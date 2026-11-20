# Copyright (C) 2015-2021 Regents of the University of California
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import math
import os
import re
import shlex
import tempfile
from argparse import ArgumentParser, _ArgumentGroup
from queue import Empty, Queue
from shlex import quote

from toil.batchSystems.abstractBatchSystem import UpdatedBatchJobInfo
from toil.batchSystems.abstractGridEngineBatchSystem import (
    AbstractGridEngineBatchSystem,
    JobTuple,
)
from toil.batchSystems.accelerator_args import AcceleratorArgsOption
from toil.batchSystems.allocation import pbs_hosts
from toil.batchSystems.options import OptionSetter
from toil.job import AcceleratorRequirement
from toil.lib.conversions import hms_duration_to_seconds
from toil.lib.misc import CalledProcessErrorStderr, call_command

logger = logging.getLogger(__name__)

TORQUE_ACCELERATOR_ARGS = AcceleratorArgsOption(
    "Torque", "--torqueAcceleratorArgs", "torque_accelerator_args"
)


def refuse_toil_resources(args: str, processes: int = 1, walltime: int = 0) -> None:
    """
    Raise ValueError if the qsub arguments set resources Toil sets itself,
    including the placement of a job of more than one process and the
    walltime of a job that has one.
    """
    toil_resources = ["mem=", "nodes=", "ppn=", "select="]
    if processes > 1:
        toil_resources.append("place=")
    if walltime:
        toil_resources.append("walltime=")
    if any(resource in args for resource in toil_resources):
        raise ValueError(
            f"Incompatible resource arguments ({', '.join(map(repr, toil_resources))}): {args}"
        )


def pbs_job_name(jobID: int, jobName: str) -> str:
    """
    Name a job as Slurm's batch system does, toil_job_<ID>_<name>, in what
    every PBS accepts for -N: OpenPBS's check_job_name takes at most 230
    characters, each a letter, a digit or one of -_+. (src/lib/Libcmds/
    ck_job_name.c, PBS_MAXJOBNAME in src/include/pbs_ifl.h), and Torque any
    printable ones up to 256 (src/lib/Libcmds/ck_job_name.c).

    >>> pbs_job_name(5, "CWLJob")
    'toil_job_5_CWLJob'
    >>> pbs_job_name(5, "Unknown Job/step:1")
    'toil_job_5_Unknown_Job_step_1'
    >>> len(pbs_job_name(5, "x" * 300))
    230
    """
    return re.sub(r"[^A-Za-z0-9_+.-]", "_", f"toil_job_{jobID}_{jobName}")[:230]


class TorqueBatchSystem(AbstractGridEngineBatchSystem):
    accelerator_args = TORQUE_ACCELERATOR_ARGS

    # class-specific Worker
    class GridEngineThread(AbstractGridEngineBatchSystem.GridEngineThread[list[str]]):
        def __init__(
            self,
            newJobsQueue: Queue[JobTuple | None],
            updatedJobsQueue: Queue[UpdatedBatchJobInfo],
            killQueue: Queue[int],
            killedJobsQueue: Queue[int],
            boss: AbstractGridEngineBatchSystem,
        ) -> None:
            super().__init__(
                newJobsQueue, updatedJobsQueue, killQueue, killedJobsQueue, boss
            )
            self._version = self._pbsVersion()

        def _pbsVersion(self) -> str:
            """Determines PBS/Torque version via pbsnodes"""
            try:
                out = call_command(["pbsnodes", "--version"])
                if "PBSPro" in out or out.lstrip().startswith("pbs_version ="):
                    logger.debug("PBS Pro / OpenPBS version detected")
                    self._version = "pro"
                else:
                    logger.debug("Torque OSS version detected")
                    self._version = "oss"
            except CalledProcessErrorStderr as e:
                if e.returncode != 0:
                    logger.error("Could not determine PBS/Torque version")

            return self._version

        """
        Torque-specific AbstractGridEngineWorker methods
        """

        def getRunningJobIDs(self) -> dict[int, float]:
            times: dict[int, float] = {}
            with self.runningJobsLock:
                currentjobs = {
                    str(self.batchJobIDs[x][0]).strip().split(".")[0]: x
                    for x in self.runningJobs
                }
            logger.debug("getRunningJobIDs current jobs are: %s", currentjobs)
            # Skip running qstat if we don't have any current jobs
            if not currentjobs:
                return times
            # Only query for job IDs to avoid clogging the batch system on heavily loaded clusters
            # PBS plain qstat will return every running job on the system.
            jobids = sorted(list(currentjobs.keys()))
            if self._version == "pro":
                stdout = call_command(["qstat", "-x"] + jobids)
            elif self._version == "oss":
                stdout = call_command(["qstat"] + jobids)

            # qstat supports XML output which is more comprehensive, but PBSPro does not support it
            # so instead we stick with plain commandline qstat tabular outputs
            for currline in stdout.split("\n"):
                items = currline.strip().split()
                if items:
                    jobid = items[0].strip().split(".")[0]
                    if jobid in currentjobs:
                        logger.debug("getRunningJobIDs job status for is: %s", items[4])
                    if jobid in currentjobs and items[4] == "R":
                        walltime = items[3].strip()
                        logger.debug(
                            "getRunningJobIDs qstat reported walltime is: %s", walltime
                        )
                        # normal qstat has a quirk with job time where it reports '0'
                        # when initially running; this catches this case
                        if walltime == "0":
                            seconds = 0.0
                        elif not walltime:
                            # Sometimes we don't get any data here.
                            # See https://github.com/DataBiosphere/toil/issues/3715
                            logger.warning(
                                "Assuming 0 walltime due to missing field in qstat line: %s",
                                currline,
                            )
                            seconds = 0.0
                        else:
                            seconds = hms_duration_to_seconds(walltime)
                        times[currentjobs[jobid]] = seconds

            logger.debug("Job times from qstat are: %s", times)
            return times

        def killJob(self, jobID: int) -> None:
            call_command(["qdel", self.getBatchSystemID(jobID)])

        def prepareSubmission(
            self,
            cpu: float,
            memory: int,
            disk: int,
            walltime: int,
            jobID: int,
            command: str,
            jobName: str,
            job_environment: dict[str, str] | None = None,
            accelerators: list[AcceleratorRequirement] | None = None,
            processes: int = 1,
        ) -> list[str]:
            return self.prepareQsub(
                cpu,
                memory,
                jobID,
                jobName,
                job_environment,
                accelerators,
                processes,
                walltime,
            ) + [self.generateTorqueWrapper(command, jobID)]

        def submitJob(self, subLine: list[str]) -> str:
            return call_command(subLine).strip()

        def submission_queue(self, subLine: list[str]) -> str | None:
            queue = None
            for i, arg in enumerate(subLine):
                if arg == "-q" and i + 1 < len(subLine):
                    queue = subLine[i + 1]
                elif arg.startswith("-q") and len(arg) > 2:
                    queue = arg[2:]
            return queue

        def getJobExitCode(self, torqueJobID: str) -> int | None:
            if self._version == "pro":
                args = ["qstat", "-x", "-f", str(torqueJobID).split(".")[0]]
            elif self._version == "oss":
                args = ["qstat", "-f", str(torqueJobID).split(".")[0]]

            stdout = call_command(args)
            for line in stdout.split("\n"):
                line = line.strip()
                # Case differences due to PBSPro vs OSS Torque qstat outputs
                if (
                    line.startswith("failed")
                    or line.startswith("FAILED")
                    and int(line.split()[1]) == 1
                ):
                    return 1
                if line.startswith("exit_status") or line.startswith("Exit_status"):
                    status = line.split(" = ")[1]
                    logger.debug("Exit Status: %s", status)
                    return int(status)
                if "unknown job id" in line.lower():
                    # some clusters configure Torque to forget everything about just
                    # finished jobs instantly, apparently for performance reasons
                    logger.debug(
                        "Batch system no longer remembers about job %s", torqueJobID
                    )
                    # return assumed success; status files should reveal failure
                    return 0
            return None

        """
        Implementation-specific helper methods
        """

        def prepareQsub(
            self,
            cpu: float,
            mem: int,
            jobID: int,
            jobName: str,
            job_environment: dict[str, str] | None,
            accelerators: list[AcceleratorRequirement] | None = None,
            processes: int = 1,
            walltime: int = 0,
        ) -> list[str]:
            """
            Returns the qsub command line to run to queue the job, without its
            script.

            On PBS Pro, a job of more than one process is submitted as a chunk
            per process sharing its cores and memory, which PBS may place on
            different nodes. OSS Torque jobs can only have one process.
            """

            # TODO: passing $PWD on command line not working for -d, resorting to
            # $PBS_O_WORKDIR but maybe should fix this here instead of in script?

            qsubline = [
                "qsub",
                "-S",
                "/bin/sh",
                "-V",
                "-N",
                pbs_job_name(jobID, jobName),
            ]

            environment = self.boss.environment.copy()
            if job_environment:
                environment.update(job_environment)

            if environment:
                qsubline.append("-v")
                qsubline.append(
                    ",".join(
                        k + "=" + quote(os.environ[k] if v is None else v)
                        for k, v in environment.items()
                    )
                )

            reqline = list()
            if self._version == "pro" and processes > 1:
                request = (
                    f"select={processes}:ncpus={math.ceil(cpu / processes)}:mpiprocs=1"
                )
                if mem is not None:
                    request += f":mem={math.ceil(mem / 1024 / processes)}K"
                reqline.append(request)
            elif self._version == "pro":
                request = "select=1"
                if mem is not None:
                    request += f":mem={mem//1024}K"
                if cpu is not None and math.ceil(cpu) > 1:
                    request += ":ncpus=" + str(int(math.ceil(cpu)))
                reqline.append(request)
            elif processes > 1:
                raise ValueError(
                    f"Cannot start the {processes} processes of job {jobID} on "
                    f"OSS Torque, which Toil only gives jobs of one process"
                )
            else:
                if mem is not None:
                    reqline.append(f"mem={mem//1024}K")
                if cpu is not None and math.ceil(cpu) > 1:
                    reqline.append("nodes=1:ppn=" + str(int(math.ceil(cpu))))

            if walltime:
                minutes, seconds = divmod(math.ceil(walltime), 60)
                hours, minutes = divmod(minutes, 60)
                reqline.append(f"walltime={hours}:{minutes:02d}:{seconds:02d}")

            # Other resource requirements can be passed through the environment (see man qsub)
            reqlineEnv = os.getenv("TOIL_TORQUE_REQS")
            if reqlineEnv is not None:
                logger.debug(
                    "Additional Torque resource requirements appended to qsub from "
                    "TOIL_TORQUE_REQS env. variable: %s",
                    reqlineEnv,
                )
                refuse_toil_resources(reqlineEnv, processes, walltime)

                reqline.append(reqlineEnv)

            if reqline:
                qsubline += ["-l", ",".join(reqline)]

            # All other qsub parameters can be passed through the environment (see man qsub).
            # No attempt is made to parse them out here and check that they do not conflict
            # with those that we already constructed above
            arglineEnv = os.getenv("TOIL_TORQUE_ARGS")
            if arglineEnv is not None:
                logger.debug(
                    "Native Torque options appended to qsub from TOIL_TORQUE_ARGS env. variable: %s",
                    arglineEnv,
                )
                refuse_toil_resources(arglineEnv, processes, walltime)
                qsubline += shlex.split(arglineEnv)

            accelerator_args = TORQUE_ACCELERATOR_ARGS.args_for(
                accelerators or [], self.boss.config
            )
            refuse_toil_resources(shlex.join(accelerator_args), processes, walltime)
            qsubline += accelerator_args

            return qsubline

        def generateTorqueWrapper(self, command: str, jobID: int) -> str:
            """
            A very simple script generator that just wraps the command given; for
            now this goes to default tempdir
            """
            stdoutfile: str = self.boss.format_std_out_err_path(
                jobID, r"${PBS_JOBID}", "out"
            )
            stderrfile: str = self.boss.format_std_out_err_path(
                jobID, r"${PBS_JOBID}", "err"
            )

            fd, tmp_file = tempfile.mkstemp(suffix=".sh", prefix="torque_wrapper")
            with open(tmp_file, "w") as f:
                f.write("#!/bin/sh\n")
                f.write(f"#PBS -o {stdoutfile}\n")
                f.write(f"#PBS -e {stderrfile}\n")
                f.write("cd $PBS_O_WORKDIR\n\n")
                f.write(command + "\n")
            os.close(fd)
            return tmp_file

    @classmethod
    def process_placement_variables(cls) -> list[str]:
        return super().process_placement_variables() + ["PBS_.*"]

    @classmethod
    def process_hosts(cls, processes: int) -> list[str]:
        # A job of more than one process is submitted as a chunk per process,
        # each with one MPI process and so one line in PBS_NODEFILE.
        return pbs_hosts()

    @classmethod
    def add_options(cls, parser: ArgumentParser | _ArgumentGroup) -> None:
        TORQUE_ACCELERATOR_ARGS.add_to(
            parser,
            help="qsub arguments for Torque or PBS jobs that ask for an "
            "accelerator, for one accelerator kind, such as the queue that has "
            "it (fpga=-q fpga-nodes) or a site's own resource for it. Give it "
            "once per kind. A {name} field in them is filled with the job's "
            "accelerator property of that name. Jobs asking for an accelerator "
            "kind not given here, or lacking a property it needs, are refused. "
            "KIND= with no arguments says a job gets the kind wherever it runs.",
        )

    @classmethod
    def setOptions(cls, setOption: OptionSetter) -> None:
        setOption("torque_accelerator_args")
