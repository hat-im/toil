import sys
from pathlib import Path
from queue import Queue
from uuid import uuid4

import pytest

import toil.batchSystems.torque
from toil.batchSystems.abstractBatchSystem import (
    BatchSystemSupport,
    InsufficientSystemResources,
)
from toil.common import Config
from toil.job import Job, Requirer
from toil.test import ToilTest


class FakeBatchSystem(BatchSystemSupport):
    """
    Class that implements a minimal Batch System, needed to create a Worker (see below).
    """

    def __init__(self):
        config = Config()
        config.workflowID = str(uuid4())
        config.cleanWorkDir = "always"
        toil.batchSystems.torque.TorqueBatchSystem.setOptions(
            lambda o: setattr(config, o, None)
        )
        super().__init__(config, float("inf"), sys.maxsize, sys.maxsize)

    def getWaitDuration(self):
        return 10


FakeBatchSystem.__abstractmethods__ = set()


def call_pbsnodes(args, **_) -> str:
    if args[0] == "pbsnodes":
        return "pbs_version = 2022.1.1"
    raise RuntimeError(f"Cannot fake command call: {args}")


class TorqueTest(ToilTest):
    """
    Class for unit-testing TorqueBatchSystem
    """

    def setUp(self):
        self.monkeypatch = pytest.MonkeyPatch()
        self.monkeypatch.setattr(toil.batchSystems.torque, "call_command", call_pbsnodes)
        self.monkeypatch.delenv("TOIL_TORQUE_ARGS", raising=False)
        self.monkeypatch.delenv("TOIL_TORQUE_REQS", raising=False)
        self.worker = toil.batchSystems.torque.TorqueBatchSystem.GridEngineThread(
            newJobsQueue=Queue(),
            updatedJobsQueue=Queue(),
            killQueue=Queue(),
            killedJobsQueue=Queue(),
            boss=FakeBatchSystem(),
        )

    def tearDown(self):
        self.monkeypatch.undo()

    def test_prepareQsub_accelerator_args(self):
        self.worker.boss.config.torque_accelerator_args = [
            toil.batchSystems.torque.TORQUE_ACCELERATOR_ARGS.parse(
                "qpu=-q quantum -l qpu={backend}"
            )
        ]
        qpu = [{"kind": "qpu", "count": 1, "properties": {"backend": "sim"}}]

        # A job's accelerator reaches its qsub with the arguments its kind is mapped to
        command = self.worker.prepareQsub(1, 1024, 5, None, qpu)
        assert command[-4:] == ["-q", "quantum", "-l", "qpu=sim"]

        # Jobs without that accelerator don't get them
        command = self.worker.prepareQsub(1, 1024, 5, None, None)
        assert "quantum" not in command

        # A kind a job starts itself needs no arguments
        self.worker.boss.config.torque_accelerator_args = [
            toil.batchSystems.torque.TORQUE_ACCELERATOR_ARGS.parse("qpu=")
        ]
        assert self.worker.prepareQsub(1, 1024, 5, None, qpu) == self.worker.prepareQsub(
            1, 1024, 5, None, None
        )

        # Resources Toil sets itself can't be given for an accelerator
        self.worker.boss.config.torque_accelerator_args = [("qpu", ["-l", "mem=10G"])]
        with self.assertRaises(ValueError):
            self.worker.prepareQsub(1, 1024, 5, None, qpu)

    def test_prepareSubmission_accelerator_args(self):
        self.worker.boss.config.torque_accelerator_args = [("fpga", ["-q", "fpgas"])]
        command = self.worker.prepareSubmission(
            1, 1024, 0, 0, 5, "true", "job5", None, [{"kind": "fpga", "count": 1}]
        )
        assert command[-3:-1] == ["-q", "fpgas"]

    def test_prepareQsub_processes(self):
        self.worker._version = "pro"
        command = self.worker.prepareQsub(4, 4 * 2**30, 5, None)
        assert command[command.index("-l") + 1] == "select=1:mem=4194304K:ncpus=4"

        command = self.worker.prepareQsub(96, 96 * 2**30, 5, None, None, 2)
        assert (
            command[command.index("-l") + 1]
            == "select=2:ncpus=48:mpiprocs=1:mem=50331648K"
        )

        command = self.worker.prepareQsub(1, 2**30, 5, None, None, 3)
        assert (
            command[command.index("-l") + 1]
            == "select=3:ncpus=1:mpiprocs=1:mem=349526K"
        )

    def test_prepareSubmission_walltime(self):
        self.worker._version = "pro"
        command = self.worker.prepareSubmission(1, 2**30, 0, 3725, 5, "true", "job5")
        assert (
            command[command.index("-l") + 1] == "select=1:mem=1048576K,walltime=1:02:05"
        )

        command = self.worker.prepareSubmission(1, 2**30, 0, 0, 5, "true", "job5")
        assert command[command.index("-l") + 1] == "select=1:mem=1048576K"

    def test_prepareSubmission_walltime_args(self):
        self.worker._version = "pro"
        for variable in ("TOIL_TORQUE_REQS", "TOIL_TORQUE_ARGS"):
            with self.subTest(variable=variable):
                self.monkeypatch.setenv(variable, "-l walltime=2:00:00")
                with self.assertRaisesRegex(ValueError, "Incompatible"):
                    self.worker.prepareSubmission(
                        1, 2**30, 0, 60, 5, "true", "job5"
                    )
                command = self.worker.prepareSubmission(
                    1, 2**30, 0, 0, 5, "true", "job5"
                )
                assert "walltime=2:00:00" in " ".join(command)
                self.monkeypatch.delenv(variable)

    def test_prepareSubmission_oss_walltime(self):
        self.worker._version = "oss"
        command = self.worker.prepareSubmission(1, 2**30, 0, 90000, 5, "true", "job5")
        assert command[command.index("-l") + 1] == "mem=1048576K,walltime=25:00:00"

    def test_prepareSubmission_processes(self):
        self.worker._version = "pro"
        command = self.worker.prepareSubmission(
            2, 2**30, 0, 0, 5, "true", "job5", None, None, 2
        )
        assert command[command.index("-l") + 1].startswith("select=2:ncpus=1:")

    def test_prepareQsub_placement_args(self):
        self.worker._version = "pro"
        for variable in ("TOIL_TORQUE_ARGS", "TOIL_TORQUE_REQS"):
            for args in ("-l select=2:ncpus=4", "-l place=scatter"):
                with self.subTest(variable=variable, args=args):
                    self.monkeypatch.setenv(variable, args)
                    with self.assertRaisesRegex(ValueError, "Incompatible"):
                        self.worker.prepareQsub(2, 2**30, 5, None, None, 2)
                    self.monkeypatch.delenv(variable)

        self.monkeypatch.setenv("TOIL_TORQUE_ARGS", "-l select=2")
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            self.worker.prepareQsub(2, 2**30, 5, None)
        self.monkeypatch.setenv("TOIL_TORQUE_ARGS", "-l place=excl")
        command = self.worker.prepareQsub(2, 2**30, 5, None)
        assert command[-2:] == ["-l", "place=excl"]

    def test_pbsVersion(self):
        for out, version in [
            ("pbs_version = 2026.1.0.20260821184519\n", "pro"),
            ("", "oss"),
        ]:
            self.monkeypatch.setattr(
                toil.batchSystems.torque, "call_command", lambda args, **_: out
            )
            assert self.worker._pbsVersion() == version, out

    def test_prepareQsub_oss_processes(self):
        self.worker._version = "oss"
        command = self.worker.prepareQsub(2, 2**30, 5, None)
        assert command[command.index("-l") + 1] == "mem=1048576K,nodes=1:ppn=2"
        with self.assertRaisesRegex(ValueError, "OSS Torque"):
            self.worker.prepareQsub(2, 2**30, 5, None, None, 2)

    def test_process_placement_variables(self):
        assert toil.batchSystems.torque.TorqueBatchSystem.process_placement_variables() == [
            "TOIL_PROCESS_HOSTFILE",
            "PBS_.*",
        ]

    def test_check_accelerator_request(self):
        boss = toil.batchSystems.torque.TorqueBatchSystem.__new__(
            toil.batchSystems.torque.TorqueBatchSystem
        )
        boss.config = FakeBatchSystem().config
        boss.config.torque_accelerator_args = [("fpga", ["-q", "fpgas"])]

        def check(accelerators):
            boss._check_accelerator_request(Requirer({"accelerators": accelerators}))

        # Mapped kinds can be provided
        check([])
        check([{"kind": "fpga", "count": 1}])

        # Unmapped kinds, GPUs among them, can't, and the error says how to map them
        with self.assertRaisesRegex(
            InsufficientSystemResources, "--torqueAcceleratorArgs qpu=ARGS"
        ):
            check([{"kind": "qpu", "count": 1}])
        with self.assertRaisesRegex(
            InsufficientSystemResources, "--torqueAcceleratorArgs gpu=ARGS"
        ):
            check([{"kind": "gpu", "count": 1}])

    def test_torqueAcceleratorArgs_option(self):
        parser = Job.Runner.getDefaultArgumentParser()
        options = parser.parse_args(
            [
                "file:store",
                "--batchSystem=torque",
                "--torqueAcceleratorArgs",
                "qpu=",
                "--torqueAcceleratorArgs",
                "fpga=-q fpgas",
            ]
        )
        config = Config()
        config.setOptions(options)
        assert config.torque_accelerator_args == [("qpu", []), ("fpga", ["-q", "fpgas"])]


def test_process_hosts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    A PBS job's processes must be named on the hosts of its PBS_NODEFILE.
    """
    nodefile = tmp_path / "nodefile"
    nodefile.write_text("node1.cluster\nnode1.cluster\nnode2.cluster\n")
    monkeypatch.setenv("PBS_NODEFILE", str(nodefile))
    assert toil.batchSystems.torque.TorqueBatchSystem.process_hosts(3) == [
        "node1.cluster",
        "node1.cluster",
        "node2.cluster",
    ]
