# Copyright (C) 2026 Regents of the University of California
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
from __future__ import annotations

import shlex
import string
from argparse import ArgumentParser, _ArgumentGroup

from toil.batchSystems.abstractBatchSystem import InsufficientSystemResources
from toil.common import Config
from toil.job import AcceleratorRequirement, Requirer


class KeyedArgsOption:
    """
    A batch system's option giving, once per key as ``KEY=ARGS``, submission
    arguments. ARGS are split as a shell would split them.
    """

    def __init__(self, flag: str, dest: str, key_name: str) -> None:
        """
        :param flag: The command line flag of the option.
        :param dest: The name of the option in the Toil config.
        :param key_name: What the keys name, as the help and errors show it.
        """
        self.flag = flag
        self.dest = dest
        self.key_name = key_name

    def add_to(self, parser: ArgumentParser | _ArgumentGroup, help: str) -> None:
        parser.add_argument(
            self.flag,
            dest=self.dest,
            type=self.parse,
            action="append",
            default=[],
            help=help,
        )

    def parse(self, spec: str) -> tuple[str, list[str]]:
        """
        Parse the arguments given for one key.

        >>> KeyedArgsOption("--slurmPartitionArgs", "x", "PARTITION").parse(
        ...     "quantum=--exclusive --nodes=1")
        ('quantum', ['--exclusive', '--nodes=1'])
        >>> KeyedArgsOption("--slurmPartitionArgs", "x", "PARTITION").parse("quantum")
        Traceback (most recent call last):
        ...
        ValueError: Expected PARTITION=ARGS but got: 'quantum'

        Raises ValueError if not parseable.
        """
        key, separator, args = spec.partition("=")
        if not key or not separator:
            raise ValueError(f"Expected {self.key_name}=ARGS but got: {spec!r}")
        return key, shlex.split(args)

    def args_of(self, key: str, config: Config) -> list[str]:
        """
        Get the arguments the config gives for the key, or none.
        """
        args: dict[str, list[str]] = dict(getattr(config, self.dest, None) or [])
        return args.get(key, [])


class AcceleratorArgsOption(KeyedArgsOption):
    """
    A batch system's option giving, once per accelerator kind as
    ``KIND=ARGS``, the submission arguments that provide that kind.

    ARGS are split as a shell would split them. A ``{name}`` field in them is
    filled, for each job, with the property of that name of the job's
    accelerator; ``{{`` and ``}}`` stand for braces. ``KIND=`` with no
    arguments says a job gets the kind wherever it runs, such as when the job
    starts it itself. Kinds the batch system provides by other means can't be
    given.
    """

    def __init__(
        self,
        scheduler: str,
        flag: str,
        dest: str,
        provided_kinds: dict[str, str] | None = None,
    ) -> None:
        """
        :param scheduler: The name of the scheduler the arguments are for.
        :param flag: The command line flag of the option.
        :param dest: The name of the option in the Toil config.
        :param provided_kinds: The error to give, by kind, for each kind the
            batch system provides by other means.
        """
        super().__init__(flag, dest, "KIND")
        self.scheduler = scheduler
        self.provided_kinds = provided_kinds or {}

    def parse(self, spec: str) -> tuple[str, list[str]]:
        """
        Parse the arguments that provide one kind of accelerator.

        >>> option = AcceleratorArgsOption("Torque", "--torqueAcceleratorArgs", "x",
        ...     {"gpu": "GPUs are requested by other means"})
        >>> option.parse("fpga=-q fpgas")
        ('fpga', ['-q', 'fpgas'])
        >>> option.parse("qpu=-q quantum -N 'quantum job'")
        ('qpu', ['-q', 'quantum', '-N', 'quantum job'])
        >>> option.parse("qpu=-l qpu={backend}:qubits={qubits}")
        ('qpu', ['-l', 'qpu={backend}:qubits={qubits}'])
        >>> option.parse("qpu=")
        ('qpu', [])
        >>> option.parse("gpu=-q gpus")
        Traceback (most recent call last):
        ...
        ValueError: GPUs are requested by other means
        >>> option.parse("fpga")
        Traceback (most recent call last):
        ...
        ValueError: Expected KIND=ARGS but got: 'fpga'
        >>> option.parse("qpu=-l qpu={0}")
        Traceback (most recent call last):
        ...
        ValueError: Expected only {name} fields, each naming an accelerator property, in qpu arguments but got: 'qpu={0}'

        Raises ValueError if not parseable.
        """
        kind, split = super().parse(spec)
        if kind in self.provided_kinds:
            raise ValueError(self.provided_kinds[kind])
        for arg in split:
            for _, name, format_spec, conversion in string.Formatter().parse(arg):
                if name is not None and (
                    not name.isidentifier() or format_spec or conversion
                ):
                    raise ValueError(
                        f"Expected only {{name}} fields, each naming an accelerator "
                        f"property, in {kind} arguments but got: {arg!r}"
                    )
        return kind, split

    def args_for(
        self, accelerators: list[AcceleratorRequirement], config: Config
    ) -> list[str]:
        """
        Get the arguments that provide the given accelerators.

        Every kind of accelerator the batch system doesn't provide by other
        means is provided by the arguments the config maps it to, with their
        fields filled from the accelerator's properties. The accelerator counts
        are left to those arguments: a job submitted with them is assumed to
        get the accelerators they provide.

        >>> option = AcceleratorArgsOption("Torque", "--torqueAcceleratorArgs", "args",
        ...     {"gpu": "GPUs are requested by other means"})
        >>> config = Config()
        >>> config.args = [("fpga", ["-q", "fpgas"]), ("qpu", ["-l", "qpu={backend}"])]
        >>> option.args_for([{"kind": "gpu", "count": 1}], config)
        []
        >>> option.args_for([{"kind": "fpga", "count": 1}], config)
        ['-q', 'fpgas']
        >>> option.args_for(
        ...     [{"kind": "qpu", "count": 1, "properties": {"backend": "sim"}}], config
        ... )
        ['-l', 'qpu=sim']
        >>> option.args_for([{"kind": "tpu", "count": 1}], config)
        Traceback (most recent call last):
        ...
        ValueError: No Torque arguments are configured to provide tpu accelerators; map them with --torqueAcceleratorArgs tpu=ARGS
        >>> option.args_for([{"kind": "qpu", "count": 1}], config)
        Traceback (most recent call last):
        ...
        ValueError: The Torque arguments for qpu accelerators need the accelerator property backend, which the job's qpu accelerator does not have

        Raises ValueError if a kind has no arguments, or the accelerator lacks a
        property its arguments need.
        """
        accelerator_args = dict(getattr(config, self.dest, None) or [])
        args: list[str] = []
        for kind, properties in dict.fromkeys(
            (
                accelerator["kind"],
                tuple(sorted(accelerator.get("properties", {}).items())),
            )
            for accelerator in accelerators
        ):
            if kind in self.provided_kinds:
                continue
            if kind not in accelerator_args:
                raise ValueError(
                    f"No {self.scheduler} arguments are configured to provide {kind} "
                    f"accelerators; map them with {self.flag} {kind}=ARGS"
                )
            for arg in accelerator_args[kind]:
                try:
                    args.append(arg.format_map(dict(properties)))
                except KeyError as e:
                    raise ValueError(
                        f"The {self.scheduler} arguments for {kind} accelerators need "
                        f"the accelerator property {e.args[0]}, which the job's {kind} "
                        f"accelerator does not have"
                    ) from e
        return args

    def check(self, requirer: Requirer, config: Config) -> None:
        """
        Raise InsufficientSystemResources if the config can't provide the
        accelerators the requirer needs.
        """
        try:
            self.args_for(requirer.accelerators, config)
        except ValueError as e:
            raise InsufficientSystemResources(
                requirer,
                "accelerators",
                details=[
                    f"The accelerators {requirer.accelerators} could not be provided.",
                    str(e),
                ],
            ) from e
