import sys
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

    def test_check_accelerator_request(self):
        boss = FakeBatchSystem()
        boss.config.torque_accelerator_args = [("fpga", ["-q", "fpgas"])]

        def check(accelerators):
            requirer = Requirer({"accelerators": accelerators})
            toil.batchSystems.torque.TorqueBatchSystem._check_accelerator_request(
                boss, requirer
            )

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

