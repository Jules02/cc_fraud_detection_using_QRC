"""Synthetic benchmark tasks with known answers (see QRC-Lab toolbox), for testing the QRC pipeline.

Run them with ``python -m experiments.run_benchmarks``.
"""

from qrc.benchmarks.memory import MemoryTask
from qrc.benchmarks.narma import NARMA10Task
from qrc.benchmarks.parity import ParityTask
from qrc.benchmarks.ipc import IPCTask

__all__ = ["IPCTask", "MemoryTask", "NARMA10Task", "ParityTask"]
