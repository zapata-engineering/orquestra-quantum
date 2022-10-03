################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################

from typing import Protocol, Iterable, List, Union, Optional
from ..circuits import Circuit
from ..measurements import Measurements
from .circuit_runner import CircuitRunner
from abc import ABC, abstractmethod


class GateModelDevice(Protocol, CircuitRunner):

    @property
    def n_jobs_executed(self) -> int:
        pass

    @property
    def n_circuits_executed(self) -> int:
        pass


class BaseGateModelDevice(ABC, GateModelDevice):

    supports_batching: bool = False
    batch_size: Optional[int] = None

    def __init__(self):
        self.n_circuits_executed = 0
        self.n_jobs_executed = 0

        if self.supports_batching:
            assert isinstance(self.batch_size, int)
            assert self.batch_size > 0

    def run_and_measure(self, circuit: Circuit, n_samples: int) -> Measurements:
        """
        TODO
        """
        # TODO?
        if n_samples <= 0:
            raise ValueError("n_samples should be greater than 0.")

        return self._run_and_measure(circuit, n_samples)


    @abstractmethod
    def _run_and_measure(self, circuit: Circuit, n_samples: int) -> Measurements:
        """
        TODO
        """