################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
from abc import ABC, abstractmethod
from typing import Protocol, Sequence, List, Union, Optional
from ..circuits import Circuit
from ..measurements import Measurements
from ..distributions import MeasurementOutcomeDistribution


class CircuitRunner(Protocol):

    def run_and_measure(self, circuit: Circuit, n_samples: int) -> Measurements:
        pass

    def run_batch_and_measure(
        self, circuits_batch: Sequence[Circuit], n_samples: Union[int, Sequence[int]]
    ) -> List[Measurements]:
        pass

    def get_measurement_outcome_distribution(
        self, circuit: Circuit, n_samples: Optional[int]
    ) -> MeasurementOutcomeDistribution:
        pass

    @property
    def n_jobs_executed(self) -> int:
        pass

    @property
    def n_circuits_executed(self) -> int:
        pass


class BaseCircuitRunner(ABC, CircuitRunner):

    def __init__(self):
        super().__init__()
        self._n_circuits_executed = 0
        self._n_jobs_executed= 0

    def run_and_measure(self, circuit: Circuit, n_samples: int) -> Measurements:
        result = self._run_and_measure(circuit, n_samples)
        self._n_circuits_executed += 1
        self._n_jobs_executed += 1
        return result

    def run_batch_and_measure(
        self, circuits_batch: Sequence[Circuit], n_samples: Union[int, Sequence[int]]
    ) -> List[Measurements]:
        samples_per_circuit = (
            len(circuits_batch) * [n_samples]
            if isinstance(n_samples, int)
            else n_samples
        )
        if len(samples_per_circuit) != len(circuits_batch):
            raise ValueError(
                "Number of samples has to be an integer or a sequence of length "
                "equal to the length of batch. Length of batch: "
                f"{len(circuits_batch)}, length of n_samples: {len(n_samples)}."
            )
        return [
            self.run_and_measure(circuit, n)
            for circuit, n in zip(circuits_batch, samples_per_circuit)
        ]

    @abstractmethod
    def _run_and_measure(self, circuit: Circuit, n_samples: int) -> Measurements:
        pass

    @property
    def n_jobs_executed(self) -> int:
        return self._n_jobs_executed

    @property
    def n_circuits_executed(self) -> int:
        return self._n_circuits_executed

    def get_measurement_outcome_distribution(
        self, circuit: Circuit, n_samples: Optional[int]
    ) -> MeasurementOutcomeDistribution:
        measurements = self.run_and_measure(circuit, n_samples)
        return measurements.get_distribution()