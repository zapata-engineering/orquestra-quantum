################################################################################
# © Copyright 2020-2022 Zapata Computing Inc.
################################################################################
from typing import Optional

from orquestra.quantum.circuits import Circuit
from orquestra.quantum.distributions import MeasurementOutcomeDistribution
from orquestra.quantum.measurements import Measurements
from orquestra.quantum.runners.symbolic_simulator import SymbolicSimulator


class MockCircuitRunner:

    supports_batching = False

    def __init__(self):
        super().__init__()
        self._simulator = SymbolicSimulator()

    def run_and_measure(
        self, circuit: Circuit, n_samples: int, **kwargs
    ) -> Measurements:
        return self._simulator.run_and_measure(circuit, n_samples)

    def get_measurement_outcome_distribution(
        self, circuit: Circuit, n_samples: Optional[int]
    ) -> MeasurementOutcomeDistribution:
        measurements = self._simulator.run_and_measure(circuit, n_samples)
        return measurements.get_distribution()
