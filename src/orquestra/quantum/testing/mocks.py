################################################################################
# © Copyright 2020-2022 Zapata Computing Inc.
################################################################################
from orquestra.quantum.circuits import Circuit
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
