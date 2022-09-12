################################################################################
# © Copyright 2020-2022 Zapata Computing Inc.
################################################################################
from ..api.backend import QuantumBackend
from ..backends import SymbolicSimulator
from ..circuits import Circuit
from ..measurements import Measurements


class MockQuantumBackend(QuantumBackend):

    supports_batching = False

    def __init__(self):
        super().__init__()
        self._simulator = SymbolicSimulator()

    def run_circuit_and_measure(
        self, circuit: Circuit, n_samples: int, **kwargs
    ) -> Measurements:
        super(MockQuantumBackend, self).run_circuit_and_measure(circuit, n_samples)

        return self._simulator.run_circuit_and_measure(circuit, n_samples)
