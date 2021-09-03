import pytest
import sympy
from zquantum.core import circuits
from zquantum.core.interfaces.backend_test import (
    QuantumSimulatorGatesTest,
    QuantumSimulatorTests,
)
from zquantum.core.symbolic_simulator import SymbolicSimulator


@pytest.fixture
def backend():
    return SymbolicSimulator()


@pytest.fixture
def wf_simulator():
    return SymbolicSimulator()


class TestSymbolicSimulator(QuantumSimulatorTests):
    def test_cannot_sample_from_circuit_containing_free_symbols(self, wf_simulator):
        circuit = circuits.Circuit([circuits.XX(sympy.Symbol("theta"))(2, 1)])
        with pytest.raises(ValueError):
            wf_simulator.run_circuit_and_measure(circuit, n_samples=1000)


class TestSymbolicSimulatorGates(QuantumSimulatorGatesTest):
    pass
