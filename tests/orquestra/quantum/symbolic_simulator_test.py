################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
import numpy as np
import pytest
import sympy

from orquestra.quantum.api.backend import QuantumSimulator
from orquestra.quantum.api.backend_test import (
    QuantumSimulatorGatesTest,
    QuantumSimulatorTests,
)
from orquestra.quantum.api.circuit_runner_contracts import CIRCUIT_RUNNER_CONTRACTS
from orquestra.quantum.api.gate_model_simulator_contracts import simulator_contracts_for_tolerance
from orquestra.quantum.circuits import (
    CNOT,
    RX,
    RY,
    U3,
    XX,
    Circuit,
    GateOperation,
    MultiPhaseOperation,
    Operation,
)
from orquestra.quantum.symbolic_simulator import SymbolicSimulator


@pytest.fixture
def backend():
    return SymbolicSimulator()


@pytest.fixture
def wf_simulator() -> SymbolicSimulator:
    return SymbolicSimulator()


class TestSymbolicSimulator(QuantumSimulatorTests):
    gates_list = [
        XX(sympy.Symbol("theta"))(2, 1),
        U3(
            sympy.Symbol("alpha"),
            sympy.Symbol("beta"),
            sympy.Symbol("gamma"),
        )(1),
    ]

    @pytest.mark.parametrize(
        "gate",
        gates_list,
    )
    def test_cannot_sample_from_circuit_containing_free_symbols(self, gate):
        simulator = SymbolicSimulator()
        circuit = Circuit([gate])

        with pytest.raises(ValueError):
            simulator.run_and_measure(circuit, n_samples=1000)


class TestSymbolicSimulatorGates(QuantumSimulatorGatesTest):
    pass


@pytest.mark.parametrize("contract", CIRCUIT_RUNNER_CONTRACTS)
def test_symbolic_simulator_fullfills_circuit_runner_contracts(contract):
    simulator = SymbolicSimulator()
    assert contract(simulator)


@pytest.mark.parametrize("contract", simulator_contracts_for_tolerance())
def test_symbolic_simulator_fulfills_simulator_contracts(contract):
    simulator = SymbolicSimulator()
    assert contract(simulator)
