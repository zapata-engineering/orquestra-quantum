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
from orquestra.quantum.backends import SymbolicSimulator
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


@pytest.fixture
def backend():
    return SymbolicSimulator()


@pytest.fixture
def wf_simulator() -> SymbolicSimulator:
    return SymbolicSimulator()


class SymbolicSimulatorWithNonSupportedOperations(SymbolicSimulator):
    def is_natively_supported(self, operation: Operation) -> bool:
        if isinstance(operation, GateOperation) and operation.gate.name == "RX":
            return False
        return super().is_natively_supported(operation)


class SymbolicSimulatorWithDefaultSetOfSupportedOperations(SymbolicSimulator):
    def is_natively_supported(self, operation: Operation) -> bool:
        return QuantumSimulator.is_natively_supported(self, operation)


class TestSymbolicSimulator(QuantumSimulatorTests):
    gates_list = [
        XX(sympy.Symbol("theta"))(2, 1),
        U3(
            sympy.Symbol("alpha"),
            sympy.Symbol("beta"),
            sympy.Symbol("gamma"),
        )(1),
    ]
    incorrect_bindings = [
        {},
        {
            "alpha": sympy.pi,
            "beta": sympy.pi,
        },
    ]
    correct_bindings = [
        dict(zip(gate.free_symbols, [sympy.pi] * len(tuple(gate.free_symbols))))
        for gate in gates_list
    ]

    @pytest.mark.parametrize(
        "gate, binding",
        list(zip(gates_list, incorrect_bindings)),
    )
    def test_cannot_sample_from_circuit_containing_free_symbols(
        self, wf_simulator, gate, binding
    ):

        circuit = Circuit([gate])
        with pytest.raises(ValueError):
            wf_simulator.run_circuit_and_measure(
                circuit, n_samples=1000, symbol_map=binding
            )

    @pytest.mark.parametrize(
        "gate, binding",
        list(zip(gates_list, correct_bindings)),
    )
    def test_passes_for_complete_bindings(self, wf_simulator, gate, binding):
        circuit = Circuit([gate])
        wf_simulator.run_circuit_and_measure(
            circuit, n_samples=1000, symbol_map=binding
        )

    @pytest.mark.parametrize(
        "circuit",
        [
            Circuit([RY(0.5)(0), RX(1)(1), CNOT(0, 2), RX(np.pi)(2)]),
            Circuit([RX(1)(1), CNOT(0, 2), RX(np.pi)(2), RY(0.5)(0)]),
            Circuit([RX(1)(1), CNOT(0, 2), RX(np.pi)(2), RY(0.5)(0), RX(0.5)(0)]),
        ],
    )
    def test_quantum_simulator_switches_between_native_and_nonnative_modes_of_execution(
        self, circuit
    ):
        simulator = SymbolicSimulatorWithNonSupportedOperations()
        reference_simulator = SymbolicSimulator()

        np.testing.assert_array_equal(
            simulator.get_wavefunction(circuit).amplitudes,
            reference_simulator.get_wavefunction(circuit).amplitudes,
        )

    def test_by_default_only_gate_operations_are_supported(self):
        simulator = SymbolicSimulatorWithDefaultSetOfSupportedOperations()
        assert simulator.is_natively_supported(RX(np.pi / 2)(1))
        assert not simulator.is_natively_supported(
            MultiPhaseOperation((0.5, 0.2, 0.3, 0.1))
        )


class TestSymbolicSimulatorGates(QuantumSimulatorGatesTest):
    pass
