################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
import pytest
import sympy

from orquestra.quantum.api.circuit_runner_contracts import (
    CIRCUIT_RUNNER_CONTRACTS,
    STRICT_CIRCUIT_RUNNER_CONTRACTS,
    circuit_runner_gate_compatibility_contracts,
)
from orquestra.quantum.api.wavefunction_simulator_contracts import (
    simulator_contracts_for_tolerance,
    simulator_gate_compatibility_contracts,
)
from orquestra.quantum.circuits import U3, XX, Circuit
from orquestra.quantum.runners.symbolic_simulator import SymbolicSimulator


@pytest.fixture
def backend():
    return SymbolicSimulator()


@pytest.fixture
def wf_simulator() -> SymbolicSimulator:
    return SymbolicSimulator()


class TestSymbolicSimulator:
    @pytest.mark.parametrize(
        "gate",
        [
            XX(sympy.Symbol("theta"))(2, 1),
            U3(
                sympy.Symbol("alpha"),
                sympy.Symbol("beta"),
                sympy.Symbol("gamma"),
            )(1),
        ],
    )
    def test_cannot_sample_from_circuit_containing_free_symbols(self, gate):
        simulator = SymbolicSimulator()
        circuit = Circuit([gate])

        with pytest.raises(ValueError):
            simulator.run_and_measure(circuit, n_samples=1000)


@pytest.mark.parametrize("contract", CIRCUIT_RUNNER_CONTRACTS)
def test_symbolic_simulator_fullfills_circuit_runner_contracts(contract):
    simulator = SymbolicSimulator()
    assert contract(simulator)


@pytest.mark.parametrize("contract", simulator_contracts_for_tolerance())
def test_symbolic_simulator_fulfills_simulator_contracts(contract):
    simulator = SymbolicSimulator()
    assert contract(simulator)


@pytest.mark.parametrize("contract", STRICT_CIRCUIT_RUNNER_CONTRACTS)
def test_symbolic_simulator_fulfills_strict_circuit_runnner(contract):
    simulator = SymbolicSimulator()
    assert contract(simulator)


class TestSymbolicSimulatorUsesCorrectGateDefinitions:
    @pytest.mark.parametrize("contract", simulator_gate_compatibility_contracts())
    def test_using_target_amplitudes(self, contract):
        simulator = SymbolicSimulator()
        assert contract(simulator)

    @pytest.mark.parametrize("contract", circuit_runner_gate_compatibility_contracts())
    def test_using_expectation_values(self, contract):
        simulator = SymbolicSimulator(seed=1234)
        assert contract(simulator)
