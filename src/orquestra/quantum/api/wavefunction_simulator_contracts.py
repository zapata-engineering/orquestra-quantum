################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################


# Length of wavefunction matches 2 ** number_of_quits
# Test values of specific wavefunction
# - for default initial state
# - for initial some specific initial state
# Number of circuits run increases
# Number of jobs_executed increases

# Test distribution values for some specific case
# Number of circuits run increases
# Number of jobs_executed increases

# Test values of expectation values
# Number of circuits run increases
# Number of jobs_executed increases
import numpy as np

from orquestra.quantum.api.wavefunction_simulator import WavefunctionSimulator
from orquestra.quantum.circuits import Circuit, H, CNOT


_example_circuits = [
    Circuit([H(0)]),
    Circuit([H(0), CNOT(0, 1), CNOT(1, 2)]),
]

_corresponding_wavefunctions = [
    np.array([1, 1]) / np.sqrt(2),
    np.array([1, 0, 0, 0, 0, 0, 0, 1]) / np.sqrt(2),
]


def _verify_wavefunction_returned_by_simulator_has_correct_length(
    simulator: WavefunctionSimulator,
):
    return all(
        len(simulator.get_wavefunction(circuit)) == 2**circuit.n_qubits
        for circuit in _example_circuits
    )


def _verify_wavefunction_returned_by_simulator_has_correct_coefficients(atol):
    def _contract(simulator):
        return all(
            np.allclose(simulator.get_wavefunction(circuit), wavefunction)
            for circuit, wavefunction in zip(
                _example_circuits, _corresponding_wavefunctions
            )
        )

    return _contract


def _verify_simulator_takes_into_account_initial_state_when_computing_wavefunction(
    simulator,
):
    return True


def _verify_computing_wavefunction_increases_number_of_jobs_and_circuits_executed(
    simulator,
):
    return True


def _verify_simulator_correctly_computes_bitstring_distribution(simulator):
    return True


def _verify_obtaining_bitstring_distribution_increases_number_of_jobs_and_circuits_executed(
    simulator,
):
    return True


def _verify_simulator_correctly_computes_expectation_values(simulator):
    return True


def _verify_computing_expectation_values_increases_number_of_jobs_and_circuits_executed(
    simulator,
):
    return True


def simulator_contracts_for_tolerance(atol=1e-7):
    return [
        _verify_wavefunction_returned_by_simulator_has_correct_length,
        _verify_wavefunction_returned_by_simulator_has_correct_coefficients(atol),
    ]
