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
from orquestra.quantum.circuits import CNOT, Circuit, H, X
from orquestra.quantum.operators import PauliTerm

_EXAMPLE_CIRCUITS = [
    Circuit([H(0)]),
    Circuit([H(0), CNOT(0, 1), CNOT(1, 2)]),
]

_CORRESPONDING_WAVEFUNCTIONS = [
    np.array([1, 1]) / np.sqrt(2),
    np.array([1, 0, 0, 0, 0, 0, 0, 1]) / np.sqrt(2),
]


_CORRESPONDING_OPERATORS = [PauliTerm("X0"), PauliTerm("X0 * Z1 * Y2")]


def _verify_wavefunction_returned_by_simulator_has_correct_length(
    simulator: WavefunctionSimulator,
):
    return all(
        len(simulator.get_wavefunction(circuit)) == 2**circuit.n_qubits
        for circuit in _EXAMPLE_CIRCUITS
    )


def _verify_wavefunction_returned_by_simulator_has_correct_coefficients(atol):
    def _contract(simulator):
        return all(
            np.allclose(simulator.get_wavefunction(circuit), wavefunction, atol=atol)
            for circuit, wavefunction in zip(
                _EXAMPLE_CIRCUITS, _CORRESPONDING_WAVEFUNCTIONS
            )
        )

    return _contract


def _verify_simulator_takes_into_account_initial_state_when_computing_wavefunction(
    atol,
):
    def _contract(simulator):
        circuit = Circuit([X(0)])
        return all(
            [
                np.allclose(
                    simulator.get_wavefunction(circuit, initial_state=np.array([0, 1])),
                    np.array([1, 0]),
                    atol=atol,
                ),
                np.allclose(
                    simulator.get_wavefunction(circuit, initial_state=np.array([1, 0])),
                    np.array([0, 1]),
                    atol=atol,
                ),
            ]
        )

    return _contract


def _verify_computing_wavefunction_increases_number_of_jobs_and_circuits_executed(
    simulator,
):
    n_jobs = [simulator.n_jobs_executed]
    n_circuits = [simulator.n_circuits_executed]

    for circuit in _EXAMPLE_CIRCUITS:
        simulator.get_wavefunction(circuit)
        n_jobs.append(simulator.n_jobs_executed)
        n_circuits.append(simulator.n_circuits_executed)

    return all(n_jobs[i] < n_jobs[i + 1] for i in range(len(n_jobs) - 1)) and all(
        n_circuits[i] < n_circuits[i + 1] for i in range(len(n_jobs) - 1)
    )


def _verify_computing_expectation_values_increases_number_of_jobs_and_circuits_executed(
    simulator,
):
    n_jobs = [simulator.n_jobs_executed]
    n_circuits = [simulator.n_circuits_executed]

    for circuit, operator in zip(_EXAMPLE_CIRCUITS, _CORRESPONDING_OPERATORS):
        simulator.get_exact_expectation_values(circuit, operator)
        n_jobs.append(simulator.n_jobs_executed)
        n_circuits.append(simulator.n_circuits_executed)

    return all(n_jobs[i] < n_jobs[i + 1] for i in range(len(n_jobs) - 1)) and all(
        n_circuits[i] < n_circuits[i + 1] for i in range(len(n_jobs) - 1)
    )


def simulator_contracts_for_tolerance(atol=1e-7):
    return [
        _verify_wavefunction_returned_by_simulator_has_correct_length,
        _verify_wavefunction_returned_by_simulator_has_correct_coefficients(atol),
        _verify_simulator_takes_into_account_initial_state_when_computing_wavefunction(
            atol
        ),
        _verify_computing_wavefunction_increases_number_of_jobs_and_circuits_executed,
        _verify_computing_expectation_values_increases_number_of_jobs_and_circuits_executed,  # noqa: E501
    ]
