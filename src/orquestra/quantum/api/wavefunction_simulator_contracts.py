################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
from functools import partial
from itertools import chain

import numpy as np

from ..circuits import CNOT, Circuit, H, X, builtin_gate_by_name
from ..operators import PauliTerm
from ..testing.test_cases_for_backend_tests import (
    one_qubit_non_parametric_gates_amplitudes_test_set,
    one_qubit_parametric_gates_amplitudes_test_set,
    two_qubit_non_parametric_gates_amplitudes_test_set,
    two_qubit_parametric_gates_amplitudes_test_set,
)
from .wavefunction_simulator import WavefunctionSimulator

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


def _verify_gate_compatibility(simulator, circuit, target_amplitudes, atol):
    wavefunction = simulator.get_wavefunction(circuit)

    return np.allclose(wavefunction.amplitudes, target_amplitudes, atol=atol)


def _one_qubit_nonparametric_gate_test_cases(gates_to_exclude):
    return [
        (
            Circuit(
                [
                    builtin_gate_by_name(initial_gate)(0),
                    builtin_gate_by_name(tested_gate)(0),
                ]
            ),
            target_amplitudes,
        )
        for initial_gate, tested_gate, target_amplitudes in one_qubit_non_parametric_gates_amplitudes_test_set  # noqa: E501
        if tested_gate not in gates_to_exclude
    ]


def _two_qubit_nonparametric_gate_test_cases(gates_to_exclude):
    return [
        (
            Circuit(
                [
                    builtin_gate_by_name(initial_gates[0])(0),
                    builtin_gate_by_name(initial_gates[1])(1),
                    builtin_gate_by_name(tested_gate)(0, 1),
                ]
            ),
            target_amplitudes,
        )
        for initial_gates, tested_gate, target_amplitudes in two_qubit_non_parametric_gates_amplitudes_test_set  # noqa: E501
        if tested_gate not in gates_to_exclude
    ]


def _one_qubit_parametric_gate_test_cases(gates_to_exclude):
    return [
        (
            Circuit(
                [
                    builtin_gate_by_name(initial_gate)(0),
                    builtin_gate_by_name(tested_gate)(*params)(0),
                ]
            ),
            target_amplitudes,
        )
        for initial_gate, tested_gate, params, target_amplitudes in one_qubit_parametric_gates_amplitudes_test_set  # noqa: E501
        if tested_gate not in gates_to_exclude
    ]


def _two_qubit_parametric_gate_test_cases(gates_to_exclude):
    return [
        (
            Circuit(
                [
                    builtin_gate_by_name(initial_gates[0])(0),
                    builtin_gate_by_name(initial_gates[1])(1),
                    builtin_gate_by_name(tested_gate)(*params)(0, 1),
                ]
            ),
            target_amplitudes,
        )
        for initial_gates, tested_gate, params, target_amplitudes in two_qubit_parametric_gates_amplitudes_test_set  # noqa: E501
        if tested_gate not in gates_to_exclude
    ]


def simulator_contracts_for_tolerance(atol=1e-7):
    return [
        _verify_wavefunction_returned_by_simulator_has_correct_length,
        _verify_wavefunction_returned_by_simulator_has_correct_coefficients(atol),
        _verify_computing_wavefunction_increases_number_of_jobs_and_circuits_executed,
        _verify_computing_expectation_values_increases_number_of_jobs_and_circuits_executed,  # noqa: E501
    ]


def simulator_contracts_with_nontrivial_initial_state(atol=1e-7):
    return [
        _verify_simulator_takes_into_account_initial_state_when_computing_wavefunction(
            atol
        ),
    ]


def simulator_gate_compatibility_contracts(atol=1e-7, gates_to_exclude=None):
    gates_to_exclude = [] if gates_to_exclude is None else gates_to_exclude

    return [
        partial(
            _verify_gate_compatibility,
            circuit=circuit,
            target_amplitudes=target_amplitudes,
            atol=atol,
        )
        for circuit, target_amplitudes in chain(
            _one_qubit_nonparametric_gate_test_cases(gates_to_exclude),
            _two_qubit_nonparametric_gate_test_cases(gates_to_exclude),
            _one_qubit_parametric_gate_test_cases(gates_to_exclude),
        )
    ]
