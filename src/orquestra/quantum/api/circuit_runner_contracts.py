################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
from functools import partial
from itertools import chain
from typing import List, Tuple

import numpy as np
import pytest

from orquestra.quantum.operators import PauliRepresentation, PauliSum

from ..circuits import CNOT, RX, RY, RZ, Circuit, H, builtin_gate_by_name
from ..estimation import estimate_expectation_values_by_averaging
from ..operators import PauliTerm
from ..testing.test_cases_for_backend_tests import (
    one_qubit_non_parametric_gates_exp_vals_test_set,
    one_qubit_parametric_gates_exp_vals_test_set,
    two_qubit_non_parametric_gates_exp_vals_test_set,
    two_qubit_parametric_gates_exp_vals_test_set,
)
from . import EstimationTask
from .circuit_runner import CircuitRunner

_EXAMPLE_CIRCUITS = (
    Circuit([RY(np.pi / 5)(0)]),
    Circuit([H(0), CNOT(0, 1)]),
    Circuit([H(0), CNOT(0, 2), H(1)]),
    Circuit([RX(np.pi / 2)(0), H(2), RZ(np.pi / 3)(1), CNOT(0, 3)]),
)

_EXAMPLE_N_SAMPLES = (5, 10, 15, 20)

_INVALID_N_SAMPLES = (0, -1, -10)


class _ValidateRunAndMeasure:
    """Contracts relating to run_and_measure."""

    @staticmethod
    def returns_number_of_measurements_greater_or_equal_to_n_samples(
        runner: CircuitRunner,
    ):
        circuit = _EXAMPLE_CIRCUITS[0]

        def _count_num_measurements(n_samples):
            return len(runner.run_and_measure(circuit, n_samples=n_samples).bitstrings)

        return all(
            _count_num_measurements(n_samples) >= n_samples
            for n_samples in _EXAMPLE_N_SAMPLES
        )

    @staticmethod
    def returns_bitstrings_with_length_equal_to_number_of_qubits_in_circuit(
        runner: CircuitRunner,
    ):
        return all(
            len(bitstring) == circuit.n_qubits
            for circuit in _EXAMPLE_CIRCUITS
            for bitstring in runner.run_and_measure(circuit, n_samples=10).bitstrings
        )

    @staticmethod
    def raises_value_error_if_n_samples_is_nonpositive(runner: CircuitRunner):
        for n_samples in _INVALID_N_SAMPLES:
            try:
                runner.run_and_measure(_EXAMPLE_CIRCUITS[0], n_samples=n_samples)
                return False
            except ValueError:
                pass
            except Exception:
                return False

        return True


class _ValidateRunBatchAndMeasure:
    @staticmethod
    def returns_measurement_object_for_each_circuit_in_batch(runner: CircuitRunner):
        return len(
            runner.run_batch_and_measure(_EXAMPLE_CIRCUITS, _EXAMPLE_N_SAMPLES)
        ) == len(_EXAMPLE_CIRCUITS)

    class ReturnsNumberOfMeasurementsEqualToNSamples:
        @staticmethod
        def when_n_samples_different_for_each_circuit(runner: CircuitRunner):
            return all(
                len(measurement.bitstrings) >= n_samples
                for measurement, n_samples in zip(
                    runner.run_batch_and_measure(_EXAMPLE_CIRCUITS, _EXAMPLE_N_SAMPLES),
                    _EXAMPLE_N_SAMPLES,
                )
            )

        @staticmethod
        def when_n_samples_same_for_each_circuit(runner: CircuitRunner):
            return all(
                len(measurements.bitstrings) >= 10
                for measurements in runner.run_batch_and_measure(
                    _EXAMPLE_CIRCUITS, n_samples=10
                )
            )

    @staticmethod
    def returns_bitstrings_with_length_equal_to_number_of_qubits_in_circuit(
        runner: CircuitRunner,
    ):
        return all(
            len(bitstring) == circuit.n_qubits
            for measurements, circuit in zip(
                runner.run_batch_and_measure(_EXAMPLE_CIRCUITS, _EXAMPLE_N_SAMPLES),
                _EXAMPLE_CIRCUITS,
            )
            for bitstring in measurements.bitstrings
        )

    @staticmethod
    def raises_value_error_if_n_samples_is_nonpositive(runner: CircuitRunner):
        def _for_all_circuits():
            try:
                runner.run_batch_and_measure(
                    _EXAMPLE_CIRCUITS, n_samples=_INVALID_N_SAMPLES
                )
                return False
            except ValueError:
                pass
            except Exception:
                return False

            return True

        def _for_at_least_one_circuit():
            n_samples = (-1,) + (len(_EXAMPLE_CIRCUITS) - 1) * (10,)
            try:
                runner.run_batch_and_measure(_EXAMPLE_CIRCUITS, n_samples=n_samples)
                return False
            except ValueError:
                pass
            except Exception:
                return False

            return True

        return _for_all_circuits() and _for_at_least_one_circuit()

    @staticmethod
    def raises_value_error_if_len_of_n_samples_does_not_match_len_of_batch(
        runner: CircuitRunner,
    ):
        invalid_n_samples = (
            (len(_EXAMPLE_CIRCUITS) + 1) * (10,),
            (len(_EXAMPLE_CIRCUITS) - 1) * (10,),
        )

        for n_samples in invalid_n_samples:
            try:
                runner.run_batch_and_measure(_EXAMPLE_CIRCUITS, n_samples)
                return False
            except ValueError:
                pass
            except Exception:
                return False

        return True


class _ValidateMeasurementOutcomeDistribution:
    @staticmethod
    def returns_distribution_with_number_of_bits_same_as_circuit_n_qubits(
        runner: CircuitRunner,
    ):
        return all(
            (
                runner.get_measurement_outcome_distribution(
                    circuit, n_samples=10
                ).get_number_of_subsystems()
                == circuit.n_qubits
            )
            for circuit in _EXAMPLE_CIRCUITS
        )

    @staticmethod
    def raises_value_error_if_n_samples_is_nonpositive(runner: CircuitRunner):
        for n_samples in _INVALID_N_SAMPLES:
            try:
                runner.get_measurement_outcome_distribution(
                    _EXAMPLE_CIRCUITS[0], n_samples=n_samples
                )
                return False
            except ValueError:
                pass
            except Exception:
                return False

        return True


def strict_runner_returns_number_of_measurements_equal_to_n_samples(
    runner: CircuitRunner,
):
    def _when_n_samples_is_the_same_for_each_circuit():
        return all(
            len(measurements.bitstrings) == 10
            for measurements in runner.run_batch_and_measure(
                _EXAMPLE_CIRCUITS, n_samples=10
            )
        )

    def _when_n_samples_is_different_for_each_circuit():
        return all(
            len(measurement.bitstrings) == n_samples
            for measurement, n_samples in zip(
                runner.run_batch_and_measure(_EXAMPLE_CIRCUITS, _EXAMPLE_N_SAMPLES),
                _EXAMPLE_N_SAMPLES,
            )
        )

    # We make sure to run each subtest, otherwise short-circuiting of
    # circuits might result in missing some errors
    subtests = [
        _when_n_samples_is_different_for_each_circuit(),
        _when_n_samples_is_the_same_for_each_circuit(),
    ]
    return all(subtests)


# TODO: remove this code when it is no longer needed
# This code is temporarily copied from orquestra-vqa to facilitate estimating
# expectation values of non-ising operators. It should be removed once we
# decide what to do with the gate-compatibility contracts.


def get_context_selection_circuit_for_group(
    qubit_operator: PauliRepresentation,
) -> Tuple[Circuit, PauliSum]:
    """Get the context selection circuit for measuring the expectation value
    of a group of co-measurable Pauli terms.
    Args:
        qubit_operator: operator representing group of co-measurable Pauli term
    """
    context_selection_circuit = Circuit()
    transformed_operator = PauliSum([])
    context: List[Tuple[str, int]] = []

    for term in qubit_operator.terms:
        term_operator = PauliTerm.identity()
        for qubit, operator in term.operations:
            for existing_qubit, existing_operator in context:
                if existing_qubit == qubit and existing_operator != operator:
                    raise ValueError("Terms are not co-measurable")
            if (operator, qubit) not in context:
                context.append((operator, qubit))
            product = term_operator * PauliTerm({qubit: "Z"})
            assert isinstance(product, PauliTerm)
            term_operator = product
        transformed_operator += term_operator * term.coefficient

    for factor in context:
        if factor[0] == "X":
            context_selection_circuit += RY(-np.pi / 2)(factor[1])
        elif factor[0] == "Y":
            context_selection_circuit += RX(np.pi / 2)(factor[1])

    return context_selection_circuit, transformed_operator


def perform_context_selection(
    estimation_tasks: List[EstimationTask],
) -> List[EstimationTask]:
    """Changes the circuits in estimation tasks to involve context selection.
    Args:
        estimation_tasks: list of estimation tasks
    """
    output_estimation_tasks = []
    for estimation_task in estimation_tasks:
        (
            context_selection_circuit,
            frame_operator,
        ) = get_context_selection_circuit_for_group(estimation_task.operator)
        frame_circuit = estimation_task.circuit + context_selection_circuit
        new_estimation_task = EstimationTask(
            frame_operator, frame_circuit, estimation_task.number_of_shots
        )
        output_estimation_tasks.append(new_estimation_task)
    return output_estimation_tasks


# ----- End of code copied from orquestra-vqa


def _verify_expectation_value_by_averaging(
    runner, circuit, operator, target_value, n_samples, exp_val_spread
):
    estimation_tasks = perform_context_selection(
        [EstimationTask(operator, circuit, n_samples)]
    )

    calculated_value = estimate_expectation_values_by_averaging(
        runner, estimation_tasks
    )

    sigma = 1 / np.sqrt(n_samples)

    return calculated_value[0].values[0] == pytest.approx(
        target_value, abs=exp_val_spread * sigma * 3
    )


def _one_qubit_nonparametric_gate_test_cases(gates_to_exclude):
    operators = [
        PauliTerm.identity(),
        PauliTerm("X0"),
        PauliTerm("Y0"),
        PauliTerm("Z0"),
    ]
    return [
        (
            Circuit(
                [
                    builtin_gate_by_name(initial_gate)(0),
                    builtin_gate_by_name(tested_gate)(0),
                ]
            ),
            operator,
            target_value,
        )
        for initial_gate, tested_gate, target_values in one_qubit_non_parametric_gates_exp_vals_test_set  # noqa: E501
        for operator, target_value in zip(operators, target_values)
        if tested_gate not in gates_to_exclude
    ]


def _one_qubit_parametric_gate_test_cases(gates_to_exclude):
    operators = [
        PauliTerm.identity(),
        PauliTerm("X0"),
        PauliTerm("Y0"),
        PauliTerm("Z0"),
    ]
    return [
        (
            Circuit(
                [
                    builtin_gate_by_name(initial_gate)(0),
                    builtin_gate_by_name(tested_gate)(*params)(0),
                ]
            ),
            operator,
            target_value,
        )
        for initial_gate, tested_gate, params, target_values in one_qubit_parametric_gates_exp_vals_test_set  # noqa: E501
        for operator, target_value in zip(operators, target_values)
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
            PauliTerm(operator),
            target_value,
        )
        for initial_gates, tested_gate, operators, target_values in two_qubit_non_parametric_gates_exp_vals_test_set  # noqa: E501
        for operator, target_value in zip(operators, target_values)
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
            PauliTerm(operator),
            target_value,
        )
        for initial_gates, tested_gate, operators, params, target_values in two_qubit_parametric_gates_exp_vals_test_set  # noqa: E501
        for operator, target_value in zip(operators, target_values)
        if tested_gate not in gates_to_exclude
    ]


CIRCUIT_RUNNER_CONTRACTS = [
    _ValidateRunAndMeasure.returns_number_of_measurements_greater_or_equal_to_n_samples,  # noqa: E501
    _ValidateRunAndMeasure.returns_bitstrings_with_length_equal_to_number_of_qubits_in_circuit,  # noqa: E501
    _ValidateRunAndMeasure.raises_value_error_if_n_samples_is_nonpositive,  # noqa: E501
    _ValidateRunBatchAndMeasure.ReturnsNumberOfMeasurementsEqualToNSamples.when_n_samples_different_for_each_circuit,  # noqa: E501
    _ValidateRunBatchAndMeasure.ReturnsNumberOfMeasurementsEqualToNSamples.when_n_samples_same_for_each_circuit,  # noqa: E501
    _ValidateRunBatchAndMeasure.returns_bitstrings_with_length_equal_to_number_of_qubits_in_circuit,  # noqa: E501
    _ValidateRunBatchAndMeasure.returns_measurement_object_for_each_circuit_in_batch,  # noqa: E501
    _ValidateRunBatchAndMeasure.raises_value_error_if_n_samples_is_nonpositive,  # noqa: E501
    _ValidateRunBatchAndMeasure.raises_value_error_if_len_of_n_samples_does_not_match_len_of_batch,  # noqa: E501
    _ValidateMeasurementOutcomeDistribution.returns_distribution_with_number_of_bits_same_as_circuit_n_qubits,  # noqa: E501
    _ValidateMeasurementOutcomeDistribution.raises_value_error_if_n_samples_is_nonpositive,  # noqa: E501
]


# This set of contracts is meant to be used for runners that return exactly the
# requested number of samples when running circuits. For context: some circuit
# runners return number of samples larger than the requested one, usually
# as a result of batching.
STRICT_CIRCUIT_RUNNER_CONTRACTS = [
    strict_runner_returns_number_of_measurements_equal_to_n_samples
]


def circuit_runner_gate_compatibility_contracts(
    exp_val_spread=1.0, gates_to_exclude=None
):
    gates_to_exclude = [] if gates_to_exclude is None else gates_to_exclude
    n_samples = 1000

    return [
        partial(
            _verify_expectation_value_by_averaging,
            circuit=circuit,
            operator=operator,
            target_value=target_value,
            n_samples=n_samples,
            exp_val_spread=exp_val_spread,
        )
        for circuit, operator, target_value in chain(
            _one_qubit_nonparametric_gate_test_cases(gates_to_exclude),
            _one_qubit_parametric_gate_test_cases(gates_to_exclude),
            _two_qubit_nonparametric_gate_test_cases(gates_to_exclude),
        )
    ]
