################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
from typing import List

from orquestra.quantum.api.estimation import EstimationTask
from orquestra.quantum.api.estimator_contract import ESTIMATOR_CONTRACTS
from orquestra.quantum.api.wavefunction_simulator import WavefunctionSimulator
from orquestra.quantum.estimation import calculate_exact_expectation_values
from orquestra.quantum.measurements import ExpectationValues

# This function will be used as a mock estimator
_good_estimator = calculate_exact_expectation_values


def test_each_task_returns_one_expecation_value_test():
    def malicious_calculate_expectation_values(
        runner: WavefunctionSimulator, estimation_tasks: List[EstimationTask]
    ) -> List[ExpectationValues]:
        correct_output = calculate_exact_expectation_values(
            runner=runner, estimation_tasks=estimation_tasks
        )

        return correct_output[:-1]

    bad_estimator = malicious_calculate_expectation_values

    assert ESTIMATOR_CONTRACTS[0](_good_estimator)
    assert not ESTIMATOR_CONTRACTS[0](bad_estimator)


def test_order_of_outputs_matches_order_of_inputs_test():
    def malicious_calculate_expectation_values(
        runner: WavefunctionSimulator, estimation_tasks: List[EstimationTask]
    ) -> List[ExpectationValues]:
        correct_output = calculate_exact_expectation_values(
            runner=runner, estimation_tasks=estimation_tasks
        )

        correct_output[0], correct_output[-1] = correct_output[-1], correct_output[0]

        return correct_output

    bad_estimator = malicious_calculate_expectation_values

    assert ESTIMATOR_CONTRACTS[1](_good_estimator)
    assert not ESTIMATOR_CONTRACTS[1](bad_estimator)


def test_expectation_value_includes_coefficients_test():
    def malicious_calculate_expectation_values(
        runner: WavefunctionSimulator, estimation_tasks: List[EstimationTask]
    ) -> List[ExpectationValues]:
        correct_output = calculate_exact_expectation_values(
            runner=runner, estimation_tasks=estimation_tasks
        )

        correct_output[1] = correct_output[0]

        return correct_output

    bad_estimator = malicious_calculate_expectation_values

    assert ESTIMATOR_CONTRACTS[2](_good_estimator)
    assert not ESTIMATOR_CONTRACTS[2](bad_estimator)


def test_constant_terms_are_included_in_output_test():
    def malicious_calculate_expectation_values(
        runner: WavefunctionSimulator, estimation_tasks: List[EstimationTask]
    ) -> List[ExpectationValues]:
        correct_output = calculate_exact_expectation_values(
            runner=runner, estimation_tasks=estimation_tasks
        )

        correct_output[1] = correct_output[0]

        return correct_output

    bad_estimator = malicious_calculate_expectation_values

    assert ESTIMATOR_CONTRACTS[3](_good_estimator)
    assert not ESTIMATOR_CONTRACTS[3](bad_estimator)
