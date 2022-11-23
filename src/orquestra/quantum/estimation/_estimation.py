################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
import sympy

from ..api.circuit_runner import CircuitRunner
from ..api.estimation import EstimationTask
from ..api.wavefunction_simulator import WavefunctionSimulator
from ..measurements import ExpectationValues, expectation_values_to_real


def evaluate_estimation_circuits(
    estimation_tasks: List[EstimationTask],
    symbols_maps: List[Dict[sympy.Symbol, float]],
) -> List[EstimationTask]:
    """Evaluates circuits given in all estimation tasks using the given symbols_maps.

    If one symbols map is given, it is used to evaluate all circuits. Otherwise, the
    symbols map at index i will be used for the estimation task at index i.

    Args:
        estimation_tasks: the estimation tasks which contain the circuits to be
            evaluated
        symbols_maps: a list of dictionaries (or singular dictionary) that map the
            symbolic symbols used in the parametrized circuits to the associated values
    """
    return [
        EstimationTask(
            operator=estimation_task.operator,
            circuit=estimation_task.circuit.bind(symbols_map),
            number_of_shots=estimation_task.number_of_shots,
        )
        for estimation_task, symbols_map in zip(estimation_tasks, symbols_maps)
    ]


def split_estimation_tasks_to_measure(
    estimation_tasks: List[EstimationTask],
) -> Tuple[List[EstimationTask], List[EstimationTask], List[int], List[int]]:
    """This function splits a given list of EstimationTask into two: one that
    contains EstimationTasks that should be measured, and one that contains
    EstimationTasks with constants or with 0 shots.

    Args:
        estimation_tasks: The list of estimation tasks for which
                         Expectation Values are wanted.

    Returns:
        estimation_tasks_to_measure: A new list of estimation tasks that only
            contains the ones that should actually be submitted to the runner
        estimation_tasks_not_to_measure: A new list of estimation tasks that
            contains the EstimationTasks with only constant terms or with
            0 shot
        indices_to_measure: A list containing the indices of the EstimationTasks we will
            actually measure, i.e. the ith estimation_tasks_to_measure expectation
            value will go into the indices_to_measure[i] position.
        indices_not_to_measure: A list containing the indices of the EstimationTasks for
            constant terms or with 0 shot.
    """

    estimation_tasks_to_measure = []
    estimation_tasks_not_to_measure = []
    indices_to_measure = []
    indices_not_to_measure = []
    for i, task in enumerate(estimation_tasks):
        if task.operator.is_constant or task.number_of_shots == 0:
            indices_not_to_measure.append(i)
            estimation_tasks_not_to_measure.append(task)
        else:
            indices_to_measure.append(i)
            estimation_tasks_to_measure.append(task)

    return (
        estimation_tasks_to_measure,
        estimation_tasks_not_to_measure,
        indices_to_measure,
        indices_not_to_measure,
    )


def evaluate_non_measured_estimation_tasks(
    estimation_tasks: List[EstimationTask],
) -> List[ExpectationValues]:
    """This function evaluates a list of EstimationTask that are not
    measured, and either contain only a constant term or require 0 shot.
    Non-constant EstimationTask with 0 shot return 0.0 as their
    ExpectationValue, with a precision of 0.0 as well.

    Args:
        estimation_tasks: The list of estimation tasks for which
            Expectation Values are wanted.

    Returns:
        expectation_values: the expectation values over non-measured terms,
            with their correlations and estimator_covariances.
    """

    expectation_values = []
    for task in estimation_tasks:
        coefficient: complex
        if task.operator.is_constant:
            coefficient = task.operator.terms[0].coefficient
        else:
            if task.number_of_shots is not None and task.number_of_shots > 0:
                raise RuntimeError(
                    "An EstimationTask required shots but was classified as "
                    "a non-measured task"
                )
            else:
                coefficient = 0.0

        expectation_values.append(
            ExpectationValues(
                np.asarray([coefficient]),
                correlations=[np.asarray([[0.0]])],
                estimator_covariances=[np.asarray([[0.0]])],
            )
        )

    return expectation_values


def estimate_expectation_values_by_averaging(
    runner: CircuitRunner,
    estimation_tasks: List[EstimationTask],
) -> List[ExpectationValues]:
    """Basic method for estimating expectation values for list of estimation tasks.

    It executes specified circuit and calculates expectation values based on the
    measurements.

    Args:
        runner: runner used for executing circuits
        estimation_tasks: list of estimation tasks
    """

    (
        estimation_tasks_to_measure,
        estimation_tasks_not_to_measure,
        indices_to_measure,
        indices_not_to_measure,
    ) = split_estimation_tasks_to_measure(estimation_tasks)

    non_measured_expectation_values_list = evaluate_non_measured_estimation_tasks(
        estimation_tasks_not_to_measure
    )

    if not estimation_tasks_to_measure:
        measured_expectation_values_list = []
    else:
        circuits, operators, shots_per_circuit = zip(
            *[
                (e.circuit, e.operator, e.number_of_shots)
                for e in estimation_tasks_to_measure
            ]
        )
        measurements_list = runner.run_batch_and_measure(circuits, shots_per_circuit)

        measured_expectation_values_list = [
            expectation_values_to_real(
                measurements.get_expectation_values(frame_operator)
            )
            for frame_operator, measurements in zip(operators, measurements_list)
        ]

    full_expectation_values: List[Optional[ExpectationValues]] = [
        None
        for _ in range(
            len(estimation_tasks_not_to_measure) + len(estimation_tasks_to_measure)
        )
    ]

    for ex_val, final_index in zip(
        non_measured_expectation_values_list, indices_not_to_measure
    ):
        full_expectation_values[final_index] = ex_val
    for ex_val, final_index in zip(
        measured_expectation_values_list, indices_to_measure
    ):
        full_expectation_values[final_index] = ex_val

    return cast(List[ExpectationValues], full_expectation_values)


def calculate_exact_expectation_values(
    runner: WavefunctionSimulator,
    estimation_tasks: List[EstimationTask],
) -> List[ExpectationValues]:
    """Calculates exact expectation values using built-in method of a provided runner.

    Args:
        runner: runner used for executing circuits
        estimation_tasks: list of estimation tasks
    """
    expectation_values_list = [
        runner.get_exact_expectation_values(
            estimation_task.circuit, estimation_task.operator
        )
        for estimation_task in estimation_tasks
    ]
    return [ExpectationValues(np.asarray([val])) for val in expectation_values_list]
