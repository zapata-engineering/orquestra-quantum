################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
from functools import partial
from itertools import cycle

import numpy as np
import pytest
import sympy

from orquestra.quantum.api.circuit_runner import BaseCircuitRunner
from orquestra.quantum.api.estimation import EstimationTask
from orquestra.quantum.circuits import RX, RY, RZ, Circuit, H, X
from orquestra.quantum.estimation import (
    calculate_exact_expectation_values,
    estimate_expectation_values_by_averaging,
    evaluate_estimation_circuits,
    evaluate_non_measured_estimation_tasks,
    split_estimation_tasks_to_measure,
)
from orquestra.quantum.measurements import ExpectationValues, Measurements
from orquestra.quantum.operators import PauliSum, PauliTerm
from orquestra.quantum.runners.symbolic_simulator import SymbolicSimulator
from orquestra.quantum.testing import MockCircuitRunner


class TestEstimatorUtils:
    @pytest.fixture()
    def frame_operators(self):
        operators = [
            PauliTerm("2*Z1*Z2"),
            PauliTerm("2*Z0*Z3"),
            PauliTerm("Z2", -1),
        ]

        return operators

    @pytest.fixture()
    def circuits(self):
        circuits = [Circuit() for _ in range(5)]

        circuits[1] += RX(1.2)(0)
        circuits[1] += RY(1.5)(1)
        circuits[1] += RX(-0.0002)(0)
        circuits[1] += RY(0)(1)

        for circuit in circuits[2:]:
            circuit += RX(sympy.Symbol("theta_0"))(0)
            circuit += RY(sympy.Symbol("theta_1"))(1)
            circuit += RX(sympy.Symbol("theta_2"))(0)
            circuit += RY(sympy.Symbol("theta_3"))(1)

        return circuits

    def test_evaluate_estimation_circuits_all_symbols(
        self,
        circuits,
    ):
        symbols_maps = [
            [
                (sympy.Symbol("theta_0"), 0),
                (sympy.Symbol("theta_1"), 0),
                (sympy.Symbol("theta_2"), 0),
                (sympy.Symbol("theta_3"), 0),
            ]
            for _ in circuits
        ]
        evaluate_circuits = partial(
            evaluate_estimation_circuits,
            symbols_maps=symbols_maps,
        )
        operator = PauliSum()
        estimation_tasks = [
            EstimationTask(operator, circuit, 1) for circuit in circuits
        ]

        new_estimation_tasks = evaluate_circuits(estimation_tasks)

        for new_task in new_estimation_tasks:
            assert len(new_task.circuit.free_symbols) == 0

    @pytest.mark.parametrize(
        ",".join(
            [
                "estimation_tasks",
                "ref_estimation_tasks_to_measure",
                "ref_non_measured_estimation_tasks",
                "ref_indices_to_measure",
                "ref_non_measured_indices",
            ]
        ),
        [
            (
                [
                    EstimationTask(
                        PauliSum([PauliTerm("Z0", 2), PauliTerm("3*Z1*Z2")]),
                        Circuit([X(0)]),
                        10,
                    ),
                    EstimationTask(
                        PauliSum(
                            [
                                PauliTerm("Z0", 2),
                                PauliTerm("3*Z1*Z2"),
                                PauliTerm("I0", 4),
                            ]
                        ),
                        Circuit([RZ(np.pi / 2)(0)]),
                        1000,
                    ),
                    EstimationTask(
                        PauliTerm("Z3", 4),
                        Circuit([RY(np.pi / 2)(0)]),
                        17,
                    ),
                ],
                [
                    EstimationTask(
                        PauliTerm("Z0", 2) + PauliTerm("3*Z1*Z2"),
                        Circuit([X(0)]),
                        10,
                    ),
                    EstimationTask(
                        PauliTerm("I0", 4) + PauliTerm("Z0", 2) + PauliTerm("3*Z1*Z2"),
                        Circuit([RZ(np.pi / 2)(0)]),
                        1000,
                    ),
                    EstimationTask(
                        PauliTerm("Z3", 4),
                        Circuit([RY(np.pi / 2)(0)]),
                        17,
                    ),
                ],
                [],
                [0, 1, 2],
                [],
            ),
            (
                [
                    EstimationTask(
                        PauliTerm("Z0", 2) + PauliTerm("3*Z1*Z2"),
                        Circuit([X(0)]),
                        10,
                    ),
                    EstimationTask(
                        PauliTerm("I0", 4),
                        Circuit([RZ(np.pi / 2)(0)]),
                        1000,
                    ),
                    EstimationTask(
                        PauliTerm("Z3", 4),
                        Circuit([RY(np.pi / 2)(0)]),
                        17,
                    ),
                ],
                [
                    EstimationTask(
                        PauliTerm("Z0", 2) + PauliTerm("3*Z1*Z2"),
                        Circuit([X(0)]),
                        10,
                    ),
                    EstimationTask(
                        PauliTerm("Z3", 4),
                        Circuit([RY(np.pi / 2)(0)]),
                        17,
                    ),
                ],
                [EstimationTask(PauliTerm("I0", 4), Circuit([RZ(np.pi / 2)(0)]), 1000)],
                [0, 2],
                [1],
            ),
            (
                [
                    EstimationTask(PauliTerm("I0", -3), Circuit([X(0)]), 0),
                    EstimationTask(
                        PauliTerm("I0", 4) + PauliTerm("Z0", 2) + PauliTerm("3*Z1*Z2"),
                        Circuit([RZ(np.pi / 2)(0)]),
                        1000,
                    ),
                    EstimationTask(
                        PauliTerm("Z3", 4),
                        Circuit([RY(np.pi / 2)(0)]),
                        17,
                    ),
                ],
                [
                    EstimationTask(
                        PauliTerm("I0", 4) + PauliTerm("Z0", 2) + PauliTerm("3*Z1*Z2"),
                        Circuit([RZ(np.pi / 2)(0)]),
                        1000,
                    ),
                    EstimationTask(
                        PauliTerm("Z3", 4),
                        Circuit([RY(np.pi / 2)(0)]),
                        17,
                    ),
                ],
                [
                    EstimationTask(PauliTerm("I0", -3), Circuit([X(0)]), 0),
                ],
                [1, 2],
                [0],
            ),
            (
                [
                    EstimationTask(PauliTerm("I0", -3), Circuit([X(0)]), 0),
                    EstimationTask(
                        PauliTerm("I0", 4) + PauliTerm("Z0", 2) + PauliTerm("3*Z1*Z2"),
                        Circuit([RZ(np.pi / 2)(0)]),
                        1000,
                    ),
                    EstimationTask(
                        PauliTerm("Z3", 4),
                        Circuit([RY(np.pi / 2)(0)]),
                        0,
                    ),
                ],
                [
                    EstimationTask(
                        PauliTerm("I0", 4) + PauliTerm("Z0", 2) + PauliTerm("3*Z1*Z2"),
                        Circuit([RZ(np.pi / 2)(0)]),
                        1000,
                    ),
                ],
                [
                    EstimationTask(PauliTerm("I0", -3), Circuit([X(0)]), 0),
                    EstimationTask(
                        PauliTerm("Z3", 4),
                        Circuit([RY(np.pi / 2)(0)]),
                        0,
                    ),
                ],
                [1],
                [0, 2],
            ),
        ],
    )
    def test_split_estimation_tasks_to_measure(
        self,
        estimation_tasks,
        ref_estimation_tasks_to_measure,
        ref_non_measured_estimation_tasks,
        ref_indices_to_measure,
        ref_non_measured_indices,
    ):

        (
            estimation_task_to_measure,
            non_measured_estimation_tasks,
            indices_to_measure,
            indices_for_non_measureds,
        ) = split_estimation_tasks_to_measure(estimation_tasks)

        assert estimation_task_to_measure == ref_estimation_tasks_to_measure
        assert non_measured_estimation_tasks == ref_non_measured_estimation_tasks
        assert indices_to_measure == ref_indices_to_measure
        assert ref_non_measured_indices == indices_for_non_measureds

    @pytest.mark.parametrize(
        "estimation_tasks,ref_expectation_values",
        [
            (
                [
                    EstimationTask(
                        PauliTerm("I0", 4),
                        Circuit([RZ(np.pi / 2)(0)]),
                        1000,
                    ),
                ],
                [
                    ExpectationValues(
                        np.asarray([4.0]),
                        correlations=[np.asarray([[0.0]])],
                        estimator_covariances=[np.asarray([[0.0]])],
                    ),
                ],
            ),
            (
                [
                    EstimationTask(
                        PauliTerm("I0", -0.5) + PauliTerm("I0", -2.5),
                        Circuit([X(0)]),
                        0,
                    ),
                    EstimationTask(
                        PauliTerm("I0", 0.001), Circuit([RZ(np.pi / 2)(0)]), 2
                    ),
                    EstimationTask(
                        PauliTerm("Z1", 2.5) + PauliTerm("1*Z2*Z3"),
                        Circuit([RY(np.pi / 2)(0)]),
                        0,
                    ),
                ],
                [
                    ExpectationValues(
                        np.asarray([-3.0]),
                        correlations=[np.asarray([[0.0]])],
                        estimator_covariances=[np.asarray([[0.0]])],
                    ),
                    ExpectationValues(
                        np.asarray([0.001]),
                        correlations=[np.asarray([[0.0]])],
                        estimator_covariances=[np.asarray([[0.0]])],
                    ),
                    ExpectationValues(
                        np.asarray([0.0]),
                        correlations=[np.asarray([[0.0]])],
                        estimator_covariances=[np.asarray([[0.0]])],
                    ),
                ],
            ),
        ],
    )
    def test_evaluate_non_measured_estimation_tasks(
        self, estimation_tasks, ref_expectation_values
    ):

        expectation_values = evaluate_non_measured_estimation_tasks(estimation_tasks)

        for ex_val, ref_ex_val in zip(expectation_values, ref_expectation_values):
            assert np.allclose(ex_val.values, ref_ex_val.values)
            assert np.allclose(ex_val.correlations, ref_ex_val.correlations)
            assert np.allclose(
                ex_val.estimator_covariances, ref_ex_val.estimator_covariances
            )

    @pytest.mark.parametrize(
        "estimation_tasks",
        [
            (
                [
                    EstimationTask(
                        PauliSum([PauliTerm("I0", -2.5), PauliTerm("Z1", -0.5)]),
                        Circuit([X(0)]),
                        1,
                    ),
                ]
            ),
            (
                [
                    EstimationTask(
                        PauliTerm("Z0", 0.001),
                        Circuit([RZ(np.pi / 2)(0)]),
                        0,
                    ),
                    EstimationTask(
                        PauliTerm("I0", 2.0), Circuit([RZ(np.pi / 2)(0)]), 2
                    ),
                    EstimationTask(
                        PauliTerm("1.5*Z0*Z1"),
                        Circuit([RY(np.pi / 2)(0)]),
                        10,
                    ),
                ]
            ),
        ],
    )
    def test_evaluate_non_measured_estimation_tasks_fails_with_non_zero_shots(
        self, estimation_tasks
    ):
        with pytest.raises(RuntimeError):
            _ = evaluate_non_measured_estimation_tasks(estimation_tasks)


TEST_CASES_EIGENSTATES = [
    (
        [
            EstimationTask(
                PauliTerm("Z0"), circuit=Circuit([X(0)]), number_of_shots=10
            ),
            EstimationTask(
                PauliTerm("I0", coefficient=2.0),
                circuit=Circuit([RY(np.pi / 4)(0)]),
                number_of_shots=30,
            ),
            # tests correlation
            EstimationTask(
                PauliSum([PauliTerm("Z0"), PauliTerm("Z1")]),
                circuit=Circuit([X(0), X(1)]),
                number_of_shots=10,
            ),
            # tests negative correlation
            EstimationTask(
                PauliSum([PauliTerm("Z0"), PauliTerm("Z1")]),
                circuit=Circuit([X(1)]),
                number_of_shots=10,
            ),
        ],
        [
            ExpectationValues(np.array([-1]), [np.array([[1]])], [np.array([[0]])]),
            ExpectationValues(np.array([2]), [np.array([[0]])], [np.array([[0]])]),
            ExpectationValues(
                np.array([-1, -1]),
                [np.array([[1, 1], [1, 1]])],
                [np.array([[0, 0], [0, 0]])],
            ),
            ExpectationValues(
                np.array([1, -1]),
                [np.array([[1, -1], [-1, 1]])],
                [np.array([[0, 0], [0, 0]])],
            ),
        ],
    ),
]
TEST_CASES_NONEIGENSTATES = [
    (
        [
            EstimationTask(
                PauliTerm("Z0"),
                circuit=Circuit([H(0)]),
                number_of_shots=10000,
            ),
            EstimationTask(
                PauliTerm("Z0", coefficient=-2),
                circuit=Circuit([RY(np.pi / 4)(0)]),
                number_of_shots=10000,
            ),
        ],
        [
            ExpectationValues(np.array([0]), [np.array([[1]])], [np.array([[0.001]])]),
            ExpectationValues(
                np.array([-2 * (np.cos(np.pi / 8) ** 2 - 0.5) * 2]),
                [np.array([[4]])],
                [np.array([[0.002]])],
            ),
        ],
    ),
]
""" When number_of_shots is high, the covariance will tend to zero. So we need extra
test cases to ensure covariance is being caluclated correctly.

Since we generate the data with MockBackendForTestingCovariancewhenNumberOfShotsIsLow
the estimation tasks can be arbitrary. They just need a valid circuit, number of shots
and have 2 terms in the ising operator."""
TEST_CASES_NONEIGENSTATES_WITH_LOW_NUMBER_OF_SHOTS = [
    (
        4
        * [
            EstimationTask(
                PauliSum([PauliTerm("Z0"), PauliTerm("Z1")]),
                circuit=Circuit([X(0), X(1)]),
                number_of_shots=10,
            ),
        ],
        [
            ExpectationValues(
                np.array([0]),
                [np.array([[1, 0], [0, 1]])],
                [np.array([[0, 0], [0, 0.5]])],
            ),
            ExpectationValues(
                np.array([0]),
                [np.array([[1, 0], [0, 1]])],
                [np.array([[0.5, 0], [0, 0]])],
            ),
            ExpectationValues(
                np.array([0]),
                [np.array([[1, -1], [-1, 1]])],
                [np.array([[0.5, -0.5], [-0.5, 0.5]])],
            ),
            ExpectationValues(
                np.array([0]),
                [np.array([[1, 1], [1, 1]])],
                [np.array([[0.05, 0.05], [0.05, 0.05]])],
            ),
        ],
    ),
]


# needs its own class otherwise issues arise with calling run_circuitset_and_measure.
class MockBackendForTestingCovariancewhenNumberOfShotsIsLow(BaseCircuitRunner):
    def __init__(self):
        super().__init__()
        self._measurements = iter(
            cycle(
                [
                    Measurements([(0, 1), (0, 0)]),
                    Measurements([(1, 0), (0, 0)]),
                    Measurements([(1, 0), (0, 1)]),
                    Measurements([(1, 1), (0, 0)] * 10),
                ]
            )
        )

    def _run_and_measure(self, circuit, shots_per_circuit):
        return next(self._measurements)


class TestBasicEstimationMethods:
    @pytest.fixture()
    def simulator(self):
        return SymbolicSimulator()

    @pytest.fixture()
    def estimation_tasks(self):
        task_1 = EstimationTask(
            PauliTerm("Z0"), circuit=Circuit([X(0)]), number_of_shots=10
        )
        task_2 = EstimationTask(
            PauliTerm("Z0"),
            circuit=Circuit([RY(np.pi / 2)(0)]),
            number_of_shots=20,
        )
        task_3 = EstimationTask(
            PauliTerm("I0", coefficient=2.0),
            circuit=Circuit([RY(np.pi / 4)(0)]),
            number_of_shots=30,
        )
        return [task_1, task_2, task_3]

    @pytest.fixture()
    def target_expectation_values(self):
        return [ExpectationValues(-1), ExpectationValues(0), ExpectationValues(2)]

    @pytest.mark.parametrize(
        "estimation_tasks,target_expectations", TEST_CASES_EIGENSTATES
    )
    def test_estimate_expectation_values_by_averaging_for_eigenstates(
        self, simulator, estimation_tasks, target_expectations
    ):
        expectation_values_list = estimate_expectation_values_by_averaging(
            simulator, estimation_tasks
        )
        for expectation_values, target, task in zip(
            expectation_values_list, target_expectations, estimation_tasks
        ):
            assert len(expectation_values.values) == len(task.operator.terms)
            np.testing.assert_array_equal(expectation_values.values, target.values)

    @pytest.mark.parametrize(
        "estimation_tasks,target_expectations", TEST_CASES_NONEIGENSTATES
    )
    def test_estimate_expectation_values_by_averaging_for_non_eigenstates(
        self, simulator, estimation_tasks, target_expectations
    ):

        expectation_values_list = estimate_expectation_values_by_averaging(
            simulator, estimation_tasks
        )
        for expectation_values, target, task in zip(
            expectation_values_list, target_expectations, estimation_tasks
        ):
            assert len(expectation_values.values) == len(task.operator.terms)
            np.testing.assert_allclose(
                expectation_values.values, target.values, atol=0.1
            )

    @pytest.mark.skip(reason="Temporary disabled")
    @pytest.mark.parametrize(
        "estimation_tasks,target_expectations",
        TEST_CASES_EIGENSTATES + TEST_CASES_NONEIGENSTATES,
    )
    def test_calculate_exact_expectation_values(
        self, simulator, estimation_tasks, target_expectations
    ):
        expectation_values_list = calculate_exact_expectation_values(
            simulator, estimation_tasks
        )
        for expectation_values, target, task in zip(
            expectation_values_list, target_expectations, estimation_tasks
        ):
            assert len(expectation_values.values) == len(task.operator.terms)
            np.testing.assert_array_almost_equal(
                expectation_values.values, target.values
            )

    @pytest.mark.parametrize(
        "estimation_tasks,target_expectations", TEST_CASES_EIGENSTATES
    )
    def test_covariance_and_correlations_when_averaging_for_eigenstates(
        self, simulator, estimation_tasks, target_expectations
    ):
        expectation_values_list = estimate_expectation_values_by_averaging(
            simulator, estimation_tasks
        )
        for expectation_values, target, task in zip(
            expectation_values_list, target_expectations, estimation_tasks
        ):
            for cov, corr, target_corr, target_cov in zip(
                expectation_values.estimator_covariances,
                expectation_values.correlations,
                target.correlations,
                target.estimator_covariances,
            ):
                assert len(corr) == len(task.operator.terms)
                assert len(cov) == len(task.operator.terms)

                np.testing.assert_array_almost_equal(corr, target_corr, decimal=2)
                np.testing.assert_array_almost_equal(cov, target_cov, decimal=2)

    @pytest.mark.parametrize(
        "estimation_tasks,target_expectations",
        TEST_CASES_NONEIGENSTATES,
    )
    def test_correlation_and_covariance_when_averaging_for_non_eigenstates(
        self, simulator, estimation_tasks, target_expectations
    ):

        expectation_values_list = estimate_expectation_values_by_averaging(
            simulator, estimation_tasks
        )
        for expectation_values, target, task in zip(
            expectation_values_list, target_expectations, estimation_tasks
        ):
            for cov, corr, target_corr, target_cov in zip(
                expectation_values.estimator_covariances,
                expectation_values.correlations,
                target.correlations,
                target.estimator_covariances,
            ):
                assert len(corr) == len(task.operator.terms)
                assert len(cov) == len(task.operator.terms)

                np.testing.assert_array_almost_equal(corr, target_corr, decimal=1)
                # All covariances should be close to 0 when number_of_shots is high.
                np.testing.assert_array_almost_equal(target_cov, cov, decimal=2)

    @pytest.mark.parametrize(
        "mock_estimation_tasks, target_expectations",
        TEST_CASES_NONEIGENSTATES_WITH_LOW_NUMBER_OF_SHOTS,
    )
    def test_covariance_when_averaging_for_non_eigenstates_and_number_of_shots_is_low(
        self, mock_estimation_tasks, target_expectations
    ):

        expectation_values_list = estimate_expectation_values_by_averaging(
            MockBackendForTestingCovariancewhenNumberOfShotsIsLow(),
            mock_estimation_tasks,
        )
        for expectation_values, target in zip(
            expectation_values_list, target_expectations
        ):
            np.testing.assert_array_almost_equal(
                expectation_values.correlations,
                target.correlations,
                decimal=2,
            )
            np.testing.assert_array_almost_equal(
                expectation_values.estimator_covariances,
                target.estimator_covariances,
                decimal=2,
            )

    def test_calculate_exact_expectation_values_fails_with_non_simulator(
        self, estimation_tasks
    ):
        backend = MockCircuitRunner()
        with pytest.raises(AttributeError):
            _ = calculate_exact_expectation_values(backend, estimation_tasks)

    def test_no_measured_estimation_tasks_provided(simulator):
        estimation_tasks = [
            EstimationTask(PauliTerm("Z0"), Circuit([H(0)]), 0),
        ]

        target = [
            ExpectationValues(np.array([0.0]), [np.array([[0.0]])], [np.array([0.0])])
        ]

        expectation_values_list = estimate_expectation_values_by_averaging(
            simulator, estimation_tasks
        )

        assert len(expectation_values_list) == 1
        assert expectation_values_list[0] == target[0]
