################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
import os

import numpy as np

from orquestra.quantum.measurements import (
    ExpectationValues,
    Parities,
    concatenate_expectation_values,
    expectation_values_to_real,
    get_expectation_values_from_parities,
    load_expectation_values,
    save_expectation_values,
)


def remove_file_if_exists(filename):
    try:
        os.remove(filename)
    except OSError:
        pass


def test_expectation_values_io():
    expectation_values = np.array([0.0, 0.0, -1.0])
    correlations = []
    correlations.append(np.array([[1.0, -1.0], [-1.0, 1.0]]))
    correlations.append(np.array([[1.0]]))

    estimator_covariances = []
    estimator_covariances.append(np.array([[0.1, -0.1], [-0.1, 0.1]]))
    estimator_covariances.append(np.array([[0.1]]))

    expectation_values_object = ExpectationValues(
        expectation_values, correlations, estimator_covariances
    )

    save_expectation_values(expectation_values_object, "expectation_values.json")
    expectation_values_object_loaded = load_expectation_values(
        "expectation_values.json"
    )

    assert np.allclose(
        expectation_values_object.values,
        expectation_values_object_loaded.values,
    )
    assert len(expectation_values_object.correlations) == len(
        expectation_values_object_loaded.correlations
    )
    assert len(expectation_values_object.estimator_covariances) == len(
        expectation_values_object_loaded.estimator_covariances
    )
    for i in range(len(expectation_values_object.correlations)):
        assert np.allclose(
            expectation_values_object.correlations[i],
            expectation_values_object_loaded.correlations[i],
        )
    for i in range(len(expectation_values_object.estimator_covariances)):
        assert np.allclose(
            expectation_values_object.estimator_covariances[i],
            expectation_values_object_loaded.estimator_covariances[i],
        )

    remove_file_if_exists("expectation_values.json")


def test_get_expectation_values_from_parities():
    parities = Parities(values=np.array([[18, 50], [120, 113], [75, 26]]))
    expectation_values = get_expectation_values_from_parities(parities)

    assert len(expectation_values.values) == 3
    assert np.isclose(expectation_values.values[0], -0.47058823529411764)
    assert np.isclose(expectation_values.values[1], 0.030042918454935622)
    assert np.isclose(expectation_values.values[2], 0.48514851485148514)

    assert len(expectation_values.estimator_covariances) == 3
    assert np.allclose(
        expectation_values.estimator_covariances[0],
        np.array([[0.014705882352941176]]),
    )
    assert np.allclose(
        expectation_values.estimator_covariances[1], np.array([[0.00428797]])
    )

    assert np.allclose(
        expectation_values.estimator_covariances[2], np.array([[0.0075706]])
    )


def test_expectation_values_to_real():
    # Given
    expectation_values = ExpectationValues(np.array([0.0 + 0.1j, 0.0 + 1e-10j, -1.0]))
    target_expectation_values = ExpectationValues(np.array([0.0, 0.0, -1.0]))

    # When
    real_expectation_values = expectation_values_to_real(expectation_values)

    # Then
    for value in expectation_values.values:
        assert not isinstance(value, complex)
    np.testing.assert_array_equal(
        real_expectation_values.values, target_expectation_values.values
    )


def test_concatenate_expectation_values():
    expectation_values_set = [
        ExpectationValues(np.array([1.0, 2.0])),
        ExpectationValues(np.array([3.0, 4.0])),
    ]

    combined_expectation_values = concatenate_expectation_values(expectation_values_set)
    assert combined_expectation_values.correlations is None
    assert combined_expectation_values.estimator_covariances is None
    assert np.allclose(combined_expectation_values.values, [1.0, 2.0, 3.0, 4.0])


def test_concatenate_expectation_values_with_cov_and_corr():
    expectation_values_set = [
        ExpectationValues(
            np.array([1.0, 2.0]),
            estimator_covariances=[np.array([[0.1, 0.2], [0.3, 0.4]])],
            correlations=[np.array([[-0.1, -0.2], [-0.3, -0.4]])],
        ),
        ExpectationValues(
            np.array([3.0, 4.0]),
            estimator_covariances=[np.array([[0.1]]), np.array([[0.2]])],
            correlations=[np.array([[-0.1]]), np.array([[-0.2]])],
        ),
    ]
    combined_expectation_values = concatenate_expectation_values(expectation_values_set)
    assert len(combined_expectation_values.estimator_covariances) == 3
    assert np.allclose(
        combined_expectation_values.estimator_covariances[0],
        [[0.1, 0.2], [0.3, 0.4]],
    )
    assert np.allclose(combined_expectation_values.estimator_covariances[1], [[0.1]])
    assert np.allclose(combined_expectation_values.estimator_covariances[2], [[0.2]])

    assert len(combined_expectation_values.correlations) == 3
    assert np.allclose(
        combined_expectation_values.correlations[0],
        [[-0.1, -0.2], [-0.3, -0.4]],
    )
    assert np.allclose(combined_expectation_values.correlations[1], [[-0.1]])
    assert np.allclose(combined_expectation_values.correlations[2], [[-0.2]])

    assert np.allclose(combined_expectation_values.values, [1.0, 2.0, 3.0, 4.0])
