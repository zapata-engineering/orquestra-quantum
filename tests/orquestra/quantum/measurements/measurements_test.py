################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
import json
import os
import random
from collections import Counter

import numpy as np
import pytest

from orquestra.quantum.distributions import MeasurementOutcomeDistribution
from orquestra.quantum.measurements import (
    Measurements,
    _check_sample_elimination,
    convert_bitstring_to_int,
    get_expectation_value_from_frequencies,
)
from orquestra.quantum.operators import PauliTerm
from orquestra.quantum.utils import (
    RNDSEED,
    convert_tuples_to_bitstrings,
    get_ordered_list_of_bitstrings,
)


def remove_file_if_exists(filename):
    try:
        os.remove(filename)
    except OSError:
        pass


def test_convert_bitstring_to_int():
    bitstring = (0, 1, 0, 1, 0, 1)
    assert convert_bitstring_to_int(bitstring) == 42


def test_get_expectation_value_from_frequencies():
    bitstrings = ["001", "001", "110", "000"]
    bitstring_frequencies = dict(Counter(bitstrings))
    marked_qubits = (1, 2)
    assert np.isclose(
        get_expectation_value_from_frequencies(marked_qubits, bitstring_frequencies),
        -0.5,
    )


class TestMeasurements:
    @pytest.fixture
    def bitstrings(self):
        return [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1],
            [1, 0, 1],
            [0, 0, 1],
        ]

    @pytest.fixture
    def counts(self):
        return {
            key: value
            for key, value in zip(
                get_ordered_list_of_bitstrings(3), [1, 2, 1, 1, 1, 1, 1, 1]
            )
        }

    @pytest.fixture
    def measurements_data(self, counts, bitstrings):
        return {
            "counts": counts,
            "bitstrings": bitstrings,
        }

    def test_io(self, measurements_data):
        # Given
        input_filename = "measurements_input_test.json"
        output_filename = "measurements_output_test.json"

        with open(input_filename, "w") as f:
            f.write(json.dumps(measurements_data, indent=2))

        # When
        measurements = Measurements.load_from_file(input_filename)
        measurements.save(output_filename)

        # Then
        with open(output_filename, "r") as f:
            output_data = json.load(f)
        assert measurements_data == output_data

        remove_file_if_exists(input_filename)
        remove_file_if_exists(output_filename)

    def test_save_for_numpy_integers(self):
        # Given
        target_bitstrings = [(0, 0, 0)]
        input_bitstrings = [(np.int8(0), np.int8(0), np.int8(0))]

        filename = "measurementstest.json"
        measurements = Measurements(input_bitstrings)
        target_measurements = Measurements(target_bitstrings)

        # When
        measurements.save(filename)

        # Then
        recreated_measurements = Measurements.load_from_file(filename)
        assert target_measurements.bitstrings == recreated_measurements.bitstrings
        remove_file_if_exists("measurementstest.json")

    def test_intialize_with_bitstrings(self):
        # Given
        bitstrings = [
            (0, 0, 0),
            (0, 0, 1),
            (0, 0, 1),
            (0, 1, 0),
            (0, 1, 1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, 0),
            (1, 1, 1),
        ]

        # When
        measurements = Measurements(bitstrings=bitstrings)

        # Then
        assert measurements.bitstrings == [
            (0, 0, 0),
            (0, 0, 1),
            (0, 0, 1),
            (0, 1, 0),
            (0, 1, 1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, 0),
            (1, 1, 1),
        ]

    def test_intialize_with_counts(self, counts):
        # When
        measurements = Measurements.from_counts(counts)

        # Then
        assert measurements.bitstrings == [
            (0, 0, 0),
            (0, 0, 1),
            (0, 0, 1),
            (0, 1, 0),
            (0, 1, 1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, 0),
            (1, 1, 1),
        ]

    def test_bitstrings(self, measurements_data):
        input_filename = "measurements_input_test.json"

        with open(input_filename, "w") as f:
            f.write(json.dumps(measurements_data, indent=2))
        measurements = Measurements.load_from_file(input_filename)

        # When/Then
        assert measurements.bitstrings == [
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 0),
            (0, 1, 1),
            (1, 0, 0),
            (1, 1, 0),
            (1, 1, 1),
            (1, 0, 1),
            (0, 0, 1),
        ]

        remove_file_if_exists(input_filename)

    def test_get_counts(self, measurements_data):
        # Given
        input_filename = "measurements_input_test.json"

        with open(input_filename, "w") as f:
            f.write(json.dumps(measurements_data, indent=2))
        measurements = Measurements.load_from_file(input_filename)

        # When
        counts = measurements.get_counts()

        # Then
        assert measurements_data["counts"] == counts

        remove_file_if_exists(input_filename)

    def test_get_distribution(self, measurements_data):
        # Given
        input_filename = "measurements_input_test.json"

        with open(input_filename, "w") as f:
            f.write(json.dumps(measurements_data, indent=2))
        measurements = Measurements.load_from_file(input_filename)

        # When
        distribution = measurements.get_distribution()

        # Then
        assert distribution.distribution_dict == {
            (0, 0, 0): 1 / 9,
            (0, 0, 1): 2 / 9,
            (0, 1, 0): 1 / 9,
            (0, 1, 1): 1 / 9,
            (1, 0, 0): 1 / 9,
            (1, 0, 1): 1 / 9,
            (1, 1, 0): 1 / 9,
            (1, 1, 1): 1 / 9,
        }

        remove_file_if_exists(input_filename)

    def test_add_counts(self, counts):
        # Given
        measurements = Measurements()
        measurements_counts = counts

        # When
        measurements.add_counts(measurements_counts)

        # Then
        assert measurements.bitstrings == [
            (0, 0, 0),
            (0, 0, 1),
            (0, 0, 1),
            (0, 1, 0),
            (0, 1, 1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, 0),
            (1, 1, 1),
        ]
        assert measurements.get_counts() == counts

    def test_add_measurements(self):
        # Given
        measurements = Measurements()
        bitstrings = [
            (0, 0, 0),
            (0, 0, 1),
            (0, 0, 1),
            (0, 1, 0),
            (0, 1, 1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, 0),
            (1, 1, 1),
        ]

        # When
        measurements.bitstrings = bitstrings

        # Then
        assert measurements.bitstrings == [
            (0, 0, 0),
            (0, 0, 1),
            (0, 0, 1),
            (0, 1, 0),
            (0, 1, 1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, 0),
            (1, 1, 1),
        ]

        # When
        measurements.bitstrings += bitstrings

        # Then
        assert measurements.bitstrings == [
            (0, 0, 0),
            (0, 0, 1),
            (0, 0, 1),
            (0, 1, 0),
            (0, 1, 1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, 0),
            (1, 1, 1),
            (0, 0, 0),
            (0, 0, 1),
            (0, 0, 1),
            (0, 1, 0),
            (0, 1, 1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, 0),
            (1, 1, 1),
        ]

    def test_get_expectation_values_from_measurements(self):
        # Given
        measurements = Measurements(
            [(0, 1, 0), (0, 1, 0), (0, 0, 0), (1, 0, 0), (1, 1, 1)]
        )
        ising_operator = (
            PauliTerm("I0", 10) + PauliTerm("1*Z0*Z1") - PauliTerm("15*Z1*Z2")
        )
        target_expectation_values = np.array([10, -0.2, -3])
        target_correlations = np.array([[100, -2, -30], [-2, 1, -9], [-30, -9, 225]])
        denominator = len(measurements.bitstrings)
        covariance_11 = (
            target_correlations[1, 1] - target_expectation_values[1] ** 2
        ) / denominator
        covariance_12 = (
            target_correlations[1, 2]
            - target_expectation_values[1] * target_expectation_values[2]
        ) / denominator
        covariance_22 = (
            target_correlations[2, 2] - target_expectation_values[2] ** 2
        ) / denominator

        target_covariances = np.array(
            [
                [0, 0, 0],
                [0, covariance_11, covariance_12],
                [0, covariance_12, covariance_22],
            ]
        )

        # When
        expectation_values = measurements.get_expectation_values(ising_operator, False)
        # Then
        np.testing.assert_allclose(expectation_values.values, target_expectation_values)
        assert len(expectation_values.correlations) == 1
        np.testing.assert_allclose(
            expectation_values.correlations[0], target_correlations
        )
        assert len(expectation_values.estimator_covariances) == 1
        np.testing.assert_allclose(
            expectation_values.estimator_covariances[0], target_covariances
        )

    def test_get_expectation_values_from_measurements_with_bessel_correction(self):
        # Given
        measurements = Measurements(
            [(0, 1, 0), (0, 1, 0), (0, 0, 0), (1, 0, 0), (1, 1, 1)]
        )
        ising_operator = (
            PauliTerm("I0", 10) + PauliTerm("1*Z0*Z1") - PauliTerm("15*Z1*Z2")
        )
        target_expectation_values = np.array([10, -0.2, -3])
        target_correlations = np.array([[100, -2, -30], [-2, 1, -9], [-30, -9, 225]])
        denominator = len(measurements.bitstrings) - 1
        covariance_11 = (
            target_correlations[1, 1] - target_expectation_values[1] ** 2
        ) / denominator
        covariance_12 = (
            target_correlations[1, 2]
            - target_expectation_values[1] * target_expectation_values[2]
        ) / denominator
        covariance_22 = (
            target_correlations[2, 2] - target_expectation_values[2] ** 2
        ) / denominator

        target_covariances = np.array(
            [
                [0, 0, 0],
                [0, covariance_11, covariance_12],
                [0, covariance_12, covariance_22],
            ]
        )

        # When
        expectation_values = measurements.get_expectation_values(ising_operator, True)
        # Then
        np.testing.assert_allclose(expectation_values.values, target_expectation_values)
        assert len(expectation_values.correlations) == 1
        np.testing.assert_allclose(
            expectation_values.correlations[0], target_correlations
        )
        assert len(expectation_values.estimator_covariances) == 1
        np.testing.assert_allclose(
            expectation_values.estimator_covariances[0], target_covariances
        )

    @pytest.mark.parametrize(
        "bitstring_distribution, number_of_samples",
        [
            (MeasurementOutcomeDistribution({"00": 0.5, "11": 0.5}), 1),
            (MeasurementOutcomeDistribution({"00": 0.5, "11": 0.5}), 10),
            (MeasurementOutcomeDistribution({"00": 0.5, "11": 0.5}), 51),
            (MeasurementOutcomeDistribution({"00": 0.5, "11": 0.5}), 137),
            (MeasurementOutcomeDistribution({"00": 0.5, "11": 0.5}), 5000),
            (MeasurementOutcomeDistribution({"0000": 0.137, "0001": 0.863}), 100),
            (
                MeasurementOutcomeDistribution(
                    {"00": 0.1234, "01": 0.5467, "10": 0.0023, "11": 0.3276}
                ),
                100,
            ),
            (
                MeasurementOutcomeDistribution(
                    {
                        "0000": 0.06835580857498666,
                        "1000": 0.060975627112613416,
                        "0100": 0.05976605586194627,
                        "1100": 0.07138587439957303,
                        "0010": 0.06474168297455969,
                        "1010": 0.0825036470378936,
                        "0110": 0.09861252446183953,
                        "1110": 0.0503013698630137,
                        "0001": 0.04496424123821384,
                        "1001": 0.07317221135029355,
                        "0101": 0.08171161714997331,
                        "1101": 0.03753940579967977,
                        "0011": 0.05157676570005337,
                        "1011": 0.05,
                        "0111": 0.04964419142501335,
                        "1111": 0.05474897705034692,
                    }
                ),
                5621,
            ),
        ],
    )
    def test_get_measurements_representing_distribution_returns_right_number_of_samples(
        self, bitstring_distribution, number_of_samples
    ):
        measurements = Measurements.get_measurements_representing_distribution(
            bitstring_distribution, number_of_samples
        )
        assert len(measurements.bitstrings) == number_of_samples

    @pytest.mark.parametrize(
        "bitstring_distribution, number_of_samples, expected_counts",
        [
            (
                MeasurementOutcomeDistribution(
                    {"01": 0.3333333, "11": (1 - 0.3333333)}
                ),
                3,
                {"01": 1, "11": 2},
            ),
            (
                MeasurementOutcomeDistribution(
                    {"01": 0.9999999, "11": (1 - 0.9999999)}
                ),
                1,
                {"01": 1},
            ),
        ],
    )
    def test_get_measurements_samples_correctly_leftover_bitstrings(
        self, bitstring_distribution, number_of_samples, expected_counts
    ):
        random.seed(RNDSEED)
        measurements = Measurements.get_measurements_representing_distribution(
            bitstring_distribution, number_of_samples
        )
        assert measurements.get_counts() == expected_counts

    def test_get_measurements_samples_correctly_leftover_bitstrings_for_equal_probas(
        self,
    ):
        random.seed(RNDSEED)
        bitstring_distribution = MeasurementOutcomeDistribution({"00": 0.5, "11": 0.5})
        number_of_samples = 51
        max_number_of_trials = 10
        got_different_measurements = False
        previous_measurements = Measurements.get_measurements_representing_distribution(
            bitstring_distribution, number_of_samples
        )

        while not got_different_measurements:
            measurements = Measurements.get_measurements_representing_distribution(
                bitstring_distribution, number_of_samples
            )

            assert measurements.get_counts() == {
                "00": 25,
                "11": 26,
            } or measurements.get_counts() == {"00": 26, "11": 25}

            if measurements.get_counts() != previous_measurements.get_counts():
                got_different_measurements = True

            max_number_of_trials -= 1
            if max_number_of_trials == 0:
                break
        assert got_different_measurements

    @pytest.mark.parametrize(
        "bitstring_distribution",
        [
            MeasurementOutcomeDistribution({"00": 0.5, "11": 0.5}),
            MeasurementOutcomeDistribution({"000": 0.5, "101": 0.5}),
            MeasurementOutcomeDistribution({"0000": 0.137, "0001": 0.863}),
            MeasurementOutcomeDistribution(
                {"00": 0.1234, "01": 0.5467, "10": 0.0023, "11": 0.3276}
            ),
        ],
    )
    def test_get_measurements_representing_distribution_gives_exactly_right_counts(
        self, bitstring_distribution
    ):
        number_of_samples = 10000
        measurements = Measurements.get_measurements_representing_distribution(
            bitstring_distribution, number_of_samples
        )

        counts = measurements.get_counts()
        for bitstring, probability in bitstring_distribution.distribution_dict.items():
            assert (
                probability * number_of_samples
                == counts[convert_tuples_to_bitstrings([bitstring])[0]]
            )

    @pytest.mark.parametrize(
        "bitstring_distribution",
        [
            MeasurementOutcomeDistribution(
                {
                    "0011": 0.0049,
                    "1100": 0.0049,
                    "1111": 0.008,
                    "0000": 0.008,
                    "0001": 0.008,
                    "0010": 0.008,
                    "0100": 0.008,
                    "1000": 0.008,
                    "1001": 0.008,
                    "1010": 0.008,
                    "1110": 0.008,
                    "1101": 0.008,
                    "0110": 0.9102,
                }
            )
        ],
    )
    def test_get_measurements_representing_distribution_doesnt_raise(
        self, bitstring_distribution
    ):
        number_of_samples = 100
        max_number_of_trials = 100
        for _ in range(max_number_of_trials):
            _ = Measurements.get_measurements_representing_distribution(
                bitstring_distribution, number_of_samples
            )

    @pytest.mark.parametrize(
        "samples, bitstring_samples, leftover_distribution",
        [
            (
                Counter({"0011": 3, "1100": 1, "0101": 2}),
                [
                    (0, 0, 1, 1),
                    (1, 1, 0, 0),
                    (0, 1, 0, 1),
                    (0, 1, 0, 1),
                    (0, 1, 0, 1),
                    (1, 1, 0, 0),
                    (0, 0, 1, 1),
                ],
                MeasurementOutcomeDistribution({"0011": 0.3, "1100": 0.3, "0101": 0.3}),
            ),
            (
                Counter({"0011": 3, "1100": 1, "0101": 2, "0001": 1}),
                [
                    (0, 0, 1, 1),
                    (1, 1, 0, 0),
                    (0, 1, 0, 1),
                    (0, 1, 0, 1),
                    (0, 1, 0, 1),
                    (1, 1, 0, 0),
                    (0, 0, 1, 1),
                ],
                MeasurementOutcomeDistribution(
                    {"0011": 0.3, "1100": 0.001, "0101": 0.001, "0001": 0.698}
                ),
            ),
        ],
    )
    def test_check_sample_elimination(
        self, samples, bitstring_samples, leftover_distribution
    ):
        correct_samples = _check_sample_elimination(
            samples, bitstring_samples, leftover_distribution
        )

        bitstring_counts = Counter(bitstring_samples)
        for sample in correct_samples:
            bitstring = tuple([int(measurement_value) for measurement_value in sample])
            assert correct_samples[sample] <= bitstring_counts[bitstring]
