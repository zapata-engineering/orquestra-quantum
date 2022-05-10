################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
import unittest

from orquestra.quantum.distributions import MeasurementOutcomeDistribution
from orquestra.quantum.distributions.BAS_dataset import (
    bars_and_stripes_zigzag,
    get_bars_and_stripes_target_distribution,
    get_num_bars_and_stripes_patterns,
)


class TestTarget(unittest.TestCase):
    def setUp(self):
        self.methods = ["zigzag"]
        self.grids = [[1, 1], [1, 2], [2, 1], [2, 2], [3, 2], [2, 3], [3, 3]]
        self.number_of_patterns = [2, 4, 4, 6, 10, 10, 14]
        self.fractions = [1.0, 0.75, 0.5, 0.25, 0.1]

    def test_get_num_bars_and_stripes_patterns(self):
        for i, grid in enumerate(self.grids):
            # When
            given_num_patterns = get_num_bars_and_stripes_patterns(grid[0], grid[1])

            # Then
            self.assertEqual(given_num_patterns, self.number_of_patterns[i])

    def test_get_bars_and_stripes_target_distribution_all_methods(self):
        for method in self.methods:
            for i, grid in enumerate(self.grids):
                for fraction in self.fractions:
                    # When
                    bars_and_stripes_distribution = (
                        get_bars_and_stripes_target_distribution(
                            grid[0], grid[1], fraction=fraction, method=method
                        )
                    )

                    # Then
                    self.assertIsInstance(
                        bars_and_stripes_distribution, MeasurementOutcomeDistribution
                    )
                    expected_num_patters = max(
                        int(self.number_of_patterns[i] * fraction), 1
                    )
                    self.assertEqual(
                        len(bars_and_stripes_distribution.distribution_dict.keys()),
                        expected_num_patters,
                    )

    def test_get_bars_and_stripes_target_distribution_unsupported_method(self):
        # Given
        method = "METHOD TYPE NOT SUPPORTED"
        nrows = 2
        ncols = 2

        # When/Then
        self.assertRaises(
            RuntimeError,
            lambda: get_bars_and_stripes_target_distribution(
                nrows, ncols, fraction=1.0, method=method
            ),
        )

    def test_bars_and_stripes_zigzag_2_by_2(self):
        # Given
        ncols = 2
        nrows = 2
        expected_patterns = ["0000", "0011", "1100", "1010", "0101", "1111"]

        # When
        pattern_list = bars_and_stripes_zigzag(nrows, ncols)

        # Then
        self.assertEqual(len(expected_patterns), len(pattern_list))
        for pattern in pattern_list:
            bitstring = ""
            for qubit in pattern:
                bitstring += str(qubit)
            self.assertIn(bitstring, expected_patterns)
