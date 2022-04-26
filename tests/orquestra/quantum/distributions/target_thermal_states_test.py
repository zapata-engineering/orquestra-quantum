################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
import random
import unittest

import numpy as np

from orquestra.quantum.distributions import compute_mmd
from orquestra.quantum.distributions.target_thermal_states import (
    _get_random_ising_hamiltonian_parameters,
    convert_integer_to_ising_bitstring,
    convert_ising_bitstring_to_integer,
    get_cardinality_distribution,
    get_thermal_sampled_distribution,
    get_thermal_target_measurement_outcome_distribution,
)
from orquestra.quantum.utils import dec2bin

SEED = 14943


class TestThermalTarget(unittest.TestCase):
    def test_convert_integer_to_ising_bitstring(self):
        # Given
        number = 4
        length = 5
        expected_ising_bitstring = [-1, -1, 1, -1, -1]

        # When
        ising_bitstring = convert_integer_to_ising_bitstring(number, length)

        # Then
        self.assertListEqual(ising_bitstring, expected_ising_bitstring)

        # When
        ising_bitstring_zero = convert_integer_to_ising_bitstring(0, length)

        # Then
        self.assertListEqual(ising_bitstring_zero, [-1, -1, -1, -1, -1])

    def test_convert_ising_bitstring_to_integer(self):
        # Given
        ising_bitstring = [-1, -1, 1, -1, -1]
        expected_number = 4
        zero_bitstring = [-1, -1, -1, -1, -1]

        # When
        number = convert_ising_bitstring_to_integer(ising_bitstring)

        # Then
        self.assertEqual(number, expected_number)

        # When
        number = convert_ising_bitstring_to_integer(zero_bitstring)

        # Then
        self.assertEqual(number, 0)

    def test_ising2int2ising(self):
        # Given
        expected_number = 14
        length = 5
        zero_number = 0
        zero_length = 0
        # When
        number = convert_ising_bitstring_to_integer(
            convert_integer_to_ising_bitstring(expected_number, length)
        )

        # Then
        self.assertEqual(number, expected_number)

        # When
        number = convert_ising_bitstring_to_integer(
            convert_integer_to_ising_bitstring(zero_number, length)
        )

        # Then
        self.assertEqual(number, zero_number)

        # When
        number = convert_ising_bitstring_to_integer(
            convert_integer_to_ising_bitstring(zero_number, zero_length)
        )

        # Then
        self.assertEqual(number, zero_number)

    def test_get_random_ising_hamiltonian_parameters(self):
        # Given
        np.random.seed(seed=SEED)
        n_spins = 15

        # When
        hamiltonian_parameters = _get_random_ising_hamiltonian_parameters(n_spins)

        # Then
        self.assertEqual(
            len(hamiltonian_parameters[0]),
            n_spins,
        )
        self.assertEqual(
            hamiltonian_parameters[1].shape,
            (n_spins, n_spins),
        )

    def test_get_thermal_target_distribution_dict(self):
        # Given
        n_spins = 5
        temperature = 1.0
        expected_bitstrings = [
            tuple(dec2bin(number, n_spins)) for number in range(int(2**n_spins))
        ]
        expected_keys = expected_bitstrings[len(expected_bitstrings) :: -1]
        np.random.seed(SEED)
        external_fields = np.random.rand(n_spins)
        two_body_couplings = np.random.rand(n_spins, n_spins)
        hamiltonian_parameters = [external_fields, two_body_couplings]

        # When
        target_distribution = get_thermal_target_measurement_outcome_distribution(
            n_spins, temperature, hamiltonian_parameters
        )

        # Then
        self.assertListEqual(
            sorted(list(target_distribution.distribution_dict.keys())),
            sorted(expected_keys),
        )

        self.assertAlmostEqual(
            sum(list(target_distribution.distribution_dict.values())),
            1.0,
        )

    def test_thermal_sampled_distribution(self):
        # Given
        n_samples = 5000
        n_spins = 2
        temperature = 0.85
        np.random.seed(SEED)  # needed by our samplers
        random.seed(SEED)  # needed to make lea reproducible

        external_fields = np.random.rand(n_spins)
        two_body_couplings = np.random.rand(n_spins, n_spins)
        hamiltonian_parameters = [external_fields, two_body_couplings]
        expected_bitstrings = [
            tuple(dec2bin(number, n_spins)) for number in range(int(2**n_spins))
        ]
        expected_keys = expected_bitstrings[len(expected_bitstrings) :: -1]

        # When
        sample_distribution = get_thermal_sampled_distribution(
            n_samples, n_spins, temperature, hamiltonian_parameters
        )

        # Then
        self.assertListEqual(
            sorted(list(sample_distribution.distribution_dict.keys())),
            sorted(expected_keys),
        )

        self.assertAlmostEqual(
            sum(list(sample_distribution.distribution_dict.values())), 1
        )

    def test_samples_from_distribution(self):
        # Given
        n_samples = 10000
        n_spins = 4
        temperature = 1.0
        distance_measure_parameters = {"sigma": 1.0}
        np.random.seed(SEED)  # needed by our samplers
        random.seed(SEED)  # needed to make lea reproducible

        external_fields = np.random.rand(n_spins)
        two_body_couplings = np.random.rand(n_spins, n_spins)
        hamiltonian_parameters = [external_fields, two_body_couplings]

        actual = get_thermal_target_measurement_outcome_distribution(
            n_spins, temperature, hamiltonian_parameters
        )
        model = get_thermal_sampled_distribution(
            n_samples, n_spins, temperature, hamiltonian_parameters
        )

        # When
        mmd = compute_mmd(actual, model, distance_measure_parameters)

        # Then
        self.assertLess(mmd, 1e-4)

    def test_cardinality_distribution(self):
        # Given
        n_samples = 10000
        n_spins = 4
        temperature = 1.0
        np.random.seed(SEED)  # needed by our samplers
        random.seed(SEED)  # needed to make lea reproducible

        external_fields = np.random.rand(n_spins)
        two_body_couplings = np.random.rand(n_spins, n_spins)
        hamiltonian_parameters = [external_fields, two_body_couplings]
        sampled_distribution = get_thermal_sampled_distribution(
            n_samples, n_spins, temperature, hamiltonian_parameters
        )

        # When
        cardinality_distr = get_cardinality_distribution(
            n_samples, n_spins, sampled_distribution
        )

        # Then
        assert all([spin_cardinality >= 0 for spin_cardinality in cardinality_distr])


if __name__ == "__main__":
    unittest.main()
