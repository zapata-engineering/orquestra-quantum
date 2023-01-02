################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
from typing import Dict, List, Tuple, Union

import numpy as np

from ..distributions import MeasurementOutcomeDistribution
from ..utils import (
    bin2dec,
    dec2bin,
    sample_from_probability_distribution,
    tuple_to_bitstring,
)


def convert_integer_to_ising_bitstring(number: int, length: int) -> List[int]:
    """Converts an integer into a +/-1s bitstring (also called Ising bitstring).

    Args:
        number: positive number to be converted into its corresponding
            Ising bitstring representation
        length: length of the Ising bitstring (i.e. positive number of spins
            in the Ising system)
    Returns:
        Ising bitstring representation (1D array of +/-1)."""

    if length < 0:
        raise ValueError("Length cannot be negative.")
    binary_bitstring = dec2bin(number, length)
    ising_bitstring = [bit * 2 - 1 for bit in binary_bitstring]
    return ising_bitstring


def convert_ising_bitstring_to_integer(ising_bitstring: List[int]) -> int:
    """Converts a +/-1s bitstring (also called Ising bitstring) into an integer.
    Args:
        ising_bitstring: 1D array of +/-1.
    Returns:
        Integer number representation of the Ising bitstring."""

    binary_bitstring = [int((bit + 1) / 2) for bit in ising_bitstring]
    number = bin2dec(binary_bitstring)
    return number


def _get_random_ising_hamiltonian_parameters(
    n_spins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates random h, J, and where h and J are arrays of random coefficients
    sampled from a normal distribution with zero mean and sqrt(n_spins) sd.
    For reproducibilty, fix random generator seed in the higher level from which
    this function is called.
    Useful in the following Ising Hamiltonian: 1D Nearest Neighbor Ising Model with
    open boundary conditions, where h's are the external field values and J's
    the two-body couplings coefficients.
    Args:
        n_spins: positive number of spins in the Ising system.
    Returns:
       external_fields: n_spin coefficients sampled from a normal distribution with
        zero mean and sqrt(n_spins) sd
       two_body_couplings: n_spin x n_spin symmetric array of coefficients sampled
        from a normal distribution with zero mean and sqrt(n_spins) sd
    """
    external_fields = np.zeros((n_spins))
    two_body_couplings = np.zeros((n_spins, n_spins))
    for i in range(n_spins):
        external_fields[i] = np.random.normal(0, np.sqrt(n_spins))
        for j in range(i, n_spins):
            if i == j:
                continue
            two_body_couplings[i, j] = two_body_couplings[j, i] = np.random.normal(
                0, np.sqrt(n_spins)
            )
    return external_fields, two_body_couplings


def get_thermal_target_measurement_outcome_distribution(
    n_spins: int,
    temperature: float,
    hamiltonian_parameters: Tuple[np.ndarray, np.ndarray],
) -> MeasurementOutcomeDistribution:
    """Generates thermal states target distribution, saved in a dict where keys are
    bitstrings and values are corresponding probabilities according to the Boltzmann
    distribution formula.

    Args:
        n_spins: positive number of spins in the Ising system
        temperature: temperature factor in the boltzman distribution
        hamiltonian_parameters: values of hamiltonian parameters, namely external
            fields and two body couplings.

    Returns:
       Thermal target distribution.
       Number of positive spins in the spin state.
    """
    partition_function = 0
    external_fields, two_body_couplings = hamiltonian_parameters
    beta = 1.0 / temperature
    distribution = {}

    for spin in range(int(2**n_spins)):
        ising_bitstring = convert_integer_to_ising_bitstring(spin, n_spins)
        energy = 0
        for i in range(n_spins):
            energy -= ising_bitstring[i] * external_fields[i]
            if i != n_spins - 1:
                energy -= (
                    ising_bitstring[i]
                    * ising_bitstring[i + 1]
                    * two_body_couplings[i, i + 1]
                )
        boltzmann_factor = np.exp(energy * beta)
        partition_function += boltzmann_factor

        binary_bitstring = tuple_to_bitstring(tuple(dec2bin(spin, n_spins)))
        distribution[binary_bitstring] = boltzmann_factor

    normalized_distribution: Dict[Union[str, Tuple[int, ...]], float] = {
        key: value / partition_function for key, value in distribution.items()
    }

    return MeasurementOutcomeDistribution(normalized_distribution)


def get_thermal_sampled_distribution(
    n_samples: int,
    n_spins: int,
    temperature: float,
    hamiltonian_parameters: Tuple[np.ndarray, np.ndarray],
) -> MeasurementOutcomeDistribution:
    """Generates thermal states sample distribution
    Args:
        n_samples: the number of samples from the original distribution
        n_spins: number of spins in the Ising system
        temperature: temperature factor in the Boltzmann distribution
    Returns:
       histogram_samples: keys are binary string representations and values
        are corresponding probabilities.
    """
    distribution = get_thermal_target_measurement_outcome_distribution(
        n_spins, temperature, hamiltonian_parameters
    ).distribution_dict
    temp_sample_distribution_dict = sample_from_probability_distribution(
        distribution, n_samples
    )
    histogram_samples = np.zeros(2**n_spins)
    for samples, counts in temp_sample_distribution_dict.items():
        integer_list: List[int] = []
        for elem in samples:
            integer_list.append(int(elem))
        idx = convert_ising_bitstring_to_integer(integer_list)
        histogram_samples[idx] += counts / n_samples

    sample_distribution_dict: Dict[Union[str, Tuple[int, ...]], float] = {}
    for spin in range(int(2**n_spins)):
        binary_bitstring = tuple_to_bitstring(tuple(dec2bin(spin, n_spins)))
        sample_distribution_dict[binary_bitstring] = histogram_samples[spin]

    return MeasurementOutcomeDistribution(sample_distribution_dict)


def get_cardinality_distribution(
    n_samples: int, n_spins: int, sampled_distribution: MeasurementOutcomeDistribution
) -> List[int]:
    """Generates a list with all the occurrences associated to different cardinalities
        in a sampled distribution.

    Args:
        n_samples: the number of samples used to build the sampled distribution
            (used for normalization purposes)
        n_spins: positive number of spins in the Ising system
        sampled_distribution: measurement outcome distribution built of samples drawn
            for a target distribution

    Returns:
        cardinality_list: a list with the cardinalities of all the sampled bitstrings.

    """
    histogram_samples = np.zeros(2**n_spins)
    cardinality_list: List[int] = []
    for samples, counts in sampled_distribution.distribution_dict.items():
        integer_list: List[int] = []
        for elem in samples:
            integer_list.append(int(elem))
        idx = convert_ising_bitstring_to_integer(integer_list)
        histogram_samples[idx] += counts / n_samples
        cardinality = 0
        for elem in integer_list:
            if elem == 1:
                cardinality += 1
        for _ in range(int(counts)):
            cardinality_list.append(cardinality)

    return cardinality_list
