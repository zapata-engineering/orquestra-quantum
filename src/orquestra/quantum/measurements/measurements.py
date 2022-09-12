################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
from __future__ import annotations

import copy
import json
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, TextIO, Tuple

import numpy as np

from ..distributions import MeasurementOutcomeDistribution
from ..operators import PauliRepresentation
from ..typing import AnyPath
from ..utils import convert_tuples_to_bitstrings, sample_from_probability_distribution
from .expectation_values import ExpectationValues
from .parities import check_parity_of_vector


def convert_bitstring_to_int(bitstring: Sequence[int]) -> int:
    """Convert a bitstring to an integer.

    Args:
        bitstring (list): A list of integers.
    Returns:
        int: The value of the bitstring, where the first bit in the least
            significant (little endian).
    """
    return int("".join(str(bit) for bit in bitstring[::-1]), 2)


def _convert_bitstrings_to_vector(bitstrings: Iterable[str]) -> np.ndarray:
    """Converts bitstrings to vector so parity can be checked."""
    n_qubits = len([*bitstrings][0])
    all_bits = "".join(bitstrings)
    bitstring_1d_array = np.frombuffer(all_bits.encode("utf-8"), "u1") - ord("0")
    return bitstring_1d_array.astype(int).reshape(-1, n_qubits)


def get_expectation_value_from_frequencies(
    marked_qubits: Iterable[int], bitstring_frequencies: Dict[str, int]
) -> float:
    """Get the expectation value the product of Z operators on selected qubits
    from bitstring frequencies.

    Args:
        marked_qubits: The qubits that the Z operators act on.
        bitstring_frequences: The frequencies of the bitstrings.

    Returns:
        The expectation value of the product of Z operators on selected qubits.
    """

    parity = (
        check_parity_of_vector(
            _convert_bitstrings_to_vector(bitstring_frequencies.keys()),
            marked_qubits,
        )
        * 2
        - 1
    )
    num_measurements = sum(bitstring_frequencies.values())
    expectation_values: np.ndarray = (
        np.fromiter(bitstring_frequencies.values(), dtype=int)
        * parity
        / num_measurements
    )

    # The item method converts a numpy float to a native Python float
    return expectation_values.sum().item()


def _check_sample_elimination(
    samples: Counter,
    bitstring_samples: List[Tuple[int, ...]],
    leftover_distribution: MeasurementOutcomeDistribution,
) -> Counter:
    """This is a function that checks that all elements in samples
    are present in bitstring_samples. If they are not, we eliminate the
    elements not in bitstring samples, set their probability to 0 in
    leftover_distribution and resample the appropriate number of times.
    Then, we re-check the new samples.
    Args:
        samples: The bitstrings to eliminate and how many times to eliminate them
        bitstring_samples: the bitstring distribution from where the bitstrings
                           should be removed
        leftover_distribution: the distribution from which samples to eliminate are
                               sampled
    Returns:
        correct_samples: A sample object that only contains bitstrings that can actually
                     be removed from bitstring_samples
    """
    bitstring_counts = Counter(bitstring_samples)

    nresamples = 1  # Initializing so that the loop starts
    corrected_leftover_distribution = MeasurementOutcomeDistribution(
        leftover_distribution.distribution_dict
    )
    correct_samples = samples.copy()
    while nresamples != 0:
        new_samples = None
        nresamples = 0
        for sample in correct_samples:
            bitstring = tuple([int(measurement_value) for measurement_value in sample])
            if correct_samples[sample] > bitstring_counts[bitstring]:
                nresamples = correct_samples[sample] - bitstring_counts[bitstring]
                correct_samples[sample] = bitstring_counts[bitstring]
                distribution_dict = corrected_leftover_distribution.distribution_dict
                distribution_dict[sample] = 0
                corrected_leftover_distribution = MeasurementOutcomeDistribution(
                    distribution_dict, True
                )
                new_samples = sample_from_probability_distribution(
                    corrected_leftover_distribution.distribution_dict, nresamples
                )
                correct_samples = correct_samples + new_samples
                break

    return correct_samples


class Measurements:
    """A class representing measurements from a quantum circuit. The bitstrings variable
    represents the internal data structure of the Measurements class. It is expressed as
    a list of tuples wherein each tuple is a measurement and the value of the tuple at a
    given index is the measured bit-value of the qubit (indexed from 0 -> N-1)"""

    def __init__(self, bitstrings: Optional[List[Tuple[int, ...]]] = None):
        if bitstrings is None:
            self.bitstrings = []
        else:
            self.bitstrings = bitstrings

    @classmethod
    def from_counts(cls, counts: Dict[str, int]):
        """Create an instance of the Measurements class from a dictionary

        Args:
            counts: mapping of bitstrings to integers representing the number of times
                the bitstring was measured
        """
        measurements = cls()
        measurements.add_counts(counts)
        return measurements

    @classmethod
    def get_measurements_representing_distribution(
        cls,
        measurement_outcome_distribution: MeasurementOutcomeDistribution,
        number_of_samples: int,
    ):
        """Create an instance of the Measurements class that exactly (or as closely as
        possible) resembles the input bitstring distribution.

        Args:
            measurement_outcome_distribution: the bitstring distribution to be sampled
            number_of_samples: the number of measurements
        """
        distribution = copy.deepcopy(measurement_outcome_distribution.distribution_dict)

        bitstring_samples = []
        # Rounding gives the closest integer to the observed frequency
        for state in distribution:
            bitstring = tuple([int(measurement_value) for measurement_value in state])

            bitstring_samples += [bitstring] * int(
                round(distribution[state] * number_of_samples)
            )

        # If frequencies are inconsistent with number of samples, we may need to
        # add or delete samples. The bitstrings to correct are chosen at random,
        # giving more weight to those with non-integer part closest to 0.5
        if len(bitstring_samples) != number_of_samples:
            leftover_distribution = MeasurementOutcomeDistribution(
                {
                    states: 0.5
                    - abs(0.5 - (distribution[states] * number_of_samples) % 1)
                    for states in distribution.keys()
                },
                True,
            )

            samples = sample_from_probability_distribution(
                leftover_distribution.distribution_dict,
                abs(number_of_samples - len(bitstring_samples)),
            )
            if number_of_samples - len(bitstring_samples) > 0:
                for sample in samples:
                    bitstring_samples += [
                        tuple([int(measurement_value) for measurement_value in sample])
                    ] * samples[sample]
            else:
                # Eliminating samples: need to ensure they are present in the
                # bitstring_samples list
                samples = _check_sample_elimination(
                    samples, bitstring_samples, leftover_distribution
                )
                for sample in samples:
                    for _ in range(samples[sample]):
                        bitstring_samples.remove(
                            tuple(
                                [int(measurement_value) for measurement_value in sample]
                            )
                        )

        return cls(bitstring_samples)

    @classmethod
    def load_from_file(cls, file: TextIO):
        """Load a set of measurements from file

        Args:
            file (str or file-like object): the name of the file, or a file-like object
        """
        if isinstance(file, str):
            with open(file, "r") as f:
                data = json.load(f)
        else:
            data = json.load(file)

        bitstrings = []
        for bitstring in data["bitstrings"]:
            bitstrings.append(tuple(bitstring))

        return cls(bitstrings=bitstrings)

    def save(self, filename: AnyPath):
        """Serialize the Measurements object into a file in JSON format.

        Args:
            filename (string): filename to save the data to
        """
        data = {
            "counts": self.get_counts(),
            # This step is required if bistrings contain np.int8 instead of regular int.
            "bitstrings": [
                list(map(int, list(bitstring))) for bitstring in self.bitstrings
            ],
        }
        with open(filename, "w") as f:
            f.write(json.dumps(data, indent=2))

    def get_counts(self):
        """Get the measurements as a histogram

        Returns:
            A dictionary mapping bitstrings to integers representing the number of times
            the bitstring was measured
        """
        bitstrings = convert_tuples_to_bitstrings(self.bitstrings)
        return dict(Counter(bitstrings))

    def add_counts(self, counts: Dict[str, int]):
        """Add measurements from a histogram

        Args:
            counts: mapping of bitstrings to integers representing the number of times
                the bitstring was measured
                NOTE: bitstrings are also indexed from 0 -> N-1, where the "001"
                bitstring represents a measurement of qubit 2 in the 1 state
        """
        for bitstring in counts.keys():
            measurement = []
            for bitvalue in bitstring:
                measurement.append(int(bitvalue))

            self.bitstrings += [tuple(measurement)] * counts[bitstring]

    def get_distribution(self) -> MeasurementOutcomeDistribution:
        """Get the normalized probability distribution representing the measurements

        Returns:
            distribution: bitstring distribution based on the frequency of measurements
        """
        counts = self.get_counts()
        num_measurements = len(self.bitstrings)

        distribution = {}
        for bitstring in counts.keys():
            distribution[bitstring] = counts[bitstring] / num_measurements

        return MeasurementOutcomeDistribution(distribution)

    def get_expectation_values(
        self, ising_operator: PauliRepresentation, use_bessel_correction: bool = False
    ) -> ExpectationValues:
        """Get the expectation values of an operator from the measurements.

        Args:
            ising_operator: the operator
            use_bessel_correction: Whether to use Bessel's correction when
                when estimating the covariance of operators. Using the
                correction provides an unbiased estimate for covariances, but
                diverges when only one sample is taken.

        Returns:
            expectation values of each term in the operator
        """
        # We require operator to be ising because measurements are always performed in
        # the Z basis, so we need the operator to be Ising (containing only Z terms).
        # A general Qubit Operator could have X or Y terms which don’t get directly
        # measured.
        if not ising_operator.is_ising:
            raise TypeError("Input operator is not ising.")

        # Count number of occurrences of bitstrings
        bitstring_frequencies = self.get_counts()
        num_measurements = len(self.bitstrings)

        # Perform weighted average
        expectation_values_list = [
            term.coefficient
            * get_expectation_value_from_frequencies(term.qubits, bitstring_frequencies)
            for term in ising_operator.terms
        ]
        expectation_values = np.array(expectation_values_list)

        correlations = np.zeros((len(ising_operator.terms),) * 2, dtype=complex)
        for i, first_term in enumerate(ising_operator.terms):
            correlations[i, i] = first_term.coefficient**2
            for j in range(i):
                second_term = ising_operator.terms[j]
                marked_qubits = first_term.qubits.symmetric_difference(
                    second_term.qubits
                )
                correlations[i, j] = (
                    first_term.coefficient
                    * second_term.coefficient
                    * get_expectation_value_from_frequencies(
                        marked_qubits, bitstring_frequencies
                    )
                )
                correlations[j, i] = correlations[i, j]

        denominator = (
            num_measurements - 1 if use_bessel_correction else num_measurements
        )

        estimator_covariances = (
            correlations
            - expectation_values[:, np.newaxis] * expectation_values[np.newaxis, :]
        ) / denominator

        return ExpectationValues(
            expectation_values, [correlations], [estimator_covariances]
        )
