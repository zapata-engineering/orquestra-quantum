################################################################################
# © Copyright 2020-2022 Zapata Computing Inc.
################################################################################
"""General-purpose utilities."""
import collections
import json
import os
import sys
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import sympy

from .typing import AnyPath, DumpTarget, LoadSource

RNDSEED = 12345


def convert_dict_to_array(dictionary: dict) -> np.ndarray:
    """Convert a dictionary to a numpy array.

    Args:
        dictionary (dict): the dict containing the data

    Returns:
        array (numpy.array): a numpy array
    """

    array = np.array(dictionary["real"])

    if dictionary.get("imag"):
        array = array + 1j * np.array(dictionary["imag"])

    return array


def convert_array_to_dict(array: np.ndarray) -> dict:
    """Convert a numpy array to a dictionary.

    Args:
        array (numpy.array): a numpy array

    Returns:
        dictionary (dict): the dict containing the data
    """

    dictionary = {}
    if np.iscomplexobj(array):
        dictionary["real"] = array.real.tolist()
        dictionary["imag"] = array.imag.tolist()
    else:
        dictionary["real"] = array.tolist()

    return dictionary


def dec2bin(number: int, length: int) -> List[int]:
    """Converts a decimal number into a binary representation
    of fixed number of bits.

    Args:
        number: (int) the input decimal number
        length: (int) number of bits in the output string

    Returns:
        A list of binary numbers
    """

    if pow(2, length) < number:
        sys.exit(
            "Insufficient number of bits for representing the number {}".format(number)
        )

    bit_str = bin(number)
    bit_str = bit_str[2 : len(bit_str)]  # chop off the first two chars
    bit_string = [int(x) for x in list(bit_str)]
    if len(bit_string) < length:
        len_zeros = length - len(bit_string)
        bit_string = [int(x) for x in list(np.zeros(len_zeros))] + bit_string

    return bit_string


def bin2dec(x: List[int]) -> int:
    """Converts a binary vector to an integer, with the 0-th
    element being the most significant digit.

    Args:
        x: (list) a binary vector

    Returns:
        An integer
    """

    dec = 0
    coeff = 1
    for i in range(len(x)):
        dec = dec + coeff * x[len(x) - 1 - i]
        coeff = coeff * 2
    return dec


# The functions PAULI_X, PAULI_Y, PAULI_Z and IDENTITY below are used for
# generating the generators of the Pauli group, which include Pauli X, Y, Z
# operators as well as identity operator

pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]])
pauli_y = np.array([[0.0, -1.0j], [1.0j, 0.0]])
pauli_z = np.array([[1.0, 0.0], [0.0, -1.0]])
identity = np.array([[1.0, 0.0], [0.0, 1.0]])


def is_identity(u: np.ndarray, tol=1e-15) -> bool:
    """Test if a matrix is identity.

    Args:
        u: np.ndarray
            Matrix to be checked.
        tol: float
            Threshold below which two matrix elements are considered equal.
    """

    dims = np.array(u).shape
    if dims[0] != dims[1]:
        raise Exception("Input matrix is not square.")

    return np.allclose(u, np.eye(u.shape[0]), atol=tol)


def is_unitary(u: np.ndarray, tol=1e-15) -> bool:
    """Test if a matrix is unitary.

    Args:
        u: array
            Matrix to be checked.
        tol: float
            Threshold below which two matrix elements are considered equal.
    """

    dims = np.array(u).shape
    if dims[0] != dims[1]:
        raise Exception("Input matrix is not square.")

    test_matrix = np.dot(np.array(u).T.conj(), u)
    return is_identity(test_matrix, tol)


def compare_unitary(u1: np.ndarray, u2: np.ndarray, tol: float = 1e-15) -> bool:
    """Compares two unitary operators to see if they are equal to within a phase.

    Args:
        u1 (numpy.ndarray): First unitary operator.
        u2 (numpy.ndarray): Second unitary operator.
        tol (float): Threshold below which two matrix elements are considered equal.

    Returns:
        bool: True if the unitaries are equal to within the tolerance, ignoring
            differences in global phase.
    """

    if not is_unitary(u1, tol):
        raise Exception("The first input matrix is not unitary.")
    if not is_unitary(u2, tol):
        raise Exception("The second input matrix is not unitary.")

    test_matrix = np.dot(u1.conj().T, u2)
    phase = test_matrix.item((0, 0)) ** -1
    return is_identity(phase * test_matrix, tol)


def sample_from_probability_distribution(
    probability_distribution: dict, n_samples: int
) -> collections.Counter:
    """
    Samples events from a discrete probability distribution

    Args:
        probabilty_distribution: The discrete probability distribution to be used
        for sampling. This should be a dictionary

        n_samples (int): The number of samples desired

    Returns:
        A dictionary of the outcomes sampled. The key values are the things be sampled
        and values are how many times those things appeared in the sampling
    """
    if isinstance(probability_distribution, dict):
        # Need to do this preprocessing to handle different types of dict keys
        keys_as_array = np.empty(len(probability_distribution), dtype=object)
        keys_as_array[:] = list(probability_distribution.keys())

        result = np.random.choice(
            keys_as_array,
            n_samples,
            replace=True,
            p=list(probability_distribution.values()),
        )
        sampled_dict: collections.Counter = collections.Counter(result)

        return sampled_dict
    else:
        raise RuntimeError(
            "Probability distribution should be a dictionary with key value \
        being the thing being sampled and the value being probability of getting \
        sampled "
        )


def convert_bitstrings_to_tuples(bitstrings: Iterable[str]) -> List[Tuple[int, ...]]:
    """Given the measured bitstrings, convert each bitstring to tuple format

    Args:
        bitstrings (list of strings): the measured bitstrings
    Returns:
        A list of tuples
    """
    measurements = [bitstring_to_tuple(bitstring) for bitstring in bitstrings]
    return measurements


@lru_cache()
def bitstring_to_tuple(bitstring: str) -> Tuple[int, ...]:
    """Given a bitstring, convert it to tuple format

    Args:
        bitstring (string): the measured bitstring
    Returns:
        A tuple of 0s and 1s
    """
    measurement = tuple(int(bit) for bit in bitstring[::-1])
    return measurement


def convert_tuples_to_bitstrings(tuples: List[Tuple[int]]) -> List[str]:
    """Given a set of measurement tuples, convert each to a little endian
    string.

    Args:
        tuples (list of tuples): the measurement tuples
    Returns:
        A list of bitstrings
    """
    bitstrings = [tuple_to_bitstring(tup) for tup in tuples]
    return bitstrings


@lru_cache()
def tuple_to_bitstring(tup: Tuple[int, ...]) -> str:
    """Given a tuple, convert to an equivalent string.

    Args:
        tup (tuple): the measurement tuple
    Returns:
        A string with binary digits
    """
    return "".join(map(str, tup))


class ValueEstimate(float):
    """A class representing a numerical value and its precision corresponding
        to an observable or an objective function

    Args:
        value (np.float): the numerical value or a value that can be converted to float
        precision (np.float): its precision

    Attributes:
        value (np.float): the numerical value
        precision (np.float): its precision
    """

    def __init__(self, value, precision: Optional[float] = None):
        super().__init__()
        self.precision = precision

    def __new__(cls, value, precision=None):
        return super().__new__(cls, value)

    def __eq__(self, other):
        super_eq = super().__eq__(other)
        if super_eq is NotImplemented:
            return super_eq
        return super_eq and self.precision == getattr(other, "precision", None)

    def __ne__(self, other):
        return not self == other

    def __str__(self):
        value_str = super().__str__()
        if self.precision is not None:
            return f"{value_str} ± {self.precision}"
        else:
            return f"{value_str}"

    def to_dict(self):
        """Convert to a dictionary"""

        data = {}
        if type(self).__module__ == np.__name__:
            data["value"] = self.value.item()
        else:
            data["value"] = self

        if type(self.precision).__module__ == np.__name__:
            data["precision"] = self.precision.item()
        else:
            data["precision"] = self.precision

        return data

    @classmethod
    def from_dict(cls, dictionary):
        """Create an ExpectationValues object from a dictionary."""

        value = dictionary["value"]
        if "precision" in dictionary:
            precision = dictionary["precision"]
            return cls(value, precision)
        else:
            return cls(value)


def load_value_estimate(file: LoadSource) -> ValueEstimate:
    """Loads value estimate from a failed.

    Args:
        file (str or file-like object): the name of the file, or a file-like object.

    Returns:
        array (numpy.array): the array
    """

    if isinstance(file, str):
        with open(file, "r") as f:
            data = json.load(f)
    else:
        data = json.load(file)  # type: ignore

    return ValueEstimate.from_dict(data)


def save_value_estimate(value_estimate: ValueEstimate, filename: AnyPath):
    """Saves value estimate to a file.

    Args:
        value_estimate (orquestra.quantum.utils.ValueEstimate): the value estimate
        file (str or file-like object): the name of the file, or a file-like object
    """
    dictionary = value_estimate.to_dict()

    with open(filename, "w") as f:
        f.write(json.dumps(dictionary, indent=2))


def load_list(file: LoadSource) -> List:
    """Load an array from a file.

    Args:
        file (str or file-like object): the name of the file, or a file-like object.

    Returns:
        array (list): the list
    """

    if isinstance(file, str):
        with open(file, "r") as f:
            data = json.load(f)
    else:
        data = json.load(file)  # type: ignore

    return data["list"]


def save_list(array: List, filename: AnyPath):
    """Save expectation values to a file.

    Args:
        array (list): the list to be saved
        file (str or file-like object): the name of the file, or a file-like object
    """
    dictionary: Dict[str, Any] = {}
    dictionary["list"] = array

    with open(filename, "w") as f:
        f.write(json.dumps(dictionary, indent=2))


def save_generic_dict(dictionary: Dict, filename: AnyPath):
    """Save dictionary as json

    Args:
        dictionary (dict): the dict containing the data
    """
    dictionary_stored = {}
    dictionary_stored.update(dictionary)

    with open(filename, "w") as f:
        f.write(json.dumps(dictionary_stored, indent=2))


def create_symbols_map(
    symbols: List[sympy.Symbol], params: np.ndarray
) -> Dict[sympy.Symbol, float]:
    """
    Creates a map to be used for evaluating sympy expressions.

    Args:
        symbols: list of sympy Symbols to be evaluated
        params: numpy array containing numerical value for the symbols
    """
    if len(symbols) != len(params):
        raise (
            ValueError(
                "Length of symbols: {0} doesn't match length of params: {1}".format(
                    len(symbols), len(params)
                )
            )
        )
    return {symbol: param for symbol, param in zip(symbols, params.tolist())}


def save_timing(walltime: float, filename: AnyPath) -> None:
    """
    Saves timing information.

    Args:
        walltime: The execution time.
    """

    with open(filename, "w") as f:
        f.write(json.dumps({"walltime": walltime}))


def save_nmeas_estimate(
    nmeas: float,
    nterms: int,
    filename: AnyPath,
    frame_meas: Optional[np.ndarray] = None,
) -> None:
    """Save an estimate of the number of measurements to a file

    Args:
        nmeas: total number of measurements for epsilon = 1.0
        nterms: number of terms (groups) in the objective function
        frame_meas: A list of the number of measurements per frame for epsilon = 1.0
    """

    data: Dict[str, Any] = {}
    data["K"] = nmeas
    data["nterms"] = nterms
    if frame_meas is not None:
        data["frame_meas"] = convert_array_to_dict(frame_meas)

    with open(filename, "w") as f:
        f.write(json.dumps(data, indent=2))


def load_nmeas_estimate(filename: AnyPath) -> Tuple[float, int, np.ndarray]:
    """Load an estimate of the number of measurements from a file.

    Args:
        filename: the name of the file

    Returns:
        nmeas: number of measurements for epsilon = 1.0
        nterms: number of terms in the hamiltonian
        frame_meas: frame measurements (number of measurements per group)
    """

    with open(filename, "r") as f:
        data = json.load(f)

    frame_meas = convert_dict_to_array(data["frame_meas"])
    K_coeff = data["K"]
    nterms = data["nterms"]

    return K_coeff, nterms, frame_meas


def scale_and_discretize(values: Iterable[float], total: int) -> List[int]:
    """Convert a list of floats to a list of integers such that the total equals
    a given value and the ratios of elements are approximately preserved.

    Args:
        values: The list of floats to be scaled and discretized.
        total: The desired total which the resulting values should sum to.

    Returns:
        A list of integers whose sum is equal to the given total, where the
            ratios of the list elements are approximately equal to the ratios
            of the input list elements.
    """

    # Note: combining the two lines below breaks type checking; see mypy #6040
    value_sum = sum(values)
    scale_factor = total / value_sum

    result = [np.floor(value * scale_factor) for value in values]
    remainders = [
        value * scale_factor - np.floor(value * scale_factor) for value in values
    ]
    indexes_sorted_by_remainder = np.argsort(remainders)[::-1]
    for index in range(int(round(total - sum(result)))):
        result[indexes_sorted_by_remainder[index]] += 1

    result = [int(value) for value in result]

    assert sum(result) == total, "The scaled list does not sum to the desired total."

    return result


def get_ordered_list_of_bitstrings(num_qubits: int) -> List[str]:
    """Create list of binary strings corresponding to 2^num_qubits integers
    and save them in ascending order.

    Args:
        num_qubits: number of binary digits in each bitstring

    Returns:
        The ordered bitstring representations of the integers
    """
    bitstrings = []
    for i in range(2**num_qubits):
        bitstring = "{0:b}".format(i)
        while len(bitstring) < num_qubits:
            bitstring = "0" + bitstring
        bitstrings.append(bitstring)
    return bitstrings


@contextmanager
def ensure_open(path_like: Union[LoadSource, DumpTarget], mode="r", encoding="utf-8"):
    # str | bytes | PathLike | Readable
    if isinstance(path_like, (str, bytes, os.PathLike)):
        with open(path_like, mode, encoding=encoding if "b" not in mode else None) as f:
            yield f
    else:
        # Readable | Writable
        if set(mode).intersection(set("wxa+")) and not path_like.writable():
            raise ValueError(f"File isn't writable, can't ensure mode {mode}")
        yield path_like
