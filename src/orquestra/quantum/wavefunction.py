################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
import json
from functools import lru_cache
from math import log2
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
from warnings import warn

import numpy as np
from sympy import Matrix, Symbol

from .typing import AnyPath, LoadSource, ParameterizedVector
from .utils import (
    convert_array_to_dict,
    convert_bitstrings_to_tuples,
    convert_dict_to_array,
    ensure_open,
)


def _is_number(possible_number):
    try:
        complex(possible_number)
        return True
    except Exception:
        return False


def _cast_sympy_matrix_to_numpy(sympy_matrix, complex=False):
    new_type = np.complex128 if complex else np.float64

    try:
        return np.array(sympy_matrix, dtype=new_type).flatten()
    except TypeError:
        return np.array(sympy_matrix, dtype=object).flatten()


def _get_next_number_with_same_hamming_weight(val):
    # Copied from:
    # http://graphics.stanford.edu/~seander/bithacks.html#NextBitPermutation
    t = (val | (val - 1)) + 1
    return t | ((((t & -t) // (val & -val)) >> 1) - 1)


def _most_significant_set_bit(val):
    bin_string = bin(val)
    return len(bin_string) - 2


class Wavefunction:
    """
    A simple wavefunction data structure that can
    be used to calculate amplitudes of quantum states.

    Args:
        amplitude_vector: the initial amplitudes of the system,
            can either be a NumPy ndarray or a SymPy Matrix
    """

    def __init__(self, amplitude_vector: ParameterizedVector) -> None:
        if bin(len(amplitude_vector)).count("1") != 1:
            raise ValueError(
                "Provided wavefunction does not have a size of a power of 2."
            )

        try:
            self._amplitude_vector = np.asarray(amplitude_vector, dtype=complex)
        except TypeError:
            self._amplitude_vector = Matrix(amplitude_vector)

        self._check_normalization(self._amplitude_vector)

    @property
    def amplitudes(self) -> Union[np.ndarray, Matrix]:
        if self.free_symbols:
            return _cast_sympy_matrix_to_numpy(self._amplitude_vector, complex=True)

        return self._amplitude_vector

    @property
    def n_qubits(self):
        return int(log2(len(self)))

    @property
    def free_symbols(self) -> Set[Symbol]:
        return getattr(self._amplitude_vector, "free_symbols", set())

    @staticmethod
    def _check_normalization(arr: ParameterizedVector):

        if (
            isinstance(arr, np.ndarray)
            or isinstance(arr, Matrix)
            and not arr.free_symbols
        ):
            probs_of_ground_entries = np.sum(np.abs(arr) ** 2)

            if not np.isclose(probs_of_ground_entries, 1.0):
                raise ValueError("Vector does not result in a unit probability.")
        else:
            numbers = np.array(
                [elem for elem in arr if _is_number(elem)], dtype=np.complex128
            )
            probs_of_ground_entries = np.sum(np.abs(numbers) ** 2)

            if probs_of_ground_entries > 1.0:
                raise ValueError(
                    "Ground entries in vector already exceeding probability of 1.0!"
                )

    def __len__(self) -> int:
        return len(self._amplitude_vector)

    def __iter__(self):
        return iter(self._amplitude_vector)

    def __getitem__(self, idx):
        return self._amplitude_vector[idx]

    def __setitem__(self, idx, val):
        old_val = self._amplitude_vector[idx]
        self._amplitude_vector[idx] = val

        try:
            self._check_normalization(self._amplitude_vector)
        except ValueError:
            self._amplitude_vector[idx] = old_val

            raise ValueError("This assignment violates probability unity.")

    def __str__(self) -> str:
        cast_wf = _cast_sympy_matrix_to_numpy(self._amplitude_vector, complex=True)
        return f"Wavefunction({cast_wf})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Wavefunction):
            return False

        return np.array_equal(self.amplitudes, other.amplitudes)

    @staticmethod
    def zero_state(n_qubits: int) -> "Wavefunction":
        if not isinstance(n_qubits, int):
            warn(
                f"Non-integer value {n_qubits} passed as number of qubits! "
                "Will be cast to integer."
            )
            n_qubits = int(n_qubits)

        if n_qubits <= 0:
            raise ValueError(f"Invalid number of qubits in system. Got {n_qubits}.")

        np_arr = np.zeros(2**n_qubits, dtype=np.complex128)
        np_arr[0] = 1.0
        return Wavefunction(np_arr)

    @staticmethod
    def dicke_state(n_qubits: int, hamming_weight: int) -> "Wavefunction":
        initial_wf = Wavefunction.zero_state(n_qubits)

        if hamming_weight < 0 or not isinstance(hamming_weight, int):
            raise ValueError(f"Invalid hamming weight value. Got {hamming_weight}.")

        if hamming_weight > n_qubits:
            raise ValueError(
                f"Hamming weight larger than number of qubits. \
                    Got {hamming_weight}. Max can be {n_qubits}."
            )

        if hamming_weight == 0:
            return initial_wf
        else:
            del initial_wf

            # Get first value with hamming weight
            current_value = int("1" * hamming_weight, base=2)

            counter: int = 1
            indices: List[int] = [current_value]
            while True:
                current_value = _get_next_number_with_same_hamming_weight(current_value)
                if not _most_significant_set_bit(current_value) <= n_qubits:
                    break
                indices.append(current_value)
                counter += 1

            amplitude = 1 / np.sqrt(counter)
            wf = np.zeros(2**n_qubits, dtype=np.complex128)
            wf[indices] = amplitude

            return Wavefunction(wf)

    def bind(self, symbol_map: Dict[Symbol, Any]) -> "Wavefunction":
        if not self.free_symbols:
            return self
        assert isinstance(self._amplitude_vector, Matrix)
        result = self._amplitude_vector.subs(symbol_map)

        try:
            return type(self)(result)
        except ValueError:
            raise ValueError("Passed map results in a violation of probability unity.")

    def get_probabilities(self) -> np.ndarray:
        return np.abs(self.amplitudes) ** 2

    def get_outcome_probs(self) -> Dict[str, float]:
        values = [
            format(i, "0" + str(self.n_qubits) + "b")[::-1] for i in range(len(self))
        ]

        probs = self.get_probabilities()

        return dict(zip(values, probs))


def flip_wavefunction(wavefunction: Wavefunction):
    return Wavefunction(flip_amplitudes(wavefunction.amplitudes))


def flip_amplitudes(amplitudes: Union[Sequence[complex], np.ndarray]) -> np.ndarray:
    number_of_states = len(amplitudes)
    ordering = _get_ordering(number_of_states)
    return np.asarray(amplitudes)[ordering]


@lru_cache
def _get_ordering(number_of_states: int) -> np.ndarray:
    num_bits = number_of_states.bit_length() - 1
    ordering = (
        np.arange(2**num_bits)
        .reshape(num_bits * [2])
        .transpose(*reversed(range(num_bits)))
        .reshape(2**num_bits)
    )
    return ordering


def load_wavefunction(file: LoadSource) -> Wavefunction:
    """Load a qubit wavefunction from a file.

    Args:
        file (str or file-like object): the name of the file, or a file-like object.

    Returns:
        wavefunction (orquestra.quantum.Wavefunction): the wavefunction object
    """

    with ensure_open(file) as f:
        data = json.load(f)

    wavefunction = Wavefunction(convert_dict_to_array(data["amplitudes"]))
    return wavefunction


def save_wavefunction(wavefunction: Wavefunction, filename: AnyPath) -> None:
    """Save a wavefunction object to a file.

    Args:
        wavefunction (orquestra.quantum.Wavefunction): the wavefunction object
        filename (str): the name of the file
    """

    data: Dict[str, Any] = {}
    data["amplitudes"] = convert_array_to_dict(wavefunction.amplitudes)
    with open(filename, "w") as f:
        f.write(json.dumps(data, indent=2))


def sample_from_wavefunction(
    wavefunction: Wavefunction,
    n_samples: int,
    seed: Optional[int] = None,
) -> List[Tuple[int, ...]]:
    """Sample bitstrings from a wavefunction.

    Args:
        wavefunction: the wavefunction to sample from.
        n_samples: the number of samples taken. Needs to be greater than 0.
        seed: the seed of the sampler

    Returns:
        List[Tuple[int]]: A list of tuples where the each tuple is a sampled bitstring.
    """
    if n_samples < 1:
        raise ValueError("Must sample from wavefunction at least once.")
    rng = np.random.default_rng(seed)
    outcome_strings, probabilities_np = zip(*wavefunction.get_outcome_probs().items())
    probabilities = [
        x[0] if isinstance(x, (list, np.ndarray)) else x for x in list(probabilities_np)
    ]
    # accelerate sampling by smartly choosing when formatting of samples is done
    if len(wavefunction) < n_samples:
        outcome_tuples: List[Union[Tuple[int, ...], int]] = []
        outcome_tuples += convert_bitstrings_to_tuples(outcome_strings)
        outcome_tuples += [0]  # adding non tuple forces rng.choice to return tuples
        probabilities += [0]  # need to add corresponding probability of 0
        samples = rng.choice(
            a=np.array(outcome_tuples, dtype=object), size=n_samples, p=probabilities
        ).tolist()
    else:
        string_samples = rng.choice(a=outcome_strings, size=n_samples, p=probabilities)
        samples = convert_bitstrings_to_tuples(string_samples)
    return samples
