################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
import json
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..operators import PauliRepresentation
from ..typing import AnyPath, LoadSource
from ..utils import convert_array_to_dict, convert_dict_to_array, ensure_open


class Parities:
    """A class representing counts of parities for Pauli terms.

    Args:
        values (np.array): Number of observations of parities. See Attributes.
        correlations (list): Number of observations of pairwise products of terms.
            See Attributes.

    Attributes:
        values (np.array): an array of dimension N x 2 indicating how many times
            each Pauli term was observed with even and odd parity, where N is the
            number of Pauli terms. Here values[i][0] and values[i][1] correspond
            to the number of samples with even and odd parities for term P_i,
            respectively.
        correlations (list): a list of 3-dimensional numpy arrays indicating how
            many times each product of Pauli terms was observed with even and odd
            parity. Here correlations[i][j][k][0] and correlations[i][j][k][1]
            correspond to the number of samples with even and odd parities term P_j P_k
            in frame i, respectively.
    """

    def __init__(
        self, values: np.ndarray, correlations: Optional[List[np.ndarray]] = None
    ):
        self.values = values
        self.correlations = correlations

    def to_dict(self) -> dict:
        data: Dict[str, Any] = {"values": convert_array_to_dict(self.values)}
        if self.correlations:
            data["correlations"] = [
                convert_array_to_dict(arr) for arr in self.correlations
            ]
        return data

    @classmethod
    def from_dict(cls, data: dict):
        values = convert_dict_to_array(data["values"])
        if data.get("correlations"):
            correlations: Optional[List] = [
                convert_dict_to_array(arr) for arr in data["correlations"]
            ]
        else:
            correlations = None
        return cls(values, correlations)


def save_parities(parities: Parities, filename: AnyPath) -> None:
    """Save parities to a file.

    Args:
        parities (orquestra.quantum.measurement.Parities): the parities
        file (str or file-like object): the name of the file, or a file-like object
    """
    data = parities.to_dict()

    with open(filename, "w") as f:
        f.write(json.dumps(data, indent=2))


def load_parities(file: LoadSource) -> Parities:
    """Load parities from a file.

    Args:
        file (str or file-like object): the name of the file, or a file-like object.

    Returns:
        orquestra.quantum.measurement.Parities: the parities
    """

    with ensure_open(file) as f:
        data = json.load(f)

    return Parities.from_dict(data)


def check_parity(
    bitstring: Union[str, Sequence[int]], marked_qubits: Iterable[int]
) -> bool:
    """Determine if the marked qubits have even parity for the given bitstring.

    Args:
        bitstring: The bitstring, either as a tuple or little endian string.
        marked_qubits: The qubits whose parity is to be determined.

    Returns:
        True if an even number of the marked qubits are in the 1 state, False
            otherwise.
    """
    result = True
    for qubit_index in marked_qubits:
        if bitstring[qubit_index] == "1" or bitstring[qubit_index] == 1:
            result = not result
    return result


def check_parity_of_vector(
    bitstrings_vector: np.ndarray, marked_qubits: Iterable[int]
) -> np.ndarray:
    """Determine if the marked qubits have even parity for each bitstring in the given
        vector.
        NOTE: This performs the same functionality as `check_parity` but is much
        faster as it uses vectorization to find the parity of multiple bitstrings
        at once.

    Args:
        bitstring: A 2d array of bitstrings whose size is number of bistrings * number
            of qubits
        marked_qubits: The qubits whose parity is to be determined.

    Returns:
        A 1d array with size equal to number of bitstrings. Each entry is 1 if an even
        number of the marked qubits of the corresponding bitstring are in the 1 state
        and 0 if otherwise.
    """
    if not marked_qubits:
        return np.ones(bitstrings_vector.shape[0])

    # Check if an even number of the marked qubits of each bitstring are in the 1 state
    bitstring_subset = bitstrings_vector[:, np.fromiter(marked_qubits, dtype=int)]
    return (bitstring_subset.sum(axis=1) + 1) % 2


def get_parities_from_measurements(
    measurements: List[Tuple[int]], ising_operator: PauliRepresentation
) -> Parities:
    """Get expectation values from bitstrings.

    Args:
        measurements (list): the measured bitstrings
        ising_operator: the operator

    Returns:
        orquestra.quantum.measurement.Parities: parities of each term in the operator
    """

    # check input format
    if not ising_operator.is_ising:
        raise TypeError("Input operator is not an ising operator")

    # Count number of occurrences of bitstrings
    bitstring_frequencies = Counter(measurements)
    bitstrings_vector = np.array([*bitstring_frequencies.keys()])
    bitstring_counts: np.ndarray = np.fromiter(
        bitstring_frequencies.values(), dtype=int
    )

    # Count parity occurrences
    values = []
    for _, term in enumerate(ising_operator.terms):
        parity = check_parity_of_vector(bitstrings_vector, term.qubits)

        true_parity_count: int = (parity * bitstring_counts).sum()
        false_parity_count: int = ((1 - parity) * bitstring_counts).sum()
        values.append([true_parity_count, false_parity_count])

    # Count parity occurrences for pairwise products of operators
    correlations = [np.zeros((len(ising_operator.terms), len(ising_operator.terms), 2))]
    for term1_index, term1 in enumerate(ising_operator.terms):
        for term2_index, term2 in enumerate(ising_operator.terms):

            parity1 = check_parity_of_vector(bitstrings_vector, term1.qubits)
            parity2 = check_parity_of_vector(bitstrings_vector, term2.qubits)

            # 0 if parities are equal, 1 otherwise
            equal_parities = np.abs(parity1 - parity2)

            # Counts of bitstrings where parity is equal
            correlations[0][term1_index, term2_index][0] += (
                (1 - equal_parities) * bitstring_counts
            ).sum()

            # Counts of bitstrings where parity is not equal
            correlations[0][term1_index, term2_index][1] += (
                (equal_parities) * bitstring_counts
            ).sum()

    return Parities(np.array(values), correlations)
