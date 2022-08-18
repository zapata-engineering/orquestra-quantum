################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
import os

import numpy as np

from orquestra.quantum.measurements import (
    check_parity,
    check_parity_of_vector,
    get_parities_from_measurements,
    load_parities,
    save_parities,
)
from orquestra.quantum.operators import PauliSum


def remove_file_if_exists(filename):
    try:
        os.remove(filename)
    except OSError:
        pass


def test_parities_io():
    measurements = [(1, 0), (1, 0), (0, 1), (0, 0)]
    op = PauliSum("Z0 + Z1 + Z0*Z1")
    parities = get_parities_from_measurements(measurements, op)
    save_parities(parities, "parities.json")
    loaded_parities = load_parities("parities.json")
    assert np.allclose(parities.values, loaded_parities.values)
    assert len(parities.correlations) == len(loaded_parities.correlations)
    for i in range(len(parities.correlations)):
        assert np.allclose(parities.correlations[i], loaded_parities.correlations[i])
    remove_file_if_exists("parities.json")


def test_check_parity_odd_string():
    bitstring = "01001"
    marked_qubits = (1, 2, 3)
    assert not check_parity(bitstring, marked_qubits)


def test_check_parity_even_string():
    bitstring = "01101"
    marked_qubits = (1, 2, 3)
    assert check_parity(bitstring, marked_qubits)


def test_check_parity_odd_tuple():
    bitstring = (0, 1, 0, 0, 1)
    marked_qubits = (1, 2, 3)
    assert not check_parity(bitstring, marked_qubits)


def test_check_parity_even_tuple():
    bitstring = (0, 1, 1, 0, 1)
    marked_qubits = (1, 2, 3)
    assert check_parity(bitstring, marked_qubits)


def test_check_parity_of_vector():
    bitstrings = np.array([[0, 1, 1, 0, 1], [0, 1, 0, 0, 1]])
    marked_qubits = (1, 2, 3)
    assert np.allclose(
        check_parity_of_vector(bitstrings, marked_qubits), np.array([1, 0])
    )


def test_check_parity_of_vector_with_no_marked_qubits():
    bitstring = np.array([[1, 0], [0, 0]])
    marked_qubits = []
    assert all(check_parity_of_vector(bitstring, marked_qubits))
