#   Copyright 2017 The OpenFermion Developers
#   Modifications copyright 2022 Zapata Computing, Inc. for compatibility reasons.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""This module provides functions to interface with scipy.sparse."""
from functools import reduce
from typing import List, Optional

import numpy
import scipy
import scipy.sparse
import scipy.sparse.linalg

from .._pauli_operators import PauliRepresentation, PauliSum, PauliTerm

# Make global definitions.
identity_csc = scipy.sparse.identity(2, format="csc", dtype=complex)
pauli_x_csc = scipy.sparse.csc_matrix([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
pauli_y_csc = scipy.sparse.csc_matrix([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
pauli_z_csc = scipy.sparse.csc_matrix([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
pauli_matrix_map = {
    "I": identity_csc,
    "X": pauli_x_csc,
    "Y": pauli_y_csc,
    "Z": pauli_z_csc,
}


def get_sparse_operator(operator: PauliRepresentation, n_qubits: Optional[int] = None):
    """Initialize a Scipy sparse matrix from a PauliTerm or PauliSum.

    Args:
        operator: the pauli representation to convert to matrix representation.
        n_qubits (int): Number of qubits.

    Returns:
        The corresponding Scipy sparse matrix.
    """
    if not isinstance(operator, (PauliSum, PauliTerm)):
        raise TypeError(
            "Failed to convert a {} to a sparse matrix.".format(type(operator).__name__)
        )

    if n_qubits is None:
        n_qubits = operator.n_qubits
    if n_qubits < operator.n_qubits:
        raise ValueError("Invalid number of qubits specified.")

    # Construct the Scipy sparse matrix.
    n_hilbert = 2**n_qubits
    values_list: List[numpy.ndarray] = []
    row_list: List[numpy.ndarray] = []
    column_list: List[numpy.ndarray] = []

    # Loop through the terms.
    for qubit_term in operator.terms:
        tensor_factor = 0
        coefficient = qubit_term.coefficient
        sparse_operators = [coefficient]
        for qubit_num, operator_str in sorted(qubit_term.operations):

            # Grow space for missing identity operators.
            if qubit_num > tensor_factor:
                identity_qubits = qubit_num - tensor_factor
                identity = scipy.sparse.identity(
                    2**identity_qubits, dtype=complex, format="csc"
                )
                sparse_operators += [identity]

            # Add actual operator to the list.
            sparse_operators += [pauli_matrix_map[operator_str]]
            tensor_factor = qubit_num + 1

        # Grow space at end of string unless operator acted on final qubit.
        if tensor_factor < n_qubits or not qubit_term:
            identity_qubits = n_qubits - tensor_factor
            identity = scipy.sparse.identity(
                2**identity_qubits, dtype=complex, format="csc"
            )
            sparse_operators += [identity]

        # Extract triplets from sparse_term.
        sparse_matrix = _kronecker_operators(sparse_operators)
        values_list.append(sparse_matrix.tocoo(copy=False).data)
        (column, row) = sparse_matrix.nonzero()
        column_list.append(column)
        row_list.append(row)

    # Create sparse operator.
    values_list = numpy.concatenate(values_list)
    row_list = numpy.concatenate(row_list)
    column_list = numpy.concatenate(column_list)
    sparse_operator = scipy.sparse.coo_matrix(
        (values_list, (row_list, column_list)), shape=(n_hilbert, n_hilbert)
    ).tocsc(copy=False)
    sparse_operator.eliminate_zeros()
    return sparse_operator


def _kronecker_operators(*args):
    """Return the Kronecker product of multiple sparse.csc_matrix operators."""
    return reduce(_wrapped_kronecker, *args)


def _wrapped_kronecker(operator_1, operator_2):
    """Return the Kronecker product of two sparse.csc_matrix operators."""
    return scipy.sparse.kron(operator_1, operator_2, "csc")


def expectation(operator, state):
    """Compute the expectation value of an operator with a state.

    Args:
        operator(scipy.sparse.spmatrix or scipy.sparse.linalg.LinearOperator):
            The operator whose expectation value is desired.
        state(numpy.ndarray or scipy.sparse.spmatrix): A numpy array
            representing a pure state or a sparse matrix representing a density
            matrix. If `operator` is a LinearOperator, then this must be a
            numpy array.

    Returns:
        A complex number giving the expectation value.

    Raises:
        ValueError: Input state has invalid format.
    """

    if isinstance(state, scipy.sparse.spmatrix):
        # Handle density matrix.
        if isinstance(operator, scipy.sparse.linalg.LinearOperator):
            raise ValueError(
                "Taking the expectation of a LinearOperator with "
                "a density matrix is not supported."
            )
        product = state * operator
        expectation = numpy.sum(product.diagonal())

    elif isinstance(state, numpy.ndarray):
        # Handle state vector.
        if len(state.shape) == 1:
            # Row vector
            expectation = numpy.dot(numpy.conjugate(state), operator * state)
        else:
            # Column vector
            expectation = numpy.dot(numpy.conjugate(state.T), operator * state)[0, 0]

    else:
        # Handle exception.
        raise ValueError("Input state must be a numpy array or a sparse matrix.")

    # Return.
    return expectation
