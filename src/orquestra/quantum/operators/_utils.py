################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
import random
from typing import List, Optional

import numpy as np

from ..utils import ValueEstimate, bin2dec, dec2bin
from ..wavefunction import Wavefunction
from ._openfermion_utils.sparse_tools import expectation, get_sparse_operator
from ._pauli_operators import PauliRepresentation, PauliSum, PauliTerm


def get_pauliop_from_matrix(operator: List[List]) -> PauliSum:
    r"""Expands a 2^n by 2^n matrix into n-qubit Pauli basis. The runtime of
    this function is O(2^2n).

    Args:
        operator: a list of lists (rows) representing a 2^n by 2^n
            matrix.

    Returns:
        A PauliSum instance corresponding to the expansion of
        the input operator as a sum of Pauli strings:

        O = 2^-n \sum_P tr(O*P) P
    """

    nrows = len(operator)
    ncols = len(operator[0])

    # Check if the input operator is square
    if nrows != ncols:
        raise Exception("The input operator is not square")

    # Check if the dimensions are powers of 2
    if not (((nrows & (nrows - 1)) == 0) and nrows > 0):
        raise Exception("The number of rows is not a power of 2")
    if not (((ncols & (ncols - 1)) == 0) and ncols > 0):
        raise Exception("The number of cols is not a power of 2")

    n = int(np.log2(nrows))  # number of qubits

    def decode(bit_string):  # Helper function for converting any 2n-bit
        # string to a label vector representing a Pauli
        # string of length n

        if len(bit_string) != 2 * n:
            raise Exception("LH_expand:decode: input bit string length not 2n")

        output_label = list(np.zeros(n))
        for i in range(0, n):
            output_label[i] = bin2dec(bit_string[2 * i : 2 * i + 2])

        return output_label

    def trace_product(label_vec):  # Helper function for computing tr(OP)
        # where O is the input operator and P is a
        # Pauli string operator

        def f(j):  # Function which computes the index of the nonzero
            # element in P for a given column j

            j_str = dec2bin(j, n)
            for index in range(0, n):
                if label_vec[index] in [1, 2]:  # flip if X or Y
                    j_str[index] = int(not j_str[index])
            return bin2dec(j_str)

        def nz(j):  # Function which computes the value of the nonzero
            # element in P on the column j

            val_nz = 1.0
            j_str = dec2bin(j, n)
            for index in range(0, n):
                if label_vec[index] == 2:
                    if j_str[index] == 0:
                        val_nz = val_nz * (1j)
                    if j_str[index] == 1:
                        val_nz = val_nz * (-1j)
                if label_vec[index] == 3:
                    if j_str[index] == 1:
                        val_nz = val_nz * (-1)
            return val_nz

        # Compute the trace
        tr = 0.0
        for j in range(0, 2**n):  # loop over the columns
            tr = tr + operator[j][f(j)] * nz(j)

        return tr / 2**n

    # Expand the operator in Pauli basis
    coeffs = list(np.zeros(4**n))
    labels = list(np.zeros(4**n))
    for i in range(0, 4**n):  # loop over all 2n-bit strings
        current_string = dec2bin(i, 2 * n)  # see util.py
        current_label = decode(current_string)
        coeffs[i] = trace_product(current_label)
        labels[i] = current_label

    return get_pauliop_from_coeffs_and_labels(coeffs, labels)


def get_pauliop_from_coeffs_and_labels(
    coeffs: List[float], labels: List[List[int]]
) -> PauliSum:
    """Generates a PauliSum based on a coefficient vector and
    a label matrix.

    Args:
        coeffs: a list of floats representing the coefficients
            for the terms in the Hamiltonian
        labels: a list of lists (a matrix) where each list
            is a vector of integers representing the Pauli
            string. See pauliutil.py for details.

    Example:

        The Hamiltonian H = 0.1 X1 X2 - 0.4 Y1 Y2 Z3 Z4 can be
        initiated by calling

        H = get_pauliop_from_coeffs_and_labels([0.1, -0.4],  # coefficients
            [[1 1 0 0],  # label matrix
            [2 2 3 3]])
    """

    output = PauliSum()
    for i in range(0, len(labels)):
        string_term = ""
        for ind, elem in enumerate(labels[i]):
            pauli_symbol = ""
            if elem == 1:
                pauli_symbol = "*X" + str(ind)
            elif elem == 2:
                pauli_symbol = "*Y" + str(ind)
            elif elem == 3:
                pauli_symbol = "*Z" + str(ind)
            string_term += pauli_symbol

        output += PauliTerm(f"{coeffs[i]}{string_term}")

    return output


def generate_random_pauliop(
    nqubits: int,
    nterms: int,
    nlocality: int,
    max_coeff: float,
    fixed_coeff: bool = False,
) -> PauliSum:
    """Generates a Hamiltonian with term coefficients uniformly distributed
    in [-max_coeff, max_coeff].

    Args:
        nqubits   - number of qubits
        nterms    - number of terms in the Hamiltonian
        nlocality - locality of the Hamiltonian
        max_coeff - bound for generating the term coefficients
        fixed_coeff (bool) - If true, all the terms are assign the
            max_coeff as coefficient.

    Returns:
        A PauliSum with the appropriate coefficient vector
        and label matrix.
    """
    # generate random coefficient vector
    if fixed_coeff:
        coeffs = [max_coeff] * nterms
    else:
        coeffs = list(np.zeros(nterms))
        for j in range(0, nterms):
            coeffs[j] = random.uniform(-max_coeff, max_coeff)

    # generate random label vector
    labels = list(np.zeros(nterms, dtype=int))
    label_set = set()
    j = 0
    while j < nterms:
        inds_nontrivial = sorted(random.sample(range(0, nqubits), nlocality))
        label = list(np.zeros(nqubits, dtype=int))
        for ind in inds_nontrivial:
            label[ind] = random.randint(1, 3)
        if str(label) not in label_set:
            labels[j] = label
            j += 1
            label_set.add(str(label))
    return get_pauliop_from_coeffs_and_labels(coeffs, labels)


def evaluate_operator(
    operator: PauliRepresentation, expectation_values
) -> ValueEstimate:
    """Evaluate the expectation value of a qubit operator using expectation values for
    the terms.

    Args:
        operator: the operator
        expectation_values: the expectation values

    Returns:
        value_estimate: stores the value of the expectation and its precision
    """

    # Sum the contributions from all terms
    total = 0

    # Add all non-trivial terms
    for i, term in enumerate(operator.terms):
        total += np.real(term.coefficient * expectation_values.values[i])

    value_estimate = ValueEstimate(total)
    return value_estimate


def evaluate_operator_list(
    operator_list: List[PauliRepresentation],
    expectation_values,
) -> ValueEstimate:
    """Evaluate the expectation value of an operator list using expectation values for
    the terms. The expectation values should be in the order given by the qubit operator
    list, and the value returned is the sum of all terms in the qubit operator list.

    Args:
        operator_list: the operator list
        expectation_values: the expectation values

    Returns:
        value_estimate: stores the value of the expectation and its precision
    """

    # Sum the contributions from all terms
    total = 0

    # Add all non-trivial terms
    term_index = 0
    for operator in operator_list:
        for term in operator.terms:
            total += np.real(term.coefficient * expectation_values.values[term_index])
            term_index += 1

    value_estimate = ValueEstimate(total)
    return value_estimate


def reverse_qubit_order(
    qubit_operator: PauliRepresentation, n_qubits: Optional[int] = None
):
    """Reverse the order of qubit indices in a qubit operator.

    Args:
        qubit_operator: the operator to be reversed
        n_qubits (int): total number of qubits. Needs to be provided when
            the size of the system of interest is greater than the size of qubit
            operator (optional)

    Returns:
        reversed_op: the reversed operator
    """

    reversed_op = PauliSum()

    if n_qubits is None:
        n_qubits = qubit_operator.n_qubits
    if n_qubits < qubit_operator.n_qubits:
        raise ValueError("Invalid number of qubits specified.")

    for term in qubit_operator.terms:
        new_term = {}
        for qubit_num, operator_str in term.operations:
            new_qubit_num = n_qubits - 1 - qubit_num
            new_term[new_qubit_num] = operator_str
        reversed_op += PauliTerm(new_term, term.coefficient)
    return reversed_op


def get_expectation_value(
    qubit_op: PauliRepresentation,
    wavefunction: Wavefunction,
    reverse_operator: bool = False,
) -> complex:
    """Get the expectation value of a qubit operator with respect to a wavefunction.

    Args:
        qubit_op: the operator
        wavefunction: the wavefunction
        reverse_operator: whether to reverse order of qubit operator
            before computing expectation value. This should be True if the convention
            of the basis states used for the wavefunction is the opposite of the one in
            the qubit operator. This is the case when the wavefunction uses
            Rigetti convention (https://arxiv.org/abs/1711.02086) of ordering qubits.
    Returns:
        the expectation value
    """
    n_qubits = wavefunction.amplitudes.shape[0].bit_length() - 1

    # Convert the qubit operator to a sparse matrix. Note that the qubit indices
    # must be reversed because OpenFermion and our Wavefunction use
    # different conventions for how to order the computational basis states!
    if reverse_operator:
        qubit_op = reverse_qubit_order(qubit_op, n_qubits=n_qubits)
    sparse_op = get_sparse_operator(qubit_op, n_qubits=n_qubits)

    # Computer the expectation value
    exp_val = expectation(sparse_op, wavefunction.amplitudes)
    return exp_val
