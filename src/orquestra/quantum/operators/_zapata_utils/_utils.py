from typing import Optional

from ...wavefunction import Wavefunction
from .._openfermion_utils.sparse_tools import expectation, get_sparse_operator
from .._pauli_operators import PauliRepresentation, PauliSum, PauliTerm


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
