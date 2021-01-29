from zquantum.core.circuit import Circuit, Gate, Qubit
from zquantum.core.evolution import time_evolution
from zquantum.core.openfermion import qubitop_to_pyquilpauli
from openfermion import (
    jordan_wigner,
    bravyi_kitaev,
    FermionOperator,
    InteractionOperator,
    get_fermion_operator,
)
import numpy as np
from typing import Optional, Union


def exponentiate_fermion_operator(
    fermion_generator: Union[FermionOperator, InteractionOperator],
    transformation: str = "Jordan-Wigner",
    number_of_qubits: Optional[int] = None,
) -> Circuit:
    """Create a circuit corresponding to the exponentiation of an operator. Works only for antihermitian fermionic operators.

    Args:
        fermion_generator (openfermion.FermionOperator or
            openfermion.InteractionOperator): fermionic generator.
        transformation (str): The name of the qubit-to-fermion transformation to use.
        number_of_qubits (int|None): This can be used to force the number of qubits in the resulting operator
            above the number that appears in the input operator. Defaults to None and the number of qubits in
            the resulting operator will match the number that appears in the input operator.

    Returns:
        zquantum.core.circuit.Circuit: Circuit corresponding to the exponentiation of the
            transformed operator.
    """
    if transformation not in ["Jordan-Wigner", "Bravyi-Kitaev"]:
        raise RuntimeError(f"Unrecognized transformation {transformation}")

    # Transform generator to qubits
    if transformation == "Jordan-Wigner":
        qubit_generator = jordan_wigner(fermion_generator)
    else:
        if isinstance(fermion_generator, InteractionOperator):
            fermion_generator = get_fermion_operator(fermion_generator)
        qubit_generator = bravyi_kitaev(fermion_generator, n_qubits=number_of_qubits)

    for term in qubit_generator.terms:
        if not np.isclose(qubit_generator.terms[term].real, 0.0):
            raise RuntimeError("Transformed fermion_generator is not anti-hermitian.")
        qubit_generator.terms[term] = float(qubit_generator.terms[term].imag)
    qubit_generator.compress()

    # Quantum circuit implementing the excitation operators
    circuit = time_evolution(
        qubitop_to_pyquilpauli(qubit_generator), 1, method="Trotter", trotter_order=1
    )

    return circuit


def build_hartree_fock_circuit(
    number_of_qubits: int,
    number_of_alpha_electrons: int,
    number_of_beta_electrons: int,
    transformation: str,
    spin_ordering: str = "interleaved",
) -> Circuit:
    """Creates a circuit that prepares the Hartree-Fock state.

    Args:
        number_of_qubits (int): the number of qubits in the system.
        number_of_alpha_electrons (int): the number of alpha electrons in the system.
        number_of_beta_electrons (int): the number of beta electrons in the system.
        transformation (str): the Hamiltonian transformation to use.
        spin_ordering (str, optional): the spin ordering convention to use. Defaults to "interleaved".

    Returns:
        zquantum.core.circuit.Circuit: a circuit that prepares the Hartree-Fock state.
    """
    if spin_ordering != "interleaved":
        raise RuntimeError(
            f"{spin_ordering} is not supported at this time. Interleaved is the only supported spin-ordering."
        )
    circuit = Circuit()
    circuit.gates = []
    alpha_indexes = list(range(0, number_of_qubits, 2))
    beta_indexes = list(range(1, number_of_qubits, 2))
    index_list = []
    for index in alpha_indexes[:number_of_alpha_electrons]:
        index_list.append(index)
    for index in beta_indexes[:number_of_beta_electrons]:
        index_list.append(index)
    index_list.sort()
    op_list = [(x, 1) for x in index_list]
    fermion_op = FermionOperator(tuple(op_list), 1.0)
    if transformation == "Jordan-Wigner":
        transformed_op = jordan_wigner(fermion_op)
    elif transformation == "Bravyi-Kitaev":
        transformed_op = bravyi_kitaev(fermion_op, n_qubits=number_of_qubits)
    else:
        raise RuntimeError(
            f"{transformation} is not a supported transformation. Jordan-Wigner and Bravyi-Kitaev are supported at this time."
        )
    term = next(iter(transformed_op.terms.items()))
    for op in term[0]:
        if op[1] != "Z":
            circuit.gates.append(Gate("X", [Qubit(op[0])]))
    return circuit
