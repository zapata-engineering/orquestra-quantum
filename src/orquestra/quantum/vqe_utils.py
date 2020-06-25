from zquantum.core.circuit import Circuit, Gate, Qubit
from forestopenfermion import exponentiate
from openfermion import (
    jordan_wigner,
    bravyi_kitaev,
    FermionOperator,
    InteractionOperator,
)
from typing import Union


def exponentiate_fermion_operator(
    fermion_generator: Union[FermionOperator, InteractionOperator],
    transformation: str = "Jordan-Wigner",
) -> Circuit:
    """
    Create a pyQuil circuit corresponding to the exponentiation of an operator.

    Args:
        fermion_generator (openfermion.FermionOperator or 
            openfermion.InteractionOperator): fermionic generator.
        fermion_transform (str): The name of the qubit to fermion transformation
            to use.

    Returns:
        zquantum.core.circuit.Circuit: Circuit corresponding to the exponentiation of the
            transformed operator. 
    """
    if transformation not in ["Jordan-Wigner", "Bravyi-Kitaev"]:
        raise RuntimeError(f"Unrecognized transformation {transformation}")

    # Transform generator to qubits
    if transformation == "Jordan-Wigner":
        transformation = jordan_wigner
    elif transformation == "Bravyi-Kitaev":
        transformation = bravyi_kitaev

    qubit_generator = transformation(fermion_generator)

    for term in qubit_generator.terms:
        qubit_generator.terms[term] = float(qubit_generator.terms[term].imag)
    qubit_generator.compress()

    # Quantum circuit implementing the excitation operators
    circuit = exponentiate(qubit_generator)

    return circuit


def create_layer_of_gates(number_of_qubits: int, gate_name: str) -> Circuit:
    """
    Creates a circuit consisting of a single layer of specific gate.
    """
    circuit = Circuit()
    qubits = [Qubit(i) for i in range(0, number_of_qubits)]
    circuit.qubits = qubits
    gates = []
    for i in range(number_of_qubits):
        gates.append(Gate(gate_name, [qubits[i]]))
    circuit.gates = gates
    return circuit
