import unittest
from zquantum.core.circuit import Circuit, Qubit, Gate
from zquantum.vqe.utils import exponentiate_fermion_operator, build_hartree_fock_circuit


class TestVQEUtils:
    def test_build_hartree_fock_circuit_jordan_wigner(self):
        number_of_qubits = 4
        number_of_alpha_electrons = 1
        number_of_beta_electrons = 1
        transformation = "Jordan-Wigner"
        expected_circuit = Circuit()
        expected_circuit.qubits = [Qubit(0), Qubit(1), Qubit(2), Qubit(3)]
        expected_circuit.gates = [Gate("X", [Qubit(0)]), Gate("X", [Qubit(1)])]
        actual_circuit = build_hartree_fock_circuit(
            number_of_qubits,
            number_of_alpha_electrons,
            number_of_beta_electrons,
            transformation,
        )
        assert actual_circuit == expected_circuit

    def test_build_hartree_fock_circuit_bravyi_kitaev(self):
        number_of_qubits = 4
        number_of_alpha_electrons = 1
        number_of_beta_electrons = 1
        transformation = "Bravyi-Kitaev"
        expected_circuit = Circuit()
        expected_circuit.qubits = [Qubit(0), Qubit(1), Qubit(2), Qubit(3)]
        expected_circuit.gates = [Gate("X", [Qubit(0)])]
        actual_circuit = build_hartree_fock_circuit(
            number_of_qubits,
            number_of_alpha_electrons,
            number_of_beta_electrons,
            transformation,
        )
        assert actual_circuit == expected_circuit
