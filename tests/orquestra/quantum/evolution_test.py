################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
import numpy as np
import pytest
import sympy
from scipy.linalg import fractional_matrix_power

from orquestra.quantum import circuits
from orquestra.quantum.circuits import XX, YY, ZZ, Circuit
from orquestra.quantum.evolution import (
    _generate_circuit_sequence,
    time_evolution,
    time_evolution_derivatives,
    time_evolution_for_term,
)
from orquestra.quantum.operators import PauliSum, PauliTerm
from orquestra.quantum.utils import compare_unitary

TUPLE_TERM_TO_ORQUESTRA_GATE = {
    ((0, "X"), (1, "X")): XX,
    ((0, "Y"), (1, "Y")): YY,
    ((0, "Z"), (1, "Z")): ZZ,
}


def _orquestra_exponentiate_qubit_hamiltonian_term(term, time, n_steps):
    base_exponent = 2 * time / n_steps / np.pi

    base_gate = TUPLE_TERM_TO_ORQUESTRA_GATE[tuple(sorted(term.operations))]
    # This introduces a phase to the gate, but that's fine
    # since `compare_unitary` accounts for that
    mat = base_gate(np.pi)(0, 1).lifted_matrix(2)
    if term.coefficient.real == term.coefficient:
        coefficient = term.coefficient.real
    else:
        raise ValueError("Cannot raise matrix to the power of a complex coefficinet")
    zquant_mat = fractional_matrix_power(mat, coefficient * base_exponent)

    return zquant_mat


def _orquestra_exponentiate_hamiltonian(hamiltonian, time, n_steps):
    ops = []
    for term in hamiltonian.terms:
        mat = _orquestra_exponentiate_qubit_hamiltonian_term(term, time, n_steps)
        ops.append(
            circuits.CustomGateDefinition(
                gate_name="custom_a",
                matrix=sympy.Matrix(mat),
                params_ordering=(),
            )()(0, 1)
        )

    circuit = Circuit(operations=ops * n_steps)

    return circuit


@pytest.mark.parametrize(
    "term, time, expected_unitary",
    [
        (PauliTerm("X0*X1"), np.pi, -np.eye(4)),
        (
            PauliTerm("0.5*Y0*Y1"),
            np.pi,
            np.diag([1j, -1j, -1j, 1j])[::-1],
        ),
        (PauliTerm("Z0*Z1"), np.pi, -np.eye(4)),
    ],
)
class TestTimeEvolutionOfTerm:
    def test_evolving_pauli_term_with_numerical_time_gives_correct_unitary(
        self, term, time, expected_unitary
    ):
        actual_unitary = time_evolution_for_term(term, time).to_unitary()
        np.testing.assert_array_almost_equal(actual_unitary, expected_unitary)

    def test_evolving_pauli_term_with_symbolic_time_gives_correct_unitary(
        self, term, time, expected_unitary
    ):
        time_symbol = sympy.Symbol("t")
        symbol_map = {time_symbol: time}
        evolution_circuit = time_evolution_for_term(term, time_symbol)

        actual_unitary = evolution_circuit.bind(symbol_map).to_unitary()
        np.testing.assert_array_almost_equal(actual_unitary, expected_unitary)


def test_complex_coefficent_throws_error():
    with pytest.raises(ValueError):
        time_evolution_for_term(PauliTerm("X0", 1j), 1)


class TestTimeEvolutionOfConstantTerm:
    # This test is added to make sure that constant terms in qubit operators
    # do not cause errors.
    def test_evolving_constant_term_qubit_operator_gives_empty_circuit(self):
        evolution_circuit = time_evolution_for_term(PauliTerm.identity() * 3, np.pi)
        assert evolution_circuit == circuits.Circuit()


class TestTimeEvolutionOfPauliSum:
    @pytest.fixture
    def hamiltonian(self):
        return PauliSum("X0*X1 + 0.5*Y0*Y1 + 0.3*Z0*Z1")

    @pytest.mark.parametrize("time", [0.1, 0.4, 1.0])
    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_evolution_with_numerical_time_produces_correct_result(
        self, hamiltonian, time, order
    ):
        expected_orquestra_circuit = _orquestra_exponentiate_hamiltonian(
            hamiltonian, time, order
        )

        reference_unitary = expected_orquestra_circuit.to_unitary()
        unitary = time_evolution(hamiltonian, time, n_steps=order).to_unitary()

        assert compare_unitary(unitary, reference_unitary, tol=1e-10)

    @pytest.mark.parametrize("time_value", [0.1, 0.4, 1.0])
    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_time_evolution_with_symbolic_time_produces_correct_unitary(
        self, hamiltonian, time_value, order
    ):
        time_symbol = sympy.Symbol("t")
        symbols_map = {time_symbol: time_value}

        expected_orquestra_circuit = _orquestra_exponentiate_hamiltonian(
            hamiltonian, time_value, order
        )

        reference_unitary = expected_orquestra_circuit.to_unitary()

        unitary = (
            time_evolution(hamiltonian, time_symbol, n_steps=order)
            .bind(symbols_map)
            .to_unitary()
        )

        assert compare_unitary(unitary, reference_unitary, tol=1e-10)


class TestGeneratingCircuitSequence:
    @pytest.mark.parametrize(
        "repeated_circuit, different_circuit, length, position, expected_result",
        [
            (
                circuits.Circuit([circuits.X(0), circuits.Y(1)]),
                circuits.Circuit([circuits.Z(1)]),
                5,
                1,
                circuits.Circuit(
                    [
                        *[circuits.X(0), circuits.Y(1)],
                        circuits.Z(1),
                        *([circuits.X(0), circuits.Y(1)] * 3),
                    ]
                ),
            ),
            (
                circuits.Circuit([circuits.RX(0.5)(1)]),
                circuits.Circuit([circuits.CNOT(0, 2)]),
                3,
                0,
                circuits.Circuit(
                    [circuits.CNOT(0, 2), circuits.RX(0.5)(1), circuits.RX(0.5)(1)]
                ),
            ),
            (
                circuits.Circuit([circuits.RX(0.5)(1)]),
                circuits.Circuit([circuits.CNOT(0, 2)]),
                3,
                2,
                circuits.Circuit(
                    [
                        circuits.RX(0.5)(1),
                        circuits.RX(0.5)(1),
                        circuits.CNOT(0, 2),
                    ]
                ),
            ),
        ],
    )
    def test_produces_correct_sequence(
        self, repeated_circuit, different_circuit, position, length, expected_result
    ):
        actual_result = _generate_circuit_sequence(
            repeated_circuit, different_circuit, length, position
        )

        assert actual_result == expected_result

    @pytest.mark.parametrize("length, invalid_position", [(5, 5), (4, 6)])
    def test_raises_error_if_position_is_larger_or_equal_to_length(
        self, length, invalid_position
    ):
        repeated_circuit = circuits.Circuit([circuits.X(0)])
        different_circuit = circuits.Circuit([circuits.Y(0)])

        with pytest.raises(ValueError):
            _generate_circuit_sequence(
                repeated_circuit, different_circuit, length, invalid_position
            )


class TestTimeEvolutionDerivatives:
    @pytest.fixture
    def hamiltonian(self):
        return PauliSum("X0*X1 + 0.5*Y0*Y1 + 0.3*Z0*Z1")

    @pytest.mark.parametrize("time", [0.4, sympy.Symbol("t")])
    def test_gives_correct_number_of_derivatives_and_factors(self, time, hamiltonian):

        order = 3
        reference_factors_1 = [1.0 / order, 0.5 / order, 0.3 / order] * 3
        reference_factors_2 = [-1.0 * x for x in reference_factors_1]

        derivatives, factors = time_evolution_derivatives(
            hamiltonian, time, n_steps=order
        )

        assert len(derivatives) == order * 2 * len(hamiltonian.terms)
        assert len(factors) == order * 2 * len(hamiltonian.terms)
        assert factors[0:18:2] == reference_factors_1
        assert factors[1:18:2] == reference_factors_2
