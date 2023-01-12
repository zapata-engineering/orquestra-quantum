################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
"""Test cases for _gates module."""
from unittest.mock import Mock

import numpy as np
import pytest
import sympy

from orquestra.quantum.circuits import _builtin_gates, _gates
from orquestra.quantum.circuits._gates import (
    Exponential,
    GateOperation,
    MatrixFactoryGate,
    Power,
)

GATES_REPRESENTATIVES = [
    _builtin_gates.X,
    _builtin_gates.Y,
    _builtin_gates.Z,
    _builtin_gates.T,
    _builtin_gates.H,
    _builtin_gates.I,
    _builtin_gates.RX(sympy.Symbol("theta")),
    _builtin_gates.RY(0.5),
    _builtin_gates.RZ(0),
    _builtin_gates.PHASE(sympy.pi / 5),
    _builtin_gates.U3(np.pi, sympy.pi / 2, sympy.Symbol("x")),
    _builtin_gates.CZ,
    _builtin_gates.CNOT,
    _builtin_gates.SWAP,
    _builtin_gates.ISWAP,
    _builtin_gates.XX(sympy.cos(sympy.Symbol("phi"))),
    _builtin_gates.YY(sympy.pi),
    _builtin_gates.ZZ(sympy.Symbol("x") + sympy.Symbol("y")),
    _builtin_gates.CPHASE(1.5),
]

POWER_GATE_EXPONENTS = [-2.0, 0, 0.5, 1.0]


def example_one_qubit_matrix_factory(a, b):
    return sympy.Matrix([[a, b], [b, a]])


def example_two_qubit_matrix_factory(a, b, c):
    return sympy.Matrix([[a, 0, 0, 0], [0, b, 0, 0], [0, 0, c, 0], [0, 0, 0, 1]])


class TestMatrixFactoryGate:
    @pytest.mark.parametrize(
        "params, factory, num_qubits",
        [
            ((0.5, sympy.Symbol("theta")), example_one_qubit_matrix_factory, 1),
            (
                (sympy.Symbol("alpha"), sympy.Symbol("beta"), 1),
                example_two_qubit_matrix_factory,
                2,
            ),
        ],
    )
    def test_constructs_its_matrix_by_calling_factory_with_bound_parameter(
        self, params, factory, num_qubits
    ):
        wrapped_factory = Mock(wraps=factory)
        gate = MatrixFactoryGate("U", wrapped_factory, params, num_qubits)
        assert gate.matrix == factory(*params)
        wrapped_factory.assert_called_once_with(*params)

    def test_binding_parameters_creates_new_instance_with_substituted_free_params(self):
        gamma, theta, x, y = sympy.symbols("gamma, theta, x, y")
        params = (theta, x + y)
        gate = MatrixFactoryGate("U", example_one_qubit_matrix_factory, params, 1)

        new_gate = gate.bind({theta: 0.5, x: gamma, y: 3})

        assert new_gate.name == gate.name
        assert new_gate.matrix_factory == gate.matrix_factory
        assert new_gate.num_qubits == gate.num_qubits
        assert new_gate.params == (0.5, gamma + 3)

    def test_binding_parameters_with_symbol_outside_of_free_symbols_does_not_raise(
        self,
    ):
        gamma, theta = sympy.symbols("gamma, theta")
        params = (theta, 2 * theta)
        gate = MatrixFactoryGate("U", example_one_qubit_matrix_factory, params, 1)

        new_gate = gate.bind({gamma: 0.5, theta: 1})

        assert new_gate.params == (1, 2)

    def test_binding_parameters_does_not_change_parameters_without_free_symbols(self):
        theta = sympy.Symbol("theta")
        gate = MatrixFactoryGate("V", example_one_qubit_matrix_factory, (1, 2), 1)

        new_gate = gate.bind({theta: 5.0})

        assert new_gate.params == (1, 2)

    def test_replace_parameters_correctly_gives_instance_with_correctly_set_parameters(
        self,
    ):
        theta = sympy.Symbol("theta")
        gate = MatrixFactoryGate("V", example_one_qubit_matrix_factory, (1, 2), 1)

        new_gate = gate.replace_params((theta, 0.5))

        assert new_gate == MatrixFactoryGate(
            "V", example_one_qubit_matrix_factory, (theta, 0.5), 1
        )

    def test_daggers_matrix_is_adjoint_of_original_gates_matrix(self):
        gate = MatrixFactoryGate("V", example_one_qubit_matrix_factory, (1, 2), 1)
        assert gate.dagger.matrix == gate.matrix.adjoint()

    def test_matrix_exponential_is_exponential_of_original_gates_matrix(self):
        gate = MatrixFactoryGate("V", example_one_qubit_matrix_factory, (1, 2), 1)
        assert gate.exp.matrix == gate.matrix.exp()

    def test_dagger_has_the_same_params_and_num_qubits_as_wrapped_gate(self):
        gate = MatrixFactoryGate(
            "U", example_two_qubit_matrix_factory, (0.5, 0.1, sympy.Symbol("a")), 2
        )
        assert gate.dagger.num_qubits == gate.num_qubits
        assert gate.dagger.params == gate.params

    def test_dagger_of_hermitian_gate_is_the_same_gate(self):
        gate = MatrixFactoryGate(
            "V", example_one_qubit_matrix_factory, (1, 0), 1, is_hermitian=True
        )
        assert gate.dagger is gate

    def test_power_of_dagger_is_dagger_wrapped_by_power(self):
        gate = MatrixFactoryGate("V", example_one_qubit_matrix_factory, (1, 0), 1)
        assert gate.dagger.power(0.5) == Power(gate.dagger, 0.5)

    def test_exponential_of_dagger_is_dagger_wrapped_by_exponential(self):
        gate = MatrixFactoryGate("V", example_one_qubit_matrix_factory, (1, 0), 1)
        assert gate.dagger.exp == Exponential(gate.dagger)

    def test_binding_gates_in_dagger_is_propagated_to_wrapped_gate(self):
        theta = sympy.Symbol("theta")
        gate = MatrixFactoryGate("V", example_one_qubit_matrix_factory, (theta, 0), 1)

        assert gate.dagger.bind({theta: 0.5}) == gate.bind({theta: 0.5}).dagger

    def test_dagger_of_dagger_is_the_same_as_original_gate(self):
        gate = MatrixFactoryGate("V", example_one_qubit_matrix_factory, (1, 0), 1)
        assert gate.dagger.dagger is gate

    def test_applying_dagger_and_replacing_parameters_commutes(self):
        gate = MatrixFactoryGate("V", example_one_qubit_matrix_factory, (1, 0), 1)
        new_params = (sympy.Symbol("theta"), 4.2)
        assert (
            gate.dagger.replace_params(new_params)
            == gate.replace_params(new_params).dagger
        )

    def test_applying_gate_returns_operation_with_correct_gate_and_indices(self):
        theta = sympy.Symbol("theta")
        gamma = sympy.Symbol("gamma")
        gate = MatrixFactoryGate(
            "A", example_two_qubit_matrix_factory, (theta, gamma, 42), 2
        )
        operation = gate(4, 1)

        assert operation.gate == gate
        assert operation.qubit_indices == (4, 1)

    def test_str_on_dagger_gives_correct_representation(self):
        gate = MatrixFactoryGate("V", example_one_qubit_matrix_factory, (1, 0), 1)
        assert str(gate.dagger) == "V†(1, 0)"

    def test_wrapping_two_times_gives_correct_string(self):
        phi = sympy.Symbol("phi")
        op = MatrixFactoryGate("U", example_one_qubit_matrix_factory, (phi, 1), 1)
        T_op = _builtin_gates.T  # need a gate without params for exponentiation

        # assert all permutations produce correct string
        assert str(op.controlled(2).dagger(0)) == "c-c-U†(phi, 1)(0)"
        assert str(T_op.controlled(2).exp(0)) == "exp^{c-c-T}(0)"
        assert str(T_op.controlled(2).power(3)(0)) == "c-c-T^3(0)"
        assert str(op.controlled(2).controlled(2)(0)) == "c-c-c-c-U(phi, 1)(0)"

        assert str(op.dagger.controlled(2)(0)) == "c-c-U†(phi, 1)(0)"
        assert str(T_op.dagger.exp(0)) == "exp^T†(0)"
        assert str(T_op.dagger.power(3)(0)) == "T†^3(0)"
        assert str(op.dagger.dagger(0)) == "U(phi, 1)(0)"

        assert str(T_op.exp.dagger(0)) == "exp^T†(0)"
        assert str(T_op.exp.controlled(2)(0)) == "c-c-exp^T(0)"
        assert str(T_op.exp.power(3)(0)) == "{exp^T}^3(0)"
        assert str(T_op.exp.exp(0)) == "exp^exp^T(0)"

        assert str(T_op.power(3).dagger(0)) == "T†^3(0)"
        assert str(T_op.power(3).controlled(2)(0)) == "c-c-T^3(0)"
        assert str(T_op.power(3).exp(0)) == "exp^{T^3}(0)"
        assert str(T_op.power(3).power(3)(0)) == "T^3^3(0)"

    def test_wrapping_three_times_gives_correct_string(self):
        T_op = _builtin_gates.T  # need a gate without params for exponentiation

        # assert some permutations because there are too many to test exhaustively
        assert str(T_op.controlled(2).dagger.exp(0)) == "exp^{c-c-T†}(0)"
        assert str(T_op.dagger.controlled(2).power(3)(0)) == "c-c-T†^3(0)"
        assert str(T_op.exp.controlled(2).exp(0)) == "exp^{c-c-exp^T}(0)"


@pytest.mark.parametrize("gate", GATES_REPRESENTATIVES)
class TestControlledGate:
    def test_num_qubits_equal_to_wrapped_gates_num_qubits_plus_num_controlled_qubits(
        self, gate
    ):
        assert gate.controlled(3).num_qubits == gate.num_qubits + 3

    def test_has_matrix_with_eye_and_wrapped_gates_matrix_as_bottom_left_block(
        self, gate
    ):
        controlled_gate = gate.controlled(2)
        n = gate.matrix.shape[0]
        assert gate.matrix.shape[1] == n
        assert controlled_gate.matrix[0:-n, 0:-n] == sympy.eye(
            2**controlled_gate.num_qubits - n
        )
        assert controlled_gate.matrix[-n:, -n:] == gate.matrix

    def test_controlled_of_controlled_gate_has_summed_number_of_control_qubits(
        self, gate
    ):
        controlled_gate = gate.controlled(2)
        double_controlled_gate = controlled_gate.controlled(3)

        assert double_controlled_gate.wrapped_gate == gate
        assert double_controlled_gate.num_qubits == gate.num_qubits + 2 + 3
        assert double_controlled_gate.num_control_qubits == 2 + 3
        assert double_controlled_gate.matrix.shape == 2 * (
            2 ** (gate.num_qubits + 2 + 3),
        )

    def test_has_the_same_parameters_as_wrapped_gate(self, gate):
        controlled_gate = gate.controlled(4)

        assert controlled_gate.params == gate.params

    def test_dagger_of_controlled_gate_is_controlled_gate_wrapping_dagger(self, gate):
        controlled_gate = gate.controlled(4)

        assert controlled_gate.dagger == gate.dagger.controlled(4)

    def test_exp_of_controlled_gate_is_not_controlled_gate_wrapping_exp(self, gate):
        if len(gate.free_symbols) == 0:
            controlled_gate = gate.controlled(2)
            assert controlled_gate.exp != gate.exp.controlled(2)

    def test_power_of_controlled_gate_is_controlled_gate_wrapping_power(self, gate):
        if len(gate.free_symbols) == 0:
            controlled_gate = gate.controlled(2)
            assert controlled_gate.power(0.5) == gate.power(0.5).controlled(2)

    def test_binding_parameters_in_control_gate_is_propagated_to_wrapped_gate(
        self, gate
    ):
        controlled_gate = gate.controlled(2)
        symbols_map = {sympy.Symbol("theta"): 0.5, sympy.Symbol("x"): 3}
        assert controlled_gate.bind(symbols_map) == gate.bind(symbols_map).controlled(2)

    def test_constructing_controlled_gate_and_replacing_parameters_commute(self, gate):
        controlled_gate = gate.controlled(2)
        new_params = tuple(3 * param for param in controlled_gate.params)

        assert controlled_gate.replace_params(new_params) == gate.replace_params(
            new_params
        ).controlled(2)

    def test_constructing_controlled_gate_with_zero_control_raises_error(self, gate):
        with pytest.raises(ValueError):
            gate.controlled(0)

    def test_str_gives_correct_string_for_one_control(self, gate):
        controlled_gate = gate.controlled(1)
        assert str(controlled_gate) == "c-" + str(gate)

    def test_str_gives_correct_string_for_multiple_controls(self, gate):
        controlled_gate_2 = gate.controlled(2)
        controlled_gate_5 = gate.controlled(5)
        assert str(controlled_gate_2) == "c-" * 2 + str(gate)
        assert str(controlled_gate_5) == "c-" * 5 + str(gate)

    def test_str_gives_correct_string_for_stacking_controls(self, gate):
        double_controlled_gate = gate.controlled(1).controlled(1)
        assert str(double_controlled_gate) == "c-" * 2 + str(gate)


@pytest.mark.parametrize("gate", GATES_REPRESENTATIVES)
@pytest.mark.parametrize("exponent", POWER_GATE_EXPONENTS)
class TestPowerGate:
    def test_constructing_a_power_gate_with_free_symbols_raises_error(
        self, gate, exponent
    ):
        if len(gate.free_symbols) > 0:
            with pytest.raises(ValueError):
                gate.power(exponent)

    def test_power_gate_naming_scheme(self, gate, exponent):
        if len(gate.free_symbols) == 0:
            power_gate = gate.power(exponent)
            assert power_gate.name == f"{gate.name}{_gates.POWER_GATE_SYMBOL}{exponent}"

    def test_has_same_parameters_as_wrapped_gate(self, gate, exponent):
        if len(gate.free_symbols) == 0:
            assert gate.power(exponent).params == gate.params

    def test_has_same_free_symbols_as_wrapped_gate(self, gate, exponent):
        if len(gate.free_symbols) == 0:
            assert gate.power(exponent).free_symbols == gate.free_symbols

    def test_has_same_number_of_qubits_as_wrapped_gate(self, gate, exponent):
        if len(gate.free_symbols) == 0:
            assert gate.power(exponent).num_qubits == gate.num_qubits

    def test_has_matrix_equal_to_wrapped_gate_matrix_exponentiated(
        self, gate, exponent
    ):
        if len(gate.free_symbols) == 0:
            wrapped_gate_matrix_exponentiated = gate.matrix**exponent
            assert gate.power(exponent).matrix == wrapped_gate_matrix_exponentiated

    def test_dagger_of_power_gate_power_gate_of_dagger(self, gate, exponent):
        if len(gate.free_symbols) == 0:
            power_gate = gate.power(exponent)
            assert power_gate.dagger == gate.dagger.power(exponent)

    def test_creating_power_gate_from_power_gate(self, gate, exponent):
        if len(gate.free_symbols) == 0:
            power_gate = gate.power(exponent)
            powered_power_gate = power_gate.power(exponent)
            assert powered_power_gate.matrix == power_gate.matrix**exponent

    def test_parameter_binding_not_implemented_for_power_gates(self, gate, exponent):
        if len(gate.free_symbols) == 0:
            power_gate = gate.power(exponent)
            symbols_map = {sympy.Symbol("theta"): 0.5, sympy.Symbol("x"): 3}
            with pytest.raises(NotImplementedError):
                power_gate.bind(symbols_map)

    def test_constructing_power_gate_and_replacing_parameters_commute(
        self, gate, exponent
    ):
        if len(gate.free_symbols) == 0:
            power_gate = gate.power(exponent)
            new_params = tuple(3 * param for param in power_gate.params)

            assert power_gate.replace_params(new_params) == gate.replace_params(
                new_params
            ).power(exponent)

    def test_str_gives_correct_string(self, gate, exponent):
        if len(gate.free_symbols) == 0:
            power_gate = gate.power(exponent)
            correct_str = str(gate) + f"{_gates.POWER_GATE_SYMBOL}{exponent}"
            assert str(power_gate) == correct_str


@pytest.mark.parametrize("gate", GATES_REPRESENTATIVES[:10])
class TestGateExponential:
    def test_constructing_a_gate_exponential_with_free_symbols_raises_error(self, gate):
        if len(gate.free_symbols) > 0:
            with pytest.raises(ValueError):
                gate.exp

    def test_gate_exponential_naming_scheme(self, gate):
        if len(gate.free_symbols) == 0:
            gate_exponential = gate.exp
            assert gate_exponential.name == _gates.EXPONENTIAL_GATE_NAME

    def test_has_same_parameters_as_wrapped_gate(self, gate):
        if len(gate.free_symbols) == 0:
            assert gate.exp.params == gate.params

    def test_has_same_free_symbols_as_wrapped_gate(self, gate):
        if len(gate.free_symbols) == 0:
            assert gate.exp.free_symbols == gate.free_symbols

    def test_has_same_number_of_qubits_as_wrapped_gate(self, gate):
        if len(gate.free_symbols) == 0:
            assert gate.exp.num_qubits == gate.num_qubits

    def test_matrix_exponential_equal_to_wrapped_gate_matrix_exponential(self, gate):
        if len(gate.free_symbols) == 0 and gate.name != "T":
            wrapped_gate_matrix_exponential = gate.matrix.exp()
            assert gate.exp.matrix == wrapped_gate_matrix_exponential

    def test_dagger_of_gate_exponential_gate_exponential_of_dagger(self, gate):
        if len(gate.free_symbols) == 0 and gate.name != "T":
            assert gate.exp.dagger == gate.dagger.exp

    def test_power_of_gate_exponential_gate_exponential_of_power(self, gate):
        if len(gate.free_symbols) == 0 and gate.name != "T":
            assert gate.exp.power(2.0) != gate.power(2.0).exp

    def test_parameter_binding_not_implemented_for_gate_exponential(self, gate):
        if len(gate.free_symbols) == 0:
            gate_exponential = gate.exp
            symbols_map = {sympy.Symbol("theta"): 0.5, sympy.Symbol("x"): 3}
            with pytest.raises(NotImplementedError):
                gate_exponential.bind(symbols_map)

    def test_constructing_gate_exponential_and_replacing_parameters_commute(self, gate):
        if len(gate.free_symbols) == 0:
            gate_exponential = gate.exp
            new_params = tuple(3 * param for param in gate_exponential.params)

            assert (
                gate_exponential.replace_params(new_params)
                == gate.replace_params(new_params).exp
            )

    def test_str_gives_correct_string(self, gate):
        if len(gate.free_symbols) == 0:
            assert str(gate.exp) == "exp^" + str(gate)


@pytest.mark.parametrize("gate", GATES_REPRESENTATIVES)
class TestGateOperation:
    def test_bound_symbols_are_not_present_in_gate_parameters(self, gate):
        op = GateOperation(gate, tuple(range(gate.num_qubits)))
        symbols_map = {sympy.Symbol("phi"): 0.5, sympy.Symbol("y"): 1.1}
        assert all(
            symbol not in sympy.sympify(param).atoms(sympy.Symbol)
            for symbol in symbols_map
            for param in op.bind(symbols_map).params
        )

    def test_replacing_parameters_constructs_operation_of_gate_with_new_parameters(
        self, gate
    ):
        op = GateOperation(gate, tuple(range(gate.num_qubits)))
        new_params = tuple(-1 * param for param in op.params)

        assert op.replace_params(new_params).params == new_params

    def test_free_symbols_of_gate_operation_are_the_same_as_the_ones_in_wrapped_gate(
        self, gate
    ):
        op = GateOperation(gate, tuple(range(gate.num_qubits)))
        assert op.free_symbols == gate.free_symbols

    def test_cannot_be_applied_to_vector_of_not_power_of_two_length(self, gate):
        state_vector = np.array([0.1 for _ in range(2**gate.num_qubits + 1)])
        with pytest.raises(ValueError):
            GateOperation(gate, tuple(range(gate.num_qubits))).apply(state_vector)
