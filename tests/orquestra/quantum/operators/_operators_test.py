################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
from itertools import product
from typing import List
from unittest.mock import Mock

import pytest

from orquestra.quantum.circuits import Circuit, X, Y, Z
from orquestra.quantum.operators._pauli_operators import PauliSum, PauliTerm


@pytest.fixture
def pauli_term():
    correct_list = [("X", 0), ("Y", 1), ("Z", 12)]
    return PauliTerm.from_iterable(correct_list, coefficient=2.0)


@pytest.fixture
def pauli_sum():
    return 0.5 * PauliTerm("X0") + 0.5j * PauliTerm("Y0")


class TestPauliTermInitialization:
    @pytest.mark.parametrize(
        "operator_dict, coefficient",
        [({1: "X"}, 1.5), ({0: "X", 1: "X"}, 0.5), ({0: "Y", 10: "Z"}, 0.5 + 0.5j)],
    )
    def test_term_can_be_initialized_with_dictionary(self, operator_dict, coefficient):
        term = PauliTerm(operator_dict, coefficient)
        assert term.operations == frozenset(operator_dict.items())
        assert term.coefficient == coefficient
        assert term.qubits == frozenset(operator_dict)

    @pytest.mark.parametrize(
        "pauli_str, coefficient, qubit_index, operator",
        [
            ("X0", 1.0, 0, "X"),
            ("Z0", -1j, 0, "Z"),
            ("z1", -2.0, 1, "Z"),
            ("y2", 1.5, 2, "Y"),
            ("X123", 4 + 0.5j, 123, "X"),
        ],
    )
    def test_term_can_be_initialized_using_single_operator_and_coefficient(
        self, pauli_str, coefficient, qubit_index, operator
    ):
        term = PauliTerm(pauli_str, coefficient)
        assert term.operations == frozenset([(qubit_index, operator)])
        assert term.coefficient == coefficient
        assert term.qubits == {qubit_index}

    @pytest.mark.parametrize(
        "pauli_str, coefficient, qubit_indices, operators",
        [
            ("X0 * Z1", -1.0, (0, 1), ("X", "Z")),
            ("X3 * Y12", -0.5, (3, 12), ("X", "Y")),
        ],
    )
    def test_term_can_be_initialized_using_multiple_operators_and_coefficient(
        self, pauli_str, coefficient, qubit_indices, operators
    ):
        term = PauliTerm(pauli_str, coefficient)
        assert term.operations == frozenset(zip(qubit_indices, operators))
        assert term.coefficient == coefficient
        assert term.qubits == frozenset(qubit_indices)

    @pytest.mark.parametrize(
        "pauli_str, coefficient, qubit_indices, operators",
        [
            ("-1.0 * X0 * Z1", -1.0, (0, 1), ("X", "Z")),
            ("-0.5 * X3 * Y12", -0.5, (3, 12), ("X", "Y")),
            ("(1-0.5j) * X3 * Y12", 1 - 0.5j, (3, 12), ("X", "Y")),
            ("(2 + 3j) * y0 * z2", 2 + 3j, (0, 2), ("Y", "Z")),
        ],
    )
    def test_term_can_be_initialized_by_passing_operators_and_coef_in_string(
        self, pauli_str, coefficient, qubit_indices, operators
    ):
        term = PauliTerm(pauli_str)
        assert term.operations == frozenset(zip(qubit_indices, operators))
        assert term.coefficient == coefficient
        assert term.qubits == frozenset(qubit_indices)

    @pytest.mark.parametrize(
        "pauli_str, expected_ops",
        [
            ("I0", {}),
            ("I12", {}),
            ("X0 * I1 * I34", {(0, "X")}),
            ("X0 * Z1 * I34", {(0, "X"), (1, "Z")}),
        ],
    )
    def test_identity_operators_are_not_stored_in_iterm(self, pauli_str, expected_ops):
        assert PauliTerm(pauli_str).operations == frozenset(expected_ops)

    @pytest.mark.parametrize(
        "pauli_str", ["1 X", "X - 1254", "A0", "5.0 + Z1", "0.3+0.5j * X0", "X0 Z1"]
    )
    def test_term_cannot_be_constructed_from_badly_formatted_string(self, pauli_str):
        with pytest.raises(ValueError) as e:
            PauliTerm(pauli_str)

        assert "Badly formatted" in str(e.value)

    @pytest.mark.parametrize("operator_dict", [{0: "XX"}, {0: "X", 1: "A"}])
    def test_term_cannot_be_constructed_from_dictionary_containing_incorrect_operators(
        self, operator_dict
    ):
        with pytest.raises(ValueError):
            PauliTerm(operator_dict)

    @pytest.mark.parametrize("pauli_str", ["X0 * Y0", "X0 * Y1 * Z1"])
    def test_term_cannot_be_constructed_if_duplicate_qubits_are_present_in_pauli_str(
        self, pauli_str
    ):
        with pytest.raises(ValueError):
            PauliTerm(pauli_str)

    @pytest.mark.parametrize(
        "pauli_str, coefficient", [("0.5 * X0", 0.5), ("2.0 * Y12", 3.0)]
    )
    def test_term_cannot_be_constructed_if_coefficient_is_passed_via_both_ways(
        self, pauli_str, coefficient
    ):
        with pytest.raises(ValueError) as e:
            PauliTerm(pauli_str, coefficient)

        assert "Coefficient can be provided either in an argument " in str(e.value)

    @pytest.mark.parametrize(
        "term",
        [
            PauliTerm("X0", -5.0),
            PauliTerm("-5.0 * Z1"),
            PauliTerm("2.0 * X0 * Y2 * Z3"),
            PauliTerm("X0 * Y2 * Z3", 2.0),
        ],
    )
    def test_creating_term_from_its_str_representation_gives_the_same_term(self, term):
        assert PauliTerm(str(term)) == term


class TestConstructingPauliTermFromIterable:
    @pytest.mark.parametrize(
        "ops, coefficient",
        [
            ([("X", 0), ("Y", 1), ("Z", 12)], -0.5),
            ({("X", 0), ("Y", 1), ("Z", 12)}, -0.5),
        ],
    )
    def test_term_constructed_from_iterable_has_expected_operators_and_coefficient(
        self, ops, coefficient
    ):
        term = PauliTerm.from_iterable(ops, coefficient)
        # The reverse (pair[::-1]) is from the fact that from_iterable accepts input in
        # the reverse direction then the operations produces it.
        assert term.operations == frozenset([pair[::-1] for pair in ops])
        assert term.coefficient == coefficient

    def test_identity_term_is_constructed_from_empty_iterable(self):
        term = PauliTerm.from_iterable([])
        assert term.operations == frozenset()
        assert term.coefficient == 1.0
        assert term == PauliTerm.identity()

        term_with_coeff = PauliTerm.from_iterable([], 2.0 + 3.0j)
        assert term_with_coeff.operations == frozenset()
        assert term_with_coeff.coefficient == 2.0 + 3.0j
        assert term_with_coeff == PauliTerm.identity() * (2.0 + 3.0j)

    def test_term_cannot_be_constructed_from_iterable_of_incorrectly_shaped_tuples(
        self,
    ):
        with pytest.raises(ValueError) as e:
            PauliTerm.from_iterable([("X0")])

        assert "list can only contain" in str(e.value)

    def test_term_cannot_be_initialized_if_any_index_in_list_is_incorrect(self):
        with pytest.raises(ValueError) as e:
            PauliTerm.from_iterable([("X", -1)])

        assert "Invalid qubit index" in str(e.value)

    def test_term_cannot_be_constructed_if_qubit_indices_are_duplicate(self):
        with pytest.raises(ValueError) as e:
            PauliTerm.from_iterable([("X", 0), ("Y", 0)])
        assert "Duplicate" in str(e.value)


class TestPauliTermIdentity:
    def test_identity_stores_coefficient_of_one_and_no_operators(self):
        identity_term = PauliTerm.identity()

        assert len(identity_term) == 0
        assert identity_term.coefficient == 1.0

    @pytest.mark.parametrize(
        "term",
        [PauliTerm("X1"), PauliTerm("X0 * X1", -0.5), PauliTerm("(2.5+1j) * X0 * Z12")],
    )
    def test_identity_is_neutral_element_for_pauli_terms_multiplication(self, term):
        assert PauliTerm.identity() * term == term
        assert term * PauliTerm.identity() == term


class TestCopyingPauliTerm:
    def test_copying_term_returns_new_object(self):
        original = PauliTerm("X0", 2.0)
        copy = original.copy(new_coefficient=3.0)

        assert copy is not original
        assert copy._ops is not original._ops

    def test_copy_of_term_has_the_same_coef_if_new_coef_is_not_provided(self):
        assert PauliTerm("(2.0+3j) * Y12").copy().coefficient == 2.0 + 3j

    def test_copying_terms_with_new_coefficient_does_not_modify_original_term(self):
        original = PauliTerm("Y0 * Z10", -1.0)
        copy = original.copy(new_coefficient=2.5)

        assert original.coefficient == -1.0
        assert copy.coefficient == 2.5


class TestPauliOperatorProperties:
    @pytest.mark.parametrize(
        "term",
        [
            PauliTerm("Z0", -2),
            PauliTerm("-0.5 * Z1 * Z3"),
            PauliTerm.from_iterable([("Z", 3), ("Z", 4)], 5.0),
        ],
    )
    def test_term_is_ising_if_it_contains_only_z_operators(self, term):
        assert term.is_ising

    @pytest.mark.parametrize(
        "term", [PauliTerm("X0"), PauliTerm("Z0 * X1"), PauliTerm("X1 * Y2 * Z3")]
    )
    def test_term_is_not_ising_if_any_of_its_nontrivial_operators_are_not_z(self, term):
        assert not term.is_ising

    @pytest.mark.parametrize(
        "term", [PauliTerm("I0", 2.0), PauliTerm({}), PauliTerm({0: "I", 1: "I"})]
    )
    def test_term_is_constant_if_it_comprises_only_trivial_operators(self, term):
        assert term.is_constant

    @pytest.mark.parametrize(
        "term",
        [
            PauliTerm("X0"),
            PauliTerm("2.5 * X0 * Z2 * Y10"),
            PauliTerm("0.5 * I0 * I1 * Z3"),
        ],
    )
    def test_term_is_not_constant_if_it_comprises_nontrivial_operators(self, term):
        assert not term.is_constant

    @pytest.mark.parametrize(
        "operator,n_qubits",
        [
            (PauliTerm("X0"), 1),
            (PauliSum("2"), 0),
            (PauliTerm("Z0*Z3"), 4),
            (PauliTerm("X0*X1"), 2),
            (PauliSum("2*Z2 + X5"), 6),
        ],
    )
    def test_n_qubits(self, operator, n_qubits):
        assert operator.n_qubits == n_qubits


class TestPauliTermToCircuitConversion:
    def test_circuit_constructed_from_pauli_term_is_cached(self):
        term = PauliTerm("X0 * Z1 * Y3", -1.0)
        assert not hasattr(term, "_circuit")
        circuit = term.circuit
        assert hasattr(term, "_circuit")
        assert term.circuit is circuit

    def test_circuit_obtained_from_term_has_correct_gates(self):
        term = PauliTerm("-3.5 * X0 * Z1 * Y3")
        assert term.circuit == Circuit([X(0), Z(1), Y(3)])


class TestPauliTermIndexingAndIteration:
    @pytest.mark.parametrize(
        "term", [PauliTerm("X0 * I2 * Y3 * Z4"), PauliTerm("X0 * Y3 * Z4")]
    )
    def test_accessing_undefined_or_trivial_indices_warns_and_returns_identity(
        self, term
    ):
        with pytest.warns(UserWarning):
            returned_op = term[2]

        assert returned_op == "I"

    def test_iterative_over_terms_yields_correct_pairs_of_operator_and_index(self):
        term = PauliTerm("X0 * Y1 * Z12")
        expected_items = [("X", 0), ("Y", 1), ("Z", 12)]

        assert list(term) == expected_items


class TestPauliTermAlgebra:
    def test_pauliterm_equality(self, pauli_term):
        # Check copying creates an equal term
        copied_term = pauli_term.copy()

        assert copied_term == pauli_term
        assert pauli_term == PauliSum([copied_term])

        # Check __eq__ of PauliSum is called
        old_paulisum_eq = PauliSum.__eq__
        PauliSum.__eq__ = Mock()
        pauli_term == PauliSum([copied_term])
        PauliSum.__eq__.assert_called()
        PauliSum.__eq__.assert_called_with(pauli_term)
        PauliSum.__eq__ = old_paulisum_eq

        # Check removing a term makes the terms unequal
        copied_term *= PauliTerm("X0")

        assert copied_term != pauli_term

        # Check constant term is comparable with numbers
        assert PauliTerm.identity() == 1.0

        # Check error is raised for invalid type
        with pytest.raises(TypeError):
            pauli_term == "X0*Y1*Z12"

    def test_pauliterm_equality_zero(self, pauli_term: PauliTerm):
        # Test that various representations of zero are equal
        zero_paulisum = PauliSum()
        zero_identity_term = PauliTerm("I0", 0)
        mul_0 = pauli_term * 0
        copy_0 = pauli_term.copy(0)
        other_0 = PauliTerm("X5*Z11", 0)
        paulisum_with_zero_term = PauliSum([zero_identity_term]).simplify()
        subtracted = pauli_term - pauli_term

        assert (
            zero_paulisum
            == zero_identity_term
            == mul_0
            == copy_0
            == other_0
            == paulisum_with_zero_term
            == subtracted
        )

    def test_pauliterm_add(self, pauli_term):
        # Test adding to PauliSum calls function of PauliSum
        old_paulisum_add = PauliSum.__add__
        PauliSum.__add__ = Mock()
        pauli_term + PauliSum.identity()
        PauliSum.__add__.assert_called_with(pauli_term)
        PauliSum.__add__ = old_paulisum_add

        # Test adding to term creates PauliSum with both terms
        summation = pauli_term + PauliTerm.identity()
        assert len(summation) == 2
        assert pauli_term in summation
        assert PauliTerm.identity() in summation

        # Test adding to a constant
        summation = pauli_term + 3.0
        assert PauliTerm("I0", 3.0) in summation

    def test_pauliterm_radd(self, pauli_term):
        summation = 3.0 + pauli_term
        assert PauliTerm("I0", 3.0) in summation

        old_pauliterm_add = PauliTerm.__add__
        PauliTerm.__add__ = Mock()
        3.0 + pauli_term
        PauliTerm.__add__.assert_called_once_with(3.0)
        PauliTerm.__add__ = old_pauliterm_add

    def test_pauliterm_sub(self, pauli_term):
        assert pauli_term - PauliTerm("X3") == pauli_term + PauliTerm("X3", -1)

        assert pauli_term - pauli_term == PauliSum()

    def test_pauliterm_rsub(self):
        assert 3.0 - PauliTerm("X3") == 3.0 + PauliTerm("X3", -1)

    def test_pauliterm_mul(self, pauli_term):
        # Test constant just creates similar term with multiplied coeffs
        pauli_term.coefficiet = 2.0
        constant_mul = pauli_term * 2.0j
        assert constant_mul._ops == pauli_term._ops
        assert constant_mul.coefficient == 4.0j
        assert constant_mul is not pauli_term

        # Test mul with PauliSum delegates to mul of PauliSum
        old_paulisum_mul = PauliSum.__mul__
        PauliSum.__mul__ = Mock()
        pauli_term * PauliSum.identity()
        PauliSum.__mul__.assert_called_once_with(PauliSum.identity())
        PauliSum.__mul__ = old_paulisum_mul

        # Test different scenarios for PauliTerm
        for combination in product(("X", "Y", "Z", "I"), repeat=2):
            left, right = combination[0], combination[1]

            result = PauliTerm(f"{left}0") * PauliTerm(f"{right}0")

            if left == "I" or right == "I":
                other_op = left if right == "I" else right
                assert result == PauliTerm(f"{other_op}0")
                continue

            if left == right:
                assert result == PauliTerm.identity()
                continue

            if "X" in combination and "Y" in combination:
                assert result._ops[0] == "Z"

                if left == "X":
                    assert result.coefficient == 1.0j
                else:
                    assert result.coefficient == -1.0j

            if "Y" in combination and "Z" in combination:
                assert result._ops[0] == "X"

                if left == "Y":
                    assert result.coefficient == 1.0j
                else:
                    assert result.coefficient == -1.0j

            if "X" in combination and "Z" in combination:
                assert result._ops[0] == "Y"

                if left == "X":
                    assert result.coefficient == -1.0j
                else:
                    assert result.coefficient == 1.0j

    def test_pauliterm_rmul(self, pauli_term):
        multiplication = 3.0 * pauli_term
        assert multiplication.coefficient == 6.0

        old_pauliterm_mul = PauliTerm.__mul__  # Need to recover later
        PauliTerm.__mul__ = Mock(return_value=PauliTerm.identity())
        3.0 * pauli_term
        PauliTerm.__mul__.assert_called_with(3.0)
        PauliTerm.__mul__ = old_pauliterm_mul

    def test_pauliterm_pow(self, pauli_term):
        with pytest.raises(ValueError):
            pauli_term**-10
        with pytest.raises(ValueError):
            pauli_term**0.25

        assert pauli_term**0 == PauliTerm.identity()
        assert pauli_term**1 == pauli_term

    def test_pauliterm_repr(self, pauli_term: PauliTerm):
        # Check identity is printed correctly (integer coefficients)
        assert str(PauliTerm.identity()) == "1.0*I"

        # Check identity is printed correctly (decimal coefficients)
        assert (
            str(PauliTerm.identity().copy(new_coefficient=1.5 + 2.3j)) == "(1.5+2.3j)*I"
        )

        # Check Pauli Term is printed correctly (integer coefficients)
        assert str(pauli_term) == "2.0*X0*Y1*Z12"

        # Check Pauli Term is printed correctly (decimal coefficients)
        assert (
            str(pauli_term.copy(new_coefficient=3.3 + 5.2j)) == "(3.3+5.2j)*X0*Y1*Z12"
        )

    def test_pauliterm_multiply_by_zero(self, pauli_term: PauliTerm):
        zero_op = pauli_term * 0
        assert isinstance(zero_op, PauliTerm)
        assert zero_op.coefficient == 0


class TestPauliSumOperations:
    @pytest.fixture
    def pauli_sum(self):
        return (
            0.5 * PauliTerm("X0")
            + 0.5j * PauliTerm("Y0")
            + PauliTerm("Z1") * PauliTerm("Z2")
        )

    def test_paulisum_native_constructor(self):
        # Empty parameters list returns empty sum
        assert PauliSum().terms == []

        # Incorrect list contents
        with pytest.raises(ValueError):
            PauliSum([PauliTerm.identity(), 1])

    def test_paulisum_cannot_be_constructed_from_term_not_wrapped_in_list(self):
        with pytest.raises(ValueError):
            PauliSum(PauliTerm.identity())

    def test_paulisum_fromstr(self, pauli_sum):
        assert PauliSum(str(pauli_sum)) == pauli_sum

    def test_paulisum_qubits(self, pauli_sum):
        assert pauli_sum.qubits == {0, 1, 2}

    def test_paulisum_isising(self, pauli_sum):
        assert not pauli_sum.is_ising

        ising_sum = PauliSum(
            [PauliTerm("Z1") * PauliTerm("Z2"), PauliTerm("Z12"), PauliTerm("Z34")]
        )

        assert ising_sum.is_ising

    def test_paulisum_circuits(self, pauli_sum):
        assert not hasattr(pauli_sum, "_circuits")
        circuits: List[Circuit] = pauli_sum.circuits
        assert hasattr(pauli_sum, "_circuits")

        for circuit, term in zip(circuits, pauli_sum):
            for op in circuit.operations:
                assert op.gate.name == term[op.qubit_indices[0]]

    def test_paulisum_identity(self):
        sum_id = PauliSum.identity()

        assert len(sum_id) == 1
        assert sum_id[0] == PauliTerm.identity()

    def test_paulisum_eq(self, pauli_sum):
        assert PauliSum.identity() == 1.0
        assert PauliSum.identity() == PauliTerm.identity()

        assert pauli_sum != PauliSum.identity()
        assert pauli_sum != PauliSum(pauli_sum[:-1])

        assert pauli_sum == PauliSum(pauli_sum[::-1])

        with pytest.raises(TypeError):
            pauli_sum == "(0.5)*X0 + (0.5j)*Y0 + (1.0)*Z1*Z2"

    def test_paulisum_add(self, pauli_sum):
        # Check simplify is called during sum
        old_paulisum_simplify = PauliSum.simplify
        PauliSum.simplify = Mock()
        pauli_sum + pauli_sum
        PauliSum.simplify.assert_called_once()
        PauliSum.simplify = old_paulisum_simplify

        # Check simplify works correctly
        summation = pauli_sum + pauli_sum
        assert len(summation) == len(pauli_sum)
        for term in pauli_sum:
            assert str(term.copy(new_coefficient=term.coefficient * 2)) in str(
                summation
            )

        # Check summation with constant
        constant_sum = PauliSum.identity() + 3.0
        assert len(constant_sum) == 1
        assert constant_sum[0].coefficient == 4

        # Check summation with pauliterm
        constant_sum = PauliSum.identity() + PauliTerm.identity()
        assert len(constant_sum) == 1
        assert constant_sum[0].coefficient == 2

    def test_paulisum_radd(self, pauli_sum):
        old_paulisum_radd = PauliSum.__radd__
        old_paulisum_add = PauliSum.__add__

        PauliSum.__radd__ = Mock(side_effect=PauliSum.__radd__)
        PauliSum.__add__ = Mock()

        # Actual call
        PauliSum.__radd__(pauli_sum, other=5)

        PauliSum.__radd__.assert_called_once()
        PauliSum.__add__.assert_called_once_with(5)

        PauliSum.__radd__ = old_paulisum_radd
        PauliSum.__add__ = old_paulisum_add

    def test_paulisum_sub(self, pauli_sum):
        assert pauli_sum - pauli_sum == PauliSum()

    def test_paulisum_rsub(self, pauli_sum):
        subtraction = 0 - pauli_sum

        for neg_term, term in zip(subtraction, pauli_sum):
            assert neg_term.coefficient == -1 * term.coefficient

    def test_paulisum_mul(self, pauli_sum):
        multiplication = pauli_sum * PauliTerm("X0")

        for mul_term, term in zip(multiplication, pauli_sum):
            assert mul_term == term * PauliTerm("X0")

        assert pauli_sum * 5 == 5 * pauli_sum

    def test_paulisum_rmul(self, pauli_sum):
        mul_by_constant = 5.0j * pauli_sum

        assert all(
            [
                mul_term.coefficient == term.coefficient * 5j
                for mul_term, term in zip(mul_by_constant, pauli_sum)
            ]
        )

    def test_paulisum_pow(self, pauli_sum):
        with pytest.raises(ValueError):
            pauli_sum**-1
        with pytest.raises(ValueError):
            pauli_sum**0.25

        assert pauli_sum**0 == PauliTerm.identity()
        assert pauli_sum**1 == pauli_sum

    def test_paulisum_repr(self, pauli_sum):
        repr_str = str(pauli_sum)

        assert all([str(term) in repr_str for term in pauli_sum.terms])

    def test_paulisum_multiply_by_zero(self, pauli_sum: PauliSum):
        zero_op = pauli_sum * 0
        assert zero_op == PauliSum()

    def test_paulisum_constant_term_zero(self, pauli_sum):
        assert pauli_sum.constant_term == 0

    @pytest.mark.parametrize("constant_term", [10, -10, 0.5, -0.5])
    def test_paulisum_constant(self, pauli_sum, constant_term):
        pauli_sum = pauli_sum + constant_term
        assert pauli_sum.constant_term == constant_term

    @pytest.mark.parametrize("constant_term", [10, -10, 0.5, -0.5])
    def test_paulisum_identity_operator(self, pauli_sum, constant_term):
        pauli_sum = pauli_sum + constant_term * PauliTerm("I0")
        assert pauli_sum.constant_term == constant_term
