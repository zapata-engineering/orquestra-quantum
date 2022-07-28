from itertools import product
from typing import List
from unittest.mock import Mock

import pytest

from orquestra.quantum.circuits import Circuit
from orquestra.quantum.wip.operators._pauli_operators import PauliSum, PauliTerm


class TestPauliTermOperations:
    @pytest.fixture
    def pauli_term(self):
        correct_list = [("X", 0), ("Y", 1), ("Z", 12)]
        return PauliTerm.from_list(correct_list, coefficient=2.0)

    @pytest.fixture
    def pauli_sum(self):
        return 0.5 * PauliTerm("X0") + 0.5j * PauliTerm("Y0")

    def test_pauliterm_native_constructor(self):
        # Test creating terms from different Pauli ops
        assert PauliTerm("X0", 1.0)._ops[0] == "X"
        assert PauliTerm("Y0", 1.0)._ops[0] == "Y"
        assert PauliTerm("Z0", 1.0)._ops[0] == "Z"
        assert PauliTerm("z0", 1.0)._ops[0] == "Z"

        # Test multi-digit qubit indices work
        assert 123 in PauliTerm("X123", 1.0)._ops

        # Test different types of coefficients work
        assert PauliTerm("Z0", 1).coefficient == 1
        assert isinstance(PauliTerm("Z0", 1).coefficient, complex)

        assert PauliTerm("Z0", 2.0).coefficient == 2.0
        assert isinstance(PauliTerm("Z0", 2.0).coefficient, complex)

        assert PauliTerm("Z0", 3.4 + 1.5j).coefficient == 3.4 + 1.5j

        # Test constant does not get included in the dictionary
        constant = PauliTerm("I0", 2.0)
        assert 0 not in constant
        assert constant.coefficient == 2.0

    def test_pauliterm_passing_wrong_input_to_constructor_raises_error(self):
        # Test passing wrongly formatted string
        with pytest.raises(ValueError) as e:
            PauliTerm("1 X")
        assert "Badly formatted" in str(e.value)

        # Test passing invalid qubit number
        with pytest.raises(ValueError) as e:
            PauliTerm("X-1254")
        assert "Badly formatted" in str(e.value)

        # Test passing wrong operator
        with pytest.raises(ValueError) as e:
            PauliTerm("A0")
        assert "Badly formatted" in str(e.value)

    def test_pauliterm_from_list_constructor(self, pauli_term):
        assert len(pauli_term._ops) == 3
        assert pauli_term[0] == "X"
        assert pauli_term[1] == "Y"
        assert pauli_term[12] == "Z"

        # Error for wrong tuple shape
        with pytest.raises(ValueError) as e:
            PauliTerm.from_list([("X0")])
        assert "PauliTerm.from_str" in str(e.value)

        # Error for bad qubit index
        with pytest.raises(ValueError) as e:
            PauliTerm.from_list([("X", -1)])
        assert "Invalid qubit index" in str(e.value)

        # Error for duplicate qubits
        with pytest.raises(ValueError) as e:
            PauliTerm.from_list([("X", 0), ("Y", 0)])
        assert "Duplicate" in str(e.value)

    def test_pauliterm_from_str_constructor(self, pauli_term):
        assert PauliTerm(str(pauli_term)) == pauli_term

        with pytest.raises(ValueError):
            PauliTerm("(5.0j + 9)*I")

        with pytest.raises(ValueError):
            PauliTerm("X*(5.0 + 9j)")

    def test_pauliterm_identity(self, pauli_term):
        identity_term = PauliTerm.identity()

        assert len(identity_term) == 0
        assert identity_term.coefficient == 1.0

        # Test multiplying by identity returns term
        assert PauliTerm.identity() * pauli_term == pauli_term
        assert pauli_term * PauliTerm.identity() == pauli_term

    def test_pauliterm_copy_function(self, pauli_term):
        copied_term = pauli_term.copy(new_coefficient=2.0)

        assert copied_term is not pauli_term
        assert copied_term._ops is not pauli_term._ops

        assert copied_term.coefficient == 2.0
        assert copied_term._ops == pauli_term._ops

    def test_pauliterm_qubits(self, pauli_term):
        assert pauli_term.qubits == {0, 1, 12}

    def test_pauliterm_isising(self, pauli_term):
        assert not pauli_term.is_ising

        assert PauliTerm.from_list([("Z", 0), ("Z", 1), ("Z", 3)]).is_ising

        pauli_term_minus_X = pauli_term * PauliTerm("Y0")
        pauli_term_only_Zs = pauli_term_minus_X * PauliTerm("Y1")

        assert pauli_term_only_Zs.is_ising

    def test_pauliterm_circuit(self, pauli_term):
        assert not hasattr(pauli_term, "_circuit")
        circuit: Circuit = pauli_term.circuit
        assert hasattr(pauli_term, "_circuit")

        for op in circuit.operations:
            assert op.gate.name == pauli_term[op.qubit_indices[0]]

    def test_pauliterm_operations_as_set(self, pauli_term):
        frozen_set = pauli_term.operations_as_set()

        assert isinstance(frozen_set, frozenset)
        assert frozen_set.difference(pauli_term._ops.items()) == set()

    def test_pauliterm_getitem_raises_warning_for_invalid_index(self, pauli_term):
        with pytest.warns(UserWarning):
            returned_id = pauli_term[sum(pauli_term._ops.keys())]

        assert returned_id == "I"

    def test_pauliterm_iterable_returns_op_idx_pairs(self, pauli_term):
        expected = [("X", 0), ("Y", 1), ("Z", 12)]

        for (exp_left, exp_right), (left, right) in zip(expected, pauli_term):
            assert exp_left == left
            assert exp_right == right

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
        assert str(PauliTerm.identity()) == "(1+0j)*I"

        # Check identity is printed correctly (decimal coefficients)
        assert (
            str(PauliTerm.identity().copy(new_coefficient=1.5 + 2.3j)) == "(1.5+2.3j)*I"
        )

        # Check Pauli Term is printed correctly (integer coefficients)
        assert str(pauli_term) == "(2+0j)*X0*Y1*Z12"

        # Check Pauli Term is printed correctly (decimal coefficients)
        assert (
            str(pauli_term.copy(new_coefficient=3.3 + 5.2j)) == "(3.3+5.2j)*X0*Y1*Z12"
        )


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
