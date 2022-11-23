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

import re
import warnings
from collections import OrderedDict
from itertools import chain, product
from numbers import Number
from typing import (
    Any,
    Dict,
    FrozenSet,
    Hashable,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

import numpy as np

from ..circuits import Circuit, Operation, builtin_gate_by_name

PauliRepresentation = Union["PauliTerm", "PauliSum"]

ALLOWED_OPERATORS = ["X", "Y", "Z", "I"]

# Use ASCII values instead of actual strings
# as keys to allow for commutativity
OPERATOR_MAP = {
    ord("X") + ord("Y"): "Z",
    ord("Y") + ord("Z"): "X",
    ord("X") + ord("Z"): "Y",
}

COEFF_MAP = {
    "XY": 1.0j,
    "XZ": -1.0j,
    "YX": -1.0j,
    "YZ": 1.0j,
    "ZX": 1.0j,
    "ZY": -1.0j,
}

HASH_PRECISION = 1e6


def _efficient_exponentiation(
    pauli_rep: PauliRepresentation, power: int
) -> PauliRepresentation:
    """Efficiently exponentiate PauliTerm or PauliSum.

    This function assumes validation checks have already been done from parent routine.
    """
    if power == 0:
        return type(pauli_rep).identity()

    if power % 2 == 1:
        return pauli_rep * _efficient_exponentiation(pauli_rep, power - 1)

    intermediate_result = _efficient_exponentiation(pauli_rep, power // 2)
    return intermediate_result * intermediate_result


def _validate_type(object: Any) -> None:
    if not isinstance(object, (PauliSum, PauliTerm, int, float, complex)):
        raise TypeError(
            f"Can't carry out operation with object of type {type(object)}."
        )


def _is_in_brackets(string: str) -> bool:
    return string.startswith("(") and string.endswith(")")


def _parse_complex(complex_str: str) -> complex:
    value = complex(complex_str.replace(" ", ""))
    if value.real != 0 and value.imag != 0 and not _is_in_brackets(complex_str):
        raise ValueError(
            "Complex number with nonzero real and imaginary part has to be "
            "enclosed in bracket."
        )
    return value


def _parse_operator(op_str: str) -> Tuple[int, str]:
    match = re.match(r"([XYZI])([0-9]+)$", op_str, re.I)

    if not match:
        raise ValueError("Badly formatted string representation passed.")

    return int(match.group(2)), match.group(1).upper()


def _parse_operators_and_coefficient(
    term_str: str,
) -> Tuple[Optional[complex], Dict[int, str]]:
    parts = re.split(r"\ *\*\ *", term_str.strip(" "))
    try:
        coef = _parse_complex(parts[0])
        operators_strs = parts[1:]
    except ValueError:
        coef = None
        operators_strs = parts

    operators_dict = dict([_parse_operator(op_str) for op_str in operators_strs])

    if len(operators_dict) != len(operators_strs):
        raise ValueError("Duplicate qubit index in a term detected.")

    return coef, dict([_parse_operator(op_str) for op_str in operators_strs])


class PauliTerm:
    """Representation of a single Pauli Term.

    If coefficient is not provided neither directly nor in the string repr,
    it defaults to 1.0
    """

    def __init__(
        self,
        operator: Union[str, Dict[int, str]],
        coefficient: Optional[complex] = None,
    ):
        if isinstance(operator, str):
            _parsed_coefficient, operator = _parse_operators_and_coefficient(operator)
            if _parsed_coefficient is not None and coefficient is not None:
                raise ValueError(
                    "Coefficient can be provided either in an argument or string "
                    "representation (but not both)."
                )
            if _parsed_coefficient is not None:
                coefficient = _parsed_coefficient

        if not all([qubit_idx >= 0 for qubit_idx in operator]):
            raise ValueError(
                "Invalid qubit index in dictionary keys. "
                "Make sure all qubit indices are non-negative integers."
            )

        # Verify value values
        if not all([op in ALLOWED_OPERATORS for op in operator.values()]):
            raise ValueError(
                "Invalid operators in dictionary. "
                f"Allowed ones are {ALLOWED_OPERATORS}."
            )

        self._ops: Dict[int, str] = {
            idx: op for idx, op in operator.items() if op != "I"
        }
        self.coefficient = 1.0 if coefficient is None else coefficient

    @staticmethod
    def from_iterable(
        terms: Iterable[Tuple[str, int]], coefficient: complex = 1.0
    ) -> "PauliTerm":
        """Construct PauliTerm from a list of operators.

        A slightly more efficient constructor when all the elements of the term are
        known beforehand. Users should employ this function instead of creating
        individual terms and multiplying.
        """

        ############### Some checks on input first ###############
        if terms:
            _, idx_list = zip(*terms)

            if not all([isinstance(op, tuple) for op in terms]):
                raise ValueError(
                    "The list can only contain tuples of the form (index, op). If you "
                    "want to initialize from strings, use PauliTerm's constructor."
                )

            if len(set(idx_list)) != len(idx_list):
                raise ValueError(
                    "Duplicate indices used in list. Manually create terms"
                    "and multiply them instead."
                )

        ##########################################################

        result_dict = {idx: op for op, idx in terms if op != "I"}

        return PauliTerm(result_dict, coefficient)

    @staticmethod
    def identity() -> "PauliTerm":
        return PauliTerm("I0", 1.0)

    def copy(self, new_coefficient: Optional[complex] = None) -> "PauliTerm":
        """Copy PauliTerm, possibly changing its coefficient to a new one.

        The created copy is deep, in particular internal dictionary storing map
        from qubit indices to operators is also copied.
        """
        if new_coefficient is None:
            new_coefficient = self.coefficient

        return PauliTerm(operator=self._ops, coefficient=new_coefficient)

    @property
    def qubits(self) -> Set[int]:
        """The list of qubit indices associated with this term."""
        return set(self._ops.keys())

    @property
    def is_ising(self) -> bool:
        """True iff this term is Ising Operator (i.e. contains no X or Y operators)"""

        return set(self._ops.values()) == {"Z"} or self.is_constant

    @property
    def circuit(self) -> Circuit:
        """Circuit implementing this Pauli term.

        For efficiency constructed circuit is cached after the first invocation of
        this property.
        """
        if not hasattr(self, "_circuit"):
            self._circuit = Circuit(
                [cast(Operation, builtin_gate_by_name(op)(index)) for op, index in self]
            )

        return self._circuit

    @property
    def operations(self) -> FrozenSet[Tuple[int, str]]:
        return frozenset(self._ops.items())

    def __len__(self) -> int:
        return len(self._ops)

    def __getitem__(self, i: int) -> str:
        if i not in self._ops:
            warnings.warn("Index not in term! Identity is returned.", UserWarning)

        return self._ops.get(i, "I")

    def __iter__(self) -> Iterator[Tuple[str, int]]:
        for i in self.qubits:
            yield self[i], i

    def __hash__(self) -> int:
        if isinstance(self.coefficient, complex):
            coefficient_real = self.coefficient.real
            coefficient_imag = self.coefficient.imag
        else:
            coefficient_real = self.coefficient
            coefficient_imag = 0
        return hash(
            (
                round(coefficient_real * HASH_PRECISION),
                round(coefficient_imag * HASH_PRECISION),
                self.operations,
            )
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (PauliSum, PauliTerm, int, float, complex)):
            raise TypeError(f"Can't compare with object of type {type(other)}")

        if isinstance(other, (int, float, complex)):
            return self == PauliTerm("I0", other)

        if isinstance(other, PauliSum):
            return other == self

        cast_other = cast(PauliTerm, other)
        return np.allclose(self.coefficient, cast_other.coefficient) and (
            np.allclose(self.coefficient, 0) or self.operations == cast_other.operations
        )

    def __add__(self, other: Union[PauliRepresentation, complex]) -> "PauliSum":
        _validate_type(other)

        if isinstance(other, PauliSum):
            return other + self

        if isinstance(other, PauliTerm):
            return PauliSum([self, other]).simplify()

        return self + PauliTerm("I0", other)

    def __radd__(self, other: complex) -> "PauliSum":
        return self + PauliTerm("I0", other)

    def __sub__(self, other: Union[PauliRepresentation, complex]) -> "PauliSum":
        return self + -1.0 * other

    def __rsub__(self, other: Union[PauliRepresentation, complex]) -> "PauliSum":
        return other + -1.0 * self

    def _multiply_by_operator(self, op: str, index: int) -> "PauliTerm":
        result_coeff = self.coefficient
        result_ops = self._ops.copy()

        if index not in result_ops:
            # Case 1: qubit not used yet
            result_ops[index] = op
        elif result_ops[index] == op:
            # Case 2: equal operators cancel
            del result_ops[index]
        elif op in ["X", "Y", "Z"]:
            # Case 3: different pauli ops return third op
            result_ops[index] = OPERATOR_MAP[ord(self[index]) + ord(op)]
            result_coeff *= COEFF_MAP[self[index] + op]
        else:
            raise ValueError(f"Unsupported operation. Got {op}.")

        return PauliTerm(result_ops, result_coeff)

    def __mul__(
        self, other: Union[PauliRepresentation, complex]
    ) -> PauliRepresentation:
        """Multiply this Pauli Term with another PauliTerm, PauliSum, or number.

        This method performs simplifications according to Pauli Algebra rules.
        """
        _validate_type(other)

        if isinstance(other, PauliSum):
            return (PauliSum([self]) * other).simplify()
        elif isinstance(other, PauliTerm):
            result_term = self.copy(new_coefficient=1)

            for op, index in other:
                if op != "I":
                    result_term = result_term._multiply_by_operator(op, index)

            new_coeff = self.coefficient * other.coefficient
            return result_term.copy(new_coefficient=result_term.coefficient * new_coeff)

        return self.copy(self.coefficient * complex(other))

    def __rmul__(self, other: complex) -> "PauliTerm":
        result = self * other
        assert isinstance(result, PauliTerm)
        return result

    def __truediv__(self, other: complex) -> "PauliTerm":
        result = self * (1.0 / other)
        assert isinstance(result, PauliTerm)
        return result

    def __pow__(self, power: int) -> "PauliTerm":
        """Raise this PauliTerm to integral power."""
        if not isinstance(power, int) or power < 0:
            raise ValueError("The power must be a non-negative integer.")

        return cast(PauliTerm, _efficient_exponentiation(self.copy(), power))

    def __repr__(self) -> str:
        term_strs = [f"{self[index]}{index}" for index in self._ops]

        if len(term_strs) == 0:
            term_strs.append("I")

        return f"{self.coefficient}*{'*'.join(term_strs)}"

    @property
    def terms(self) -> List["PauliTerm"]:
        return [self]

    @property
    def is_constant(self) -> bool:
        return self._ops == {}

    @property
    def n_qubits(self) -> int:
        """Number of qubits used in this PauliTerm.

        Follows the convention of openfermion's `count_qubits`.
        Note that this is different from the number of operations. For example,
        PauliTerm("Z0*Z3").n_qubits = 4, but len(PauliTerm("Z0*Z3").qubits) = 2.

        """
        return 0 if self.is_constant else max(self.qubits) + 1


class PauliSum:
    def __init__(self, terms: Optional[Union[str, Sequence[PauliTerm]]] = None):
        if isinstance(terms, str):
            terms = [PauliTerm(s.strip()) for s in re.split(r"\+(?![^(]*\))", terms)]
        if terms is None:
            # If no terms is given, the PauliSum has a value of zero.
            terms = []

        if not (
            isinstance(terms, Sequence)
            and all([isinstance(term, PauliTerm) for term in terms])
        ):
            raise ValueError(
                "PauliSums can be constructed only from Sequences of PauliTerms."
            )
        self.terms: Sequence[PauliTerm] = cast(Sequence[PauliTerm], terms)

    def __len__(self) -> int:
        return len(self.terms)

    def __getitem__(self, idx: int) -> PauliTerm:
        return self.terms[idx]

    def __iter__(self) -> Iterator[PauliTerm]:
        return self.terms.__iter__()

    @property
    def qubits(self) -> Set[int]:
        return set(chain.from_iterable([term.qubits for term in self.terms]))

    @property
    def is_ising(self) -> bool:
        """Returns whether the full operator represents an Ising model."""
        if not hasattr(self, "_is_ising"):
            self._is_ising = all([term.is_ising for term in self.terms])
        return self._is_ising

    @property
    def circuits(self) -> List[Circuit]:
        if not hasattr(self, "_circuits"):
            self._circuits = [term.circuit for term in self.terms]

        return self._circuits

    @staticmethod
    def identity() -> "PauliSum":
        return PauliSum([PauliTerm.identity()])

    def __eq__(self, other: object) -> bool:
        _validate_type(other)

        if isinstance(other, (int, float, complex)):
            constant_term = PauliTerm("I0", complex(other))
            return self == PauliSum([constant_term])

        if isinstance(other, PauliTerm):
            if len(self) == 0:
                return np.allclose(other.coefficient, 0)
            return self == PauliSum([other])

        other = cast(PauliSum, other)
        if len(self) != len(other):
            return False

        return set(self.terms) == set(other.terms)

    def __add__(self, other: Union[PauliRepresentation, complex]) -> "PauliSum":
        _validate_type(other)

        if isinstance(other, PauliTerm):
            other = PauliSum([other])
        elif isinstance(other, (int, float, complex)):
            other = PauliSum([PauliTerm("I0", other)])

        new_op = PauliSum([term.copy() for term in chain(self.terms, other.terms)])

        return new_op.simplify()

    def __radd__(self, other: complex) -> "PauliSum":
        assert isinstance(other, (int, float, complex))
        return self + other

    def __sub__(self, other: Union[PauliRepresentation, complex]) -> "PauliSum":
        # No need to check for types here since it will be
        # carried out in the __add__ function
        return self + -1.0 * other

    def __rsub__(self, other: complex) -> "PauliSum":
        return other + -1.0 * self

    def __mul__(self, other: Union[PauliRepresentation, complex]) -> "PauliSum":
        _validate_type(other)

        other_terms = (
            other.terms
            if isinstance(other, PauliSum)
            else [cast(PauliTerm, PauliTerm.identity() * other)]
        )

        new_paulisum = PauliSum(
            cast(
                Sequence[PauliTerm],
                [
                    left_term * right_term
                    for left_term, right_term in product(self.terms, other_terms)
                ],
            )
        )

        return new_paulisum.simplify()

    def __rmul__(self, other: complex) -> "PauliSum":
        assert isinstance(other, (int, float, complex))

        new_terms = [cast(PauliTerm, term.copy() * other) for term in self.terms]

        return PauliSum(new_terms).simplify()

    def __truediv__(self, other: complex) -> "PauliSum":
        return self * (1.0 / other)

    def __pow__(self, power: int) -> "PauliSum":
        if not isinstance(power, int) or power < 0:
            raise ValueError(f"Power must be a non-negative integer. Got {power}.")

        return cast(PauliSum, _efficient_exponentiation(self, power))

    def simplify(self) -> "PauliSum":
        like_terms: Dict[Hashable, List[PauliTerm]] = OrderedDict()
        for term in self.terms:
            key = term.operations
            if key in like_terms:
                like_terms[key].append(term)
            else:
                like_terms[key] = [term]

        terms = []
        for term_list in like_terms.values():
            first_term = term_list[0]
            if len(term_list) == 1 and not np.isclose(first_term.coefficient, 0.0):
                terms.append(first_term)
            else:
                coeff = sum(t.coefficient for t in term_list)
                if not np.isclose(coeff, 0.0):  # type: ignore
                    terms.append(term_list[0].copy(new_coefficient=coeff))
        return PauliSum(terms)

    def __repr__(self):
        if len(self) == 0:
            zero_identity_term = PauliTerm("I0", 0)
            return str(zero_identity_term)
        return " + ".join([str(term) for term in self.terms])

    def __hash__(self):
        return hash(tuple(self.terms))

    @property
    def is_constant(self) -> bool:
        return len(self.terms) == 0 or all([term.is_constant for term in self.terms])

    @property
    def constant_term(self) -> Union[Number, complex]:
        return sum([term.coefficient for term in self.terms if term.is_constant])

    @property
    def n_qubits(self) -> int:
        """Number of qubits used in this PauliSum.

        Follows the convention of openfermion's `count_qubits`, so that a qubit is
        counted even if there are no operations on it. For example,
        PauliSum("Z0+Z3").n_qubits = 4, but len(PauliSum("Z0+Z3").qubits) = 2.

        """
        return 0 if self.is_constant else max(self.qubits) + 1
