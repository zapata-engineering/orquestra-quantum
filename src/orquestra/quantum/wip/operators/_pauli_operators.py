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
from typing import (
    Any,
    Dict,
    FrozenSet,
    Hashable,
    Iterator,
    List,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

import numpy as np

from orquestra.quantum.circuits import Circuit, Operation, builtin_gate_by_name

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
    """
    A more efficient implementation of exponentiation.
    Assumes validation checks already done from parent routine.
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


class PauliTerm:
    """
    A datastructure for storing information about a single Pauli Term
    """

    def __init__(
        self,
        operator: Union[str, Dict[int, str]],
        coefficient: complex = 1.0,
    ):
        self.coefficient = complex(coefficient)

        if isinstance(operator, dict):
            # Verify key values
            if not all([qubit_idx >= 0 for qubit_idx in operator]):
                raise ValueError(
                    "Invalid qubit index in dictionary keys."
                    " Make sure all qubit indices are non-negative integers."
                )

            # Verify value values
            if not all([op in ALLOWED_OPERATORS for op in operator.values()]):
                raise ValueError(
                    "Invalid operators in dictionary."
                    f" Allowed ones are {ALLOWED_OPERATORS}."
                )
            self._ops: Dict[int, str] = {
                idx: op for idx, op in operator.items() if op != "I"
            }
        elif isinstance(operator, str):
            matched_pattern = re.match(r"([XYZI])([0-9]+)", operator, re.I)

            if not matched_pattern:
                raise ValueError("Badly formatted string representation passed.")

            op, qubit_idx = matched_pattern.groups()

            self._ops = {}
            if op != "I":
                self._ops[int(qubit_idx)] = op.upper()
        else:
            raise TypeError(f"Wrong type of operator passed. Got {type(operator)}.")

    @staticmethod
    def from_list(
        list_of_terms: List[Tuple[str, int]],
        coefficient: complex = 1.0,
    ) -> "PauliTerm":
        """
        A slightly more efficient constructor when all the elements of the term are
         known beforehand. Users should employ this function instead of creating
         individual terms and multiplying.
        """

        ############### Some checks on input first ###############
        _, idx_list = zip(*list_of_terms)

        if not all([isinstance(op, tuple) for op in list_of_terms]):
            raise ValueError(
                "The list can only contain tuples of the form (op, index). If you"
                " want to initialize from strings, check the"
                " PauliTerm.from_str function."
            )

        if len(set(idx_list)) != len(idx_list):
            raise ValueError(
                "Duplicate indices used in list. Manually create terms"
                " and multiply them instead."
            )

        ##########################################################

        result_dict = {idx: op for op, idx in list_of_terms if op != "I"}

        return PauliTerm(result_dict, coefficient)

    @staticmethod
    def from_str(str_pauli_term: str) -> "PauliTerm":
        """Construct a PauliTerm from the result of str(pauli_term)"""
        # split into str_coef, str_op at first '*'' outside parenthesis
        try:
            str_coef, str_op = re.split(r"\*(?![^(]*\))", str_pauli_term, maxsplit=1)
        except ValueError:
            raise ValueError(
                "Could not separate the pauli string into "
                f"coefficient and operator. {str_pauli_term} does"
                " not match <coefficient>*<operator>"
            )
        # parse the coefficient into complex
        try:
            coef = complex(str_coef.replace(" ", ""))
        except ValueError:
            raise ValueError(f"Could not parse the coefficient {str_coef}")

        result_term = PauliTerm.identity() * coef
        if str_op == "I":
            assert isinstance(result_term, PauliTerm)
            return result_term

        for op in str_op.split("*"):
            result_term *= PauliTerm(op)

        assert isinstance(result_term, PauliTerm)
        return result_term

    @staticmethod
    def identity() -> "PauliTerm":
        return PauliTerm("I0", 1.0)

    def copy(self, new_coefficient: complex = None) -> "PauliTerm":
        """
        Properly creates a new PauliTerm, with a completely new dictionary
        of operators
        """
        new_coefficient = new_coefficient if new_coefficient else self.coefficient

        return PauliTerm(operator=self._ops, coefficient=new_coefficient)

    @property
    def qubits(self) -> Set[int]:
        """
        Returns the list of qubit indices associated with this term.
        """
        return set(self._ops.keys())

    @property
    def is_ising(self) -> bool:
        """
        Returns whether the term represents an ising model
        (i.e. contains only Z terms)
        """

        return set(self._ops.values()) == {"Z"}

    @property
    def circuit(self) -> Circuit:
        """
        Returns the circuit implementing this Pauli Term. Since the public API
        treats the object as immutable, we store the circuit representation when
        the function is called for the first time, for efficiency.
        """
        if not hasattr(self, "_circuit"):
            self._circuit = Circuit(
                [cast(Operation, builtin_gate_by_name(op)(index)) for op, index in self]
            )

        return self._circuit

    def operations_as_set(self) -> FrozenSet[Tuple[int, str]]:
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
        assert isinstance(self.coefficient, complex)
        return hash(
            (
                round(self.coefficient.real * HASH_PRECISION),
                round(self.coefficient.imag * HASH_PRECISION),
                self.operations_as_set(),
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
        return (
            self.operations_as_set() == cast_other.operations_as_set()
            and np.allclose(self.coefficient, cast_other.coefficient)
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
        """Multiplies this Pauli Term with another PauliTerm, PauliSum, or number
        according to the Pauli algebra rules.
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

    def __pow__(self, power: int) -> "PauliTerm":
        """
        Raises this PauliTerm to power.
        """
        if not isinstance(power, int) or power < 0:
            raise ValueError("The power must be a non-negative integer.")

        return cast(PauliTerm, _efficient_exponentiation(self.copy(), power))

    def __repr__(self) -> str:
        term_strs = [f"{self[index]}{index}" for index in self._ops]

        if len(term_strs) == 0:
            term_strs.append("I")

        return f"{self.coefficient}*{'*'.join(term_strs)}"


class PauliSum:
    def __init__(self, terms: Sequence[PauliTerm] = None):
        if terms is None:
            terms = []

        if not (
            isinstance(terms, Sequence)
            and all([isinstance(term, PauliTerm) for term in terms])
        ):
            raise ValueError(
                "PauliSums can be constructed only from Sequences of PauliTerms."
            )
        self.terms: Sequence[PauliTerm] = terms

    @staticmethod
    def from_str(str_pauli_sum: str) -> "PauliSum":
        """Construct a PauliSum from the result of str(pauli_sum)"""
        # split str_pauli_sum only at "+" outside of parenthesis to allow
        # e.g. "(0.5)*X0 + (0.5+0j)*Z2"
        str_terms = re.split(r"\+(?![^(]*\))", str_pauli_sum)
        str_terms = [s.strip() for s in str_terms]
        terms = [PauliTerm.from_str(term) for term in str_terms]
        return PauliSum(terms).simplify()

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
        """
        Returns whether the full operator represents an ising model.
        """
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

    def __pow__(self, power: int) -> "PauliSum":
        if not isinstance(power, int) or power < 0:
            raise ValueError(f"Power must be a non-negative integer. Got {power}.")

        return cast(PauliSum, _efficient_exponentiation(self, power))

    def simplify(self) -> "PauliSum":
        like_terms: Dict[Hashable, List[PauliTerm]] = OrderedDict()
        for term in self.terms:
            key = term.operations_as_set()
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
        return " + ".join([str(term) for term in self.terms])
