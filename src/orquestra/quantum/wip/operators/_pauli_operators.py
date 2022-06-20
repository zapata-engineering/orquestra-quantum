import copy
import re
import warnings
from collections import OrderedDict
from typing import Dict, FrozenSet, Iterator, List, Sequence, Tuple, Union, cast
import numpy as np

from orquestra.quantum.circuits import Circuit, builtin_gate_by_name, Operation

_coefficient_types = Union[int, float, complex]

allowed_operators = ["X", "Y", "Z", "I"]

PauliRepresentation = Union["PauliTerm", "PauliOp"]

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
"""
# TODO:
- __hash__
- __rmul__
- __add__
- __radd__
- __sub__
- __rsub__
- compact_str

- convenience methods (maybe put them in __init__)

- all paulisums
"""


class PauliTerm:
    """
    A datastructure for storing information about a single Pauli Term
    """

    def __init__(
        self,
        operator: Union[Tuple[str, int], str],
        coefficient: _coefficient_types = 1.0,
    ):
        if isinstance(operator, str):
            matched_pattern = re.match(r"([a-zA-Z]+)(-?[0-9]+)", operator, re.I)

            if not matched_pattern:
                raise ValueError("Badly formatted string representation passed.")

            op, qubit_idx = matched_pattern.groups()
        elif isinstance(operator, tuple):
            op, qubit_idx = operator

        qubit_idx = int(qubit_idx)

        if op not in allowed_operators:
            raise ValueError(f"Passed operator not supported. Got {op}.")

        if qubit_idx < 0:
            raise ValueError(
                f"Only positive qubit indices are allowed. Got {qubit_idx}."
            )

        self._ops: Dict[int, str] = OrderedDict()
        if op != "I":
            self._ops[qubit_idx] = op

        self.coefficient = complex(coefficient)

    @classmethod
    def from_list(
        cls,
        list_of_terms: List[Tuple[str, int]],
        coefficient: _coefficient_types = 1.0,
    ) -> "PauliTerm":
        """
        A slightly more efficient constructor when all the elements of the term are known beforehand.
        Users should employ this function instead of creating individual terms and multiplying.
        """

        ############### Some checks on input first ###############
        op_list, idx_list = zip(*list_of_terms)

        if not all([isinstance(op, tuple) for op in list_of_terms]):
            raise ValueError(
                "The list can only contain tuples of the form (op, index). If you want to initialize from strings,"
                " check the PauliTerm.from_str function."
            )

        valid_input = all(
            [
                all([op in allowed_operators for op in op_list]),
                all([idx >= 0 for idx in idx_list]),
            ]
        )

        if not valid_input:
            raise ValueError(
                "Some of the input was invalid. Make sure you only pass Pauli operators"
                " and non-negative qubit indices."
            )

        if len(set(idx_list)) != len(idx_list):
            raise ValueError(
                "Duplicate indices used in list. Manually create terms and multiply them instead."
            )

        ##########################################################

        result_term = PauliTerm("I0")
        result_term.coefficient = complex(coefficient)
        for op, idx in list_of_terms:
            if op != "I":
                result_term._ops[idx] = op

        return result_term

    @classmethod
    def from_str(cls, pauli_term_str: str) -> "PauliTerm":
        pass

    def copy(self, new_coefficient: _coefficient_types = None) -> "PauliTerm":
        """
        Properly creates a new PauliTerm, with a completely new dictionary
        of operators
        """
        new_term = PauliTerm("I0", 1.0)  # create new object
        # manually copy all attributes over
        for key in self.__dict__.keys():
            val = self.__dict__[key]
            if isinstance(val, (dict, list, set)):  # mutable types
                new_term.__dict__[key] = copy.copy(val)
            else:  # immutable types
                new_term.__dict__[key] = val

        if new_coefficient:
            new_term.coefficient = new_coefficient

        return new_term

    @property
    def qubits(self) -> List[int]:
        """
        Returns the list of qubit indices associated with this term.
        """
        return list(self._ops.keys())

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
        if i not in self._ops.keys():
            warnings.warn("Index not in term! Identity is returned.", UserWarning)

        return self._ops.get(i, "I")

    def __iter__(self) -> Iterator[Tuple[str, int]]:
        for i in self.qubits:
            yield self[i], i

    def __eq__(self, other: object) -> bool:
        if not isinstance(self, (PauliOp, PauliTerm)):
            raise ValueError(f"Can't compare with object of type {type(other)}")

        if isinstance(other, PauliOp):
            return other == self

        cast_other = cast(PauliTerm, other)
        return (
            self.operations_as_set() == cast_other.operations_as_set()
            and np.allclose(self.coefficient, cast_other.coefficient)
        )

    def _multiply_by_operator(self, op: str, index: int) -> "PauliTerm":
        result_term = PauliTerm("I0", 0)
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

        result_term._ops = result_ops
        result_term.coefficient = result_coeff

        return result_term

    def __mul__(
        self, term: Union[PauliRepresentation, _coefficient_types]
    ) -> PauliRepresentation:
        """Multiplies this Pauli Term with another PauliTerm, PauliSum, or number according to the
        Pauli algebra rules.
        :param term: (PauliTerm or PauliSum or Number) A term to multiply by.
        :returns: The product of this PauliTerm and term.
        """
        if isinstance(term, PauliOp):
            pass
            # return (PauliOp([self]) * term).simplify()
        elif isinstance(term, PauliTerm):
            result_term = PauliTerm("I0", 1.0)
            result_term._ops = self._ops.copy()

            new_coeff = self.coefficient * term.coefficient
            for op, index in term:
                if op != "I":
                    result_term = result_term._multiply_by_operator(op, index)

            return result_term.copy(new_coefficient=result_term.coefficient * new_coeff)

        return self.copy(self.coefficient * cast(complex, term))

    def __pow__(self, power: int) -> "PauliTerm":
        """Raises this PauliTerm to power.
        :param power: The power to raise this PauliTerm to.
        :return: The power-fold product of power.
        """
        if not isinstance(power, int) or power < 0:
            raise ValueError("The power must be a non-negative integer.")

        if len(self.qubits) == 0:
            # There weren't any nontrivial operators
            return self.copy(new_coefficient=1.0)

        result = PauliTerm("I0", 1)
        for _ in range(power):
            result = cast(PauliTerm, result * self)
        return result

    def __repr__(self) -> str:
        term_strs = []
        for index in self._ops.keys():
            term_strs.append("%s%s" % (self[index], index))

        if len(term_strs) == 0:
            term_strs.append("I")
        out = "%s*%s" % (self.coefficient, "*".join(term_strs))
        return out


class PauliOp:
    def __init__(self, terms: Sequence[PauliTerm]):
        self.terms = terms


tmp = PauliTerm.from_list([("I", 0), ("X", 1), ("Y", 2)])
tmp2 = PauliTerm("X1", 1.0)
breakpoint()
