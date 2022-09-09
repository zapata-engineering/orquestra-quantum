################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
from typing import Any, Dict, List, Tuple

import rapidjson as json

from ..typing import AnyPath, LoadSource
from ._pauli_operators import PauliRepresentation, PauliSum, PauliTerm


def convert_dict_to_op(dictionary: dict) -> PauliSum:
    """Get a PauliSum from a dictionary.
    Args:
        dictionary: the dictionary representation
    Returns:
        op: the operator
    """
    full_operator = PauliSum()
    for term_dict in dictionary["terms"]:
        operator: List[Tuple[str, int]] = []
        for pauli_op in term_dict["pauli_ops"]:
            operator.append((pauli_op["op"], pauli_op["qubit"]))
        coefficient = term_dict["coefficient"]["real"]
        if term_dict["coefficient"].get("imag"):
            coefficient += 1j * term_dict["coefficient"]["imag"]
        full_operator += PauliTerm.from_iterable(operator, coefficient)

    return full_operator


def convert_op_to_dict(op: PauliRepresentation) -> Dict[str, Any]:
    """Convert a PauliTerm or PauliSum to a dictionary.
    Args:
        op: the operator
    Returns:
        dictionary: the dictionary representation
    """

    dictionary: Dict[str, Any] = {}
    dictionary["terms"] = []
    for term in op.terms:
        term_dict: Dict[str, Any] = {
            "pauli_ops": [{"qubit": op[0], "op": op[1]} for op in term.operations]
        }

        if isinstance(term.coefficient, complex):
            term_dict["coefficient"] = {
                "real": term.coefficient.real,
                "imag": term.coefficient.imag,
            }
        else:
            term_dict["coefficient"] = {"real": term.coefficient.real}

        dictionary["terms"].append(term_dict)

    return dictionary


def save_operator(operator: PauliRepresentation, filename: AnyPath) -> None:
    """Save a qubit operator to file.
    Args:
        operator: the operator to be saved
        filename: the name of the file
    """

    with open(filename, "w") as f:
        f.write(json.dumps(convert_op_to_dict(operator), indent=2))


def load_operator(file: LoadSource) -> PauliSum:
    """Load an operator object from a file.
    Args:
        file: the name of the file, or a file-like object.
    Returns:
        op: the operator.
    """

    if isinstance(file, str):
        with open(file, "r") as f:
            data = json.load(f)
    else:
        data = json.load(file)

    return convert_dict_to_op(data)


def save_operator_set(operator_set: List[PauliSum], filename: AnyPath) -> None:
    """Save a set of qubit operators to a file.

    Args:
        operator_set: a list of QubitOperator to be saved
        file: the name of the file
    """
    dictionary: Dict[str, Any] = {}
    dictionary["operators"] = []
    for operator in operator_set:
        dictionary["operators"].append(convert_op_to_dict(operator))
    with open(filename, "w") as f:
        f.write(json.dumps(dictionary, indent=2))


def load_operator_set(file: LoadSource) -> List[PauliSum]:
    """Load a set of qubit operators from a file.

    Args:
        file: the name of the file, or a file-like object.

    Returns:
        operator_set: a list of QubitOperator objects
    """
    if isinstance(file, str):
        with open(file, "r") as f:
            data = json.load(f)
    else:
        data = json.load(file)

    operator_set = []
    for operator_dict in data["operators"]:
        operator_set.append(convert_dict_to_op(operator_dict))
    return operator_set


def get_pauli_strings(operator: PauliRepresentation) -> List[str]:
    """Convert a operator into a list of Pauli strings.

    Args:
        operator: an operator to be converted

    Returns:
        pauli_strings: list of Pauli strings
    """
    pauli_strings = []
    for term in operator.terms:
        pauli_string = ""
        for pauli in sorted(term.operations):
            pauli_string += pauli[1] + str(pauli[0])
        pauli_strings.append(pauli_string)

    return pauli_strings
