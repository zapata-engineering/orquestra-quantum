################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
import os
import unittest

import numpy as np

from orquestra.quantum.operators import PauliTerm, hermitian_conjugated
from orquestra.quantum.operators._io import (
    convert_dict_to_op,
    convert_op_to_dict,
    get_pauli_strings,
    load_operator,
    load_operator_set,
    save_operator,
    save_operator_set,
)


class Testoperator(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_op_to_dict_io(self):
        # Given
        op = PauliTerm("Y0*X1*Z2*X4", 3.0j)
        op += hermitian_conjugated(op)

        # When
        op_dict = convert_op_to_dict(op)
        recreated_op = convert_dict_to_op(op_dict)

        # Then
        self.assertEqual(recreated_op, op)

    def test_operator_io(self):
        # Given
        op = PauliTerm("Y0*X3*Z8*X11", 3.0j)

        # When
        save_operator(op, "op.json")
        loaded_op = load_operator("op.json")

        # Then
        self.assertEqual(op, loaded_op)
        os.remove("op.json")

    def test_operator_set_io(self):
        op1 = PauliTerm("Y0*X3*Z8*X11", 3.0j)
        op2 = PauliTerm("Y0*X1*Z7*X14", 1.0j)

        operator_set = [op1, op2]
        save_operator_set(operator_set, "operator_set.json")
        loaded_operator_set = load_operator_set("operator_set.json")
        for i in range(len(operator_set)):
            self.assertEqual(operator_set[i], loaded_operator_set[i])
        os.remove("operator_set.json")

    def test_op_io(self):
        # Given
        op = PauliTerm("Y0*X1*Z2*X4", 3.0j)

        # When
        save_operator(op, "op.json")
        loaded_op = load_operator("op.json")

        # Then
        self.assertEqual(op, loaded_op)
        os.remove("op.json")

    def test_get_pauli_strings(self):
        operator = (
            PauliTerm("X0*Y1") - 0.5 * PauliTerm("Y1") + 0.5 * PauliTerm.identity()
        )
        constructed_list = get_pauli_strings(operator)
        target_list = ["X0Y1", "Y1", ""]
        self.assertListEqual(constructed_list, target_list)
