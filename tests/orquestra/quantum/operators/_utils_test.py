################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
import random
import unittest

import numpy as np

from orquestra.quantum.operators import PauliSum, PauliTerm, get_sparse_operator
from orquestra.quantum.utils import RNDSEED
from orquestra.quantum.wavefunction import Wavefunction


class Testpaulioperator(unittest.TestCase):
    def test_build_paulioperator_from_coeffs_and_labels(self):
        from orquestra.quantum.operators._utils import (
            get_pauliop_from_coeffs_and_labels,
        )

        # Given
        test_op = PauliTerm("(3j)*Y0*X1*Z2*X4")
        coeffs = [3.0j]
        labels = [[2, 1, 3, 0, 1]]

        # When
        build_op = get_pauliop_from_coeffs_and_labels(coeffs, labels)

        # Then
        self.assertEqual(test_op, build_op)

    def test_pauliop_matrix_converion(self):
        from orquestra.quantum.operators._utils import get_pauliop_from_matrix

        # Given
        m = 4
        n = 2**m
        TOL = 10**-15
        random.seed(RNDSEED)
        A = np.array([[random.uniform(-1, 1) for x in range(n)] for y in range(n)])

        # When
        A_pauliop = get_pauliop_from_matrix(A)
        A_pauliop_matrix = np.array(get_sparse_operator(A_pauliop).todense())
        test_matrix = A_pauliop_matrix - A

        # Then
        for row in test_matrix:
            for elem in row:
                self.assertEqual(abs(elem) < TOL, True)

    def test_generate_random_pauliop(self):
        from orquestra.quantum.operators._utils import generate_random_pauliop

        # Given
        nqubits = 4
        nterms = 5
        nlocality = 2
        max_coeff = 1.5
        fixed_coeff = False

        # When
        op = generate_random_pauliop(nqubits, nterms, nlocality, max_coeff, fixed_coeff)
        # Then
        self.assertEqual(len(op.terms), nterms)
        for term in op.terms:
            self.assertLess(sorted(term.qubits)[-1], nqubits)  # max qubit < nqubits
            self.assertEqual(len(term), nlocality)
            self.assertLessEqual(np.abs(term.coefficient), max_coeff)

        # Given
        fixed_coeff = True
        # When
        op = generate_random_pauliop(nqubits, nterms, nlocality, max_coeff, fixed_coeff)
        # Then
        self.assertEqual(len(op.terms), nterms)
        for term in op.terms:
            self.assertEqual(np.abs(term.coefficient), max_coeff)

    def test_evaluate_operator(self):
        from orquestra.quantum.measurements import ExpectationValues
        from orquestra.quantum.operators._utils import evaluate_operator

        # Given
        qubit_op = PauliSum("0.5 + 0.5*Z1")
        expectation_values = ExpectationValues([0.5, 0.5])
        # When
        value_estimate = evaluate_operator(qubit_op, expectation_values)
        # Then
        self.assertAlmostEqual(value_estimate, 0.5)

    def test_evaluate_operator_list(self):
        from orquestra.quantum.measurements import ExpectationValues
        from orquestra.quantum.operators._utils import evaluate_operator_list

        # Given
        qubit_op_list = [PauliSum("0.5 + 0.5*Z1"), PauliSum("0.3*X1 + 0.2*Y2")]
        expectation_values = ExpectationValues([0.5, 0.5, 0.4, 0.6])
        # When
        value_estimate = evaluate_operator_list(qubit_op_list, expectation_values)
        # Then
        self.assertAlmostEqual(value_estimate, 0.74)

    def test_reverse_qubit_order(self):
        from orquestra.quantum.operators._utils import reverse_qubit_order

        # Given
        op1 = PauliTerm("Z0*Z1")
        op2 = PauliTerm("Z1*Z0")

        # When/Then
        self.assertEqual(op1, reverse_qubit_order(op2))

        # Given
        op1 = PauliTerm("Z0")
        op2 = PauliTerm("Z1")

        # When/Then
        self.assertEqual(op1, reverse_qubit_order(op2, n_qubits=2))
        self.assertEqual(op2, reverse_qubit_order(op1, n_qubits=2))

    def test_get_expectation_value(self):
        from orquestra.quantum.operators._utils import get_expectation_value

        """Check <Z0> and <Z1> for the state |100>"""
        # Given
        wf = Wavefunction([0, 1, 0, 0, 0, 0, 0, 0])
        op1 = PauliTerm("Z0")
        op2 = PauliTerm("Z1")

        # When
        exp_op1 = get_expectation_value(op1, wf, True)
        exp_op2 = get_expectation_value(op2, wf, True)

        # Then
        self.assertAlmostEqual(-1, exp_op1)
        self.assertAlmostEqual(1, exp_op2)
