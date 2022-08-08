import unittest

import numpy
from scipy.sparse import csc_matrix

from orquestra.quantum.operators import (
    PauliSum,
    PauliTerm,
    hermitian_conjugated,
    is_hermitian,
)


class HermitianConjugatedTest(unittest.TestCase):
    def test_hermitian_conjugated_pauli_term(self):
        """Test conjugating PauliTerms."""
        op = PauliTerm("X0*Y1", 2.0)
        op_hc = hermitian_conjugated(op)
        correct_op = op
        self.assertEqual(op_hc, correct_op)

        op = PauliTerm("X0*Y1", 2.0j)
        op_hc = hermitian_conjugated(op)
        correct_op = PauliTerm("X0*Y1", -2.0j)
        self.assertEqual(op_hc, correct_op)

    def test_hermitian_conjugated_pauli_sum(self):
        """Test conjugating PauliSums."""
        op = PauliTerm("X0*Y1", 2.0) + PauliTerm("Z4*X5*Y7", 3.0j)
        op_hc = hermitian_conjugated(op)
        correct_op = PauliTerm("X0*Y1", 2.0) + PauliTerm("Z4*X5*Y7", -3.0j)
        self.assertEqual(op_hc, correct_op)

    def test_hermitian_conjugate_empty(self):
        op = PauliSum()
        op = hermitian_conjugated(op)
        self.assertEqual(op, PauliSum())

    def test_exceptions(self):
        with self.assertRaises(TypeError):
            _ = hermitian_conjugated(1)


class IsHermitianTest(unittest.TestCase):
    def test_pauli_operator_zero(self):
        op = PauliSum()
        self.assertTrue(is_hermitian(op))

    def test_pauli_operator_identity(self):
        op = PauliTerm.identity()
        self.assertTrue(is_hermitian(op))

    def test_pauli_operator_nonhermitian(self):
        op = PauliTerm("X0*Y2*Z5", 1.0 + 2.0j)
        self.assertFalse(is_hermitian(op))

    def test_pauli_operator_hermitian(self):
        op = PauliTerm("X0*Y2*Z5", 1.0 + 2.0j)
        op += PauliTerm("X0*Y2*Z5", 1.0 - 2.0j)
        self.assertTrue(is_hermitian(op))

    def test_sparse_matrix_and_numpy_array_zero(self):
        op = numpy.zeros((4, 4))
        self.assertTrue(is_hermitian(op))
        op = csc_matrix(op)
        self.assertTrue(is_hermitian(op))

    def test_sparse_matrix_and_numpy_array_identity(self):
        op = numpy.eye(4)
        self.assertTrue(is_hermitian(op))
        op = csc_matrix(op)
        self.assertTrue(is_hermitian(op))

    def test_sparse_matrix_and_numpy_array_nonhermitian(self):
        op = numpy.arange(16).reshape((4, 4))
        self.assertFalse(is_hermitian(op))
        op = csc_matrix(op)
        self.assertFalse(is_hermitian(op))

    def test_sparse_matrix_and_numpy_array_hermitian(self):
        op = numpy.arange(16, dtype=complex).reshape((4, 4))
        op += 1.0j * op
        op += op.T.conj()
        self.assertTrue(is_hermitian(op))
        op = csc_matrix(op)
        self.assertTrue(is_hermitian(op))

    def test_exceptions(self):
        with self.assertRaises(TypeError):
            _ = is_hermitian("a")
