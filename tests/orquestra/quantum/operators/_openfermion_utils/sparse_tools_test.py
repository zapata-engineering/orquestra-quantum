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
"""Tests for sparse_tools.py."""
import unittest

import numpy
import scipy
from scipy.sparse import csc_matrix

from orquestra.quantum.operators import (
    PauliSum,
    PauliTerm,
    expectation,
    get_sparse_operator,
)


class GetSparseOperatorTest(unittest.TestCase):
    def test_get_sparse_operator_n_qubits_too_small(self):
        with self.assertRaises(ValueError):
            get_sparse_operator(PauliTerm("X3"), 1)

    def test_get_sparse_operator_n_qubits_not_specified(self):
        expected = csc_matrix(
            ([1, 1, 1, 1], ([1, 0, 3, 2], [0, 1, 2, 3])), shape=(4, 4)
        )
        # Test PauliTerm.
        self.assertTrue(
            numpy.allclose(get_sparse_operator(PauliTerm("X1")).A, expected.A)
        )
        # Test PauliSum.
        self.assertTrue(
            numpy.allclose(
                get_sparse_operator(PauliSum([PauliTerm("X1")])).A, expected.A
            )
        )

    def test_get_sparse_operator_with_different_qubit_order(self):
        op = PauliTerm("(2+0j)*Z0*Z3")
        op2 = PauliTerm("(2+0j)*Z3*Z0")
        self.assertTrue(
            numpy.allclose(get_sparse_operator(op).A, get_sparse_operator(op2).A)
        )


class ExpectationTest(unittest.TestCase):
    def test_expectation_correct_sparse_matrix(self):
        operator = get_sparse_operator(PauliTerm("X0"), n_qubits=2)
        vector = numpy.array([0.0, 1.0j, 0.0, 1.0j])
        self.assertAlmostEqual(expectation(operator, vector), 2.0)

        density_matrix = scipy.sparse.csc_matrix(
            numpy.outer(vector, numpy.conjugate(vector))
        )
        self.assertAlmostEqual(expectation(operator, density_matrix), 2.0)

    def test_expectation_handles_column_vector(self):
        operator = get_sparse_operator(PauliTerm("X0"), n_qubits=2)
        vector = numpy.array([[0.0], [1.0j], [0.0], [1.0j]])
        self.assertAlmostEqual(expectation(operator, vector), 2.0)

    def test_expectation_correct_zero(self):
        operator = get_sparse_operator(PauliTerm("X0"), n_qubits=2)
        vector = numpy.array([1j, -1j, -1j, -1j])
        self.assertAlmostEqual(expectation(operator, vector), 0.0)
