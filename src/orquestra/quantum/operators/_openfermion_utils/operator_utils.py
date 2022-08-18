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
"""This module provides generic tools for classes in orquestra.quantum.operators"""

import numpy
from scipy.sparse import spmatrix

from .._pauli_operators import PauliSum, PauliTerm
from .config import EQ_TOLERANCE


def hermitian_conjugated(operator):
    """Return Hermitian conjugate of operator."""
    # Handle PauliSum
    if isinstance(operator, PauliSum):
        conjugate_operator = PauliSum()
        for term in operator.terms:
            conjugate_operator += term.copy(term.coefficient.conjugate())

    # Handle PauliTerm
    elif isinstance(operator, PauliTerm):
        conjugate_operator = operator.copy(operator.coefficient.conjugate())

    # Handle sparse matrix
    elif isinstance(operator, spmatrix):
        conjugate_operator = operator.getH()

    # Handle numpy array
    elif isinstance(operator, numpy.ndarray):
        conjugate_operator = operator.T.conj()

    # Unsupported type
    else:
        raise TypeError(
            "Taking the hermitian conjugate of a {} is not "
            "supported.".format(type(operator).__name__)
        )

    return conjugate_operator


def is_hermitian(operator):
    """Test if operator is Hermitian."""
    # Handle QubitOperator
    if isinstance(operator, (PauliSum, PauliTerm)):
        return operator == hermitian_conjugated(operator)

    # Handle sparse matrix
    elif isinstance(operator, spmatrix):
        difference = operator - hermitian_conjugated(operator)
        discrepancy = 0.0
        if difference.nnz:
            discrepancy = max(abs(difference.data))
        return discrepancy < EQ_TOLERANCE

    # Handle numpy array
    elif isinstance(operator, numpy.ndarray):
        difference = operator - hermitian_conjugated(operator)
        discrepancy = numpy.amax(abs(difference))
        return discrepancy < EQ_TOLERANCE

    # Unsupported type
    else:
        raise TypeError(
            "Checking whether a {} is hermitian is not "
            "supported.".format(type(operator).__name__)
        )
