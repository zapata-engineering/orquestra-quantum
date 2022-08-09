################################################################################
# © Copyright 2020-2022 Zapata Computing Inc.
################################################################################
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

"""This module contains functions from openfermion adapted for use with Orquestra's
PauliTerm and PauliSum classes.
"""
from .config import EQ_TOLERANCE
from .operator_utils import hermitian_conjugated, is_hermitian
from .sparse_tools import expectation, get_sparse_operator
