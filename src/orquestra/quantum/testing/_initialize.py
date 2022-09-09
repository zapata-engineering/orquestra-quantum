################################################################################
# © Copyright 2020-2022 Zapata Computing Inc.
################################################################################

import numpy as np

from ..wavefunction import Wavefunction


def create_random_wavefunction(n_qubits, seed=None):
    """Create a random wavefunction for testing purposes."""
    if seed:
        np.random.seed(seed)

    random_vector = [
        complex(a, b)
        for a, b in zip(np.random.rand(2**n_qubits), np.random.rand(2**n_qubits))
    ]
    normalization_factor = np.sqrt(np.sum(np.abs(random_vector) ** 2))
    random_vector /= normalization_factor

    return Wavefunction(random_vector)
