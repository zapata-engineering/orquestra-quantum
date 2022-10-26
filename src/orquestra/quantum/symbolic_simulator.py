################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
from typing import Any, Dict, Optional

from orquestra.quantum.api.backend import StateVector
from orquestra.quantum.api.wavefunction_simulator import BaseWavefunctionSimulator
from orquestra.quantum.circuits import Circuit, Operation


class SymbolicSimulator(BaseWavefunctionSimulator):
    """A simulator computing wavefunction by consecutive gate matrix multiplication.

    Args:
        seed: the seed of the sampler
    """

    def __init__(
        self,
        *,
        seed: Optional[int] = None,
    ):
        super().__init__(seed=seed)

    def _get_wavefunction_from_native_circuit(
        self, circuit: Circuit, initial_state: StateVector
    ) -> StateVector:
        state = initial_state

        for operation in circuit.operations:
            state = operation.apply(state)

        return state

    def is_natively_supported(self, operation: Operation) -> bool:
        return True
