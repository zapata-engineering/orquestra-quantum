################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
from typing import Any, Dict, Optional

from sympy import Symbol

from orquestra.quantum.api.backend import QuantumSimulator, StateVector
from orquestra.quantum.api.gate_model_simulator import GateModelSimulator
from orquestra.quantum.circuits import Circuit, Operation
from orquestra.quantum.circuits.layouts import CircuitConnectivity
from orquestra.quantum.measurements import Measurements
from orquestra.quantum.wavefunction import sample_from_wavefunction


class SymbolicSimulator(GateModelSimulator):
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

    def run_circuit_and_measure(
        self,
        circuit: Circuit,
        n_samples: int,
    ) -> Measurements:
        """Run a circuit and measure a certain number of bitstrings

        Args:
            circuit: the circuit to prepare the state
            n_samples: the number of bitstrings to sample
        """
        wavefunction = self.get_wavefunction(circuit)

        if wavefunction.free_symbols:
            raise ValueError(
                "Cannot sample from wavefunction with symbolic parameters."
            )

        bitstrings = sample_from_wavefunction(wavefunction, n_samples, self.seed)
        return Measurements(bitstrings)

    def _get_wavefunction_from_native_circuit(
        self, circuit: Circuit, initial_state: StateVector
    ) -> StateVector:
        state = initial_state

        for operation in circuit.operations:
            state = operation.apply(state)

        return state

    def is_natively_supported(self, operation: Operation) -> bool:
        return True
