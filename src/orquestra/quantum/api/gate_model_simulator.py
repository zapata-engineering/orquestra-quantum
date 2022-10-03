################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################

import numpy as np
from typing import Protocol, Optional
from ..circuits import Circuit, split_circuit, Operation, GateOperation
from ..measurements import Measurements
from .circuit_runner import CircuitRunner, BaseCircuitRunner
from abc import abstractmethod
from ..wavefunction import sample_from_wavefunction, Wavefunction
from ..operators import PauliRepresentation, get_expectation_value
from ..distributions import (
    MeasurementOutcomeDistribution,
    create_bitstring_distribution_from_probability_distribution,
)

# TODO: Statevector definition should be moved to typing (I think?)
from .backend import StateVector


class WavefunctionSimulator(CircuitRunner, Protocol):

    def get_wavefunction(
        self, circuit: Circuit, initial_state: Optional[StateVector] = None
    ) -> Wavefunction:
        pass

    def get_exact_expectation_values(
        self, circuit: Circuit, operator: PauliRepresentation
    ) -> complex:
        pass


class BaseWavefunctionSimulator(BaseCircuitRunner, WavefunctionSimulator):
    def __init__(self, *, seed: Optional[int] = None):
        super().__init__()
        self.seed = seed

    def run_and_measure(self, circuit: Circuit, n_samples: int) -> Measurements:
        return self._run_and_measure(circuit, n_samples)

    def _run_and_measure(self, circuit: Circuit, n_samples: int) -> Measurements:
        wavefunction = self.get_wavefunction(circuit)
        return Measurements(
            sample_from_wavefunction(wavefunction, n_samples, self.seed)
        )

    def get_wavefunction(
        self, circuit: Circuit, initial_state: Optional[StateVector] = None
    ) -> Wavefunction:
        state: StateVector
        if initial_state is None:
            state = np.zeros(2**circuit.n_qubits)
            state[0] = 1
        else:
            state = initial_state

        for is_supported, subcircuit in split_circuit(
            circuit, self.is_natively_supported
        ):
            # Native subcircuits are passed through to the underlying simulator.
            # They also count towards number of circuits and number of jobs run.
            self._n_jobs_executed += 1
            if is_supported:
                self._n_circuits_executed += 1
                state = self._get_wavefunction_from_native_circuit(subcircuit, state)
            else:
                for operation in subcircuit.operations:
                    state = operation.apply(state)

        return Wavefunction(state)

    @abstractmethod
    def _get_wavefunction_from_native_circuit(
        self, circuit: Circuit, initial_state: StateVector
    ) -> StateVector:
        pass

    def get_exact_expectation_values(
        self, circuit: Circuit, operator: PauliRepresentation
    ) -> complex:
        wavefunction = self.get_wavefunction(circuit)
        # Casting to real, because any non-zero imaginary part must mean some
        # numerical inaccuracy.
        return get_expectation_value(operator, wavefunction).real

    @abstractmethod
    def is_natively_supported(self, operation: Operation) -> bool:
        return isinstance(operation, GateOperation)

    def get_measurement_outcome_distribution(
        self, circuit: Circuit, n_samples: Optional[int] = None
    ) -> MeasurementOutcomeDistribution:
        if n_samples is None:
            wavefunction = self.get_wavefunction(circuit)
            return create_bitstring_distribution_from_probability_distribution(
                wavefunction.get_probabilities()
            )
        else:
            # Get the expectation values
            measurements = self.run_and_measure(circuit, n_samples)
            return measurements.get_distribution()
