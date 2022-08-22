################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################

import numpy as np
from typing import Protocol, Iterable, List, Union, Optional
from ..circuits import Circuit, split_circuit, Operation
from ..measurements import Measurements, ExpectationValues
from .circuit_runner import CircuitRunner
from abc import ABC, abstractmethod
from ..wavefunction import sample_from_wavefunction, Wavefunction
from ..operators import PauliRepresentation, get_expectation_value
from ..distributions import (
    MeasurementOutcomeDistribution,
    create_bitstring_distribution_from_probability_distribution,
)

# TODO: Statevector definition should be moved to typing (I think?)
from .backend import StateVector


class GateModelSimulator(ABC, CircuitRunner):
    def __init__(self, *, seed: Optional[int] = None):
        # TODO
        self.n_circuits_executed = 0
        self.n_jobs_executed = 0
        self.supports_batching = False
        self.seed = None

    def run_and_measure(self, circuit: Circuit, n_samples: int) -> Measurements:
        """
        TODO
        """

        return self._run_and_measure(circuit, n_samples)

    def _run_and_measure(self, circuit: Circuit, n_samples: int) -> Measurements:
        """
        TODO
        """
        wavefunction = self.get_wavefunction(circuit)

        bitstrings = sample_from_wavefunction(wavefunction, n_samples, self.seed)
        return Measurements(bitstrings)

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
            self.n_jobs_executed += 1
            if is_supported:
                self.n_circuits_executed += 1
                state = self._get_wavefunction_from_native_circuit(subcircuit, state)
            else:
                for operation in subcircuit.operations:
                    state = operation.apply(state)

        return Wavefunction(state)

    @abstractmethod
    def _get_wavefunction_from_native_circuit(
        self, circuit: Circuit, initial_state: StateVector
    ) -> StateVector:
        """Get wavefunction from circuit comprising only natively-supported operations.
        TODO
        Args:
            circuit: circuit to simulate. Implementers of this function might assume
              that this circuit comprises only natively-supported operations as decided
              by self._is_supported predicate.
            initial_state: amplitudes of the initial state.
        Returns:
            StateVector representing amplitudes of final circuit state.
        """

    def get_exact_expectation_values(
        self, circuit: Circuit, operator: PauliRepresentation
    ) -> ExpectationValues:
        """Calculates the expectation values for given operator, based on the exact
        quantum state produced by circuit.

        Args:
            circuit: quantum circuit to be executed.
            operator: Operator for which we calculate the expectation value.
        """
        wavefunction = self.get_wavefunction(circuit)
        return get_expectation_value(operator, wavefunction)

    @abstractmethod
    def is_natively_supported(self, operation: Operation) -> bool:
        """TODO"""

    def get_measurement_outcome_distribution(
        self, circuit: Circuit, n_samples: Optional[int] = None
    ) -> MeasurementOutcomeDistribution:
        """Calculates a bitstring distribution.

        Args:
            circuit: quantum circuit to be executed.

        Returns:
            Probability distribution of getting specific bistrings.
        """
        if n_samples is None:
            wavefunction = self.get_wavefunction(circuit)
            return create_bitstring_distribution_from_probability_distribution(
                wavefunction.get_probabilities()
            )
        else:
            # Get the expectation values
            measurements = self.run_circuit_and_measure(circuit, n_samples)
            return measurements.get_distribution()
