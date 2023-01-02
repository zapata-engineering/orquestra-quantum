################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
"""The WavefunctionSimulator protocol and ABC for implementing it."""
from abc import abstractmethod
from typing import Optional, Protocol

import numpy as np

from ..circuits import Circuit, GateOperation, Operation, split_circuit
from ..distributions import (
    MeasurementOutcomeDistribution,
    create_bitstring_distribution_from_probability_distribution,
)
from ..measurements import Measurements
from ..operators import PauliRepresentation, get_expectation_value
from ..typing import StateVector
from ..wavefunction import Wavefunction, sample_from_wavefunction
from .circuit_runner import BaseCircuitRunner, CircuitRunner


class WavefunctionSimulator(CircuitRunner, Protocol):
    """The protocol for objects able to compute the wavefunction."""

    def get_wavefunction(
        self, circuit: Circuit, initial_state: Optional[StateVector] = None
    ) -> Wavefunction:
        """Get a wavefunction of a circuit starting from a given initial state.

        Args:
            circuit: circuit whose wavefunction is to be computed.
            initial_state: state used as an input to the `circuit`. If not provided,
              `|0...0>` will be used.
        Returns:
            Wavefunction comprising 2 ** n amplitudes, where n is number of qubits in
            `circuit`.
        """

    def get_exact_expectation_values(
        self, circuit: Circuit, operator: PauliRepresentation
    ) -> float:
        """Get an exact expectation value of an operator.

        Args:
            circuit: circuit constructing state for which expectation value
              is to be computed.
            operator: operator of which expectation value should be computed.
        Returns:
            An expectation value as a single floating point number.
        """


class BaseWavefunctionSimulator(BaseCircuitRunner, WavefunctionSimulator):
    """ABC for implementing simple wavefunction simulators.

    This ABC is build around _get_wavefunction_from_native_circuit
    method. In general, most simulators wrap some third-party resource (
    a library, service, API etc.), which can only consume circuits comprising
    operations from a given set. Such operations are called "native" to
    the given simulator, whereas other operations are called "nonnative".

    The idea of simulating arbitrary circuit is thus as follows:

    - split circuit into alternating consecutive parts of only native and
      only "nonnative" operations.

    - start with some initial state

    - for each part:

      - if it is native, run it via third-party resource, save the new
        statevector

      - if it is nonnative, apply each operation in the sequence using
        operation.apply(previous_statevector). Save the new statevector.

      Last saved statevector is the wavefunction of the total circuit.

      For this to work, subclasses of this ABC should implement
      _get_wavefunction_from_the_native_circuit method.

    Note:
        Since this class inherits all the limitations of BaseCircuitRunner.
        The _run_and_measure function is implemented via sampling from the
        wavefunction. Care must be taken when using third-party service that
        implements more sophisticated/more performant sampling method not
        involving direct computation of the whole wavefunction. In such cases,
        using BaseWavefunctionSimulator ABC will likely result in huge
        performance hit.
    """

    def __init__(self, *, seed: Optional[int] = None):
        super().__init__()
        self.seed = seed

    def run_and_measure(self, circuit: Circuit, n_samples: int) -> Measurements:
        """Run circuit given number of times and return obtained measurements.

        See docstrings of CircuitRunner protocol for exact description of
        parameters.

        Note:
            For BaseWavefunctionSimulator, running a circuit comprising computing
            its wavefunction and then sampling from the probability distribution
            that it defines.
        """
        if n_samples <= 0:
            raise ValueError(f"Number of samples has to be positive, got {n_samples}")
        result = self._run_and_measure(circuit, n_samples)
        return result

    def _run_and_measure(self, circuit: Circuit, n_samples: int) -> Measurements:
        if circuit.free_symbols:
            raise ValueError("Cannot sample from circuit with unbound free symbols")
        wavefunction = self.get_wavefunction(circuit)
        return Measurements(
            sample_from_wavefunction(wavefunction, n_samples, self.seed)
        )

    def get_wavefunction(
        self, circuit: Circuit, initial_state: Optional[StateVector] = None
    ) -> Wavefunction:
        """Get a wavefunction of a circuit starting from a given initial state.

        See docstrings of WavefunctionSimulator protocol for exact description of
        parameters.
        """
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
        """Get a wavefunction of a native circuit starting from given initial state.

        This is the only mandatory method to be implemented by classes extending
        this ABC.

        Args:
            circuit: circuit whose wavefunction is to be computed. This method might
              assume that it comprises only operations native to this simulator.
            initial_state: state used as an input to the `circuit`. If not provided,
              |0...0> will be used.
        Returns:
            The computed wavefunction.
        """

    def get_exact_expectation_values(
        self, circuit: Circuit, operator: PauliRepresentation
    ) -> float:
        """Get an exact expectation value of an operator.

        See docstrings of WavefunctionSimulator protocol for exact description of
        parameters.
        """
        wavefunction = self.get_wavefunction(circuit)
        # Casting to real, because any non-zero imaginary part must mean some
        # numerical inaccuracy.
        return get_expectation_value(operator, wavefunction).real

    def is_natively_supported(self, operation: Operation) -> bool:
        """Determine if given operation is natively supported by this simulator.

        By default, operation is natively supported iff it is a GateOperation.
        However, this doesn't have to be true for every simulator.
        If the set of natively supported operations is different for
        some simulator, this method should be changed accordingly.

        Args:
            operation: operation to be checked.
        Returns:
              True if `operation` is natively supported and False otherwise.
        """
        return isinstance(operation, GateOperation)

    def get_measurement_outcome_distribution(
        self, circuit: Circuit, n_samples: Optional[int] = None
    ) -> MeasurementOutcomeDistribution:
        """Get a distribution of measurement outcomes from a given circuit.

        Args:
            circuit: circuit to be sampled
            n_samples: number of times `circuit` should be sampled or None, if
            exact computation should be performed.
        Returns:
            An object representing empirical (for integral n_samples) or theoretical
            (n_samples is None) distribution of measured bitstrings
        """
        if n_samples is None:
            wavefunction = self.get_wavefunction(circuit)
            return create_bitstring_distribution_from_probability_distribution(
                wavefunction.get_probabilities()
            )
        else:
            # Get the expectation values
            measurements = self.run_and_measure(circuit, n_samples)
            return measurements.get_distribution()
