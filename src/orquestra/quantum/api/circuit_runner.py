################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
"""Definition of CircuitRunner protocol and ABC for implementing it."""
from abc import ABC, abstractmethod
from typing import List, Optional, Protocol, Sequence, Union

from ..circuits import Circuit
from ..distributions import MeasurementOutcomeDistribution
from ..measurements import Measurements


class CircuitRunner(Protocol):
    """Protocol for objects able to run circuits and collect measurements.

    This protocol is a pure interface and does not contain any default
    implementations. See BaseCircuitRunner ABC that can be extended to
    implement simple CircuitRunner.

    Note:
        This protocol was previously known as QuantumBackend.
    """

    def run_and_measure(self, circuit: Circuit, n_samples: int) -> Measurements:
        """Run circuit given number of times and return obtained measurements.

        Args:
            circuit: a circuit to be run.
            n_samples: minimum number of times the circuit should be run. This
              is sometimes called "number of shots" in other SDKs.
        Returns:
            A Measurements object containing *at least* n_samples.
        Raises:
            ValueError: if n_samples is not positive.

        Note:
            For some runners, the returned number of measurements might be
            actually larger than n_samples. Runner - independent code should
            take it into account and not rely on the assumption
            len(measurement.bitstrings) == n_samples
        """

    def run_batch_and_measure(
        self, circuits_batch: Sequence[Circuit], n_samples: Union[int, Sequence[int]]
    ) -> List[Measurements]:
        """Run multiple circuits and return measurements for each circuit.

        Essentially, this method performs the same as `run_and_measure` for
        a single circuit. However, some runners may be able to submit multiple
        circuits at once (thus possibly reducing costs), or run multiple
        circuits in parallel (thus increasing performance). For this reasons,
        usage of run_batch_and_measure is always recommended over
        run_and_measure if one needs to run multiple circuits.

        Args:
            circuits_batch: sequence of circuits to be run.
            n_samples: minimum number of times each circuit should be run. If
              a single integer N is passed, each circuit will be run at least
              N times. If a sequence of integers is passed, it must be of the
              same length as circuits_batch and i-th number is treated as
              minimum number of runs for i-th circuit.
        Returns:
            A list of measurements such that i-th object corresponds to i-th
            circuit.
        Raises:
            ValueError:
            - for integral n_samples if it is not positive
            - if any element of n_samples sequence is not positive, or n_samples
              has length differing from circuits_batch.

        Note:
            Similarly as with `run_and_measure` method, clients of circuit
            runners can only assume number of returned samples is greater
            or equal to the requested one. Furthermore, clients should not
            assume that each cicruict has been run the same number of times.
        """

    def get_measurement_outcome_distribution(
        self, circuit: Circuit, n_samples: Optional[int]
    ) -> MeasurementOutcomeDistribution:
        """Get a distribution of measurement outcomes from a given circuit.

        This is essentially the same as `run_and_measure`, except that
        a frequency of obtained outcomes is returned rather than a
        Measurements object.

        If `n_samples` is equal to None, supported samplers may returned
        exact distribution instead of an empirical one.
        """

    @property
    def n_jobs_executed(self) -> int:
        """Number of jobs executed by this runner.

        For circuit runners wrapping external resources or simulators,
        this number determines how many calls were made to the external
        library or API.
        """

    @property
    def n_circuits_executed(self) -> int:
        """Total number of circuits executed by this runner.

        Note:
            This number is counted with repetitions. For instance, if the
            same circuit has to be run multiple times to obtain required number
            of samples.
        """


class BaseCircuitRunner(ABC, CircuitRunner):
    """ABC for implementing simple circuit runners.

    To implement this ABC, override at least `_run_and_measure` method.
    In addition, if your runner supports more sophisticated way of
    running multiple circuits in a batch, you can override the
    `_run_batch_and_measure` method.

    Note:
        For purpose of counting executed jobs and circuits, this ABC
        makes assumption that each call to _run_and_measure constitutes
        a single job. If this assumption is not valid, you should avoid
        inheriting this class and instead implement CircuitRunner protocol
        separately.
    """

    def __init__(self):
        super().__init__()
        self._n_circuits_executed = 0
        self._n_jobs_executed = 0

    def run_and_measure(self, circuit: Circuit, n_samples: int) -> Measurements:
        """Run circuit given number of times and return obtained measurements.

        See docstrings of CircuitRunner protocol for exact description of
        parameters.

        Note:
            Subclasses of `BaseCircuitRunner` ABC should not override this method.
            Instead, implement `_run_and_measure`.
        """
        if n_samples <= 0:
            raise ValueError(f"Number of samples has to be positive, got {n_samples}")
        result = self._run_and_measure(circuit, n_samples)
        self._n_circuits_executed += 1
        self._n_jobs_executed += 1
        return result

    def run_batch_and_measure(
        self, circuits_batch: Sequence[Circuit], n_samples: Union[int, Sequence[int]]
    ) -> List[Measurements]:
        """Run multiple circuits and return measurements for each circuit.

        See docstrings of CircuitRunner protocol for exact description of
        parameters.

        Note:
            Subclasses of `BaseCircuitRunner` ABC should not override this method.
            Instead, implement `_run_and_measure` method and, if you have a
            dedicated way of running circuits' batch, `_run_batch_and_measure`
            method.
        """
        samples_per_circuit = (
            len(circuits_batch) * [n_samples]
            if isinstance(n_samples, int)
            else n_samples
        )
        if len(samples_per_circuit) != len(circuits_batch):
            raise ValueError(
                "Number of samples has to be an integer or a sequence of length "
                "equal to the length of batch. Length of batch: "
                f"{len(circuits_batch)}, length of n_samples: "
                f"{len(samples_per_circuit)}."
            )
        if any(n <= 0 for n in samples_per_circuit):
            raise ValueError(
                f"All numbers of samples have to be positive. Got: {n_samples}"
            )

        return self._run_batch_and_measure(circuits_batch, samples_per_circuit)

    @abstractmethod
    def _run_and_measure(self, circuit: Circuit, n_samples: int) -> Measurements:
        """Underlying implementation of running and measuring single circuit.

        Implementations of this method can assume that n_samples is positive.
        Furthermore, implementations of this method should not modify
        `self._n_jobs_executed` and `self._n_circuits_executed` fields.
        """

    def _run_batch_and_measure(
        self, batch: Sequence[Circuit], samples_per_circuit: Sequence[int]
    ):
        """Underlying implementation of running and measuring multiple circuits.

        Implementations of this method can assume that samples_per_circuit has
        the same length as `batch`, and each element in `samples_per_circuit`
        is positive.

        The basic implementation just runs `run_and_measure` for each circuit.
        It should only be overriden if runner implements some dedicated logic
        for running circuits' batch.
        """
        return [
            self.run_and_measure(circuit, n)
            for circuit, n in zip(batch, samples_per_circuit)
        ]

    @property
    def n_jobs_executed(self) -> int:
        """Number of executed jobs."""
        return self._n_jobs_executed

    @property
    def n_circuits_executed(self) -> int:
        """Number of executed circuits."""
        return self._n_circuits_executed

    def get_measurement_outcome_distribution(
        self, circuit: Circuit, n_samples: Optional[int]
    ) -> MeasurementOutcomeDistribution:
        """Get a distribution of measurement outcomes from a given circuit.

        Args:
            circuit: circuit to be sampled
            n_samples: number of times `circuit` should be sampled
        Returns:
            An object representing empirical distribution of measured bitstrings
        Raises:
            ValueError: if n_samples is None or n_samples is not positive.
        """
        if n_samples is None:
            raise ValueError(
                "This runner needs n_samples to compute measurement outcome"
                "distribution"
            )
        measurements = self.run_and_measure(circuit, n_samples)
        return measurements.get_distribution()
