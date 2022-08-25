################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################

from typing import Protocol, Sequence, List, Union, Optional
from ..circuits import Circuit
from ..measurements import Measurements
from ..distributions import MeasurementOutcomeDistribution


class CircuitRunner(Protocol):
    # check_if_n_samples_is_greater_or_equal_number_of_measurements
    # check_if_len_of_bitstrings_is_equal_to_number_of_qubits_in_bitstrings
    # check_if_fails_when_n_samples_is_less_or_equal_0
    # check_if_fails_when_n_samples_is_less_or_equal_0
    def run_and_measure(self, circuit: Circuit, n_samples: int) -> Measurements:
        """
        TODO
        """

    # Same as run_and_measure
    # check_if_len_of_output_is_equal_number_of_circuits
    # check_if_fails_if_len_of_n_samples_doesnt_match_len_of_circuits_batch
    def run_batch_and_measure(
        self, circuits_batch: Sequence[Circuit], n_samples: Union[int, Sequence[int]]
    ) -> List[Measurements]:
        """
        TODO
        """
        # TODO: mention that this default implementation is kind of optional.
        samples_per_circuit = (
            len(circuits_batch) * [n_samples]
            if isinstance(n_samples, int)
            else n_samples
        )
        if len(samples_per_circuit) != len(circuits_batch):
            raise ValueError(
                "Number of samples has to be an integer or a sequence of length "
                "equal to the length of batch. Length of batch: "
                f"{len(circuits_batch)}, length of n_samples: {len(n_samples)}."
            )
        return [
            self.run_and_measure(circuit, n)
            for circuit, n in zip(circuits_batch, samples_per_circuit)
        ]

    # check_if_number_bits_in_distribution_is_equal_to_number_of_qubits_in_circuit
    # check_if_fails_if_n_samples_is_less_or_equal_0
    def get_measurement_outcome_distribution(
        self, circuit: Circuit, n_samples: Optional[int]
    ) -> MeasurementOutcomeDistribution:
        """
        TODO
        Calculates a measurement outcome distribution.

        Args:
            circuit: quantum circuit to be executed.

        Returns:
            Distribution of getting specific bistrings.

        """
        # TODO
        measurements = self.run_and_measure(circuit, n_samples)
        return measurements.get_distribution()
