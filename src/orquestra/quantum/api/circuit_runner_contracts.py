################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
import numpy as np

from .circuit_runner import CircuitRunner
from ..circuits import RX, RY, RZ, Circuit, H, CNOT


_example_circuits = (
    Circuit([RY(np.pi / 5)(0)]),
    Circuit([H(0), CNOT(0, 1)]),
    Circuit([H(0), CNOT(0, 2), H(1)]),
    Circuit([RX(np.pi / 2)(0), H(2), RZ(np.pi / 3)(1), CNOT(0, 3)]),
)

_example_n_samples = (5, 10, 15, 20)

_invalid_n_samples = (0, -1, -10)


class _ValidateRunAndMeasure:
    """Contracts relating to run_and_measure."""

    @staticmethod
    def returns_number_of_measurements_greater_or_equal_to_n_samples(
        runner: CircuitRunner,
    ):
        circuit = _example_circuits[0]

        def _count_num_measurements(n_samples):
            return len(runner.run_and_measure(circuit, n_samples=n_samples).bitstrings)

        return all(
            _count_num_measurements(n_samples) >= n_samples
            for n_samples in _example_n_samples
        )

    @staticmethod
    def returns_bitstrings_with_length_equal_to_number_of_qubits_in_circuit(
        runner: CircuitRunner,
    ):
        return all(
            len(bitstring) == circuit.n_qubits
            for circuit in _example_circuits
            for bitstring in runner.run_and_measure(circuit, n_samples=10).bitstrings
        )

    @staticmethod
    def raises_value_error_if_n_samples_is_nonpositive(runner: CircuitRunner):
        for n_samples in _invalid_n_samples:
            try:
                runner.run_and_measure(_example_circuits[0], n_samples=n_samples)
                return False
            except ValueError:
                pass
            except Exception:
                return False

        return True


class _ValidateRunBatchAndMeasure:
    @staticmethod
    def returns_measurement_object_for_each_circuit_in_batch(runner: CircuitRunner):
        return len(
            runner.run_batch_and_measure(_example_circuits, _example_n_samples)
        ) == len(_example_circuits)

    @staticmethod
    def returns_number_of_measurements_greater_or_equal_to_n_samples(
        runner: CircuitRunner,
    ):
        def _when_n_samples_is_the_same_for_each_circuit():
            return all(
                len(measurements.bitstrings) >= 10
                for measurements in runner.run_batch_and_measure(
                    _example_circuits, n_samples=10
                )
            )

        def _when_n_samples_is_different_for_each_circuit():
            return all(
                len(measurement.bitstrings) >= n_samples
                for measurement, n_samples in zip(
                    runner.run_batch_and_measure(_example_circuits, _example_n_samples),
                    _example_n_samples,
                )
            )

        # We make sure to run each subtest, otherwise short-circuiting of
        # circuits might result in missing some errors
        subtests = [
            _when_n_samples_is_different_for_each_circuit(),
            _when_n_samples_is_the_same_for_each_circuit(),
        ]
        return all(subtests)

    @staticmethod
    def returns_bitstrings_with_length_equal_to_number_of_qubits_in_circuit(
        runner: CircuitRunner,
    ):
        return all(
            len(bitstring) == circuit.n_qubits
            for measurements, circuit in zip(
                runner.run_batch_and_measure(_example_circuits, _example_n_samples),
                _example_circuits,
            )
            for bitstring in measurements.bitstrings
        )

    @staticmethod
    def raises_value_error_if_n_samples_is_nonpositive(runner: CircuitRunner):
        def _for_all_circuits():
            try:
                runner.run_batch_and_measure(
                    _example_circuits, n_samples=_invalid_n_samples
                )
                return False
            except ValueError:
                pass
            except Exception:
                return False

            return True

        def _for_at_least_one_circuit():
            n_samples = (-1,) + (len(_example_circuits) - 1) * (10,)
            try:
                runner.run_batch_and_measure(_example_circuits, n_samples=n_samples)
                return False
            except ValueError:
                pass
            except Exception:
                return False

            return True

        return _for_all_circuits() and _for_at_least_one_circuit()

    @staticmethod
    def raises_value_error_if_len_of_n_samples_does_not_match_len_of_batch(
        runner: CircuitRunner,
    ):
        invalid_n_samples = (
            (len(_example_circuits) + 1) * (10,),
            (len(_example_circuits) - 1) * (10,),
        )

        for n_samples in invalid_n_samples:
            try:
                runner.run_batch_and_measure(_example_circuits, n_samples)
                return False
            except ValueError:
                pass
            except Exception:
                return False

        return True


class _ValidateMeasurementOutcomeDistribution:
    @staticmethod
    def returns_distribution_with_number_of_bits_same_as_circuit_n_qubits(
        runner: CircuitRunner,
    ):
        return all(
            (
                runner.get_measurement_outcome_distribution(
                    circuit, n_samples=10
                ).get_number_of_subsystems()
                == circuit.n_qubits
            )
            for circuit in _example_circuits
        )

    @staticmethod
    def raises_value_error_if_n_samples_is_nonpositive(runner: CircuitRunner):
        for n_samples in _invalid_n_samples:
            try:
                runner.get_measurement_outcome_distribution(
                    _example_circuits[0], n_samples=n_samples
                )
                return False
            except ValueError:
                pass
            except Exception:
                return False

        return True


CIRCUIT_RUNNER_CONTRACTS = [
    _ValidateRunAndMeasure.returns_number_of_measurements_greater_or_equal_to_n_samples,
    _ValidateRunAndMeasure.returns_bitstrings_with_length_equal_to_number_of_qubits_in_circuit,
    _ValidateRunAndMeasure.raises_value_error_if_n_samples_is_nonpositive,
    _ValidateRunBatchAndMeasure.returns_number_of_measurements_greater_or_equal_to_n_samples,
    _ValidateRunBatchAndMeasure.returns_bitstrings_with_length_equal_to_number_of_qubits_in_circuit,
    _ValidateRunBatchAndMeasure.returns_measurement_object_for_each_circuit_in_batch,
    _ValidateRunBatchAndMeasure.raises_value_error_if_n_samples_is_nonpositive,
    _ValidateRunBatchAndMeasure.raises_value_error_if_len_of_n_samples_does_not_match_len_of_batch,
    _ValidateMeasurementOutcomeDistribution.returns_distribution_with_number_of_bits_same_as_circuit_n_qubits,
    _ValidateMeasurementOutcomeDistribution.raises_value_error_if_n_samples_is_nonpositive,
]
