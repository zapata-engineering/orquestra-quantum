################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
import numpy as np

from ..circuits import CNOT, RX, RY, RZ, Circuit, H
from .circuit_runner import CircuitRunner

_EXAMPLE_CIRCUITS = (
    Circuit([RY(np.pi / 5)(0)]),
    Circuit([H(0), CNOT(0, 1)]),
    Circuit([H(0), CNOT(0, 2), H(1)]),
    Circuit([RX(np.pi / 2)(0), H(2), RZ(np.pi / 3)(1), CNOT(0, 3)]),
)

_EXAMPLE_N_SAMPLES = (5, 10, 15, 20)

_INVALID_N_SAMPLES = (0, -1, -10)


class _ValidateRunAndMeasure:
    """Contracts relating to run_and_measure."""

    @staticmethod
    def returns_number_of_measurements_greater_or_equal_to_n_samples(
        runner: CircuitRunner,
    ):
        circuit = _EXAMPLE_CIRCUITS[0]

        def _count_num_measurements(n_samples):
            return len(runner.run_and_measure(circuit, n_samples=n_samples).bitstrings)

        return all(
            _count_num_measurements(n_samples) >= n_samples
            for n_samples in _EXAMPLE_N_SAMPLES
        )

    @staticmethod
    def returns_bitstrings_with_length_equal_to_number_of_qubits_in_circuit(
        runner: CircuitRunner,
    ):
        return all(
            len(bitstring) == circuit.n_qubits
            for circuit in _EXAMPLE_CIRCUITS
            for bitstring in runner.run_and_measure(circuit, n_samples=10).bitstrings
        )

    @staticmethod
    def raises_value_error_if_n_samples_is_nonpositive(runner: CircuitRunner):
        for n_samples in _INVALID_N_SAMPLES:
            try:
                runner.run_and_measure(_EXAMPLE_CIRCUITS[0], n_samples=n_samples)
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
            runner.run_batch_and_measure(_EXAMPLE_CIRCUITS, _EXAMPLE_N_SAMPLES)
        ) == len(_EXAMPLE_CIRCUITS)

    @staticmethod
    def returns_number_of_measurements_greater_or_equal_to_n_samples(
        runner: CircuitRunner,
    ):
        def _when_n_samples_is_the_same_for_each_circuit():
            return all(
                len(measurements.bitstrings) >= 10
                for measurements in runner.run_batch_and_measure(
                    _EXAMPLE_CIRCUITS, n_samples=10
                )
            )

        def _when_n_samples_is_different_for_each_circuit():
            return all(
                len(measurement.bitstrings) >= n_samples
                for measurement, n_samples in zip(
                    runner.run_batch_and_measure(_EXAMPLE_CIRCUITS, _EXAMPLE_N_SAMPLES),
                    _EXAMPLE_N_SAMPLES,
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
                runner.run_batch_and_measure(_EXAMPLE_CIRCUITS, _EXAMPLE_N_SAMPLES),
                _EXAMPLE_CIRCUITS,
            )
            for bitstring in measurements.bitstrings
        )

    @staticmethod
    def raises_value_error_if_n_samples_is_nonpositive(runner: CircuitRunner):
        def _for_all_circuits():
            try:
                runner.run_batch_and_measure(
                    _EXAMPLE_CIRCUITS, n_samples=_INVALID_N_SAMPLES
                )
                return False
            except ValueError:
                pass
            except Exception:
                return False

            return True

        def _for_at_least_one_circuit():
            n_samples = (-1,) + (len(_EXAMPLE_CIRCUITS) - 1) * (10,)
            try:
                runner.run_batch_and_measure(_EXAMPLE_CIRCUITS, n_samples=n_samples)
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
            (len(_EXAMPLE_CIRCUITS) + 1) * (10,),
            (len(_EXAMPLE_CIRCUITS) - 1) * (10,),
        )

        for n_samples in invalid_n_samples:
            try:
                runner.run_batch_and_measure(_EXAMPLE_CIRCUITS, n_samples)
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
            for circuit in _EXAMPLE_CIRCUITS
        )

    @staticmethod
    def raises_value_error_if_n_samples_is_nonpositive(runner: CircuitRunner):
        for n_samples in _INVALID_N_SAMPLES:
            try:
                runner.get_measurement_outcome_distribution(
                    _EXAMPLE_CIRCUITS[0], n_samples=n_samples
                )
                return False
            except ValueError:
                pass
            except Exception:
                return False

        return True


def strict_runner_returns_number_of_measurements_greater_or_equal_to_n_samples(
    runner: CircuitRunner,
):
    def _when_n_samples_is_the_same_for_each_circuit():
        return all(
            len(measurements.bitstrings) == 10
            for measurements in runner.run_batch_and_measure(
                _EXAMPLE_CIRCUITS, n_samples=10
            )
        )

    def _when_n_samples_is_different_for_each_circuit():
        return all(
            len(measurement.bitstrings) == n_samples
            for measurement, n_samples in zip(
                runner.run_batch_and_measure(_EXAMPLE_CIRCUITS, _EXAMPLE_N_SAMPLES),
                _EXAMPLE_N_SAMPLES,
            )
        )

    # We make sure to run each subtest, otherwise short-circuiting of
    # circuits might result in missing some errors
    subtests = [
        _when_n_samples_is_different_for_each_circuit(),
        _when_n_samples_is_the_same_for_each_circuit(),
    ]
    return all(subtests)


CIRCUIT_RUNNER_CONTRACTS = [
    _ValidateRunAndMeasure.returns_number_of_measurements_greater_or_equal_to_n_samples,  # noqa: E501
    _ValidateRunAndMeasure.returns_bitstrings_with_length_equal_to_number_of_qubits_in_circuit,  # noqa: E501
    _ValidateRunAndMeasure.raises_value_error_if_n_samples_is_nonpositive,  # noqa: E501
    _ValidateRunBatchAndMeasure.returns_number_of_measurements_greater_or_equal_to_n_samples,  # noqa: E501
    _ValidateRunBatchAndMeasure.returns_bitstrings_with_length_equal_to_number_of_qubits_in_circuit,  # noqa: E501
    _ValidateRunBatchAndMeasure.returns_measurement_object_for_each_circuit_in_batch,  # noqa: E501
    _ValidateRunBatchAndMeasure.raises_value_error_if_n_samples_is_nonpositive,  # noqa: E501
    _ValidateRunBatchAndMeasure.raises_value_error_if_len_of_n_samples_does_not_match_len_of_batch,  # noqa: E501
    _ValidateMeasurementOutcomeDistribution.returns_distribution_with_number_of_bits_same_as_circuit_n_qubits,  # noqa: E501
    _ValidateMeasurementOutcomeDistribution.raises_value_error_if_n_samples_is_nonpositive,  # noqa: E501
]


# This set of contracts is meant to be used for runners that return exactly the
# requested number of samples when running circuits. For context: some circuit
# runners return number of samples larger than the requested one, usually
# as a result of batching.
STRICT_CIRCUIT_RUNNER_CONTRACTS = [
    strict_runner_returns_number_of_measurements_greater_or_equal_to_n_samples
]
