import pytest

from orquestra.quantum.circuits import H, CNOT, X
from orquestra.quantum.circuits._circuit import Circuit
from orquestra.quantum.circuits._itertools import split_into_batches, expand_sample_sizes


class TestBatchingWithInvalidInputs:

    @pytest.mark.parametrize(
        "n_samples_per_circuit",
        [
            [100],
            [50, 100, 50]
        ]
    )
    def test_circuits_and_samples_per_circuit_need_to_have_the_same_length(
        self,
        n_samples_per_circuit
    ):
        circuits = [
            Circuit([H(0), CNOT(0, 1)]),
            Circuit([X(2)])
        ]

        with pytest.raises(ValueError):
            split_into_batches(circuits, n_samples_per_circuit, 100)

    @pytest.mark.parametrize("max_circuits_per_batch", [-10, 0])
    def test_max_circuits_per_batch_has_to_be_positive(self, max_circuits_per_batch):
        circuits = [Circuit([X(0), CNOT(1, 2)]), Circuit([X(1), H(1)])]

        with pytest.raises(ValueError):
            split_into_batches(circuits, [100, 100], max_circuits_per_batch)


class TestBatching:

    @pytest.mark.parametrize("max_batch_size", [100, 1000])
    def test_one_item_is_returned_if_all_circuits_fit_into_single_batch(
        self, max_batch_size
    ):
        circuits = (Circuit([CNOT(0, 1)]), Circuit([X(2)]))

        batches = list(
            split_into_batches(circuits, [100, 50], max_batch_size)
        )

        assert len(batches) == 1
        assert batches[0] == (circuits, 100)

    def test_batch_size_equals_to_max_of_requested_sample_sizes(self):
        circuits = (Circuit([CNOT(0, 1)]),) * 10
        samples_per_circuit = [5, 10, 21, 37, 12, 1, 10, 10, 10, 200]

        batches = list(split_into_batches(circuits, samples_per_circuit, max_batch_size=3))

        assert batches == [
            (circuits[0:3], 21),
            (circuits[3:6], 37),
            (circuits[6:9], 10),
            (circuits[9:], 200)
        ]


class TestExpandingSampleSizes:

    def test_same_seqs_are_returned_if_all_sample_sizes_le_max_sample_size(
        self
    ):
        circuits = [
            Circuit([X(0), CNOT(0, 1)]),
            Circuit([X(0), H(0)]),
            Circuit([H(0), H(1)])
        ]
        n_samples_per_circuit = [20, 30, 40]

        new_circuits, new_n_samples, multiplicities = expand_sample_sizes(
            circuits, n_samples_per_circuit, 100
        )

        assert multiplicities == [1, 1, 1]
        assert new_n_samples == n_samples_per_circuit
        assert new_circuits == circuits

    def test_n_samples_and_multiplicities_are_correctly_computed(self):
        circuits = [Circuit([X(i)]) for i in range(6)]
        n_samples_per_circuit = [10, 20, 30, 20, 45, 40]
        max_sample_size = 20
        expected_new_circuits = [
            circuits[i] for i in (0, 1, 2, 2, 3, 4, 4, 4, 5, 5)
        ]
        expected_new_n_samples = [
            10, 20, 20, 10, 20, 20, 20, 5, 20, 20
        ]
        expected_multiplicities = [1, 1, 2, 1, 3, 2]

        new_circuits, new_n_samples, multiplicities = expand_sample_sizes(
            circuits, n_samples_per_circuit, max_sample_size
        )

        assert new_circuits == expected_new_circuits
        assert new_n_samples == expected_new_n_samples
        assert expected_multiplicities == multiplicities
