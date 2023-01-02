from collections import Counter
from functools import reduce
from itertools import islice
from math import ceil
from typing import Dict, Iterable, Sequence, Tuple, TypeVar

T = TypeVar("T")


def _iterate_in_batches(items: Iterable[T], batch_size: int) -> Iterable[Tuple[T, ...]]:
    it = iter(items)
    while chunk := tuple(islice(it, batch_size)):
        yield chunk


def split_into_batches(
    circuits: Sequence[T],
    n_samples_per_circuit: Sequence[int],
    max_batch_size: int,
) -> Iterable[Tuple[Sequence[T], int]]:
    """Split sequence of circuits and corresponding sample sizes into batches.

    The circuits are split into batches of length `max_batch_size`, except
    possibly the last batch that can be smaller. The number of samples
    for each batch is a maximum number of samples requested for the circuits
    in the chunk. To illustrate, consider the following input:

    - circuits=[c1, c2, c3, c4, c5]
    - n_samples_per_circuit=[n1, n2, n3, n4, n5]
    - max_batch_size=2

    Then the returned iterable contains the following tuples:

    ((c1, c2), max([n1, n2]))
    ((c3, c4), max([n3, n4]))
    ((c5,), n5)

    Args:
        circuits: a sequence of circuits to be batched. Please note that
          the exact type of the circuits does not matter, and hence
          this function works for native Orquestra circuits as well as
          circuits from other libraries/SDKs.
        n_samples_per_circuit: a sequence of numbers of length equal to
          len(circuits), s.t. n_samples_per_circuit[i] is number of
          samples requested for circuits[i].
        max_batch_size: maximum allowable size of batch. Typically it comes
          from specific runner's restrictions.
    Returns:
        An iterable yielding tuples of the forms (circuits_chunk, n_samples)
        satisfying the following constraints:

        - len(circuits_chunk) <= max_batch_size
        - n_samples is large enough to accommodate sample sizes requested
          for each circuit in circuits_chunk

    Raises:
        ValueError: if input sequences are not of equal length, or if
          max_batch_size is not positive.
    """
    if len(circuits) != len(n_samples_per_circuit):
        raise ValueError(
            "Mismatched lengths of `circuits` and `n_samples_per_circuit: "
            f"({len(circuits)} and {len(n_samples_per_circuit)} respectively)."
            "Both sequences need to have the same length"
        )

    if max_batch_size <= 0:
        raise ValueError(
            "Max circuits per batch has to be positive, got " f"{max_batch_size}"
        )

    return (
        (circuits_chunk, max(samples_chunk))
        for circuits_chunk, samples_chunk in zip(
            _iterate_in_batches(circuits, max_batch_size),
            _iterate_in_batches(n_samples_per_circuit, max_batch_size),
        )
    )


def _expand_sample_size(n_samples, max_sample_size):
    multiplicities = ceil(n_samples / max_sample_size)
    new_n_samples = (
        multiplicities * (max_sample_size,)
        if n_samples % max_sample_size == 0
        else (multiplicities - 1) * (max_sample_size,) + (n_samples % max_sample_size,)
    )
    return new_n_samples, multiplicities


def expand_sample_sizes(
    circuits: Sequence[T], n_samples_per_circuit: Sequence[int], max_sample_size: int
) -> Tuple[Sequence[T], Sequence[int], Sequence[int]]:
    """Expand sample sizes for each circuit to fit maximum sample size.

    Args:
        circuits: list of circuits to be expanded
        n_samples_per_circuit: list of sample sizes corresponding to each
          circuit
        max_sample_size: maximum allowable sample size
    Returns:
      Tuple of three sequences (new_circuits, new_sample_sizes, multiplicities):

      - new_circuits: sequence of circuits containing all of the original, possibly
        duplicated, circuits
      - new_n_samples: list of integers of the same length as new_circuits,
        s.t. sum of all sample sizes corresponding to given circuit (counting
        duplicates) is equal to the original sample size
      - multiplicities: sequence of the same length as original sequence of
        circuits holding information on how many times given circuit has
        been duplicated in `new_circuits`
    """
    n_samples_and_multiplicities = [
        _expand_sample_size(n_samples, max_sample_size)
        for n_samples in n_samples_per_circuit
    ]

    new_n_samples = [
        n for n_samples, _ in n_samples_and_multiplicities for n in n_samples
    ]

    multiplicities = [multi for _, multi in n_samples_and_multiplicities]
    new_circuits = [
        circuit
        for circuit, multi in zip(circuits, multiplicities)
        for _ in range(multi)
    ]

    return new_circuits, new_n_samples, multiplicities


def _combine_measurements(
    first: Dict[str, int], second: Dict[str, int]
) -> Dict[str, int]:
    result = Counter(first)
    for bitstring, count in second.items():
        result[bitstring] += count
    return dict(result)


def combine_measurement_counts(
    all_measurements: Sequence[Dict[str, int]], multiplicities: Sequence[int]
) -> Sequence[Dict[str, int]]:
    """Combine (aggregate) measurements of the same circuits run several times.

    Suppose multiplicities is a list [1, 2 ,3]. Then, the all_measurements should
    be a sequence of 1+2+3=6 elements m0,m1,m2,m3,m4,m5, and the results will
    contain three dictionaries M0, M1, M2 s.t.

    - M0 contains counts from m0 only
    - M1 comprises combined counts from m1 and m2
    - M2 comprises combined counts from m3, m4 and m5

    Args:
        all_measurements: sequence of measurements containing measurements
          gathered from some, possibly duplicated, circuits. The Measurement
          objects corresponding to the same circuit should be placed next to
          each other. Should have the same length as sum(multiplicities).
        multiplicities: sequence of positive integers marking groups of
          consecutive measurements corresponding to the same circuit. For
          instance, multiplicities [1, 2, 3] mean that first group of
          measurements comprises 1 Measurement, second group comprises 2
          consecutive Measurements, third group contains 3 consecutive
          Measurements and so on.
    Returns:
        Sequence of combined measurements of length equal len(multiplicities)
    Raises:
        ValueError: if len(all_measurements != sum(multiplicities)
    """
    if len(all_measurements) != (sum_multiplicities := sum(multiplicities)):
        raise ValueError(
            "Mismatch between multiplicities and number of measurements to combine. "
            f"Got {len(all_measurements)} Measurements objects to combine "
            f"but multiplicities sum to {sum_multiplicities}"
        )
    measurements_it = iter(all_measurements)
    return [
        reduce(_combine_measurements, islice(measurements_it, multiplicity))
        for multiplicity in multiplicities
    ]
