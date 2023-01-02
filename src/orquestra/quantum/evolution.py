################################################################################
# © Copyright 2020-2022 Zapata Computing Inc.
################################################################################
"""Functions for constructing circuits simulating evolution under given Hamiltonian."""
import operator
import warnings
from functools import reduce
from itertools import chain
from typing import List, Optional, Tuple, Union

import numpy as np
import sympy

from .circuits import CNOT, RX, RZ, Circuit, GateOperation, H
from .operators import PauliRepresentation, PauliTerm


def time_evolution(
    hamiltonian: PauliRepresentation,
    time: Union[float, sympy.Expr],
    method: str = "Trotter",
    trotter_order: int = 1,
) -> Circuit:
    """Create a circuit simulating evolution under given Hamiltonian.

    Args:
        hamiltonian: The Hamiltonian to be evolved under.
        time: Time duration of the evolution.
        method: Time evolution method. Currently the only option is 'Trotter'.
        trotter_order: order of Trotter evolution (1 by default).

    Returns:
        Circuit approximating evolution under `hamiltonian`.
        Circuit's unitary i approximately equal to exp(-i * time * hamiltonian).
    """
    if method != "Trotter":
        raise ValueError(f"Currently the method {method} is not supported.")

    return reduce(
        operator.add,
        (
            time_evolution_for_term(term, time / trotter_order)
            for _index_order in range(trotter_order)
            for term in hamiltonian.terms
        ),
    )


def time_evolution_for_term(term: PauliTerm, time: Union[float, sympy.Expr]) -> Circuit:
    """Evolves a Pauli term for a given time and returns a circuit representing it.
    Based on section 4 from https://arxiv.org/abs/1001.3855 .
    Args:
        term: Pauli term to be evolved
        time: time of evolution
    Returns:
        Circuit representing evolved term.
    """

    base_changes = []
    base_reversals = []
    cnot_gates = []
    central_gate: Optional[GateOperation] = None
    qubit_indices = sorted(term.qubits)

    circuit = Circuit()

    # If constant term, return empty circuit.
    if term.is_constant:
        return circuit

    if term.coefficient.imag > 1e-9:
        raise ValueError("Coefficients of terms must be real for Trotterization.")

    for i, qubit_id in enumerate(qubit_indices):
        term_type = term[qubit_id]
        if term_type == "X":
            base_changes.append(H(qubit_id))
            base_reversals.append(H(qubit_id))
        elif term_type == "Y":
            base_changes.append(RX(np.pi / 2)(qubit_id))
            base_reversals.append(RX(-np.pi / 2)(qubit_id))
        if i == len(term.operations) - 1:
            central_gate = RZ(2 * time * term.coefficient.real)(qubit_id)
        else:
            cnot_gates.append(CNOT(qubit_id, qubit_indices[i + 1]))

    for gate in base_changes:
        circuit += gate

    for gate in cnot_gates:
        circuit += gate

    if central_gate is not None:
        circuit += central_gate

    for gate in reversed(cnot_gates):
        circuit += gate

    for gate in base_reversals:
        circuit += gate

    return circuit


def time_evolution_derivatives(
    hamiltonian: PauliRepresentation,
    time: float,
    method: str = "Trotter",
    trotter_order: int = 1,
) -> Tuple[List[Circuit], List[float]]:
    """Generates derivative circuits for the time evolution operator defined in
    function time_evolution

    Args:
        hamiltonian: The Hamiltonian to be evolved under. It should contain numeric
            coefficients, symbolic expressions aren't supported.
        time: time duration of the evolution.
        method: time evolution method. Currently the only option is 'Trotter'.
        trotter_order: order of Trotter evolution

    Returns:
        A Circuit simulating time evolution.
    """
    if method != "Trotter":
        raise ValueError(f"The method {method} is currently not supported.")

    single_trotter_derivatives = []
    factors = [1.0, -1.0]
    output_factors = []
    terms = hamiltonian.terms

    for i, term_1 in enumerate(terms):
        for factor in factors:
            output = Circuit()

            if term_1.coefficient.real != term_1.coefficient:
                warnings.warn(
                    "Only real coefficients are supported. The imaginary part of the "
                    "term {} will be ignored.".format(term_1)
                )
            r = term_1.coefficient.real / trotter_order
            output_factors.append(r * factor)
            shift = factor * (np.pi / (4.0 * r))

            for j, term_2 in enumerate(terms):
                output += time_evolution_for_term(
                    term_2,
                    (time + shift) / trotter_order if i == j else time / trotter_order,
                )

            single_trotter_derivatives.append(output)

    if trotter_order > 1:
        output_circuits = []
        final_factors = []

        repeated_circuit = time_evolution(
            hamiltonian, time, method="Trotter", trotter_order=1
        )

        for position in range(trotter_order):
            for factor, different_circuit in zip(
                output_factors, single_trotter_derivatives
            ):
                output_circuits.append(
                    _generate_circuit_sequence(
                        repeated_circuit, different_circuit, trotter_order, position
                    )
                )
                final_factors.append(factor)
        return output_circuits, final_factors
    else:
        return single_trotter_derivatives, output_factors


def _generate_circuit_sequence(
    repeated_circuit: Circuit,
    different_circuit: Circuit,
    length: int,
    position: int,
):
    """Join multiple copies of circuit, replacing one copy with a different circuit.

    Args:
        repeated_circuit: circuit which copies should be concatenated
        different_circuit: circuit that will replace one copy of `repeated_circuit
        length: total number of circuits to join
        position: which copy of repeated_circuit should be replaced by
        `different_circuit`.
    Returns:
        Concatenation of circuits C_1, ..., C_length, where C_i = `repeated_circuit`
        if i != position and C_i = `different_circuit` if i == position.
    """
    if position >= length:
        raise ValueError(f"Position {position} should be < {length}")

    return Circuit(
        list(
            chain.from_iterable(
                [
                    (
                        repeated_circuit if i != position else different_circuit
                    ).operations
                    for i in range(length)
                ]
            )
        )
    )
