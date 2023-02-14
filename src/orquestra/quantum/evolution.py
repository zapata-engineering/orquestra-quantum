################################################################################
# © Copyright 2020-2022 Zapata Computing Inc.
################################################################################
"""Functions for constructing circuits simulating evolution under given Hamiltonian."""
import warnings
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
    n_steps: int = 1,
) -> Circuit:
    """Create a circuit simulating evolution under given Hamiltonian. First, we
    split the hamiltonian into n_steps with e^{-i t H} = prod_{j=1}^n e^{-i t/n H}.
    We approximate each term e^{-i t/n H} ≈ prod_{k=1}^m e^{-i t/n H_k} + O((t/n)^2).
    So then e^{-i t H} ≈ prod_{j=1}^n prod_{k=1}^m e^{-i t/n H_k} + O((t/n)^2).
    Thus, we can approximate the given hamiltonian to precision O((time / n_steps)^2).

    Args:
        hamiltonian: The Hamiltonian to be evolved under.
        time: Time duration of the evolution.
        method: Time evolution method. Currently the only option is 'Trotter'.
        n_steps: number of time steps in the approximation (1 by default).

    Returns:
        Circuit approximating exp(-i * time * hamiltonian) to order
          O((time / n_steps)^2).
    """
    if method != "Trotter":
        raise ValueError(f"Currently the method {method} is not supported.")

    # concatenate the circuits for each term
    circuit = Circuit()
    for _ in range(n_steps):
        for term in hamiltonian.terms:
            circuit += time_evolution_for_term(term, time / n_steps)
    return circuit


def time_evolution_for_term(term: PauliTerm, time: Union[float, sympy.Expr]) -> Circuit:
    """Returns a circuit which evolves a Pauli term for a given time.
    Based on section 4 from https://arxiv.org/abs/1001.3855 .
    Args:
        term: Pauli term to be evolved
        time: time of evolution
    Returns:
        Circuit representing evolved term.
    """

    basis_change = Circuit()
    cnot_gates = Circuit()
    qubit_indices = sorted(term.qubits)

    circuit = Circuit()

    # If constant term, return empty circuit.
    if term.is_constant:
        return circuit

    if term.coefficient.imag > 1e-9:
        raise ValueError("Coefficients of terms must be real for Trotterization.")

    for i, qubit_id in enumerate(qubit_indices):
        if term[qubit_id] == "X":
            basis_change += H(qubit_id)
        elif term[qubit_id] == "Y":
            basis_change += RX(np.pi / 2)(qubit_id)
        if i == len(term.operations) - 1:
            central_gate = RZ(2 * time * term.coefficient.real)(qubit_id)
        else:
            cnot_gates += CNOT(qubit_id, qubit_indices[i + 1])

    # implement e^(-i * time * Z_1 * Z_2 * ... * Z_n)
    all_z_rotation = cnot_gates + central_gate + cnot_gates.inverse()
    # change the rotation to be in the diagonal basis of term
    circuit = basis_change + all_z_rotation + basis_change.inverse()

    return circuit


def time_evolution_derivatives(
    hamiltonian: PauliRepresentation,
    time: float,
    method: str = "Trotter",
    n_steps: int = 1,
) -> Tuple[List[Circuit], List[float]]:
    """Generates derivative circuits for the time evolution operator defined in
    function time_evolution

    Args:
        hamiltonian: The Hamiltonian to be evolved under. It should contain numeric
            coefficients, symbolic expressions aren't supported.
        time: time duration of the evolution.
        method: time evolution method. Currently the only option is 'Trotter'.
        n_steps: number of time steps in the approximation (1 by default).

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
            r = term_1.coefficient.real / n_steps
            output_factors.append(r * factor)
            shift = factor * (np.pi / (4.0 * r))

            for j, term_2 in enumerate(terms):
                output += time_evolution_for_term(
                    term_2,
                    (time + shift) / n_steps if i == j else time / n_steps,
                )

            single_trotter_derivatives.append(output)

    if n_steps > 1:
        output_circuits = []
        final_factors = []

        repeated_circuit = time_evolution(
            hamiltonian, time, method="Trotter", n_steps=1
        )

        for position in range(n_steps):
            for factor, different_circuit in zip(
                output_factors, single_trotter_derivatives
            ):
                output_circuits.append(
                    _generate_circuit_sequence(
                        repeated_circuit, different_circuit, n_steps, position
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
