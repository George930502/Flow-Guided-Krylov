"""
CUDA-Q quantum circuits for UCCSD-like ansatz.

Uses particle-number-conserving Givens rotations as the building block.
Each Givens rotation G(θ) acts on adjacent qubits (p, p+1) as:
  |00⟩ → |00⟩
  |01⟩ → cos(θ)|01⟩ - sin(θ)|10⟩
  |10⟩ → sin(θ)|01⟩ + cos(θ)|10⟩
  |11⟩ → |11⟩

This preserves particle number exactly (no post-selection needed).

Circuit decomposition for G(θ) on adjacent qubits p, p+1:
  CNOT(p+1, p)
  Ry(θ, p+1)
  CNOT(p, p+1)
  Ry(-θ, p+1)
  CNOT(p, p+1)
  CNOT(p+1, p)

The full UCCSD ansatz applies Givens rotations in a brick-wall pattern
(even pairs then odd pairs) within each spin sector, repeated n_layers times.
"""

import cudaq
import numpy as np


@cudaq.kernel
def uccsd_ansatz(n_qubits: int, n_alpha: int, n_beta: int,
                 n_layers: int, thetas: list[float]):
    """
    UCCSD-like ansatz using particle-number-conserving Givens rotations.

    Brick-wall pattern: even pairs (0,1),(2,3),... then odd pairs (1,2),(3,4),...
    Applied separately to alpha and beta spin sectors.
    Preserves particle number exactly.
    """
    q = cudaq.qvector(n_qubits)
    n_orb = n_qubits // 2

    # Prepare Hartree-Fock state
    for i in range(n_alpha):
        x(q[i])
    for i in range(n_beta):
        x(q[n_orb + i])

    param_idx = 0

    for layer in range(n_layers):
        # --- Alpha sector Givens rotations ---
        # Even pairs: (0,1), (2,3), ...
        for i in range(0, n_orb - 1, 2):
            theta = thetas[param_idx]
            # Givens rotation G(θ) on q[i], q[i+1]
            cx(q[i + 1], q[i])
            ry(theta, q[i + 1])
            cx(q[i], q[i + 1])
            ry(-theta, q[i + 1])
            cx(q[i], q[i + 1])
            cx(q[i + 1], q[i])
            param_idx += 1

        # Odd pairs: (1,2), (3,4), ...
        for i in range(1, n_orb - 1, 2):
            theta = thetas[param_idx]
            cx(q[i + 1], q[i])
            ry(theta, q[i + 1])
            cx(q[i], q[i + 1])
            ry(-theta, q[i + 1])
            cx(q[i], q[i + 1])
            cx(q[i + 1], q[i])
            param_idx += 1

        # --- Beta sector Givens rotations ---
        # Even pairs
        for i in range(0, n_orb - 1, 2):
            j = n_orb + i
            theta = thetas[param_idx]
            cx(q[j + 1], q[j])
            ry(theta, q[j + 1])
            cx(q[j], q[j + 1])
            ry(-theta, q[j + 1])
            cx(q[j], q[j + 1])
            cx(q[j + 1], q[j])
            param_idx += 1

        # Odd pairs
        for i in range(1, n_orb - 1, 2):
            j = n_orb + i
            theta = thetas[param_idx]
            cx(q[j + 1], q[j])
            ry(theta, q[j + 1])
            cx(q[j], q[j + 1])
            ry(-theta, q[j + 1])
            cx(q[j], q[j + 1])
            cx(q[j + 1], q[j])
            param_idx += 1


def count_uccsd_params(n_orbitals, n_layers):
    """Count parameters for UCCSD ansatz."""
    # Per layer: (even + odd) pairs for alpha + beta
    # Even pairs per sector: ceil((n_orb-1)/2)
    # Odd pairs per sector: floor((n_orb-1)/2)
    # Total per sector per layer: n_orb - 1
    # Total per layer: 2 * (n_orb - 1)
    return n_layers * 2 * (n_orbitals - 1)
