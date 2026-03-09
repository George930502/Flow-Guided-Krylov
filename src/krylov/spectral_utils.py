"""
Spectral range utilities for SKQD time step computation.

Provides shared function to compute the optimal Krylov time step
from the spectral range of a molecular Hamiltonian's subspace.

Per SKQD paper (Theorem 3.1, Epperly et al.):
    dt_optimal = pi / (E_max - E_min)

Three strategies for different config-space sizes:
  - Small (≤5K): Dense diagonalization
  - Medium (5K-10K): Sparse eigsh on full matrix
  - Large (10K-100K): LinearOperator + sparse eigsh (matrix-free)
  - Very large (>100K): Diagonal approximation O(n_configs)
"""

import numpy as np
import torch
from itertools import combinations
from typing import Tuple
from math import comb


def _enumerate_basis(hamiltonian) -> torch.Tensor:
    """Enumerate all particle-conserving configurations."""
    n_orb = hamiltonian.n_orbitals
    n_alpha = hamiltonian.n_alpha
    n_beta = hamiltonian.n_beta
    device = hamiltonian.device

    alpha_configs = list(combinations(range(n_orb), n_alpha))
    beta_configs = list(combinations(range(n_orb), n_beta))

    basis = []
    for ac in alpha_configs:
        for bc in beta_configs:
            cfg = torch.zeros(2 * n_orb, dtype=torch.long, device=device)
            for o in ac:
                cfg[o] = 1
            for o in bc:
                cfg[o + n_orb] = 1
            basis.append(cfg)
    return torch.stack(basis)


def _spectral_range_dense(hamiltonian, basis_tensor: torch.Tensor) -> float:
    """Dense diagonalization for small subspaces (≤5K configs)."""
    H_sub = hamiltonian.matrix_elements(basis_tensor, basis_tensor)
    H_np = H_sub.cpu().numpy().real.astype(np.float64)
    H_np = 0.5 * (H_np + H_np.T)
    evals = np.linalg.eigvalsh(H_np)
    return float(evals[-1] - evals[0])


def _spectral_range_sparse(hamiltonian, basis_tensor: torch.Tensor) -> float:
    """Sparse eigsh on full matrix for medium subspaces (5K-10K)."""
    from scipy.sparse.linalg import eigsh as scipy_eigsh
    from scipy.sparse import csr_matrix

    H_sub = hamiltonian.matrix_elements(basis_tensor, basis_tensor)
    H_gpu = H_sub.to(dtype=torch.float64)
    H_gpu = 0.5 * (H_gpu + H_gpu.T)

    H_np = H_gpu.cpu().numpy()
    H_sp = csr_matrix(H_np)
    del H_np, H_gpu

    E_min = float(scipy_eigsh(H_sp, k=1, which='SA', return_eigenvectors=False)[0])
    E_max = float(scipy_eigsh(H_sp, k=1, which='LA', return_eigenvectors=False)[0])
    return E_max - E_min


def _spectral_range_matfree(hamiltonian, basis_tensor: torch.Tensor) -> float:
    """
    Matrix-free spectral range via LinearOperator + sparse eigsh.

    Computes H×v on-the-fly in batches without storing the full matrix.
    Memory: O(batch_size × n) instead of O(n²). Enables 10K-100K configs.
    """
    from scipy.sparse.linalg import eigsh as scipy_eigsh, LinearOperator

    n = len(basis_tensor)
    batch_size = min(500, n)

    def matvec(v_np):
        v_torch = torch.tensor(v_np, dtype=torch.float64, device=hamiltonian.device)
        result = torch.zeros(n, dtype=torch.float64, device=hamiltonian.device)

        # Process in row batches to avoid building full n×n matrix
        for i_start in range(0, n, batch_size):
            i_end = min(i_start + batch_size, n)
            row_basis = basis_tensor[i_start:i_end]
            # H_block: (batch_size, n) — one batch of rows
            H_block = hamiltonian.matrix_elements(row_basis, basis_tensor)
            H_real = H_block.real.to(torch.float64)
            result[i_start:i_end] = H_real @ v_torch
            del H_block, H_real

        return result.cpu().numpy()

    linop = LinearOperator((n, n), matvec=matvec, dtype=np.float64)

    E_min = float(scipy_eigsh(linop, k=1, which='SA', return_eigenvectors=False)[0])
    E_max = float(scipy_eigsh(linop, k=1, which='LA', return_eigenvectors=False)[0])
    return E_max - E_min


def _spectral_range_diagonal(hamiltonian, basis_tensor: torch.Tensor) -> float:
    """
    Diagonal approximation for very large config spaces (>100K).

    Estimates spectral range from diagonal elements H_ii only.
    O(n_configs) time and memory. Less accurate but always feasible.
    """
    n = len(basis_tensor)
    batch_size = 5000

    diag_min = float('inf')
    diag_max = float('-inf')

    for i_start in range(0, n, batch_size):
        i_end = min(i_start + batch_size, n)
        batch = basis_tensor[i_start:i_end]
        # Diagonal elements: H_ii for each config
        diag_block = hamiltonian.matrix_elements(batch, batch)
        diag_vals = diag_block.diagonal().real.cpu().numpy()
        diag_min = min(diag_min, float(diag_vals.min()))
        diag_max = max(diag_max, float(diag_vals.max()))
        del diag_block

    spectral_range = diag_max - diag_min

    # Diagonal approximation tends to underestimate; apply safety factor
    return spectral_range * 1.2


def compute_optimal_dt(hamiltonian) -> Tuple[float, float]:
    """
    Compute optimal Krylov time step from spectral range of subspace Hamiltonian.

    Enumerates all particle-conserving configurations, builds the subspace
    Hamiltonian, and computes its spectral range to derive the optimal dt.

    Per SKQD paper (Theorem 3.1, Epperly et al.):
        dt_optimal = pi / (E_max - E_min)

    Strategy selection by config-space size:
      ≤5K:    Dense eigvalsh (exact, fast)
      5K-10K: Sparse eigsh on full matrix (exact extremal)
      10K-100K: Matrix-free LinearOperator + sparse eigsh
      >100K:  Diagonal approximation

    Args:
        hamiltonian: MolecularHamiltonian with n_orbitals, n_alpha, n_beta attributes

    Returns:
        (optimal_dt, spectral_range) where optimal_dt = pi / spectral_range
    """
    n_orb = hamiltonian.n_orbitals
    n_alpha = hamiltonian.n_alpha
    n_beta = hamiltonian.n_beta
    n = comb(n_orb, n_alpha) * comb(n_orb, n_beta)

    if n > 100_000:
        # Very large: diagonal approximation (no full basis enumeration needed
        # for diagonal, but we still need configs). For truly huge spaces,
        # sample a subset of configs for diagonal estimation.
        print(f"  Spectral range: diagonal approximation ({n:,} configs)")
        basis_tensor = _enumerate_basis(hamiltonian)
        spectral_range = _spectral_range_diagonal(hamiltonian, basis_tensor)
    elif n > 10_000:
        print(f"  Spectral range: matrix-free eigsh ({n:,} configs)")
        basis_tensor = _enumerate_basis(hamiltonian)
        spectral_range = _spectral_range_matfree(hamiltonian, basis_tensor)
    elif n > 5000:
        basis_tensor = _enumerate_basis(hamiltonian)
        spectral_range = _spectral_range_sparse(hamiltonian, basis_tensor)
    else:
        basis_tensor = _enumerate_basis(hamiltonian)
        spectral_range = _spectral_range_dense(hamiltonian, basis_tensor)

    optimal_dt = np.pi / spectral_range
    return optimal_dt, spectral_range
