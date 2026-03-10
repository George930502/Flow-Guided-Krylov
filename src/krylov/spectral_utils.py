"""
Spectral range utilities for SKQD time step computation.

Provides shared function to compute the optimal Krylov time step
from the spectral range of a molecular Hamiltonian's subspace.

Per SKQD paper (Theorem 3.1, Epperly et al.):
    dt_optimal = pi / (E_max - E_min)

Four strategies for different config-space sizes:
  - Small (≤5K): Dense diagonalization
  - Medium (5K-20K): Sparse eigsh on full matrix (built once via matrix_elements_fast)
  - Medium-large (20K-200K): Matrix-free Lanczos (accurate, no full matrix)
  - Very large (>200K): Diagonal approximation O(n_configs)
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
    """Sparse eigsh on full matrix for medium subspaces (5K-20K).

    Builds the full matrix once via matrix_elements_fast (optimized Hermitian
    construction), converts to sparse, then runs eigsh for extremal eigenvalues.
    """
    from scipy.sparse.linalg import eigsh as scipy_eigsh
    from scipy.sparse import csr_matrix

    # Use matrix_elements_fast (bra==ket optimized path) instead of
    # matrix_elements(bra, ket) which takes the slow general path
    H_sub = hamiltonian.matrix_elements_fast(basis_tensor)
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
    Matrix-free Lanczos for medium-large config spaces (20K-200K).

    Uses scipy.sparse.linalg.eigsh with a LinearOperator backed by the
    Hamiltonian's Slater-Condon rules. No full matrix construction needed.
    O(n_configs) per Lanczos iteration, ~20 iterations total.
    """
    from scipy.sparse.linalg import eigsh as scipy_eigsh, LinearOperator

    n = len(basis_tensor)
    device = basis_tensor.device

    # Build integer-encoded lookup for fast basis matching
    n_sites = basis_tensor.shape[1]
    # Encode each config as an integer for O(1) lookup
    powers = torch.pow(2, torch.arange(n_sites - 1, -1, -1, device=device, dtype=torch.int64))
    basis_ints = (basis_tensor * powers.unsqueeze(0)).sum(dim=1)  # shape (n,)
    int_to_idx = {}
    for i, val in enumerate(basis_ints.cpu().numpy()):
        int_to_idx[int(val)] = i

    # Pre-compute diagonal elements
    diag = hamiltonian.diagonal_elements_batch(basis_tensor).cpu().numpy().astype(np.float64)

    def matvec(v):
        """H @ v using diagonal + off-diagonal Slater-Condon connections."""
        result = diag * v  # diagonal part

        # Off-diagonal: iterate over basis states, get connections
        v_torch = torch.from_numpy(v).to(device=device, dtype=torch.float64)
        result_torch = torch.from_numpy(result).to(device=device, dtype=torch.float64)

        # Process in batches to manage memory
        batch_size = min(2000, n)
        for i_start in range(0, n, batch_size):
            i_end = min(i_start + batch_size, n)
            batch = basis_tensor[i_start:i_end]

            for local_idx in range(len(batch)):
                global_idx = i_start + local_idx
                cfg = batch[local_idx]

                connected, h_elements = hamiltonian.get_connections(cfg)
                if connected is None or len(connected) == 0:
                    continue

                for conn_cfg, h_elem in zip(connected, h_elements):
                    conn_int = int((conn_cfg * powers).sum().item())
                    j = int_to_idx.get(conn_int)
                    if j is not None and j != global_idx:
                        result_torch[global_idx] += h_elem.real * v_torch[j]

        return result_torch.cpu().numpy()

    H_op = LinearOperator((n, n), matvec=matvec, dtype=np.float64)

    try:
        E_min = float(scipy_eigsh(H_op, k=1, which='SA', return_eigenvectors=False)[0])
        E_max = float(scipy_eigsh(H_op, k=1, which='LA', return_eigenvectors=False)[0])
        return E_max - E_min
    except Exception as e:
        print(f"  Matrix-free Lanczos failed ({e}), falling back to diagonal approx")
        return _spectral_range_diagonal(hamiltonian, basis_tensor)


def _spectral_range_diagonal(hamiltonian, basis_tensor: torch.Tensor) -> float:
    """
    Diagonal approximation for large config spaces (>20K).

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
        # Use diagonal_elements_batch directly — O(batch) instead of O(batch²)
        diag_vals = hamiltonian.diagonal_elements_batch(batch).cpu().numpy()
        diag_min = min(diag_min, float(diag_vals.min()))
        diag_max = max(diag_max, float(diag_vals.max()))

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
      ≤5K:      Dense eigvalsh (exact, fast)
      5K-20K:   Sparse eigsh on full matrix (built once via matrix_elements_fast)
      20K-200K: Matrix-free Lanczos (accurate, no full matrix construction)
      >200K:    Diagonal approximation with 1.2x safety factor

    Args:
        hamiltonian: MolecularHamiltonian with n_orbitals, n_alpha, n_beta attributes

    Returns:
        (optimal_dt, spectral_range) where optimal_dt = pi / spectral_range
    """
    n_orb = hamiltonian.n_orbitals
    n_alpha = hamiltonian.n_alpha
    n_beta = hamiltonian.n_beta
    n = comb(n_orb, n_alpha) * comb(n_orb, n_beta)

    if n > 200_000:
        # Very large: diagonal approximation — O(n) time
        print(f"  Spectral range: diagonal approximation ({n:,} configs)")
        basis_tensor = _enumerate_basis(hamiltonian)
        spectral_range = _spectral_range_diagonal(hamiltonian, basis_tensor)
    elif n > 20_000:
        # Medium-large (20K-200K): matrix-free Lanczos — accurate, no full matrix
        print(f"  Spectral range: matrix-free Lanczos ({n:,} configs)")
        basis_tensor = _enumerate_basis(hamiltonian)
        spectral_range = _spectral_range_matfree(hamiltonian, basis_tensor)
    elif n > 5000:
        # Medium: build full matrix once via optimized matrix_elements_fast,
        # then sparse eigsh. Much faster than matrix-free LinearOperator
        # (which requires O(n_iterations) expensive Slater-Condon matvecs).
        print(f"  Spectral range: sparse eigsh ({n:,} configs)")
        basis_tensor = _enumerate_basis(hamiltonian)
        spectral_range = _spectral_range_sparse(hamiltonian, basis_tensor)
    else:
        basis_tensor = _enumerate_basis(hamiltonian)
        spectral_range = _spectral_range_dense(hamiltonian, basis_tensor)

    optimal_dt = np.pi / spectral_range
    return optimal_dt, spectral_range
