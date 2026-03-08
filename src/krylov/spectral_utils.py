"""
Spectral range utilities for SKQD time step computation.

Provides shared function to compute the optimal Krylov time step
from the spectral range of a molecular Hamiltonian's subspace.

Per SKQD paper (Theorem 3.1, Epperly et al.):
    dt_optimal = pi / (E_max - E_min)
"""

import numpy as np
import torch
from itertools import combinations
from typing import Tuple


def compute_optimal_dt(hamiltonian) -> Tuple[float, float]:
    """
    Compute optimal Krylov time step from spectral range of subspace Hamiltonian.

    Enumerates all particle-conserving configurations, builds the subspace
    Hamiltonian, and computes its spectral range to derive the optimal dt.

    Per SKQD paper (Theorem 3.1, Epperly et al.):
        dt_optimal = pi / (E_max - E_min)

    For large subspaces (>5000 configs), uses scipy sparse eigsh to find
    only extremal eigenvalues without dense diagonalization.

    Args:
        hamiltonian: MolecularHamiltonian with n_orbitals, n_alpha, n_beta attributes

    Returns:
        (optimal_dt, spectral_range) where optimal_dt = pi / spectral_range
    """
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
    basis_tensor = torch.stack(basis)

    n = len(basis)

    if n <= 5000:
        H_sub = hamiltonian.matrix_elements(basis_tensor, basis_tensor)
        H_np = H_sub.cpu().numpy().real.astype(np.float64)
        H_np = 0.5 * (H_np + H_np.T)
        evals = np.linalg.eigvalsh(H_np)
        spectral_range = float(evals[-1] - evals[0])
    else:
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
        spectral_range = E_max - E_min

    optimal_dt = np.pi / spectral_range
    return optimal_dt, spectral_range
