"""
Eigenvalue solvers for projected Hamiltonians.

Provides interfaces to sparse eigensolvers for finding ground state
energies from projected Hamiltonian matrices.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any

try:
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix as csr_matrix_gpu
    from cupyx.scipy.sparse.linalg import eigsh as eigsh_gpu
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

from scipy.sparse import csr_matrix as csr_matrix_cpu
from scipy.sparse.linalg import eigsh as eigsh_cpu


def solve_generalized_eigenvalue(
    H: "csr_matrix",
    S: Optional["csr_matrix"] = None,
    k: int = 2,
    which: str = "SA",
    use_gpu: bool = True,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve generalized eigenvalue problem Hv = ESv.

    For SKQD, S is typically the identity (standard eigenvalue problem)
    since the sampled basis states are orthonormal in the computational basis.

    For KQD with Krylov overlaps, S would be the overlap matrix.

    Args:
        H: Hamiltonian matrix (sparse)
        S: Overlap matrix (optional, default: identity)
        k: Number of eigenvalues to compute
        which: Which eigenvalues ("SA" = smallest algebraic)
        use_gpu: Whether to use GPU acceleration
        **kwargs: Additional arguments for eigsh

    Returns:
        (eigenvalues, eigenvectors) sorted by eigenvalue
    """
    use_gpu = use_gpu and CUPY_AVAILABLE

    if use_gpu:
        eigsh = eigsh_gpu
        # Ensure matrices are on GPU
        if not isinstance(H, csr_matrix_gpu):
            H = csr_matrix_gpu(H)
        if S is not None and not isinstance(S, csr_matrix_gpu):
            S = csr_matrix_gpu(S)
    else:
        eigsh = eigsh_cpu
        # Ensure matrices are on CPU
        if hasattr(H, 'get'):
            H = csr_matrix_cpu(H.get())
        if S is not None and hasattr(S, 'get'):
            S = csr_matrix_cpu(S.get())

    # Solve eigenvalue problem
    if S is None:
        eigenvalues, eigenvectors = eigsh(
            H, k=k, which=which, return_eigenvectors=True, **kwargs
        )
    else:
        eigenvalues, eigenvectors = eigsh(
            H, M=S, k=k, which=which, return_eigenvectors=True, **kwargs
        )

    # Convert to numpy if on GPU
    if use_gpu:
        eigenvalues = eigenvalues.get()
        eigenvectors = eigenvectors.get()

    # Sort by eigenvalue
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors


def compute_ground_state_energy(
    H: "csr_matrix",
    use_gpu: bool = True,
    **kwargs,
) -> float:
    """
    Compute ground state energy of a Hamiltonian.

    Args:
        H: Hamiltonian matrix (sparse)
        use_gpu: Whether to use GPU
        **kwargs: Additional arguments for eigsh

    Returns:
        Ground state energy
    """
    eigenvalues, _ = solve_generalized_eigenvalue(
        H, k=1, which="SA", use_gpu=use_gpu, **kwargs
    )
    return float(eigenvalues[0])


def analyze_spectrum(
    H: "csr_matrix",
    k: int = 10,
    use_gpu: bool = True,
) -> Dict[str, Any]:
    """
    Analyze low-energy spectrum of Hamiltonian.

    Args:
        H: Hamiltonian matrix
        k: Number of low-lying states to compute
        use_gpu: Whether to use GPU

    Returns:
        Dictionary with spectral analysis:
            - eigenvalues: Array of k lowest eigenvalues
            - gaps: Energy gaps E_i - E_0
            - ground_state_energy: E_0
    """
    eigenvalues, eigenvectors = solve_generalized_eigenvalue(
        H, k=k, which="SA", use_gpu=use_gpu
    )

    E0 = eigenvalues[0]
    gaps = eigenvalues - E0

    return {
        "eigenvalues": eigenvalues,
        "gaps": gaps,
        "ground_state_energy": E0,
        "first_gap": gaps[1] if len(gaps) > 1 else None,
        "eigenvectors": eigenvectors,
    }


def regularize_overlap_matrix(
    S: "csr_matrix",
    threshold: float = 1e-10,
    use_gpu: bool = True,
) -> "csr_matrix":
    """
    Regularize an overlap matrix to ensure positive definiteness.

    For ill-conditioned Krylov overlap matrices, small eigenvalues
    can cause numerical instability. This function removes components
    with eigenvalues below a threshold.

    Args:
        S: Overlap matrix
        threshold: Minimum eigenvalue to retain
        use_gpu: Whether to use GPU

    Returns:
        Regularized overlap matrix
    """
    use_gpu = use_gpu and CUPY_AVAILABLE

    if use_gpu:
        S_dense = cp.asarray(S.toarray())
        eigenvalues, eigenvectors = cp.linalg.eigh(S_dense)

        # Zero out small eigenvalues
        eigenvalues = cp.maximum(eigenvalues, threshold)

        # Reconstruct
        S_reg = eigenvectors @ cp.diag(eigenvalues) @ eigenvectors.T
        return csr_matrix_gpu(S_reg)
    else:
        S_dense = np.asarray(S.toarray())
        eigenvalues, eigenvectors = np.linalg.eigh(S_dense)

        eigenvalues = np.maximum(eigenvalues, threshold)

        S_reg = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        return csr_matrix_cpu(S_reg)
