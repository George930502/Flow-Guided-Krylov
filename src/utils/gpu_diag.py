"""GPU-accelerated diagonalization adapter — drop-in replacement for IBM's solve_fermion.

Replaces qiskit_addon_sqd.fermion.solve_fermion with:
- torch.linalg.eigh for dense matrices (< SPARSE_THRESHOLD)
- scipy.sparse.linalg.eigsh (Lanczos iterative) for larger matrices
- CuPy sparse eigsh if available (GPU Lanczos, ~60x speedup)

Computes orbital occupancies from the eigenvector in IBM-compatible format:
  tuple(occ_alpha[n_orb], occ_beta[n_orb])
"""

import logging
import warnings

import numpy as np
import torch

from scipy.sparse import csr_matrix as scipy_csr
from scipy.sparse.linalg import eigsh as scipy_eigsh

try:
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix as cupy_csr
    from cupyx.scipy.sparse.linalg import eigsh as cupy_eigsh

    try:
        cp.cuda.Device(0).compute_capability
        CUPY_AVAILABLE = True
    except Exception:
        CUPY_AVAILABLE = False
except ImportError:
    CUPY_AVAILABLE = False

logger = logging.getLogger(__name__)

# Maximum basis size for dense diag (10K x 10K x 8 bytes = 800 MB)
MAX_DENSE_CONFIGS = 10000

# Use iterative (Lanczos) eigensolver above this; dense eigh below
SPARSE_THRESHOLD = 3000


def compute_occupancies(configs, v0, n_orb=None):
    """Compute orbital occupancies from eigenvector.

    occ_p = sum_i |c_i|^2 * n_p(x_i)

    Args:
        configs: (n_configs, 2*n_orb) array — occupation numbers.
                 Format: [alpha_0..alpha_{n-1}, beta_0..beta_{n-1}].
        v0: (n_configs,) array — eigenvector coefficients.
        n_orb: int — number of spatial orbitals. If None, inferred as configs.shape[1] // 2.

    Returns:
        tuple(occ_alpha, occ_beta): each np.ndarray of shape (n_orb,).
        This matches IBM's solve_fermion occupancy format.
    """
    configs_np = np.asarray(configs, dtype=np.float64)
    v0_np = np.asarray(v0)
    # Handle complex eigenvectors: |c_i|^2, not c_i^2
    probs = np.abs(v0_np) ** 2
    occ_flat = (probs[:, None] * configs_np).sum(axis=0)

    if n_orb is None:
        n_orb = configs_np.shape[1] // 2

    occ_alpha = occ_flat[:n_orb]
    occ_beta = occ_flat[n_orb:]
    return (occ_alpha, occ_beta)


def compute_occupancies_flat(configs, v0):
    """Compute orbital occupancies as a flat (n_qubits,) array.

    Convenience function for cases where IBM tuple format is not needed.

    Args:
        configs: (n_configs, 2*n_orb) array.
        v0: (n_configs,) array — eigenvector coefficients.

    Returns:
        np.ndarray of shape (2*n_orb,) — flat orbital occupancies.
    """
    configs_np = np.asarray(configs, dtype=np.float64)
    v0_np = np.asarray(v0, dtype=np.float64)
    probs = v0_np ** 2
    return (probs[:, None] * configs_np).sum(axis=0)


def gpu_solve_fermion(configs, hamiltonian, max_dense=MAX_DENSE_CONFIGS):
    """GPU-accelerated diagonalization — drop-in for IBM's solve_fermion.

    Builds the projected Hamiltonian in the config subspace and diagonalizes
    using the best available solver (GPU dense, GPU iterative, or CPU iterative).

    Args:
        configs: torch.Tensor (n_configs, 2*n_orb) — configurations in our format.
        hamiltonian: MolecularHamiltonian with matrix_elements() method.
        max_dense: int — above this, use iterative eigsh for diag (OOM guard).
                   Note: H construction is always dense via matrix_elements().
                   Sparse H construction requires get_sparse_matrix_elements()
                   which is planned for PR #2 (SKQD integration).

    Returns:
        (energy, eigenvector, occupancies):
            energy: float — ground state energy (total, including nuclear repulsion).
            eigenvector: np.ndarray (n_configs,) — ground state coefficients.
            occupancies: tuple(occ_alpha, occ_beta) — IBM-compatible occupancy format.
    """
    if isinstance(configs, np.ndarray):
        configs = torch.from_numpy(configs).long()
    else:
        configs = configs.detach().cpu().long()

    n = len(configs)
    n_orb = configs.shape[1] // 2

    if n == 0:
        raise ValueError("Empty basis — cannot diagonalize")

    # ── Single config: trivial ──
    if n == 1:
        e = float(hamiltonian.diagonal_element(configs[0]))
        v0 = np.array([1.0])
        occ = compute_occupancies(configs.numpy(), v0, n_orb)
        return e, v0, occ

    # ── Build projected Hamiltonian (dense) ──
    H_proj = hamiltonian.matrix_elements(configs, configs)
    H_np = H_proj.detach().cpu().numpy()

    # Take real part (molecular Hamiltonians are real)
    if np.iscomplexobj(H_np):
        H_np = H_np.real

    H_np = H_np.astype(np.float64)

    # Symmetrize (numerical errors can break Hermiticity)
    H_np = 0.5 * (H_np + H_np.T)

    # ── Diagonalize ──
    if n <= min(SPARSE_THRESHOLD, max_dense):
        E0, v0 = _dense_diag(H_np)
    else:
        # Use Lanczos iterative solver (avoids O(n^3) dense diag)
        E0, v0 = _iterative_diag(H_np)

    # ── Compute occupancies in IBM-compatible format ──
    occ = compute_occupancies(configs.numpy(), v0, n_orb)

    return E0, v0, occ


def _dense_diag(H_np):
    """Dense diagonalization via torch.linalg.eigh (GPU) or numpy.linalg.eigh (CPU)."""
    n = H_np.shape[0]

    # Try GPU torch.linalg.eigh
    if torch.cuda.is_available() and n <= 8000:
        try:
            H_gpu = torch.from_numpy(H_np).to("cuda")
            eigenvalues, eigenvectors = torch.linalg.eigh(H_gpu)
            E0 = float(eigenvalues[0].cpu())
            v0 = eigenvectors[:, 0].cpu().numpy()
            return E0, v0
        except Exception as e:
            logger.debug(f"GPU torch.linalg.eigh failed ({e}), falling back to CPU")

    # CPU numpy eigh
    eigenvalues, eigenvectors = np.linalg.eigh(H_np)
    return float(eigenvalues[0]), eigenvectors[:, 0]


def _iterative_diag(H_np):
    """Iterative (Lanczos) diagonalization for large matrices.

    Uses CuPy GPU eigsh if available, else SciPy CPU eigsh.
    The projected Hamiltonian is typically dense (not truly sparse), but
    Lanczos iterative solvers are faster than O(n^3) full eigh when only
    k=1 eigenvalue is needed. We use a LinearOperator to avoid CSR
    conversion overhead for dense matrices.
    """
    n = H_np.shape[0]

    # Try CuPy GPU: use dense eigh for moderate sizes, sparse eigsh for large
    if CUPY_AVAILABLE:
        try:
            H_gpu = cp.asarray(H_np)
            if n <= 8000:
                # Dense CuPy eigh (faster than CSR conversion for moderate matrices)
                eigenvalues, eigenvectors = cp.linalg.eigh(H_gpu)
                E0 = float(cp.asnumpy(eigenvalues[0]))
                v0 = cp.asnumpy(eigenvectors[:, 0])
            else:
                # Large: use CuPy sparse eigsh (Lanczos)
                H_sparse = cupy_csr(H_gpu)
                eigenvalues, eigenvectors = cupy_eigsh(H_sparse, k=1, which="SA")
                E0 = float(cp.asnumpy(eigenvalues[0]))
                v0 = cp.asnumpy(eigenvectors[:, 0])
            del H_gpu
            return E0, v0
        except Exception as e:
            warnings.warn(f"CuPy eigsh failed ({e}), falling back to SciPy")

    # SciPy CPU Lanczos eigsh via LinearOperator (avoids CSR conversion overhead)
    from scipy.sparse.linalg import LinearOperator
    matvec = lambda x: H_np @ x
    H_op = LinearOperator((n, n), matvec=matvec, dtype=H_np.dtype)
    eigenvalues, eigenvectors = scipy_eigsh(H_op, k=1, which="SA")
    return float(eigenvalues[0]), eigenvectors[:, 0]
