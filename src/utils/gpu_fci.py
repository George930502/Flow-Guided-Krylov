"""
GPU-accelerated FCI via gpu4pyscf.

Wraps gpu4pyscf's CUDA-kernel-accelerated FCI solver (Davidson iteration
with GPU contract_2e matvec) behind a clean interface. The key speedup
comes from custom CUDA kernels (_build_t1, _gather) that tile the CI
vector product in 32×32 blocks on GPU.

Two entry points:
  - compute_gpu_fci(): from geometry + basis (builds mol/mf internally)
  - compute_gpu_fci_from_integrals(): from pre-computed MO integrals

Falls back gracefully when gpu4pyscf or CuPy is not available.

Reference:
    gpu4pyscf: https://github.com/pyscf/gpu4pyscf
"""

import numpy as np
from typing import Tuple, Optional

# Detect gpu4pyscf availability
GPU4PYSCF_AVAILABLE = False
_GPU4PYSCF_IMPORT_ERROR = None

try:
    import cupy as cp
    from pyscf import fci as pyscf_fci
    from pyscf.fci import direct_spin1 as cpu_direct_spin1
    # Try importing gpu4pyscf's FCI module
    from gpu4pyscf.fci.direct_spin1 import FCI as GPU_FCI, contract_2e as gpu_contract_2e
    GPU4PYSCF_AVAILABLE = True
except ImportError as e:
    _GPU4PYSCF_IMPORT_ERROR = str(e)
except Exception as e:
    _GPU4PYSCF_IMPORT_ERROR = str(e)


def compute_gpu_fci(
    geometry: list,
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    max_memory: int = 8000,
    conv_tol: float = 1e-10,
    max_cycle: int = 300,
) -> float:
    """
    Compute FCI energy on GPU using gpu4pyscf's Davidson solver.

    Builds mol/mf from scratch, transforms integrals to MO basis,
    then runs GPU-accelerated Davidson iteration where the H×CI
    matvec uses CUDA kernels.

    Args:
        geometry: List of (atom, (x, y, z)) tuples
        basis: Gaussian basis set name
        charge: Molecular charge
        spin: Spin multiplicity (2S)
        max_memory: Max memory in MB for FCI solver
        conv_tol: Energy convergence tolerance
        max_cycle: Maximum Davidson iterations

    Returns:
        FCI ground state energy in Hartree

    Raises:
        RuntimeError: If gpu4pyscf is not available
    """
    if not GPU4PYSCF_AVAILABLE:
        raise RuntimeError(
            f"gpu4pyscf not available: {_GPU4PYSCF_IMPORT_ERROR}. "
            "Install with: pip install gpu4pyscf"
        )

    from pyscf import gto, scf, ao2mo

    mol = gto.Mole()
    mol.atom = geometry
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.verbose = 0
    mol.build()

    if spin == 0:
        mf = scf.RHF(mol)
    else:
        mf = scf.ROHF(mol)
    mf.kernel()

    # Transform integrals to MO basis
    h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    # ao2mo.kernel returns compressed 4-fold symmetric form by default
    eri = ao2mo.kernel(mol, mf.mo_coeff)
    norb = mf.mo_coeff.shape[1]
    nelec = mol.nelec

    # Create GPU FCI solver
    cisolver = GPU_FCI(mol)
    cisolver.max_memory = max_memory
    cisolver.conv_tol = conv_tol
    cisolver.max_cycle = max_cycle

    e_corr, fcivec = cisolver.kernel(h1e, eri, norb, nelec)
    total_energy = e_corr + mol.energy_nuc()

    return float(total_energy)


def compute_gpu_fci_from_integrals(
    h1e: np.ndarray,
    h2e: np.ndarray,
    n_orbitals: int,
    n_alpha: int,
    n_beta: int,
    nuclear_repulsion: float,
    max_memory: int = 8000,
    conv_tol: float = 1e-10,
    max_cycle: int = 300,
) -> float:
    """
    Compute FCI energy on GPU from pre-computed MO integrals.

    Takes the same integrals stored in MolecularIntegrals and runs
    gpu4pyscf's GPU Davidson solver.

    Args:
        h1e: One-electron integrals in MO basis (n_orb, n_orb)
        h2e: Two-electron integrals (n_orb, n_orb, n_orb, n_orb) or compressed
        n_orbitals: Number of spatial orbitals
        n_alpha: Number of alpha electrons
        n_beta: Number of beta electrons
        nuclear_repulsion: Nuclear repulsion energy
        max_memory: Max memory in MB
        conv_tol: Convergence tolerance
        max_cycle: Maximum iterations

    Returns:
        FCI ground state energy (including nuclear repulsion) in Hartree

    Raises:
        RuntimeError: If gpu4pyscf is not available
    """
    if not GPU4PYSCF_AVAILABLE:
        raise RuntimeError(
            f"gpu4pyscf not available: {_GPU4PYSCF_IMPORT_ERROR}. "
            "Install with: pip install gpu4pyscf"
        )

    from pyscf import ao2mo

    h1e_np = np.asarray(h1e, dtype=np.float64)
    h2e_np = np.asarray(h2e, dtype=np.float64)

    norb = n_orbitals
    nelec = (n_alpha, n_beta)

    # Convert h2e to compressed 4-fold symmetric form if needed
    # gpu4pyscf's contract_2e expects (nnorb, nnorb) where nnorb = norb*(norb+1)//2
    nnorb = norb * (norb + 1) // 2
    if h2e_np.ndim == 4:
        # Full 4-index tensor -> compressed
        eri = ao2mo.restore(4, h2e_np.reshape(norb**2, norb**2), norb)
    elif h2e_np.ndim == 2 and h2e_np.shape == (nnorb, nnorb):
        # Already compressed
        eri = h2e_np
    elif h2e_np.ndim == 2 and h2e_np.shape[0] == norb**2:
        # norb^2 x norb^2 -> compressed
        eri = ao2mo.restore(4, h2e_np, norb)
    else:
        # Try ao2mo.restore as-is
        eri = ao2mo.restore(4, h2e_np, norb)

    # Create GPU FCI solver without mol object
    # We need to construct a minimal solver manually
    cisolver = GPU_FCI(mol=None)
    cisolver.max_memory = max_memory
    cisolver.conv_tol = conv_tol
    cisolver.max_cycle = max_cycle

    e_corr, fcivec = cisolver.kernel(h1e_np, eri, norb, nelec)
    total_energy = float(e_corr) + nuclear_repulsion

    return total_energy
