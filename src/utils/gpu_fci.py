"""
GPU-accelerated FCI with embedded CUDA kernels.

Replaces PySCF's CPU contract_2e (the H×CI matvec bottleneck inside
Davidson iteration) with custom CUDA kernels running via CuPy.
No external gpu4pyscf package required — only CuPy + PySCF.

The CUDA kernels (_build_t1, _gather) process determinant strings in
32×32 tiles on GPU. The Davidson eigensolver loop stays in PySCF;
only the matvec (which dominates compute) runs on GPU.

Two entry points:
  - compute_gpu_fci(): from geometry + basis (builds mol/mf internally)
  - compute_gpu_fci_from_integrals(): from pre-computed MO integrals

Falls back gracefully when CuPy is not available.

Based on gpu4pyscf (Apache 2.0 License):
    https://github.com/pyscf/gpu4pyscf
"""

import numpy as np
from typing import Optional

# ---------------------------------------------------------------------------
# Availability detection
# ---------------------------------------------------------------------------

GPU_FCI_AVAILABLE = False
_GPU_FCI_IMPORT_ERROR = None

try:
    import cupy as cp
    from pyscf.fci import direct_spin1 as _cpu_direct_spin1

    # ---------------------------------------------------------------------------
    # Embedded CUDA kernels for contract_2e (from gpu4pyscf/fci/direct_spin1.py)
    # ---------------------------------------------------------------------------

    _TILE = 32

    _CUDA_CODE = r'''
#define TILE 32
extern "C" {
__global__
void _build_t1(double *ci0, double *t1,
    long long strb0, long long na, long long nb, long long nnorb,
    unsigned short *addra, unsigned short *addrb, char *signa, char *signb)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int stra0 = blockIdx.y * blockDim.y;
    int strb = strb0 + tx;
    int stra = stra0 + ty;

    int nab = na * TILE;
    int ab_id = stra * TILE + tx;
    __shared__ unsigned short _addra[TILE*TILE];
    __shared__ unsigned short _addrb[TILE*TILE];
    __shared__ char _signa[TILE*TILE];
    __shared__ char _signb[TILE*TILE];
    int sign, str1, j0, j;
    int dj = TILE;
    double val;

    for (j0 = 0; j0 < nnorb; j0+=TILE) {
        _addra[ty*TILE+tx] = addra[(j0+ty)*na+stra0+tx];
        _addrb[ty*TILE+tx] = addrb[(j0+ty)*nb+strb0+tx];
        _signa[ty*TILE+tx] = signa[(j0+ty)*na+stra0+tx];
        _signb[ty*TILE+tx] = signb[(j0+ty)*nb+strb0+tx];
        if (j0 + TILE > nnorb) {
            dj = nnorb - j0;
        }
        __syncthreads();
        if (stra < na && strb < nb) {
            for (j = 0; j < dj; j++) {
                val = 0;
                sign = _signa[j*TILE+ty];
                str1 = _addra[j*TILE+ty];
                if (sign != 0) {
                    val = sign * ci0[str1*nb+strb];
                }

                sign = _signb[j*TILE+tx];
                str1 = _addrb[j*TILE+tx];
                if (sign != 0) {
                    val += sign * ci0[stra*nb+str1];
                }
                t1[(j0+j)*nab + ab_id] = val;
            }
        }
        __syncthreads();
    }
}

__global__
void _gather(double *out, double *t1,
    long long strb0, long long na, long long nb, long long nnorb,
    unsigned short *addra, unsigned short *addrb, char *signa, char *signb)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int stra0 = blockIdx.y * blockDim.y;
    int strb = strb0 + tx;
    int stra = stra0 + ty;
    int nab = na * TILE;
    int ab_id = stra * TILE + tx;
    __shared__ unsigned short _addra[TILE*TILE];
    __shared__ unsigned short _addrb[TILE*TILE];
    __shared__ char _signa[TILE*TILE];
    __shared__ char _signb[TILE*TILE];
    int sign, str1, j0, j;
    int dj = TILE;
    double val = 0.;

    for (j0 = 0; j0 < nnorb; j0+=TILE) {
        _addra[ty*TILE+tx] = addra[(j0+ty)*na+stra0+tx];
        _addrb[ty*TILE+tx] = addrb[(j0+ty)*nb+strb0+tx];
        _signa[ty*TILE+tx] = signa[(j0+ty)*na+stra0+tx];
        _signb[ty*TILE+tx] = signb[(j0+ty)*nb+strb0+tx];
        if (j0 + TILE > nnorb) {
            dj = nnorb - j0;
        }
        __syncthreads();
        if (stra < na && strb < nb) {
            for (j = 0; j < dj; j++) {
                sign = _signa[j*TILE+ty];
                str1 = _addra[j*TILE+ty];
                if (sign != 0) {
                    val += sign * t1[(j0+j)*nab + (str1*TILE+tx)];
                }

                sign = _signb[j*TILE+tx];
                str1 = _addrb[j*TILE+tx];
                if (sign != 0) {
                    out[stra*nb+str1] += sign * t1[(j0+j)*nab + ab_id];
                }
            }
        }
        __syncthreads();
    }
    out[stra*nb+strb] += val;
}
}'''

    _cuda_module = cp.RawModule(code=_CUDA_CODE)
    _kernel_build_t1 = _cuda_module.get_function('_build_t1')
    _kernel_gather = _cuda_module.get_function('_gather')

    def _link_index_to_addrs(link_index, nnorb):
        """Convert PySCF link_index to GPU address/sign arrays."""
        na = link_index.shape[0]
        ia = link_index[:, :, 0].T
        addr = np.zeros((nnorb, na), dtype=np.uint16)
        sign = np.zeros((nnorb, na), dtype=np.int8)
        idx = np.arange(na)
        addr[ia, idx] = link_index[:, :, 2].T
        sign[ia, idx] = link_index[:, :, 3].T
        # Pad to avoid out-of-bounds in kernel
        _addr = cp.empty((nnorb + _TILE - 1, na), dtype=np.uint16)[:nnorb]
        _sign = cp.empty((nnorb + _TILE - 1, na), dtype=np.int8)[:nnorb]
        _addr.set(addr)
        _sign.set(sign)
        return _addr, _sign

    def _gpu_contract_2e(eri, ci0, norb, nelec, link_index):
        """GPU-accelerated H×CI vector product using CUDA kernels."""
        ci0 = cp.asarray(ci0)
        original_shape = ci0.shape
        link_indexa, link_indexb = link_index
        na, nb = link_indexa.shape[0], link_indexb.shape[0]
        ci0 = ci0.reshape(na, nb)
        out = cp.zeros((na, nb), dtype=ci0.dtype)
        nnorb = norb * (norb + 1) // 2
        assert eri.shape == (nnorb, nnorb), (
            f"ERI shape {eri.shape} != expected ({nnorb}, {nnorb})"
        )
        eri = cp.asarray(eri)

        addra, signa = _link_index_to_addrs(link_indexa, nnorb)
        if link_indexa is link_indexb:
            addrb, signb = addra, signa
        else:
            addrb, signb = _link_index_to_addrs(link_indexb, nnorb)

        threads = (_TILE, _TILE)
        blocks = (1, (na + _TILE - 1) // _TILE)
        rest_args = (na, nb, nnorb, addra, addrb, signa, signb)
        t1 = cp.empty((nnorb, na * _TILE))
        gt1 = cp.empty((nnorb, na * _TILE))

        for strb0 in range(0, nb, _TILE):
            _kernel_build_t1(blocks, threads, (ci0, t1, strb0) + rest_args)
            eri.dot(t1, out=gt1)
            _kernel_gather(blocks, threads, (out, gt1, strb0) + rest_args)

        return out.reshape(original_shape).get()

    class GPUFCISolver(_cpu_direct_spin1.FCI):
        """FCI solver with GPU-accelerated contract_2e."""
        contract_2e = staticmethod(_gpu_contract_2e)

    GPU_FCI_AVAILABLE = True

except ImportError as e:
    _GPU_FCI_IMPORT_ERROR = str(e)
except Exception as e:
    _GPU_FCI_IMPORT_ERROR = str(e)


# Keep backward-compatible alias
GPU4PYSCF_AVAILABLE = GPU_FCI_AVAILABLE


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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
    Compute FCI energy on GPU using CUDA-accelerated Davidson solver.

    Builds mol/mf from scratch, transforms integrals to MO basis,
    then runs Davidson iteration with GPU contract_2e matvec.

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
        RuntimeError: If CuPy or PySCF is not available
    """
    if not GPU_FCI_AVAILABLE:
        raise RuntimeError(
            f"GPU FCI not available: {_GPU_FCI_IMPORT_ERROR}. "
            "Requires: cupy + pyscf"
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

    h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.kernel(mol, mf.mo_coeff)
    norb = mf.mo_coeff.shape[1]
    nelec = mol.nelec

    cisolver = GPUFCISolver(mol)
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
        RuntimeError: If CuPy or PySCF is not available
    """
    if not GPU_FCI_AVAILABLE:
        raise RuntimeError(
            f"GPU FCI not available: {_GPU_FCI_IMPORT_ERROR}. "
            "Requires: cupy + pyscf"
        )

    from pyscf import ao2mo

    h1e_np = np.asarray(h1e, dtype=np.float64)
    h2e_np = np.asarray(h2e, dtype=np.float64)

    norb = n_orbitals
    nelec = (n_alpha, n_beta)

    # Convert h2e to compressed 4-fold symmetric form
    # GPU contract_2e expects (nnorb, nnorb) where nnorb = norb*(norb+1)//2
    nnorb = norb * (norb + 1) // 2
    if h2e_np.ndim == 4:
        eri = ao2mo.restore(4, h2e_np.reshape(norb**2, norb**2), norb)
    elif h2e_np.ndim == 2 and h2e_np.shape == (nnorb, nnorb):
        eri = h2e_np
    elif h2e_np.ndim == 2 and h2e_np.shape[0] == norb**2:
        eri = ao2mo.restore(4, h2e_np, norb)
    else:
        eri = ao2mo.restore(4, h2e_np, norb)

    cisolver = GPUFCISolver(mol=None)
    cisolver.max_memory = max_memory
    cisolver.conv_tol = conv_tol
    cisolver.max_cycle = max_cycle

    e_corr, fcivec = cisolver.kernel(h1e_np, eri, norb, nelec)
    total_energy = float(e_corr) + nuclear_repulsion

    return total_energy
