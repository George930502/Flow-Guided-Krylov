"""
Quantum Circuit SKQD following the NVIDIA CUDA-Q tutorial.

Implements Sample-Based Krylov Quantum Diagonalization using quantum circuits
for Krylov state generation (Trotterized time evolution + computational basis
measurement). Falls back to classical Trotterized simulation when CUDA-Q is
not available.

This module provides the quantum circuit counterpart to the classical SKQD
in skqd.py. The key difference:
    - Classical SKQD: exact matrix exponential (gpu_expm_multiply), no Trotter error
    - Quantum SKQD: Trotterized exp_pauli circuit, finite shot noise

GPU Acceleration:
    All matrix operations (Pauli matrix construction, Trotter unitaries, time
    evolution, eigensolving) use PyTorch on GPU when available. The only CPU
    operations are bitstring sampling (multinomial) and final energy extraction.

Reference:
    NVIDIA CUDA-Q SKQD tutorial:
    nvidia.github.io/cuda-quantum/latest/applications/python/skqd.html
"""

import math
import numpy as np
import torch
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field

# Check CUDA-Q availability
try:
    import cudaq
    CUDAQ_AVAILABLE = True
except ImportError:
    CUDAQ_AVAILABLE = False

# Check CuPy availability (for fused CUDA kernel)
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Popcount parity utilities (Phase 2 optimization)
# ---------------------------------------------------------------------------

# Use torch.Tensor.bitwise_count if available (PyTorch 2.3+), else byte LUT
_HAS_BITWISE_COUNT = hasattr(torch.Tensor, "bitwise_count")

_PARITY_LUT_CACHE: Dict[torch.device, torch.Tensor] = {}


def _get_parity_lut(device: torch.device) -> torch.Tensor:
    """256-entry byte parity LUT: lut[b] = popcount(b) & 1."""
    if device not in _PARITY_LUT_CACHE:
        lut = torch.zeros(256, dtype=torch.int8, device=device)
        for i in range(256):
            lut[i] = bin(i).count("1") & 1
        _PARITY_LUT_CACHE[device] = lut
    return _PARITY_LUT_CACHE[device]


def _popcount_parity(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Compute popcount(x) & 1 for int64 tensor. Returns int8 tensor of 0s and 1s.

    Fast path: torch.Tensor.bitwise_count (PyTorch 2.3+) — single kernel call.
    Fallback: byte-level LUT decomposition (6 lookups per int64).
    """
    if _HAS_BITWISE_COUNT:
        return (x.bitwise_count() & 1).to(torch.int8)

    lut = _get_parity_lut(device)
    p = lut[(x & 0xFF).long()]
    for shift in (8, 16, 24, 32, 40, 48, 56):
        p = p ^ lut[((x >> shift) & 0xFF).long()]
    return p


# ---------------------------------------------------------------------------
# CuPy fused Pauli matvec kernel (Phase 4 optimization)
# ---------------------------------------------------------------------------

_CUPY_MATVEC_KERNEL = None
_CUPY_MATVEC_KERNEL_VERSION = 2  # Bump to invalidate cached kernels


def _get_cupy_matvec_kernel():
    """Compile and cache the CuPy fused Pauli matvec kernel."""
    global _CUPY_MATVEC_KERNEL
    if _CUPY_MATVEC_KERNEL is not None:
        return _CUPY_MATVEC_KERNEL
    if not CUPY_AVAILABLE:
        return None

    kernel_code = r"""
    extern "C" __global__
    void pauli_matvec(
        const double* psi_real, const double* psi_imag,
        double* result_real, double* result_imag,
        const long long* flip_masks, const long long* yz_masks,
        const double* coeff_real, const double* coeff_imag,
        const double* iny_real, const double* iny_imag,
        double constant_real, double constant_imag,
        int dim, int n_terms
    ) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        if (x >= dim) return;

        // Start with constant * psi[x]
        double acc_r = constant_real * psi_real[x] - constant_imag * psi_imag[x];
        double acc_i = constant_real * psi_imag[x] + constant_imag * psi_real[x];

        for (int k = 0; k < n_terms; k++) {
            int target = x ^ (int)flip_masks[k];

            // Phase at TARGET index (not source): phase(x^flip)
            int parity = __popc((long long)target & yz_masks[k]) & 1;
            double sign = 1.0 - 2.0 * parity;

            // phase = iny[k] * sign
            double phase_r = iny_real[k] * sign;
            double phase_i = iny_imag[k] * sign;

            // p_psi = phase * psi[target]
            double pt_r = psi_real[target];
            double pt_i = psi_imag[target];
            double ppsi_r = phase_r * pt_r - phase_i * pt_i;
            double ppsi_i = phase_r * pt_i + phase_i * pt_r;

            // contrib = coeff[k] * p_psi  (coeff is raw, NOT pre-multiplied by iny)
            double cr = coeff_real[k], ci = coeff_imag[k];
            acc_r += cr * ppsi_r - ci * ppsi_i;
            acc_i += cr * ppsi_i + ci * ppsi_r;
        }
        result_real[x] = acc_r;
        result_imag[x] = acc_i;
    }
    """
    try:
        _CUPY_MATVEC_KERNEL = cp.RawKernel(kernel_code, "pauli_matvec")
        return _CUPY_MATVEC_KERNEL
    except Exception:
        return None


# ---------------------------------------------------------------------------
# CuPy fused Trotter rotation kernel (eliminates Python-loop overhead)
# ---------------------------------------------------------------------------

_CUPY_TROTTER_KERNEL = None


def _get_cupy_trotter_kernel():
    """
    Compile and cache the CuPy in-place Pauli rotation kernel.

    Applies exp(-iθP)|ψ⟩ in-place for a single Pauli term.
    For off-diagonal terms (flip≠0), thread x (where x < target) processes
    both elements of the pair, avoiding race conditions.
    """
    global _CUPY_TROTTER_KERNEL
    if _CUPY_TROTTER_KERNEL is not None:
        return _CUPY_TROTTER_KERNEL
    if not CUPY_AVAILABLE:
        return None

    kernel_code = r"""
    extern "C" __global__
    void pauli_rotation_inplace(
        double* psi_r, double* psi_i,
        long long flip_mask, long long yz_mask,
        double iny_r, double iny_i,
        double cos_t, double sin_t,
        int dim
    ) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        if (x >= dim) return;

        int target = x ^ (int)flip_mask;

        if (flip_mask == 0) {
            // Diagonal: P|x⟩ = phase(x)|x⟩, so exp(-iθP)|x⟩ = (cosθ - i·sinθ·phase)|x⟩
            // Phase at target=x (same index)
            int parity = __popc((long long)x & yz_mask) & 1;
            double sign = 1.0 - 2.0 * parity;
            double ph_r = iny_r * sign;
            double ph_i = iny_i * sign;

            double pr = psi_r[x], pi = psi_i[x];
            // ppsi = phase * psi = (ph_r + i*ph_i)(pr + i*pi)
            double ppsi_r = ph_r * pr - ph_i * pi;
            double ppsi_i = ph_r * pi + ph_i * pr;
            // result = cos_t * psi - i * sin_t * ppsi
            // -i*(ppsi_r + i*ppsi_i) = ppsi_i - i*ppsi_r
            psi_r[x] = cos_t * pr + sin_t * ppsi_i;
            psi_i[x] = cos_t * pi - sin_t * ppsi_r;
        } else if (x < target) {
            // Off-diagonal pair: process both x and target
            double pr_x = psi_r[x], pi_x = psi_i[x];
            double pr_t = psi_r[target], pi_t = psi_i[target];

            // P|ψ⟩[x] = phase(target) * ψ[target]  (phase at TARGET index)
            int par_t = __popc((long long)target & yz_mask) & 1;
            double sign_t = 1.0 - 2.0 * par_t;
            double ph_t_r = iny_r * sign_t;
            double ph_t_i = iny_i * sign_t;
            double ppsi_x_r = ph_t_r * pr_t - ph_t_i * pi_t;
            double ppsi_x_i = ph_t_r * pi_t + ph_t_i * pr_t;

            // P|ψ⟩[target] = phase(x) * ψ[x]  (phase at x, which is target's TARGET)
            int par_x = __popc((long long)x & yz_mask) & 1;
            double sign_x = 1.0 - 2.0 * par_x;
            double ph_x_r = iny_r * sign_x;
            double ph_x_i = iny_i * sign_x;
            double ppsi_t_r = ph_x_r * pr_x - ph_x_i * pi_x;
            double ppsi_t_i = ph_x_r * pi_x + ph_x_i * pr_x;

            // result = cos_t * psi - i * sin_t * ppsi
            psi_r[x] = cos_t * pr_x + sin_t * ppsi_x_i;
            psi_i[x] = cos_t * pi_x - sin_t * ppsi_x_r;
            psi_r[target] = cos_t * pr_t + sin_t * ppsi_t_i;
            psi_i[target] = cos_t * pi_t - sin_t * ppsi_t_r;
        }
        // else x > target: skip (partner thread handles this pair)
    }
    """
    try:
        _CUPY_TROTTER_KERNEL = cp.RawKernel(kernel_code, "pauli_rotation_inplace")
        return _CUPY_TROTTER_KERNEL
    except Exception:
        return None


# ---------------------------------------------------------------------------
# CuPy fused diagonal Trotter kernel: fuses ALL diagonal Pauli rotations
# into a single kernel launch (Z/I-only strings, flip_mask==0).
# ---------------------------------------------------------------------------

_CUPY_FUSED_DIAG_KERNEL = None


def _get_cupy_fused_diag_kernel():
    """
    Compile and cache a CUDA kernel that fuses all diagonal Pauli rotations.

    For each basis state x, accumulates all diagonal exp(-iθ_k P_k) rotations
    in registers before writing result. Reduces diagonal kernel launches from
    n_diag_terms to 1 per half-step.
    """
    global _CUPY_FUSED_DIAG_KERNEL
    if _CUPY_FUSED_DIAG_KERNEL is not None:
        return _CUPY_FUSED_DIAG_KERNEL
    if not CUPY_AVAILABLE:
        return None

    kernel_code = r"""
    extern "C" __global__
    void fused_diagonal_trotter(
        double* psi_r, double* psi_i,
        const long long* yz_masks,
        const double* cos_vals,
        const double* sin_vals,
        const double* iny_r_vals,
        const double* iny_i_vals,
        int n_diag_terms,
        int dim
    ) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        if (x >= dim) return;

        // Accumulate combined rotation as (acc_r + i*acc_i) multiplier
        double acc_r = 1.0, acc_i = 0.0;

        for (int k = 0; k < n_diag_terms; k++) {
            int parity = __popc((long long)x & yz_masks[k]) & 1;
            double sign = 1.0 - 2.0 * parity;
            double ph_r = iny_r_vals[k] * sign;
            double ph_i = iny_i_vals[k] * sign;

            // rot = cos_t - i * sin_t * phase
            // -i * (ph_r + i*ph_i) = ph_i - i*ph_r
            double rot_r = cos_vals[k] + sin_vals[k] * ph_i;
            double rot_i = -sin_vals[k] * ph_r;

            // acc = acc * rot
            double new_r = acc_r * rot_r - acc_i * rot_i;
            double new_i = acc_r * rot_i + acc_i * rot_r;
            acc_r = new_r;
            acc_i = new_i;
        }

        // Apply accumulated rotation to psi[x]
        double pr = psi_r[x], pi = psi_i[x];
        psi_r[x] = acc_r * pr - acc_i * pi;
        psi_i[x] = acc_r * pi + acc_i * pr;
    }
    """
    try:
        _CUPY_FUSED_DIAG_KERNEL = cp.RawKernel(kernel_code, "fused_diagonal_trotter")
        return _CUPY_FUSED_DIAG_KERNEL
    except Exception:
        return None


# Threshold for dense Trotter unitary (Phase 3 — dim × dim complex128 ~ dim² × 16 bytes)
TROTTER_UNITARY_DIM_LIMIT = 8192  # ~13 qubits, U is ~1GB


# ---------------------------------------------------------------------------
# GPU Pauli matrices (cached, shared across instances)
# ---------------------------------------------------------------------------

_PAULI_CACHE: Dict[torch.device, Dict[str, torch.Tensor]] = {}


def _get_pauli_matrices(device: torch.device) -> Dict[str, torch.Tensor]:
    """Get single-qubit Pauli matrices on the specified device (cached)."""
    if device not in _PAULI_CACHE:
        _PAULI_CACHE[device] = {
            "I": torch.eye(2, dtype=torch.complex128, device=device),
            "X": torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128, device=device),
            "Y": torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128, device=device),
            "Z": torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128, device=device),
        }
    return _PAULI_CACHE[device]


@dataclass
class QuantumSKQDConfig:
    """Configuration for quantum circuit SKQD."""

    # Krylov parameters
    max_krylov_dim: int = 15  # Paper: d=15 (Ising simulation, Fig. 1)
    total_evolution_time: float = np.pi
    num_trotter_steps: int = 1  # Paper: single S₂(Δt) per evolution ([S₂(Δt)]^k)
    trotter_order: int = 2  # 1=first-order, 2=second-order Suzuki-Trotter (paper default)

    # Sampling
    shots: int = 100_000

    # Eigensolver
    num_eigenvalues: int = 2
    which_eigenvalues: str = "SA"

    # CUDA-Q target
    cudaq_target: str = "nvidia"
    cudaq_option: str = "fp64"  # fp64 for chemistry accuracy, fp32 for speed
    seed: int = 42

    # GPU postprocessing
    use_gpu: bool = True

    # Initial state
    initial_state: str = "hf"  # "hf" for molecular, "neel" for spin

    # Backend selection: "auto" (CUDA-Q if available), "cudaq", "classical",
    # "exact" (CuPy Trotter on GPU, Lanczos CPU fallback), "lanczos" (force Lanczos)
    backend: str = "auto"


class QuantumCircuitSKQD:
    """
    SKQD using quantum circuits for Krylov state generation.

    Follows the NVIDIA CUDA-Q tutorial algorithm:
    1. Prepare reference state |psi_0> (HF or Neel)
    2. For each k, apply Trotterized U^k = (e^{-iH*dt})^k via exp_pauli
    3. Sample in computational basis (cudaq.sample or classical fallback)
    4. Accumulate basis states across Krylov dimensions
    5. Project H onto basis and diagonalize classically

    For molecular systems, the Hamiltonian must be provided in Pauli form
    via Jordan-Wigner transformation (see hamiltonians.pauli_mapping).

    Args:
        pauli_coefficients: Coefficients for each Pauli term in H
        pauli_words: Pauli strings for each term (e.g., "XXIZI")
        n_qubits: Number of qubits
        config: Quantum SKQD configuration
        constant_energy: Energy offset from identity Pauli term + nuclear repulsion
        hamiltonian: Optional Hamiltonian object for Slater-Condon post-processing
    """

    def __init__(
        self,
        pauli_coefficients: List[float],
        pauli_words: List[str],
        n_qubits: int,
        config: Optional[QuantumSKQDConfig] = None,
        constant_energy: float = 0.0,
        hamiltonian=None,
        initial_state_vector: Optional[np.ndarray] = None,
    ):
        self.pauli_coefficients = pauli_coefficients
        self.pauli_words = pauli_words
        self.n_qubits = n_qubits
        self.config = config or QuantumSKQDConfig()
        self.constant_energy = constant_energy
        self.hamiltonian = hamiltonian
        self.initial_state_vector = initial_state_vector

        self.dt = self.config.total_evolution_time / self.config.num_trotter_steps

        # Select device
        self._device = (
            torch.device("cuda")
            if self.config.use_gpu and torch.cuda.is_available()
            else torch.device("cpu")
        )

        # Cached GPU tensors (lazy init)
        self._H_pauli_gpu: Optional[torch.Tensor] = None
        self._trotter_U_gpu: Optional[torch.Tensor] = None  # Combined Trotter unitary
        self._psi0_gpu: Optional[torch.Tensor] = None

        # State caching for Krylov evolution (Phase 1 optimization)
        # Maps krylov_power -> evolved state |ψₖ⟩ = (e^{-iHΔt})^k|ψ₀⟩
        self._cached_exact_states: Dict[int, torch.Tensor] = {}
        self._cached_trotter_states: Dict[int, torch.Tensor] = {}

        # Initialize CUDA-Q target ONCE (not per-sample call)
        self._cudaq_initialized = False
        self._cudaq_kernels_built = False

    # ------------------------------------------------------------------
    # GPU matrix construction
    # ------------------------------------------------------------------

    def _precompute_pauli_actions(self) -> None:
        """
        Precompute the action of each Pauli string on computational basis states.

        Each n-qubit Pauli string P acts on |x> as:
            P|x> = phase(x) |x'>
        where x' is obtained by flipping bits where X or Y acts, and
        phase is determined by Z and Y operators acting on the original state.

        This avoids building full dim×dim matrices entirely, enabling
        O(n_terms × dim) state-vector Trotter evolution instead of
        O(n_terms × dim²) dense matrix construction.

        Stores:
            _pauli_flip_masks: (n_terms,) integer masks for bit flips (X, Y positions)
            _pauli_phase_tables: (n_terms, dim) complex phase for each basis state
        """
        if hasattr(self, "_pauli_flip_masks") and self._pauli_flip_masks is not None:
            return

        device = self._device
        n_qubits = self.n_qubits
        n_terms = len(self.pauli_words)
        dim = 2 ** n_qubits

        # Precompute all basis state bit arrays: (dim, n_qubits) boolean
        indices = torch.arange(dim, device=device, dtype=torch.int64)
        # bit_array[i, q] = (i >> (n_qubits - 1 - q)) & 1
        shifts = torch.arange(n_qubits - 1, -1, -1, device=device, dtype=torch.int64)
        bit_array = ((indices.unsqueeze(1) >> shifts.unsqueeze(0)) & 1).to(torch.int8)
        # (dim, n_qubits)

        # Convert Pauli strings to numeric: I=0, X=1, Y=2, Z=3
        pauli_to_int = {"I": 0, "X": 1, "Y": 2, "Z": 3}
        pauli_ops = torch.tensor(
            [[pauli_to_int[p] for p in pw] for pw in self.pauli_words],
            dtype=torch.int8, device=device,
        )  # (n_terms, n_qubits)

        # Flip mask: integer whose bits are 1 where X or Y acts
        flip_positions = ((pauli_ops == 1) | (pauli_ops == 2)).to(torch.int64)
        # (n_terms, n_qubits)
        flip_masks = (flip_positions * (1 << shifts).unsqueeze(0)).sum(dim=1)
        # (n_terms,) integer flip masks

        # Phase table: for each (term, basis_state), compute the complex phase
        # P|x> = phase * |x ^ flip_mask>
        # Phase contributions:
        #   Z on bit=1: factor of -1 (i.e., phase *= (-1))
        #   Y on bit=0: factor of +i (phase *= i)
        #   Y on bit=1: factor of -i (phase *= -i)
        #   X on bit=0: factor of 1
        #   X on bit=1: factor of 1
        #   I: factor of 1

        # Compute phase index mod 4 for each (term, state) pair
        # Z on 1: contributes 2 (i.e. -1 = i^2)
        # Y on 0: contributes 1 (i.e. +i = i^1)
        # Y on 1: contributes 3 (i.e. -i = i^3)
        # All accumulated mod 4, then mapped to {1, i, -1, -i}

        y_mask = (pauli_ops == 2)  # (n_terms, n_qubits)
        z_mask = (pauli_ops == 3)  # (n_terms, n_qubits)

        # Broadcast: bit_array (dim, nq) vs masks (n_terms, nq)
        # Work in chunks to avoid OOM for large dim × n_terms
        chunk_size = max(1, min(dim, 65536 // max(n_terms, 1)))
        phase_table = torch.empty(n_terms, dim, dtype=torch.complex128, device=device)

        phase_lookup = torch.tensor(
            [1.0 + 0j, 0.0 + 1j, -1.0 + 0j, 0.0 - 1j],
            dtype=torch.complex128, device=device,
        )

        for start in range(0, dim, chunk_size):
            end = min(start + chunk_size, dim)
            bits_chunk = bit_array[start:end]  # (chunk, nq)

            # (chunk, 1, nq) vs (1, n_terms, nq)
            bits_exp = bits_chunk[:, None, :]
            y_exp = y_mask[None, :, :]
            z_exp = z_mask[None, :, :]

            n_y0 = (y_exp & (bits_exp == 0)).sum(dim=2, dtype=torch.int32)
            n_y1 = (y_exp & (bits_exp == 1)).sum(dim=2, dtype=torch.int32)
            n_z1 = (z_exp & (bits_exp == 1)).sum(dim=2, dtype=torch.int32)

            phase_idx = (n_y0 - n_y1 + 2 * n_z1) % 4  # (chunk, n_terms)
            phases = phase_lookup[phase_idx.long()]  # (chunk, n_terms)

            phase_table[:, start:end] = phases.T  # (n_terms, chunk)

        self._pauli_flip_masks = flip_masks  # (n_terms,)
        self._pauli_phase_tables = phase_table  # (n_terms, dim)

    def _apply_pauli_exp_to_state(
        self, psi: torch.Tensor, term_idx: int, coeff: float, dt_scale: float = 1.0
    ) -> torch.Tensor:
        """
        Apply exp(-i * coeff * dt * dt_scale * P_k) to state vector psi in O(dim) time.

        Uses P^2 = I identity:
            exp(-i*theta*P)|psi> = cos(theta)|psi> - i*sin(theta)*P|psi>

        P|psi> is computed via flip_mask and phase_table without building
        the full dim×dim matrix.

        Args:
            dt_scale: Multiplier for dt (0.5 for second-order Trotter half-steps).
        """
        theta = coeff * self.dt * dt_scale
        cos_t = torch.tensor(np.cos(theta), dtype=torch.complex128, device=psi.device)
        sin_t = torch.tensor(np.sin(theta), dtype=torch.complex128, device=psi.device)

        if abs(sin_t.real.item()) < 1e-15:
            return cos_t * psi

        # P|psi>: flip bits and apply phase
        flip_mask = self._pauli_flip_masks[term_idx]
        phases = self._pauli_phase_tables[term_idx]  # (dim,)

        # XOR with flip_mask to get target indices (cached arange)
        if not hasattr(self, '_arange_cache') or self._arange_cache.shape[0] != len(psi):
            self._arange_cache = torch.arange(len(psi), device=psi.device, dtype=torch.int64)
        target_indices = self._arange_cache ^ flip_mask

        # P|psi>[x] = phase[x ^ flip] * psi[x ^ flip]
        # But we want: result[x] = sum_y P[x,y] * psi[y]
        # P[x,y] = phase(y) * delta(x, y ^ flip)
        # So result[x] = phase(x ^ flip) * psi[x ^ flip]
        p_psi = phases[target_indices] * psi[target_indices]

        return cos_t * psi - 1j * sin_t * p_psi

    def _build_pauli_matrix_gpu(self) -> torch.Tensor:
        """Build full Hamiltonian matrix from Pauli decomposition on GPU."""
        if self._H_pauli_gpu is not None:
            return self._H_pauli_gpu

        device = self._device
        dim = 2 ** self.n_qubits

        # Use precomputed Pauli actions for O(n_terms * dim) construction
        self._precompute_pauli_actions()

        H = torch.zeros((dim, dim), dtype=torch.complex128, device=device)
        indices = torch.arange(dim, device=device, dtype=torch.int64)

        for k, (coeff, pw) in enumerate(zip(self.pauli_coefficients, self.pauli_words)):
            flip_mask = self._pauli_flip_masks[k]
            phases = self._pauli_phase_tables[k]
            target_indices = indices ^ flip_mask

            # P[x, x^flip] = phase[x^flip], so H[x, x^flip] += coeff * phase[x^flip]
            H[indices, target_indices] += coeff * phases[target_indices]

        H.add_(torch.eye(dim, dtype=torch.complex128, device=device), alpha=self.constant_energy)

        self._H_pauli_gpu = H
        return H

    def _build_trotter_unitary_gpu(self) -> torch.Tensor:
        """
        Build combined single-Trotter-step unitary on GPU via state-vector method.

        Instead of building the full Trotter unitary as a dense matrix product,
        applies each exp(-i*c_k*dt*P_k) column-by-column using the O(dim)
        Pauli action. Total cost: O(n_terms * dim^2) with O(dim) per column,
        vs O(n_terms * dim^2 * dim) for naive dense matrix multiply.

        Pre-computed once and reused for all Krylov powers.
        """
        if self._trotter_U_gpu is not None:
            return self._trotter_U_gpu

        device = self._device
        dim = 2 ** self.n_qubits

        self._precompute_pauli_actions()

        # Build U by applying Trotter sequence to each basis vector
        # U[:, j] = prod_k exp(-i*c_k*dt*P_k) |e_j>
        U = torch.eye(dim, dtype=torch.complex128, device=device)

        n_terms = len(self.pauli_coefficients)
        for j in range(dim):
            col = U[:, j].clone()
            if self.config.trotter_order == 2:
                # Second-order Suzuki-Trotter: forward half + backward half
                for k, coeff in enumerate(self.pauli_coefficients):
                    col = self._apply_pauli_exp_to_state(col, k, coeff, dt_scale=0.5)
                for k in range(n_terms - 1, -1, -1):
                    col = self._apply_pauli_exp_to_state(
                        col, k, self.pauli_coefficients[k], dt_scale=0.5
                    )
            else:
                for k, coeff in enumerate(self.pauli_coefficients):
                    col = self._apply_pauli_exp_to_state(col, k, coeff)
            U[:, j] = col

        self._trotter_U_gpu = U
        return U

    def _get_initial_state_gpu(self) -> torch.Tensor:
        """Get initial state vector on GPU."""
        if self._psi0_gpu is not None:
            return self._psi0_gpu.clone()

        device = self._device
        dim = 2 ** self.n_qubits
        psi = torch.zeros(dim, dtype=torch.complex128, device=device)

        if self.initial_state_vector is not None:
            psi = torch.from_numpy(self.initial_state_vector).to(
                dtype=torch.complex128, device=device
            )
        elif self.config.initial_state == "hf" and self.hamiltonian is not None:
            hf = self.hamiltonian.get_hf_state().cpu().numpy()
            idx = int("".join(str(int(b)) for b in hf), 2)
            psi[idx] = 1.0
        elif self.config.initial_state == "neel":
            bits = [1 if i % 2 == 0 else 0 for i in range(self.n_qubits)]
            idx = int("".join(str(b) for b in bits), 2)
            psi[idx] = 1.0
        else:
            psi[0] = 1.0

        self._psi0_gpu = psi
        return psi.clone()

    # ------------------------------------------------------------------
    # Quantum circuit Krylov state generation (CUDA-Q)
    # ------------------------------------------------------------------

    def _init_cudaq(self) -> None:
        """
        Initialize CUDA-Q target and compile kernels ONCE.

        Per CUDA-Q docs, set_target should be called once before the first
        kernel execution, not per-sample call. Kernel definitions are also
        compiled once and reused.

        IMPORTANT: Pre-computes exp_pauli angles = [-coeff * dt] outside the
        kernel. CUDA-Q's JIT compiler miscompiles `coeffs[i] * dt_val` when
        coeffs is a list[float] kernel argument (produces zero rotation).
        Workaround: pass pre-computed angles directly to exp_pauli(angles[i]).
        """
        if self._cudaq_initialized:
            return

        if not CUDAQ_AVAILABLE:
            raise RuntimeError("CUDA-Q is not available. Install cudaq package.")

        # Set target ONCE with precision option
        target = self.config.cudaq_target
        option = self.config.cudaq_option
        if option:
            cudaq.set_target(target, option=option)
        else:
            cudaq.set_target(target)

        # Pre-convert Pauli words (reused across all Krylov steps)
        pauli_words_raw = [cudaq.pauli_word(pw) for pw in self.pauli_words]

        # Pre-compute exp_pauli angles outside kernel to avoid JIT bug
        # exp_pauli(angle, q, P) applies exp(i * angle * P)
        # We want exp(-i * coeff * dt * P), so angle = -coeff * dt
        if self.config.trotter_order == 2:
            # Second-order Suzuki-Trotter: forward half + reversed half per step
            # Combined list so the kernel loop structure stays identical
            half = [-c * self.dt / 2 for c in self.pauli_coefficients]
            self._exp_pauli_angles = half + half[::-1]
            self._pauli_words_cudaq = pauli_words_raw + pauli_words_raw[::-1]
        else:
            self._exp_pauli_angles = [-c * self.dt for c in self.pauli_coefficients]
            self._pauli_words_cudaq = pauli_words_raw

        # Determine occupied qubits for HF state (reused across all Krylov steps)
        if self.config.initial_state == "hf":
            if self.hamiltonian is not None and hasattr(self.hamiltonian, "get_hf_state"):
                hf = self.hamiltonian.get_hf_state().cpu().numpy()
                self._occupied_qubits = [i for i in range(self.n_qubits) if hf[i] == 1]
            else:
                self._occupied_qubits = list(range(self.n_qubits // 2))

        self._cudaq_initialized = True

    def _build_cudaq_kernels(self) -> None:
        """
        Build and cache CUDA-Q kernels ONCE.

        Three kernel types:
        1. init_hf / init_neel: Prepare initial state (no evolution, no measurement)
        2. evolve_from_state: Take cudaq.State, apply ONE Trotter step (no measurement)
        3. Legacy krylov_circuit_*: Full from-scratch evolution + measurement (fallback)

        State-cached kernels (1+2) use cudaq.get_state() for state extraction,
        then manual sampling from |ψ|². This reduces total Trotter applications
        from O(k²) to O(k) by reusing cached intermediate states.
        """
        if self._cudaq_kernels_built:
            return

        # --- State-cached kernels (no measurement, for get_state) ---

        @cudaq.kernel
        def init_hf_kernel(
            num_qubits: int,
            occ_qubits: list[int],
        ):
            """Prepare HF initial state (no evolution, no measurement)."""
            qubits = cudaq.qvector(num_qubits)
            for oq in range(len(occ_qubits)):
                x(qubits[occ_qubits[oq]])

        @cudaq.kernel
        def init_neel_kernel(
            num_qubits: int,
        ):
            """Prepare Neel initial state (no evolution, no measurement)."""
            qubits = cudaq.qvector(num_qubits)
            for qubit_index in range(num_qubits):
                if qubit_index % 2 == 0:
                    x(qubits[qubit_index])

        @cudaq.kernel
        def evolve_one_step(
            state: cudaq.State,
            trotter_steps: int,
            H_pauli_words: list[cudaq.pauli_word],
            angles: list[float],
        ):
            """Apply ONE Trotter step to an existing state (no measurement)."""
            qubits = cudaq.qvector(state)
            for _ in range(trotter_steps):
                for i in range(len(angles)):
                    exp_pauli(angles[i], qubits, H_pauli_words[i])

        self._kernel_init_hf = init_hf_kernel
        self._kernel_init_neel = init_neel_kernel
        self._kernel_evolve_one_step = evolve_one_step

        # --- Legacy kernels (full from-scratch, for fallback) ---

        @cudaq.kernel
        def krylov_circuit_hf(
            num_qubits: int,
            krylov_power: int,
            trotter_steps: int,
            H_pauli_words: list[cudaq.pauli_word],
            angles: list[float],
            occ_qubits: list[int],
        ):
            qubits = cudaq.qvector(num_qubits)
            for oq in range(len(occ_qubits)):
                x(qubits[occ_qubits[oq]])
            for _ in range(krylov_power):
                for _ in range(trotter_steps):
                    for i in range(len(angles)):
                        exp_pauli(angles[i], qubits, H_pauli_words[i])
            mz(qubits)

        @cudaq.kernel
        def krylov_circuit_neel(
            num_qubits: int,
            krylov_power: int,
            trotter_steps: int,
            H_pauli_words: list[cudaq.pauli_word],
            angles: list[float],
        ):
            qubits = cudaq.qvector(num_qubits)
            for qubit_index in range(num_qubits):
                if qubit_index % 2 == 0:
                    x(qubits[qubit_index])
            for _ in range(krylov_power):
                for _ in range(trotter_steps):
                    for i in range(len(angles)):
                        exp_pauli(angles[i], qubits, H_pauli_words[i])
            mz(qubits)

        self._kernel_hf = krylov_circuit_hf
        self._kernel_neel = krylov_circuit_neel
        self._cudaq_kernels_built = True

        # State cache for incremental evolution
        self._cached_cudaq_states: Dict[int, Any] = {}

    def _sample_cudaq(self, krylov_power: int) -> Dict[str, int]:
        """
        Sample from Krylov state U^k|ψ₀⟩ using CUDA-Q with state caching.

        State-cached approach (O(k) total Trotter steps instead of O(k²)):
        1. k=0: get_state of initial state (HF/Neel, no evolution)
        2. k>0: evolve cached state k-1 by ONE Trotter step via get_state
        3. Sample from |ψ|² via torch.multinomial (same as Path C)

        Falls back to legacy from-scratch sampling if get_state fails.
        """
        self._init_cudaq()
        self._build_cudaq_kernels()

        cudaq.set_random_seed(self.config.seed + krylov_power)

        try:
            return self._sample_cudaq_cached(krylov_power)
        except Exception as e:
            if krylov_power == 0:
                print(f"  CUDA-Q state caching unavailable ({e}), "
                      f"using legacy from-scratch sampling")
            return self._sample_cudaq_legacy(krylov_power)

    def _sample_cudaq_cached(self, krylov_power: int) -> Dict[str, int]:
        """State-cached CUDA-Q: get_state → cache → manual sampling."""
        if krylov_power == 0:
            # Get initial state (no evolution)
            if self.config.initial_state == "hf":
                state = cudaq.get_state(
                    self._kernel_init_hf,
                    self.n_qubits,
                    self._occupied_qubits,
                )
            else:
                state = cudaq.get_state(
                    self._kernel_init_neel,
                    self.n_qubits,
                )

            if not hasattr(self, "_cudaq_cached_info_printed"):
                self._cudaq_cached_info_printed = True
                n_terms = len(self.pauli_coefficients)
                dim = 2 ** self.n_qubits
                print(f"  CUDA-Q state-cached mode ({n_terms} Pauli terms, "
                      f"dim={dim:,}, 1 Trotter step/k)")
        else:
            # Evolve cached state by ONE Trotter step
            prev_state = self._cached_cudaq_states[krylov_power - 1]
            state = cudaq.get_state(
                self._kernel_evolve_one_step,
                prev_state,
                self.config.num_trotter_steps,
                self._pauli_words_cudaq,
                self._exp_pauli_angles,
            )

        # Cache state for next step, evict previous
        self._cached_cudaq_states[krylov_power] = state
        if krylov_power > 0 and (krylov_power - 1) in self._cached_cudaq_states:
            del self._cached_cudaq_states[krylov_power - 1]

        # Convert cudaq.State → torch tensor → sample from |ψ|²
        psi_np = np.array(state, dtype=np.complex128)
        psi = torch.from_numpy(psi_np).to(self._device)
        return self._sample_from_state(psi, krylov_power)

    def _sample_cudaq_legacy(self, krylov_power: int) -> Dict[str, int]:
        """Legacy from-scratch CUDA-Q sampling (O(k²) total, no state caching)."""
        if self.config.initial_state == "hf":
            result = cudaq.sample(
                self._kernel_hf,
                self.n_qubits,
                krylov_power,
                self.config.num_trotter_steps,
                self._pauli_words_cudaq,
                self._exp_pauli_angles,
                self._occupied_qubits,
                shots_count=self.config.shots,
            )
        else:
            result = cudaq.sample(
                self._kernel_neel,
                self.n_qubits,
                krylov_power,
                self.config.num_trotter_steps,
                self._pauli_words_cudaq,
                self._exp_pauli_angles,
                shots_count=self.config.shots,
            )

        return dict(result.items())

    # ------------------------------------------------------------------
    # Classical Trotterized fallback (GPU-accelerated, no CUDA-Q needed)
    # ------------------------------------------------------------------

    def _apply_trotter_step(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Apply one full Trotter step (all num_trotter_steps sub-steps) to psi.

        Supports first-order and second-order Suzuki-Trotter decomposition.
        Second-order (paper default): symmetric split
            S₂(dt) = ∏_k exp(-i·c_k·dt/2·P_k) · ∏_k^{rev} exp(-i·c_k·dt/2·P_k)

        Total cost per Trotter step:
            First-order:  O(num_trotter_steps * n_terms * dim)
            Second-order: O(num_trotter_steps * 2 * n_terms * dim)
        """
        n_terms = len(self.pauli_coefficients)
        if self.config.trotter_order == 2:
            # Second-order Suzuki-Trotter (matches SKQD paper experiments)
            for _ in range(self.config.num_trotter_steps):
                # Forward half-step
                for k in range(n_terms):
                    psi = self._apply_pauli_exp_to_state(
                        psi, k, self.pauli_coefficients[k], dt_scale=0.5
                    )
                # Backward half-step (reversed order)
                for k in range(n_terms - 1, -1, -1):
                    psi = self._apply_pauli_exp_to_state(
                        psi, k, self.pauli_coefficients[k], dt_scale=0.5
                    )
        else:
            # First-order Trotter (NVIDIA tutorial default)
            for _ in range(self.config.num_trotter_steps):
                for k, coeff in enumerate(self.pauli_coefficients):
                    psi = self._apply_pauli_exp_to_state(psi, k, coeff)
        return psi

    def _sample_classical_trotterized(self, krylov_power: int) -> Dict[str, int]:
        """
        Classical fallback: Trotterized state-vector evolution on GPU.

        Phase 1: State caching (7x fewer Trotter applications).
        Phase 3: Auto-selects dense unitary for small systems (≤13 qubits)
                 or lightweight per-term Trotter for large systems (no phase_table).
        """
        dim = 2 ** self.n_qubits
        use_dense = dim <= TROTTER_UNITARY_DIM_LIMIT

        if use_dense:
            # Dense unitary: precompute U once, then U @ psi per step
            self._precompute_pauli_actions()
        else:
            # Lightweight: no phase_table, O(n_terms) memory
            self._precompute_pauli_masks_lightweight()

        if krylov_power == 0 and not hasattr(self, "_trotter_info_printed"):
            n_terms = len(self.pauli_coefficients)
            order = self.config.trotter_order
            mode = "dense-U" if use_dense else "lightweight"
            print(f"  State-vector Trotter-{order} ({n_terms} terms, dim={dim:,}, "
                  f"{self.config.num_trotter_steps} steps/evolution, {mode}, state-cached)")
            self._trotter_info_printed = True

        # Phase 3: Pre-compute trig values for lightweight path
        if not use_dense and not hasattr(self, "_trotter_cos_sin"):
            self._precompute_trotter_trig()

        # Phase 1: Use cached evolved state from previous Krylov step
        if krylov_power == 0:
            psi = self._get_initial_state_gpu()
        elif krylov_power - 1 in self._cached_trotter_states:
            psi = self._cached_trotter_states[krylov_power - 1].clone()
            if use_dense:
                psi = self._apply_trotter_step(psi)
            else:
                psi = self._apply_trotter_step_lightweight(psi)
        else:
            # Fallback: evolve from scratch
            psi = self._get_initial_state_gpu()
            step_fn = self._apply_trotter_step if use_dense else self._apply_trotter_step_lightweight
            for _ in range(krylov_power):
                psi = step_fn(psi)

        psi = psi / torch.linalg.norm(psi)
        self._cached_trotter_states[krylov_power] = psi.clone()

        # Evict previous state to free VRAM (only state k-1 is ever needed)
        if krylov_power > 0 and (krylov_power - 1) in self._cached_trotter_states:
            del self._cached_trotter_states[krylov_power - 1]

        return self._sample_from_state(psi, krylov_power)

    def _precompute_trotter_trig(self) -> None:
        """Pre-compute cos/sin values for all Pauli terms (Phase 3c)."""
        n_terms = len(self.pauli_coefficients)

        if self.config.trotter_order == 2:
            # Half-step angles for second-order Trotter
            thetas = torch.tensor(
                [c * self.dt * 0.5 for c in self.pauli_coefficients],
                dtype=torch.float64,
                device=self._device,
            )
        else:
            thetas = torch.tensor(
                [c * self.dt for c in self.pauli_coefficients],
                dtype=torch.float64,
                device=self._device,
            )

        self._trotter_cos_sin = (
            torch.cos(thetas).to(torch.complex128),
            torch.sin(thetas).to(torch.complex128),
        )

    def _apply_pauli_exp_lightweight(
        self, psi: torch.Tensor, term_idx: int, cos_t: torch.Tensor, sin_t: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply exp(-iθP_k)|ψ⟩ using lightweight masks (Phase 3a).

        Uses _lw_flip_masks + _lw_yz_masks + _lw_i_ny instead of phase_tables.
        Memory: O(dim) working space instead of O(n_terms × dim) phase_table.
        Enables Path B for 18-20+ qubit systems.
        """
        if abs(sin_t.real.item()) < 1e-15:
            return cos_t * psi

        flip_mask = self._lw_flip_masks[term_idx].item()
        yz_mask = self._lw_yz_masks[term_idx].item()
        i_ny = self._lw_i_ny[term_idx]

        if not hasattr(self, "_arange_cache") or self._arange_cache.shape[0] != len(psi):
            self._arange_cache = torch.arange(len(psi), device=psi.device, dtype=torch.int64)

        target_indices = self._arange_cache ^ flip_mask
        psi_flipped = psi[target_indices]

        # Compute sign from bit parity at TARGET index (not source)
        masked = target_indices & yz_mask
        parity = _popcount_parity(masked, psi.device)
        sign = (1.0 - 2.0 * parity.to(torch.float64))

        p_psi = i_ny * sign * psi_flipped

        return cos_t * psi - 1j * sin_t * p_psi

    def _apply_trotter_step_lightweight(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Apply one full Trotter step using lightweight masks (Phase 3).

        Auto-selects CuPy fused kernel (if available) or PyTorch fallback.
        """
        # Try CuPy fused path first (eliminates Python-loop overhead)
        if psi.device.type == "cuda" and CUPY_AVAILABLE:
            kernel = _get_cupy_trotter_kernel()
            if kernel is not None:
                return self._apply_trotter_step_cupy(psi, kernel)

        # PyTorch fallback
        cos_vals, sin_vals = self._trotter_cos_sin
        n_terms = len(self.pauli_coefficients)

        if self.config.trotter_order == 2:
            for _ in range(self.config.num_trotter_steps):
                for k in range(n_terms):
                    psi = self._apply_pauli_exp_lightweight(
                        psi, k, cos_vals[k], sin_vals[k]
                    )
                for k in range(n_terms - 1, -1, -1):
                    psi = self._apply_pauli_exp_lightweight(
                        psi, k, cos_vals[k], sin_vals[k]
                    )
        else:
            for _ in range(self.config.num_trotter_steps):
                for k in range(n_terms):
                    psi = self._apply_pauli_exp_lightweight(
                        psi, k, cos_vals[k], sin_vals[k]
                    )
        return psi

    def _apply_trotter_step_cupy(self, psi: torch.Tensor, kernel) -> torch.Tensor:
        """
        Apply one full Trotter step via CuPy fused kernel.

        Uses a fused diagonal kernel for all diagonal Pauli terms (flip_mask==0)
        in a single launch, then individual launches for off-diagonal terms.
        This reduces kernel launches by ~2.5x for typical molecular Hamiltonians.
        """
        dim = len(psi)
        n_terms = len(self.pauli_coefficients)
        block = 256
        grid = (dim + block - 1) // block

        # Pre-extract masks as CPU Python ints (avoid .item() per call)
        if not hasattr(self, "_cupy_trotter_precomputed"):
            self._cupy_trotter_precomputed = True
            self._cupy_flip_masks_cpu = [int(m.item()) for m in self._lw_flip_masks]
            self._cupy_yz_masks_cpu = [int(m.item()) for m in self._lw_yz_masks]
            self._cupy_iny_r_cpu = [float(v.real.item()) for v in self._lw_i_ny]
            self._cupy_iny_i_cpu = [float(v.imag.item()) for v in self._lw_i_ny]
            cos_vals, sin_vals = self._trotter_cos_sin
            self._cupy_cos_cpu = [float(v.real.item()) for v in cos_vals]
            self._cupy_sin_cpu = [float(v.real.item()) for v in sin_vals]

            # Split into diagonal (flip_mask==0) and off-diagonal indices
            self._cupy_diag_indices = [
                k for k in range(n_terms)
                if self._cupy_flip_masks_cpu[k] == 0 and abs(self._cupy_sin_cpu[k]) >= 1e-15
            ]
            self._cupy_offdiag_indices = [
                k for k in range(n_terms)
                if self._cupy_flip_masks_cpu[k] != 0 and abs(self._cupy_sin_cpu[k]) >= 1e-15
            ]

            # Pre-allocate CuPy arrays for fused diagonal kernel
            n_diag = len(self._cupy_diag_indices)
            if n_diag > 0:
                self._cupy_diag_yz = cp.array(
                    [self._cupy_yz_masks_cpu[k] for k in self._cupy_diag_indices],
                    dtype=cp.int64,
                )
                self._cupy_diag_cos = cp.array(
                    [self._cupy_cos_cpu[k] for k in self._cupy_diag_indices],
                    dtype=cp.float64,
                )
                self._cupy_diag_sin = cp.array(
                    [self._cupy_sin_cpu[k] for k in self._cupy_diag_indices],
                    dtype=cp.float64,
                )
                self._cupy_diag_iny_r = cp.array(
                    [self._cupy_iny_r_cpu[k] for k in self._cupy_diag_indices],
                    dtype=cp.float64,
                )
                self._cupy_diag_iny_i = cp.array(
                    [self._cupy_iny_i_cpu[k] for k in self._cupy_diag_indices],
                    dtype=cp.float64,
                )

        flip_masks = self._cupy_flip_masks_cpu
        yz_masks = self._cupy_yz_masks_cpu
        iny_r = self._cupy_iny_r_cpu
        iny_i = self._cupy_iny_i_cpu
        cos_v = self._cupy_cos_cpu
        sin_v = self._cupy_sin_cpu
        diag_indices = self._cupy_diag_indices
        offdiag_indices = self._cupy_offdiag_indices

        # Convert psi to separate real/imag CuPy arrays (in-place via DLPack)
        psi_r_torch = psi.real.contiguous()
        psi_i_torch = psi.imag.contiguous()

        # Work on CuPy arrays (zero-copy from torch)
        psi_r = cp.from_dlpack(psi_r_torch.detach())
        psi_i = cp.from_dlpack(psi_i_torch.detach())

        np_int64 = np.int64
        np_float64 = np.float64
        np_int32 = np.int32

        fused_diag_kernel = _get_cupy_fused_diag_kernel()
        n_diag = len(diag_indices)

        def _apply_half_step(term_order):
            """Apply diagonal (fused) + off-diagonal (individual) for one half-step."""
            # Fused diagonal: single kernel launch for all Z/I-only terms
            if n_diag > 0 and fused_diag_kernel is not None:
                fused_diag_kernel(
                    (grid,), (block,),
                    (psi_r, psi_i,
                     self._cupy_diag_yz,
                     self._cupy_diag_cos, self._cupy_diag_sin,
                     self._cupy_diag_iny_r, self._cupy_diag_iny_i,
                     np_int32(n_diag), np_int32(dim)),
                )
            elif n_diag > 0:
                # Fallback: individual diagonal launches
                for k in diag_indices:
                    kernel(
                        (grid,), (block,),
                        (psi_r, psi_i,
                         np_int64(flip_masks[k]), np_int64(yz_masks[k]),
                         np_float64(iny_r[k]), np_float64(iny_i[k]),
                         np_float64(cos_v[k]), np_float64(sin_v[k]),
                         np_int32(dim)),
                    )

            # Off-diagonal: individual kernel launches (non-commutative)
            for k in term_order:
                kernel(
                    (grid,), (block,),
                    (psi_r, psi_i,
                     np_int64(flip_masks[k]), np_int64(yz_masks[k]),
                     np_float64(iny_r[k]), np_float64(iny_i[k]),
                     np_float64(cos_v[k]), np_float64(sin_v[k]),
                     np_int32(dim)),
                )

        if self.config.trotter_order == 2:
            for _ in range(self.config.num_trotter_steps):
                # Forward half-step
                _apply_half_step(offdiag_indices)
                # Backward half-step (reversed off-diagonal, same fused diagonal)
                _apply_half_step(list(reversed(offdiag_indices)))
        else:
            for _ in range(self.config.num_trotter_steps):
                _apply_half_step(offdiag_indices)

        # Reconstruct complex torch tensor from real/imag CuPy arrays
        result = torch.complex(
            torch.from_dlpack(psi_r),
            torch.from_dlpack(psi_i),
        )
        return result

    # ------------------------------------------------------------------
    # Exact evolution backend (Lanczos, no Trotter decomposition)
    # ------------------------------------------------------------------

    def _precompute_pauli_masks_lightweight(self) -> None:
        """
        Precompute lightweight Pauli masks for Hamiltonian matvec.

        Unlike _precompute_pauli_actions() which stores O(n_terms × dim) phase tables,
        this stores only O(n_terms) integer masks and coefficients. The per-state
        phase is computed on-the-fly from bit parity operations.

        Enables exact time evolution for systems where 2^n is large (≥18 qubits)
        without the memory overhead of phase tables.
        """
        if hasattr(self, "_lw_flip_masks") and self._lw_flip_masks is not None:
            return

        device = self._device
        n_qubits = self.n_qubits

        flip_masks = []
        yz_masks = []
        n_y_counts = []

        for pw in self.pauli_words:
            flip = 0
            yz = 0
            ny = 0
            for q, p in enumerate(pw):
                bit = 1 << (n_qubits - 1 - q)
                if p in ("X", "Y"):
                    flip |= bit
                if p in ("Y", "Z"):
                    yz |= bit
                if p == "Y":
                    ny += 1
            flip_masks.append(flip)
            yz_masks.append(yz)
            n_y_counts.append(ny)

        self._lw_flip_masks = torch.tensor(flip_masks, dtype=torch.int64, device=device)
        self._lw_yz_masks = torch.tensor(yz_masks, dtype=torch.int64, device=device)

        # Precompute i^{n_Y} for each term
        i_powers = [1.0 + 0j, 0.0 + 1j, -1.0 + 0j, 0.0 - 1j]
        self._lw_i_ny = torch.tensor(
            [i_powers[ny % 4] for ny in n_y_counts],
            dtype=torch.complex128,
            device=device,
        )
        self._lw_coeffs = torch.tensor(
            self.pauli_coefficients, dtype=torch.complex128, device=device
        )

    def _apply_hamiltonian_matvec(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Compute H|ψ⟩ = E_const|ψ⟩ + Σ_k c_k P_k|ψ⟩.

        Three backends (auto-selected):
        1. CuPy fused kernel: single GPU kernel, hardware __popc, zero intermediate allocs
        2. PyTorch bitwise_count: single-kernel popcount (PyTorch 2.3+)
        3. PyTorch LUT fallback: byte-level popcount parity via 256-entry LUT
        """
        self._precompute_pauli_masks_lightweight()

        dim = len(psi)
        device = psi.device
        n_terms = len(self.pauli_coefficients)

        # --- Fast path: CuPy fused CUDA kernel (Phase 4) ---
        if device.type == "cuda" and CUPY_AVAILABLE:
            kernel = _get_cupy_matvec_kernel()
            if kernel is not None:
                return self._apply_hamiltonian_matvec_cupy(psi, kernel)

        # --- PyTorch path (Phase 2 optimized) ---
        result = self.constant_energy * psi.clone()

        # Cache arange for index computation
        if not hasattr(self, "_matvec_arange") or self._matvec_arange.shape[0] != dim:
            self._matvec_arange = torch.arange(dim, device=device, dtype=torch.int64)
        indices = self._matvec_arange

        # Adaptive chunk size based on available GPU memory
        if device.type == "cuda" and torch.cuda.is_available():
            free_mem = torch.cuda.mem_get_info(device)[0]
            # Each term in chunk needs: targets(int64) + psi_gathered(c128) + masked(int64) + sign(f64)
            bytes_per_term = dim * (8 + 16 + 8 + 8)
            chunk_size = max(8, min(n_terms, int(free_mem * 0.3 / bytes_per_term)))
        else:
            chunk_size = min(64, n_terms)

        # Process all terms in single pass if they fit
        if chunk_size >= n_terms:
            chunk_size = n_terms

        for start in range(0, n_terms, chunk_size):
            end = min(start + chunk_size, n_terms)

            fmasks = self._lw_flip_masks[start:end]
            yzmasks = self._lw_yz_masks[start:end]
            iny = self._lw_i_ny[start:end]
            coeffs = self._lw_coeffs[start:end]

            targets = indices.unsqueeze(0) ^ fmasks.unsqueeze(1)
            psi_gathered = psi[targets]

            # Popcount parity at TARGET index (not source) — phase(x^flip)
            masked = targets & yzmasks.unsqueeze(1)
            parity = _popcount_parity(masked, device).to(torch.float64)
            sign = 1.0 - 2.0 * parity

            phase = iny[:, None] * sign
            contrib = coeffs[:, None] * phase * psi_gathered

            result += contrib.sum(dim=0)

        return result

    def _apply_hamiltonian_matvec_cupy(
        self, psi: torch.Tensor, kernel
    ) -> torch.Tensor:
        """
        Fused CuPy CUDA kernel for H|ψ⟩. Single kernel launch, hardware __popc.

        Zero intermediate tensor allocations — all work done inside one kernel.
        ~100x faster than chunked PyTorch for large systems.
        """
        dim = len(psi)
        n_terms = len(self.pauli_coefficients)

        # Prepare contiguous real/imag arrays via DLPack (zero-copy)
        psi_r = cp.from_dlpack(psi.real.contiguous().detach())
        psi_i = cp.from_dlpack(psi.imag.contiguous().detach())

        result_r = cp.zeros(dim, dtype=cp.float64)
        result_i = cp.zeros(dim, dtype=cp.float64)

        flip_masks_cp = cp.from_dlpack(self._lw_flip_masks.contiguous().detach())
        yz_masks_cp = cp.from_dlpack(self._lw_yz_masks.contiguous().detach())

        # Raw coefficients (NOT pre-multiplied by iny — kernel handles phase separately)
        coeff_r = cp.from_dlpack(self._lw_coeffs.real.contiguous().detach())
        coeff_i = cp.from_dlpack(self._lw_coeffs.imag.contiguous().detach())

        # i^{n_Y} for phase computation inside kernel
        iny_r = cp.from_dlpack(self._lw_i_ny.real.contiguous().detach())
        iny_i = cp.from_dlpack(self._lw_i_ny.imag.contiguous().detach())

        block = 256
        grid = (dim + block - 1) // block

        kernel(
            (grid,),
            (block,),
            (
                psi_r, psi_i, result_r, result_i,
                flip_masks_cp, yz_masks_cp,
                coeff_r, coeff_i,
                iny_r, iny_i,
                np.float64(self.constant_energy), np.float64(0.0),
                np.int32(dim), np.int32(n_terms),
            ),
        )

        # Convert back to torch complex128 (zero-copy via DLPack)
        result = torch.complex(
            torch.from_dlpack(result_r),
            torch.from_dlpack(result_i),
        )
        return result

    def _lanczos_exact_evolution(
        self, psi: torch.Tensor, t: float, krylov_dim: int = 30
    ) -> torch.Tensor:
        """
        Compute e^{-iHt}|ψ⟩ using Lanczos approximation.

        Optimized (Phase 2c):
        - Pre-allocated V matrix (no Python list + torch.stack)
        - alpha/beta kept as GPU tensors (no .item() per iteration)
        - Convergence checked every 5 iterations (reduce GPU→CPU syncs)

        Cost: O(krylov_dim × n_terms × dim) for matvecs + O(krylov_dim³) for exp.
        """
        device = psi.device
        n = len(psi)
        norm_psi = torch.linalg.norm(psi).real

        if norm_psi.item() < 1e-15:
            return psi.clone()

        actual_dim = min(krylov_dim, n)

        # Pre-allocate Lanczos basis matrix on GPU (Phase 2c)
        V = torch.zeros(actual_dim, n, dtype=torch.complex128, device=device)
        alpha = torch.zeros(actual_dim, dtype=torch.float64, device=device)
        beta = torch.zeros(actual_dim, dtype=torch.float64, device=device)

        v = psi / norm_psi
        V[0] = v

        w = self._apply_hamiltonian_matvec(v)
        alpha[0] = torch.vdot(v, w).real
        w = w - alpha[0] * v

        m = 1  # actual Lanczos dimension built so far

        for j in range(1, actual_dim):
            b = torch.linalg.norm(w).real
            if b.item() < 1e-12:
                break
            beta[j] = b
            v_new = w / b
            V[j] = v_new

            w = self._apply_hamiltonian_matvec(v_new)
            alpha[j] = torch.vdot(v_new, w).real
            w = w - alpha[j] * v_new - b * V[j - 1]

            m = j + 1

        # Build tridiagonal T (m × m) from GPU tensors — no CPU round-trip
        T = torch.zeros(m, m, dtype=torch.complex128, device=device)
        T.diagonal().copy_(alpha[:m].to(torch.complex128))
        if m > 1:
            T.diagonal(1).copy_(beta[1:m].to(torch.complex128))
            T.diagonal(-1).copy_(beta[1:m].to(torch.complex128))

        # Compute exp(-i*t*T) on small matrix
        expT = torch.linalg.matrix_exp(-1j * t * T)

        # Project back: result = norm_psi * V^T @ expT[:, 0]
        coeffs = expT[:, 0] * norm_psi
        result = coeffs @ V[:m]  # (n,) — uses pre-allocated V directly

        return result

    def _lanczos_exact_evolution_lowmem(
        self, psi: torch.Tensor, t: float, krylov_dim: int = 30
    ) -> torch.Tensor:
        """
        Low-memory Lanczos: compute e^{-iHt}|ψ⟩ storing only 3 vectors at a time.

        Two-pass algorithm:
        - Pass 1: Lanczos iteration storing only (v_prev, v_curr, w) + tridiagonal T.
          Memory: 3 × n × 16 bytes instead of krylov_dim × n × 16 bytes.
        - Compute expansion coefficients c = expm(-i*t*T)[:, 0].
        - Pass 2: Re-run Lanczos, accumulating result += c[j] * v_j on the fly.

        10x memory reduction vs standard Lanczos (3 vectors vs 30).
        Enables Path C for 24-26 qubit systems on 16 GB VRAM.
        """
        device = psi.device
        n = len(psi)
        norm_psi = torch.linalg.norm(psi).real

        if norm_psi.item() < 1e-15:
            return psi.clone()

        actual_dim = min(krylov_dim, n)

        # --- Pass 1: Build tridiagonal T using only 3 vectors ---
        alpha = torch.zeros(actual_dim, dtype=torch.float64, device=device)
        beta = torch.zeros(actual_dim, dtype=torch.float64, device=device)

        v_curr = psi / norm_psi
        w = self._apply_hamiltonian_matvec(v_curr)
        alpha[0] = torch.vdot(v_curr, w).real
        w = w - alpha[0] * v_curr

        m = 1
        for j in range(1, actual_dim):
            b = torch.linalg.norm(w).real
            if b.item() < 1e-12:
                break
            beta[j] = b
            v_prev = v_curr
            v_curr = w / b

            w = self._apply_hamiltonian_matvec(v_curr)
            alpha[j] = torch.vdot(v_curr, w).real
            w = w - alpha[j] * v_curr - b * v_prev

            m = j + 1

        # Free working vectors from pass 1
        del v_curr, v_prev, w

        # Build tridiagonal T and compute expansion coefficients
        T = torch.zeros(m, m, dtype=torch.complex128, device=device)
        T.diagonal().copy_(alpha[:m].to(torch.complex128))
        if m > 1:
            T.diagonal(1).copy_(beta[1:m].to(torch.complex128))
            T.diagonal(-1).copy_(beta[1:m].to(torch.complex128))

        expT = torch.linalg.matrix_exp(-1j * t * T)
        coeffs = expT[:, 0] * norm_psi

        # --- Pass 2: Re-run Lanczos, accumulating result on the fly ---
        result = torch.zeros(n, dtype=torch.complex128, device=device)

        v_curr = psi / norm_psi
        result += coeffs[0] * v_curr

        w = self._apply_hamiltonian_matvec(v_curr)
        w = w - alpha[0] * v_curr

        for j in range(1, m):
            b = beta[j]
            v_prev = v_curr
            v_curr = w / b
            result += coeffs[j] * v_curr

            w = self._apply_hamiltonian_matvec(v_curr)
            w = w - alpha[j] * v_curr - b * v_prev

        return result

    def _should_use_lowmem_lanczos(self, krylov_dim: int = 30) -> bool:
        """Auto-select low-memory Lanczos when V matrix would use >40% of VRAM."""
        dim = 2 ** self.n_qubits
        v_matrix_bytes = krylov_dim * dim * 16  # complex128

        if torch.cuda.is_available():
            free_vram = torch.cuda.mem_get_info()[0]
            return v_matrix_bytes > free_vram * 0.4
        else:
            # On CPU, use lowmem for >4 GB V matrix
            return v_matrix_bytes > 4 * 1024**3

    def _sample_exact(self, krylov_power: int) -> Dict[str, int]:
        """
        Exact time evolution in full 2^n Hilbert space via Lanczos.

        Phase 1 optimization: caches evolved state |ψₖ⟩ between Krylov steps.
        For k, reuses cached |ψₖ₋₁⟩ and applies one Lanczos evolution instead
        of k evolutions from scratch. Reduces total Lanczos calls from
        0+1+...+14 = 105 to just 15 (one per step).
        """
        self._precompute_pauli_masks_lightweight()

        # Auto-select Lanczos variant based on VRAM
        use_lowmem = self._should_use_lowmem_lanczos()

        if krylov_power == 0 and not hasattr(self, "_exact_info_printed"):
            n_terms = len(self.pauli_coefficients)
            dim = 2**self.n_qubits
            T = self.config.total_evolution_time
            mode = "low-memory 2-pass" if use_lowmem else "standard"
            print(
                f"  Exact Lanczos evolution ({n_terms} Pauli terms, dim={dim:,}, "
                f"T={T:.6f} per step, state-cached, {mode})"
            )
            self._exact_info_printed = True

        T = self.config.total_evolution_time
        evolve_fn = self._lanczos_exact_evolution_lowmem if use_lowmem else self._lanczos_exact_evolution

        # Phase 1: Use cached evolved state from previous Krylov step
        if krylov_power == 0:
            psi = self._get_initial_state_gpu()
        elif krylov_power - 1 in self._cached_exact_states:
            # Reuse |ψₖ₋₁⟩ and apply one evolution: |ψₖ⟩ = e^{-iHT}|ψₖ₋₁⟩
            psi = self._cached_exact_states[krylov_power - 1].clone()
            psi = evolve_fn(psi, T)
            psi = psi / torch.linalg.norm(psi)
        else:
            # Fallback: evolve from scratch (shouldn't happen with sequential k)
            psi = self._get_initial_state_gpu()
            for _ in range(krylov_power):
                psi = evolve_fn(psi, T)
                psi = psi / torch.linalg.norm(psi)

        # Cache for next step
        self._cached_exact_states[krylov_power] = psi.clone()

        # Evict previous state to free VRAM (only state k-1 is ever needed)
        if krylov_power > 0 and (krylov_power - 1) in self._cached_exact_states:
            del self._cached_exact_states[krylov_power - 1]

        # Sample from |ψ|²
        return self._sample_from_state(psi, krylov_power)

    def _sample_from_state(self, psi: torch.Tensor, krylov_power: int) -> Dict[str, int]:
        """Shared sampling logic: |ψ|² → multinomial → bitstring counts."""
        probs = torch.abs(psi) ** 2
        probs = probs / probs.sum()

        # Move to CPU for multinomial to avoid CUDA device-side asserts
        # on edge cases (small systems, near-zero probs in float32)
        probs_cpu = probs.cpu().float()
        probs_cpu = probs_cpu.clamp(min=0.0)
        psum = probs_cpu.sum()
        if psum < 1e-30:
            probs_cpu = torch.ones_like(probs_cpu)
            psum = probs_cpu.sum()
        probs_cpu = probs_cpu / psum

        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.config.seed + krylov_power + 1000)
        indices = torch.multinomial(probs_cpu, self.config.shots, replacement=True)

        unique, counts = torch.unique(indices, return_counts=True)
        unique_cpu = unique.cpu().numpy()
        counts_cpu = counts.cpu().numpy()

        results = {}
        for idx, count in zip(unique_cpu, counts_cpu):
            bitstring = format(int(idx), f"0{self.n_qubits}b")
            results[bitstring] = int(count)

        return results

    # ------------------------------------------------------------------
    # Core SKQD algorithm
    # ------------------------------------------------------------------

    def generate_krylov_samples(
        self, progress: bool = True
    ) -> Tuple[List[Dict[str, int]], List[Dict[str, int]]]:
        """
        Generate samples from each Krylov state and build cumulative basis.

        Returns:
            (all_samples, cumulative_results) following NVIDIA tutorial structure
        """
        cfg = self.config
        max_k = cfg.max_krylov_dim

        # Choose sampling backend based on config
        requested = cfg.backend
        if requested == "cudaq":
            if not CUDAQ_AVAILABLE:
                raise RuntimeError("backend='cudaq' requested but cudaq is not installed")
            sample_fn = self._sample_cudaq
            backend = "CUDA-Q"
        elif requested == "classical":
            sample_fn = self._sample_classical_trotterized
            backend = f"Classical Trotterized (GPU: {self._device.type})"
        elif requested == "exact":
            # Prefer CuPy Trotter for speed; fall back to Lanczos on CPU
            if CUPY_AVAILABLE and self._device.type == "cuda":
                sample_fn = self._sample_classical_trotterized
                backend = f"CuPy Trotterized (GPU, fused-diagonal)"
            else:
                sample_fn = self._sample_exact
                backend = f"Exact Lanczos (CPU)"
        elif requested == "lanczos":
            sample_fn = self._sample_exact
            backend = f"Exact Lanczos ({self._device.type})"
        else:  # "auto"
            if CUDAQ_AVAILABLE:
                sample_fn = self._sample_cudaq
                backend = "CUDA-Q"
            else:
                sample_fn = self._sample_classical_trotterized
                backend = f"Classical Trotterized (GPU: {self._device.type})"

        print(f"Quantum SKQD backend: {backend}")
        print(f"  Krylov dim: {max_k}, Trotter-{cfg.trotter_order}, "
              f"{cfg.num_trotter_steps} steps, dt: {self.dt:.6f}, "
              f"T/step: {cfg.total_evolution_time:.6f}, shots: {cfg.shots:,}")

        all_samples = []
        cumulative: Dict[str, int] = {}
        cumulative_results = []

        # Pre-compute essential configs (HF + singles + doubles) for injection
        essential_bitstrings = self._get_essential_bitstrings()
        if essential_bitstrings:
            print(f"  Essential config injection: {len(essential_bitstrings)} configs "
                  f"(HF + singles + doubles)")

        for k in range(max_k):
            if progress:
                print(f"  Generating Krylov state U^{k}...")
            samples = sample_fn(k)
            all_samples.append(samples)

            # Accumulate (union of bitstrings across Krylov powers)
            for bs, count in samples.items():
                cumulative[bs] = cumulative.get(bs, 0) + count

            # Inject essential configs to ensure ground-state coverage
            for bs in essential_bitstrings:
                if bs not in cumulative:
                    cumulative[bs] = 1

            cumulative_results.append(dict(cumulative))

        self._all_samples = all_samples
        self._cumulative_results = cumulative_results

        return all_samples, cumulative_results

    def _get_essential_bitstrings(self) -> List[str]:
        """Generate HF + single + double excitation bitstrings for config injection."""
        if self.hamiltonian is None or not hasattr(self.hamiltonian, "n_orbitals"):
            return []

        from itertools import combinations

        H = self.hamiltonian
        n_orb = H.n_orbitals
        n_alpha = H.n_alpha
        n_beta = H.n_beta
        n_qubits = self.n_qubits

        hf = H.get_hf_state().cpu().numpy()
        def _cfg_to_bs(c):
            return "".join(str(int(v)) for v in c)

        bitstrings = [_cfg_to_bs(hf)]

        occ_alpha = list(range(n_alpha))
        occ_beta = list(range(n_beta))
        virt_alpha = list(range(n_alpha, n_orb))
        virt_beta = list(range(n_beta, n_orb))

        # Single excitations
        for i in occ_alpha:
            for a in virt_alpha:
                cfg = hf.copy()
                cfg[i] = 0
                cfg[a] = 1
                bitstrings.append(_cfg_to_bs(cfg))
        for i in occ_beta:
            for a in virt_beta:
                cfg = hf.copy()
                cfg[i + n_orb] = 0
                cfg[a + n_orb] = 1
                bitstrings.append(_cfg_to_bs(cfg))

        # Double excitations (capped at 5000)

        max_doubles = 5000
        count = 0
        for i, j in combinations(occ_alpha, 2):
            for a, b_orb in combinations(virt_alpha, 2):
                if count >= max_doubles:
                    break
                cfg = hf.copy()
                cfg[i] = 0
                cfg[j] = 0
                cfg[a] = 1
                cfg[b_orb] = 1
                bitstrings.append(_cfg_to_bs(cfg))
                count += 1
        for i, j in combinations(occ_beta, 2):
            for a, b_orb in combinations(virt_beta, 2):
                if count >= max_doubles:
                    break
                cfg = hf.copy()
                cfg[i + n_orb] = 0
                cfg[j + n_orb] = 0
                cfg[a + n_orb] = 1
                cfg[b_orb + n_orb] = 1
                bitstrings.append(_cfg_to_bs(cfg))
                count += 1
        for i in occ_alpha:
            for j in occ_beta:
                for a in virt_alpha:
                    for b_orb in virt_beta:
                        if count >= max_doubles:
                            break
                        cfg = hf.copy()
                        cfg[i] = 0
                        cfg[j + n_orb] = 0
                        cfg[a] = 1
                        cfg[b_orb + n_orb] = 1
                        bitstrings.append(_cfg_to_bs(cfg))
                        count += 1

        # Deduplicate
        return list(set(bitstrings))

    def _basis_from_samples(self, sample_dict: Dict[str, int]) -> torch.Tensor:
        """Convert sample dictionary to basis state tensor on GPU (vectorized)."""
        bitstrings = list(sample_dict.keys())
        n = len(bitstrings)
        nq = self.n_qubits

        # Vectorized: convert all bitstrings to byte array at once
        flat = np.frombuffer(
            "".join(bitstrings).encode("ascii"), dtype=np.uint8
        ) - ord("0")
        basis = torch.from_numpy(flat.reshape(n, nq).astype(np.int64)).to(self._device)
        return basis

    def compute_energies(
        self, progress: bool = True
    ) -> List[float]:
        """
        Compute ground state energy at each Krylov dimension.

        Uses the projected Hamiltonian approach:
        1. Extract basis states from cumulative samples
        2. Build H_eff[i,j] = <s_i|H|s_j> in the basis
        3. Diagonalize via GPU eigensolver

        Returns:
            List of ground state energy estimates, one per Krylov dimension (k=1..max_k-1)
        """
        if not hasattr(self, "_cumulative_results"):
            raise RuntimeError("Call generate_krylov_samples() first")

        max_k = self.config.max_krylov_dim
        energies = []

        for k in range(1, max_k):
            cum_samples = self._cumulative_results[k]
            basis = self._basis_from_samples(cum_samples)
            subspace_dim = len(cum_samples)

            if progress:
                print(f"  k={k+1}: {subspace_dim} basis states, ", end="")

            # Build projected Hamiltonian
            if self.hamiltonian is not None:
                E0 = self._diagonalize_slater_condon(basis)
            else:
                E0 = self._diagonalize_pauli_gpu(basis)

            energies.append(E0)
            if progress:
                print(f"E = {E0:.8f} Ha")

        self.energies = energies
        return energies

    def _diagonalize_slater_condon(self, basis: torch.Tensor) -> float:
        """Diagonalize using Hamiltonian's matrix_elements (Slater-Condon rules)."""
        device = self.hamiltonian.device if hasattr(self.hamiltonian, "device") else self._device
        basis = basis.to(device)

        H_proj = self.hamiltonian.matrix_elements(basis, basis)

        # Symmetrize and diagonalize on GPU
        H_proj = H_proj.real.double()
        H_proj = 0.5 * (H_proj + H_proj.T)

        n = H_proj.shape[0]
        if n <= 1:
            return float(H_proj[0, 0].cpu()) if n == 1 else float("inf")

        eigenvalues = torch.linalg.eigh(H_proj)[0]
        return float(eigenvalues[0].cpu())

    def _diagonalize_pauli_gpu(self, basis: torch.Tensor) -> float:
        """
        Diagonalize using fully-vectorized Pauli string evaluation on GPU.

        Follows the NVIDIA tutorial's vectorized_projected_hamiltonian algorithm:
        1. Encode Pauli ops as numeric array (I=0, X=1, Y=2, Z=3)
        2. Broadcast over all (basis_state, pauli_term) pairs simultaneously
        3. Compute transformed states and phase factors in parallel
        4. Use searchsorted for O(n log n) basis matching (no Python loops)
        5. Build sparse projected H, then eigsh

        This is fully vectorized on GPU — no Python loops over basis states.
        """
        n = len(basis)
        if n == 0:
            return float("inf")

        device = self._device
        basis = basis.to(device)
        n_qubits = self.n_qubits
        n_terms = len(self.pauli_coefficients)

        if n_terms == 0:
            return self.constant_energy

        # Convert Pauli strings to numeric: I=0, X=1, Y=2, Z=3
        pauli_to_int = {"I": 0, "X": 1, "Y": 2, "Z": 3}
        pauli_ops = torch.tensor(
            [[pauli_to_int[p] for p in pw] for pw in self.pauli_words],
            dtype=torch.int8, device=device,
        )  # (n_terms, n_qubits)

        coefficients = torch.tensor(
            self.pauli_coefficients, dtype=torch.complex128, device=device
        )  # (n_terms,)

        # Broadcast: states (n, 1, nq) x pauli_ops (1, n_terms, nq)
        states_exp = basis[:, None, :].to(torch.int8)  # (n, 1, nq)
        pauli_exp = pauli_ops[None, :, :]               # (1, n_terms, nq)

        # Step 1: Compute transformed states (X and Y flip bits)
        flip_mask = (pauli_exp == 1) | (pauli_exp == 2)
        transformed = torch.where(flip_mask, 1 - states_exp, states_exp)
        # (n, n_terms, nq)

        # Step 2: Compute phase factors
        y_mask = (pauli_exp == 2)
        z_mask = (pauli_exp == 3)

        n_y0 = (y_mask & (states_exp == 0)).sum(dim=2, dtype=torch.int32)  # (n, n_terms)
        n_y1 = (y_mask & (states_exp == 1)).sum(dim=2, dtype=torch.int32)
        n_z1 = (z_mask & (states_exp == 1)).sum(dim=2, dtype=torch.int32)

        phase_index = (n_y0 - n_y1 + 2 * n_z1) % 4  # (n, n_terms)
        phase_lookup = torch.tensor(
            [1.0 + 0j, 0.0 + 1j, -1.0 + 0j, 0.0 - 1j],
            dtype=torch.complex128, device=device,
        )
        phase_factors = phase_lookup[phase_index.long()]  # (n, n_terms)

        # Step 3: H elements = coeff * phase
        h_elements = coefficients[None, :] * phase_factors  # (n, n_terms)

        # Step 4: Convert states to integers for searchsorted matching
        powers = (1 << torch.arange(n_qubits - 1, -1, -1, device=device, dtype=torch.int64))
        basis_ints = (basis.to(torch.int64) * powers).sum(dim=1)  # (n,)
        transformed_ints = (transformed.to(torch.int64) * powers).sum(dim=2)  # (n, n_terms)

        # Step 5: Sorted search for O(n log n) matching
        sorted_indices = torch.argsort(basis_ints)
        sorted_basis_ints = basis_ints[sorted_indices]

        transformed_flat = transformed_ints.reshape(-1)  # (n * n_terms,)
        search_pos = torch.searchsorted(sorted_basis_ints, transformed_flat)

        in_bounds = search_pos < n
        search_pos_clipped = torch.minimum(search_pos, torch.tensor(n - 1, device=device))
        actually_found = in_bounds & (sorted_basis_ints[search_pos_clipped] == transformed_flat)

        # Map back to original indices
        row_indices = sorted_indices[search_pos_clipped]
        col_indices = torch.arange(n, device=device).repeat_interleave(n_terms)

        # Filter valid entries
        valid_rows = row_indices[actually_found]
        valid_cols = col_indices[actually_found]
        valid_elements = h_elements.reshape(-1)[actually_found]

        # Step 6: Accumulate into dense matrix
        H_eff = torch.zeros((n, n), dtype=torch.complex128, device=device)
        H_eff.index_put_((valid_rows, valid_cols), valid_elements, accumulate=True)

        # Add constant energy
        H_eff.add_(torch.eye(n, dtype=torch.complex128, device=device), alpha=self.constant_energy)

        # Symmetrize
        H_eff = 0.5 * (H_eff + H_eff.conj().T)

        if n <= 1:
            return float(H_eff[0, 0].real.cpu())

        eigenvalues = torch.linalg.eigh(H_eff.real.double())[0]
        return float(eigenvalues[0].cpu())

    # ------------------------------------------------------------------
    # Full run
    # ------------------------------------------------------------------

    def run(self, progress: bool = True) -> Dict[str, Any]:
        """
        Run full quantum circuit SKQD pipeline.

        Returns:
            Dictionary with energies, basis sizes, and diagnostics
        """
        # Step 1: Generate Krylov samples
        all_samples, cumulative = self.generate_krylov_samples(progress=progress)

        # Step 2: Compute energies at each Krylov dimension
        energies = self.compute_energies(progress=progress)

        # Results
        basis_sizes = [len(cumulative[k]) for k in range(1, self.config.max_krylov_dim)]

        results = {
            "energies": energies,
            "krylov_dims": list(range(2, self.config.max_krylov_dim + 1)),
            "basis_sizes": basis_sizes,
            "final_energy": energies[-1] if energies else float("inf"),
            "best_energy": min(energies) if energies else float("inf"),
            "backend": "CUDA-Q" if CUDAQ_AVAILABLE else "Classical Trotterized",
            "device": str(self._device),
            "config": {
                "max_krylov_dim": self.config.max_krylov_dim,
                "num_trotter_steps": self.config.num_trotter_steps,
                "trotter_order": self.config.trotter_order,
                "total_evolution_time": self.config.total_evolution_time,
                "shots": self.config.shots,
                "initial_state": self.config.initial_state,
            },
            "constant_energy": self.constant_energy,
            "n_pauli_terms": len(self.pauli_coefficients),
        }

        print(f"\nQuantum SKQD Results:")
        print(f"  Best energy: {results['best_energy']:.8f} Ha")
        print(f"  Final energy (k={self.config.max_krylov_dim}): {results['final_energy']:.8f} Ha")
        print(f"  Final basis size: {basis_sizes[-1] if basis_sizes else 0}")
        print(f"  Device: {self._device}")

        return results

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_molecular_hamiltonian(
        cls,
        hamiltonian,
        config: Optional[QuantumSKQDConfig] = None,
    ) -> "QuantumCircuitSKQD":
        """
        Create QuantumCircuitSKQD from a MolecularHamiltonian.

        Performs Jordan-Wigner transformation to get Pauli representation.
        """
        try:
            from ..hamiltonians.pauli_mapping import molecular_hamiltonian_to_pauli
        except ImportError:
            from hamiltonians.pauli_mapping import molecular_hamiltonian_to_pauli

        cfg = config or QuantumSKQDConfig()
        cfg.initial_state = "hf"

        h1e = hamiltonian.h1e.cpu().numpy()
        h2e = hamiltonian.h2e.cpu().numpy()
        n_orb = hamiltonian.n_orbitals

        print(f"Jordan-Wigner transformation: {n_orb} orbitals -> {2 * n_orb} qubits...")
        coefficients, pauli_words, constant = molecular_hamiltonian_to_pauli(
            h1e, h2e, hamiltonian.nuclear_repulsion, n_orb
        )
        print(f"  {len(coefficients)} Pauli terms + constant = {constant:.8f}")

        return cls(
            pauli_coefficients=coefficients,
            pauli_words=pauli_words,
            n_qubits=2 * n_orb,
            config=cfg,
            constant_energy=constant,
            hamiltonian=hamiltonian,
        )

    @classmethod
    def from_heisenberg(
        cls,
        n_spins: int,
        Jx: float = 1.0,
        Jy: float = 1.0,
        Jz: float = 1.0,
        config: Optional[QuantumSKQDConfig] = None,
    ) -> "QuantumCircuitSKQD":
        """
        Create QuantumCircuitSKQD for a Heisenberg spin chain.

        Matches the NVIDIA CUDA-Q tutorial setup exactly.
        """
        try:
            from ..hamiltonians.pauli_mapping import heisenberg_hamiltonian_pauli
        except ImportError:
            from hamiltonians.pauli_mapping import heisenberg_hamiltonian_pauli

        cfg = config or QuantumSKQDConfig()
        cfg.initial_state = "neel"

        hx = np.ones(n_spins)
        hy = np.ones(n_spins)
        hz = np.ones(n_spins)

        coefficients, pauli_words, constant = heisenberg_hamiltonian_pauli(
            n_spins, Jx, Jy, Jz, hx, hy, hz
        )

        return cls(
            pauli_coefficients=coefficients,
            pauli_words=pauli_words,
            n_qubits=n_spins,
            config=cfg,
            constant_energy=constant,
        )
