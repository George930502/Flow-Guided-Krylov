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

    # Backend selection: "auto" (CUDA-Q if available), "cudaq", "classical"
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
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        if abs(sin_t) < 1e-15:
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

        Kernel compilation is expensive; defining inside a per-call method
        forces recompilation at each Krylov step. Cache as instance attributes.

        NOTE: Angles are pre-computed outside the kernel (in _init_cudaq) and
        passed as list[float]. CUDA-Q's JIT miscompiles `coeffs[i] * dt`
        arithmetic inside kernels, so we pass final angles directly.
        """
        if self._cudaq_kernels_built:
            return

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

    def _sample_cudaq(self, krylov_power: int) -> Dict[str, int]:
        """
        Sample from Krylov state U^k|psi_0> using CUDA-Q quantum circuit.

        Follows the NVIDIA tutorial's quantum_krylov_evolution_circuit exactly.
        Target and kernels are initialized once; only the random seed
        changes per Krylov step.

        Uses pre-computed angles (from _init_cudaq) to avoid CUDA-Q JIT bug
        with in-kernel list[float] arithmetic.
        """
        self._init_cudaq()
        self._build_cudaq_kernels()

        cudaq.set_random_seed(self.config.seed + krylov_power)

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

        Faithfully simulates the quantum circuit's Trotterized evolution
        using O(dim) per-term Pauli action instead of building dense unitary.

        Total cost per Krylov step: O(k * num_trotter_steps * n_terms * dim)
        vs previous: O(dim²) per matmul with precomputed unitary.

        For LiH (dim=4096, 630 terms, 8 steps): 630*8*4096 = 20M ops per k
        vs previous: 4096²*630 kron build = hanging.
        """
        # Precompute Pauli actions (flip masks + phase tables)
        self._precompute_pauli_actions()

        if krylov_power == 0 and not hasattr(self, "_trotter_info_printed"):
            n_terms = len(self.pauli_coefficients)
            dim = 2 ** self.n_qubits
            order = self.config.trotter_order
            print(f"  State-vector Trotter-{order} ({n_terms} terms, dim={dim}, "
                  f"{self.config.num_trotter_steps} steps/evolution)")
            self._trotter_info_printed = True

        # Get initial state on GPU
        psi = self._get_initial_state_gpu()

        # Apply U^k via k Trotter steps directly to state vector
        for _ in range(krylov_power):
            psi = self._apply_trotter_step(psi)

        # Normalize
        psi = psi / torch.linalg.norm(psi)

        # Sample from |psi|^2 on GPU
        probs = torch.abs(psi) ** 2
        probs = probs / probs.sum()

        gen = torch.Generator(device=self._device)
        gen.manual_seed(self.config.seed + krylov_power + 1000)
        indices = torch.multinomial(probs.float(), self.config.shots, replacement=True)

        # Count unique indices (on CPU — small data)
        unique, counts = torch.unique(indices, return_counts=True)
        unique_cpu = unique.cpu().numpy()
        counts_cpu = counts.cpu().numpy()

        results = {}
        for idx, count in zip(unique_cpu, counts_cpu):
            bitstring = format(int(idx), f"0{self.n_qubits}b")
            results[bitstring] = int(count)

        return results

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
        Compute H|ψ⟩ = E_const|ψ⟩ + Σ_k c_k P_k|ψ⟩ using lightweight masks.

        Processes Pauli terms in chunks for GPU efficiency without materializing
        O(n_terms × dim) intermediate tensors.

        Phase computation uses bit parity: for Pauli string P_k acting on |x⟩,
            phase(x) = i^{n_Y_k} · (-1)^{popcount(x & yz_mask_k)}
        where yz_mask has 1s at positions with Y or Z operators.
        """
        self._precompute_pauli_masks_lightweight()

        dim = len(psi)
        device = psi.device
        result = self.constant_energy * psi.clone()

        indices = torch.arange(dim, device=device, dtype=torch.int64)
        n_terms = len(self.pauli_coefficients)

        # Adaptive chunk size to manage GPU memory:
        # Each chunk uses O(chunk_size × dim × 16) bytes for complex128 intermediates
        if dim <= 100_000:
            chunk_size = 64
        elif dim <= 500_000:
            chunk_size = 16
        else:
            chunk_size = 8

        for start in range(0, n_terms, chunk_size):
            end = min(start + chunk_size, n_terms)

            fmasks = self._lw_flip_masks[start:end]  # (chunk,)
            yzmasks = self._lw_yz_masks[start:end]  # (chunk,)
            iny = self._lw_i_ny[start:end]  # (chunk,) complex128
            coeffs = self._lw_coeffs[start:end]  # (chunk,) complex128

            # Target indices: (chunk, dim) = indices ^ flip_mask per term
            targets = indices.unsqueeze(0) ^ fmasks.unsqueeze(1)

            # Gather psi values at target positions
            psi_gathered = psi[targets]  # (chunk, dim) complex128

            # Compute parity of popcount(indices & yz_mask) via XOR fold
            masked = indices.unsqueeze(0) & yzmasks.unsqueeze(1)  # (chunk, dim)
            masked = masked ^ (masked >> 32)
            masked = masked ^ (masked >> 16)
            masked = masked ^ (masked >> 8)
            masked = masked ^ (masked >> 4)
            masked = masked ^ (masked >> 2)
            masked = masked ^ (masked >> 1)
            parity = (masked & 1).to(torch.float64)  # (chunk, dim)
            sign = 1.0 - 2.0 * parity  # +1 or -1

            # Phase = i^{n_Y} * sign; contribution = coeff * phase * psi_flipped
            phase = iny[:, None] * sign  # (chunk, dim) complex128
            contrib = coeffs[:, None] * phase * psi_gathered  # (chunk, dim)

            result += contrib.sum(dim=0)

        return result

    def _lanczos_exact_evolution(
        self, psi: torch.Tensor, t: float, krylov_dim: int = 30
    ) -> torch.Tensor:
        """
        Compute e^{-iHt}|ψ⟩ using Lanczos approximation.

        Builds a Krylov basis {v, Hv, H²v, ...} of dimension krylov_dim,
        projects H onto this basis to get tridiagonal T, computes exp(-itT)
        on the small matrix, and projects back.

        Cost: O(krylov_dim × n_terms × dim) for matvecs + O(krylov_dim³) for exp.
        """
        device = psi.device
        n = len(psi)
        norm_psi = torch.linalg.norm(psi).real.item()

        if norm_psi < 1e-15:
            return psi.clone()

        actual_dim = min(krylov_dim, n)

        # Lanczos iteration
        V: List[torch.Tensor] = []
        alpha_list: List[float] = []
        beta_list: List[float] = []

        v = psi / norm_psi
        V.append(v)

        w = self._apply_hamiltonian_matvec(v)
        a = torch.vdot(v, w).real.item()
        alpha_list.append(a)
        w = w - a * v

        for j in range(1, actual_dim):
            b = torch.linalg.norm(w).real.item()
            if b < 1e-12:
                break
            beta_list.append(b)
            v_new = w / b
            V.append(v_new)

            w = self._apply_hamiltonian_matvec(v_new)
            a = torch.vdot(v_new, w).real.item()
            alpha_list.append(a)
            w = w - a * v_new - b * V[-2]

        m = len(alpha_list)

        # Build tridiagonal T (m × m) — vectorized
        T = torch.zeros(m, m, dtype=torch.complex128, device=device)
        T.diagonal().copy_(torch.tensor(alpha_list, dtype=torch.complex128, device=device))
        if beta_list:
            beta_t = torch.tensor(beta_list, dtype=torch.complex128, device=device)
            T.diagonal(1).copy_(beta_t)
            T.diagonal(-1).copy_(beta_t)

        # Compute exp(-i*t*T) on small matrix
        expT = torch.linalg.matrix_exp(-1j * t * T)

        # Project back: result = norm_psi * Σ_j expT[j,0] * V[j]
        coeffs = expT[:, 0] * norm_psi

        V_matrix = torch.stack(V[:m])  # (m, n)
        result = coeffs @ V_matrix     # (n,)

        return result

    def _sample_exact(self, krylov_power: int) -> Dict[str, int]:
        """
        Exact time evolution in full 2^n Hilbert space via Lanczos.

        Structurally identical to _sample_classical_trotterized but replaces
        Trotter decomposition with Lanczos-based exact e^{-iHt}. Uses
        lightweight Pauli masks (no O(n_terms × dim) phase tables).

        Shares with _sample_classical_trotterized:
            - Same initial state (_get_initial_state_gpu)
            - Same RNG seeding (seed + k + 1000)
            - Same sampling (torch.multinomial on |ψ|²)
        Only difference: exact evolution vs Trotter approximation.
        """
        self._precompute_pauli_masks_lightweight()

        if krylov_power == 0 and not hasattr(self, "_exact_info_printed"):
            n_terms = len(self.pauli_coefficients)
            dim = 2**self.n_qubits
            T = self.config.total_evolution_time
            print(
                f"  Exact Lanczos evolution ({n_terms} Pauli terms, dim={dim:,}, "
                f"T={T:.6f} per step)"
            )
            self._exact_info_printed = True

        # Same initial state as Trotter path
        psi = self._get_initial_state_gpu()

        # Apply exact e^{-iHT} k times
        T = self.config.total_evolution_time
        for _ in range(krylov_power):
            psi = self._lanczos_exact_evolution(psi, T)
            psi = psi / torch.linalg.norm(psi)

        # Sample from |ψ|² — SAME mechanism as _sample_classical_trotterized
        probs = torch.abs(psi) ** 2
        probs = probs / probs.sum()

        gen = torch.Generator(device=self._device)
        gen.manual_seed(self.config.seed + krylov_power + 1000)
        indices = torch.multinomial(probs.float(), self.config.shots, replacement=True)

        # Count unique indices (on CPU — small data)
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
            sample_fn = self._sample_exact
            backend = f"Exact Lanczos (GPU: {self._device.type})"
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

        for k in range(max_k):
            if progress:
                print(f"  Generating Krylov state U^{k}...")
            samples = sample_fn(k)
            all_samples.append(samples)

            # Accumulate (union of bitstrings across Krylov powers)
            for bs, count in samples.items():
                cumulative[bs] = cumulative.get(bs, 0) + count
            cumulative_results.append(dict(cumulative))

        self._all_samples = all_samples
        self._cumulative_results = cumulative_results

        return all_samples, cumulative_results

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
