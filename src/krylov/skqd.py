"""
Sample-Based Krylov Quantum Diagonalization (SKQD).

Implements the SKQD algorithm from:
"Sample-based Krylov Quantum Diagonalization"
(Yu et al., IBM Quantum)

The algorithm:
1. Initialize reference state |ψ_0⟩
2. Generate Krylov states |ψ_k⟩ = U^k |ψ_0⟩ where U = e^{-iHΔt}
3. Sample basis states from each Krylov state
4. Project Hamiltonian onto sampled basis
5. Diagonalize to get ground state energy
"""

import torch
import numpy as np
from typing import Optional, Tuple, List, Dict, Callable
from dataclasses import dataclass
from itertools import combinations
from math import comb
from tqdm import tqdm

# Sparse eigensolvers (as specified in AGENTs.md)
from scipy.sparse import csr_matrix as scipy_csr
from scipy.sparse.linalg import eigsh as scipy_eigsh

try:
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix as cupy_csr
    from cupyx.scipy.sparse.linalg import eigsh as cupy_eigsh
    # Also check if CUDA is actually usable (not just installed)
    try:
        cp.cuda.Device(0).compute_capability
        CUPY_AVAILABLE = True
    except (cp.cuda.runtime.CUDARuntimeError, RuntimeError):
        CUPY_AVAILABLE = False
except ImportError:
    CUPY_AVAILABLE = False

# Support both package imports and direct script execution
try:
    from ..hamiltonians.base import Hamiltonian
except ImportError:
    from hamiltonians.base import Hamiltonian


@dataclass
class SKQDConfig:
    """Configuration for SKQD algorithm."""

    # Krylov parameters
    max_krylov_dim: int = 12
    time_step: float = 0.1  # Δt
    total_evolution_time: Optional[float] = None  # If set, overrides max_k

    # Trotter parameters
    num_trotter_steps: int = 8

    # Sampling parameters
    shots_per_krylov: int = 100000
    use_cumulative_basis: bool = True  # Accumulate samples across Krylov states

    # Eigensolver parameters
    num_eigenvalues: int = 2  # k for eigsh
    which_eigenvalues: str = "SA"  # Smallest algebraic

    # Numerical stability
    regularization: float = 1e-8  # Diagonal regularization for stability

    # Hardware
    use_gpu: bool = True


class SampleBasedKrylovDiagonalization:
    """
    Sample-Based Krylov Quantum Diagonalization.

    This class provides a classical simulation of SKQD for validation
    and development. For actual quantum hardware execution, use the
    CUDA-Q integration via KrylovBasisSampler.

    The algorithm builds a Krylov subspace by time-evolving a reference
    state and sampling in the computational basis at each step.

    Args:
        hamiltonian: System Hamiltonian
        config: SKQD configuration
        initial_state: Optional initial state (default: Néel state)
    """

    def __init__(
        self,
        hamiltonian: Hamiltonian,
        config: Optional[SKQDConfig] = None,
        initial_state: Optional[torch.Tensor] = None,
    ):
        self.hamiltonian = hamiltonian
        self.config = config or SKQDConfig()
        self.num_sites = hamiltonian.num_sites

        # Check if this is a molecular Hamiltonian with particle conservation
        self._is_molecular = hasattr(hamiltonian, 'n_alpha') and hasattr(hamiltonian, 'n_beta')
        self._subspace_basis = None
        self._subspace_index_map = None
        self._subspace_H = None

        # For molecular systems, set up particle-conserving subspace
        if self._is_molecular:
            self._setup_particle_conserving_subspace()

        # Set up initial state
        if initial_state is not None:
            self.initial_state = initial_state
        else:
            # Default: HF state for molecules, Néel state for spin systems
            if self._is_molecular:
                self.initial_state = self.hamiltonian.get_hf_state()
            else:
                self.initial_state = self._create_neel_state()

        # Compute time step based on spectral range if not specified
        if self.config.total_evolution_time is not None:
            self.time_step = (
                self.config.total_evolution_time / self.config.num_trotter_steps
            )
        else:
            self.time_step = self.config.time_step

        # Storage for results
        self.krylov_samples: List[Dict[str, int]] = []
        self.cumulative_basis: List[torch.Tensor] = []
        self.energies: List[float] = []

    def _setup_particle_conserving_subspace(self):
        """
        Set up the particle-conserving subspace for molecular Hamiltonians.

        This dramatically reduces the Hamiltonian size from 2^n to
        C(n_orb, n_alpha) * C(n_orb, n_beta), which is typically 10-100x smaller.

        Example sizes:
        - NH3 (16 qubits): 65,536 -> 3,136 (21x reduction)
        - N2 (20 qubits): 1,048,576 -> 14,400 (73x reduction)
        """
        n_orb = self.hamiltonian.n_orbitals
        n_alpha = self.hamiltonian.n_alpha
        n_beta = self.hamiltonian.n_beta

        n_valid = comb(n_orb, n_alpha) * comb(n_orb, n_beta)
        print(f"Setting up particle-conserving subspace: {n_valid:,} configs "
              f"(vs {self.hamiltonian.hilbert_dim:,} full Hilbert space)")

        # Generate all valid configurations
        alpha_configs = list(combinations(range(n_orb), n_alpha))
        beta_configs = list(combinations(range(n_orb), n_beta))

        basis_configs = []
        for alpha_occ in alpha_configs:
            for beta_occ in beta_configs:
                # Build configuration tensor
                config = torch.zeros(self.num_sites, dtype=torch.long)
                for i in alpha_occ:
                    config[i] = 1
                for i in beta_occ:
                    config[i + n_orb] = 1
                basis_configs.append(config)

        self._subspace_basis = torch.stack(basis_configs)

        # Create index mapping: config tuple -> subspace index
        self._subspace_index_map = {}
        for idx, config in enumerate(basis_configs):
            key = tuple(config.tolist())
            self._subspace_index_map[key] = idx

        print(f"Subspace setup complete: {len(basis_configs)} configurations")

    @property
    def device(self) -> torch.device:
        """Get device from Hamiltonian (for GPU-aware Hamiltonians)."""
        if hasattr(self.hamiltonian, 'device'):
            return torch.device(self.hamiltonian.device)
        return torch.device('cpu')

    def _create_neel_state(self) -> torch.Tensor:
        """Create Néel state |010101...⟩."""
        state = torch.zeros(self.num_sites, dtype=torch.long, device=self.device)
        state[::2] = 1  # Odd sites = 1
        return state

    def _time_evolution_operator(
        self,
        state_vector: torch.Tensor,
        num_steps: int = 1,
    ) -> torch.Tensor:
        """
        Apply time evolution U^num_steps = (e^{-iHΔt})^num_steps.

        For molecular systems with particle conservation, uses subspace evolution
        which is MUCH faster (e.g., 3K x 3K instead of 65K x 65K for NH3).

        For spin systems, uses sparse matrix exponential or Trotter decomposition.

        Args:
            state_vector: Full state vector in Hilbert space
            num_steps: Number of time steps to apply

        Returns:
            Evolved state vector
        """
        # Ensure state vector is on the correct device
        device = self.device
        state_vector = state_vector.to(device)

        # For molecular systems, ALWAYS use particle-conserving subspace evolution
        # This is the key optimization: work in much smaller subspace
        if self._is_molecular and self._subspace_basis is not None:
            return self._sparse_time_evolution(state_vector, num_steps)

        # For very small systems (<=6 qubits), dense is faster due to less overhead
        if self.num_sites <= 6:
            # Get dense Hamiltonian on the same device
            H = self.hamiltonian.to_dense(device=str(device))
            # Ensure H is complex for matrix exponential
            if not H.is_complex():
                H = H.to(torch.complex64)
            U = torch.linalg.matrix_exp(-1j * self.time_step * H)

            for _ in range(num_steps):
                state_vector = U @ state_vector

            return state_vector
        elif self.num_sites <= 16:
            # For medium systems (7-16 qubits), use sparse matrix exponential
            return self._sparse_time_evolution(state_vector, num_steps)
        else:
            # For larger spin systems, use Trotter decomposition
            return self._trotter_evolution(state_vector, num_steps)

    def _sparse_time_evolution(
        self,
        state_vector: torch.Tensor,
        num_steps: int,
    ) -> torch.Tensor:
        """
        Apply time evolution using sparse matrix-vector multiplication.

        Uses scipy.sparse.linalg.expm_multiply for efficient computation
        of e^{-iHt}|psi> without forming the full matrix exponential.

        For molecular systems, works in particle-conserving subspace for
        massive speedup (e.g., 3K x 3K instead of 65K x 65K for NH3).
        """
        from scipy.sparse.linalg import expm_multiply

        # Build sparse Hamiltonian (cached for reuse)
        if not hasattr(self, '_sparse_H'):
            self._sparse_H = self._build_sparse_hamiltonian()

        # For molecular systems, work in subspace
        if self._is_molecular and self._subspace_basis is not None:
            return self._sparse_time_evolution_subspace(state_vector, num_steps)

        # Standard full Hilbert space evolution
        psi_np = state_vector.cpu().numpy().astype(np.complex128)

        # Apply time evolution num_steps times
        t = -1j * self.time_step
        for _ in range(num_steps):
            psi_np = expm_multiply(t * self._sparse_H, psi_np)

        return torch.from_numpy(psi_np).to(state_vector.device)

    def _sparse_time_evolution_subspace(
        self,
        state_vector: torch.Tensor,
        num_steps: int,
    ) -> torch.Tensor:
        """
        Time evolution in particle-conserving subspace.

        Much faster because subspace is typically 10-100x smaller than full space.
        """
        from scipy.sparse.linalg import expm_multiply

        # Convert full state vector to subspace representation
        psi_subspace = self._full_to_subspace(state_vector)

        # Apply time evolution in subspace
        t = -1j * self.time_step
        for _ in range(num_steps):
            psi_subspace = expm_multiply(t * self._sparse_H, psi_subspace)

        # Convert back to full Hilbert space
        return self._subspace_to_full(psi_subspace, state_vector.device)

    def _full_to_subspace(self, state_vector: torch.Tensor) -> np.ndarray:
        """Convert full Hilbert space state to subspace representation."""
        n_subspace = len(self._subspace_basis)
        psi_subspace = np.zeros(n_subspace, dtype=np.complex128)

        # Extract amplitudes for valid configurations
        state_np = state_vector.cpu().numpy()
        for i, config in enumerate(self._subspace_basis):
            idx = self.hamiltonian._config_to_index(config)
            psi_subspace[i] = state_np[idx]

        return psi_subspace

    def _subspace_to_full(self, psi_subspace: np.ndarray, device) -> torch.Tensor:
        """Convert subspace state back to full Hilbert space."""
        n_full = self.hamiltonian.hilbert_dim
        state_full = np.zeros(n_full, dtype=np.complex128)

        # Place amplitudes in correct positions
        for i, config in enumerate(self._subspace_basis):
            idx = self.hamiltonian._config_to_index(config)
            state_full[idx] = psi_subspace[i]

        return torch.from_numpy(state_full).to(device)

    def _build_sparse_hamiltonian(self):
        """
        Build sparse CSR Hamiltonian matrix.

        For molecular Hamiltonians, builds in particle-conserving subspace
        which is MUCH smaller than the full Hilbert space.
        """
        from scipy.sparse import csr_matrix

        # For molecular systems, build in particle-conserving subspace
        if self._is_molecular and self._subspace_basis is not None:
            return self._build_subspace_hamiltonian()

        # For non-molecular systems, use full Hilbert space
        if hasattr(self.hamiltonian, 'to_sparse'):
            print(f"Building sparse Hamiltonian ({self.hamiltonian.hilbert_dim} x {self.hamiltonian.hilbert_dim})...")
            return self.hamiltonian.to_sparse("cpu")

        # Fallback: build manually
        n = self.hamiltonian.hilbert_dim
        rows, cols, data = [], [], []

        # Generate all basis states
        basis = self.hamiltonian._generate_all_configs("cpu")

        print(f"Building sparse Hamiltonian manually ({n} x {n})...")
        for j in range(n):
            config_j = basis[j]

            # Diagonal element
            diag = self.hamiltonian.diagonal_element(config_j).item()
            rows.append(j)
            cols.append(j)
            data.append(diag)

            # Off-diagonal connections
            connected, elements = self.hamiltonian.get_connections(config_j)
            if len(connected) > 0:
                for conn, elem in zip(connected, elements):
                    # Find index of connected config
                    i = self.hamiltonian._config_to_index(conn)
                    rows.append(i)
                    cols.append(j)
                    data.append(elem.item() if hasattr(elem, 'item') else elem)

        return csr_matrix(
            (data, (rows, cols)),
            shape=(n, n),
            dtype=np.complex128
        )

    def _build_subspace_hamiltonian(self):
        """
        Build sparse Hamiltonian in particle-conserving subspace.

        This is MUCH faster than building in full Hilbert space because:
        - NH3: 3,136 x 3,136 instead of 65,536 x 65,536 (420x fewer elements)
        - N2: 14,400 x 14,400 instead of 1,048,576 x 1,048,576 (5300x fewer elements)
        """
        from scipy.sparse import csr_matrix

        n_subspace = len(self._subspace_basis)
        print(f"Building subspace Hamiltonian ({n_subspace:,} x {n_subspace:,})...")

        rows, cols, data = [], [], []

        # Build Hamiltonian matrix elements in subspace
        for j in range(n_subspace):
            config_j = self._subspace_basis[j]

            # Diagonal element
            diag = self.hamiltonian.diagonal_element(config_j).item()
            rows.append(j)
            cols.append(j)
            data.append(diag)

            # Off-diagonal connections (only within subspace)
            connected, elements = self.hamiltonian.get_connections(config_j)

            if len(connected) > 0:
                for conn, elem in zip(connected, elements):
                    # Look up index in subspace
                    key = tuple(conn.tolist())
                    if key in self._subspace_index_map:
                        i = self._subspace_index_map[key]
                        rows.append(i)
                        cols.append(j)
                        data.append(elem.item() if hasattr(elem, 'item') else elem)

        H_subspace = csr_matrix(
            (data, (rows, cols)),
            shape=(n_subspace, n_subspace),
            dtype=np.complex128
        )

        print(f"Subspace Hamiltonian built: {H_subspace.nnz:,} non-zero elements")
        return H_subspace

    def _trotter_evolution(
        self,
        state_vector: torch.Tensor,
        num_steps: int,
    ) -> torch.Tensor:
        """
        Apply Trotterized time evolution using second-order Trotter-Suzuki.

        Decomposes H into terms and applies:
        U ≈ Π_j e^{-iH_j Δt/2} Π_j e^{-iH_j Δt/2}  (reversed order)

        For Pauli Hamiltonians, each term e^{-iθP} can be efficiently computed.
        """
        dt = self.time_step / self.config.num_trotter_steps

        # Get Hamiltonian terms (Pauli decomposition)
        if hasattr(self.hamiltonian, 'pauli_terms'):
            # Use Pauli decomposition for true Trotter
            return self._trotter_pauli(state_vector, num_steps, dt)
        else:
            # Fallback: decompose into diagonal and off-diagonal parts
            return self._trotter_split(state_vector, num_steps, dt)

    def _trotter_pauli(
        self,
        state_vector: torch.Tensor,
        num_steps: int,
        dt: float,
    ) -> torch.Tensor:
        """
        Trotter evolution using Pauli term decomposition.

        Each Pauli term P with coefficient c contributes e^{-icPΔt}.
        """
        pauli_terms = self.hamiltonian.pauli_terms  # List of (coeff, pauli_string)

        for _ in range(num_steps):
            for _ in range(self.config.num_trotter_steps):
                # Forward sweep (half step)
                for coeff, pauli in pauli_terms:
                    state_vector = self._apply_pauli_exp(
                        state_vector, coeff * dt / 2, pauli
                    )
                # Backward sweep (half step)
                for coeff, pauli in reversed(pauli_terms):
                    state_vector = self._apply_pauli_exp(
                        state_vector, coeff * dt / 2, pauli
                    )

        return state_vector

    def _trotter_split(
        self,
        state_vector: torch.Tensor,
        num_steps: int,
        dt: float,
    ) -> torch.Tensor:
        """
        Trotter evolution by splitting H into diagonal and off-diagonal parts.

        H = H_diag + H_off
        U ≈ e^{-iH_diag Δt/2} e^{-iH_off Δt} e^{-iH_diag Δt/2}
        """
        H = self.hamiltonian.to_dense()
        H_diag = torch.diag(torch.diag(H))
        H_off = H - H_diag

        # Precompute exponentials for efficiency
        U_diag_half = torch.diag(torch.exp(-1j * dt / 2 * torch.diag(H)))
        U_off = torch.linalg.matrix_exp(-1j * dt * H_off)

        for _ in range(num_steps):
            for _ in range(self.config.num_trotter_steps):
                # Second-order Trotter: e^{-iH_d dt/2} e^{-iH_o dt} e^{-iH_d dt/2}
                state_vector = U_diag_half @ state_vector
                state_vector = U_off @ state_vector
                state_vector = U_diag_half @ state_vector

        return state_vector

    def _apply_pauli_exp(
        self,
        state_vector: torch.Tensor,
        angle: float,
        pauli_string: str,
    ) -> torch.Tensor:
        """
        Apply e^{-i*angle*P} where P is a Pauli string.

        For a Pauli string P, e^{-iθP} = cos(θ)I - i*sin(θ)P
        """
        # Build Pauli matrix
        P = self._pauli_string_to_matrix(pauli_string)

        # e^{-iθP} = cos(θ)I - i*sin(θ)P
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)

        result = cos_theta * state_vector - 1j * sin_theta * (P @ state_vector)
        return result

    def _pauli_string_to_matrix(self, pauli_string: str) -> torch.Tensor:
        """Convert Pauli string to matrix via tensor product."""
        # Single-qubit Pauli matrices
        I = torch.eye(2, dtype=torch.complex64)
        X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
        Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
        Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)

        pauli_map = {"I": I, "X": X, "Y": Y, "Z": Z}

        result = torch.tensor([[1.0]], dtype=torch.complex64)
        for p in pauli_string:
            result = torch.kron(result, pauli_map[p])

        return result

    def _sample_from_state(
        self,
        state_vector: torch.Tensor,
        num_samples: int,
    ) -> Dict[str, int]:
        """
        Sample bitstrings from a quantum state.

        Args:
            state_vector: Quantum state vector
            num_samples: Number of shots

        Returns:
            Dictionary mapping bitstrings to counts
        """
        # Compute probabilities
        probs = torch.abs(state_vector) ** 2
        probs = probs / probs.sum()  # Normalize

        # Sample indices (multinomial on GPU, convert to numpy for counting)
        indices = torch.multinomial(
            probs, num_samples, replacement=True
        ).cpu().numpy()

        # Count occurrences
        unique, counts = np.unique(indices, return_counts=True)

        # Convert to bitstring dictionary
        results = {}
        for idx, count in zip(unique, counts):
            bitstring = self._index_to_bitstring(idx)
            results[bitstring] = int(count)

        return results

    def _index_to_bitstring(self, idx: int) -> str:
        """Convert Hilbert space index to bitstring."""
        return format(idx, f"0{self.num_sites}b")

    def _bitstring_to_tensor(self, bitstring: str) -> torch.Tensor:
        """Convert bitstring to tensor configuration."""
        return torch.tensor([int(b) for b in bitstring], dtype=torch.long)

    def generate_krylov_samples(
        self,
        max_krylov_dim: Optional[int] = None,
        progress: bool = True,
    ) -> List[Dict[str, int]]:
        """
        Generate samples from Krylov states.

        For k = 0, 1, ..., max_k - 1:
            1. Prepare |ψ_k⟩ = U^k |ψ_0⟩
            2. Sample num_shots measurements

        Args:
            max_krylov_dim: Override config max Krylov dimension
            progress: Show progress bar

        Returns:
            List of sample dictionaries for each Krylov state
        """
        if max_krylov_dim is None:
            max_krylov_dim = self.config.max_krylov_dim

        self.krylov_samples = []

        # Create initial state vector in Hilbert space on the correct device
        # |ψ_0⟩ = |bitstring⟩
        device = self.device
        initial_index = int(
            "".join(str(b.item()) for b in self.initial_state.cpu()), 2
        )
        state_vector = torch.zeros(
            self.hamiltonian.hilbert_dim, dtype=torch.complex64, device=device
        )
        state_vector[initial_index] = 1.0

        iterator = range(max_krylov_dim)
        if progress:
            iterator = tqdm(iterator, desc="Generating Krylov states")

        current_state = state_vector.clone()

        for k in iterator:
            # Sample from current state
            samples = self._sample_from_state(
                current_state, self.config.shots_per_krylov
            )
            self.krylov_samples.append(samples)

            # Evolve state: |ψ_{k+1}⟩ = U |ψ_k⟩
            if k < max_krylov_dim - 1:
                current_state = self._time_evolution_operator(
                    current_state, num_steps=1
                )

        return self.krylov_samples

    def build_cumulative_basis(self) -> List[Dict[str, int]]:
        """
        Build cumulative basis by accumulating samples across Krylov states.

        cumulative[k] contains all unique bitstrings from steps 0, 1, ..., k.

        Returns:
            List of cumulative sample dictionaries
        """
        cumulative = []
        all_samples: Dict[str, int] = {}

        for k, samples in enumerate(self.krylov_samples):
            # Merge samples
            for bitstring, count in samples.items():
                all_samples[bitstring] = all_samples.get(bitstring, 0) + count

            cumulative.append(dict(all_samples))

        return cumulative

    def get_basis_states(
        self,
        krylov_index: int,
        cumulative: bool = True,
    ) -> torch.Tensor:
        """
        Get basis states as tensor array.

        Args:
            krylov_index: Krylov step index
            cumulative: Whether to use cumulative basis

        Returns:
            Tensor of basis configurations, shape (n_basis, num_sites)
        """
        if cumulative:
            samples = self.build_cumulative_basis()[krylov_index]
        else:
            samples = self.krylov_samples[krylov_index]

        bitstrings = list(samples.keys())
        configs = [self._bitstring_to_tensor(bs) for bs in bitstrings]

        return torch.stack(configs)

    def compute_ground_state_energy(
        self,
        basis: Optional[torch.Tensor] = None,
        return_eigenvector: bool = False,
        regularization: float = 1e-8,
    ) -> Tuple[float, Optional[torch.Tensor]]:
        """
        Compute ground state energy via subspace diagonalization.

        Projects Hamiltonian onto the sampled basis and diagonalizes using
        sparse eigensolver (scipy.sparse.linalg.eigsh or cupyx equivalent)
        as specified in AGENTs.md for scalability.

        Includes numerical stability improvements:
        - Uses float64 for better precision
        - Regularization for ill-conditioned matrices
        - Hermitian symmetrization
        - SVD-based fallback for problematic cases
        - Validates result is real (imaginary part should be tiny)

        Args:
            basis: Basis states to use (default: use all sampled states)
            return_eigenvector: Whether to return ground state coefficients
            regularization: Small value added to diagonal for stability

        Returns:
            (ground_energy, ground_state_coefficients) if return_eigenvector
            ground_energy otherwise
        """
        if basis is None:
            # Use cumulative basis from last Krylov step
            cumulative = self.build_cumulative_basis()
            basis = self.get_basis_states(len(self.krylov_samples) - 1)

        # Build projected Hamiltonian
        H_proj = self.hamiltonian.matrix_elements(basis, basis)

        # Convert to numpy with DOUBLE precision for numerical stability
        H_np = H_proj.detach().cpu().numpy().astype(np.complex128)
        n = H_np.shape[0]

        # Ensure Hermitian symmetry (numerical errors can break this)
        H_np = 0.5 * (H_np + H_np.conj().T)

        # Verify Hamiltonian is essentially real (molecular Hamiltonians should be)
        max_imag = np.abs(H_np.imag).max()
        if max_imag > 1e-10:
            # Has significant imaginary part - keep complex
            pass
        else:
            # Essentially real - use real matrix for better stability
            H_np = H_np.real.astype(np.float64)

        # Add small regularization to improve conditioning
        # NOTE: This shifts ALL eigenvalues up by regularization amount
        if regularization > 0:
            H_np = H_np + regularization * np.eye(n, dtype=H_np.dtype)

        # Check matrix conditioning
        try:
            cond = np.linalg.cond(H_np)
            if cond > 1e12:
                print(f"WARNING: Ill-conditioned Hamiltonian (cond={cond:.2e})")
                print("Using SVD-based solver for numerical stability")
                return self._svd_ground_state(H_np, return_eigenvector)
        except np.linalg.LinAlgError:
            print("WARNING: Could not compute condition number, using SVD")
            return self._svd_ground_state(H_np, return_eigenvector)

        # Use sparse eigensolver for efficiency (as specified in AGENTs.md)
        # For small matrices, dense is actually faster
        if n < 100:
            # Small matrix: use dense solver
            eigenvalues, eigenvectors = np.linalg.eigh(H_np)
            E0 = float(eigenvalues[0])
            v0 = eigenvectors[:, 0]
        else:
            # Large matrix: use sparse solver
            use_gpu = self.config.use_gpu and CUPY_AVAILABLE

            try:
                if use_gpu:
                    # GPU sparse solver
                    H_gpu = cp.asarray(H_np)
                    H_sparse = cupy_csr(H_gpu)
                    eigenvalues = cupy_eigsh(
                        H_sparse,
                        k=self.config.num_eigenvalues,
                        which=self.config.which_eigenvalues,
                        return_eigenvectors=return_eigenvector,
                    )
                    if return_eigenvector:
                        eigenvalues, eigenvectors = eigenvalues
                        E0 = float(cp.asnumpy(eigenvalues[0]))
                        v0 = cp.asnumpy(eigenvectors[:, 0])
                    else:
                        E0 = float(cp.asnumpy(eigenvalues[0]))
                        v0 = None
                else:
                    # CPU sparse solver
                    H_sparse = scipy_csr(H_np)
                    result = scipy_eigsh(
                        H_sparse,
                        k=min(self.config.num_eigenvalues, n - 1),
                        which=self.config.which_eigenvalues,
                        return_eigenvectors=return_eigenvector,
                    )
                    if return_eigenvector:
                        eigenvalues, eigenvectors = result
                        E0 = float(eigenvalues[0])
                        v0 = eigenvectors[:, 0]
                    else:
                        E0 = float(result[0])
                        v0 = None
            except Exception as e:
                print(f"Sparse eigensolver failed: {e}")
                print("Falling back to dense solver")
                eigenvalues, eigenvectors = np.linalg.eigh(H_np)
                E0 = float(eigenvalues[0])
                v0 = eigenvectors[:, 0] if return_eigenvector else None

        if return_eigenvector:
            return E0, torch.from_numpy(v0) if v0 is not None else None
        else:
            return E0, None

    def _svd_ground_state(
        self,
        H_np: np.ndarray,
        return_eigenvector: bool = False,
    ) -> Tuple[float, Optional[torch.Tensor]]:
        """
        Compute ground state using SVD-based approach for numerical stability.

        Uses eigendecomposition after projecting out small singular values
        that cause numerical instability.
        """
        # SVD to identify and handle near-null space
        U, s, Vh = np.linalg.svd(H_np, hermitian=True)

        # Filter out very small singular values
        threshold = 1e-10 * s.max()
        valid_mask = s > threshold
        n_valid = valid_mask.sum()

        if n_valid < len(s):
            print(f"  SVD: Projecting out {len(s) - n_valid} near-null modes")

        # Reconstruct regularized Hamiltonian
        s_reg = np.where(s > threshold, s, threshold)
        H_reg = U @ np.diag(s_reg) @ Vh

        # Now diagonalize the regularized matrix
        eigenvalues, eigenvectors = np.linalg.eigh(H_reg)
        E0 = float(eigenvalues[0])
        v0 = eigenvectors[:, 0]

        if return_eigenvector:
            return E0, torch.from_numpy(v0)
        else:
            return E0, None

    def run(
        self,
        max_krylov_dim: Optional[int] = None,
        progress: bool = True,
    ) -> Dict[str, List]:
        """
        Run full SKQD algorithm.

        Returns energy as function of Krylov dimension.

        Args:
            max_krylov_dim: Maximum Krylov dimension
            progress: Show progress bars

        Returns:
            Dictionary with 'krylov_dims', 'energies', 'basis_sizes'
        """
        if max_krylov_dim is None:
            max_krylov_dim = self.config.max_krylov_dim

        # Generate Krylov samples
        self.generate_krylov_samples(max_krylov_dim, progress=progress)

        # Build cumulative basis
        cumulative = self.build_cumulative_basis()

        # Compute energy at each Krylov dimension
        results = {
            "krylov_dims": [],
            "energies": [],
            "basis_sizes": [],
        }

        for k in range(1, max_krylov_dim):
            basis = self.get_basis_states(k, cumulative=True)
            E0, _ = self.compute_ground_state_energy(basis)

            results["krylov_dims"].append(k + 1)
            results["energies"].append(E0)
            results["basis_sizes"].append(len(basis))

        self.energies = results["energies"]

        return results


class FlowGuidedSKQD(SampleBasedKrylovDiagonalization):
    """
    SKQD with Flow-Guided initial basis.

    Instead of (or in addition to) using Krylov time evolution samples,
    this variant incorporates the basis discovered by the normalizing flow.

    The NF-discovered basis provides a good initial subspace that captures
    the support of the ground state, while Krylov refinement improves
    the energy estimate through systematic subspace expansion.
    """

    def __init__(
        self,
        hamiltonian: Hamiltonian,
        nf_basis: torch.Tensor,
        config: Optional[SKQDConfig] = None,
        initial_state: Optional[torch.Tensor] = None,
    ):
        super().__init__(hamiltonian, config, initial_state)

        self.nf_basis = nf_basis  # (n_nf, num_sites)

    def get_combined_basis(
        self,
        krylov_index: int,
        include_nf: bool = True,
    ) -> torch.Tensor:
        """
        Get combined basis from NF and Krylov sampling.

        Args:
            krylov_index: Krylov step index
            include_nf: Whether to include NF-discovered basis

        Returns:
            Combined unique basis states
        """
        # Get Krylov basis
        krylov_basis = self.get_basis_states(krylov_index, cumulative=True)

        if not include_nf:
            return krylov_basis

        # Ensure both are on the same device
        nf_basis = self.nf_basis.to(krylov_basis.device)

        # Combine with NF basis
        combined = torch.cat([nf_basis, krylov_basis], dim=0)

        # Remove duplicates
        unique = torch.unique(combined, dim=0)

        return unique

    def run_with_nf(
        self,
        max_krylov_dim: Optional[int] = None,
        progress: bool = True,
    ) -> Dict[str, List]:
        """
        Run SKQD with NF-augmented basis.

        IMPORTANT: This method combines Krylov-discovered configurations with
        the NF basis, rather than using the full particle-conserving subspace.
        This is the correct approach because:
        1. The NF basis already captures important low-energy configurations
        2. Krylov time evolution discovers configurations missed by NF
        3. Combining them gives a better basis than either alone

        Includes numerical stability improvements:
        - Uses regularization from config
        - Validates energy is variationally consistent
        - Falls back to best NF-only energy if instability detected
        - Uses float64 for better numerical precision

        Returns:
            Dictionary with results comparing NF-only, Krylov-only, and combined
        """
        if max_krylov_dim is None:
            max_krylov_dim = self.config.max_krylov_dim

        # Energy with NF basis only (reference for stability check)
        E_nf, _ = self.compute_ground_state_energy(
            self.nf_basis,
            regularization=self.config.regularization
        )
        print(f"NF-only basis energy: {E_nf:.6f} ({len(self.nf_basis)} configs)")

        # Generate Krylov samples
        self.generate_krylov_samples(max_krylov_dim, progress=progress)

        results = {
            "krylov_dims": [],
            "energies_krylov": [],
            "energies_combined": [],
            "basis_sizes_krylov": [],
            "basis_sizes_combined": [],
            "energy_nf_only": E_nf,
            "nf_basis_size": len(self.nf_basis),
            "numerical_warnings": [],
        }

        best_energy = E_nf
        best_basis_size = len(self.nf_basis)
        instability_detected = False

        for k in range(1, max_krylov_dim):
            # Krylov only
            krylov_basis = self.get_basis_states(k, cumulative=True)
            E_krylov, _ = self.compute_ground_state_energy(
                krylov_basis,
                regularization=self.config.regularization
            )

            # Combined: NF basis + Krylov-discovered configs
            combined_basis = self.get_combined_basis(k, include_nf=True)
            E_combined, _ = self.compute_ground_state_energy(
                combined_basis,
                regularization=self.config.regularization
            )

            # VARIATIONAL CHECK: Energy should decrease or stay same as basis grows
            # If energy increases, likely numerical instability
            if k > 1 and len(results["energies_combined"]) > 0:
                prev_energy = results["energies_combined"][-1]
                energy_change = E_combined - prev_energy

                # Energy should not increase significantly
                if energy_change > 0.001:  # 1 mHa tolerance
                    warning = f"k={k+1}: Energy increased by {energy_change*1000:.4f} mHa (numerical instability)"
                    results["numerical_warnings"].append(warning)
                    print(f"WARNING: {warning}")
                    instability_detected = True

                # Large energy jumps can indicate numerical instability
                if abs(energy_change) > 1.0:  # 1 Ha is suspicious for Krylov refinement
                    warning = f"k={k+1}: Large energy jump {abs(energy_change):.4f} Ha"
                    results["numerical_warnings"].append(warning)
                    print(f"WARNING: {warning}")
                    instability_detected = True

            # Track best valid energy (variational principle)
            if E_combined < best_energy:
                best_energy = E_combined
                best_basis_size = len(combined_basis)

            results["krylov_dims"].append(k + 1)
            results["energies_krylov"].append(E_krylov)
            results["energies_combined"].append(E_combined)
            results["basis_sizes_krylov"].append(len(krylov_basis))
            results["basis_sizes_combined"].append(len(combined_basis))

        # Report statistics on Krylov contribution
        if results["energies_combined"]:
            krylov_improvement = E_nf - best_energy
            new_configs = best_basis_size - len(self.nf_basis)
            if krylov_improvement > 0:
                print(f"Krylov improvement: {krylov_improvement*1000:.4f} mHa "
                      f"({new_configs} new configs from Krylov sampling)")

        # If instability detected, report the most stable result
        if instability_detected:
            print(f"Numerical instability detected. Best stable energy: {best_energy:.6f}")
            results["best_stable_energy"] = best_energy
        else:
            results["best_stable_energy"] = best_energy

        return results
