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

        # Set up initial state
        if initial_state is not None:
            self.initial_state = initial_state
        else:
            # Default: Néel state |010101...⟩
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

        Uses exact exponentiation for classical simulation.
        For quantum hardware, this would use Trotter decomposition.

        Args:
            state_vector: Full state vector in Hilbert space
            num_steps: Number of time steps to apply

        Returns:
            Evolved state vector
        """
        # Ensure state vector is on the correct device
        device = self.device
        state_vector = state_vector.to(device)

        # For small systems, use exact evolution
        if self.num_sites <= 14:
            # Get dense Hamiltonian on the same device
            H = self.hamiltonian.to_dense(device=str(device))
            # Ensure H is complex for matrix exponential
            if not H.is_complex():
                H = H.to(torch.complex64)
            U = torch.linalg.matrix_exp(-1j * self.time_step * H)

            for _ in range(num_steps):
                state_vector = U @ state_vector

            return state_vector
        else:
            # For larger systems, use Trotter decomposition
            return self._trotter_evolution(state_vector, num_steps)

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
    ) -> Tuple[float, Optional[torch.Tensor]]:
        """
        Compute ground state energy via subspace diagonalization.

        Projects Hamiltonian onto the sampled basis and diagonalizes using
        sparse eigensolver (scipy.sparse.linalg.eigsh or cupyx equivalent)
        as specified in AGENTs.md for scalability.

        Args:
            basis: Basis states to use (default: use all sampled states)
            return_eigenvector: Whether to return ground state coefficients

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

        # Convert to numpy for sparse solver
        H_np = H_proj.detach().cpu().numpy()
        n = H_np.shape[0]

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

        if return_eigenvector:
            return E0, torch.from_numpy(v0) if v0 is not None else None
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

        Returns:
            Dictionary with results comparing NF-only, Krylov-only, and combined
        """
        if max_krylov_dim is None:
            max_krylov_dim = self.config.max_krylov_dim

        # Energy with NF basis only
        E_nf, _ = self.compute_ground_state_energy(self.nf_basis)

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
        }

        for k in range(1, max_krylov_dim):
            # Krylov only
            krylov_basis = self.get_basis_states(k, cumulative=True)
            E_krylov, _ = self.compute_ground_state_energy(krylov_basis)

            # Combined
            combined_basis = self.get_combined_basis(k, include_nf=True)
            E_combined, _ = self.compute_ground_state_energy(combined_basis)

            results["krylov_dims"].append(k + 1)
            results["energies_krylov"].append(E_krylov)
            results["energies_combined"].append(E_combined)
            results["basis_sizes_krylov"].append(len(krylov_basis))
            results["basis_sizes_combined"].append(len(combined_basis))

        return results
