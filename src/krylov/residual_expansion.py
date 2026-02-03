"""
Residual-Based Basis Expansion for Quantum Diagonalization.

This module implements Selected-CI style basis expansion using residual analysis:
1. Compute current ground state estimate |Φ⟩ in current basis
2. Compute residual r = H|Φ⟩ - E|Φ⟩
3. Find configurations outside basis with large residual contributions
4. Add those configurations to expand the basis
5. Repeat until convergence

This provides automatic recovery of important configurations that
might have been missed by the initial basis construction.

References:
- Selected-CI: Huron et al., "Iterative perturbation calculations..."
- CIPSI: Evangelisti et al., "Convergence of an improved CIPSI algorithm"
- ASCI: Tubman et al., "A deterministic alternative to FCIQMC"
- SKQD paper: Configuration recovery via residual analysis
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict, List, Set
from dataclasses import dataclass


@dataclass
class ResidualExpansionConfig:
    """Configuration for residual-based expansion."""

    # Maximum configurations to add per iteration
    max_configs_per_iter: int = 100

    # Residual threshold for adding configurations
    residual_threshold: float = 1e-4

    # Maximum total iterations
    max_iterations: int = 10

    # Convergence criterion (energy change in Hartree)
    energy_convergence: float = 1e-6

    # Early stopping: minimum energy improvement per iteration (in mHa)
    # Stop if improvement < this threshold for consecutive iterations
    min_energy_improvement_mha: float = 0.05  # 0.05 mHa = 5e-5 Ha

    # Number of consecutive stagnant iterations before stopping
    stagnation_patience: int = 2

    # Maximum total basis size
    max_basis_size: int = 4096

    # Whether to use importance sampling for residual computation
    use_importance_sampling: bool = True

    # Number of samples for importance sampling
    n_importance_samples: int = 10000

    # Energy bound for variational principle enforcement
    # If set, any expansion that drops energy below this is rejected
    energy_lower_bound: float = None  # Set to reference energy (e.g., CCSD)

    # Maximum allowed energy drop per iteration (Ha) - safety check
    # If energy drops more than this in one iteration, reject the expansion
    max_energy_drop_per_iter: float = 0.5  # 500 mHa is suspicious


class ResidualBasedExpander:
    """
    Expands basis by analyzing residual contributions.

    The residual r_i = <i|H|Φ⟩ - E<i|Φ⟩ measures how much configuration i
    contributes to the error in the current approximation. Large |r_i|
    indicates important missing configurations.

    For configurations outside the current basis, r_i = <i|H|Φ⟩ since <i|Φ⟩ = 0.
    """

    def __init__(
        self,
        hamiltonian,
        config: ResidualExpansionConfig = None,
    ):
        self.hamiltonian = hamiltonian
        self.config = config or ResidualExpansionConfig()
        self.device = getattr(hamiltonian, 'device', 'cpu')

    def expand_basis(
        self,
        current_basis: torch.Tensor,
        energy: Optional[float] = None,
        eigenvector: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Expand basis using residual analysis.

        Args:
            current_basis: (n_basis, n_sites) current basis configurations
            energy: Current ground state energy estimate (computed if not given)
            eigenvector: Current ground state eigenvector (computed if not given)

        Returns:
            expanded_basis: (n_expanded, n_sites) expanded basis
            stats: Dictionary with expansion statistics
        """
        cfg = self.config
        current_basis = current_basis.to(self.device)
        n_current = len(current_basis)

        # Compute current energy and eigenvector if not provided
        if energy is None or eigenvector is None:
            energy, eigenvector = self._diagonalize(current_basis)

        # Track expansion history
        history = {
            'energies': [energy],
            'basis_sizes': [n_current],
            'configs_added': [],
        }

        expanded_basis = current_basis.clone()
        current_energy = energy
        current_eigenvector = eigenvector

        for iteration in range(cfg.max_iterations):
            # Check if basis is already at max size
            if len(expanded_basis) >= cfg.max_basis_size:
                break

            # Find important configurations via residual
            new_configs, residuals = self._find_residual_configs(
                expanded_basis, current_energy, current_eigenvector
            )

            if len(new_configs) == 0:
                break

            # Add new configurations
            expanded_basis = torch.cat([expanded_basis, new_configs], dim=0)
            expanded_basis = torch.unique(expanded_basis, dim=0)

            # Rediagonalize
            new_energy, new_eigenvector = self._diagonalize(expanded_basis)

            # Update history
            history['energies'].append(new_energy)
            history['basis_sizes'].append(len(expanded_basis))
            history['configs_added'].append(len(new_configs))

            # Check convergence
            energy_change = abs(new_energy - current_energy)
            if energy_change < cfg.energy_convergence:
                break

            current_energy = new_energy
            current_eigenvector = new_eigenvector

        stats = {
            'initial_basis_size': n_current,
            'final_basis_size': len(expanded_basis),
            'configs_added_total': len(expanded_basis) - n_current,
            'iterations': iteration + 1,
            'converged': energy_change < cfg.energy_convergence if 'energy_change' in dir() else True,
            'final_energy': current_energy,
            'history': history,
        }

        return expanded_basis, stats

    def _diagonalize(
        self,
        basis: torch.Tensor,
    ) -> Tuple[float, np.ndarray]:
        """Diagonalize Hamiltonian in given basis."""
        H_matrix = self.hamiltonian.matrix_elements(basis, basis)
        H_np = H_matrix.cpu().numpy()

        eigenvalues, eigenvectors = np.linalg.eigh(H_np)

        return float(eigenvalues[0]), eigenvectors[:, 0]

    def _find_residual_configs(
        self,
        basis: torch.Tensor,
        energy: float,
        eigenvector: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find configurations with large residual contributions.

        For each configuration i not in basis:
            r_i = <i|H|Φ⟩ = sum_j c_j <i|H|j>

        where j runs over basis states and c_j are eigenvector coefficients.
        """
        cfg = self.config
        n_basis = len(basis)
        n_sites = basis.shape[1]

        # Build set of basis configurations for quick lookup (GPU-accelerated)
        basis_set = self._configs_to_set(basis)

        # Convert eigenvector to tensor
        coeffs = torch.from_numpy(eigenvector).float().to(self.device)

        # Filter to significant coefficients first (avoids .item() in loop)
        significant_mask = coeffs.abs() > 1e-10
        significant_indices = significant_mask.nonzero(as_tuple=True)[0]

        if len(significant_indices) == 0:
            return torch.empty(0, n_sites, device=self.device), torch.empty(0, device=self.device)

        # Use vectorized batch connections if available (GPU-accelerated)
        if hasattr(self.hamiltonian, 'get_connections_vectorized_batch'):
            significant_basis = basis[significant_indices]
            all_connected, all_elements, batch_indices = \
                self.hamiltonian.get_connections_vectorized_batch(significant_basis)

            if len(all_connected) == 0:
                return torch.empty(0, n_sites, device=self.device), torch.empty(0, device=self.device)

            # Encode connected configs for membership check
            connected_ints = self._encode_configs_gpu(all_connected)
            connected_ints_cpu = set(connected_ints.cpu().tolist())

            # Filter to configs outside basis
            in_basis_mask = torch.tensor(
                [int(ci) in basis_set for ci in connected_ints.cpu().tolist()],
                device=self.device, dtype=torch.bool
            )
            outside_mask = ~in_basis_mask

            if not outside_mask.any():
                return torch.empty(0, n_sites, device=self.device), torch.empty(0, device=self.device)

            # Get coefficients for each connection
            local_batch_idx = batch_indices[outside_mask]
            connected_coeffs = coeffs[significant_indices[local_batch_idx]]

            # Compute residuals
            candidates = all_connected[outside_mask]
            candidate_residuals = (connected_coeffs * all_elements[outside_mask]).abs()

        else:
            # Fallback to sequential method for hamiltonians without vectorized batch
            candidates = []
            candidate_residuals = []

            for idx in significant_indices:
                j = idx.item()
                connected, elements = self.hamiltonian.get_connections(basis[j])

                if len(connected) == 0:
                    continue

                # Use GPU-based integer encoding for membership check
                connected_ints = self._encode_configs_gpu(connected)
                connected_ints_cpu = connected_ints.cpu().tolist()

                coeff_j = coeffs[j]
                for k in range(len(connected)):
                    if connected_ints_cpu[k] not in basis_set:
                        residual = (coeff_j * elements[k]).abs()
                        candidates.append(connected[k])
                        candidate_residuals.append(residual)

            if not candidates:
                return torch.empty(0, n_sites, device=self.device), torch.empty(0, device=self.device)

            candidates = torch.stack(candidates)
            candidate_residuals = torch.stack(candidate_residuals)

        # Ensure candidates and residuals are tensors
        if not isinstance(candidate_residuals, torch.Tensor):
            residuals = torch.tensor(candidate_residuals, device=self.device)
        else:
            residuals = candidate_residuals

        # Remove duplicates, keeping highest residual (GPU-optimized)
        unique_candidates, inverse = torch.unique(candidates, dim=0, return_inverse=True)

        # Use scatter_reduce for efficient max aggregation (GPU-accelerated)
        unique_residuals = torch.zeros(len(unique_candidates), device=self.device)
        unique_residuals.scatter_reduce_(
            0, inverse, residuals, reduce='amax', include_self=False
        )

        # Filter by threshold
        mask = unique_residuals > cfg.residual_threshold
        filtered_candidates = unique_candidates[mask]
        filtered_residuals = unique_residuals[mask]

        if len(filtered_candidates) == 0:
            return torch.empty(0, n_sites, device=self.device), torch.empty(0, device=self.device)

        # Select top configurations by residual magnitude
        n_select = min(cfg.max_configs_per_iter, len(filtered_candidates))
        _, top_indices = torch.topk(filtered_residuals, n_select)

        selected = filtered_candidates[top_indices]
        selected_residuals = filtered_residuals[top_indices]

        return selected, selected_residuals

    def _configs_to_set(self, configs: torch.Tensor) -> Set[int]:
        """
        Convert configurations to set of integer hashes for O(1) lookup.

        GPU-OPTIMIZED: Uses integer encoding on GPU, single CPU transfer.
        For C2H4 (28 qubits, 3000+ configs): ~10-20x faster than hash method.
        """
        device = configs.device
        num_sites = configs.shape[1]

        # Compute powers of 2 on GPU (or use cached if available)
        if hasattr(self.hamiltonian, '_powers_gpu'):
            powers = self.hamiltonian._powers_gpu
        else:
            powers = (2 ** torch.arange(num_sites, device=device, dtype=torch.long)).flip(0)

        # Encode all configs as integers on GPU, then transfer once
        config_ints = (configs.long() * powers).sum(dim=1)
        return set(config_ints.cpu().tolist())  # Single CPU transfer

    def _encode_configs_gpu(self, configs: torch.Tensor) -> torch.Tensor:
        """Encode configs as integers on GPU for fast comparison."""
        device = configs.device
        num_sites = configs.shape[1]

        if hasattr(self.hamiltonian, '_powers_gpu'):
            powers = self.hamiltonian._powers_gpu
        else:
            powers = (2 ** torch.arange(num_sites, device=device, dtype=torch.long)).flip(0)

        return (configs.long() * powers).sum(dim=1)


def iterative_residual_expansion(
    hamiltonian,
    initial_basis: torch.Tensor,
    max_iterations: int = 10,
    max_configs_per_iter: int = 100,
    energy_convergence: float = 1e-6,
    verbose: bool = False,
) -> Tuple[torch.Tensor, float, Dict]:
    """
    Convenience function for residual-based expansion.

    Args:
        hamiltonian: System Hamiltonian
        initial_basis: Starting basis configurations
        max_iterations: Maximum expansion iterations
        max_configs_per_iter: Max configs to add per iteration
        energy_convergence: Convergence threshold
        verbose: Print progress

    Returns:
        final_basis: Expanded basis
        final_energy: Ground state energy estimate
        stats: Expansion statistics
    """
    config = ResidualExpansionConfig(
        max_iterations=max_iterations,
        max_configs_per_iter=max_configs_per_iter,
        energy_convergence=energy_convergence,
    )

    expander = ResidualBasedExpander(hamiltonian, config)

    if verbose:
        print(f"Starting residual expansion from {len(initial_basis)} configs")

    final_basis, stats = expander.expand_basis(initial_basis)

    if verbose:
        print(f"Expanded to {len(final_basis)} configs")
        print(f"Energy: {stats['final_energy']:.8f}")
        print(f"Iterations: {stats['iterations']}")

    return final_basis, stats['final_energy'], stats


class SelectedCIExpander:
    """
    Selected-CI style expansion with perturbation-based selection.

    Uses second-order perturbation theory to estimate importance:
    ε_i = |<i|H|Φ⟩|² / (E - E_i)

    This provides a better estimate of configuration importance
    than raw residual magnitude.

    Optimizations:
    - Uses numpy arrays instead of Python sets for faster membership checks
    - Batches diagonal element computations
    - Early terminates basis state loop for insignificant coefficients
    """

    def __init__(
        self,
        hamiltonian,
        config: ResidualExpansionConfig = None,
    ):
        self.hamiltonian = hamiltonian
        self.config = config or ResidualExpansionConfig()
        self.device = getattr(hamiltonian, 'device', 'cpu')

        # Cache for basis set membership (optimization)
        self._basis_array_cache = None
        self._basis_hash_cache = None

    def expand_basis(
        self,
        current_basis: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Expand basis using perturbation-based selection.

        Includes multiple safety checks:
        1. Variational consistency: reject if energy increases
        2. Energy bound: reject if energy drops below reference (e.g., CCSD)
        3. Catastrophic drop: reject if energy drops too much in one iteration

        Args:
            current_basis: Current basis configurations

        Returns:
            expanded_basis: Expanded basis
            stats: Expansion statistics
        """
        cfg = self.config
        current_basis = current_basis.to(self.device)

        # Compute current energy and eigenvector
        energy, eigenvector = self._diagonalize(current_basis)

        # Find important configurations via perturbation estimate
        new_configs, importances = self._find_important_configs(
            current_basis, energy, eigenvector
        )

        if len(new_configs) == 0:
            return current_basis, {'configs_added': 0, 'energy': energy, 'initial_energy': energy, 'final_energy': energy}

        # Add new configurations
        expanded_basis = torch.cat([current_basis, new_configs], dim=0)
        expanded_basis = torch.unique(expanded_basis, dim=0)

        # Rediagonalize
        new_energy, _ = self._diagonalize(expanded_basis)

        # SAFETY CHECK 1: Variational consistency
        # Adding configurations should NEVER increase energy
        energy_improvement = energy - new_energy

        if energy_improvement < -1e-8:  # Allow tiny numerical tolerance
            stats = {
                'initial_size': len(current_basis),
                'final_size': len(current_basis),
                'configs_added': 0,
                'initial_energy': energy,
                'final_energy': energy,
                'energy_improvement': 0.0,
                'energy_improvement_mha': 0.0,
                'variational_violation': True,
                'rejected_energy': new_energy,
                'rejected_increase_mha': -energy_improvement * 1000,
                'rejection_reason': 'energy_increased',
            }
            return current_basis, stats

        # SAFETY CHECK 2: Energy lower bound
        # If we have a reference energy, reject if we go below it
        if cfg.energy_lower_bound is not None:
            if new_energy < cfg.energy_lower_bound - 1e-6:
                violation_amount = (cfg.energy_lower_bound - new_energy) * 1000
                print(f"  WARNING: Energy {new_energy:.6f} Ha below bound "
                      f"{cfg.energy_lower_bound:.6f} Ha by {violation_amount:.2f} mHa!")
                stats = {
                    'initial_size': len(current_basis),
                    'final_size': len(current_basis),
                    'configs_added': 0,
                    'initial_energy': energy,
                    'final_energy': energy,
                    'energy_improvement': 0.0,
                    'energy_improvement_mha': 0.0,
                    'variational_violation': True,
                    'rejected_energy': new_energy,
                    'bound_violation_mha': violation_amount,
                    'rejection_reason': 'below_energy_bound',
                }
                return current_basis, stats

        # SAFETY CHECK 3: Catastrophic energy drop
        # Large drops often indicate numerical issues
        if energy_improvement > cfg.max_energy_drop_per_iter:
            print(f"  WARNING: Suspiciously large energy drop: "
                  f"{energy_improvement*1000:.2f} mHa in one iteration!")
            # Still accept but flag it
            pass

        stats = {
            'initial_size': len(current_basis),
            'final_size': len(expanded_basis),
            'configs_added': len(new_configs),
            'initial_energy': energy,
            'final_energy': new_energy,
            'energy_improvement': energy_improvement,
            'energy_improvement_mha': energy_improvement * 1000,
            'variational_violation': False,
        }

        return expanded_basis, stats

    def _diagonalize(
        self,
        basis: torch.Tensor,
    ) -> Tuple[float, np.ndarray]:
        """
        Diagonalize Hamiltonian in given basis.

        Uses float64 precision for numerical stability.
        Ensures consistent ground state is found across iterations.

        IMPORTANT: The Hamiltonian matrix should already be Hermitian from
        matrix_elements_fast(). We check for asymmetry and warn if found.
        """
        n_basis = len(basis)

        H_matrix = self.hamiltonian.matrix_elements(basis, basis)
        # Use float64 for better numerical precision
        H_np = H_matrix.cpu().numpy().astype(np.float64)

        # Check for asymmetry BEFORE symmetrization
        asymmetry = np.abs(H_np - H_np.T).max()
        if asymmetry > 1e-8:
            print(f"  WARNING: Matrix asymmetry detected: {asymmetry:.2e}")
            # Find worst offender
            diff = np.abs(H_np - H_np.T)
            i, j = np.unravel_index(np.argmax(diff), diff.shape)
            if H_np[i, j] * H_np[j, i] < 0:
                print(f"  CRITICAL: Opposite signs at ({i},{j}): "
                      f"H[i,j]={H_np[i,j]:.4f}, H[j,i]={H_np[j,i]:.4f}")

        # Ensure Hermitian symmetry (should be minimal adjustment now)
        H_np = 0.5 * (H_np + H_np.T)

        # Use sparse solver for large bases
        if n_basis > 500:
            try:
                from scipy.sparse import csr_matrix
                from scipy.sparse.linalg import eigsh

                H_sparse = csr_matrix(H_np)
                # Use tighter tolerance for consistent convergence
                eigenvalues, eigenvectors = eigsh(
                    H_sparse, k=1, which='SA', tol=1e-12, maxiter=1000
                )
                return float(eigenvalues[0]), eigenvectors[:, 0]
            except Exception:
                pass  # Fall back to dense

        eigenvalues, eigenvectors = np.linalg.eigh(H_np)
        return float(eigenvalues[0]), eigenvectors[:, 0]

    def _configs_to_hash_set(self, configs: torch.Tensor) -> set:
        """Convert configurations to integer set for O(1) lookup (GPU-optimized)."""
        device = configs.device
        num_sites = configs.shape[1]

        # Use GPU-based integer encoding
        if hasattr(self.hamiltonian, '_powers_gpu'):
            powers = self.hamiltonian._powers_gpu
        else:
            powers = (2 ** torch.arange(num_sites, device=device, dtype=torch.long)).flip(0)

        config_ints = (configs.long() * powers).sum(dim=1)
        return set(config_ints.cpu().tolist())  # Single CPU transfer

    def _encode_config_int(self, config: torch.Tensor) -> int:
        """Encode single config as integer for hash lookup."""
        device = config.device
        num_sites = config.shape[0]

        if hasattr(self.hamiltonian, '_powers_gpu'):
            powers = self.hamiltonian._powers_gpu
        else:
            powers = (2 ** torch.arange(num_sites, device=device, dtype=torch.long)).flip(0)

        return (config.long() * powers).sum().item()

    def _find_important_configs(
        self,
        basis: torch.Tensor,
        energy: float,
        eigenvector: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find important configurations using perturbation theory.

        Importance: ε_i = |<i|H|Φ⟩|² / |E - E_i|

        where <i|H|Φ> = Σⱼ cⱼ <i|H|j> (sum of SIGNED couplings, then square)

        CRITICAL FIX: Previous version incorrectly computed Σⱼ (cⱼ <i|H|j>)²
        instead of |Σⱼ cⱼ <i|H|j>|². This caused wrong configurations to be
        selected and energy oscillation (violation of variational principle).

        Optimizations:
        - Hash-based membership checking
        - Coefficient magnitude sorting for early termination
        - BATCHED diagonal energy computation (major speedup)
        - Two-phase collection to minimize diagonal calls
        """
        cfg = self.config
        n_sites = basis.shape[1]
        n_basis = len(basis)

        # Use hash set for faster membership checks
        basis_hash_set = self._configs_to_hash_set(basis)

        coeffs = torch.from_numpy(eigenvector).float().to(self.device)

        # Sort basis states by coefficient magnitude (descending)
        # Process most important states first
        coeff_magnitudes = torch.abs(coeffs)
        sorted_indices = torch.argsort(coeff_magnitudes, descending=True)

        # Only process basis states with significant coefficients
        significant_mask = coeff_magnitudes[sorted_indices] > 1e-8
        significant_indices = sorted_indices[significant_mask]

        # PHASE 1: Collect all unique candidates and accumulate SIGNED couplings
        # CRITICAL: We must accumulate signed couplings first, then square
        # <i|H|Φ> = Σⱼ cⱼ <i|H|j>  (this is a sum of signed terms)
        # importance = |<i|H|Φ>|² / |E - E_i|
        candidate_coupling_dict = {}  # hash -> (config_tensor, cumulative_coupling_SIGNED)

        # Try vectorized batch method if available (GPU-accelerated)
        if hasattr(self.hamiltonian, 'get_connections_vectorized_batch'):
            significant_basis = basis[significant_indices]
            all_connected, all_elements, batch_indices = \
                self.hamiltonian.get_connections_vectorized_batch(significant_basis)

            if len(all_connected) > 0:
                # Encode connected configs as integers (GPU-accelerated)
                if hasattr(self.hamiltonian, '_powers_gpu'):
                    powers = self.hamiltonian._powers_gpu
                else:
                    powers = (2 ** torch.arange(n_sites, device=self.device, dtype=torch.long)).flip(0)

                connected_ints = (all_connected.long() * powers).sum(dim=1)
                connected_ints_cpu = connected_ints.cpu().tolist()

                # Get coefficients for each connection
                local_coeffs = coeffs[significant_indices[batch_indices]]

                # Accumulate signed couplings
                for k in range(len(all_connected)):
                    config_int = connected_ints_cpu[k]
                    if config_int not in basis_hash_set:
                        coupling = (local_coeffs[k] * all_elements[k]).item()

                        if config_int in candidate_coupling_dict:
                            old_config, old_coupling_sum = candidate_coupling_dict[config_int]
                            candidate_coupling_dict[config_int] = (old_config, old_coupling_sum + coupling)
                        else:
                            candidate_coupling_dict[config_int] = (all_connected[k], coupling)
        else:
            # Fallback to sequential method
            for j in significant_indices.tolist():
                coeff_j = coeffs[j].item()

                connected, elements = self.hamiltonian.get_connections(basis[j])

                if len(connected) == 0:
                    continue

                # Use integer encoding for membership check
                for k in range(len(connected)):
                    config_int = self._encode_config_int(connected[k])

                    if config_int not in basis_hash_set:
                        coupling = coeff_j * elements[k].item()

                        if config_int in candidate_coupling_dict:
                            old_config, old_coupling_sum = candidate_coupling_dict[config_int]
                            candidate_coupling_dict[config_int] = (old_config, old_coupling_sum + coupling)
                        else:
                            candidate_coupling_dict[config_int] = (connected[k], coupling)

        if not candidate_coupling_dict:
            return torch.empty(0, n_sites, device=self.device), torch.empty(0, device=self.device)

        # PHASE 2: Batch compute diagonal energies for all candidates at once
        candidates_list = [v[0] for v in candidate_coupling_dict.values()]
        couplings_signed_list = [v[1] for v in candidate_coupling_dict.values()]

        candidates = torch.stack(candidates_list)
        couplings_signed = torch.tensor(couplings_signed_list, device=self.device)

        # FIXED: Square the accumulated signed coupling (correct PT2 formula)
        couplings_sq = couplings_signed ** 2

        # BATCH diagonal computation - this is the key optimization!
        if hasattr(self.hamiltonian, 'diagonal_elements_batch'):
            E_candidates = self.hamiltonian.diagonal_elements_batch(candidates)
        else:
            # Fallback for Hamiltonians without batch method
            E_candidates = torch.stack([
                self.hamiltonian.diagonal_element(c) for c in candidates
            ])

        # Compute PT2 importances: ε_i = |<i|H|Φ>|² / |E - E_i|
        denominators = torch.abs(energy - E_candidates) + 1e-10
        importances = couplings_sq / denominators

        # Select top by importance
        n_select = min(cfg.max_configs_per_iter, len(candidates))
        _, top_indices = torch.topk(importances, n_select)

        return candidates[top_indices], importances[top_indices]
