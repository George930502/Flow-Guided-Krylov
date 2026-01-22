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

    # Convergence criterion (energy change)
    energy_convergence: float = 1e-6

    # Maximum total basis size
    max_basis_size: int = 4096

    # Whether to use importance sampling for residual computation
    use_importance_sampling: bool = True

    # Number of samples for importance sampling
    n_importance_samples: int = 10000


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

        # Build set of basis configurations for quick lookup
        basis_set = self._configs_to_set(basis)

        # Collect candidate configurations from connections
        candidates = []
        candidate_residuals = []

        # Convert eigenvector to tensor
        coeffs = torch.from_numpy(eigenvector).float().to(self.device)

        # For each basis state, find its connections
        for j in range(n_basis):
            if abs(coeffs[j].item()) < 1e-10:
                continue

            connected, elements = self.hamiltonian.get_connections(basis[j])

            if len(connected) == 0:
                continue

            for k in range(len(connected)):
                config_tuple = tuple(connected[k].cpu().tolist())

                # Only consider configurations outside current basis
                if config_tuple not in basis_set:
                    # Residual contribution: c_j * <i|H|j>
                    residual = coeffs[j].item() * elements[k].item()

                    candidates.append(connected[k])
                    candidate_residuals.append(abs(residual))

        if not candidates:
            return torch.empty(0, n_sites, device=self.device), torch.empty(0, device=self.device)

        # Stack candidates
        candidates = torch.stack(candidates)
        residuals = torch.tensor(candidate_residuals, device=self.device)

        # Remove duplicates, keeping highest residual
        unique_candidates, inverse = torch.unique(candidates, dim=0, return_inverse=True)
        unique_residuals = torch.zeros(len(unique_candidates), device=self.device)

        for i, inv in enumerate(inverse):
            if residuals[i] > unique_residuals[inv]:
                unique_residuals[inv] = residuals[i]

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

    def _configs_to_set(self, configs: torch.Tensor) -> Set[tuple]:
        """Convert configurations to set of tuples for fast lookup."""
        return {tuple(c.cpu().tolist()) for c in configs}


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
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Expand basis using perturbation-based selection.

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
            return current_basis, {'configs_added': 0, 'energy': energy}

        # Add new configurations
        expanded_basis = torch.cat([current_basis, new_configs], dim=0)
        expanded_basis = torch.unique(expanded_basis, dim=0)

        # Rediagonalize
        new_energy, _ = self._diagonalize(expanded_basis)

        stats = {
            'initial_size': len(current_basis),
            'final_size': len(expanded_basis),
            'configs_added': len(new_configs),
            'initial_energy': energy,
            'final_energy': new_energy,
            'energy_improvement': energy - new_energy,
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

    def _find_important_configs(
        self,
        basis: torch.Tensor,
        energy: float,
        eigenvector: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find important configurations using perturbation theory.

        Importance: ε_i = |<i|H|Φ⟩|² / |E - E_i|

        where E_i is approximated by the diagonal element <i|H|i>.
        """
        cfg = self.config
        n_sites = basis.shape[1]

        basis_set = {tuple(c.cpu().tolist()) for c in basis}
        coeffs = torch.from_numpy(eigenvector).float().to(self.device)

        candidates = []
        importances = []

        # Collect candidates from connections
        for j in range(len(basis)):
            if abs(coeffs[j].item()) < 1e-10:
                continue

            connected, elements = self.hamiltonian.get_connections(basis[j])

            if len(connected) == 0:
                continue

            for k in range(len(connected)):
                config_tuple = tuple(connected[k].cpu().tolist())

                if config_tuple not in basis_set:
                    # Coupling: <i|H|Φ⟩ contribution from this basis state
                    coupling = coeffs[j].item() * elements[k].item()

                    # Diagonal energy of candidate
                    E_i = self.hamiltonian.diagonal_element(connected[k]).item()

                    # Perturbation importance
                    denominator = abs(energy - E_i) + 1e-10
                    importance = coupling**2 / denominator

                    candidates.append(connected[k])
                    importances.append(importance)

        if not candidates:
            return torch.empty(0, n_sites, device=self.device), torch.empty(0, device=self.device)

        candidates = torch.stack(candidates)
        importances = torch.tensor(importances, device=self.device)

        # Aggregate importances for duplicate candidates
        unique_candidates, inverse = torch.unique(candidates, dim=0, return_inverse=True)
        unique_importances = torch.zeros(len(unique_candidates), device=self.device)
        unique_importances.scatter_add_(0, inverse, importances)

        # Select top by importance
        n_select = min(cfg.max_configs_per_iter, len(unique_candidates))
        _, top_indices = torch.topk(unique_importances, n_select)

        return unique_candidates[top_indices], unique_importances[top_indices]
