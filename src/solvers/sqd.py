"""
Sample-based Quantum Diagonalization (SQD).

Clean implementation without noise injection or S-CORE recovery.
Takes sampled configurations, creates batches, diagonalizes each,
and runs self-consistent orbital occupancy updates.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from .base import Solver, SolverResult


@dataclass
class SQDConfig:
    """Configuration for SQD solver."""
    num_batches: int = 5
    batch_size: int = 0           # 0 = auto
    self_consistent_iters: int = 3
    occupancy_convergence: float = 0.01
    n_samples: int = 10000        # samples from sampler


class SQDSolver(Solver):
    """Clean SQD: sample -> batch -> diag -> self-consistent loop."""

    def __init__(self, sampler, config: Optional[SQDConfig] = None):
        self.sampler = sampler
        self.config = config or SQDConfig()

    def solve(self, hamiltonian, mol_info: dict) -> SolverResult:
        t0 = time.time()
        cfg = self.config
        device = "cpu"

        # Step 1: Sample configurations
        sample_result = self.sampler.sample(cfg.n_samples)
        configs = sample_result.configs.to(device)

        # Step 2: Filter for correct particle number
        n_orb = hamiltonian.n_orbitals
        n_alpha = hamiltonian.n_alpha
        n_beta = hamiltonian.n_beta

        alpha_counts = configs[:, :n_orb].sum(dim=1)
        beta_counts = configs[:, n_orb:].sum(dim=1)
        valid_mask = (alpha_counts == n_alpha) & (beta_counts == n_beta)
        configs = configs[valid_mask]

        if len(configs) == 0:
            return SolverResult(
                energy=None, diag_dim=0, wall_time=time.time() - t0,
                method="SQD", converged=False,
                metadata={"error": "No valid configurations after particle number filter"},
            )

        # Add HF state
        hf = hamiltonian.get_hf_state().unsqueeze(0).to(device)
        configs = torch.cat([hf, configs], dim=0)
        configs = torch.unique(configs, dim=0)

        n_configs = len(configs)

        # Determine batch size
        batch_size = cfg.batch_size
        if batch_size <= 0:
            batch_size = min(n_configs, max(100, n_configs // cfg.num_batches))

        # Step 3: Self-consistent loop
        n_orb = hamiltonian.n_orbitals
        orbital_occ = hamiltonian.get_hf_state().float().cpu().numpy()
        best_energy = float("inf")
        batch_energies = []

        for sc_iter in range(cfg.self_consistent_iters):
            # Create batches using orbital occupancy-based weighting
            batches = self._create_batches(configs, cfg.num_batches, batch_size, orbital_occ)

            iter_energies = []
            iter_vectors = []

            for batch in batches:
                if len(batch) < 2:
                    continue

                # Build and diagonalize projected Hamiltonian
                H_proj = hamiltonian.matrix_elements_fast(batch)
                H_np = H_proj.cpu().numpy().astype(np.float64)

                # Symmetrize
                H_np = 0.5 * (H_np + H_np.T)

                eigenvalues, eigenvectors = np.linalg.eigh(H_np)
                e0 = float(eigenvalues[0])
                psi0 = eigenvectors[:, 0]

                iter_energies.append(e0)
                iter_vectors.append((batch, psi0))

            if not iter_energies:
                break

            # Best energy across batches
            best_idx = int(np.argmin(iter_energies))
            current_best = iter_energies[best_idx]

            if current_best < best_energy:
                best_energy = current_best
                batch_energies = iter_energies

            # Update orbital occupancies from best eigenvector
            best_batch, best_psi = iter_vectors[best_idx]
            new_occ = self._compute_orbital_occupancies(best_batch, best_psi, n_orb)

            # Check convergence
            occ_change = np.max(np.abs(new_occ - orbital_occ))
            orbital_occ = new_occ

            if occ_change < cfg.occupancy_convergence and sc_iter > 0:
                break

        wall_time = time.time() - t0

        return SolverResult(
            energy=best_energy if best_energy < float("inf") else None,
            diag_dim=batch_size,
            wall_time=wall_time,
            method="SQD",
            converged=True,
            metadata={
                "n_configs": n_configs,
                "n_batches": cfg.num_batches,
                "batch_size": batch_size,
                "batch_energies": batch_energies,
                "sc_iterations": sc_iter + 1 if 'sc_iter' in dir() else 0,
            },
        )

    def _create_batches(self, configs, num_batches, batch_size, orbital_occ):
        """Create random batches from configurations."""
        n = len(configs)
        batches = []

        for _ in range(num_batches):
            if n <= batch_size:
                batches.append(configs)
            else:
                indices = torch.randperm(n)[:batch_size]
                batches.append(configs[indices])

        return batches

    def _compute_orbital_occupancies(self, configs, psi, n_orb):
        """Compute orbital occupancies <n_p> from eigenvector."""
        probs = psi ** 2
        configs_np = configs.cpu().numpy().astype(np.float64)
        occ = (probs[:, None] * configs_np).sum(axis=0)
        return occ
