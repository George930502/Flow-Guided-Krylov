"""
Sample-based Krylov Quantum Diagonalization (SKQD).

Method B: H-connection expansion from initial NF samples.
Method C: Subspace Krylov + iterative expansion.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from .base import Solver, SolverResult
from ..utils.config_hash import config_integer_hash


@dataclass
class SKQDConfig:
    """Configuration for SKQD solvers."""
    n_samples: int = 5000
    max_iterations: int = 20
    expansion_size: int = 500
    max_basis_size: int = 10000
    convergence_threshold: float = 1e-5  # Ha


class SKQDSolverB(Solver):
    """SKQD Method B: H-connection expansion.

    1. NF sample -> initial basis B₀
    2. Build projected H, diag -> |Φ⟩, E₀
    3. For configs in B₀: find H-connections via get_connections()
    4. Rank new configs by PT2 importance
    5. Add top-k, re-diag
    6. Repeat until convergence
    """

    def __init__(self, sampler, config: Optional[SKQDConfig] = None):
        self.sampler = sampler
        self.config = config or SKQDConfig()

    def solve(self, hamiltonian, mol_info: dict) -> SolverResult:
        t0 = time.time()
        cfg = self.config
        device = "cpu"

        # Step 1: Get initial basis from sampler
        sample_result = self.sampler.sample(cfg.n_samples)
        basis = sample_result.configs.to(device).long()

        # Filter particle number
        n_orb = hamiltonian.n_orbitals
        n_alpha = hamiltonian.n_alpha
        n_beta = hamiltonian.n_beta
        alpha_counts = basis[:, :n_orb].sum(dim=1)
        beta_counts = basis[:, n_orb:].sum(dim=1)
        valid = (alpha_counts == n_alpha) & (beta_counts == n_beta)
        basis = basis[valid]

        # Ensure HF is in basis
        hf = hamiltonian.get_hf_state().unsqueeze(0).to(device)
        basis = torch.cat([hf, basis], dim=0)
        basis = torch.unique(basis, dim=0)

        # Track basis hashes for dedup
        basis_hashes = set(config_integer_hash(basis))

        prev_energy = float("inf")
        converged = False

        for iteration in range(cfg.max_iterations):
            if len(basis) > cfg.max_basis_size:
                break

            # Build projected Hamiltonian and diagonalize
            H_proj = hamiltonian.matrix_elements_fast(basis)
            H_np = H_proj.cpu().numpy().astype(np.float64)
            H_np = 0.5 * (H_np + H_np.T)

            if len(H_np) <= 2000:
                eigenvalues, eigenvectors = np.linalg.eigh(H_np)
            else:
                from scipy.sparse.linalg import eigsh
                from scipy.sparse import csr_matrix
                eigenvalues, eigenvectors = eigsh(csr_matrix(H_np), k=1, which="SA")

            e0 = float(eigenvalues[0])
            psi0 = eigenvectors[:, 0]

            # Check convergence
            delta_e = abs(e0 - prev_energy)
            if delta_e < cfg.convergence_threshold and iteration > 0:
                converged = True
                break
            prev_energy = e0

            # Find new configs via H-connections
            new_configs = []
            new_importance = []

            # Process configs with largest coefficients
            sorted_idx = np.argsort(np.abs(psi0))[::-1]
            n_process = min(len(sorted_idx), 200)

            for idx in sorted_idx[:n_process]:
                c_i = psi0[idx]
                if abs(c_i) < 1e-8:
                    continue

                connected, elements = hamiltonian.get_connections(basis[idx])
                if len(connected) == 0:
                    continue

                conn_hashes = config_integer_hash(connected)
                for k, h in enumerate(conn_hashes):
                    if h not in basis_hashes:
                        # PT2 importance: |c_i * H_{x',x_i}|^2 / |E0 - H_{x'x'}|
                        h_elem = float(elements[k])
                        coupling = c_i * h_elem
                        h_xx = float(hamiltonian.diagonal_element(connected[k]))
                        denom = abs(e0 - h_xx) + 1e-12
                        importance = coupling ** 2 / denom

                        new_configs.append(connected[k])
                        new_importance.append(importance)
                        basis_hashes.add(h)

            if not new_configs:
                converged = True
                break

            # Add top-k most important new configs
            importance_arr = np.array(new_importance)
            top_k = min(cfg.expansion_size, len(new_configs))
            top_indices = np.argsort(importance_arr)[-top_k:]

            new_batch = torch.stack([new_configs[i] for i in top_indices])
            basis = torch.cat([basis, new_batch], dim=0)

        wall_time = time.time() - t0

        return SolverResult(
            energy=e0 if 'e0' in dir() else None,
            diag_dim=len(basis),
            wall_time=wall_time,
            method="NF-SKQD-B",
            converged=converged,
            metadata={
                "iterations": iteration + 1 if 'iteration' in dir() else 0,
                "final_basis_size": len(basis),
                "delta_e": delta_e if 'delta_e' in dir() else None,
            },
        )


class SKQDSolverC(Solver):
    """SKQD Method C: Subspace Krylov + expansion.

    1. NF sample -> initial basis B₀
    2. Build projected H_proj in B₀
    3. Compute ground state |Φ⟩ in subspace
    4. Identify "boundary" configs: those with large |c_i| whose H-connections go outside B₀
    5. Add highest-coupling external configs
    6. Rebuild H_proj, re-diag
    7. Repeat until convergence
    """

    def __init__(self, sampler, config: Optional[SKQDConfig] = None):
        self.sampler = sampler
        self.config = config or SKQDConfig()

    def solve(self, hamiltonian, mol_info: dict) -> SolverResult:
        t0 = time.time()
        cfg = self.config
        device = "cpu"

        # Step 1: Get initial basis from sampler
        sample_result = self.sampler.sample(cfg.n_samples)
        basis = sample_result.configs.to(device).long()

        # Filter particle number
        n_orb = hamiltonian.n_orbitals
        n_alpha = hamiltonian.n_alpha
        n_beta = hamiltonian.n_beta
        alpha_counts = basis[:, :n_orb].sum(dim=1)
        beta_counts = basis[:, n_orb:].sum(dim=1)
        valid = (alpha_counts == n_alpha) & (beta_counts == n_beta)
        basis = basis[valid]

        # Ensure HF is in basis
        hf = hamiltonian.get_hf_state().unsqueeze(0).to(device)
        basis = torch.cat([hf, basis], dim=0)
        basis = torch.unique(basis, dim=0)

        basis_hashes = set(config_integer_hash(basis))
        prev_energy = float("inf")
        converged = False

        for iteration in range(cfg.max_iterations):
            if len(basis) > cfg.max_basis_size:
                break

            # Build projected Hamiltonian and diagonalize
            H_proj = hamiltonian.matrix_elements_fast(basis)
            H_np = H_proj.cpu().numpy().astype(np.float64)
            H_np = 0.5 * (H_np + H_np.T)

            if len(H_np) <= 2000:
                eigenvalues, eigenvectors = np.linalg.eigh(H_np)
            else:
                from scipy.sparse.linalg import eigsh
                from scipy.sparse import csr_matrix
                eigenvalues, eigenvectors = eigsh(csr_matrix(H_np), k=1, which="SA")

            e0 = float(eigenvalues[0])
            psi0 = eigenvectors[:, 0]

            # Check convergence
            delta_e = abs(e0 - prev_energy)
            if delta_e < cfg.convergence_threshold and iteration > 0:
                converged = True
                break
            prev_energy = e0

            # Identify boundary configs: those with significant amplitude
            # whose H-connections extend outside the current basis
            sorted_idx = np.argsort(np.abs(psi0))[::-1]
            n_boundary = min(len(sorted_idx), 100)

            # Collect external connections with coupling scores
            external_configs = []
            external_scores = []

            for idx in sorted_idx[:n_boundary]:
                c_i = abs(psi0[idx])
                if c_i < 1e-8:
                    continue

                connected, elements = hamiltonian.get_connections(basis[idx])
                if len(connected) == 0:
                    continue

                conn_hashes = config_integer_hash(connected)
                for k, h in enumerate(conn_hashes):
                    if h not in basis_hashes:
                        # Coupling score: |c_i| * |H_{x',x_i}|
                        score = c_i * abs(float(elements[k]))
                        external_configs.append(connected[k])
                        external_scores.append(score)
                        basis_hashes.add(h)

            if not external_configs:
                converged = True
                break

            # Add top-k external configs by coupling score
            scores_arr = np.array(external_scores)
            top_k = min(cfg.expansion_size, len(external_configs))
            top_indices = np.argsort(scores_arr)[-top_k:]

            new_batch = torch.stack([external_configs[i] for i in top_indices])
            basis = torch.cat([basis, new_batch], dim=0)

        wall_time = time.time() - t0

        return SolverResult(
            energy=e0 if 'e0' in dir() else None,
            diag_dim=len(basis),
            wall_time=wall_time,
            method="NF-SKQD-C",
            converged=converged,
            metadata={
                "iterations": iteration + 1 if 'iteration' in dir() else 0,
                "final_basis_size": len(basis),
                "delta_e": delta_e if 'delta_e' in dir() else None,
            },
        )
