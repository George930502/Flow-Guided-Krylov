"""
Iterative NF-SQD: Neural Flow with SQD in a unified training loop.

Inspired by HI-VQE (Pellow-Jarman et al., 2025), but replaces the quantum
circuit with a classical normalizing flow. The key insight is that the NF
and SQD are trained together in a self-consistent loop:

    Loop:
      1. NF samples configurations
      2. (Optional) Expand basis via H-connections (SKQD-style)
      3. SQD builds projected Hamiltonian and diagonalizes → E₀, |Φ⟩
      4. Use |Φ⟩ to update NF weights (weighted MLE + energy gradient)
      5. Check convergence → repeat

This differs from the two-stage approach (train NF → then SQD) because:
- NF learns FROM SQD's ground state |Φ⟩, not just from Hamiltonian expectations
- SQD benefits from progressively better NF samples
- The feedback loop allows the NF to discover important configs that
  pure physics-guided training would miss

References:
- Pellow-Jarman et al. (2025) "HIVQE: Handover Iterative VQE", arXiv:2503.06292
- Robledo-Moreno et al. (2024) "Chemistry beyond exact solutions on a quantum-centric supercomputer"
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
import torch
import torch.nn.functional as F

from .base import Solver, SolverResult
from ..utils.config_hash import config_integer_hash

logger = logging.getLogger(__name__)


@dataclass
class IterativeNFSQDConfig:
    """Configuration for iterative NF-SQD solver."""

    # Outer loop
    max_outer_iterations: int = 20
    convergence_threshold: float = 1e-5  # Ha

    # NF sampling
    n_samples: int = 5000

    # SQD diagonalization
    max_basis_size: int = 5000

    # SKQD-style expansion (optional, per outer iteration)
    do_expansion: bool = True
    expansion_size: int = 200
    n_expansion_configs: int = 100  # top-amplitude configs to expand from

    # NF update
    nf_update_steps: int = 50       # gradient steps per outer iteration
    nf_lr: float = 1e-3
    wavefunction_weight: float = 1.0  # weight for |Φ⟩ matching loss
    energy_weight: float = 0.1        # weight for REINFORCE energy loss
    entropy_weight: float = 0.01      # weight for entropy regularization


class IterativeNFSQDSolver(Solver):
    """
    Iterative NF-SQD: unified training loop for NF + SQD.

    The NF and SQD improve each other iteratively:
    - NF provides better samples as it learns from |Φ⟩
    - SQD finds lower energies as the sample quality improves
    """

    def __init__(self, flow_model, config: Optional[IterativeNFSQDConfig] = None):
        """
        Args:
            flow_model: A flow model with:
                - sample(n) -> configs, log_probs
                - log_prob(configs) -> log_probs (differentiable)
                - parameters() -> iterable of parameters
            config: Solver configuration
        """
        self.flow = flow_model
        self.config = config or IterativeNFSQDConfig()

    def solve(self, hamiltonian, mol_info: dict) -> SolverResult:
        t0 = time.time()
        cfg = self.config
        device = "cpu"

        n_orb = hamiltonian.n_orbitals
        n_alpha = hamiltonian.n_alpha
        n_beta = hamiltonian.n_beta

        # Optimizer for NF parameters
        optimizer = torch.optim.Adam(self.flow.parameters(), lr=cfg.nf_lr)

        # Track history
        energy_history = []
        basis_size_history = []
        loss_history = []

        # Persistent basis: accumulate good configs across iterations
        hf = hamiltonian.get_hf_state().unsqueeze(0).to(device)
        persistent_basis = hf.clone()
        persistent_hashes = set(config_integer_hash(persistent_basis))

        prev_energy = float("inf")
        best_energy = float("inf")
        best_basis = None
        converged = False

        for outer_iter in range(cfg.max_outer_iterations):
            # =========================================================
            # Step 1: Sample from NF
            # =========================================================
            with torch.no_grad():
                sample_out = self.flow.sample(cfg.n_samples)
                # Handle both (configs, log_probs) and (log_probs, configs) conventions
                if sample_out[0].dim() == 1:
                    # (log_probs, configs) convention
                    configs = sample_out[1].long().to(device)
                else:
                    configs = sample_out[0].long().to(device)

                # Filter for correct particle number
                alpha_counts = configs[:, :n_orb].sum(dim=1)
                beta_counts = configs[:, n_orb:].sum(dim=1)
                valid = (alpha_counts == n_alpha) & (beta_counts == n_beta)
                new_configs = configs[valid]

            # Merge with persistent basis
            if len(new_configs) > 0:
                combined = torch.cat([persistent_basis, new_configs], dim=0)
                combined = torch.unique(combined, dim=0)
            else:
                combined = persistent_basis

            # Limit basis size
            if len(combined) > cfg.max_basis_size:
                combined = combined[:cfg.max_basis_size]

            # =========================================================
            # Step 2: (Optional) SKQD-style expansion
            # =========================================================
            if cfg.do_expansion and len(combined) > 1:
                combined, persistent_hashes = self._expand_basis(
                    combined, persistent_hashes, hamiltonian, cfg
                )

            # =========================================================
            # Step 3: SQD diagonalization
            # =========================================================
            H_proj = hamiltonian.matrix_elements_fast(combined)
            H_np = H_proj.cpu().numpy().astype(np.float64)
            H_np = 0.5 * (H_np + H_np.T)

            if len(H_np) <= 2000:
                eigenvalues, eigenvectors = np.linalg.eigh(H_np)
            else:
                from scipy.sparse.linalg import eigsh
                from scipy.sparse import csr_matrix
                eigenvalues, eigenvectors = eigsh(
                    csr_matrix(H_np), k=1, which="SA"
                )

            e0 = float(eigenvalues[0])
            psi0 = eigenvectors[:, 0]

            energy_history.append(e0)
            basis_size_history.append(len(combined))

            if e0 < best_energy:
                best_energy = e0
                best_basis = combined.clone()

            # Update persistent basis: keep configs with significant amplitude
            significant = np.abs(psi0) > 1e-6
            if significant.sum() > 0:
                persistent_basis = combined[significant]
                persistent_hashes = set(config_integer_hash(persistent_basis))
                # Always keep HF
                hf_hash = config_integer_hash(hf)[0]
                if hf_hash not in persistent_hashes:
                    persistent_basis = torch.cat([hf, persistent_basis], dim=0)
                    persistent_hashes.add(hf_hash)

            # =========================================================
            # Step 4: Update NF using |Φ⟩
            # =========================================================
            iter_losses = self._update_nf(
                combined, psi0, e0, hamiltonian, optimizer, cfg
            )
            loss_history.extend(iter_losses)

            # =========================================================
            # Step 5: Check convergence
            # =========================================================
            delta_e = abs(e0 - prev_energy)
            prev_energy = e0

            logger.info(
                f"Iter {outer_iter}: E={e0:.10f} Ha, "
                f"basis={len(combined)}, ΔE={delta_e:.2e}"
            )
            print(f"    Iter {outer_iter:>3d}: E={e0:.10f} Ha, "
                  f"basis={len(combined):>6d}, ΔE={delta_e:.2e}, "
                  f"loss={iter_losses[-1]:.4f}" if iter_losses else "")

            if delta_e < cfg.convergence_threshold and outer_iter > 0:
                converged = True
                break

        wall_time = time.time() - t0

        return SolverResult(
            energy=best_energy if best_energy < float("inf") else None,
            diag_dim=len(best_basis) if best_basis is not None else 0,
            wall_time=wall_time,
            method="Iter-NF-SQD",
            converged=converged,
            metadata={
                "outer_iterations": outer_iter + 1 if "outer_iter" in dir() else 0,
                "energy_history": energy_history,
                "basis_size_history": basis_size_history,
                "final_basis_size": len(best_basis) if best_basis is not None else 0,
            },
        )

    def _expand_basis(self, basis, basis_hashes, hamiltonian, cfg):
        """SKQD-style expansion: add H-connected configs."""
        # Quick diag to get amplitudes
        if len(basis) > 2000:
            return basis, basis_hashes

        H_proj = hamiltonian.matrix_elements_fast(basis)
        H_np = H_proj.cpu().numpy().astype(np.float64)
        H_np = 0.5 * (H_np + H_np.T)
        _, eigvecs = np.linalg.eigh(H_np)
        psi0 = eigvecs[:, 0]

        # Process top-amplitude configs
        sorted_idx = np.argsort(np.abs(psi0))[::-1]
        n_process = min(len(sorted_idx), cfg.n_expansion_configs)

        new_configs = []
        new_scores = []
        e0 = float(np.dot(psi0, H_np @ psi0))

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
                    h_elem = float(elements[k])
                    h_xx = float(hamiltonian.diagonal_element(connected[k]))
                    denom = abs(e0 - h_xx) + 1e-12
                    score = (c_i * h_elem) ** 2 / denom
                    new_configs.append(connected[k])
                    new_scores.append(score)
                    basis_hashes.add(h)

        if new_configs:
            scores_arr = np.array(new_scores)
            top_k = min(cfg.expansion_size, len(new_configs))
            top_indices = np.argsort(scores_arr)[-top_k:]
            new_batch = torch.stack([new_configs[i] for i in top_indices])
            basis = torch.cat([basis, new_batch], dim=0)

        # Enforce max size
        if len(basis) > cfg.max_basis_size:
            basis = basis[:cfg.max_basis_size]

        return basis, basis_hashes

    def _update_nf(self, basis_configs, psi0, e0, hamiltonian, optimizer, cfg):
        """
        Update NF parameters using SQD ground state |Φ⟩ AND energy E₀.

        Three loss components:
        1. Wavefunction matching: minimize KL(|Φ⟩² || p_NF)
           → teaches NF WHICH configs are important (from |Φ⟩ coefficients)
        2. Energy REINFORCE: E[(H_diag(x) - E₀) · log p_NF(x)]
           → teaches NF to LOWER the energy (from E₀)
        3. Entropy regularization: prevent distribution collapse
        """
        losses = []
        basis_configs_float = basis_configs.float()

        # From |Φ⟩: probability weights
        weights = torch.from_numpy(psi0 ** 2).float()
        weights = weights / weights.sum()

        # From E₀: compute diagonal energies for REINFORCE advantage
        with torch.no_grad():
            diag_energies = torch.from_numpy(
                np.asarray(hamiltonian.diagonal_elements_batch(basis_configs), dtype=np.float64)
            ).float()
            # Advantage: how much worse each config's diagonal energy is vs E₀
            # Configs with lower diagonal energy should be sampled MORE
            advantage = diag_energies - e0  # positive = bad, negative = good

        for step in range(cfg.nf_update_steps):
            optimizer.zero_grad()

            log_probs = self.flow.log_prob(basis_configs_float)

            # --- Loss 1: Wavefunction matching ---
            # Maximize Σ |c_i|² log p_NF(x_i)
            loss_wf = -(weights * log_probs).sum()

            # --- Loss 2: Energy REINFORCE ---
            # Minimize E_p[H_diag - E₀] ≈ Σ p_NF(x_i) · (H_ii - E₀)
            # Gradient: Σ (H_ii - E₀) · p_NF(x_i) · ∇log p_NF(x_i)
            # Using |Φ⟩² as importance weights instead of p_NF for stability
            loss_energy = (weights * advantage * log_probs).sum()

            # --- Loss 3: Entropy regularization ---
            loss_entropy = log_probs.mean()

            # --- Combined loss ---
            loss = (cfg.wavefunction_weight * loss_wf
                    + cfg.energy_weight * loss_energy
                    + cfg.entropy_weight * loss_entropy)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.flow.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach()))

        return losses


class IterativeNFSKQDSolver(Solver):
    """
    Iterative NF-SKQD: same as IterativeNFSQD but with stronger
    SKQD expansion at each iteration.

    This is the recommended variant — combines NF's global sampling
    with SKQD's systematic local expansion in a feedback loop.
    """

    def __init__(self, flow_model, config: Optional[IterativeNFSQDConfig] = None):
        self.flow = flow_model
        if config is None:
            config = IterativeNFSQDConfig(
                do_expansion=True,
                expansion_size=500,
                n_expansion_configs=200,
            )
        self.config = config

    def solve(self, hamiltonian, mol_info: dict) -> SolverResult:
        # Delegate to IterativeNFSQDSolver with expansion enabled
        inner = IterativeNFSQDSolver(self.flow, self.config)
        result = inner.solve(hamiltonian, mol_info)
        # Override method name
        return SolverResult(
            energy=result.energy,
            diag_dim=result.diag_dim,
            wall_time=result.wall_time,
            method="Iter-NF-SKQD",
            converged=result.converged,
            metadata=result.metadata,
        )
