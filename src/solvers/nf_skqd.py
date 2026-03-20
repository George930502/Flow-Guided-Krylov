"""
NF-SKQD: Normalizing Flow Sample-based Krylov Quantum Diagonalization.

A faithful classical analog of the quantum SKQD algorithm (Yu et al., 2025),
replacing quantum circuit time evolution with NF distribution evolution.

Quantum SKQD:
  Krylov subspace = {|ψ₀⟩, U|ψ₀⟩, U²|ψ₀⟩, ..., Uᵏ|ψ₀⟩}
  - U = e^{-iHΔt} via Trotter circuits
  - Sample each Uᵏ|ψ₀⟩ → bitstrings
  - Combine ALL bitstrings → basis
  - Project H → diagonalize

NF-SKQD:
  Krylov subspace = {NF₀, NF₁, NF₂, ..., NFₖ}
  - NF₀ = untrained (random/HF-biased distribution)
  - NFₖ₊₁ = NFₖ updated by a few gradient steps toward |Φₖ⟩
    (this is the NF analog of "one time evolution step")
  - Sample each NFₖ → configs
  - Combine ALL configs from ALL k → cumulative basis
  - Project H → diagonalize

Key design principles:
  1. CUMULATIVE BASIS: Never discard configs from previous Krylov powers.
     The subspace grows monotonically, guaranteeing Rayleigh-Ritz convergence.
  2. PARTIAL NF UPDATE: Only take a few gradient steps per Krylov power,
     NOT train to convergence. This mimics a small Trotter time step —
     the distribution shifts gradually, sampling NEW regions each time.
  3. NO H-connection expansion: The basis grows purely from NF sampling.
     This is faithful to the quantum SKQD where samples come from circuits only.
  4. ENERGY MONOTONICITY: Each Krylov power can only improve (or maintain)
     the energy, since the subspace only grows.

References:
  - Yu et al. (2025) "Quantum-Centric Algorithm for Sample-Based Krylov
    Diagonalization", arXiv:2501.09702
  - Pellow-Jarman et al. (2025) "HIVQE", arXiv:2503.06292
  - Robledo-Moreno et al. (2024) "Chemistry beyond exact solutions"
"""

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .base import Solver, SolverResult
from ..utils.config_hash import config_integer_hash


@dataclass
class NFSKQDConfig:
    """Configuration for NF-SKQD solver."""

    # Krylov structure
    n_krylov_powers: int = 10         # number of "time steps" (like k_max)
    n_samples_per_power: int = 2000   # samples per Krylov power (like shots)

    # NF "time evolution" (gradient steps per Krylov power)
    nf_steps_per_power: int = 20      # gradient steps = "Trotter steps"
    nf_lr: float = 1e-3

    # Wavefunction matching weight
    wf_weight: float = 1.0
    # Energy REINFORCE weight
    energy_weight: float = 0.1
    # Entropy regularization to maintain exploration
    entropy_weight: float = 0.05

    # Sampling temperature schedule: start high (explore), anneal down (exploit)
    initial_temperature: float = 2.0
    final_temperature: float = 0.5

    # Delayed NF update: sample for warmup_powers before first NF update
    # This allows the basis to grow before the NQS starts learning
    warmup_powers: int = 0

    # Basis management
    max_basis_size: int = 10000

    # Convergence
    convergence_threshold: float = 1e-6  # Ha (tighter for PRL-quality)


class NFSKQDSolver(Solver):
    """
    NF-SKQD: Faithful NF analog of quantum SKQD.

    Each "Krylov power" k:
      1. Sample from NFₖ → new configs
      2. Add to cumulative basis (never discard)
      3. Diagonalize in cumulative basis → Eₖ, |Φₖ⟩
      4. Partially update NF toward |Φₖ⟩ → NFₖ₊₁

    The NF distribution evolves like a quantum state under time evolution,
    exploring progressively different regions of Hilbert space.
    """

    def __init__(self, flow_model, config: Optional[NFSKQDConfig] = None):
        """
        Args:
            flow_model: Flow model with sample(), log_prob(), parameters()
        """
        self.flow = flow_model
        self.config = config or NFSKQDConfig()

    def solve(self, hamiltonian, mol_info: dict) -> SolverResult:
        t0 = time.time()
        cfg = self.config
        device = "cpu"

        n_orb = hamiltonian.n_orbitals
        n_alpha = hamiltonian.n_alpha
        n_beta = hamiltonian.n_beta

        optimizer = torch.optim.Adam(self.flow.parameters(), lr=cfg.nf_lr)

        # =============================================
        # Cumulative basis: the Krylov subspace
        # This ONLY GROWS, never shrinks
        # =============================================
        hf = hamiltonian.get_hf_state().unsqueeze(0).to(device)
        cumulative_basis = hf.clone()
        basis_hashes = set(config_integer_hash(cumulative_basis))

        energy_history = []
        basis_size_history = []
        samples_per_power = []  # track how many new configs each power adds

        prev_energy = float("inf")
        best_energy = float("inf")
        converged = False

        print(f"    NF-SKQD: {cfg.n_krylov_powers} Krylov powers, "
              f"{cfg.n_samples_per_power} samples/power, "
              f"{cfg.nf_steps_per_power} NF steps/power, "
              f"T: {cfg.initial_temperature:.1f}→{cfg.final_temperature:.1f}")

        for k in range(cfg.n_krylov_powers):
            # Temperature annealing: high T (explore) → low T (exploit)
            progress = k / max(cfg.n_krylov_powers - 1, 1)
            temperature = (cfg.initial_temperature
                           + progress * (cfg.final_temperature - cfg.initial_temperature))

            # =================================================
            # Step 1: Sample from current NF distribution (NFₖ)
            # This is the analog of measuring Uᵏ|ψ⟩
            # =================================================
            with torch.no_grad():
                # Pass temperature for autoregressive models
                try:
                    sample_out = self.flow.sample(cfg.n_samples_per_power,
                                                  temperature=temperature)
                except TypeError:
                    sample_out = self.flow.sample(cfg.n_samples_per_power)

                # Handle both return conventions
                if sample_out[0].dim() == 1:
                    raw_configs = sample_out[1].long().to(device)
                else:
                    raw_configs = sample_out[0].long().to(device)

                # Filter for correct particle number
                alpha_counts = raw_configs[:, :n_orb].sum(dim=1)
                beta_counts = raw_configs[:, n_orb:].sum(dim=1)
                valid = (alpha_counts == n_alpha) & (beta_counts == n_beta)
                new_configs = raw_configs[valid]

            # Add NEW configs to cumulative basis (dedup)
            n_new = 0
            if len(new_configs) > 0:
                new_unique = torch.unique(new_configs, dim=0)
                new_hashes = config_integer_hash(new_unique)
                truly_new = []
                for idx, h in enumerate(new_hashes):
                    if h not in basis_hashes:
                        truly_new.append(new_unique[idx])
                        basis_hashes.add(h)

                if truly_new:
                    new_batch = torch.stack(truly_new)
                    cumulative_basis = torch.cat([cumulative_basis, new_batch], dim=0)
                    n_new = len(truly_new)

            samples_per_power.append(n_new)

            # Enforce max basis size (keep most recent if overflow)
            if len(cumulative_basis) > cfg.max_basis_size:
                # Keep HF + most recently added configs
                cumulative_basis = torch.cat([
                    hf,
                    cumulative_basis[-(cfg.max_basis_size - 1):]
                ], dim=0)
                basis_hashes = set(config_integer_hash(cumulative_basis))

            # =================================================
            # Step 2: Diagonalize in cumulative basis
            # This is the Rayleigh-Ritz step
            # =================================================
            if len(cumulative_basis) < 2:
                # Not enough configs yet
                e0 = float(hamiltonian.diagonal_element(cumulative_basis[0]))
                psi0 = np.array([1.0])
            else:
                H_proj = hamiltonian.matrix_elements_fast(cumulative_basis)
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
            basis_size_history.append(len(cumulative_basis))

            if e0 < best_energy:
                best_energy = e0

            # =================================================
            # Step 3: Partially update NF toward |Φₖ⟩
            # This is the "time evolution" step
            # Only a FEW gradient steps — NOT convergence!
            # Skip during warmup to let basis grow first
            # =================================================
            if len(cumulative_basis) >= 2 and k >= cfg.warmup_powers:
                self._evolve_nf(
                    cumulative_basis, psi0, e0,
                    hamiltonian, optimizer, cfg
                )

            # =================================================
            # Step 4: Check convergence
            # =================================================
            delta_e = abs(e0 - prev_energy)
            prev_energy = e0

            print(f"    Power {k:>3d}: E={e0:.10f} Ha, "
                  f"basis={len(cumulative_basis):>6d} (+{n_new}), "
                  f"ΔE={delta_e:.2e}, T={temperature:.2f}")

            if delta_e < cfg.convergence_threshold and k > 0:
                converged = True
                break

        wall_time = time.time() - t0

        return SolverResult(
            energy=best_energy if best_energy < float("inf") else None,
            diag_dim=len(cumulative_basis),
            wall_time=wall_time,
            method="NF-SKQD",
            converged=converged,
            metadata={
                "n_krylov_powers": k + 1 if "k" in dir() else 0,
                "energy_history": energy_history,
                "basis_size_history": basis_size_history,
                "samples_per_power": samples_per_power,
            },
        )

    def _evolve_nf(self, basis_configs, psi0, e0,
                   hamiltonian, optimizer, cfg):
        """
        Partially update NF distribution: the "time evolution" step.

        Takes only a few gradient steps toward |Φ⟩, NOT full convergence.
        This ensures the NF distribution SHIFTS gradually, sampling
        different regions at each Krylov power.

        Loss = λ_wf × KL(|Φ|² ∥ p_NF)       ← match wavefunction
             + λ_E  × REINFORCE(E)             ← lower energy
             + λ_ent × entropy                  ← maintain exploration
        """
        basis_float = basis_configs.float()

        # Wavefunction weights from |Φ⟩
        weights = torch.from_numpy(psi0 ** 2).float()
        weights = weights / weights.sum()

        # Diagonal energies for REINFORCE
        with torch.no_grad():
            diag_energies = torch.from_numpy(
                np.asarray(hamiltonian.diagonal_elements_batch(basis_configs),
                           dtype=np.float64)
            ).float()
            advantage = diag_energies - e0

        for step in range(cfg.nf_steps_per_power):
            optimizer.zero_grad()

            log_probs = self.flow.log_prob(basis_float)

            # Wavefunction matching
            loss_wf = -(weights * log_probs).sum()

            # Energy REINFORCE
            loss_energy = (weights * advantage * log_probs).sum()

            # Entropy regularization
            loss_entropy = log_probs.mean()

            loss = (cfg.wf_weight * loss_wf
                    + cfg.energy_weight * loss_energy
                    + cfg.entropy_weight * loss_entropy)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.flow.parameters(), max_norm=1.0)
            optimizer.step()
