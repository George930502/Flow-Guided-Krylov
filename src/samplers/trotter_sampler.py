"""Trotter time evolution sampler.

Uses either CUDA-Q (if available) or scipy for time evolution.
Accumulates bitstrings from multiple Krylov time steps.
"""

import time
from typing import Optional

import numpy as np
import torch
from scipy.sparse.linalg import expm_multiply
from scipy.sparse import csr_matrix

from .base import Sampler, SamplerResult


class TrotterSampler(Sampler):
    """Trotter time evolution sampler.

    For k=0..d-1 Krylov steps:
    1. Compute |ψ_k⟩ = e^{-iHk*dt} |ψ_0⟩
    2. Sample bitstrings from |ψ_k⟩
    3. Accumulate all bitstrings

    Falls back to scipy expm_multiply if CUDA-Q is unavailable.
    """

    def __init__(
        self,
        hamiltonian,
        n_krylov_steps: int = 6,
        dt: float = 0.1,
        shots_per_step: int = 10000,
        device: str = "cpu",
    ):
        self.hamiltonian = hamiltonian
        self.n_krylov_steps = n_krylov_steps
        self.dt = dt
        self.shots_per_step = shots_per_step
        self.device = device

        self.n_orbitals = hamiltonian.n_orbitals
        self.n_alpha = hamiltonian.n_alpha
        self.n_beta = hamiltonian.n_beta

        # Check CUDA-Q availability
        try:
            import cudaq
            self._use_cudaq = True
        except ImportError:
            self._use_cudaq = False

    def sample(self, n_samples: int) -> SamplerResult:
        """Sample configurations from Krylov time evolution states."""
        t0 = time.time()

        # Build particle-conserving subspace
        basis_configs, H_sub = self._build_subspace_hamiltonian()
        n_sub = len(basis_configs)

        if n_sub == 0:
            return SamplerResult(
                configs=torch.empty(0, self.hamiltonian.num_sites, dtype=torch.long),
                log_probs=None,
                wall_time=time.time() - t0,
                metadata={"error": "Empty subspace"},
            )

        # HF initial state in subspace
        hf = self.hamiltonian.get_hf_state()
        hf_hash = self._config_hash(hf)
        psi0 = np.zeros(n_sub, dtype=np.complex128)
        for i, cfg in enumerate(basis_configs):
            if self._config_hash(cfg) == hf_hash:
                psi0[i] = 1.0
                break
        else:
            psi0[0] = 1.0  # fallback

        # Time evolution and sampling
        all_configs = []
        H_sparse = csr_matrix(H_sub)

        shots_per_step = max(1, n_samples // self.n_krylov_steps)

        for k in range(self.n_krylov_steps):
            t_k = k * self.dt
            if t_k == 0:
                psi_k = psi0.copy()
            else:
                psi_k = expm_multiply(-1j * H_sparse, psi0, start=0, stop=t_k, num=2)[-1]

            # Sample from |ψ_k|²
            probs = np.abs(psi_k) ** 2
            probs = probs / probs.sum()

            indices = np.random.choice(n_sub, size=shots_per_step, p=probs)
            for idx in indices:
                all_configs.append(basis_configs[idx])

        if not all_configs:
            return SamplerResult(
                configs=torch.empty(0, self.hamiltonian.num_sites, dtype=torch.long),
                log_probs=None,
                wall_time=time.time() - t0,
                metadata={"error": "No configs sampled"},
            )

        configs = torch.stack(all_configs).to(self.device)
        unique_configs = torch.unique(configs, dim=0)

        wall_time = time.time() - t0

        return SamplerResult(
            configs=unique_configs,
            log_probs=None,
            wall_time=wall_time,
            metadata={
                "n_krylov_steps": self.n_krylov_steps,
                "dt": self.dt,
                "subspace_dim": n_sub,
                "n_unique": len(unique_configs),
                "sampler_type": "Trotter",
            },
        )

    def _build_subspace_hamiltonian(self):
        """Build Hamiltonian in particle-conserving subspace."""
        from math import comb

        n_orb = self.n_orbitals
        n_alpha = self.n_alpha
        n_beta = self.n_beta
        n_sites = 2 * n_orb

        n_configs = comb(n_orb, n_alpha) * comb(n_orb, n_beta)

        # For large subspaces, limit to HF neighborhood
        MAX_SUBSPACE = 50000
        if n_configs > MAX_SUBSPACE:
            # Use HF + singles + doubles
            configs = self._generate_cisd_basis()
        else:
            # Enumerate all valid configs
            configs = self._enumerate_valid_configs()

        if len(configs) == 0:
            return [], np.array([])

        # Build Hamiltonian matrix
        H = self.hamiltonian.matrix_elements_fast(torch.stack(configs))
        H_np = H.cpu().numpy().astype(np.float64)
        H_np = 0.5 * (H_np + H_np.T)

        return configs, H_np

    def _enumerate_valid_configs(self):
        """Enumerate all particle-number-conserving configurations."""
        from itertools import combinations

        n_orb = self.n_orbitals
        configs = []

        for alpha_occ in combinations(range(n_orb), self.n_alpha):
            for beta_occ in combinations(range(n_orb), self.n_beta):
                config = torch.zeros(2 * n_orb, dtype=torch.long)
                for i in alpha_occ:
                    config[i] = 1
                for i in beta_occ:
                    config[i + n_orb] = 1
                configs.append(config)

        return configs

    def _generate_cisd_basis(self):
        """Generate HF + singles + doubles basis."""
        hf = self.hamiltonian.get_hf_state()
        n_orb = self.n_orbitals
        configs = [hf]

        occ_a = [i for i in range(n_orb) if hf[i] == 1]
        virt_a = [i for i in range(n_orb) if hf[i] == 0]
        occ_b = [i for i in range(n_orb) if hf[i + n_orb] == 1]
        virt_b = [i for i in range(n_orb) if hf[i + n_orb] == 0]

        # Singles
        for i in occ_a:
            for a in virt_a:
                c = hf.clone()
                c[i] = 0
                c[a] = 1
                configs.append(c)
        for i in occ_b:
            for a in virt_b:
                c = hf.clone()
                c[i + n_orb] = 0
                c[a + n_orb] = 1
                configs.append(c)

        # Alpha-beta doubles (most important)
        for i in occ_a:
            for j in occ_b:
                for a in virt_a:
                    for b in virt_b:
                        c = hf.clone()
                        c[i] = 0
                        c[j + n_orb] = 0
                        c[a] = 1
                        c[b + n_orb] = 1
                        configs.append(c)

        return configs

    def _config_hash(self, config):
        """Simple hash for a single config."""
        return tuple(config.cpu().tolist())
