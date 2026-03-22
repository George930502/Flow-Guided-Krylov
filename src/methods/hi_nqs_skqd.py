"""
HI+NQS+SKQD: Iterative NQS sampling + Krylov expansion + GPU diag.

Replaces IBM's solve_fermion with:
1. NQS sampling (GPU) → configs
2. Krylov expansion from HF seed → discover H-connected configs
3. Post-Krylov merge: union(NQS configs, Krylov configs)
4. GPU sparse eigsh on merged basis → energy + eigenvector
5. |c_i|² from eigenvector → NQS training weights
"""

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from ..solvers.base import SolverResult
from ..nqs.transformer import AutoregressiveTransformer
from ..utils.format_utils import configs_to_ibm_format, ibm_format_to_configs, vectorized_dedup
from ..utils.gpu_diag import gpu_solve_fermion
from ..utils.krylov_expand import expand_basis_via_connections


@dataclass
class HINQSSKQDConfig:
    """Configuration for HI+NQS+SKQD."""
    max_iterations: int = 30
    convergence_threshold: float = 1e-6
    convergence_window: int = 3

    # NQS sampling
    n_samples: int = 5000
    max_basis_size: int = 0  # 0 = no limit

    # Krylov expansion
    krylov_max_new: int = 200  # Max new configs from H-connections per iteration
    krylov_n_ref: int = 50     # Reference configs for expansion

    # NQS update
    nf_steps: int = 10
    nf_lr: float = 1e-3

    # Loss weights
    wf_weight: float = 1.0
    energy_weight: float = 0.1
    entropy_weight: float = 0.05

    # Temperature
    initial_temperature: float = 1.0
    final_temperature: float = 0.3

    # Eigenvector weights for NQS training
    use_eigvec_weights: bool = True


def run_hi_nqs_skqd(hamiltonian, mol_info,
                    config: Optional[HINQSSKQDConfig] = None) -> SolverResult:
    """Run HI+NQS+SKQD: NQS sampling + Krylov expansion + GPU diag."""
    t0 = time.time()
    cfg = config or HINQSSKQDConfig()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_orb = hamiltonian.n_orbitals
    n_alpha = hamiltonian.n_alpha
    n_beta = hamiltonian.n_beta
    n_qubits = 2 * n_orb

    # Auto-scale transformer
    if n_orb <= 5:
        embed, heads, layers = 64, 4, 3
    elif n_orb <= 7:
        embed, heads, layers = 128, 4, 4
    elif n_orb <= 10:
        embed, heads, layers = 128, 8, 6
    elif n_orb <= 15:
        embed, heads, layers = 192, 8, 6
    elif n_orb <= 20:
        embed, heads, layers = 256, 8, 8
    else:
        embed, heads, layers = 256, 8, 10

    nqs = AutoregressiveTransformer(
        n_orbitals=n_orb, n_alpha=n_alpha, n_beta=n_beta,
        embed_dim=embed, n_heads=heads, n_layers=layers,
    ).to(device)

    optimizer = torch.optim.Adam(nqs.parameters(), lr=cfg.nf_lr)

    n_params = sum(p.numel() for p in nqs.parameters())
    print(f"    HI+NQS+SKQD (device={device}): arch={embed}/{heads}/{layers}, "
          f"params={n_params:,}, samples={cfg.n_samples}, krylov_expand={cfg.krylov_max_new}")

    # State
    energy_history = []
    basis_size_history = []
    prev_energy = float("inf")
    best_energy = float("inf")
    converged = False
    converge_count = 0
    iteration = -1

    # Cumulative configs in our format (torch.Tensor)
    cumulative_configs = None

    for iteration in range(cfg.max_iterations):
        iter_t0 = time.time()

        progress = iteration / max(cfg.max_iterations - 1, 1)
        temperature = (cfg.initial_temperature
                       + progress * (cfg.final_temperature - cfg.initial_temperature))

        # =====================================================
        # Step 1: Sample from NQS on GPU
        # =====================================================
        sample_t0 = time.time()
        with torch.no_grad():
            configs_gpu, _ = nqs.sample(cfg.n_samples, temperature=temperature)
            configs_cpu = configs_gpu.long().cpu()

            alpha_counts = configs_cpu[:, :n_orb].sum(dim=1)
            beta_counts = configs_cpu[:, n_orb:].sum(dim=1)
            valid = (alpha_counts == n_alpha) & (beta_counts == n_beta)
            new_configs = configs_cpu[valid]
        sample_time = time.time() - sample_t0

        # ── Accumulate NQS configs ──
        n_new_nqs = 0
        if len(new_configs) > 0:
            new_unique = torch.unique(new_configs, dim=0)
            if cumulative_configs is None:
                cumulative_configs = new_unique
                n_new_nqs = len(new_unique)
            else:
                # Dedup via IBM format (reuse vectorized_dedup)
                existing_bs = configs_to_ibm_format(cumulative_configs.numpy(), n_orb, n_qubits)
                new_bs = configs_to_ibm_format(new_unique.numpy(), n_orb, n_qubits)
                truly_new_bs = vectorized_dedup(existing_bs, new_bs)
                if len(truly_new_bs) > 0:
                    truly_new = ibm_format_to_configs(truly_new_bs, n_orb, n_qubits)
                    cumulative_configs = torch.cat([cumulative_configs, truly_new], dim=0)
                    n_new_nqs = len(truly_new)

        # Add HF if not present
        if cumulative_configs is None:
            cumulative_configs = hamiltonian.get_hf_state().cpu().unsqueeze(0)

        # =====================================================
        # Step 2: Krylov expansion from HF (post-Krylov merge)
        # =====================================================
        krylov_t0 = time.time()
        hf_seed = hamiltonian.get_hf_state().cpu().unsqueeze(0)
        if cfg.krylov_max_new > 0:
            krylov_configs = expand_basis_via_connections(
                hf_seed, hamiltonian,
                max_new=cfg.krylov_max_new,
                n_ref=cfg.krylov_n_ref,
            )
            # Post-Krylov merge: union(NQS, Krylov)
            merged = torch.unique(
                torch.cat([cumulative_configs.cpu(), krylov_configs.cpu()], dim=0),
                dim=0,
            )
        else:
            # NQS-only mode: no Krylov expansion
            krylov_configs = cumulative_configs[:0]  # empty
            merged = cumulative_configs.cpu()

        # Enforce max size
        if cfg.max_basis_size > 0 and len(merged) > cfg.max_basis_size:
            merged = merged[-cfg.max_basis_size:]

        krylov_time = time.time() - krylov_t0
        n_krylov = len(merged) - len(cumulative_configs)

        # =====================================================
        # Step 3: GPU diag on merged basis
        # =====================================================
        diag_t0 = time.time()
        try:
            e0, eigvec, occ = gpu_solve_fermion(merged, hamiltonian)
        except Exception as ex:
            print(f"    Iter {iteration:>3d}: GPU diag failed ({ex})")
            continue
        diag_time = time.time() - diag_t0

        if e0 < best_energy:
            best_energy = e0

        energy_history.append(e0)
        basis_size_history.append(len(merged))

        # =====================================================
        # Step 4: Update NQS using eigenvector feedback
        # =====================================================
        update_t0 = time.time()
        _update_nqs(nqs, optimizer, merged, e0, eigvec, hamiltonian, cfg, device)
        update_time = time.time() - update_t0

        # =====================================================
        # Step 5: Convergence check
        # =====================================================
        delta_e = abs(e0 - prev_energy)
        prev_energy = e0
        iter_time = time.time() - iter_t0

        if delta_e < cfg.convergence_threshold and iteration > 0:
            converge_count += 1
        else:
            converge_count = 0

        print(f"    Iter {iteration:>3d}: E={e0:.10f}, "
              f"basis={len(merged):>6d}(nqs+{n_new_nqs},kry+{max(0,n_krylov)}), "
              f"ΔE={delta_e:.2e}, "
              f"t={iter_time:.1f}s [samp={sample_time:.1f} kry={krylov_time:.1f} "
              f"diag={diag_time:.1f} upd={update_time:.1f}]")

        if converge_count >= cfg.convergence_window:
            converged = True
            break

    wall_time = time.time() - t0

    return SolverResult(
        energy=best_energy if best_energy < float("inf") else None,
        diag_dim=len(merged) if "merged" in locals() else 0,
        wall_time=wall_time,
        method="HI+NQS+SKQD",
        converged=converged,
        metadata={
            "iterations": iteration + 1,
            "energy_history": energy_history,
            "basis_size_history": basis_size_history,
            "device": device,
            "diag_mode": "gpu_diag+krylov",
        },
    )


def _update_nqs(nqs, optimizer, basis, e0, eigvec, hamiltonian, cfg, device):
    """Update NQS using eigenvector or diagonal energy feedback."""
    n_total = len(basis)

    with torch.no_grad():
        if cfg.use_eigvec_weights and eigvec is not None:
            weights = torch.from_numpy(np.abs(eigvec) ** 2).float()
            weights = weights / weights.sum()
        else:
            diag_e_raw = hamiltonian.diagonal_elements_batch(basis)
            if isinstance(diag_e_raw, torch.Tensor):
                diag_e_np = diag_e_raw.detach().cpu().numpy().astype(np.float64)
            else:
                diag_e_np = np.asarray(diag_e_raw, dtype=np.float64)
            diag_e_t = torch.tensor(diag_e_np, dtype=torch.float32)
            advantage_all = diag_e_t - e0
            weights = torch.softmax(-advantage_all / max(abs(e0) * 0.01, 0.1), dim=0)

        diag_e_raw = hamiltonian.diagonal_elements_batch(basis)
        if isinstance(diag_e_raw, torch.Tensor):
            diag_e_np = diag_e_raw.detach().cpu().numpy().astype(np.float64)
        else:
            diag_e_np = np.asarray(diag_e_raw, dtype=np.float64)
        diag_e_t = torch.tensor(diag_e_np, dtype=torch.float32)
        advantage = diag_e_t - e0

    configs_float = basis.float()
    max_batch = min(5000, n_total)

    for step in range(cfg.nf_steps):
        optimizer.zero_grad()

        if n_total > max_batch:
            idx = torch.randperm(n_total)[:max_batch]
            batch_configs = configs_float[idx].to(device)
            batch_weights = weights[idx].to(device)
            batch_weights = batch_weights / batch_weights.sum()
            batch_advantage = advantage[idx].to(device)
        else:
            batch_configs = configs_float.to(device)
            batch_weights = weights.to(device)
            batch_advantage = advantage.to(device)

        log_probs = nqs.log_prob(batch_configs)

        loss_wf = -(batch_weights * log_probs).sum()
        loss_energy = (batch_weights * batch_advantage * log_probs).sum()
        loss_entropy = log_probs.mean()

        loss = (cfg.wf_weight * loss_wf
                + cfg.energy_weight * loss_energy
                + cfg.entropy_weight * loss_entropy)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(nqs.parameters(), max_norm=1.0)
        optimizer.step()

        del batch_configs, log_probs, loss

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
