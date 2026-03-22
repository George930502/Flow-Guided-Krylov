"""
HI+NQS+SQD: Iterative self-consistent NQS-SQD loop — GPU accelerated.

NQS (Transformer autoregressive): torch on CUDA → sample configs
SQD backend: GPU diag (default) or IBM solve_fermion (fallback)
Feedback: eigenvector |c_i|² from diag updates NQS weights via backprop

This is the classical analog of HI-VQE (Pellow-Jarman et al., 2025),
replacing quantum circuits with NQS while using GPU-accelerated diag.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from ..solvers.base import SolverResult
from ..nqs.transformer import AutoregressiveTransformer
from ..utils.config_hash import config_integer_hash
from ..utils.format_utils import configs_to_ibm_format, ibm_format_to_configs, vectorized_dedup
from ..utils.gpu_diag import gpu_solve_fermion, compute_occupancies

# IBM SQD tools — optional, used as fallback when use_gpu_diag=False
try:
    from qiskit_addon_sqd.fermion import solve_fermion
    from qiskit_addon_sqd.configuration_recovery import recover_configurations
    IBM_SQD_AVAILABLE = True
except ImportError:
    IBM_SQD_AVAILABLE = False


@dataclass
class HINQSSQDConfig:
    """Configuration for HI+NQS+SQD."""
    max_iterations: int = 30
    convergence_threshold: float = 1e-6
    convergence_window: int = 3

    # NQS sampling
    n_samples: int = 5000
    max_basis_size: int = 0  # 0 = no limit

    # SQD batching (only used when use_gpu_diag=False)
    num_batches: int = 5
    samples_per_batch: int = 0   # 0 = auto

    # Configuration recovery
    configuration_recovery: bool = True

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

    # GPU diag (PR #1): replaces IBM solve_fermion with sparse eigsh
    use_gpu_diag: bool = True

    # Eigenvector weights: use |c_i|² instead of softmax(-H_ii) for NQS training
    use_eigvec_weights: bool = True


def run_hi_nqs_sqd(hamiltonian, mol_info,
                   config: Optional[HINQSSQDConfig] = None) -> SolverResult:
    """Run HI+NQS+SQD: NQS sampling (GPU) + diagonalization."""
    t0 = time.time()
    cfg = config or HINQSSQDConfig()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_orb = hamiltonian.n_orbitals
    n_alpha = hamiltonian.n_alpha
    n_beta = hamiltonian.n_beta
    n_qubits = 2 * n_orb

    # Molecular integrals (needed for IBM fallback and config recovery)
    integrals = hamiltonian.integrals
    hcore = np.asarray(integrals.h1e, dtype=np.float64)
    eri = np.asarray(integrals.h2e, dtype=np.float64)
    nuclear_repulsion = float(integrals.nuclear_repulsion)

    # Validate GPU diag availability
    if not cfg.use_gpu_diag and not IBM_SQD_AVAILABLE:
        print("    WARNING: use_gpu_diag=False but qiskit-addon-sqd not installed. "
              "Falling back to GPU diag.")
        cfg.use_gpu_diag = True

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
    diag_mode = "GPU-diag" if cfg.use_gpu_diag else "IBM-SQD"
    print(f"    HI+NQS+SQD ({diag_mode}, device={device}): arch={embed}/{heads}/{layers}, "
          f"params={n_params:,}, samples={cfg.n_samples}")

    # Auto batch size (only for IBM SQD mode)
    samples_per_batch = cfg.samples_per_batch
    if samples_per_batch <= 0:
        samples_per_batch = max(50, cfg.n_samples // cfg.num_batches)

    # State
    energy_history = []
    basis_size_history = []
    prev_energy = float("inf")
    best_energy = float("inf")
    converged = False
    converge_count = 0
    avg_occupancies = None
    best_eigvec = None  # Store eigenvector for NQS training

    # Cumulative basis: IBM format (bool ndarray) for config recovery compatibility
    cumulative_bs = None

    for iteration in range(cfg.max_iterations):
        iter_t0 = time.time()

        # Temperature schedule
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

        # ── Vectorized format conversion + dedup (PR #1) ──
        n_new = 0
        if len(new_configs) > 0:
            new_unique = torch.unique(new_configs, dim=0)
            new_bs = configs_to_ibm_format(new_unique, n_orb, n_qubits)

            truly_new = vectorized_dedup(cumulative_bs, new_bs)

            if len(truly_new) > 0:
                if cumulative_bs is None:
                    cumulative_bs = truly_new
                else:
                    cumulative_bs = np.concatenate([cumulative_bs, truly_new], axis=0)
                n_new = len(truly_new)

        # Add HF if not present
        if cumulative_bs is None:
            hf = hamiltonian.get_hf_state()
            hf_bs = configs_to_ibm_format(hf.unsqueeze(0), n_orb, n_qubits)
            cumulative_bs = hf_bs

        # Enforce max size (0 = no limit)
        if cfg.max_basis_size > 0 and len(cumulative_bs) > cfg.max_basis_size:
            cumulative_bs = cumulative_bs[-cfg.max_basis_size:]

        # =====================================================
        # Step 2: Configuration recovery (IBM's qiskit-addon-sqd)
        # =====================================================
        if cfg.configuration_recovery and avg_occupancies is not None and IBM_SQD_AVAILABLE:
            try:
                probs = np.ones(len(cumulative_bs)) / len(cumulative_bs)
                cumulative_bs, probs = recover_configurations(
                    cumulative_bs, probs,
                    avg_occupancies,
                    num_elec_a=n_alpha,
                    num_elec_b=n_beta,
                    rand_seed=iteration,
                )
            except Exception:
                pass

        # =====================================================
        # Step 3: Diagonalization
        # =====================================================
        sqd_t0 = time.time()

        if cfg.use_gpu_diag:
            # ── GPU diag: single full-basis diag (no batching needed) ──
            configs_tensor = ibm_format_to_configs(cumulative_bs, n_orb, n_qubits)
            try:
                e0, eigvec, avg_occupancies = gpu_solve_fermion(configs_tensor, hamiltonian)
                best_eigvec = eigvec
            except Exception as ex:
                print(f"    Iter {iteration:>3d}: GPU diag failed ({ex})")
                sqd_time = time.time() - sqd_t0
                continue
        else:
            # ── IBM solve_fermion: 5-batch mode (original) ──
            batch_energies = []
            batch_occs = []
            batch_states = []
            batch_size = min(samples_per_batch, len(cumulative_bs))

            for b in range(cfg.num_batches):
                if len(cumulative_bs) <= batch_size:
                    batch = cumulative_bs
                else:
                    idx = np.random.choice(len(cumulative_bs), size=batch_size, replace=False)
                    batch = cumulative_bs[idx]

                if len(batch) < 2:
                    continue

                try:
                    e, sci_state, occ, spin_sq = solve_fermion(
                        batch, hcore, eri, spin_sq=0,
                    )
                    e_total = e + nuclear_repulsion
                    batch_energies.append(e_total)
                    batch_occs.append(occ)
                    batch_states.append(sci_state)
                except Exception:
                    continue

            if not batch_energies:
                print(f"    Iter {iteration:>3d}: SQD failed (0/{cfg.num_batches} batches)")
                sqd_time = time.time() - sqd_t0
                continue

            best_batch_idx = int(np.argmin(batch_energies))
            e0 = batch_energies[best_batch_idx]
            avg_occupancies = batch_occs[best_batch_idx]
            best_eigvec = None  # IBM mode: no eigenvector available

        sqd_time = time.time() - sqd_t0

        if e0 < best_energy:
            best_energy = e0

        energy_history.append(e0)
        basis_size_history.append(len(cumulative_bs))

        # =====================================================
        # Step 4: Update NQS on GPU using diag results
        # =====================================================
        update_t0 = time.time()

        _update_nqs_from_sqd(
            nqs, optimizer, cumulative_bs, e0, best_eigvec,
            hamiltonian, cfg, device, n_orb, n_qubits,
        )
        update_time = time.time() - update_t0

        # =====================================================
        # Step 5: Check convergence
        # =====================================================
        delta_e = abs(e0 - prev_energy)
        prev_energy = e0
        iter_time = time.time() - iter_t0

        if delta_e < cfg.convergence_threshold and iteration > 0:
            converge_count += 1
        else:
            converge_count = 0

        print(f"    Iter {iteration:>3d}: E={e0:.10f}, "
              f"basis={len(cumulative_bs):>6d}(+{n_new}), "
              f"ΔE={delta_e:.2e}, "
              f"t={iter_time:.1f}s [sample={sample_time:.1f} diag={sqd_time:.1f} update={update_time:.1f}]")

        if converge_count >= cfg.convergence_window:
            converged = True
            break

    wall_time = time.time() - t0

    return SolverResult(
        energy=best_energy if best_energy < float("inf") else None,
        diag_dim=len(cumulative_bs) if cumulative_bs is not None else 0,
        wall_time=wall_time,
        method="HI+NQS+SQD",
        converged=converged,
        metadata={
            "iterations": iteration + 1 if "iteration" in dir() else 0,
            "energy_history": energy_history,
            "basis_size_history": basis_size_history,
            "device": device,
            "diag_mode": "gpu_diag" if cfg.use_gpu_diag else "ibm_sqd",
        },
    )


def _update_nqs_from_sqd(nqs, optimizer, cumulative_bs, e0, eigvec,
                         hamiltonian, cfg, device, n_orb, n_qubits):
    """Update NQS using diag feedback with mini-batching.

    When eigvec is available (GPU diag mode), uses |c_i|² as weights.
    Otherwise falls back to softmax(-diagonal_energy) weights.
    """
    configs = ibm_format_to_configs(cumulative_bs, n_orb, n_qubits)
    n_total = len(configs)

    with torch.no_grad():
        if cfg.use_eigvec_weights and eigvec is not None:
            # ── PR #1: Use eigenvector |c_i|² as weights ──
            # This is the NQS-SC distillation approach: exact ground state
            # probabilities from diag, strictly better than diagonal energy proxy.
            weights = torch.from_numpy(eigvec ** 2).float()
            weights = weights / weights.sum()

            # Advantage still uses diagonal energies for REINFORCE term
            diag_e = hamiltonian.diagonal_elements_batch(configs)
            diag_e_t = torch.tensor(np.asarray(diag_e, dtype=np.float64), dtype=torch.float32)
            advantage = diag_e_t - e0
        else:
            # ── Original: softmax(-advantage) weights ──
            diag_e = hamiltonian.diagonal_elements_batch(configs)
            diag_e_t = torch.tensor(np.asarray(diag_e, dtype=np.float64), dtype=torch.float32)
            advantage = diag_e_t - e0
            weights = torch.softmax(-advantage / max(abs(e0) * 0.01, 0.1), dim=0)

    max_batch = min(5000, n_total)

    for step in range(cfg.nf_steps):
        optimizer.zero_grad()

        if n_total > max_batch:
            idx = torch.randperm(n_total)[:max_batch]
            batch_configs = configs[idx].float().to(device)
            batch_weights = weights[idx].to(device)
            batch_weights = batch_weights / batch_weights.sum()
            batch_advantage = advantage[idx].to(device)
        else:
            batch_configs = configs.float().to(device)
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


# ── Legacy format conversion (kept for backward compatibility) ──
# New code should use utils.format_utils directly.

def _configs_to_ibm_format(configs, n_orb, n_qubits):
    """Convert our config tensors to IBM bitstring matrix (bool array).

    DEPRECATED: Use utils.format_utils.configs_to_ibm_format instead.
    """
    return configs_to_ibm_format(configs, n_orb, n_qubits)


def _ibm_format_to_configs(bs_matrix, n_orb, n_qubits):
    """Convert IBM bitstring matrix back to our config tensor format.

    DEPRECATED: Use utils.format_utils.ibm_format_to_configs instead.
    """
    return ibm_format_to_configs(bs_matrix, n_orb, n_qubits)
