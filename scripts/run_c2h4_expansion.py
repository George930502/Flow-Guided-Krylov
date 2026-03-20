#!/usr/bin/env python3
"""
C2H4 (28Q): Test HI+NQS+SQD with two expansion strategies:
1. No PT2: add all singles/doubles from top-amplitude configs
2. With PT2: rank by PT2 importance, only keep top-k

Both use IBM's solve_fermion for SQD.
"""
import sys, numpy as np, torch, time
from math import comb
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.molecules import get_molecule
from src.nqs.transformer import AutoregressiveTransformer
from src.utils.config_hash import config_integer_hash
from qiskit_addon_sqd.fermion import solve_fermion
from qiskit_addon_sqd.configuration_recovery import recover_configurations

print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

H, info = get_molecule("C2H4")
n_orb = H.n_orbitals
n_alpha = H.n_alpha
n_beta = H.n_beta
n_qubits = 2 * n_orb
integrals = H.integrals
hcore = np.asarray(integrals.h1e, dtype=np.float64)
eri = np.asarray(integrals.h2e, dtype=np.float64)
nuc = float(integrals.nuclear_repulsion)

print(f"C2H4 (28Q, {n_orb} orb, Hilbert={comb(n_orb, n_alpha)*comb(n_orb, n_beta):,})", flush=True)
device = "cuda"


def configs_to_ibm(configs, n_orb, n_qubits):
    n = len(configs)
    bs = np.zeros((n, n_qubits), dtype=bool)
    for s in range(n):
        c = configs[s]
        for j in range(n_orb):
            bs[s, n_orb - 1 - j] = bool(c[j])
            bs[s, n_qubits - 1 - j] = bool(c[j + n_orb])
    return bs


def ibm_to_configs(bs, n_orb, n_qubits):
    n = len(bs)
    configs = torch.zeros(n, n_qubits, dtype=torch.long)
    for s in range(n):
        for j in range(n_orb):
            configs[s, j] = int(bs[s, n_orb - 1 - j])
            configs[s, j + n_orb] = int(bs[s, n_qubits - 1 - j])
    return configs


def expand_no_pt2(basis_configs, hamiltonian, top_k_configs=100, max_new=0):
    """Expand by adding ALL singles/doubles from top-amplitude configs. No PT2.
    max_new=0 means no limit."""
    basis_hashes = set(config_integer_hash(basis_configs))
    new_configs = []

    for idx in range(min(top_k_configs, len(basis_configs))):
        connected, elements = hamiltonian.get_connections(basis_configs[idx])
        if len(connected) == 0:
            continue
        conn_hashes = config_integer_hash(connected)
        for k, h in enumerate(conn_hashes):
            if h not in basis_hashes:
                new_configs.append(connected[k])
                basis_hashes.add(h)

    if new_configs:
        return torch.cat([basis_configs, torch.stack(new_configs)], dim=0)
    return basis_configs


def expand_with_pt2(basis_configs, psi0, e0, hamiltonian, top_k_configs=100, max_new=2000):
    """Expand by adding singles/doubles ranked by PT2 importance."""
    basis_hashes = set(config_integer_hash(basis_configs))
    candidates = []
    scores = []

    sorted_idx = np.argsort(np.abs(psi0))[::-1]

    for idx in sorted_idx[:top_k_configs]:
        c_i = psi0[idx]
        if abs(c_i) < 1e-8:
            continue
        connected, elements = hamiltonian.get_connections(basis_configs[idx])
        if len(connected) == 0:
            continue
        conn_hashes = config_integer_hash(connected)
        for k, h in enumerate(conn_hashes):
            if h not in basis_hashes:
                h_elem = float(elements[k])
                h_xx = float(hamiltonian.diagonal_element(connected[k]))
                denom = abs(e0 - h_xx) + 1e-12
                score = (c_i * h_elem) ** 2 / denom
                candidates.append(connected[k])
                scores.append(score)
                basis_hashes.add(h)

    if candidates:
        scores_arr = np.array(scores)
        top_k = min(max_new, len(candidates))
        top_indices = np.argsort(scores_arr)[-top_k:]
        new_batch = torch.stack([candidates[i] for i in top_indices])
        return torch.cat([basis_configs, new_batch], dim=0)
    return basis_configs


def run_hi_nqs_sqd_with_expansion(expansion_mode="none", max_iters=20, n_samples=20000,
                                   max_basis=10000, expansion_configs=100, expansion_new=2000):
    """Run HI+NQS+SQD with optional expansion."""
    t0 = time.time()

    nqs = AutoregressiveTransformer(
        n_orbitals=n_orb, n_alpha=n_alpha, n_beta=n_beta,
        embed_dim=192, n_heads=8, n_layers=6,
    ).to(device)
    optimizer = torch.optim.Adam(nqs.parameters(), lr=1e-3)

    hf = H.get_hf_state().unsqueeze(0)
    cumulative = hf.clone()
    cumulative_hashes = set(config_integer_hash(cumulative))

    energy_history = []
    best_energy = float("inf")
    prev_energy = float("inf")
    converge_count = 0

    for iteration in range(max_iters):
        iter_t0 = time.time()
        progress = iteration / max(max_iters - 1, 1)
        temperature = 1.0 + progress * (0.3 - 1.0)

        # Step 1: NQS sample
        with torch.no_grad():
            configs_gpu, _ = nqs.sample(n_samples, temperature=temperature)
            configs_cpu = configs_gpu.long().cpu()
            alpha_c = configs_cpu[:, :n_orb].sum(dim=1)
            beta_c = configs_cpu[:, n_orb:].sum(dim=1)
            valid = (alpha_c == n_alpha) & (beta_c == n_beta)
            new_configs = configs_cpu[valid]

        # Add to cumulative
        n_new = 0
        if len(new_configs) > 0:
            new_unique = torch.unique(new_configs, dim=0)
            new_hashes = config_integer_hash(new_unique)
            truly_new = []
            for idx, h in enumerate(new_hashes):
                if h not in cumulative_hashes:
                    truly_new.append(new_unique[idx])
                    cumulative_hashes.add(h)
            if truly_new:
                cumulative = torch.cat([cumulative, torch.stack(truly_new)], dim=0)
                n_new = len(truly_new)

        # Step 2: Expansion (if enabled)
        if expansion_mode != "none" and len(cumulative) >= 2:
            pre_expand = len(cumulative)

            if expansion_mode == "no_pt2":
                cumulative = expand_no_pt2(cumulative, H, expansion_configs, expansion_new)
            elif expansion_mode == "pt2" and iteration > 0:
                # Need eigenvector from previous SQD
                cumulative = expand_with_pt2(
                    cumulative, prev_psi0, prev_e0, H, expansion_configs, expansion_new
                )

            cumulative_hashes = set(config_integer_hash(cumulative))
            n_expanded = len(cumulative) - pre_expand
        else:
            n_expanded = 0

        # Enforce max basis
        if max_basis > 0 and len(cumulative) > max_basis:
            cumulative = cumulative[-max_basis:]
            cumulative_hashes = set(config_integer_hash(cumulative))

        # Step 3: SQD (IBM solve_fermion)
        bs_matrix = configs_to_ibm(cumulative, n_orb, n_qubits)
        batch_size = min(len(bs_matrix), max_basis)

        try:
            e, sci_state, occ, _ = solve_fermion(bs_matrix[:batch_size], hcore, eri, spin_sq=0)
            e0 = e + nuc
        except Exception as ex:
            print(f"    Iter {iteration}: SQD failed: {ex}", flush=True)
            continue

        # Save for PT2 expansion next iteration
        prev_e0 = e0
        # Get eigenvector for PT2 (need to diag ourselves for the coefficients)
        if expansion_mode == "pt2":
            H_proj = H.matrix_elements_fast(cumulative[:batch_size])
            H_np = H_proj.cpu().numpy().astype(np.float64)
            H_np = 0.5 * (H_np + H_np.T)
            _, eigvecs = np.linalg.eigh(H_np)
            prev_psi0 = eigvecs[:, 0]

        if e0 < best_energy:
            best_energy = e0
        energy_history.append(e0)

        # Step 4: Update NQS (mini-batch)
        configs_for_update = ibm_to_configs(bs_matrix[:min(5000, len(bs_matrix))], n_orb, n_qubits)
        with torch.no_grad():
            diag_e = H.diagonal_elements_batch(configs_for_update)
            diag_t = torch.tensor(np.asarray(diag_e, dtype=np.float64), dtype=torch.float32)
            advantage = diag_t - e0
            weights = torch.softmax(-advantage / max(abs(e0) * 0.01, 0.1), dim=0)

        for step in range(10):
            optimizer.zero_grad()
            batch_gpu = configs_for_update.float().to(device)
            lp = nqs.log_prob(batch_gpu)
            loss = -(weights.to(device) * lp).sum() + 0.05 * lp.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(nqs.parameters(), max_norm=1.0)
            optimizer.step()
            del batch_gpu, lp, loss
            torch.cuda.empty_cache()

        delta_e = abs(e0 - prev_energy)
        prev_energy = e0
        iter_time = time.time() - iter_t0

        print(f"    Iter {iteration:>3d}: E={e0:.10f}, basis={len(cumulative):>6d}"
              f"(+{n_new}+{n_expanded}exp), ΔE={delta_e:.2e}, t={iter_time:.1f}s", flush=True)

        if delta_e < 1e-6 and iteration > 0:
            converge_count += 1
        else:
            converge_count = 0
        if converge_count >= 3:
            break

    total_time = time.time() - t0
    return best_energy, len(cumulative), total_time, energy_history


# ============================================================
# Run 3 versions
# ============================================================

print(f"\n{'='*60}", flush=True)
print("Version 1: HI+NQS+SQD (no expansion)", flush=True)
print(f"{'='*60}", flush=True)
np.random.seed(42); torch.manual_seed(42)
e1, b1, t1, h1 = run_hi_nqs_sqd_with_expansion("none", max_iters=15, max_basis=10000)
print(f"  Result: E={e1:.10f}, basis={b1}, time={t1:.1f}s", flush=True)

print(f"\n{'='*60}", flush=True)
print("Version 2: HI+NQS+SQD + expansion (no PT2, all S/D)", flush=True)
print(f"{'='*60}", flush=True)
np.random.seed(42); torch.manual_seed(42)
e2, b2, t2, h2 = run_hi_nqs_sqd_with_expansion("no_pt2", max_iters=15, max_basis=10000,
                                                   expansion_configs=50, expansion_new=0)
print(f"  Result: E={e2:.10f}, basis={b2}, time={t2:.1f}s", flush=True)

print(f"\n{'='*60}", flush=True)
print("Version 3: HI+NQS+SQD + expansion (with PT2)", flush=True)
print(f"{'='*60}", flush=True)
np.random.seed(42); torch.manual_seed(42)
e3, b3, t3, h3 = run_hi_nqs_sqd_with_expansion("pt2", max_iters=15, max_basis=10000,
                                                   expansion_configs=50, expansion_new=1000)
print(f"  Result: E={e3:.10f}, basis={b3}, time={t3:.1f}s", flush=True)

# Summary
sci_e = -77.2351408123
print(f"\n{'='*60}", flush=True)
print(f"SUMMARY (C2H4 28Q, SCI ref = {sci_e:.10f})", flush=True)
print(f"{'Method':<35} {'Energy':>16} {'vs SCI (mHa)':>14} {'Basis':>8} {'Time':>8}", flush=True)
print(f"{'SCI':<35} {sci_e:>16.10f} {'0.000':>14} {'10000':>8} {'744':>8}", flush=True)
print(f"{'HI+NQS+SQD (no exp)':<35} {e1:>16.10f} {(e1-sci_e)*1000:>14.3f} {b1:>8} {t1:>8.1f}", flush=True)
print(f"{'HI+NQS+SQD + exp (no PT2)':<35} {e2:>16.10f} {(e2-sci_e)*1000:>14.3f} {b2:>8} {t2:>8.1f}", flush=True)
print(f"{'HI+NQS+SQD + exp (PT2)':<35} {e3:>16.10f} {(e3-sci_e)*1000:>14.3f} {b3:>8} {t3:>8.1f}", flush=True)
