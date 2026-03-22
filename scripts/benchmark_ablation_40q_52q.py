#!/usr/bin/env python3
"""Ablation Study: HI+NQS+SKQD on 24Q → 52Q.

消融實驗設計：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  實驗 A: 方法對比 (固定系統 = N₂ CAS(10,20) 40Q)
    A1: Krylov-only    — 只用 Krylov expansion from HF, 不用 NQS
    A2: NQS-only       — 只用 NQS sampling, 不用 Krylov expansion
    A3: NQS+SKQD       — 完整 pipeline (NQS + Krylov + 迭代)
  → 回答: NQS 和 Krylov 各自的貢獻是什麼？

  實驗 B: Scaling (固定方法 = NQS+SKQD)
    B1: N₂ CAS(10,12) 24Q
    B2: N₂ CAS(10,15) 30Q
    B3: N₂ CAS(10,20) 40Q
    B4: N₂ CAS(10,26) 52Q
  → 回答: 能量和時間如何隨 qubit 數 scaling？

  實驗 C: Krylov 深度 (固定系統 = 40Q, 固定方法 = NQS+SKQD)
    C1: krylov_max_new = 500
    C2: krylov_max_new = 1000
    C3: krylov_max_new = 2000
    C4: krylov_max_new = 5000
  → 回答: Krylov 擴展的 configs 數量對能量的影響？

控制變因:
  - seed = 42 (torch.manual_seed + np.random.seed)
  - n_samples = 5000 (NQS)
  - max_iterations = 15
  - convergence_threshold = 1e-5
  - nf_steps = 8
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys
import os
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np

from src.methods.hi_nqs_skqd import run_hi_nqs_skqd, HINQSSKQDConfig
from src.utils.gpu_diag import gpu_solve_fermion
from src.utils.krylov_expand import expand_basis_via_connections


SEED = 42


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_mol_info(H):
    return {
        "n_orbitals": H.n_orbitals, "n_alpha": H.n_alpha,
        "n_beta": H.n_beta, "n_qubits": H.num_sites,
        "nuclear_repulsion": H.nuclear_repulsion,
    }


def krylov_only(H, max_new):
    """A1: Krylov expansion from HF — no NQS."""
    set_seed()
    hf = H.get_hf_state().cpu().unsqueeze(0)
    t0 = time.perf_counter()
    expanded = expand_basis_via_connections(hf, H, max_new=max_new, n_ref=50)
    e, v0, occ = gpu_solve_fermion(expanded, H)
    t = time.perf_counter() - t0
    return {"energy": e, "basis_size": len(expanded), "time_s": t, "method": "Krylov-only"}


def nqs_only(H, n_samples, max_iterations):
    """A2: NQS sampling only — no Krylov expansion."""
    set_seed()
    cfg = HINQSSKQDConfig(
        max_iterations=max_iterations, n_samples=n_samples,
        krylov_max_new=0,  # Disable Krylov
        nf_steps=8, convergence_threshold=1e-5,
        max_basis_size=15000,
    )
    result = run_hi_nqs_skqd(H, make_mol_info(H), config=cfg)
    return {
        "energy": result.energy, "basis_size": result.diag_dim,
        "time_s": result.wall_time, "iterations": result.metadata["iterations"],
        "method": "NQS-only",
    }


def nqs_skqd(H, n_samples, krylov_max_new, max_iterations):
    """A3: Full NQS + SKQD pipeline."""
    set_seed()
    cfg = HINQSSKQDConfig(
        max_iterations=max_iterations, n_samples=n_samples,
        krylov_max_new=krylov_max_new, krylov_n_ref=50,
        nf_steps=8, convergence_threshold=1e-5,
        max_basis_size=15000,
    )
    result = run_hi_nqs_skqd(H, make_mol_info(H), config=cfg)
    return {
        "energy": result.energy, "basis_size": result.diag_dim,
        "time_s": result.wall_time, "iterations": result.metadata["iterations"],
        "converged": result.converged, "method": "NQS+SKQD",
    }


def hf_baseline(H):
    """HF energy baseline."""
    hf = H.get_hf_state().cpu()
    return float(H.diagonal_element(hf))


def run_experiment_a(H_40q, e_hf):
    """Experiment A: Method comparison on 40Q."""
    print("\n" + "=" * 70)
    print("  Experiment A: Method Comparison (N₂ CAS(10,20) 40Q)")
    print("=" * 70)

    results = []

    # A1: Krylov-only
    for max_new in [2000, 5000]:
        print(f"\n  A1: Krylov-only (max_new={max_new})...")
        r = krylov_only(H_40q, max_new)
        r["delta_hf_mha"] = (r["energy"] - e_hf) * 1000
        r["label"] = f"Krylov-only (max={max_new})"
        results.append(r)
        print(f"    E={r['energy']:.8f}, ΔE_HF={r['delta_hf_mha']:.2f} mHa, "
              f"basis={r['basis_size']}, t={r['time_s']:.1f}s")

    # A2: NQS-only
    print(f"\n  A2: NQS-only (5000 samples, 15 iters)...")
    r = nqs_only(H_40q, n_samples=5000, max_iterations=15)
    r["delta_hf_mha"] = (r["energy"] - e_hf) * 1000
    r["label"] = "NQS-only"
    results.append(r)
    print(f"    E={r['energy']:.8f}, ΔE_HF={r['delta_hf_mha']:.2f} mHa, "
          f"basis={r['basis_size']}, t={r['time_s']:.1f}s")

    # A3: NQS+SKQD
    print(f"\n  A3: NQS+SKQD (5000 samples + 2000 krylov, 15 iters)...")
    r = nqs_skqd(H_40q, n_samples=5000, krylov_max_new=2000, max_iterations=15)
    r["delta_hf_mha"] = (r["energy"] - e_hf) * 1000
    r["label"] = "NQS+SKQD"
    results.append(r)
    print(f"    E={r['energy']:.8f}, ΔE_HF={r['delta_hf_mha']:.2f} mHa, "
          f"basis={r['basis_size']}, t={r['time_s']:.1f}s")

    return results


def run_experiment_b():
    """Experiment B: Qubit scaling (24→52)."""
    from src.hamiltonians.molecular import create_n2_cas_hamiltonian

    print("\n" + "=" * 70)
    print("  Experiment B: Qubit Scaling (24→52)")
    print("=" * 70)

    systems = [
        ("24Q", (10, 12), 500),
        ("30Q", (10, 15), 1000),
        ("40Q", (10, 20), 2000),
        ("52Q", (10, 26), 3000),
    ]

    results = []
    for label, cas, krylov_max in systems:
        print(f"\n  B: N₂ CAS{cas} cc-pVDZ {label}...")
        H = create_n2_cas_hamiltonian(bond_length=1.10, basis="cc-pvdz", cas=cas)
        e_hf = hf_baseline(H)

        r = nqs_skqd(H, n_samples=5000, krylov_max_new=krylov_max, max_iterations=15)
        r["delta_hf_mha"] = (r["energy"] - e_hf) * 1000
        r["label"] = f"N₂ {label} CAS{cas}"
        r["n_qubits"] = H.num_sites
        r["hf_energy"] = e_hf
        results.append(r)
        print(f"    E={r['energy']:.8f}, ΔE_HF={r['delta_hf_mha']:.2f} mHa, "
              f"basis={r['basis_size']}, iters={r.get('iterations','?')}, t={r['time_s']:.1f}s")

    return results


def run_experiment_c(H_40q, e_hf):
    """Experiment C: Krylov depth on 40Q."""
    print("\n" + "=" * 70)
    print("  Experiment C: Krylov Depth (N₂ CAS(10,20) 40Q)")
    print("=" * 70)

    results = []
    for krylov_max in [500, 1000, 2000, 5000]:
        print(f"\n  C: krylov_max_new={krylov_max}...")
        r = nqs_skqd(H_40q, n_samples=5000, krylov_max_new=krylov_max, max_iterations=15)
        r["delta_hf_mha"] = (r["energy"] - e_hf) * 1000
        r["krylov_max_new"] = krylov_max
        r["label"] = f"krylov_max={krylov_max}"
        results.append(r)
        print(f"    E={r['energy']:.8f}, ΔE_HF={r['delta_hf_mha']:.2f} mHa, "
              f"basis={r['basis_size']}, iters={r.get('iterations','?')}, t={r['time_s']:.1f}s")

    return results


def main():
    from src.hamiltonians.molecular import create_n2_cas_hamiltonian

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("=" * 70)
    print("  Ablation Study: HI+NQS+SKQD (24Q → 52Q)")
    print(f"  Device: {device}")
    print(f"  Seed: {SEED}")
    print("=" * 70)

    # Build 40Q system (shared by Experiments A and C)
    H_40q = create_n2_cas_hamiltonian(bond_length=1.10, basis="cc-pvdz", cas=(10, 20))
    e_hf_40q = hf_baseline(H_40q)
    print(f"\n  40Q HF energy: {e_hf_40q:.6f}")

    all_results = {}

    # Experiment A
    all_results["experiment_A"] = run_experiment_a(H_40q, e_hf_40q)

    # Experiment B
    all_results["experiment_B"] = run_experiment_b()

    # Experiment C
    all_results["experiment_C"] = run_experiment_c(H_40q, e_hf_40q)

    # Summary tables
    print(f"\n\n{'='*80}")
    print("  ABLATION STUDY SUMMARY")
    print(f"{'='*80}")

    print("\n  ── Experiment A: Method Comparison (40Q) ──")
    print(f"  {'Method':<30} {'ΔE_HF (mHa)':>12} {'Basis':>7} {'Time':>8}")
    print(f"  {'-'*60}")
    for r in all_results["experiment_A"]:
        print(f"  {r['label']:<30} {r['delta_hf_mha']:>12.2f} {r['basis_size']:>7} {r['time_s']:>7.1f}s")

    print("\n  ── Experiment B: Qubit Scaling ──")
    print(f"  {'System':<25} {'Q':>3} {'ΔE_HF (mHa)':>12} {'Basis':>7} {'Iters':>5} {'Time':>8}")
    print(f"  {'-'*65}")
    for r in all_results["experiment_B"]:
        print(f"  {r['label']:<25} {r['n_qubits']:>3} {r['delta_hf_mha']:>12.2f} "
              f"{r['basis_size']:>7} {r.get('iterations','?'):>5} {r['time_s']:>7.1f}s")

    print("\n  ── Experiment C: Krylov Depth (40Q) ──")
    print(f"  {'krylov_max_new':>15} {'ΔE_HF (mHa)':>12} {'Basis':>7} {'Iters':>5} {'Time':>8}")
    print(f"  {'-'*50}")
    for r in all_results["experiment_C"]:
        print(f"  {r['krylov_max_new']:>15} {r['delta_hf_mha']:>12.2f} "
              f"{r['basis_size']:>7} {r.get('iterations','?'):>5} {r['time_s']:>7.1f}s")

    # Save
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ablation_40q_52q.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
