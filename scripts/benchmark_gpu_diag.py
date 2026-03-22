#!/usr/bin/env python3
"""Benchmark: gpu_solve_fermion vs IBM solve_fermion.

Compares energy accuracy, wall time, and occupancies across molecular systems.
"""

import sys
import os
import time
import json

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from itertools import combinations
from utils.gpu_diag import gpu_solve_fermion
from utils.format_utils import configs_to_ibm_format, ibm_format_to_configs

# IBM's solve_fermion
from qiskit_addon_sqd.fermion import solve_fermion


def build_full_basis(H):
    """Build full particle-conserving basis."""
    n_orb = H.n_orbitals
    configs = []
    for a_occ in combinations(range(n_orb), H.n_alpha):
        for b_occ in combinations(range(n_orb), H.n_beta):
            c = torch.zeros(H.num_sites, dtype=torch.long)
            for i in a_occ:
                c[i] = 1
            for i in b_occ:
                c[i + n_orb] = 1
            configs.append(c)
    return torch.stack(configs)


def benchmark_system(name, H, max_basis=None):
    """Benchmark one molecular system."""
    n_orb = H.n_orbitals
    n_qubits = H.num_sites
    nuclear_repulsion = H.nuclear_repulsion if hasattr(H, "nuclear_repulsion") else 0.0

    # Build basis (use fixed seed for reproducibility)
    basis = build_full_basis(H)
    if max_basis and len(basis) > max_basis:
        gen = torch.Generator().manual_seed(42)
        idx = torch.randperm(len(basis), generator=gen)[:max_basis]
        basis = basis[idx]

    n_configs = len(basis)
    fci = H.fci_energy() if hasattr(H, "fci_energy") else None

    # Extract integrals for IBM solve_fermion
    hcore = H.h1e.cpu().numpy() if hasattr(H.h1e, "cpu") else np.array(H.h1e)
    eri = H.h2e.cpu().numpy() if hasattr(H.h2e, "cpu") else np.array(H.h2e)

    # Convert to IBM format
    ibm_bs = configs_to_ibm_format(basis.numpy(), n_orb, n_qubits)

    print(f"\n{'='*60}")
    print(f"  {name} | {n_qubits}Q | {n_configs:,} configs")
    print(f"{'='*60}")

    # ── IBM solve_fermion ──
    t0 = time.perf_counter()
    try:
        e_ibm, sci_state, occ_ibm, spin_sq = solve_fermion(ibm_bs, hcore, eri, spin_sq=0)
        e_ibm_total = e_ibm + nuclear_repulsion
        t_ibm = time.perf_counter() - t0
        ibm_ok = True
    except Exception as ex:
        e_ibm_total = None
        t_ibm = time.perf_counter() - t0
        occ_ibm = None
        ibm_ok = False
        print(f"  IBM solve_fermion FAILED: {ex}")

    # ── GPU solve_fermion ──
    t0 = time.perf_counter()
    e_gpu, v0_gpu, occ_gpu = gpu_solve_fermion(basis, H)
    t_gpu = time.perf_counter() - t0

    # ── Results ──
    result = {
        "system": name,
        "n_qubits": n_qubits,
        "n_configs": n_configs,
        "fci_energy": fci,
    }

    print(f"\n  {'Method':<25} {'Energy (Ha)':>16} {'Error (mHa)':>12} {'Time (s)':>10}")
    print(f"  {'-'*65}")

    if fci:
        print(f"  {'FCI (reference)':<25} {fci:>16.8f} {'0.000':>12} {'—':>10}")

    if ibm_ok:
        err_ibm = abs(e_ibm_total - fci) * 1000 if fci else None
        print(f"  {'IBM solve_fermion':<25} {e_ibm_total:>16.8f} {err_ibm:>12.4f} {t_ibm:>10.3f}")
        result["ibm_energy"] = e_ibm_total
        result["ibm_time_s"] = t_ibm
        result["ibm_error_mha"] = err_ibm
    else:
        result["ibm_energy"] = None
        result["ibm_time_s"] = t_ibm

    err_gpu = abs(e_gpu - fci) * 1000 if fci else None
    print(f"  {'GPU gpu_solve_fermion':<25} {e_gpu:>16.8f} {err_gpu:>12.4f} {t_gpu:>10.3f}")
    result["gpu_energy"] = e_gpu
    result["gpu_time_s"] = t_gpu
    result["gpu_error_mha"] = err_gpu

    # Speedup
    if ibm_ok:
        speedup = t_ibm / t_gpu if t_gpu > 0 else float("inf")
        print(f"\n  Speedup: {speedup:.1f}x")
        result["speedup"] = speedup

        # Energy match
        energy_diff = abs(e_gpu - e_ibm_total) * 1000
        print(f"  Energy diff (GPU vs IBM): {energy_diff:.6f} mHa")
        result["energy_diff_mha"] = energy_diff

        # Occupancy match (IBM returns tuple(alpha, beta), GPU also returns tuple)
        if occ_ibm is not None and isinstance(occ_ibm, tuple) and isinstance(occ_gpu, tuple):
            alpha_diff = np.max(np.abs(occ_gpu[0] - occ_ibm[0]))
            beta_diff = np.max(np.abs(occ_gpu[1] - occ_ibm[1]))
            occ_diff = max(alpha_diff, beta_diff)
            print(f"  Max occ diff: {occ_diff:.6f} (α={alpha_diff:.6f}, β={beta_diff:.6f})")
            result["max_occ_diff"] = float(occ_diff)

    return result


def main():
    from hamiltonians.molecular import (
        create_h2_hamiltonian,
        create_lih_hamiltonian,
        create_h2o_hamiltonian,
        create_beh2_hamiltonian,
    )

    systems = [
        ("H2 (4Q)", create_h2_hamiltonian(bond_length=0.74), None),
        ("LiH (12Q)", create_lih_hamiltonian(bond_length=1.6), None),
        ("H2O (14Q)", create_h2o_hamiltonian(), None),
        ("BeH2 (14Q)", create_beh2_hamiltonian(), None),
    ]

    # Try larger systems if available
    try:
        from hamiltonians.molecular import create_nh3_hamiltonian
        systems.append(("NH3 (16Q)", create_nh3_hamiltonian(), None))
    except Exception:
        pass

    try:
        from hamiltonians.molecular import create_ch4_hamiltonian
        systems.append(("CH4 (18Q)", create_ch4_hamiltonian(), 5000))
    except Exception:
        pass

    try:
        from hamiltonians.molecular import create_n2_hamiltonian
        systems.append(("N2 (20Q)", create_n2_hamiltonian(), 5000))
    except Exception:
        pass

    print("=" * 60)
    print("  GPU Diag vs IBM solve_fermion Benchmark")
    print(f"  Device: {'CUDA ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)

    results = []
    for name, H, max_basis in systems:
        try:
            r = benchmark_system(name, H, max_basis)
            results.append(r)
        except Exception as ex:
            import traceback
            print(f"\n  {name}: FAILED ({ex})")
            traceback.print_exc()

    # Summary table
    print(f"\n\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")
    print(f"  {'System':<15} {'Q':>3} {'Configs':>8} {'IBM (s)':>8} {'GPU (s)':>8} {'Speedup':>8} {'ΔE (mHa)':>10}")
    print(f"  {'-'*70}")
    for r in results:
        ibm_t = f"{r.get('ibm_time_s', 0):.3f}" if r.get("ibm_energy") else "FAIL"
        gpu_t = f"{r['gpu_time_s']:.3f}"
        speedup = f"{r.get('speedup', 0):.1f}x" if r.get("speedup") else "—"
        diff = f"{r.get('energy_diff_mha', 0):.6f}" if r.get("energy_diff_mha") is not None else "—"
        print(f"  {r['system']:<15} {r['n_qubits']:>3} {r['n_configs']:>8,} {ibm_t:>8} {gpu_t:>8} {speedup:>8} {diff:>10}")

    # Save JSON
    out_path = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_path, exist_ok=True)
    json_path = os.path.join(out_path, "gpu_diag_benchmark.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {json_path}")


if __name__ == "__main__":
    main()
