#!/usr/bin/env python3
"""40Q Benchmark: HI+NQS+SKQD vs HI+NQS+SQD (GPU diag) on N₂ CAS systems.

Tests scaling from 20Q to 40Q with the new GPU diag + Krylov expansion.
"""

import sys
import os
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ["PYTHONPATH"] = os.path.join(os.path.dirname(__file__), "..")

import numpy as np
import torch

from utils.gpu_diag import gpu_solve_fermion
from utils.krylov_expand import expand_basis_via_connections
from utils.format_utils import configs_to_ibm_format, ibm_format_to_configs


def benchmark_krylov_expansion(name, H, max_new_list=[50, 100, 200, 500]):
    """Benchmark Krylov expansion from HF on a single system."""
    n_orb = H.n_orbitals
    n_qubits = H.num_sites

    hf = H.get_hf_state().cpu()
    e_hf = float(H.diagonal_element(hf))

    fci = None
    try:
        fci = H.fci_energy()
    except Exception:
        pass

    print(f"\n{'='*70}")
    print(f"  {name} | {n_qubits}Q | n_orb={n_orb} | HF={e_hf:.6f}")
    if fci:
        print(f"  FCI={fci:.6f}")
    print(f"{'='*70}")

    results = []

    # HF only baseline
    e_hf_diag, _, _ = gpu_solve_fermion(hf.unsqueeze(0), H)
    print(f"\n  {'Method':<35} {'Basis':>7} {'Energy (Ha)':>16} {'ΔE_HF (mHa)':>12} {'Time':>8}")
    print(f"  {'-'*80}")
    print(f"  {'HF only':<35} {1:>7} {e_hf_diag:>16.8f} {'0.000':>12} {'<0.1s':>8}")

    for max_new in max_new_list:
        t0 = time.perf_counter()
        expanded = expand_basis_via_connections(
            hf.unsqueeze(0), H, max_new=max_new, n_ref=50
        )
        t_expand = time.perf_counter() - t0

        t0 = time.perf_counter()
        e, v0, occ = gpu_solve_fermion(expanded, H)
        t_diag = time.perf_counter() - t0

        t_total = t_expand + t_diag
        delta_hf = (e - e_hf) * 1000

        label = f"Krylov expand (max_new={max_new})"
        print(f"  {label:<35} {len(expanded):>7} {e:>16.8f} {delta_hf:>12.2f} {t_total:>7.1f}s")

        r = {
            "method": label,
            "max_new": max_new,
            "basis_size": len(expanded),
            "energy": e,
            "delta_hf_mha": delta_hf,
            "expand_time_s": t_expand,
            "diag_time_s": t_diag,
            "total_time_s": t_total,
        }

        if fci:
            r["delta_fci_mha"] = abs(e - fci) * 1000
            r["recovery_pct"] = (e - e_hf) / (fci - e_hf) * 100 if fci != e_hf else 100

        results.append(r)

    return {
        "system": name,
        "n_qubits": n_qubits,
        "hf_energy": e_hf,
        "fci_energy": fci,
        "results": results,
    }


def main():
    from hamiltonians.molecular import (
        create_n2_hamiltonian,
        create_n2_cas_hamiltonian,
    )

    print("=" * 70)
    print("  40Q Benchmark: GPU Diag + Krylov Expansion")
    device = "CUDA " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"  Device: {device}")
    print("=" * 70)

    all_results = []

    # Tier 1: N₂ STO-3G 20Q (sanity check — should match our pipeline results)
    print("\n\n--- Tier 1: N₂ STO-3G 20Q ---")
    try:
        H = create_n2_hamiltonian(bond_length=1.10)
        r = benchmark_krylov_expansion("N₂ STO-3G (20Q)", H, [50, 100, 200, 500])
        all_results.append(r)
    except Exception as ex:
        print(f"  FAILED: {ex}")

    # Tier 2: N₂ CAS(10,12) cc-pVDZ 24Q
    print("\n\n--- Tier 2: N₂ CAS(10,12) cc-pVDZ 24Q ---")
    try:
        H = create_n2_cas_hamiltonian(bond_length=1.10, basis="cc-pvdz", cas=(10, 12))
        r = benchmark_krylov_expansion("N₂ CAS(10,12) (24Q)", H, [100, 500, 1000, 2000])
        all_results.append(r)
    except Exception as ex:
        print(f"  FAILED: {ex}")

    # Tier 3: N₂ CAS(10,15) cc-pVDZ 30Q
    print("\n\n--- Tier 3: N₂ CAS(10,15) cc-pVDZ 30Q ---")
    try:
        H = create_n2_cas_hamiltonian(bond_length=1.10, basis="cc-pvdz", cas=(10, 15))
        r = benchmark_krylov_expansion("N₂ CAS(10,15) (30Q)", H, [200, 500, 1000, 3000])
        all_results.append(r)
    except Exception as ex:
        print(f"  FAILED: {ex}")

    # Tier 4: N₂ CAS(10,20) cc-pVDZ 40Q — THE TARGET
    print("\n\n--- Tier 4: N₂ CAS(10,20) cc-pVDZ 40Q ---")
    try:
        H = create_n2_cas_hamiltonian(bond_length=1.10, basis="cc-pvdz", cas=(10, 20))
        r = benchmark_krylov_expansion("N₂ CAS(10,20) (40Q)", H, [500, 1000, 3000, 5000])
        all_results.append(r)
    except Exception as ex:
        print(f"  FAILED: {ex}")

    # Summary
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'System':<25} {'Q':>3} {'Best Basis':>10} {'ΔE_HF (mHa)':>12} {'Time':>8}")
    print(f"  {'-'*65}")
    for r in all_results:
        best = min(r["results"], key=lambda x: x["energy"])
        print(f"  {r['system']:<25} {r['n_qubits']:>3} {best['basis_size']:>10} "
              f"{best['delta_hf_mha']:>12.2f} {best['total_time_s']:>7.1f}s")

    # Save
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "benchmark_40q_krylov.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
