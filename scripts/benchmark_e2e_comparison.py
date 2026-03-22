#!/usr/bin/env python3
"""E2E Benchmark: HI+NQS+SKQD iterative pipeline on molecular systems.

Runs the full pipeline and reports energy, basis size, iterations, wall time.
"""

import sys
import os
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["PYTHONPATH"] = os.path.join(os.path.dirname(__file__), "..")

import torch
import numpy as np

from src.methods.hi_nqs_skqd import run_hi_nqs_skqd, HINQSSKQDConfig


def benchmark_system(name, H, cfg):
    """Run HI+NQS+SKQD on a single system."""
    mol_info = {
        "n_orbitals": H.n_orbitals, "n_alpha": H.n_alpha,
        "n_beta": H.n_beta, "n_qubits": H.num_sites,
        "nuclear_repulsion": H.nuclear_repulsion,
    }

    hf = H.get_hf_state().cpu()
    e_hf = float(H.diagonal_element(hf))
    fci = None
    try:
        fci = H.fci_energy()
    except Exception:
        pass

    print(f"\n{'='*60}")
    print(f"  {name} | {H.num_sites}Q | HF={e_hf:.6f}" +
          (f" | FCI={fci:.6f}" if fci else ""))
    print(f"{'='*60}")

    result = run_hi_nqs_skqd(H, mol_info, config=cfg)

    delta_hf = (result.energy - e_hf) * 1000 if result.energy else 0
    delta_fci = abs(result.energy - fci) * 1000 if (result.energy and fci) else None

    print(f"\n  Result: E={result.energy:.8f}")
    print(f"  ΔE_HF = {delta_hf:.2f} mHa")
    if delta_fci is not None:
        print(f"  ΔE_FCI = {delta_fci:.2f} mHa")
    print(f"  Basis = {result.diag_dim}, Iters = {result.metadata['iterations']}, "
          f"Time = {result.wall_time:.1f}s, Converged = {result.converged}")

    return {
        "system": name, "n_qubits": H.num_sites,
        "energy": result.energy, "hf_energy": e_hf, "fci_energy": fci,
        "delta_hf_mha": delta_hf,
        "delta_fci_mha": delta_fci,
        "basis_size": result.diag_dim,
        "iterations": result.metadata["iterations"],
        "wall_time_s": result.wall_time,
        "converged": result.converged,
        "energy_history": result.metadata["energy_history"],
    }


def main():
    from src.hamiltonians.molecular import (
        create_h2_hamiltonian, create_lih_hamiltonian,
        create_h2o_hamiltonian, create_beh2_hamiltonian,
    )

    print("=" * 60)
    print("  HI+NQS+SKQD E2E Iterative Pipeline Benchmark")
    device = "CUDA " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"  Device: {device}")
    print("=" * 60)

    systems = [
        ("H₂ (4Q)", create_h2_hamiltonian(0.74),
         HINQSSKQDConfig(max_iterations=15, n_samples=300, krylov_max_new=50, nf_steps=5)),
        ("LiH (12Q)", create_lih_hamiltonian(1.6),
         HINQSSKQDConfig(max_iterations=15, n_samples=1000, krylov_max_new=150, nf_steps=8)),
        ("H₂O (14Q)", create_h2o_hamiltonian(),
         HINQSSKQDConfig(max_iterations=15, n_samples=1000, krylov_max_new=200, nf_steps=8)),
        ("BeH₂ (14Q)", create_beh2_hamiltonian(),
         HINQSSKQDConfig(max_iterations=15, n_samples=1000, krylov_max_new=200, nf_steps=8)),
    ]

    # Try larger systems
    try:
        from src.hamiltonians.molecular import create_nh3_hamiltonian
        systems.append(("NH₃ (16Q)", create_nh3_hamiltonian(),
            HINQSSKQDConfig(max_iterations=15, n_samples=2000, krylov_max_new=300, nf_steps=8)))
    except Exception:
        pass

    try:
        from src.hamiltonians.molecular import create_n2_hamiltonian
        systems.append(("N₂ (20Q)", create_n2_hamiltonian(1.10),
            HINQSSKQDConfig(max_iterations=15, n_samples=3000, krylov_max_new=500, nf_steps=8)))
    except Exception:
        pass

    results = []
    for name, H, cfg in systems:
        try:
            r = benchmark_system(name, H, cfg)
            results.append(r)
        except Exception as ex:
            import traceback
            print(f"\n  {name}: FAILED ({ex})")
            traceback.print_exc()

    # Summary
    print(f"\n\n{'='*80}")
    print(f"  E2E SUMMARY: HI+NQS+SKQD Iterative Pipeline")
    print(f"{'='*80}")
    print(f"  {'System':<15} {'Q':>3} {'Basis':>7} {'Iters':>5} {'ΔE_HF':>10} {'ΔE_FCI':>10} {'Time':>7} {'Conv':>5}")
    print(f"  {'-'*70}")
    for r in results:
        fci_str = f"{r['delta_fci_mha']:.2f}" if r['delta_fci_mha'] is not None else "—"
        conv_str = "✓" if r["converged"] else "✗"
        print(f"  {r['system']:<15} {r['n_qubits']:>3} {r['basis_size']:>7} {r['iterations']:>5} "
              f"{r['delta_hf_mha']:>9.2f} {fci_str:>10} {r['wall_time_s']:>6.1f}s {conv_str:>5}")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "benchmark_e2e_hi_nqs_skqd.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
