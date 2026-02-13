"""
Subspace Method A/B Comparison: SKQD vs SQD.

Runs both subspace construction modes on the same molecular systems
and prints a side-by-side comparison table.

Approach 1 (SKQD - Krylov-based):
    NF-NQS generates initial configs, Krylov time evolution
    U^k = e^{-iHdt} expands subspace iteratively, union and diagonalize.

Approach 2 (SQD - Sampling-based):
    Following the IBM "Chemistry Beyond Exact Diagonalization" paper.
    NF-NQS replaces the quantum circuit as the sampler, then apply
    SQD's configuration recovery + batch diagonalization +
    self-consistent orbital occupancy loop.

Usage:
    docker-compose run --rm flow-krylov-gpu python examples/subspace_comparison.py
    docker-compose run --rm flow-krylov-gpu python examples/subspace_comparison.py --systems h2o beh2
    docker-compose run --rm flow-krylov-gpu python examples/subspace_comparison.py --systems h2 lih h2o beh2 nh3
"""

import sys
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np

from hamiltonians.molecular import (
    create_h2_hamiltonian,
    create_lih_hamiltonian,
    create_h2o_hamiltonian,
    create_beh2_hamiltonian,
    create_nh3_hamiltonian,
    create_ch4_hamiltonian,
)
from pipeline import FlowGuidedKrylovPipeline, PipelineConfig


@dataclass
class ComparisonResult:
    """Result from comparing SKQD vs SQD on one system."""
    system: str
    n_qubits: int
    n_configs: int
    fci_energy: float
    nf_energy: float
    skqd_energy: float
    sqd_energy: float
    skqd_error_mha: float
    sqd_error_mha: float
    skqd_time: float
    sqd_time: float


SYSTEMS = {
    "h2": ("H2 (STO-3G)", create_h2_hamiltonian, 0.74),
    "lih": ("LiH (STO-3G)", create_lih_hamiltonian, 1.6),
    "h2o": ("H2O (STO-3G)", create_h2o_hamiltonian, None),
    "beh2": ("BeH2 (STO-3G)", create_beh2_hamiltonian, None),
    "nh3": ("NH3 (STO-3G)", create_nh3_hamiltonian, None),
    "ch4": ("CH4 (STO-3G)", create_ch4_hamiltonian, None),
}


def run_comparison(system_key: str, verbose: bool = True) -> ComparisonResult:
    """Run both SKQD and SQD on a single system and compare."""
    name, create_fn, bond_length = SYSTEMS[system_key]

    print(f"\n{'='*70}")
    print(f"COMPARING: {name}")
    print(f"{'='*70}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create Hamiltonian
    if bond_length is not None:
        H = create_fn(bond_length=bond_length, device=device)
    else:
        H = create_fn(device=device)

    n_qubits = H.num_sites
    from math import comb
    n_configs = comb(H.n_orbitals, H.n_alpha) * comb(H.n_orbitals, H.n_beta)

    # FCI reference
    E_fci = H.fci_energy()
    print(f"  Qubits: {n_qubits}, Configs: {n_configs}, FCI: {E_fci:.8f} Ha")

    # --- SKQD Mode ---
    print(f"\n  Running SKQD (Krylov time evolution)...")
    t0 = time.time()

    config_skqd = PipelineConfig(
        subspace_mode="skqd",
        skip_nf_training=True,
        device=device,
    )
    config_skqd.adapt_to_system_size(n_configs)

    pipeline_skqd = FlowGuidedKrylovPipeline(H, config=config_skqd, exact_energy=E_fci)
    results_skqd = pipeline_skqd.run(progress=verbose)

    skqd_energy = results_skqd.get(
        'combined_energy',
        results_skqd.get('skqd_energy',
        results_skqd.get('sqd_energy', float('inf')))
    )
    skqd_time = time.time() - t0
    nf_energy = results_skqd.get('nf_nqs_energy', 0.0)

    skqd_error = abs(skqd_energy - E_fci) * 1000
    print(f"  SKQD: {skqd_energy:.8f} Ha (error: {skqd_error:.4f} mHa, time: {skqd_time:.1f}s)")

    # --- SQD Mode ---
    print(f"\n  Running SQD (sampling-based batch diag)...")
    t0 = time.time()

    config_sqd = PipelineConfig(
        subspace_mode="sqd",
        skip_nf_training=True,
        sqd_num_batches=5,
        sqd_self_consistent_iters=3,
        device=device,
    )
    config_sqd.adapt_to_system_size(n_configs)

    pipeline_sqd = FlowGuidedKrylovPipeline(H, config=config_sqd, exact_energy=E_fci)
    results_sqd = pipeline_sqd.run(progress=verbose)

    sqd_energy = results_sqd.get(
        'combined_energy',
        results_sqd.get('sqd_energy',
        results_sqd.get('skqd_energy', float('inf')))
    )
    sqd_time = time.time() - t0

    sqd_error = abs(sqd_energy - E_fci) * 1000
    print(f"  SQD:  {sqd_energy:.8f} Ha (error: {sqd_error:.4f} mHa, time: {sqd_time:.1f}s)")

    return ComparisonResult(
        system=name,
        n_qubits=n_qubits,
        n_configs=n_configs,
        fci_energy=E_fci,
        nf_energy=nf_energy,
        skqd_energy=skqd_energy,
        sqd_energy=sqd_energy,
        skqd_error_mha=skqd_error,
        sqd_error_mha=sqd_error,
        skqd_time=skqd_time,
        sqd_time=sqd_time,
    )


def main():
    parser = argparse.ArgumentParser(description="SKQD vs SQD Subspace Comparison")
    parser.add_argument(
        "--systems",
        nargs="+",
        default=["h2", "lih", "h2o", "beh2"],
        choices=list(SYSTEMS.keys()),
        help="Systems to compare (default: h2 lih h2o beh2)",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce verbosity")
    args = parser.parse_args()

    print("=" * 90)
    print("SUBSPACE METHOD COMPARISON: SKQD vs SQD")
    print("=" * 90)
    print("SKQD: Krylov time evolution (U^k = e^{-iHdt})")
    print("SQD:  Sampling-based batch diagonalization (IBM paper)")
    print("=" * 90)

    results = []
    for key in args.systems:
        try:
            r = run_comparison(key, verbose=not args.quiet)
            results.append(r)
        except Exception as e:
            print(f"\nERROR on {key}: {e}")
            import traceback
            traceback.print_exc()

    # Print comparison table
    print("\n" + "=" * 110)
    print("COMPARISON TABLE")
    print("=" * 110)
    print(f"{'System':<18} {'Qubits':>6} {'Configs':>8} {'FCI Energy':>14} "
          f"{'SKQD err':>10} {'SQD err':>10} {'SKQD t':>8} {'SQD t':>8} {'Better':>8}")
    print("-" * 110)

    for r in results:
        better = "SKQD" if r.skqd_error_mha <= r.sqd_error_mha else "SQD"
        print(f"{r.system:<18} {r.n_qubits:>6} {r.n_configs:>8} {r.fci_energy:>14.8f} "
              f"{r.skqd_error_mha:>10.4f} {r.sqd_error_mha:>10.4f} "
              f"{r.skqd_time:>7.1f}s {r.sqd_time:>7.1f}s {better:>8}")

    print("-" * 110)

    # Chemical accuracy check
    chem_acc_threshold = 1.6  # mHa
    skqd_pass = sum(1 for r in results if r.skqd_error_mha < chem_acc_threshold)
    sqd_pass = sum(1 for r in results if r.sqd_error_mha < chem_acc_threshold)
    total = len(results)

    print(f"\nChemical accuracy (<{chem_acc_threshold} mHa):")
    print(f"  SKQD: {skqd_pass}/{total} systems")
    print(f"  SQD:  {sqd_pass}/{total} systems")

    # Average errors
    if results:
        avg_skqd = np.mean([r.skqd_error_mha for r in results])
        avg_sqd = np.mean([r.sqd_error_mha for r in results])
        print(f"\nAverage error:")
        print(f"  SKQD: {avg_skqd:.4f} mHa")
        print(f"  SQD:  {avg_sqd:.4f} mHa")

        avg_skqd_t = np.mean([r.skqd_time for r in results])
        avg_sqd_t = np.mean([r.sqd_time for r in results])
        print(f"\nAverage time:")
        print(f"  SKQD: {avg_skqd_t:.1f}s")
        print(f"  SQD:  {avg_sqd_t:.1f}s")

    print("=" * 110)


if __name__ == "__main__":
    main()
