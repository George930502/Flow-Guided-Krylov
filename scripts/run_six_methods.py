#!/usr/bin/env python3
"""
Unified benchmark: 6 methods for sample-based quantum diagonalization.

Classical (NQS):           Quantum (QC):
  NQS+SQD                   QC+SQD
  NQS+SKQD                  QC+SKQD
  HI+NQS+SQD                HI-VQE

Usage:
  python scripts/run_six_methods.py --molecules H2,LiH
  python scripts/run_six_methods.py --tier 1
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.molecules import get_molecule
from src.solvers.fci import FCISolver
from src.methods.nqs_sqd import run_nqs_sqd
from src.methods.nqs_skqd import run_nqs_skqd
from src.methods.hi_nqs_sqd import run_hi_nqs_sqd, HINQSSQDConfig
from src.methods.qc_sqd import run_qc_sqd
from src.methods.qc_skqd import run_qc_skqd
from src.methods.hi_vqe import run_hi_vqe, HIVQEConfig

# Molecule tiers
TIERS = {
    1: ["H2", "LiH", "H2O", "BeH2"],
    2: ["NH3", "CH4"],
    3: ["N2", "CO"],
}

# Hyperparameters per qubit range
def _get_params(nq):
    if nq <= 14:
        return dict(n_epochs=200, n_samples=2000, samples_per_epoch=512, max_basis=5000)
    elif nq <= 20:
        return dict(n_epochs=400, n_samples=5000, samples_per_epoch=1024, max_basis=10000)
    else:
        return dict(n_epochs=500, n_samples=8000, samples_per_epoch=2048, max_basis=10000)


def run_benchmark(molecule_names, methods=None):
    if methods is None:
        methods = ["FCI", "QC+SQD", "QC+SKQD", "HI-VQE",
                    "NQS+SQD", "NQS+SKQD", "HI+NQS+SQD"]

    results = []
    total_t0 = time.time()

    for mol_name in molecule_names:
        try:
            H, info = get_molecule(mol_name)
        except Exception as e:
            print(f"\nSKIP {mol_name}: {e}")
            continue

        nq = info["n_qubits"]
        hp = _get_params(nq)
        fci_energy = None

        print(f"\n{'='*80}")
        print(f"{mol_name} ({nq}Q)")
        print(f"  {'Method':<16} {'Energy (Ha)':>18} {'Error (mHa)':>12} "
              f"{'Basis':>8} {'Time (s)':>10}")
        print(f"  {'-'*70}")

        for method in methods:
            try:
                if method == "FCI":
                    r = FCISolver().solve(H, info)
                    if r.energy is not None:
                        fci_energy = r.energy

                elif method == "QC+SQD":
                    r = run_qc_sqd(H, info, n_samples=hp["n_samples"])

                elif method == "QC+SKQD":
                    r = run_qc_skqd(H, info, max_basis_size=hp["max_basis"])

                elif method == "HI-VQE":
                    cfg = HIVQEConfig(
                        shots=hp["n_samples"],
                        max_basis_size=hp["max_basis"],
                    )
                    r = run_hi_vqe(H, info, config=cfg)

                elif method == "NQS+SQD":
                    r = run_nqs_sqd(H, info,
                                    n_epochs=hp["n_epochs"],
                                    n_samples=hp["n_samples"],
                                    samples_per_epoch=hp["samples_per_epoch"])

                elif method == "NQS+SKQD":
                    r = run_nqs_skqd(H, info,
                                     n_epochs=hp["n_epochs"],
                                     n_samples=hp["n_samples"],
                                     max_basis_size=hp["max_basis"],
                                     samples_per_epoch=hp["samples_per_epoch"])

                elif method == "HI+NQS+SQD":
                    cfg = HINQSSQDConfig(
                        n_samples=hp["n_samples"],
                        max_basis_size=hp["max_basis"],
                    )
                    r = run_hi_nqs_sqd(H, info, config=cfg)

                else:
                    continue

                # Compute error
                err = None
                if r.energy is not None and fci_energy is not None:
                    err = (r.energy - fci_energy) * 1000

                e_str = f"{r.energy:.10f}" if r.energy is not None else "N/A"
                err_str = f"{err:.3f}" if err is not None else "N/A"
                status = "OK" if r.converged else ("SKIP" if r.energy is None else "FAIL")

                print(f"  {method:<16} {e_str:>18} {err_str:>12} "
                      f"{r.diag_dim:>8} {r.wall_time:>10.2f}  {status}")

                results.append({
                    "molecule": mol_name,
                    "method": method,
                    "energy": r.energy,
                    "error_mha": err,
                    "diag_dim": r.diag_dim,
                    "wall_time": r.wall_time,
                    "converged": r.converged,
                })

            except Exception as e:
                print(f"  {method:<16} ERROR: {e}")
                results.append({
                    "molecule": mol_name,
                    "method": method,
                    "energy": None,
                    "error_mha": None,
                    "diag_dim": 0,
                    "wall_time": 0,
                    "converged": False,
                })

    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "six_methods.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    csv_path = results_dir / "six_methods.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "molecule", "method", "energy", "error_mha",
            "diag_dim", "wall_time", "converged",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    total_time = time.time() - total_t0
    print(f"\n\nTotal time: {total_time:.1f}s")
    print(f"Results: {results_dir / 'six_methods.json'}")

    return results


def main():
    parser = argparse.ArgumentParser(description="6-method SQD benchmark")
    parser.add_argument("--tier", type=int, default=None)
    parser.add_argument("--molecules", type=str, default=None)
    parser.add_argument("--methods", type=str, default=None,
                        help="Comma-separated methods (default: all)")
    args = parser.parse_args()

    # Parse molecules
    molecule_names = None
    if args.molecules:
        parts, depth, current = [], 0, []
        for ch in args.molecules:
            if ch == '(': depth += 1; current.append(ch)
            elif ch == ')': depth -= 1; current.append(ch)
            elif ch == ',' and depth == 0: parts.append(''.join(current).strip()); current = []
            else: current.append(ch)
        if current: parts.append(''.join(current).strip())
        molecule_names = [p for p in parts if p]
    elif args.tier:
        molecule_names = []
        for t in range(1, args.tier + 1):
            molecule_names.extend(TIERS.get(t, []))
    else:
        molecule_names = TIERS[1]  # default: Tier 1

    methods = None
    if args.methods:
        methods = [m.strip() for m in args.methods.split(",")]

    run_benchmark(molecule_names, methods)


if __name__ == "__main__":
    main()
