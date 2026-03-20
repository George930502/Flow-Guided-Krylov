#!/usr/bin/env python3
"""Phase 3: Run quantum circuit methods (Q-SQD, Q-SKQD) on molecules."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.molecules import get_molecule
from src.solvers.sqd import SQDSolver, SQDConfig
from src.solvers.skqd import SKQDSolverC, SKQDConfig


def run_quantum_methods(molecule_names=None, output_path=None):
    """Run Q-SQD and Q-SKQD on specified molecules."""

    if molecule_names is None:
        molecule_names = ["H2", "LiH", "H2O", "BeH2"]

    results = []

    print("=" * 100)
    print(f"{'Molecule':<12} {'Method':<12} {'Energy (Ha)':>16} "
          f"{'Diag Dim':>10} {'Time (s)':>10} {'Status':<10}")
    print("=" * 100)

    for mol_name in molecule_names:
        try:
            H, mol_info = get_molecule(mol_name)
        except Exception as e:
            print(f"{mol_name:<12} SKIP: {e}")
            continue

        # Q-SQD: LUCJ sampler + SQD solver
        try:
            from src.samplers.lucj_sampler import LUCJSampler
            lucj_sampler = LUCJSampler(H, n_reps=2)
            q_sqd = SQDSolver(lucj_sampler, SQDConfig(n_samples=10000))
            result = q_sqd.solve(H, mol_info)

            e_str = f"{result.energy:.10f}" if result.energy is not None else "N/A"
            status = "OK" if result.converged else "FAIL"
            print(f"{mol_name:<12} {'Q-SQD':<12} {e_str:>16} "
                  f"{result.diag_dim:>10} {result.wall_time:>10.2f} {status:<10}")

            results.append({
                "molecule": mol_name, "method": "Q-SQD",
                "energy": result.energy, "diag_dim": result.diag_dim,
                "wall_time": result.wall_time, "converged": result.converged,
                "metadata": result.metadata,
            })
        except ImportError as e:
            print(f"{mol_name:<12} {'Q-SQD':<12} SKIP: {e}")
        except Exception as e:
            print(f"{mol_name:<12} {'Q-SQD':<12} ERROR: {e}")

        # Q-SKQD: Trotter sampler + SKQD-C solver
        try:
            from src.samplers.trotter_sampler import TrotterSampler
            trotter_sampler = TrotterSampler(H, n_krylov_steps=6, dt=0.1)
            q_skqd = SKQDSolverC(trotter_sampler, SKQDConfig(n_samples=5000))
            result = q_skqd.solve(H, mol_info)

            e_str = f"{result.energy:.10f}" if result.energy is not None else "N/A"
            status = "OK" if result.converged else "FAIL"
            print(f"{mol_name:<12} {'Q-SKQD':<12} {e_str:>16} "
                  f"{result.diag_dim:>10} {result.wall_time:>10.2f} {status:<10}")

            results.append({
                "molecule": mol_name, "method": "Q-SKQD",
                "energy": result.energy, "diag_dim": result.diag_dim,
                "wall_time": result.wall_time, "converged": result.converged,
                "metadata": result.metadata,
            })
        except Exception as e:
            print(f"{mol_name:<12} {'Q-SKQD':<12} ERROR: {e}")

        print("-" * 100)

    # Save results
    if output_path is None:
        output_path = Path(__file__).parent.parent / "results" / "quantum_methods.json"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")
    return results


if __name__ == "__main__":
    molecules = sys.argv[1:] if len(sys.argv) > 1 else None
    run_quantum_methods(molecules)
