#!/usr/bin/env python3
"""Phase 1: Run baseline methods (FCI, CCSD, CCSD(T), SCI) on all molecules."""

import json
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.molecules import get_molecule, list_molecules
from src.solvers.fci import FCISolver
from src.solvers.ccsd import CCSDSolver, CCSDTSolver
from src.solvers.sci import CIPSISolver


def run_baselines(molecule_names=None, output_path=None):
    """Run FCI, CCSD, CCSD(T), SCI on specified molecules."""

    if molecule_names is None:
        # All 16 molecules ordered by n_qubits (small to large)
        molecule_names = [
            "H2", "LiH", "H2O", "BeH2",           # Tier 1: ≤14Q
            "NH3", "CH4",                            # Tier 2: 16-18Q
            "N2", "CO",                              # Tier 3: 20Q
            "HCN", "C2H2", "N2-CAS(10,12)", "Cr2",  # Tier 4: 22-24Q
            "H2S", "C2H4", "Benzene", "N2-CAS(10,15)",  # Tier 5: 26-30Q
        ]

    solvers = {
        "FCI": FCISolver(),
        "CCSD": CCSDSolver(),
        "CCSD(T)": CCSDTSolver(),
        "SCI": CIPSISolver(),
    }

    results = []
    fci_energies = {}  # for computing errors

    print("=" * 100)
    print(f"{'Molecule':<12} {'Method':<10} {'Energy (Ha)':>16} {'Error (mHa)':>12} "
          f"{'Diag Dim':>10} {'Time (s)':>10} {'Status':<10}")
    print("=" * 100)

    for mol_name in molecule_names:
        try:
            H, mol_info = get_molecule(mol_name)
        except Exception as e:
            print(f"{mol_name:<12} SKIP: {e}")
            continue

        for solver_name, solver in solvers.items():
            try:
                result = solver.solve(H, mol_info)

                # Track FCI energy for error computation
                if solver_name == "FCI" and result.energy is not None:
                    fci_energies[mol_name] = result.energy

                # Compute error vs FCI
                error_mha = None
                if result.energy is not None and mol_name in fci_energies:
                    error_mha = (result.energy - fci_energies[mol_name]) * 1000

                # Print row
                e_str = f"{result.energy:.10f}" if result.energy is not None else "N/A"
                err_str = f"{error_mha:.3f}" if error_mha is not None else "N/A"
                status = "OK" if result.converged else "FAIL"

                print(f"{mol_name:<12} {solver_name:<10} {e_str:>16} {err_str:>12} "
                      f"{result.diag_dim:>10} {result.wall_time:>10.2f} {status:<10}")

                results.append({
                    "molecule": mol_name,
                    "method": solver_name,
                    "energy": result.energy,
                    "error_mha": error_mha,
                    "diag_dim": result.diag_dim,
                    "wall_time": result.wall_time,
                    "converged": result.converged,
                    "metadata": result.metadata,
                })

            except Exception as e:
                print(f"{mol_name:<12} {solver_name:<10} ERROR: {e}")
                results.append({
                    "molecule": mol_name,
                    "method": solver_name,
                    "energy": None,
                    "error_mha": None,
                    "diag_dim": 0,
                    "wall_time": 0,
                    "converged": False,
                    "metadata": {"error": str(e)},
                })

        print("-" * 100)

    # Save results
    if output_path is None:
        output_path = Path(__file__).parent.parent / "results" / "baselines.json"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")
    return results


if __name__ == "__main__":
    # Allow specifying molecules on command line
    molecules = sys.argv[1:] if len(sys.argv) > 1 else None
    run_baselines(molecules)
