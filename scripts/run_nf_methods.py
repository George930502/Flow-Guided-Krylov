#!/usr/bin/env python3
"""Phase 2: Run NF-based methods (NF-SQD, NF-SKQD-B, NF-SKQD-C) on molecules."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.molecules import get_molecule
from src.samplers.nf_sampler import NFSampler, NFSamplerConfig
from src.solvers.sqd import SQDSolver, SQDConfig
from src.solvers.skqd import SKQDSolverB, SKQDSolverC, SKQDConfig


def run_nf_methods(molecule_names=None, output_path=None):
    """Run NF-SQD, NF-SKQD-B, NF-SKQD-C on specified molecules."""

    if molecule_names is None:
        # All 16 molecules ordered by n_qubits
        molecule_names = [
            "H2", "LiH", "H2O", "BeH2",           # Tier 1: ≤14Q
            "NH3", "CH4",                            # Tier 2: 16-18Q
            "N2", "CO",                              # Tier 3: 20Q
            "HCN", "C2H2", "N2-CAS(10,12)", "Cr2",  # Tier 4: 22-24Q
            "H2S", "C2H4", "Benzene", "N2-CAS(10,15)",  # Tier 5: 26-30Q
        ]

    # Auto-tune hyperparameters based on system size
    NF_HYPERPARAMS = {
        # (max_nqubits): (epochs, samples_per_epoch, n_samples, skqd_max_basis)
        14: (200,  512,  3000,  5000),
        18: (400,  1024, 5000,  10000),
        20: (400,  1024, 5000,  10000),
        24: (500,  2048, 8000,  10000),
        30: (600,  2048, 10000, 10000),
    }

    def _get_nf_params(n_qubits):
        for max_nq in sorted(NF_HYPERPARAMS.keys()):
            if n_qubits <= max_nq:
                return NF_HYPERPARAMS[max_nq]
        return NF_HYPERPARAMS[30]

    results = []

    print("=" * 100)
    print(f"{'Molecule':<16} {'Method':<14} {'Energy (Ha)':>16} "
          f"{'Diag Dim':>10} {'Time (s)':>10} {'Status':<10}")
    print("=" * 100)

    for mol_name in molecule_names:
        try:
            H, mol_info = get_molecule(mol_name)
        except Exception as e:
            print(f"{mol_name:<16} SKIP: {e}")
            continue

        nq = mol_info["n_qubits"]
        epochs, spe, n_samples, skqd_max = _get_nf_params(nq)

        # Train NF sampler once per molecule
        print(f"\n--- Training NF for {mol_name} ({nq}Q, {epochs} epochs, {spe} samples/epoch) ---")
        nf_config = NFSamplerConfig(
            n_epochs=epochs,
            samples_per_epoch=spe,
        )
        nf_sampler = NFSampler(H, config=nf_config, device="cpu")
        nf_sampler.train(verbose=True)

        # Run each method with tuned hyperparameters
        methods = {
            "NF-SQD": SQDSolver(nf_sampler, SQDConfig(n_samples=n_samples)),
            "NF-SKQD-B": SKQDSolverB(nf_sampler, SKQDConfig(
                n_samples=n_samples, max_basis_size=skqd_max)),
            "NF-SKQD-C": SKQDSolverC(nf_sampler, SKQDConfig(
                n_samples=n_samples, max_basis_size=skqd_max)),
        }

        for method_name, solver in methods.items():
            try:
                result = solver.solve(H, mol_info)

                e_str = f"{result.energy:.10f}" if result.energy is not None else "N/A"
                status = "OK" if result.converged else "FAIL"

                print(f"{mol_name:<16} {method_name:<14} {e_str:>16} "
                      f"{result.diag_dim:>10} {result.wall_time:>10.2f} {status:<10}")

                results.append({
                    "molecule": mol_name,
                    "method": method_name,
                    "energy": result.energy,
                    "diag_dim": result.diag_dim,
                    "wall_time": result.wall_time,
                    "converged": result.converged,
                    "metadata": result.metadata,
                })

            except Exception as e:
                print(f"{mol_name:<16} {method_name:<14} ERROR: {e}")
                results.append({
                    "molecule": mol_name,
                    "method": method_name,
                    "energy": None,
                    "diag_dim": 0,
                    "wall_time": 0,
                    "converged": False,
                    "metadata": {"error": str(e)},
                })

        print("-" * 100)

    # Save results
    if output_path is None:
        output_path = Path(__file__).parent.parent / "results" / "nf_methods.json"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")
    return results


if __name__ == "__main__":
    molecules = sys.argv[1:] if len(sys.argv) > 1 else None
    run_nf_methods(molecules)
