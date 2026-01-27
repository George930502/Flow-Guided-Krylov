"""
Molecular Benchmark for Flow-Guided Krylov Pipeline.

This script benchmarks the pipeline on various molecules with chemical accuracy.

Usage:
    python examples/benchmark.py --molecule all
    python examples/benchmark.py --molecule lih
    python examples/benchmark.py --molecule h2 lih h2o

Available molecules:
    h2, lih, h2o, beh2, nh3, n2, ch4

Docker usage:
    docker-compose run --rm flow-krylov-gpu python examples/benchmark.py --molecule all
"""

import sys
from pathlib import Path
import argparse
import time
from math import comb

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

try:
    from hamiltonians.molecular import MolecularHamiltonian
    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False
    print("ERROR: PySCF required. Install with: pip install pyscf")
    sys.exit(1)

from pipeline import FlowGuidedKrylovPipeline, PipelineConfig


# Molecule configurations
MOLECULES = {
    'h2': {
        'name': 'H₂',
        'create_fn': 'create_h2_hamiltonian',
        'kwargs': {'bond_length': 0.74},
        'description': '2 electrons, 2 orbitals',
    },
    'lih': {
        'name': 'LiH',
        'create_fn': 'create_lih_hamiltonian',
        'kwargs': {'bond_length': 1.6},
        'description': '4 electrons, 6 orbitals',
    },
    'h2o': {
        'name': 'H₂O',
        'create_fn': 'create_h2o_hamiltonian',
        'kwargs': {'oh_length': 0.96, 'angle': 104.5},
        'description': '10 electrons, 7 orbitals',
    },
    'beh2': {
        'name': 'BeH₂',
        'create_fn': 'create_beh2_hamiltonian',
        'kwargs': {},
        'description': '6 electrons, 7 orbitals',
    },
    'nh3': {
        'name': 'NH₃',
        'create_fn': 'create_nh3_hamiltonian',
        'kwargs': {},
        'description': '10 electrons, 8 orbitals',
    },
    'n2': {
        'name': 'N₂',
        'create_fn': 'create_n2_hamiltonian',
        'kwargs': {},
        'description': '14 electrons, 10 orbitals',
    },
    'ch4': {
        'name': 'CH₄',
        'create_fn': 'create_ch4_hamiltonian',
        'kwargs': {},
        'description': '10 electrons, 9 orbitals',
    },
}


def create_hamiltonian(molecule_key: str) -> MolecularHamiltonian:
    """Create a molecular Hamiltonian."""
    from hamiltonians import molecular

    mol_config = MOLECULES[molecule_key]
    create_fn = getattr(molecular, mol_config['create_fn'])
    return create_fn(**mol_config['kwargs'])


def benchmark_molecule(molecule_key: str, verbose: bool = True, compare_krylov: bool = True) -> dict:
    """
    Benchmark the pipeline on a single molecule.

    Args:
        molecule_key: Key from MOLECULES dict (e.g., 'h2', 'lih')
        verbose: Print progress
        compare_krylov: If True, run both with and without Krylov to show improvement

    Returns:
        Results dictionary
    """
    mol_config = MOLECULES[molecule_key]

    print("\n" + "=" * 70)
    print(f"BENCHMARK: {mol_config['name']} ({mol_config['description']})")
    print("=" * 70)

    # Create Hamiltonian
    H = create_hamiltonian(molecule_key)

    n_valid = comb(H.n_orbitals, H.n_alpha) * comb(H.n_orbitals, H.n_beta)
    print(f"Qubits: {H.num_sites}")
    print(f"Electrons: {H.n_alpha}α + {H.n_beta}β = {H.n_electrons}")
    print(f"Orbitals: {H.n_orbitals}")
    print(f"Valid configurations: {n_valid:,}")
    print(f"Hilbert space: {H.hilbert_dim:,} ({100*n_valid/H.hilbert_dim:.2f}% valid)")

    # Get exact energy
    print("\nComputing FCI energy...")
    E_exact = H.fci_energy()
    print(f"FCI energy: {E_exact:.8f} Ha")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    results_dict = {
        'molecule': mol_config['name'],
        'molecule_key': molecule_key,
        'qubits': H.num_sites,
        'n_valid': n_valid,
        'exact_energy': E_exact,
    }

    # ========== Run WITHOUT Krylov (NF-only) ==========
    print("\n" + "-" * 50)
    print("Mode 1: NF Sampling ONLY (no Krylov)")
    print("-" * 50)

    config_no_krylov = PipelineConfig(
        use_local_energy=True,
        use_ci_seeding=False,
        skip_skqd=True,  # Skip Krylov step
        device=device,
    )

    start_time = time.time()
    pipeline_no_krylov = FlowGuidedKrylovPipeline(H, config=config_no_krylov, exact_energy=E_exact)
    results_no_krylov = pipeline_no_krylov.run(progress=verbose)
    time_no_krylov = time.time() - start_time

    # Get NF-only energy (from basis diagonalization)
    E_nf_only = results_no_krylov.get('nf_basis_energy', results_no_krylov.get('nf_nqs_energy'))
    error_nf_mha = abs(E_nf_only - E_exact) * 1000
    error_nf_kcal = abs(E_nf_only - E_exact) * 627.5

    print(f"NF-only energy:   {E_nf_only:.8f} Ha")
    print(f"Error: {error_nf_mha:.4f} mHa ({error_nf_kcal:.4f} kcal/mol)")
    print(f"Time: {time_no_krylov:.1f}s")

    results_dict['nf_only_energy'] = E_nf_only
    results_dict['nf_only_error_mha'] = error_nf_mha
    results_dict['nf_only_error_kcal'] = error_nf_kcal
    results_dict['nf_only_time'] = time_no_krylov

    # ========== Run WITH Krylov ==========
    if compare_krylov:
        print("\n" + "-" * 50)
        print("Mode 2: NF Sampling + Krylov Refinement")
        print("-" * 50)

        config_with_krylov = PipelineConfig(
            use_local_energy=True,
            use_ci_seeding=False,
            skip_skqd=False,  # Include Krylov step
            device=device,
        )

        start_time = time.time()
        pipeline_with_krylov = FlowGuidedKrylovPipeline(H, config=config_with_krylov, exact_energy=E_exact)
        results_with_krylov = pipeline_with_krylov.run(progress=verbose)
        time_with_krylov = time.time() - start_time

        E_with_krylov = results_with_krylov.get('combined_energy', results_with_krylov.get('skqd_energy'))
        error_krylov_mha = abs(E_with_krylov - E_exact) * 1000
        error_krylov_kcal = abs(E_with_krylov - E_exact) * 627.5

        print(f"NF+Krylov energy: {E_with_krylov:.8f} Ha")
        print(f"Error: {error_krylov_mha:.4f} mHa ({error_krylov_kcal:.4f} kcal/mol)")
        print(f"Time: {time_with_krylov:.1f}s")

        results_dict['krylov_energy'] = E_with_krylov
        results_dict['krylov_error_mha'] = error_krylov_mha
        results_dict['krylov_error_kcal'] = error_krylov_kcal
        results_dict['krylov_time'] = time_with_krylov

        # Compute improvement from Krylov
        krylov_improvement_mha = error_nf_mha - error_krylov_mha
        results_dict['krylov_improvement_mha'] = krylov_improvement_mha

    # ========== Summary ==========
    print("\n" + "=" * 60)
    print(f"{mol_config['name']} COMPARISON RESULTS:")
    print("=" * 60)
    print(f"{'Method':<25} {'Energy (Ha)':<16} {'Error (mHa)':<14} {'Error (kcal/mol)':<16} {'Time (s)':<10}")
    print("-" * 81)
    print(f"{'Exact (FCI)':<25} {E_exact:<16.8f} {0:<14.4f} {0:<16.4f} {'-':<10}")
    print(f"{'NF Sampling Only':<25} {E_nf_only:<16.8f} {error_nf_mha:<14.4f} {error_nf_kcal:<16.4f} {time_no_krylov:<10.1f}")

    if compare_krylov:
        print(f"{'NF + Krylov':<25} {E_with_krylov:<16.8f} {error_krylov_mha:<14.4f} {error_krylov_kcal:<16.4f} {time_with_krylov:<10.1f}")
        print("-" * 81)
        print(f"Krylov improvement: {krylov_improvement_mha:.4f} mHa ({krylov_improvement_mha * 0.6275:.4f} kcal/mol)")

        # Determine best result
        best_error_kcal = min(error_nf_kcal, error_krylov_kcal)
        best_method = "NF + Krylov" if error_krylov_kcal < error_nf_kcal else "NF Only"
        results_dict['pipeline_energy'] = E_with_krylov
        results_dict['error_mha'] = error_krylov_mha
        results_dict['error_kcal'] = error_krylov_kcal
        results_dict['time'] = time_with_krylov
    else:
        best_error_kcal = error_nf_kcal
        best_method = "NF Only"
        results_dict['pipeline_energy'] = E_nf_only
        results_dict['error_mha'] = error_nf_mha
        results_dict['error_kcal'] = error_nf_kcal
        results_dict['time'] = time_no_krylov

    passed = best_error_kcal < 1.0
    results_dict['passed'] = passed
    results_dict['best_method'] = best_method

    print(f"\nBest method: {best_method}")
    print(f"Chemical accuracy (<1 kcal/mol): {'PASS' if passed else 'FAIL'}")

    return results_dict


def run_benchmarks(molecules: list, verbose: bool = True) -> list:
    """
    Run benchmarks on multiple molecules.

    Args:
        molecules: List of molecule keys
        verbose: Print progress

    Returns:
        List of results dictionaries
    """
    print("\n" + "=" * 70)
    print("FLOW-GUIDED KRYLOV MOLECULAR BENCHMARK")
    print("=" * 70)

    results = []
    for mol_key in molecules:
        if mol_key not in MOLECULES:
            print(f"WARNING: Unknown molecule '{mol_key}', skipping.")
            continue

        try:
            result = benchmark_molecule(mol_key, verbose=verbose)
            results.append(result)
        except Exception as e:
            print(f"\nERROR benchmarking {mol_key}: {e}")
            import traceback
            traceback.print_exc()

    # Summary table
    if len(results) > 0:
        print("\n" + "=" * 100)
        print("BENCHMARK SUMMARY: NF-Only vs NF+Krylov Comparison")
        print("=" * 100)
        print(f"{'Molecule':<10} {'Qubits':<8} {'NF Error':<14} {'Krylov Error':<14} "
              f"{'Improvement':<14} {'Best Method':<15} {'Status':<8}")
        print(f"{'':10} {'':8} {'(mHa)':14} {'(mHa)':14} {'(mHa)':14} {'':15} {'':8}")
        print("-" * 100)

        for r in results:
            status = "PASS" if r['passed'] else "FAIL"
            nf_err = r.get('nf_only_error_mha', r['error_mha'])
            krylov_err = r.get('krylov_error_mha', '-')
            improvement = r.get('krylov_improvement_mha', 0)
            best_method = r.get('best_method', 'N/A')

            if isinstance(krylov_err, float):
                print(f"{r['molecule']:<10} {r['qubits']:<8} {nf_err:<14.4f} {krylov_err:<14.4f} "
                      f"{improvement:<14.4f} {best_method:<15} {status:<8}")
            else:
                print(f"{r['molecule']:<10} {r['qubits']:<8} {nf_err:<14.4f} {'-':<14} "
                      f"{'-':<14} {best_method:<15} {status:<8}")

        print("-" * 100)
        total_time = sum(r['time'] for r in results)
        n_passed = sum(1 for r in results if r['passed'])

        # Calculate average improvement from Krylov
        improvements = [r.get('krylov_improvement_mha', 0) for r in results if 'krylov_improvement_mha' in r]
        if improvements:
            avg_improvement = sum(improvements) / len(improvements)
            print(f"Average Krylov improvement: {avg_improvement:.4f} mHa ({avg_improvement * 0.6275:.4f} kcal/mol)")

        print(f"Total time: {total_time:.1f}s | Passed: {n_passed}/{len(results)}")

        all_passed = all(r['passed'] for r in results)
        print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
        print("=" * 100)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Molecular Benchmark for Flow-Guided Krylov Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python examples/benchmark.py --molecule all
    python examples/benchmark.py --molecule lih
    python examples/benchmark.py --molecule h2 lih h2o

Available molecules:
    h2    - H₂ (2 electrons, 2 orbitals)
    lih   - LiH (4 electrons, 6 orbitals)
    h2o   - H₂O (10 electrons, 7 orbitals)
    beh2  - BeH₂ (6 electrons, 7 orbitals)
    nh3   - NH₃ (10 electrons, 8 orbitals)
    n2    - N₂ (14 electrons, 10 orbitals)
    ch4   - CH₄ (10 electrons, 9 orbitals)
        """
    )
    parser.add_argument(
        "--molecule", "-m",
        type=str,
        nargs="+",
        default=["all"],
        help="Molecule(s) to benchmark. Use 'all' for all molecules."
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce output verbosity"
    )

    args = parser.parse_args()

    if not PYSCF_AVAILABLE:
        print("ERROR: PySCF is required for molecular benchmarks.")
        print("Install with: pip install pyscf")
        sys.exit(1)

    # Determine molecules to run
    if "all" in args.molecule:
        molecules = list(MOLECULES.keys())
    else:
        molecules = [m.lower() for m in args.molecule]

    # Run benchmarks
    run_benchmarks(molecules, verbose=not args.quiet)


if __name__ == "__main__":
    main()
