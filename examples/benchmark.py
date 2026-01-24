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


def benchmark_molecule(molecule_key: str, verbose: bool = True) -> dict:
    """
    Benchmark the pipeline on a single molecule.

    Args:
        molecule_key: Key from MOLECULES dict (e.g., 'h2', 'lih')
        verbose: Print progress

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

    # Configure pipeline (auto-adapts to system size)
    config = PipelineConfig(
        use_particle_conserving_flow=True,
        use_diversity_selection=True,
        use_residual_expansion=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print(f"\nDevice: {config.device}")

    # Run pipeline
    start_time = time.time()
    pipeline = FlowGuidedKrylovPipeline(H, config=config, exact_energy=E_exact)
    results = pipeline.run(progress=verbose)
    elapsed = time.time() - start_time

    # Compute accuracy
    E_pipeline = results.get('combined_energy', results.get('skqd_energy'))
    error_ha = abs(E_pipeline - E_exact)
    error_mha = error_ha * 1000
    error_kcal = error_ha * 627.5

    print("\n" + "-" * 50)
    print(f"{mol_config['name']} RESULTS:")
    print("-" * 50)
    print(f"Exact energy:     {E_exact:.8f} Ha")
    print(f"Pipeline energy:  {E_pipeline:.8f} Ha")
    print(f"Error: {error_mha:.4f} mHa ({error_kcal:.4f} kcal/mol)")
    print(f"Time: {elapsed:.1f}s")

    passed = error_kcal < 1.0
    print(f"\nChemical accuracy (<1 kcal/mol): {'PASS' if passed else 'FAIL'}")

    return {
        'molecule': mol_config['name'],
        'molecule_key': molecule_key,
        'qubits': H.num_sites,
        'n_valid': n_valid,
        'exact_energy': E_exact,
        'pipeline_energy': E_pipeline,
        'error_mha': error_mha,
        'error_kcal': error_kcal,
        'passed': passed,
        'time': elapsed,
    }


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
    if len(results) > 1:
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        print(f"{'Molecule':<10} {'Qubits':<8} {'Valid':<12} {'Error (mHa)':<12} "
              f"{'Error (kcal/mol)':<16} {'Time (s)':<10} {'Status':<8}")
        print("-" * 80)

        for r in results:
            status = "PASS" if r['passed'] else "FAIL"
            print(f"{r['molecule']:<10} {r['qubits']:<8} {r['n_valid']:<12,} "
                  f"{r['error_mha']:<12.4f} {r['error_kcal']:<16.4f} "
                  f"{r['time']:<10.1f} {status:<8}")

        print("-" * 80)
        total_time = sum(r['time'] for r in results)
        n_passed = sum(1 for r in results if r['passed'])
        print(f"Total time: {total_time:.1f}s | Passed: {n_passed}/{len(results)}")

        all_passed = all(r['passed'] for r in results)
        print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
        print("=" * 80)

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
