"""
Molecular Benchmark: Pure SKQD vs NF-only vs Combined Method

This script compares three energy estimation methods on molecular systems:
1. Pure SKQD: Standard Krylov-based sampling without neural network guidance
2. NF-only: Using only the NF-discovered basis (diagonalization in NF subspace)
3. Combined (Our Method): NF basis + Krylov expansion (best of both)

Run with:
    python examples/molecular_benchmark.py [--molecule h2|lih|all]

Docker usage:
    docker-compose run --rm flow-krylov-gpu python examples/molecular_benchmark.py --molecule all
"""

import sys
from pathlib import Path
import argparse
import time

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np

try:
    from hamiltonians.molecular import (
        create_h2_hamiltonian,
        create_lih_hamiltonian,
    )
    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False
    print("WARNING: PySCF not available. Install with: pip install pyscf")

from pipeline import FlowGuidedKrylovPipeline, PipelineConfig
from krylov.skqd import SampleBasedKrylovDiagonalization, SKQDConfig


def run_pure_skqd(H, E_exact, config, molecule_name):
    """
    Run pure SKQD without NF basis.
    """
    print("\n" + "-" * 60)
    print(f"Method 1: Pure SKQD (No Neural Network)")
    print("-" * 60)

    skqd_config = SKQDConfig(
        max_krylov_dim=config.max_krylov_dim,
        time_step=config.time_step,
        num_trotter_steps=config.num_trotter_steps,
        shots_per_krylov=config.shots_per_krylov,
        use_gpu=config.use_gpu,
    )

    start_time = time.time()
    skqd = SampleBasedKrylovDiagonalization(H, config=skqd_config)
    results = skqd.run(progress=True)
    elapsed = time.time() - start_time

    E_skqd = results["energies"][-1]
    error_ha = abs(E_skqd - E_exact)
    error_mha = error_ha * 1000
    error_kcal = error_ha * 627.5

    print(f"\nPure SKQD Results:")
    print(f"  Energy: {E_skqd:.8f} Ha")
    print(f"  Error: {error_mha:.4f} mHa ({error_kcal:.4f} kcal/mol)")
    print(f"  Time: {elapsed:.1f}s")

    return {
        'method': 'Pure SKQD',
        'energy': E_skqd,
        'error_ha': error_ha,
        'error_mha': error_mha,
        'error_kcal': error_kcal,
        'time': elapsed,
    }


def run_nf_only(H, E_exact, config, molecule_name):
    """
    Run NF-NQS training and use only the NF basis for energy.
    """
    print("\n" + "-" * 60)
    print(f"Method 2: NF-only (No Krylov Expansion)")
    print("-" * 60)

    start_time = time.time()

    # Train NF-NQS
    pipeline = FlowGuidedKrylovPipeline(H, config=config, exact_energy=E_exact)
    pipeline.train_nf_nqs(progress=True)
    pipeline.extract_basis()

    # Compute energy in NF basis only (diagonalize H in NF subspace)
    basis = pipeline.nf_basis.cpu()
    H_matrix = H.matrix_elements(basis, basis).cpu().numpy()
    eigenvalues, _ = np.linalg.eigh(H_matrix)
    E_nf_only = eigenvalues[0]

    elapsed = time.time() - start_time

    error_ha = abs(E_nf_only - E_exact)
    error_mha = error_ha * 1000
    error_kcal = error_ha * 627.5

    print(f"\nNF-only Results:")
    print(f"  Energy: {E_nf_only:.8f} Ha")
    print(f"  Basis Size: {len(basis)} states")
    print(f"  Error: {error_mha:.4f} mHa ({error_kcal:.4f} kcal/mol)")
    print(f"  Time: {elapsed:.1f}s")

    return {
        'method': 'NF-only',
        'energy': E_nf_only,
        'error_ha': error_ha,
        'error_mha': error_mha,
        'error_kcal': error_kcal,
        'time': elapsed,
        'basis_size': len(basis),
        'pipeline': pipeline,
    }


def run_combined(pipeline, E_exact, nf_time):
    """
    Run combined method: NF basis + SKQD Krylov expansion.
    """
    print("\n" + "-" * 60)
    print(f"Method 3: Combined (NF + Krylov) - Our Method")
    print("-" * 60)

    start_time = time.time()

    # Run SKQD with NF basis
    skqd_results = pipeline.run_skqd(use_nf_basis=True, progress=True)

    skqd_time = time.time() - start_time
    total_time = nf_time + skqd_time

    E_combined = skqd_results["energies_combined"][-1]
    E_krylov_only = skqd_results["energies_krylov"][-1]
    E_nf_only = skqd_results["energy_nf_only"]

    error_ha = abs(E_combined - E_exact)
    error_mha = error_ha * 1000
    error_kcal = error_ha * 627.5

    print(f"\nCombined Results:")
    print(f"  Energy: {E_combined:.8f} Ha")
    print(f"  (NF-only: {E_nf_only:.8f}, Krylov-only: {E_krylov_only:.8f})")
    print(f"  Error: {error_mha:.4f} mHa ({error_kcal:.4f} kcal/mol)")
    print(f"  Time: {total_time:.1f}s (NF: {nf_time:.1f}s + SKQD: {skqd_time:.1f}s)")

    return {
        'method': 'Combined (NF+Krylov)',
        'energy': E_combined,
        'error_ha': error_ha,
        'error_mha': error_mha,
        'error_kcal': error_kcal,
        'time': total_time,
        'energy_nf_only': E_nf_only,
        'energy_krylov_only': E_krylov_only,
    }


def benchmark_h2():
    """Benchmark on H2 molecule (4 qubits)."""
    if not PYSCF_AVAILABLE:
        print("ERROR: PySCF required. Install with: pip install pyscf")
        return None

    print("\n" + "=" * 70)
    print("BENCHMARK: H2 Molecule")
    print("=" * 70)

    bond_length = 0.74  # Angstrom (equilibrium)
    H = create_h2_hamiltonian(bond_length=bond_length)
    E_exact, _ = H.exact_ground_state()

    print(f"Bond length: {bond_length} Angstrom")
    print(f"Basis: STO-3G")
    print(f"Qubits: {H.num_sites}")
    print(f"Hilbert dimension: {H.hilbert_dim}")
    print(f"Exact ground state: {E_exact:.8f} Ha")

    # Configuration for H2 (small system)
    config = PipelineConfig(
        nf_coupling_layers=3,
        nf_hidden_dims=[128, 128],
        nqs_hidden_dims=[128, 128, 128],
        samples_per_batch=1000,
        num_batches=1,
        max_epochs=300,
        min_epochs=100,
        convergence_threshold=0.15,
        max_krylov_dim=8,
        shots_per_krylov=30000,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print(f"\nUsing device: {config.device}")

    results = []

    # 1. Pure SKQD
    r1 = run_pure_skqd(H, E_exact, config, "H2")
    results.append(r1)

    # 2. NF-only
    r2 = run_nf_only(H, E_exact, config, "H2")
    results.append(r2)

    # 3. Combined
    r3 = run_combined(r2['pipeline'], E_exact, r2['time'])
    results.append(r3)

    return {
        'molecule': 'H2',
        'qubits': H.num_sites,
        'hilbert_dim': H.hilbert_dim,
        'exact_energy': E_exact,
        'bond_length': bond_length,
        'results': results,
    }


def benchmark_lih():
    """Benchmark on LiH molecule (12 qubits)."""
    if not PYSCF_AVAILABLE:
        print("ERROR: PySCF required. Install with: pip install pyscf")
        return None

    print("\n" + "=" * 70)
    print("BENCHMARK: LiH Molecule")
    print("=" * 70)

    bond_length = 1.6  # Angstrom
    H = create_lih_hamiltonian(bond_length=bond_length)

    print(f"Bond length: {bond_length} Angstrom")
    print(f"Basis: STO-3G")
    print(f"Qubits: {H.num_sites}")
    print(f"Hilbert dimension: {H.hilbert_dim}")
    print("Computing exact ground state energy...")

    E_exact, _ = H.exact_ground_state()
    print(f"Exact ground state: {E_exact:.8f} Ha")

    # Configuration for LiH (optimized for speed + accuracy)
    config = PipelineConfig(
        nf_coupling_layers=4,
        nf_hidden_dims=[256, 256],
        nqs_hidden_dims=[256, 256, 256, 256],
        samples_per_batch=1200,
        num_batches=1,
        max_epochs=250,
        min_epochs=80,
        convergence_threshold=0.22,
        max_krylov_dim=8,
        shots_per_krylov=50000,
        device="cuda" if torch.cuda.is_available() else "cpu",
        # Basis management for large systems
        max_accumulated_basis=1536,
        accumulated_energy_interval=8,
    )

    print(f"\nUsing device: {config.device}")

    results = []

    # 1. Pure SKQD
    r1 = run_pure_skqd(H, E_exact, config, "LiH")
    results.append(r1)

    # 2. NF-only
    r2 = run_nf_only(H, E_exact, config, "LiH")
    results.append(r2)

    # 3. Combined
    r3 = run_combined(r2['pipeline'], E_exact, r2['time'])
    results.append(r3)

    return {
        'molecule': 'LiH',
        'qubits': H.num_sites,
        'hilbert_dim': H.hilbert_dim,
        'exact_energy': E_exact,
        'bond_length': bond_length,
        'results': results,
    }


def print_comparison_table(benchmarks):
    """Print comparison table for all benchmarks."""
    print("\n" + "=" * 90)
    print("MOLECULAR BENCHMARK COMPARISON SUMMARY")
    print("=" * 90)

    for b in benchmarks:
        print(f"\n{b['molecule']} ({b['qubits']} qubits, dim={b['hilbert_dim']})")
        print(f"Bond length: {b['bond_length']} A | Exact energy: {b['exact_energy']:.8f} Ha")
        print("-" * 90)
        print(f"{'Method':<25} {'Energy (Ha)':<16} {'Error (mHa)':<14} {'Error (kcal/mol)':<16} {'Time (s)':<10}")
        print("-" * 90)

        for r in b['results']:
            chem_acc = "PASS" if r['error_kcal'] < 1.0 else "FAIL"
            print(f"{r['method']:<25} {r['energy']:<16.8f} {r['error_mha']:<14.4f} {r['error_kcal']:<16.4f} {r['time']:<10.1f}")

        print("-" * 90)

        # Find best method
        best = min(b['results'], key=lambda x: x['error_mha'])
        print(f"Best method: {best['method']} ({best['error_mha']:.4f} mHa)")

        # Check chemical accuracy for combined method
        combined = [r for r in b['results'] if 'Combined' in r['method']][0]
        if combined['error_kcal'] < 1.0:
            print(f"Chemical accuracy (<1 kcal/mol): PASS ({combined['error_kcal']:.4f} kcal/mol)")
        else:
            print(f"Chemical accuracy (<1 kcal/mol): FAIL ({combined['error_kcal']:.4f} kcal/mol)")

    # Final summary table for README
    print("\n" + "=" * 90)
    print("SUMMARY TABLE (for README)")
    print("=" * 90)
    print(f"{'Molecule':<10} {'Qubits':<8} {'Exact (Ha)':<14} {'Pure SKQD':<14} {'NF-only':<14} {'Combined':<14} {'Chem Acc':<10}")
    print("-" * 90)
    for b in benchmarks:
        pure = [r for r in b['results'] if r['method'] == 'Pure SKQD'][0]
        nf = [r for r in b['results'] if r['method'] == 'NF-only'][0]
        comb = [r for r in b['results'] if 'Combined' in r['method']][0]
        chem = "PASS" if comb['error_kcal'] < 1.0 else "FAIL"
        print(f"{b['molecule']:<10} {b['qubits']:<8} {b['exact_energy']:<14.6f} {pure['error_mha']:<14.2f} {nf['error_mha']:<14.2f} {comb['error_mha']:<14.2f} {chem:<10}")
    print("-" * 90)
    print("Error values in mHa (milliHartree). Chemical accuracy: <1.6 mHa (<1 kcal/mol)")


def main():
    parser = argparse.ArgumentParser(description="Molecular Benchmark Comparison")
    parser.add_argument(
        "--molecule",
        type=str,
        default="all",
        choices=["h2", "lih", "all"],
        help="Which molecule to benchmark (default: all)"
    )

    args = parser.parse_args()

    if not PYSCF_AVAILABLE:
        print("ERROR: PySCF is required for molecular benchmarks.")
        print("Install with: pip install pyscf")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("Flow-Guided Krylov: Molecular Method Comparison")
    print("=" * 70)
    print("\nComparing three methods:")
    print("  1. Pure SKQD: Krylov expansion only (no neural network)")
    print("  2. NF-only: Neural network basis only (no Krylov expansion)")
    print("  3. Combined: NF basis + Krylov expansion (our method)")
    print("\nChemical accuracy threshold: 1 kcal/mol = 1.6 mHa")
    print("=" * 70)

    benchmarks = []

    if args.molecule in ["h2", "all"]:
        b = benchmark_h2()
        if b:
            benchmarks.append(b)

    if args.molecule in ["lih", "all"]:
        b = benchmark_lih()
        if b:
            benchmarks.append(b)

    if benchmarks:
        print_comparison_table(benchmarks)


if __name__ == "__main__":
    main()
