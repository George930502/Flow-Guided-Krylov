"""
Enhanced Pipeline Benchmark: Testing Physical Constraints.

This script benchmarks the enhanced pipeline with:
1. Particle-conserving flow (valid molecular configurations)
2. Physics-guided training (energy importance weighting)
3. Diversity-aware basis selection (excitation rank bucketing)
4. Residual-based expansion (Selected-CI style recovery)

Run with:
    python examples/enhanced_benchmark.py [--molecule h2|lih|all]

Docker usage:
    docker-compose run --rm flow-krylov-gpu python examples/enhanced_benchmark.py --molecule lih
"""

import sys
from pathlib import Path
import argparse
import time
from math import comb

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np

try:
    from hamiltonians.molecular import (
        create_h2_hamiltonian,
        create_lih_hamiltonian,
        create_h2o_hamiltonian,
    )
    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False
    print("WARNING: PySCF not available. Install with: pip install pyscf")

from enhanced_pipeline import EnhancedFlowKrylovPipeline, EnhancedPipelineConfig


def benchmark_h2_enhanced():
    """Benchmark enhanced pipeline on H2."""
    if not PYSCF_AVAILABLE:
        print("ERROR: PySCF required.")
        return None

    print("\n" + "=" * 70)
    print("ENHANCED BENCHMARK: H2 Molecule (4 qubits)")
    print("=" * 70)

    bond_length = 0.74
    H = create_h2_hamiltonian(bond_length=bond_length)

    print(f"Bond length: {bond_length} Angstrom")
    print(f"Qubits: {H.num_sites}")
    print(f"Electrons: {H.n_alpha}α + {H.n_beta}β = {H.n_electrons}")

    E_exact, _ = H.exact_ground_state()
    print(f"Exact ground state: {E_exact:.8f} Ha")

    # H2 is small - use lighter config
    config = EnhancedPipelineConfig(
        use_particle_conserving_flow=True,
        use_diversity_selection=True,
        use_residual_expansion=True,
        # Lighter settings for H2
        nf_hidden_dims=[128, 128],
        nqs_hidden_dims=[128, 128, 128],
        samples_per_batch=1000,
        max_epochs=250,
        min_epochs=80,
        max_diverse_configs=512,
        residual_iterations=3,
        max_krylov_dim=6,
        shots_per_krylov=20000,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print(f"\nUsing device: {config.device}")

    start_time = time.time()
    pipeline = EnhancedFlowKrylovPipeline(H, config=config, exact_energy=E_exact)
    results = pipeline.run(progress=True)
    elapsed = time.time() - start_time

    # Compute accuracy
    E_pipeline = results.get('combined_energy', results.get('skqd_energy'))
    error_ha = abs(E_pipeline - E_exact)
    error_mha = error_ha * 1000
    error_kcal = error_ha * 627.5

    print("\n" + "-" * 50)
    print("H2 ENHANCED RESULTS:")
    print("-" * 50)
    print(f"Exact energy:     {E_exact:.8f} Ha")
    print(f"Pipeline energy:  {E_pipeline:.8f} Ha")
    print(f"Error: {error_mha:.4f} mHa ({error_kcal:.4f} kcal/mol)")
    print(f"Time: {elapsed:.1f}s")
    print(f"Chemical accuracy: {'PASS' if error_kcal < 1.0 else 'FAIL'}")

    return {
        'molecule': 'H2',
        'qubits': H.num_sites,
        'exact_energy': E_exact,
        'pipeline_energy': E_pipeline,
        'error_mha': error_mha,
        'error_kcal': error_kcal,
        'passed': error_kcal < 1.0,
        'time': elapsed,
    }


def benchmark_h2o_enhanced():
    """Benchmark enhanced pipeline on H2O - the critical test case."""
    if not PYSCF_AVAILABLE:
        print("ERROR: PySCF required.")
        return None

    print("\n" + "=" * 70)
    print("ENHANCED BENCHMARK: H2O Molecule (14 qubits)")
    print("=" * 70)

    oh_length = 0.96
    angle = 104.5
    H = create_h2o_hamiltonian(oh_length=oh_length, angle=angle)

    print(f"O-H bond length: {oh_length} Angstrom")
    print(f"H-O-H angle: {angle} degrees")
    print(f"Qubits: {H.num_sites}")
    print(f"Electrons: {H.n_alpha}α + {H.n_beta}β = {H.n_electrons}")
    print(f"Orbitals: {H.n_orbitals}")

    # Key insight: valid configs = C(7,5)^2 = 441
    n_valid = comb(H.n_orbitals, H.n_alpha) * comb(H.n_orbitals, H.n_beta)
    print(f"Valid particle-conserving configurations: {n_valid}")
    print(f"Total Hilbert space: {H.hilbert_dim} (only {100*n_valid/H.hilbert_dim:.1f}% valid)")

    print("Computing exact ground state energy...")
    E_exact, _ = H.exact_ground_state()
    print(f"Exact ground state: {E_exact:.8f} Ha")

    # Enhanced config for H2O - key insight: small valid subspace
    # With only 441 valid configs, we can be very thorough
    config = EnhancedPipelineConfig(
        # Critical: particle-conserving flow for H2O (5α + 5β)
        use_particle_conserving_flow=True,

        # Physics-guided training - higher physics weight for correlation
        teacher_weight=0.35,
        physics_weight=0.55,  # Higher physics weight for strongly correlated H2O
        entropy_weight=0.10,

        # Architecture - moderate size, H2O correlations are complex
        nf_hidden_dims=[256, 256],
        nqs_hidden_dims=[256, 256, 256, 256],

        # Training - more epochs for convergence
        samples_per_batch=1500,
        max_epochs=400,
        min_epochs=150,
        convergence_threshold=0.15,  # Tighter convergence

        # Diversity selection - with only 441 valid configs, can be thorough
        use_diversity_selection=True,
        max_diverse_configs=400,  # Can nearly enumerate the valid space!
        rank_2_fraction=0.55,  # Doubles are important for H2O

        # Residual expansion - critical for recovering missing configs
        use_residual_expansion=True,
        residual_iterations=8,  # More iterations for thorough coverage
        residual_configs_per_iter=50,
        residual_threshold=1e-6,

        # SKQD
        max_krylov_dim=10,
        shots_per_krylov=60000,

        # Basis management
        max_accumulated_basis=512,  # Can hold most valid configs

        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print(f"\nUsing device: {config.device}")
    print("\nKey enhancements enabled:")
    print(f"  - Particle-conserving flow ({H.n_alpha}α + {H.n_beta}β electrons)")
    print("  - Physics-guided training (w_physics=0.55)")
    print("  - Diversity selection (emphasize doubles)")
    print("  - Residual expansion (8 iterations)")
    print(f"  - NOTE: Only {n_valid} valid configs exist - thorough coverage possible!")

    start_time = time.time()
    pipeline = EnhancedFlowKrylovPipeline(H, config=config, exact_energy=E_exact)
    results = pipeline.run(progress=True)
    elapsed = time.time() - start_time

    # Compute accuracy
    E_pipeline = results.get('combined_energy', results.get('skqd_energy'))
    error_ha = abs(E_pipeline - E_exact)
    error_mha = error_ha * 1000
    error_kcal = error_ha * 627.5

    print("\n" + "-" * 50)
    print("H2O ENHANCED RESULTS:")
    print("-" * 50)
    print(f"Exact energy:     {E_exact:.8f} Ha")
    print(f"Pipeline energy:  {E_pipeline:.8f} Ha")
    print(f"Error: {error_mha:.4f} mHa ({error_kcal:.4f} kcal/mol)")
    print(f"Time: {elapsed:.1f}s")

    # Additional stats
    if 'nf_basis_size' in results:
        print(f"NF basis size: {results['nf_basis_size']}")
    if 'residual_expansion_stats' in results:
        stats = results['residual_expansion_stats']
        print(f"Expanded basis: {stats.get('final_basis_size', 'N/A')}")

    passed = error_kcal < 1.0
    print(f"\nChemical accuracy (<1 kcal/mol): {'PASS [OK]' if passed else 'FAIL [X]'}")

    # Compare with original method
    print("\n" + "-" * 50)
    print("COMPARISON (from original molecular_test.py):")
    print("-" * 50)
    print(f"Original NF-NQS:  -75.71 Ha (error: 5416 mHa) - MASSIVE FAIL")
    print(f"Original SKQD:    -79.40 Ha (error: 1726 mHa) - FAIL")
    print(f"Enhanced Pipeline: {E_pipeline:.2f} Ha (error: {error_mha:.2f} mHa) - {'PASS' if passed else 'FAIL'}")

    if error_mha < 1726:
        improvement = ((1726 - error_mha) / 1726) * 100
        print(f"\nImprovement: {improvement:.1f}% reduction in error!")

    return {
        'molecule': 'H2O',
        'qubits': H.num_sites,
        'exact_energy': E_exact,
        'pipeline_energy': E_pipeline,
        'error_mha': error_mha,
        'error_kcal': error_kcal,
        'passed': passed,
        'time': elapsed,
        'results': results,
    }


def benchmark_lih_enhanced():
    """Benchmark enhanced pipeline on LiH."""
    if not PYSCF_AVAILABLE:
        print("ERROR: PySCF required.")
        return None

    print("\n" + "=" * 70)
    print("ENHANCED BENCHMARK: LiH Molecule (12 qubits)")
    print("=" * 70)

    bond_length = 1.6
    H = create_lih_hamiltonian(bond_length=bond_length)

    print(f"Bond length: {bond_length} Angstrom")
    print(f"Qubits: {H.num_sites}")
    print(f"Electrons: {H.n_alpha}α + {H.n_beta}β = {H.n_electrons}")
    print(f"Orbitals: {H.n_orbitals}")

    print("Computing exact ground state energy...")
    E_exact, _ = H.exact_ground_state()
    print(f"Exact ground state: {E_exact:.8f} Ha")

    # Enhanced config for LiH
    config = EnhancedPipelineConfig(
        # Key enhancement: particle-conserving flow
        use_particle_conserving_flow=True,

        # Physics-guided training
        teacher_weight=0.4,  # Reduce teacher weight
        physics_weight=0.5,  # Increase physics weight
        entropy_weight=0.1,

        # Architecture
        nf_hidden_dims=[256, 256],
        nqs_hidden_dims=[256, 256, 256, 256],

        # Training
        samples_per_batch=1500,
        max_epochs=350,
        min_epochs=100,
        convergence_threshold=0.20,

        # Diversity selection
        use_diversity_selection=True,
        max_diverse_configs=1536,
        rank_2_fraction=0.55,  # Emphasize doubles

        # Residual expansion
        use_residual_expansion=True,
        residual_iterations=5,
        residual_configs_per_iter=75,
        residual_threshold=1e-5,

        # SKQD
        max_krylov_dim=8,
        shots_per_krylov=50000,

        # Basis management
        max_accumulated_basis=2048,

        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print(f"\nUsing device: {config.device}")
    print("\nKey enhancements enabled:")
    print("  - Particle-conserving flow (2α + 2β electrons)")
    print("  - Physics-guided training (w_physics=0.5)")
    print("  - Diversity selection (emphasize doubles)")
    print("  - Residual expansion (5 iterations)")

    start_time = time.time()
    pipeline = EnhancedFlowKrylovPipeline(H, config=config, exact_energy=E_exact)
    results = pipeline.run(progress=True)
    elapsed = time.time() - start_time

    # Compute accuracy
    E_pipeline = results.get('combined_energy', results.get('skqd_energy'))
    error_ha = abs(E_pipeline - E_exact)
    error_mha = error_ha * 1000
    error_kcal = error_ha * 627.5

    print("\n" + "-" * 50)
    print("LiH ENHANCED RESULTS:")
    print("-" * 50)
    print(f"Exact energy:     {E_exact:.8f} Ha")
    print(f"Pipeline energy:  {E_pipeline:.8f} Ha")
    print(f"Error: {error_mha:.4f} mHa ({error_kcal:.4f} kcal/mol)")
    print(f"Time: {elapsed:.1f}s")

    # Additional stats
    if 'nf_basis_size' in results:
        print(f"NF basis size: {results['nf_basis_size']}")
    if 'residual_expansion_stats' in results:
        stats = results['residual_expansion_stats']
        print(f"Expanded basis: {stats.get('final_basis_size', 'N/A')}")

    passed = error_kcal < 1.0
    print(f"\nChemical accuracy (<1 kcal/mol): {'PASS [OK]' if passed else 'FAIL [X]'}")

    # Compare with original method
    print("\n" + "-" * 50)
    print("COMPARISON (from previous benchmark):")
    print("-" * 50)
    print(f"Original NF-only:     1.64 mHa (1.03 kcal/mol) - FAIL")
    print(f"Original Combined:    1.66 mHa (1.04 kcal/mol) - FAIL")
    print(f"Enhanced Pipeline:    {error_mha:.2f} mHa ({error_kcal:.2f} kcal/mol) - {'PASS' if passed else 'FAIL'}")

    if error_mha < 1.64:
        improvement = ((1.64 - error_mha) / 1.64) * 100
        print(f"\nImprovement: {improvement:.1f}% reduction in error!")

    return {
        'molecule': 'LiH',
        'qubits': H.num_sites,
        'exact_energy': E_exact,
        'pipeline_energy': E_pipeline,
        'error_mha': error_mha,
        'error_kcal': error_kcal,
        'passed': passed,
        'time': elapsed,
        'results': results,
    }


def run_all_enhanced():
    """Run all enhanced benchmarks."""
    print("\n" + "=" * 70)
    print("ENHANCED FLOW-GUIDED KRYLOV: MOLECULAR BENCHMARK SUITE")
    print("=" * 70)
    print("\nEnhancements over original method:")
    print("  1. Particle-conserving flow (enforces N_e, Sz conservation)")
    print("  2. Physics-guided training (energy importance weighting)")
    print("  3. Diversity-aware selection (excitation rank bucketing)")
    print("  4. Residual expansion (Selected-CI style recovery)")
    print("=" * 70)

    results = []

    # H2
    h2_result = benchmark_h2_enhanced()
    if h2_result:
        results.append(h2_result)

    # LiH
    lih_result = benchmark_lih_enhanced()
    if lih_result:
        results.append(lih_result)

    # H2O
    h2o_result = benchmark_h2o_enhanced()
    if h2o_result:
        results.append(h2o_result)

    # Summary table
    print("\n" + "=" * 70)
    print("ENHANCED BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Molecule':<10} {'Qubits':<8} {'Exact (Ha)':<14} {'Error (mHa)':<12} {'Error (kcal/mol)':<16} {'Status':<8}")
    print("-" * 70)

    for r in results:
        status = "PASS" if r['passed'] else "FAIL"
        print(f"{r['molecule']:<10} {r['qubits']:<8} {r['exact_energy']:<14.6f} "
              f"{r['error_mha']:<12.4f} {r['error_kcal']:<16.4f} {status:<8}")

    print("-" * 70)
    total_time = sum(r['time'] for r in results)
    print(f"Total time: {total_time:.1f}s")

    all_passed = all(r['passed'] for r in results)
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(description="Enhanced Pipeline Benchmark")
    parser.add_argument(
        "--molecule",
        type=str,
        default="all",
        choices=["h2", "lih", "h2o", "all"],
        help="Which molecule to benchmark (default: all)"
    )

    args = parser.parse_args()

    if not PYSCF_AVAILABLE:
        print("ERROR: PySCF is required for molecular benchmarks.")
        print("Install with: pip install pyscf")
        sys.exit(1)

    if args.molecule == "h2":
        benchmark_h2_enhanced()
    elif args.molecule == "lih":
        benchmark_lih_enhanced()
    elif args.molecule == "h2o":
        benchmark_h2o_enhanced()
    else:
        run_all_enhanced()


if __name__ == "__main__":
    main()
