"""
Test Script: Molecular Hamiltonians (H2, LiH, H2O)

This script verifies the Flow-Guided Krylov pipeline on molecular systems.
Reference energies are compared against exact diagonalization (FCI for small systems).

Run with:
    python examples/molecular_test.py [--molecule h2|lih|h2o|all]
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
        create_h2o_hamiltonian,
        MolecularHamiltonian,
    )
    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False

from pipeline import FlowGuidedKrylovPipeline, PipelineConfig


def test_h2(verbose: bool = True):
    """
    Test H2 molecule at equilibrium bond length.

    H2 with STO-3G: 4 spin orbitals -> 4 qubits
    Exact ground state energy: ~-1.137 Hartree
    """
    if not PYSCF_AVAILABLE:
        print("PySCF not available. Install with: pip install pyscf")
        return None

    print("\n" + "=" * 70)
    print("TEST: H2 Molecule (4 qubits)")
    print("=" * 70)

    bond_length = 0.74  # Angstrom
    print(f"Bond length: {bond_length} Angstrom")
    print("Basis: STO-3G (4 spin orbitals)")

    # Create Hamiltonian
    H = create_h2_hamiltonian(bond_length=bond_length)

    print(f"Number of qubits: {H.num_sites}")
    print(f"Hilbert space dimension: {H.hilbert_dim}")

    # Get exact energy
    E_exact, _ = H.exact_ground_state()
    print(f"Exact ground state energy: {E_exact:.8f} Ha")

    # Configure pipeline (small system - use lighter config)
    config = PipelineConfig(
        nf_coupling_layers=3,
        nf_hidden_dims=[128, 128],
        nqs_hidden_dims=[128, 128, 128],
        samples_per_batch=1000,
        num_batches=1,
        max_epochs=300,
        min_epochs=100,
        convergence_threshold=0.15,
        inference_samples=1000,
        inference_iterations=500,
        max_krylov_dim=8,
        shots_per_krylov=30000,
        skip_inference=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print(f"\nUsing device: {config.device}")

    # Run pipeline
    start_time = time.time()
    pipeline = FlowGuidedKrylovPipeline(H, config=config, exact_energy=E_exact)
    results = pipeline.run(progress=True)
    elapsed = time.time() - start_time

    # Compute accuracy metrics
    E_pipeline = results.get('combined_energy', results.get('skqd_energy'))
    error_ha = abs(E_pipeline - E_exact)
    error_mha = error_ha * 1000
    error_kcal = error_ha * 627.5

    print("\n" + "-" * 50)
    print("H2 TEST RESULTS:")
    print("-" * 50)
    print(f"Exact energy:    {E_exact:.8f} Ha")
    print(f"Pipeline energy: {E_pipeline:.8f} Ha")
    print(f"Error: {error_mha:.4f} mHa ({error_kcal:.4f} kcal/mol)")
    print(f"Time elapsed: {elapsed:.1f}s")

    # Chemical accuracy threshold: 1 kcal/mol
    passed = error_kcal < 1.0
    print(f"\nChemical accuracy (<1 kcal/mol): {'PASS [OK]' if passed else 'FAIL [X]'}")

    # Save convergence plot
    pipeline.plot_convergence(save_path='h2_test_convergence.png', show=False)
    print("Saved convergence plot to h2_test_convergence.png")

    return {
        'molecule': 'H2',
        'qubits': H.num_sites,
        'exact_energy': E_exact,
        'pipeline_energy': E_pipeline,
        'error_mha': error_mha,
        'error_kcal': error_kcal,
        'passed': passed,
        'time': elapsed,
    }


def test_lih(verbose: bool = True):
    """
    Test LiH molecule at equilibrium bond length.

    LiH with STO-3G: 12 spin orbitals -> 12 qubits
    Exact ground state energy: ~-7.882 Hartree
    """
    if not PYSCF_AVAILABLE:
        print("PySCF not available. Install with: pip install pyscf")
        return None

    print("\n" + "=" * 70)
    print("TEST: LiH Molecule (12 qubits)")
    print("=" * 70)

    bond_length = 1.6  # Angstrom
    print(f"Bond length: {bond_length} Angstrom")
    print("Basis: STO-3G (12 spin orbitals)")

    # Create Hamiltonian
    H = create_lih_hamiltonian(bond_length=bond_length)

    print(f"Number of qubits: {H.num_sites}")
    print(f"Hilbert space dimension: {H.hilbert_dim}")

    # Get exact energy (feasible for 12 qubits)
    E_exact, _ = H.exact_ground_state()
    print(f"Exact ground state energy: {E_exact:.8f} Ha")

    # Configure pipeline (medium system - optimized for speed)
    config = PipelineConfig(
        nf_coupling_layers=4,
        nf_hidden_dims=[256, 256],
        nqs_hidden_dims=[256, 256, 256, 256],
        samples_per_batch=1000,  # Reduced for faster epochs
        num_batches=1,
        max_epochs=300,
        min_epochs=100,
        convergence_threshold=0.25,  # More lenient for faster convergence
        inference_samples=2000,
        inference_iterations=800,
        max_krylov_dim=10,
        shots_per_krylov=50000,
        skip_inference=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        # Basis management for large Hilbert spaces
        max_accumulated_basis=1024,  # Cap basis size
        accumulated_energy_interval=5,  # Compute full energy every 5 epochs
    )

    print(f"\nUsing device: {config.device}")

    # Run pipeline
    start_time = time.time()
    pipeline = FlowGuidedKrylovPipeline(H, config=config, exact_energy=E_exact)
    results = pipeline.run(progress=True)
    elapsed = time.time() - start_time

    # Compute accuracy metrics
    E_pipeline = results.get('combined_energy', results.get('skqd_energy'))
    error_ha = abs(E_pipeline - E_exact)
    error_mha = error_ha * 1000
    error_kcal = error_ha * 627.5

    print("\n" + "-" * 50)
    print("LiH TEST RESULTS:")
    print("-" * 50)
    print(f"Exact energy:    {E_exact:.8f} Ha")
    print(f"Pipeline energy: {E_pipeline:.8f} Ha")
    print(f"Error: {error_mha:.4f} mHa ({error_kcal:.4f} kcal/mol)")
    print(f"Time elapsed: {elapsed:.1f}s")

    # Chemical accuracy threshold: 1 kcal/mol
    passed = error_kcal < 1.0
    print(f"\nChemical accuracy (<1 kcal/mol): {'PASS [OK]' if passed else 'FAIL [X]'}")

    # Save convergence plot
    pipeline.plot_convergence(save_path='lih_test_convergence.png', show=False)
    print("Saved convergence plot to lih_test_convergence.png")

    return {
        'molecule': 'LiH',
        'qubits': H.num_sites,
        'exact_energy': E_exact,
        'pipeline_energy': E_pipeline,
        'error_mha': error_mha,
        'error_kcal': error_kcal,
        'passed': passed,
        'time': elapsed,
    }


def test_h2o(verbose: bool = True):
    """
    Test H2O molecule with standard geometry.

    H2O with STO-3G: 14 spin orbitals -> 14 qubits
    Exact ground state energy: ~-75.01 Hartree
    """
    if not PYSCF_AVAILABLE:
        print("PySCF not available. Install with: pip install pyscf")
        return None

    print("\n" + "=" * 70)
    print("TEST: H2O Molecule (14 qubits)")
    print("=" * 70)

    oh_length = 0.96  # Angstrom
    angle = 104.5  # degrees
    print(f"O-H bond length: {oh_length} Angstrom")
    print(f"H-O-H angle: {angle} degrees")
    print("Basis: STO-3G (14 spin orbitals)")

    # Create Hamiltonian
    H = create_h2o_hamiltonian(oh_length=oh_length, angle=angle)

    print(f"Number of qubits: {H.num_sites}")
    print(f"Hilbert space dimension: {H.hilbert_dim}")

    # Get exact energy (feasible for 14 qubits, may take a while)
    print("Computing exact ground state energy (this may take a moment)...")
    E_exact, _ = H.exact_ground_state()
    print(f"Exact ground state energy: {E_exact:.8f} Ha")

    # Configure pipeline (larger system - more resources)
    config = PipelineConfig(
        nf_coupling_layers=4,
        nf_hidden_dims=[512, 512],
        nqs_hidden_dims=[512, 512, 512, 512],
        samples_per_batch=3000,
        num_batches=1,
        max_epochs=500,
        min_epochs=200,
        convergence_threshold=0.20,
        inference_samples=3000,
        inference_iterations=1000,
        max_krylov_dim=12,
        shots_per_krylov=80000,
        skip_inference=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print(f"\nUsing device: {config.device}")

    # Run pipeline
    start_time = time.time()
    pipeline = FlowGuidedKrylovPipeline(H, config=config, exact_energy=E_exact)
    results = pipeline.run(progress=True)
    elapsed = time.time() - start_time

    # Compute accuracy metrics
    E_pipeline = results.get('combined_energy', results.get('skqd_energy'))
    error_ha = abs(E_pipeline - E_exact)
    error_mha = error_ha * 1000
    error_kcal = error_ha * 627.5

    print("\n" + "-" * 50)
    print("H2O TEST RESULTS:")
    print("-" * 50)
    print(f"Exact energy:    {E_exact:.8f} Ha")
    print(f"Pipeline energy: {E_pipeline:.8f} Ha")
    print(f"Error: {error_mha:.4f} mHa ({error_kcal:.4f} kcal/mol)")
    print(f"Time elapsed: {elapsed:.1f}s")

    # Chemical accuracy threshold: 1 kcal/mol
    passed = error_kcal < 1.0
    print(f"\nChemical accuracy (<1 kcal/mol): {'PASS [OK]' if passed else 'FAIL [X]'}")

    # Save convergence plot
    pipeline.plot_convergence(save_path='h2o_test_convergence.png', show=False)
    print("Saved convergence plot to h2o_test_convergence.png")

    return {
        'molecule': 'H2O',
        'qubits': H.num_sites,
        'exact_energy': E_exact,
        'pipeline_energy': E_pipeline,
        'error_mha': error_mha,
        'error_kcal': error_kcal,
        'passed': passed,
        'time': elapsed,
    }


def run_all_tests():
    """Run all molecular tests and print summary."""
    print("\n" + "=" * 70)
    print("MOLECULAR HAMILTONIAN VERIFICATION SUITE")
    print("Flow-Guided Krylov Quantum Diagonalization")
    print("=" * 70)

    results = []

    # Test H2
    h2_result = test_h2()
    if h2_result:
        results.append(h2_result)

    # Test LiH
    lih_result = test_lih()
    if lih_result:
        results.append(lih_result)

    # Test H2O
    h2o_result = test_h2o()
    if h2o_result:
        results.append(h2o_result)

    # Print summary table
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"{'Molecule':<10} {'Qubits':<8} {'Exact (Ha)':<14} {'Error (mHa)':<12} {'Error (kcal/mol)':<16} {'Status':<8}")
    print("-" * 70)

    all_passed = True
    for r in results:
        status = "PASS [OK]" if r['passed'] else "FAIL [X]"
        if not r['passed']:
            all_passed = False
        print(f"{r['molecule']:<10} {r['qubits']:<8} {r['exact_energy']:<14.6f} {r['error_mha']:<12.4f} {r['error_kcal']:<16.4f} {status:<8}")

    print("-" * 70)
    total_time = sum(r['time'] for r in results)
    print(f"Total time: {total_time:.1f}s")
    print(f"\nOverall: {'ALL TESTS PASSED [OK]' if all_passed else 'SOME TESTS FAILED [X]'}")
    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(description="Molecular Hamiltonian Tests")
    parser.add_argument(
        "--molecule",
        type=str,
        default="all",
        choices=["h2", "lih", "h2o", "all"],
        help="Which molecule to test (default: all)"
    )

    args = parser.parse_args()

    if not PYSCF_AVAILABLE:
        print("ERROR: PySCF is required for molecular tests.")
        print("Install with: pip install pyscf")
        sys.exit(1)

    if args.molecule == "h2":
        test_h2()
    elif args.molecule == "lih":
        test_lih()
    elif args.molecule == "h2o":
        test_h2o()
    else:
        run_all_tests()


if __name__ == "__main__":
    main()
