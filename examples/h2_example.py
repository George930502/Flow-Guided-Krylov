"""
Example: Ground State Energy of H2 Molecule

This example demonstrates the Flow-Guided Krylov pipeline on the
hydrogen molecule (H2) in the STO-3G basis.

The H2 molecule is a standard benchmark for quantum chemistry methods.
With STO-3G basis (2 spatial orbitals, 4 spin orbitals), we have a
16-dimensional Hilbert space (4 qubits with Jordan-Wigner mapping).

Run with:
    python examples/h2_example.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
import matplotlib.pyplot as plt

try:
    from hamiltonians.molecular import (
        create_h2_hamiltonian,
        compute_molecular_integrals,
        MolecularHamiltonian,
    )
    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False

from pipeline import FlowGuidedKrylovPipeline, PipelineConfig


def run_h2_binding_curve():
    """Compute H2 binding curve using Flow-Guided Krylov."""
    if not PYSCF_AVAILABLE:
        print("PySCF not available. Install with: pip install pyscf")
        return

    print("=" * 70)
    print("Flow-Guided Krylov: H2 Binding Curve")
    print("=" * 70)

    # Bond lengths to scan (in Angstrom)
    bond_lengths = np.linspace(0.5, 2.0, 10)

    # Store results
    energies_exact = []
    energies_pipeline = []

    for r in bond_lengths:
        print(f"\n--- Bond length: {r:.2f} Å ---")

        # Create H2 Hamiltonian
        H = create_h2_hamiltonian(bond_length=r)

        # Get exact energy
        E_exact, _ = H.exact_ground_state()
        energies_exact.append(E_exact)
        print(f"Exact energy: {E_exact:.6f} Ha")

        # Small config for fast testing
        config = PipelineConfig(
            nf_coupling_layers=2,
            nqs_hidden_dims=[64, 64],
            samples_per_batch=500,
            num_batches=10,
            max_epochs=100,
            inference_samples=500,
            inference_iterations=500,
            max_krylov_dim=6,
            shots_per_krylov=10000,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        pipeline = FlowGuidedKrylovPipeline(H, config=config, exact_energy=E_exact)

        try:
            results = pipeline.run(progress=False)
            E_pipeline = results.get('combined_energy', results.get('skqd_energy'))
            energies_pipeline.append(E_pipeline)
            print(f"Pipeline energy: {E_pipeline:.6f} Ha")
            print(f"Error: {abs(E_pipeline - E_exact) * 1000:.2f} mHa")
        except Exception as e:
            print(f"Pipeline failed: {e}")
            energies_pipeline.append(np.nan)

    # Plot results
    plt.figure(figsize=(10, 6))

    plt.plot(bond_lengths, energies_exact, 'b-o', label='Exact (FCI)', linewidth=2)
    plt.plot(bond_lengths, energies_pipeline, 'r-s', label='Flow-Guided Krylov', linewidth=2)

    plt.xlabel('Bond Length (Å)', fontsize=12)
    plt.ylabel('Energy (Hartree)', fontsize=12)
    plt.title('H₂ Binding Curve: Flow-Guided Krylov vs Exact', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('h2_binding_curve.png', dpi=150)
    print("\nSaved binding curve to h2_binding_curve.png")

    return bond_lengths, energies_exact, energies_pipeline


def run_single_h2():
    """Run a single H2 calculation at equilibrium."""
    if not PYSCF_AVAILABLE:
        print("PySCF not available. Install with: pip install pyscf")
        return

    print("=" * 70)
    print("Flow-Guided Krylov: H2 at Equilibrium")
    print("=" * 70)

    # Equilibrium bond length
    r_eq = 0.74  # Angstrom

    print(f"\nH2 molecule at bond length {r_eq} Å")
    print("Basis: STO-3G (4 spin orbitals)")

    # Create Hamiltonian
    H = create_h2_hamiltonian(bond_length=r_eq)

    print(f"Number of qubits: {H.num_sites}")
    print(f"Hilbert space dimension: {H.hilbert_dim}")

    # Get exact energy
    E_exact, psi_exact = H.exact_ground_state()
    print(f"\nExact ground state energy: {E_exact:.8f} Ha")

    # Pipeline configuration
    config = PipelineConfig(
        nf_coupling_layers=3,
        nqs_hidden_dims=[128, 128, 128],
        samples_per_batch=1000,
        num_batches=20,
        max_epochs=200,
        inference_samples=1000,
        inference_iterations=1000,
        max_krylov_dim=8,
        shots_per_krylov=50000,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Run pipeline
    pipeline = FlowGuidedKrylovPipeline(H, config=config, exact_energy=E_exact)
    results = pipeline.run(progress=True)

    # Analysis
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    E_pipeline = results.get('combined_energy', results.get('skqd_energy'))
    error_ha = abs(E_pipeline - E_exact)
    error_mha = error_ha * 1000
    error_kcal = error_ha * 627.5

    print(f"\nExact energy:    {E_exact:.8f} Ha")
    print(f"Pipeline energy: {E_pipeline:.8f} Ha")
    print(f"Error: {error_mha:.4f} mHa ({error_kcal:.2f} kcal/mol)")

    # Chemical accuracy is 1 kcal/mol
    if error_kcal < 1.0:
        print("\n✓ Chemical accuracy achieved!")
    else:
        print(f"\n✗ Chemical accuracy not achieved (error > 1 kcal/mol)")

    # Plot convergence
    pipeline.plot_convergence(save_path='h2_convergence.png', show=False)
    print("\nSaved convergence plot to h2_convergence.png")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="H2 molecule example")
    parser.add_argument(
        "--curve",
        action="store_true",
        help="Run full binding curve (takes longer)"
    )

    args = parser.parse_args()

    if args.curve:
        run_h2_binding_curve()
    else:
        run_single_h2()
