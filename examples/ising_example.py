"""
Example: Ground State Energy of Transverse Field Ising Model

This example demonstrates the Flow-Guided Krylov pipeline on the
transverse field Ising model:

    H = -V Σ_{⟨i,j⟩} σ_i^z σ_j^z - h Σ_i σ_i^x

The pipeline:
1. Trains NF-NQS to discover high-probability basis states
2. Extracts a data-driven basis from the trained flow
3. Refines amplitudes using a fresh NQS on the fixed basis
4. Uses SKQD to systematically improve energy via Krylov expansion

Run with:
    python examples/ising_example.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import matplotlib.pyplot as plt

from pipeline import FlowGuidedKrylovPipeline, PipelineConfig
from hamiltonians.spin import TransverseFieldIsing


def main():
    """Run Ising model example."""
    print("=" * 70)
    print("Flow-Guided Krylov Quantum Diagonalization")
    print("Example: Transverse Field Ising Model")
    print("=" * 70)

    # System parameters
    num_spins = 10
    V = 1.0  # ZZ interaction strength
    h = 1.0  # Transverse field strength
    L = 5    # Interaction range (L=1 for nearest neighbor)

    print(f"\nSystem: {num_spins} spins, V={V}, h={h}, L={L}")

    # Create Hamiltonian
    H = TransverseFieldIsing(
        num_spins=num_spins,
        V=V,
        h=h,
        L=L,
        periodic=True,
    )

    # Get exact ground state energy for comparison (only for small systems)
    if num_spins <= 14:
        E_exact, _ = H.exact_ground_state()
        print(f"Exact ground state energy: {E_exact:.6f}")
    else:
        E_exact = None
        print("System too large for exact diagonalization")

    # Pipeline configuration (uses optimized training defaults)
    config = PipelineConfig(
        # NF-NQS architecture
        nf_coupling_layers=4,
        nf_hidden_dims=[512, 512],
        nqs_hidden_dims=[512, 512, 512, 512],

        # Training: uses optimized defaults from PipelineConfig
        # - samples_per_batch=2000 (fast iterations)
        # - num_batches=1 (single batch per epoch)
        # - nf_lr=1e-3, nqs_lr=1e-3 (faster convergence)
        # - use_local_energy=True (fast energy estimation)

        # SKQD
        max_krylov_dim=12,
        time_step=0.1,
        num_trotter_steps=8,
        shots_per_krylov=100000,

        # Hardware
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print(f"\nUsing device: {config.device}")

    # Create pipeline
    pipeline = FlowGuidedKrylovPipeline(H, config=config, exact_energy=E_exact)

    # Run full pipeline
    print("\nRunning pipeline...")
    results = pipeline.run(progress=True)

    # Print summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(f"\nNF-NQS Training:")
    print(f"  Final energy: {results['nf_nqs_energy']:.6f}")
    print(f"  Basis size: {results['nf_basis_size']}")

    print(f"\nInference NQS:")
    print(f"  Final energy: {results['inference_energy']:.6f}")

    print(f"\nSKQD Refinement:")
    skqd = results['skqd_results']
    if 'energies_combined' in skqd:
        print(f"  NF-only energy: {skqd['energy_nf_only']:.6f}")
        print(f"  Krylov-only (final): {skqd['energies_krylov'][-1]:.6f}")
        print(f"  Combined (final): {skqd['energies_combined'][-1]:.6f}")
    else:
        print(f"  SKQD energy: {skqd['energies'][-1]:.6f}")

    if E_exact is not None:
        best_energy = results.get('combined_energy', results.get('skqd_energy'))
        error = abs(best_energy - E_exact)
        error_pct = 100 * error / abs(E_exact)
        print(f"\nComparison to exact:")
        print(f"  Exact energy: {E_exact:.6f}")
        print(f"  Best estimate: {best_energy:.6f}")
        print(f"  Error: {error:.6f} ({error_pct:.4f}%)")

    # Plot convergence
    print("\nGenerating convergence plot...")
    pipeline.plot_convergence(
        save_path="ising_convergence.png",
        show=False,
    )
    print("Saved to ising_convergence.png")

    return results


if __name__ == "__main__":
    main()
