"""Test script for Flow-Guided Krylov pipeline with stabilization."""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch

from pipeline import FlowGuidedKrylovPipeline, PipelineConfig
from hamiltonians.spin import TransverseFieldIsing


def main():
    """Run test with stabilized training."""
    print("=" * 70)
    print("Flow-Guided Krylov Quantum Diagonalization")
    print("Testing Stabilization Mechanisms")
    print("=" * 70)

    # System parameters
    num_spins = 10
    V = 1.0
    h = 1.0
    L = 5

    print(f"\nSystem: {num_spins} spins, V={V}, h={h}, L={L}")

    # Create Hamiltonian
    H = TransverseFieldIsing(
        num_spins=num_spins,
        V=V,
        h=h,
        L=L,
        periodic=True,
    )

    # Get exact ground state energy
    E_exact, _ = H.exact_ground_state()
    print(f"Exact ground state energy: {E_exact:.6f}")

    # Pipeline configuration with stability parameters
    config = PipelineConfig(
        # NF-NQS architecture
        nf_coupling_layers=4,
        nf_hidden_dims=[512, 512],
        nqs_hidden_dims=[512, 512, 512, 512],

        # Stability parameters (new!)
        nf_lr=5e-4,  # Slower flow LR
        nqs_lr=1e-3,  # Faster NQS LR
        use_accumulated_energy=True,  # Energy on accumulated basis
        ema_decay=0.95,  # EMA tracking
        entropy_weight=0.01,  # Entropy regularization

        # SKQD
        max_krylov_dim=12,
        time_step=0.1,
        num_trotter_steps=8,
        shots_per_krylov=100000,

        # Hardware
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print(f"\nUsing device: {config.device}")
    print(f"Stability parameters:")
    print(f"  - Flow LR: {config.nf_lr}")
    print(f"  - NQS LR: {config.nqs_lr}")
    print(f"  - Use accumulated energy: {config.use_accumulated_energy}")
    print(f"  - EMA decay: {config.ema_decay}")
    print(f"  - Entropy weight: {config.entropy_weight}")

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

    print(f"\nCo-trained NQS on basis:")
    print(f"  Energy: {results['inference_energy']:.6f}")

    print(f"\nSKQD Refinement:")
    skqd = results['skqd_results']
    if 'energies_combined' in skqd:
        print(f"  NF-only energy: {skqd['energy_nf_only']:.6f}")
        print(f"  Krylov-only (final): {skqd['energies_krylov'][-1]:.6f}")
        print(f"  Combined (final): {skqd['energies_combined'][-1]:.6f}")

        # Verify the expected pattern: E_combined <= min(E_NF, E_Krylov)
        print(f"\n  Verification of variational principle:")
        print(f"    E_NF_only = {skqd['energy_nf_only']:.6f}")
        print(f"    E_Krylov_only = {skqd['energies_krylov'][-1]:.6f}")
        print(f"    E_Combined = {skqd['energies_combined'][-1]:.6f}")
        print(f"    E_Combined <= E_NF_only: {skqd['energies_combined'][-1] <= skqd['energy_nf_only']}")
        print(f"    E_Combined <= E_Krylov_only: {skqd['energies_combined'][-1] <= skqd['energies_krylov'][-1]}")

    if E_exact is not None:
        best_energy = results.get('combined_energy', results.get('skqd_energy'))
        error = abs(best_energy - E_exact)
        error_pct = 100 * error / abs(E_exact)
        print(f"\nComparison to exact:")
        print(f"  Exact energy: {E_exact:.6f}")
        print(f"  Best estimate: {best_energy:.6f}")
        print(f"  Error: {error:.6f} ({error_pct:.4f}%)")

    return results


if __name__ == "__main__":
    main()
