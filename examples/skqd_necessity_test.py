"""
SKQD Necessity Test: When is SKQD needed vs. residual expansion alone?

This script tests larger molecules to determine when SKQD provides
additional benefit over residual expansion alone.

Hypothesis:
- Small valid config space (< 1000): Residual expansion sufficient
- Medium valid config space (1000-10000): SKQD may help
- Large valid config space (> 10000): SKQD likely essential

Molecules tested:
- BeH2: 6 electrons, 7 orbitals → 1,225 valid configs
- NH3: 10 electrons, 8 orbitals → 3,136 valid configs
- N2: 14 electrons, 10 orbitals → 14,400 valid configs
- CH4: 10 electrons, 9 orbitals → 15,876 valid configs
"""

import sys
from pathlib import Path
import time
from math import comb

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np

try:
    from hamiltonians.molecular import (
        create_beh2_hamiltonian,
        create_nh3_hamiltonian,
        create_n2_hamiltonian,
        create_ch4_hamiltonian,
    )
    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False
    print("ERROR: PySCF required. Install with: pip install pyscf")
    sys.exit(1)

from pipeline import FlowGuidedKrylovPipeline as EnhancedFlowKrylovPipeline
from pipeline import PipelineConfig as EnhancedPipelineConfig


def test_molecule(name, create_fn, config_overrides=None):
    """Test a molecule with both residual-only and residual+SKQD."""
    print(f"\n{'='*70}")
    print(f"TESTING: {name}")
    print("=" * 70)

    # Create Hamiltonian
    H = create_fn()
    n_valid = comb(H.n_orbitals, H.n_alpha) * comb(H.n_orbitals, H.n_beta)

    print(f"Qubits: {H.num_sites}")
    print(f"Electrons: {H.n_alpha}α + {H.n_beta}β = {H.n_electrons}")
    print(f"Orbitals: {H.n_orbitals}")
    print(f"Valid configurations: {n_valid:,}")
    print(f"Hilbert space: {H.hilbert_dim:,} (valid: {100*n_valid/H.hilbert_dim:.2f}%)")

    # Compute exact FCI energy using PySCF (fast and efficient)
    print("\nComputing exact FCI energy using PySCF...")
    start = time.time()
    try:
        # Use the efficient FCI solver (works for any size molecule)
        E_exact = H.fci_energy()
        print(f"FCI energy: {E_exact:.8f} Ha (computed in {time.time()-start:.1f}s)")
    except Exception as e:
        print(f"Could not compute FCI energy: {e}")
        import traceback
        traceback.print_exc()
        print("Will compare residual-only vs residual+SKQD without exact reference.")
        E_exact = None

    # Base config for this molecule - ADAPTIVE parameters will be set automatically
    # by EnhancedPipelineConfig.adapt_to_system_size()
    base_config = dict(
        use_particle_conserving_flow=True,
        teacher_weight=0.35,
        physics_weight=0.50,
        entropy_weight=0.15,
        # Network architecture will be adapted based on system size
        nf_hidden_dims=[256, 256],
        nqs_hidden_dims=[256, 256, 256],
        # Training parameters
        samples_per_batch=2000,
        max_epochs=300,
        min_epochs=100,
        convergence_threshold=0.20,
        # Basis management - will be scaled by adapt_to_system_size()
        use_diversity_selection=True,
        max_diverse_configs=2048,  # Base value, will be scaled
        rank_2_fraction=0.50,
        # Residual expansion - will be scaled
        use_residual_expansion=True,
        residual_iterations=8,
        residual_configs_per_iter=150,
        residual_threshold=1e-6,
        use_perturbative_selection=True,  # Use 2nd-order PT for better selection
        # SKQD parameters
        max_krylov_dim=10,
        shots_per_krylov=50000,
        skqd_regularization=1e-8,  # Numerical stability
        # Base accumulated basis - will be scaled
        max_accumulated_basis=4096,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    if config_overrides:
        base_config.update(config_overrides)

    # ============================================================
    # Test 1: Residual expansion ONLY (no SKQD)
    # ============================================================
    print("\n" + "-" * 50)
    print("TEST 1: Residual Expansion ONLY (no SKQD)")
    print("-" * 50)

    config_no_skqd = EnhancedPipelineConfig(**base_config)
    config_no_skqd.max_krylov_dim = 0  # Disable SKQD

    start = time.time()
    pipeline1 = EnhancedFlowKrylovPipeline(H, config=config_no_skqd, exact_energy=E_exact)

    # Run full pipeline (SKQD will be skipped due to max_krylov_dim=0)
    results1 = pipeline1.run(progress=True)

    E_residual_only = results1.get("residual_energy", results1.get("combined_energy"))

    time_residual_only = time.time() - start

    print(f"\nResidual-only energy: {E_residual_only:.8f} Ha")
    print(f"Time: {time_residual_only:.1f}s")
    if E_exact:
        error_residual = abs(E_residual_only - E_exact) * 1000
        print(f"Error: {error_residual:.4f} mHa ({error_residual * 0.6275:.4f} kcal/mol)")

    # ============================================================
    # Test 2: Residual expansion + SKQD
    # ============================================================
    print("\n" + "-" * 50)
    print("TEST 2: Residual Expansion + SKQD")
    print("-" * 50)

    config_with_skqd = EnhancedPipelineConfig(**base_config)

    start = time.time()
    pipeline2 = EnhancedFlowKrylovPipeline(H, config=config_with_skqd, exact_energy=E_exact)
    results2 = pipeline2.run(progress=True)

    E_with_skqd = results2.get('combined_energy', results2.get('skqd_energy'))
    time_with_skqd = time.time() - start

    print(f"\nResidual+SKQD energy: {E_with_skqd:.8f} Ha")
    print(f"Time: {time_with_skqd:.1f}s")
    if E_exact:
        error_skqd = abs(E_with_skqd - E_exact) * 1000
        print(f"Error: {error_skqd:.4f} mHa ({error_skqd * 0.6275:.4f} kcal/mol)")

    # ============================================================
    # Comparison
    # ============================================================
    print("\n" + "-" * 50)
    print("COMPARISON")
    print("-" * 50)

    skqd_improvement = (E_residual_only - E_with_skqd) * 1000  # mHa (lower is better)
    print(f"SKQD improvement: {skqd_improvement:.4f} mHa")
    print(f"  (positive = SKQD found lower energy)")

    if E_exact:
        error_residual = abs(E_residual_only - E_exact) * 1000
        error_skqd = abs(E_with_skqd - E_exact) * 1000

        print(f"\nResidual-only error: {error_residual:.4f} mHa")
        print(f"Residual+SKQD error: {error_skqd:.4f} mHa")

        if error_residual > 0.01:  # Avoid division by near-zero
            pct_improvement = ((error_residual - error_skqd) / error_residual) * 100
            print(f"Error reduction: {pct_improvement:.1f}%")

        chemical_accuracy = 1.6  # mHa
        residual_passes = error_residual < chemical_accuracy
        skqd_passes = error_skqd < chemical_accuracy

        print(f"\nChemical accuracy (< 1.6 mHa):")
        print(f"  Residual-only: {'PASS' if residual_passes else 'FAIL'}")
        print(f"  Residual+SKQD: {'PASS' if skqd_passes else 'FAIL'}")

        skqd_needed = not residual_passes and skqd_passes
        print(f"\nSKQD NECESSARY? {'YES' if skqd_needed else 'NO'}")
    else:
        # Without exact energy, judge by SKQD improvement
        print("\n(No exact energy available for comparison)")
        print(f"SKQD {'HELPFUL' if skqd_improvement > 0.1 else 'NOT HELPFUL'} (improvement > 0.1 mHa)")

    return {
        'molecule': name,
        'n_valid': n_valid,
        'hilbert_dim': H.hilbert_dim,
        'E_exact': E_exact,
        'E_residual_only': E_residual_only,
        'E_with_skqd': E_with_skqd,
        'time_residual_only': time_residual_only,
        'time_with_skqd': time_with_skqd,
        'error_residual': error_residual if E_exact else None,
        'error_skqd': error_skqd if E_exact else None,
        'skqd_improvement_mha': (E_residual_only - E_with_skqd) * 1000,
    }


def main():
    print("=" * 70)
    print("SKQD NECESSITY TEST")
    print("When is SKQD needed beyond residual expansion?")
    print("=" * 70)

    results = []

    # Test molecules in order of increasing valid config space
    # Note: Adaptive scaling is now automatic via EnhancedPipelineConfig.adapt_to_system_size()
    molecules = [
        # ("BeH2", create_beh2_hamiltonian, None),  # 1,225 valid - small tier
        ("NH3", create_nh3_hamiltonian, None),    # 3,136 valid - medium tier
        ("N2", create_n2_hamiltonian, {           # 14,400 valid - large tier
            # Override for strongly correlated system
            'max_epochs': 500,
            'physics_weight': 0.55,  # More physics guidance for N2 triple bond
        }),
        ("CH4", create_ch4_hamiltonian, {         # 15,876 valid - large tier
            # CH4 has weaker correlation, default should work
            'max_epochs': 400,
        }),
    ]

    for name, create_fn, overrides in molecules:
        try:
            result = test_molecule(name, create_fn, overrides)
            results.append(result)
        except Exception as e:
            print(f"\nERROR testing {name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY: SKQD NECESSITY BY MOLECULE SIZE")
    print("=" * 80)
    print(f"{'Molecule':<8} {'Valid':<10} {'Hilbert':<12} {'Res (mHa)':<12} {'+SKQD (mHa)':<12} {'Δ (mHa)':<10} {'Needed?':<8}")
    print("-" * 80)

    for r in results:
        err_res = f"{r['error_residual']:.2f}" if r['error_residual'] is not None else "N/A"
        err_skqd = f"{r['error_skqd']:.2f}" if r['error_skqd'] is not None else "N/A"
        delta = f"{r['skqd_improvement_mha']:.2f}"

        if r['error_residual'] is not None and r['error_skqd'] is not None:
            needed = "YES" if r['error_residual'] > 1.6 and r['error_skqd'] <= 1.6 else "NO"
        elif r['skqd_improvement_mha'] > 0.5:
            needed = "LIKELY"
        else:
            needed = "NO"

        print(f"{r['molecule']:<8} {r['n_valid']:<10,} {r['hilbert_dim']:<12,} {err_res:<12} {err_skqd:<12} {delta:<10} {needed:<8}")

    print("-" * 80)
    print("\nConclusion:")
    print("- Small valid space (<1000): Residual expansion often sufficient")
    print("- Medium valid space (1000-10000): SKQD may provide marginal improvement")
    print("- Large valid space (>10000): SKQD likely needed for chemical accuracy")


if __name__ == "__main__":
    main()
