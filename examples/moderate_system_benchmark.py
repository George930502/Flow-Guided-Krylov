"""
Moderate System Benchmark for SKQD Scaling Validation.

Tests molecular systems in the 20-30 qubit range to validate
SKQD necessity scaling between CH4 and very large systems.

Systems tested:
- CO (carbon monoxide): 20 qubits, 14 electrons
- C2H2 (acetylene): 24 qubits, 14 electrons
- H2O 6-31G: 26 qubits, 10 electrons
- C2H4 (ethylene): 28 qubits, 16 electrons
- NH3 6-31G: 30 qubits, 10 electrons
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hamiltonians.molecular import MolecularHamiltonian
from nqs.flow_nqs import FlowNQS
from skqd.krylov import SKQD
from pipeline.flow_krylov_pipeline import FlowKrylovPipeline


def create_co_hamiltonian():
    """
    Carbon monoxide (CO) in STO-3G basis.

    - 14 electrons (6 from C, 8 from O)
    - 10 spatial orbitals = 20 spin-orbitals (qubits)
    - Estimated valid configs: ~15,000-20,000
    """
    geometry = [
        ("C", (0.0, 0.0, 0.0)),
        ("O", (0.0, 0.0, 1.128)),  # Bond length ~1.128 Å
    ]

    hamiltonian = MolecularHamiltonian(
        geometry=geometry,
        basis="sto-3g",
        charge=0,
        spin=0,
    )

    return hamiltonian, "CO (STO-3G)"


def create_c2h2_hamiltonian():
    """
    Acetylene (C2H2) in STO-3G basis.

    - 14 electrons
    - 12 spatial orbitals = 24 spin-orbitals (qubits)
    - Linear geometry with triple bond
    """
    geometry = [
        ("C", (0.0, 0.0, 0.0)),
        ("C", (0.0, 0.0, 1.203)),  # C≡C bond ~1.203 Å
        ("H", (0.0, 0.0, -1.063)),  # C-H bond ~1.063 Å
        ("H", (0.0, 0.0, 2.266)),
    ]

    hamiltonian = MolecularHamiltonian(
        geometry=geometry,
        basis="sto-3g",
        charge=0,
        spin=0,
    )

    return hamiltonian, "C2H2 (STO-3G)"


def create_h2o_631g_hamiltonian():
    """
    Water (H2O) in 6-31G basis.

    - 10 electrons
    - 13 spatial orbitals = 26 spin-orbitals (qubits)
    - Larger basis than STO-3G for more correlation
    """
    geometry = [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (0.757, 0.587, 0.0)),
        ("H", (-0.757, 0.587, 0.0)),
    ]

    hamiltonian = MolecularHamiltonian(
        geometry=geometry,
        basis="6-31g",
        charge=0,
        spin=0,
    )

    return hamiltonian, "H2O (6-31G)"


def create_c2h4_hamiltonian():
    """
    Ethylene (C2H4) in STO-3G basis.

    - 16 electrons
    - 14 spatial orbitals = 28 spin-orbitals (qubits)
    - Planar geometry with double bond
    """
    geometry = [
        ("C", (0.0, 0.0, 0.667)),
        ("C", (0.0, 0.0, -0.667)),  # C=C bond ~1.334 Å
        ("H", (0.0, 0.923, 1.238)),
        ("H", (0.0, -0.923, 1.238)),
        ("H", (0.0, 0.923, -1.238)),
        ("H", (0.0, -0.923, -1.238)),
    ]

    hamiltonian = MolecularHamiltonian(
        geometry=geometry,
        basis="sto-3g",
        charge=0,
        spin=0,
    )

    return hamiltonian, "C2H4 (STO-3G)"


def create_nh3_631g_hamiltonian():
    """
    Ammonia (NH3) in 6-31G basis.

    - 10 electrons
    - 15 spatial orbitals = 30 spin-orbitals (qubits)
    - Pyramidal geometry
    """
    # NH3 pyramidal geometry
    bond_length = 1.012  # N-H bond in Å
    angle = 106.7 * np.pi / 180  # H-N-H angle

    # Calculate H positions
    h_z = bond_length * np.cos(np.pi - angle/2)
    h_r = bond_length * np.sin(np.pi - angle/2)

    geometry = [
        ("N", (0.0, 0.0, 0.0)),
        ("H", (h_r, 0.0, h_z)),
        ("H", (-h_r/2, h_r*np.sqrt(3)/2, h_z)),
        ("H", (-h_r/2, -h_r*np.sqrt(3)/2, h_z)),
    ]

    hamiltonian = MolecularHamiltonian(
        geometry=geometry,
        basis="6-31g",
        charge=0,
        spin=0,
    )

    return hamiltonian, "NH3 (6-31G)"


def create_hcn_hamiltonian():
    """
    Hydrogen cyanide (HCN) in STO-3G basis.

    - 14 electrons
    - 11 spatial orbitals = 22 spin-orbitals (qubits)
    - Linear geometry
    """
    geometry = [
        ("H", (0.0, 0.0, 0.0)),
        ("C", (0.0, 0.0, 1.066)),  # H-C bond ~1.066 Å
        ("N", (0.0, 0.0, 2.222)),  # C≡N bond ~1.156 Å
    ]

    hamiltonian = MolecularHamiltonian(
        geometry=geometry,
        basis="sto-3g",
        charge=0,
        spin=0,
    )

    return hamiltonian, "HCN (STO-3G)"


def create_h2s_hamiltonian():
    """
    Hydrogen sulfide (H2S) in STO-3G basis.

    - 18 electrons
    - 13 spatial orbitals = 26 spin-orbitals (qubits)
    - Bent geometry similar to H2O
    """
    geometry = [
        ("S", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.967, 0.923)),  # S-H bond ~1.336 Å
        ("H", (0.0, -0.967, 0.923)),  # H-S-H angle ~92.1°
    ]

    hamiltonian = MolecularHamiltonian(
        geometry=geometry,
        basis="sto-3g",
        charge=0,
        spin=0,
    )

    return hamiltonian, "H2S (STO-3G)"


def run_moderate_benchmark(
    hamiltonian: MolecularHamiltonian,
    system_name: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    nf_epochs: int = 200,
    residual_iterations: int = 15,
    krylov_dim: int = 8,
    krylov_shots: int = 5000,
    max_residual_configs: int = 300,
):
    """
    Run benchmark comparing NF, Residual, and SKQD contributions.

    Returns detailed metrics for SKQD necessity analysis.
    """
    print(f"\n{'='*60}")
    print(f"MODERATE SYSTEM BENCHMARK: {system_name}")
    print(f"{'='*60}")

    # System info
    n_qubits = hamiltonian.num_qubits
    n_electrons = hamiltonian.n_electrons

    print(f"\nSystem Information:")
    print(f"  Qubits: {n_qubits}")
    print(f"  Electrons: {n_electrons}")

    # Count valid configurations
    print(f"\nCounting valid configurations...")
    valid_configs = hamiltonian.get_valid_configurations()
    n_valid = len(valid_configs)
    print(f"  Valid configurations: {n_valid:,}")

    # Check if FCI is feasible
    fci_feasible = n_valid <= 50000
    fci_energy = None

    if fci_feasible:
        print(f"\nComputing FCI reference (feasible with {n_valid:,} configs)...")
        try:
            fci_energy = hamiltonian.compute_fci_energy()
            print(f"  FCI Energy: {fci_energy:.6f} Ha")
        except Exception as e:
            print(f"  FCI computation failed: {e}")
            fci_feasible = False
    else:
        print(f"\nFCI not feasible ({n_valid:,} configs > 50,000)")
        # Use CCSD as reference
        try:
            ccsd_energy = hamiltonian.compute_ccsd_energy()
            print(f"  CCSD Energy: {ccsd_energy:.6f} Ha (reference)")
        except:
            pass

    # Initialize pipeline components
    print(f"\nInitializing pipeline...")

    # Create NQS
    nqs = FlowNQS(
        num_sites=n_qubits,
        num_particles=n_electrons,
        hidden_dim=128,
        num_flow_layers=8,
        device=device,
    )

    # Create SKQD
    skqd = SKQD(
        hamiltonian=hamiltonian,
        krylov_dim=krylov_dim,
        num_shots=krylov_shots,
        dt=0.1,
        device=device,
    )

    # Create pipeline
    pipeline = FlowKrylovPipeline(
        hamiltonian=hamiltonian,
        nqs=nqs,
        skqd=skqd,
        device=device,
    )

    # Track configurations from each method
    results = {
        "system": system_name,
        "n_qubits": n_qubits,
        "n_electrons": n_electrons,
        "n_valid_configs": n_valid,
        "fci_energy": fci_energy,
    }

    # Stage 1: NF Training
    print(f"\n--- Stage 1: NF Training ({nf_epochs} epochs) ---")
    start_time = time.time()

    pipeline.train_nqs(
        num_epochs=nf_epochs,
        batch_size=min(512, n_valid),
        learning_rate=1e-3,
    )

    nf_time = time.time() - start_time
    nf_configs = pipeline.get_basis_configurations()
    results["nf_configs"] = len(nf_configs)
    results["nf_time"] = nf_time

    print(f"  NF found {len(nf_configs)} configurations in {nf_time:.1f}s")

    # Compute NF-only energy
    nf_energy = pipeline.compute_basis_energy(nf_configs)
    results["nf_energy"] = nf_energy
    if fci_energy:
        nf_error = (nf_energy - fci_energy) * 1000  # mHa
        results["nf_error_mha"] = nf_error
        print(f"  NF Energy: {nf_energy:.6f} Ha (error: {nf_error:.4f} mHa)")

    # Stage 2: Residual Expansion
    print(f"\n--- Stage 2: Residual Expansion ({residual_iterations} iterations) ---")
    start_time = time.time()

    residual_configs_before = set(map(tuple, nf_configs.cpu().numpy()))

    pipeline.expand_basis_residual(
        max_iterations=residual_iterations,
        max_new_configs_per_iter=max_residual_configs,
        convergence_threshold=1e-4,
    )

    residual_time = time.time() - start_time
    all_configs_after_residual = pipeline.get_basis_configurations()
    residual_configs_after = set(map(tuple, all_configs_after_residual.cpu().numpy()))

    # Configs found by residual
    residual_new = residual_configs_after - residual_configs_before
    results["residual_new_configs"] = len(residual_new)
    results["residual_time"] = residual_time

    print(f"  Residual found {len(residual_new)} new configurations in {residual_time:.1f}s")
    print(f"  Total after residual: {len(residual_configs_after)}")

    # Compute NF+Residual energy
    nf_residual_energy = pipeline.compute_basis_energy(all_configs_after_residual)
    results["nf_residual_energy"] = nf_residual_energy
    if fci_energy:
        nf_residual_error = (nf_residual_energy - fci_energy) * 1000
        results["nf_residual_error_mha"] = nf_residual_error
        print(f"  NF+Residual Energy: {nf_residual_energy:.6f} Ha (error: {nf_residual_error:.4f} mHa)")

    # Stage 3: SKQD
    print(f"\n--- Stage 3: SKQD (dim={krylov_dim}, shots={krylov_shots}) ---")
    start_time = time.time()

    # Set initial state from current best eigenvector
    eigenvector = pipeline.get_ground_state_vector()
    if eigenvector is not None:
        # Find dominant configuration
        max_idx = torch.argmax(torch.abs(eigenvector))
        initial_config = all_configs_after_residual[max_idx]
        skqd.initial_state = initial_config

    # Run SKQD
    skqd_configs = skqd.run()
    skqd_time = time.time() - start_time

    skqd_configs_set = set(map(tuple, skqd_configs.cpu().numpy()))

    # Krylov-unique configs (not found by NF or Residual)
    krylov_unique = skqd_configs_set - residual_configs_after
    results["krylov_new_configs"] = len(skqd_configs_set)
    results["krylov_unique_configs"] = len(krylov_unique)
    results["krylov_time"] = skqd_time

    print(f"  SKQD found {len(skqd_configs_set)} configurations in {skqd_time:.1f}s")
    print(f"  Krylov-unique (not in NF+Residual): {len(krylov_unique)}")

    # Combined basis
    combined_configs_set = residual_configs_after | skqd_configs_set
    combined_configs = torch.tensor(list(combined_configs_set), device=device)
    results["combined_configs"] = len(combined_configs_set)

    # Compute NF+Krylov energy (without residual)
    nf_krylov_set = residual_configs_before | skqd_configs_set
    nf_krylov_configs = torch.tensor(list(nf_krylov_set), device=device)
    nf_krylov_energy = pipeline.compute_basis_energy(nf_krylov_configs)
    results["nf_krylov_energy"] = nf_krylov_energy
    if fci_energy:
        nf_krylov_error = (nf_krylov_energy - fci_energy) * 1000
        results["nf_krylov_error_mha"] = nf_krylov_error
        print(f"  NF+Krylov Energy: {nf_krylov_energy:.6f} Ha (error: {nf_krylov_error:.4f} mHa)")

    # Compute combined energy
    combined_energy = pipeline.compute_basis_energy(combined_configs)
    results["combined_energy"] = combined_energy
    if fci_energy:
        combined_error = (combined_energy - fci_energy) * 1000
        results["combined_error_mha"] = combined_error
        print(f"  Combined Energy: {combined_energy:.6f} Ha (error: {combined_error:.4f} mHa)")

    # Determine SKQD necessity
    print(f"\n--- SKQD Necessity Analysis ---")

    if len(krylov_unique) > 0:
        # Calculate energy improvement from Krylov-unique configs
        krylov_improvement = (nf_residual_energy - combined_energy) * 1000  # mHa
        results["krylov_improvement_mha"] = krylov_improvement

        if krylov_improvement > 0.1:  # More than 0.1 mHa improvement
            verdict = "NECESSARY"
            reason = f"Found {len(krylov_unique)} unique configs, {krylov_improvement:.2f} mHa improvement"
        else:
            verdict = "HELPFUL"
            reason = f"Found {len(krylov_unique)} unique configs, but minimal energy improvement"
    else:
        verdict = "REDUNDANT"
        reason = "All Krylov configs already found by NF+Residual"
        results["krylov_improvement_mha"] = 0.0

    results["verdict"] = verdict
    results["reason"] = reason

    print(f"\n  Verdict: {verdict}")
    print(f"  Reason: {reason}")

    # Summary table
    print(f"\n{'='*60}")
    print(f"SUMMARY: {system_name}")
    print(f"{'='*60}")
    print(f"  Qubits: {n_qubits}")
    print(f"  Valid Configs: {n_valid:,}")
    print(f"  NF Configs: {results['nf_configs']}")
    print(f"  Residual New: {results['residual_new_configs']}")
    print(f"  Krylov New: {results['krylov_new_configs']}")
    print(f"  Krylov Unique: {results['krylov_unique_configs']}")
    if fci_energy:
        print(f"  NF Error: {results.get('nf_error_mha', 'N/A'):.4f} mHa")
        print(f"  NF+Residual Error: {results.get('nf_residual_error_mha', 'N/A'):.4f} mHa")
        print(f"  NF+Krylov Error: {results.get('nf_krylov_error_mha', 'N/A'):.4f} mHa")
        print(f"  Combined Error: {results.get('combined_error_mha', 'N/A'):.4f} mHa")
    print(f"  SKQD Verdict: {verdict}")
    print(f"{'='*60}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Moderate System SKQD Benchmark")
    parser.add_argument(
        "--system",
        type=str,
        default="all",
        choices=["co", "c2h2", "h2o_631g", "c2h4", "nh3_631g", "hcn", "h2s", "all"],
        help="System to benchmark",
    )
    parser.add_argument("--nf-epochs", type=int, default=200, help="NF training epochs")
    parser.add_argument("--krylov-dim", type=int, default=8, help="Krylov dimension")
    parser.add_argument("--krylov-shots", type=int, default=5000, help="Krylov shots")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # System factory
    systems = {
        "co": create_co_hamiltonian,
        "hcn": create_hcn_hamiltonian,
        "c2h2": create_c2h2_hamiltonian,
        "h2o_631g": create_h2o_631g_hamiltonian,
        "h2s": create_h2s_hamiltonian,
        "c2h4": create_c2h4_hamiltonian,
        "nh3_631g": create_nh3_631g_hamiltonian,
    }

    # Order by qubit count (ascending)
    system_order = ["co", "hcn", "c2h2", "h2o_631g", "h2s", "c2h4", "nh3_631g"]

    if args.system == "all":
        systems_to_run = system_order
    else:
        systems_to_run = [args.system]

    all_results = []

    for system_key in systems_to_run:
        try:
            hamiltonian, name = systems[system_key]()

            result = run_moderate_benchmark(
                hamiltonian=hamiltonian,
                system_name=name,
                device=args.device,
                nf_epochs=args.nf_epochs,
                krylov_dim=args.krylov_dim,
                krylov_shots=args.krylov_shots,
            )

            all_results.append(result)

        except Exception as e:
            print(f"\nError running {system_key}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Final summary
    if len(all_results) > 1:
        print("\n" + "="*80)
        print("OVERALL RESULTS SUMMARY")
        print("="*80)
        print(f"\n{'System':<20} {'Qubits':>8} {'Valid':>12} {'Krylov-Unique':>15} {'Verdict':<12}")
        print("-"*80)

        for r in all_results:
            print(f"{r['system']:<20} {r['n_qubits']:>8} {r['n_valid_configs']:>12,} "
                  f"{r['krylov_unique_configs']:>15} {r['verdict']:<12}")

        print("-"*80)

        # Count verdicts
        necessary_count = sum(1 for r in all_results if r['verdict'] == 'NECESSARY')
        helpful_count = sum(1 for r in all_results if r['verdict'] == 'HELPFUL')
        redundant_count = sum(1 for r in all_results if r['verdict'] == 'REDUNDANT')

        print(f"\nSKQD Necessity Summary:")
        print(f"  NECESSARY: {necessary_count} systems")
        print(f"  HELPFUL: {helpful_count} systems")
        print(f"  REDUNDANT: {redundant_count} systems")

        # Find threshold
        sorted_results = sorted(all_results, key=lambda x: x['n_valid_configs'])
        threshold_found = False
        for i, r in enumerate(sorted_results):
            if r['krylov_unique_configs'] > 0 and not threshold_found:
                print(f"\n  SKQD becomes necessary around {r['n_valid_configs']:,} valid configs")
                print(f"  ({r['system']}, {r['n_qubits']} qubits)")
                threshold_found = True
                break


if __name__ == "__main__":
    main()
