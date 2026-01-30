"""
Moderate System Benchmark for SKQD Scaling Validation.

Tests molecular systems in the 20-30 qubit range to validate
SKQD necessity scaling between CH4 and very large systems.

Systems tested (ordered by qubit count):
- CO (carbon monoxide): 20 qubits, 14 electrons
- HCN (hydrogen cyanide): 22 qubits, 14 electrons
- C2H2 (acetylene): 24 qubits, 14 electrons
- H2O 6-31G: 26 qubits, 10 electrons
- H2S (hydrogen sulfide): 26 qubits, 18 electrons
- C2H4 (ethylene): 28 qubits, 16 electrons
- NH3 6-31G: 30 qubits, 10 electrons

Usage:
    docker-compose run --rm flow-krylov-gpu python examples/moderate_system_benchmark.py --system all
    docker-compose run --rm flow-krylov-gpu python examples/moderate_system_benchmark.py --system c2h4
"""

import sys
from pathlib import Path
import argparse
import time
from math import comb
from typing import Dict, Any, Set, Tuple, Optional
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np

try:
    from pyscf import gto, scf, ao2mo, cc
    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False
    print("ERROR: PySCF required")
    sys.exit(1)

from hamiltonians.molecular import MolecularHamiltonian, MolecularIntegrals
from pipeline import FlowGuidedKrylovPipeline, PipelineConfig
from krylov.skqd import SampleBasedKrylovDiagonalization, FlowGuidedSKQD, SKQDConfig
from krylov.residual_expansion import SelectedCIExpander, ResidualExpansionConfig


@dataclass
class MoleculeData:
    """Container for molecule data including Hamiltonian and reference energies."""
    hamiltonian: MolecularHamiltonian
    hf_energy: float
    ccsd_energy: Optional[float] = None
    geometry: list = None
    basis: str = "sto-3g"


@dataclass
class BenchmarkResult:
    """Result of a benchmark test."""
    system: str
    n_qubits: int
    n_electrons: int
    n_valid_configs: int
    exact_energy: float
    energy_type: str = "FCI"  # FCI, CCSD, or HF

    # Configuration counts
    nf_configs: int = 0
    residual_new_configs: int = 0
    krylov_new_configs: int = 0
    krylov_unique_configs: int = 0

    # Energies
    nf_energy: float = 0.0
    nf_residual_energy: float = 0.0
    nf_krylov_energy: float = 0.0
    combined_energy: float = 0.0

    # Verdict
    verdict: str = ""
    krylov_improvement_mha: float = 0.0


def print_banner(title: str):
    """Print a formatted banner."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def configs_to_set(configs: torch.Tensor) -> Set[tuple]:
    """Convert tensor of configurations to set of tuples."""
    return {tuple(c.cpu().tolist()) for c in configs}


def set_to_configs(config_set: Set[tuple], n_sites: int, device: str) -> torch.Tensor:
    """Convert set of tuples back to tensor."""
    configs = [list(c) for c in config_set]
    return torch.tensor(configs, dtype=torch.long, device=device)


def compute_basis_energy(H: MolecularHamiltonian, basis: torch.Tensor) -> float:
    """Compute ground state energy by diagonalizing H in given basis."""
    H_matrix = H.matrix_elements(basis, basis)
    H_np = H_matrix.cpu().numpy().astype(np.float64)
    H_np = 0.5 * (H_np + H_np.T)
    eigenvalues, _ = np.linalg.eigh(H_np)
    return float(eigenvalues[0])


# =============================================================================
# Helper: Create Hamiltonian with Reference Energies
# =============================================================================

def create_molecule_data(
    geometry: list,
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    compute_ccsd: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> MoleculeData:
    """
    Create MoleculeData with Hamiltonian and reference energies.

    Returns MoleculeData containing:
    - MolecularHamiltonian
    - HF energy
    - CCSD energy (if requested)
    """
    # Build PySCF molecule
    mol = gto.Mole()
    mol.atom = geometry
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.build()

    # Run HF
    if spin == 0:
        mf = scf.RHF(mol)
    else:
        mf = scf.ROHF(mol)
    mf.kernel()
    hf_energy = float(mf.e_tot)

    # Run CCSD if requested
    ccsd_energy = None
    if compute_ccsd:
        try:
            mycc = cc.CCSD(mf)
            mycc.kernel()
            ccsd_energy = float(mycc.e_tot)
        except Exception as e:
            print(f"  CCSD failed: {e}")

    # Get integrals in MO basis
    h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    h2e = ao2mo.kernel(mol, mf.mo_coeff)
    h2e = ao2mo.restore(1, h2e, mol.nao)

    n_electrons = mol.nelectron
    n_orbitals = mol.nao
    n_alpha = (n_electrons + spin) // 2
    n_beta = (n_electrons - spin) // 2

    integrals = MolecularIntegrals(
        h1e=h1e,
        h2e=h2e,
        nuclear_repulsion=mol.energy_nuc(),
        n_electrons=n_electrons,
        n_orbitals=n_orbitals,
        n_alpha=n_alpha,
        n_beta=n_beta,
    )

    hamiltonian = MolecularHamiltonian(integrals, device=device)

    return MoleculeData(
        hamiltonian=hamiltonian,
        hf_energy=hf_energy,
        ccsd_energy=ccsd_energy,
        geometry=geometry,
        basis=basis,
    )


# =============================================================================
# Hamiltonian Factories for Moderate Systems
# =============================================================================

def create_co_molecule(
    bond_length: float = 1.128,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> MoleculeData:
    """
    Create CO (carbon monoxide) molecule data.

    14 electrons, 10 orbitals in STO-3G = 20 qubits
    Valid configs: C(10,7)² = 14,400
    """
    geometry = [
        ("C", (0.0, 0.0, 0.0)),
        ("O", (0.0, 0.0, bond_length)),
    ]
    return create_molecule_data(geometry, basis="sto-3g", device=device)


def create_hcn_molecule(
    ch_length: float = 1.066,
    cn_length: float = 1.156,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> MoleculeData:
    """
    Create HCN (hydrogen cyanide) molecule data.

    14 electrons, 11 orbitals in STO-3G = 22 qubits
    Linear geometry: H-C≡N
    """
    geometry = [
        ("H", (0.0, 0.0, 0.0)),
        ("C", (0.0, 0.0, ch_length)),
        ("N", (0.0, 0.0, ch_length + cn_length)),
    ]
    return create_molecule_data(geometry, basis="sto-3g", device=device)


def create_c2h2_molecule(
    cc_length: float = 1.203,
    ch_length: float = 1.063,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> MoleculeData:
    """
    Create C2H2 (acetylene) molecule data.

    14 electrons, 12 orbitals in STO-3G = 24 qubits
    Linear geometry: H-C≡C-H
    """
    geometry = [
        ("H", (0.0, 0.0, -ch_length - cc_length/2)),
        ("C", (0.0, 0.0, -cc_length/2)),
        ("C", (0.0, 0.0, cc_length/2)),
        ("H", (0.0, 0.0, ch_length + cc_length/2)),
    ]
    return create_molecule_data(geometry, basis="sto-3g", device=device)


def create_h2o_631g_molecule(
    oh_length: float = 0.96,
    angle: float = 104.5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> MoleculeData:
    """
    Create H2O (water) molecule data with 6-31G basis.

    10 electrons, 13 orbitals in 6-31G = 26 qubits
    """
    angle_rad = np.radians(angle)
    geometry = [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (oh_length, 0.0, 0.0)),
        ("H", (oh_length * np.cos(angle_rad), oh_length * np.sin(angle_rad), 0.0)),
    ]
    return create_molecule_data(geometry, basis="6-31g", device=device)


def create_h2s_molecule(
    sh_length: float = 1.336,
    angle: float = 92.1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> MoleculeData:
    """
    Create H2S (hydrogen sulfide) molecule data.

    18 electrons, 13 orbitals in STO-3G = 26 qubits
    """
    angle_rad = np.radians(angle)
    geometry = [
        ("S", (0.0, 0.0, 0.0)),
        ("H", (sh_length, 0.0, 0.0)),
        ("H", (sh_length * np.cos(angle_rad), sh_length * np.sin(angle_rad), 0.0)),
    ]
    return create_molecule_data(geometry, basis="sto-3g", device=device)


def create_c2h4_molecule(
    cc_length: float = 1.339,
    ch_length: float = 1.087,
    hcc_angle: float = 121.3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> MoleculeData:
    """
    Create C2H4 (ethylene) molecule data.

    16 electrons, 14 orbitals in STO-3G = 28 qubits
    Planar geometry
    """
    angle_rad = np.radians(hcc_angle)

    geometry = [
        # Carbon atoms along z-axis
        ("C", (0.0, 0.0, -cc_length/2)),
        ("C", (0.0, 0.0, cc_length/2)),
        # H atoms on first carbon (in xz plane)
        ("H", (ch_length * np.sin(angle_rad), 0.0, -cc_length/2 - ch_length * np.cos(angle_rad))),
        ("H", (-ch_length * np.sin(angle_rad), 0.0, -cc_length/2 - ch_length * np.cos(angle_rad))),
        # H atoms on second carbon
        ("H", (ch_length * np.sin(angle_rad), 0.0, cc_length/2 + ch_length * np.cos(angle_rad))),
        ("H", (-ch_length * np.sin(angle_rad), 0.0, cc_length/2 + ch_length * np.cos(angle_rad))),
    ]
    return create_molecule_data(geometry, basis="sto-3g", device=device)


def create_nh3_631g_molecule(
    nh_length: float = 1.012,
    hnh_angle: float = 106.7,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> MoleculeData:
    """
    Create NH3 (ammonia) molecule data with 6-31G basis.

    10 electrons, 15 orbitals in 6-31G = 30 qubits
    """
    angle_rad = np.radians(hnh_angle)
    # Height of N above H plane
    h = nh_length * np.cos(np.arcsin(np.sin(angle_rad/2) / np.sin(np.radians(60))))
    r = np.sqrt(nh_length**2 - h**2)  # Radius of H triangle

    geometry = [
        ("N", (0.0, 0.0, h)),
        ("H", (r, 0.0, 0.0)),
        ("H", (r * np.cos(np.radians(120)), r * np.sin(np.radians(120)), 0.0)),
        ("H", (r * np.cos(np.radians(240)), r * np.sin(np.radians(240)), 0.0)),
    ]
    return create_molecule_data(geometry, basis="6-31g", device=device)


# =============================================================================
# Benchmark Runner
# =============================================================================

def run_benchmark(
    molecule_key: str,
    verbose: bool = True,
) -> BenchmarkResult:
    """
    Run SKQD necessity benchmark for a single molecule.

    Args:
        molecule_key: Key for molecule factory
        verbose: Print detailed progress
    """
    # System factories
    factories = {
        'co': (create_co_molecule, "CO (STO-3G)"),
        'hcn': (create_hcn_molecule, "HCN (STO-3G)"),
        'c2h2': (create_c2h2_molecule, "C2H2 (STO-3G)"),
        'h2o_631g': (create_h2o_631g_molecule, "H2O (6-31G)"),
        'h2s': (create_h2s_molecule, "H2S (STO-3G)"),
        'c2h4': (create_c2h4_molecule, "C2H4 (STO-3G)"),
        'nh3_631g': (create_nh3_631g_molecule, "NH3 (6-31G)"),
    }

    if molecule_key not in factories:
        raise ValueError(f"Unknown molecule: {molecule_key}")

    factory_fn, system_name = factories[molecule_key]

    print_banner(f"BENCHMARK: {system_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create molecule data with Hamiltonian and reference energies
    print("Creating Hamiltonian and computing reference energies...")
    mol_data = factory_fn(device=device)
    H = mol_data.hamiltonian

    n_qubits = H.num_sites
    n_electrons = H.n_electrons
    n_valid = comb(H.n_orbitals, H.n_alpha) * comb(H.n_orbitals, H.n_beta)

    print(f"  System: {system_name}")
    print(f"  Qubits: {n_qubits}")
    print(f"  Electrons: {n_electrons}")
    print(f"  Valid configs: {n_valid:,}")
    print(f"  HF Energy: {mol_data.hf_energy:.8f} Ha")
    if mol_data.ccsd_energy:
        print(f"  CCSD Energy: {mol_data.ccsd_energy:.8f} Ha")

    # Determine reference energy
    energy_type = "HF"
    if n_valid <= 100000:
        print("Computing FCI energy...")
        try:
            E_exact = H.fci_energy()
            energy_type = "FCI"
            print(f"  FCI Energy: {E_exact:.8f} Ha")
        except Exception as e:
            print(f"  FCI failed: {e}")
            if mol_data.ccsd_energy:
                E_exact = mol_data.ccsd_energy
                energy_type = "CCSD"
                print(f"  Using CCSD Energy: {E_exact:.8f} Ha (reference)")
            else:
                E_exact = mol_data.hf_energy
                print(f"  Using HF Energy: {E_exact:.8f} Ha (reference)")
    else:
        print("FCI not feasible...")
        if mol_data.ccsd_energy:
            E_exact = mol_data.ccsd_energy
            energy_type = "CCSD"
            print(f"  Using CCSD Energy: {E_exact:.8f} Ha (reference)")
        else:
            E_exact = mol_data.hf_energy
            print(f"  Using HF Energy: {E_exact:.8f} Ha (reference)")

    result = BenchmarkResult(
        system=system_name,
        n_qubits=n_qubits,
        n_electrons=n_electrons,
        n_valid_configs=n_valid,
        exact_energy=E_exact,
        energy_type=energy_type,
    )

    # =======================================================================
    # Step 1: NF-NQS Training
    # =======================================================================
    print("\n--- Step 1: NF-NQS Training ---")

    # Use PipelineConfig with adapt_to_system_size for parameter tuning
    config = PipelineConfig(
        use_residual_expansion=False,
        skip_skqd=True,
        max_epochs=400,
        device=device,
    )
    config.adapt_to_system_size(n_valid)

    pipeline = FlowGuidedKrylovPipeline(H, config=config, exact_energy=E_exact)
    pipeline.train_flow_nqs(progress=verbose)
    nf_basis = pipeline.extract_and_select_basis()

    nf_set = configs_to_set(nf_basis)
    result.nf_configs = len(nf_set)
    result.nf_energy = compute_basis_energy(H, nf_basis)

    nf_error = abs(result.nf_energy - E_exact) * 1000
    print(f"  NF found: {len(nf_set)} configs")
    print(f"  NF energy: {result.nf_energy:.8f} Ha (error: {nf_error:.4f} mHa)")

    # =======================================================================
    # Step 2: Residual Expansion
    # =======================================================================
    print("\n--- Step 2: Residual (PT2) Expansion ---")
    residual_config = ResidualExpansionConfig(
        max_configs_per_iter=config.residual_configs_per_iter,
        max_iterations=config.residual_iterations,
        residual_threshold=config.residual_threshold,
    )
    expander = SelectedCIExpander(H, residual_config)

    print(f"  Using params: {config.residual_iterations} iterations, "
          f"{config.residual_configs_per_iter} configs/iter")

    expanded_basis = nf_basis.clone()
    for i in range(config.residual_iterations):
        old_size = len(expanded_basis)
        expanded_basis, stats = expander.expand_basis(expanded_basis)
        added = stats['configs_added']
        if added == 0:
            break
        if verbose:
            print(f"  Iter {i+1}: {old_size} -> {len(expanded_basis)} (+{added})")

    residual_set = configs_to_set(expanded_basis)
    residual_new = residual_set - nf_set
    result.residual_new_configs = len(residual_new)
    result.nf_residual_energy = compute_basis_energy(H, expanded_basis)

    residual_error = abs(result.nf_residual_energy - E_exact) * 1000
    print(f"  Residual found: {len(residual_new)} NEW configs")
    print(f"  NF+Residual energy: {result.nf_residual_energy:.8f} Ha (error: {residual_error:.4f} mHa)")

    # =======================================================================
    # Step 3: Krylov Time Evolution
    # =======================================================================
    print("\n--- Step 3: Krylov Time Evolution ---")

    # Skip SKQD for very large systems where building the full Hamiltonian is infeasible
    # For C2H4 (9M configs), building the 9M x 9M Hamiltonian takes hours
    MAX_SKQD_SUBSPACE_SIZE = 100000  # 100k configs is manageable

    if n_valid > MAX_SKQD_SUBSPACE_SIZE:
        print(f"  SKIPPING: Subspace too large ({n_valid:,} > {MAX_SKQD_SUBSPACE_SIZE:,})")
        print(f"  Building {n_valid:,} x {n_valid:,} Hamiltonian is not feasible")
        print(f"  Using NF+Residual basis only for this system")

        # Set Krylov results to match residual results (no improvement)
        result.krylov_new_configs = 0
        result.nf_krylov_energy = result.nf_residual_energy
        krylov_set = set()
        nf_krylov_set = residual_set
    else:
        # Default SKQD parameters
        krylov_dim = 8
        dt = 0.1
        shots_per_krylov = 50000

        skqd_config = SKQDConfig(
            max_krylov_dim=krylov_dim,
            time_step=dt,
            shots_per_krylov=shots_per_krylov,
        )

        print(f"  Using params: krylov_dim={krylov_dim}, "
              f"dt={dt:.4f}, shots={shots_per_krylov:,}")

        skqd = FlowGuidedSKQD(H, nf_basis, skqd_config)
        skqd_results = skqd.run_with_nf(max_krylov_dim=krylov_dim, progress=verbose)

        # Collect Krylov configs
        krylov_set = set()
        cumulative = skqd.build_cumulative_basis()
        if cumulative:
            for bitstring in cumulative[-1].keys():
                config = tuple(int(b) for b in bitstring)
                krylov_set.add(config)

        krylov_new = krylov_set - nf_set
        result.krylov_new_configs = len(krylov_new)

        # Compute NF+Krylov energy
        nf_krylov_set = nf_set | krylov_set
        nf_krylov_basis = set_to_configs(nf_krylov_set, H.num_sites, device)
        result.nf_krylov_energy = compute_basis_energy(H, nf_krylov_basis)

        nf_krylov_error = abs(result.nf_krylov_energy - E_exact) * 1000
        print(f"  Krylov found: {len(krylov_new)} NEW configs")
        print(f"  NF+Krylov energy: {result.nf_krylov_energy:.8f} Ha (error: {nf_krylov_error:.4f} mHa)")

    # =======================================================================
    # Step 4: Krylov-Unique Analysis
    # =======================================================================
    print("\n--- Step 4: Krylov-Unique Analysis ---")

    krylov_unique = krylov_set - residual_set
    result.krylov_unique_configs = len(krylov_unique)

    print(f"  Krylov-UNIQUE configs: {len(krylov_unique)}")
    print(f"  (Found by Krylov but NOT by NF+Residual)")

    # Combined energy
    all_configs = nf_set | residual_set | krylov_set
    all_basis = set_to_configs(all_configs, H.num_sites, device)
    result.combined_energy = compute_basis_energy(H, all_basis)

    combined_error = abs(result.combined_energy - E_exact) * 1000
    print(f"  Combined basis: {len(all_configs)} configs")
    print(f"  Combined energy: {result.combined_energy:.8f} Ha (error: {combined_error:.4f} mHa)")

    # =======================================================================
    # Step 5: Verdict
    # =======================================================================
    print("\n--- Step 5: SKQD Necessity Verdict ---")

    # Handle skipped SKQD case
    if n_valid > MAX_SKQD_SUBSPACE_SIZE:
        result.verdict = "SKIPPED"
        result.krylov_improvement_mha = 0.0
        reason = f"SKQD skipped due to large subspace ({n_valid:,} configs)"
    elif len(krylov_unique) > 0:
        krylov_improvement = (result.nf_residual_energy - result.combined_energy) * 1000
        result.krylov_improvement_mha = krylov_improvement

        if krylov_improvement > 0.1:
            result.verdict = "NECESSARY"
            reason = f"Found {len(krylov_unique)} unique configs, {krylov_improvement:.2f} mHa improvement"
        else:
            result.verdict = "HELPFUL"
            reason = f"Found {len(krylov_unique)} unique configs, minimal energy impact"
    else:
        result.verdict = "REDUNDANT"
        result.krylov_improvement_mha = 0.0
        reason = "All Krylov configs already found by NF+Residual"

    print(f"\n  VERDICT: {result.verdict}")
    print(f"  Reason: {reason}")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: {system_name}")
    print(f"{'='*70}")
    print(f"  Qubits: {n_qubits}")
    print(f"  Valid Configs: {n_valid:,}")
    print(f"  NF Configs: {result.nf_configs}")
    print(f"  Residual New: {result.residual_new_configs}")
    print(f"  Krylov New: {result.krylov_new_configs}")
    print(f"  Krylov-UNIQUE: {result.krylov_unique_configs}")
    print(f"  SKQD Verdict: {result.verdict}")
    print(f"{'='*70}\n")

    return result


def main():
    parser = argparse.ArgumentParser(description="Moderate System SKQD Benchmark")
    parser.add_argument(
        "--system",
        type=str,
        default="all",
        choices=["co", "hcn", "c2h2", "h2o_631g", "h2s", "c2h4", "nh3_631g", "all"],
        help="System to benchmark",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")

    args = parser.parse_args()

    # Order by qubit count (ascending)
    system_order = ["co", "hcn", "c2h2", "h2o_631g", "h2s", "c2h4", "nh3_631g"]

    if args.system == "all":
        systems_to_run = system_order
    else:
        systems_to_run = [args.system]

    all_results = []

    for system_key in systems_to_run:
        try:
            result = run_benchmark(system_key, verbose=not args.quiet)
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
        print(f"\n{'System':<20} {'Qubits':>8} {'Valid':>15} {'Krylov-Unique':>15} {'Verdict':<12}")
        print("-"*80)

        for r in all_results:
            print(f"{r.system:<20} {r.n_qubits:>8} {r.n_valid_configs:>15,} "
                  f"{r.krylov_unique_configs:>15} {r.verdict:<12}")

        print("-"*80)

        # Count verdicts
        necessary_count = sum(1 for r in all_results if r.verdict == 'NECESSARY')
        helpful_count = sum(1 for r in all_results if r.verdict == 'HELPFUL')
        redundant_count = sum(1 for r in all_results if r.verdict == 'REDUNDANT')

        print(f"\nSKQD Necessity Summary:")
        print(f"  NECESSARY: {necessary_count} systems")
        print(f"  HELPFUL: {helpful_count} systems")
        print(f"  REDUNDANT: {redundant_count} systems")

        # Find threshold
        sorted_results = sorted(all_results, key=lambda x: x.n_valid_configs)
        for r in sorted_results:
            if r.krylov_unique_configs > 0:
                print(f"\n  SKQD becomes necessary around {r.n_valid_configs:,} valid configs")
                print(f"  ({r.system}, {r.n_qubits} qubits)")
                break


if __name__ == "__main__":
    main()
