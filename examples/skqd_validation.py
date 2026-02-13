"""
SKQD Validation Experiments: Demonstrating Krylov's Unique Contribution

This benchmark is designed to validate the research hypothesis:
"SKQD provides important configurations not found by NF sampling alone"

Experiment Modes:
1. ISOLATED_SKQD: Test SKQD in isolation (NF only vs NF + SKQD)
2. PROVENANCE_TRACKING: Track which method discovers each configuration
3. STRETCHED_BONDS: Test strongly correlated systems where Krylov should help
4. POOR_INITIAL_STATE: Test with random/degraded initial states
5. LARGER_BASIS: Test with larger basis sets

Usage:
    docker-compose run --rm flow-krylov-gpu python examples/skqd_validation.py --mode all
    docker-compose run --rm flow-krylov-gpu python examples/skqd_validation.py --mode isolated
    docker-compose run --rm flow-krylov-gpu python examples/skqd_validation.py --mode stretched --molecule h2o
"""

import sys
from pathlib import Path
import argparse
import time
from math import comb
from typing import Dict, Any, Tuple, Set, Optional
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np

try:
    from hamiltonians.molecular import (
        MolecularHamiltonian,
        compute_molecular_integrals,
        create_h2_hamiltonian,
        create_lih_hamiltonian,
        create_h2o_hamiltonian,
        create_n2_hamiltonian,
    )
    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False
    print("ERROR: PySCF required. Install with: pip install pyscf")
    sys.exit(1)

from pipeline import FlowGuidedKrylovPipeline, PipelineConfig
from krylov.skqd import SampleBasedKrylovDiagonalization, FlowGuidedSKQD, SKQDConfig


@dataclass
class ExperimentResult:
    """Container for experiment results."""
    name: str
    molecule: str
    exact_energy: float
    nf_only_energy: Optional[float] = None
    skqd_only_energy: Optional[float] = None
    combined_energy: Optional[float] = None
    nf_basis_size: int = 0
    skqd_unique_configs: int = 0
    total_basis_size: int = 0
    time_seconds: float = 0.0
    notes: str = ""


def print_banner(title: str):
    """Print a formatted banner."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# =============================================================================
# Experiment 1: Isolated SKQD (NF only vs NF + SKQD)
# =============================================================================

def run_isolated_skqd_experiment(molecule: str = "lih") -> ExperimentResult:
    """
    Test SKQD in isolation: NF only vs NF + SKQD.

    This shows what Krylov time evolution contributes on top of NF sampling.
    """
    print_banner(f"EXPERIMENT: Isolated SKQD on {molecule.upper()}")

    H = create_hamiltonian(molecule)
    E_exact = H.fci_energy()

    n_valid = comb(H.n_orbitals, H.n_alpha) * comb(H.n_orbitals, H.n_beta)
    print(f"System: {H.num_sites} qubits, {n_valid:,} valid configs")
    print(f"FCI energy: {E_exact:.8f} Ha")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    start_time = time.time()

    # Mode A: NF sampling only (no SKQD)
    print("\n--- Mode A: NF Sampling Only (baseline) ---")
    config_nf_only = PipelineConfig(
        subspace_mode="skqd",
        skip_skqd=True,
        max_epochs=400,
        device=device,
    )
    pipeline_nf = FlowGuidedKrylovPipeline(H, config=config_nf_only, exact_energy=E_exact)
    results_nf = pipeline_nf.run(progress=True)
    E_nf_only = results_nf.get('combined_energy', results_nf.get('nf_nqs_energy'))
    nf_basis_size = results_nf.get('nf_basis_size', 0)

    # Mode B: NF + SKQD
    print("\n--- Mode B: NF + SKQD ---")
    config_with_skqd = PipelineConfig(
        subspace_mode="skqd",
        skip_skqd=False,
        max_krylov_dim=12,
        shots_per_krylov=100000,
        max_epochs=400,
        device=device,
    )
    pipeline_skqd = FlowGuidedKrylovPipeline(H, config=config_with_skqd, exact_energy=E_exact)
    results_skqd = pipeline_skqd.run(progress=True)
    E_with_skqd = results_skqd.get('combined_energy', results_skqd.get('skqd_energy'))

    elapsed = time.time() - start_time

    error_nf = abs(E_nf_only - E_exact) * 1000
    error_skqd = abs(E_with_skqd - E_exact) * 1000
    improvement = error_nf - error_skqd

    print("\n" + "=" * 60)
    print("ISOLATED SKQD RESULTS:")
    print("=" * 60)
    print(f"{'Method':<25} {'Energy (Ha)':<16} {'Error (mHa)':<14}")
    print("-" * 55)
    print(f"{'Exact (FCI)':<25} {E_exact:<16.8f} {0:<14.4f}")
    print(f"{'NF Only':<25} {E_nf_only:<16.8f} {error_nf:<14.4f}")
    print(f"{'NF + SKQD':<25} {E_with_skqd:<16.8f} {error_skqd:<14.4f}")
    print("-" * 55)
    print(f"SKQD improvement: {improvement:.4f} mHa")
    print(f"SKQD provides {'SIGNIFICANT' if improvement > 0.1 else 'MINIMAL'} benefit")

    return ExperimentResult(
        name="isolated_skqd",
        molecule=molecule,
        exact_energy=E_exact,
        nf_only_energy=E_nf_only,
        combined_energy=E_with_skqd,
        nf_basis_size=nf_basis_size,
        time_seconds=elapsed,
        notes=f"SKQD improvement: {improvement:.4f} mHa"
    )


# =============================================================================
# Experiment 2: Configuration Provenance Tracking
# =============================================================================

def run_provenance_experiment(molecule: str = "lih") -> ExperimentResult:
    """
    Track which method discovers each configuration.

    This quantifies the unique contribution of each method:
    - Configurations found ONLY by NF
    - Configurations found ONLY by Krylov
    - Configurations found by both
    """
    print_banner(f"EXPERIMENT: Configuration Provenance on {molecule.upper()}")

    H = create_hamiltonian(molecule)
    E_exact = H.fci_energy()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    start_time = time.time()

    # Step 1: Run NF-NQS training to get NF basis
    print("\n--- Step 1: NF-NQS Training ---")
    config = PipelineConfig(
        skip_skqd=True,
        max_epochs=400,
        device=device,
    )
    pipeline = FlowGuidedKrylovPipeline(H, config=config, exact_energy=E_exact)
    pipeline.train_flow_nqs(progress=True)
    nf_basis = pipeline.extract_and_select_basis()

    nf_configs_set = configs_to_set(nf_basis)
    print(f"NF discovered {len(nf_configs_set)} unique configurations")

    # Step 2: Run pure SKQD from HF reference
    print("\n--- Step 2: Pure SKQD from HF State ---")
    skqd_config = SKQDConfig(
        max_krylov_dim=12,
        time_step=0.1,
        shots_per_krylov=100000,
    )
    skqd = SampleBasedKrylovDiagonalization(H, skqd_config)
    skqd.generate_krylov_samples(max_krylov_dim=12, progress=True)

    krylov_configs_set = set()
    cumulative = skqd.build_cumulative_basis()
    for bitstring in cumulative[-1].keys():
        config = tuple(int(b) for b in bitstring)
        krylov_configs_set.add(config)

    print(f"Krylov discovered {len(krylov_configs_set)} unique configurations")

    # Analyze overlap
    nf_only = nf_configs_set - krylov_configs_set
    krylov_only = krylov_configs_set - nf_configs_set
    both = nf_configs_set & krylov_configs_set
    combined = nf_configs_set | krylov_configs_set

    print("\n--- Configuration Provenance Analysis ---")
    print(f"Found ONLY by NF:     {len(nf_only):,}")
    print(f"Found ONLY by Krylov: {len(krylov_only):,}")
    print(f"Found by BOTH:        {len(both):,}")
    print(f"Combined unique:      {len(combined):,}")

    # Step 3: Compute energies for each basis
    print("\n--- Step 3: Computing Energies ---")

    E_nf_only = compute_basis_energy(H, nf_basis)

    krylov_basis = set_to_configs(krylov_configs_set, H.num_sites, device)
    E_krylov_only = compute_basis_energy(H, krylov_basis)

    combined_basis = set_to_configs(combined, H.num_sites, device)
    E_combined = compute_basis_energy(H, combined_basis)

    elapsed = time.time() - start_time

    error_nf = abs(E_nf_only - E_exact) * 1000
    error_krylov = abs(E_krylov_only - E_exact) * 1000
    error_combined = abs(E_combined - E_exact) * 1000

    print("\n" + "=" * 70)
    print("PROVENANCE EXPERIMENT RESULTS:")
    print("=" * 70)
    print(f"{'Basis':<25} {'Size':<10} {'Energy (Ha)':<16} {'Error (mHa)':<14}")
    print("-" * 65)
    print(f"{'Exact (FCI)':<25} {'-':<10} {E_exact:<16.8f} {0:<14.4f}")
    print(f"{'NF Only':<25} {len(nf_configs_set):<10} {E_nf_only:<16.8f} {error_nf:<14.4f}")
    print(f"{'Krylov Only':<25} {len(krylov_configs_set):<10} {E_krylov_only:<16.8f} {error_krylov:<14.4f}")
    print(f"{'Combined (NF+Krylov)':<25} {len(combined):<10} {E_combined:<16.8f} {error_combined:<14.4f}")
    print("-" * 65)
    print(f"\nKrylov-unique configs: {len(krylov_only)} ({100*len(krylov_only)/max(1,len(combined)):.1f}% of combined)")
    print(f"Combined improvement over NF: {error_nf - error_combined:.4f} mHa")
    print(f"Combined improvement over Krylov: {error_krylov - error_combined:.4f} mHa")

    return ExperimentResult(
        name="provenance",
        molecule=molecule,
        exact_energy=E_exact,
        nf_only_energy=E_nf_only,
        skqd_only_energy=E_krylov_only,
        combined_energy=E_combined,
        nf_basis_size=len(nf_configs_set),
        skqd_unique_configs=len(krylov_only),
        total_basis_size=len(combined),
        time_seconds=elapsed,
        notes=f"Krylov-unique: {len(krylov_only)}, Combined improvement: {error_nf - error_combined:.4f} mHa"
    )


# =============================================================================
# Experiment 3: Stretched Bond Geometries (Strong Correlation)
# =============================================================================

def create_stretched_h2o(stretch_factor: float = 2.0) -> MolecularHamiltonian:
    """Create H2O with stretched OH bonds (strongly correlated)."""
    oh_eq = 0.96
    oh_stretched = oh_eq * stretch_factor
    angle = 104.5

    angle_rad = np.radians(angle)
    geometry = [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (oh_stretched, 0.0, 0.0)),
        ("H", (oh_stretched * np.cos(angle_rad), oh_stretched * np.sin(angle_rad), 0.0)),
    ]
    integrals = compute_molecular_integrals(geometry, basis="sto-3g")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return MolecularHamiltonian(integrals, device=device)


def create_stretched_n2(stretch_factor: float = 2.0) -> MolecularHamiltonian:
    """Create N2 with stretched NN bond (strongly correlated triple bond breaking)."""
    nn_eq = 1.10
    nn_stretched = nn_eq * stretch_factor

    geometry = [
        ("N", (0.0, 0.0, 0.0)),
        ("N", (0.0, 0.0, nn_stretched)),
    ]
    integrals = compute_molecular_integrals(geometry, basis="sto-3g")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return MolecularHamiltonian(integrals, device=device)


def run_stretched_bond_experiment(molecule: str = "h2o", stretch_factor: float = 2.0) -> ExperimentResult:
    """
    Test on stretched geometries where ground state becomes strongly correlated.

    At stretched geometries:
    - Single-reference methods (HF, MP2) fail badly
    - Ground state has multi-reference character
    - NF might struggle to capture all important configurations
    - Krylov time evolution may discover correlations NF misses
    """
    print_banner(f"EXPERIMENT: Stretched {molecule.upper()} (factor={stretch_factor}x)")

    if molecule == "h2o":
        H = create_stretched_h2o(stretch_factor)
    elif molecule == "n2":
        H = create_stretched_n2(stretch_factor)
    else:
        raise ValueError(f"Stretched geometry not implemented for {molecule}")

    E_exact = H.fci_energy()
    n_valid = comb(H.n_orbitals, H.n_alpha) * comb(H.n_orbitals, H.n_beta)

    print(f"System: {H.num_sites} qubits, {n_valid:,} valid configs")
    print(f"Stretch factor: {stretch_factor}x equilibrium")
    print(f"FCI energy: {E_exact:.8f} Ha")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    start_time = time.time()

    # Mode A: NF only (no SKQD)
    print("\n--- Mode A: NF Only (baseline) ---")
    config_nf = PipelineConfig(
        subspace_mode="skqd",
        skip_skqd=True,
        max_epochs=600,
        device=device,
    )
    pipeline_nf = FlowGuidedKrylovPipeline(H, config=config_nf, exact_energy=E_exact)
    results_nf = pipeline_nf.run(progress=True)
    E_nf = results_nf.get('combined_energy', results_nf.get('nf_nqs_energy'))

    # Mode B: NF + SKQD
    print("\n--- Mode B: NF + SKQD ---")
    config_skqd = PipelineConfig(
        subspace_mode="skqd",
        skip_skqd=False,
        max_krylov_dim=15,
        shots_per_krylov=150000,
        max_epochs=600,
        device=device,
    )
    pipeline_skqd = FlowGuidedKrylovPipeline(H, config=config_skqd, exact_energy=E_exact)
    results_skqd = pipeline_skqd.run(progress=True)
    E_skqd = results_skqd.get('combined_energy', results_skqd.get('skqd_energy'))

    elapsed = time.time() - start_time

    error_nf = abs(E_nf - E_exact) * 1000
    error_skqd = abs(E_skqd - E_exact) * 1000
    improvement = error_nf - error_skqd

    print("\n" + "=" * 70)
    print(f"STRETCHED {molecule.upper()} RESULTS (stretch={stretch_factor}x):")
    print("=" * 70)
    print(f"{'Method':<30} {'Energy (Ha)':<16} {'Error (mHa)':<14}")
    print("-" * 60)
    print(f"{'Exact (FCI)':<30} {E_exact:<16.8f} {0:<14.4f}")
    print(f"{'NF Only':<30} {E_nf:<16.8f} {error_nf:<14.4f}")
    print(f"{'NF + SKQD':<30} {E_skqd:<16.8f} {error_skqd:<14.4f}")
    print("-" * 60)
    print(f"SKQD improvement: {improvement:.4f} mHa")

    return ExperimentResult(
        name=f"stretched_{molecule}_{stretch_factor}x",
        molecule=molecule,
        exact_energy=E_exact,
        nf_only_energy=E_nf,
        combined_energy=E_skqd,
        time_seconds=elapsed,
        notes=f"Stretch={stretch_factor}x, SKQD improvement={improvement:.4f} mHa"
    )


# =============================================================================
# Experiment 4: Poor Initial State
# =============================================================================

def run_poor_initial_state_experiment(molecule: str = "lih") -> ExperimentResult:
    """
    Test with degraded initial states where NF might struggle.

    This simulates scenarios where:
    - The NF training doesn't converge well
    - The initial guess is far from the ground state
    - Krylov time evolution should help explore more of the Hilbert space
    """
    print_banner(f"EXPERIMENT: Poor Initial State on {molecule.upper()}")

    H = create_hamiltonian(molecule)
    E_exact = H.fci_energy()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    start_time = time.time()

    # Mode A: Limited NF training (simulates poor convergence)
    print("\n--- Mode A: Limited NF Training (poor initial) ---")
    config_limited = PipelineConfig(
        max_epochs=50,
        min_epochs=50,
        skip_skqd=True,
        device=device,
    )
    pipeline_limited = FlowGuidedKrylovPipeline(H, config=config_limited, exact_energy=E_exact)
    results_limited = pipeline_limited.run(progress=True)
    E_limited = results_limited.get('combined_energy', results_limited.get('nf_nqs_energy'))

    # Mode B: Limited NF + SKQD
    print("\n--- Mode B: Limited NF + SKQD ---")
    config_limited_skqd = PipelineConfig(
        max_epochs=50,
        min_epochs=50,
        skip_skqd=False,
        max_krylov_dim=15,
        shots_per_krylov=150000,
        device=device,
    )
    pipeline_limited_skqd = FlowGuidedKrylovPipeline(H, config=config_limited_skqd, exact_energy=E_exact)
    results_limited_skqd = pipeline_limited_skqd.run(progress=True)
    E_limited_skqd = results_limited_skqd.get('combined_energy', results_limited_skqd.get('skqd_energy'))

    # Mode C: Full training for comparison
    print("\n--- Mode C: Full NF Training (reference) ---")
    config_full = PipelineConfig(
        max_epochs=400,
        skip_skqd=True,
        device=device,
    )
    pipeline_full = FlowGuidedKrylovPipeline(H, config=config_full, exact_energy=E_exact)
    results_full = pipeline_full.run(progress=True)
    E_full = results_full.get('combined_energy', results_full.get('nf_nqs_energy'))

    elapsed = time.time() - start_time

    error_limited = abs(E_limited - E_exact) * 1000
    error_limited_skqd = abs(E_limited_skqd - E_exact) * 1000
    error_full = abs(E_full - E_exact) * 1000

    skqd_recovery = error_limited - error_limited_skqd

    print("\n" + "=" * 70)
    print("POOR INITIAL STATE RESULTS:")
    print("=" * 70)
    print(f"{'Method':<35} {'Energy (Ha)':<16} {'Error (mHa)':<14}")
    print("-" * 65)
    print(f"{'Exact (FCI)':<35} {E_exact:<16.8f} {0:<14.4f}")
    print(f"{'Limited NF (50 epochs)':<35} {E_limited:<16.8f} {error_limited:<14.4f}")
    print(f"{'Limited NF + SKQD':<35} {E_limited_skqd:<16.8f} {error_limited_skqd:<14.4f}")
    print(f"{'Full NF (400 epochs, reference)':<35} {E_full:<16.8f} {error_full:<14.4f}")
    print("-" * 65)
    print(f"SKQD recovery from poor initial: {skqd_recovery:.4f} mHa")
    print(f"SKQD {'SIGNIFICANTLY' if skqd_recovery > 1.0 else 'PARTIALLY'} compensates for poor NF training")

    return ExperimentResult(
        name="poor_initial",
        molecule=molecule,
        exact_energy=E_exact,
        nf_only_energy=E_limited,
        combined_energy=E_limited_skqd,
        time_seconds=elapsed,
        notes=f"SKQD recovery: {skqd_recovery:.4f} mHa"
    )


# =============================================================================
# Experiment 5: Larger Basis Set (6-31G)
# =============================================================================

def create_h2_631g(bond_length: float = 0.74) -> MolecularHamiltonian:
    """Create H2 with 6-31G basis (larger than STO-3G)."""
    geometry = [
        ("H", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.0, bond_length)),
    ]
    integrals = compute_molecular_integrals(geometry, basis="6-31g")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return MolecularHamiltonian(integrals, device=device)


def create_lih_631g(bond_length: float = 1.6) -> MolecularHamiltonian:
    """Create LiH with 6-31G basis."""
    geometry = [
        ("Li", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.0, bond_length)),
    ]
    integrals = compute_molecular_integrals(geometry, basis="6-31g")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return MolecularHamiltonian(integrals, device=device)


def run_larger_basis_experiment(molecule: str = "h2") -> ExperimentResult:
    """
    Test with larger basis sets where configuration space is bigger.

    With 6-31G basis:
    - H2: 4 orbitals -> C(4,1)^2 = 16 configs
    - LiH: 11 orbitals -> C(11,2) * C(11,2) = 3025 configs
    """
    print_banner(f"EXPERIMENT: Larger Basis (6-31G) for {molecule.upper()}")

    if molecule == "h2":
        H = create_h2_631g()
    elif molecule == "lih":
        H = create_lih_631g()
    else:
        raise ValueError(f"6-31G not implemented for {molecule}")

    E_exact = H.fci_energy()
    n_valid = comb(H.n_orbitals, H.n_alpha) * comb(H.n_orbitals, H.n_beta)

    print(f"Basis: 6-31G")
    print(f"Orbitals: {H.n_orbitals}")
    print(f"Qubits: {H.num_sites}")
    print(f"Valid configs: {n_valid:,}")
    print(f"FCI energy: {E_exact:.8f} Ha")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    start_time = time.time()

    # Mode A: NF only
    print("\n--- Mode A: NF Only ---")
    config_nf = PipelineConfig(
        skip_skqd=True,
        max_epochs=400,
        device=device,
    )
    pipeline_nf = FlowGuidedKrylovPipeline(H, config=config_nf, exact_energy=E_exact)
    results_nf = pipeline_nf.run(progress=True)
    E_nf = results_nf.get('combined_energy', results_nf.get('nf_nqs_energy'))

    # Mode B: NF + SKQD
    print("\n--- Mode B: NF + SKQD ---")
    config_skqd = PipelineConfig(
        subspace_mode="skqd",
        skip_skqd=False,
        max_krylov_dim=12,
        shots_per_krylov=100000,
        max_epochs=400,
        device=device,
    )
    pipeline_skqd = FlowGuidedKrylovPipeline(H, config=config_skqd, exact_energy=E_exact)
    results_skqd = pipeline_skqd.run(progress=True)
    E_skqd = results_skqd.get('combined_energy', results_skqd.get('skqd_energy'))

    elapsed = time.time() - start_time

    error_nf = abs(E_nf - E_exact) * 1000
    error_skqd = abs(E_skqd - E_exact) * 1000
    improvement = error_nf - error_skqd

    print("\n" + "=" * 70)
    print(f"LARGER BASIS (6-31G) RESULTS for {molecule.upper()}:")
    print("=" * 70)
    print(f"{'Method':<25} {'Energy (Ha)':<16} {'Error (mHa)':<14}")
    print("-" * 55)
    print(f"{'Exact (FCI)':<25} {E_exact:<16.8f} {0:<14.4f}")
    print(f"{'NF Only':<25} {E_nf:<16.8f} {error_nf:<14.4f}")
    print(f"{'NF + SKQD':<25} {E_skqd:<16.8f} {error_skqd:<14.4f}")
    print("-" * 55)
    print(f"SKQD improvement: {improvement:.4f} mHa")

    best = min(error_nf, error_skqd)
    best_method = "SKQD" if error_skqd < error_nf else "NF"
    print(f"Best method: {best_method}")

    return ExperimentResult(
        name=f"larger_basis_{molecule}",
        molecule=molecule,
        exact_energy=E_exact,
        nf_only_energy=E_nf,
        skqd_only_energy=E_skqd,
        time_seconds=elapsed,
        notes=f"6-31G basis, {n_valid} configs, Best={best_method}"
    )


# =============================================================================
# Helper Functions
# =============================================================================

def create_hamiltonian(molecule: str) -> MolecularHamiltonian:
    """Create a molecular Hamiltonian by name."""
    creators = {
        'h2': create_h2_hamiltonian,
        'lih': create_lih_hamiltonian,
        'h2o': create_h2o_hamiltonian,
        'n2': create_n2_hamiltonian,
    }
    if molecule not in creators:
        raise ValueError(f"Unknown molecule: {molecule}. Available: {list(creators.keys())}")
    return creators[molecule]()


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
    H_np = 0.5 * (H_np + H_np.T)  # Ensure Hermitian
    eigenvalues, _ = np.linalg.eigh(H_np)
    return float(eigenvalues[0])


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SKQD Validation Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Experiment Modes:
    isolated    - Test SKQD alone (NF only vs NF + SKQD)
    provenance  - Track which method finds each configuration
    stretched   - Test on stretched geometries (strong correlation)
    poor_init   - Test with limited NF training
    larger_basis- Test with 6-31G basis set
    all         - Run all experiments
        """
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        default="all",
        choices=["isolated", "provenance", "stretched", "poor_init",
                 "larger_basis", "all"],
        help="Experiment mode"
    )
    parser.add_argument(
        "--molecule",
        type=str,
        default="lih",
        help="Molecule for experiment (h2, lih, h2o, n2)"
    )
    parser.add_argument(
        "--stretch",
        type=float,
        default=2.0,
        help="Stretch factor for stretched geometry experiment"
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("SKQD VALIDATION EXPERIMENTS")
    print("Goal: Demonstrate Krylov's unique contribution to the pipeline")
    print("=" * 70)

    results = []

    if args.mode in ["isolated", "all"]:
        results.append(run_isolated_skqd_experiment(args.molecule))

    if args.mode in ["provenance", "all"]:
        results.append(run_provenance_experiment(args.molecule))

    if args.mode in ["stretched", "all"]:
        results.append(run_stretched_bond_experiment(
            "h2o" if args.mode == "all" else args.molecule,
            args.stretch
        ))

    if args.mode in ["poor_init", "all"]:
        results.append(run_poor_initial_state_experiment(args.molecule))

    if args.mode in ["larger_basis", "all"]:
        results.append(run_larger_basis_experiment("lih" if args.mode == "all" else args.molecule))

    # Final summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    for r in results:
        print(f"\n{r.name.upper()}: {r.notes}")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("""
These experiments demonstrate:
1. ISOLATED_SKQD: How much Krylov helps on top of NF sampling
2. PROVENANCE: What percentage of configs are uniquely Krylov-discovered
3. STRETCHED: Whether Krylov helps more for strongly correlated systems
4. POOR_INIT: Whether Krylov compensates for poor NF training
5. LARGER_BASIS: Scalability to larger configuration spaces

Look for:
- Krylov-unique configs > 0 (Krylov finds things NF misses)
- Energy improvement from Krylov on top of NF
- Better performance on stretched/correlated systems
    """)


if __name__ == "__main__":
    main()
