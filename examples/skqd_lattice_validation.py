"""
SKQD Validation on Lattice Models (Original Use Case)

The SKQD paper demonstrates the method on:
- Transverse Field Ising Model (TFIM)
- Heisenberg Model
- Anderson Impurity Model

These lattice models are the IDEAL use case for SKQD because:
1. Ground states are sparse in computational basis (TFIM with small h)
2. Time evolution can be efficiently Trotterized
3. Krylov subspace naturally captures correlation growth

This benchmark tests whether SKQD provides value beyond what NF sampling achieves
on the systems where SKQD was originally designed to excel.

Usage:
    docker-compose run --rm flow-krylov-gpu python examples/skqd_lattice_validation.py --system all
    docker-compose run --rm flow-krylov-gpu python examples/skqd_lattice_validation.py --system tfim --spins 14
"""

import sys
from pathlib import Path
import argparse
import time
from typing import Dict, Any, Set, Tuple, Optional
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np

from hamiltonians.spin import HeisenbergHamiltonian, TransverseFieldIsing
from hamiltonians.base import Hamiltonian
from krylov.skqd import SampleBasedKrylovDiagonalization, SKQDConfig
from pipeline import FlowGuidedKrylovPipeline, PipelineConfig


@dataclass
class LatticeResult:
    """Container for lattice experiment results."""
    name: str
    system: str
    n_spins: int
    exact_energy: float
    nf_energy: Optional[float] = None
    skqd_energy: Optional[float] = None
    nf_basis_size: int = 0
    skqd_basis_size: int = 0
    krylov_unique_configs: int = 0
    time_seconds: float = 0.0
    notes: str = ""


def print_banner(title: str):
    """Print a formatted banner."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def exact_ground_state_energy(H: Hamiltonian) -> float:
    """Compute exact ground state energy via full diagonalization."""
    n_sites = H.num_sites
    if n_sites > 18:
        print(f"WARNING: System size {n_sites} may be too large for exact diag")

    # Generate all basis states
    n_states = 2 ** n_sites
    device = "cpu"  # Always use CPU for exact diag

    print(f"Computing exact energy ({n_states:,} basis states)...")
    start = time.time()

    # Build full Hamiltonian matrix
    H_full = np.zeros((n_states, n_states))

    for i in range(n_states):
        # Convert index to configuration
        config = torch.tensor(
            [(i >> bit) & 1 for bit in range(n_sites - 1, -1, -1)],
            dtype=torch.long, device=device
        )

        # Diagonal element
        H_full[i, i] = H.diagonal_element(config).item()

        # Off-diagonal elements
        connected, elements = H.get_connections(config)
        if len(connected) > 0:
            for conn, elem in zip(connected, elements):
                # Convert connected config to index
                j = sum(conn[bit].item() << (n_sites - 1 - bit) for bit in range(n_sites))
                H_full[j, i] = elem.item()

    # Diagonalize
    eigenvalues = np.linalg.eigvalsh(H_full)
    E0 = float(eigenvalues[0])
    elapsed = time.time() - start
    print(f"Exact E0 = {E0:.8f} (computed in {elapsed:.1f}s)")

    return E0


def configs_to_set(configs: torch.Tensor) -> Set[tuple]:
    """Convert tensor of configurations to set of tuples."""
    return {tuple(c.cpu().tolist()) for c in configs}


def set_to_configs(config_set: Set[tuple], n_sites: int, device: str) -> torch.Tensor:
    """Convert set of tuples back to tensor."""
    configs = [list(c) for c in config_set]
    return torch.tensor(configs, dtype=torch.long, device=device)


def compute_basis_energy(H: Hamiltonian, basis: torch.Tensor) -> float:
    """Compute ground state energy by diagonalizing H in given basis."""
    n_configs = len(basis)
    device = basis.device

    # Build Hamiltonian matrix in this basis
    H_matrix = np.zeros((n_configs, n_configs))

    # Build config -> index map
    config_to_idx = {}
    for i, config in enumerate(basis):
        config_to_idx[tuple(config.cpu().tolist())] = i

    for j in range(n_configs):
        config_j = basis[j]

        # Diagonal
        H_matrix[j, j] = H.diagonal_element(config_j).cpu().item()

        # Off-diagonal
        connected, elements = H.get_connections(config_j)
        if len(connected) > 0:
            for conn, elem in zip(connected, elements):
                key = tuple(conn.cpu().tolist())
                if key in config_to_idx:
                    i = config_to_idx[key]
                    H_matrix[i, j] = elem.item()

    # Ensure Hermitian
    H_matrix = 0.5 * (H_matrix + H_matrix.T)

    # Diagonalize
    eigenvalues = np.linalg.eigvalsh(H_matrix)
    return float(eigenvalues[0])


# =============================================================================
# Experiment 1: Transverse Field Ising Model (TFIM)
# =============================================================================

def run_tfim_experiment(
    n_spins: int = 12,
    h_field: float = 0.5,  # Small transverse field -> sparse ground state
) -> LatticeResult:
    """
    Test SKQD on Transverse Field Ising Model.

    The TFIM with small transverse field h is the canonical example from the
    SKQD paper because:
    - Ground state is sparse (dominated by |0...0> and |1...1> for h=0)
    - Krylov time evolution efficiently explores nearby configurations
    - This is where SKQD should show clear advantage

    H = -V Σ σ_i^z σ_j^z - h Σ σ_i^x
    """
    print_banner(f"EXPERIMENT: TFIM ({n_spins} spins, h={h_field})")

    H = TransverseFieldIsing(n_spins, V=1.0, h=h_field, periodic=True)

    E_exact = exact_ground_state_energy(H)
    print(f"TFIM {n_spins} spins, h={h_field}")
    print(f"Hilbert space: {2**n_spins:,}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    start_time = time.time()

    # Mode A: Pure SKQD from Néel-like initial state
    print("\n--- Mode A: Pure SKQD from |0...0> ---")
    initial_state = torch.zeros(n_spins, dtype=torch.long, device=device)

    skqd_config = SKQDConfig(
        max_krylov_dim=15,
        time_step=0.1,
        shots_per_krylov=100000,
    )
    skqd = SampleBasedKrylovDiagonalization(H, skqd_config)

    # Set initial state (all zeros - close to ground state for small h)
    skqd.initial_state = initial_state  # 1D tensor expected

    # Generate Krylov samples
    skqd.generate_krylov_samples(max_krylov_dim=15, progress=True)
    cumulative = skqd.build_cumulative_basis()

    # Analyze Krylov basis growth
    print("\nKrylov basis growth:")
    krylov_energies = []
    for k in range(1, len(cumulative)):
        configs = cumulative[k]
        basis_tensor = torch.tensor(
            [[int(b) for b in bs] for bs in configs.keys()],
            dtype=torch.long, device=device
        )
        E_k = compute_basis_energy(H, basis_tensor)
        krylov_energies.append(E_k)
        error_k = abs(E_k - E_exact) * 1000
        print(f"  k={k}: {len(configs)} configs, E={E_k:.6f}, error={error_k:.4f} mHa")

    E_skqd = krylov_energies[-1] if krylov_energies else float('nan')
    skqd_basis_size = len(cumulative[-1]) if cumulative else 0

    # Mode B: NF sampling (no Krylov)
    print("\n--- Mode B: NF Sampling Only ---")
    config_nf = PipelineConfig(
        use_particle_conserving_flow=False,  # No particle conservation for spin systems
        use_residual_expansion=False,
        skip_skqd=True,
        max_epochs=400,
        device=device,
    )
    pipeline_nf = FlowGuidedKrylovPipeline(H, config=config_nf, exact_energy=E_exact)
    results_nf = pipeline_nf.run(progress=True)
    E_nf = results_nf.get('combined_energy', results_nf.get('nf_nqs_energy'))
    nf_basis_size = results_nf.get('nf_basis_size', 0)

    # Mode C: NF + SKQD Combined
    print("\n--- Mode C: NF + SKQD Combined ---")
    config_combined = PipelineConfig(
        use_particle_conserving_flow=False,
        use_residual_expansion=False,
        skip_skqd=False,
        max_krylov_dim=12,
        shots_per_krylov=100000,
        max_epochs=400,
        device=device,
    )
    pipeline_combined = FlowGuidedKrylovPipeline(H, config=config_combined, exact_energy=E_exact)
    results_combined = pipeline_combined.run(progress=True)
    E_combined = results_combined.get('combined_energy')

    elapsed = time.time() - start_time

    # Results
    error_skqd = abs(E_skqd - E_exact) * 1000
    error_nf = abs(E_nf - E_exact) * 1000
    error_combined = abs(E_combined - E_exact) * 1000

    print("\n" + "=" * 70)
    print(f"TFIM RESULTS ({n_spins} spins, h={h_field}):")
    print("=" * 70)
    print(f"{'Method':<25} {'Basis Size':<12} {'Energy':<16} {'Error (mHa)':<14}")
    print("-" * 67)
    print(f"{'Exact':<25} {'-':<12} {E_exact:<16.8f} {0:<14.4f}")
    print(f"{'Pure SKQD':<25} {skqd_basis_size:<12} {E_skqd:<16.8f} {error_skqd:<14.4f}")
    print(f"{'NF Only':<25} {nf_basis_size:<12} {E_nf:<16.8f} {error_nf:<14.4f}")
    print(f"{'NF + SKQD':<25} {'-':<12} {E_combined:<16.8f} {error_combined:<14.4f}")
    print("-" * 67)

    best = min(error_skqd, error_nf, error_combined)
    best_method = "SKQD" if best == error_skqd else ("NF" if best == error_nf else "Combined")
    print(f"Best method: {best_method}")

    skqd_wins = error_skqd < error_nf
    print(f"\nSKQD {'OUTPERFORMS' if skqd_wins else 'underperforms'} NF on this system")

    return LatticeResult(
        name=f"tfim_{n_spins}_{h_field}",
        system="TFIM",
        n_spins=n_spins,
        exact_energy=E_exact,
        nf_energy=E_nf,
        skqd_energy=E_skqd,
        nf_basis_size=nf_basis_size,
        skqd_basis_size=skqd_basis_size,
        time_seconds=elapsed,
        notes=f"h={h_field}, Best={best_method}, SKQD_wins={skqd_wins}"
    )


# =============================================================================
# Experiment 2: Heisenberg Model
# =============================================================================

def run_heisenberg_experiment(
    n_spins: int = 10,
    Jz: float = 1.0,
    h_perturb: float = 0.1,  # Small perturbation to break degeneracy
) -> LatticeResult:
    """
    Test SKQD on Heisenberg XXZ Model.

    The Heisenberg model with small field perturbation:
    - Has antiferromagnetic ground state (Néel-like for 1D)
    - Ground state is sparse when h is small
    - Krylov explores spin-flip excitations systematically
    """
    print_banner(f"EXPERIMENT: Heisenberg ({n_spins} spins, h_perturb={h_perturb})")

    h_z = np.zeros(n_spins)
    h_z[0] = h_perturb  # Small perturbation on first site

    H = HeisenbergHamiltonian(
        n_spins,
        Jx=1.0, Jy=1.0, Jz=Jz,
        h_x=np.zeros(n_spins),
        h_y=np.zeros(n_spins),
        h_z=h_z,
        periodic=False
    )

    E_exact = exact_ground_state_energy(H)
    print(f"Heisenberg {n_spins} spins, Jz={Jz}, h_perturb={h_perturb}")
    print(f"Hilbert space: {2**n_spins:,}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    start_time = time.time()

    # Mode A: Pure SKQD from Néel state |0101...>
    print("\n--- Mode A: Pure SKQD from Néel State ---")
    neel_state = torch.tensor(
        [i % 2 for i in range(n_spins)],
        dtype=torch.long, device=device
    )

    skqd_config = SKQDConfig(
        max_krylov_dim=15,
        time_step=0.1,
        shots_per_krylov=100000,
    )
    skqd = SampleBasedKrylovDiagonalization(H, skqd_config)
    skqd.initial_state = neel_state  # 1D tensor expected

    skqd.generate_krylov_samples(max_krylov_dim=15, progress=True)
    cumulative = skqd.build_cumulative_basis()

    # Compute energy at final Krylov dimension
    if cumulative:
        final_configs = cumulative[-1]
        basis_tensor = torch.tensor(
            [[int(b) for b in bs] for bs in final_configs.keys()],
            dtype=torch.long, device=device
        )
        E_skqd = compute_basis_energy(H, basis_tensor)
        skqd_basis_size = len(final_configs)
    else:
        E_skqd = float('nan')
        skqd_basis_size = 0

    # Mode B: NF sampling only
    print("\n--- Mode B: NF Sampling Only ---")
    config_nf = PipelineConfig(
        use_particle_conserving_flow=False,
        use_residual_expansion=False,
        skip_skqd=True,
        max_epochs=400,
        device=device,
    )
    pipeline_nf = FlowGuidedKrylovPipeline(H, config=config_nf, exact_energy=E_exact)
    results_nf = pipeline_nf.run(progress=True)
    E_nf = results_nf.get('combined_energy', results_nf.get('nf_nqs_energy'))
    nf_basis_size = results_nf.get('nf_basis_size', 0)

    # Mode C: NF + SKQD
    print("\n--- Mode C: NF + SKQD ---")
    config_combined = PipelineConfig(
        use_particle_conserving_flow=False,
        use_residual_expansion=False,
        skip_skqd=False,
        max_krylov_dim=12,
        shots_per_krylov=100000,
        max_epochs=400,
        device=device,
    )
    pipeline_combined = FlowGuidedKrylovPipeline(H, config=config_combined, exact_energy=E_exact)
    results_combined = pipeline_combined.run(progress=True)
    E_combined = results_combined.get('combined_energy')

    elapsed = time.time() - start_time

    # Results
    error_skqd = abs(E_skqd - E_exact) * 1000
    error_nf = abs(E_nf - E_exact) * 1000
    error_combined = abs(E_combined - E_exact) * 1000

    print("\n" + "=" * 70)
    print(f"HEISENBERG RESULTS ({n_spins} spins):")
    print("=" * 70)
    print(f"{'Method':<25} {'Basis Size':<12} {'Energy':<16} {'Error (mHa)':<14}")
    print("-" * 67)
    print(f"{'Exact':<25} {'-':<12} {E_exact:<16.8f} {0:<14.4f}")
    print(f"{'Pure SKQD':<25} {skqd_basis_size:<12} {E_skqd:<16.8f} {error_skqd:<14.4f}")
    print(f"{'NF Only':<25} {nf_basis_size:<12} {E_nf:<16.8f} {error_nf:<14.4f}")
    print(f"{'NF + SKQD':<25} {'-':<12} {E_combined:<16.8f} {error_combined:<14.4f}")
    print("-" * 67)

    best = min(error_skqd, error_nf, error_combined)
    best_method = "SKQD" if best == error_skqd else ("NF" if best == error_nf else "Combined")
    print(f"Best method: {best_method}")

    return LatticeResult(
        name=f"heisenberg_{n_spins}",
        system="Heisenberg",
        n_spins=n_spins,
        exact_energy=E_exact,
        nf_energy=E_nf,
        skqd_energy=E_skqd,
        nf_basis_size=nf_basis_size,
        skqd_basis_size=skqd_basis_size,
        time_seconds=elapsed,
        notes=f"Jz={Jz}, Best={best_method}"
    )


# =============================================================================
# Experiment 3: Krylov Convergence Analysis
# =============================================================================

def run_krylov_convergence_experiment(
    n_spins: int = 10,
    h_values: list = None,
) -> Dict[str, Any]:
    """
    Analyze how Krylov convergence depends on ground state sparsity.

    The SKQD paper shows that convergence depends on:
    1. Overlap of initial state with ground state |γ₀|²
    2. Sparsity of ground state (α_L, β_L parameters)

    This experiment tests convergence across different transverse field strengths
    to validate the theoretical predictions.
    """
    if h_values is None:
        h_values = [0.1, 0.3, 0.5, 1.0, 2.0]  # Range from sparse to non-sparse

    print_banner(f"EXPERIMENT: Krylov Convergence Analysis ({n_spins} spins)")
    print(f"Testing h = {h_values}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}

    for h in h_values:
        print(f"\n--- h = {h} ---")

        H = TransverseFieldIsing(n_spins, V=1.0, h=h, periodic=True)
        E_exact = exact_ground_state_energy(H)

        # Run Krylov with increasing dimensions
        skqd_config = SKQDConfig(
            max_krylov_dim=15,
            time_step=0.1,
            shots_per_krylov=100000,
        )
        skqd = SampleBasedKrylovDiagonalization(H, skqd_config)

        # Initialize from |0...0>
        initial_state = torch.zeros(n_spins, dtype=torch.long, device=device)
        skqd.initial_state = initial_state  # 1D tensor expected

        skqd.generate_krylov_samples(max_krylov_dim=15, progress=False)
        cumulative = skqd.build_cumulative_basis()

        # Track convergence
        convergence = []
        for k in range(1, len(cumulative)):
            configs = cumulative[k]
            basis_tensor = torch.tensor(
                [[int(b) for b in bs] for bs in configs.keys()],
                dtype=torch.long, device=device
            )
            E_k = compute_basis_energy(H, basis_tensor)
            error_k = abs(E_k - E_exact) * 1000
            convergence.append({
                'k': k,
                'basis_size': len(configs),
                'energy': E_k,
                'error_mha': error_k
            })
            print(f"  k={k}: {len(configs):4d} configs, error={error_k:.4f} mHa")

        results[f"h={h}"] = {
            'h': h,
            'exact_energy': E_exact,
            'convergence': convergence,
            'final_error': convergence[-1]['error_mha'] if convergence else float('nan')
        }

    # Summary
    print("\n" + "=" * 70)
    print("KRYLOV CONVERGENCE SUMMARY:")
    print("=" * 70)
    print(f"{'h':<10} {'Final Error (mHa)':<20} {'Final Basis Size':<20}")
    print("-" * 50)
    for key, data in results.items():
        if data['convergence']:
            final = data['convergence'][-1]
            print(f"{data['h']:<10} {final['error_mha']:<20.4f} {final['basis_size']:<20}")

    print("\nExpected: Lower h → sparser ground state → faster convergence")

    return results


# =============================================================================
# Experiment 4: Configuration Discovery Comparison
# =============================================================================

def run_discovery_comparison(n_spins: int = 10, h_field: float = 0.5) -> LatticeResult:
    """
    Compare what configurations each method discovers.

    Track:
    - Configs found only by Krylov
    - Configs found only by NF
    - Overlap between methods
    """
    print_banner(f"EXPERIMENT: Discovery Comparison (TFIM {n_spins} spins)")

    H = TransverseFieldIsing(n_spins, V=1.0, h=h_field, periodic=True)
    E_exact = exact_ground_state_energy(H)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    start_time = time.time()

    # Step 1: Collect Krylov-discovered configs
    print("\n--- Krylov Configuration Discovery ---")
    skqd_config = SKQDConfig(max_krylov_dim=12, time_step=0.1, shots_per_krylov=100000)
    skqd = SampleBasedKrylovDiagonalization(H, skqd_config)
    skqd.initial_state = torch.zeros(n_spins, dtype=torch.long, device=device)  # 1D tensor
    skqd.generate_krylov_samples(max_krylov_dim=12, progress=True)

    krylov_set = set()
    cumulative = skqd.build_cumulative_basis()
    for bitstring in cumulative[-1].keys():
        config = tuple(int(b) for b in bitstring)
        krylov_set.add(config)
    print(f"Krylov discovered: {len(krylov_set)} configs")

    # Step 2: Collect NF-discovered configs
    print("\n--- NF Configuration Discovery ---")
    config_nf = PipelineConfig(
        use_particle_conserving_flow=False,
        use_residual_expansion=False,
        skip_skqd=True,
        max_epochs=400,
        device=device,
    )
    pipeline = FlowGuidedKrylovPipeline(H, config=config_nf, exact_energy=E_exact)
    pipeline.train_flow_nqs(progress=True)
    nf_basis = pipeline.extract_and_select_basis()

    nf_set = configs_to_set(nf_basis)
    print(f"NF discovered: {len(nf_set)} configs")

    # Step 3: Analyze overlap
    krylov_only = krylov_set - nf_set
    nf_only = nf_set - krylov_set
    both = krylov_set & nf_set
    combined = krylov_set | nf_set

    print("\n--- Discovery Analysis ---")
    print(f"Krylov-only configs:  {len(krylov_only)}")
    print(f"NF-only configs:      {len(nf_only)}")
    print(f"Both methods:         {len(both)}")
    print(f"Combined unique:      {len(combined)}")

    # Step 4: Compute energies
    nf_basis_tensor = set_to_configs(nf_set, n_spins, device)
    krylov_basis_tensor = set_to_configs(krylov_set, n_spins, device)
    combined_basis_tensor = set_to_configs(combined, n_spins, device)

    E_nf = compute_basis_energy(H, nf_basis_tensor)
    E_krylov = compute_basis_energy(H, krylov_basis_tensor)
    E_combined = compute_basis_energy(H, combined_basis_tensor)

    elapsed = time.time() - start_time

    # Results
    error_nf = abs(E_nf - E_exact) * 1000
    error_krylov = abs(E_krylov - E_exact) * 1000
    error_combined = abs(E_combined - E_exact) * 1000

    print("\n" + "=" * 70)
    print("DISCOVERY COMPARISON RESULTS:")
    print("=" * 70)
    print(f"{'Basis':<25} {'Size':<10} {'Energy':<16} {'Error (mHa)':<14}")
    print("-" * 65)
    print(f"{'Exact':<25} {'-':<10} {E_exact:<16.8f} {0:<14.4f}")
    print(f"{'NF Only':<25} {len(nf_set):<10} {E_nf:<16.8f} {error_nf:<14.4f}")
    print(f"{'Krylov Only':<25} {len(krylov_set):<10} {E_krylov:<16.8f} {error_krylov:<14.4f}")
    print(f"{'Combined':<25} {len(combined):<10} {E_combined:<16.8f} {error_combined:<14.4f}")
    print("-" * 65)

    improvement_from_krylov = error_nf - error_combined
    print(f"\nKrylov-unique configs improve energy by: {improvement_from_krylov:.4f} mHa")

    if len(krylov_only) > 0:
        print(f"\n>>> KRYLOV FOUND {len(krylov_only)} CONFIGS THAT NF MISSED <<<")

    return LatticeResult(
        name="discovery_comparison",
        system="TFIM",
        n_spins=n_spins,
        exact_energy=E_exact,
        nf_energy=E_nf,
        skqd_energy=E_krylov,
        nf_basis_size=len(nf_set),
        skqd_basis_size=len(krylov_set),
        krylov_unique_configs=len(krylov_only),
        time_seconds=elapsed,
        notes=f"Krylov-unique: {len(krylov_only)}, Combined improvement: {improvement_from_krylov:.4f} mHa"
    )


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SKQD Validation on Lattice Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Systems:
    tfim        - Transverse Field Ising Model
    heisenberg  - Heisenberg XXZ Model
    convergence - Krylov convergence analysis
    discovery   - Configuration discovery comparison
    all         - Run all experiments
        """
    )
    parser.add_argument(
        "--system", "-s",
        type=str,
        default="all",
        choices=["tfim", "heisenberg", "convergence", "discovery", "all"],
        help="System to test"
    )
    parser.add_argument(
        "--spins", "-n",
        type=int,
        default=10,
        help="Number of spins (default: 10)"
    )
    parser.add_argument(
        "--h-field",
        type=float,
        default=0.5,
        help="Transverse field strength for TFIM (default: 0.5)"
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("SKQD LATTICE VALIDATION EXPERIMENTS")
    print("Testing Krylov on its original use case: spin lattice models")
    print("=" * 70)

    results = []

    if args.system in ["tfim", "all"]:
        results.append(run_tfim_experiment(args.spins, args.h_field))

    if args.system in ["heisenberg", "all"]:
        results.append(run_heisenberg_experiment(args.spins))

    if args.system in ["convergence", "all"]:
        run_krylov_convergence_experiment(args.spins)

    if args.system in ["discovery", "all"]:
        results.append(run_discovery_comparison(args.spins, args.h_field))

    # Final summary
    print("\n" + "=" * 80)
    print("LATTICE EXPERIMENT SUMMARY")
    print("=" * 80)
    for r in results:
        print(f"\n{r.name}: {r.notes}")

    print("\n" + "=" * 80)
    print("THEORETICAL EXPECTATIONS")
    print("=" * 80)
    print("""
According to the SKQD paper (Yu et al.), SKQD should excel when:
1. Ground state is sparse (small h in TFIM)
2. Initial state has good overlap with ground state
3. Hamiltonian has efficient Trotter decomposition

Look for:
- Better SKQD performance at small transverse field h
- Krylov-unique configurations that NF misses
- Exponential convergence with Krylov dimension k

If SKQD does NOT outperform NF on these systems, the pipeline design
may need revision (NF is already discovering the important configs).
    """)


if __name__ == "__main__":
    main()
