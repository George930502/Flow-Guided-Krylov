"""
SKQD Necessity Test: When Is Krylov Actually Needed?

This experiment is designed to answer the core research question:
"Under what conditions does SKQD provide configurations that other methods miss?"

The test systematically varies:
1. System size (number of valid configurations)
2. NF training quality (epochs)
3. Residual expansion thoroughness

And measures:
- Unique configs discovered by each method
- Energy improvement from each method
- Whether SKQD is NECESSARY vs merely helpful

Usage:
    docker-compose run --rm flow-krylov-gpu python examples/skqd_necessity_test.py
"""

import sys
from pathlib import Path
import argparse
import time
from math import comb
from typing import Dict, Any, Set, Tuple, List
from dataclasses import dataclass, field
import json

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np

try:
    from hamiltonians.molecular import (
        MolecularHamiltonian,
        create_h2_hamiltonian,
        create_lih_hamiltonian,
        create_h2o_hamiltonian,
        create_beh2_hamiltonian,
        create_n2_hamiltonian,
        create_ch4_hamiltonian,
    )
    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False
    print("ERROR: PySCF required")
    sys.exit(1)

from pipeline import FlowGuidedKrylovPipeline, PipelineConfig
from krylov.skqd import SampleBasedKrylovDiagonalization, FlowGuidedSKQD, SKQDConfig
from krylov.residual_expansion import SelectedCIExpander, ResidualExpansionConfig


@dataclass
class NecessityResult:
    """Result of a necessity test."""
    molecule: str
    n_valid_configs: int
    n_qubits: int
    exact_energy: float

    # Configuration counts
    nf_configs: int = 0
    residual_new_configs: int = 0
    krylov_new_configs: int = 0
    krylov_unique_configs: int = 0  # Found by Krylov but NOT by NF+Residual

    # Energies
    nf_energy: float = 0.0
    nf_residual_energy: float = 0.0
    nf_krylov_energy: float = 0.0
    all_combined_energy: float = 0.0

    # Error improvements (mHa)
    residual_improvement: float = 0.0
    krylov_improvement: float = 0.0
    krylov_unique_contribution: float = 0.0  # Improvement from Krylov-unique configs

    # Necessity verdict
    skqd_necessary: bool = False  # True if Krylov found essential configs
    skqd_helpful: bool = False    # True if Krylov improved energy at all

    notes: str = ""


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


def run_necessity_test(molecule_key: str, verbose: bool = True) -> NecessityResult:
    """
    Run comprehensive necessity test for a single molecule.

    This test determines whether SKQD is:
    - NECESSARY: Finds configs that NF+Residual cannot find
    - HELPFUL: Improves energy but configs could be found by Residual
    - REDUNDANT: Doesn't add value beyond what other methods provide
    """
    print_banner(f"NECESSITY TEST: {molecule_key.upper()}")

    # Create Hamiltonian
    creators = {
        'h2': create_h2_hamiltonian,
        'lih': create_lih_hamiltonian,
        'h2o': create_h2o_hamiltonian,
        'beh2': create_beh2_hamiltonian,
        'n2': create_n2_hamiltonian,
        'ch4': create_ch4_hamiltonian,
    }
    H = creators[molecule_key]()

    n_valid = comb(H.n_orbitals, H.n_alpha) * comb(H.n_orbitals, H.n_beta)
    E_exact = H.fci_energy()

    print(f"Molecule: {molecule_key}")
    print(f"Qubits: {H.num_sites}")
    print(f"Valid configs: {n_valid:,}")
    print(f"FCI energy: {E_exact:.8f} Ha")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    result = NecessityResult(
        molecule=molecule_key,
        n_valid_configs=n_valid,
        n_qubits=H.num_sites,
        exact_energy=E_exact,
    )

    # =======================================================================
    # Step 1: NF-NQS Training and Basis Extraction
    # =======================================================================
    print("\n--- Step 1: NF-NQS Training ---")
    config_nf = PipelineConfig(
        use_residual_expansion=False,
        skip_skqd=True,
        max_epochs=400,
        device=device,
    )
    pipeline = FlowGuidedKrylovPipeline(H, config=config_nf, exact_energy=E_exact)
    pipeline.train_flow_nqs(progress=verbose)
    nf_basis = pipeline.extract_and_select_basis()

    nf_set = configs_to_set(nf_basis)
    result.nf_configs = len(nf_set)
    result.nf_energy = compute_basis_energy(H, nf_basis)

    print(f"NF discovered: {len(nf_set)} configs")
    print(f"NF energy: {result.nf_energy:.8f} Ha")

    # =======================================================================
    # Step 2: Residual Expansion (PT2-based)
    # =======================================================================
    print("\n--- Step 2: Residual (PT2) Expansion ---")
    residual_config = ResidualExpansionConfig(
        max_configs_per_iter=300,
        max_iterations=15,  # Thorough expansion
    )
    expander = SelectedCIExpander(H, residual_config)

    expanded_basis = nf_basis.clone()
    for i in range(15):
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

    print(f"Residual found: {len(residual_new)} NEW configs")
    print(f"NF+Residual energy: {result.nf_residual_energy:.8f} Ha")

    # =======================================================================
    # Step 3: Krylov Time Evolution Sampling
    # =======================================================================
    print("\n--- Step 3: Krylov Time Evolution ---")
    skqd_config = SKQDConfig(
        max_krylov_dim=12,
        time_step=0.1,
        shots_per_krylov=100000,
    )

    # Run Flow-Guided SKQD (starts from NF basis)
    skqd = FlowGuidedSKQD(H, nf_basis, skqd_config)
    skqd_results = skqd.run_with_nf(max_krylov_dim=12, progress=verbose)

    # Collect all Krylov-discovered configs
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

    print(f"Krylov found: {len(krylov_new)} NEW configs")
    print(f"NF+Krylov energy: {result.nf_krylov_energy:.8f} Ha")

    # =======================================================================
    # Step 4: Identify Krylov-UNIQUE configs
    # =======================================================================
    print("\n--- Step 4: Analyzing Config Provenance ---")

    # Configs found by Krylov but NOT by NF+Residual
    krylov_unique = krylov_set - residual_set
    result.krylov_unique_configs = len(krylov_unique)

    # Configs found by Residual but NOT by Krylov
    residual_unique = residual_set - krylov_set

    # Configs found by both (beyond NF)
    both_new = krylov_new & residual_new

    print(f"Krylov-UNIQUE configs: {len(krylov_unique)}")
    print(f"Residual-UNIQUE configs: {len(residual_unique)}")
    print(f"Found by BOTH: {len(both_new)}")

    # =======================================================================
    # Step 5: Compute Combined Energy (All Methods)
    # =======================================================================
    print("\n--- Step 5: Computing Combined Energy ---")

    all_configs = nf_set | residual_set | krylov_set
    all_basis = set_to_configs(all_configs, H.num_sites, device)
    result.all_combined_energy = compute_basis_energy(H, all_basis)

    print(f"Combined basis: {len(all_configs)} configs")
    print(f"Combined energy: {result.all_combined_energy:.8f} Ha")

    # =======================================================================
    # Step 6: Compute Improvements
    # =======================================================================
    nf_error = abs(result.nf_energy - E_exact) * 1000
    nf_residual_error = abs(result.nf_residual_energy - E_exact) * 1000
    nf_krylov_error = abs(result.nf_krylov_energy - E_exact) * 1000
    combined_error = abs(result.all_combined_energy - E_exact) * 1000

    result.residual_improvement = nf_error - nf_residual_error
    result.krylov_improvement = nf_error - nf_krylov_error

    # Contribution from Krylov-unique configs
    # (Combined vs NF+Residual tells us what Krylov added beyond Residual)
    result.krylov_unique_contribution = nf_residual_error - combined_error

    # =======================================================================
    # Step 7: Determine Necessity
    # =======================================================================
    print("\n--- Step 7: Necessity Verdict ---")

    # SKQD is NECESSARY if:
    # 1. It found configs that Residual missed (krylov_unique > 0)
    # 2. Those configs improve energy (krylov_unique_contribution > 0.01 mHa)
    result.skqd_necessary = (
        result.krylov_unique_configs > 0 and
        result.krylov_unique_contribution > 0.01
    )

    # SKQD is HELPFUL if:
    # It improves energy even if Residual could find the same configs
    result.skqd_helpful = result.krylov_improvement > 0.01

    if result.skqd_necessary:
        result.notes = f"NECESSARY: {result.krylov_unique_configs} unique configs, " \
                      f"{result.krylov_unique_contribution:.4f} mHa improvement"
    elif result.skqd_helpful:
        result.notes = f"HELPFUL but not necessary: " \
                      f"{result.krylov_improvement:.4f} mHa improvement (Residual could match)"
    else:
        result.notes = "REDUNDANT: Residual expansion sufficient"

    # =======================================================================
    # Results Summary
    # =======================================================================
    print("\n" + "=" * 70)
    print(f"NECESSITY TEST RESULTS: {molecule_key.upper()}")
    print("=" * 70)
    print(f"{'Metric':<35} {'Value':<20}")
    print("-" * 55)
    print(f"{'Valid configurations':<35} {n_valid:<20,}")
    print(f"{'NF configs':<35} {result.nf_configs:<20,}")
    print(f"{'Residual NEW configs':<35} {result.residual_new_configs:<20,}")
    print(f"{'Krylov NEW configs':<35} {result.krylov_new_configs:<20,}")
    print(f"{'Krylov UNIQUE configs':<35} {result.krylov_unique_configs:<20,}")
    print("-" * 55)
    print(f"{'NF error (mHa)':<35} {nf_error:<20.4f}")
    print(f"{'NF+Residual error (mHa)':<35} {nf_residual_error:<20.4f}")
    print(f"{'NF+Krylov error (mHa)':<35} {nf_krylov_error:<20.4f}")
    print(f"{'All combined error (mHa)':<35} {combined_error:<20.4f}")
    print("-" * 55)
    print(f"{'Residual improvement (mHa)':<35} {result.residual_improvement:<20.4f}")
    print(f"{'Krylov improvement (mHa)':<35} {result.krylov_improvement:<20.4f}")
    print(f"{'Krylov-unique contribution (mHa)':<35} {result.krylov_unique_contribution:<20.4f}")
    print("-" * 55)

    if result.skqd_necessary:
        print(f"\n>>> VERDICT: SKQD is NECESSARY <<<")
        print(f"    Found {result.krylov_unique_configs} configs that Residual missed")
        print(f"    These configs improved energy by {result.krylov_unique_contribution:.4f} mHa")
    elif result.skqd_helpful:
        print(f"\n>>> VERDICT: SKQD is HELPFUL but not strictly necessary <<<")
        print(f"    Krylov improved energy by {result.krylov_improvement:.4f} mHa")
        print(f"    But Residual could find the same configs with more iterations")
    else:
        print(f"\n>>> VERDICT: SKQD is REDUNDANT for this system <<<")
        print(f"    Residual expansion is sufficient")

    return result


def run_scaling_analysis(molecules: List[str] = None, verbose: bool = True) -> List[NecessityResult]:
    """
    Run necessity test across multiple molecules to understand scaling.

    This helps answer: "At what system size does SKQD become necessary?"
    """
    if molecules is None:
        molecules = ['h2', 'lih', 'beh2', 'h2o', 'n2', 'ch4']

    print_banner("SKQD NECESSITY SCALING ANALYSIS")

    results = []
    for mol in molecules:
        try:
            result = run_necessity_test(mol, verbose=verbose)
            results.append(result)
        except Exception as e:
            print(f"\nERROR testing {mol}: {e}")
            import traceback
            traceback.print_exc()

    # Summary table
    print("\n" + "=" * 100)
    print("SCALING ANALYSIS SUMMARY")
    print("=" * 100)
    print(f"{'Molecule':<10} {'Valid':<12} {'NF':<8} {'Res New':<10} {'Kry New':<10} "
          f"{'Kry Unique':<12} {'Verdict':<15}")
    print("-" * 100)

    necessary_count = 0
    helpful_count = 0

    for r in results:
        verdict = "NECESSARY" if r.skqd_necessary else ("HELPFUL" if r.skqd_helpful else "REDUNDANT")
        print(f"{r.molecule:<10} {r.n_valid_configs:<12,} {r.nf_configs:<8,} "
              f"{r.residual_new_configs:<10,} {r.krylov_new_configs:<10,} "
              f"{r.krylov_unique_configs:<12,} {verdict:<15}")

        if r.skqd_necessary:
            necessary_count += 1
        elif r.skqd_helpful:
            helpful_count += 1

    print("-" * 100)
    print(f"NECESSARY: {necessary_count}/{len(results)} systems")
    print(f"HELPFUL: {helpful_count}/{len(results)} systems")
    print(f"REDUNDANT: {len(results) - necessary_count - helpful_count}/{len(results)} systems")

    # Conclusion
    print("\n" + "=" * 80)
    print("CONCLUSIONS")
    print("=" * 80)

    if necessary_count > 0:
        necessary_mols = [r.molecule for r in results if r.skqd_necessary]
        print(f"\nSKQD is NECESSARY for: {', '.join(necessary_mols)}")
        print("These systems have configurations that Krylov time evolution discovers")
        print("but PT2 residual expansion does not.")
    else:
        print("\nSKQD was not strictly NECESSARY for any tested system.")
        print("This suggests PT2 residual expansion is thorough enough for these molecules.")

    if any(r.krylov_unique_configs > 0 for r in results):
        print("\n>>> IMPORTANT: Krylov DID find unique configurations in some systems <<<")
        print("This validates the research hypothesis that Krylov explores differently.")
    else:
        print("\n>>> WARNING: Krylov found NO unique configurations <<<")
        print("Either NF+Residual is exhaustive, or SKQD needs better initial states.")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="SKQD Necessity Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This test determines when SKQD is truly necessary vs. merely helpful.

Verdicts:
    NECESSARY: SKQD finds configs that NF+Residual cannot find
    HELPFUL:   SKQD improves energy but same configs reachable by Residual
    REDUNDANT: SKQD adds no value beyond what other methods provide
        """
    )
    parser.add_argument(
        "--molecule", "-m",
        type=str,
        default="all",
        help="Molecule to test (h2, lih, beh2, h2o, n2, ch4, or 'all')"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce output verbosity"
    )

    args = parser.parse_args()

    if args.molecule == "all":
        run_scaling_analysis(verbose=not args.quiet)
    else:
        run_necessity_test(args.molecule, verbose=not args.quiet)


if __name__ == "__main__":
    main()
