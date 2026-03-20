"""
Large System Benchmark: Testing SKQD Scaling on Challenging Molecules

This benchmark tests the SKQD necessity hypothesis on larger molecular systems
with active space selection, including:

1. Cr₂ (chromium dimer) - Notorious multi-reference system
2. Benzene - 6π electrons in 6 orbitals
3. Fe(II)-porphyrin model - Transition metal with d-orbital correlations
4. N₂ with larger basis (cc-pVDZ)
5. Butadiene - Extended π system

The goal is to confirm:
- SKQD becomes necessary as system size increases
- Krylov finds unique configurations for >20,000 valid configs

Usage:
    docker-compose run --rm flow-krylov-gpu python examples/large_system_benchmark.py --system all
    docker-compose run --rm flow-krylov-gpu python examples/large_system_benchmark.py --system cr2
    docker-compose run --rm flow-krylov-gpu python examples/large_system_benchmark.py --system fe_porphyrin
"""

import sys
from pathlib import Path
import argparse
import time
from math import comb
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np

try:
    from pyscf import gto, scf, mcscf, ao2mo, fci
    from pyscf.tools import molden
    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False
    print("ERROR: PySCF required. Install with: pip install pyscf")
    sys.exit(1)

from hamiltonians.molecular import MolecularHamiltonian, MolecularIntegrals
from pipeline import FlowGuidedKrylovPipeline, PipelineConfig
from krylov.skqd import FlowGuidedSKQD, SKQDConfig
from krylov.residual_expansion import SelectedCIExpander, ResidualExpansionConfig


@dataclass
class LargeSystemResult:
    """Container for large system benchmark results."""
    name: str
    n_orbitals: int
    n_electrons: int
    n_alpha: int
    n_beta: int
    n_qubits: int
    n_valid_configs: int

    # Energies
    hf_energy: float = 0.0
    casci_energy: Optional[float] = None  # Reference if computable
    nf_energy: float = 0.0
    nf_residual_energy: float = 0.0
    nf_krylov_energy: float = 0.0
    combined_energy: float = 0.0

    # Config discovery
    nf_configs: int = 0
    residual_new_configs: int = 0
    krylov_new_configs: int = 0
    krylov_unique_configs: int = 0

    # Timing
    time_nf: float = 0.0
    time_residual: float = 0.0
    time_krylov: float = 0.0

    # Verdict
    skqd_necessary: bool = False
    notes: str = ""


def print_banner(title: str):
    """Print a formatted banner."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def compute_active_space_integrals(
    mol,
    mf,
    n_active_orbitals: int,
    n_active_electrons: int,
    active_orbitals: Optional[List[int]] = None,
) -> Tuple[MolecularIntegrals, float]:
    """
    Compute molecular integrals in an active space.

    Args:
        mol: PySCF molecule object
        mf: Converged HF object
        n_active_orbitals: Number of orbitals in active space
        n_active_electrons: Number of electrons in active space
        active_orbitals: Specific orbital indices (if None, uses HOMO-centered selection)

    Returns:
        (MolecularIntegrals, core_energy)
    """
    n_orbitals = mol.nao
    n_electrons = mol.nelectron

    # Determine active orbital indices
    if active_orbitals is None:
        # Select orbitals around HOMO/LUMO
        n_occ = n_electrons // 2
        n_active_occ = n_active_electrons // 2
        n_active_virt = n_active_orbitals - n_active_occ

        start_occ = max(0, n_occ - n_active_occ)
        end_virt = min(n_orbitals, n_occ + n_active_virt)
        active_orbitals = list(range(start_occ, end_virt))

    # Get MO coefficients for active space
    mo_coeff = mf.mo_coeff
    mo_active = mo_coeff[:, active_orbitals]

    # Core orbitals (frozen)
    core_orbitals = [i for i in range(min(active_orbitals))]
    n_core = len(core_orbitals)

    # Compute core energy contribution
    if n_core > 0:
        mo_core = mo_coeff[:, core_orbitals]
        dm_core = 2.0 * mo_core @ mo_core.T
        h1_ao = mf.get_hcore()
        j_core, k_core = mf.get_jk(dm=dm_core)
        core_energy = mol.energy_nuc() + 0.5 * np.einsum('ij,ji->', h1_ao + 0.5 * j_core - 0.25 * k_core, dm_core)

        # Effective one-electron integrals include core contribution
        h1e_eff = mo_active.T @ (h1_ao + j_core - 0.5 * k_core) @ mo_active
    else:
        core_energy = mol.energy_nuc()
        h1e_eff = mo_active.T @ mf.get_hcore() @ mo_active

    # Two-electron integrals in active space
    h2e_active = ao2mo.kernel(mol, mo_active)
    h2e_active = ao2mo.restore(1, h2e_active, n_active_orbitals)

    # Compute number of alpha/beta electrons in active space
    n_alpha = (n_active_electrons + mol.spin) // 2
    n_beta = (n_active_electrons - mol.spin) // 2

    integrals = MolecularIntegrals(
        h1e=h1e_eff,
        h2e=h2e_active,
        nuclear_repulsion=core_energy,
        n_electrons=n_active_electrons,
        n_orbitals=n_active_orbitals,
        n_alpha=n_alpha,
        n_beta=n_beta,
    )

    return integrals, core_energy


def configs_to_set(configs: torch.Tensor) -> set:
    """Convert tensor of configurations to set of tuples."""
    return {tuple(c.cpu().tolist()) for c in configs}


def set_to_configs(config_set: set, n_sites: int, device: str) -> torch.Tensor:
    """Convert set of tuples back to tensor."""
    configs = [list(c) for c in config_set]
    return torch.tensor(configs, dtype=torch.long, device=device)


def compute_basis_energy(H: MolecularHamiltonian, basis: torch.Tensor) -> float:
    """Compute ground state energy by diagonalizing H in given basis."""
    if len(basis) == 0:
        return float('inf')
    H_matrix = H.matrix_elements(basis, basis)
    H_np = H_matrix.cpu().numpy().astype(np.float64)
    H_np = 0.5 * (H_np + H_np.T)
    eigenvalues, _ = np.linalg.eigh(H_np)
    return float(eigenvalues[0])


# =============================================================================
# Large Molecule Definitions
# =============================================================================

def create_cr2_hamiltonian(
    bond_length: float = 1.68,  # Equilibrium ~1.68 Å
    n_active_orbitals: int = 12,
    n_active_electrons: int = 12,
    device: str = "cuda",
) -> Tuple[MolecularHamiltonian, dict]:
    """
    Create Cr₂ (chromium dimer) Hamiltonian with active space.

    Cr₂ is notoriously difficult due to:
    - Formal sextuple bond (12 electrons in 12 orbitals: 3d + 4s)
    - Strong multi-reference character
    - Requires large active spaces for accurate description

    Common active spaces:
    - (12e, 12o): Minimal for 3d-3d correlation
    - (12e, 20o): Includes 4d double-shell
    - (12e, 28o): Includes 4d + 4f
    """
    print(f"Creating Cr₂ with bond length {bond_length} Å")
    print(f"Active space: ({n_active_electrons}e, {n_active_orbitals}o)")

    mol = gto.Mole()
    mol.atom = [
        ('Cr', (0.0, 0.0, 0.0)),
        ('Cr', (0.0, 0.0, bond_length)),
    ]
    mol.basis = 'cc-pvdz'
    mol.spin = 0  # Singlet ground state
    mol.symmetry = False
    mol.verbose = 3
    mol.build()

    # Run HF
    mf = scf.RHF(mol)
    mf.max_cycle = 200
    mf.kernel()
    hf_energy = mf.e_tot
    print(f"HF energy: {hf_energy:.8f} Ha")

    # Compute active space integrals
    integrals, core_energy = compute_active_space_integrals(
        mol, mf, n_active_orbitals, n_active_electrons
    )

    H = MolecularHamiltonian(integrals, device=device)

    info = {
        'hf_energy': hf_energy,
        'core_energy': core_energy,
        'mol': mol,
        'mf': mf,
    }

    return H, info


def create_benzene_hamiltonian(
    n_active_orbitals: int = 6,
    n_active_electrons: int = 6,
    device: str = "cuda",
) -> Tuple[MolecularHamiltonian, dict]:
    """
    Create benzene Hamiltonian with π-electron active space.

    Benzene π system:
    - 6 π electrons in 6 π orbitals (3 bonding + 3 antibonding)
    - Well-studied system for testing methods
    - (6e, 6o): C(6,3)² = 400 valid configs
    - (6e, 12o): With σ* → ~40,000 configs
    """
    print(f"Creating benzene with ({n_active_electrons}e, {n_active_orbitals}o) active space")

    # Benzene geometry (regular hexagon, C-C = 1.40 Å)
    cc_bond = 1.40
    ch_bond = 1.09

    # Hexagon coordinates
    angles = [i * 60 for i in range(6)]
    carbon_coords = []
    hydrogen_coords = []

    for i, angle in enumerate(angles):
        rad = np.radians(angle)
        x = cc_bond * np.cos(rad)
        y = cc_bond * np.sin(rad)
        carbon_coords.append(('C', (x, y, 0.0)))

        # H atoms at larger radius
        hx = (cc_bond + ch_bond) * np.cos(rad)
        hy = (cc_bond + ch_bond) * np.sin(rad)
        hydrogen_coords.append(('H', (hx, hy, 0.0)))

    mol = gto.Mole()
    mol.atom = carbon_coords + hydrogen_coords
    mol.basis = 'sto-3g'  # Use minimal basis for speed
    mol.spin = 0
    mol.verbose = 3
    mol.build()

    # Run HF
    mf = scf.RHF(mol)
    mf.kernel()
    hf_energy = mf.e_tot
    print(f"HF energy: {hf_energy:.8f} Ha")

    # For benzene, we want the π orbitals
    # In minimal basis, these are typically orbitals around HOMO/LUMO
    integrals, core_energy = compute_active_space_integrals(
        mol, mf, n_active_orbitals, n_active_electrons
    )

    H = MolecularHamiltonian(integrals, device=device)

    info = {
        'hf_energy': hf_energy,
        'core_energy': core_energy,
        'mol': mol,
        'mf': mf,
    }

    return H, info


def create_fe_porphyrin_model_hamiltonian(
    n_active_orbitals: int = 10,
    n_active_electrons: int = 8,
    device: str = "cuda",
) -> Tuple[MolecularHamiltonian, dict]:
    """
    Create a simplified Fe-porphyrin model Hamiltonian.

    Full Fe-porphyrin (FeN₄C₂₀H₁₂) is too large, so we use:
    - Fe(NH₃)₄²⁺ as a model (Fe with 4 N ligands in square planar)
    - Active space: Fe 3d orbitals + ligand orbitals

    Active spaces:
    - (8e, 10o): Fe 3d + N lone pairs → ~6,300 configs
    - (8e, 12o): Includes 4s → ~15,000 configs
    - (10e, 12o): → ~20,000+ configs
    """
    print(f"Creating Fe-porphyrin model with ({n_active_electrons}e, {n_active_orbitals}o) active space")

    # Fe(NH₃)₄²⁺ square planar geometry
    fe_n_dist = 2.0  # Å

    mol = gto.Mole()
    mol.atom = [
        ('Fe', (0.0, 0.0, 0.0)),
        ('N', (fe_n_dist, 0.0, 0.0)),
        ('N', (-fe_n_dist, 0.0, 0.0)),
        ('N', (0.0, fe_n_dist, 0.0)),
        ('N', (0.0, -fe_n_dist, 0.0)),
        # Simplified: no H atoms on NH₃ to reduce size
    ]
    mol.basis = 'sto-3g'
    mol.charge = 2  # Fe²⁺
    mol.spin = 4  # High-spin Fe(II) has 4 unpaired electrons
    mol.verbose = 3
    mol.build()

    # Run ROHF for open-shell
    mf = scf.ROHF(mol)
    mf.max_cycle = 200
    mf.kernel()
    hf_energy = mf.e_tot
    print(f"HF energy: {hf_energy:.8f} Ha")

    # Compute active space integrals
    integrals, core_energy = compute_active_space_integrals(
        mol, mf, n_active_orbitals, n_active_electrons
    )

    H = MolecularHamiltonian(integrals, device=device)

    info = {
        'hf_energy': hf_energy,
        'core_energy': core_energy,
        'mol': mol,
        'mf': mf,
    }

    return H, info


def create_n2_large_basis_hamiltonian(
    bond_length: float = 1.10,
    basis: str = "cc-pvdz",
    n_active_orbitals: int = 14,
    n_active_electrons: int = 10,
    device: str = "cuda",
) -> Tuple[MolecularHamiltonian, dict]:
    """
    Create N₂ with larger basis set and active space.

    N₂ with cc-pVDZ:
    - Full space: 28 orbitals, 14 electrons
    - Active (10e, 14o): Valence space → 26,334 configs
    - Active (10e, 10o): Minimal valence → 14,400 configs
    """
    print(f"Creating N₂ ({basis}) with ({n_active_electrons}e, {n_active_orbitals}o) active space")

    mol = gto.Mole()
    mol.atom = [
        ('N', (0.0, 0.0, 0.0)),
        ('N', (0.0, 0.0, bond_length)),
    ]
    mol.basis = basis
    mol.spin = 0
    mol.verbose = 3
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()
    hf_energy = mf.e_tot
    print(f"HF energy: {hf_energy:.8f} Ha")

    integrals, core_energy = compute_active_space_integrals(
        mol, mf, n_active_orbitals, n_active_electrons
    )

    H = MolecularHamiltonian(integrals, device=device)

    info = {
        'hf_energy': hf_energy,
        'core_energy': core_energy,
        'mol': mol,
        'mf': mf,
    }

    return H, info


def create_butadiene_hamiltonian(
    n_active_orbitals: int = 8,
    n_active_electrons: int = 8,
    device: str = "cuda",
) -> Tuple[MolecularHamiltonian, dict]:
    """
    Create butadiene (C₄H₆) Hamiltonian.

    trans-1,3-Butadiene π system:
    - 4 π electrons in 4 π orbitals (minimal)
    - Extended to (8e, 8o) including σ → ~4,900 configs
    - (8e, 12o) → ~34,650 configs
    """
    print(f"Creating butadiene with ({n_active_electrons}e, {n_active_orbitals}o) active space")

    # trans-Butadiene geometry
    cc_single = 1.46  # C-C single bond
    cc_double = 1.34  # C=C double bond
    ch_bond = 1.09

    mol = gto.Mole()
    mol.atom = [
        ('C', (0.0, 0.0, 0.0)),
        ('C', (cc_double, 0.0, 0.0)),
        ('C', (cc_double + cc_single, 0.0, 0.0)),
        ('C', (2*cc_double + cc_single, 0.0, 0.0)),
        ('H', (-ch_bond * 0.866, ch_bond * 0.5, 0.0)),
        ('H', (-ch_bond * 0.866, -ch_bond * 0.5, 0.0)),
        ('H', (2*cc_double + cc_single + ch_bond * 0.866, ch_bond * 0.5, 0.0)),
        ('H', (2*cc_double + cc_single + ch_bond * 0.866, -ch_bond * 0.5, 0.0)),
        ('H', (cc_double + cc_single/2, ch_bond, 0.0)),
        ('H', (cc_double + cc_single/2, -ch_bond, 0.0)),
    ]
    mol.basis = 'sto-3g'
    mol.spin = 0
    mol.verbose = 3
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()
    hf_energy = mf.e_tot
    print(f"HF energy: {hf_energy:.8f} Ha")

    integrals, core_energy = compute_active_space_integrals(
        mol, mf, n_active_orbitals, n_active_electrons
    )

    H = MolecularHamiltonian(integrals, device=device)

    info = {
        'hf_energy': hf_energy,
        'core_energy': core_energy,
        'mol': mol,
        'mf': mf,
    }

    return H, info


def create_ozone_hamiltonian(
    n_active_orbitals: int = 12,
    n_active_electrons: int = 12,
    device: str = "cuda",
) -> Tuple[MolecularHamiltonian, dict]:
    """
    Create ozone (O₃) Hamiltonian.

    Ozone is a classic multi-reference system:
    - Bent geometry (116.8°)
    - Significant diradical character
    - (12e, 9o): ~6,000 configs
    - (12e, 12o): ~62,000 configs
    """
    print(f"Creating ozone with ({n_active_electrons}e, {n_active_orbitals}o) active space")

    # Ozone geometry
    oo_bond = 1.278  # Å
    angle = 116.8  # degrees

    angle_rad = np.radians(angle / 2)

    mol = gto.Mole()
    mol.atom = [
        ('O', (0.0, 0.0, 0.0)),
        ('O', (oo_bond * np.cos(angle_rad), oo_bond * np.sin(angle_rad), 0.0)),
        ('O', (oo_bond * np.cos(angle_rad), -oo_bond * np.sin(angle_rad), 0.0)),
    ]
    mol.basis = 'cc-pvdz'
    mol.spin = 0
    mol.verbose = 3
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()
    hf_energy = mf.e_tot
    print(f"HF energy: {hf_energy:.8f} Ha")

    integrals, core_energy = compute_active_space_integrals(
        mol, mf, n_active_orbitals, n_active_electrons
    )

    H = MolecularHamiltonian(integrals, device=device)

    info = {
        'hf_energy': hf_energy,
        'core_energy': core_energy,
        'mol': mol,
        'mf': mf,
    }

    return H, info


# =============================================================================
# Benchmark Function
# =============================================================================

def run_large_system_benchmark(
    system_name: str,
    H: MolecularHamiltonian,
    info: dict,
    max_residual_iters: int = 20,
    residual_configs_per_iter: int = 500,
    max_krylov_dim: int = 15,
    shots_per_krylov: int = 200000,
    compute_casci: bool = False,
) -> LargeSystemResult:
    """
    Run comprehensive benchmark on a large molecular system.
    """
    n_valid = comb(H.n_orbitals, H.n_alpha) * comb(H.n_orbitals, H.n_beta)

    print_banner(f"BENCHMARK: {system_name}")
    print(f"Active space: ({H.n_alpha + H.n_beta}e, {H.n_orbitals}o)")
    print(f"Qubits: {H.num_sites}")
    print(f"Valid configurations: {n_valid:,}")
    print(f"Full Hilbert space: {2**H.num_sites:,}")
    print(f"Compression ratio: {n_valid / 2**H.num_sites * 100:.2f}%")

    result = LargeSystemResult(
        name=system_name,
        n_orbitals=H.n_orbitals,
        n_electrons=H.n_alpha + H.n_beta,
        n_alpha=H.n_alpha,
        n_beta=H.n_beta,
        n_qubits=H.num_sites,
        n_valid_configs=n_valid,
        hf_energy=info.get('hf_energy', 0.0),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Optionally compute CASCI reference
    if compute_casci and n_valid <= 50000:
        try:
            print("\nComputing CASCI reference...")
            start = time.time()
            casci_energy = H.fci_energy()
            result.casci_energy = casci_energy
            print(f"CASCI energy: {casci_energy:.8f} Ha ({time.time()-start:.1f}s)")
        except Exception as e:
            print(f"CASCI failed: {e}")
            result.casci_energy = None

    # =======================================================================
    # Step 1: NF-NQS Training
    # =======================================================================
    print("\n--- Step 1: NF-NQS Training ---")
    start_nf = time.time()

    config_nf = PipelineConfig(
        use_residual_expansion=False,
        skip_skqd=True,
        max_epochs=800,
        samples_per_batch=4000,
        device=device,
    )
    pipeline = FlowGuidedKrylovPipeline(H, config=config_nf, exact_energy=result.casci_energy)
    pipeline.train_flow_nqs(progress=True)
    nf_basis = pipeline.extract_and_select_basis()

    nf_set = configs_to_set(nf_basis)
    result.nf_configs = len(nf_set)
    result.nf_energy = compute_basis_energy(H, nf_basis)
    result.time_nf = time.time() - start_nf

    print(f"NF discovered: {result.nf_configs:,} configs")
    print(f"NF energy: {result.nf_energy:.8f} Ha")

    # =======================================================================
    # Step 2: Residual (PT2) Expansion
    # =======================================================================
    print("\n--- Step 2: Residual (PT2) Expansion ---")
    start_residual = time.time()

    residual_config = ResidualExpansionConfig(
        max_configs_per_iter=residual_configs_per_iter,
        max_iterations=max_residual_iters,
    )
    expander = SelectedCIExpander(H, residual_config)

    expanded_basis = nf_basis.clone()
    for i in range(max_residual_iters):
        old_size = len(expanded_basis)
        expanded_basis, stats = expander.expand_basis(expanded_basis)
        added = stats['configs_added']
        if added == 0:
            print(f"  Iter {i+1}: Converged (no new configs)")
            break
        print(f"  Iter {i+1}: {old_size:,} -> {len(expanded_basis):,} (+{added})")

    residual_set = configs_to_set(expanded_basis)
    residual_new = residual_set - nf_set
    result.residual_new_configs = len(residual_new)
    result.nf_residual_energy = compute_basis_energy(H, expanded_basis)
    result.time_residual = time.time() - start_residual

    print(f"Residual found: {result.residual_new_configs:,} NEW configs")
    print(f"NF+Residual energy: {result.nf_residual_energy:.8f} Ha")

    # =======================================================================
    # Step 3: Krylov Time Evolution
    # =======================================================================
    print("\n--- Step 3: Krylov Time Evolution ---")
    start_krylov = time.time()

    skqd_config = SKQDConfig(
        max_krylov_dim=max_krylov_dim,
        time_step=0.1,
        shots_per_krylov=shots_per_krylov,
    )

    skqd = FlowGuidedSKQD(H, nf_basis, skqd_config)
    skqd_results = skqd.run_with_nf(max_krylov_dim=max_krylov_dim, progress=True)

    # Collect Krylov-discovered configs
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
    result.time_krylov = time.time() - start_krylov

    print(f"Krylov found: {result.krylov_new_configs:,} NEW configs")
    print(f"NF+Krylov energy: {result.nf_krylov_energy:.8f} Ha")

    # =======================================================================
    # Step 4: Analyze Unique Contributions
    # =======================================================================
    print("\n--- Step 4: Configuration Analysis ---")

    krylov_unique = krylov_set - residual_set
    result.krylov_unique_configs = len(krylov_unique)

    # Combined basis
    all_configs = nf_set | residual_set | krylov_set
    all_basis = set_to_configs(all_configs, H.num_sites, device)
    result.combined_energy = compute_basis_energy(H, all_basis)

    print(f"Krylov-UNIQUE configs: {result.krylov_unique_configs:,}")
    print(f"Combined basis: {len(all_configs):,} configs")
    print(f"Combined energy: {result.combined_energy:.8f} Ha")

    # =======================================================================
    # Step 5: Determine Necessity
    # =======================================================================
    result.skqd_necessary = result.krylov_unique_configs > 0

    if result.casci_energy is not None:
        nf_error = abs(result.nf_energy - result.casci_energy) * 1000
        residual_error = abs(result.nf_residual_energy - result.casci_energy) * 1000
        krylov_error = abs(result.nf_krylov_energy - result.casci_energy) * 1000
        combined_error = abs(result.combined_energy - result.casci_energy) * 1000

        krylov_unique_contribution = residual_error - combined_error

        result.notes = (
            f"Errors (mHa): NF={nf_error:.2f}, Res={residual_error:.2f}, "
            f"Kry={krylov_error:.2f}, Comb={combined_error:.2f}; "
            f"Krylov-unique contribution: {krylov_unique_contribution:.2f} mHa"
        )
    else:
        result.notes = f"Krylov-unique: {result.krylov_unique_configs}"

    # =======================================================================
    # Results Summary
    # =======================================================================
    print("\n" + "=" * 80)
    print(f"RESULTS: {system_name}")
    print("=" * 80)
    print(f"{'Metric':<35} {'Value':<20}")
    print("-" * 55)
    print(f"{'Valid configurations':<35} {n_valid:<20,}")
    print(f"{'NF configs':<35} {result.nf_configs:<20,}")
    print(f"{'Residual NEW configs':<35} {result.residual_new_configs:<20,}")
    print(f"{'Krylov NEW configs':<35} {result.krylov_new_configs:<20,}")
    print(f"{'Krylov UNIQUE configs':<35} {result.krylov_unique_configs:<20,}")
    print("-" * 55)
    print(f"{'HF energy (Ha)':<35} {result.hf_energy:<20.8f}")
    print(f"{'NF energy (Ha)':<35} {result.nf_energy:<20.8f}")
    print(f"{'NF+Residual energy (Ha)':<35} {result.nf_residual_energy:<20.8f}")
    print(f"{'NF+Krylov energy (Ha)':<35} {result.nf_krylov_energy:<20.8f}")
    print(f"{'Combined energy (Ha)':<35} {result.combined_energy:<20.8f}")
    if result.casci_energy is not None:
        print(f"{'CASCI energy (Ha)':<35} {result.casci_energy:<20.8f}")
    print("-" * 55)
    print(f"{'Time NF (s)':<35} {result.time_nf:<20.1f}")
    print(f"{'Time Residual (s)':<35} {result.time_residual:<20.1f}")
    print(f"{'Time Krylov (s)':<35} {result.time_krylov:<20.1f}")
    print("-" * 55)

    if result.skqd_necessary:
        print(f"\n>>> VERDICT: SKQD is NECESSARY <<<")
        print(f"    Found {result.krylov_unique_configs:,} configs that Residual missed")
    else:
        print(f"\n>>> VERDICT: SKQD is HELPFUL but not strictly necessary <<<")

    return result


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Large System SKQD Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Systems available:
    cr2           - Chromium dimer (12e, 12o) - ~43,000 configs
    benzene       - Benzene π system (6e, 6o) - 400 configs
    benzene_large - Benzene extended (6e, 12o) - ~40,000 configs
    fe_porphyrin  - Fe-porphyrin model (8e, 10o) - ~6,300 configs
    n2_large      - N₂ cc-pVDZ (10e, 14o) - ~26,000 configs
    butadiene     - Butadiene (8e, 8o) - ~4,900 configs
    ozone         - Ozone (12e, 12o) - ~62,000 configs
    all           - Run all benchmarks
        """
    )
    parser.add_argument(
        "--system", "-s",
        type=str,
        default="all",
        help="System to benchmark"
    )
    parser.add_argument(
        "--compute-casci",
        action="store_true",
        help="Compute CASCI reference (slow for large systems)"
    )

    args = parser.parse_args()

    print_banner("LARGE SYSTEM SKQD BENCHMARK")
    print("Testing SKQD necessity on challenging molecular systems")
    print("Hypothesis: SKQD becomes necessary for systems with >20,000 configs")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    results = []

    # Define systems to test
    systems = {
        'benzene': lambda: create_benzene_hamiltonian(6, 6, device),
        'butadiene': lambda: create_butadiene_hamiltonian(8, 8, device),
        'fe_porphyrin': lambda: create_fe_porphyrin_model_hamiltonian(10, 8, device),
        'n2_large': lambda: create_n2_large_basis_hamiltonian(1.10, 'cc-pvdz', 12, 10, device),
        'ozone': lambda: create_ozone_hamiltonian(9, 12, device),
        'benzene_large': lambda: create_benzene_hamiltonian(12, 6, device),
        'cr2': lambda: create_cr2_hamiltonian(1.68, 12, 12, device),
    }

    if args.system == "all":
        # Run in order of increasing size
        test_order = ['benzene', 'butadiene', 'fe_porphyrin', 'n2_large', 'ozone']
    else:
        test_order = [args.system]

    for system_name in test_order:
        if system_name not in systems:
            print(f"Unknown system: {system_name}")
            continue

        try:
            H, info = systems[system_name]()
            result = run_large_system_benchmark(
                system_name,
                H,
                info,
                max_residual_iters=20,
                residual_configs_per_iter=500,
                max_krylov_dim=15,
                shots_per_krylov=200000,
                compute_casci=args.compute_casci,
            )
            results.append(result)
        except Exception as e:
            print(f"\nERROR running {system_name}: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    if results:
        print("\n" + "=" * 100)
        print("LARGE SYSTEM BENCHMARK SUMMARY")
        print("=" * 100)
        print(f"{'System':<15} {'Configs':<12} {'NF':<8} {'Res New':<10} {'Kry New':<10} "
              f"{'Kry Unique':<12} {'Verdict':<15}")
        print("-" * 100)

        necessary_count = 0
        for r in results:
            verdict = "NECESSARY" if r.skqd_necessary else "HELPFUL"
            print(f"{r.name:<15} {r.n_valid_configs:<12,} {r.nf_configs:<8,} "
                  f"{r.residual_new_configs:<10,} {r.krylov_new_configs:<10,} "
                  f"{r.krylov_unique_configs:<12,} {verdict:<15}")
            if r.skqd_necessary:
                necessary_count += 1

        print("-" * 100)
        print(f"NECESSARY: {necessary_count}/{len(results)} systems")

        print("\n" + "=" * 80)
        print("SCALING TREND ANALYSIS")
        print("=" * 80)
        print("Expected: Krylov-unique configs increase with system size")
        print("\nResults sorted by system size:")
        sorted_results = sorted(results, key=lambda r: r.n_valid_configs)
        for r in sorted_results:
            print(f"  {r.n_valid_configs:>10,} configs: {r.krylov_unique_configs:>6,} Krylov-unique "
                  f"({'NECESSARY' if r.skqd_necessary else 'helpful'})")


if __name__ == "__main__":
    main()
