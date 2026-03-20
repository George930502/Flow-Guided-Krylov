"""
Molecule registry mapping molecule names to factory functions and metadata.

Provides a unified interface for creating molecular Hamiltonians with
known qubit counts, basis sets, and descriptions. Includes all STO-3G
systems up to 30 qubits plus CAS active-space systems.

Usage:
    from src.molecules import get_molecule, list_molecules

    names = list_molecules()
    H, info = get_molecule("H2O", device="cpu")
"""

import numpy as np
from functools import partial
from typing import Dict, List, Tuple

from .hamiltonians.molecular import (
    MolecularHamiltonian,
    MolecularIntegrals,
    compute_molecular_integrals,
    create_h2_hamiltonian,
    create_lih_hamiltonian,
    create_h2o_hamiltonian,
    create_beh2_hamiltonian,
    create_nh3_hamiltonian,
    create_n2_hamiltonian,
    create_ch4_hamiltonian,
    create_n2_cas_hamiltonian,
    create_cr2_hamiltonian,
    create_benzene_hamiltonian,
)


# =============================================================================
# New molecule factory functions (use compute_molecular_integrals directly)
# =============================================================================


def create_co_hamiltonian(
    bond_length: float = 1.13,
    basis: str = "sto-3g",
    device: str = "cpu",
) -> MolecularHamiltonian:
    """
    Create CO (carbon monoxide) Hamiltonian.

    Linear molecule with triple bond character.
    14 electrons, 10 orbitals in STO-3G -> 20 qubits.
    """
    geometry = [
        ("C", (0.0, 0.0, 0.0)),
        ("O", (0.0, 0.0, bond_length)),
    ]
    integrals = compute_molecular_integrals(geometry, basis=basis)
    return MolecularHamiltonian(integrals, device=device)


def create_hcn_hamiltonian(
    hc_length: float = 1.06,
    cn_length: float = 1.16,
    basis: str = "sto-3g",
    device: str = "cpu",
) -> MolecularHamiltonian:
    """
    Create HCN (hydrogen cyanide) Hamiltonian.

    Linear molecule: H-C-N with triple bond between C and N.
    14 electrons, 11 orbitals in STO-3G -> 22 qubits.
    """
    geometry = [
        ("H", (0.0, 0.0, 0.0)),
        ("C", (0.0, 0.0, hc_length)),
        ("N", (0.0, 0.0, hc_length + cn_length)),
    ]
    integrals = compute_molecular_integrals(geometry, basis=basis)
    return MolecularHamiltonian(integrals, device=device)


def create_c2h2_hamiltonian(
    cc_length: float = 1.20,
    ch_length: float = 1.06,
    basis: str = "sto-3g",
    device: str = "cpu",
) -> MolecularHamiltonian:
    """
    Create C2H2 (acetylene) Hamiltonian.

    Linear molecule: H-C-C-H with triple bond between carbons.
    14 electrons, 12 orbitals in STO-3G -> 24 qubits.
    """
    geometry = [
        ("H", (0.0, 0.0, 0.0)),
        ("C", (0.0, 0.0, ch_length)),
        ("C", (0.0, 0.0, ch_length + cc_length)),
        ("H", (0.0, 0.0, ch_length + cc_length + ch_length)),
    ]
    integrals = compute_molecular_integrals(geometry, basis=basis)
    return MolecularHamiltonian(integrals, device=device)


def create_h2s_hamiltonian(
    sh_length: float = 1.34,
    hsh_angle: float = 92.1,
    basis: str = "sto-3g",
    device: str = "cpu",
) -> MolecularHamiltonian:
    """
    Create H2S (hydrogen sulfide) Hamiltonian.

    Bent molecule similar to H2O but with smaller bond angle.
    18 electrons, 13 orbitals in STO-3G -> 26 qubits.
    """
    angle_rad = np.radians(hsh_angle)
    geometry = [
        ("S", (0.0, 0.0, 0.0)),
        ("H", (sh_length, 0.0, 0.0)),
        ("H", (sh_length * np.cos(angle_rad), sh_length * np.sin(angle_rad), 0.0)),
    ]
    integrals = compute_molecular_integrals(geometry, basis=basis)
    return MolecularHamiltonian(integrals, device=device)


def create_c2h4_hamiltonian(
    cc_length: float = 1.33,
    ch_length: float = 1.09,
    basis: str = "sto-3g",
    device: str = "cpu",
) -> MolecularHamiltonian:
    """
    Create C2H4 (ethylene) Hamiltonian.

    Planar molecule with C=C double bond. All atoms in the xy-plane.
    16 electrons, 14 orbitals in STO-3G -> 28 qubits.
    """
    # HCH angle ~117.4 degrees for ethylene
    hch_angle_rad = np.radians(121.3)  # H-C-C angle

    # First carbon at origin, second along x-axis
    c1 = (0.0, 0.0, 0.0)
    c2 = (cc_length, 0.0, 0.0)

    # Hydrogens on C1 (pointing left)
    h1 = (-ch_length * np.cos(np.pi - hch_angle_rad), ch_length * np.sin(np.pi - hch_angle_rad), 0.0)
    h2 = (-ch_length * np.cos(np.pi - hch_angle_rad), -ch_length * np.sin(np.pi - hch_angle_rad), 0.0)

    # Hydrogens on C2 (pointing right)
    h3 = (cc_length + ch_length * np.cos(np.pi - hch_angle_rad), ch_length * np.sin(np.pi - hch_angle_rad), 0.0)
    h4 = (cc_length + ch_length * np.cos(np.pi - hch_angle_rad), -ch_length * np.sin(np.pi - hch_angle_rad), 0.0)

    geometry = [
        ("C", c1),
        ("C", c2),
        ("H", h1),
        ("H", h2),
        ("H", h3),
        ("H", h4),
    ]
    integrals = compute_molecular_integrals(geometry, basis=basis)
    return MolecularHamiltonian(integrals, device=device)


# =============================================================================
# Molecule Registry
# =============================================================================

MOLECULE_REGISTRY: Dict[str, dict] = {
    # --- Small systems (4-14 qubits) ---
    "H2": {
        "factory": create_h2_hamiltonian,
        "n_qubits": 4,
        "description": "Hydrogen (H2)",
        "basis": "STO-3G",
        "is_cas": False,
    },
    "LiH": {
        "factory": create_lih_hamiltonian,
        "n_qubits": 12,
        "description": "Lithium hydride (LiH)",
        "basis": "STO-3G",
        "is_cas": False,
    },
    "H2O": {
        "factory": create_h2o_hamiltonian,
        "n_qubits": 14,
        "description": "Water (H2O)",
        "basis": "STO-3G",
        "is_cas": False,
    },
    "BeH2": {
        "factory": create_beh2_hamiltonian,
        "n_qubits": 14,
        "description": "Beryllium hydride (BeH2)",
        "basis": "STO-3G",
        "is_cas": False,
    },
    # --- Medium systems (16-18 qubits) ---
    "NH3": {
        "factory": create_nh3_hamiltonian,
        "n_qubits": 16,
        "description": "Ammonia (NH3)",
        "basis": "STO-3G",
        "is_cas": False,
    },
    "CH4": {
        "factory": create_ch4_hamiltonian,
        "n_qubits": 18,
        "description": "Methane (CH4)",
        "basis": "STO-3G",
        "is_cas": False,
    },
    # --- Large systems (20-30 qubits) ---
    "N2": {
        "factory": create_n2_hamiltonian,
        "n_qubits": 20,
        "description": "Nitrogen (N2)",
        "basis": "STO-3G",
        "is_cas": False,
    },
    "CO": {
        "factory": create_co_hamiltonian,
        "n_qubits": 20,
        "description": "Carbon monoxide (CO)",
        "basis": "STO-3G",
        "is_cas": False,
    },
    "HCN": {
        "factory": create_hcn_hamiltonian,
        "n_qubits": 22,
        "description": "Hydrogen cyanide (HCN)",
        "basis": "STO-3G",
        "is_cas": False,
    },
    "C2H2": {
        "factory": create_c2h2_hamiltonian,
        "n_qubits": 24,
        "description": "Acetylene (C2H2)",
        "basis": "STO-3G",
        "is_cas": False,
    },
    "N2-CAS(10,12)": {
        "factory": partial(create_n2_cas_hamiltonian, cas=(10, 12)),
        "n_qubits": 24,
        "description": "Nitrogen CAS(10,12) (N2)",
        "basis": "cc-pVDZ",
        "is_cas": True,
    },
    "Cr2": {
        "factory": create_cr2_hamiltonian,
        "n_qubits": 24,
        "description": "Chromium dimer (Cr2) CAS(12,12)",
        "basis": "STO-3G",
        "is_cas": True,
    },
    "H2S": {
        "factory": create_h2s_hamiltonian,
        "n_qubits": 26,
        "description": "Hydrogen sulfide (H2S)",
        "basis": "STO-3G",
        "is_cas": False,
    },
    "C2H4": {
        "factory": create_c2h4_hamiltonian,
        "n_qubits": 28,
        "description": "Ethylene (C2H4)",
        "basis": "STO-3G",
        "is_cas": False,
    },
    "N2-CAS(10,15)": {
        "factory": partial(create_n2_cas_hamiltonian, cas=(10, 15)),
        "n_qubits": 30,
        "description": "Nitrogen CAS(10,15) (N2)",
        "basis": "cc-pVDZ",
        "is_cas": True,
    },
    "Benzene": {
        "factory": create_benzene_hamiltonian,
        "n_qubits": 30,
        "description": "Benzene CAS(6,15) (C6H6)",
        "basis": "STO-3G",
        "is_cas": True,
    },
    # --- Large systems (34-40 qubits) ---
    "N2-CAS(10,17)": {
        "factory": partial(create_n2_cas_hamiltonian, cas=(10, 17)),
        "n_qubits": 34,
        "description": "Nitrogen CAS(10,17) cc-pVDZ (N2)",
        "basis": "cc-pVDZ",
        "is_cas": True,
    },
    "Cr2-CAS(12,18)": {
        "factory": partial(create_cr2_hamiltonian, basis="cc-pvdz", cas=(12, 18)),
        "n_qubits": 36,
        "description": "Chromium dimer CAS(12,18) cc-pVDZ (Cr2)",
        "basis": "cc-pVDZ",
        "is_cas": True,
    },
    "N2-CAS(10,20)": {
        "factory": partial(create_n2_cas_hamiltonian, basis="cc-pvtz", cas=(10, 20)),
        "n_qubits": 40,
        "description": "Nitrogen CAS(10,20) cc-pVTZ (N2)",
        "basis": "cc-pVTZ",
        "is_cas": True,
    },
    "Cr2-CAS(12,20)": {
        "factory": partial(create_cr2_hamiltonian, basis="cc-pvdz", cas=(12, 20)),
        "n_qubits": 40,
        "description": "Chromium dimer CAS(12,20) cc-pVDZ (Cr2)",
        "basis": "cc-pVDZ",
        "is_cas": True,
    },
    # --- Ultra-large systems (52-58 qubits) ---
    "N2-CAS(10,26)": {
        "factory": partial(create_n2_cas_hamiltonian, basis="cc-pvtz", cas=(10, 26)),
        "n_qubits": 52,
        "description": "Nitrogen CAS(10,26) cc-pVTZ (N2)",
        "basis": "cc-pVTZ",
        "is_cas": True,
    },
    "Cr2-CAS(12,26)": {
        "factory": partial(create_cr2_hamiltonian, basis="cc-pvdz", cas=(12, 26)),
        "n_qubits": 52,
        "description": "Chromium dimer CAS(12,26) cc-pVDZ (Cr2)",
        "basis": "cc-pVDZ",
        "is_cas": True,
    },
    "Cr2-CAS(12,28)": {
        "factory": partial(create_cr2_hamiltonian, basis="cc-pvdz", cas=(12, 28)),
        "n_qubits": 56,
        "description": "Chromium dimer CAS(12,28) cc-pVDZ (Cr2)",
        "basis": "cc-pVDZ",
        "is_cas": True,
    },
    "Cr2-CAS(12,29)": {
        "factory": partial(create_cr2_hamiltonian, basis="cc-pvdz", cas=(12, 29)),
        "n_qubits": 58,
        "description": "Chromium dimer CAS(12,29) cc-pVDZ (Cr2)",
        "basis": "cc-pVDZ",
        "is_cas": True,
    },
}


# =============================================================================
# Public API
# =============================================================================


def get_molecule(name: str, device: str = "cpu") -> Tuple[MolecularHamiltonian, dict]:
    """
    Create a molecular Hamiltonian by name.

    Args:
        name: Molecule name (case-sensitive). Use list_molecules() for valid names.
        device: Computation device ("cpu" or "cuda").

    Returns:
        Tuple of (MolecularHamiltonian, info_dict) where info_dict contains:
            - name: molecule name
            - n_qubits: number of qubits (2 * n_orbitals)
            - description: human-readable name
            - basis: basis set used
            - is_cas: whether this is a CAS active-space system

    Raises:
        KeyError: If molecule name is not in the registry.
    """
    if name not in MOLECULE_REGISTRY:
        available = ", ".join(sorted(MOLECULE_REGISTRY.keys()))
        raise KeyError(f"Unknown molecule '{name}'. Available: {available}")

    entry = MOLECULE_REGISTRY[name]
    factory = entry["factory"]
    hamiltonian = factory(device=device)

    info = {
        "name": name,
        "n_qubits": entry["n_qubits"],
        "description": entry["description"],
        "basis": entry["basis"],
        "is_cas": entry["is_cas"],
    }

    # Add geometry info for CCSD solver (reconstructing PySCF mol)
    integrals = hamiltonian.integrals
    if integrals._geometry is not None:
        info["geometry"] = integrals._geometry
        info["charge"] = integrals._charge
        info["spin"] = integrals._spin

    return hamiltonian, info


def list_molecules() -> List[str]:
    """
    Return sorted list of all available molecule names.

    Returns:
        List of molecule name strings that can be passed to get_molecule().
    """
    return sorted(MOLECULE_REGISTRY.keys())
