"""Molecular Hamiltonians using PySCF."""

import torch
import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass

try:
    from .base import Hamiltonian
except ImportError:
    from hamiltonians.base import Hamiltonian


@dataclass
class MolecularIntegrals:
    """Container for molecular integrals."""

    h1e: np.ndarray  # One-electron integrals (n_orb, n_orb)
    h2e: np.ndarray  # Two-electron integrals (n_orb, n_orb, n_orb, n_orb)
    nuclear_repulsion: float
    n_electrons: int
    n_orbitals: int
    n_alpha: int  # Number of alpha electrons
    n_beta: int   # Number of beta electrons


class MolecularHamiltonian(Hamiltonian):
    """
    Second-quantized molecular Hamiltonian.

    H = Σ_{pq,σ} h_pq a†_{pσ} a_{qσ}
        + 1/2 Σ_{pqrs,στ} h_pqrs a†_{pσ} a†_{rτ} a_{sτ} a_{qσ}
        + E_nuc

    Uses Jordan-Wigner transformation to map to qubits:
    - α-spin orbitals on sites 0, 1, ..., n_orb-1
    - β-spin orbitals on sites n_orb, ..., 2*n_orb-1

    Args:
        integrals: MolecularIntegrals object
        active_space: Optional (n_electrons, n_orbitals) active space
    """

    def __init__(
        self,
        integrals: MolecularIntegrals,
        active_space: Optional[Tuple[int, int]] = None,
    ):
        n_qubits = 2 * integrals.n_orbitals  # Spin orbitals

        super().__init__(n_qubits, local_dim=2)

        self.integrals = integrals
        self.h1e = torch.from_numpy(integrals.h1e).float()
        self.h2e = torch.from_numpy(integrals.h2e).float()
        self.nuclear_repulsion = integrals.nuclear_repulsion
        self.n_orbitals = integrals.n_orbitals
        self.n_electrons = integrals.n_electrons

        # Precompute some useful quantities
        self._precompute_interactions()

    def _precompute_interactions(self):
        """Precompute interaction terms for efficient evaluation."""
        n_orb = self.n_orbitals

        # Store non-zero one-electron terms
        self.one_body_terms = []
        for p in range(n_orb):
            for q in range(n_orb):
                if abs(self.h1e[p, q].item()) > 1e-12:
                    self.one_body_terms.append((p, q, self.h1e[p, q].item()))

        # Store non-zero two-electron terms (physicist notation)
        # h_pqrs = (pq|rs) = ∫∫ φ_p*(1)φ_q(1) 1/r12 φ_r*(2)φ_s(2)
        self.two_body_terms = []
        for p in range(n_orb):
            for q in range(n_orb):
                for r in range(n_orb):
                    for s in range(n_orb):
                        val = self.h2e[p, q, r, s].item()
                        if abs(val) > 1e-12:
                            self.two_body_terms.append((p, q, r, s, val))

    def _orbital_to_qubit(self, orbital: int, spin: str) -> int:
        """Map orbital index and spin to qubit index."""
        if spin == "alpha" or spin == "a":
            return orbital
        else:  # beta
            return orbital + self.n_orbitals

    def _qubit_to_orbital(self, qubit: int) -> Tuple[int, str]:
        """Map qubit index to orbital and spin."""
        if qubit < self.n_orbitals:
            return qubit, "alpha"
        else:
            return qubit - self.n_orbitals, "beta"

    def diagonal_element(self, config: torch.Tensor) -> torch.Tensor:
        """
        Compute diagonal element ⟨x|H|x⟩.

        For a Slater determinant |x⟩, the diagonal element is:
        E = E_nuc + Σ_{p∈occ} h_pp + 1/2 Σ_{p,q∈occ} (h_pqpq - h_pqqp δ_σpσq)
        """
        device = config.device

        # Get occupied orbitals
        occ_alpha = torch.where(config[:self.n_orbitals] == 1)[0]
        occ_beta = torch.where(config[self.n_orbitals:] == 1)[0]

        energy = self.nuclear_repulsion

        # One-body terms
        for p in occ_alpha:
            energy += self.h1e[p, p].item()
        for p in occ_beta:
            energy += self.h1e[p, p].item()

        # Two-body terms: Coulomb - Exchange
        # Same spin (alpha-alpha)
        for p in occ_alpha:
            for q in occ_alpha:
                if p != q:
                    # Coulomb
                    energy += 0.5 * self.h2e[p, p, q, q].item()
                    # Exchange
                    energy -= 0.5 * self.h2e[p, q, q, p].item()

        # Same spin (beta-beta)
        for p in occ_beta:
            for q in occ_beta:
                if p != q:
                    energy += 0.5 * self.h2e[p, p, q, q].item()
                    energy -= 0.5 * self.h2e[p, q, q, p].item()

        # Different spin (alpha-beta)
        for p in occ_alpha:
            for q in occ_beta:
                # Coulomb only (no exchange for different spins)
                energy += self.h2e[p, p, q, q].item()

        return torch.tensor(energy, device=device)

    def get_connections(
        self, config: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get off-diagonal connections.

        Off-diagonal elements arise from:
        1. Single excitations: a†_p a_q with p ≠ q (one-body terms)
        2. Double excitations: a†_p a†_r a_s a_q (two-body terms)

        Returns only configurations with non-zero matrix elements.
        """
        device = config.device
        connected = []
        elements = []

        occ_alpha = torch.where(config[:self.n_orbitals] == 1)[0].tolist()
        occ_beta = torch.where(config[self.n_orbitals:] == 1)[0].tolist()
        virt_alpha = torch.where(config[:self.n_orbitals] == 0)[0].tolist()
        virt_beta = torch.where(config[self.n_orbitals:] == 0)[0].tolist()

        # ===== SINGLE EXCITATIONS (one-body terms) =====
        # Alpha spin: a†_p a_q for p ∈ virtual, q ∈ occupied
        for q in occ_alpha:
            for p in virt_alpha:
                if abs(self.h1e[p, q].item()) > 1e-12:
                    new_config = config.clone()
                    new_config[q] = 0
                    new_config[p] = 1

                    sign = self._jw_sign(config, p, q)
                    element = sign * self.h1e[p, q].item()

                    connected.append(new_config)
                    elements.append(element)

        # Beta spin single excitations
        for q in occ_beta:
            for p in virt_beta:
                if abs(self.h1e[p, q].item()) > 1e-12:
                    new_config = config.clone()
                    new_config[q + self.n_orbitals] = 0
                    new_config[p + self.n_orbitals] = 1

                    sign = self._jw_sign(
                        config, p + self.n_orbitals, q + self.n_orbitals
                    )
                    element = sign * self.h1e[p, q].item()

                    connected.append(new_config)
                    elements.append(element)

        # ===== DOUBLE EXCITATIONS (two-body terms) =====
        # The two-electron term: 1/2 Σ h_pqrs a†_p a†_r a_s a_q
        # Off-diagonal: excite from (q,s) occupied to (p,r) virtual

        # Alpha-Alpha double excitations
        for i, q in enumerate(occ_alpha):
            for s in occ_alpha[i+1:]:  # s > q to avoid double counting
                for j, p in enumerate(virt_alpha):
                    for r in virt_alpha[j+1:]:  # r > p to avoid double counting
                        # h_pqrs - h_pqsr (exchange)
                        val = self.h2e[p, q, r, s].item() - self.h2e[p, s, r, q].item()
                        if abs(val) > 1e-12:
                            new_config = config.clone()
                            new_config[q] = 0
                            new_config[s] = 0
                            new_config[p] = 1
                            new_config[r] = 1

                            # JW sign for double excitation
                            sign = self._jw_sign_double(config, p, r, q, s)
                            connected.append(new_config)
                            elements.append(sign * val)

        # Beta-Beta double excitations
        for i, q in enumerate(occ_beta):
            for s in occ_beta[i+1:]:
                for j, p in enumerate(virt_beta):
                    for r in virt_beta[j+1:]:
                        val = self.h2e[p, q, r, s].item() - self.h2e[p, s, r, q].item()
                        if abs(val) > 1e-12:
                            new_config = config.clone()
                            q_idx = q + self.n_orbitals
                            s_idx = s + self.n_orbitals
                            p_idx = p + self.n_orbitals
                            r_idx = r + self.n_orbitals
                            new_config[q_idx] = 0
                            new_config[s_idx] = 0
                            new_config[p_idx] = 1
                            new_config[r_idx] = 1

                            sign = self._jw_sign_double(config, p_idx, r_idx, q_idx, s_idx)
                            connected.append(new_config)
                            elements.append(sign * val)

        # Alpha-Beta double excitations (no exchange term)
        for q in occ_alpha:
            for s in occ_beta:
                for p in virt_alpha:
                    for r in virt_beta:
                        val = self.h2e[p, q, r, s].item()
                        if abs(val) > 1e-12:
                            new_config = config.clone()
                            s_idx = s + self.n_orbitals
                            r_idx = r + self.n_orbitals
                            new_config[q] = 0
                            new_config[s_idx] = 0
                            new_config[p] = 1
                            new_config[r_idx] = 1

                            sign = self._jw_sign_double(config, p, r_idx, q, s_idx)
                            connected.append(new_config)
                            elements.append(sign * val)

        if len(connected) == 0:
            return torch.tensor([], device=device), torch.tensor([], device=device)

        return torch.stack(connected), torch.tensor(elements, device=device)

    def _jw_sign_double(
        self, config: torch.Tensor, p: int, r: int, q: int, s: int
    ) -> int:
        """
        Compute Jordan-Wigner sign for double excitation a†_p a†_r a_s a_q.

        The sign comes from anticommutation of fermionic operators through
        the JW string.
        """
        # Order the operators: a†_p a†_r a_s a_q
        # Need to track parity of permutations and JW strings
        indices = sorted([p, r, q, s])

        # Count occupied sites in ranges
        total_count = 0

        # For a†_p: count occupied below p
        total_count += torch.sum(config[:p]).item()

        # For a†_r: count occupied below r (excluding q if q < r)
        count_r = torch.sum(config[:r]).item()
        if q < r:
            count_r -= config[q].item()
        total_count += count_r

        # For a_s: count occupied below s (excluding contributions from p, r, q)
        count_s = torch.sum(config[:s]).item()
        if p < s:
            count_s += 1  # p is now occupied
        if r < s:
            count_s += 1  # r is now occupied
        if q < s:
            count_s -= config[q].item()
        total_count += count_s

        # For a_q: similar logic
        count_q = torch.sum(config[:q]).item()
        if p < q:
            count_q += 1
        if r < q:
            count_q += 1
        if s < q:
            count_q -= config[s].item()
        total_count += count_q

        return (-1) ** int(total_count)

    def _jw_sign(self, config: torch.Tensor, p: int, q: int) -> int:
        """
        Compute Jordan-Wigner sign for a†_p a_q.

        Sign = (-1)^(number of occupied sites between p and q)
        """
        if p == q:
            return 1

        low, high = min(p, q), max(p, q)
        count = torch.sum(config[low + 1:high]).item()
        return (-1) ** int(count)

    def to_pauli_strings(self) -> Tuple[List[float], List[str]]:
        """
        Convert molecular Hamiltonian to Pauli string representation.

        Uses Jordan-Wigner transformation to map fermionic operators to
        Pauli strings for CUDA-Q integration.

        Returns:
            (coefficients, pauli_words): Lists of coefficients and Pauli strings
        """
        n_qubits = self.num_sites
        coefficients = []
        pauli_words = []

        # Nuclear repulsion contributes to identity term
        coefficients.append(self.nuclear_repulsion)
        pauli_words.append("I" * n_qubits)

        # One-body terms: h_pq (a†_pσ a_qσ)
        # JW: a†_p a_q = (X_p - iY_p)/2 * Z_{p+1}...Z_{q-1} * (X_q + iY_q)/2
        for p in range(self.n_orbitals):
            for q in range(self.n_orbitals):
                h_pq = self.h1e[p, q].item()
                if abs(h_pq) < 1e-12:
                    continue

                # Add terms for both alpha and beta spin
                for spin_offset in [0, self.n_orbitals]:
                    p_qubit = p + spin_offset
                    q_qubit = q + spin_offset

                    if p_qubit == q_qubit:
                        # Number operator: n_p = (I - Z_p)/2
                        # Contributes h_pp * (1/2) to identity
                        coefficients.append(h_pq / 2)
                        pauli_words.append("I" * n_qubits)

                        # Contributes -h_pp * (1/2) to Z_p
                        pauli = ["I"] * n_qubits
                        pauli[p_qubit] = "Z"
                        coefficients.append(-h_pq / 2)
                        pauli_words.append("".join(pauli))
                    else:
                        # Hopping term: a†_p a_q + h.c.
                        # = (1/2)(X_p Z... X_q + Y_p Z... Y_q)
                        low, high = min(p_qubit, q_qubit), max(p_qubit, q_qubit)

                        # XX term
                        pauli = ["I"] * n_qubits
                        pauli[p_qubit] = "X"
                        pauli[q_qubit] = "X"
                        for k in range(low + 1, high):
                            pauli[k] = "Z"
                        coefficients.append(h_pq / 2)
                        pauli_words.append("".join(pauli))

                        # YY term
                        pauli = ["I"] * n_qubits
                        pauli[p_qubit] = "Y"
                        pauli[q_qubit] = "Y"
                        for k in range(low + 1, high):
                            pauli[k] = "Z"
                        coefficients.append(h_pq / 2)
                        pauli_words.append("".join(pauli))

        # Two-body terms (simplified - diagonal contributions)
        # Full JW transformation of two-body terms is complex
        # Here we add the diagonal Coulomb/exchange contributions
        for p in range(self.n_orbitals):
            for q in range(self.n_orbitals):
                # Coulomb: (1/2) h_pqpq n_p n_q for different spins
                h_pqpq = self.h2e[p, p, q, q].item()
                if abs(h_pqpq) > 1e-12 and p != q:
                    # Same spin ZZ
                    for spin_offset in [0, self.n_orbitals]:
                        pauli = ["I"] * n_qubits
                        pauli[p + spin_offset] = "Z"
                        pauli[q + spin_offset] = "Z"
                        coefficients.append(h_pqpq / 8)
                        pauli_words.append("".join(pauli))

                    # Different spin ZZ (alpha-beta)
                    pauli = ["I"] * n_qubits
                    pauli[p] = "Z"
                    pauli[q + self.n_orbitals] = "Z"
                    coefficients.append(h_pqpq / 4)
                    pauli_words.append("".join(pauli))

        # Consolidate terms with same Pauli string
        consolidated = {}
        for coeff, pauli in zip(coefficients, pauli_words):
            if pauli in consolidated:
                consolidated[pauli] += coeff
            else:
                consolidated[pauli] = coeff

        # Filter out near-zero terms
        final_coeffs = []
        final_paulis = []
        for pauli, coeff in consolidated.items():
            if abs(coeff) > 1e-12:
                final_coeffs.append(coeff)
                final_paulis.append(pauli)

        return final_coeffs, final_paulis

    def get_hf_state(self) -> torch.Tensor:
        """
        Get Hartree-Fock reference state configuration.

        Returns the occupation pattern corresponding to the HF determinant,
        suitable for use as initial state in SKQD.

        Returns:
            Configuration tensor with 1s for occupied orbitals
        """
        config = torch.zeros(self.num_sites, dtype=torch.long)

        # Fill alpha orbitals
        for i in range(self.integrals.n_alpha):
            config[i] = 1

        # Fill beta orbitals
        for i in range(self.integrals.n_beta):
            config[i + self.n_orbitals] = 1

        return config


def compute_molecular_integrals(
    geometry: List[Tuple[str, Tuple[float, float, float]]],
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
) -> MolecularIntegrals:
    """
    Compute molecular integrals using PySCF.

    Args:
        geometry: List of (atom_symbol, (x, y, z)) tuples
        basis: Basis set name
        charge: Molecular charge
        spin: 2S (number of unpaired electrons)

    Returns:
        MolecularIntegrals object
    """
    try:
        from pyscf import gto, scf, ao2mo
    except ImportError:
        raise ImportError("PySCF is required for molecular Hamiltonians")

    # Build molecule
    mol = gto.Mole()
    mol.atom = geometry
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.build()

    # Run HF to get orbitals
    if spin == 0:
        mf = scf.RHF(mol)
    else:
        mf = scf.ROHF(mol)
    mf.kernel()

    # Get integrals in MO basis
    h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff

    # Two-electron integrals
    h2e = ao2mo.kernel(mol, mf.mo_coeff)
    h2e = ao2mo.restore(1, h2e, mol.nao)  # Restore to 4-index tensor

    n_electrons = mol.nelectron
    n_orbitals = mol.nao
    n_alpha = (n_electrons + spin) // 2
    n_beta = (n_electrons - spin) // 2

    return MolecularIntegrals(
        h1e=h1e,
        h2e=h2e,
        nuclear_repulsion=mol.energy_nuc(),
        n_electrons=n_electrons,
        n_orbitals=n_orbitals,
        n_alpha=n_alpha,
        n_beta=n_beta,
    )


def create_h2_hamiltonian(bond_length: float = 0.74) -> MolecularHamiltonian:
    """Create H2 Hamiltonian at given bond length."""
    geometry = [
        ("H", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.0, bond_length)),
    ]
    integrals = compute_molecular_integrals(geometry, basis="sto-3g")
    return MolecularHamiltonian(integrals)


def create_lih_hamiltonian(bond_length: float = 1.6) -> MolecularHamiltonian:
    """Create LiH Hamiltonian at given bond length."""
    geometry = [
        ("Li", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.0, bond_length)),
    ]
    integrals = compute_molecular_integrals(geometry, basis="sto-3g")
    return MolecularHamiltonian(integrals)


def create_h2o_hamiltonian(
    oh_length: float = 0.96,
    angle: float = 104.5,
) -> MolecularHamiltonian:
    """Create H2O Hamiltonian."""
    import numpy as np

    angle_rad = np.radians(angle)
    geometry = [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (oh_length, 0.0, 0.0)),
        ("H", (oh_length * np.cos(angle_rad), oh_length * np.sin(angle_rad), 0.0)),
    ]
    integrals = compute_molecular_integrals(geometry, basis="sto-3g")
    return MolecularHamiltonian(integrals)
