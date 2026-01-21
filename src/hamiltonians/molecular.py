"""
Molecular Hamiltonians using PySCF.

Optimizations included:
- Fully vectorized diagonal element computation (no Python loops)
- Precomputed Coulomb/Exchange tensors for GPU acceleration
- Hash-based configuration lookup for O(1) matrix element access
- Batched matrix construction
"""

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
    Second-quantized molecular Hamiltonian with GPU acceleration.

    H = sum_{pq,s} h_pq a+_{ps} a_{qs}
        + 1/2 sum_{pqrs,st} h_pqrs a+_{ps} a+_{rt} a_{st} a_{qs}
        + E_nuc

    Uses Jordan-Wigner transformation to map to qubits:
    - alpha-spin orbitals on sites 0, 1, ..., n_orb-1
    - beta-spin orbitals on sites n_orb, ..., 2*n_orb-1

    Includes optimizations:
    - Vectorized batch diagonal computation
    - Precomputed Coulomb/Exchange tensors
    - Hash-based configuration lookup

    Args:
        integrals: MolecularIntegrals object
        device: Torch device for GPU acceleration
    """

    def __init__(
        self,
        integrals: MolecularIntegrals,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        n_qubits = 2 * integrals.n_orbitals  # Spin orbitals

        super().__init__(n_qubits, local_dim=2)

        self.device = device
        self.integrals = integrals
        self.h1e = torch.from_numpy(integrals.h1e).float().to(device)
        self.h2e = torch.from_numpy(integrals.h2e).float().to(device)
        self.nuclear_repulsion = integrals.nuclear_repulsion
        self.n_orbitals = integrals.n_orbitals
        self.n_electrons = integrals.n_electrons
        self.n_alpha = integrals.n_alpha
        self.n_beta = integrals.n_beta

        # Precompute vectorized integral tensors
        self._precompute_vectorized_integrals()

        # Precompute single excitation data
        self._precompute_single_excitation_data()

    def _precompute_vectorized_integrals(self):
        """Precompute tensors for vectorized energy evaluation."""
        n_orb = self.n_orbitals
        device = self.device

        # One-body diagonal: h_pp
        self.h1_diag = torch.diag(self.h1e)  # (n_orb,)

        # Two-body Coulomb tensor: J_pq = h2e[p,p,q,q]
        self.J_tensor = torch.zeros(n_orb, n_orb, device=device)
        for p in range(n_orb):
            for q in range(n_orb):
                self.J_tensor[p, q] = self.h2e[p, p, q, q]

        # Two-body Exchange tensor: K_pq = h2e[p,q,q,p]
        self.K_tensor = torch.zeros(n_orb, n_orb, device=device)
        for p in range(n_orb):
            for q in range(n_orb):
                self.K_tensor[p, q] = self.h2e[p, q, q, p]

        # Precompute nonzero off-diagonal h1e indices
        tol = 1e-12
        h1_offdiag_mask = (torch.abs(self.h1e) > tol) & ~torch.eye(n_orb, device=device, dtype=torch.bool)
        self.h1_offdiag_indices = torch.nonzero(h1_offdiag_mask)
        self.h1_offdiag_values = self.h1e[h1_offdiag_mask]

    def _precompute_single_excitation_data(self):
        """Precompute data for fast single excitation enumeration."""
        self.single_exc_data = []
        for idx in range(len(self.h1_offdiag_indices)):
            p, q = self.h1_offdiag_indices[idx]
            h_pq = self.h1_offdiag_values[idx]
            self.single_exc_data.append((p.item(), q.item(), h_pq.item()))

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

    @torch.no_grad()
    def diagonal_elements_batch(self, configs: torch.Tensor) -> torch.Tensor:
        """
        Fully vectorized diagonal energy computation for a batch.

        E_diag = E_nuc + sum_p h_pp * n_p + 0.5 * sum_{p!=q} (J_pq - K_pq*delta_s) * n_p * n_q

        Args:
            configs: (batch_size, num_sites) occupation numbers

        Returns:
            (batch_size,) diagonal energies
        """
        configs = configs.to(self.device).float()
        batch_size = configs.shape[0]
        n_orb = self.n_orbitals

        # Split into alpha and beta
        n_alpha = configs[:, :n_orb]  # (batch, n_orb)
        n_beta = configs[:, n_orb:]   # (batch, n_orb)

        # Nuclear repulsion
        energies = torch.full((batch_size,), self.nuclear_repulsion,
                             device=self.device, dtype=torch.float32)

        # One-body: sum_p h_pp * (n_p^alpha + n_p^beta)
        energies += (n_alpha + n_beta) @ self.h1_diag

        # Two-body Coulomb (J)
        # alpha-alpha: 0.5 * sum_{p!=q} J_pq * n_p^a * n_q^a
        J_aa = 0.5 * (torch.einsum('bp,pq,bq->b', n_alpha, self.J_tensor, n_alpha)
                      - torch.sum(n_alpha * torch.diag(self.J_tensor), dim=1))

        # beta-beta
        J_bb = 0.5 * (torch.einsum('bp,pq,bq->b', n_beta, self.J_tensor, n_beta)
                      - torch.sum(n_beta * torch.diag(self.J_tensor), dim=1))

        # alpha-beta (no self-exclusion needed)
        J_ab = torch.einsum('bp,pq,bq->b', n_alpha, self.J_tensor, n_beta)

        energies += J_aa + J_bb + J_ab

        # Two-body Exchange (K): same spin only
        K_aa = -0.5 * (torch.einsum('bp,pq,bq->b', n_alpha, self.K_tensor, n_alpha)
                       - torch.sum(n_alpha * torch.diag(self.K_tensor), dim=1))

        K_bb = -0.5 * (torch.einsum('bp,pq,bq->b', n_beta, self.K_tensor, n_beta)
                       - torch.sum(n_beta * torch.diag(self.K_tensor), dim=1))

        energies += K_aa + K_bb

        return energies

    def diagonal_element(self, config: torch.Tensor) -> torch.Tensor:
        """
        Compute diagonal element <x|H|x> for single configuration.

        Uses vectorized batch computation internally.
        """
        return self.diagonal_elements_batch(config.unsqueeze(0))[0]

    def get_connections(
        self, config: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get off-diagonal connections for a configuration.

        Off-diagonal elements arise from:
        1. Single excitations: a+_p a_q with p != q (one-body terms)
        2. Double excitations: a+_p a+_r a_s a_q (two-body terms)

        Returns only configurations with non-zero matrix elements.
        """
        device = self.device
        config = config.to(device)
        n_orb = self.n_orbitals

        connected = []
        elements = []

        # Get occupied/virtual orbitals
        occ_alpha = (config[:n_orb] == 1).nonzero(as_tuple=True)[0]
        occ_beta = (config[n_orb:] == 1).nonzero(as_tuple=True)[0]
        virt_alpha = (config[:n_orb] == 0).nonzero(as_tuple=True)[0]
        virt_beta = (config[n_orb:] == 0).nonzero(as_tuple=True)[0]

        occ_alpha_list = occ_alpha.tolist()
        occ_beta_list = occ_beta.tolist()
        virt_alpha_list = virt_alpha.tolist()
        virt_beta_list = virt_beta.tolist()

        # ===== SINGLE EXCITATIONS (one-body terms) =====
        for p, q, h_pq in self.single_exc_data:
            # Alpha: q -> p
            if q in occ_alpha_list and p in virt_alpha_list:
                new_config = config.clone()
                new_config[q] = 0
                new_config[p] = 1
                sign = self._jw_sign(config, p, q)
                connected.append(new_config)
                elements.append(sign * h_pq)

            # Beta: q -> p
            if q in occ_beta_list and p in virt_beta_list:
                new_config = config.clone()
                new_config[q + n_orb] = 0
                new_config[p + n_orb] = 1
                sign = self._jw_sign(config, p + n_orb, q + n_orb)
                connected.append(new_config)
                elements.append(sign * h_pq)

        # ===== DOUBLE EXCITATIONS (two-body terms) =====
        # Alpha-Alpha
        for i, q in enumerate(occ_alpha_list):
            for s in occ_alpha_list[i+1:]:
                for j, p in enumerate(virt_alpha_list):
                    for r in virt_alpha_list[j+1:]:
                        val = self.h2e[p, q, r, s].item() - self.h2e[p, s, r, q].item()
                        if abs(val) > 1e-12:
                            new_config = config.clone()
                            new_config[q] = 0
                            new_config[s] = 0
                            new_config[p] = 1
                            new_config[r] = 1
                            sign = self._jw_sign_double(config, p, r, q, s)
                            connected.append(new_config)
                            elements.append(sign * val)

        # Beta-Beta
        for i, q in enumerate(occ_beta_list):
            for s in occ_beta_list[i+1:]:
                for j, p in enumerate(virt_beta_list):
                    for r in virt_beta_list[j+1:]:
                        val = self.h2e[p, q, r, s].item() - self.h2e[p, s, r, q].item()
                        if abs(val) > 1e-12:
                            new_config = config.clone()
                            q_idx = q + n_orb
                            s_idx = s + n_orb
                            p_idx = p + n_orb
                            r_idx = r + n_orb
                            new_config[q_idx] = 0
                            new_config[s_idx] = 0
                            new_config[p_idx] = 1
                            new_config[r_idx] = 1
                            sign = self._jw_sign_double(config, p_idx, r_idx, q_idx, s_idx)
                            connected.append(new_config)
                            elements.append(sign * val)

        # Alpha-Beta (no exchange term)
        for q in occ_alpha_list:
            for s in occ_beta_list:
                for p in virt_alpha_list:
                    for r in virt_beta_list:
                        val = self.h2e[p, q, r, s].item()
                        if abs(val) > 1e-12:
                            new_config = config.clone()
                            s_idx = s + n_orb
                            r_idx = r + n_orb
                            new_config[q] = 0
                            new_config[s_idx] = 0
                            new_config[p] = 1
                            new_config[r_idx] = 1
                            sign = self._jw_sign_double(config, p, r_idx, q, s_idx)
                            connected.append(new_config)
                            elements.append(sign * val)

        if len(connected) == 0:
            return torch.empty(0, self.num_sites, device=device), torch.empty(0, device=device)

        return torch.stack(connected), torch.tensor(elements, device=device)

    def _jw_sign_double(
        self, config: torch.Tensor, p: int, r: int, q: int, s: int
    ) -> int:
        """
        Compute Jordan-Wigner sign for double excitation a+_p a+_r a_s a_q.
        """
        total_count = 0
        total_count += config[:p].sum().item()

        count_r = config[:r].sum().item()
        if q < r:
            count_r -= config[q].item()
        total_count += count_r

        count_s = config[:s].sum().item()
        if p < s:
            count_s += 1
        if r < s:
            count_s += 1
        if q < s:
            count_s -= config[q].item()
        total_count += count_s

        count_q = config[:q].sum().item()
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
        Compute Jordan-Wigner sign for a+_p a_q.

        Sign = (-1)^(number of occupied sites between p and q)
        """
        if p == q:
            return 1
        low, high = min(p, q), max(p, q)
        count = config[low + 1:high].sum().item()
        return (-1) ** int(count)

    @torch.no_grad()
    def matrix_elements_fast(
        self,
        configs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fast Hamiltonian matrix construction.

        Uses vectorized diagonal and hash-based off-diagonal lookup.

        Args:
            configs: (n_configs, num_sites) basis configurations

        Returns:
            (n_configs, n_configs) Hamiltonian matrix
        """
        configs = configs.to(self.device)
        n_configs = configs.shape[0]

        H = torch.zeros(n_configs, n_configs, device=self.device)

        # Vectorized diagonal
        H.diagonal().copy_(self.diagonal_elements_batch(configs))

        # Build hash table for O(1) config lookup
        config_hash = {}
        for i in range(n_configs):
            key = tuple(configs[i].cpu().tolist())
            config_hash[key] = i

        # Off-diagonal elements
        for j in range(n_configs):
            connected, elements = self.get_connections(configs[j])
            if len(connected) > 0:
                for k in range(len(connected)):
                    key = tuple(connected[k].cpu().tolist())
                    if key in config_hash:
                        i = config_hash[key]
                        H[i, j] = elements[k]

        return H

    def matrix_elements(
        self,
        configs_bra: torch.Tensor,
        configs_ket: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute matrix of elements H_ij = <x_i|H|x_j>.

        Uses fast path when bra == ket.
        """
        # Fast path for same bra/ket
        if (configs_bra.shape == configs_ket.shape and
            torch.all(configs_bra == configs_ket)):
            return self.matrix_elements_fast(configs_bra)

        # General case
        configs_bra = configs_bra.to(self.device)
        configs_ket = configs_ket.to(self.device)
        n_bra = configs_bra.shape[0]
        n_ket = configs_ket.shape[0]

        H = torch.zeros(n_bra, n_ket, device=self.device)

        # Build bra hash
        bra_hash = {tuple(configs_bra[i].cpu().tolist()): i
                    for i in range(n_bra)}

        for j in range(n_ket):
            config_j = configs_ket[j]
            key_j = tuple(config_j.cpu().tolist())

            # Diagonal
            if key_j in bra_hash:
                i = bra_hash[key_j]
                H[i, j] = self.diagonal_elements_batch(config_j.unsqueeze(0))[0]

            # Off-diagonal
            connected, elements = self.get_connections(config_j)
            if len(connected) > 0:
                for k in range(len(connected)):
                    key = tuple(connected[k].cpu().tolist())
                    if key in bra_hash:
                        i = bra_hash[key]
                        H[i, j] = elements[k]

        return H

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

        # One-body terms
        for p in range(self.n_orbitals):
            for q in range(self.n_orbitals):
                h_pq = self.h1e[p, q].item()
                if abs(h_pq) < 1e-12:
                    continue

                for spin_offset in [0, self.n_orbitals]:
                    p_qubit = p + spin_offset
                    q_qubit = q + spin_offset

                    if p_qubit == q_qubit:
                        coefficients.append(h_pq / 2)
                        pauli_words.append("I" * n_qubits)

                        pauli = ["I"] * n_qubits
                        pauli[p_qubit] = "Z"
                        coefficients.append(-h_pq / 2)
                        pauli_words.append("".join(pauli))
                    else:
                        low, high = min(p_qubit, q_qubit), max(p_qubit, q_qubit)

                        pauli = ["I"] * n_qubits
                        pauli[p_qubit] = "X"
                        pauli[q_qubit] = "X"
                        for k in range(low + 1, high):
                            pauli[k] = "Z"
                        coefficients.append(h_pq / 2)
                        pauli_words.append("".join(pauli))

                        pauli = ["I"] * n_qubits
                        pauli[p_qubit] = "Y"
                        pauli[q_qubit] = "Y"
                        for k in range(low + 1, high):
                            pauli[k] = "Z"
                        coefficients.append(h_pq / 2)
                        pauli_words.append("".join(pauli))

        # Two-body terms (diagonal contributions)
        for p in range(self.n_orbitals):
            for q in range(self.n_orbitals):
                h_pqpq = self.h2e[p, p, q, q].item()
                if abs(h_pqpq) > 1e-12 and p != q:
                    for spin_offset in [0, self.n_orbitals]:
                        pauli = ["I"] * n_qubits
                        pauli[p + spin_offset] = "Z"
                        pauli[q + spin_offset] = "Z"
                        coefficients.append(h_pqpq / 8)
                        pauli_words.append("".join(pauli))

                    pauli = ["I"] * n_qubits
                    pauli[p] = "Z"
                    pauli[q + self.n_orbitals] = "Z"
                    coefficients.append(h_pqpq / 4)
                    pauli_words.append("".join(pauli))

        # Consolidate terms
        consolidated = {}
        for coeff, pauli in zip(coefficients, pauli_words):
            if pauli in consolidated:
                consolidated[pauli] += coeff
            else:
                consolidated[pauli] = coeff

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

        Returns the occupation pattern corresponding to the HF determinant.
        """
        config = torch.zeros(self.num_sites, dtype=torch.long, device=self.device)

        for i in range(self.n_alpha):
            config[i] = 1

        for i in range(self.n_beta):
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


def create_h2_hamiltonian(
    bond_length: float = 0.74,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> MolecularHamiltonian:
    """Create H2 Hamiltonian at given bond length."""
    geometry = [
        ("H", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.0, bond_length)),
    ]
    integrals = compute_molecular_integrals(geometry, basis="sto-3g")
    return MolecularHamiltonian(integrals, device=device)


def create_lih_hamiltonian(
    bond_length: float = 1.6,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> MolecularHamiltonian:
    """Create LiH Hamiltonian at given bond length."""
    geometry = [
        ("Li", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.0, bond_length)),
    ]
    integrals = compute_molecular_integrals(geometry, basis="sto-3g")
    return MolecularHamiltonian(integrals, device=device)


def create_h2o_hamiltonian(
    oh_length: float = 0.96,
    angle: float = 104.5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> MolecularHamiltonian:
    """Create H2O Hamiltonian."""
    angle_rad = np.radians(angle)
    geometry = [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (oh_length, 0.0, 0.0)),
        ("H", (oh_length * np.cos(angle_rad), oh_length * np.sin(angle_rad), 0.0)),
    ]
    integrals = compute_molecular_integrals(geometry, basis="sto-3g")
    return MolecularHamiltonian(integrals, device=device)
