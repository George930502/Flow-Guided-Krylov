"""Spin system Hamiltonians."""

import torch
import numpy as np
from typing import Tuple, List, Optional

try:
    from .base import Hamiltonian, PauliString
except ImportError:
    from hamiltonians.base import Hamiltonian, PauliString


class HeisenbergHamiltonian(Hamiltonian):
    """
    Heisenberg XXZ Hamiltonian.

    H = Σ_{⟨i,j⟩} [J_xy (S_i^x S_j^x + S_i^y S_j^y) + J_z S_i^z S_j^z]
        + Σ_i [h_x S_i^x + h_y S_i^y + h_z S_i^z]

    In terms of Pauli operators:
    H = Σ_{⟨i,j⟩} [J_xy/2 (σ_i^+ σ_j^- + σ_i^- σ_j^+) + J_z/4 σ_i^z σ_j^z]
        + Σ_i h · σ_i / 2

    Args:
        num_spins: Number of spins
        Jx, Jy, Jz: Exchange couplings
        h_x, h_y, h_z: Magnetic field components (can be arrays)
        periodic: Whether to use periodic boundary conditions
    """

    def __init__(
        self,
        num_spins: int,
        Jx: float = 1.0,
        Jy: float = 1.0,
        Jz: float = 1.0,
        h_x: Optional[np.ndarray] = None,
        h_y: Optional[np.ndarray] = None,
        h_z: Optional[np.ndarray] = None,
        periodic: bool = False,
    ):
        super().__init__(num_spins, local_dim=2)

        self.Jx = Jx
        self.Jy = Jy
        self.Jz = Jz
        self.periodic = periodic

        # Default field values
        self.h_x = h_x if h_x is not None else np.zeros(num_spins)
        self.h_y = h_y if h_y is not None else np.zeros(num_spins)
        self.h_z = h_z if h_z is not None else np.zeros(num_spins)

        # Build list of bonds
        self.bonds = []
        for i in range(num_spins - 1):
            self.bonds.append((i, i + 1))
        if periodic and num_spins > 2:
            self.bonds.append((num_spins - 1, 0))

    def diagonal_element(self, config: torch.Tensor) -> torch.Tensor:
        """
        Compute ⟨x|H|x⟩.

        Diagonal contributions:
        - J_z S_i^z S_j^z = J_z/4 * (2*s_i - 1)(2*s_j - 1)
        - h_z S_i^z = h_z/2 * (2*s_i - 1)
        """
        device = config.device
        spins = 2 * config.float() - 1  # Map {0,1} → {-1,+1}

        # ZZ interaction
        diag = 0.0
        for i, j in self.bonds:
            diag += self.Jz / 4.0 * spins[i] * spins[j]

        # Z field
        for i in range(self.num_sites):
            diag += self.h_z[i] / 2.0 * spins[i]

        return torch.tensor(diag, device=device)

    def diagonal_elements_batch(self, configs: torch.Tensor) -> torch.Tensor:
        """
        Vectorized diagonal energy computation for a batch.

        Args:
            configs: (batch_size, num_sites) occupation numbers

        Returns:
            (batch_size,) diagonal energies
        """
        device = configs.device
        spins = 2 * configs.float() - 1  # Map {0,1} → {-1,+1}

        # ZZ interaction: sum over bonds
        diag = torch.zeros(configs.shape[0], device=device)
        for i, j in self.bonds:
            diag += self.Jz / 4.0 * spins[:, i] * spins[:, j]

        # Z field
        h_z_tensor = torch.tensor(self.h_z, device=device, dtype=torch.float32)
        diag += (spins * h_z_tensor / 2.0).sum(dim=1)

        return diag

    def matrix_elements(
        self, configs_bra: torch.Tensor, configs_ket: torch.Tensor
    ) -> torch.Tensor:
        """Compute Hamiltonian matrix elements between configurations."""
        device = configs_bra.device
        n_bra = configs_bra.shape[0]
        n_ket = configs_ket.shape[0]

        H = torch.zeros(n_bra, n_ket, device=device)

        # Build config hash for fast lookup
        config_to_bra = {tuple(configs_bra[i].cpu().tolist()): i for i in range(n_bra)}

        for j in range(n_ket):
            config_j = configs_ket[j]
            key_j = tuple(config_j.cpu().tolist())

            # Diagonal
            if key_j in config_to_bra:
                i = config_to_bra[key_j]
                H[i, j] = self.diagonal_element(config_j)

            # Off-diagonal
            connected, elements = self.get_connections(config_j)
            if len(connected) > 0:
                for k in range(len(connected)):
                    key = tuple(connected[k].cpu().tolist())
                    if key in config_to_bra:
                        i = config_to_bra[key]
                        H[i, j] = elements[k]

        return H

    def get_connections(
        self, config: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get off-diagonal connections.

        Off-diagonal contributions:
        - XX: flips both spins (only if they're different)
        - YY: flips both spins with phase
        - X field: single spin flip
        - Y field: single spin flip with phase
        """
        device = config.device
        connected = []
        elements = []

        spins = 2 * config.float() - 1

        # Exchange interactions (XX + YY)
        for i, j in self.bonds:
            # XX + YY = 2(S+S- + S-S+) only connects if spins are antiparallel
            if config[i] != config[j]:
                new_config = config.clone()
                new_config[i] = 1 - new_config[i]
                new_config[j] = 1 - new_config[j]
                connected.append(new_config)
                # Coefficient: (Jx + Jy)/4 for the flip
                elements.append((self.Jx + self.Jy) / 4.0)

        # X field (single spin flips)
        for i in range(self.num_sites):
            if abs(self.h_x[i]) > 1e-10:
                new_config = config.clone()
                new_config[i] = 1 - new_config[i]
                connected.append(new_config)
                elements.append(self.h_x[i] / 2.0)

        if len(connected) == 0:
            return torch.tensor([], device=device), torch.tensor([], device=device)

        return torch.stack(connected), torch.tensor(elements, device=device)


class TransverseFieldIsing(Hamiltonian):
    """
    Transverse Field Ising Model.

    H = -V Σ_{⟨i,j⟩∈E} σ_i^z σ_j^z - Σ_i σ_i^x

    With optional long-range interactions up to distance L.

    Args:
        num_spins: Number of spins
        V: Interaction strength
        h: Transverse field strength
        L: Interaction range (1 = nearest neighbor)
        periodic: Use periodic boundary conditions
    """

    def __init__(
        self,
        num_spins: int,
        V: float = 1.0,
        h: float = 1.0,
        L: int = 1,
        periodic: bool = True,
    ):
        super().__init__(num_spins, local_dim=2)

        self.V = V
        self.h = h
        self.L = L
        self.periodic = periodic

        # Build edge list for interactions up to distance L
        self.edges = []
        for i in range(num_spins):
            for d in range(1, L + 1):
                j = (i + d) % num_spins if periodic else i + d
                if j < num_spins and (i, j) not in self.edges and (j, i) not in self.edges:
                    self.edges.append((i, j))

    def diagonal_element(self, config: torch.Tensor) -> torch.Tensor:
        """
        Compute diagonal ⟨x|H|x⟩ = -V Σ σ_i^z σ_j^z.
        """
        device = config.device
        spins = 2 * config.float() - 1  # {0,1} → {-1,+1}

        diag = torch.tensor(0.0, device=device)
        for i, j in self.edges:
            diag = diag - self.V * spins[i] * spins[j]

        return diag

    def diagonal_elements_batch(self, configs: torch.Tensor) -> torch.Tensor:
        """
        Vectorized diagonal energy computation for a batch.

        Args:
            configs: (batch_size, num_sites) occupation numbers

        Returns:
            (batch_size,) diagonal energies
        """
        device = configs.device
        spins = 2 * configs.float() - 1  # {0,1} → {-1,+1}

        diag = torch.zeros(configs.shape[0], device=device)
        for i, j in self.edges:
            diag -= self.V * spins[:, i] * spins[:, j]

        return diag

    def matrix_elements(
        self, configs_bra: torch.Tensor, configs_ket: torch.Tensor
    ) -> torch.Tensor:
        """Compute Hamiltonian matrix elements between configurations."""
        device = configs_bra.device
        n_bra = configs_bra.shape[0]
        n_ket = configs_ket.shape[0]

        H = torch.zeros(n_bra, n_ket, device=device)

        # Build config hash for fast lookup
        config_to_bra = {tuple(configs_bra[i].cpu().tolist()): i for i in range(n_bra)}

        for j in range(n_ket):
            config_j = configs_ket[j]
            key_j = tuple(config_j.cpu().tolist())

            # Diagonal
            if key_j in config_to_bra:
                i = config_to_bra[key_j]
                H[i, j] = self.diagonal_element(config_j)

            # Off-diagonal
            connected, elements = self.get_connections(config_j)
            if len(connected) > 0:
                for k in range(len(connected)):
                    key = tuple(connected[k].cpu().tolist())
                    if key in config_to_bra:
                        i = config_to_bra[key]
                        H[i, j] = elements[k]

        return H

    def get_connections(
        self, config: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get off-diagonal connections from transverse field.

        -σ_i^x flips spin i with coefficient -1.
        """
        device = config.device

        if abs(self.h) < 1e-10:
            return torch.tensor([], device=device), torch.tensor([], device=device)

        connected = []
        elements = []

        for i in range(self.num_sites):
            new_config = config.clone()
            new_config[i] = 1 - new_config[i]
            connected.append(new_config)
            elements.append(-self.h)

        return torch.stack(connected), torch.tensor(elements, device=device)


def create_heisenberg_hamiltonian(
    num_spins: int,
    Jx: float,
    Jy: float,
    Jz: float,
    h_x: np.ndarray,
    h_y: np.ndarray,
    h_z: np.ndarray,
) -> HeisenbergHamiltonian:
    """
    Factory function for Heisenberg Hamiltonian.

    Compatible with the CUDA-Q example interface.

    Args:
        num_spins: Number of spins
        Jx, Jy, Jz: Exchange couplings
        h_x, h_y, h_z: Field arrays

    Returns:
        HeisenbergHamiltonian instance
    """
    return HeisenbergHamiltonian(
        num_spins=num_spins,
        Jx=Jx,
        Jy=Jy,
        Jz=Jz,
        h_x=h_x,
        h_y=h_y,
        h_z=h_z,
        periodic=False,
    )


def extract_coeffs_and_paulis(
    hamiltonian: Hamiltonian,
) -> Tuple[List[float], List[str]]:
    """
    Extract Pauli coefficients and strings from Hamiltonian.

    For CUDA-Q integration.

    Returns:
        (coefficients, pauli_words)
    """
    if isinstance(hamiltonian, HeisenbergHamiltonian):
        coeffs = []
        paulis = []

        # ZZ interactions
        for i, j in hamiltonian.bonds:
            if abs(hamiltonian.Jz) > 1e-10:
                pauli = ["I"] * hamiltonian.num_sites
                pauli[i] = "Z"
                pauli[j] = "Z"
                coeffs.append(hamiltonian.Jz / 4.0)
                paulis.append("".join(pauli))

        # XX + YY = (X⊗X + Y⊗Y) interactions
        for i, j in hamiltonian.bonds:
            Jxy = (hamiltonian.Jx + hamiltonian.Jy) / 2.0
            if abs(Jxy) > 1e-10:
                # XX term
                pauli = ["I"] * hamiltonian.num_sites
                pauli[i] = "X"
                pauli[j] = "X"
                coeffs.append(hamiltonian.Jx / 4.0)
                paulis.append("".join(pauli))

                # YY term
                pauli = ["I"] * hamiltonian.num_sites
                pauli[i] = "Y"
                pauli[j] = "Y"
                coeffs.append(hamiltonian.Jy / 4.0)
                paulis.append("".join(pauli))

        # Single-site fields
        for i in range(hamiltonian.num_sites):
            if abs(hamiltonian.h_x[i]) > 1e-10:
                pauli = ["I"] * hamiltonian.num_sites
                pauli[i] = "X"
                coeffs.append(hamiltonian.h_x[i] / 2.0)
                paulis.append("".join(pauli))

            if abs(hamiltonian.h_y[i]) > 1e-10:
                pauli = ["I"] * hamiltonian.num_sites
                pauli[i] = "Y"
                coeffs.append(hamiltonian.h_y[i] / 2.0)
                paulis.append("".join(pauli))

            if abs(hamiltonian.h_z[i]) > 1e-10:
                pauli = ["I"] * hamiltonian.num_sites
                pauli[i] = "Z"
                coeffs.append(hamiltonian.h_z[i] / 2.0)
                paulis.append("".join(pauli))

        return coeffs, paulis

    elif isinstance(hamiltonian, TransverseFieldIsing):
        coeffs = []
        paulis = []

        # ZZ interactions
        for i, j in hamiltonian.edges:
            pauli = ["I"] * hamiltonian.num_sites
            pauli[i] = "Z"
            pauli[j] = "Z"
            coeffs.append(-hamiltonian.V)
            paulis.append("".join(pauli))

        # X field
        for i in range(hamiltonian.num_sites):
            pauli = ["I"] * hamiltonian.num_sites
            pauli[i] = "X"
            coeffs.append(-hamiltonian.h)
            paulis.append("".join(pauli))

        return coeffs, paulis

    else:
        raise NotImplementedError(
            f"Pauli extraction not implemented for {type(hamiltonian)}"
        )
