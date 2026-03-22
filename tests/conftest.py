"""Shared fixtures for Flow-Guided-Krylov test suite."""

import sys
import os
import pytest
import numpy as np
import torch

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture(scope="session")
def h2_hamiltonian():
    """H2 molecule (4Q, 2e, 2 orbitals) — smallest molecular system."""
    from hamiltonians.molecular import create_h2_hamiltonian

    return create_h2_hamiltonian(bond_length=0.74)


@pytest.fixture(scope="session")
def lih_hamiltonian():
    """LiH molecule (12Q, 4e, 6 orbitals) — small but non-trivial."""
    from hamiltonians.molecular import create_lih_hamiltonian

    return create_lih_hamiltonian(bond_length=1.6)


@pytest.fixture(scope="session")
def h2o_hamiltonian():
    """H2O molecule (14Q, 10e, 7 orbitals)."""
    from hamiltonians.molecular import create_h2o_hamiltonian

    return create_h2o_hamiltonian()


@pytest.fixture(scope="session")
def h2_mol_info(h2_hamiltonian):
    """Molecular info dict for H2."""
    H = h2_hamiltonian
    return {
        "n_orbitals": H.n_sites // 2,
        "n_alpha": H.n_alpha if hasattr(H, "n_alpha") else 1,
        "n_beta": H.n_beta if hasattr(H, "n_beta") else 1,
        "n_qubits": H.n_sites,
        "nuclear_repulsion": H.nuclear_repulsion if hasattr(H, "nuclear_repulsion") else 0.0,
    }


@pytest.fixture(scope="session")
def lih_mol_info(lih_hamiltonian):
    """Molecular info dict for LiH."""
    H = lih_hamiltonian
    return {
        "n_orbitals": H.n_sites // 2,
        "n_alpha": H.n_alpha if hasattr(H, "n_alpha") else 2,
        "n_beta": H.n_beta if hasattr(H, "n_beta") else 2,
        "n_qubits": H.n_sites,
        "nuclear_repulsion": H.nuclear_repulsion if hasattr(H, "nuclear_repulsion") else 0.0,
    }


@pytest.fixture
def device():
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_configs_lih():
    """Sample configs for LiH (12Q = 6 alpha + 6 beta orbitals)."""
    n_orb = 6
    # HF state: alpha=[1,1,0,0,0,0], beta=[1,1,0,0,0,0]
    hf = np.zeros(2 * n_orb, dtype=np.int64)
    hf[:2] = 1  # alpha occupied
    hf[n_orb : n_orb + 2] = 1  # beta occupied

    # Single excitation: alpha 1->2
    single = hf.copy()
    single[1] = 0
    single[2] = 1

    # Double excitation: alpha 1->2, beta 1->2
    double = hf.copy()
    double[1] = 0
    double[2] = 1
    double[n_orb + 1] = 0
    double[n_orb + 2] = 1

    configs = torch.tensor(np.array([hf, single, double]), dtype=torch.long)
    return configs, n_orb
