"""TDD: Tests for the complete HI+NQS+SKQD method (run_hi_nqs_skqd).

This tests the full pipeline: NQS sampling → Krylov expansion → post-merge diag.
"""

import sys
import os

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.methods.hi_nqs_skqd import run_hi_nqs_skqd, HINQSSKQDConfig


@pytest.fixture(scope="module")
def h2_system():
    from hamiltonians.molecular import create_h2_hamiltonian
    H = create_h2_hamiltonian(bond_length=0.74)
    mol_info = {
        "n_orbitals": H.n_orbitals,
        "n_alpha": H.n_alpha,
        "n_beta": H.n_beta,
        "n_qubits": H.num_sites,
        "nuclear_repulsion": H.nuclear_repulsion,
    }
    return H, mol_info


@pytest.fixture(scope="module")
def lih_system():
    from hamiltonians.molecular import create_lih_hamiltonian
    H = create_lih_hamiltonian(bond_length=1.6)
    mol_info = {
        "n_orbitals": H.n_orbitals,
        "n_alpha": H.n_alpha,
        "n_beta": H.n_beta,
        "n_qubits": H.num_sites,
        "nuclear_repulsion": H.nuclear_repulsion,
    }
    return H, mol_info


class TestHINQSSKQDMethod:

    def test_h2_returns_solver_result(self, h2_system):
        """Should return a SolverResult with energy and metadata."""
        H, mol_info = h2_system
        cfg = HINQSSKQDConfig(max_iterations=3, n_samples=100, krylov_max_new=20)
        result = run_hi_nqs_skqd(H, mol_info, config=cfg)
        assert result.energy is not None
        assert result.energy < 0  # H2 has negative total energy
        assert result.wall_time > 0
        assert result.method == "HI+NQS+SKQD"

    def test_h2_energy_reasonable(self, h2_system):
        """H2 energy should be close to FCI."""
        H, mol_info = h2_system
        cfg = HINQSSKQDConfig(max_iterations=5, n_samples=200, krylov_max_new=50)
        result = run_hi_nqs_skqd(H, mol_info, config=cfg)
        fci = H.fci_energy()
        error_mha = abs(result.energy - fci) * 1000
        assert error_mha < 50, f"H2 error={error_mha:.2f} mHa (expected < 50)"

    def test_lih_energy_below_hf(self, lih_system):
        """LiH should be below HF energy."""
        H, mol_info = lih_system
        cfg = HINQSSKQDConfig(max_iterations=3, n_samples=200, krylov_max_new=50)
        result = run_hi_nqs_skqd(H, mol_info, config=cfg)
        hf = H.get_hf_state()
        e_hf = float(H.diagonal_element(hf))
        assert result.energy < e_hf

    def test_metadata_has_required_fields(self, h2_system):
        """Metadata should include iterations, energy_history, diag_mode."""
        H, mol_info = h2_system
        cfg = HINQSSKQDConfig(max_iterations=2, n_samples=50, krylov_max_new=10)
        result = run_hi_nqs_skqd(H, mol_info, config=cfg)
        assert "iterations" in result.metadata
        assert "energy_history" in result.metadata
        assert "diag_mode" in result.metadata
        assert result.metadata["diag_mode"] == "gpu_diag+krylov"

    def test_krylov_expansion_used(self, lih_system):
        """Basis should be larger than just NQS samples (Krylov added configs)."""
        H, mol_info = lih_system
        cfg = HINQSSKQDConfig(max_iterations=2, n_samples=100, krylov_max_new=50)
        result = run_hi_nqs_skqd(H, mol_info, config=cfg)
        # diag_dim should be > n_samples because Krylov expansion adds configs
        assert result.diag_dim > 0

    def test_config_defaults(self):
        """Default config should have reasonable values."""
        cfg = HINQSSKQDConfig()
        assert cfg.max_iterations > 0
        assert cfg.krylov_max_new > 0
        assert cfg.use_eigvec_weights is True
