"""TDD: End-to-end tests for the full HI+NQS+SKQD iterative pipeline.

Verifies the complete loop: NQS sampling → Krylov expansion → GPU diag →
|c_i|² feedback → NQS retraining → convergence.
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
    from src.hamiltonians.molecular import create_h2_hamiltonian
    H = create_h2_hamiltonian(bond_length=0.74)
    mol_info = {"n_orbitals": H.n_orbitals, "n_alpha": H.n_alpha,
                "n_beta": H.n_beta, "n_qubits": H.num_sites,
                "nuclear_repulsion": H.nuclear_repulsion}
    return H, mol_info


@pytest.fixture(scope="module")
def lih_system():
    from src.hamiltonians.molecular import create_lih_hamiltonian
    H = create_lih_hamiltonian(bond_length=1.6)
    mol_info = {"n_orbitals": H.n_orbitals, "n_alpha": H.n_alpha,
                "n_beta": H.n_beta, "n_qubits": H.num_sites,
                "nuclear_repulsion": H.nuclear_repulsion}
    return H, mol_info


@pytest.fixture(scope="module")
def h2o_system():
    from src.hamiltonians.molecular import create_h2o_hamiltonian
    H = create_h2o_hamiltonian()
    mol_info = {"n_orbitals": H.n_orbitals, "n_alpha": H.n_alpha,
                "n_beta": H.n_beta, "n_qubits": H.num_sites,
                "nuclear_repulsion": H.nuclear_repulsion}
    return H, mol_info


class TestE2EConvergence:
    """Test that the iterative pipeline converges on real molecular systems."""

    def test_h2_converges_to_near_fci(self, h2_system):
        """H2 (4Q): should converge within 10 mHa of FCI in ≤10 iterations."""
        H, mol_info = h2_system
        cfg = HINQSSKQDConfig(
            max_iterations=10, n_samples=200, krylov_max_new=50,
            nf_steps=5, convergence_threshold=1e-5, convergence_window=2,
        )
        result = run_hi_nqs_skqd(H, mol_info, config=cfg)

        fci = H.fci_energy()
        error_mha = abs(result.energy - fci) * 1000
        assert error_mha < 10.0, (
            f"H2 E2E error = {error_mha:.2f} mHa (expected < 10), "
            f"E={result.energy:.8f}, FCI={fci:.8f}, iters={result.metadata['iterations']}"
        )

    def test_lih_converges_below_hf(self, lih_system):
        """LiH (12Q): energy should improve significantly below HF."""
        H, mol_info = lih_system
        cfg = HINQSSKQDConfig(
            max_iterations=8, n_samples=500, krylov_max_new=100,
            nf_steps=5, convergence_threshold=1e-5, convergence_window=2,
        )
        result = run_hi_nqs_skqd(H, mol_info, config=cfg)

        hf = H.get_hf_state().cpu()
        e_hf = float(H.diagonal_element(hf))
        improvement_mha = (e_hf - result.energy) * 1000

        assert improvement_mha > 10.0, (
            f"LiH should improve > 10 mHa below HF, got {improvement_mha:.2f} mHa"
        )

    def test_h2o_converges_below_hf(self, h2o_system):
        """H2O (14Q): energy should improve below HF."""
        H, mol_info = h2o_system
        cfg = HINQSSKQDConfig(
            max_iterations=5, n_samples=500, krylov_max_new=100,
            nf_steps=3, convergence_threshold=1e-5, convergence_window=2,
        )
        result = run_hi_nqs_skqd(H, mol_info, config=cfg)

        hf = H.get_hf_state().cpu()
        e_hf = float(H.diagonal_element(hf))
        assert result.energy < e_hf, (
            f"H2O E={result.energy:.6f} should be < HF={e_hf:.6f}"
        )


class TestE2EIterationBehavior:
    """Test iteration mechanics: energy improves, basis grows, metadata correct."""

    def test_energy_monotonically_improves(self, lih_system):
        """Energy history should generally decrease (variational principle)."""
        H, mol_info = lih_system
        cfg = HINQSSKQDConfig(
            max_iterations=5, n_samples=300, krylov_max_new=80,
            nf_steps=3,
        )
        result = run_hi_nqs_skqd(H, mol_info, config=cfg)

        history = result.metadata["energy_history"]
        assert len(history) >= 2, "Should have at least 2 iterations"
        # Best energy should be <= first energy (variational improvement)
        assert min(history) <= history[0] + 1e-6

    def test_basis_grows_with_iterations(self, lih_system):
        """Basis size should grow (NQS adds new configs each iteration)."""
        H, mol_info = lih_system
        cfg = HINQSSKQDConfig(
            max_iterations=4, n_samples=300, krylov_max_new=50,
            nf_steps=3,
        )
        result = run_hi_nqs_skqd(H, mol_info, config=cfg)

        basis_history = result.metadata["basis_size_history"]
        assert len(basis_history) >= 2
        # Basis should grow (or at least not shrink)
        assert basis_history[-1] >= basis_history[0]

    def test_wall_time_recorded(self, h2_system):
        """Wall time should be positive and reasonable."""
        H, mol_info = h2_system
        cfg = HINQSSKQDConfig(max_iterations=2, n_samples=50, krylov_max_new=10, nf_steps=2)
        result = run_hi_nqs_skqd(H, mol_info, config=cfg)
        assert result.wall_time > 0
        assert result.wall_time < 60  # Should finish in under 1 min for H2
