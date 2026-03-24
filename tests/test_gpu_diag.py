"""TDD Phase 1d: Tests for GPU diag adapter (solve_fermion replacement).

Tests gpu_solve_fermion which replaces IBM's solve_fermion with
sparse eigsh / torch.linalg.eigh, computing energy + eigenvector + occupancies.
"""

import sys
import os

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.gpu_diag import gpu_solve_fermion, compute_occupancies, compute_occupancies_flat


# ── Fixtures ──

@pytest.fixture(scope="module")
def h2_system():
    from hamiltonians.molecular import create_h2_hamiltonian
    H = create_h2_hamiltonian(bond_length=0.74)
    basis = []
    for a in range(2):
        for b in range(2):
            c = torch.zeros(4, dtype=torch.long)
            c[a] = 1
            c[2 + b] = 1
            basis.append(c)
    basis = torch.stack(basis)
    return H, basis


@pytest.fixture(scope="module")
def lih_system():
    from hamiltonians.molecular import create_lih_hamiltonian
    H = create_lih_hamiltonian(bond_length=1.6)
    from krylov.skqd import SampleBasedKrylovDiagonalization
    skqd = SampleBasedKrylovDiagonalization(H)
    return H, skqd._subspace_basis


# ── Core Functionality ──

class TestGPUSolveFermion:

    def test_h2_energy_matches_exact(self, h2_system):
        H, basis = h2_system
        energy, v0, occ = gpu_solve_fermion(basis, H)
        fci = H.fci_energy()
        assert abs(energy - fci) < 1e-6, f"E={energy:.8f}, FCI={fci:.8f}"

    def test_lih_energy_below_hf(self, lih_system):
        H, basis = lih_system
        energy, v0, occ = gpu_solve_fermion(basis, H)
        hf = H.get_hf_state()
        e_hf = float(H.diagonal_element(hf))
        assert energy < e_hf

    def test_lih_energy_matches_eigh(self, lih_system):
        H, basis = lih_system
        energy_gpu, _, _ = gpu_solve_fermion(basis, H)
        H_proj = H.matrix_elements(basis, basis)
        H_np = H_proj.detach().cpu().numpy().real.astype(np.float64)
        H_np = 0.5 * (H_np + H_np.T)
        eigenvalues = np.linalg.eigh(H_np)[0]
        assert abs(energy_gpu - eigenvalues[0]) < 1e-8

    def test_returns_correct_types(self, h2_system):
        H, basis = h2_system
        energy, v0, occ = gpu_solve_fermion(basis, H)
        assert isinstance(energy, float)
        assert isinstance(v0, np.ndarray)
        assert isinstance(occ, tuple)
        assert len(occ) == 2
        assert isinstance(occ[0], np.ndarray)  # occ_alpha
        assert isinstance(occ[1], np.ndarray)  # occ_beta

    def test_eigenvector_normalized(self, lih_system):
        H, basis = lih_system
        _, v0, _ = gpu_solve_fermion(basis, H)
        assert abs(np.sum(v0 ** 2) - 1.0) < 1e-6

    def test_eigenvector_correct_length(self, lih_system):
        H, basis = lih_system
        _, v0, _ = gpu_solve_fermion(basis, H)
        assert len(v0) == len(basis)

    def test_max_dense_guard(self, lih_system):
        H, basis = lih_system
        energy, v0, occ = gpu_solve_fermion(basis, H, max_dense=10)
        fci = H.fci_energy()
        assert abs(energy - fci) < 1e-4

    def test_single_config_basis(self, h2_system):
        H, basis = h2_system
        single = basis[:1]
        energy, v0, occ = gpu_solve_fermion(single, H)
        e_diag = float(H.diagonal_element(single[0]))
        assert abs(energy - e_diag) < 1e-8


# ── Orbital Occupancies (IBM-compatible tuple format) ──

class TestOrbitalOccupancies:

    def test_occupancies_tuple_format(self, lih_system):
        """Occupancies should be tuple(occ_alpha, occ_beta) matching IBM format."""
        H, basis = lih_system
        _, _, occ = gpu_solve_fermion(basis, H)
        assert isinstance(occ, tuple)
        assert len(occ) == 2
        assert occ[0].shape == (H.n_orbitals,)
        assert occ[1].shape == (H.n_orbitals,)

    def test_occupancies_sum_to_nelec(self, lih_system):
        """Sum of all occupancies should equal n_alpha + n_beta."""
        H, basis = lih_system
        _, _, occ = gpu_solve_fermion(basis, H)
        total = occ[0].sum() + occ[1].sum()
        expected = H.n_alpha + H.n_beta
        assert abs(total - expected) < 0.1, f"sum(occ)={total:.3f}, expected={expected}"

    def test_alpha_occupancies_sum_to_nalpha(self, lih_system):
        """Alpha occupancies should sum to n_alpha."""
        H, basis = lih_system
        _, _, occ = gpu_solve_fermion(basis, H)
        assert abs(occ[0].sum() - H.n_alpha) < 0.1

    def test_beta_occupancies_sum_to_nbeta(self, lih_system):
        """Beta occupancies should sum to n_beta."""
        H, basis = lih_system
        _, _, occ = gpu_solve_fermion(basis, H)
        assert abs(occ[1].sum() - H.n_beta) < 0.1

    def test_occupancies_between_0_and_1(self, lih_system):
        H, basis = lih_system
        _, _, occ = gpu_solve_fermion(basis, H)
        for spin_occ in occ:
            assert np.all(spin_occ >= -0.01)
            assert np.all(spin_occ <= 1.01)

    def test_hf_dominant_occupancies(self, h2_system):
        H, basis = h2_system
        _, _, occ = gpu_solve_fermion(basis, H)
        assert occ[0][0] > 0.5, f"alpha occ[0]={occ[0][0]:.3f}, expected > 0.5"
        assert occ[1][0] > 0.5, f"beta occ[0]={occ[1][0]:.3f}, expected > 0.5"

    def test_compute_occupancies_standalone(self):
        """Test compute_occupancies returns IBM-compatible tuple."""
        configs = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ], dtype=np.float64)
        v0 = np.array([0.8, 0.6])
        occ = compute_occupancies(configs, v0, n_orb=2)
        # alpha: occ[0] = 0.64*1 + 0.36*0 = 0.64, occ[1] = 0.64*0 + 0.36*1 = 0.36
        # beta:  occ[0] = 0.64*1 + 0.36*0 = 0.64, occ[1] = 0.64*0 + 0.36*1 = 0.36
        assert isinstance(occ, tuple)
        np.testing.assert_allclose(occ[0], [0.64, 0.36], atol=1e-10)
        np.testing.assert_allclose(occ[1], [0.64, 0.36], atol=1e-10)

    def test_compute_occupancies_flat(self):
        """Test compute_occupancies_flat returns flat array."""
        configs = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.float64)
        v0 = np.array([0.8, 0.6])
        occ = compute_occupancies_flat(configs, v0)
        assert occ.shape == (4,)
        np.testing.assert_allclose(occ, [0.64, 0.36, 0.64, 0.36], atol=1e-10)

    def test_occupancies_normalized_sum(self):
        """With normalized eigvec, occ sum = n_elec."""
        configs = np.array([
            [1, 1, 0, 1, 1, 0],
            [1, 0, 1, 1, 0, 1],
        ], dtype=np.float64)
        v0 = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2)])
        occ = compute_occupancies(configs, v0, n_orb=3)
        total = occ[0].sum() + occ[1].sum()
        assert abs(total - 4.0) < 1e-10
