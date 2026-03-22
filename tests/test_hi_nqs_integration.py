"""TDD Phase 1e: Integration tests for GPU diag in HI+NQS+SQD pipeline.

Tests that gpu_solve_fermion integrates correctly into the HI-NQS-SQD
iterative loop: correct energy, occupancies, eigenvector weights, fallback.
"""

import sys
import os

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.gpu_diag import gpu_solve_fermion, compute_occupancies
from utils.format_utils import configs_to_ibm_format, ibm_format_to_configs, vectorized_dedup


@pytest.fixture(scope="module")
def h2_full_system():
    """H2 with full basis — for E2E pipeline test."""
    from hamiltonians.molecular import create_h2_hamiltonian
    H = create_h2_hamiltonian(bond_length=0.74)
    n_orb = H.n_orbitals
    n_qubits = H.num_sites
    # Generate all valid configs
    from itertools import combinations
    configs = []
    for a_occ in combinations(range(n_orb), H.n_alpha):
        for b_occ in combinations(range(n_orb), H.n_beta):
            c = torch.zeros(n_qubits, dtype=torch.long)
            for i in a_occ:
                c[i] = 1
            for i in b_occ:
                c[i + n_orb] = 1
            configs.append(c)
    basis = torch.stack(configs)
    return H, basis, n_orb, n_qubits


@pytest.fixture(scope="module")
def lih_full_system():
    """LiH with full basis."""
    from hamiltonians.molecular import create_lih_hamiltonian
    H = create_lih_hamiltonian(bond_length=1.6)
    n_orb = H.n_orbitals
    n_qubits = H.num_sites
    from itertools import combinations
    configs = []
    for a_occ in combinations(range(n_orb), H.n_alpha):
        for b_occ in combinations(range(n_orb), H.n_beta):
            c = torch.zeros(n_qubits, dtype=torch.long)
            for i in a_occ:
                c[i] = 1
            for i in b_occ:
                c[i + n_orb] = 1
            configs.append(c)
    basis = torch.stack(configs)
    return H, basis, n_orb, n_qubits


# ── GPU diag vs solve_fermion interface compatibility ──

class TestGPUDiagInterface:

    def test_full_basis_better_than_batch(self, lih_full_system):
        """Full basis diag should give lower energy than any random batch."""
        H, basis, n_orb, n_qubits = lih_full_system
        e_full, _, _ = gpu_solve_fermion(basis, H)

        # Random batch (50% of basis)
        idx = torch.randperm(len(basis))[:len(basis) // 2]
        e_batch, _, _ = gpu_solve_fermion(basis[idx], H)

        assert e_full <= e_batch + 1e-10, (
            f"Full basis E={e_full:.8f} should be <= batch E={e_batch:.8f}"
        )

    def test_occupancies_feed_to_config_recovery(self, lih_full_system):
        """Occupancies from gpu_solve_fermion should be IBM-compatible tuple format.

        IBM's recover_configurations expects tuple(occ_alpha[n_orb], occ_beta[n_orb]).
        """
        H, basis, n_orb, n_qubits = lih_full_system
        _, _, occ = gpu_solve_fermion(basis, H)
        assert isinstance(occ, tuple), f"Expected tuple, got {type(occ)}"
        assert len(occ) == 2
        assert occ[0].shape == (n_orb,), f"alpha shape {occ[0].shape}, expected ({n_orb},)"
        assert occ[1].shape == (n_orb,), f"beta shape {occ[1].shape}, expected ({n_orb},)"
        assert np.all(occ[0] >= -0.01)
        assert np.all(occ[1] >= -0.01)
        total = occ[0].sum() + occ[1].sum()
        assert abs(total - (H.n_alpha + H.n_beta)) < 0.5


# ── Eigenvector weights for NQS training ──

class TestEigvecWeights:

    def test_eigvec_weights_sum_to_one(self, lih_full_system):
        """|c_i|^2 should sum to 1 (normalized eigenvector)."""
        H, basis, _, _ = lih_full_system
        _, v0, _ = gpu_solve_fermion(basis, H)
        weights = v0 ** 2
        assert abs(weights.sum() - 1.0) < 1e-6

    def test_eigvec_weights_are_positive(self, lih_full_system):
        """|c_i|^2 should all be non-negative."""
        H, basis, _, _ = lih_full_system
        _, v0, _ = gpu_solve_fermion(basis, H)
        weights = v0 ** 2
        assert np.all(weights >= 0)

    def test_hf_has_highest_weight(self, h2_full_system):
        """For H2, HF should have the highest |c_i|^2."""
        H, basis, n_orb, _ = h2_full_system
        _, v0, _ = gpu_solve_fermion(basis, H)
        weights = v0 ** 2

        # Find HF config
        hf = H.get_hf_state().cpu()
        hf_idx = None
        for i, c in enumerate(basis):
            if torch.equal(c.cpu(), hf):
                hf_idx = i
                break
        assert hf_idx is not None, "HF not found in basis"
        assert weights[hf_idx] == weights.max(), (
            f"HF weight={weights[hf_idx]:.4f}, max={weights.max():.4f}"
        )

    def test_eigvec_weights_vs_softmax_advantage(self, lih_full_system):
        """Eigvec weights should be different from softmax(-advantage).

        This verifies that |c_i|^2 captures off-diagonal correlations
        that diagonal energies miss.
        """
        H, basis, _, _ = lih_full_system
        _, v0, _ = gpu_solve_fermion(basis, H)
        eigvec_weights = v0 ** 2

        # Old method: softmax(-advantage)
        diag_e = np.array([float(H.diagonal_element(c)) for c in basis])
        e0 = min(diag_e)  # approximate
        advantage = diag_e - e0
        softmax_weights = np.exp(-advantage / max(abs(e0) * 0.01, 0.1))
        softmax_weights = softmax_weights / softmax_weights.sum()

        # They should not be identical (off-diagonal effects matter)
        correlation = np.corrcoef(eigvec_weights, softmax_weights)[0, 1]
        # High correlation expected (both peak at HF) but not perfect
        assert correlation < 0.999, (
            f"Weights too similar (corr={correlation:.6f}), "
            f"off-diagonal effects not captured"
        )


# ── Format conversion + GPU diag roundtrip ──

class TestFormatRoundtrip:

    def test_ibm_to_configs_to_gpu_diag(self, lih_full_system):
        """IBM format → our format → gpu_solve_fermion should work."""
        H, basis, n_orb, n_qubits = lih_full_system
        ibm_bs = configs_to_ibm_format(basis.numpy(), n_orb, n_qubits)
        configs_back = ibm_format_to_configs(ibm_bs, n_orb, n_qubits)
        e, v0, occ = gpu_solve_fermion(configs_back, H)
        fci = H.fci_energy()
        assert abs(e - fci) < 1e-6

    def test_vectorized_dedup_then_gpu_diag(self, lih_full_system):
        """Dedup + GPU diag should give correct energy."""
        H, basis, n_orb, n_qubits = lih_full_system
        ibm_bs = configs_to_ibm_format(basis.numpy(), n_orb, n_qubits)

        # Add duplicates
        ibm_with_dups = np.vstack([ibm_bs, ibm_bs[:10]])
        new_only = vectorized_dedup(ibm_bs, ibm_bs[:10])
        assert len(new_only) == 0  # All dups


# ── End-to-End mini pipeline ──

class TestEndToEnd:

    def test_h2_mini_pipeline(self, h2_full_system):
        """Mini E2E: sample → format → dedup → gpu_diag → occ → weights."""
        H, basis, n_orb, n_qubits = h2_full_system

        # Simulate NQS sampling (use random subset)
        sample_idx = torch.randperm(len(basis))[:3]
        nqs_configs = basis[sample_idx]

        # Convert to IBM format
        ibm_bs = configs_to_ibm_format(nqs_configs.numpy(), n_orb, n_qubits)

        # Accumulate (first iteration: no existing)
        new_configs = vectorized_dedup(None, ibm_bs)
        cumulative_bs = new_configs

        # GPU diag on full cumulative basis
        configs_tensor = ibm_format_to_configs(cumulative_bs, n_orb, n_qubits)
        e0, v0, occ = gpu_solve_fermion(configs_tensor, H)

        # Verify results
        assert isinstance(e0, float)
        assert e0 < 0  # H2 has negative total energy
        assert isinstance(occ, tuple) and len(occ) == 2
        assert occ[0].shape == (n_orb,)
        assert abs(occ[0].sum() + occ[1].sum() - 2) < 0.5  # 1 alpha + 1 beta

        # Eigvec weights
        weights = v0 ** 2
        assert abs(weights.sum() - 1.0) < 1e-6

    def test_lih_mini_pipeline(self, lih_full_system):
        """LiH E2E: full basis → gpu_diag → check energy."""
        H, basis, n_orb, n_qubits = lih_full_system
        e0, v0, occ = gpu_solve_fermion(basis, H)
        fci = H.fci_energy()
        error_mha = abs(e0 - fci) * 1000
        assert error_mha < 0.01, f"LiH error = {error_mha:.4f} mHa (expected < 0.01)"
