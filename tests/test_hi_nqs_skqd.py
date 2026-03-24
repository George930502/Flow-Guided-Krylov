"""TDD Phase 2b: Tests for HI+NQS+SKQD method.

Tests the new method that combines NQS sampling with Krylov expansion
and post-merge diagonalization.
"""

import sys
import os

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.gpu_diag import gpu_solve_fermion
from utils.krylov_expand import expand_basis_via_connections
from utils.format_utils import configs_to_ibm_format, ibm_format_to_configs


@pytest.fixture(scope="module")
def lih_system():
    from hamiltonians.molecular import create_lih_hamiltonian
    H = create_lih_hamiltonian(bond_length=1.6)
    return H


@pytest.fixture(scope="module")
def h2o_system():
    from hamiltonians.molecular import create_h2o_hamiltonian
    H = create_h2o_hamiltonian()
    return H


class TestPostKrylovMerge:
    """Test the post-Krylov merge pattern: SKQD from HF + merge NQS configs."""

    def test_krylov_expansion_improves_hf_energy(self, lih_system):
        """Expanding HF via connections should give better energy than HF alone."""
        H = lih_system
        hf = H.get_hf_state().unsqueeze(0)
        e_hf, _, _ = gpu_solve_fermion(hf, H)

        expanded = expand_basis_via_connections(hf, H, max_new=50)
        e_expanded, _, _ = gpu_solve_fermion(expanded, H)

        assert e_expanded < e_hf - 1e-6, (
            f"Expanded E={e_expanded:.8f} should be < HF E={e_hf:.8f}"
        )

    def test_merge_nqs_and_krylov_better_than_either(self, lih_system):
        """Merging NQS + Krylov configs should give E <= min(E_nqs, E_krylov)."""
        H = lih_system
        n_orb = H.n_orbitals

        # Simulate NQS configs: HF + a few random valid configs
        hf = H.get_hf_state()
        nqs_configs = [hf]
        # Add some single excitations
        for p in range(H.n_alpha):
            for q in range(n_orb):
                if q not in range(H.n_alpha):
                    single = hf.clone()
                    single[p] = 0
                    single[q] = 1
                    nqs_configs.append(single)
                    if len(nqs_configs) >= 10:
                        break
            if len(nqs_configs) >= 10:
                break
        nqs_basis = torch.unique(torch.stack(nqs_configs).cpu(), dim=0)

        # Krylov expansion from HF (returns CPU tensors)
        krylov_basis = expand_basis_via_connections(hf.cpu().unsqueeze(0), H, max_new=30)

        # Merge (post-Krylov merge: union of both, ensure same device)
        merged = torch.unique(torch.cat([nqs_basis.cpu(), krylov_basis.cpu()], dim=0), dim=0)

        # Diag each
        e_nqs, _, _ = gpu_solve_fermion(nqs_basis, H)
        e_krylov, _, _ = gpu_solve_fermion(krylov_basis, H)
        e_merged, _, _ = gpu_solve_fermion(merged, H)

        # Variational principle: merged (larger subspace) <= both subsets
        assert e_merged <= e_nqs + 1e-10, f"E_merged={e_merged:.8f} > E_nqs={e_nqs:.8f}"
        assert e_merged <= e_krylov + 1e-10, f"E_merged={e_merged:.8f} > E_krylov={e_krylov:.8f}"

    def test_krylov_discovers_configs_nqs_misses(self, lih_system):
        """Krylov expansion should find configs not in a random NQS sample."""
        H = lih_system
        hf = H.get_hf_state()

        # NQS basis: just HF + one single
        single = hf.clone()
        single[0] = 0
        single[2] = 1
        nqs_basis = torch.stack([hf, single])

        # Krylov expansion finds more
        krylov_basis = expand_basis_via_connections(hf.unsqueeze(0), H, max_new=30)

        nqs_set = {c.cpu().numpy().tobytes() for c in nqs_basis}
        krylov_only = [c for c in krylov_basis if c.cpu().numpy().tobytes() not in nqs_set]

        assert len(krylov_only) > 0, "Krylov should find configs NQS doesn't have"

    def test_eigvec_from_merged_basis(self, lih_system):
        """Eigenvector from merged basis should be normalized."""
        H = lih_system
        hf = H.get_hf_state().unsqueeze(0)
        expanded = expand_basis_via_connections(hf, H, max_new=30)
        _, v0, occ = gpu_solve_fermion(expanded, H)

        assert abs(np.sum(v0 ** 2) - 1.0) < 1e-6
        assert isinstance(occ, tuple) and len(occ) == 2


class TestEnergyImprovement:
    """Test that Krylov expansion meaningfully improves energy."""

    def test_lih_expansion_reaches_near_fci(self, lih_system):
        """LiH: expansion from HF should reach close to FCI."""
        H = lih_system
        hf = H.get_hf_state().unsqueeze(0)
        expanded = expand_basis_via_connections(hf, H, max_new=200)
        e_expanded, _, _ = gpu_solve_fermion(expanded, H)
        fci = H.fci_energy()

        error_mha = abs(e_expanded - fci) * 1000
        # With ~200 configs from HF expansion, should be within 10 mHa of FCI
        assert error_mha < 10.0, (
            f"LiH expansion error = {error_mha:.2f} mHa (expected < 10 mHa), "
            f"basis={len(expanded)} configs"
        )

    def test_h2o_expansion_improves_energy(self, h2o_system):
        """H2O: expansion should improve energy vs HF alone."""
        H = h2o_system
        hf = H.get_hf_state().unsqueeze(0)
        e_hf, _, _ = gpu_solve_fermion(hf, H)

        expanded = expand_basis_via_connections(hf, H, max_new=100)
        e_expanded, _, _ = gpu_solve_fermion(expanded, H)

        improvement = (e_hf - e_expanded) * 1000
        assert improvement > 1.0, (
            f"H2O expansion should improve by > 1 mHa, got {improvement:.2f} mHa"
        )
