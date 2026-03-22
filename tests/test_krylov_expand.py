"""TDD Phase 2a: Tests for Hamiltonian connection expansion.

Tests expand_basis_via_connections which discovers new configurations
by computing Hamiltonian connections (singles/doubles) from existing basis.
"""

import sys
import os
from itertools import combinations

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.krylov_expand import expand_basis_via_connections


@pytest.fixture(scope="module")
def h2_system():
    from hamiltonians.molecular import create_h2_hamiltonian
    H = create_h2_hamiltonian(bond_length=0.74)
    return H


@pytest.fixture(scope="module")
def lih_system():
    from hamiltonians.molecular import create_lih_hamiltonian
    H = create_lih_hamiltonian(bond_length=1.6)
    return H


class TestExpandBasis:

    def test_h2_hf_expands(self, h2_system):
        """HF seed should expand to connected configs via H connections."""
        H = h2_system
        hf = H.get_hf_state().unsqueeze(0)
        expanded = expand_basis_via_connections(hf, H)
        # H2 has 4 total configs. HF should connect to at least 1 more.
        assert len(expanded) > len(hf), (
            f"Expanded {len(expanded)} should be > seed {len(hf)}"
        )

    def test_h2_full_expansion(self, h2_system):
        """H2 HF connects to 1 double excitation (Slater-Condon: only singles+doubles).

        H2 has 4 configs but HF only connects to [0,1,0,1] (alpha-beta double).
        [1,0,0,1] and [0,1,1,0] are in a separate Slater-Condon component.
        """
        H = h2_system
        hf = H.get_hf_state().unsqueeze(0)
        expanded = expand_basis_via_connections(hf, H, max_new=100)
        # HF + 1 double excitation = 2 connected configs
        assert len(expanded) == 2, f"H2 HF component should have 2 configs, got {len(expanded)}"

    def test_lih_hf_discovers_new(self, lih_system):
        """LiH HF should discover singles and doubles."""
        H = lih_system
        hf = H.get_hf_state().unsqueeze(0)
        expanded = expand_basis_via_connections(hf, H, max_new=50)
        assert len(expanded) > 1
        # Should find at least some single excitations
        assert len(expanded) >= 5, f"Expected >=5 configs from HF expansion, got {len(expanded)}"

    def test_expansion_respects_max_new(self, lih_system):
        """Should not add more than max_new configs."""
        H = lih_system
        hf = H.get_hf_state().unsqueeze(0)
        expanded = expand_basis_via_connections(hf, H, max_new=10)
        # expanded = seed (1) + new (<=10) = <=11
        assert len(expanded) <= 11

    def test_expansion_dedup(self, lih_system):
        """Expanded basis should have no duplicates."""
        H = lih_system
        hf = H.get_hf_state().unsqueeze(0)
        expanded = expand_basis_via_connections(hf, H, max_new=50)
        unique = torch.unique(expanded, dim=0)
        assert len(unique) == len(expanded), "Duplicates found in expansion"

    def test_expansion_preserves_particle_number(self, lih_system):
        """All expanded configs should have correct electron count."""
        H = lih_system
        hf = H.get_hf_state().unsqueeze(0)
        expanded = expand_basis_via_connections(hf, H, max_new=50)
        n_orb = H.n_orbitals
        for c in expanded:
            n_alpha = c[:n_orb].sum().item()
            n_beta = c[n_orb:].sum().item()
            assert n_alpha == H.n_alpha, f"Wrong alpha count: {n_alpha} != {H.n_alpha}"
            assert n_beta == H.n_beta, f"Wrong beta count: {n_beta} != {H.n_beta}"

    def test_expansion_from_larger_seed(self, lih_system):
        """Expansion from multiple seed configs should discover more."""
        H = lih_system
        n_orb = H.n_orbitals

        # Build seed: HF + 2 single excitations
        hf = H.get_hf_state()
        configs = [hf]

        # Single excitation: alpha orbital 1→2
        single = hf.clone()
        single[1] = 0
        single[2] = 1
        configs.append(single)

        seed = torch.stack(configs)
        expanded = expand_basis_via_connections(seed, H, max_new=30)

        # Should find more configs than single-HF expansion
        hf_only_expanded = expand_basis_via_connections(hf.unsqueeze(0), H, max_new=30)
        assert len(expanded) >= len(hf_only_expanded)

    def test_returns_tensor(self, h2_system):
        """Should return a torch.Tensor."""
        H = h2_system
        hf = H.get_hf_state().unsqueeze(0)
        result = expand_basis_via_connections(hf, H)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.long

    def test_empty_seed_raises(self, h2_system):
        """Empty seed should raise ValueError."""
        H = h2_system
        empty = torch.zeros(0, H.num_sites, dtype=torch.long)
        with pytest.raises(ValueError):
            expand_basis_via_connections(empty, H)
