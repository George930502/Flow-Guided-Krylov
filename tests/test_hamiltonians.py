"""Tests for Hamiltonian implementations."""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hamiltonians.spin import TransverseFieldIsing, HeisenbergHamiltonian
from hamiltonians.base import Hamiltonian


class TestTransverseFieldIsing:
    """Test cases for Transverse Field Ising model."""

    def test_construction(self):
        """Test basic construction."""
        H = TransverseFieldIsing(num_spins=4, V=1.0, h=1.0)
        assert H.num_sites == 4
        assert H.V == 1.0
        assert H.h == 1.0

    def test_diagonal_element(self):
        """Test diagonal matrix elements."""
        H = TransverseFieldIsing(num_spins=4, V=1.0, h=0.0, L=1, periodic=False)

        # All spins aligned: should give negative energy
        config_aligned = torch.tensor([0, 0, 0, 0])
        diag = H.diagonal_element(config_aligned)

        # -V * sum(sigma_i * sigma_j) for aligned spins
        # sigma_i = sigma_j = -1 for all zeros
        # E = -V * 3 * (-1)(-1) = -3
        assert abs(diag.item() + 3.0) < 1e-10

    def test_connections(self):
        """Test off-diagonal connections."""
        H = TransverseFieldIsing(num_spins=4, V=1.0, h=1.0)

        config = torch.tensor([0, 0, 0, 0])
        connected, elements = H.get_connections(config)

        # Should have 4 connections (one per spin flip)
        assert len(connected) == 4
        assert len(elements) == 4

        # All elements should be -h = -1
        for elem in elements:
            assert abs(elem.item() + 1.0) < 1e-10

    def test_dense_matrix(self):
        """Test dense matrix construction for small system."""
        H = TransverseFieldIsing(num_spins=4, V=1.0, h=1.0, periodic=False)
        H_dense = H.to_dense()

        # Should be 2^4 x 2^4 = 16 x 16
        assert H_dense.shape == (16, 16)

        # Should be Hermitian
        assert torch.allclose(H_dense, H_dense.T, atol=1e-10)

    def test_exact_ground_state(self):
        """Test exact diagonalization."""
        H = TransverseFieldIsing(num_spins=4, V=1.0, h=0.1, periodic=False)
        E0, psi0 = H.exact_ground_state()

        # Energy should be negative (ferromagnetic ground state)
        assert E0 < 0

        # Ground state should be normalized
        assert abs(torch.sum(torch.abs(psi0)**2) - 1.0) < 1e-10


class TestHeisenbergHamiltonian:
    """Test cases for Heisenberg model."""

    def test_construction(self):
        """Test basic construction."""
        H = HeisenbergHamiltonian(num_spins=4, Jx=1.0, Jy=1.0, Jz=1.0)
        assert H.num_sites == 4

    def test_diagonal_antiferromagnetic(self):
        """Test diagonal for antiferromagnetic state."""
        H = HeisenbergHamiltonian(num_spins=4, Jx=0.0, Jy=0.0, Jz=1.0)

        # Néel state |0101⟩
        config = torch.tensor([0, 1, 0, 1])
        diag = H.diagonal_element(config)

        # ZZ interaction for alternating spins
        # Jz/4 * (σ_i^z σ_j^z) where σ = ±1
        # 3 bonds: (+1)(-1), (-1)(+1), (+1)(-1) = 3 * (-1) = -3
        # Energy = Jz/4 * (-3) = -0.75
        assert abs(diag.item() + 0.75) < 1e-10

    def test_exchange_connections(self):
        """Test XX+YY connections."""
        H = HeisenbergHamiltonian(num_spins=4, Jx=1.0, Jy=1.0, Jz=0.0)

        # State |0100⟩ - antiparallel spins at positions 1,2
        config = torch.tensor([0, 1, 0, 0])
        connected, elements = H.get_connections(config)

        # Should flip antiparallel pairs
        assert len(connected) > 0


class TestMatrixElements:
    """Test matrix element computation."""

    def test_matrix_elements_symmetric(self):
        """Test that H_ij = H_ji^* for Hermitian Hamiltonian."""
        H = TransverseFieldIsing(num_spins=4, V=1.0, h=1.0)

        configs = torch.tensor([
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
        ])

        H_mat = H.matrix_elements(configs, configs)

        # Should be symmetric for real Hamiltonian
        assert torch.allclose(H_mat, H_mat.T, atol=1e-10)


class TestPauliExtraction:
    """Test Pauli string extraction."""

    def test_ising_paulis(self):
        """Test Pauli extraction for Ising model."""
        from hamiltonians.spin import extract_coeffs_and_paulis

        H = TransverseFieldIsing(num_spins=3, V=1.0, h=1.0, L=1, periodic=False)
        coeffs, paulis = extract_coeffs_and_paulis(H)

        # Should have ZZ terms and X terms
        assert len(coeffs) == len(paulis)
        assert any("ZZ" in p or "ZZI" in p or "IZZ" in p for p in paulis)
        assert any(p.count("X") == 1 for p in paulis)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
