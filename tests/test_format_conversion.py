"""TDD Phase 1a: Tests for vectorized format conversion.

Tests the conversion between our config format and IBM bitstring format.
Our format: [α₀, α₁, ..., α_{n-1}, β₀, β₁, ..., β_{n-1}]
IBM format: columns [n_orb-1..0] = spin-up (reversed), [n_qubits-1..n_orb] = spin-down (reversed)
"""

import time

import numpy as np
import pytest
import torch

from src.utils.format_utils import configs_to_ibm_format, ibm_format_to_configs


# ── Reference implementations (original Python loops, known correct) ──

def _configs_to_ibm_original(configs, n_orb, n_qubits):
    n = len(configs)
    bs = np.zeros((n, n_qubits), dtype=bool)
    for s in range(n):
        c = configs[s]
        for j in range(n_orb):
            bs[s, n_orb - 1 - j] = bool(c[j])
            bs[s, n_qubits - 1 - j] = bool(c[j + n_orb])
    return bs


def _ibm_to_configs_original(bs_matrix, n_orb, n_qubits):
    n = len(bs_matrix)
    configs = torch.zeros(n, n_qubits, dtype=torch.long)
    for s in range(n):
        for j in range(n_orb):
            configs[s, j] = int(bs_matrix[s, n_orb - 1 - j])
            configs[s, j + n_orb] = int(bs_matrix[s, n_qubits - 1 - j])
    return configs


# ── Tests for configs_to_ibm_format ──

class TestConfigsToIBMFormat:

    def test_single_config_h2(self):
        """H2: 2 orbitals. HF = [1,0, 1,0]."""
        n_orb, n_qubits = 2, 4
        configs = np.array([[1, 0, 1, 0]])
        result = configs_to_ibm_format(configs, n_orb, n_qubits)
        # α₀=1 → pos 1, α₁=0 → pos 0; β₀=1 → pos 3, β₁=0 → pos 2
        expected = np.array([[False, True, False, True]])
        np.testing.assert_array_equal(result, expected)

    def test_batch_configs_lih(self):
        """LiH: 6 orbitals, multiple configs."""
        n_orb, n_qubits = 6, 12
        hf = [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
        single = [1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0]
        configs = np.array([hf, single])
        result = configs_to_ibm_format(configs, n_orb, n_qubits)
        assert result.shape == (2, 12)
        assert result.dtype == bool

    def test_roundtrip_identity(self):
        """to_ibm -> from_ibm should recover original configs."""
        n_orb, n_qubits = 6, 12
        np.random.seed(42)
        configs_list = []
        for _ in range(20):
            c = np.zeros(n_qubits, dtype=np.int64)
            c[np.random.choice(n_orb, 2, replace=False)] = 1
            c[np.random.choice(n_orb, 2, replace=False) + n_orb] = 1
            configs_list.append(c)
        configs = np.array(configs_list)
        ibm = configs_to_ibm_format(configs, n_orb, n_qubits)
        recovered = ibm_format_to_configs(ibm, n_orb, n_qubits)
        np.testing.assert_array_equal(recovered.numpy(), configs)

    def test_matches_original(self):
        """Vectorized must match original Python loop output."""
        n_orb, n_qubits = 7, 14
        np.random.seed(123)
        configs = np.random.randint(0, 2, size=(50, n_qubits))
        result_new = configs_to_ibm_format(configs, n_orb, n_qubits)
        result_old = _configs_to_ibm_original(configs, n_orb, n_qubits)
        np.testing.assert_array_equal(result_new, result_old)

    def test_large_batch_performance(self):
        """10K configs x 26Q should take < 0.1s."""
        n_orb, n_qubits = 13, 26
        np.random.seed(0)
        configs = np.random.randint(0, 2, size=(10000, n_qubits))
        start = time.perf_counter()
        configs_to_ibm_format(configs, n_orb, n_qubits)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.1, f"Took {elapsed:.3f}s, expected < 0.1s"

    def test_empty_input(self):
        """Empty configs → empty array."""
        result = configs_to_ibm_format(np.zeros((0, 12), dtype=np.int64), 6, 12)
        assert result.shape == (0, 12)

    def test_accepts_torch_tensor(self):
        """Should accept torch.Tensor input as well as numpy."""
        n_orb, n_qubits = 3, 6
        configs = torch.tensor([[1, 0, 1, 1, 0, 1]])
        result = configs_to_ibm_format(configs, n_orb, n_qubits)
        assert isinstance(result, np.ndarray)
        assert result.dtype == bool


# ── Tests for ibm_format_to_configs ──

class TestIBMFormatToConfigs:

    def test_single_config_h2(self):
        """H2: reverse of to_ibm test."""
        n_orb, n_qubits = 2, 4
        ibm = np.array([[False, True, False, True]])
        result = ibm_format_to_configs(ibm, n_orb, n_qubits)
        expected = torch.tensor([[1, 0, 1, 0]], dtype=torch.long)
        torch.testing.assert_close(result, expected)

    def test_batch_configs(self):
        """Multiple configs."""
        n_orb, n_qubits = 6, 12
        np.random.seed(42)
        ibm = np.random.randint(0, 2, size=(30, n_qubits)).astype(bool)
        result = ibm_format_to_configs(ibm, n_orb, n_qubits)
        assert result.shape == (30, n_qubits)
        assert result.dtype == torch.long

    def test_roundtrip_from_ibm(self):
        """from_ibm -> to_ibm recovers original IBM format."""
        n_orb, n_qubits = 7, 14
        np.random.seed(42)
        ibm_original = np.random.randint(0, 2, size=(25, n_qubits)).astype(bool)
        configs = ibm_format_to_configs(ibm_original, n_orb, n_qubits)
        ibm_recovered = configs_to_ibm_format(configs.numpy(), n_orb, n_qubits)
        np.testing.assert_array_equal(ibm_recovered, ibm_original)

    def test_matches_original(self):
        """Vectorized must match original Python loop output."""
        n_orb, n_qubits = 7, 14
        np.random.seed(456)
        ibm = np.random.randint(0, 2, size=(50, n_qubits)).astype(bool)
        result_new = ibm_format_to_configs(ibm, n_orb, n_qubits)
        result_old = _ibm_to_configs_original(ibm, n_orb, n_qubits)
        torch.testing.assert_close(result_new, result_old)

    def test_large_batch_performance(self):
        """10K configs x 26Q should take < 0.1s."""
        n_orb, n_qubits = 13, 26
        np.random.seed(0)
        ibm = np.random.randint(0, 2, size=(10000, n_qubits)).astype(bool)
        start = time.perf_counter()
        ibm_format_to_configs(ibm, n_orb, n_qubits)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.1, f"Took {elapsed:.3f}s, expected < 0.1s"

    def test_empty_input(self):
        """Empty IBM matrix → empty tensor."""
        result = ibm_format_to_configs(np.zeros((0, 12), dtype=bool), 6, 12)
        assert result.shape == (0, 12)
