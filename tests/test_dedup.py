"""TDD Phase 1b: Tests for vectorized basis deduplication."""

import time

import numpy as np
import pytest

from src.utils.format_utils import vectorized_dedup


class TestVectorizedDedup:

    def test_basic_dedup(self):
        """Remove rows in new_bs that already exist in existing_bs."""
        existing = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=bool)
        new = np.array([[1, 0, 1, 0], [1, 1, 0, 0], [0, 1, 0, 1]], dtype=bool)
        result = vectorized_dedup(existing, new)
        # Only [1,1,0,0] is truly new
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], [True, True, False, False])

    def test_no_duplicates_unchanged(self):
        """If no duplicates, all new_bs rows are kept."""
        existing = np.array([[1, 0, 0, 0]], dtype=bool)
        new = np.array([[0, 1, 0, 0], [0, 0, 1, 0]], dtype=bool)
        result = vectorized_dedup(existing, new)
        assert len(result) == 2

    def test_all_duplicates(self):
        """If all new_bs rows are duplicates, return empty."""
        existing = np.array([[1, 0], [0, 1]], dtype=bool)
        new = np.array([[0, 1], [1, 0]], dtype=bool)
        result = vectorized_dedup(existing, new)
        assert len(result) == 0

    def test_empty_existing(self):
        """If existing is None/empty, all new rows are kept."""
        new = np.array([[1, 0], [0, 1], [1, 0]], dtype=bool)
        result = vectorized_dedup(None, new)
        # Dedup within new: [1,0] appears twice → keep unique = 2
        assert len(result) == 2

    def test_empty_new(self):
        """If new is empty, return empty."""
        existing = np.array([[1, 0]], dtype=bool)
        new = np.zeros((0, 2), dtype=bool)
        result = vectorized_dedup(existing, new)
        assert len(result) == 0

    def test_dedup_within_new(self):
        """Duplicates within new_bs should also be removed."""
        existing = np.array([[1, 0, 0, 0]], dtype=bool)
        new = np.array([
            [0, 1, 0, 0],
            [0, 1, 0, 0],  # duplicate within new
            [0, 0, 1, 0],
        ], dtype=bool)
        result = vectorized_dedup(existing, new)
        assert len(result) == 2  # [0,1,0,0] and [0,0,1,0]

    @pytest.mark.slow
    def test_large_batch_performance(self):
        """50K existing + 10K new should complete in reasonable time."""
        np.random.seed(0)
        existing = np.random.randint(0, 2, size=(50000, 26)).astype(bool)
        new = np.random.randint(0, 2, size=(10000, 26)).astype(bool)
        start = time.perf_counter()
        vectorized_dedup(existing, new)
        elapsed = time.perf_counter() - start
        # Use generous threshold (1s) to avoid CI flakiness
        assert elapsed < 1.0, f"Took {elapsed:.3f}s, expected < 1.0s"

    def test_preserves_new_order(self):
        """Truly-new rows should maintain their original order from new_bs."""
        existing = np.array([[1, 0, 0]], dtype=bool)
        new = np.array([
            [0, 0, 1],  # new
            [1, 0, 0],  # dup
            [0, 1, 0],  # new
        ], dtype=bool)
        result = vectorized_dedup(existing, new)
        assert len(result) == 2
        np.testing.assert_array_equal(result[0], [False, False, True])
        np.testing.assert_array_equal(result[1], [False, True, False])
