"""Vectorized format conversion between our config format and IBM bitstring format.

Our format: [α₀, α₁, ..., α_{n-1}, β₀, β₁, ..., β_{n-1}]  (length 2*n_orb)
IBM format: bool array, columns [n_orb-1..0] = spin-up (reversed),
            columns [n_qubits-1..n_orb] = spin-down (reversed)

The mapping is a bit-reversal within each spin sector:
  α_j  →  IBM position (n_orb - 1 - j)
  β_j  →  IBM position (n_qubits - 1 - j)
"""

import numpy as np
import torch


def configs_to_ibm_format(configs, n_orb, n_qubits):
    """Convert config array/tensor to IBM bitstring matrix (bool ndarray).

    Args:
        configs: (n_configs, 2*n_orb) array-like — our format.
                 Accepts numpy ndarray or torch.Tensor.
        n_orb: number of spatial orbitals.
        n_qubits: 2 * n_orb.

    Returns:
        np.ndarray of shape (n_configs, n_qubits), dtype=bool — IBM format.
    """
    if isinstance(configs, torch.Tensor):
        configs_np = configs.cpu().numpy()
    else:
        configs_np = np.asarray(configs)

    n = len(configs_np)
    if n == 0:
        return np.zeros((0, n_qubits), dtype=bool)

    bs = np.zeros((n, n_qubits), dtype=bool)
    # Alpha: our [0..n_orb) → IBM [n_orb-1..0] (reversed)
    bs[:, :n_orb] = configs_np[:, :n_orb][:, ::-1].astype(bool)
    # Beta: our [n_orb..2*n_orb) → IBM [n_qubits-1..n_orb] (reversed)
    bs[:, n_orb:] = configs_np[:, n_orb:][:, ::-1].astype(bool)
    return bs


def ibm_format_to_configs(bs_matrix, n_orb, n_qubits):
    """Convert IBM bitstring matrix back to our config tensor format.

    Args:
        bs_matrix: (n_configs, n_qubits) bool ndarray — IBM format.
        n_orb: number of spatial orbitals.
        n_qubits: 2 * n_orb.

    Returns:
        torch.Tensor of shape (n_configs, n_qubits), dtype=torch.long — our format.
    """
    bs = np.asarray(bs_matrix)
    n = len(bs)
    if n == 0:
        return torch.zeros(0, n_qubits, dtype=torch.long)

    configs = np.zeros((n, n_qubits), dtype=np.int64)
    # Alpha: IBM [n_orb-1..0] → our [0..n_orb) (reverse back)
    configs[:, :n_orb] = bs[:, :n_orb][:, ::-1].astype(np.int64)
    # Beta: IBM [n_qubits-1..n_orb] → our [n_orb..2*n_orb) (reverse back)
    configs[:, n_orb:] = bs[:, n_orb:][:, ::-1].astype(np.int64)
    return torch.from_numpy(configs)


def vectorized_dedup(existing_bs, new_bs):
    """Return rows in new_bs that are not in existing_bs (and unique within new_bs).

    Uses numpy void-view hashing for O(n) amortized dedup — no Python loops.

    Args:
        existing_bs: (n_existing, n_cols) bool ndarray, or None if empty.
        new_bs: (n_new, n_cols) bool ndarray.

    Returns:
        np.ndarray of truly-new rows (preserving order from new_bs).
    """
    if len(new_bs) == 0:
        return new_bs

    n_cols = new_bs.shape[1]

    # Step 1: dedup within new_bs (keep first occurrence, preserve order)
    _, first_idx = np.unique(
        np.ascontiguousarray(new_bs).view(np.dtype((np.void, new_bs.dtype.itemsize * n_cols))),
        return_index=True,
    )
    first_idx.sort()  # np.unique scrambles index order
    new_unique = new_bs[first_idx]

    if existing_bs is None or len(existing_bs) == 0:
        return new_unique

    # Step 2: find rows in new_unique that are NOT in existing_bs
    # Use tobytes() for hashable row representation
    existing_set = {row.tobytes() for row in np.ascontiguousarray(existing_bs)}
    mask = np.array(
        [row.tobytes() not in existing_set for row in np.ascontiguousarray(new_unique)],
        dtype=bool,
    )
    return new_unique[mask]
