"""Hamiltonian connection expansion for basis enrichment.

Discovers new configurations by computing off-diagonal Hamiltonian connections
(single and double excitations) from existing basis configs. This is the core
mechanism of SKQD's Krylov expansion, extracted as a standalone utility.

Usage in HI+NQS+SKQD pipeline:
1. NQS samples configs → cumulative basis
2. expand_basis_via_connections(basis, H) → expanded basis with H-connected configs
3. Diagonalize expanded basis → better energy

This is equivalent to one "hop" of Krylov expansion: H|basis⟩ discovers
configs connected via Slater-Condon rules (singles + doubles).
"""

import numpy as np
import torch


def expand_basis_via_connections(
    basis,
    hamiltonian,
    max_new=500,
    n_ref=None,
    coupling_rank=True,
):
    """Expand basis by discovering Hamiltonian-connected configurations.

    For each reference config in basis, computes get_connections() to find
    single/double excitations, then adds the highest-coupling new configs.

    Args:
        basis: torch.Tensor (n_configs, 2*n_orb) — seed configurations.
        hamiltonian: MolecularHamiltonian with get_connections() method.
        max_new: int — maximum new configs to add.
        n_ref: int — number of reference configs to expand from.
                     If None, uses min(len(basis), 50).
        coupling_rank: bool — if True, rank new configs by |H_ij| coupling
                       strength and keep top max_new. If False, keep first found.

    Returns:
        torch.Tensor (n_expanded, 2*n_orb) — expanded basis (seed + new).
        Guaranteed: no duplicates, correct particle number, seed configs preserved.

    Raises:
        ValueError: if basis is empty.
    """
    if len(basis) == 0:
        raise ValueError("Empty basis — cannot expand")

    if isinstance(basis, np.ndarray):
        basis = torch.from_numpy(basis).long()
    basis = basis.cpu().long()

    n_seed = len(basis)
    n_sites = basis.shape[1]

    # Build set of existing configs for O(1) dedup
    existing = {row.tobytes() for row in basis.numpy()}

    if n_ref is None:
        n_ref = min(n_seed, 50)

    # Select reference configs (first n_ref — typically HF + low-energy)
    refs = basis[:n_ref]

    # Discover connected configs
    new_configs = []
    new_couplings = []

    for ref in refs:
        try:
            connected, elements = hamiltonian.get_connections(ref)
        except Exception:
            continue

        if connected is None or len(connected) == 0:
            continue

        connected = connected.cpu().long()
        if elements is not None:
            elements_np = elements.detach().cpu().numpy()
        else:
            elements_np = np.ones(len(connected))

        for i in range(len(connected)):
            key = connected[i].numpy().tobytes()
            if key not in existing:
                existing.add(key)
                new_configs.append(connected[i])
                coupling = abs(float(elements_np[i])) if i < len(elements_np) else 0.0
                new_couplings.append(coupling)

    if not new_configs:
        return basis

    new_tensor = torch.stack(new_configs)
    new_couplings_np = np.array(new_couplings)

    # Rank by coupling strength and keep top max_new
    if coupling_rank and len(new_configs) > max_new:
        top_idx = np.argsort(new_couplings_np)[::-1][:max_new].copy()
        new_tensor = new_tensor[top_idx]
    elif len(new_configs) > max_new:
        new_tensor = new_tensor[:max_new]

    # Concatenate seed + new
    expanded = torch.cat([basis, new_tensor], dim=0)

    # Multi-hop: expand again from new configs to discover 2-hop connections
    # This finds configs reachable via 2 excitations (e.g., quadruples from HF)
    if len(new_tensor) > 0 and max_new > len(new_tensor):
        remaining = max_new - len(new_tensor)
        second_hop = _single_hop(new_tensor, hamiltonian, existing, remaining, coupling_rank)
        if len(second_hop) > 0:
            expanded = torch.cat([expanded, second_hop], dim=0)

    return expanded


def _single_hop(refs, hamiltonian, existing, max_new, coupling_rank):
    """One hop of connection expansion from refs, excluding existing."""
    new_configs = []
    new_couplings = []

    for ref in refs:
        try:
            connected, elements = hamiltonian.get_connections(ref)
        except Exception:
            continue
        if connected is None or len(connected) == 0:
            continue

        connected = connected.cpu().long()
        elements_np = elements.detach().cpu().numpy() if elements is not None else np.ones(len(connected))

        for i in range(len(connected)):
            key = connected[i].numpy().tobytes()
            if key not in existing:
                existing.add(key)
                new_configs.append(connected[i])
                coupling = abs(float(elements_np[i])) if i < len(elements_np) else 0.0
                new_couplings.append(coupling)

    if not new_configs:
        return torch.zeros(0, refs.shape[1], dtype=torch.long)

    new_tensor = torch.stack(new_configs)
    if coupling_rank and len(new_configs) > max_new:
        top_idx = np.argsort(np.array(new_couplings))[::-1][:max_new].copy()
        new_tensor = new_tensor[top_idx]
    elif len(new_configs) > max_new:
        new_tensor = new_tensor[:max_new]

    return new_tensor
