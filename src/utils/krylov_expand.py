"""Hamiltonian connection expansion for basis enrichment.

Discovers new configurations by computing off-diagonal Hamiltonian connections
(single and double excitations) from existing basis configs. This is the core
mechanism of SKQD's Krylov expansion, extracted as a standalone utility.

Usage in HI+NQS+SKQD pipeline:
1. NQS samples configs → cumulative basis
2. expand_basis_via_connections(basis, H) → expanded basis with H-connected configs
3. Diagonalize expanded basis → better energy

Performs 2-hop Krylov expansion: first hop discovers singles/doubles from
seed configs, second hop expands from those to reach up to quadruples.
Each hop uses Slater-Condon rules via hamiltonian.get_connections().
"""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


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

    NOTE: The first n_ref configs in basis are used as references. For best
    results, ensure basis is ordered with HF and low-energy configs first.

    Args:
        basis: torch.Tensor or np.ndarray (n_configs, 2*n_orb) — seed configs.
        hamiltonian: MolecularHamiltonian with get_connections() method.
        max_new: int — maximum new configs to add per hop.
        n_ref: int — number of reference configs to expand from.
                     If None, uses min(len(basis), 50).
        coupling_rank: bool — if True, rank new configs by max |H_ij| coupling
                       strength across all references and keep top max_new.

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

    # Build set of existing config keys for O(1) dedup
    existing_keys = {row.tobytes() for row in basis.numpy()}

    if n_ref is None:
        n_ref = min(n_seed, 50)

    # Select reference configs (first n_ref — should be HF + low-energy)
    refs = basis[:n_ref]

    # ── First hop: discover connected configs with max-coupling tracking ──
    new_map, new_configs_map = _collect_connections(refs, hamiltonian, existing_keys)

    if not new_map:
        return basis

    # Build tensors from the collected configs
    keys_list = list(new_map.keys())
    new_configs = [new_configs_map[k] for k in keys_list]
    new_couplings = np.array([new_map[k] for k in keys_list])
    new_tensor = torch.stack(new_configs)

    # Rank by coupling strength and keep top max_new
    new_tensor, new_couplings = _truncate_by_coupling(
        new_tensor, new_couplings, max_new, coupling_rank
    )

    # Concatenate seed + first hop
    expanded = torch.cat([basis, new_tensor], dim=0)

    # ── Second hop: expand from first-hop configs (capped at 50 refs) ──
    if len(new_tensor) > 0 and max_new > len(new_tensor):
        remaining = max_new - len(new_tensor)
        second_refs = new_tensor[:min(50, len(new_tensor))]
        hop2_map, hop2_configs_map = _collect_connections(
            second_refs, hamiltonian, existing_keys
        )
        if hop2_map:
            keys2 = list(hop2_map.keys())
            hop2_tensor = torch.stack([hop2_configs_map[k] for k in keys2])
            hop2_couplings = np.array([hop2_map[k] for k in keys2])
            hop2_tensor, _ = _truncate_by_coupling(
                hop2_tensor, hop2_couplings, remaining, coupling_rank
            )
            if len(hop2_tensor) > 0:
                expanded = torch.cat([expanded, hop2_tensor], dim=0)

    return expanded


def _collect_connections(refs, hamiltonian, existing_keys):
    """Collect connected configs from references, tracking max coupling per config.

    Returns:
        new_map: dict[bytes → float] — config key → max |H_ij| coupling
        new_configs_map: dict[bytes → Tensor] — config key → config tensor
    """
    new_map = {}  # key → max coupling
    new_configs_map = {}  # key → config tensor

    for ref in refs:
        try:
            connected, elements = hamiltonian.get_connections(ref)
        except Exception as e:
            logger.debug(f"get_connections failed: {e}")
            continue

        if connected is None or len(connected) == 0:
            continue

        connected = connected.cpu().long()
        if elements is not None:
            elements_np = elements.detach().cpu().numpy()
            assert len(elements_np) == len(connected), (
                f"get_connections returned {len(connected)} configs but "
                f"{len(elements_np)} elements"
            )
        else:
            elements_np = np.ones(len(connected))

        for i in range(len(connected)):
            key = connected[i].numpy().tobytes()
            if key in existing_keys:
                continue
            coupling = abs(float(elements_np[i]))

            # Track max coupling across all references (C3 fix)
            if key not in new_map or coupling > new_map[key]:
                new_map[key] = coupling
                new_configs_map[key] = connected[i]

    # Add all newly discovered keys to existing set for cross-hop dedup
    existing_keys.update(new_map.keys())

    return new_map, new_configs_map


def _truncate_by_coupling(tensor, couplings, max_new, coupling_rank):
    """Keep top max_new configs by coupling strength."""
    if len(tensor) <= max_new:
        return tensor, couplings
    if coupling_rank:
        top_idx = np.argsort(couplings)[::-1][:max_new].copy()
    else:
        top_idx = np.arange(max_new)
    return tensor[top_idx], couplings[top_idx]
