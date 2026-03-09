"""
Quantum Circuit vs Classical Krylov: 3-Way Comparison.

Compares three SKQD implementations on the same molecular systems:

1. Path C (Classical SKQD): exact e^{-iHt} in particle-conserving subspace (no Trotter error)
2. Path B (Classical Trotterized): second-order Suzuki-Trotter on GPU state-vector
3. Path A (CUDA-Q Circuit): real quantum circuit via exp_pauli gates, second-order Trotter

Paper-compliant parameters (Yu et al., arXiv:2501.09702):
- Optimal time step: dt = pi / spectral_range (Theorem 3.1, Epperly et al.)
- Second-order Suzuki-Trotter decomposition (paper Section IV)
- Cumulative basis across all Krylov states
- Standard eigenvalue problem (S=I, computational basis is orthonormal)

Supports systems from 4 to 30 qubits. Uses FCI for small/medium systems,
CCSD(T) for large systems where FCI is infeasible.

Reference:
    NVIDIA CUDA-Q SKQD tutorial:
    nvidia.github.io/cuda-quantum/latest/applications/python/skqd.html

Usage:
    python examples/quantum_vs_classical_krylov.py --systems h2 lih
    python examples/quantum_vs_classical_krylov.py --tier small
    python examples/quantum_vs_classical_krylov.py --tier medium --paths C B --profile
    python examples/quantum_vs_classical_krylov.py --tier all --paths C B --profile
    python examples/quantum_vs_classical_krylov.py --paths B --systems h2o_631g --profile
    docker-compose run --rm flow-krylov-gpu python examples/quantum_vs_classical_krylov.py
"""

import sys
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Set, Tuple
from math import comb

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch

from hamiltonians.molecular import (
    create_h2_hamiltonian,
    create_lih_hamiltonian,
    create_h2o_hamiltonian,
    create_beh2_hamiltonian,
    create_nh3_hamiltonian,
    create_ch4_hamiltonian,
    create_n2_hamiltonian,
)
from krylov.quantum_skqd import QuantumCircuitSKQD, QuantumSKQDConfig, CUDAQ_AVAILABLE

# ---------------------------------------------------------------------------
# System registry
# ---------------------------------------------------------------------------

# Tier definitions
TIER_SMALL = ["h2", "lih", "h2o", "beh2", "nh3", "ch4", "n2"]
TIER_MEDIUM = ["co", "hcn", "c2h2"]
TIER_LARGE = ["h2o_631g", "h2s", "c2h4", "nh3_631g"]

SYSTEMS = {
    # --- Small tier (4-20 qubits, factory functions in hamiltonians.molecular) ---
    "h2": {
        "name": "H2",
        "factory": create_h2_hamiltonian,
        "kwargs": {"bond_length": 0.74},
    },
    "lih": {
        "name": "LiH",
        "factory": create_lih_hamiltonian,
        "kwargs": {"bond_length": 1.6},
    },
    "h2o": {
        "name": "H2O",
        "factory": create_h2o_hamiltonian,
        "kwargs": {},
    },
    "beh2": {
        "name": "BeH2",
        "factory": create_beh2_hamiltonian,
        "kwargs": {},
    },
    "nh3": {
        "name": "NH3",
        "factory": create_nh3_hamiltonian,
        "kwargs": {},
    },
    "ch4": {
        "name": "CH4",
        "factory": create_ch4_hamiltonian,
        "kwargs": {},
    },
    "n2": {
        "name": "N2",
        "factory": create_n2_hamiltonian,
        "kwargs": {"bond_length": 1.10},
    },
    # --- Medium tier (20-24 qubits, factory via moderate_system_benchmark) ---
    "co": {
        "name": "CO",
        "molecule_factory": "create_co_molecule",
    },
    "hcn": {
        "name": "HCN",
        "molecule_factory": "create_hcn_molecule",
    },
    "c2h2": {
        "name": "C2H2",
        "molecule_factory": "create_c2h2_molecule",
    },
    # --- Large tier (26-30 qubits) ---
    "h2o_631g": {
        "name": "H2O(6-31G)",
        "molecule_factory": "create_h2o_631g_molecule",
    },
    "h2s": {
        "name": "H2S",
        "molecule_factory": "create_h2s_molecule",
    },
    "c2h4": {
        "name": "C2H4",
        "molecule_factory": "create_c2h4_molecule",
    },
    "nh3_631g": {
        "name": "NH3(6-31G)",
        "molecule_factory": "create_nh3_631g_molecule",
    },
}


@dataclass
class ProfileData:
    """Per-path profiling data."""
    jw_time_s: float = 0.0
    dt_time_s: float = 0.0
    sampling_time_s: float = 0.0
    energy_time_s: float = 0.0
    peak_vram_mb: float = 0.0


@dataclass
class ComparisonResult:
    """Results from a comparison run."""
    system: str
    n_qubits: int
    n_configs: int
    n_pauli_terms: int
    ref_energy: float
    ref_type: str  # "FCI" or "CCSD(T)"
    spectral_range: float
    optimal_dt: float
    # Direct diag baseline
    direct_energy: float
    direct_error_mha: float
    # Path C: Classical SKQD (exact time evolution)
    classical_energy: Optional[float] = None
    classical_error_mha: Optional[float] = None
    classical_time_s: Optional[float] = None
    classical_basis_size: Optional[int] = None
    classical_skip_reason: Optional[str] = None
    # Path B: Classical Trotterized (state-vector, second-order)
    pathB_energy: Optional[float] = None
    pathB_error_mha: Optional[float] = None
    pathB_time_s: Optional[float] = None
    pathB_basis_size: Optional[int] = None
    pathB_skip_reason: Optional[str] = None
    # Path A: CUDA-Q circuit (if available, second-order)
    pathA_energy: Optional[float] = None
    pathA_error_mha: Optional[float] = None
    pathA_time_s: Optional[float] = None
    pathA_basis_size: Optional[int] = None
    pathA_available: bool = False
    # Profiling
    profile_C: Optional[ProfileData] = None
    profile_B: Optional[ProfileData] = None
    profile_A: Optional[ProfileData] = None


def _generate_essential_configs(hamiltonian) -> torch.Tensor:
    """Generate HF + singles + doubles configs for a molecular Hamiltonian."""
    from itertools import combinations

    n_orb = hamiltonian.n_orbitals
    n_alpha = hamiltonian.n_alpha
    n_beta = hamiltonian.n_beta
    device = hamiltonian.device

    hf_state = hamiltonian.get_hf_state()
    essential = [hf_state.clone()]

    occ_alpha = list(range(n_alpha))
    occ_beta = list(range(n_beta))
    virt_alpha = list(range(n_alpha, n_orb))
    virt_beta = list(range(n_beta, n_orb))

    for i in occ_alpha:
        for a in virt_alpha:
            cfg = hf_state.clone(); cfg[i] = 0; cfg[a] = 1
            essential.append(cfg)
    for i in occ_beta:
        for a in virt_beta:
            cfg = hf_state.clone(); cfg[i + n_orb] = 0; cfg[a + n_orb] = 1
            essential.append(cfg)

    max_doubles = 5000
    count = 0
    for i, j in combinations(occ_alpha, 2):
        for a, b in combinations(virt_alpha, 2):
            if count >= max_doubles: break
            cfg = hf_state.clone(); cfg[i] = 0; cfg[j] = 0; cfg[a] = 1; cfg[b] = 1
            essential.append(cfg); count += 1
    for i, j in combinations(occ_beta, 2):
        for a, b in combinations(virt_beta, 2):
            if count >= max_doubles: break
            cfg = hf_state.clone(); cfg[i+n_orb] = 0; cfg[j+n_orb] = 0; cfg[a+n_orb] = 1; cfg[b+n_orb] = 1
            essential.append(cfg); count += 1
    for i in occ_alpha:
        for j in occ_beta:
            for a in virt_alpha:
                for b in virt_beta:
                    if count >= max_doubles: break
                    cfg = hf_state.clone(); cfg[i] = 0; cfg[j+n_orb] = 0; cfg[a] = 1; cfg[b+n_orb] = 1
                    essential.append(cfg); count += 1

    return torch.unique(torch.stack(essential).to(device), dim=0)


from krylov.spectral_utils import compute_optimal_dt as _compute_optimal_dt


def _estimate_vram_bytes(n_qubits: int, path: str) -> int:
    """Estimate VRAM requirement for a given path and qubit count."""
    dim = 2 ** n_qubits
    state_bytes = dim * 16  # complex128
    if path == "B":
        # 2 working states + 1 cached + arange cache
        return state_bytes * 3 + dim * 8
    elif path == "C":
        # Low-memory Lanczos: 3 vectors + 1 cached + 2 working
        return state_bytes * 6
    elif path == "A":
        # CUDA-Q manages its own memory; estimate similar to B
        return state_bytes * 3
    return state_bytes * 4


def _get_free_vram() -> Optional[int]:
    """Get free VRAM in bytes, or None if no GPU."""
    if torch.cuda.is_available():
        return torch.cuda.mem_get_info()[0]
    return None


def _check_vram_feasibility(
    n_qubits: int, path: str, max_vram_gb: Optional[float] = None
) -> Optional[str]:
    """Check if a path is feasible given VRAM. Returns skip reason or None."""
    estimate = _estimate_vram_bytes(n_qubits, path)

    if max_vram_gb is not None:
        limit = int(max_vram_gb * 1024**3)
        if estimate > limit * 0.8:
            return (f"estimated {estimate / 1024**3:.1f} GB > "
                    f"80% of --max-vram {max_vram_gb:.1f} GB")

    free = _get_free_vram()
    if free is not None and estimate > free * 0.8:
        return (f"estimated {estimate / 1024**3:.1f} GB > "
                f"80% of free VRAM {free / 1024**3:.1f} GB")

    return None


def _get_molecule_data(system_key: str):
    """
    Load a molecule for medium/large tier systems.

    Returns (MoleculeData, is_molecule_data) where is_molecule_data=True
    means it has .hamiltonian, .ccsd_t_energy, etc.
    """
    # Lazy import to avoid requiring pyscf for small-tier systems
    from moderate_system_benchmark import (
        create_co_molecule,
        create_hcn_molecule,
        create_c2h2_molecule,
        create_h2o_631g_molecule,
        create_h2s_molecule,
        create_c2h4_molecule,
        create_nh3_631g_molecule,
        compute_pyscf_fci,
    )

    factory_map = {
        "create_co_molecule": create_co_molecule,
        "create_hcn_molecule": create_hcn_molecule,
        "create_c2h2_molecule": create_c2h2_molecule,
        "create_h2o_631g_molecule": create_h2o_631g_molecule,
        "create_h2s_molecule": create_h2s_molecule,
        "create_c2h4_molecule": create_c2h4_molecule,
        "create_nh3_631g_molecule": create_nh3_631g_molecule,
    }

    factory_name = SYSTEMS[system_key]["molecule_factory"]
    mol_data = factory_map[factory_name]()
    return mol_data


def _run_quantum_skqd(
    hamiltonian,
    max_krylov_dim: int,
    krylov_dt: float,
    num_trotter_steps: int,
    trotter_order: int,
    shots: int,
    backend: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run quantum-circuit SKQD with specified backend."""
    config = QuantumSKQDConfig(
        max_krylov_dim=max_krylov_dim,
        total_evolution_time=krylov_dt,
        num_trotter_steps=num_trotter_steps,
        trotter_order=trotter_order,
        shots=shots,
        initial_state="hf",
        backend=backend,
    )

    solver = QuantumCircuitSKQD.from_molecular_hamiltonian(hamiltonian, config=config)
    results = solver.run(progress=verbose)
    return results


def run_comparison(
    system_key: str,
    max_krylov_dim: int = 15,
    quantum_shots: int = 100_000,
    num_trotter_steps: int = 1,
    trotter_order: int = 2,
    verbose: bool = True,
    enabled_paths: Optional[Set[str]] = None,
    profile: bool = False,
    max_vram_gb: Optional[float] = None,
) -> ComparisonResult:
    """Run comparison for selected paths: C, B, A (default: all)."""
    paths = enabled_paths or {"C", "B", "A"}
    info = SYSTEMS[system_key]
    name = info["name"]

    n_enabled = sum(1 for p in ["C", "B", "A"] if p in paths)
    enabled_label = "/".join(p for p in ["C", "B", "A"] if p in paths)

    print(f"\n{'=' * 70}")
    print(f"  {name}: SKQD Comparison — Path {enabled_label} (Paper-Compliant)")
    print(f"{'=' * 70}")

    # --- Create Hamiltonian ---
    ref_type = "FCI"
    ref_energy = None
    ccsd_t_energy = None

    if "factory" in info:
        # Small tier: direct factory
        H = info["factory"](**info["kwargs"])
    else:
        # Medium/Large tier: molecule data with CCSD(T) reference
        mol_data = _get_molecule_data(system_key)
        H = mol_data.hamiltonian
        ccsd_t_energy = mol_data.ccsd_t_energy

    n_qubits = H.num_sites
    n_orb = H.n_orbitals
    n_configs = comb(n_orb, H.n_alpha) * comb(n_orb, H.n_beta)

    # --- Reference energy hierarchy ---
    if n_configs <= 100_000:
        try:
            ref_energy = H.fci_energy()
            ref_type = "FCI"
            print(f"  FCI energy: {ref_energy:.8f} Ha")
        except Exception as e:
            print(f"  Matrix FCI failed: {e}")

    if ref_energy is None:
        try:
            from utils.gpu_fci import GPU4PYSCF_AVAILABLE, compute_gpu_fci
            if GPU4PYSCF_AVAILABLE:
                geometry = mol_data.geometry
                basis = mol_data.basis
                print(f"  Computing FCI via GPU4PySCF Davidson ({n_configs:,} configs)...")
                t0 = time.time()
                ref_energy = compute_gpu_fci(geometry, basis)
                ref_type = "FCI"
                print(f"  GPU FCI energy: {ref_energy:.8f} Ha ({time.time() - t0:.1f}s)")
        except Exception as e:
            print(f"  GPU FCI failed: {e}")

    if ref_energy is None and n_configs <= 15_000_000:
        try:
            from moderate_system_benchmark import compute_pyscf_fci
            geometry = mol_data.geometry
            basis = mol_data.basis
            print(f"  Computing FCI via PySCF CPU Davidson ({n_configs:,} configs)...")
            t0 = time.time()
            ref_energy = compute_pyscf_fci(geometry, basis)
            ref_type = "FCI"
            print(f"  PySCF FCI energy: {ref_energy:.8f} Ha ({time.time() - t0:.1f}s)")
        except Exception as e:
            print(f"  PySCF CPU FCI failed: {e}")

    if ref_energy is None:
        if ccsd_t_energy is not None:
            ref_energy = ccsd_t_energy
            ref_type = "CCSD(T)"
            print(f"  CCSD(T) energy: {ref_energy:.8f} Ha (NOT a variational bound)")
        else:
            raise RuntimeError(f"No reference energy available for {name}")

    print(f"  Qubits: {n_qubits}, Orbitals: {n_orb}, Configs: {n_configs:,}")
    print(f"  Reference: {ref_type} = {ref_energy:.8f} Ha")

    # Compute optimal time step from spectral range (paper Theorem 3.1)
    t_dt0 = time.time()
    optimal_dt, spectral_range = _compute_optimal_dt(H)
    dt_time = time.time() - t_dt0
    print(f"  Spectral range: {spectral_range:.4f} Ha")
    print(f"  Optimal dt (pi/dE): {optimal_dt:.6f}")
    if profile:
        print(f"  compute_optimal_dt time: {dt_time:.1f}s")
    print(f"  Trotter order: {trotter_order}, Trotter steps: {num_trotter_steps}")

    # Direct diag baseline
    nf_basis = _generate_essential_configs(H)
    print(f"  Essential configs: {len(nf_basis)}")
    H_proj = H.matrix_elements(nf_basis, nf_basis)
    H_np = H_proj.detach().cpu().numpy().real
    H_np = 0.5 * (H_np + H_np.T)
    direct_energy = float(np.linalg.eigh(H_np)[0][0])
    direct_error = abs(direct_energy - ref_energy) * 1000
    print(f"  Direct diag (no Krylov): {direct_energy:.8f} Ha, error: {direct_error:.4f} mHa")

    n_pauli_terms = 0
    step_idx = 0

    # Helper for profiling
    def _reset_vram_stats():
        if profile and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def _get_peak_vram_mb():
        if profile and torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024**2)
        return 0.0

    # ------------------------------------------------------------------
    # 1. Path C: Exact evolution (Lanczos, same 2^n space as Path A/B)
    # ------------------------------------------------------------------
    classical_energy = None
    classical_error = None
    classical_time = None
    classical_basis_size = None
    classical_skip_reason = None
    prof_C = ProfileData(dt_time_s=dt_time) if profile else None

    if "C" in paths:
        skip = _check_vram_feasibility(n_qubits, "C", max_vram_gb)
        if skip:
            classical_skip_reason = skip
            step_idx += 1
            print(f"\n{'─' * 60}")
            print(f"  [{step_idx}/{n_enabled}] Path C: SKIPPED ({skip})")
            print(f"{'─' * 60}")
        else:
            step_idx += 1
            print(f"\n{'─' * 60}")
            print(f"  [{step_idx}/{n_enabled}] Path C: Exact Lanczos "
                  f"(no Trotter, dt={optimal_dt:.6f})")
            print(f"{'─' * 60}")

            _reset_vram_stats()
            t0 = time.time()
            pathC_results = _run_quantum_skqd(
                H, max_krylov_dim, optimal_dt, num_trotter_steps, trotter_order,
                quantum_shots, backend="exact", verbose=verbose,
            )
            classical_time = time.time() - t0
            classical_energy = pathC_results["best_energy"]
            classical_error = abs(classical_energy - ref_energy) * 1000
            classical_basis_size = (
                pathC_results["basis_sizes"][-1] if pathC_results["basis_sizes"] else 0
            )
            if n_pauli_terms == 0:
                n_pauli_terms = pathC_results["n_pauli_terms"]

            if prof_C:
                prof_C.peak_vram_mb = _get_peak_vram_mb()
                prof_C.sampling_time_s = classical_time

            print(f"  Energy: {classical_energy:.8f} Ha")
            err_label = "Error" if ref_type == "FCI" else f"vs {ref_type}"
            print(f"  {err_label}: {classical_error:.4f} mHa | "
                  f"Time: {classical_time:.1f}s | Basis: {classical_basis_size}")
            if prof_C:
                print(f"  Peak VRAM: {prof_C.peak_vram_mb:.0f} MB")

    # ------------------------------------------------------------------
    # 2. Path B: Classical Trotterized (state-vector, GPU, 2nd-order)
    # ------------------------------------------------------------------
    pathB_energy = None
    pathB_error = None
    pathB_time = None
    pathB_basis_size = None
    pathB_skip_reason = None
    prof_B = ProfileData(dt_time_s=dt_time) if profile else None

    if "B" in paths:
        skip = _check_vram_feasibility(n_qubits, "B", max_vram_gb)
        if skip:
            pathB_skip_reason = skip
            step_idx += 1
            print(f"\n{'─' * 60}")
            print(f"  [{step_idx}/{n_enabled}] Path B: SKIPPED ({skip})")
            print(f"{'─' * 60}")
        else:
            step_idx += 1
            print(f"\n{'─' * 60}")
            print(f"  [{step_idx}/{n_enabled}] Path B: Trotterized State-Vector "
                  f"(order={trotter_order}, dt={optimal_dt:.6f})")
            print(f"{'─' * 60}")

            _reset_vram_stats()
            t0 = time.time()
            pathB_results = _run_quantum_skqd(
                H, max_krylov_dim, optimal_dt, num_trotter_steps, trotter_order,
                quantum_shots, backend="classical", verbose=verbose,
            )
            pathB_time = time.time() - t0
            pathB_energy = pathB_results["best_energy"]
            pathB_error = abs(pathB_energy - ref_energy) * 1000
            pathB_basis_size = (
                pathB_results["basis_sizes"][-1] if pathB_results["basis_sizes"] else 0
            )
            n_pauli_terms = pathB_results["n_pauli_terms"]

            if prof_B:
                prof_B.peak_vram_mb = _get_peak_vram_mb()
                prof_B.sampling_time_s = pathB_time

            print(f"  Energy: {pathB_energy:.8f} Ha")
            err_label = "Error" if ref_type == "FCI" else f"vs {ref_type}"
            print(f"  {err_label}: {pathB_error:.4f} mHa | "
                  f"Time: {pathB_time:.1f}s | Basis: {pathB_basis_size} | "
                  f"Pauli terms: {n_pauli_terms}")
            if prof_B:
                print(f"  Peak VRAM: {prof_B.peak_vram_mb:.0f} MB")

    # ------------------------------------------------------------------
    # 3. Path A: CUDA-Q Circuit (if available, 2nd-order)
    # ------------------------------------------------------------------
    pathA_energy = None
    pathA_error = None
    pathA_time = None
    pathA_basis_size = None
    pathA_available = CUDAQ_AVAILABLE
    prof_A = ProfileData(dt_time_s=dt_time) if profile else None

    if "A" in paths:
        if CUDAQ_AVAILABLE:
            step_idx += 1
            print(f"\n{'─' * 60}")
            print(f"  [{step_idx}/{n_enabled}] Path A: CUDA-Q Circuit "
                  f"(order={trotter_order}, dt={optimal_dt:.6f})")
            print(f"{'─' * 60}")

            _reset_vram_stats()
            t0 = time.time()
            pathA_results = _run_quantum_skqd(
                H, max_krylov_dim, optimal_dt, num_trotter_steps, trotter_order,
                quantum_shots, backend="cudaq", verbose=verbose,
            )
            pathA_time = time.time() - t0
            pathA_energy = pathA_results["best_energy"]
            pathA_error = abs(pathA_energy - ref_energy) * 1000
            pathA_basis_size = (
                pathA_results["basis_sizes"][-1] if pathA_results["basis_sizes"] else 0
            )
            if n_pauli_terms == 0:
                n_pauli_terms = pathA_results["n_pauli_terms"]

            if prof_A:
                prof_A.peak_vram_mb = _get_peak_vram_mb()
                prof_A.sampling_time_s = pathA_time

            print(f"  Energy: {pathA_energy:.8f} Ha")
            err_label = "Error" if ref_type == "FCI" else f"vs {ref_type}"
            print(f"  {err_label}: {pathA_error:.4f} mHa | Time: {pathA_time:.1f}s | "
                  f"Basis: {pathA_basis_size}")
            if prof_A:
                print(f"  Peak VRAM: {prof_A.peak_vram_mb:.0f} MB")
        else:
            step_idx += 1
            print(f"\n{'─' * 60}")
            print(f"  [{step_idx}/{n_enabled}] Path A: CUDA-Q — SKIPPED "
                  f"(cudaq not installed)")
            print(f"{'─' * 60}")

    # ------------------------------------------------------------------
    # Error analysis
    # ------------------------------------------------------------------
    print(f"\n{'─' * 60}")
    print(f"  Error Analysis")
    print(f"{'─' * 60}")
    print(f"  Reference ({ref_type}):       {ref_energy:.8f} Ha")
    if ref_type != "FCI":
        print(f"  NOTE: {ref_type} is NOT a variational lower bound")
    print(f"  Optimal dt:               {optimal_dt:.6f} (spectral range: {spectral_range:.4f})")
    print(f"  Direct diag error:        {direct_error:.4f} mHa")
    if classical_error is not None:
        print(f"  Path C (exact) error:     {classical_error:.4f} mHa")
    elif classical_skip_reason:
        print(f"  Path C: SKIPPED ({classical_skip_reason})")
    if pathB_error is not None:
        print(f"  Path B (Trotter-{trotter_order}) error: {pathB_error:.4f} mHa")
    elif pathB_skip_reason:
        print(f"  Path B: SKIPPED ({pathB_skip_reason})")
    elif "B" in paths:
        print(f"  Path B (Trotter-{trotter_order}) error: SKIPPED (memory)")
    if pathA_error is not None:
        print(f"  Path A (CUDA-Q-{trotter_order}) error: {pathA_error:.4f} mHa")
    elif "A" in paths and not CUDAQ_AVAILABLE:
        print(f"  Path A: SKIPPED (cudaq not installed)")
    print(f"")
    if pathB_energy is not None and classical_energy is not None:
        print(f"  Trotter effect (B-C): "
              f"{abs(pathB_energy - classical_energy) * 1000:.4f} mHa")
    if pathA_energy is not None and pathB_energy is not None:
        print(f"  Circuit effect (A-B): "
              f"{abs(pathA_energy - pathB_energy) * 1000:.4f} mHa")
    elif pathA_energy is not None and classical_energy is not None:
        print(f"  Circuit vs exact (A-C): "
              f"{abs(pathA_energy - classical_energy) * 1000:.4f} mHa")
    basis_parts = []
    if classical_basis_size is not None:
        basis_parts.append(f"C={classical_basis_size}")
    if pathB_basis_size is not None:
        basis_parts.append(f"B={pathB_basis_size}")
    if pathA_basis_size is not None:
        basis_parts.append(f"A={pathA_basis_size}")
    if basis_parts:
        print(f"  Basis sizes: {', '.join(basis_parts)}")

    chem_acc = 1.594
    if ref_type == "FCI":
        print(f"\n  Chemical accuracy (< {chem_acc:.3f} mHa):")
        if classical_error is not None:
            print(f"    Path C: {'PASS' if classical_error < chem_acc else 'FAIL'}")
        if pathB_error is not None:
            print(f"    Path B: {'PASS' if pathB_error < chem_acc else 'FAIL'}")
        elif "B" in paths and pathB_energy is None:
            print(f"    Path B: SKIPPED")
        if pathA_error is not None:
            print(f"    Path A: {'PASS' if pathA_error < chem_acc else 'FAIL'}")
    else:
        print(f"\n  Deviation from {ref_type} (not pass/fail — {ref_type} is not variational):")
        if classical_error is not None:
            print(f"    Path C: {classical_error:.4f} mHa")
        if pathB_error is not None:
            print(f"    Path B: {pathB_error:.4f} mHa")
        if pathA_error is not None:
            print(f"    Path A: {pathA_error:.4f} mHa")

    return ComparisonResult(
        system=name,
        n_qubits=n_qubits,
        n_configs=n_configs,
        n_pauli_terms=n_pauli_terms,
        ref_energy=ref_energy,
        ref_type=ref_type,
        spectral_range=spectral_range,
        optimal_dt=optimal_dt,
        direct_energy=direct_energy,
        direct_error_mha=direct_error,
        classical_energy=classical_energy,
        classical_error_mha=classical_error,
        classical_time_s=classical_time,
        classical_basis_size=classical_basis_size,
        classical_skip_reason=classical_skip_reason,
        pathB_energy=pathB_energy,
        pathB_error_mha=pathB_error,
        pathB_time_s=pathB_time,
        pathB_basis_size=pathB_basis_size,
        pathB_skip_reason=pathB_skip_reason,
        pathA_energy=pathA_energy,
        pathA_error_mha=pathA_error,
        pathA_time_s=pathA_time,
        pathA_basis_size=pathA_basis_size,
        pathA_available=pathA_available,
        profile_C=prof_C,
        profile_B=prof_B,
        profile_A=prof_A,
    )


def print_summary_table(results: List[ComparisonResult], enabled_paths: Set[str]) -> None:
    """Print summary comparison table for enabled paths."""
    has_C = "C" in enabled_paths and any(r.classical_error_mha is not None for r in results)
    has_B = "B" in enabled_paths
    has_A = "A" in enabled_paths and any(r.pathA_available for r in results)
    has_profile = any(
        r.profile_C is not None or r.profile_B is not None for r in results
    )

    print(f"\n{'=' * 115}")
    enabled_label = "/".join(p for p in ["C", "B", "A"] if p in enabled_paths)
    print(f"  SUMMARY: SKQD Comparison — Path {enabled_label} (Paper-Compliant)")
    print(f"{'=' * 115}")

    # Header
    hdr = f"{'System':<12} {'Qubits':<7} {'Configs':<10} {'Ref':>6} {'dt_opt':>8} "
    hdr_sub = f"{'':12} {'':7} {'':10} {'':>6} {'':>8} "
    if has_C:
        hdr += f"{'Path C':>12} "
        hdr_sub += f"{'Error(mHa)':>12} "
    if has_B:
        hdr += f"{'Path B':>12} "
        hdr_sub += f"{'Error(mHa)':>12} "
    if has_A:
        hdr += f"{'Path A':>12} "
        hdr_sub += f"{'Error(mHa)':>12} "
    if has_C:
        hdr += f"{'C':>7}"
        hdr_sub += f"{'Time':>7}"
    if has_B:
        hdr += f" {'B':>7}"
        hdr_sub += f" {'Time':>7}"
    if has_A:
        hdr += f" {'A':>7}"
        hdr_sub += f" {'Time':>7}"
    if has_profile:
        hdr += f"  {'C VRAM':>8} {'B VRAM':>8}"
        hdr_sub += f"  {'(MB)':>8} {'(MB)':>8}"

    print(hdr)
    print(hdr_sub)
    print("─" * 115)

    for r in results:
        line = (f"{r.system:<12} {r.n_qubits:<7} {r.n_configs:<10,} "
                f"{r.ref_type:>6} {r.optimal_dt:>8.5f} ")
        if has_C:
            if r.classical_error_mha is not None:
                line += f"{r.classical_error_mha:>12.4f} "
            elif r.classical_skip_reason:
                line += f"{'skip':>12} "
            else:
                line += f"{'N/A':>12} "
        if has_B:
            if r.pathB_error_mha is not None:
                line += f"{r.pathB_error_mha:>12.4f} "
            elif r.pathB_skip_reason:
                line += f"{'skip':>12} "
            else:
                line += f"{'skip':>12} "
        if has_A:
            if r.pathA_error_mha is not None:
                line += f"{r.pathA_error_mha:>12.4f} "
            else:
                line += f"{'N/A':>12} "
        if has_C:
            if r.classical_time_s is not None:
                line += f"{r.classical_time_s:>6.1f}s"
            else:
                line += f"{'skip':>7}"
        if has_B:
            if r.pathB_time_s is not None:
                line += f" {r.pathB_time_s:>6.1f}s"
            else:
                line += f" {'skip':>7}"
        if has_A:
            if r.pathA_time_s is not None:
                line += f" {r.pathA_time_s:>6.1f}s"
            else:
                line += f" {'N/A':>7}"
        if has_profile:
            c_vram = (f"{r.profile_C.peak_vram_mb:>7.0f}" if r.profile_C
                      and r.profile_C.peak_vram_mb > 0 else f"{'---':>7}")
            b_vram = (f"{r.profile_B.peak_vram_mb:>7.0f}" if r.profile_B
                      and r.profile_B.peak_vram_mb > 0 else f"{'---':>7}")
            line += f"  {c_vram}  {b_vram}"
        print(line)

    print("─" * 115)

    chem_acc = 1.594

    # FCI systems: chemical accuracy check
    fci_results = [r for r in results if r.ref_type == "FCI"]
    if fci_results:
        summary_parts = []
        if has_C:
            pathC_valid = [r for r in fci_results if r.classical_error_mha is not None]
            n_pass_c = sum(1 for r in pathC_valid if r.classical_error_mha < chem_acc)
            summary_parts.append(f"Path C {n_pass_c}/{len(pathC_valid)}")
        if has_B:
            pathB_valid = [r for r in fci_results if r.pathB_error_mha is not None]
            n_pass_b = sum(1 for r in pathB_valid if r.pathB_error_mha < chem_acc)
            summary_parts.append(f"Path B {n_pass_b}/{len(pathB_valid)}")
        if has_A:
            pathA_valid = [r for r in fci_results if r.pathA_error_mha is not None]
            n_pass_a = sum(1 for r in pathA_valid if r.pathA_error_mha < chem_acc)
            summary_parts.append(f"Path A {n_pass_a}/{len(pathA_valid)}")
        if summary_parts:
            print(f"\nChemical accuracy (< {chem_acc:.3f} mHa, FCI ref): "
                  f"{', '.join(summary_parts)}")

    # CCSD(T) systems: deviation report
    ccsdt_results = [r for r in results if r.ref_type == "CCSD(T)"]
    if ccsdt_results:
        print(f"\nDeviation from CCSD(T) (not pass/fail — CCSD(T) is not variational):")
        for r in ccsdt_results:
            parts = [f"  {r.system}:"]
            if r.classical_error_mha is not None:
                parts.append(f"C={r.classical_error_mha:.4f}")
            if r.pathB_error_mha is not None:
                parts.append(f"B={r.pathB_error_mha:.4f}")
            if r.pathA_error_mha is not None:
                parts.append(f"A={r.pathA_error_mha:.4f}")
            print(" ".join(parts) + " mHa")

    print(f"\nKey findings:")
    if results:
        # Trotter analysis (needs both C and B)
        pathB_with_C = [r for r in results
                        if r.pathB_energy is not None and r.classical_energy is not None]
        if pathB_with_C:
            avg_trotter = np.mean([abs(r.pathB_energy - r.classical_energy) * 1000
                                   for r in pathB_with_C])
            print(f"  Avg Trotter-2 error (B vs C): {avg_trotter:.4f} mHa")

            pathB_worse = sum(1 for r in pathB_with_C
                              if r.pathB_error_mha > r.classical_error_mha)
            print(f"  Path B worse than Path C: {pathB_worse}/{len(pathB_with_C)} systems")

        if has_A:
            pathA_results = [r for r in results if r.pathA_energy is not None]
            if pathA_results:
                pathA_with_B = [r for r in pathA_results if r.pathB_energy is not None]
                if pathA_with_B:
                    avg_circuit = np.mean([abs(r.pathA_energy - r.pathB_energy) * 1000
                                           for r in pathA_with_B])
                    print(f"  Avg circuit effect (A vs B): {avg_circuit:.4f} mHa")
                pathA_with_C = [r for r in pathA_results if r.classical_energy is not None]
                if pathA_with_C:
                    avg_circuit_vs_c = np.mean(
                        [abs(r.pathA_energy - r.classical_energy) * 1000
                         for r in pathA_with_C])
                    print(f"  Avg circuit effect (A vs C): {avg_circuit_vs_c:.4f} mHa")

                if pathA_with_B:
                    pathA_worse = sum(1 for r in pathA_with_B
                                      if r.pathA_error_mha > r.pathB_error_mha)
                    print(f"  Path A worse than Path B: "
                          f"{pathA_worse}/{len(pathA_with_B)} systems")

    # Skipped systems summary
    skipped = [(r.system, r.n_qubits, r.classical_skip_reason, r.pathB_skip_reason)
               for r in results
               if r.classical_skip_reason or r.pathB_skip_reason]
    if skipped:
        print(f"\n  Skipped paths (VRAM limits):")
        for sys_name, nq, c_skip, b_skip in skipped:
            if c_skip:
                print(f"    {sys_name} ({nq}q) Path C: {c_skip}")
            if b_skip:
                print(f"    {sys_name} ({nq}q) Path B: {b_skip}")

    print(f"\nPaper compliance:")
    print(f"  Time step:     dt = pi / spectral_range (Epperly Theorem 3.1)")
    print(f"  Trotter order: 2nd-order Suzuki-Trotter (paper Section IV)")
    print(f"  Eigenvalue:    Standard (S=I, orthonormal computational basis)")
    print(f"  Basis:         Cumulative union across all Krylov states")
    print(f"\nControlled experiment design:")
    print(f"  All paths share: full 2^n Hilbert space, HF initial state,")
    print(f"  torch.multinomial sampling (seed+k+1000), Slater-Condon diag")
    print(f"  Only variable: time evolution method")
    print(f"\nBackends:")
    if has_C:
        print(f"  Path C: exact e^{{-iHt}} via Lanczos (full 2^n space, no Trotter)")
    if has_B:
        print(f"  Path B: state-vector Trotter-2 on GPU (cos(t)I - i*sin(t)P)")
    if has_A:
        print(f"  Path A: CUDA-Q circuit (nvidia target, fp64, exp_pauli gates)")
        print(f"  Note: Path A uses CUDA-Q internal RNG (seed+k), not torch")


def main():
    all_system_keys = list(SYSTEMS.keys())

    parser = argparse.ArgumentParser(
        description="SKQD Comparison (Paper-Compliant): selectable Path C / B / A"
    )
    parser.add_argument(
        "--systems", nargs="+", default=None,
        choices=all_system_keys,
        help="Molecular systems to compare",
    )
    parser.add_argument(
        "--tier", default=None,
        choices=["small", "medium", "large", "all"],
        help="System tier: small (4-20q), medium (20-24q), large (26-30q), all",
    )
    parser.add_argument(
        "--paths", nargs="+", default=["C", "B", "A"],
        choices=["C", "B", "A"],
        help="Paths to run: C (exact Lanczos), B (classical Trotter), A (CUDA-Q). "
             "Default: all three. Example: --paths A  or  --paths C A",
    )
    parser.add_argument("--krylov-dim", type=int, default=15,
                        help="Max Krylov dimension (default: 15, paper Fig. 1)")
    parser.add_argument("--shots", type=int, default=100_000,
                        help="Shots per Krylov state (default: 100000, paper Section V)")
    parser.add_argument("--trotter-steps", type=int, default=1,
                        help="Trotter sub-steps per Krylov step (default: 1)")
    parser.add_argument("--trotter-order", type=int, default=2, choices=[1, 2],
                        help="Trotter order: 1=first, 2=second (default: 2)")
    parser.add_argument("--profile", action="store_true",
                        help="Report timing breakdown and peak VRAM per path")
    parser.add_argument("--max-vram", type=float, default=None,
                        help="Max VRAM in GB (auto-skip paths exceeding 80%% of this)")
    args = parser.parse_args()

    # Resolve system list from --tier or --systems
    if args.tier is not None:
        if args.tier == "small":
            systems = TIER_SMALL
        elif args.tier == "medium":
            systems = TIER_MEDIUM
        elif args.tier == "large":
            systems = TIER_LARGE
        elif args.tier == "all":
            systems = TIER_SMALL + TIER_MEDIUM + TIER_LARGE
    elif args.systems is not None:
        systems = args.systems
    else:
        # Default: small tier (backward compatible)
        systems = TIER_SMALL

    enabled_paths: Set[str] = set(args.paths)
    enabled_label = "/".join(p for p in ["C", "B", "A"] if p in enabled_paths)

    print(f"SKQD Comparison — Path {enabled_label} (Controlled Experiment)")
    if "C" in enabled_paths:
        print(f"  Path C: Exact Lanczos (no Trotter, full 2^n space)")
    if "B" in enabled_paths:
        print(f"  Path B: State-vector Trotter-{args.trotter_order} (full 2^n space)")
    if "A" in enabled_paths:
        print(f"  Path A: CUDA-Q Trotter-{args.trotter_order} (full 2^n space)")
    print(f"Systems: {', '.join(systems)}")
    print(f"Krylov dim: {args.krylov_dim}, Shots: {args.shots:,}, "
          f"Trotter-{args.trotter_order} ({args.trotter_steps} steps)")
    print(f"Time step: optimal dt = pi / spectral_range (computed per system)")
    if args.profile:
        print(f"Profiling: ENABLED (timing + peak VRAM)")
    if args.max_vram:
        print(f"Max VRAM: {args.max_vram:.1f} GB")
    if "A" in enabled_paths:
        print(f"CUDA-Q available: {CUDAQ_AVAILABLE}")

    results = []
    for system_key in systems:
        try:
            result = run_comparison(
                system_key,
                max_krylov_dim=args.krylov_dim,
                quantum_shots=args.shots,
                num_trotter_steps=args.trotter_steps,
                trotter_order=args.trotter_order,
                enabled_paths=enabled_paths,
                profile=args.profile,
                max_vram_gb=args.max_vram,
            )
            results.append(result)
        except Exception as e:
            print(f"\nERROR on {system_key}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    if results:
        print_summary_table(results, enabled_paths)


if __name__ == "__main__":
    main()
