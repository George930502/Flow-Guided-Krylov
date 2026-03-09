#!/usr/bin/env python3
"""
Profile CPU FCI vs GPU FCI across all molecular systems.

Finds the crossover point where GPU FCI becomes faster than CPU FCI,
then verifies both give the same energy (within tolerance).

Usage:
    uv run python examples/profile_fci_cpu_vs_gpu.py
"""

import sys
import os
import time
import numpy as np
import torch
from math import comb
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from hamiltonians.molecular import (
    create_h2_hamiltonian,
    create_lih_hamiltonian,
    create_h2o_hamiltonian,
    create_beh2_hamiltonian,
    create_nh3_hamiltonian,
    create_ch4_hamiltonian,
    create_n2_hamiltonian,
)
from moderate_system_benchmark import (
    create_co_molecule,
    create_hcn_molecule,
    create_c2h2_molecule,
)


@dataclass
class FCIProfile:
    system: str
    n_qubits: int
    n_configs: int
    cpu_time: Optional[float]
    gpu_time: Optional[float]
    cpu_energy: Optional[float]
    gpu_energy: Optional[float]
    energy_diff_mha: Optional[float]  # |CPU - GPU| in mHa


def cpu_fci_energy(H) -> tuple[float, float]:
    """Compute FCI on CPU via matrix diag. Returns (energy, time_seconds)."""
    from itertools import combinations
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh

    n_orb = H.n_orbitals
    n_alpha = H.n_alpha
    n_beta = H.n_beta

    alpha_configs = list(combinations(range(n_orb), n_alpha))
    beta_configs = list(combinations(range(n_orb), n_beta))

    basis_configs = []
    for ac in alpha_configs:
        for bc in beta_configs:
            config = torch.zeros(H.num_sites, dtype=torch.long)
            for i in ac:
                config[i] = 1
            for i in bc:
                config[i + n_orb] = 1
            basis_configs.append(config)

    basis_tensor = torch.stack(basis_configs).to(H.device)
    n = len(basis_configs)

    t0 = time.time()

    # Build matrix (on GPU if available, same as H.fci_energy())
    H_fci = H.matrix_elements(basis_tensor, basis_tensor)
    H_np = H_fci.cpu().numpy().astype(np.float64)
    H_np = 0.5 * (H_np + H_np.T)

    # Diag on CPU
    if n <= 2000:
        eigenvalues, _ = np.linalg.eigh(H_np)
        energy = float(eigenvalues[0])
    else:
        H_sp = csr_matrix(H_np)
        eigenvalues, _ = eigsh(H_sp, k=1, which='SA', tol=1e-12)
        energy = float(eigenvalues[0])

    elapsed = time.time() - t0
    return energy, elapsed


def gpu_fci_energy(H) -> tuple[float, float]:
    """Compute FCI on GPU via CuPy CUDA kernels + PySCF Davidson. Returns (energy, time)."""
    from utils.gpu_fci import compute_gpu_fci_from_integrals, GPU_FCI_AVAILABLE

    if not GPU_FCI_AVAILABLE:
        raise RuntimeError("GPU FCI not available (CuPy not installed)")

    h1e = H.h1e.cpu().numpy() if hasattr(H.h1e, 'cpu') else H.h1e
    h2e = H.h2e.cpu().numpy() if hasattr(H.h2e, 'cpu') else H.h2e

    t0 = time.time()
    energy = compute_gpu_fci_from_integrals(
        h1e=h1e,
        h2e=h2e,
        n_orbitals=H.n_orbitals,
        n_alpha=H.n_alpha,
        n_beta=H.n_beta,
        nuclear_repulsion=H.nuclear_repulsion,
        conv_tol=1e-10,
        max_cycle=300,
    )
    elapsed = time.time() - t0
    return energy, elapsed


def profile_system(name, H, n_qubits) -> FCIProfile:
    """Profile CPU vs GPU FCI for a single system."""
    n_configs = comb(H.n_orbitals, H.n_alpha) * comb(H.n_orbitals, H.n_beta)
    print(f"\n{'='*60}")
    print(f"  {name}: {n_qubits} qubits, {n_configs:,} configs")
    print(f"{'='*60}")

    # --- CPU FCI ---
    cpu_energy_val, cpu_time = None, None
    try:
        print(f"  CPU FCI (matrix diag)... ", end="", flush=True)
        cpu_energy_val, cpu_time = cpu_fci_energy(H)
        print(f"{cpu_time:.3f}s  E = {cpu_energy_val:.8f} Ha")
    except Exception as e:
        print(f"FAILED: {e}")

    # --- GPU FCI ---
    gpu_energy_val, gpu_time = None, None
    try:
        # Warmup run (JIT compilation of CUDA kernels on first call)
        print(f"  GPU FCI warmup... ", end="", flush=True)
        _, warmup_t = gpu_fci_energy(H)
        print(f"{warmup_t:.3f}s")

        print(f"  GPU FCI (CuPy Davidson)... ", end="", flush=True)
        gpu_energy_val, gpu_time = gpu_fci_energy(H)
        print(f"{gpu_time:.3f}s  E = {gpu_energy_val:.8f} Ha")
    except Exception as e:
        print(f"FAILED: {e}")

    # --- Compare ---
    energy_diff = None
    if cpu_energy_val is not None and gpu_energy_val is not None:
        energy_diff = abs(cpu_energy_val - gpu_energy_val) * 1000  # mHa
        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        winner = "GPU" if gpu_time < cpu_time else "CPU"
        print(f"  Energy diff: {energy_diff:.6f} mHa")
        print(f"  Speedup: {speedup:.2f}x ({winner} faster)")

    return FCIProfile(
        system=name,
        n_qubits=n_qubits,
        n_configs=n_configs,
        cpu_time=cpu_time,
        gpu_time=gpu_time,
        cpu_energy=cpu_energy_val,
        gpu_energy=gpu_energy_val,
        energy_diff_mha=energy_diff,
    )


def main():
    print("=" * 60)
    print("  FCI Profiling: CPU (matrix diag) vs GPU (CuPy Davidson)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Check GPU FCI availability
    try:
        from utils.gpu_fci import GPU_FCI_AVAILABLE
        print(f"  GPU FCI available: {GPU_FCI_AVAILABLE}")
    except ImportError:
        GPU_FCI_AVAILABLE = False
        print(f"  GPU FCI available: False")

    # --- Systems ordered by size ---
    systems = [
        ("H2",   4,  lambda: create_h2_hamiltonian(device=device)),
        ("LiH",  12, lambda: create_lih_hamiltonian(device=device)),
        ("H2O",  14, lambda: create_h2o_hamiltonian(device=device)),
        ("BeH2", 14, lambda: create_beh2_hamiltonian(device=device)),
        ("NH3",  16, lambda: create_nh3_hamiltonian(device=device)),
        ("N2",   20, lambda: create_n2_hamiltonian(device=device)),
        ("CH4",  18, lambda: create_ch4_hamiltonian(device=device)),
    ]

    # Add medium-tier systems (need mol_data factory)
    medium_systems = [
        ("CO",   20, lambda: create_co_molecule(device=device).hamiltonian),
        ("HCN",  22, lambda: create_hcn_molecule(device=device).hamiltonian),
        ("C2H2", 24, lambda: create_c2h2_molecule(device=device).hamiltonian),
    ]

    results = []

    for name, n_qubits, factory in systems + medium_systems:
        try:
            H = factory()
            result = profile_system(name, H, n_qubits)
            results.append(result)
        except Exception as e:
            print(f"\n  {name}: SKIPPED ({e})")

        # Free GPU memory between systems
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Summary table ---
    print("\n\n" + "=" * 90)
    print("  SUMMARY: CPU vs GPU FCI Profiling")
    print("=" * 90)
    print(f"  {'System':<8} {'Qubits':>6} {'Configs':>10} "
          f"{'CPU (s)':>10} {'GPU (s)':>10} {'Speedup':>10} {'dE (mHa)':>10} {'Winner':>8}")
    print("-" * 90)

    crossover_qubits = None

    for r in sorted(results, key=lambda x: x.n_configs):
        cpu_str = f"{r.cpu_time:.3f}" if r.cpu_time is not None else "---"
        gpu_str = f"{r.gpu_time:.3f}" if r.gpu_time is not None else "---"

        if r.cpu_time is not None and r.gpu_time is not None:
            speedup = r.cpu_time / r.gpu_time
            speedup_str = f"{speedup:.2f}x"
            winner = "GPU" if r.gpu_time < r.cpu_time else "CPU"
            if winner == "GPU" and crossover_qubits is None:
                crossover_qubits = r.n_qubits
        else:
            speedup_str = "---"
            winner = "---"

        de_str = f"{r.energy_diff_mha:.6f}" if r.energy_diff_mha is not None else "---"

        print(f"  {r.system:<8} {r.n_qubits:>6} {r.n_configs:>10,} "
              f"{cpu_str:>10} {gpu_str:>10} {speedup_str:>10} {de_str:>10} {winner:>8}")

    print("-" * 90)
    if crossover_qubits is not None:
        print(f"\n  GPU becomes faster at {crossover_qubits} qubits.")
        print(f"  Recommended threshold: use GPU FCI for >= {crossover_qubits} qubits")
    else:
        print("\n  CPU was faster for all tested systems (GPU crossover not found).")

    # Check energy agreement
    max_diff = max((r.energy_diff_mha for r in results if r.energy_diff_mha is not None), default=0)
    if max_diff < 0.001:  # < 0.001 mHa = exact agreement
        print(f"  Energy agreement: EXACT (max diff = {max_diff:.6f} mHa)")
    elif max_diff < 1.594:
        print(f"  Energy agreement: within chemical accuracy (max diff = {max_diff:.6f} mHa)")
    else:
        print(f"  WARNING: Energy disagreement exceeds chemical accuracy! (max diff = {max_diff:.6f} mHa)")


if __name__ == "__main__":
    main()
