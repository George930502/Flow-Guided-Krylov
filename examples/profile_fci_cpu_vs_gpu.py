#!/usr/bin/env python3
"""
Profile FCI computation methods across all molecular systems.

Compares two reliable FCI paths:
1. CPU matrix diag: H.fci_energy() — build full matrix, numpy/scipy eigensolver
2. PySCF Davidson: compute_pyscf_fci(geometry, basis) — iterative Davidson solver

Note: GPU FCI via embedded CUDA kernels (gpu_fci.py) was found to give WRONG
energies for several systems (H2O 2.6 Ha off, N2 15 mHa). It is excluded
from the reference energy hierarchy until the kernel bugs are fixed.

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
    compute_pyscf_fci,
    create_co_molecule,
    create_hcn_molecule,
    create_c2h2_molecule,
)


@dataclass
class FCIProfile:
    system: str
    n_qubits: int
    n_configs: int
    matrix_time: Optional[float]     # H.fci_energy() CPU matrix diag
    pyscf_time: Optional[float]      # PySCF iterative Davidson
    matrix_energy: Optional[float]
    pyscf_energy: Optional[float]
    energy_diff_mha: Optional[float]


def cpu_matrix_fci(H) -> tuple[float, float]:
    """Compute FCI via H.fci_energy() (matrix diag). Returns (energy, time)."""
    t0 = time.time()
    energy = H.fci_energy()
    elapsed = time.time() - t0
    return energy, elapsed


def pyscf_fci(geometry, basis) -> tuple[float, float]:
    """Compute FCI via PySCF iterative Davidson. Returns (energy, time)."""
    t0 = time.time()
    energy = compute_pyscf_fci(geometry, basis)
    elapsed = time.time() - t0
    return energy, elapsed


def _get_geometry(name, **kwargs):
    """Get geometry + basis for small-tier systems."""
    if name == "H2":
        bl = kwargs.get("bond_length", 0.74)
        return [("H", (0., 0., 0.)), ("H", (0., 0., bl))], "sto-3g"
    elif name == "LiH":
        bl = kwargs.get("bond_length", 1.6)
        return [("Li", (0., 0., 0.)), ("H", (0., 0., bl))], "sto-3g"
    elif name == "H2O":
        oh, ang = 0.96, np.radians(104.5)
        return [("O", (0., 0., 0.)), ("H", (oh, 0., 0.)),
                ("H", (oh*np.cos(ang), oh*np.sin(ang), 0.))], "sto-3g"
    elif name == "BeH2":
        bl = 1.33
        return [("Be", (0., 0., 0.)), ("H", (0., 0., bl)), ("H", (0., 0., -bl))], "sto-3g"
    elif name == "NH3":
        nh, ang = 1.01, np.radians(107.8)
        h = nh * np.cos(np.arcsin(np.sin(ang/2) / np.sin(np.radians(60))))
        r = np.sqrt(nh**2 - h**2)
        return [("N", (0., 0., h)), ("H", (r, 0., 0.)),
                ("H", (r*np.cos(np.radians(120)), r*np.sin(np.radians(120)), 0.)),
                ("H", (r*np.cos(np.radians(240)), r*np.sin(np.radians(240)), 0.))], "sto-3g"
    elif name == "CH4":
        a = 1.09 / np.sqrt(3)
        return [("C", (0., 0., 0.)), ("H", (a, a, a)), ("H", (a, -a, -a)),
                ("H", (-a, a, -a)), ("H", (-a, -a, a))], "sto-3g"
    elif name == "N2":
        bl = kwargs.get("bond_length", 1.10)
        return [("N", (0., 0., 0.)), ("N", (0., 0., bl))], "sto-3g"
    return None, None


def profile_system(name, H, n_qubits, geometry, basis) -> FCIProfile:
    """Profile matrix diag vs PySCF Davidson for a single system."""
    n_configs = comb(H.n_orbitals, H.n_alpha) * comb(H.n_orbitals, H.n_beta)
    print(f"\n{'='*60}")
    print(f"  {name}: {n_qubits} qubits, {n_configs:,} configs")
    print(f"{'='*60}")

    # --- CPU matrix diag ---
    matrix_energy, matrix_time = None, None
    try:
        print(f"  Matrix diag (H.fci_energy)... ", end="", flush=True)
        matrix_energy, matrix_time = cpu_matrix_fci(H)
        print(f"{matrix_time:.3f}s  E = {matrix_energy:.8f} Ha")
    except Exception as e:
        print(f"FAILED: {e}")

    # --- PySCF Davidson ---
    pyscf_energy_val, pyscf_time_val = None, None
    if geometry is not None:
        try:
            print(f"  PySCF Davidson... ", end="", flush=True)
            pyscf_energy_val, pyscf_time_val = pyscf_fci(geometry, basis)
            print(f"{pyscf_time_val:.3f}s  E = {pyscf_energy_val:.8f} Ha")
        except Exception as e:
            print(f"FAILED: {e}")

    # --- Compare ---
    energy_diff = None
    if matrix_energy is not None and pyscf_energy_val is not None:
        energy_diff = abs(matrix_energy - pyscf_energy_val) * 1000
        speedup = matrix_time / pyscf_time_val if pyscf_time_val > 0 else float('inf')
        winner = "PySCF" if pyscf_time_val < matrix_time else "Matrix"
        print(f"  Energy diff: {energy_diff:.6f} mHa")
        print(f"  Speedup: {speedup:.2f}x ({winner} faster)")

    return FCIProfile(
        system=name, n_qubits=n_qubits, n_configs=n_configs,
        matrix_time=matrix_time, pyscf_time=pyscf_time_val,
        matrix_energy=matrix_energy, pyscf_energy=pyscf_energy_val,
        energy_diff_mha=energy_diff,
    )


def main():
    print("=" * 60)
    print("  FCI Profiling: Matrix Diag vs PySCF Davidson")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    systems = [
        ("H2",   4,  lambda: create_h2_hamiltonian(device=device),  *_get_geometry("H2")),
        ("LiH",  12, lambda: create_lih_hamiltonian(device=device), *_get_geometry("LiH")),
        ("H2O",  14, lambda: create_h2o_hamiltonian(device=device), *_get_geometry("H2O")),
        ("BeH2", 14, lambda: create_beh2_hamiltonian(device=device), *_get_geometry("BeH2")),
        ("NH3",  16, lambda: create_nh3_hamiltonian(device=device), *_get_geometry("NH3")),
        ("CH4",  18, lambda: create_ch4_hamiltonian(device=device), *_get_geometry("CH4")),
        ("N2",   20, lambda: create_n2_hamiltonian(device=device),  *_get_geometry("N2")),
    ]

    medium_factories = [
        ("CO",   20, lambda: create_co_molecule(device=device)),
        ("HCN",  22, lambda: create_hcn_molecule(device=device)),
        ("C2H2", 24, lambda: create_c2h2_molecule(device=device)),
    ]

    results = []

    for name, n_qubits, factory, geometry, basis in systems:
        try:
            H = factory()
            result = profile_system(name, H, n_qubits, geometry, basis)
            results.append(result)
        except Exception as e:
            print(f"\n  {name}: SKIPPED ({e})")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for name, n_qubits, mol_factory in medium_factories:
        try:
            mol_data = mol_factory()
            result = profile_system(name, mol_data.hamiltonian, n_qubits,
                                    mol_data.geometry, mol_data.basis)
            results.append(result)
        except Exception as e:
            print(f"\n  {name}: SKIPPED ({e})")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Summary ---
    print("\n\n" + "=" * 95)
    print("  SUMMARY: Matrix Diag vs PySCF Davidson")
    print("=" * 95)
    print(f"  {'System':<8} {'Qubits':>6} {'Configs':>10} "
          f"{'Matrix(s)':>10} {'PySCF(s)':>10} {'Speedup':>10} {'dE (mHa)':>10} {'Winner':>8}")
    print("-" * 95)

    crossover_configs = None

    for r in sorted(results, key=lambda x: x.n_configs):
        m_str = f"{r.matrix_time:.3f}" if r.matrix_time is not None else "OOM"
        p_str = f"{r.pyscf_time:.3f}" if r.pyscf_time is not None else "---"

        if r.matrix_time is not None and r.pyscf_time is not None:
            speedup = r.matrix_time / r.pyscf_time
            speedup_str = f"{speedup:.2f}x"
            winner = "PySCF" if r.pyscf_time < r.matrix_time else "Matrix"
            if winner == "PySCF" and crossover_configs is None:
                crossover_configs = r.n_configs
        else:
            speedup_str = "---"
            winner = "PySCF" if r.pyscf_time is not None else "---"

        de_str = f"{r.energy_diff_mha:.6f}" if r.energy_diff_mha is not None else "---"

        print(f"  {r.system:<8} {r.n_qubits:>6} {r.n_configs:>10,} "
              f"{m_str:>10} {p_str:>10} {speedup_str:>10} {de_str:>10} {winner:>8}")

    print("-" * 95)

    if crossover_configs:
        print(f"\n  PySCF Davidson faster at >= {crossover_configs:,} configs.")
    print(f"  Recommendation: use matrix diag for <=5K, PySCF Davidson for >5K.")

    # Energy agreement check
    max_diff = max((r.energy_diff_mha for r in results if r.energy_diff_mha is not None), default=0)
    if max_diff < 0.01:
        print(f"  Energy agreement: EXACT (max diff = {max_diff:.6f} mHa)")
    elif max_diff < 1.594:
        print(f"  Energy agreement: within chemical accuracy (max diff = {max_diff:.3f} mHa)")
    else:
        print(f"  WARNING: Energy disagreement > chemical accuracy (max diff = {max_diff:.3f} mHa)")


if __name__ == "__main__":
    main()
