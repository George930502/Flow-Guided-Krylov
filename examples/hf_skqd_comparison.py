"""
HF-State SKQD: Classical vs Quantum Krylov from Hartree-Fock.

Both classical and quantum SKQD start from the Hartree-Fock (HF) determinant
and build a Krylov subspace via time evolution:

    |ψ_k⟩ = (e^{-iHΔt})^k |HF⟩,   k = 0, 1, ..., d-1

The Hamiltonian is projected onto the sampled basis and diagonalized to
obtain the ground-state energy.

Two solvers:
  - Classical SKQD: exact matrix exponential in particle-conserving subspace
  - Quantum SKQD:  Trotterized evolution in full 2^n Hilbert space
                   (CUDA-Q circuits when available, else classical state-vector)

Paper-compliant parameters (Yu et al., arXiv:2501.09702):
  - dt = π / spectral_range  (Epperly Theorem 3.1)
  - Krylov dimension d = 15
  - Single 2nd-order Suzuki-Trotter step per evolution
  - 10^5 shots per Krylov state

Usage:
    # Run all small systems (H2 through N2)
    uv run python examples/hf_skqd_comparison.py

    # Specific systems
    uv run python examples/hf_skqd_comparison.py --systems h2 lih h2o

    # Only classical SKQD
    uv run python examples/hf_skqd_comparison.py --mode classical

    # Only quantum SKQD
    uv run python examples/hf_skqd_comparison.py --mode quantum

    # Custom Krylov dimension
    uv run python examples/hf_skqd_comparison.py --krylov-dim 10

    # Docker
    docker-compose run --rm flow-krylov-gpu python examples/hf_skqd_comparison.py
"""

import sys
import time
import argparse
import traceback
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

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
from krylov.skqd import SampleBasedKrylovDiagonalization, SKQDConfig
from krylov.quantum_skqd import QuantumCircuitSKQD, QuantumSKQDConfig, CUDAQ_AVAILABLE
from krylov.spectral_utils import compute_optimal_dt


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHEMICAL_ACCURACY_HA = 1.594e-3  # 1 kcal/mol in Hartree

SYSTEM_REGISTRY = {
    "h2": ("H₂", create_h2_hamiltonian, {}),
    "lih": ("LiH", create_lih_hamiltonian, {}),
    "h2o": ("H₂O", create_h2o_hamiltonian, {}),
    "beh2": ("BeH₂", create_beh2_hamiltonian, {}),
    "nh3": ("NH₃", create_nh3_hamiltonian, {}),
    "ch4": ("CH₄", create_ch4_hamiltonian, {}),
    "n2": ("N₂", create_n2_hamiltonian, {}),
}

DEFAULT_SYSTEMS = ["h2", "lih", "h2o", "beh2", "nh3", "ch4", "n2"]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SKQDResult:
    """Result from a single SKQD run."""

    system_name: str
    mode: str  # "classical" or "quantum"
    energy: float
    fci_energy: float
    error_mha: float
    basis_size: int
    krylov_dim: int
    time_step: float
    elapsed_sec: float
    passed_chemical_accuracy: bool


# ---------------------------------------------------------------------------
# Classical SKQD (particle-conserving subspace, exact exp)
# ---------------------------------------------------------------------------

def run_classical_skqd(
    hamiltonian,
    max_krylov_dim: int = 15,
    shots_per_krylov: int = 100_000,
) -> Dict[str, Any]:
    """
    Run classical SKQD starting from HF state.

    Time evolution uses exact matrix exponential via Lanczos
    in the particle-conserving subspace (10-100x smaller than full 2^n).
    """
    # Compute optimal time step
    dt, spectral_range = compute_optimal_dt(hamiltonian)
    print(f"  dt = π/ΔE = {dt:.6f}  (spectral range: {spectral_range:.4f} Ha)")

    config = SKQDConfig(
        max_krylov_dim=max_krylov_dim,
        time_step=dt,
        shots_per_krylov=shots_per_krylov,
        use_gpu=torch.cuda.is_available(),
        seed=42,
    )

    solver = SampleBasedKrylovDiagonalization(
        hamiltonian=hamiltonian,
        config=config,
        initial_state=hamiltonian.get_hf_state(),
    )

    results = solver.run(progress=True)
    best_energy = min(results["energies"])
    final_basis_size = results["basis_sizes"][-1] if results["basis_sizes"] else 0

    return {
        "energy": best_energy,
        "energies": results["energies"],
        "krylov_dims": results["krylov_dims"],
        "basis_sizes": results["basis_sizes"],
        "final_basis_size": final_basis_size,
        "time_step": dt,
    }


# ---------------------------------------------------------------------------
# Quantum SKQD (full Hilbert space, Trotterized)
# ---------------------------------------------------------------------------

def run_quantum_skqd(
    hamiltonian,
    max_krylov_dim: int = 15,
    num_trotter_steps: int = 1,
    shots: int = 100_000,
) -> Dict[str, Any]:
    """
    Run quantum SKQD starting from HF state.

    Time evolution uses 2nd-order Suzuki-Trotter decomposition
    in the full 2^n Hilbert space. Three backends:
      - CUDA-Q circuits (Path A) when available
      - Classical state-vector Trotter (Path B) for small systems
      - GPU Lanczos (Path C) as gold-standard fallback
    """
    # Compute optimal time step
    dt, spectral_range = compute_optimal_dt(hamiltonian)
    print(f"  dt = π/ΔE = {dt:.6f}  (spectral range: {spectral_range:.4f} Ha)")

    config = QuantumSKQDConfig(
        max_krylov_dim=max_krylov_dim,
        total_evolution_time=dt,
        num_trotter_steps=num_trotter_steps,
        shots=shots,
        initial_state="hf",
        backend="auto",
    )

    solver = QuantumCircuitSKQD.from_molecular_hamiltonian(
        hamiltonian, config=config
    )

    results = solver.run(progress=True)

    return {
        "energy": results["best_energy"],
        "energies": results["energies"],
        "krylov_dims": results["krylov_dims"],
        "basis_sizes": results["basis_sizes"],
        "final_basis_size": results["basis_sizes"][-1] if results["basis_sizes"] else 0,
        "time_step": dt,
        "backend": results["backend"],
    }


# ---------------------------------------------------------------------------
# Main comparison loop
# ---------------------------------------------------------------------------

def run_comparison(
    systems: List[str],
    mode: str = "both",
    max_krylov_dim: int = 15,
    num_trotter_steps: int = 1,
    shots: int = 100_000,
) -> List[SKQDResult]:
    """
    Run HF-state SKQD comparison across molecular systems.

    Args:
        systems: List of system keys (e.g., ["h2", "lih", "h2o"])
        mode: "classical", "quantum", or "both"
        max_krylov_dim: Krylov subspace dimension
        num_trotter_steps: Trotter steps per evolution (quantum only)
        shots: Measurement shots per Krylov state

    Returns:
        List of SKQDResult objects
    """
    all_results: List[SKQDResult] = []

    print("=" * 72)
    print("HF-State SKQD Comparison: Classical vs Quantum")
    print("=" * 72)
    print(f"  Mode:       {mode}")
    print(f"  Krylov dim: {max_krylov_dim}")
    print(f"  Shots:      {shots:,}")
    if mode in ("quantum", "both"):
        print(f"  Trotter:    {num_trotter_steps} step(s), 2nd-order Suzuki-Trotter")
        print(f"  CUDA-Q:     {'available' if CUDAQ_AVAILABLE else 'not available (classical fallback)'}")
    print(f"  GPU:        {'CUDA ' + torch.version.cuda if torch.cuda.is_available() else 'CPU'}")
    print()

    for sys_key in systems:
        if sys_key not in SYSTEM_REGISTRY:
            print(f"Unknown system: {sys_key}, skipping.")
            continue

        display_name, factory_fn, kwargs = SYSTEM_REGISTRY[sys_key]

        print("-" * 72)
        print(f"System: {display_name} ({sys_key})")
        print("-" * 72)

        # Build Hamiltonian
        H = factory_fn(**kwargs)
        n_qubits = H.num_sites
        n_configs = H.hilbert_dim
        print(f"  Qubits: {n_qubits}  |  Configs: {n_configs:,}")

        # FCI reference energy
        fci_energy = H.fci_energy()
        print(f"  FCI energy: {fci_energy:.8f} Ha")

        # --- Classical SKQD ---
        if mode in ("classical", "both"):
            print(f"\n  [Classical SKQD] (particle-conserving subspace, exact exp)")
            t0 = time.perf_counter()
            try:
                c_results = run_classical_skqd(
                    H,
                    max_krylov_dim=max_krylov_dim,
                    shots_per_krylov=shots,
                )
                elapsed = time.perf_counter() - t0
                error_mha = abs(c_results["energy"] - fci_energy) * 1000
                passed = error_mha < CHEMICAL_ACCURACY_HA * 1000

                result = SKQDResult(
                    system_name=display_name,
                    mode="classical",
                    energy=c_results["energy"],
                    fci_energy=fci_energy,
                    error_mha=error_mha,
                    basis_size=c_results["final_basis_size"],
                    krylov_dim=max_krylov_dim,
                    time_step=c_results["time_step"],
                    elapsed_sec=elapsed,
                    passed_chemical_accuracy=passed,
                )
                all_results.append(result)

                status = "PASS" if passed else "FAIL"
                print(f"  Energy: {c_results['energy']:.8f} Ha")
                print(f"  Error:  {error_mha:.4f} mHa  [{status}]")
                print(f"  Basis:  {c_results['final_basis_size']:,} configs")
                print(f"  Time:   {elapsed:.1f}s")
            except Exception as e:
                print(f"  FAILED: {e}")
                traceback.print_exc()

        # --- Quantum SKQD ---
        if mode in ("quantum", "both"):
            print(f"\n  [Quantum SKQD] (full 2^n space, Trotterized)")
            t0 = time.perf_counter()
            try:
                q_results = run_quantum_skqd(
                    H,
                    max_krylov_dim=max_krylov_dim,
                    num_trotter_steps=num_trotter_steps,
                    shots=shots,
                )
                elapsed = time.perf_counter() - t0
                error_mha = abs(q_results["energy"] - fci_energy) * 1000
                passed = error_mha < CHEMICAL_ACCURACY_HA * 1000

                result = SKQDResult(
                    system_name=display_name,
                    mode="quantum",
                    energy=q_results["energy"],
                    fci_energy=fci_energy,
                    error_mha=error_mha,
                    basis_size=q_results["final_basis_size"],
                    krylov_dim=max_krylov_dim,
                    time_step=q_results["time_step"],
                    elapsed_sec=elapsed,
                    passed_chemical_accuracy=passed,
                )
                all_results.append(result)

                status = "PASS" if passed else "FAIL"
                backend = q_results.get("backend", "unknown")
                print(f"  Backend: {backend}")
                print(f"  Energy:  {q_results['energy']:.8f} Ha")
                print(f"  Error:   {error_mha:.4f} mHa  [{status}]")
                print(f"  Basis:   {q_results['final_basis_size']:,} configs")
                print(f"  Time:    {elapsed:.1f}s")
            except Exception as e:
                print(f"  FAILED: {e}")
                traceback.print_exc()

        print()

    # --- Summary table ---
    print_summary_table(all_results)

    return all_results


def print_summary_table(results: List[SKQDResult]) -> None:
    """Print a formatted summary table of all results."""
    if not results:
        print("No results to display.")
        return

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)

    # Group by system
    systems_seen = []
    for r in results:
        if r.system_name not in systems_seen:
            systems_seen.append(r.system_name)

    # Header
    has_classical = any(r.mode == "classical" for r in results)
    has_quantum = any(r.mode == "quantum" for r in results)

    header = f"{'System':<8}"
    if has_classical:
        header += f"  {'Classical (mHa)':>15}  {'Time':>6}"
    if has_quantum:
        header += f"  {'Quantum (mHa)':>14}  {'Time':>6}"
    header += f"  {'Status':>6}"
    print(header)
    print("-" * len(header))

    for sys_name in systems_seen:
        sys_results = [r for r in results if r.system_name == sys_name]
        c_result = next((r for r in sys_results if r.mode == "classical"), None)
        q_result = next((r for r in sys_results if r.mode == "quantum"), None)

        row = f"{sys_name:<8}"
        all_passed = True

        if has_classical:
            if c_result:
                row += f"  {c_result.error_mha:>15.4f}  {c_result.elapsed_sec:>5.1f}s"
                if not c_result.passed_chemical_accuracy:
                    all_passed = False
            else:
                row += f"  {'---':>15}  {'---':>6}"

        if has_quantum:
            if q_result:
                row += f"  {q_result.error_mha:>14.4f}  {q_result.elapsed_sec:>5.1f}s"
                if not q_result.passed_chemical_accuracy:
                    all_passed = False
            else:
                row += f"  {'---':>14}  {'---':>6}"

        status = "PASS" if all_passed else "FAIL"
        row += f"  {status:>6}"
        print(row)

    # Chemical accuracy threshold
    n_passed = sum(1 for r in results if r.passed_chemical_accuracy)
    n_total = len(results)
    print()
    print(f"Chemical accuracy threshold: {CHEMICAL_ACCURACY_HA * 1000:.3f} mHa (1 kcal/mol)")
    print(f"Passed: {n_passed}/{n_total}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="HF-State SKQD: Classical vs Quantum Krylov from Hartree-Fock",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples/hf_skqd_comparison.py                           # All systems, both modes
  python examples/hf_skqd_comparison.py --systems h2 lih          # Specific systems
  python examples/hf_skqd_comparison.py --mode classical           # Classical only
  python examples/hf_skqd_comparison.py --mode quantum             # Quantum only
  python examples/hf_skqd_comparison.py --krylov-dim 10            # Custom Krylov dim
        """,
    )

    parser.add_argument(
        "--systems",
        nargs="+",
        default=DEFAULT_SYSTEMS,
        choices=list(SYSTEM_REGISTRY.keys()),
        help="Molecular systems to benchmark (default: all)",
    )
    parser.add_argument(
        "--mode",
        choices=["classical", "quantum", "both"],
        default="both",
        help="Which SKQD solver(s) to run (default: both)",
    )
    parser.add_argument(
        "--krylov-dim",
        type=int,
        default=15,
        help="Maximum Krylov subspace dimension (default: 15, paper value)",
    )
    parser.add_argument(
        "--trotter-steps",
        type=int,
        default=1,
        help="Trotter steps per evolution for quantum SKQD (default: 1)",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=100_000,
        help="Measurement shots per Krylov state (default: 100,000)",
    )

    args = parser.parse_args()

    run_comparison(
        systems=args.systems,
        mode=args.mode,
        max_krylov_dim=args.krylov_dim,
        num_trotter_steps=args.trotter_steps,
        shots=args.shots,
    )


if __name__ == "__main__":
    main()
