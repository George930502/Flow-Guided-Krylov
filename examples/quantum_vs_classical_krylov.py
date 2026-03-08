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

Reference:
    NVIDIA CUDA-Q SKQD tutorial:
    nvidia.github.io/cuda-quantum/latest/applications/python/skqd.html

Usage:
    python examples/quantum_vs_classical_krylov.py --systems h2 lih
    python examples/quantum_vs_classical_krylov.py --systems h2 lih h2o beh2
    docker-compose run --rm flow-krylov-gpu python examples/quantum_vs_classical_krylov.py
"""

import sys
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

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

SYSTEMS = {
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
}


@dataclass
class ComparisonResult:
    """Results from a 3-way comparison run."""
    system: str
    n_qubits: int
    n_configs: int
    n_pauli_terms: int
    fci_energy: float
    spectral_range: float
    optimal_dt: float
    # Direct diag baseline
    direct_energy: float
    direct_error_mha: float
    # Path C: Classical SKQD (exact time evolution)
    classical_energy: float
    classical_error_mha: float
    classical_time_s: float
    classical_basis_size: int
    # Path B: Classical Trotterized (state-vector, second-order)
    pathB_energy: float
    pathB_error_mha: float
    pathB_time_s: float
    pathB_basis_size: int
    # Path A: CUDA-Q circuit (if available, second-order)
    pathA_energy: Optional[float]
    pathA_error_mha: Optional[float]
    pathA_time_s: Optional[float]
    pathA_basis_size: Optional[int]
    pathA_available: bool


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
    """
    Run quantum-circuit SKQD with specified backend.

    Args:
        krylov_dt: Evolution time per Krylov step (= total_evolution_time)
        num_trotter_steps: Number of Trotter sub-steps per Krylov step
        trotter_order: 1 or 2 (Suzuki-Trotter order)
        backend: "cudaq" for Path A, "classical" for Path B
    """
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
) -> ComparisonResult:
    """Run 3-way comparison: Path C (Classical) vs Path B (Trotter) vs Path A (CUDA-Q)."""
    info = SYSTEMS[system_key]
    name = info["name"]

    print(f"\n{'=' * 70}")
    print(f"  {name}: 3-Way SKQD Comparison (Paper-Compliant)")
    print(f"{'=' * 70}")

    H = info["factory"](**info["kwargs"])
    n_qubits = H.num_sites
    n_orb = H.n_orbitals
    from math import comb
    n_configs = comb(n_orb, H.n_alpha) * comb(n_orb, H.n_beta)
    fci_energy = H.fci_energy()

    print(f"  Qubits: {n_qubits}, Orbitals: {n_orb}, Configs: {n_configs:,}")
    print(f"  FCI energy: {fci_energy:.8f} Ha")

    # Compute optimal time step from spectral range (paper Theorem 3.1)
    optimal_dt, spectral_range = _compute_optimal_dt(H)
    print(f"  Spectral range: {spectral_range:.4f} Ha")
    print(f"  Optimal dt (pi/dE): {optimal_dt:.6f}")
    print(f"  Trotter order: {trotter_order}, Trotter steps: {num_trotter_steps}")

    # Direct diag baseline
    nf_basis = _generate_essential_configs(H)
    print(f"  Essential configs: {len(nf_basis)}")
    H_proj = H.matrix_elements(nf_basis, nf_basis)
    H_np = H_proj.detach().cpu().numpy().real
    H_np = 0.5 * (H_np + H_np.T)
    direct_energy = float(np.linalg.eigh(H_np)[0][0])
    direct_error = abs(direct_energy - fci_energy) * 1000
    print(f"  Direct diag (no Krylov): {direct_energy:.8f} Ha, error: {direct_error:.4f} mHa")

    n_pauli_terms = 0

    # ------------------------------------------------------------------
    # 1. Path C: Exact evolution (Lanczos, same 2^n space as Path A/B)
    # ------------------------------------------------------------------
    print(f"\n{'─' * 60}")
    print(f"  [1/3] Path C: Exact Lanczos (no Trotter, dt={optimal_dt:.6f})")
    print(f"{'─' * 60}")

    t0 = time.time()
    pathC_results = _run_quantum_skqd(
        H, max_krylov_dim, optimal_dt, num_trotter_steps, trotter_order,
        quantum_shots, backend="exact", verbose=verbose,
    )
    classical_time = time.time() - t0
    classical_energy = pathC_results["best_energy"]
    classical_error = abs(classical_energy - fci_energy) * 1000
    classical_basis_size = (
        pathC_results["basis_sizes"][-1] if pathC_results["basis_sizes"] else 0
    )
    if n_pauli_terms == 0:
        n_pauli_terms = pathC_results["n_pauli_terms"]

    print(f"  Energy: {classical_energy:.8f} Ha")
    print(f"  Error:  {classical_error:.4f} mHa | Time: {classical_time:.1f}s | "
          f"Basis: {classical_basis_size}")

    # ------------------------------------------------------------------
    # 2. Path B: Classical Trotterized (state-vector, GPU, 2nd-order)
    #    Skip when 2^n > 100K — phase_table OOMs (n_terms × dim × 16 bytes)
    # ------------------------------------------------------------------
    # Estimate phase_table memory: n_pauli_terms not yet known, use heuristic
    full_dim = 2 ** n_qubits
    pathB_feasible = full_dim <= 100_000  # ~16 qubits max
    pathB_energy = None
    pathB_error = None
    pathB_time = None
    pathB_basis_size = None

    if pathB_feasible:
        print(f"\n{'─' * 60}")
        print(f"  [2/3] Path B: Trotterized State-Vector (order={trotter_order}, "
              f"dt={optimal_dt:.6f})")
        print(f"{'─' * 60}")

        t0 = time.time()
        pathB_results = _run_quantum_skqd(
            H, max_krylov_dim, optimal_dt, num_trotter_steps, trotter_order,
            quantum_shots, backend="classical", verbose=verbose,
        )
        pathB_time = time.time() - t0
        pathB_energy = pathB_results["best_energy"]
        pathB_error = abs(pathB_energy - fci_energy) * 1000
        pathB_basis_size = (
            pathB_results["basis_sizes"][-1] if pathB_results["basis_sizes"] else 0
        )
        n_pauli_terms = pathB_results["n_pauli_terms"]

        print(f"  Energy: {pathB_energy:.8f} Ha")
        print(f"  Error:  {pathB_error:.4f} mHa | Time: {pathB_time:.1f}s | "
              f"Basis: {pathB_basis_size} | Pauli terms: {n_pauli_terms}")
    else:
        print(f"\n{'─' * 60}")
        print(f"  [2/3] Path B: SKIPPED (2^{n_qubits}={full_dim:,} state-vector "
              f"too large for phase_table)")
        print(f"{'─' * 60}")

    # ------------------------------------------------------------------
    # 3. Path A: CUDA-Q Circuit (if available, 2nd-order)
    # ------------------------------------------------------------------
    pathA_energy = None
    pathA_error = None
    pathA_time = None
    pathA_basis_size = None
    pathA_available = CUDAQ_AVAILABLE

    if CUDAQ_AVAILABLE:
        print(f"\n{'─' * 60}")
        print(f"  [3/3] Path A: CUDA-Q Circuit (order={trotter_order}, "
              f"dt={optimal_dt:.6f})")
        print(f"{'─' * 60}")

        t0 = time.time()
        pathA_results = _run_quantum_skqd(
            H, max_krylov_dim, optimal_dt, num_trotter_steps, trotter_order,
            quantum_shots, backend="cudaq", verbose=verbose,
        )
        pathA_time = time.time() - t0
        pathA_energy = pathA_results["best_energy"]
        pathA_error = abs(pathA_energy - fci_energy) * 1000
        pathA_basis_size = (
            pathA_results["basis_sizes"][-1] if pathA_results["basis_sizes"] else 0
        )
        if n_pauli_terms == 0:
            n_pauli_terms = pathA_results["n_pauli_terms"]

        print(f"  Energy: {pathA_energy:.8f} Ha")
        print(f"  Error:  {pathA_error:.4f} mHa | Time: {pathA_time:.1f}s | "
              f"Basis: {pathA_basis_size}")
    else:
        print(f"\n{'─' * 60}")
        print(f"  [3/3] Path A: CUDA-Q — SKIPPED (cudaq not installed)")
        print(f"{'─' * 60}")

    # ------------------------------------------------------------------
    # Error analysis
    # ------------------------------------------------------------------
    print(f"\n{'─' * 60}")
    print(f"  Error Analysis")
    print(f"{'─' * 60}")
    print(f"  FCI energy:               {fci_energy:.8f} Ha")
    print(f"  Optimal dt:               {optimal_dt:.6f} (spectral range: {spectral_range:.4f})")
    print(f"  Direct diag error:        {direct_error:.4f} mHa")
    print(f"  Path C (exact) error:     {classical_error:.4f} mHa")
    if pathB_error is not None:
        print(f"  Path B (Trotter-{trotter_order}) error: {pathB_error:.4f} mHa")
    else:
        print(f"  Path B (Trotter-{trotter_order}) error: SKIPPED")
    if pathA_error is not None:
        print(f"  Path A (CUDA-Q-{trotter_order}) error: {pathA_error:.4f} mHa")
    print(f"")
    if pathB_energy is not None:
        print(f"  Trotter effect (B-C): "
              f"{abs(pathB_energy - classical_energy) * 1000:.4f} mHa")
    if pathA_energy is not None and pathB_energy is not None:
        print(f"  Circuit effect (A-B): "
              f"{abs(pathA_energy - pathB_energy) * 1000:.4f} mHa")
    elif pathA_energy is not None:
        print(f"  Circuit vs exact (A-C): "
              f"{abs(pathA_energy - classical_energy) * 1000:.4f} mHa")
    basis_str = f"  Basis sizes: C={classical_basis_size}"
    if pathB_basis_size is not None:
        basis_str += f", B={pathB_basis_size}"
    if pathA_basis_size is not None:
        basis_str += f", A={pathA_basis_size}"
    print(basis_str)

    chem_acc = 1.594
    print(f"\n  Chemical accuracy (< {chem_acc:.3f} mHa):")
    print(f"    Path C: {'PASS' if classical_error < chem_acc else 'FAIL'}")
    if pathB_error is not None:
        print(f"    Path B: {'PASS' if pathB_error < chem_acc else 'FAIL'}")
    else:
        print(f"    Path B: SKIPPED")
    if pathA_error is not None:
        print(f"    Path A: {'PASS' if pathA_error < chem_acc else 'FAIL'}")

    return ComparisonResult(
        system=name,
        n_qubits=n_qubits,
        n_configs=n_configs,
        n_pauli_terms=n_pauli_terms,
        fci_energy=fci_energy,
        spectral_range=spectral_range,
        optimal_dt=optimal_dt,
        direct_energy=direct_energy,
        direct_error_mha=direct_error,
        classical_energy=classical_energy,
        classical_error_mha=classical_error,
        classical_time_s=classical_time,
        classical_basis_size=classical_basis_size,
        pathB_energy=pathB_energy,
        pathB_error_mha=pathB_error,
        pathB_time_s=pathB_time,
        pathB_basis_size=pathB_basis_size,
        pathA_energy=pathA_energy,
        pathA_error_mha=pathA_error,
        pathA_time_s=pathA_time,
        pathA_basis_size=pathA_basis_size,
        pathA_available=pathA_available,
    )


def print_summary_table(results: List[ComparisonResult]) -> None:
    """Print 3-way summary comparison table."""
    has_pathA = any(r.pathA_available for r in results)

    print(f"\n{'=' * 115}")
    print(f"  SUMMARY: 3-Way SKQD Comparison (Paper-Compliant)")
    print(f"{'=' * 115}")

    # Header
    hdr = (f"{'System':<8} {'Qubits':<7} {'Paulis':<8} {'dt_opt':>8} "
           f"{'Path C':>12} {'Path B':>12} ")
    hdr_sub = (f"{'':8} {'':7} {'':8} {'':>8} "
               f"{'Error(mHa)':>12} {'Error(mHa)':>12} ")
    if has_pathA:
        hdr += f"{'Path A':>12} "
        hdr_sub += f"{'Error(mHa)':>12} "
    hdr += f"{'Trotter':>10} {'C':>7} {'B':>7}"
    hdr_sub += f"{'B-C':>10} {'Time':>7} {'Time':>7}"
    if has_pathA:
        hdr += f" {'A':>7}"
        hdr_sub += f" {'Time':>7}"

    print(hdr)
    print(hdr_sub)
    print("─" * 115)

    for r in results:
        pathB_err_str = (f"{r.pathB_error_mha:>12.4f}"
                         if r.pathB_error_mha is not None else f"{'skip':>12}")
        line = (f"{r.system:<8} {r.n_qubits:<7} {r.n_pauli_terms:<8} "
                f"{r.optimal_dt:>8.5f} "
                f"{r.classical_error_mha:>12.4f} {pathB_err_str} ")
        if has_pathA:
            if r.pathA_error_mha is not None:
                line += f"{r.pathA_error_mha:>12.4f} "
            else:
                line += f"{'N/A':>12} "
        if r.pathB_energy is not None:
            trotter_eff = abs(r.pathB_energy - r.classical_energy) * 1000
            line += f"{trotter_eff:>10.4f} "
        else:
            line += f"{'---':>10} "
        line += f"{r.classical_time_s:>6.1f}s"
        if r.pathB_time_s is not None:
            line += f" {r.pathB_time_s:>6.1f}s"
        else:
            line += f" {'skip':>7}"
        if has_pathA:
            if r.pathA_time_s is not None:
                line += f" {r.pathA_time_s:>6.1f}s"
            else:
                line += f" {'N/A':>7}"
        print(line)

    print("─" * 115)

    chem_acc = 1.594
    n_pass_c = sum(1 for r in results if r.classical_error_mha < chem_acc)
    pathB_results_valid = [r for r in results if r.pathB_error_mha is not None]
    n_pass_b = sum(1 for r in pathB_results_valid if r.pathB_error_mha < chem_acc)
    summary = (f"\nChemical accuracy (< {chem_acc:.3f} mHa): "
               f"Path C {n_pass_c}/{len(results)}, "
               f"Path B {n_pass_b}/{len(pathB_results_valid)}")
    if has_pathA:
        n_pass_a = sum(1 for r in results if r.pathA_error_mha is not None
                       and r.pathA_error_mha < chem_acc)
        n_avail = sum(1 for r in results if r.pathA_available)
        summary += f", Path A {n_pass_a}/{n_avail}"
    print(summary)

    print(f"\nKey findings:")
    if results:
        pathB_valid = [r for r in results
                       if r.pathB_energy is not None]
        if pathB_valid:
            avg_trotter = np.mean([abs(r.pathB_energy - r.classical_energy) * 1000
                                   for r in pathB_valid])
            print(f"  Avg Trotter-2 error (B vs C): {avg_trotter:.4f} mHa")

            pathB_worse = sum(1 for r in pathB_valid
                              if r.pathB_error_mha > r.classical_error_mha)
            print(f"  Path B worse than Path C: {pathB_worse}/{len(pathB_valid)} systems")

        if has_pathA:
            pathA_results = [r for r in results if r.pathA_energy is not None]
            if pathA_results:
                # Compare A vs B when B is available, otherwise A vs C
                pathA_with_B = [r for r in pathA_results if r.pathB_energy is not None]
                if pathA_with_B:
                    avg_circuit = np.mean([abs(r.pathA_energy - r.pathB_energy) * 1000
                                           for r in pathA_with_B])
                    print(f"  Avg circuit effect (A vs B): {avg_circuit:.4f} mHa")
                avg_circuit_vs_c = np.mean([abs(r.pathA_energy - r.classical_energy) * 1000
                                            for r in pathA_results])
                print(f"  Avg circuit effect (A vs C): {avg_circuit_vs_c:.4f} mHa")

                if pathA_with_B:
                    pathA_worse = sum(1 for r in pathA_with_B
                                      if r.pathA_error_mha > r.pathB_error_mha)
                    print(f"  Path A worse than Path B: {pathA_worse}/{len(pathA_with_B)} systems")

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
    print(f"  Path C: exact e^{{-iHt}} via Lanczos (full 2^n space, no Trotter)")
    print(f"  Path B: state-vector Trotter-2 on GPU (cos(t)I - i*sin(t)P)")
    if has_pathA:
        print(f"  Path A: CUDA-Q circuit (nvidia target, fp64, exp_pauli gates)")
        print(f"  Note: Path A uses CUDA-Q internal RNG (seed+k), not torch")
    else:
        print(f"  Path A: NOT AVAILABLE (install cuda-quantum-cu12)")


def main():
    parser = argparse.ArgumentParser(
        description="3-Way SKQD Comparison (Paper-Compliant): Path C vs Path B vs Path A"
    )
    parser.add_argument(
        "--systems", nargs="+", default=["h2", "lih", "h2o", "beh2", "nh3", "ch4", "n2"],
        choices=list(SYSTEMS.keys()),
        help="Molecular systems to compare (default: all 7)",
    )
    parser.add_argument("--krylov-dim", type=int, default=15,
                        help="Max Krylov dimension (default: 15, paper Fig. 1)")
    parser.add_argument("--shots", type=int, default=100_000,
                        help="Shots per Krylov state (default: 100000, paper Section V)")
    parser.add_argument("--trotter-steps", type=int, default=1,
                        help="Trotter sub-steps per Krylov step (default: 1, paper single S₂(Δt))")
    parser.add_argument("--trotter-order", type=int, default=2, choices=[1, 2],
                        help="Trotter order: 1=first, 2=second (default: 2)")
    args = parser.parse_args()

    print("3-Way SKQD Comparison (Controlled Experiment)")
    print(f"  Path C: Exact Lanczos (no Trotter, full 2^n space)")
    print(f"  Path B: State-vector Trotter-{args.trotter_order} (full 2^n space)")
    print(f"  Path A: CUDA-Q Trotter-{args.trotter_order} (full 2^n space)")
    print(f"Systems: {', '.join(args.systems)}")
    print(f"Krylov dim: {args.krylov_dim}, Shots: {args.shots:,}, "
          f"Trotter-{args.trotter_order} ({args.trotter_steps} steps)")
    print(f"Time step: optimal dt = pi / spectral_range (computed per system)")
    print(f"CUDA-Q available: {CUDAQ_AVAILABLE}")

    results = []
    for system_key in args.systems:
        try:
            result = run_comparison(
                system_key,
                max_krylov_dim=args.krylov_dim,
                quantum_shots=args.shots,
                num_trotter_steps=args.trotter_steps,
                trotter_order=args.trotter_order,
            )
            results.append(result)
        except Exception as e:
            print(f"\nERROR on {system_key}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    if results:
        print_summary_table(results)


if __name__ == "__main__":
    main()
