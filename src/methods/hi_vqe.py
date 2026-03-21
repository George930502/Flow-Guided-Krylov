"""
HI-VQE: Handover Iterative VQE using CUDA-Q UCCSD circuits + IBM's qiskit-addon-sqd.

Quantum circuit: CUDA-Q UCCSD (particle-number-conserving Givens rotations)
SQD solver: IBM's solve_fermion from qiskit-addon-sqd

Algorithm (Pellow-Jarman et al., 2025, arXiv:2503.06292):
  Loop:
    1. CUDA-Q UCCSD circuit U(θ)|HF⟩ → sample bitstrings
    2. IBM's solve_fermion → diagonalize → E₀, |Φ⟩, occupancies
    3. Configuration recovery using occupancies
    4. COBYLA optimizer updates θ to minimize E₀
    5. Check convergence (3 consecutive steps ΔE < 10⁻⁴)
"""

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from ..solvers.base import SolverResult
from ..samplers.cudaq_sampler import CUDAQCircuitSampler, CUDAQSamplerConfig
from ..utils.config_hash import config_integer_hash

from qiskit_addon_sqd.fermion import solve_fermion
from qiskit_addon_sqd.configuration_recovery import recover_configurations


@dataclass
class HIVQEConfig:
    """Configuration for HI-VQE."""
    max_iterations: int = 25
    shots: int = 1000
    convergence_count: int = 3
    convergence_abstol: float = 1e-4

    # Circuit
    n_layers: int = 2

    # SQD batching
    num_batches: int = 5
    samples_per_batch: int = 0    # 0 = auto
    max_basis_size: int = 10000

    # Configuration recovery
    configuration_recovery: bool = True

    # COBYLA optimizer
    cobyla_maxiter: int = 3       # iterations per outer loop
    cobyla_rhobeg: float = 0.1


def run_hi_vqe(hamiltonian, mol_info,
               config: Optional[HIVQEConfig] = None) -> SolverResult:
    """Run HI-VQE with CUDA-Q UCCSD + IBM SQD."""
    t0 = time.time()
    cfg = config or HIVQEConfig()

    n_orb = hamiltonian.n_orbitals
    n_alpha = hamiltonian.n_alpha
    n_beta = hamiltonian.n_beta
    n_qubits = 2 * n_orb

    # Molecular integrals for IBM's solve_fermion
    integrals = hamiltonian.integrals
    hcore = np.asarray(integrals.h1e, dtype=np.float64)
    eri = np.asarray(integrals.h2e, dtype=np.float64)
    nuclear_repulsion = float(integrals.nuclear_repulsion)

    # CUDA-Q UCCSD sampler
    cudaq_config = CUDAQSamplerConfig(n_layers=cfg.n_layers, shots=cfg.shots)
    sampler = CUDAQCircuitSampler(hamiltonian, cudaq_config)

    # Auto batch size
    samples_per_batch = cfg.samples_per_batch
    if samples_per_batch <= 0:
        samples_per_batch = max(50, cfg.shots // cfg.num_batches)

    energy_history = []
    best_energy = float("inf")
    converged = False
    converge_count = 0
    avg_occupancies = None

    print(f"    HI-VQE (CUDA-Q UCCSD): params={sampler.n_params}, "
          f"shots={cfg.shots}, layers={cfg.n_layers}")

    def _evaluate_energy(params):
        """Evaluate energy for given circuit parameters (for COBYLA)."""
        sampler.set_params(params)
        sr = sampler.sample(cfg.shots)
        configs = sr.configs  # already particle-number-conserving

        if len(configs) < 2:
            return 0.0

        # Convert to IBM format
        bs_matrix = _configs_to_ibm_format(configs, n_orb, n_qubits)

        try:
            e, _, _, _ = solve_fermion(bs_matrix, hcore, eri, spin_sq=0)
            return e + nuclear_repulsion
        except:
            return 0.0

    for iteration in range(cfg.max_iterations):
        # ==========================================================
        # Step 1: Sample from CUDA-Q UCCSD circuit
        # ==========================================================
        sample_result = sampler.sample(cfg.shots)
        configs = sample_result.configs
        # UCCSD preserves particle number — all configs are valid

        # ==========================================================
        # Step 2: Configuration recovery (optional, for noisy hardware)
        # ==========================================================
        bs_matrix = _configs_to_ibm_format(configs, n_orb, n_qubits)
        probs = np.ones(len(bs_matrix)) / len(bs_matrix)

        if cfg.configuration_recovery and avg_occupancies is not None:
            try:
                bs_matrix, probs = recover_configurations(
                    bs_matrix, probs,
                    avg_occupancies,
                    num_elec_a=n_alpha,
                    num_elec_b=n_beta,
                    rand_seed=iteration,
                )
            except:
                pass

        if len(bs_matrix) < 2:
            print(f"    Iter {iteration:>3d}: Too few configs ({len(bs_matrix)})")
            sampler.params += np.random.randn(sampler.n_params) * 0.1
            continue

        # ==========================================================
        # Step 3: SQD diagonalization (IBM's solve_fermion)
        # ==========================================================
        batch_energies = []
        batch_occs = []
        batch_size = min(samples_per_batch, len(bs_matrix))

        for b in range(cfg.num_batches):
            if len(bs_matrix) <= batch_size:
                batch = bs_matrix
            else:
                idx = np.random.choice(len(bs_matrix), size=batch_size, replace=False)
                batch = bs_matrix[idx]

            if len(batch) < 2:
                continue

            try:
                e, _, occ, _ = solve_fermion(batch, hcore, eri, spin_sq=0)
                batch_energies.append(e + nuclear_repulsion)
                batch_occs.append(occ)
            except:
                continue

        if not batch_energies:
            print(f"    Iter {iteration:>3d}: SQD failed")
            sampler.params += np.random.randn(sampler.n_params) * 0.05
            continue

        best_batch_idx = int(np.argmin(batch_energies))
        e_current = batch_energies[best_batch_idx]
        avg_occupancies = batch_occs[best_batch_idx]

        if e_current < best_energy:
            best_energy = e_current

        energy_history.append(e_current)

        # ==========================================================
        # Step 4: Update circuit parameters (COBYLA)
        # ==========================================================
        from scipy.optimize import minimize as scipy_minimize
        try:
            opt_result = scipy_minimize(
                _evaluate_energy,
                sampler.params,
                method='COBYLA',
                options={'maxiter': cfg.cobyla_maxiter, 'rhobeg': cfg.cobyla_rhobeg},
            )
            sampler.set_params(opt_result.x)
        except:
            sampler.params += np.random.randn(sampler.n_params) * 0.05

        # ==========================================================
        # Step 5: Convergence
        # ==========================================================
        if len(energy_history) >= 2:
            delta_e = abs(energy_history[-1] - energy_history[-2])
        else:
            delta_e = float("inf")

        if delta_e < cfg.convergence_abstol:
            converge_count += 1
        else:
            converge_count = 0

        print(f"    Iter {iteration:>3d}: E={e_current:.10f} Ha, "
              f"configs={len(bs_matrix)}, "
              f"ΔE={delta_e:.2e}, conv={converge_count}/{cfg.convergence_count}")

        if converge_count >= cfg.convergence_count:
            converged = True
            break

    wall_time = time.time() - t0

    return SolverResult(
        energy=best_energy if best_energy < float("inf") else None,
        diag_dim=len(bs_matrix) if "bs_matrix" in dir() else 0,
        wall_time=wall_time,
        method="HI-VQE",
        converged=converged,
        metadata={
            "iterations": iteration + 1 if "iteration" in dir() else 0,
            "energy_history": energy_history,
            "n_params": sampler.n_params,
        },
    )


def _configs_to_ibm_format(configs, n_orb, n_qubits):
    """Convert config tensors to IBM bitstring matrix format.

    IBM format: bool array where columns [N..N/2] = spin-up (alpha),
    columns [N/2..0] = spin-down (beta).
    """
    n = len(configs)
    bs = np.zeros((n, n_qubits), dtype=bool)
    for s in range(n):
        c = configs[s]
        for j in range(n_orb):
            bs[s, n_orb - 1 - j] = bool(c[j])
            bs[s, n_qubits - 1 - j] = bool(c[j + n_orb])
    return bs
