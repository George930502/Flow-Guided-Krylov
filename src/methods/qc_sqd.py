"""
QC+SQD: CUDA-Q UCCSD circuit + IBM's SQD.

One-shot: sample from particle-number-conserving UCCSD circuit,
then diagonalize with IBM's solve_fermion.
"""

import time
import numpy as np
import torch

from ..solvers.base import SolverResult
from ..samplers.cudaq_sampler import CUDAQCircuitSampler, CUDAQSamplerConfig

from qiskit_addon_sqd.fermion import solve_fermion


def run_qc_sqd(hamiltonian, mol_info, n_samples=10000,
               num_batches=5, n_layers=2) -> SolverResult:
    """Run QC+SQD: CUDA-Q UCCSD → sample → IBM SQD."""
    t0 = time.time()

    n_orb = hamiltonian.n_orbitals
    n_qubits = 2 * n_orb

    integrals = hamiltonian.integrals
    hcore = np.asarray(integrals.h1e, dtype=np.float64)
    eri = np.asarray(integrals.h2e, dtype=np.float64)
    nuclear_repulsion = float(integrals.nuclear_repulsion)

    # CUDA-Q UCCSD sampling (particle-number-conserving, no post-selection needed)
    cudaq_config = CUDAQSamplerConfig(n_layers=n_layers, shots=n_samples)
    sampler = CUDAQCircuitSampler(hamiltonian, cudaq_config)
    sampler.set_params(np.random.randn(sampler.n_params) * 0.3)

    sample_result = sampler.sample(n_samples)
    configs = sample_result.configs

    if len(configs) < 2:
        return SolverResult(
            energy=None, diag_dim=len(configs), wall_time=time.time()-t0,
            method="QC+SQD", converged=False, metadata={},
        )

    # Convert to IBM format and diagonalize
    bs_matrix = _configs_to_ibm_format(configs, n_orb, n_qubits)

    batch_size = max(10, len(bs_matrix) // num_batches)
    best_energy = float("inf")

    for b in range(num_batches):
        if len(bs_matrix) <= batch_size:
            batch = bs_matrix
        else:
            idx = np.random.choice(len(bs_matrix), size=batch_size, replace=False)
            batch = bs_matrix[idx]

        try:
            e, _, _, _ = solve_fermion(batch, hcore, eri, spin_sq=0)
            e_total = e + nuclear_repulsion
            if e_total < best_energy:
                best_energy = e_total
        except:
            continue

    wall_time = time.time() - t0

    return SolverResult(
        energy=best_energy if best_energy < float("inf") else None,
        diag_dim=len(configs),
        wall_time=wall_time,
        method="QC+SQD",
        converged=best_energy < float("inf"),
        metadata={"n_unique": len(configs), "n_samples": n_samples},
    )


def _configs_to_ibm_format(configs, n_orb, n_qubits):
    n = len(configs)
    bs = np.zeros((n, n_qubits), dtype=bool)
    for s in range(n):
        c = configs[s]
        for j in range(n_orb):
            bs[s, n_orb - 1 - j] = bool(c[j])
            bs[s, n_qubits - 1 - j] = bool(c[j + n_orb])
    return bs
