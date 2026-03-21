"""
QC+SKQD: Quantum circuit Trotter Krylov sampler + SKQD diagonalization.

Uses CUDA-Q Trotter time evolution to generate Krylov subspace,
then diagonalizes in the cumulative sample basis.
"""

import time
from ..solvers.base import SolverResult
from ..solvers.skqd import SKQDSolverC, SKQDConfig
from ..samplers.cudaq_sampler import TrotterKrylovSampler, TrotterKrylovConfig


def run_qc_skqd(hamiltonian, mol_info, n_krylov_steps=6, dt=0.1,
                shots_per_step=10000, max_basis_size=10000) -> SolverResult:
    """Run QC+SKQD: Trotter Krylov sampling + SKQD."""
    t0 = time.time()

    sampler = TrotterKrylovSampler(hamiltonian, TrotterKrylovConfig(
        n_krylov_steps=n_krylov_steps,
        dt=dt,
        shots_per_step=shots_per_step,
    ))
    skqd = SKQDSolverC(sampler, SKQDConfig(
        n_samples=shots_per_step * n_krylov_steps,
        max_basis_size=max_basis_size,
    ))
    result = skqd.solve(hamiltonian, mol_info)

    total_time = time.time() - t0

    return SolverResult(
        energy=result.energy,
        diag_dim=result.diag_dim,
        wall_time=total_time,
        method="QC+SKQD",
        converged=result.converged,
        metadata=result.metadata,
    )
