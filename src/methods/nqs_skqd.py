"""
NQS+SKQD: Two-stage method with Krylov expansion.
Stage 1: Train NQS independently
Stage 2: Sample → SKQD Krylov diagonalization
"""

import time
import numpy as np
import torch

from ..solvers.base import SolverResult
from ..solvers.skqd import SKQDSolverC, SKQDConfig
from ..samplers.transformer_nf_sampler import TransformerNFSampler, TransformerSamplerConfig


def run_nqs_skqd(hamiltonian, mol_info, n_epochs=400, n_samples=5000,
                 max_basis_size=10000, samples_per_epoch=512) -> SolverResult:
    """Run NQS+SKQD: train NQS then sample for SKQD."""
    t0 = time.time()

    # Stage 1: Train NQS
    tf_config = TransformerSamplerConfig(
        n_epochs=n_epochs,
        samples_per_epoch=samples_per_epoch,
    )
    sampler = TransformerNFSampler(hamiltonian, config=tf_config, device="cpu")
    sampler.train(verbose=True)
    train_time = time.time() - t0

    # Stage 2: SKQD
    skqd = SKQDSolverC(sampler, SKQDConfig(
        n_samples=n_samples, max_basis_size=max_basis_size,
    ))
    result = skqd.solve(hamiltonian, mol_info)

    total_time = time.time() - t0

    return SolverResult(
        energy=result.energy,
        diag_dim=result.diag_dim,
        wall_time=total_time,
        method="NQS+SKQD",
        converged=result.converged,
        metadata={**result.metadata, "train_time": train_time},
    )
