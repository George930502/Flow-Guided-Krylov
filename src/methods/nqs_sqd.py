"""
NQS+SQD: Two-stage method.
Stage 1: Train NQS (autoregressive transformer) independently
Stage 2: Sample from trained NQS → SQD diagonalization
"""

import time
import numpy as np
import torch

from ..solvers.base import SolverResult
from ..solvers.sqd import SQDSolver, SQDConfig
from ..nqs.transformer import AutoregressiveTransformer
from ..samplers.transformer_nf_sampler import TransformerNFSampler, TransformerSamplerConfig


def run_nqs_sqd(hamiltonian, mol_info, n_epochs=400, n_samples=5000,
                samples_per_epoch=512) -> SolverResult:
    """Run NQS+SQD: train NQS then sample for SQD."""
    t0 = time.time()

    # Stage 1: Train NQS
    tf_config = TransformerSamplerConfig(
        n_epochs=n_epochs,
        samples_per_epoch=samples_per_epoch,
    )
    sampler = TransformerNFSampler(hamiltonian, config=tf_config, device="cpu")
    sampler.train(verbose=True)
    train_time = time.time() - t0

    # Stage 2: SQD
    sqd = SQDSolver(sampler, SQDConfig(n_samples=n_samples))
    result = sqd.solve(hamiltonian, mol_info)

    total_time = time.time() - t0

    return SolverResult(
        energy=result.energy,
        diag_dim=result.diag_dim,
        wall_time=total_time,
        method="NQS+SQD",
        converged=result.converged,
        metadata={**result.metadata, "train_time": train_time},
    )
