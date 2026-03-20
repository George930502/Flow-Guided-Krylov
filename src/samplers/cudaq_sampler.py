"""
CUDA-Q quantum circuit sampler using UCCSD ansatz.

Uses particle-number-conserving Givens rotations — no post-selection needed.
"""

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import cudaq

from .base import Sampler, SamplerResult
from .cudaq_circuits import uccsd_ansatz, count_uccsd_params


@dataclass
class CUDAQSamplerConfig:
    """Configuration for CUDA-Q UCCSD sampler."""
    n_layers: int = 2
    shots: int = 10000
    target: str = "qpp-cpu"


class CUDAQCircuitSampler(Sampler):
    """
    CUDA-Q UCCSD circuit sampler.

    Particle-number-conserving: all samples have exactly n_alpha + n_beta electrons.
    No post-selection needed.
    """

    def __init__(self, hamiltonian, config: Optional[CUDAQSamplerConfig] = None):
        self.hamiltonian = hamiltonian
        self.config = config or CUDAQSamplerConfig()

        self.n_orbitals = hamiltonian.n_orbitals
        self.n_alpha = hamiltonian.n_alpha
        self.n_beta = hamiltonian.n_beta
        self.n_qubits = 2 * self.n_orbitals

        cudaq.set_target(self.config.target)

        self.n_params = count_uccsd_params(self.n_orbitals, self.config.n_layers)
        self.params = np.random.randn(self.n_params) * 0.01

    def set_params(self, params):
        self.params = np.array(params, dtype=np.float64)

    def sample(self, n_samples: int) -> SamplerResult:
        t0 = time.time()

        thetas = self.params.tolist()

        result = cudaq.sample(
            uccsd_ansatz,
            self.n_qubits, self.n_alpha, self.n_beta,
            self.config.n_layers, thetas,
            shots_count=n_samples,
        )

        # Convert bitstrings to config tensors
        configs_list = []
        for bitstring in result:
            count = result.count(bitstring)
            config = torch.zeros(self.n_qubits, dtype=torch.long)
            for i, bit in enumerate(bitstring):
                config[i] = int(bit)
            for _ in range(count):
                configs_list.append(config)

        if not configs_list:
            return SamplerResult(
                configs=torch.empty(0, self.n_qubits, dtype=torch.long),
                log_probs=None,
                wall_time=time.time() - t0,
                metadata={"error": "No samples"},
            )

        configs = torch.stack(configs_list)
        unique_configs = torch.unique(configs, dim=0)
        wall_time = time.time() - t0

        return SamplerResult(
            configs=unique_configs,
            log_probs=None,
            wall_time=wall_time,
            metadata={
                "n_raw_samples": n_samples,
                "n_unique": len(unique_configs),
                "n_params": self.n_params,
                "sampler_type": "CUDAQ-UCCSD",
            },
        )
