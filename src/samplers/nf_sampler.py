"""Normalizing Flow sampler for molecular configurations."""

import time
from dataclasses import dataclass
from typing import Optional

import torch

from .base import Sampler, SamplerResult
from ..flows.particle_conserving_flow import ParticleConservingFlowSampler
from ..flows.training import PhysicsGuidedConfig, PhysicsGuidedFlowTrainer
from ..nqs.dense import DenseNQS


@dataclass
class NFSamplerConfig:
    """Configuration for NF sampler."""
    n_epochs: int = 400
    samples_per_epoch: int = 512
    flow_lr: float = 5e-4
    nqs_lr: float = 1e-3
    initial_temperature: float = 1.0
    final_temperature: float = 0.3
    hidden_dims: Optional[list] = None
    nqs_hidden_dims: Optional[list] = None


class NFSampler(Sampler):
    """Normalizing Flow sampler using ParticleConservingFlow + physics-guided training."""

    def __init__(
        self,
        hamiltonian,
        config: Optional[NFSamplerConfig] = None,
        device: str = "cpu",
    ):
        self.hamiltonian = hamiltonian
        self.config = config or NFSamplerConfig()
        self.device = device

        n_sites = hamiltonian.num_sites
        n_alpha = hamiltonian.n_alpha
        n_beta = hamiltonian.n_beta

        # Create flow
        self.flow = ParticleConservingFlowSampler(
            num_sites=n_sites,
            n_alpha=n_alpha,
            n_beta=n_beta,
            hidden_dims=self.config.hidden_dims,
            temperature=self.config.initial_temperature,
        ).to(device)

        # Create NQS — auto-scale hidden dims by system size
        if self.config.nqs_hidden_dims is not None:
            nqs_hidden = self.config.nqs_hidden_dims
        elif n_sites <= 20:
            nqs_hidden = [256, 256]
        elif n_sites <= 40:
            nqs_hidden = [384, 384, 256]
        elif n_sites <= 52:
            nqs_hidden = [512, 512, 384]
        else:
            nqs_hidden = [512, 512, 512, 384]
        self.nqs = DenseNQS(
            num_sites=n_sites,
            hidden_dims=nqs_hidden,
        ).to(device)

        self._trained = False

    def train(self, verbose: bool = True) -> dict:
        """Train the NF using physics-guided training.

        Returns:
            Training history dict
        """
        cfg = self.config
        train_config = PhysicsGuidedConfig(
            samples_per_batch=cfg.samples_per_epoch,
            num_epochs=cfg.n_epochs,
            flow_lr=cfg.flow_lr,
            nqs_lr=cfg.nqs_lr,
            initial_temperature=cfg.initial_temperature,
            final_temperature=cfg.final_temperature,
        )

        trainer = PhysicsGuidedFlowTrainer(
            flow=self.flow,
            nqs=self.nqs,
            hamiltonian=self.hamiltonian,
            config=train_config,
            device=self.device,
        )

        history = trainer.train()
        self._trained = True
        return history

    def sample(self, n_samples: int) -> SamplerResult:
        """Sample configurations from the trained flow.

        Args:
            n_samples: Number of samples to draw (before deduplication)

        Returns:
            SamplerResult with unique configurations
        """
        if not self._trained:
            self.train()

        t0 = time.time()

        with torch.no_grad():
            # Sample from flow
            configs, log_probs = self.flow.flow.sample(n_samples, hard=True)
            configs = configs.long()

            # Deduplicate
            unique_configs, inverse = torch.unique(configs, dim=0, return_inverse=True)

            # Recompute log probs for unique configs
            unique_log_probs = self.flow.log_prob(unique_configs.float())

        wall_time = time.time() - t0

        return SamplerResult(
            configs=unique_configs,
            log_probs=unique_log_probs,
            wall_time=wall_time,
            metadata={
                "n_raw_samples": n_samples,
                "n_unique": len(unique_configs),
                "unique_ratio": len(unique_configs) / n_samples,
            },
        )
