"""
Transformer-based Normalizing Flow sampler.

Uses an autoregressive transformer to sample molecular configurations,
replacing the product-of-marginals ParticleConservingFlow with a
fully autoregressive model that captures inter-orbital correlations.

The transformer architecture is inspired by:
- Psiformer (von Glehn et al., 2023) — self-attention for quantum chemistry
- Autoregressive NQS (Sharir et al., 2020) — autoregressive factorization
- Barrett et al. (2022) — autoregressive wavefunctions for ab initio QC

Key advantage over ParticleConservingFlow:
- Each orbital P(σ_i | σ_{<i}) is conditioned on ALL previous decisions
- Self-attention captures long-range orbital correlations
- Beta channel cross-attends to full alpha config
"""

import time
from dataclasses import dataclass
from typing import Optional

import torch

from .base import Sampler, SamplerResult
from ..nqs.transformer import AutoregressiveTransformer, TransformerNQS
from ..flows.training import PhysicsGuidedConfig, PhysicsGuidedFlowTrainer


def _auto_scale_transformer(n_orbitals: int) -> dict:
    """Auto-scale transformer hyperparameters by system size."""
    if n_orbitals <= 10:  # ≤20Q
        return {"embed_dim": 64, "n_heads": 4, "n_layers": 4}
    elif n_orbitals <= 15:  # ≤30Q
        return {"embed_dim": 128, "n_heads": 4, "n_layers": 4}
    elif n_orbitals <= 20:  # ≤40Q
        return {"embed_dim": 128, "n_heads": 8, "n_layers": 6}
    elif n_orbitals <= 26:  # ≤52Q
        return {"embed_dim": 192, "n_heads": 8, "n_layers": 6}
    else:  # 54Q+
        return {"embed_dim": 256, "n_heads": 8, "n_layers": 8}


@dataclass
class TransformerSamplerConfig:
    """Configuration for Transformer NF sampler."""
    n_epochs: int = 400
    samples_per_epoch: int = 512
    flow_lr: float = 3e-4
    nqs_lr: float = 1e-3
    initial_temperature: float = 1.0
    final_temperature: float = 0.3

    # Transformer architecture (None = auto-scale)
    embed_dim: Optional[int] = None
    n_heads: Optional[int] = None
    n_layers: Optional[int] = None

    # NQS architecture
    nqs_embed_dim: Optional[int] = None
    nqs_n_heads: Optional[int] = None
    nqs_n_layers: Optional[int] = None


class _TransformerFlowWrapper(torch.nn.Module):
    """
    Wrapper to make AutoregressiveTransformer compatible with
    the ParticleConservingFlowSampler interface expected by the trainer.
    """

    def __init__(self, transformer: AutoregressiveTransformer):
        super().__init__()
        self.flow = transformer
        self.num_sites = transformer.n_qubits
        self.temperature = 1.0

    def sample(self, n_samples: int, hard: bool = True):
        configs, log_probs = self.flow.sample(n_samples, hard=hard, temperature=self.temperature)
        # Match ParticleConservingFlowSampler interface: (log_probs, unique_configs)
        unique_configs = torch.unique(configs.long(), dim=0)
        return log_probs, unique_configs

    def log_prob(self, config: torch.Tensor) -> torch.Tensor:
        return self.flow.log_prob(config)


class TransformerNFSampler(Sampler):
    """
    Transformer-based NF sampler using autoregressive generation.

    Drop-in replacement for NFSampler with stronger architecture.
    """

    def __init__(
        self,
        hamiltonian,
        config: Optional[TransformerSamplerConfig] = None,
        device: str = "cpu",
    ):
        self.hamiltonian = hamiltonian
        self.config = config or TransformerSamplerConfig()
        self.device = device

        n_sites = hamiltonian.num_sites
        n_orbitals = hamiltonian.n_orbitals
        n_alpha = hamiltonian.n_alpha
        n_beta = hamiltonian.n_beta

        # Auto-scale if not specified
        auto = _auto_scale_transformer(n_orbitals)
        embed_dim = self.config.embed_dim or auto["embed_dim"]
        n_heads = self.config.n_heads or auto["n_heads"]
        n_layers = self.config.n_layers or auto["n_layers"]

        # Create autoregressive transformer
        self.transformer = AutoregressiveTransformer(
            n_orbitals=n_orbitals,
            n_alpha=n_alpha,
            n_beta=n_beta,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=n_layers,
        ).to(device)

        # Wrap for trainer compatibility
        self.flow = _TransformerFlowWrapper(self.transformer).to(device)

        # Create NQS (also transformer-based)
        nqs_embed = self.config.nqs_embed_dim or embed_dim
        nqs_heads = self.config.nqs_n_heads or n_heads
        nqs_layers = self.config.nqs_n_layers or max(n_layers - 1, 2)

        self.nqs = TransformerNQS(
            num_sites=n_sites,
            embed_dim=nqs_embed,
            n_heads=nqs_heads,
            n_layers=nqs_layers,
        ).to(device)

        self._trained = False

    def train(self, verbose: bool = True) -> dict:
        """Train using physics-guided training."""
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
        """Sample configurations from the trained transformer."""
        if not self._trained:
            self.train()

        t0 = time.time()

        with torch.no_grad():
            self.flow.temperature = self.config.final_temperature
            configs, log_probs = self.transformer.sample(
                n_samples, temperature=self.config.final_temperature
            )
            configs = configs.long()

            # Deduplicate
            unique_configs = torch.unique(configs, dim=0)

            # Recompute log probs for unique configs
            unique_log_probs = self.transformer.log_prob(unique_configs)

        wall_time = time.time() - t0

        return SamplerResult(
            configs=unique_configs,
            log_probs=unique_log_probs,
            wall_time=wall_time,
            metadata={
                "n_raw_samples": n_samples,
                "n_unique": len(unique_configs),
                "unique_ratio": len(unique_configs) / n_samples,
                "architecture": "transformer",
            },
        )
