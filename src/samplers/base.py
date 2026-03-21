"""Base classes for configuration samplers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class SamplerResult:
    """Result from a sampler run."""
    configs: torch.Tensor           # (n, num_sites) binary
    log_probs: Optional[torch.Tensor]  # (n,) or None
    wall_time: float                # seconds
    metadata: dict = field(default_factory=dict)


class Sampler(ABC):
    """Abstract base class for configuration samplers."""

    @abstractmethod
    def sample(self, n_samples: int) -> SamplerResult:
        """Sample configurations.

        Args:
            n_samples: Number of samples to draw

        Returns:
            SamplerResult with unique configurations and metadata
        """
        ...
