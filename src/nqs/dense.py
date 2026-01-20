"""Dense (fully-connected) Neural Quantum State implementation."""

import torch
import torch.nn as nn
from typing import Optional

try:
    from .base import NeuralQuantumState
except ImportError:
    from nqs.base import NeuralQuantumState


class DenseNQS(NeuralQuantumState):
    """
    Dense Neural Quantum State using fully-connected layers.

    Architecture:
        Input (num_sites) → Hidden layers → Output (log_amplitude, [phase])

    This is the architecture described in the NF-NQS paper for the Ising model,
    using ReLU activations and a final tanh for bounded output.

    Args:
        num_sites: Number of sites in the system
        hidden_dims: List of hidden layer dimensions
        local_dim: Local Hilbert space dimension (default: 2 for qubits)
        complex_output: Whether to output complex amplitudes
        activation: Activation function to use
    """

    def __init__(
        self,
        num_sites: int,
        hidden_dims: list[int] = [512, 512, 512, 512],
        local_dim: int = 2,
        complex_output: bool = False,
        activation: str = "relu",
    ):
        super().__init__(num_sites, local_dim, complex_output)

        self.hidden_dims = hidden_dims

        # Choose activation
        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "tanh":
            act_fn = nn.Tanh
        elif activation == "gelu":
            act_fn = nn.GELU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build amplitude network
        layers = []
        in_dim = num_sites
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                act_fn(),
            ])
            in_dim = hidden_dim

        # Output layer for log amplitude
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Tanh())  # Bounded output for stability

        self.amplitude_net = nn.Sequential(*layers)

        # Build phase network if complex output is needed
        if complex_output:
            phase_layers = []
            in_dim = num_sites
            for hidden_dim in hidden_dims:
                phase_layers.extend([
                    nn.Linear(in_dim, hidden_dim),
                    act_fn(),
                ])
                in_dim = hidden_dim
            phase_layers.append(nn.Linear(in_dim, 1))
            self.phase_net = nn.Sequential(*phase_layers)
        else:
            self.phase_net = None

        # Learnable scale for log amplitude
        self.log_amp_scale = nn.Parameter(torch.tensor(1.0))

    def log_amplitude(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log|ψ_θ(x)|.

        Args:
            x: Batch of configurations, shape (batch_size, num_sites)

        Returns:
            Log amplitudes, shape (batch_size,)
        """
        x = self.encode_configuration(x)
        out = self.amplitude_net(x)  # (batch_size, 1)
        return self.log_amp_scale * out.squeeze(-1)

    def phase(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute phase φ_θ(x).

        Args:
            x: Batch of configurations, shape (batch_size, num_sites)

        Returns:
            Phases, shape (batch_size,)
        """
        if self.phase_net is None:
            # For real NQS, use sign network or return zero
            return torch.zeros(x.shape[0], device=x.device)

        x = self.encode_configuration(x)
        out = self.phase_net(x)  # (batch_size, 1)
        return out.squeeze(-1)


class SignedDenseNQS(NeuralQuantumState):
    """
    Dense NQS with explicit sign structure for real wavefunctions.

    Instead of outputting phase, outputs a sign ∈ {-1, +1}.
    This is useful for systems like the Ising model with real ground states.
    """

    def __init__(
        self,
        num_sites: int,
        hidden_dims: list[int] = [512, 512, 512, 512],
        local_dim: int = 2,
        activation: str = "relu",
    ):
        super().__init__(num_sites, local_dim, complex_output=False)

        self.hidden_dims = hidden_dims

        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "tanh":
            act_fn = nn.Tanh
        elif activation == "gelu":
            act_fn = nn.GELU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Shared feature extractor
        feature_layers = []
        in_dim = num_sites
        for hidden_dim in hidden_dims[:-1]:
            feature_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                act_fn(),
            ])
            in_dim = hidden_dim

        self.feature_net = nn.Sequential(*feature_layers)

        # Amplitude head
        self.amplitude_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dims[-1]),
            act_fn(),
            nn.Linear(hidden_dims[-1], 1),
            nn.Tanh(),
        )

        # Sign head (outputs logit for sign)
        self.sign_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dims[-1]),
            act_fn(),
            nn.Linear(hidden_dims[-1], 1),
            nn.Tanh(),
        )

        self.log_amp_scale = nn.Parameter(torch.tensor(1.0))

    def log_amplitude(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode_configuration(x)
        features = self.feature_net(x)
        out = self.amplitude_head(features)
        return self.log_amp_scale * out.squeeze(-1)

    def phase(self, x: torch.Tensor) -> torch.Tensor:
        """Return 0 or π based on sign prediction."""
        x = self.encode_configuration(x)
        features = self.feature_net(x)
        sign_logit = self.sign_head(features).squeeze(-1)
        # Map tanh output to {0, π}
        return torch.pi * (sign_logit < 0).float()

    def get_sign(self, x: torch.Tensor) -> torch.Tensor:
        """Get the sign ∈ {-1, +1} directly."""
        x = self.encode_configuration(x)
        features = self.feature_net(x)
        sign_logit = self.sign_head(features).squeeze(-1)
        return torch.sign(sign_logit + 1e-10)
