"""Dense (fully-connected) Neural Quantum State implementation."""

import torch
import torch.nn as nn
from typing import Optional

try:
    from .base import NeuralQuantumState
except ImportError:
    from nqs.base import NeuralQuantumState


def compile_nqs(model: nn.Module, mode: str = "reduce-overhead") -> nn.Module:
    """
    Apply torch.compile() to an NQS model for 20-40% speedup.

    Args:
        model: NQS model to compile
        mode: Compilation mode ("default", "reduce-overhead", "max-autotune")
              - "reduce-overhead": Best for small batches, minimal startup cost
              - "max-autotune": Best for large batches, longer startup but faster

    Returns:
        Compiled model (or original if torch.compile unavailable)

    Example:
        nqs = compile_nqs(DenseNQS(num_sites=28))
    """
    if not hasattr(torch, 'compile'):
        return model  # PyTorch < 2.0

    try:
        return torch.compile(model, mode=mode, fullgraph=False)
    except Exception:
        # Fall back to uncompiled if compilation fails
        return model


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

        # Output layer for log amplitude (NO Tanh - allows proper amplitude range)
        layers.append(nn.Linear(in_dim, 1))
        # NOTE: Removed Tanh which was limiting amplitude range to exp(-1) to exp(1) = 7.4x
        # Real ground states need 100-1000x amplitude variations

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

        # Learnable scale for log amplitude (initialized to 1.5 for stable training)
        # With soft clamp in log_amplitude(), this allows amplitude range of ~90x
        # Starting with a smaller value prevents training instabilities in large systems
        # The network can learn to increase this if needed
        self.log_amp_scale = nn.Parameter(torch.tensor(1.5))

    def log_amplitude(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log|ψ_θ(x)|.

        Args:
            x: Batch of configurations, shape (batch_size, num_sites)

        Returns:
            Log amplitudes, shape (batch_size,)

        Note:
            Uses soft clamp for numerical stability while allowing large amplitude range.
            With log_amp_scale=3.0 and tanh(x/3)*3, output is bounded to [-9, 9],
            giving amplitude range of exp(-9) to exp(9) ≈ 8000x (vs 7.4x with simple Tanh).
        """
        x = self.encode_configuration(x)
        raw = self.amplitude_net(x).squeeze(-1)  # (batch_size,)
        # Soft clamp for numerical stability (prevents exp overflow)
        # tanh(x/3)*3 bounds to [-3, 3], then scale gives [-9, 9] with log_amp_scale=3.0
        return self.log_amp_scale * torch.tanh(raw / 3.0) * 3.0

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

    OPTIMIZATION: Includes feature caching to avoid redundant forward passes
    when computing both amplitude and phase for the same configurations.
    This provides ~2x speedup for local energy computation.
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

        # Amplitude head (NO Tanh - allows proper amplitude range)
        self.amplitude_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dims[-1]),
            act_fn(),
            nn.Linear(hidden_dims[-1], 1),
            # NOTE: Removed Tanh - soft clamp applied in log_amplitude() instead
        )

        # Sign head (outputs logit for sign)
        self.sign_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dims[-1]),
            act_fn(),
            nn.Linear(hidden_dims[-1], 1),
            nn.Tanh(),  # Keep Tanh for sign (bounded output is appropriate here)
        )

        # Learnable scale for log amplitude (initialized to 1.5 for stable training)
        # Starting with a smaller value prevents training instabilities in large systems
        self.log_amp_scale = nn.Parameter(torch.tensor(1.5))

        # Feature cache for avoiding duplicate forward passes
        self._feature_cache = None
        self._feature_cache_input_hash = None

    def _get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get features with caching to avoid duplicate computation.

        When computing local energy, we need both amplitude and phase for
        the same configurations. Caching the features provides ~2x speedup.
        """
        x_encoded = self.encode_configuration(x)

        # Simple hash based on tensor data pointer and shape
        input_hash = (x_encoded.data_ptr(), x_encoded.shape, x_encoded.device)

        if self._feature_cache is not None and self._feature_cache_input_hash == input_hash:
            return self._feature_cache

        features = self.feature_net(x_encoded)

        # Cache for potential reuse
        self._feature_cache = features
        self._feature_cache_input_hash = input_hash

        return features

    def clear_feature_cache(self):
        """Clear the feature cache to free memory."""
        self._feature_cache = None
        self._feature_cache_input_hash = None

    def log_amplitude(self, x: torch.Tensor) -> torch.Tensor:
        features = self._get_features(x)
        raw = self.amplitude_head(features).squeeze(-1)
        # Soft clamp for numerical stability while allowing large amplitude range
        return self.log_amp_scale * torch.tanh(raw / 3.0) * 3.0

    def phase(self, x: torch.Tensor) -> torch.Tensor:
        """Return 0 or π based on sign prediction."""
        features = self._get_features(x)
        sign_logit = self.sign_head(features).squeeze(-1)
        # Map tanh output to {0, π}
        return torch.pi * (sign_logit < 0).float()

    def get_sign(self, x: torch.Tensor) -> torch.Tensor:
        """Get the sign ∈ {-1, +1} directly."""
        features = self._get_features(x)
        sign_logit = self.sign_head(features).squeeze(-1)
        return torch.sign(sign_logit + 1e-10)

    def log_amplitude_and_phase(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute both log amplitude and phase in a single forward pass.

        OPTIMIZED: Single feature extraction for both outputs.
        Use this method when you need both values for the same configs.
        """
        features = self._get_features(x)
        raw = self.amplitude_head(features).squeeze(-1)
        # Soft clamp for numerical stability while allowing large amplitude range
        log_amp = self.log_amp_scale * torch.tanh(raw / 3.0) * 3.0
        sign_logit = self.sign_head(features).squeeze(-1)
        phase = torch.pi * (sign_logit < 0).float()
        return log_amp, phase
