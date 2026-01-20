"""Complex-valued Neural Quantum State for systems with non-trivial phases."""

import torch
import torch.nn as nn
from typing import Optional

try:
    from .base import NeuralQuantumState
except ImportError:
    from nqs.base import NeuralQuantumState


class ComplexNQS(NeuralQuantumState):
    """
    Complex-valued Neural Quantum State.

    Uses separate networks for amplitude and phase, with optional weight sharing.
    Suitable for systems with complex ground state wavefunctions.

    Architecture:
        Amplitude: Input → Shared → Amplitude head → log|ψ|
        Phase: Input → Shared → Phase head → φ

    Args:
        num_sites: Number of sites in the system
        hidden_dims: Hidden layer dimensions for shared network
        amplitude_dims: Additional hidden dims for amplitude head
        phase_dims: Additional hidden dims for phase head
        local_dim: Local Hilbert space dimension
    """

    def __init__(
        self,
        num_sites: int,
        hidden_dims: list[int] = [256, 256],
        amplitude_dims: list[int] = [128],
        phase_dims: list[int] = [128],
        local_dim: int = 2,
    ):
        super().__init__(num_sites, local_dim, complex_output=True)

        # Shared feature extractor
        shared_layers = []
        in_dim = num_sites
        for hidden_dim in hidden_dims:
            shared_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
            ])
            in_dim = hidden_dim
        self.shared_net = nn.Sequential(*shared_layers)

        # Amplitude head
        amp_layers = []
        for amp_dim in amplitude_dims:
            amp_layers.extend([
                nn.Linear(in_dim, amp_dim),
                nn.GELU(),
            ])
            in_dim = amp_dim
        amp_layers.append(nn.Linear(in_dim, 1))
        self.amplitude_head = nn.Sequential(*amp_layers)

        # Phase head
        in_dim = hidden_dims[-1] if hidden_dims else num_sites
        phase_layers = []
        for phase_dim in phase_dims:
            phase_layers.extend([
                nn.Linear(in_dim, phase_dim),
                nn.GELU(),
            ])
            in_dim = phase_dim
        phase_layers.append(nn.Linear(in_dim, 1))
        self.phase_head = nn.Sequential(*phase_layers)

        self.log_amp_scale = nn.Parameter(torch.tensor(1.0))

    def log_amplitude(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode_configuration(x)
        features = self.shared_net(x)
        log_amp = self.amplitude_head(features).squeeze(-1)
        return self.log_amp_scale * log_amp

    def phase(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode_configuration(x)
        features = self.shared_net(x)
        phi = self.phase_head(features).squeeze(-1)
        return phi  # Unbounded phase


class RBMQuantumState(NeuralQuantumState):
    """
    Restricted Boltzmann Machine-based Neural Quantum State.

    Classic architecture from Carleo & Troyer (2017).
    Uses a single hidden layer with complex-valued weights.

    ψ(x) = exp(∑_j a_j x_j) ∏_i cosh(b_i + ∑_j W_ij x_j)

    Args:
        num_sites: Number of visible units (sites)
        num_hidden: Number of hidden units
        local_dim: Local Hilbert space dimension
        complex_weights: Whether to use complex weights
    """

    def __init__(
        self,
        num_sites: int,
        num_hidden: Optional[int] = None,
        local_dim: int = 2,
        complex_weights: bool = False,
    ):
        super().__init__(num_sites, local_dim, complex_output=complex_weights)

        if num_hidden is None:
            num_hidden = num_sites  # Default: same as visible

        self.num_hidden = num_hidden
        self.complex_weights = complex_weights

        if complex_weights:
            # Complex parameters as (real, imag) pairs
            self.a_real = nn.Parameter(torch.randn(num_sites) * 0.01)
            self.a_imag = nn.Parameter(torch.randn(num_sites) * 0.01)
            self.b_real = nn.Parameter(torch.randn(num_hidden) * 0.01)
            self.b_imag = nn.Parameter(torch.randn(num_hidden) * 0.01)
            self.W_real = nn.Parameter(torch.randn(num_hidden, num_sites) * 0.01)
            self.W_imag = nn.Parameter(torch.randn(num_hidden, num_sites) * 0.01)
        else:
            self.a = nn.Parameter(torch.randn(num_sites) * 0.01)
            self.b = nn.Parameter(torch.randn(num_hidden) * 0.01)
            self.W = nn.Parameter(torch.randn(num_hidden, num_sites) * 0.01)

    def _theta(self, x: torch.Tensor) -> torch.Tensor:
        """Compute θ_i = b_i + ∑_j W_ij x_j."""
        if self.complex_weights:
            theta_real = self.b_real + torch.matmul(x, self.W_real.T)
            theta_imag = self.b_imag + torch.matmul(x, self.W_imag.T)
            return theta_real, theta_imag
        else:
            return self.b + torch.matmul(x, self.W.T)

    def log_amplitude(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode_configuration(x)

        if self.complex_weights:
            # For complex RBM, amplitude involves |cosh(θ)|
            theta_real, theta_imag = self._theta(x)

            # Visible bias contribution (real part)
            vis_real = torch.sum(x * self.a_real, dim=-1)

            # Hidden units contribution: log|cosh(θ)|
            # |cosh(a+ib)|² = cosh²(a) - sin²(b)
            log_cosh = torch.sum(
                torch.log(torch.cosh(theta_real).abs() + 1e-10)
                + 0.5 * torch.log(1 + torch.tanh(theta_real)**2 * torch.tan(theta_imag)**2 + 1e-10),
                dim=-1
            )
            return vis_real + log_cosh
        else:
            vis = torch.sum(x * self.a, dim=-1)
            theta = self._theta(x)
            log_cosh = torch.sum(torch.log(torch.cosh(theta) + 1e-10), dim=-1)
            return vis + log_cosh

    def phase(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode_configuration(x)

        if self.complex_weights:
            theta_real, theta_imag = self._theta(x)

            # Visible bias phase contribution
            vis_phase = torch.sum(x * self.a_imag, dim=-1)

            # Hidden units phase contribution: arg(cosh(θ))
            # arg(cosh(a+ib)) = atan(tanh(a)*tan(b))
            hidden_phase = torch.sum(
                torch.atan(torch.tanh(theta_real) * torch.tan(theta_imag)),
                dim=-1
            )
            return vis_phase + hidden_phase
        else:
            return torch.zeros(x.shape[0], device=x.device)
