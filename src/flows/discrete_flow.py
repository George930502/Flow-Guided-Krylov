"""
Discrete Flow Sampler for Neural Quantum States.

Implements the discretization scheme from the NF-NQS paper:
- Continuous normalizing flow maps prior to posterior
- Posterior is discretized into regions corresponding to basis states
- Enables efficient sampling from high-dimensional discrete distributions

Reference:
    "Improved Ground State Estimation in Quantum Field Theories
     via Normalising Flow-Assisted Neural Quantum States"
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
import normflows as nf


class DiscreteFlowSampler(nn.Module):
    """
    Normalizing Flow-based sampler that maps continuous distributions to discrete basis states.

    The flow learns to place probability mass in regions R_x corresponding to
    high-amplitude basis states in the ground state wavefunction.

    Discretization scheme for spin-1/2:
        y_i > 0 → spin up (1)
        y_i < 0 → spin down (0)

    Args:
        num_sites: Number of sites/qubits
        num_coupling_layers: Number of RealNVP coupling layers
        hidden_dims: Hidden layer dimensions for coupling networks
        prior_std: Standard deviation of Gaussian prior modes at ±1
        n_mc_samples: Number of MC samples for probability estimation
    """

    def __init__(
        self,
        num_sites: int,
        num_coupling_layers: int = 4,
        hidden_dims: list[int] = [512, 512],
        prior_std: float = 0.33,
        n_mc_samples: int = 25,
    ):
        super().__init__()

        self.num_sites = num_sites
        self.n_mc_samples = n_mc_samples
        self.prior_std = prior_std

        # Build the normalizing flow using RealNVP architecture
        # as described in the NF-NQS paper

        # Prior: Mixture of two Gaussians at ±1 for each dimension
        # This ensures initial uniform sampling over all configurations
        self.base_dist = self._create_mixture_prior(num_sites, prior_std)

        # Create RealNVP flow
        flows = []
        for i in range(num_coupling_layers):
            # Alternate which half is transformed
            mask = self._create_mask(num_sites, i)

            # Scale and translate networks
            s_net = self._create_coupling_net(num_sites, hidden_dims)
            t_net = self._create_coupling_net(num_sites, hidden_dims)

            flows.append(
                nf.flows.MaskedAffineFlow(
                    mask,
                    t_net,
                    s_net,
                )
            )

        self.flow = nf.NormalizingFlow(self.base_dist, flows)

        # Final tanh to bound outputs to [-1, 1]
        self.final_activation = nn.Tanh()

    def _create_mixture_prior(
        self, dim: int, std: float
    ) -> "MultiModalPrior":
        """
        Create a mixture of Gaussians prior centered at ±1.

        As specified in the NF-NQS paper, the prior is a bimodal Gaussian
        at ±1 for each dimension to ensure uniform initial coverage of
        all 2^n configurations. This is essential for proper training
        convergence.
        """
        return MultiModalPrior(dim, std=std)

    def _create_mask(self, dim: int, layer_idx: int) -> torch.Tensor:
        """Create alternating mask for coupling layers."""
        mask = torch.zeros(dim)
        if layer_idx % 2 == 0:
            mask[:dim // 2] = 1
        else:
            mask[dim // 2:] = 1
        return mask

    def _create_coupling_net(
        self, dim: int, hidden_dims: list[int]
    ) -> nn.Sequential:
        """Create network for scale/translate in coupling layer.

        Note: normflows MaskedAffineFlow passes the FULL input to the network,
        using the mask only to determine which outputs to apply. So the network
        must accept full dimension as input and output.
        """
        layers = []
        in_dim = dim  # Full input dimension (normflows passes all dims)

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim

        # Output dimension is also full (mask handles selection)
        layers.extend([
            nn.Linear(in_dim, dim),
            nn.Tanh(),
        ])

        return nn.Sequential(*layers)

    def sample_continuous(self, batch_size: int) -> torch.Tensor:
        """
        Sample continuous posterior y from the flow.

        Args:
            batch_size: Number of samples

        Returns:
            Continuous samples bounded in [-1, 1], shape (batch_size, num_sites)
        """
        # Sample from prior and transform through flow
        z = self.base_dist.sample(batch_size)

        # Move z to same device as flow parameters
        device = next(self.flow.parameters()).device
        z = z.to(device)

        # Forward through flow (returns tuple, unpack first element)
        result = self.flow.forward(z)
        y = result[0] if isinstance(result, tuple) else result

        # Apply final tanh to bound outputs
        y = self.final_activation(y)

        return y

    def discretize(self, y: torch.Tensor) -> torch.Tensor:
        """
        Map continuous samples to discrete basis states.

        For spin-1/2: sign(y_i) determines spin up/down

        Args:
            y: Continuous samples, shape (batch_size, num_sites)

        Returns:
            Discrete configurations {0, 1}^n, shape (batch_size, num_sites)
        """
        return (y > 0).long()

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample discrete basis states from the flow.

        Args:
            batch_size: Number of samples

        Returns:
            (configs, unique_configs): All sampled configs and unique ones
        """
        y = self.sample_continuous(batch_size)
        configs = self.discretize(y)

        # Get unique configurations
        unique_configs = torch.unique(configs, dim=0)

        return configs, unique_configs

    def log_prob_continuous(self, y: torch.Tensor) -> torch.Tensor:
        """
        Compute log p_Y(y) for continuous samples.

        Args:
            y: Continuous samples, shape (batch_size, num_sites)

        Returns:
            Log probabilities, shape (batch_size,)
        """
        # Need to invert tanh to get pre-activation values
        # y = tanh(y_raw) → y_raw = arctanh(y)
        # Clamp to avoid numerical issues at boundaries
        y_clamped = torch.clamp(y, -1 + 1e-6, 1 - 1e-6)
        y_raw = torch.arctanh(y_clamped)

        # Jacobian of tanh: d(tanh(x))/dx = 1 - tanh²(x)
        # log|det J| = sum(log(1 - y²))
        log_det_tanh = torch.sum(torch.log(1 - y_clamped**2 + 1e-10), dim=-1)

        # Get log prob from flow
        log_prob_raw = self.flow.log_prob(y_raw)

        # Adjust for tanh transformation
        return log_prob_raw - log_det_tanh

    def estimate_discrete_prob(
        self,
        config: torch.Tensor,
        n_mc_samples: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Estimate p_φ(x) = ∫_{R_x} p_Y(y) dy using Monte Carlo.

        As specified in the NF-NQS paper, sample UNIFORMLY within region R_x
        and compute: p̂_φ(x) = (Vol(R_x)/M) Σ_{m=1}^{M} p_Y(y_m)

        For discretization y_i > 0 → 1, y_i < 0 → 0:
        - Region R_x for config x is the hypercube quadrant where signs match x
        - R_x = {y : sign(y_i) = 2*x_i - 1 for all i}
        - For bounded [-1, 1] space: R_x has volume (1)^n = 1 for each quadrant

        Args:
            config: Discrete configuration, shape (num_sites,) or (batch, num_sites)
            n_mc_samples: Number of MC samples per region

        Returns:
            Estimated discrete probability
        """
        if n_mc_samples is None:
            n_mc_samples = self.n_mc_samples

        if config.dim() == 1:
            config = config.unsqueeze(0)

        batch_size = config.shape[0]
        device = config.device

        # Region R_x for config x:
        # - If x_i = 1: y_i ∈ (0, 1)
        # - If x_i = 0: y_i ∈ (-1, 0)
        # Sample UNIFORMLY within each region (paper requirement)

        # Generate uniform samples in [0, 1]
        u_samples = torch.rand(
            n_mc_samples, batch_size, self.num_sites, device=device
        )

        # Map to correct region based on config:
        # x_i = 1: y_i = u * 1.0 + 0 = u ∈ (0, 1) → map to (eps, 1-eps)
        # x_i = 0: y_i = u * 1.0 - 1 = u - 1 ∈ (-1, 0) → map to (-1+eps, -eps)
        config_expanded = config.float().unsqueeze(0)  # (1, batch, num_sites)

        # For x=1: y ∈ (0, 1), for x=0: y ∈ (-1, 0)
        # y = u * (upper - lower) + lower
        # x=1: lower=eps, upper=1-eps
        # x=0: lower=-1+eps, upper=-eps
        eps = 1e-4  # Small margin to avoid boundary issues
        lower = config_expanded * eps + (1 - config_expanded) * (-1 + eps)
        upper = config_expanded * (1 - eps) + (1 - config_expanded) * (-eps)

        y_samples = u_samples * (upper - lower) + lower

        # Reshape for batch processing
        y_flat = y_samples.view(-1, self.num_sites)

        # Compute log probabilities p_Y(y)
        log_probs = self.log_prob_continuous(y_flat)
        log_probs = log_probs.view(n_mc_samples, batch_size)

        # MC estimate: p̂(x) = Vol(R_x) * (1/M) Σ p_Y(y_m)
        # Vol(R_x) = 1 for each quadrant in [-1,1]^n with half-space per dimension
        # Actually Vol(R_x) = 1 since each dimension contributes factor of 1
        # Use logsumexp for numerical stability
        # p̂(x) = (1/M) Σ p_Y(y_m) = exp(logsumexp(log_probs) - log(M))
        log_prob_estimate = torch.logsumexp(log_probs, dim=0) - np.log(n_mc_samples)

        return torch.exp(log_prob_estimate)

    def get_unique_samples(
        self, n_samples: int, max_attempts: int = 10
    ) -> torch.Tensor:
        """
        Sample until we get a desired number of unique configurations.

        Args:
            n_samples: Target number of unique samples
            max_attempts: Maximum sampling attempts

        Returns:
            Unique configurations, shape (n_unique, num_sites)
        """
        all_configs = []

        for _ in range(max_attempts):
            _, unique = self.sample(n_samples)
            all_configs.append(unique)

            # Combine all unique configs
            combined = torch.cat(all_configs, dim=0)
            combined = torch.unique(combined, dim=0)

            if combined.shape[0] >= n_samples:
                return combined[:n_samples]

        return torch.unique(torch.cat(all_configs, dim=0), dim=0)


class MultiModalPrior(nf.distributions.BaseDistribution):
    """
    Mixture of Gaussians prior for better initialization.

    Places modes at ±1 for each dimension to ensure uniform initial
    coverage of all 2^n configurations.
    """

    def __init__(self, dim: int, std: float = 0.33):
        super().__init__()
        self.dim = dim
        self.std = std
        log_const = -0.5 * dim * np.log(2 * np.pi * std**2)
        self.register_buffer("_log_const", torch.tensor(log_const, dtype=torch.float32))

    def sample(self, num_samples: int) -> torch.Tensor:
        """Sample from mixture of Gaussians at ±1."""
        device = self._log_const.device

        # Randomly choose mode (±1) for each dimension
        modes = 2 * torch.randint(0, 2, (num_samples, self.dim), device=device) - 1
        modes = modes.float()

        # Add Gaussian noise around modes
        noise = torch.randn(num_samples, self.dim, device=device) * self.std

        return modes + noise

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """Compute log probability under mixture."""
        # For each component, compute log prob
        # p(z) = 0.5 * N(z; -1, σ²) + 0.5 * N(z; +1, σ²)

        log_p_minus = -0.5 * ((z + 1) / self.std)**2
        log_p_plus = -0.5 * ((z - 1) / self.std)**2

        # Sum over dimensions
        log_p_minus = torch.sum(log_p_minus, dim=-1)
        log_p_plus = torch.sum(log_p_plus, dim=-1)

        # Mix with logsumexp
        log_prob = torch.logsumexp(
            torch.stack([log_p_minus, log_p_plus], dim=-1), dim=-1
        ) - np.log(2)

        return log_prob + self._log_const
