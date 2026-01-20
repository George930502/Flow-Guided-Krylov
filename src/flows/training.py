"""
Training procedures for Flow-assisted Neural Quantum States.

Implements the co-training algorithm from the NF-NQS paper:
1. Sample from NF to get discrete configurations
2. Evaluate NQS probabilities on sampled configurations
3. Update NF to maximize probability in high-weight regions
4. Update NQS to minimize energy expectation

Reference:
    "Improved Ground State Estimation in Quantum Field Theories
     via Normalising Flow-Assisted Neural Quantum States"
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
from tqdm import tqdm

# Support both package imports and direct script execution
try:
    from .discrete_flow import DiscreteFlowSampler
    from ..nqs.base import NeuralQuantumState
except ImportError:
    from flows.discrete_flow import DiscreteFlowSampler
    from nqs.base import NeuralQuantumState


@dataclass
class TrainingConfig:
    """Configuration for NF-NQS co-training."""

    # Sampling
    samples_per_batch: int = 3000  # Good coverage of basis states
    num_batches: int = 1  # Single batch per epoch
    n_mc_samples: int = 10  # MC samples for probability estimation

    # Optimization - slower flow LR prevents chasing NQS too aggressively
    flow_lr: float = 5e-4  # Slower for stability
    nqs_lr: float = 1e-3  # Faster for convergence
    grad_clip: float = 1.0

    # Training
    num_epochs: int = 500
    min_epochs: int = 150  # Sufficient training before checking convergence
    convergence_threshold: float = 0.20  # Train until flow concentrates well (<20% unique)

    # Energy computation
    use_local_energy: bool = False  # Use accurate subspace energy (local energy is buggy)
    use_accumulated_energy: bool = True  # Compute energy on accumulated basis for stability

    # Stability - EMA and entropy regularization
    ema_decay: float = 0.95  # Exponential moving average decay for energy tracking
    entropy_weight: float = 0.01  # Entropy regularization to prevent premature collapse

    # Logging
    log_interval: int = 10
    save_interval: int = 100


class FlowNQSTrainer:
    """
    Trainer for co-training Normalizing Flow and Neural Quantum State.

    The training alternates between:
    1. Sampling configurations from the flow
    2. Computing NQS probabilities and local energies
    3. Updating flow parameters to sample high-probability regions
    4. Updating NQS parameters to minimize energy

    Args:
        flow: DiscreteFlowSampler instance
        nqs: NeuralQuantumState instance
        hamiltonian: Callable that computes H|x⟩ for configurations
        config: Training configuration
        device: Torch device
    """

    def __init__(
        self,
        flow: DiscreteFlowSampler,
        nqs: NeuralQuantumState,
        hamiltonian: "Hamiltonian",
        config: Optional[TrainingConfig] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.flow = flow.to(device)
        self.nqs = nqs.to(device)
        self.hamiltonian = hamiltonian
        self.config = config or TrainingConfig()
        self.device = device

        # Optimizers
        self.flow_optimizer = optim.Adam(
            self.flow.parameters(), lr=self.config.flow_lr
        )
        self.nqs_optimizer = optim.Adam(
            self.nqs.parameters(), lr=self.config.nqs_lr
        )

        # History
        self.history = {
            "energies": [],
            "ema_energies": [],  # EMA-smoothed energies for stable monitoring
            "flow_loss": [],
            "nqs_loss": [],
            "unique_ratio": [],
        }

        # Accumulated basis from training (important for inference)
        self.accumulated_basis = None

        # EMA energy tracking for stability monitoring
        self.ema_energy = None

        # Cached Hamiltonian matrix for accumulated basis (updated when basis grows)
        self._cached_H_matrix = None
        self._cached_basis_size = 0

    def compute_energy_expectation(
        self,
        configs: torch.Tensor,
        use_subspace: bool = True,
    ) -> torch.Tensor:
        """
        Compute energy expectation ⟨ψ|H|ψ⟩ over sampled configurations.

        If use_subspace=True, computes energy in the subspace spanned by configs.
        Otherwise, uses local energy estimation.

        Args:
            configs: Unique configurations, shape (n_configs, num_sites)
            use_subspace: Whether to use subspace energy calculation

        Returns:
            Energy expectation value
        """
        n_configs = configs.shape[0]

        if use_subspace:
            # Compute full Hamiltonian in subspace
            # H_ij = ⟨x_i|H|x_j⟩
            H_matrix = self.hamiltonian.matrix_elements(configs.cpu(), configs.cpu())

            # Ensure H_matrix is on the correct device
            H_matrix = H_matrix.to(self.device)

            # Get wavefunction amplitudes
            psi = self.nqs.psi(configs)  # (n_configs,)

            # Normalize
            psi = psi / torch.sqrt(torch.sum(torch.abs(psi)**2))

            # E = ψ† H ψ
            if psi.is_complex():
                energy = torch.real(torch.conj(psi) @ H_matrix @ psi)
            else:
                energy = psi @ H_matrix @ psi

            return energy
        else:
            # Local energy estimation
            return self._compute_local_energy(configs)

    def _compute_local_energy(self, configs: torch.Tensor) -> torch.Tensor:
        """
        Compute energy using local energy estimator (fast version).

        Uses sampling and vectorized diagonal computation for speed.
        E_loc(x) = ⟨x|H|x⟩ + Σ_{x'≠x} ⟨x|H|x'⟩ ψ(x') / ψ(x)
        """
        n_configs = configs.shape[0]

        # Sample subset of configs for speed (max 200)
        max_samples = min(200, n_configs)
        if n_configs > max_samples:
            indices = torch.randperm(n_configs, device=configs.device)[:max_samples]
            configs_sample = configs[indices]
        else:
            configs_sample = configs

        # Get NQS amplitudes for sampled configs
        psi_x = self.nqs.psi(configs_sample)
        prob_x = torch.abs(psi_x)**2
        prob_x = prob_x / (prob_x.sum() + 1e-10)

        # Compute diagonal energies (vectorized)
        local_energies = []
        for i, config in enumerate(configs_sample):
            diag = self.hamiltonian.diagonal_element(config)

            # Get off-diagonal contributions (transverse field terms)
            connected, h_elements = self.hamiltonian.get_connections(config)

            if len(connected) > 0:
                h_elements = h_elements.to(self.device)
                connected = connected.to(self.device)
                psi_connected = self.nqs.psi(connected)
                off_diag = torch.sum(h_elements * psi_connected) / (psi_x[i] + 1e-10)
                E_loc = diag + off_diag
            else:
                E_loc = diag

            local_energies.append(E_loc)

        local_energies = torch.stack(local_energies)

        # E = Σ_x p(x) E_loc(x)
        return torch.sum(prob_x * local_energies.real if local_energies.is_complex() else prob_x * local_energies)

    def compute_nqs_probabilities(
        self, configs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute normalized NQS probabilities p_θ(x) = |ψ_θ(x)|² / Z.

        Args:
            configs: Configurations, shape (n_configs, num_sites)

        Returns:
            Normalized probabilities, shape (n_configs,)
        """
        log_amp = self.nqs.log_amplitude(configs)
        log_prob = 2 * log_amp

        # Normalize over sampled configurations
        log_Z = torch.logsumexp(log_prob, dim=0)
        prob = torch.exp(log_prob - log_Z)

        return prob

    def compute_flow_loss(
        self,
        configs: torch.Tensor,
        nqs_probs: torch.Tensor,
        energy: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute NF loss: cross-entropy weighted by energy magnitude with entropy regularization.

        L_φ = -|E| / |S| Σ_x p_θ(x) log(p̂_φ(x)) - β * H(p_φ)

        The entropy term H(p_φ) = -Σ_x p_φ(x) log(p_φ(x)) prevents premature collapse
        by encouraging the flow to maintain exploration of the configuration space.

        Args:
            configs: Unique configurations
            nqs_probs: NQS probabilities p_θ(x)
            energy: Current energy estimate

        Returns:
            Flow loss value
        """
        n_configs = configs.shape[0]

        # Estimate discrete probabilities from flow
        flow_probs = self.flow.estimate_discrete_prob(configs)

        # Cross-entropy loss weighted by NQS probability
        log_flow_probs = torch.log(flow_probs + 1e-10)
        cross_entropy = -torch.sum(nqs_probs * log_flow_probs)

        # Weight by energy magnitude (helps convergence per paper)
        loss = torch.abs(energy) * cross_entropy / n_configs

        # Entropy regularization to prevent premature collapse
        # H(p) = -Σ p(x) log(p(x)), we want to maximize entropy (subtract from loss)
        entropy = -torch.sum(flow_probs * log_flow_probs)
        loss = loss - self.config.entropy_weight * entropy

        return loss

    def train_step(self) -> Dict[str, float]:
        """
        Perform one training step with stabilization mechanisms.

        Includes:
        - Energy computation on accumulated basis (optional, more stable)
        - EMA energy tracking for smooth convergence monitoring
        - Entropy regularization in flow loss

        Returns:
            Dictionary with loss values and metrics
        """
        total_flow_loss = 0.0
        total_nqs_loss = 0.0
        total_energy = 0.0
        total_unique_ratio = 0.0

        for batch_idx in range(self.config.num_batches):
            # Sample configurations from flow
            all_configs, unique_configs = self.flow.sample(
                self.config.samples_per_batch
            )

            n_unique = unique_configs.shape[0]
            unique_ratio = n_unique / self.config.samples_per_batch
            total_unique_ratio += unique_ratio

            # Accumulate basis states (important for inference phase)
            if self.accumulated_basis is None:
                self.accumulated_basis = unique_configs.clone()
            else:
                combined = torch.cat([self.accumulated_basis, unique_configs], dim=0)
                self.accumulated_basis = torch.unique(combined, dim=0)

            # Compute NQS probabilities on current sample (for flow loss)
            nqs_probs = self.compute_nqs_probabilities(unique_configs)

            # Compute energy - either on accumulated basis (stable) or current sample
            if self.config.use_accumulated_energy and self.accumulated_basis is not None:
                # Use accumulated basis for more stable energy estimation
                energy = self.compute_energy_expectation(
                    self.accumulated_basis,
                    use_subspace=not self.config.use_local_energy
                )
            else:
                # Use current sample only
                energy = self.compute_energy_expectation(
                    unique_configs,
                    use_subspace=not self.config.use_local_energy
                )
            total_energy += energy.item()

            # Update EMA energy
            if self.ema_energy is None:
                self.ema_energy = energy.item()
            else:
                self.ema_energy = (
                    self.config.ema_decay * self.ema_energy +
                    (1 - self.config.ema_decay) * energy.item()
                )

            # Compute flow loss (includes entropy regularization)
            flow_loss = self.compute_flow_loss(unique_configs, nqs_probs, energy)
            total_flow_loss += flow_loss.item()

            # Update flow
            self.flow_optimizer.zero_grad()
            flow_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(
                self.flow.parameters(), self.config.grad_clip
            )
            self.flow_optimizer.step()

            # Update NQS with energy as loss
            nqs_loss = energy
            total_nqs_loss += nqs_loss.item()

            self.nqs_optimizer.zero_grad()
            nqs_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.nqs.parameters(), self.config.grad_clip
            )
            self.nqs_optimizer.step()

        return {
            "flow_loss": total_flow_loss / self.config.num_batches,
            "nqs_loss": total_nqs_loss / self.config.num_batches,
            "energy": total_energy / self.config.num_batches,
            "ema_energy": self.ema_energy,
            "unique_ratio": total_unique_ratio / self.config.num_batches,
        }

    def train(
        self,
        num_epochs: Optional[int] = None,
        callback: Optional[Callable[[int, Dict], None]] = None,
    ) -> Dict[str, list]:
        """
        Run full training loop with stabilization monitoring.

        Args:
            num_epochs: Number of training epochs (overrides config)
            callback: Optional callback called after each epoch

        Returns:
            Training history dictionary
        """
        if num_epochs is None:
            num_epochs = self.config.num_epochs

        pbar = tqdm(range(num_epochs), desc="Training NF-NQS")

        for epoch in pbar:
            metrics = self.train_step()

            # Record history
            self.history["energies"].append(metrics["energy"])
            self.history["ema_energies"].append(metrics["ema_energy"])
            self.history["flow_loss"].append(metrics["flow_loss"])
            self.history["nqs_loss"].append(metrics["nqs_loss"])
            self.history["unique_ratio"].append(metrics["unique_ratio"])

            # Update progress bar with EMA energy (more stable display)
            pbar.set_postfix({
                "E": f"{metrics['energy']:.4f}",
                "EMA": f"{metrics['ema_energy']:.4f}",
                "unique": f"{metrics['unique_ratio']:.2f}",
                "basis": len(self.accumulated_basis) if self.accumulated_basis is not None else 0,
            })

            # Callback
            if callback is not None:
                callback(epoch, metrics)

            # Check convergence (unique ratio drops below threshold)
            # Only check after minimum epochs to allow training to progress
            if epoch >= self.config.min_epochs and metrics["unique_ratio"] < self.config.convergence_threshold:
                print(f"\nConverged at epoch {epoch}: unique ratio = {metrics['unique_ratio']:.2f}, "
                      f"EMA energy = {metrics['ema_energy']:.4f}")
                break

        return self.history

    def extract_basis(self, n_samples: int = 10000) -> torch.Tensor:
        """
        Extract basis states from the trained flow.

        Uses accumulated basis from training if available (preferred),
        otherwise samples fresh from the frozen flow.

        Args:
            n_samples: Number of samples to draw (only used if no accumulated basis)

        Returns:
            Unique configurations forming the basis
        """
        self.flow.eval()

        # Prefer accumulated basis from training (preserves important states)
        if self.accumulated_basis is not None and len(self.accumulated_basis) > 0:
            print(f"Using accumulated basis from training: {len(self.accumulated_basis)} states")
            return self.accumulated_basis

        # Fallback: sample fresh from flow
        with torch.no_grad():
            _, unique_configs = self.flow.sample(n_samples)

        return unique_configs

    def save_checkpoint(self, path: str):
        """Save model checkpoints and training state."""
        torch.save({
            "flow_state_dict": self.flow.state_dict(),
            "nqs_state_dict": self.nqs.state_dict(),
            "flow_optimizer": self.flow_optimizer.state_dict(),
            "nqs_optimizer": self.nqs_optimizer.state_dict(),
            "history": self.history,
            "config": self.config,
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoints and training state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.flow.load_state_dict(checkpoint["flow_state_dict"])
        self.nqs.load_state_dict(checkpoint["nqs_state_dict"])
        self.flow_optimizer.load_state_dict(checkpoint["flow_optimizer"])
        self.nqs_optimizer.load_state_dict(checkpoint["nqs_optimizer"])
        self.history = checkpoint["history"]


class InferenceNQSTrainer:
    """
    Trainer for the inference phase after flow convergence.

    After the flow has converged, we freeze it and train a fresh NQS
    to accurately learn the amplitudes in the discovered subspace.
    """

    def __init__(
        self,
        flow: DiscreteFlowSampler,
        nqs: NeuralQuantumState,
        hamiltonian: "Hamiltonian",
        lr: float = 1e-3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.flow = flow.to(device)
        self.flow.eval()  # Freeze flow
        self.nqs = nqs.to(device)
        self.hamiltonian = hamiltonian
        self.device = device

        self.optimizer = optim.Adam(self.nqs.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.5, patience=20
        )

    def train(
        self,
        num_iters: int = 2000,
        n_samples: int = 5000,
    ) -> Dict[str, list]:
        """
        Train NQS on fixed subspace from flow.

        Args:
            num_iters: Number of training iterations
            n_samples: Number of samples per iteration

        Returns:
            Training history
        """
        history = {"energies": []}

        # Sample fixed basis from frozen flow
        with torch.no_grad():
            _, basis = self.flow.sample(n_samples)

        # Precompute Hamiltonian matrix (it doesn't change during training)
        # Compute on CPU then move to device for consistency
        H_matrix = self.hamiltonian.matrix_elements(basis.cpu(), basis.cpu())
        H_matrix = H_matrix.to(self.device)

        # Ensure basis is on correct device
        basis = basis.to(self.device)

        pbar = tqdm(range(num_iters), desc="Inference NQS Training")

        for iteration in pbar:
            # Compute energy in subspace
            psi = self.nqs.psi(basis)
            psi_norm = psi / torch.sqrt(torch.sum(torch.abs(psi)**2))

            if psi.is_complex():
                energy = torch.real(torch.conj(psi_norm) @ H_matrix @ psi_norm)
            else:
                energy = psi_norm @ H_matrix @ psi_norm

            # Update NQS
            self.optimizer.zero_grad()
            energy.backward()
            self.optimizer.step()
            self.scheduler.step(energy)

            history["energies"].append(energy.item())
            pbar.set_postfix({"E": f"{energy.item():.6f}"})

        return history
