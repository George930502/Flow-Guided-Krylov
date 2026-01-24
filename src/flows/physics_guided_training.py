"""
Physics-Guided Training for Normalizing Flows.

This module implements mixed-objective training that combines:
1. Teacher signal: Match NQS probability distribution (standard approach)
2. Physics signal: Energy-weighted importance (not just probability)
3. Exploration bonus: Entropy regularization to avoid collapse

The key insight is that NF should learn to sample configurations that:
- Have high NQS probability (teacher)
- Have LOW local energy (physics - ground state has lowest energy)
- Maintain diversity (exploration)

References:
- NF-NQS paper: "Improved Ground State Estimation via NF-Assisted NQS"
- Importance sampling: configurations with low local energy matter more
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from tqdm import tqdm


@dataclass
class PhysicsGuidedConfig:
    """Configuration for physics-guided flow training."""

    # Batch sizes
    samples_per_batch: int = 2000
    num_batches: int = 1

    # Learning rates
    flow_lr: float = 5e-4
    nqs_lr: float = 1e-3

    # Training epochs
    num_epochs: int = 500
    min_epochs: int = 150
    convergence_threshold: float = 0.20

    # Loss weights (sum to 1.0 recommended)
    teacher_weight: float = 0.5  # Match NQS probability
    physics_weight: float = 0.4  # Energy-based importance
    entropy_weight: float = 0.1  # Exploration bonus

    # Energy baseline for physics signal
    use_energy_baseline: bool = True  # Subtract baseline for variance reduction

    # Accumulated basis for energy computation
    use_accumulated_energy: bool = True
    max_accumulated_basis: int = 2048
    accumulated_energy_interval: int = 4
    prune_basis_threshold: float = 1e-6

    # EMA for stable tracking
    ema_decay: float = 0.95

    # Temperature annealing for particle-conserving flow
    initial_temperature: float = 1.0
    final_temperature: float = 0.1
    temperature_decay_epochs: int = 200


class PhysicsGuidedFlowTrainer:
    """
    Trainer with mixed-objective loss for normalizing flows.

    The training objective combines three signals:
    1. Teacher (KL divergence): Flow matches NQS probability
    2. Physics (energy importance): Flow favors low-energy configurations
    3. Exploration (entropy): Flow maintains sampling diversity

    Loss = w_teacher * L_teacher + w_physics * L_physics - w_entropy * H(flow)
    """

    def __init__(
        self,
        flow: nn.Module,
        nqs: nn.Module,
        hamiltonian: Any,
        config: PhysicsGuidedConfig,
        device: str = "cuda",
    ):
        self.flow = flow
        self.nqs = nqs
        self.hamiltonian = hamiltonian
        self.config = config
        self.device = device

        # Optimizers
        self.flow_optimizer = torch.optim.AdamW(
            flow.parameters(), lr=config.flow_lr, weight_decay=1e-5
        )
        self.nqs_optimizer = torch.optim.AdamW(
            nqs.parameters(), lr=config.nqs_lr, weight_decay=1e-5
        )

        # Schedulers
        self.flow_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.flow_optimizer, T_max=config.num_epochs, eta_min=1e-6
        )
        self.nqs_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.nqs_optimizer, T_max=config.num_epochs, eta_min=1e-6
        )

        # Accumulated basis
        self.accumulated_basis = None

        # Tracking
        self.energy_ema = None
        self.history = {
            'energies': [],
            'accumulated_energies': [],
            'teacher_losses': [],
            'physics_losses': [],
            'entropy_values': [],
            'unique_ratios': [],
            'basis_sizes': [],
        }

    def train(self) -> Dict[str, list]:
        """Run physics-guided training loop."""
        config = self.config

        print(f"Starting physics-guided NF-NQS training")
        print(f"  Teacher weight: {config.teacher_weight}")
        print(f"  Physics weight: {config.physics_weight}")
        print(f"  Entropy weight: {config.entropy_weight}")

        pbar = tqdm(range(config.num_epochs), desc="Training")

        for epoch in pbar:
            # Temperature annealing for particle-conserving flow
            if hasattr(self.flow, 'set_temperature'):
                progress = min(1.0, epoch / config.temperature_decay_epochs)
                temperature = config.initial_temperature + progress * (
                    config.final_temperature - config.initial_temperature
                )
                self.flow.set_temperature(temperature)

            # Training step
            metrics = self._train_epoch(epoch)

            # Update history
            self.history['energies'].append(metrics['energy'])
            self.history['teacher_losses'].append(metrics['teacher_loss'])
            self.history['physics_losses'].append(metrics['physics_loss'])
            self.history['entropy_values'].append(metrics['entropy'])
            self.history['unique_ratios'].append(metrics['unique_ratio'])

            if 'accumulated_energy' in metrics:
                self.history['accumulated_energies'].append(metrics['accumulated_energy'])
            if self.accumulated_basis is not None:
                self.history['basis_sizes'].append(len(self.accumulated_basis))

            # Update schedulers
            self.flow_scheduler.step()
            self.nqs_scheduler.step()

            # Progress bar update
            pbar.set_postfix({
                'E': f"{metrics['energy']:.4f}",
                'unique': f"{metrics['unique_ratio']:.2f}",
                'T_loss': f"{metrics['teacher_loss']:.4f}",
                'P_loss': f"{metrics['physics_loss']:.4f}",
            })

            # Check convergence
            if epoch >= config.min_epochs:
                if metrics['unique_ratio'] < config.convergence_threshold:
                    print(f"\nConverged at epoch {epoch}: unique_ratio={metrics['unique_ratio']:.3f}")
                    break

        return self.history

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Single training epoch."""
        config = self.config
        self.flow.train()
        self.nqs.train()

        total_metrics = {
            'energy': 0.0,
            'teacher_loss': 0.0,
            'physics_loss': 0.0,
            'entropy': 0.0,
            'unique_ratio': 0.0,
        }

        for batch_idx in range(config.num_batches):
            # Sample from flow
            if hasattr(self.flow, 'sample_with_probs'):
                configs, log_probs, unique_configs = self.flow.sample_with_probs(
                    config.samples_per_batch
                )
            else:
                log_probs, unique_configs = self.flow.sample(config.samples_per_batch)
                configs = unique_configs

            n_unique = len(unique_configs)
            unique_ratio = n_unique / config.samples_per_batch

            # Compute NQS probabilities
            with torch.no_grad():
                nqs_log_amp = self.nqs.log_amplitude(unique_configs.float())
                nqs_probs = torch.exp(2 * nqs_log_amp)  # |psi|^2 = exp(2*log|psi|)
                nqs_probs = nqs_probs / nqs_probs.sum()

            # Compute local energies for physics signal
            local_energies = self._compute_local_energies(unique_configs)

            # Compute NQS energy estimate
            energy = (local_energies * nqs_probs).sum()

            # Update accumulated basis
            self._update_accumulated_basis(unique_configs)

            # Compute flow loss with mixed objectives
            flow_loss, loss_components = self._compute_flow_loss(
                configs, unique_configs, nqs_probs, local_energies, energy
            )

            # NQS loss (minimize energy)
            nqs_loss = self._compute_nqs_loss(unique_configs, nqs_probs, local_energies)

            # Backward pass
            self.flow_optimizer.zero_grad()
            self.nqs_optimizer.zero_grad()

            flow_loss.backward(retain_graph=True)
            nqs_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.flow.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.nqs.parameters(), max_norm=1.0)

            self.flow_optimizer.step()
            self.nqs_optimizer.step()

            # Accumulate metrics
            total_metrics['energy'] += energy.item()
            total_metrics['teacher_loss'] += loss_components['teacher'].item()
            total_metrics['physics_loss'] += loss_components['physics'].item()
            total_metrics['entropy'] += loss_components['entropy'].item()
            total_metrics['unique_ratio'] += unique_ratio

        # Average metrics
        for key in total_metrics:
            total_metrics[key] /= config.num_batches

        # Update EMA
        if self.energy_ema is None:
            self.energy_ema = total_metrics['energy']
        else:
            self.energy_ema = (config.ema_decay * self.energy_ema +
                             (1 - config.ema_decay) * total_metrics['energy'])
        total_metrics['energy_ema'] = self.energy_ema

        # Compute accumulated energy periodically
        if (config.use_accumulated_energy and
            epoch % config.accumulated_energy_interval == 0 and
            self.accumulated_basis is not None):
            acc_energy = self._compute_accumulated_energy()
            total_metrics['accumulated_energy'] = acc_energy

        return total_metrics

    def _compute_local_energies(self, configs: torch.Tensor) -> torch.Tensor:
        """
        Compute local energies E_loc(x) = <x|H|psi>/<x|psi>.

        For configurations in the NQS support.
        """
        n_configs = len(configs)
        local_energies = torch.zeros(n_configs, device=self.device)

        with torch.no_grad():
            # Diagonal elements
            diag = self.hamiltonian.diagonal_elements_batch(configs)

            # NQS amplitudes (use log_amplitude method)
            log_psi = self.nqs.log_amplitude(configs.float())

            for i in range(n_configs):
                # Off-diagonal connections
                connected, elements = self.hamiltonian.get_connections(configs[i])

                e_loc = diag[i]
                if len(connected) > 0:
                    log_psi_connected = self.nqs.log_amplitude(connected.float())
                    ratio = torch.exp(log_psi_connected - log_psi[i])
                    e_loc = e_loc + (elements * ratio).sum()

                local_energies[i] = e_loc.real if torch.is_complex(e_loc) else e_loc

        return local_energies

    def _compute_flow_loss(
        self,
        all_configs: torch.Tensor,
        unique_configs: torch.Tensor,
        nqs_probs: torch.Tensor,
        local_energies: torch.Tensor,
        energy: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute mixed-objective flow loss.

        L = w_t * L_teacher + w_p * L_physics - w_e * H(flow)

        Teacher loss: Cross-entropy between flow and NQS distributions
        Physics loss: Encourage sampling low-energy configurations
        Entropy: Maintain exploration diversity
        """
        config = self.config

        # Get flow probabilities
        flow_probs = self.flow.estimate_discrete_prob(unique_configs)
        flow_probs = flow_probs / (flow_probs.sum() + 1e-10)
        log_flow_probs = torch.log(flow_probs + 1e-10)

        # === Teacher Loss ===
        # KL(NQS || Flow) = sum p_nqs * (log p_nqs - log p_flow)
        teacher_loss = -torch.sum(nqs_probs.detach() * log_flow_probs)

        # === Physics Loss ===
        # Encourage flow to sample low-energy configurations
        # Use energy importance: w_i propto exp(-beta * E_loc_i)
        if config.use_energy_baseline:
            # Subtract baseline for variance reduction
            energy_deviation = local_energies - energy.detach()
        else:
            energy_deviation = local_energies

        # Soft importance weighting (lower energy = higher importance)
        # We want flow to assign higher probability to lower energy configs
        # Physics loss = E_flow[E_loc] (minimize expected energy under flow)
        physics_loss = (flow_probs * energy_deviation.detach()).sum()

        # === Entropy Bonus ===
        # H(flow) = -sum p_flow * log p_flow
        entropy = -torch.sum(flow_probs * log_flow_probs)

        # Combined loss
        total_loss = (
            config.teacher_weight * teacher_loss +
            config.physics_weight * physics_loss -
            config.entropy_weight * entropy
        )

        # Scale by energy magnitude for stability
        total_loss = total_loss / (torch.abs(energy.detach()) + 1.0)

        components = {
            'teacher': teacher_loss,
            'physics': physics_loss,
            'entropy': entropy,
        }

        return total_loss, components

    def _compute_nqs_loss(
        self,
        configs: torch.Tensor,
        probs: torch.Tensor,
        local_energies: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute NQS loss (energy minimization).

        Uses variance-reduced estimator:
        L = <E_loc> + lambda * Var(E_loc)
        """
        # Recompute with gradients
        log_amp = self.nqs.log_amplitude(configs.float())
        log_probs = 2 * log_amp  # log(|psi|^2) = 2 * log|psi|

        # REINFORCE-style gradient
        # d<E>/d_theta = 2 * Re[<(E_loc - <E>) * d log psi/d_theta>]
        energy = (local_energies.detach() * probs.detach()).sum()
        centered_energies = local_energies.detach() - energy

        # Policy gradient loss
        loss = (centered_energies * log_probs * probs.detach()).sum()

        return loss

    def _update_accumulated_basis(self, new_configs: torch.Tensor):
        """Add new configurations to accumulated basis."""
        if self.accumulated_basis is None:
            self.accumulated_basis = new_configs.clone()
        else:
            combined = torch.cat([self.accumulated_basis, new_configs], dim=0)
            self.accumulated_basis = torch.unique(combined, dim=0)

        # Prune if too large
        max_size = self.config.max_accumulated_basis
        if len(self.accumulated_basis) > max_size:
            # Keep most recent and random subset
            n_keep = max_size
            indices = torch.randperm(len(self.accumulated_basis))[:n_keep]
            self.accumulated_basis = self.accumulated_basis[indices]

    def _compute_accumulated_energy(self) -> float:
        """Compute energy in accumulated basis via diagonalization."""
        if self.accumulated_basis is None or len(self.accumulated_basis) == 0:
            return float('inf')

        with torch.no_grad():
            H_matrix = self.hamiltonian.matrix_elements(
                self.accumulated_basis, self.accumulated_basis
            )
            H_np = H_matrix.cpu().numpy()

            # Diagonalize
            eigenvalues, _ = np.linalg.eigh(H_np)
            return float(eigenvalues[0])


def create_physics_guided_trainer(
    flow: nn.Module,
    nqs: nn.Module,
    hamiltonian: Any,
    device: str = "cuda",
    teacher_weight: float = 0.5,
    physics_weight: float = 0.4,
    entropy_weight: float = 0.1,
    **kwargs,
) -> PhysicsGuidedFlowTrainer:
    """
    Factory function to create physics-guided trainer.

    Args:
        flow: Normalizing flow model
        nqs: Neural quantum state model
        hamiltonian: System Hamiltonian
        device: Compute device
        teacher_weight: Weight for teacher signal (match NQS)
        physics_weight: Weight for physics signal (favor low energy)
        entropy_weight: Weight for entropy bonus (exploration)
        **kwargs: Additional config parameters

    Returns:
        Configured PhysicsGuidedFlowTrainer
    """
    config = PhysicsGuidedConfig(
        teacher_weight=teacher_weight,
        physics_weight=physics_weight,
        entropy_weight=entropy_weight,
        **kwargs,
    )

    return PhysicsGuidedFlowTrainer(
        flow=flow,
        nqs=nqs,
        hamiltonian=hamiltonian,
        config=config,
        device=device,
    )
