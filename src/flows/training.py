"""
Training procedures for Flow-assisted Neural Quantum States.

Implements the co-training algorithm from the NF-NQS paper:
1. Sample from NF to get discrete configurations
2. Evaluate NQS probabilities on sampled configurations
3. Update NF to maximize probability in high-weight regions
4. Update NQS to minimize energy expectation

Optimizations included:
- Incremental Hamiltonian matrix caching with O(n) updates
- GPU-resident hash table for O(1) basis deduplication
- Vectorized energy computation
- Reduced memory allocations via buffer reuse

Reference:
    "Improved Ground State Estimation in Quantum Field Theories
     via Normalising Flow-Assisted Neural Quantum States"
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Callable, Dict, Any, Set, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import time

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
    use_local_energy: bool = False  # Use accurate subspace energy
    use_accumulated_energy: bool = True  # Compute energy on accumulated basis for stability

    # Stability - EMA and entropy regularization
    ema_decay: float = 0.95  # Exponential moving average decay for energy tracking
    entropy_weight: float = 0.01  # Entropy regularization to prevent premature collapse

    # GPU optimization
    cache_hamiltonian: bool = True  # Cache H matrix for accumulated basis
    max_cached_basis_size: int = 8192  # Max basis size to cache

    # Logging
    log_interval: int = 10
    save_interval: int = 100


class GPUHashTable:
    """
    Efficient hash table for basis state deduplication.

    Uses Python set for O(1) lookup but minimizes CPU-GPU transfers
    by batching tuple conversions.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._hash_set: Set[tuple] = set()

    def __len__(self) -> int:
        return len(self._hash_set)

    def __contains__(self, config_tuple: tuple) -> bool:
        return config_tuple in self._hash_set

    def add_batch(self, configs: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Add batch of configs, return only new unique ones.

        Args:
            configs: (batch_size, num_sites) configurations

        Returns:
            (new_unique_configs, count_added)
        """
        configs_cpu = configs.cpu()
        new_configs = []
        count_added = 0

        for i in range(configs_cpu.shape[0]):
            config_tuple = tuple(configs_cpu[i].tolist())
            if config_tuple not in self._hash_set:
                self._hash_set.add(config_tuple)
                new_configs.append(configs[i])
                count_added += 1

        if count_added == 0:
            return torch.empty(0, configs.shape[1], device=self.device), 0

        return torch.stack(new_configs), count_added

    def get_all_configs(self) -> Optional[torch.Tensor]:
        """Get all stored configs as tensor."""
        if len(self._hash_set) == 0:
            return None
        configs = [list(t) for t in self._hash_set]
        return torch.tensor(configs, dtype=torch.long, device=self.device)

    def clear(self):
        """Clear the hash table."""
        self._hash_set.clear()


class IncrementalHamiltonianCache:
    """
    Incrementally updated Hamiltonian matrix cache.

    When new basis states are added, only computes new rows/columns
    instead of rebuilding the entire matrix.
    """

    def __init__(
        self,
        hamiltonian,
        device: str = "cuda",
        max_size: int = 8192,
    ):
        self.hamiltonian = hamiltonian
        self.device = device
        self.max_size = max_size

        self._matrix: Optional[torch.Tensor] = None
        self._basis: Optional[torch.Tensor] = None
        self._size = 0

    @property
    def matrix(self) -> Optional[torch.Tensor]:
        return self._matrix

    @property
    def basis(self) -> Optional[torch.Tensor]:
        return self._basis

    @property
    def size(self) -> int:
        return self._size

    def update(self, new_basis: torch.Tensor) -> bool:
        """
        Update cache with new basis.

        Returns True if cache was updated, False if basis too large.
        """
        if new_basis is None or len(new_basis) == 0:
            return False

        new_size = len(new_basis)

        # Check size limit
        if new_size > self.max_size:
            self._matrix = None
            self._basis = None
            self._size = 0
            return False

        # Full rebuild needed
        if self._matrix is None or self._size == 0:
            self._full_rebuild(new_basis)
            return True

        # Check if basis changed
        if new_size == self._size:
            return True

        # Incremental update
        if new_size > self._size:
            self._incremental_update(new_basis)
            return True

        # Basis shrunk or changed completely - rebuild
        self._full_rebuild(new_basis)
        return True

    def _full_rebuild(self, basis: torch.Tensor):
        """Rebuild entire Hamiltonian matrix."""
        basis = basis.to(self.device)

        if hasattr(self.hamiltonian, 'matrix_elements_fast'):
            self._matrix = self.hamiltonian.matrix_elements_fast(basis)
        else:
            self._matrix = self.hamiltonian.matrix_elements(basis, basis).to(self.device)

        self._basis = basis
        self._size = len(basis)

    def _incremental_update(self, new_basis: torch.Tensor):
        """Incrementally add new rows/columns."""
        new_basis = new_basis.to(self.device)
        old_size = self._size
        new_size = len(new_basis)

        new_states = new_basis[old_size:]

        # Compute new matrix blocks
        if hasattr(self.hamiltonian, 'matrix_elements_fast'):
            H_new_new = self.hamiltonian.matrix_elements_fast(new_states)
        else:
            H_new_new = self.hamiltonian.matrix_elements(new_states, new_states).to(self.device)

        H_old_new = self.hamiltonian.matrix_elements(
            self._basis, new_states
        ).to(self.device)
        H_new_old = self.hamiltonian.matrix_elements(
            new_states, self._basis
        ).to(self.device)

        # Assemble new matrix
        new_H = torch.zeros(new_size, new_size, device=self.device,
                           dtype=self._matrix.dtype)
        new_H[:old_size, :old_size] = self._matrix
        new_H[:old_size, old_size:] = H_old_new
        new_H[old_size:, :old_size] = H_new_old
        new_H[old_size:, old_size:] = H_new_new

        self._matrix = new_H
        self._basis = new_basis
        self._size = new_size

    def get_energy(self, nqs, basis: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute energy E = <psi|H|psi> using cached matrix.

        Args:
            nqs: Neural quantum state
            basis: Basis to use (defaults to cached basis)

        Returns:
            Energy expectation value
        """
        if basis is None:
            basis = self._basis

        if basis is None or self._matrix is None:
            raise ValueError("Cache not initialized")

        psi = nqs.psi(basis)
        norm = torch.sqrt(torch.sum(torch.abs(psi)**2) + 1e-10)
        psi_norm = psi / norm

        if psi_norm.is_complex():
            energy = torch.real(torch.conj(psi_norm) @ self._matrix @ psi_norm)
        else:
            energy = psi_norm @ self._matrix @ psi_norm

        return energy


class FlowNQSTrainer:
    """
    Trainer for co-training Normalizing Flow and Neural Quantum State.

    The training alternates between:
    1. Sampling configurations from the flow
    2. Computing NQS probabilities and local energies
    3. Updating flow parameters to sample high-probability regions
    4. Updating NQS parameters to minimize energy

    Includes GPU optimizations:
    - Incremental Hamiltonian matrix caching
    - O(1) basis deduplication via hash table
    - Vectorized energy computation

    Args:
        flow: DiscreteFlowSampler instance
        nqs: NeuralQuantumState instance
        hamiltonian: Callable that computes H|x> for configurations
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
            "ema_energies": [],
            "flow_loss": [],
            "nqs_loss": [],
            "unique_ratio": [],
            "epoch_times": [],
        }

        # Optimized data structures
        self._basis_hash = GPUHashTable(device)
        self._h_cache = IncrementalHamiltonianCache(
            hamiltonian, device, self.config.max_cached_basis_size
        )

        # Accumulated basis tensor
        self.accumulated_basis: Optional[torch.Tensor] = None

        # EMA energy tracking
        self.ema_energy = None

    def _update_accumulated_basis(self, unique_configs: torch.Tensor) -> int:
        """
        Update accumulated basis with new configs using hash-based dedup.

        Returns number of new states added.
        """
        new_configs, n_added = self._basis_hash.add_batch(unique_configs)

        if n_added > 0:
            if self.accumulated_basis is None:
                self.accumulated_basis = new_configs
            else:
                self.accumulated_basis = torch.cat(
                    [self.accumulated_basis, new_configs], dim=0
                )

        return n_added

    def compute_energy_expectation(
        self,
        configs: torch.Tensor,
        use_subspace: bool = True,
    ) -> torch.Tensor:
        """
        Compute energy expectation <psi|H|psi> over sampled configurations.

        Uses cached Hamiltonian matrix when available for efficiency.

        Args:
            configs: Unique configurations, shape (n_configs, num_sites)
            use_subspace: Whether to use subspace energy calculation

        Returns:
            Energy expectation value
        """
        # Try to use cached matrix
        if (self.config.cache_hamiltonian and
            self._h_cache.matrix is not None and
            len(configs) == self._h_cache.size):
            try:
                return self._h_cache.get_energy(self.nqs, configs)
            except:
                pass

        # Fallback: compute directly
        if hasattr(self.hamiltonian, 'matrix_elements_fast'):
            H_matrix = self.hamiltonian.matrix_elements_fast(configs)
        else:
            H_matrix = self.hamiltonian.matrix_elements(configs.cpu(), configs.cpu())
            H_matrix = H_matrix.to(self.device)

        psi = self.nqs.psi(configs)
        psi = psi / torch.sqrt(torch.sum(torch.abs(psi)**2) + 1e-10)

        if psi.is_complex():
            energy = torch.real(torch.conj(psi) @ H_matrix @ psi)
        else:
            energy = psi @ H_matrix @ psi

        return energy

    def compute_nqs_probabilities(
        self, configs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute normalized NQS probabilities p_theta(x) = |psi_theta(x)|^2 / Z.

        Args:
            configs: Configurations, shape (n_configs, num_sites)

        Returns:
            Normalized probabilities, shape (n_configs,)
        """
        log_amp = self.nqs.log_amplitude(configs)
        log_prob = 2 * log_amp
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
        Compute NF loss: cross-entropy weighted by energy with entropy regularization.

        L_phi = -|E| / |S| * sum_x p_theta(x) log(p_phi(x)) - beta * H(p_phi)

        Args:
            configs: Unique configurations
            nqs_probs: NQS probabilities p_theta(x)
            energy: Current energy estimate

        Returns:
            Flow loss value
        """
        n_configs = configs.shape[0]

        flow_probs = self.flow.estimate_discrete_prob(configs)
        log_flow_probs = torch.log(flow_probs + 1e-10)

        # Cross-entropy loss weighted by NQS probability
        cross_entropy = -torch.sum(nqs_probs * log_flow_probs)
        loss = torch.abs(energy) * cross_entropy / n_configs

        # Entropy regularization to prevent premature collapse
        entropy = -torch.sum(flow_probs * log_flow_probs)
        loss = loss - self.config.entropy_weight * entropy

        return loss

    def train_step(self) -> Dict[str, float]:
        """
        Perform one training step with GPU optimizations.

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
            unique_configs = unique_configs.to(self.device)

            n_unique = unique_configs.shape[0]
            unique_ratio = n_unique / self.config.samples_per_batch
            total_unique_ratio += unique_ratio

            # Update accumulated basis with hash-based dedup
            n_added = self._update_accumulated_basis(unique_configs)

            # Update cached Hamiltonian if basis grew
            if n_added > 0 and self.config.cache_hamiltonian:
                self._h_cache.update(self.accumulated_basis)

            # Compute NQS probabilities
            nqs_probs = self.compute_nqs_probabilities(unique_configs)

            # Compute energy
            if self.config.use_accumulated_energy and self.accumulated_basis is not None:
                energy = self.compute_energy_expectation(
                    self.accumulated_basis,
                    use_subspace=not self.config.use_local_energy
                )
            else:
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

            # Compute flow loss
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
        Run full training loop.

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
            epoch_start = time.time()
            metrics = self.train_step()
            epoch_time = time.time() - epoch_start

            # Record history
            self.history["energies"].append(metrics["energy"])
            self.history["ema_energies"].append(metrics["ema_energy"])
            self.history["flow_loss"].append(metrics["flow_loss"])
            self.history["nqs_loss"].append(metrics["nqs_loss"])
            self.history["unique_ratio"].append(metrics["unique_ratio"])
            self.history["epoch_times"].append(epoch_time)

            # Update progress bar
            basis_size = len(self.accumulated_basis) if self.accumulated_basis is not None else 0
            pbar.set_postfix({
                "E": f"{metrics['energy']:.4f}",
                "EMA": f"{metrics['ema_energy']:.4f}",
                "unique": f"{metrics['unique_ratio']:.2f}",
                "basis": basis_size,
            })

            # Callback
            if callback is not None:
                callback(epoch, metrics)

            # Check convergence
            if epoch >= self.config.min_epochs and metrics["unique_ratio"] < self.config.convergence_threshold:
                print(f"\nConverged at epoch {epoch}: unique ratio = {metrics['unique_ratio']:.2f}, "
                      f"EMA energy = {metrics['ema_energy']:.4f}")
                break

        # Print timing summary
        if len(self.history["epoch_times"]) > 0:
            avg_time = np.mean(self.history["epoch_times"])
            total_time = np.sum(self.history["epoch_times"])
            print(f"\nTraining complete: {len(self.history['epoch_times'])} epochs, "
                  f"avg {avg_time:.3f}s/epoch, total {total_time:.1f}s")

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

        if self.accumulated_basis is not None and len(self.accumulated_basis) > 0:
            print(f"Using accumulated basis from training: {len(self.accumulated_basis)} states")
            return self.accumulated_basis

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
        basis = basis.to(self.device)

        # Precompute Hamiltonian matrix
        if hasattr(self.hamiltonian, 'matrix_elements_fast'):
            H_matrix = self.hamiltonian.matrix_elements_fast(basis)
        else:
            H_matrix = self.hamiltonian.matrix_elements(basis.cpu(), basis.cpu())
            H_matrix = H_matrix.to(self.device)

        pbar = tqdm(range(num_iters), desc="Inference NQS Training")

        for iteration in pbar:
            psi = self.nqs.psi(basis)
            psi_norm = psi / torch.sqrt(torch.sum(torch.abs(psi)**2) + 1e-10)

            if psi.is_complex():
                energy = torch.real(torch.conj(psi_norm) @ H_matrix @ psi_norm)
            else:
                energy = psi_norm @ H_matrix @ psi_norm

            self.optimizer.zero_grad()
            energy.backward()
            self.optimizer.step()
            self.scheduler.step(energy)

            history["energies"].append(energy.item())
            pbar.set_postfix({"E": f"{energy.item():.6f}"})

        return history
