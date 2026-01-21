"""
Flow-Guided Krylov Quantum Diagonalization Pipeline.

This module provides the end-to-end pipeline that combines:
1. Normalizing Flow-Assisted Neural Quantum States (NF-NQS) for basis discovery
2. Sample-Based Krylov Quantum Diagonalization (SKQD) for energy refinement

The pipeline achieves systematic improvement toward ground-truth energies
by combining learned importance sampling with Krylov subspace projection.
"""

import torch
import numpy as np
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import matplotlib.pyplot as plt

# Support both package imports and direct script execution
try:
    from .nqs.dense import DenseNQS
    from .flows.discrete_flow import DiscreteFlowSampler
    from .flows.training import FlowNQSTrainer, TrainingConfig, InferenceNQSTrainer
    from .hamiltonians.base import Hamiltonian
    from .hamiltonians.spin import TransverseFieldIsing, HeisenbergHamiltonian
    from .hamiltonians.molecular import MolecularHamiltonian
    from .krylov.skqd import (
        SampleBasedKrylovDiagonalization,
        FlowGuidedSKQD,
        SKQDConfig,
    )
except ImportError:
    from nqs.dense import DenseNQS
    from flows.discrete_flow import DiscreteFlowSampler
    from flows.training import FlowNQSTrainer, TrainingConfig, InferenceNQSTrainer
    from hamiltonians.base import Hamiltonian
    from hamiltonians.spin import TransverseFieldIsing, HeisenbergHamiltonian
    from hamiltonians.molecular import MolecularHamiltonian
    from krylov.skqd import (
        SampleBasedKrylovDiagonalization,
        FlowGuidedSKQD,
        SKQDConfig,
    )


@dataclass
class PipelineConfig:
    """Configuration for the full NF-NQS-SKQD pipeline."""

    # NF-NQS training parameters
    nf_coupling_layers: int = 4
    nf_hidden_dims: list = field(default_factory=lambda: [512, 512])
    nqs_hidden_dims: list = field(default_factory=lambda: [512, 512, 512, 512])

    # Training parameters (balanced for accuracy and speed)
    samples_per_batch: int = 3000  # Good coverage of basis states
    num_batches: int = 1  # Single batch per epoch
    nf_lr: float = 5e-4  # Slower flow LR for stability
    nqs_lr: float = 1e-3  # Faster NQS LR for convergence
    max_epochs: int = 500
    min_epochs: int = 150  # Sufficient training before checking convergence
    convergence_threshold: float = 0.20  # Train until flow concentrates well (<20% unique)
    use_local_energy: bool = False  # Use accurate subspace energy (local energy is buggy)

    # Stability parameters (prevent energy drifting)
    use_accumulated_energy: bool = True  # Compute energy on accumulated basis
    ema_decay: float = 0.95  # EMA decay for stable energy tracking
    entropy_weight: float = 0.01  # Entropy regularization to prevent collapse

    # Basis management (CRITICAL for large systems like LiH, H2O)
    max_accumulated_basis: int = 2048  # Hard cap on accumulated basis size
    accumulated_energy_interval: int = 1  # Compute accumulated energy every N epochs
    prune_basis_threshold: float = 1e-6  # Prune low-importance states

    # Inference parameters
    inference_samples: int = 5000
    inference_iterations: int = 2000
    inference_lr: float = 1e-3
    skip_inference: bool = True  # Skip inference phase (use co-trained NQS directly)

    # SKQD parameters
    max_krylov_dim: int = 12
    time_step: float = 0.1
    num_trotter_steps: int = 8
    shots_per_krylov: int = 100000

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_gpu: bool = True


class FlowGuidedKrylovPipeline:
    """
    End-to-end pipeline for ground state energy computation.

    This pipeline combines NF-NQS for discovering the ground state support
    with SKQD for systematic energy refinement via Krylov subspace projection.

    The workflow is:
    1. **NF-NQS Co-Training**: Learn the ground state distribution
       - Normalizing Flow learns to sample high-probability basis states
       - NQS learns amplitude/phase structure

    2. **Basis Extraction**: Freeze NF and sample a high-quality basis set

    3. **Inference NQS Training**: Refine amplitudes on the fixed basis

    4. **SKQD Refinement**: Use Krylov time evolution to systematically
       expand and improve the basis, achieving convergence to ground truth

    Example usage:
    ```python
    # Create Hamiltonian
    H = TransverseFieldIsing(num_spins=20, V=1.0, h=1.0, L=10)

    # Create and run pipeline
    pipeline = FlowGuidedKrylovPipeline(H)
    results = pipeline.run()

    # Access results
    print(f"NF-NQS energy: {results['nf_nqs_energy']:.6f}")
    print(f"SKQD energy: {results['skqd_energy']:.6f}")
    print(f"Combined energy: {results['combined_energy']:.6f}")
    ```

    Args:
        hamiltonian: System Hamiltonian
        config: Pipeline configuration
        exact_energy: Known exact energy for validation (optional)
    """

    def __init__(
        self,
        hamiltonian: Hamiltonian,
        config: Optional[PipelineConfig] = None,
        exact_energy: Optional[float] = None,
    ):
        self.hamiltonian = hamiltonian
        self.config = config or PipelineConfig()
        self.exact_energy = exact_energy
        self.num_sites = hamiltonian.num_sites

        # Initialize components
        self._init_components()

        # Results storage
        self.results: Dict[str, Any] = {}

    def _init_components(self):
        """Initialize NF, NQS, and SKQD components."""
        device = self.config.device

        # Normalizing Flow
        self.flow = DiscreteFlowSampler(
            num_sites=self.num_sites,
            num_coupling_layers=self.config.nf_coupling_layers,
            hidden_dims=self.config.nf_hidden_dims,
        ).to(device)

        # Neural Quantum State
        self.nqs = DenseNQS(
            num_sites=self.num_sites,
            hidden_dims=self.config.nqs_hidden_dims,
        ).to(device)

        # Training configuration (including stability parameters)
        self.train_config = TrainingConfig(
            samples_per_batch=self.config.samples_per_batch,
            num_batches=self.config.num_batches,
            flow_lr=self.config.nf_lr,
            nqs_lr=self.config.nqs_lr,
            num_epochs=self.config.max_epochs,
            min_epochs=self.config.min_epochs,
            convergence_threshold=self.config.convergence_threshold,
            use_local_energy=self.config.use_local_energy,
            # Stability parameters
            use_accumulated_energy=self.config.use_accumulated_energy,
            ema_decay=self.config.ema_decay,
            entropy_weight=self.config.entropy_weight,
            # Basis management (for large systems)
            max_accumulated_basis=self.config.max_accumulated_basis,
            accumulated_energy_interval=self.config.accumulated_energy_interval,
            prune_basis_threshold=self.config.prune_basis_threshold,
        )

        # SKQD configuration
        self.skqd_config = SKQDConfig(
            max_krylov_dim=self.config.max_krylov_dim,
            time_step=self.config.time_step,
            num_trotter_steps=self.config.num_trotter_steps,
            shots_per_krylov=self.config.shots_per_krylov,
            use_gpu=self.config.use_gpu,
        )

    def train_nf_nqs(
        self,
        progress: bool = True,
    ) -> Dict[str, list]:
        """
        Stage 1: Train NF-NQS to discover ground state support.

        Returns:
            Training history with energies, losses, and convergence metrics
        """
        print("=" * 60)
        print("Stage 1: NF-NQS Co-Training")
        print("=" * 60)

        self.trainer = FlowNQSTrainer(
            flow=self.flow,
            nqs=self.nqs,
            hamiltonian=self.hamiltonian,
            config=self.train_config,
            device=self.config.device,
        )

        history = self.trainer.train()

        self.results["nf_nqs_history"] = history
        self.results["nf_nqs_energy"] = history["energies"][-1]

        return history

    def extract_basis(
        self,
        n_samples: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Stage 2: Extract basis from trained NF.

        Prefers accumulated basis from training (preserves important states),
        otherwise samples fresh from the frozen flow.

        Returns:
            Unique basis configurations, shape (n_basis, num_sites)
        """
        print("=" * 60)
        print("Stage 2: Basis Extraction")
        print("=" * 60)

        if n_samples is None:
            n_samples = self.config.inference_samples

        self.flow.eval()

        # Prefer accumulated basis from training (preserves important states)
        if hasattr(self, 'trainer') and self.trainer.accumulated_basis is not None:
            unique_basis = self.trainer.accumulated_basis
            print(f"Using accumulated basis from training: {len(unique_basis)} states")
        else:
            # Fallback: sample fresh from flow
            with torch.no_grad():
                _, unique_basis = self.flow.sample(n_samples)
            print(f"Extracted {len(unique_basis)} unique basis states (fresh sample)")

        self.nf_basis = unique_basis
        self.results["nf_basis_size"] = len(unique_basis)

        return unique_basis

    def refine_nqs(
        self,
        basis: Optional[torch.Tensor] = None,
        progress: bool = True,
    ) -> Dict[str, list]:
        """
        Stage 3: Train fresh NQS on fixed basis.

        After the flow has converged, we train a new NQS to accurately
        learn the amplitudes within the discovered subspace.

        Returns:
            Inference training history
        """
        print("=" * 60)
        print("Stage 3: Inference NQS Training")
        print("=" * 60)

        if basis is None:
            basis = self.nf_basis

        # Create fresh NQS for inference
        inference_nqs = DenseNQS(
            num_sites=self.num_sites,
            hidden_dims=self.config.nqs_hidden_dims,
        ).to(self.config.device)

        trainer = InferenceNQSTrainer(
            flow=self.flow,
            nqs=inference_nqs,
            hamiltonian=self.hamiltonian,
            lr=self.config.inference_lr,
            device=self.config.device,
        )

        history = trainer.train(
            num_iters=self.config.inference_iterations,
            n_samples=self.config.inference_samples,
        )

        self.inference_nqs = inference_nqs
        self.results["inference_history"] = history
        self.results["inference_energy"] = history["energies"][-1]

        return history

    def _compute_cotrained_energy(self):
        """
        Compute energy using co-trained NQS on accumulated basis.

        Used when skip_inference=True to get energy estimate without
        training a fresh NQS (which often performs worse).
        """
        import torch

        basis = self.nf_basis.to(self.config.device)

        # Compute Hamiltonian matrix in basis
        H_matrix = self.hamiltonian.matrix_elements(basis.cpu(), basis.cpu())
        H_matrix = H_matrix.to(self.config.device)

        # Get wavefunction amplitudes from co-trained NQS
        with torch.no_grad():
            psi = self.nqs.psi(basis)
            psi_norm = psi / torch.sqrt(torch.sum(torch.abs(psi)**2))

            if psi.is_complex():
                energy = torch.real(torch.conj(psi_norm) @ H_matrix @ psi_norm)
            else:
                energy = psi_norm @ H_matrix @ psi_norm

        self.results["inference_energy"] = energy.item()
        print(f"Co-trained NQS energy on basis: {energy.item():.6f}")

    def run_skqd(
        self,
        use_nf_basis: bool = True,
        progress: bool = True,
    ) -> Dict[str, list]:
        """
        Stage 4: Run SKQD for systematic energy refinement.

        Uses Krylov time evolution to expand the basis and improve
        the energy estimate through subspace projection.

        Args:
            use_nf_basis: Whether to include NF-discovered basis
            progress: Show progress bars

        Returns:
            SKQD results with energies vs Krylov dimension
        """
        print("=" * 60)
        print("Stage 4: Sample-Based Krylov Quantum Diagonalization")
        print("=" * 60)

        if use_nf_basis and hasattr(self, "nf_basis"):
            # Use flow-guided SKQD
            skqd = FlowGuidedSKQD(
                hamiltonian=self.hamiltonian,
                nf_basis=self.nf_basis,
                config=self.skqd_config,
            )
            results = skqd.run_with_nf(progress=progress)

            self.results["skqd_results"] = results
            self.results["skqd_energy"] = results["energies_combined"][-1]
            self.results["combined_energy"] = results["energies_combined"][-1]
        else:
            # Standard SKQD
            skqd = SampleBasedKrylovDiagonalization(
                hamiltonian=self.hamiltonian,
                config=self.skqd_config,
            )
            results = skqd.run(progress=progress)

            self.results["skqd_results"] = results
            self.results["skqd_energy"] = results["energies"][-1]

        return results

    def run(
        self,
        skip_training: bool = False,
        progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline.

        Args:
            skip_training: If True, skip NF-NQS training (use pre-trained)
            progress: Show progress bars

        Returns:
            Complete results dictionary
        """
        print("\n" + "=" * 60)
        print("Flow-Guided Krylov Quantum Diagonalization Pipeline")
        print("=" * 60)
        print(f"System: {self.num_sites} sites")
        print(f"Hamiltonian: {type(self.hamiltonian).__name__}")
        if self.exact_energy is not None:
            print(f"Exact ground state energy: {self.exact_energy:.6f}")
        print("=" * 60 + "\n")

        if not skip_training:
            # Stage 1: NF-NQS training
            self.train_nf_nqs(progress=progress)

        # Stage 2: Basis extraction
        self.extract_basis()

        # Stage 3: Inference NQS training (optional)
        if not self.config.skip_inference:
            self.refine_nqs(progress=progress)
        else:
            print("=" * 60)
            print("Stage 3: Skipped (using co-trained NQS)")
            print("=" * 60)
            # Use energy from co-trained NQS on accumulated basis
            self._compute_cotrained_energy()

        # Stage 4: SKQD refinement
        self.run_skqd(use_nf_basis=True, progress=progress)

        # Summary
        self._print_summary()

        return self.results

    def _print_summary(self):
        """Print results summary."""
        print("\n" + "=" * 60)
        print("Results Summary")
        print("=" * 60)

        if "nf_nqs_energy" in self.results:
            print(f"NF-NQS Energy:      {self.results['nf_nqs_energy']:.6f}")

        if "inference_energy" in self.results:
            print(f"Inference Energy:   {self.results['inference_energy']:.6f}")

        if "skqd_energy" in self.results:
            print(f"SKQD Energy:        {self.results['skqd_energy']:.6f}")

        if "combined_energy" in self.results:
            print(f"Combined Energy:    {self.results['combined_energy']:.6f}")

        if self.exact_energy is not None:
            best_energy = self.results.get(
                "combined_energy",
                self.results.get("skqd_energy", self.results.get("nf_nqs_energy")),
            )
            error = abs(best_energy - self.exact_energy)
            error_pct = 100 * error / abs(self.exact_energy)
            print(f"\nError vs exact:     {error:.6f} ({error_pct:.4f}%)")

        print("=" * 60)

    def plot_convergence(
        self,
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        """
        Plot convergence of energies across pipeline stages.

        Args:
            save_path: Path to save figure (optional)
            show: Whether to display the figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # NF-NQS training
        if "nf_nqs_history" in self.results:
            ax = axes[0]
            energies = self.results["nf_nqs_history"]["energies"]
            ax.plot(energies, "b-", label="NF-NQS")
            if self.exact_energy is not None:
                ax.axhline(
                    self.exact_energy, color="r", linestyle="--", label="Exact"
                )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Energy")
            ax.set_title("Stage 1: NF-NQS Training")
            ax.legend()

        # Inference training
        if "inference_history" in self.results:
            ax = axes[1]
            energies = self.results["inference_history"]["energies"]
            ax.plot(energies, "g-", label="Inference NQS")
            if self.exact_energy is not None:
                ax.axhline(
                    self.exact_energy, color="r", linestyle="--", label="Exact"
                )
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Energy")
            ax.set_title("Stage 3: Inference Training")
            ax.legend()

        # SKQD convergence
        if "skqd_results" in self.results:
            ax = axes[2]
            skqd = self.results["skqd_results"]

            if "energies_combined" in skqd:
                ax.plot(
                    skqd["krylov_dims"],
                    skqd["energies_krylov"],
                    "b-o",
                    label="Krylov only",
                )
                ax.plot(
                    skqd["krylov_dims"],
                    skqd["energies_combined"],
                    "g-s",
                    label="NF + Krylov",
                )
                ax.axhline(
                    skqd["energy_nf_only"],
                    color="orange",
                    linestyle=":",
                    label="NF only",
                )
            else:
                ax.plot(
                    skqd["krylov_dims"], skqd["energies"], "b-o", label="SKQD"
                )

            if self.exact_energy is not None:
                ax.axhline(
                    self.exact_energy, color="r", linestyle="--", label="Exact"
                )

            ax.set_xlabel("Krylov Dimension")
            ax.set_ylabel("Energy")
            ax.set_title("Stage 4: SKQD Refinement")
            ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

    def save_checkpoint(self, path: str):
        """Save pipeline state to checkpoint."""
        checkpoint = {
            "flow_state_dict": self.flow.state_dict(),
            "nqs_state_dict": self.nqs.state_dict(),
            "config": self.config,
            "results": self.results,
        }

        if hasattr(self, "inference_nqs"):
            checkpoint["inference_nqs_state_dict"] = self.inference_nqs.state_dict()

        if hasattr(self, "nf_basis"):
            checkpoint["nf_basis"] = self.nf_basis.cpu()

        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load pipeline state from checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)

        self.flow.load_state_dict(checkpoint["flow_state_dict"])
        self.nqs.load_state_dict(checkpoint["nqs_state_dict"])
        self.results = checkpoint["results"]

        if "nf_basis" in checkpoint:
            self.nf_basis = checkpoint["nf_basis"].to(self.config.device)

        if "inference_nqs_state_dict" in checkpoint:
            self.inference_nqs = DenseNQS(
                num_sites=self.num_sites,
                hidden_dims=self.config.nqs_hidden_dims,
            ).to(self.config.device)
            self.inference_nqs.load_state_dict(
                checkpoint["inference_nqs_state_dict"]
            )

        print(f"Checkpoint loaded from {path}")
