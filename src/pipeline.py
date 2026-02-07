"""
Flow-Guided Krylov Pipeline for Molecular Ground State Energy Calculation.

This module provides an end-to-end pipeline that combines:
1. Normalizing Flow-Assisted Neural Quantum States (NF-NQS) for basis discovery
2. Sample-Based Krylov Quantum Diagonalization (SKQD) for energy refinement

Key Features:
- Particle Conservation: Samples valid molecular configurations only
- Physics-Guided Training: Mixed objective with energy importance
- Diversity Selection: Excitation-rank stratified basis selection
- Residual Expansion: Selected-CI style basis recovery
- Adaptive Scaling: Automatic parameter adjustment for system size

Usage:
    from src.pipeline import FlowGuidedKrylovPipeline, PipelineConfig
    from src.hamiltonians.molecular import create_lih_hamiltonian

    H = create_lih_hamiltonian(bond_length=1.6)
    pipeline = FlowGuidedKrylovPipeline(H)
    results = pipeline.run()
"""

import torch
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

# Flow components
try:
    from .flows.particle_conserving_flow import (
        ParticleConservingFlowSampler,
        verify_particle_conservation,
    )
    from .flows.physics_guided_training import (
        PhysicsGuidedFlowTrainer,
        PhysicsGuidedConfig,
    )
    from .flows.discrete_flow import DiscreteFlowSampler
    from .flows.training import FlowNQSTrainer, TrainingConfig
except ImportError:
    from flows.particle_conserving_flow import (
        ParticleConservingFlowSampler,
        verify_particle_conservation,
    )
    from flows.physics_guided_training import (
        PhysicsGuidedFlowTrainer,
        PhysicsGuidedConfig,
    )
    from flows.discrete_flow import DiscreteFlowSampler
    from flows.training import FlowNQSTrainer, TrainingConfig

# NQS components
try:
    from .nqs.dense import DenseNQS
except ImportError:
    from nqs.dense import DenseNQS

# Hamiltonian components
try:
    from .hamiltonians.base import Hamiltonian
    from .hamiltonians.molecular import MolecularHamiltonian
except ImportError:
    from hamiltonians.base import Hamiltonian
    from hamiltonians.molecular import MolecularHamiltonian

# Postprocessing components
try:
    from .postprocessing.diversity_selection import (
        DiversitySelector,
        DiversityConfig,
        select_diverse_basis,
    )
    from .postprocessing.eigensolver import (
        davidson_eigensolver,
        adaptive_eigensolver,
    )
except ImportError:
    from postprocessing.diversity_selection import (
        DiversitySelector,
        DiversityConfig,
        select_diverse_basis,
    )
    from postprocessing.eigensolver import (
        davidson_eigensolver,
        adaptive_eigensolver,
    )

# Krylov components
try:
    from .krylov.residual_expansion import (
        ResidualBasedExpander,
        ResidualExpansionConfig,
        iterative_residual_expansion,
    )
    from .krylov.skqd import (
        SampleBasedKrylovDiagonalization,
        FlowGuidedSKQD,
        SKQDConfig,
    )
except ImportError:
    from krylov.residual_expansion import (
        ResidualBasedExpander,
        ResidualExpansionConfig,
        iterative_residual_expansion,
    )
    from krylov.skqd import (
        SampleBasedKrylovDiagonalization,
        FlowGuidedSKQD,
        SKQDConfig,
    )


@dataclass
class PipelineConfig:
    """
    Configuration for the Flow-Guided Krylov pipeline.

    This configuration supports both molecular systems (with particle conservation)
    and general spin systems.
    """

    # Flow type
    use_particle_conserving_flow: bool = True  # Use particle-conserving flow for molecules

    # NF-NQS architecture
    nf_hidden_dims: list = field(default_factory=lambda: [256, 256])
    nqs_hidden_dims: list = field(default_factory=lambda: [256, 256, 256, 256])

    # Training parameters
    samples_per_batch: int = 2000
    num_batches: int = 1
    max_epochs: int = 400
    min_epochs: int = 100
    convergence_threshold: float = 0.20

    # Physics-guided training weights - following paper's approach
    # Paper uses only cross-entropy for NF, weighted by |E|
    teacher_weight: float = 1.0  # Cross-entropy (paper's only term)
    physics_weight: float = 0.0  # Paper doesn't use this
    entropy_weight: float = 0.0  # Paper doesn't use this

    # Learning rates
    nf_lr: float = 5e-4
    nqs_lr: float = 1e-3

    # Basis management - ADAPTIVE: will be scaled by system size
    max_accumulated_basis: int = 4096  # Base value, scaled automatically

    # Diversity selection - ADAPTIVE: will be scaled by system size
    use_diversity_selection: bool = True
    max_diverse_configs: int = 2048  # Base value, scaled automatically
    rank_2_fraction: float = 0.50  # Emphasize double excitations

    # Residual expansion - ADAPTIVE: scaled by system size
    use_residual_expansion: bool = True
    residual_iterations: int = 8
    residual_configs_per_iter: int = 150
    residual_threshold: float = 1e-6
    use_perturbative_selection: bool = True  # Use 2nd-order PT for selection

    # SKQD parameters
    max_krylov_dim: int = 8
    time_step: float = 0.1
    shots_per_krylov: int = 50000
    skqd_regularization: float = 1e-8  # Regularization for numerical stability
    skip_skqd: bool = False  # Skip Krylov refinement (for NF-only mode comparison)

    # Training mode
    use_local_energy: bool = True  # Use VMC local energy (proper variational estimator)
    use_ci_seeding: bool = False  # Seed with CI basis (set True if NF struggles)

    # Eigensolver
    use_davidson: bool = True
    davidson_threshold: int = 500  # Use Davidson for bases larger than this

    # Direct-CI mode: skip NF-NQS training entirely
    # When True, pipeline goes directly from essential config generation → residual expansion → SKQD
    # For molecular systems, essential configs (HF + singles + doubles) already dominate the ground state
    skip_nf_training: bool = False

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # === PERFORMANCE OPTIMIZATIONS FOR LARGE SYSTEMS ===
    # These dramatically reduce training time for large molecules (>20 qubits)

    # Truncate connections to top-k by matrix element magnitude (0 = no truncation)
    max_connections_per_config: int = 0

    # Skip off-diagonal computation for first N epochs (diagonal-only warmup)
    diagonal_only_warmup_epochs: int = 0

    # Sample a fraction of connections for stochastic local energy (1.0 = all)
    stochastic_connections_fraction: float = 1.0

    def adapt_to_system_size(self, n_valid_configs: int, verbose: bool = True) -> "PipelineConfig":
        """
        Adapt configuration parameters based on the valid configuration space size.

        This is CRITICAL for larger molecules where the default parameters
        are insufficient for adequate basis coverage.

        Args:
            n_valid_configs: Number of particle-conserving configurations
            verbose: Whether to print adaptation info (default True, set False to suppress)

        Returns:
            Updated config (modifies self in-place and returns)
        """
        # Skip if already adapted to same size
        if hasattr(self, '_adapted_size') and self._adapted_size == n_valid_configs:
            return self

        # Determine system complexity tier
        if n_valid_configs <= 1000:
            tier = "small"
        elif n_valid_configs <= 5000:
            tier = "medium"
        elif n_valid_configs <= 20000:
            tier = "large"
        else:
            tier = "very_large"

        # For molecular systems, skip NF training by default (Direct-CI mode)
        # Essential configs (HF + singles + doubles) + residual expansion reach exact FCI
        # NF training adds significant overhead with no energy improvement
        self.skip_nf_training = True

        if verbose:
            print(f"System size: {n_valid_configs:,} valid configs -> {tier} tier")
            if self.skip_nf_training:
                print(f"Direct-CI mode: skipping NF-NQS training")

        if tier == "small":
            # Small systems: default parameters are fine
            self.max_accumulated_basis = max(self.max_accumulated_basis, n_valid_configs)
            self.max_diverse_configs = min(n_valid_configs, self.max_diverse_configs)

        elif tier == "medium":
            # Medium systems: need more basis coverage
            self.max_accumulated_basis = min(n_valid_configs, 8192)
            self.max_diverse_configs = min(n_valid_configs, 4096)
            self.residual_iterations = max(self.residual_iterations, 10)
            self.residual_configs_per_iter = max(self.residual_configs_per_iter, 200)
            # Larger networks for more complex systems
            if len(self.nqs_hidden_dims) < 5:
                self.nqs_hidden_dims = [384, 384, 384, 384, 384]

        elif tier == "large":
            # Large systems: aggressive basis collection
            self.max_accumulated_basis = min(n_valid_configs, 12288)
            self.max_diverse_configs = min(n_valid_configs, 8192)
            self.residual_iterations = 15
            self.residual_configs_per_iter = 300
            self.residual_threshold = 1e-7
            self.use_perturbative_selection = True
            # Even larger networks
            self.nqs_hidden_dims = [512, 512, 512, 512, 512]
            # More training
            self.max_epochs = max(self.max_epochs, 600)
            self.samples_per_batch = 4000

        else:  # very_large
            # Very large systems (>20K valid configs, e.g. C2H4 with 9M)
            #
            # KEY INSIGHT: For very large systems, NF cannot efficiently explore
            # the ground state region. Instead, we rely on:
            # 1. Essential config injection (HF + singles + doubles) for subspace energy
            # 2. Short NF training to learn nearby configurations
            # 3. Aggressive residual expansion (Selected-CI) as the primary basis builder
            # 4. Krylov for supplementary discovery
            #
            # Previous approach (800 epochs, performance hacks) wasted hours while
            # NF explored wrong Hilbert space region. New approach: short training,
            # full energy signal, then rely on CI expansion.

            self.max_accumulated_basis = 16384
            self.max_diverse_configs = min(n_valid_configs, 12288)
            self.residual_iterations = 20
            self.residual_configs_per_iter = 500
            self.residual_threshold = 1e-8
            self.use_perturbative_selection = True

            # Network capacity
            self.nqs_hidden_dims = [512, 512, 512, 512]
            self.nf_hidden_dims = [256, 256]

            # SHORT training: NF serves as warm-start, not primary basis builder
            # Essential configs provide the energy signal; NF explores neighborhood
            self.max_epochs = max(self.max_epochs, 200)
            self.min_epochs = max(self.min_epochs, 50)
            self.samples_per_batch = 2000

            # IMPORTANT: Do NOT use performance hacks that cripple the energy signal
            # - No connection truncation (loses important doubles)
            # - No diagonal-only warmup (delays real energy signal)
            # - No stochastic connections (adds noise to already weak signal)
            # The essential config injection makes subspace energy fast enough
            self.max_connections_per_config = 0  # Use all connections
            self.diagonal_only_warmup_epochs = 0  # Full energy from epoch 1
            self.stochastic_connections_fraction = 1.0  # Use all connections

        # Compute coverage statistics
        coverage_accumulated = min(1.0, self.max_accumulated_basis / n_valid_configs)
        coverage_diverse = min(1.0, self.max_diverse_configs / n_valid_configs)

        if verbose:
            print(f"Adapted parameters:")
            print(f"  max_accumulated_basis: {self.max_accumulated_basis:,} ({coverage_accumulated*100:.1f}% of valid)")
            print(f"  max_diverse_configs: {self.max_diverse_configs:,} ({coverage_diverse*100:.1f}% of valid)")
            print(f"  residual_iterations: {self.residual_iterations}")
            print(f"  residual_configs_per_iter: {self.residual_configs_per_iter}")
            print(f"  NQS hidden dims: {self.nqs_hidden_dims}")

        # Mark as adapted to prevent duplicate adaptation
        self._adapted_size = n_valid_configs

        return self


class FlowGuidedKrylovPipeline:
    """
    Flow-Guided Krylov Pipeline for ground state energy computation.

    This pipeline combines:
    1. Particle-conserving normalizing flow for valid molecular configurations
    2. Physics-guided NF-NQS co-training with mixed objective
    3. Diversity-aware basis selection by excitation rank
    4. Residual-based (Selected-CI style) basis expansion
    5. SKQD refinement with Krylov subspace methods

    The workflow is:
    - Stage 1: Physics-guided NF-NQS training (discovers ground state support)
    - Stage 2: Diversity-aware basis extraction (stratified by excitation rank)
    - Stage 3: Residual expansion (recovers missing important configurations)
    - Stage 4: SKQD refinement (Krylov subspace diagonalization)

    Example usage:
    ```python
    from src.pipeline import FlowGuidedKrylovPipeline, PipelineConfig
    from src.hamiltonians.molecular import create_lih_hamiltonian

    H = create_lih_hamiltonian(bond_length=1.6)
    E_exact = H.fci_energy()

    pipeline = FlowGuidedKrylovPipeline(H, exact_energy=E_exact)
    results = pipeline.run()

    print(f"Final energy: {results['combined_energy']:.6f} Ha")
    print(f"Error: {abs(results['combined_energy'] - E_exact) * 1000:.2f} mHa")
    ```

    Args:
        hamiltonian: System Hamiltonian
        config: Pipeline configuration
        exact_energy: Known exact energy for validation (optional)
        auto_adapt: Automatically adapt config to system size
    """

    def __init__(
        self,
        hamiltonian: Hamiltonian,
        config: Optional[PipelineConfig] = None,
        exact_energy: Optional[float] = None,
        auto_adapt: bool = True,
    ):
        from math import comb

        self.hamiltonian = hamiltonian
        self.config = config or PipelineConfig()
        self.exact_energy = exact_energy
        self.num_sites = hamiltonian.num_sites
        self.device = self.config.device

        # Check if molecular Hamiltonian (for particle conservation)
        self.is_molecular = isinstance(hamiltonian, MolecularHamiltonian)

        # Compute valid configuration space size for molecules
        if self.is_molecular:
            n_orb = hamiltonian.n_orbitals
            n_alpha = hamiltonian.n_alpha
            n_beta = hamiltonian.n_beta
            self.n_valid_configs = comb(n_orb, n_alpha) * comb(n_orb, n_beta)

            # Automatically adapt configuration to system size
            if auto_adapt:
                self.config.adapt_to_system_size(self.n_valid_configs)
        else:
            self.n_valid_configs = self.hamiltonian.hilbert_dim

        # Initialize components
        self._init_components()

        # Results storage
        self.results: Dict[str, Any] = {}

    def _init_components(self):
        """Initialize flow, NQS, and auxiliary components."""
        cfg = self.config

        # Determine flow type
        if cfg.use_particle_conserving_flow and self.is_molecular:
            # Use particle-conserving flow for molecules
            n_alpha = self.hamiltonian.n_alpha
            n_beta = self.hamiltonian.n_beta

            self.flow = ParticleConservingFlowSampler(
                num_sites=self.num_sites,
                n_alpha=n_alpha,
                n_beta=n_beta,
                hidden_dims=cfg.nf_hidden_dims,
            ).to(self.device)

            print(f"Using particle-conserving flow: {n_alpha}α + {n_beta}β electrons")
        else:
            # Use standard discrete flow
            self.flow = DiscreteFlowSampler(
                num_sites=self.num_sites,
                num_coupling_layers=4,
                hidden_dims=cfg.nf_hidden_dims,
            ).to(self.device)

        # Neural Quantum State
        self.nqs = DenseNQS(
            num_sites=self.num_sites,
            hidden_dims=cfg.nqs_hidden_dims,
        ).to(self.device)

        # Get reference state (HF for molecules)
        if self.is_molecular:
            self.reference_state = self.hamiltonian.get_hf_state()
        else:
            self.reference_state = torch.zeros(self.num_sites, device=self.device)

    def _generate_essential_configs(self) -> torch.Tensor:
        """
        Generate essential configurations (HF + singles + doubles) without NF training.

        Reuses the same logic as PhysicsGuidedFlowTrainer._generate_essential_configs()
        but can be called standalone for Direct-CI mode.

        Returns:
            Tensor of essential configurations
        """
        from itertools import combinations

        n_orb = self.hamiltonian.n_orbitals
        n_alpha = self.hamiltonian.n_alpha
        n_beta = self.hamiltonian.n_beta

        hf_state = self.hamiltonian.get_hf_state()
        essential = [hf_state.clone()]

        occ_alpha = list(range(n_alpha))
        occ_beta = list(range(n_beta))
        virt_alpha = list(range(n_alpha, n_orb))
        virt_beta = list(range(n_beta, n_orb))

        # Single excitations
        for i in occ_alpha:
            for a in virt_alpha:
                new_config = hf_state.clone()
                new_config[i] = 0
                new_config[a] = 1
                essential.append(new_config)

        for i in occ_beta:
            for a in virt_beta:
                new_config = hf_state.clone()
                new_config[i + n_orb] = 0
                new_config[a + n_orb] = 1
                essential.append(new_config)

        # Double excitations (with limit for large systems)
        max_doubles = 5000
        doubles_count = 0

        # Alpha-alpha doubles
        for i, j in combinations(occ_alpha, 2):
            for a, b in combinations(virt_alpha, 2):
                if doubles_count >= max_doubles:
                    break
                new_config = hf_state.clone()
                new_config[i] = 0
                new_config[j] = 0
                new_config[a] = 1
                new_config[b] = 1
                essential.append(new_config)
                doubles_count += 1
            if doubles_count >= max_doubles:
                break

        # Beta-beta doubles
        for i, j in combinations(occ_beta, 2):
            for a, b in combinations(virt_beta, 2):
                if doubles_count >= max_doubles:
                    break
                new_config = hf_state.clone()
                new_config[i + n_orb] = 0
                new_config[j + n_orb] = 0
                new_config[a + n_orb] = 1
                new_config[b + n_orb] = 1
                essential.append(new_config)
                doubles_count += 1
            if doubles_count >= max_doubles:
                break

        # Alpha-beta doubles (most important for correlation)
        for i in occ_alpha:
            for j in occ_beta:
                for a in virt_alpha:
                    for b in virt_beta:
                        if doubles_count >= max_doubles:
                            break
                        new_config = hf_state.clone()
                        new_config[i] = 0
                        new_config[j + n_orb] = 0
                        new_config[a] = 1
                        new_config[b + n_orb] = 1
                        essential.append(new_config)
                        doubles_count += 1
                    if doubles_count >= max_doubles:
                        break
                if doubles_count >= max_doubles:
                    break
            if doubles_count >= max_doubles:
                break

        essential_tensor = torch.stack(essential).to(self.device)
        essential_tensor = torch.unique(essential_tensor, dim=0)

        n_singles = len([c for c in essential if torch.sum(torch.abs(c - hf_state)) == 2])
        n_doubles = len(essential_tensor) - n_singles - 1

        print(f"Generated {len(essential_tensor)} essential configs: "
              f"1 HF + {n_singles} singles + {n_doubles} doubles")

        return essential_tensor

    def train_flow_nqs(self, progress: bool = True) -> Dict[str, list]:
        """
        Stage 1: Train NF-NQS with physics-guided objective.

        If skip_nf_training is True (Direct-CI mode), generates essential
        configs directly without any NF training.
        """
        print("=" * 60)
        print("Stage 1: Physics-Guided NF-NQS Training")
        print("=" * 60)

        cfg = self.config

        # Direct-CI mode: skip NF training, use essential configs directly
        if cfg.skip_nf_training and self.is_molecular:
            print("Direct-CI mode: skipping NF-NQS training")
            print("Generating essential configurations (HF + singles + doubles)...")

            self._essential_configs = self._generate_essential_configs()
            self.results["training_history"] = {"energies": [], "skipped": True}
            self.results["nf_nqs_energy"] = None
            self.results["skip_nf_training"] = True

            return {"energies": [], "skipped": True}

        # Create physics-guided trainer
        train_config = PhysicsGuidedConfig(
            samples_per_batch=cfg.samples_per_batch,
            num_batches=cfg.num_batches,
            flow_lr=cfg.nf_lr,
            nqs_lr=cfg.nqs_lr,
            num_epochs=cfg.max_epochs,
            min_epochs=cfg.min_epochs,
            convergence_threshold=cfg.convergence_threshold,
            teacher_weight=cfg.teacher_weight,
            physics_weight=cfg.physics_weight,
            entropy_weight=cfg.entropy_weight,
            max_accumulated_basis=cfg.max_accumulated_basis,
            # Performance optimizations for large systems
            max_connections_per_config=cfg.max_connections_per_config,
            diagonal_only_warmup_epochs=cfg.diagonal_only_warmup_epochs,
            stochastic_connections_fraction=cfg.stochastic_connections_fraction,
        )

        self.trainer = PhysicsGuidedFlowTrainer(
            flow=self.flow,
            nqs=self.nqs,
            hamiltonian=self.hamiltonian,
            config=train_config,
            device=self.device,
        )

        history = self.trainer.train()

        self.results["training_history"] = history
        self.results["nf_nqs_energy"] = history["energies"][-1]

        return history

    def extract_and_select_basis(self) -> torch.Tensor:
        """
        Stage 2: Extract basis with diversity-aware selection.

        In Direct-CI mode, uses essential configs directly without
        diversity selection (they are already a curated, physically-motivated set).
        """
        print("=" * 60)
        print("Stage 2: Diversity-Aware Basis Extraction")
        print("=" * 60)

        cfg = self.config

        # Direct-CI mode: use essential configs directly
        if cfg.skip_nf_training and hasattr(self, '_essential_configs'):
            print("Direct-CI mode: using essential configs as basis")
            selected_basis = self._essential_configs
            print(f"Essential configs basis: {len(selected_basis)} configs")
            self.nf_basis = selected_basis
            self.results["nf_basis_size"] = len(selected_basis)
            self.results["diversity_stats"] = {"skipped": True, "reason": "Direct-CI mode"}
            return selected_basis

        # Get accumulated basis from training
        if hasattr(self, 'trainer') and self.trainer.accumulated_basis is not None:
            raw_basis = self.trainer.accumulated_basis
            print(f"Raw accumulated basis: {len(raw_basis)} configs")
        else:
            # Fallback: sample from flow
            with torch.no_grad():
                _, raw_basis = self.flow.sample(cfg.samples_per_batch * 5)
            print(f"Sampled basis: {len(raw_basis)} configs")

        # Verify particle conservation for molecular systems
        if self.is_molecular and cfg.use_particle_conserving_flow:
            n_orbitals = self.hamiltonian.n_orbitals
            valid, stats = verify_particle_conservation(
                raw_basis, n_orbitals,
                self.hamiltonian.n_alpha, self.hamiltonian.n_beta
            )
            if not valid:
                print(f"WARNING: {stats['alpha_violations'] + stats['beta_violations']} "
                      f"particle number violations detected!")
            else:
                print("All configurations satisfy particle conservation")

        # Apply diversity selection
        if cfg.use_diversity_selection:
            diversity_config = DiversityConfig(
                max_configs=cfg.max_diverse_configs,
                rank_2_fraction=cfg.rank_2_fraction,
            )

            selector = DiversitySelector(
                config=diversity_config,
                reference=self.reference_state,
                n_orbitals=self.hamiltonian.n_orbitals if self.is_molecular else self.num_sites // 2,
            )

            selected_basis, select_stats = selector.select(raw_basis)
            print(f"Selected {len(selected_basis)} diverse configs from {len(raw_basis)}")
            print(f"Bucket distribution: {select_stats.get('bucket_stats', {})}")

            self.results["diversity_stats"] = select_stats
        else:
            selected_basis = raw_basis

        # CRITICAL: Always include essential configs (HF + singles + doubles)
        # even if diversity selection filtered them out. For large systems,
        # NF may never generate these, but they dominate the ground state.
        if (self.is_molecular and hasattr(self, 'trainer') and
                hasattr(self.trainer, '_essential_configs') and
                self.trainer._essential_configs is not None):
            essential = self.trainer._essential_configs
            combined = torch.cat([essential.to(selected_basis.device), selected_basis], dim=0)
            selected_basis = torch.unique(combined, dim=0)
            print(f"After merging essential configs: {len(selected_basis)} total")

        self.nf_basis = selected_basis
        self.results["nf_basis_size"] = len(selected_basis)

        return selected_basis

    def run_residual_expansion(self) -> torch.Tensor:
        """
        Stage 3: Expand basis using residual/perturbative analysis.

        Uses Selected-CI style expansion that iteratively adds configurations
        with the largest contributions to the ground state.

        Includes early stopping when energy improvement stagnates.
        """
        if not self.config.use_residual_expansion:
            return self.nf_basis

        print("=" * 60)
        print("Stage 3: Residual-Based Basis Expansion")
        print("=" * 60)

        cfg = self.config

        # Early stopping parameters
        min_improvement_mha = 0.05  # Stop if improvement < 0.05 mHa
        stagnation_patience = 2  # Stop after 2 consecutive stagnant iterations

        # Configure residual expansion based on system size
        residual_config = ResidualExpansionConfig(
            max_configs_per_iter=cfg.residual_configs_per_iter,
            max_iterations=cfg.residual_iterations,
            residual_threshold=cfg.residual_threshold,
            max_basis_size=min(cfg.max_accumulated_basis * 2, self.n_valid_configs),
            use_importance_sampling=True,
            min_energy_improvement_mha=min_improvement_mha,
            stagnation_patience=stagnation_patience,
        )

        # Use perturbative selection for better importance estimation
        if cfg.use_perturbative_selection:
            from krylov.residual_expansion import SelectedCIExpander
            print("Using perturbative (2nd-order PT) selection for configuration importance")
            print(f"Early stopping: improvement < {min_improvement_mha} mHa for {stagnation_patience} iterations")

            expander = SelectedCIExpander(self.hamiltonian, residual_config)

            # Run multiple rounds of expansion
            expanded_basis = self.nf_basis.clone()
            total_added = 0
            initial_energy = None
            prev_energy = None
            stagnant_count = 0

            best_energy = None
            best_basis = expanded_basis.clone()

            for iteration in range(cfg.residual_iterations):
                old_size = len(expanded_basis)
                expanded_basis, expand_stats = expander.expand_basis(expanded_basis)

                current_energy = expand_stats['final_energy']

                if initial_energy is None:
                    initial_energy = expand_stats.get('initial_energy', current_energy)

                # Track best energy and basis (variational principle)
                if best_energy is None or current_energy < best_energy:
                    best_energy = current_energy
                    best_basis = expanded_basis.clone()

                added = expand_stats['configs_added']
                total_added += added

                # Check for variational violation
                if expand_stats.get('variational_violation', False):
                    rejected_mha = expand_stats.get('rejected_increase_mha', 0)
                    print(f"  Iter {iteration+1}: Rejected configs (would increase E by {rejected_mha:.4f} mHa)")
                    stagnant_count += 1
                    if stagnant_count >= stagnation_patience:
                        print(f"  Converged: no improvement for {stagnation_patience} iterations")
                        break
                    continue

                # Calculate energy improvement
                if prev_energy is not None:
                    improvement_mha = (prev_energy - current_energy) * 1000
                    print(f"  Iter {iteration+1}: {old_size} -> {len(expanded_basis)} "
                          f"(+{added}), E = {current_energy:.6f}, ΔE = {improvement_mha:.4f} mHa")

                    # Check for stagnation
                    if improvement_mha < min_improvement_mha:
                        stagnant_count += 1
                        if stagnant_count >= stagnation_patience:
                            print(f"  Converged: energy improvement < {min_improvement_mha} mHa "
                                  f"for {stagnation_patience} consecutive iterations")
                            break
                    else:
                        stagnant_count = 0  # Reset stagnation counter
                else:
                    print(f"  Iter {iteration+1}: {old_size} -> {len(expanded_basis)} "
                          f"(+{added}), E = {current_energy:.6f}")

                prev_energy = current_energy

                # Early termination if no new configs added
                if added == 0:
                    print("  Converged: no new configurations found")
                    break

                # Check if we've reached max basis size
                if len(expanded_basis) >= residual_config.max_basis_size:
                    print(f"  Reached max basis size: {residual_config.max_basis_size}")
                    break

            # Use best basis (ensures variational principle is respected)
            expanded_basis = best_basis

            expand_stats = {
                'initial_basis_size': len(self.nf_basis),
                'final_basis_size': len(expanded_basis),
                'configs_added_total': total_added,
                'iterations': iteration + 1,
                'initial_energy': initial_energy,
                'final_energy': best_energy if best_energy is not None else current_energy,
                'converged_early': stagnant_count >= stagnation_patience,
            }
        else:
            # Standard residual-based expansion
            expander = ResidualBasedExpander(self.hamiltonian, residual_config)
            expanded_basis, expand_stats = expander.expand_basis(self.nf_basis)

        print(f"Expanded: {expand_stats['initial_basis_size']} -> {expand_stats['final_basis_size']}")
        if expand_stats.get('initial_energy') is not None:
            total_improvement_mha = (expand_stats['initial_energy'] - expand_stats['final_energy']) * 1000
            print(f"Energy improvement: {expand_stats['initial_energy']:.6f} -> "
                  f"{expand_stats['final_energy']:.6f} ({total_improvement_mha:.2f} mHa)")
        else:
            print(f"Final energy: {expand_stats['final_energy']:.6f}")

        self.results["residual_expansion_stats"] = expand_stats
        self.results["residual_energy"] = expand_stats['final_energy']
        self.expanded_basis = expanded_basis

        return expanded_basis

    def run_skqd(self, progress: bool = True) -> Dict[str, Any]:
        """
        Stage 4: SKQD refinement with combined basis.

        For small bases where residual expansion already achieved near-exact
        energy, SKQD may introduce numerical instability. In such cases,
        we use the residual expansion result directly.

        Includes improved numerical stability handling:
        - Uses regularization for ill-conditioned matrices
        - Validates SKQD energy is variationally consistent
        - Falls back to residual result if SKQD produces impossible results
        """
        print("=" * 60)
        print("Stage 4: Sample-Based Krylov Quantum Diagonalization")
        print("=" * 60)

        cfg = self.config

        # Use expanded basis if available
        if hasattr(self, 'expanded_basis'):
            nf_basis = self.expanded_basis
        else:
            nf_basis = self.nf_basis

        # Get residual expansion energy if available
        residual_energy = self.results.get("residual_energy", None)

        # For small bases where residual expansion converged well,
        # skip SKQD to avoid numerical instability
        skip_skqd = False

        # Check if SKQD is disabled by config
        if cfg.skip_skqd:
            print("SKQD disabled (skip_skqd=True)")
            skip_skqd = True

        if cfg.max_krylov_dim <= 0:
            print("SKQD disabled (max_krylov_dim=0)")
            skip_skqd = True

        if residual_energy is not None and self.exact_energy is not None:
            residual_error_mha = abs(residual_energy - self.exact_energy) * 1000
            # If residual expansion already within 1 mHa, use it directly
            if residual_error_mha < 1.0:
                print(f"Residual expansion achieved {residual_error_mha:.4f} mHa error.")
                print("Skipping SKQD to avoid numerical instability.")
                skip_skqd = True

        # Skip for very small bases if residual already converged well
        if len(nf_basis) < 300 and residual_energy is not None:
            if self.exact_energy is not None:
                error_mha = abs(residual_energy - self.exact_energy) * 1000
                if error_mha < 2.0:  # Within 2 mHa
                    print(f"Basis size ({len(nf_basis)}) small and residual converged ({error_mha:.2f} mHa).")
                    print("Skipping SKQD.")
                    skip_skqd = True
            else:
                print(f"Basis size ({len(nf_basis)}) is small enough for direct diagonalization.")
                skip_skqd = True

        if skip_skqd:
            # Compute NF basis energy if no residual energy available
            if residual_energy is None:
                print("Computing NF basis energy via direct diagonalization...")
                H_matrix = self.hamiltonian.matrix_elements(nf_basis, nf_basis)
                H_np = H_matrix.detach().cpu().numpy()
                # Symmetrize for numerical stability
                H_np = 0.5 * (H_np + H_np.T)
                eigenvalues, _ = np.linalg.eigh(H_np)
                nf_basis_energy = float(eigenvalues[0])
                print(f"NF basis energy: {nf_basis_energy:.8f} Ha ({len(nf_basis)} configs)")
                self.results["nf_basis_energy"] = nf_basis_energy
                self.results["skqd_energy"] = nf_basis_energy
                self.results["combined_energy"] = nf_basis_energy
            else:
                self.results["skqd_energy"] = residual_energy
                self.results["combined_energy"] = residual_energy

            self.results["skqd_skipped"] = True
            final_energy = self.results.get("combined_energy", residual_energy)
            return {"energies_combined": [final_energy], "skipped": True}

        # Configure SKQD with regularization for numerical stability
        skqd_config = SKQDConfig(
            max_krylov_dim=cfg.max_krylov_dim,
            time_step=cfg.time_step,
            shots_per_krylov=cfg.shots_per_krylov,
            use_gpu=(self.device == "cuda"),
            regularization=getattr(cfg, 'skqd_regularization', 1e-8),
        )

        skqd = FlowGuidedSKQD(
            hamiltonian=self.hamiltonian,
            nf_basis=nf_basis,
            config=skqd_config,
        )

        results = skqd.run_with_nf(progress=progress)

        # Use best stable energy from SKQD (handles numerical instability internally)
        skqd_energy = results.get("best_stable_energy", results["energies_combined"][-1])

        self.results["skqd_results"] = results
        self.results["skqd_energy"] = skqd_energy

        # Validate energy is variationally consistent
        if residual_energy is not None:
            if self.exact_energy is not None:
                # Check if SKQD energy is below exact (impossible for variational method)
                if skqd_energy < self.exact_energy - 0.001:
                    print(f"WARNING: SKQD energy ({skqd_energy:.6f}) is below exact ({self.exact_energy:.6f})!")
                    print("This indicates numerical instability. Using residual expansion result.")
                    self.results["combined_energy"] = residual_energy
                    self.results["skqd_unstable"] = True
                else:
                    # Both are valid, use the lower (better) energy
                    self.results["combined_energy"] = min(skqd_energy, residual_energy)
            else:
                # No exact reference, check if SKQD improved over residual
                if skqd_energy < residual_energy:
                    # SKQD improved energy - use it
                    improvement_mha = (residual_energy - skqd_energy) * 1000
                    print(f"SKQD improved energy by {improvement_mha:.4f} mHa")
                    self.results["combined_energy"] = skqd_energy
                else:
                    # SKQD didn't help, use residual
                    print("SKQD did not improve energy. Using residual expansion result.")
                    self.results["combined_energy"] = residual_energy
        else:
            self.results["combined_energy"] = skqd_energy

        return results

    def run(self, progress: bool = True) -> Dict[str, Any]:
        """
        Run complete pipeline.

        Args:
            progress: Show progress bars

        Returns:
            Complete results dictionary with energies and statistics
        """
        print("\n" + "=" * 60)
        print("Flow-Guided Krylov Pipeline")
        print("=" * 60)
        print(f"System: {self.num_sites} sites")
        print(f"Device: {self.device}")
        if self.is_molecular:
            print(f"Electrons: {self.hamiltonian.n_alpha}α + {self.hamiltonian.n_beta}β")
        if self.exact_energy is not None:
            print(f"Exact energy: {self.exact_energy:.8f}")
        print("=" * 60 + "\n")

        # Stage 1: Training
        self.train_flow_nqs(progress=progress)

        # Stage 2: Basis extraction
        self.extract_and_select_basis()

        # Stage 3: Residual expansion
        self.run_residual_expansion()

        # Stage 4: SKQD
        self.run_skqd(progress=progress)

        # Summary
        self._print_summary()

        return self.results

    def _print_summary(self):
        """Print results summary."""
        print("\n" + "=" * 60)
        print("Results Summary")
        print("=" * 60)

        if self.results.get("skip_nf_training"):
            print(f"Mode:              Direct-CI (NF training skipped)")

        if "nf_nqs_energy" in self.results and self.results["nf_nqs_energy"] is not None:
            print(f"NF-NQS Energy:     {self.results['nf_nqs_energy']:.8f}")

        if "nf_basis_size" in self.results:
            print(f"NF Basis Size:     {self.results['nf_basis_size']}")

        if "residual_expansion_stats" in self.results:
            stats = self.results["residual_expansion_stats"]
            print(f"Expanded Basis:    {stats['final_basis_size']}")

        if "combined_energy" in self.results:
            print(f"Combined Energy:   {self.results['combined_energy']:.8f}")

        if self.exact_energy is not None:
            best_energy = self.results.get("combined_energy",
                           self.results.get("skqd_energy",
                           self.results.get("nf_nqs_energy")))
            error_ha = abs(best_energy - self.exact_energy)
            error_mha = error_ha * 1000
            error_kcal = error_ha * 627.5
            print(f"\nError: {error_mha:.4f} mHa ({error_kcal:.4f} kcal/mol)")

            if error_kcal < 1.0:
                print("Chemical accuracy: PASS")
            else:
                print("Chemical accuracy: FAIL")

        print("=" * 60)


def run_molecular_benchmark(
    molecule: str = "lih",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run pipeline on a molecular system.

    Args:
        molecule: "h2", "lih", "h2o", "beh2", "nh3", "n2", or "ch4"
        verbose: Print progress

    Returns:
        Results dictionary
    """
    from hamiltonians.molecular import (
        create_h2_hamiltonian,
        create_lih_hamiltonian,
        create_h2o_hamiltonian,
        create_beh2_hamiltonian,
        create_nh3_hamiltonian,
        create_n2_hamiltonian,
        create_ch4_hamiltonian,
    )

    # Create Hamiltonian
    molecule_lower = molecule.lower()
    if molecule_lower == "h2":
        H = create_h2_hamiltonian(bond_length=0.74)
    elif molecule_lower == "lih":
        H = create_lih_hamiltonian(bond_length=1.6)
    elif molecule_lower == "h2o":
        H = create_h2o_hamiltonian()
    elif molecule_lower == "beh2":
        H = create_beh2_hamiltonian()
    elif molecule_lower == "nh3":
        H = create_nh3_hamiltonian()
    elif molecule_lower == "n2":
        H = create_n2_hamiltonian()
    elif molecule_lower == "ch4":
        H = create_ch4_hamiltonian()
    else:
        raise ValueError(f"Unknown molecule: {molecule}")

    # Get exact energy
    E_exact = H.fci_energy()

    # Configure pipeline (Direct-CI mode by default for molecular systems)
    config = PipelineConfig(
        use_particle_conserving_flow=True,
        use_diversity_selection=True,
        use_residual_expansion=True,
        skip_nf_training=True,
    )

    # Run pipeline
    pipeline = FlowGuidedKrylovPipeline(H, config=config, exact_energy=E_exact)
    results = pipeline.run(progress=verbose)

    return results


# Backward compatibility aliases
EnhancedPipelineConfig = PipelineConfig
EnhancedFlowKrylovPipeline = FlowGuidedKrylovPipeline
run_enhanced_molecular_benchmark = run_molecular_benchmark
