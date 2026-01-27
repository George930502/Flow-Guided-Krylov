"""
Automatic Parameter Scaling for Flow-Guided Krylov Pipeline.

This module provides intelligent parameter determination based on system size,
using principled scaling laws derived from computational complexity analysis.

Key Features:
- Automatic system tier classification (TINY to HUGE)
- Formula-based parameter scaling using logarithmic/sqrt relationships
- Quality presets (FAST, BALANCED, ACCURATE) for user control
- Adaptive runtime adjustment based on observed metrics
- Memory-aware GPU parameter tuning

Usage:
    from utils.system_scaler import SystemScaler, QualityPreset

    scaler = SystemScaler(preset=QualityPreset.BALANCED)
    metrics = scaler.analyze_system(hamiltonian)
    params = scaler.compute_parameters(metrics)
    config = scaler.create_pipeline_config(params)
"""

import os
import numpy as np
import torch
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, Tuple
from math import comb


class QualityPreset(Enum):
    """Quality presets that trade accuracy for speed."""
    FAST = "fast"           # Quick estimates, reduced accuracy
    BALANCED = "balanced"   # Good balance for production
    ACCURATE = "accurate"   # Maximum accuracy, longer runtime


class SystemTier(Enum):
    """System complexity tiers based on valid configuration count."""
    TINY = "tiny"           # < 1,000 configs
    SMALL = "small"         # 1,000 - 10,000 configs
    MEDIUM = "medium"       # 10,000 - 100,000 configs
    LARGE = "large"         # 100,000 - 1,000,000 configs
    VERY_LARGE = "very_large"  # 1M - 10M configs
    HUGE = "huge"           # > 10M configs


# Multipliers for quality presets
PRESET_MULTIPLIERS = {
    QualityPreset.FAST: {
        "hidden_dim": 0.5,
        "num_layers": 0.5,
        "samples_per_batch": 0.5,
        "max_epochs": 0.5,
        "residual_iterations": 0.5,
        "krylov_dim": 0.5,
        "max_basis_size": 0.5,
        "shots_per_krylov": 0.5,
    },
    QualityPreset.BALANCED: {
        # All multipliers = 1.0 (default)
    },
    QualityPreset.ACCURATE: {
        "hidden_dim": 1.5,
        "num_layers": 1.5,
        "samples_per_batch": 2.0,
        "max_epochs": 2.0,
        "residual_iterations": 2.0,
        "krylov_dim": 1.5,
        "max_basis_size": 2.0,
        "shots_per_krylov": 1.5,
    }
}


@dataclass
class SystemMetrics:
    """Metrics describing the quantum system complexity."""
    n_qubits: int
    n_electrons: int
    n_alpha: int
    n_beta: int
    n_orbitals: int
    n_valid_configs: int  # Size of valid Fock space
    is_molecular: bool = True

    @property
    def tier(self) -> SystemTier:
        """Determine system tier based on configuration space size."""
        n = self.n_valid_configs
        if n < 1_000:
            return SystemTier.TINY
        elif n < 10_000:
            return SystemTier.SMALL
        elif n < 100_000:
            return SystemTier.MEDIUM
        elif n < 1_000_000:
            return SystemTier.LARGE
        elif n < 10_000_000:
            return SystemTier.VERY_LARGE
        return SystemTier.HUGE

    @property
    def log_configs(self) -> float:
        """Log2 of configuration count for scaling formulas."""
        return np.log2(max(1, self.n_valid_configs))

    @property
    def log10_configs(self) -> float:
        """Log10 of configuration count for scaling formulas."""
        return np.log10(max(1, self.n_valid_configs))

    @property
    def sqrt_configs(self) -> float:
        """Square root of configuration count for scaling formulas."""
        return np.sqrt(self.n_valid_configs)


@dataclass
class ScaledParameters:
    """All pipeline parameters determined by system size."""

    # System info
    tier: SystemTier
    n_valid_configs: int

    # NF Architecture
    hidden_dim: int = 256
    num_layers: int = 4
    num_blocks: int = 4
    nf_hidden_dims: list = field(default_factory=lambda: [256, 256])
    nqs_hidden_dims: list = field(default_factory=lambda: [256, 256, 256, 256])

    # Training
    samples_per_batch: int = 2048
    max_epochs: int = 400
    min_epochs: int = 100
    learning_rate: float = 1e-3
    nf_lr: float = 5e-4
    nqs_lr: float = 1e-3

    # Basis Management
    max_basis_size: int = 4096
    max_accumulated_basis: int = 4096
    max_diverse_configs: int = 2048
    min_weight_threshold: float = 1e-5

    # Residual Expansion
    residual_iterations: int = 8
    residual_configs_per_iter: int = 150
    energy_threshold: float = 1e-6

    # SKQD
    use_skqd: bool = True
    krylov_dim: int = 8
    trotter_steps: int = 8
    dt: float = 0.1
    shots_per_krylov: int = 50000

    # Performance
    nqs_chunk_size: int = 16384
    max_cache_size: int = 100000
    parallel_workers: int = 8
    use_parallel_connections: bool = True
    accumulated_energy_interval: int = 50
    use_torch_compile: bool = False
    compile_mode: str = "reduce-overhead"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "tier": self.tier.value,
            "n_valid_configs": self.n_valid_configs,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "samples_per_batch": self.samples_per_batch,
            "max_epochs": self.max_epochs,
            "learning_rate": self.learning_rate,
            "max_basis_size": self.max_basis_size,
            "residual_iterations": self.residual_iterations,
            "use_skqd": self.use_skqd,
            "krylov_dim": self.krylov_dim,
            "use_torch_compile": self.use_torch_compile,
        }


class SystemScaler:
    """
    Determines optimal parameters based on system complexity.

    Uses principled scaling laws:
    - hidden_dim ~ log2(n_configs) * 16
    - samples ~ sqrt(n_configs) * 32
    - epochs ~ log10(n_configs) * 200
    - residual_iter ~ -log10(coverage) * 2
    - krylov_dim ~ log2(n_configs) / 2

    Args:
        preset: Quality preset (FAST, BALANCED, ACCURATE)
        device: Torch device for GPU memory estimation
    """

    def __init__(
        self,
        preset: QualityPreset = QualityPreset.BALANCED,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.preset = preset
        self.device = device
        self.multipliers = PRESET_MULTIPLIERS.get(preset, {})

    def _apply_multiplier(self, value: float, param_name: str) -> float:
        """Apply quality preset multiplier to a parameter."""
        mult = self.multipliers.get(param_name, 1.0)
        return value * mult

    def analyze_system(self, hamiltonian) -> SystemMetrics:
        """
        Analyze a Hamiltonian to extract system metrics.

        Args:
            hamiltonian: Hamiltonian object (molecular or spin)

        Returns:
            SystemMetrics with all relevant system information
        """
        # Check if molecular Hamiltonian
        from hamiltonians.molecular import MolecularHamiltonian
        is_molecular = isinstance(hamiltonian, MolecularHamiltonian)

        if is_molecular:
            n_orb = hamiltonian.n_orbitals
            n_alpha = hamiltonian.n_alpha
            n_beta = hamiltonian.n_beta
            n_electrons = n_alpha + n_beta
            n_valid_configs = comb(n_orb, n_alpha) * comb(n_orb, n_beta)
            n_qubits = 2 * n_orb
        else:
            # Generic spin system
            n_qubits = hamiltonian.num_sites
            n_electrons = 0
            n_alpha = 0
            n_beta = 0
            n_orb = n_qubits // 2
            n_valid_configs = 2 ** n_qubits  # Full Hilbert space

        return SystemMetrics(
            n_qubits=n_qubits,
            n_electrons=n_electrons,
            n_alpha=n_alpha,
            n_beta=n_beta,
            n_orbitals=n_orb,
            n_valid_configs=n_valid_configs,
            is_molecular=is_molecular,
        )

    def compute_parameters(self, metrics: SystemMetrics) -> ScaledParameters:
        """
        Compute optimal parameters based on system metrics.

        Args:
            metrics: SystemMetrics from analyze_system()

        Returns:
            ScaledParameters with all tuned parameters
        """
        n = metrics.n_valid_configs
        log2_n = metrics.log_configs
        log10_n = metrics.log10_configs
        sqrt_n = metrics.sqrt_configs
        tier = metrics.tier

        # ========================
        # NF Architecture
        # ========================
        hidden_dim = int(min(512, max(64, log2_n * 16)))
        hidden_dim = int(self._apply_multiplier(hidden_dim, "hidden_dim"))

        num_layers = int(min(16, max(4, log2_n / 2)))
        num_layers = int(self._apply_multiplier(num_layers, "num_layers"))

        num_blocks = int(min(8, max(2, log2_n / 3)))

        # Build hidden dimensions lists
        nf_hidden_dims = [hidden_dim, hidden_dim]
        nqs_hidden_dims = [hidden_dim] * num_layers

        # ========================
        # Training Parameters
        # ========================
        base_samples = 2048
        samples_per_batch = int(min(16384, max(512, sqrt_n * 32)))
        samples_per_batch = int(self._apply_multiplier(samples_per_batch, "samples_per_batch"))

        max_epochs = int(min(2000, max(200, log10_n * 200)))
        max_epochs = int(self._apply_multiplier(max_epochs, "max_epochs"))
        min_epochs = max(100, max_epochs // 4)

        # Learning rate scales inversely with sqrt(n)
        learning_rate = 1e-3 * min(1.0, max(0.1, 1000 / sqrt_n))
        nf_lr = learning_rate * 0.5
        nqs_lr = learning_rate

        # ========================
        # Basis Management
        # ========================
        max_basis_size = int(min(100000, max(1000, n * 0.1)))
        max_basis_size = int(self._apply_multiplier(max_basis_size, "max_basis_size"))

        max_accumulated_basis = max_basis_size
        max_diverse_configs = int(max_basis_size * 0.8)

        min_weight_threshold = max(1e-6, 1e-4 / log10_n)

        # ========================
        # Residual Expansion
        # ========================
        # Iterations scale with expected coverage gap
        # Assuming NF covers ~5-10% of valid configs
        expected_coverage = 0.05
        residual_iterations = int(max(1, min(15, -np.log10(expected_coverage) * 2)))
        residual_iterations = int(self._apply_multiplier(residual_iterations, "residual_iterations"))

        # Configs per iteration scales with system size
        residual_configs_per_iter = int(min(1000, max(100, sqrt_n * 0.5)))

        energy_threshold = max(1e-8, 1e-6 / log10_n)

        # ========================
        # SKQD Parameters
        # ========================
        # SKQD becomes necessary for larger systems
        use_skqd = n > 50000 or expected_coverage < 0.05

        krylov_dim = int(min(20, max(4, log2_n / 2)))
        krylov_dim = int(self._apply_multiplier(krylov_dim, "krylov_dim"))

        trotter_steps = int(min(16, max(4, log2_n / 3)))
        dt = np.pi / (2 * krylov_dim)

        shots_per_krylov = int(min(100000, max(10000, sqrt_n * 100)))
        shots_per_krylov = int(self._apply_multiplier(shots_per_krylov, "shots_per_krylov"))

        # ========================
        # Performance Parameters
        # ========================
        # Memory management
        nqs_chunk_size = self._compute_chunk_size(metrics)
        max_cache_size = self._compute_cache_size(metrics)

        # Parallelization
        cpu_count = os.cpu_count() or 8
        parallel_workers = min(cpu_count, max(2, int(log2_n / 2)))
        use_parallel_connections = n > 10000

        # Accumulated energy interval
        accumulated_energy_interval = int(max(25, min(200, sqrt_n / 50)))

        # torch.compile
        use_torch_compile = n > 50000 and max_epochs > 500
        compile_mode = "reduce-overhead" if n < 500000 else "max-autotune"

        return ScaledParameters(
            tier=tier,
            n_valid_configs=n,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_blocks=num_blocks,
            nf_hidden_dims=nf_hidden_dims,
            nqs_hidden_dims=nqs_hidden_dims,
            samples_per_batch=samples_per_batch,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            learning_rate=learning_rate,
            nf_lr=nf_lr,
            nqs_lr=nqs_lr,
            max_basis_size=max_basis_size,
            max_accumulated_basis=max_accumulated_basis,
            max_diverse_configs=max_diverse_configs,
            min_weight_threshold=min_weight_threshold,
            residual_iterations=residual_iterations,
            residual_configs_per_iter=residual_configs_per_iter,
            energy_threshold=energy_threshold,
            use_skqd=use_skqd,
            krylov_dim=krylov_dim,
            trotter_steps=trotter_steps,
            dt=dt,
            shots_per_krylov=shots_per_krylov,
            nqs_chunk_size=nqs_chunk_size,
            max_cache_size=max_cache_size,
            parallel_workers=parallel_workers,
            use_parallel_connections=use_parallel_connections,
            accumulated_energy_interval=accumulated_energy_interval,
            use_torch_compile=use_torch_compile,
            compile_mode=compile_mode,
        )

    def _compute_chunk_size(self, metrics: SystemMetrics) -> int:
        """Compute optimal NQS chunk size based on GPU memory."""
        if self.device != "cuda" or not torch.cuda.is_available():
            return 8192  # Conservative CPU default

        try:
            props = torch.cuda.get_device_properties(0)
            available_memory = props.total_memory * 0.3  # 30% for batch processing

            # Estimate bytes per config
            config_memory = metrics.n_qubits * 8  # bytes per config

            # Compute chunk size
            chunk_size = int(available_memory / (config_memory * 100))
            chunk_size = max(1024, min(32768, chunk_size))

            return chunk_size
        except Exception:
            return 16384  # Safe default

    def _compute_cache_size(self, metrics: SystemMetrics) -> int:
        """Compute optimal cache size based on GPU memory and system size."""
        if self.device != "cuda" or not torch.cuda.is_available():
            return 50000  # Conservative CPU default

        try:
            props = torch.cuda.get_device_properties(0)
            cache_memory_budget = props.total_memory * 0.2  # 20% for cache

            # Estimate bytes per cache entry
            avg_connections = metrics.n_qubits * 2
            bytes_per_entry = metrics.n_qubits * 8 * avg_connections * 2

            cache_size = int(cache_memory_budget / max(1, bytes_per_entry))
            cache_size = max(1000, min(200000, cache_size))

            return cache_size
        except Exception:
            return 100000  # Safe default

    def create_pipeline_config(self, params: ScaledParameters):
        """
        Create a PipelineConfig from scaled parameters.

        Args:
            params: ScaledParameters from compute_parameters()

        Returns:
            PipelineConfig with all parameters set
        """
        try:
            from pipeline import PipelineConfig
        except ImportError:
            from ..pipeline import PipelineConfig

        config = PipelineConfig(
            # NF-NQS architecture
            nf_hidden_dims=params.nf_hidden_dims,
            nqs_hidden_dims=params.nqs_hidden_dims,

            # Training
            samples_per_batch=params.samples_per_batch,
            max_epochs=params.max_epochs,
            min_epochs=params.min_epochs,
            nf_lr=params.nf_lr,
            nqs_lr=params.nqs_lr,

            # Basis management
            max_accumulated_basis=params.max_accumulated_basis,
            max_diverse_configs=params.max_diverse_configs,

            # Residual expansion
            residual_iterations=params.residual_iterations,
            residual_configs_per_iter=params.residual_configs_per_iter,
            residual_threshold=params.energy_threshold,

            # SKQD
            max_krylov_dim=params.krylov_dim,
            time_step=params.dt,
            shots_per_krylov=params.shots_per_krylov,
            skip_skqd=not params.use_skqd,

            # Hardware
            device=self.device,
        )

        return config

    def print_parameters(self, params: ScaledParameters):
        """Pretty-print scaled parameters for user review."""
        print("\n" + "=" * 60)
        print(f"System Scaler: {self.preset.value.upper()} preset")
        print("=" * 60)
        print(f"System tier: {params.tier.value.upper()}")
        print(f"Valid configurations: {params.n_valid_configs:,}")
        print()

        print("NF Architecture:")
        print(f"  hidden_dim: {params.hidden_dim}")
        print(f"  num_layers: {params.num_layers}")
        print(f"  nf_hidden_dims: {params.nf_hidden_dims}")
        print(f"  nqs_hidden_dims: {params.nqs_hidden_dims}")
        print()

        print("Training:")
        print(f"  samples_per_batch: {params.samples_per_batch}")
        print(f"  max_epochs: {params.max_epochs}")
        print(f"  learning_rate: {params.learning_rate:.2e}")
        print()

        print("Basis Management:")
        print(f"  max_basis_size: {params.max_basis_size}")
        print(f"  max_diverse_configs: {params.max_diverse_configs}")
        print()

        print("Residual Expansion:")
        print(f"  iterations: {params.residual_iterations}")
        print(f"  configs_per_iter: {params.residual_configs_per_iter}")
        print(f"  energy_threshold: {params.energy_threshold:.2e}")
        print()

        print("SKQD:")
        print(f"  use_skqd: {params.use_skqd}")
        print(f"  krylov_dim: {params.krylov_dim}")
        print(f"  trotter_steps: {params.trotter_steps}")
        print(f"  dt: {params.dt:.4f}")
        print(f"  shots_per_krylov: {params.shots_per_krylov:,}")
        print()

        print("Performance:")
        print(f"  nqs_chunk_size: {params.nqs_chunk_size}")
        print(f"  max_cache_size: {params.max_cache_size:,}")
        print(f"  parallel_workers: {params.parallel_workers}")
        print(f"  use_parallel_connections: {params.use_parallel_connections}")
        print(f"  use_torch_compile: {params.use_torch_compile}")
        print("=" * 60 + "\n")


class AdaptiveAdjuster:
    """
    Runtime adaptive parameter adjustment based on observed metrics.

    Monitors training progress and suggests parameter adjustments
    when performance degrades or bottlenecks are detected.
    """

    def __init__(self, initial_params: ScaledParameters):
        self.params = initial_params
        self.metrics_history: list = []

    def record_metrics(self, metrics: Dict[str, float]):
        """Record metrics for trend analysis."""
        self.metrics_history.append(metrics)

    def check_and_adjust(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Check metrics and suggest parameter adjustments.

        Args:
            metrics: Dictionary with keys like:
                - epoch: Current training epoch
                - loss: Current loss value
                - loss_change: Change from previous epoch
                - coverage: Fraction of config space sampled
                - cache_hit_rate: Connection cache hit rate
                - residual_new_configs: New configs from last residual iteration

        Returns:
            Dictionary of suggested adjustments
        """
        self.record_metrics(metrics)
        adjustments = {}

        # Check for loss plateau early in training
        if metrics.get("epoch", 0) < 100 and metrics.get("loss_change", 1) < 1e-6:
            adjustments["hidden_dim_multiplier"] = 1.2
            adjustments["lr_multiplier"] = 0.5
            adjustments["reason"] = "early_plateau"

        # Check for low coverage
        if metrics.get("coverage", 1.0) < 0.01:
            adjustments["samples_per_batch_multiplier"] = 1.5
            adjustments["reason"] = "low_coverage"

        # Check for low cache hit rate
        if metrics.get("cache_hit_rate", 1.0) < 0.2:
            adjustments["use_cache"] = False
            adjustments["reason"] = "low_cache_hit_rate"

        # Check if residual expansion is finding many new configs
        if metrics.get("residual_new_configs", 0) > 100:
            adjustments["residual_iterations_add"] = 2
            adjustments["reason"] = "active_residual"

        return adjustments

    def get_trend(self, metric_name: str, window: int = 10) -> float:
        """Get trend for a metric over recent history."""
        if len(self.metrics_history) < 2:
            return 0.0

        recent = self.metrics_history[-window:]
        values = [m.get(metric_name, 0) for m in recent if metric_name in m]

        if len(values) < 2:
            return 0.0

        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return slope


def auto_scale_pipeline(
    hamiltonian,
    preset: QualityPreset = QualityPreset.BALANCED,
    verbose: bool = True,
):
    """
    Convenience function to automatically configure pipeline for a system.

    Args:
        hamiltonian: Hamiltonian to analyze
        preset: Quality preset
        verbose: Print parameter summary

    Returns:
        Tuple of (PipelineConfig, ScaledParameters)
    """
    scaler = SystemScaler(preset=preset)
    metrics = scaler.analyze_system(hamiltonian)
    params = scaler.compute_parameters(metrics)

    if verbose:
        scaler.print_parameters(params)

    config = scaler.create_pipeline_config(params)
    return config, params


# Convenience exports
__all__ = [
    "SystemScaler",
    "SystemMetrics",
    "ScaledParameters",
    "QualityPreset",
    "SystemTier",
    "AdaptiveAdjuster",
    "auto_scale_pipeline",
]
