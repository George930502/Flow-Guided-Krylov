# Flow-Guided Krylov Quantum Diagonalization

A hybrid quantum-classical algorithm for computing ground state energies of quantum many-body systems. This project combines two powerful techniques: **Normalizing Flow-Assisted Neural Quantum States (NF-NQS)** for discovering important quantum states, and **Sample-Based Krylov Quantum Diagonalization (SKQD)** for refining energy estimates.

## What This Project Does

Computing the ground state energy of quantum systems is a fundamental problem in physics and chemistry. This pipeline solves it by:

1. **Learning which quantum states matter most** using a neural network-based normalizing flow
2. **Building a compact basis** from those important states
3. **Refining the energy estimate** using Krylov subspace methods with direct diagonalization

The result is a highly accurate ground state energy with minimal computational resources.

## Pipeline Overview

The algorithm runs in three stages:

```
Stage 1: NF-NQS Co-Training
    |   Train two neural networks together:
    |   - Normalizing Flow: learns to sample important basis states
    |   - Neural Quantum State: learns the wavefunction structure
    v
Stage 2: Basis Extraction
    |   Collect the ~1000 most important basis states discovered during training
    v
Stage 3: SKQD Refinement
        Expand the basis using time evolution and solve for the ground state
        via direct diagonalization in the combined basis
        - NF-only energy: using states from the neural network
        - Krylov-only energy: using states from time evolution
        - Combined energy: using both (always the best result)
```

**Note**: The pipeline directly diagonalizes the Hamiltonian in the discovered basis, which gives exact variational energies. No separate amplitude refinement phase is needed because SKQD computes exact eigenvectors in the subspace.

## Key Features

- **Chemical Accuracy**: Achieves <1 kcal/mol error on H2, LiH, and H2O molecules
- **Particle Conservation**: Enforces electron number and spin constraints (critical for molecules)
- **Physics-Guided Training**: Energy importance weighting for faster convergence
- **Residual Expansion**: Selected-CI style recovery of missing configurations
- **GPU Accelerated**: Full CUDA support with vectorized operations for fast training
- **Sparse Time Evolution**: O(nnz) Krylov expansion using scipy.sparse for large systems
- **Docker Ready**: Reproducible environment with one command
- **Flexible**: Works with spin systems and molecular Hamiltonians

## Molecular Benchmark Results

### Enhanced Pipeline (Recommended)

The **enhanced pipeline** achieves chemical accuracy on all tested molecules:

| Molecule | Qubits | Electrons | Exact (Ha) | Error (mHa) | Error (kcal/mol) | Status |
|----------|--------|-----------|------------|-------------|------------------|--------|
| H2       | 4      | 2 (1α+1β) | -1.137284  | 0.00        | 0.00             | **PASS** |
| LiH      | 12     | 4 (2α+2β) | -7.963743  | 0.00        | 0.00             | **PASS** |
| H2O      | 14     | 10 (5α+5β)| -81.123161 | 0.00        | 0.00             | **PASS** |

**Key Enhancements** (over original method):
1. **Particle-conserving flow**: Enforces N_e and S_z conservation (samples only valid configurations)
2. **Physics-guided training**: Energy importance weighting for better convergence
3. **Diversity-aware selection**: Excitation rank bucketing to capture singles, doubles, etc.
4. **Residual expansion**: Selected-CI style recovery of missing important configurations

Chemical accuracy threshold: **1 kcal/mol = 1.6 mHa**

Run the enhanced benchmark:
```bash
docker-compose run --rm flow-krylov-gpu python examples/enhanced_benchmark.py --molecule all
```

### Original Pipeline Comparison

The original pipeline (without particle conservation) struggled with larger molecules:

| Molecule | Original NF-NQS | Original SKQD | Enhanced Pipeline | Improvement |
|----------|-----------------|---------------|-------------------|-------------|
| H2       | 0.00 mHa        | 0.00 mHa      | 0.00 mHa          | -           |
| LiH      | 1.64 mHa (FAIL) | 539 mHa       | 0.00 mHa (PASS)   | 100%        |
| H2O      | 5416 mHa (FAIL) | 1726 mHa      | 0.00 mHa (PASS)   | 100%        |

**Why the original failed on H2O**: H2O has 10 electrons in 7 orbitals (14 qubits). Only 441 configurations out of 16,384 (2.7%) satisfy particle number conservation. Without enforcing this constraint, the flow sampled mostly invalid states.

## Performance Optimizations

The codebase includes significant optimizations for molecular Hamiltonian calculations:

### Sparse Time Evolution (Critical for LiH and larger)

For systems with Hilbert dimension > 1000, dense matrix exponentials become prohibitive. We use sparse time evolution:

```python
# O(nnz * krylov_dim) instead of O(dim^3) dense matrix exponential
from scipy.sparse.linalg import expm_multiply
psi_evolved = expm_multiply(-1j * dt * H_sparse, psi)
```

This enables:
- LiH (4096-dimensional) to run in ~5 minutes instead of hours
- Memory-efficient handling of sparse molecular Hamiltonians

### GPU-Optimized Hamiltonian Construction

- **Vectorized Diagonal Elements**: Batch computation using `torch.einsum` with precomputed Coulomb (J) and Exchange (K) tensors
- **Hash-Based Lookups**: O(1) basis state lookups instead of O(n) linear search
- **Numpy-Based Excitation Computation**: Efficient single/double excitation generation

### GPU-Optimized Training

- **Incremental Hamiltonian Caching**: O(n) updates when new basis states are discovered
- **Vectorized Energy Computation**: Full batch energy evaluation without Python loops
- **Basis Management**: Configurable limits to prevent memory issues on large systems

### Benchmark Performance (RTX 4090)

| Operation | H2 (4 qubits) | LiH (12 qubits) |
|-----------|---------------|-----------------|
| NF-NQS Training | ~2s | ~280s |
| SKQD Refinement | ~0.2s | ~70s |
| Total Pipeline | ~2s | ~350s |

## Installation

### Prerequisites

- Docker with NVIDIA Container Toolkit (for GPU support)
- NVIDIA GPU with CUDA 12.x (recommended)

### Quick Start with Docker

```bash
# Clone the repository
git clone https://github.com/George930502/Flow-Guided-Krylov.git
cd Flow-Guided-Krylov

# Build the Docker image
docker-compose build

# Run with GPU support (recommended)
docker-compose run --rm flow-krylov-gpu

# Or run with CPU only
docker-compose run --rm flow-krylov-cpu

# Interactive shell for development
docker-compose run --rm shell
```

### Docker Services

| Service | Description |
|---------|-------------|
| `flow-krylov-gpu` | GPU-accelerated execution (requires nvidia-docker) |
| `flow-krylov-cpu` | CPU-only execution |
| `shell` | Interactive bash shell for development |

### Running Scripts

```bash
# Run the molecular benchmark
docker-compose run --rm flow-krylov-gpu python examples/molecular_benchmark.py --molecule all

# Run H2 test only
docker-compose run --rm flow-krylov-gpu python examples/molecular_test.py --molecule h2

# Run unit tests
docker-compose run --rm flow-krylov-gpu pytest tests/
```

### Environment Details

The Docker image includes:
- PyTorch 2.2.0 with CUDA 12.1
- NumPy, SciPy, Matplotlib
- normflows (for the RealNVP architecture)
- PySCF (for molecular Hamiltonians)
- CuPy (for GPU-accelerated eigensolvers)

### Manual Installation (Alternative)

If you prefer not to use Docker:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install numpy scipy matplotlib tqdm normflows pyscf
pip install cupy-cuda12x  # Optional: GPU-accelerated eigensolvers
pip install -e .
```

## Usage

### Basic Example

```python
import torch
from src.pipeline import FlowGuidedKrylovPipeline, PipelineConfig
from src.hamiltonians.molecular import create_h2_hamiltonian

# Create H2 molecular Hamiltonian
H = create_h2_hamiltonian(bond_length=0.74)  # Angstrom

# Get exact energy for comparison
exact_energy, _ = H.exact_ground_state()

# Configure the pipeline
config = PipelineConfig(
    # Neural network architecture
    nf_coupling_layers=3,
    nf_hidden_dims=[128, 128],
    nqs_hidden_dims=[128, 128, 128],

    # Training parameters
    samples_per_batch=1000,
    max_epochs=300,

    # SKQD parameters
    max_krylov_dim=8,
    shots_per_krylov=30000,

    # Hardware
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# Run the full pipeline
pipeline = FlowGuidedKrylovPipeline(H, config=config, exact_energy=exact_energy)
results = pipeline.run(progress=True)

# Print results
print(f"NF-NQS Energy:   {results['nf_nqs_energy']:.6f} Ha")
print(f"Combined Energy: {results['combined_energy']:.6f} Ha")
print(f"Exact Energy:    {exact_energy:.6f} Ha")
```

### Enhanced Pipeline (Recommended for Molecules)

For molecular systems, use the enhanced pipeline with particle conservation:

```python
import torch
from src.enhanced_pipeline import EnhancedFlowKrylovPipeline, EnhancedPipelineConfig
from src.hamiltonians.molecular import create_h2o_hamiltonian

# Create H2O molecular Hamiltonian
H = create_h2o_hamiltonian(oh_length=0.96, angle=104.5)

# Get exact energy for comparison
exact_energy, _ = H.exact_ground_state()

# Configure the enhanced pipeline
config = EnhancedPipelineConfig(
    # Critical: particle-conserving flow for molecules
    use_particle_conserving_flow=True,

    # Physics-guided training weights
    teacher_weight=0.35,
    physics_weight=0.55,  # Higher for strongly correlated systems
    entropy_weight=0.10,

    # Architecture
    nf_hidden_dims=[256, 256],
    nqs_hidden_dims=[256, 256, 256, 256],

    # Training
    samples_per_batch=1500,
    max_epochs=400,

    # Diversity selection and residual expansion
    use_diversity_selection=True,
    use_residual_expansion=True,
    residual_iterations=8,

    device="cuda" if torch.cuda.is_available() else "cpu",
)

# Run the enhanced pipeline
pipeline = EnhancedFlowKrylovPipeline(H, config=config, exact_energy=exact_energy)
results = pipeline.run(progress=True)

# Print results
E_pipeline = results.get('combined_energy', results.get('skqd_energy'))
error_mha = abs(E_pipeline - exact_energy) * 1000
print(f"Pipeline Energy: {E_pipeline:.6f} Ha")
print(f"Exact Energy:    {exact_energy:.6f} Ha")
print(f"Error: {error_mha:.4f} mHa ({error_mha * 0.6275:.4f} kcal/mol)")
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nf_coupling_layers` | 4 | Number of RealNVP coupling layers |
| `nf_hidden_dims` | [512, 512] | Hidden layer sizes for flow network |
| `nqs_hidden_dims` | [512, 512, 512, 512] | Hidden layer sizes for NQS network |
| `samples_per_batch` | 3000 | Training batch size |
| `nf_lr` | 5e-4 | Learning rate for normalizing flow |
| `nqs_lr` | 1e-3 | Learning rate for NQS |
| `max_epochs` | 500 | Maximum training epochs |
| `max_krylov_dim` | 12 | Number of Krylov expansion steps |
| `time_step` | 0.1 | Time step for Krylov evolution |
| `shots_per_krylov` | 100000 | Samples per Krylov state |

### Stability Parameters

These parameters help prevent training instabilities:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_accumulated_energy` | True | Compute energy on accumulated basis for stability |
| `ema_decay` | 0.95 | Exponential moving average for energy tracking |
| `entropy_weight` | 0.01 | Entropy regularization to prevent basis collapse |
| `max_accumulated_basis` | 2048 | Hard cap on accumulated basis size (critical for LiH+) |

## How It Works

### Stage 1: NF-NQS Co-Training

Two neural networks are trained together:

1. **Normalizing Flow (RealNVP)**: Transforms a simple distribution into one that samples important basis states. It learns which quantum configurations contribute most to the ground state.

2. **Neural Quantum State**: A dense neural network that learns the wavefunction amplitudes for each basis state.

The flow is trained to match the probability distribution implied by the NQS amplitudes, while the NQS is trained to minimize energy. This co-training discovers both the important states AND their correct amplitudes.

### Stage 2: Basis Extraction

After training, we collect all unique basis states that were sampled during training. This accumulated basis (typically ~1000 states) captures the most important configurations for the ground state.

### Stage 3: SKQD Refinement

The Krylov method expands the basis by simulating time evolution:
- Start from a reference state
- Apply sparse time evolution operator repeatedly: `|psi(t)> = exp(-iHt)|psi(0)>`
- Sample basis states from each evolved state
- Combine with the NF basis
- **Directly diagonalize** the Hamiltonian in this combined basis

This gives an exact variational energy in the subspace, which systematically improves as the basis grows.

**Why no amplitude refinement?** SKQD performs direct diagonalization in the combined basis, yielding exact eigenvectors and eigenvalues within that subspace. There's no need for a separate NQS to learn amplitudes - the diagonalization gives them exactly.

## Project Structure

```
Flow-Guided-Krylov/
├── src/
│   ├── pipeline.py              # Original pipeline orchestration
│   ├── enhanced_pipeline.py     # Enhanced pipeline with all improvements
│   ├── flows/
│   │   ├── discrete_flow.py     # Normalizing flow for discrete states
│   │   ├── training.py          # Co-training logic with stabilization
│   │   ├── particle_conserving_flow.py  # Particle-conserving flow (Gumbel-top-k)
│   │   └── physics_guided_training.py   # Physics-guided training loss
│   ├── nqs/
│   │   ├── base.py              # Neural quantum state interface
│   │   ├── dense.py             # Dense NQS implementation
│   │   └── complex_nqs.py       # Complex-valued NQS
│   ├── hamiltonians/
│   │   ├── base.py              # Hamiltonian interface
│   │   ├── spin.py              # Ising and Heisenberg models
│   │   └── molecular.py         # Molecular Hamiltonians (H2, LiH, H2O)
│   ├── krylov/
│   │   ├── skqd.py              # SKQD implementation with sparse evolution
│   │   ├── basis_sampler.py     # Krylov basis sampling
│   │   └── residual_expansion.py # Selected-CI style residual expansion
│   └── postprocessing/
│       ├── eigensolver.py       # Sparse eigensolvers (including Davidson)
│       ├── projected_hamiltonian.py
│       └── diversity_selection.py  # Excitation rank-aware basis selection
├── examples/
│   ├── enhanced_benchmark.py    # Enhanced pipeline benchmark (H2, LiH, H2O)
│   ├── molecular_benchmark.py   # Original benchmark comparison
│   ├── molecular_test.py        # Molecular system tests
│   └── h2_example.py            # H2 molecule example
├── tests/                       # Unit tests
├── Dockerfile                   # Docker configuration
└── docker-compose.yml           # Docker Compose services
```

## References

1. **NF-NQS**: "Improved Ground State Estimation in Quantum Field Theories via Normalising Flow-Assisted Neural Quantum States" - [arXiv:2506.12128](https://arxiv.org/abs/2506.12128)

2. **SKQD**: "Sample-based Krylov Quantum Diagonalization" (Yu et al., IBM Quantum) - [arXiv:2501.09702](https://arxiv.org/html/2501.09702v1)

3. **Neural Quantum States**: Carleo & Troyer, "Solving the quantum many-body problem with artificial neural networks", Science 2017 - [arXiv:1606.02318](https://arxiv.org/abs/1606.02318)

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
