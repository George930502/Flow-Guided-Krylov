# Flow-Guided Krylov Quantum Diagonalization

A hybrid quantum-classical algorithm for computing ground state energies of quantum many-body systems. This project combines two powerful techniques: **Normalizing Flow-Assisted Neural Quantum States (NF-NQS)** for discovering important quantum states, and **Sample-Based Krylov Quantum Diagonalization (SKQD)** for refining energy estimates.

## What This Project Does

Computing the ground state energy of quantum systems is a fundamental problem in physics and chemistry. This pipeline solves it by:

1. **Learning which quantum states matter most** using a neural network-based normalizing flow
2. **Building a compact basis** from those important states
3. **Refining the energy estimate** using Krylov subspace methods

The result is a highly accurate ground state energy with minimal computational resources.

## Pipeline Overview

The algorithm runs in four stages:

```
Stage 1: NF-NQS Co-Training
    │   Train two neural networks together:
    │   - Normalizing Flow: learns to sample important basis states
    │   - Neural Quantum State: learns the wavefunction structure
    ▼
Stage 2: Basis Extraction
    │   Collect the ~1000 most important basis states discovered during training
    ▼
Stage 3: Amplitude Refinement (Optional)
    │   Fine-tune the wavefunction amplitudes on the fixed basis
    ▼
Stage 4: SKQD Refinement
        Expand the basis using time evolution and solve for the ground state
        - NF-only energy: using states from the neural network
        - Krylov-only energy: using states from time evolution
        - Combined energy: using both (always the best result)
```

## Key Features

- **High Accuracy**: Achieves 0.0000% error on benchmark systems
- **GPU Accelerated**: Full CUDA support for fast training
- **Stabilized Training**: Built-in mechanisms to prevent training instabilities
- **Docker Ready**: Reproducible environment with one command
- **Flexible**: Works with spin systems and molecular Hamiltonians

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
# Run the validation test
docker-compose run --rm flow-krylov-gpu python run_test.py

# Run examples
docker-compose run --rm flow-krylov-gpu python examples/ising_example.py
docker-compose run --rm flow-krylov-gpu python examples/h2_example.py

# Run unit tests
docker-compose run --rm flow-krylov-gpu pytest tests/
```

### Environment Details

The Docker image includes:
- PyTorch 2.2.0 with CUDA 12.1
- NumPy, SciPy, Matplotlib
- normflows (for the RealNVP architecture)
- CuPy (for GPU-accelerated eigensolvers)

### Manual Installation (Alternative)

If you prefer not to use Docker:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install numpy scipy matplotlib tqdm normflows
pip install cupy-cuda12x  # Optional: GPU-accelerated eigensolvers
pip install -e .
```

## Usage

### Basic Example

```python
import torch
from src.pipeline import FlowGuidedKrylovPipeline, PipelineConfig
from src.hamiltonians.spin import TransverseFieldIsing

# Define your quantum system
hamiltonian = TransverseFieldIsing(
    num_spins=10,      # Number of spin sites
    V=1.0,             # Spin-spin interaction strength
    h=1.0,             # Transverse field strength
    L=5,               # Interaction range
    periodic=True      # Periodic boundary conditions
)

# Get exact energy for comparison (only for small systems)
exact_energy, _ = hamiltonian.exact_ground_state()

# Configure the pipeline
config = PipelineConfig(
    # Neural network architecture
    nf_coupling_layers=4,
    nf_hidden_dims=[512, 512],
    nqs_hidden_dims=[512, 512, 512, 512],

    # SKQD parameters
    max_krylov_dim=12,

    # Hardware
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# Run the full pipeline
pipeline = FlowGuidedKrylovPipeline(hamiltonian, config=config, exact_energy=exact_energy)
results = pipeline.run(progress=True)

# Print results
print(f"NF-NQS Energy:  {results['nf_nqs_energy']:.6f}")
print(f"Combined Energy: {results['combined_energy']:.6f}")
print(f"Exact Energy:    {exact_energy:.6f}")
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

## Benchmark Results

### Transverse Field Ising Model (10 spins)

**Test Configuration**:
- System: 10 spins with periodic boundaries
- Interaction strength V=1.0, field strength h=1.0, range L=5
- Hardware: NVIDIA RTX 4090 Laptop GPU (16GB)

**Training**:
- Converged at epoch 150 (of 500 max)
- Training time: ~10 minutes
- Final basis size: 1024 states

**Energy Results**:

| Method | Energy | Description |
|--------|--------|-------------|
| NF-NQS Training | -21.39 | Energy during neural network training |
| NF-only (SKQD) | -45.56 | Diagonalization using NF basis only |
| Krylov-only | -27.98 | Diagonalization using Krylov basis only |
| **Combined** | **-45.56** | Diagonalization using both bases |
| **Exact** | **-45.56** | Exact diagonalization reference |

**Accuracy**: 0.0000% error

### Understanding the Results

The pipeline produces three energy estimates from the SKQD stage:

- **NF-only**: Energy computed using only the basis states discovered by the normalizing flow. When the NF successfully finds the important states, this is already very accurate.

- **Krylov-only**: Energy computed using states generated by time evolution. This systematically explores the Hilbert space but may miss important states.

- **Combined**: Energy computed using both NF and Krylov states together. This is always the best result because a larger basis can only improve (or maintain) the energy estimate.

In the benchmark above, the NF discovered excellent basis states, so NF-only ≈ Combined ≈ Exact. The Krylov expansion serves as a safety net for harder problems where NF might miss important states.

## Project Structure

```
Flow-Guided-Krylov/
├── src/
│   ├── pipeline.py              # Main pipeline orchestration
│   ├── flows/
│   │   ├── discrete_flow.py     # Normalizing flow for discrete states
│   │   └── training.py          # Co-training logic with stabilization
│   ├── nqs/
│   │   ├── base.py              # Neural quantum state interface
│   │   ├── dense.py             # Dense NQS implementation
│   │   └── complex_nqs.py       # Complex-valued NQS
│   ├── hamiltonians/
│   │   ├── base.py              # Hamiltonian interface
│   │   ├── spin.py              # Ising and Heisenberg models
│   │   └── molecular.py         # Molecular Hamiltonians
│   ├── krylov/
│   │   ├── skqd.py              # SKQD implementation
│   │   └── basis_sampler.py     # Krylov basis sampling
│   └── postprocessing/
│       ├── eigensolver.py       # Sparse eigensolvers
│       └── projected_hamiltonian.py
├── examples/
│   ├── ising_example.py         # Transverse Field Ising Model example
│   └── h2_example.py            # H2 molecule example
├── tests/                       # Unit tests
├── run_test.py                  # Quick validation script
├── Dockerfile                   # Docker configuration
└── docker-compose.yml           # Docker Compose services
```

## How It Works

### Stage 1: NF-NQS Co-Training

Two neural networks are trained together:

1. **Normalizing Flow (RealNVP)**: Transforms a simple distribution into one that samples important basis states. It learns which quantum configurations contribute most to the ground state.

2. **Neural Quantum State**: A dense neural network that learns the wavefunction amplitudes for each basis state.

The flow is trained to match the probability distribution implied by the NQS amplitudes, while the NQS is trained to minimize energy. This co-training discovers both the important states AND their correct amplitudes.

### Stage 2: Basis Extraction

After training, we collect all unique basis states that were sampled during training. This accumulated basis (typically ~1000 states) captures the most important configurations for the ground state.

### Stage 3: Amplitude Refinement (Optional)

A fresh NQS can be trained on the fixed basis to refine the amplitudes. This stage is optional because the co-trained NQS often already has good amplitudes.

### Stage 4: SKQD Refinement

The Krylov method expands the basis by simulating time evolution:
- Start from a reference state (e.g., Néel state: |↑↓↑↓...⟩)
- Apply the time evolution operator repeatedly
- Sample basis states from each evolved state
- Combine with the NF basis
- Diagonalize the Hamiltonian in this combined basis

This gives a variational upper bound on the ground state energy that systematically improves as the basis grows.

## References

1. **NF-NQS**: "Improved Ground State Estimation in Quantum Field Theories via Normalising Flow-Assisted Neural Quantum States" - [arXiv:2506.12128](https://arxiv.org/abs/2506.12128)

2. **SKQD**: "Sample-based Krylov Quantum Diagonalization" (Yu et al., IBM Quantum) - [arXiv:2501.09702](https://arxiv.org/html/2501.09702v1)

3. **Neural Quantum States**: Carleo & Troyer, "Solving the quantum many-body problem with artificial neural networks", Science 2017 - [arXiv:1606.02318](https://arxiv.org/abs/1606.02318)

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
