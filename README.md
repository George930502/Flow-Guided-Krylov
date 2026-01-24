# Flow-Guided Krylov Quantum Diagonalization

A hybrid quantum-classical algorithm for computing ground state energies of molecular systems with **chemical accuracy** (< 1 kcal/mol). This pipeline combines [Normalizing Flow-Assisted Neural Quantum States (NF-NQS)](https://arxiv.org/abs/2506.12128) for discovering important quantum states with [Sample-Based Krylov Quantum Diagonalization (SKQD)](https://arxiv.org/abs/2501.09702) for energy refinement, using the [Gumbel-Top-K trick](https://arxiv.org/pdf/1903.06059) for particle-conserving sampling.

## Pipeline Overview

The algorithm runs in four stages:

```
Stage 1: Physics-Guided NF-NQS Training
    │   Train particle-conserving normalizing flow with NQS
    │   Mixed objective: teacher + physics + entropy
    ▼
Stage 2: Diversity-Aware Basis Extraction
    │   Select configurations by excitation rank (singles, doubles, etc.)
    │   DPP-inspired selection for maximum diversity
    ▼
Stage 3: Residual-Based Expansion (Selected-CI Style)
    │   Iteratively add missing important configurations
    │   PT2 importance estimation: ε = |⟨x|H|Φ⟩|² / |E - Eₓ|
    │   Early stopping when energy converges
    ▼
Stage 4: SKQD Refinement
        Krylov subspace diagonalization for final energy
```

## Key Features

- **Chemical Accuracy**: Achieves < 1 kcal/mol error on H₂, LiH, H₂O, and larger molecules
- **Particle Conservation**: Gumbel-Top-K sampling guarantees correct electron count
- **Physics-Guided Training**: Energy importance weighting for faster convergence
- **Adaptive Scaling**: Automatic parameter adjustment based on system size
- **Early Stopping**: Stops residual expansion when energy improvement stagnates
- **GPU Accelerated**: Full CUDA support with vectorized operations

## Molecular Benchmark Results

| Molecule | Qubits | Valid Configs | Error (mHa) | Error (kcal/mol) | Status |
|----------|--------|---------------|-------------|------------------|--------|
| H₂       | 4      | 4             | < 0.01      | < 0.01           | **PASS** |
| LiH      | 12     | 225           | < 0.5       | < 0.3            | **PASS** |
| H₂O      | 14     | 441           | < 0.5       | < 0.3            | **PASS** |
| BeH₂     | 14     | 1,225         | < 1.0       | < 0.6            | **PASS** |
| NH₃      | 16     | 3,136         | < 5.0       | < 3.1            | In progress |
| N₂       | 20     | 14,400        | < 10.0      | < 6.3            | In progress |

**Chemical accuracy threshold: 1.6 mHa (1 kcal/mol)**

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/George930502/Flow-Guided-Krylov.git
cd Flow-Guided-Krylov

# Install with Docker (recommended)
docker-compose build
docker-compose run --rm flow-krylov-gpu

# Or install manually
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install numpy scipy matplotlib tqdm normflows pyscf
pip install -e .
```

### Basic Usage

```python
from src.pipeline import FlowGuidedKrylovPipeline, PipelineConfig
from src.hamiltonians.molecular import create_lih_hamiltonian

# Create molecular Hamiltonian
H = create_lih_hamiltonian(bond_length=1.6)
E_exact = H.fci_energy()

# Run pipeline (auto-adapts to system size)
pipeline = FlowGuidedKrylovPipeline(H, exact_energy=E_exact)
results = pipeline.run()

# Check results
print(f"Energy: {results['combined_energy']:.6f} Ha")
print(f"Error: {abs(results['combined_energy'] - E_exact) * 1000:.2f} mHa")
```

### Run Molecular Benchmarks

```bash
# Run all molecules
python examples/benchmark.py --molecule all

# Run specific molecule
python examples/benchmark.py --molecule lih

# Available molecules: h2, lih, h2o, beh2, nh3, n2, ch4
```

### Configuration Options

```python
from src.pipeline import PipelineConfig

config = PipelineConfig(
    # Particle conservation (critical for molecules)
    use_particle_conserving_flow=True,

    # Physics-guided training weights
    teacher_weight=0.5,
    physics_weight=0.4,
    entropy_weight=0.1,

    # Training
    samples_per_batch=2000,
    max_epochs=400,

    # Residual expansion (Selected-CI style)
    use_residual_expansion=True,
    residual_iterations=8,
    use_perturbative_selection=True,  # PT2 importance

    # SKQD
    max_krylov_dim=8,

    # Hardware
    device="cuda",
)
```

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     FLOW-GUIDED KRYLOV PIPELINE                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   INPUT: Molecular Geometry + Basis Set                                 │
│                          │                                              │
│                          ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │  PREPROCESSING: Molecular Hamiltonian Construction              │  │
│   │  • PySCF integrals (h_pq, g_pqrs)                               │  │
│   │  • Jordan-Wigner transformation                                 │  │
│   │  • HF reference state                                           │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                          │                                              │
│                          ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │  STAGE 1: Physics-Guided NF-NQS Training                        │  │
│   │  • Gumbel-Top-K sampling (particle conservation)                │  │
│   │  • Mixed loss: L = w_t·L_teacher + w_p·L_physics - w_e·H       │  │
│   │  • Temperature annealing: T = 1.0 → 0.1                        │  │
│   │  Output: accumulated_basis                                      │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                          │                                              │
│                          ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │  STAGE 2: Diversity-Aware Basis Selection                       │  │
│   │  • Excitation rank bucketing (singles, doubles, ...)           │  │
│   │  • Budget: 5% rank-0, 25% rank-1, 50% rank-2, ...              │  │
│   │  • DPP-inspired greedy selection                                │  │
│   │  Output: selected_basis                                         │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                          │                                              │
│                          ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │  STAGE 3: Residual-Based Expansion (Selected-CI)                │  │
│   │  • PT2 importance: ε_x = |⟨x|H|Φ⟩|² / |E - E_x|                │  │
│   │  • Batch diagonal computation (optimized)                       │  │
│   │  • Early stopping on energy stagnation                          │  │
│   │  Output: expanded_basis, residual_energy                        │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                          │                                              │
│                          ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │  STAGE 4: Sample-Based Krylov Quantum Diagonalization           │  │
│   │  • Sparse time evolution: |ψ_k⟩ = e^{-ikHΔt}|ψ_0⟩              │  │
│   │  • Cumulative Krylov basis sampling                             │  │
│   │  • Regularized subspace diagonalization                         │  │
│   │  Output: combined_energy (final result)                         │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                          │                                              │
│                          ▼                                              │
│   OUTPUT: Ground State Energy with Chemical Accuracy                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
Flow-Guided-Krylov/
├── src/
│   ├── pipeline.py                    # Main pipeline
│   ├── flows/
│   │   ├── particle_conserving_flow.py  # Gumbel-Top-K flow
│   │   └── physics_guided_training.py   # Mixed-objective training
│   ├── nqs/
│   │   └── dense.py                     # Neural quantum state
│   ├── hamiltonians/
│   │   └── molecular.py                 # Molecular Hamiltonians (PySCF)
│   ├── krylov/
│   │   ├── skqd.py                      # SKQD with sparse evolution
│   │   └── residual_expansion.py        # Selected-CI expansion
│   └── postprocessing/
│       └── diversity_selection.py       # Excitation rank bucketing
├── docs/                                # Detailed documentation
├── examples/
│   └── benchmark.py                     # Molecular benchmarks
└── tests/                               # Unit tests
```

## Documentation

See the `docs/` folder for detailed documentation:

- [Pipeline Architecture](docs/PIPELINE_ARCHITECTURE.md) - Complete pipeline overview
- [Stage 1: NF-NQS Co-Training](docs/STAGE1_NF_NQS_COTRAINING.md) - 10 techniques explained
- [Stage 2: Diversity Selection](docs/STAGE2_DIVERSITY_SELECTION.md) - Excitation bucketing
- [Stage 3: Residual Expansion](docs/STAGE3_RESIDUAL_EXPANSION.md) - PT2 importance
- [Stage 4: SKQD](docs/STAGE4_SKQD.md) - Krylov subspace methods
- [Molecular Hamiltonian](docs/MODULE_MOLECULAR_HAMILTONIAN.md) - PySCF integration

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
