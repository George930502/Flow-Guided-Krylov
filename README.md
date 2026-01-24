# Flow-Guided Krylov Quantum Diagonalization

A hybrid quantum-classical algorithm for computing ground state energies of molecular systems with **chemical accuracy** (< 1 kcal/mol). This pipeline combines **Normalizing Flow-Assisted Neural Quantum States (NF-NQS)** for discovering important quantum states with **Sample-Based Krylov Quantum Diagonalization (SKQD)** for energy refinement.

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

## Key Techniques

### Stage 1: Particle-Conserving Flow

**Problem:** Standard flows sample invalid configurations with wrong electron count.

**Solution:** Gumbel-Top-K sampling guarantees exactly k electrons per spin channel:
```python
# Instead of independent Bernoulli sampling:
# samples = (probs > uniform) → wrong electron count

# We use differentiable top-k selection:
gumbel_noise = -log(-log(uniform))
perturbed_logits = logits + gumbel_noise
_, indices = topk(perturbed_logits, k=n_electrons)
samples = one_hot(indices)  # exactly k electrons guaranteed
```

### Stage 3: PT2 Importance Estimation

Configurations are selected based on second-order perturbation theory:

$$\epsilon_x = \frac{|\langle x|H|\Phi\rangle|^2}{|E - E_x|}$$

This estimates how much adding configuration x would lower the energy.

### Early Stopping for Residual Expansion

```
Iter 1: 944 -> 1053, E = -61.540, ΔE = --
Iter 2: 1053 -> 1253, E = -61.778, ΔE = 237.52 mHa
Iter 3: 1253 -> 1453, E = -61.791, ΔE = 13.32 mHa
Iter 4: 1453 -> 1653, E = -61.791, ΔE = 0.28 mHa
Iter 5: 1653 -> 1853, E = -61.791, ΔE = 0.02 mHa
Converged: improvement < 0.05 mHa for 2 iterations
```

## Project Structure

```
Flow-Guided-Krylov/
├── src/
│   ├── pipeline.py                    # Main pipeline (consolidated)
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
├── docs/
│   ├── README.md                        # Documentation index
│   ├── PIPELINE_ARCHITECTURE.md         # Full architecture details
│   ├── STAGE1_NF_NQS_COTRAINING.md     # Stage 1 techniques
│   ├── STAGE2_DIVERSITY_SELECTION.md   # Stage 2 techniques
│   ├── STAGE3_RESIDUAL_EXPANSION.md    # Stage 3 techniques
│   ├── STAGE4_SKQD.md                  # Stage 4 techniques
│   └── MODULE_MOLECULAR_HAMILTONIAN.md # Hamiltonian module
├── examples/
│   ├── enhanced_benchmark.py            # Molecular benchmarks
│   └── molecular_test.py                # Quick tests
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

## References

1. **NF-NQS**: "Improved Ground State Estimation in Quantum Field Theories via Normalising Flow-Assisted Neural Quantum States" - [arXiv:2506.12128](https://arxiv.org/abs/2506.12128)

2. **SKQD**: "Sample-based Krylov Quantum Diagonalization" (Yu et al., IBM Quantum) - [arXiv:2501.09702](https://arxiv.org/html/2501.09702v1)

3. **Selected-CI Methods**:
   - CIPSI: Huron et al. (1973)
   - ASCI: Tubman et al. (2017)
   - Heat-bath CI: Holmes et al. (2016)

4. **Gumbel-Top-K**: Kool et al., "Stochastic Beams and Where to Find Them" (2019)

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
