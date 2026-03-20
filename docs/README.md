# Flow-Guided Krylov Pipeline Documentation

## Overview

This documentation describes the Flow-Guided Krylov pipeline for computing molecular ground state energies with chemical accuracy. The pipeline combines:

- **Normalizing Flows (NF)** for efficient configuration sampling
- **Neural Quantum States (NQS)** for wavefunction amplitude learning
- **Selected-CI methods** for basis expansion
- **Krylov subspace methods (SKQD)** for energy refinement

---

## Documentation Index

### Architecture

| Document | Description |
|----------|-------------|
| [PIPELINE_ARCHITECTURE.md](PIPELINE_ARCHITECTURE.md) | Complete pipeline overview, data flow, adaptive scaling |

### Pipeline Stages

| Stage | Document | Description |
|-------|----------|-------------|
| 1 | [STAGE1_NF_NQS_COTRAINING.md](STAGE1_NF_NQS_COTRAINING.md) | Particle-conserving flow, physics-guided training, mixed-objective loss |
| 2 | [STAGE2_DIVERSITY_SELECTION.md](STAGE2_DIVERSITY_SELECTION.md) | Excitation bucketing, DPP-inspired selection, importance weighting |
| 3 | [STAGE3_RESIDUAL_EXPANSION.md](STAGE3_RESIDUAL_EXPANSION.md) | Residual-based selection, PT2 importance, iterative expansion |
| 4 | [STAGE4_SKQD.md](STAGE4_SKQD.md) | Krylov subspace, time evolution, subspace diagonalization |

### Modules

| Module | Document | Description |
|--------|----------|-------------|
| Hamiltonian | [MODULE_MOLECULAR_HAMILTONIAN.md](MODULE_MOLECULAR_HAMILTONIAN.md) | Second quantization, Jordan-Wigner, PySCF integration |

---

## Quick Start

```python
from enhanced_pipeline import EnhancedFlowKrylovPipeline
from hamiltonians.molecular import create_lih_hamiltonian

# Create molecular Hamiltonian
H = create_lih_hamiltonian(bond_length=1.6)

# Run full pipeline (auto-adapts to system size)
pipeline = EnhancedFlowKrylovPipeline(H)
results = pipeline.run(progress=True)

# Check results
print(f"Energy: {results['combined_energy']:.6f} Ha")
print(f"Error: {abs(results['combined_energy'] - H.fci_energy()) * 1000:.2f} mHa")
```

---

## Pipeline Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PIPELINE FLOW                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Molecular Geometry                                                 │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐               │
│  │   Stage 1   │──►│   Stage 2   │──►│   Stage 3   │               │
│  │  NF-NQS     │   │  Diversity  │   │  Residual   │               │
│  │  Training   │   │  Selection  │   │  Expansion  │               │
│  └─────────────┘   └─────────────┘   └─────────────┘               │
│         │                                   │                       │
│         │              ┌────────────────────┘                       │
│         │              │                                            │
│         ▼              ▼                                            │
│       ┌─────────────────────┐                                       │
│       │      Stage 4        │                                       │
│       │   SKQD Refinement   │                                       │
│       └─────────────────────┘                                       │
│                │                                                    │
│                ▼                                                    │
│        Ground State Energy                                          │
│        (Chemical Accuracy)                                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Techniques by Stage

### Stage 1: NF-NQS Co-Training

| Technique | Purpose |
|-----------|---------|
| Gumbel-Top-K | Guarantee particle conservation |
| Temperature annealing | Explore → exploit transition |
| Mixed-objective loss | Balance teacher/physics/entropy |
| REINFORCE | NQS energy minimization |
| Accumulated basis | Track discovered configs |

### Stage 2: Diversity Selection

| Technique | Purpose |
|-----------|---------|
| Excitation bucketing | Classify by physical meaning |
| Budget allocation | Prioritize doubles (50%) |
| Hamming distance | Measure configuration diversity |
| DPP-inspired selection | Maximize importance × diversity |

### Stage 3: Residual Expansion

| Technique | Purpose |
|-----------|---------|
| Residual analysis | Find missing important configs |
| PT2 importance | Energy-based prioritization |
| Iterative expansion | Systematic basis growth |
| Convergence checking | Stop when saturated |

### Stage 4: SKQD

| Technique | Purpose |
|-----------|---------|
| Particle-conserving subspace | Reduce dimension 10-100× |
| Sparse time evolution | Efficient e^{-iHt}|ψ⟩ |
| Cumulative sampling | Comprehensive basis from all Krylov states |
| Numerical regularization | Stability for ill-conditioned matrices |

---

## Supported Molecules

| Molecule | Qubits | Valid Configs | Typical Error |
|----------|--------|---------------|---------------|
| H₂ | 4 | 4 | < 0.1 mHa |
| LiH | 12 | 225 | < 0.5 mHa |
| H₂O | 14 | 441 | < 0.5 mHa |
| BeH₂ | 14 | 1,225 | < 1.0 mHa |
| NH₃ | 16 | 3,136 | < 5 mHa |
| N₂ | 20 | 14,400 | < 10 mHa |
| CH₄ | 18 | 15,876 | < 10 mHa |

Chemical accuracy threshold: **1.6 mHa** (1 kcal/mol)

---

## Requirements

- Python 3.8+
- PyTorch (CUDA recommended)
- PySCF (for molecular integrals)
- NumPy, SciPy
- CuPy (optional, for GPU acceleration)

---

## Citation

If you use this pipeline, please cite:

```bibtex
@software{flow_guided_krylov,
  title = {Flow-Guided Krylov Pipeline for Molecular Ground States},
  year = {2024},
  description = {Combining normalizing flows, neural quantum states, and Krylov methods for quantum chemistry}
}
```

---

## License

[Add license information here]
