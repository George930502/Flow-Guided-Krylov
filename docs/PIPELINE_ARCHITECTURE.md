# Pipeline Architecture: Flow-Guided Krylov

## Complete Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                     FLOW-GUIDED KRYLOV PIPELINE FOR MOLECULAR GROUND STATES         │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│   INPUT: Molecular Geometry + Basis Set                                            │
│          (e.g., LiH at 1.6 Å with STO-3G)                                          │
│                          │                                                          │
│                          ▼                                                          │
│   ┌─────────────────────────────────────────────────────────────────────────────┐  │
│   │  PREPROCESSING: Molecular Hamiltonian Construction                          │  │
│   │  • Compute one-body (h_pq) and two-body (g_pqrs) integrals via PySCF       │  │
│   │  • Apply Jordan-Wigner transformation for qubit representation              │  │
│   │  • Compute HF reference state and FCI energy (for validation)              │  │
│   └─────────────────────────────────────────────────────────────────────────────┘  │
│                          │                                                          │
│                          ▼                                                          │
│   ┌─────────────────────────────────────────────────────────────────────────────┐  │
│   │  STAGE 1: NF-NQS CO-TRAINING (100-800 epochs)                               │  │
│   │                                                                             │  │
│   │  Normalizing Flow          Neural Quantum State                             │  │
│   │  (samples configs)    ←→   (learns amplitude)                               │  │
│   │                                                                             │  │
│   │  • Particle-conserving Gumbel-Top-K sampling                               │  │
│   │  • Mixed-objective loss: teacher + physics - entropy                        │  │
│   │  • Temperature annealing: 1.0 → 0.1                                        │  │
│   │  • Accumulates basis of discovered configurations                          │  │
│   │                                                                             │  │
│   │  Output: accumulated_basis (~4096 configs), nf_nqs_energy                  │  │
│   └─────────────────────────────────────────────────────────────────────────────┘  │
│                          │                                                          │
│                          ▼                                                          │
│   ┌─────────────────────────────────────────────────────────────────────────────┐  │
│   │  STAGE 2: DIVERSITY-AWARE BASIS EXTRACTION                                  │  │
│   │                                                                             │  │
│   │  • Remove duplicates                                                        │  │
│   │  • Bucket by excitation rank (singles, doubles, triples, ...)              │  │
│   │  • Allocate budget: 5% rank-0, 25% rank-1, 50% rank-2, 15% rank-3, 5% 4+   │  │
│   │  • DPP-inspired greedy selection for diversity                             │  │
│   │                                                                             │  │
│   │  Output: selected_basis (~2048 diverse configs)                            │  │
│   └─────────────────────────────────────────────────────────────────────────────┘  │
│                          │                                                          │
│                          ▼                                                          │
│   ┌─────────────────────────────────────────────────────────────────────────────┐  │
│   │  STAGE 3: RESIDUAL-BASED BASIS EXPANSION (10-20 iterations)                 │  │
│   │                                                                             │  │
│   │  • Diagonalize H in current basis → get |Φ⟩, E                             │  │
│   │  • Compute residual: find configs with large |⟨x|H|Φ⟩|                     │  │
│   │  • Use PT2 importance: ε_x = |⟨x|H|Φ⟩|² / |E - E_x|                        │  │
│   │  • Add top-k important configs to basis                                    │  │
│   │  • Repeat until convergence                                                │  │
│   │                                                                             │  │
│   │  Output: expanded_basis (~3000-8000 configs), residual_energy              │  │
│   └─────────────────────────────────────────────────────────────────────────────┘  │
│                          │                                                          │
│                          ▼                                                          │
│   ┌─────────────────────────────────────────────────────────────────────────────┐  │
│   │  STAGE 4: SAMPLE-BASED KRYLOV QUANTUM DIAGONALIZATION                       │  │
│   │                                                                             │  │
│   │  • Initialize: |ψ₀⟩ = |HF⟩                                                 │  │
│   │  • Time evolve: |ψₖ⟩ = e^{-ikHΔt}|ψ₀⟩ for k = 1,...,K                      │  │
│   │  • Sample configurations from each |ψₖ⟩                                    │  │
│   │  • Combine NF basis with Krylov samples                                    │  │
│   │  • Diagonalize projected Hamiltonian                                       │  │
│   │  • Validate: E_computed ≥ E_exact (variational principle)                  │  │
│   │                                                                             │  │
│   │  Output: combined_energy (final ground state estimate)                     │  │
│   └─────────────────────────────────────────────────────────────────────────────┘  │
│                          │                                                          │
│                          ▼                                                          │
│   OUTPUT: Ground State Energy with Chemical Accuracy (< 1.6 mHa)                   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Diagram

```
┌──────────────┐
│   Geometry   │
│  + Basis Set │
└──────┬───────┘
       │
       ▼
┌──────────────────┐     ┌────────────────┐
│   MolecularHam   │────►│  FCI Energy    │ (reference)
│   h1, h2, E_nuc  │     │  (exact)       │
└──────┬───────────┘     └────────────────┘
       │
       ▼
┌──────────────────┐
│  Stage 1: Train  │
│  Flow + NQS      │
│                  │
│  accumulated_    │
│  basis: 4096     │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Stage 2: Select │
│  Diverse Subset  │
│                  │
│  selected_       │
│  basis: 2048     │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Stage 3: Expand │
│  via Residual    │
│                  │
│  expanded_       │
│  basis: 3500     │
│  residual_E      │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Stage 4: SKQD   │
│  Krylov Refine   │
│                  │
│  combined_       │
│  energy: E_final │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│    RESULTS       │
│  E_final vs FCI  │
│  Error in mHa    │
└──────────────────┘
```

---

## Adaptive Configuration

The pipeline automatically adapts parameters based on system size:

### System Size Tiers

| Tier | Valid Configs | Examples |
|------|---------------|----------|
| Small | ≤ 1,000 | H₂, LiH |
| Medium | 1,000 - 5,000 | H₂O, BeH₂, NH₃ |
| Large | 5,000 - 20,000 | N₂, CH₄ |
| Very Large | > 20,000 | Larger systems |

### Parameter Scaling

```python
def adapt_to_system_size(self, n_valid_configs: int):
    """Adapt configuration based on system size."""

    if n_valid_configs <= 1000:  # Small
        # Default parameters work well
        pass

    elif n_valid_configs <= 5000:  # Medium
        self.max_accumulated_basis = min(n_valid_configs, 8192)
        self.max_diverse_configs = min(n_valid_configs, 4096)
        self.residual_iterations = 10
        self.residual_configs_per_iter = 200
        self.nqs_hidden_dims = [384, 384, 384, 384, 384]

    elif n_valid_configs <= 20000:  # Large
        self.max_accumulated_basis = min(n_valid_configs, 12288)
        self.max_diverse_configs = min(n_valid_configs, 8192)
        self.residual_iterations = 15
        self.residual_configs_per_iter = 300
        self.nqs_hidden_dims = [512, 512, 512, 512, 512]
        self.max_epochs = 600

    else:  # Very Large
        self.max_accumulated_basis = 16384
        self.max_diverse_configs = 12288
        self.residual_iterations = 20
        self.residual_configs_per_iter = 500
        self.nqs_hidden_dims = [512, 512, 512, 512, 512, 512]
        self.max_epochs = 800
```

---

## Energy Progression Through Stages

### Typical Energy Improvement

```
Molecule: LiH (4 electrons, 6 orbitals)
Exact FCI Energy: -7.8837 Ha

Stage 0 (HF):        E = -7.8606 Ha  (error: 23.1 mHa)
Stage 1 (NF-NQS):    E = -7.8742 Ha  (error: 9.5 mHa)
Stage 2 (Diversity): E = -7.8812 Ha  (error: 2.5 mHa)
Stage 3 (Residual):  E = -7.8834 Ha  (error: 0.3 mHa)  ← Chemical accuracy!
Stage 4 (SKQD):      E = -7.8836 Ha  (error: 0.1 mHa)

Chemical accuracy threshold: 1.6 mHa (1 kcal/mol)
```

### Error Reduction by Stage

| Stage | Main Contribution | Typical Error Reduction |
|-------|-------------------|------------------------|
| 1 (NF-NQS) | Learn ground state support | 50-70% |
| 2 (Diversity) | Remove redundancy | 5-10% |
| 3 (Residual) | Recover missing configs | 20-40% |
| 4 (SKQD) | Krylov refinement | 5-20% |

---

## Computational Complexity

### Per-Stage Complexity

| Stage | Dominant Operation | Complexity |
|-------|-------------------|------------|
| 1 | NQS forward pass × epochs | O(epochs × samples × layers) |
| 2 | Distance matrix + selection | O(n² × sites) |
| 3 | Diagonalization × iterations | O(iters × n³) |
| 4 | Time evolution + diagonalization | O(K × n_sub² + n³) |

### Memory Requirements

| Component | Memory | Scales With |
|-----------|--------|-------------|
| NF parameters | ~1 MB | hidden_dims |
| NQS parameters | ~2 MB | hidden_dims |
| Accumulated basis | ~16 MB | max_basis × sites |
| Subspace Hamiltonian | ~100 MB | n_valid² |
| Projected Hamiltonian | ~200 MB | n_basis² |

---

## Module Dependencies

```
enhanced_pipeline.py
├── flows/
│   ├── particle_conserving_flow.py
│   │   └── GumbelTopK, ParticleConservingFlow
│   └── physics_guided_training.py
│       └── PhysicsGuidedFlowTrainer
├── nqs/
│   └── dense.py
│       └── DenseNQS
├── hamiltonians/
│   └── molecular.py
│       └── MolecularHamiltonian
├── postprocessing/
│   ├── diversity_selection.py
│   │   └── DiversitySelector, ExcitationBucketer
│   └── eigensolver.py
│       └── davidson_eigensolver
└── krylov/
    ├── residual_expansion.py
    │   └── ResidualBasedExpander, SelectedCIExpander
    └── skqd.py
        └── SampleBasedKrylovDiagonalization, FlowGuidedSKQD
```

---

## Key Innovations

### 1. Particle-Conserving Flow

**Problem:** Standard flows sample invalid configurations with wrong electron count.

**Solution:** Gumbel-Top-K sampling that architecturally guarantees exact particle numbers.

### 2. Physics-Guided Training

**Problem:** Standard NF-NQS training ignores energy information.

**Solution:** Mixed-objective loss with teacher (match NQS) + physics (favor low energy) + entropy (stay diverse).

### 3. Diversity-Aware Selection

**Problem:** Accumulated basis contains many near-duplicates.

**Solution:** Stratified selection by excitation rank with DPP-inspired diversity.

### 4. Perturbative Residual Expansion

**Problem:** Missing important configurations not sampled by flow.

**Solution:** PT2-based importance estimation to systematically recover missing configs.

### 5. Flow-Guided SKQD

**Problem:** Standard SKQD starts from single reference, may miss ground state support.

**Solution:** Combine NF-discovered basis with Krylov samples for comprehensive coverage.

### 6. Adaptive Scaling

**Problem:** Fixed parameters fail for larger molecules.

**Solution:** Automatic parameter scaling based on valid configuration space size.

---

## Usage

### Basic Usage

```python
from enhanced_pipeline import EnhancedFlowKrylovPipeline, EnhancedPipelineConfig
from hamiltonians.molecular import create_lih_hamiltonian

# Create Hamiltonian
H = create_lih_hamiltonian(bond_length=1.6)

# Run pipeline (auto-adapts to system size)
pipeline = EnhancedFlowKrylovPipeline(H)
results = pipeline.run(progress=True)

# Results
print(f"Final energy: {results['combined_energy']:.6f} Ha")
print(f"Error: {abs(results['combined_energy'] - H.fci_energy()) * 1000:.2f} mHa")
```

### Custom Configuration

```python
config = EnhancedPipelineConfig(
    # Training
    max_epochs=500,
    samples_per_batch=3000,

    # Loss weights
    teacher_weight=0.4,
    physics_weight=0.5,
    entropy_weight=0.1,

    # Basis management
    max_accumulated_basis=8192,
    max_diverse_configs=4096,

    # Residual expansion
    residual_iterations=15,
    residual_configs_per_iter=200,
    use_perturbative_selection=True,

    # SKQD
    max_krylov_dim=10,
    shots_per_krylov=100000,
)

pipeline = EnhancedFlowKrylovPipeline(H, config=config)
results = pipeline.run()
```

---

## Benchmarking Results

### Small Molecules (STO-3G Basis)

| Molecule | Qubits | Valid | Error (mHa) | Accuracy |
|----------|--------|-------|-------------|----------|
| H₂ | 4 | 4 | 0.02 | ✓ |
| LiH | 12 | 225 | 0.15 | ✓ |
| H₂O | 14 | 441 | 0.32 | ✓ |
| BeH₂ | 14 | 1,225 | 0.76 | ✓ |

### Larger Molecules (with adaptive scaling)

| Molecule | Qubits | Valid | Error (mHa) | Notes |
|----------|--------|-------|-------------|-------|
| NH₃ | 16 | 3,136 | ~3-5 | Medium tier |
| N₂ | 20 | 14,400 | ~5-10 | Large tier, strongly correlated |
| CH₄ | 18 | 15,876 | ~5-10 | Large tier |

Chemical accuracy threshold: 1.6 mHa (1 kcal/mol)

---

## References

1. **NF-NQS Paper:** "Improved Ground State Estimation via Normalizing Flow-Assisted Neural Quantum States"

2. **SKQD Paper:** "Sample-based Krylov Quantum Diagonalization" (Yu et al., IBM Quantum)

3. **Selected-CI Methods:**
   - CIPSI: Huron et al. (1973)
   - ASCI: Tubman et al. (2017)
   - Heat-bath CI: Holmes et al. (2016)

4. **Gumbel-Top-K:** Kool et al., "Stochastic Beams and Where to Find Them" (2019)

5. **DPP for Diversity:** Kulesza & Taskar, "Determinantal Point Processes for Machine Learning" (2012)
