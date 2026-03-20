# HI+NQS+SQD: Neural Quantum State Sampling for Sample-based Quantum Diagonalization

A classical analog of IBM's HI-VQE algorithm, replacing quantum circuits with autoregressive Neural Quantum States (NQS) for sample-based quantum diagonalization.

## Method Overview

**HI+NQS+SQD** (Handover Iterative NQS + Sample-based Quantum Diagonalization) is a self-consistent loop where an autoregressive Transformer NQS and IBM's SQD solver improve each other iteratively.

### Algorithm Flowchart

```
┌─────────────────────────────────────────────────────────────────┐
│                      HI+NQS+SQD Algorithm                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐                                           │
│  │  Initialize NQS   │  Autoregressive Transformer              │
│  │  (random weights) │  on GPU (CUDA)                           │
│  └────────┬─────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────────────────────────────────┐        │
│  │  ① NQS Sampling (GPU)                               │        │
│  │                                                     │        │
│  │  For each orbital i = 1, ..., N:                    │        │
│  │    P(σᵢ | σ₁,...,σᵢ₋₁) via causal self-attention   │        │
│  │    Sample σᵢ ∈ {0,1} with particle number constraint│        │
│  │                                                     │        │
│  │  → n_samples configurations (all valid)             │        │
│  └────────┬────────────────────────────────────────────┘        │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────────────────────────────────┐        │
│  │  ② Cumulative Basis Update                          │        │
│  │                                                     │        │
│  │  - Deduplicate new configs against existing basis    │        │
│  │  - Add truly new configs to cumulative set           │        │
│  │  - (Optional) Classical expansion: add singles/      │        │
│  │    doubles excitations from top-amplitude configs    │        │
│  └────────┬────────────────────────────────────────────┘        │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────────────────────────────────┐        │
│  │  ③ SQD Diagonalization (IBM qiskit-addon-sqd)       │        │
│  │                                                     │        │
│  │  - Configuration recovery (fix particle number)      │        │
│  │  - solve_fermion(bitstrings, h1e, h2e)              │        │
│  │    → Projects H into subspace of sampled configs     │        │
│  │    → PySCF selected CI kernel diagonalizes           │        │
│  │  - Returns: E₀, orbital occupancies                  │        │
│  └────────┬────────────────────────────────────────────┘        │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────────────────────────────────┐        │
│  │  ④ NQS Update (GPU backpropagation)                 │        │
│  │                                                     │        │
│  │  Loss = λ_wf × (-Σ wᵢ log p_NQS(xᵢ))              │        │
│  │       + λ_E  × (Σ wᵢ (Hᵢᵢ-E₀) log p_NQS(xᵢ))     │        │
│  │       + λ_ent × mean(log p_NQS)                     │        │
│  │                                                     │        │
│  │  where wᵢ = softmax(-(Hᵢᵢ - E₀))                   │        │
│  │                                                     │        │
│  │  Mini-batch gradient descent (avoid GPU OOM)         │        │
│  └────────┬────────────────────────────────────────────┘        │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────────────────────────────────┐        │
│  │  ⑤ Convergence Check                                │        │
│  │                                                     │        │
│  │  If |ΔE| < 10⁻⁶ for 3 consecutive iterations:      │        │
│  │    → CONVERGED, return best energy                   │        │
│  │  Else:                                              │        │
│  │    → Back to ① with updated NQS                     │        │
│  └─────────────────────────────────────────────────────┘        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Comparison: HI-VQE vs HI+NQS+SQD

```
┌─────────────────────────────┐     ┌─────────────────────────────┐
│        HI-VQE (IBM)         │     │      HI+NQS+SQD (Ours)     │
├─────────────────────────────┤     ├─────────────────────────────┤
│                             │     │                             │
│  Quantum Circuit U(θ)|HF⟩   │     │  Transformer NQS p_θ(x)    │
│  (CUDA-Q UCCSD)            │     │  (Autoregressive, GPU)      │
│         │                   │     │         │                   │
│         ▼                   │     │         ▼                   │
│  Measure → bitstrings       │     │  Sample → configurations    │
│         │                   │     │         │                   │
│         ▼                   │     │         ▼                   │
│  ┌─────────────────────┐   │     │  ┌─────────────────────┐   │
│  │  IBM solve_fermion  │   │     │  │  IBM solve_fermion  │   │
│  │  (qiskit-addon-sqd) │   │     │  │  (qiskit-addon-sqd) │   │
│  └──────────┬──────────┘   │     │  └──────────┬──────────┘   │
│             │               │     │             │               │
│             ▼               │     │             ▼               │
│  COBYLA/SPSA updates θ     │     │  Backprop updates θ         │
│  (noisy, ~3 evals/step)   │     │  (exact gradients, fast)    │
│             │               │     │             │               │
│             ▼               │     │             ▼               │
│  Repeat until convergence   │     │  Repeat until convergence   │
│                             │     │                             │
│  Requires: quantum hardware │     │  Requires: GPU only         │
└─────────────────────────────┘     └─────────────────────────────┘
```

### NQS Architecture: Autoregressive Transformer

```
Configuration: [α₁, α₂, ..., αₙ, β₁, β₂, ..., βₙ]

Alpha channel (causal self-attention):
  P(α₁)
  P(α₂ | α₁)              ← each orbital sees all previous
  P(α₃ | α₁, α₂)
  ...
  P(αₙ | α₁, ..., αₙ₋₁)

Beta channel (causal self-attention + cross-attention to alpha):
  P(β₁ | α₁, ..., αₙ)     ← sees full alpha configuration
  P(β₂ | β₁, α₁, ..., αₙ)
  ...
  P(βₙ | β₁, ..., βₙ₋₁, α₁, ..., αₙ)

┌──────────────────────────────────────────────────────────┐
│  Transformer Block (× N_layers)                          │
│  ┌────────────────────┐  ┌─────────────────────────────┐ │
│  │  Alpha:             │  │  Beta:                      │ │
│  │  Causal Self-Attn   │  │  Causal Self-Attn           │ │
│  │  + LayerNorm        │  │  + Cross-Attn (to Alpha)    │ │
│  │  + FFN              │  │  + LayerNorm + FFN          │ │
│  └────────────────────┘  └─────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘

Auto-scaling by system size:
  ≤10Q:  embed=64,  heads=4, layers=3
  12-14Q: embed=128, heads=4, layers=4
  16-20Q: embed=128, heads=8, layers=6
  22-30Q: embed=192, heads=8, layers=6
  32-40Q: embed=256, heads=8, layers=8
  42Q+:   embed=256, heads=8, layers=10
```

### Key Differences from HI-VQE

| | HI-VQE (IBM) | HI+NQS+SQD (Ours) |
|--|--------------|---------------------|
| Sampler | Quantum circuit (UCCSD) | Autoregressive Transformer NQS |
| Hardware | Quantum computer | Classical GPU (H200) |
| Parameter update | SPSA/COBYLA (noisy gradients) | Backpropagation (exact gradients) |
| Particle conservation | Post-selection or EPA gates | Constrained autoregressive sampling |
| SQD backend | qiskit-addon-sqd | qiskit-addon-sqd (same) |
| Sampling diversity | Limited by circuit depth | Autoregressive gives high diversity |

## Benchmark Design

We systematically compare **6 methods** across 11 molecules (12-30 qubits):

### Methods Tested

```
Classical (NQS-based):              Quantum (circuit-based):
┌──────────────────────┐            ┌──────────────────────┐
│ NQS+SQD              │            │ QC+SQD               │
│ (two-stage, no       │            │ (CUDA-Q UCCSD +      │
│  feedback)            │            │  IBM SQD, one-shot)  │
├──────────────────────┤            ├──────────────────────┤
│ NQS+SKQD             │            │ QC+SKQD              │
│ (two-stage, Krylov   │            │ (Trotter evolution + │
│  expansion)           │            │  Krylov sampling)    │
├──────────────────────┤            ├──────────────────────┤
│ HI+NQS+SQD ★         │            │ HI-VQE               │
│ (iterative feedback  │            │ (iterative circuit   │
│  loop, our method)    │            │  optimization)       │
└──────────────────────┘            └──────────────────────┘
```

### Classical Baselines

| Method | Description |
|--------|-------------|
| FCI | Full Configuration Interaction (exact, exponential cost) |
| CCSD | Coupled Cluster Singles and Doubles |
| CCSD(T) | CCSD with perturbative triples correction |
| SCI (CIPSI) | Selected Configuration Interaction with PT2 selection |

### Test Molecules

| Molecule | Qubits | Basis | Type | Hilbert Space |
|----------|--------|-------|------|---------------|
| LiH | 12 | STO-3G | Full | 225 |
| H2O | 14 | STO-3G | Full | 441 |
| BeH2 | 14 | STO-3G | Full | 1,225 |
| NH3 | 16 | STO-3G | Full | 3,136 |
| CH4 | 18 | STO-3G | Full | 15,876 |
| N2 | 20 | STO-3G | Full | 14,400 |
| HCN | 22 | STO-3G | Full | 108,900 |
| C2H2 | 24 | STO-3G | Full | 627,264 |
| H2S | 26 | STO-3G | Full | 3,025 |
| C2H4 | 28 | STO-3G | Full | 11,778,624 |
| Benzene | 30 | STO-3G | CAS(6,15) | 207,025 |

### Stability Testing

Each method tested with **5 random seeds** to measure mean and standard deviation.

## Results

### Accuracy Comparison (error vs FCI in mHa)

| Molecule | Qubits | FCI (Ha) | CCSD | CCSD(T) | SCI (basis) | **HI+NQS+SQD** (basis) |
|----------|--------|----------|------|---------|-------------|------------------------|
| LiH | 12 | -7.8823 | 0.011 | 0.002 | 0.000 (69) | **0.000** (225) |
| H2O | 14 | -75.0132 | 0.118 | 0.050 | 0.000 (133) | **0.000** (441) |
| BeH2 | 14 | -15.5951 | 0.397 | 0.185 | 0.000 (169) | **0.000** (1,200) |
| NH3 | 16 | -55.5177 | 0.091 | -0.033 | 0.000 (1,576) | **-0.126** (3,074) |
| CH4 | 18 | -39.8060 | 0.234 | 0.094 | 0.001 (1,629) | **0.000** (11,129) |
| N2 | 20 | -107.6541 | 3.925 | 2.201 | 0.000 (1,588) | **0.000** (12,632) |
| HCN | 22 | -91.8422 | 3.446 | 2.178 | 0.010 (3,833) | **0.000** (15,000) |
| H2S | 26 | -394.3547 | 0.066 | 0.018 | 0.000 (865) | **0.000** (3,013) |
| Benzene | 30 | -227.9565 | — | — | 0.083 (1,713) | **0.000** (10,000) |

All results within **chemical accuracy** (< 1.6 mHa).

### Molecules without FCI Reference (absolute energies in Ha)

| Molecule | Qubits | Hilbert Space | CCSD(T) | SCI (basis) | **HI+NQS+SQD** (basis) |
|----------|--------|---------------|---------|-------------|------------------------|
| C2H2 | 24 | 627,264 | -76.0227 | -76.0245 (5,302) | **-76.0246** (10,000) |
| C2H4 | 28 | 11,778,624 | -77.2348 | -77.2351 (10,000) | **-77.2353** (10,000) |

HI+NQS+SQD achieves **lower energy** than both CCSD(T) and SCI for large molecules.

### HI+NQS+SQD vs HI-VQE (quantum circuit)

| Molecule | Qubits | HI-VQE err (mHa) | HI+NQS+SQD err (mHa) |
|----------|--------|-------------------|------------------------|
| LiH | 12 | 19.612 ± 0.413 | **0.000 ± 0.000** |
| H2O | 14 | 43.282 ± 5.208 | **0.000 ± 0.000** |
| BeH2 | 14 | 30.899 ± 5.424 | **0.000 ± 0.000** |
| NH3 | 16 | 62.339 ± 1.335 | **-0.126** |

(5 random seeds, mean ± std)

### Computation Time (GPU: NVIDIA H200)

| Molecule | Qubits | SCI Time | HI+NQS+SQD Time | Bottleneck |
|----------|--------|----------|------------------|------------|
| LiH | 12 | <1s | 4s | NQS update |
| H2O | 14 | <1s | 2s | NQS update |
| BeH2 | 14 | <1s | 2s | NQS update |
| NH3 | 16 | 3s | 6s | NQS update |
| CH4 | 18 | 4s | 54s | SQD (solve_fermion) |
| N2 | 20 | 3s | 44s | SQD (solve_fermion) |
| HCN | 22 | 10s | 68s | SQD (solve_fermion) |
| C2H2 | 24 | 100s | 485s | SQD (solve_fermion) |
| H2S | 26 | <1s | 11s | NQS sampling |
| C2H4 | 28 | 744s | 485s | SQD (solve_fermion) |
| Benzene | 30 | 7s | 102s | SQD (solve_fermion) |

### Time Breakdown per Iteration (representative)

```
CH4 (18Q):  [sample=0.4s  sqd=10.7s  update=2.1s]  → SQD dominates
N2  (20Q):  [sample=0.5s  sqd=11.1s  update=1.5s]  → SQD dominates
HCN (22Q):  [sample=1.3s  sqd=9.3s   update=5.7s]  → SQD dominates
```

NQS sampling on GPU is fast (~1s). The bottleneck is IBM's `solve_fermion` on CPU.

## Project Architecture

```
nqs-sqd/
├── src/
│   ├── methods/                    # 6 comparison methods
│   │   ├── hi_nqs_sqd.py          # ★ HI+NQS+SQD (our method)
│   │   ├── hi_vqe.py              # HI-VQE (CUDA-Q + IBM SQD)
│   │   ├── nqs_sqd.py             # NQS+SQD (two-stage)
│   │   ├── nqs_skqd.py            # NQS+SKQD (two-stage)
│   │   ├── qc_sqd.py              # QC+SQD (quantum circuit)
│   │   └── qc_skqd.py             # QC+SKQD (quantum Krylov)
│   ├── samplers/
│   │   ├── cudaq_sampler.py       # CUDA-Q UCCSD circuit sampler
│   │   ├── cudaq_circuits.py      # CUDA-Q kernel definitions
│   │   ├── transformer_nf_sampler.py  # Transformer NQS sampler
│   │   └── nf_sampler.py          # Dense NF sampler
│   ├── nqs/
│   │   ├── transformer.py         # Autoregressive Transformer NQS
│   │   └── dense.py               # Dense feed-forward NQS
│   ├── solvers/
│   │   ├── sqd.py                 # SQD solver
│   │   ├── skqd.py                # SKQD Krylov solver
│   │   ├── fci.py                 # Full CI (exact)
│   │   ├── ccsd.py                # CCSD / CCSD(T)
│   │   └── sci.py                 # CIPSI selected CI
│   ├── hamiltonians/
│   │   └── molecular.py           # PySCF molecular Hamiltonians
│   ├── molecules.py               # 20+ molecule registry (4-58Q)
│   └── utils/
├── scripts/                        # Benchmark and test scripts
├── results/                        # Output data
└── README.md
```

## Dependencies

- **PyTorch** >= 2.0 (GPU acceleration for NQS)
- **PySCF** >= 2.4 (molecular integrals)
- **qiskit-addon-sqd** >= 0.12 (IBM's SQD solver: `solve_fermion`)
- **CUDA-Q** >= 0.8 (quantum circuit simulation for HI-VQE)
- **NumPy**, **SciPy**, **Numba** (numerics)

## Quick Start

```bash
# Install
pip install torch pyscf qiskit-addon-sqd cuda-quantum

# Run HI+NQS+SQD on a molecule
python -c "
from src.molecules import get_molecule
from src.methods.hi_nqs_sqd import run_hi_nqs_sqd, HINQSSQDConfig

H, info = get_molecule('H2O')
cfg = HINQSSQDConfig(n_samples=5000, max_iterations=30)
result = run_hi_nqs_sqd(H, info, config=cfg)
print(f'Energy: {result.energy:.10f} Ha')
print(f'Basis: {result.diag_dim}, Time: {result.wall_time:.1f}s')
"

# Run full benchmark
python scripts/run_six_methods.py --molecules "LiH,H2O,BeH2"
```

## References

1. Pellow-Jarman et al. (2025) "HIVQE: Handover Iterative Variational Quantum Eigensolver for Efficient Quantum Chemistry Calculations", arXiv:2503.06292
2. Robledo-Moreno et al. (2024) "Chemistry beyond exact solutions on a quantum-centric supercomputer", Nature
3. Yu et al. (2025) "Quantum-Centric Algorithm for Sample-Based Krylov Diagonalization", arXiv:2501.09702
4. von Glehn et al. (2023) "A self-attention ansatz for ab-initio quantum chemistry" (Psiformer)
5. Huron, Malrieu, Rancurel (1973) "Iterative perturbation calculations" (CIPSI)

## License

MIT
