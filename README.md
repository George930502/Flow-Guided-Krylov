# Flow-Guided Krylov Quantum Diagonalization

A quantum chemistry pipeline for computing molecular ground-state energies by combining **Normalizing Flow-Assisted Neural Quantum States (NF-NQS)** with **Krylov subspace diagonalization**. The core idea: use a normalizing flow to discover high-probability Slater determinants, then diagonalize the Hamiltonian projected onto that basis to systematically converge toward FCI-level accuracy.

Three subspace solvers are available:
- **Classical SKQD** -- Exact matrix exponential in particle-conserving subspace
- **Quantum SKQD** -- Trotterized time evolution via CUDA-Q circuits or classical state-vector simulation
- **SQD** -- IBM's sampling-based batch diagonalization with S-CORE config recovery

The quantum SKQD implementation follows the paper by Yu et al. (arXiv:2501.09702) with paper-compliant hyperparameters: `dt = pi / spectral_range` (Epperly Theorem 3.1), Krylov dimension d=15, single second-order Suzuki-Trotter step per evolution, and 10^5 shots per Krylov state.

---

## Table of Contents

- [Methodology](#methodology)
- [Quantum vs Classical SKQD (Ablation Study)](#quantum-vs-classical-skqd-ablation-study)
- [The 7 Ablation Pipelines](#the-7-ablation-pipelines)
- [Available Molecular Systems](#available-molecular-systems)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Running Examples](#running-examples)
- [GPU Acceleration](#gpu-acceleration)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [References](#references)
- [License](#license)

---

## Methodology

### 3-Stage Pipeline

```
Stage 1: Basis Generation
  ├── Direct-CI: deterministic HF + singles + doubles (Slater-Condon rules)
  ├── NF-NQS: particle-conserving flow (GumbelTopK) co-trained with NQS
  └── Hybrid: NF-learned basis merged with Direct-CI essentials
        |
        v
Stage 2: Diversity-Aware Selection
  ├── Bucket configs by excitation rank (0=HF, 1=singles, 2=doubles, ...)
  ├── DPP-greedy selection for diversity within each bucket
  └── Essential configs (HF + singles + doubles) always preserved
        |
        v
Stage 3: Subspace Diagonalization
  ├── Classical SKQD: exact e^{-iHdt} in particle-conserving subspace
  ├── Quantum SKQD: Trotterized evolution (CUDA-Q or state-vector)
  └── SQD: noise injection -> S-CORE config recovery -> batch diag
```

**Stage 1 -- Basis Generation** supports three strategies. *Direct-CI* deterministically enumerates Hartree-Fock plus all single and double excitations using Slater-Condon rules. *NF-NQS* trains a particle-conserving normalizing flow (using the Gumbel-Top-K straight-through estimator to enforce exact electron count) co-trained with a neural quantum state to learn the ground-state distribution. *Hybrid* merges NF-discovered configurations with Direct-CI essentials for robustness.

**Stage 2 -- Diversity-Aware Selection** applies excitation-rank stratification with a physics-informed budget:

| Excitation Rank | Budget | Description |
|-----------------|--------|-------------|
| 0 | 5% | HF and near-HF configurations |
| 1 | 25% | Single excitations |
| 2 | 50% | Double excitations (dominant in ground state) |
| 3 | 15% | Triple excitations |
| 4+ | 5% | Higher excitations |

Within each bucket, DPP-greedy selection maximizes `weight * hamming_distance` to ensure diversity (`min_hamming_distance=2`).

**Stage 3 -- Subspace Diagonalization** offers three solvers. *Classical SKQD* constructs a Krylov subspace via exact matrix exponential `|psi_k> = e^{-ikH*dt}|psi_0>` in the particle-conserving subspace, expanding the basis at each step. *Quantum SKQD* uses Trotterized time evolution (2nd-order Suzuki-Trotter) in the full 2^n Hilbert space, either via CUDA-Q quantum circuits (Path A) or classical state-vector simulation (Path B). *SQD* (from IBM's "Chemistry Beyond Exact Diagonalization") injects depolarizing noise, applies S-CORE configuration recovery, performs batch diagonalization with self-consistent orbital occupancy iteration, and extrapolates to zero variance.

---

## Quantum vs Classical SKQD (Ablation Study)

The `examples/quantum_vs_classical_krylov.py` script provides a controlled 3-way comparison where **only the time evolution method changes** while all other variables remain identical:

| Path | Time Evolution | Hilbert Space | Description |
|------|---------------|---------------|-------------|
| **Path C** (Classical) | Exact Lanczos `e^{-iHdt}` | Full 2^n | Gold standard, no Trotter error |
| **Path B** (Trotterized) | Classical state-vector Trotter | Full 2^n | Isolates Trotter error |
| **Path A** (CUDA-Q) | Quantum circuit Trotter | Full 2^n | Full quantum simulation |

All three paths use identical hyperparameters via `QuantumCircuitSKQD`:
- **Time step**: `dt = pi / spectral_range` (Epperly Theorem 3.1)
- **Krylov dimension**: d=15 (paper default)
- **Trotter steps**: 1 (single S_2(dt) per evolution, paper's `[S_2(dt)]^k`)
- **Shots**: 10^5 per Krylov state
- **Initial state**: Hartree-Fock
- **Basis accumulation**: Cumulative union across all Krylov states

### Benchmark Results (all 7 systems, STO-3G)

| System | Qubits | Path C (mHa) | Path B (mHa) | Path A (mHa) |
|--------|--------|-------------|-------------|-------------|
| H2     | 4      | 0.0000      | 0.0000      | 0.0000      |
| LiH    | 12     | 0.0074      | 0.0119      | 0.0099      |
| H2O    | 14     | 0.0626      | 0.0478      | 0.0437      |
| BeH2   | 14     | 0.0198      | 0.0160      | 0.0198      |
| NH3    | 16     | 0.3234      | 0.4253      | 0.3599      |
| CH4    | 18     | 0.5451      | skip        | 0.4986      |
| N2     | 20     | 1.1427      | skip        | 1.1003      |

All systems achieve chemical accuracy (< 1.594 mHa). Path B skips systems >= 18 qubits due to memory limits (phase table is n_terms x 2^n_qubits).

---

## The 7 Ablation Pipelines

The pipeline supports 7 experimental configurations spanning two ablation axes:

| # | Pipeline | Basis Strategy | Solver | Workflow |
|---|----------|---------------|--------|----------|
| 1 | CudaQ SKQD | HF only | SKQD | HF reference state -> Krylov expansion |
| 2 | Pure SKQD | Direct-CI (HF+S+D) | SKQD | HF + singles + doubles -> Krylov expansion |
| 3 | Pure SQD | Direct-CI (HF+S+D) | SQD | HF + singles + doubles -> noise -> S-CORE -> batch diag |
| 4 | NF-Trained SKQD | NF + Direct-CI | SKQD | NF training + HF+S+D -> Krylov |
| 5 | NF-Trained SQD | NF + Direct-CI | SQD | NF training + HF+S+D -> noise -> S-CORE |
| 6 | NF-only SKQD | NF only | SKQD | NF basis -> Krylov (no essential config injection) |
| 7 | NF-only SQD | NF only | SQD | NF basis -> noise -> S-CORE (no essential config injection) |

**Ablation Axis 1 -- HF-only vs Direct-CI:** Does pre-injecting singles and doubles help, or can Krylov discover them on its own? Compare CudaQ SKQD (#1) against Pure SKQD (#2).

**Ablation Axis 2 -- NF+Direct-CI vs NF-only:** Does essential config injection help when the NF provides a learned basis? Compare pipelines #4/#5 against #6/#7.

---

## Available Molecular Systems

All factory functions use the **STO-3G** basis set. Reference energies are computed from FCI at runtime via PySCF.

| Factory Function | Molecule | Electrons | Orbitals | Qubits | Configs |
|-----------------|----------|-----------|----------|--------|---------|
| `create_h2_hamiltonian(bond_length=0.74)` | H2 | 2 | 2 | 4 | 4 |
| `create_lih_hamiltonian(bond_length=1.6)` | LiH | 4 | 6 | 12 | 225 |
| `create_h2o_hamiltonian()` | H2O | 10 | 7 | 14 | 441 |
| `create_beh2_hamiltonian()` | BeH2 | 6 | 7 | 14 | 1,225 |
| `create_nh3_hamiltonian()` | NH3 | 10 | 8 | 16 | 3,136 |
| `create_ch4_hamiltonian()` | CH4 | 10 | 9 | 18 | 15,876 |
| `create_n2_hamiltonian(bond_length=1.10)` | N2 | 14 | 10 | 20 | 14,400 |

All small systems (up to 18 qubits) achieve exact FCI energy (0.0000 mHa error) in Direct-CI mode.

---

## Quick Start

```python
from src.pipeline import FlowGuidedKrylovPipeline, PipelineConfig
from src.hamiltonians.molecular import create_lih_hamiltonian

# Create molecular Hamiltonian
H = create_lih_hamiltonian(bond_length=1.6)

# Run pipeline with SKQD solver in Direct-CI mode
config = PipelineConfig(subspace_mode="skqd", skip_nf_training=True)
pipeline = FlowGuidedKrylovPipeline(H, config=config)
results = pipeline.run()

# Check results against FCI
E_exact = H.fci_energy()
print(f"Energy: {results['combined_energy']:.6f} Ha")
print(f"Error:  {abs(results['combined_energy'] - E_exact) * 1000:.4f} mHa")
```

To use the SQD solver instead:

```python
config = PipelineConfig(subspace_mode="sqd", skip_nf_training=True)
```

To enable NF-NQS training (recommended for systems with >20 qubits):

```python
config = PipelineConfig(subspace_mode="skqd", skip_nf_training=False)
```

To run the 3-way quantum vs classical SKQD comparison:

```python
from examples.quantum_vs_classical_krylov import run_comparison

results = run_comparison(systems=["h2", "lih"], max_krylov_dim=15, num_trotter_steps=1)
```

---

## Installation

### With uv (recommended)

```bash
git clone https://github.com/George930502/Flow-Guided-Krylov.git
cd Flow-Guided-Krylov

# Install core dependencies
uv sync

# Include GPU support (CuPy)
uv sync --extra cuda

# Include dev tools (pytest, black, ruff, mypy)
uv sync --extra dev
```

### With pip

```bash
git clone https://github.com/George930502/Flow-Guided-Krylov.git
cd Flow-Guided-Krylov

pip install -e .

# For GPU support
pip install cupy-cuda12x>=12.0.0
```

### With Docker (GPU)

```bash
docker-compose build
docker-compose run --rm flow-krylov-gpu
```

---

## Running Examples

```bash
# Validate all small systems (H2, LiH, H2O, BeH2, NH3, CH4)
python examples/validate_small_systems.py

# SKQD vs SQD side-by-side comparison
python examples/subspace_comparison.py

# 3-way quantum vs classical SKQD comparison (ablation study)
python examples/quantum_vs_classical_krylov.py --systems h2 lih h2o beh2
python examples/quantum_vs_classical_krylov.py --systems h2 lih --krylov-dim 15 --trotter-steps 1

# Full 7-experiment ablation study
python examples/nf_trained_comparison.py
python examples/nf_trained_comparison.py --systems h2 lih h2o

# Moderate system benchmarks (20-30 qubits)
python examples/moderate_system_benchmark.py

# Docker (GPU)
docker-compose run --rm flow-krylov-gpu python examples/quantum_vs_classical_krylov.py --systems h2 lih
docker-compose run --rm flow-krylov-gpu python examples/subspace_comparison.py
docker-compose run --rm flow-krylov-gpu python examples/nf_trained_comparison.py
```

---

## GPU Acceleration

The pipeline is designed for end-to-end GPU execution, from NF training through final diagonalization.

- **Particle-conserving subspace**: operates in the electron-number subspace (10-100x smaller than the full Hilbert space)
- **GPU Lanczos matrix exponential** (`gpu_expm_multiply`): Krylov time evolution without materializing the full matrix exponential
- **GPU eigensolvers** (`gpu_eigsh`): dense `torch.linalg.eigh` for n <= 10K, CuPy sparse `eigsh` for larger subspaces
- **DLPack zero-copy**: CuPy interop via `torch.from_dlpack()` / `cp.from_dlpack()` avoids GPU-to-CPU-to-GPU round-trips
- **ConnectionCache**: GPU integer-encoded LRU cache prevents redundant Hamiltonian connection recomputation
- **Vectorized Slater-Condon rules**: batch matrix element evaluation on GPU via `get_connections_vectorized_batch()`
- **Searchsorted-based integer encoding**: O(n log n) configuration matching replaces O(n^2) Python dict lookups
- **Batched config recovery**: SQD S-CORE uses `torch.multinomial` for vectorized orbital flipping
- **Seeded GPU sampling**: deterministic `torch.Generator` with per-Krylov-step seeds for reproducibility
- **Vectorized Lanczos projection**: single `coeffs @ V_matrix` matmul replaces Python loop
- **Cached arange tensors**: avoids redundant `torch.arange` allocation in Trotter evolution
- **Chunked S^2 matrix**: GPU-resident chunked computation bounds memory while avoiding CPU fallback
- **Vectorized DPP selection**: greedy diversity selection inner loop fully on GPU
- **TF32 matmul acceleration**: automatically enabled for CUDA matmul and cuDNN operations

All GPU features degrade gracefully: CuPy falls back to SciPy, CUDA-Q falls back to classical NumPy sampling.

---

## Architecture

```
src/
├── pipeline.py                        # PipelineConfig + FlowGuidedKrylovPipeline orchestrator
├── flows/
│   ├── particle_conserving_flow.py    # NF with exact electron count (GumbelTopK)
│   └── physics_guided_training.py     # Co-trains NF + NQS
├── nqs/
│   ├── base.py                        # NeuralQuantumState ABC
│   └── dense.py                       # DenseNQS, SignedDenseNQS
├── hamiltonians/
│   ├── base.py                        # Hamiltonian ABC (diagonal_element, get_connections)
│   ├── molecular.py                   # MolecularHamiltonian (PySCF integrals, Slater-Condon)
│   └── pauli_mapping.py              # Jordan-Wigner transformation (molecular -> Pauli strings)
├── krylov/
│   ├── skqd.py                        # Classical SKQD solver (exact matrix exponential)
│   ├── quantum_skqd.py               # Quantum SKQD (CUDA-Q + classical Trotter)
│   ├── sqd.py                         # SQD solver (IBM paper algorithm)
│   ├── basis_sampler.py              # CUDA-Q / classical Krylov sampling
│   └── spectral_utils.py            # Shared compute_optimal_dt (pi / spectral_range)
├── postprocessing/
│   ├── diversity_selection.py         # DPP-greedy diversity selection
│   ├── projected_hamiltonian.py       # H_ij = <x_i|H|x_j> construction
│   ├── eigensolver.py                 # Davidson / sparse eigsh / adaptive selection
│   └── utils.py
└── utils/
    ├── gpu_linalg.py                  # gpu_eigh, gpu_eigsh, gpu_expm_multiply
    └── connection_cache.py            # GPU-accelerated Hamiltonian connection cache
```

### Key Classes

- **`PipelineConfig`** -- Master dataclass controlling the entire pipeline. Key fields: `subspace_mode` (`"skqd"` or `"sqd"`), `skip_nf_training` (enables Direct-CI mode), `auto_time_step` (computes `dt = pi / spectral_range` from Epperly Theorem 3.1). Paper-compliant defaults: `max_krylov_dim=15`, `quantum_num_trotter_steps=1`, `shots_per_krylov=100000`.
- **`FlowGuidedKrylovPipeline`** -- Orchestrator that executes all 3 stages via `.run()`.
- **`MolecularHamiltonian`** -- Second-quantized electronic Hamiltonian from PySCF one- and two-electron integrals. Implements Slater-Condon rules for matrix element evaluation.
- **`ParticleConservingFlowSampler`** -- Normalizing flow that always produces configurations with exactly `n_alpha + n_beta` electrons via Gumbel-Top-K differentiable sampling.
- **`SampleBasedKrylovDiagonalization`** -- Classical SKQD solver. Constructs Krylov subspace via exact time evolution in the particle-conserving subspace. Seeded RNG for reproducibility. `max_diag_basis_size=15000` caps dense diagonalization to prevent OOM.
- **`QuantumCircuitSKQD`** -- Quantum SKQD solver. 3 backends: CUDA-Q circuits (Path A), classical state-vector Trotter (Path B), exact Lanczos (Path C). Jordan-Wigner transformation via `pauli_mapping.py`.
- **`SQDSolver`** -- IBM SQD implementation. Two sub-modes: SQD-Clean (`noise_rate=0`, default) and SQD-Recovery (`noise_rate>0`, injects depolarizing noise then runs S-CORE configuration recovery).
- **`PhysicsGuidedFlowTrainer`** -- Co-trains NF + NQS with cross-entropy loss weighted by `|E|`.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| Package Manager | uv (hatchling build) |
| Neural Networks | PyTorch >= 2.0 |
| Molecular Integrals | PySCF >= 2.3 |
| Eigensolvers | SciPy (CPU) / CuPy (GPU) |
| Quantum Circuits | CUDA-Q (optional, graceful fallback) |
| Formatting | black (line-length 100) |
| Linting | ruff (line-length 100) |
| Type Checking | mypy |
| Testing | pytest |
| Containerization | Docker (pytorch/pytorch:2.2.0-cuda12.1) |

---

## References

1. Yu, Robledo-Moreno et al., "Sample-based Krylov Quantum Diagonalization" ([arXiv:2501.09702](https://arxiv.org/abs/2501.09702))
2. Robledo-Moreno, Motta et al., "Chemistry Beyond the Scale of Exact Diagonalization", *Science* 2024
3. "Improved Ground State Estimation via Normalising Flow-Assisted Neural Quantum States" ([arXiv:2506.12128](https://arxiv.org/abs/2506.12128))
4. NVIDIA CUDA-Q SKQD Tutorial (Heisenberg model, Trotterized evolution)
5. Epperly et al., Theorem 3.1: optimal Krylov time step `dt = pi / spectral_range`

---

## License

MIT License
