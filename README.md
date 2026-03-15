# HF-State SKQD: Classical vs Quantum Krylov from Hartree-Fock

Sample-Based Krylov Quantum Diagonalization (SKQD) for molecular ground-state energies, starting from the **Hartree-Fock (HF) determinant** as the initial state. Two solvers are provided:

- **Classical SKQD** -- Exact matrix exponential (`gpu_expm_multiply`) in the particle-conserving subspace
- **Quantum SKQD** -- Trotterized time evolution (2nd-order Suzuki-Trotter) in the full 2^n Hilbert space, via CUDA-Q circuits or classical state-vector simulation

Both solvers construct the same Krylov subspace:

```
|psi_k> = (e^{-i H dt})^k |HF>,    k = 0, 1, ..., d-1
```

then project the Hamiltonian onto the sampled basis and diagonalize to obtain the ground-state energy.

---

## Table of Contents

- [Algorithm](#algorithm)
- [Available Molecular Systems](#available-molecular-systems)
- [Benchmark Results](#benchmark-results)
- [Installation (Docker)](#installation-docker)
- [Running the Code](#running-the-code)
  - [Basic Usage](#basic-usage)
  - [CLI Options](#cli-options)
  - [Examples](#examples)
- [Pipeline API](#pipeline-api)
- [Classical vs Quantum SKQD](#classical-vs-quantum-skqd)
- [Paper-Compliant Parameters](#paper-compliant-parameters)
- [Architecture](#architecture)
- [GPU Acceleration](#gpu-acceleration)
- [Tech Stack](#tech-stack)
- [References](#references)
- [License](#license)

---

## Algorithm

### Krylov Subspace Construction

Starting from the Hartree-Fock determinant `|HF>`, SKQD applies time evolution to build a Krylov basis:

```
|psi_0> = |HF>
|psi_1> = e^{-i H dt} |HF>
|psi_2> = (e^{-i H dt})^2 |HF>
  ...
|psi_{d-1}> = (e^{-i H dt})^{d-1} |HF>
```

At each step k, the evolved state is measured in the computational basis (with `shots` samples). The union of all sampled configurations across all Krylov steps forms the **cumulative basis**. The Hamiltonian is then projected onto this basis via Slater-Condon rules and diagonalized.

### Classical SKQD

Operates in the **particle-conserving subspace** (dimension = C(n_orb, n_alpha) x C(n_orb, n_beta)), which is 10-100x smaller than the full 2^n Hilbert space. Time evolution uses the **exact matrix exponential** via GPU-accelerated Lanczos (`gpu_expm_multiply`), so there is **no Trotter error**.

### Quantum SKQD

Operates in the **full 2^n Hilbert space**. The molecular Hamiltonian is transformed to Pauli form via **Jordan-Wigner mapping**. Time evolution uses **2nd-order Suzuki-Trotter decomposition**:

```
S_2(dt) = prod_j exp(-i c_j P_j dt/2) * prod_j exp(-i c_j P_j dt/2)  (reversed)
```

Three execution backends (auto-selected):

| Backend | Method | When Used |
|---------|--------|-----------|
| **Path A** (CUDA-Q) | `exp_pauli` quantum circuits | CUDA-Q installed + NVIDIA GPU |
| **Path B** (State-Vector) | Classical Trotter on GPU | Small systems (< 18 qubits) |
| **Path C** (Lanczos) | Exact `gpu_expm_multiply` | Fallback / large systems |

---

## Available Molecular Systems

All factory functions use the **STO-3G** basis set. Reference energies are computed from FCI at runtime.

| Key | Molecule | Electrons | Orbitals | Qubits | Configs |
|-----|----------|-----------|----------|--------|---------|
| `h2` | H2 | 2 | 2 | 4 | 4 |
| `lih` | LiH | 4 | 6 | 12 | 225 |
| `h2o` | H2O | 10 | 7 | 14 | 441 |
| `beh2` | BeH2 | 6 | 7 | 14 | 1,225 |
| `nh3` | NH3 | 10 | 8 | 16 | 3,136 |
| `ch4` | CH4 | 10 | 9 | 18 | 15,876 |
| `n2` | N2 | 14 | 10 | 20 | 14,400 |

---

## Benchmark Results

Tested via Docker on NVIDIA GPU (CUDA 12.1), with paper-compliant parameters: d=15, 10^5 shots, dt = pi/spectral_range.

All results verified against FCI reference energies computed at runtime.

| System | Qubits | FCI Energy (Ha) | Classical Error (mHa) | Classical Time | Quantum Error (mHa) | Quantum Time | Status |
|--------|--------|-----------------|----------------------|----------------|---------------------|--------------|--------|
| H2     | 4      | -1.13728383     | 0.0000               | 0.4s           | 0.0000              | 0.7s         | PASS   |
| LiH    | 12     | -7.88232438     | 0.0049               | 1.1s           | 0.0022              | 5.3s         | PASS   |
| H2O    | 14     | -75.01315470    | 0.0380               | 3.1s           | 0.0085              | 9.3s         | PASS   |
| BeH2   | 14     | -15.59511756    | 0.0147               | 9.3s           | 0.0104              | 6.0s         | PASS   |

**Chemical accuracy threshold: 1.594 mHa (1 kcal/mol). All 8/8 runs PASS.**

Quantum SKQD uses CUDA-Q backend (Path A) with `exp_pauli` quantum circuits. Classical SKQD uses exact matrix exponential in the particle-conserving subspace.

---

## Installation (Docker)

Docker provides a reproducible environment with all dependencies pre-installed (PyTorch, PySCF, CuPy, CUDA-Q).

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (for GPU support)

### Setup

```bash
git clone https://github.com/George930502/Flow-Guided-Krylov.git
cd Flow-Guided-Krylov
git checkout hf-skqd-focused

# Build the Docker image
docker-compose build

# Verify the image runs correctly
docker-compose run --rm flow-krylov-gpu python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

The Docker image (`pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime`) includes:

| Dependency | Version | Purpose |
|------------|---------|---------|
| PyTorch | 2.2.0 + CUDA 12.1 | Neural networks + GPU linear algebra |
| PySCF | >= 2.3 | Molecular integrals (SCF, FCI) |
| CuPy | 13.x | GPU sparse eigensolvers, DLPack zero-copy |
| CUDA-Q | latest | Quantum circuit simulation (`exp_pauli`) |
| SciPy | >= 1.10 | CPU fallback eigensolvers |
| NumPy | < 2 | Compatible with PyTorch 2.2.0 |

---

## Running the Code

All commands use `docker-compose run --rm flow-krylov-gpu` as the prefix.

### Basic Usage

```bash
# Run both classical and quantum SKQD on all 7 systems (H2 through N2)
docker-compose run --rm flow-krylov-gpu python examples/hf_skqd_comparison.py

# Run on specific systems
docker-compose run --rm flow-krylov-gpu python examples/hf_skqd_comparison.py --systems h2 lih h2o beh2

# Classical SKQD only (faster, no Trotter overhead)
docker-compose run --rm flow-krylov-gpu python examples/hf_skqd_comparison.py --mode classical

# Quantum SKQD only (Trotterized, uses CUDA-Q)
docker-compose run --rm flow-krylov-gpu python examples/hf_skqd_comparison.py --mode quantum
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--systems` | all 7 | Space-separated list: `h2 lih h2o beh2 nh3 ch4 n2` |
| `--mode` | `both` | `classical`, `quantum`, or `both` |
| `--krylov-dim` | `15` | Maximum Krylov subspace dimension (paper: d=15) |
| `--trotter-steps` | `1` | Trotter steps per evolution (quantum only, paper: 1) |
| `--shots` | `100000` | Measurement shots per Krylov state (paper: 10^5) |

### Examples

```bash
# Custom Krylov dimension
docker-compose run --rm flow-krylov-gpu python examples/hf_skqd_comparison.py --krylov-dim 10

# Custom shots
docker-compose run --rm flow-krylov-gpu python examples/hf_skqd_comparison.py --shots 50000

# Combine options: specific systems, quantum only, custom dim
docker-compose run --rm flow-krylov-gpu python examples/hf_skqd_comparison.py --systems h2 lih --mode quantum --krylov-dim 20

# CPU-only (no GPU required)
docker-compose run --rm flow-krylov-cpu python examples/hf_skqd_comparison.py --mode classical

# Interactive shell inside Docker
docker-compose run --rm shell
# Then inside the container:
python examples/hf_skqd_comparison.py --systems h2 lih
```

---

## Pipeline API

### Quick Start (inside Docker container)

```python
from hamiltonians.molecular import create_lih_hamiltonian
from krylov.skqd import SampleBasedKrylovDiagonalization, SKQDConfig
from krylov.spectral_utils import compute_optimal_dt

# 1. Build Hamiltonian
H = create_lih_hamiltonian(bond_length=1.6)

# 2. Compute optimal time step (Epperly Theorem 3.1)
dt, spectral_range = compute_optimal_dt(H)

# 3. Run classical SKQD from HF state
config = SKQDConfig(max_krylov_dim=15, time_step=dt)
solver = SampleBasedKrylovDiagonalization(H, config=config, initial_state=H.get_hf_state())
results = solver.run()

# 4. Check result
fci = H.fci_energy()
best = min(results["energies"])
print(f"SKQD energy: {best:.8f} Ha")
print(f"FCI energy:  {fci:.8f} Ha")
print(f"Error:       {abs(best - fci) * 1000:.4f} mHa")
```

### Using the Quantum SKQD Solver Directly

```python
from hamiltonians.molecular import create_h2o_hamiltonian
from krylov.quantum_skqd import QuantumCircuitSKQD, QuantumSKQDConfig
from krylov.spectral_utils import compute_optimal_dt

H = create_h2o_hamiltonian()
dt, _ = compute_optimal_dt(H)

config = QuantumSKQDConfig(
    max_krylov_dim=15,
    total_evolution_time=dt,
    num_trotter_steps=1,
    shots=100_000,
    initial_state="hf",
    backend="auto",  # auto-selects best available backend
)

solver = QuantumCircuitSKQD.from_molecular_hamiltonian(H, config=config)
results = solver.run()

print(f"Best energy: {results['best_energy']:.8f} Ha")
print(f"Backend:     {results['backend']}")
```

### Using the Full Pipeline

```python
from pipeline import FlowGuidedKrylovPipeline, PipelineConfig
from hamiltonians.molecular import create_lih_hamiltonian

H = create_lih_hamiltonian(bond_length=1.6)

# Classical SKQD via pipeline (Direct-CI mode skips NF training)
config = PipelineConfig(subspace_mode="skqd", skip_nf_training=True)
pipeline = FlowGuidedKrylovPipeline(H, config=config)
results = pipeline.run()

print(f"Energy: {results['combined_energy']:.8f} Ha")
print(f"FCI:    {H.fci_energy():.8f} Ha")
```

---

## Classical vs Quantum SKQD

| Aspect | Classical SKQD | Quantum SKQD |
|--------|---------------|--------------|
| **Hilbert space** | Particle-conserving subspace | Full 2^n |
| **Time evolution** | Exact matrix exponential (Lanczos) | 2nd-order Suzuki-Trotter |
| **Trotter error** | None | O(dt^3) per step |
| **Initial state** | HF determinant | HF determinant |
| **Implementation** | `src/krylov/skqd.py` | `src/krylov/quantum_skqd.py` |
| **GPU backend** | `gpu_expm_multiply` (PyTorch/CuPy) | CUDA-Q / CuPy / PyTorch |
| **Memory scaling** | O(subspace_dim^2) | O(2^n) for state vector |
| **Advantage** | Exact, fast for small systems | Mimics quantum hardware |

---

## Paper-Compliant Parameters

All defaults follow Yu et al. (arXiv:2501.09702):

| Parameter | Value | Source |
|-----------|-------|--------|
| Time step dt | pi / spectral_range | Epperly Theorem 3.1 |
| Krylov dimension d | 15 | Paper Fig. 1 (Ising simulation) |
| Trotter decomposition | 2nd-order Suzuki-Trotter | Paper Section IV |
| Trotter steps per evolution | 1 | Paper: `[S_2(dt)]^k` |
| Shots per Krylov state | 100,000 | Paper Section V |
| Basis accumulation | Cumulative union | Paper algorithm |

---

## Architecture

```
src/
├── pipeline.py                        # PipelineConfig + FlowGuidedKrylovPipeline
├── hamiltonians/
│   ├── base.py                        # Hamiltonian ABC
│   ├── molecular.py                   # MolecularHamiltonian (PySCF, Slater-Condon)
│   └── pauli_mapping.py               # Jordan-Wigner (molecular -> Pauli strings)
├── krylov/
│   ├── skqd.py                        # Classical SKQD (exact exp in subspace)
│   ├── quantum_skqd.py                # Quantum SKQD (Trotter in full 2^n)
│   ├── sqd.py                         # SQD solver (IBM paper, batch diag)
│   ├── basis_sampler.py               # CUDA-Q / classical Krylov sampling
│   └── spectral_utils.py              # compute_optimal_dt (pi / spectral_range)
├── flows/
│   ├── particle_conserving_flow.py    # NF with exact electron count (GumbelTopK)
│   └── physics_guided_training.py     # Co-trains NF + NQS
├── nqs/
│   ├── base.py                        # NeuralQuantumState ABC
│   └── dense.py                       # DenseNQS, SignedDenseNQS
├── postprocessing/
│   ├── diversity_selection.py         # DPP-greedy diversity selection
│   ├── projected_hamiltonian.py       # H_ij = <x_i|H|x_j> construction
│   ├── eigensolver.py                 # Davidson / sparse eigsh / adaptive
│   └── utils.py
└── utils/
    ├── gpu_linalg.py                  # gpu_eigh, gpu_eigsh, gpu_expm_multiply
    ├── gpu_fci.py                     # GPU FCI via gpu4pyscf (optional)
    └── connection_cache.py            # GPU-accelerated Hamiltonian connection cache

examples/
├── hf_skqd_comparison.py             # ** Main script: Classical vs Quantum SKQD from HF **
├── quantum_vs_classical_krylov.py    # 3-way comparison (Paths A/B/C)
├── subspace_comparison.py            # SKQD vs SQD side-by-side
├── validate_small_systems.py         # Small system validation
└── moderate_system_benchmark.py      # 20-30 qubit systems
```

### Key Classes

- **`SampleBasedKrylovDiagonalization`** (`src/krylov/skqd.py`) -- Classical SKQD solver. Builds Krylov subspace via exact time evolution in the particle-conserving subspace. Starts from HF state by default.
- **`QuantumCircuitSKQD`** (`src/krylov/quantum_skqd.py`) -- Quantum SKQD solver. Jordan-Wigner + Trotterized evolution. Three backends: CUDA-Q (Path A), state-vector (Path B), Lanczos (Path C).
- **`MolecularHamiltonian`** (`src/hamiltonians/molecular.py`) -- Second-quantized Hamiltonian from PySCF. Factory functions: `create_h2_hamiltonian()`, ..., `create_n2_hamiltonian()`.
- **`SKQDConfig`** / **`QuantumSKQDConfig`** -- Configuration dataclasses with paper-compliant defaults.
- **`compute_optimal_dt()`** (`src/krylov/spectral_utils.py`) -- Computes `dt = pi / spectral_range` (Epperly Theorem 3.1).

---

## GPU Acceleration

The pipeline is designed for end-to-end GPU execution:

- **GPU Lanczos matrix exponential** (`gpu_expm_multiply`): Krylov time evolution without materializing the full matrix
- **GPU eigensolvers** (`gpu_eigsh`): dense `torch.linalg.eigh` for n <= 10K, CuPy sparse for larger
- **DLPack zero-copy**: CuPy interop via `cp.from_dlpack()` avoids GPU-CPU round-trips
- **Fused CUDA kernel**: CuPy kernel for Pauli matrix-vector products (quantum SKQD)
- **ConnectionCache**: GPU integer-encoded LRU cache for Hamiltonian connections
- **Vectorized Slater-Condon rules**: batch matrix element evaluation on GPU

All GPU features degrade gracefully to CPU (CuPy -> SciPy, CUDA-Q -> classical NumPy).

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| Neural Networks | PyTorch 2.2.0 |
| Molecular Integrals | PySCF >= 2.3 |
| Eigensolvers | SciPy (CPU) / CuPy (GPU) |
| Quantum Circuits | CUDA-Q (optional, graceful fallback) |
| Containerization | Docker (pytorch/pytorch:2.2.0-cuda12.1) |

---

## References

1. Yu, Robledo-Moreno et al., "Sample-based Krylov Quantum Diagonalization" ([arXiv:2501.09702](https://arxiv.org/abs/2501.09702))
2. Robledo-Moreno, Motta et al., "Chemistry Beyond the Scale of Exact Diagonalization", *Science* 2024
3. Epperly et al., Theorem 3.1: optimal Krylov time step `dt = pi / spectral_range`
4. NVIDIA CUDA-Q SKQD Tutorial (Heisenberg model, Trotterized evolution)

---

## License

MIT License
