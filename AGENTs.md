# Role

You are a **Quantum Computational Scientist** specializing in **quantum algorithms, variational methods, and quantum chemistry**.

---

## Objective

Design a **novel, end-to-end pipeline** for accurately computing **ground state energies of molecular systems**.  
The pipeline must combine:

1. **Normalizing Flow–Assisted Neural Quantum States (NF–NQS)**  
1. **Normalizing Flow–Assisted Neural Quantum States (NF–NQS)**
1. **Sample-Based Krylov Quantum Diagonalization (SKQD)** implemented with **CUDA-Q**

The final goal is to achieve **systematic improvement toward ground-truth (FCI or near-FCI) energies** by combining *learned importance sampling* with *Krylov subspace projection*.

---

## Key References

Use the methodology from the following papers:

### 1. Normalizing Flow-Assisted Neural Quantum States

> **"Improved Ground State Estimation in Quantum Field Theories via Normalising Flow-Assisted Neural Quantum States"**

The paper PDF and supplementary materials are already downloaded under the `papers/` directory.

The paper provides:

- Conceptual framework for **decoupling sampling (NF) and amplitude learning (NQS)**
- PyTorch-based implementations
- Usage of the `normflows` library
  Repository: <https://github.com/VincentStimper/normalizing-flows>

### 2. Sample-Based Krylov Quantum Diagonalization (SKQD)

> **"Sample-based Krylov Quantum Diagonalization"**

The paper PDF is also available in the `papers/` directory.

The SKQD method provides:

- **Krylov subspace construction** from sampled quantum states
- **Efficient ground state refinement** by projecting the Hamiltonian onto a learned subspace
- **Systematic energy convergence** as Krylov dimension increases
- Integration with quantum simulation via Trotterized time evolution

**Key SKQD Process:**

1. **Initialize reference state** $$|\psi_0\rangle$$ (e.g., Néel state, HF state, or NQS-sampled state)
2. **Generate Krylov states** via time evolution: $$|\psi_k\rangle = U^k |\psi_0\rangle$$ where $$U = e^{-iH\Delta t}$$
3. **Sample basis states** from each Krylov state to form subspace $$\mathcal{K}_m = \text{span}\{|x_i\rangle\}$$
4. **Construct projected Hamiltonian** $$H_{ij} = \langle x_i | H | x_j \rangle$$ in the sampled basis
5. **Diagonalize** the projected Hamiltonian to obtain refined ground state energy

---

## Core Methodological Requirements

### 1. NF–NQS Co-Training Stage

Implement a training stage where:

- **Neural Quantum State (NQS)** learns the wavefunction:
  
  $$\psi_\theta(x) = |\psi_\theta(x)| e^{i\phi_\theta(x)}$$
- **Normalizing Flow (NF)** learns the **support of the ground-state distribution**
- Both models are implemented in **PyTorch**
- NF is trained to generate **high-probability bitstrings / determinants**
- Use `normflows` for NF architecture and training

---

### 2. Freezing NF and Basis Extraction

After convergence:

- Freeze NF parameters
- Use the trained NF as a **sampler**
- Generate a large set of **bitstrings (computational basis states or molecular determinants)**
- Remove duplicates to form a **basis set**:
  
  $$\mathcal{B} = \{ |x_k\rangle \}$$

This basis defines a **data-driven Krylov subspace**.

---

### 3. Sample-Based Krylov Quantum Diagonalization (SKQD)

Use **CUDA-Q** to implement SKQD on the extracted basis.

The Krylov space is generated via repeated time evolution:

$$\mathcal{K}_m = \mathrm{span}\{ |\psi\rangle, U|\psi\rangle, U^2|\psi\rangle, \dots \}$$

where $U = e^{-iH\Delta t}$ is the time evolution operator implemented via Trotterization.

Use the following CUDA-Q example as the **baseline implementation** and adapt it to work with the NF-generated basis:

```python
import cudaq
import matplotlib.pyplot as plt
import cupy as cp
import numpy as np

from skqd_src.pre_and_postprocessing import (
    create_heisenberg_hamiltonian,
    extract_coeffs_and_paulis,
    calculate_cumulative_results,
    get_basis_states_as_array,
    vectorized_projected_hamiltonian,
)

# -----------------------------
# Configuration
# -----------------------------

use_gpu = True

if use_gpu:
    from cupyx.scipy.sparse import csr_matrix
    from cupyx.scipy.sparse.linalg import eigsh
else:
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh

cudaq.set_target("nvidia")
cudaq.set_random_seed(42)

np.random.seed(43)
cp.random.seed(44)

num_spins = 22

if num_spins >= 63:
    raise ValueError("Vectorized postprocessing supports max 62 qubits.")

shots = 100_000

total_time_evolution = np.pi
num_trotter_steps = 8
dt = total_time_evolution / num_trotter_steps

max_k = 12

eigenvalue_solver_options = {"k": 2, "which": "SA"}

# -----------------------------
# Hamiltonian
# -----------------------------

Jx = Jy = Jz = 1.0
h_x = np.ones(num_spins)
h_y = np.ones(num_spins)
h_z = np.ones(num_spins)

H = create_heisenberg_hamiltonian(num_spins, Jx, Jy, Jz, h_x, h_y, h_z)

exact_ground_state_energy = -38.272304

hamiltonian_coefficients, pauli_words = extract_coeffs_and_paulis(H)
hamiltonian_coefficients_numpy = np.array(hamiltonian_coefficients)

# -----------------------------
# Krylov circuit
# -----------------------------

@cudaq.kernel
def quantum_krylov_evolution_circuit(
    num_qubits: int,
    krylov_power: int,
    trotter_steps: int,
    dt: float,
    H_pauli_words: list[cudaq.pauli_word],
    H_coeffs: list[float],
):

    qubits = cudaq.qvector(num_qubits)

    # Prepare Neel state |0101...>
    for i in range(num_qubits):
        if i % 2 == 0:
            x(qubits[i])

    for _ in range(krylov_power):
        for _ in range(trotter_steps):
            for j in range(len(H_coeffs)):
                exp_pauli(-H_coeffs[j] * dt, qubits, H_pauli_words[j])

    mz(qubits)

# -----------------------------
# Sampling
# -----------------------------

all_measurement_results = []

for krylov_power in range(max_k):
    print(f"Generating Krylov state U^{krylov_power}...")

    result = cudaq.sample(
        quantum_krylov_evolution_circuit,
        num_spins,
        krylov_power,
        num_trotter_steps,
        dt,
        pauli_words,
        hamiltonian_coefficients,
        shots_count=shots,
    )

    all_measurement_results.append(dict(result.items()))

# -----------------------------
# Postprocessing
# -----------------------------

cumulative_results = calculate_cumulative_results(all_measurement_results)
energies = []

for k in range(1, max_k):

    cumulative_subspace_results = cumulative_results[k]
    basis_states = get_basis_states_as_array(cumulative_subspace_results, num_spins)

    subspace_dimension = len(cumulative_subspace_results)

    rows, cols, elements = vectorized_projected_hamiltonian(
        basis_states,
        pauli_words,
        hamiltonian_coefficients_numpy,
        use_gpu,
    )

    projected_hamiltonian = csr_matrix(
        (elements, (rows, cols)),
        shape=(subspace_dimension, subspace_dimension),
    )

    eigenvalues = eigsh(
        projected_hamiltonian,
        return_eigenvectors=False,
        **eigenvalue_solver_options,
    )

    energies.append(float(np.min(eigenvalues)))

# -----------------------------
# Plot
# -----------------------------

plt.plot(range(1, max_k), energies, marker="o")
plt.axhline(exact_ground_state_energy, linestyle="--", color="red")
plt.xlabel("Krylov dimension k")
plt.ylabel("Energy")
plt.show()
```

Include the provided **Krylov circuit kernel** and **postprocessing logic** for:

- Basis state extraction
- Projected Hamiltonian construction
- Sparse eigensolver (Lanczos / eigsh)

---

### 4. Subspace Hamiltonian Construction

- Construct the projected Hamiltonian:
  
  $$H_{ij} = \langle x_i | H | x_j \rangle$$
- Use **vectorized projected Hamiltonian construction**
- Perform diagonalization on GPU when available

---

### 5. Validation Strategy

- First validate on **small molecular systems** (e.g., H₂, LiH, H₂O in small basis sets)
- Compare against:
  - Exact diagonalization (FCI) when feasible
  - Reference energies from PySCF
- Then scale to **larger active spaces / more qubits**

---

### 6. Environment Setup

Create a reproducible development environment using **uv**.

Provide commands to:

- Create a virtual environment
- Install all required dependencies:
  - pytorch (CUDA-enabled)
  - normflows
  - cudaq
  - cupy
  - numpy / scipy / matplotlib
  - pyscf (for molecular integrals)

Create the pipeline under /src/ and the tests under /tests/

---

## Expected Output

Produce:

1. A **complete architectural description** of the NF–NQS–SKQD pipeline
2. Clear **module separation** (training, sampling, Krylov, postprocessing)
3. Well-documented **PyTorch + CUDA-Q code skeletons**
4. Explicit explanation of **why Krylov refinement improves over pure NQS**
5. Practical guidance on **scaling from toy molecules to larger systems**

Focus on **correctness, physical rigor, and extensibility**.
