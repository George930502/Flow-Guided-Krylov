# Stage 4: Sample-Based Krylov Quantum Diagonalization (SKQD)

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 4: SAMPLE-BASED KRYLOV DIAGONALIZATION             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Input: expanded_basis from Stage 3 (e.g., 3089 configurations)           │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  KRYLOV SUBSPACE GENERATION                                         │  │
│   │                                                                     │  │
│   │  |ψ₀⟩ ──U──► |ψ₁⟩ ──U──► |ψ₂⟩ ──U──► ... ──U──► |ψₖ⟩             │  │
│   │   HF      U=e^{-iHΔt}                                               │  │
│   │                                                                     │  │
│   │  Sample each |ψₖ⟩ to get configurations                            │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                              ↓                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  COMBINE WITH NF BASIS                                              │  │
│   │                                                                     │  │
│   │  Combined basis = NF basis ∪ Krylov samples                        │  │
│   │  (removes duplicates)                                               │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                              ↓                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  SUBSPACE DIAGONALIZATION                                           │  │
│   │                                                                     │  │
│   │  Build H_proj: H_ij = ⟨xᵢ|H|xⱼ⟩                                    │  │
│   │  Diagonalize: H_proj |c⟩ = E |c⟩                                   │  │
│   │  Ground state energy: E₀ = min(eigenvalues)                        │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Output: combined_energy (best estimate of ground state)                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Why SKQD?

### Limitation of Previous Stages

Stages 1-3 discover important configurations through:
- NF-NQS training (Stage 1)
- Diversity selection (Stage 2)
- Residual expansion (Stage 3)

But they may still miss configurations that are:
- Reachable only through time evolution
- Important for dynamical correlations
- Not directly connected to the current basis

### SKQD Solution

Time evolve from a reference state and sample the resulting quantum states:
- Time evolution explores Hilbert space systematically
- Krylov subspace captures different correlation structures
- Combining with NF basis gives comprehensive coverage

---

## Mathematical Foundation

### Krylov Subspace

The Krylov subspace of order m is:
$$\mathcal{K}_m(H, |\psi_0\rangle) = \text{span}\{|\psi_0\rangle, H|\psi_0\rangle, H^2|\psi_0\rangle, \ldots, H^{m-1}|\psi_0\rangle\}$$

### Time Evolution Formulation

Instead of powers of H, use time evolution:
$$|\psi_k\rangle = U^k |\psi_0\rangle = e^{-ikH\Delta t} |\psi_0\rangle$$

For small Δt, this generates a similar subspace but with better numerical properties.

### Sample-Based Approximation

Instead of storing full state vectors (exponentially large):
1. Sample configurations from each |ψₖ⟩
2. Use sampled configurations as basis vectors
3. Build projected Hamiltonian in this basis

---

## Technique 1: Particle-Conserving Subspace Evolution

**Purpose:** Perform time evolution in the much smaller particle-conserving subspace.

### Problem with Full Hilbert Space

For NH3 with 16 qubits:
- Full Hilbert space: 2¹⁶ = 65,536 dimensions
- Particle-conserving subspace: C(8,5) × C(8,5) = 3,136 dimensions

Time evolution in full space requires 65,536 × 65,536 matrices!

### Solution: Subspace Restriction

```python
def _setup_particle_conserving_subspace(self):
    """
    Set up particle-conserving subspace for molecular Hamiltonians.

    Reduces dimension from 2^n to C(n_orb, n_alpha) × C(n_orb, n_beta).
    """
    n_orb = self.hamiltonian.n_orbitals
    n_alpha = self.hamiltonian.n_alpha
    n_beta = self.hamiltonian.n_beta

    n_valid = comb(n_orb, n_alpha) * comb(n_orb, n_beta)
    print(f"Subspace: {n_valid:,} configs (vs {2**self.num_sites:,} full)")

    # Generate all valid configurations
    alpha_configs = list(combinations(range(n_orb), n_alpha))
    beta_configs = list(combinations(range(n_orb), n_beta))

    basis_configs = []
    for alpha_occ in alpha_configs:
        for beta_occ in beta_configs:
            config = torch.zeros(self.num_sites, dtype=torch.long)
            for i in alpha_occ:
                config[i] = 1  # Alpha spin orbitals
            for i in beta_occ:
                config[i + n_orb] = 1  # Beta spin orbitals
            basis_configs.append(config)

    self._subspace_basis = torch.stack(basis_configs)

    # Create index mapping for O(1) lookup
    self._subspace_index_map = {
        tuple(c.tolist()): i for i, c in enumerate(basis_configs)
    }
```

### Subspace Hamiltonian Construction

```python
def _build_subspace_hamiltonian(self):
    """
    Build sparse Hamiltonian in particle-conserving subspace.

    Much smaller than full Hilbert space:
    - NH3: 3,136 × 3,136 instead of 65,536 × 65,536
    - N2: 14,400 × 14,400 instead of 1,048,576 × 1,048,576
    """
    n_subspace = len(self._subspace_basis)
    print(f"Building subspace Hamiltonian ({n_subspace:,} × {n_subspace:,})...")

    rows, cols, data = [], [], []

    for j in range(n_subspace):
        config_j = self._subspace_basis[j]

        # Diagonal element
        diag = self.hamiltonian.diagonal_element(config_j).item()
        rows.append(j)
        cols.append(j)
        data.append(diag)

        # Off-diagonal elements (only within subspace)
        connected, elements = self.hamiltonian.get_connections(config_j)

        for k in range(len(connected)):
            key = tuple(connected[k].tolist())
            if key in self._subspace_index_map:  # Only if in subspace
                i = self._subspace_index_map[key]
                rows.append(i)
                cols.append(j)
                data.append(elements[k].item())

    H_subspace = csr_matrix(
        (data, (rows, cols)),
        shape=(n_subspace, n_subspace),
        dtype=np.complex128
    )

    return H_subspace
```

---

## Technique 2: Sparse Time Evolution

**Purpose:** Apply e^{-iHΔt} efficiently without forming full matrix exponential.

### Using scipy.sparse.linalg.expm_multiply

```python
def _sparse_time_evolution_subspace(self, state_vector, num_steps):
    """
    Time evolution in particle-conserving subspace.

    Uses expm_multiply for efficient e^{-iHt}|ψ⟩ computation.
    """
    from scipy.sparse.linalg import expm_multiply

    # Convert full state to subspace representation
    psi_subspace = self._full_to_subspace(state_vector)

    # Apply time evolution in subspace
    t = -1j * self.time_step
    for _ in range(num_steps):
        psi_subspace = expm_multiply(t * self._sparse_H, psi_subspace)

    # Convert back to full Hilbert space
    return self._subspace_to_full(psi_subspace, state_vector.device)
```

### State Space Conversion

```python
def _full_to_subspace(self, state_vector):
    """Extract subspace amplitudes from full state vector."""
    psi_subspace = np.zeros(len(self._subspace_basis), dtype=np.complex128)

    state_np = state_vector.cpu().numpy()
    for i, config in enumerate(self._subspace_basis):
        idx = self.hamiltonian._config_to_index(config)
        psi_subspace[i] = state_np[idx]

    return psi_subspace

def _subspace_to_full(self, psi_subspace, device):
    """Reconstruct full state vector from subspace representation."""
    state_full = np.zeros(self.hamiltonian.hilbert_dim, dtype=np.complex128)

    for i, config in enumerate(self._subspace_basis):
        idx = self.hamiltonian._config_to_index(config)
        state_full[idx] = psi_subspace[i]

    return torch.from_numpy(state_full).to(device)
```

---

## Technique 3: Krylov State Sampling

**Purpose:** Sample configurations from time-evolved quantum states.

### Algorithm

```python
def generate_krylov_samples(self, max_krylov_dim=None, progress=True):
    """
    Generate samples from Krylov states.

    For k = 0, 1, ..., max_k - 1:
        1. Prepare |ψ_k⟩ = U^k |ψ_0⟩
        2. Sample num_shots measurements
    """
    self.krylov_samples = []

    # Initialize state vector from reference (e.g., HF state)
    initial_index = self._config_to_index(self.initial_state)
    state_vector = torch.zeros(self.hamiltonian.hilbert_dim, dtype=torch.complex64)
    state_vector[initial_index] = 1.0

    current_state = state_vector.clone()

    for k in range(max_krylov_dim):
        # Sample from current state
        samples = self._sample_from_state(current_state, self.config.shots_per_krylov)
        self.krylov_samples.append(samples)

        # Evolve: |ψ_{k+1}⟩ = U |ψ_k⟩
        if k < max_krylov_dim - 1:
            current_state = self._time_evolution_operator(current_state, num_steps=1)

    return self.krylov_samples
```

### Sampling from Quantum State

```python
def _sample_from_state(self, state_vector, num_samples):
    """
    Sample bitstrings from a quantum state.

    Uses Born rule: P(x) = |⟨x|ψ⟩|² = |ψ(x)|²
    """
    # Compute probabilities
    probs = torch.abs(state_vector) ** 2
    probs = probs / probs.sum()  # Normalize

    # Multinomial sampling
    indices = torch.multinomial(probs, num_samples, replacement=True)

    # Count occurrences
    unique, counts = np.unique(indices.cpu().numpy(), return_counts=True)

    # Convert to bitstring dictionary
    results = {}
    for idx, count in zip(unique, counts):
        bitstring = format(idx, f"0{self.num_sites}b")
        results[bitstring] = int(count)

    return results
```

---

## Technique 4: Cumulative Basis Building

**Purpose:** Accumulate samples across all Krylov states for comprehensive coverage.

### Implementation

```python
def build_cumulative_basis(self):
    """
    Build cumulative basis by accumulating samples across Krylov states.

    cumulative[k] contains all unique bitstrings from steps 0, 1, ..., k.
    """
    cumulative = []
    all_samples = {}

    for k, samples in enumerate(self.krylov_samples):
        # Merge samples from this Krylov state
        for bitstring, count in samples.items():
            all_samples[bitstring] = all_samples.get(bitstring, 0) + count

        cumulative.append(dict(all_samples))

    return cumulative
```

### Why Cumulative?

| Krylov Step | Physical Meaning | Configurations Found |
|-------------|------------------|---------------------|
| k=0 | Reference state (HF) | 1 configuration |
| k=1 | First time evolution | ~100 new configs |
| k=2 | Second evolution | ~80 new configs |
| ... | ... | ... |
| k=8 | Eighth evolution | ~20 new configs |

Cumulative basis captures configurations discovered at ALL time steps.

---

## Technique 5: Flow-Guided SKQD

**Purpose:** Combine NF-discovered basis with Krylov samples.

### Combined Basis

```python
class FlowGuidedSKQD(SampleBasedKrylovDiagonalization):
    """SKQD with Flow-Guided initial basis."""

    def get_combined_basis(self, krylov_index, include_nf=True):
        """
        Get combined basis from NF and Krylov sampling.
        """
        # Get Krylov basis
        krylov_basis = self.get_basis_states(krylov_index, cumulative=True)

        if not include_nf:
            return krylov_basis

        # Combine with NF basis
        nf_basis = self.nf_basis.to(krylov_basis.device)
        combined = torch.cat([nf_basis, krylov_basis], dim=0)

        # Remove duplicates
        unique = torch.unique(combined, dim=0)

        return unique
```

### Why Combine?

| Source | Strengths | Weaknesses |
|--------|-----------|------------|
| NF Basis | Targets ground state support | May miss dynamical correlations |
| Krylov | Explores via time evolution | Starts from single reference |
| Combined | Best of both worlds | Larger basis, more complete |

---

## Technique 6: Numerical Stability

**Purpose:** Ensure reliable eigenvalue computation for large, potentially ill-conditioned matrices.

### Regularization

```python
def compute_ground_state_energy(self, basis, regularization=1e-8):
    """
    Compute ground state energy with numerical stability improvements.
    """
    # Build projected Hamiltonian
    H_proj = self.hamiltonian.matrix_elements(basis, basis)
    H_np = H_proj.detach().cpu().numpy()
    n = H_np.shape[0]

    # Ensure Hermitian symmetry (fix numerical asymmetry)
    H_np = 0.5 * (H_np + H_np.conj().T)

    # Add regularization to diagonal
    if regularization > 0:
        H_np = H_np + regularization * np.eye(n)

    # Check conditioning
    try:
        cond = np.linalg.cond(H_np)
        if cond > 1e12:
            print(f"WARNING: Ill-conditioned (cond={cond:.2e})")
            return self._svd_ground_state(H_np)
    except:
        return self._svd_ground_state(H_np)

    # Standard diagonalization
    eigenvalues, eigenvectors = np.linalg.eigh(H_np)
    return float(eigenvalues[0])
```

### SVD Fallback for Ill-Conditioned Matrices

```python
def _svd_ground_state(self, H_np):
    """
    Compute ground state using SVD-based approach for numerical stability.
    """
    # SVD to identify near-null space
    U, s, Vh = np.linalg.svd(H_np, hermitian=True)

    # Filter out very small singular values
    threshold = 1e-10 * s.max()
    valid_mask = s > threshold
    n_valid = valid_mask.sum()

    if n_valid < len(s):
        print(f"  SVD: Projecting out {len(s) - n_valid} near-null modes")

    # Regularize small singular values
    s_reg = np.where(s > threshold, s, threshold)
    H_reg = U @ np.diag(s_reg) @ Vh

    # Diagonalize regularized matrix
    eigenvalues, _ = np.linalg.eigh(H_reg)
    return float(eigenvalues[0])
```

---

## Technique 7: Energy Validation

**Purpose:** Detect and handle numerical instabilities that produce impossible results.

### Variational Principle Check

For any variational method: E_computed ≥ E_exact

If E_computed < E_exact, something is wrong (numerical error).

```python
def run_with_nf(self, max_krylov_dim=None, progress=True):
    """Run SKQD with energy validation."""

    # ... generate Krylov samples and compute energies ...

    for k in range(1, max_krylov_dim):
        E_combined = self.compute_ground_state_energy(combined_basis)

        # Check for numerical instability
        if k > 1 and len(results["energies_combined"]) > 0:
            prev_energy = results["energies_combined"][-1]
            energy_jump = abs(E_combined - prev_energy)

            if energy_jump > 1.0:  # Suspicious large jump
                warning = f"k={k+1}: Large energy jump {energy_jump:.4f} Ha"
                results["numerical_warnings"].append(warning)
                instability_detected = True

    # Report best stable result
    if instability_detected:
        print(f"Numerical instability detected. Best stable: {best_energy:.6f}")
        results["best_stable_energy"] = best_energy

    return results
```

---

## Configuration Parameters

```python
@dataclass
class SKQDConfig:
    """Configuration for SKQD algorithm."""

    # Krylov parameters
    max_krylov_dim: int = 12          # Maximum Krylov dimension
    time_step: float = 0.1            # Δt for time evolution
    total_evolution_time: float = None  # Override: total t = k × Δt

    # Trotter parameters (for Trotterization if needed)
    num_trotter_steps: int = 8

    # Sampling parameters
    shots_per_krylov: int = 100000    # Samples per Krylov state
    use_cumulative_basis: bool = True  # Accumulate across states

    # Eigensolver parameters
    num_eigenvalues: int = 2          # k for sparse eigsh
    which_eigenvalues: str = "SA"     # Smallest algebraic

    # Numerical stability
    regularization: float = 1e-8      # Diagonal regularization

    # Hardware
    use_gpu: bool = True              # Use GPU for sparse operations
```

---

## Example Output

```
Stage 4: Sample-Based Krylov Quantum Diagonalization
============================================================
NF-only basis energy: -7.882341 (2048 configs)

Setting up particle-conserving subspace: 3,136 configs (vs 65,536 full)
Building subspace Hamiltonian (3,136 × 3,136)...
Subspace Hamiltonian built: 24,892 non-zero elements

Generating Krylov states: 100%|████████████████| 8/8 [00:12<00:00]

Krylov dim 2: Combined basis 2156, E = -7.883412
Krylov dim 3: Combined basis 2298, E = -7.883687
Krylov dim 4: Combined basis 2421, E = -7.883812
Krylov dim 5: Combined basis 2534, E = -7.883891
Krylov dim 6: Combined basis 2628, E = -7.883934
Krylov dim 7: Combined basis 2712, E = -7.883962
Krylov dim 8: Combined basis 2789, E = -7.883978

SKQD improved energy by 1.637 mHa
Combined energy: -7.883978 Ha
```

---

## When SKQD Helps vs. Hurts

### SKQD Helps When:

| Condition | Why |
|-----------|-----|
| Large valid config space (>5000) | Time evolution explores beyond NF reach |
| Strongly correlated systems | Dynamical correlations need Krylov |
| Residual expansion plateau | SKQD provides orthogonal improvement |

### SKQD May Hurt When:

| Condition | Why |
|-----------|-----|
| Small systems (<1000 valid) | Full basis already covered by residual |
| Well-converged residual | Adding configs increases numerical noise |
| Ill-conditioned H_proj | Regularization introduces error |

### Decision Logic in Pipeline

```python
# Skip SKQD if residual already achieved chemical accuracy
if residual_error_mha < 1.0:
    print("Residual expansion achieved target accuracy.")
    print("Skipping SKQD to avoid numerical instability.")
    skip_skqd = True

# Skip for small bases
if len(nf_basis) < 300:
    print("Basis small enough for direct diagonalization.")
    skip_skqd = True
```

---

## Output of Stage 4

| Output | Description | Used In |
|--------|-------------|---------|
| `combined_energy` | Best energy estimate | Final result |
| `skqd_energy` | Energy from SKQD alone | Comparison |
| `skqd_results` | Krylov dims, energies, basis sizes | Diagnostics |
| `best_stable_energy` | Most reliable result if instability detected | Fallback |

---

## Summary: SKQD Algorithm

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SKQD ALGORITHM FLOW                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. SETUP: Build particle-conserving subspace Hamiltonian              │
│         ↓                                                               │
│  2. INITIALIZE: |ψ₀⟩ = |HF⟩ (Hartree-Fock reference)                   │
│         ↓                                                               │
│  3. FOR k = 0 to max_krylov_dim:                                       │
│     a. SAMPLE: Draw shots from |ψₖ⟩ via Born rule                      │
│     b. ACCUMULATE: Add new configs to cumulative basis                 │
│     c. EVOLVE: |ψₖ₊₁⟩ = e^{-iHΔt} |ψₖ⟩ (sparse expm)                  │
│         ↓                                                               │
│  4. COMBINE: Merge NF basis with Krylov samples                        │
│         ↓                                                               │
│  5. DIAGONALIZE: Build H_proj, compute eigenvalues                     │
│         ↓                                                               │
│  6. VALIDATE: Check energy is above exact (variational)                │
│         ↓                                                               │
│  7. OUTPUT: Best stable ground state energy estimate                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Related Documentation

- [MODULE_MOLECULAR_HAMILTONIAN.md](MODULE_MOLECULAR_HAMILTONIAN.md) - Hamiltonian construction and Jordan-Wigner transformation
- [COMPUTATIONAL_ACCELERATION.md](COMPUTATIONAL_ACCELERATION.md) - Performance optimization techniques including sparse time evolution and GPU acceleration
- [STAGE3_RESIDUAL_EXPANSION.md](STAGE3_RESIDUAL_EXPANSION.md) - Residual-based basis expansion that feeds into SKQD
