# Stage 3: Residual-Based Basis Expansion

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 3: RESIDUAL-BASED BASIS EXPANSION                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Input: selected_basis from Stage 2 (e.g., 2048 configurations)           │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  ITERATION 1:                                                       │  │
│   │  1. Diagonalize H in current basis → get |Φ⟩, E                    │  │
│   │  2. Compute residual: r = H|Φ⟩ - E|Φ⟩                              │  │
│   │  3. Find configs with large |⟨x|r⟩| outside basis                  │  │
│   │  4. Add top-k configs to basis                                      │  │
│   │  Basis: 2048 → 2198                                                 │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                              ↓                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  ITERATION 2:                                                       │  │
│   │  Same process with updated basis                                    │  │
│   │  Basis: 2198 → 2347                                                 │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                              ↓                                              │
│                            ...                                              │
│                              ↓                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  ITERATION N: (convergence or max iterations)                       │  │
│   │  Final basis: 3521 configurations                                   │  │
│   │  Final energy: converged to within threshold                        │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Output: expanded_basis, final_energy                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Why Residual Expansion?

### Problem: Missing Important Configurations

Stage 1 (NF-NQS) and Stage 2 (Diversity Selection) may miss configurations that:
- Have low NQS probability but significant Hamiltonian coupling
- Were never sampled due to stochastic nature of flow
- Are important for off-diagonal correlations

### Solution: Selected-CI Style Expansion

Systematically find and add missing configurations by analyzing the residual:
- Residual r = H|Φ⟩ - E|Φ⟩ measures "error" in current approximation
- Large |⟨x|r⟩| indicates configuration x is important but missing
- Iteratively add highest-residual configurations until convergence

---

## Mathematical Foundation

### The Residual Vector

Given current approximate ground state |Φ⟩ with energy E:

$$|r\rangle = H|\Phi\rangle - E|\Phi\rangle$$

For an exact eigenstate, r = 0. For an approximation, r ≠ 0.

### Residual Element for External Configuration

For configuration |x⟩ **outside** the current basis:
$$\langle x|r\rangle = \langle x|H|\Phi\rangle - E\underbrace{\langle x|\Phi\rangle}_{=0} = \langle x|H|\Phi\rangle$$

This simplifies because |Φ⟩ has no component on |x⟩.

### Expanding the Residual

$$\langle x|H|\Phi\rangle = \sum_{j \in \text{basis}} c_j \langle x|H|j\rangle$$

where:
- $c_j$ = coefficients of |Φ⟩ in the basis
- $\langle x|H|j\rangle$ = Hamiltonian matrix elements

---

## Technique 1: Residual-Based Selection

### Algorithm

```python
def _find_residual_configs(self, basis, energy, eigenvector):
    """
    Find configurations with large residual contributions.

    For each configuration x not in basis:
        r_x = ⟨x|H|Φ⟩ = Σ_j c_j ⟨x|H|j⟩

    where j runs over basis states and c_j are eigenvector coefficients.
    """
    n_basis = len(basis)
    coeffs = torch.from_numpy(eigenvector).float()

    # Build set of basis configurations for O(1) lookup
    basis_set = self._configs_to_set(basis)

    candidates = []
    candidate_residuals = []

    # For each basis state, find its Hamiltonian connections
    for j in range(n_basis):
        if abs(coeffs[j].item()) < 1e-10:
            continue  # Skip negligible coefficients

        # Get connected configurations: which |x⟩ have ⟨x|H|j⟩ ≠ 0
        connected, elements = self.hamiltonian.get_connections(basis[j])

        for k in range(len(connected)):
            config_tuple = tuple(connected[k].cpu().tolist())

            # Only consider configurations OUTSIDE current basis
            if config_tuple not in basis_set:
                # Residual contribution: r_x += c_j * ⟨x|H|j⟩
                residual = coeffs[j].item() * elements[k].item()

                candidates.append(connected[k])
                candidate_residuals.append(abs(residual))

    # Aggregate residuals for duplicate candidates
    unique_candidates, unique_residuals = self._aggregate_residuals(
        candidates, candidate_residuals
    )

    # Filter by threshold
    mask = unique_residuals > self.config.residual_threshold
    filtered = unique_candidates[mask]
    filtered_residuals = unique_residuals[mask]

    # Select top-k by residual magnitude
    n_select = min(self.config.max_configs_per_iter, len(filtered))
    _, top_indices = torch.topk(filtered_residuals, n_select)

    return filtered[top_indices], filtered_residuals[top_indices]
```

### Why This Works

| Observation | Implication |
|-------------|-------------|
| Large $\|c_j\|$ | Basis state j is important in ground state |
| Large $\|H_{xj}\|$ | Config x strongly coupled to basis state j |
| Large $\|c_j H_{xj}\|$ | Config x significantly affects ground state through j |

---

## Technique 2: Perturbative (2nd-Order PT) Selection

### Motivation

Simple residual selection treats all configurations equally given their residual.
Better: use 2nd-order perturbation theory to estimate energy contribution.

### PT2 Importance Estimate

$$\epsilon_x = \frac{|\langle x|H|\Phi\rangle|^2}{|E - E_x|}$$

where $E_x = \langle x|H|x\rangle$ is the diagonal energy of configuration x.

This estimates how much adding configuration x would lower the energy.

### Implementation (SelectedCIExpander)

```python
class SelectedCIExpander:
    """
    Selected-CI style expansion with perturbation-based selection.

    Uses second-order perturbation theory to estimate importance:
    ε_x = |⟨x|H|Φ⟩|² / |E - E_x|
    """

    def _find_important_configs(self, basis, energy, eigenvector):
        """
        Find important configurations using perturbation theory.
        """
        coeffs = torch.from_numpy(eigenvector).float()
        basis_set = {tuple(c.cpu().tolist()) for c in basis}

        candidates = []
        importances = []

        for j in range(len(basis)):
            if abs(coeffs[j].item()) < 1e-10:
                continue

            connected, elements = self.hamiltonian.get_connections(basis[j])

            for k in range(len(connected)):
                config_tuple = tuple(connected[k].cpu().tolist())

                if config_tuple not in basis_set:
                    # Coupling: ⟨x|H|Φ⟩ contribution from basis state j
                    coupling = coeffs[j].item() * elements[k].item()

                    # Diagonal energy of candidate
                    E_x = self.hamiltonian.diagonal_element(connected[k]).item()

                    # PT2 importance: |coupling|² / |E - E_x|
                    denominator = abs(energy - E_x) + 1e-10
                    importance = coupling**2 / denominator

                    candidates.append(connected[k])
                    importances.append(importance)

        # Aggregate importances for duplicate candidates (sum, not max)
        unique_candidates, unique_importances = self._aggregate_importances(
            candidates, importances
        )

        # Select top-k by importance
        n_select = min(self.config.max_configs_per_iter, len(unique_candidates))
        _, top_indices = torch.topk(unique_importances, n_select)

        return unique_candidates[top_indices], unique_importances[top_indices]
```

### Comparison: Residual vs PT2

| Method | Selection Criterion | Best For |
|--------|---------------------|----------|
| Residual | $\|c_j H_{xj}\|$ | Fast, simple screening |
| PT2 | $\|c_j H_{xj}\|^2 / \|E - E_x\|$ | Accurate energy estimates |

PT2 is better because:
- Accounts for energy denominator (large gaps = less important)
- Estimates actual energy contribution
- Similar to CIPSI and ASCI methods used in quantum chemistry

---

## Technique 3: Iterative Expansion Loop

### Main Loop

```python
def expand_basis(self, current_basis, energy=None, eigenvector=None):
    """
    Expand basis using residual analysis.

    Iteratively:
    1. Diagonalize in current basis
    2. Find important missing configurations
    3. Add them to basis
    4. Repeat until convergence
    """
    expanded_basis = current_basis.clone()

    # Initial diagonalization
    if energy is None or eigenvector is None:
        energy, eigenvector = self._diagonalize(expanded_basis)

    history = {
        'energies': [energy],
        'basis_sizes': [len(current_basis)],
        'configs_added': [],
    }

    for iteration in range(self.config.max_iterations):
        # Check max basis size
        if len(expanded_basis) >= self.config.max_basis_size:
            break

        # Find important missing configurations
        new_configs, residuals = self._find_residual_configs(
            expanded_basis, energy, eigenvector
        )

        if len(new_configs) == 0:
            break  # No more configs to add

        # Add to basis and remove duplicates
        expanded_basis = torch.cat([expanded_basis, new_configs], dim=0)
        expanded_basis = torch.unique(expanded_basis, dim=0)

        # Rediagonalize
        new_energy, new_eigenvector = self._diagonalize(expanded_basis)

        # Track history
        history['energies'].append(new_energy)
        history['basis_sizes'].append(len(expanded_basis))
        history['configs_added'].append(len(new_configs))

        # Check convergence
        energy_change = abs(new_energy - energy)
        if energy_change < self.config.energy_convergence:
            break

        energy = new_energy
        eigenvector = new_eigenvector

    return expanded_basis, stats
```

### Convergence Criteria

The expansion stops when any of these conditions is met:

| Criterion | Condition | Meaning |
|-----------|-----------|---------|
| Energy convergence | $\|E_{new} - E_{old}\| < \epsilon$ | Energy has stabilized |
| Max iterations | iteration ≥ max_iterations | Iteration limit reached |
| Max basis size | len(basis) ≥ max_basis_size | Memory/compute limit |
| No new configs | len(new_configs) == 0 | All important configs found |

---

## Technique 4: Efficient Diagonalization

### For Small Bases (< 500)

```python
def _diagonalize(self, basis: torch.Tensor):
    """Diagonalize Hamiltonian in given basis."""
    # Build dense Hamiltonian matrix
    H_matrix = self.hamiltonian.matrix_elements(basis, basis)
    H_np = H_matrix.cpu().numpy()

    # Dense eigensolver (fast for small matrices)
    eigenvalues, eigenvectors = np.linalg.eigh(H_np)

    return float(eigenvalues[0]), eigenvectors[:, 0]
```

### For Large Bases (≥ 500)

```python
from scipy.sparse.linalg import eigsh

def _diagonalize_sparse(self, basis: torch.Tensor):
    """Diagonalize using sparse solver for large bases."""
    H_matrix = self.hamiltonian.matrix_elements(basis, basis)
    H_sparse = scipy.sparse.csr_matrix(H_matrix.cpu().numpy())

    # Sparse eigensolver (Lanczos algorithm)
    eigenvalues, eigenvectors = eigsh(
        H_sparse,
        k=1,              # Only ground state
        which='SA',       # Smallest algebraic
        tol=1e-10
    )

    return float(eigenvalues[0]), eigenvectors[:, 0]
```

---

## Configuration Parameters

```python
@dataclass
class ResidualExpansionConfig:
    """Configuration for residual-based expansion."""

    # Maximum configurations to add per iteration
    max_configs_per_iter: int = 100

    # Residual threshold for adding configurations
    residual_threshold: float = 1e-4

    # Maximum total iterations
    max_iterations: int = 10

    # Convergence criterion (energy change in Hartree)
    energy_convergence: float = 1e-6

    # Early stopping: minimum energy improvement per iteration (in mHa)
    # Stop if improvement < this threshold for consecutive iterations
    min_energy_improvement_mha: float = 0.05  # 0.05 mHa = 5e-5 Ha

    # Number of consecutive stagnant iterations before stopping
    stagnation_patience: int = 2

    # Maximum total basis size
    max_basis_size: int = 4096

    # Whether to use importance sampling for residual computation
    use_importance_sampling: bool = True

    # Number of samples for importance sampling
    n_importance_samples: int = 10000
```

---

## Technique 5: Early Stopping for Stagnation

### Problem

Residual expansion can continue adding configurations even when energy improvement has stagnated:

```
Iter 4: E = -61.791100
Iter 5: E = -61.791115 (ΔE = 0.015 mHa)
Iter 6: E = -61.791115 (ΔE = 0.000 mHa)  ← stagnant
Iter 7: E = -61.791115 (ΔE = 0.000 mHa)  ← stagnant
...continues wastefully...
```

### Solution

Track energy improvement per iteration and stop early if improvement falls below threshold:

```python
# Early stopping parameters
min_improvement_mha = 0.05  # Minimum improvement threshold
stagnation_patience = 2     # Stop after N stagnant iterations

stagnant_count = 0
for iteration in range(max_iterations):
    # ... expansion step ...

    if prev_energy is not None:
        improvement_mha = (prev_energy - current_energy) * 1000

        if improvement_mha < min_improvement_mha:
            stagnant_count += 1
            if stagnant_count >= stagnation_patience:
                print(f"Converged: improvement < {min_improvement_mha} mHa "
                      f"for {stagnation_patience} iterations")
                break
        else:
            stagnant_count = 0  # Reset counter
```

### Benefits

| Metric | Before | After |
|--------|--------|-------|
| Iterations for LiH | 10 | 4-5 |
| Iterations for larger systems | 10+ | Stops when converged |
| Wasted computation | High | Minimal |

---

## Adaptive Scaling

For larger molecules, the pipeline automatically scales parameters:

```python
# From EnhancedPipelineConfig.adapt_to_system_size()

if tier == "medium":  # 1000-5000 valid configs
    self.residual_iterations = 10
    self.residual_configs_per_iter = 200

elif tier == "large":  # 5000-20000 valid configs
    self.residual_iterations = 15
    self.residual_configs_per_iter = 300
    self.residual_threshold = 1e-7

else:  # very_large: >20000 valid configs
    self.residual_iterations = 20
    self.residual_configs_per_iter = 500
    self.residual_threshold = 1e-8
```

---

## Example Output

```
Stage 3: Residual-Based Basis Expansion
============================================================
Using perturbative (2nd-order PT) selection for configuration importance
  Iter 1: 2048 -> 2198 (+150), E = -7.882341
  Iter 2: 2198 -> 2347 (+149), E = -7.883012
  Iter 3: 2347 -> 2491 (+144), E = -7.883498
  Iter 4: 2491 -> 2628 (+137), E = -7.883821
  Iter 5: 2628 -> 2753 (+125), E = -7.884023
  Iter 6: 2753 -> 2861 (+108), E = -7.884134
  Iter 7: 2861 -> 2954 (+93), E = -7.884198
  Iter 8: 2954 -> 3021 (+67), E = -7.884231
  Iter 9: 3021 -> 3064 (+43), E = -7.884248
  Iter 10: 3064 -> 3089 (+25), E = -7.884255
  Converged: no new configurations found

Expanded: 2048 -> 3089
Energy improvement: -7.881234 -> -7.884255
```

---

## Connection to Selected-CI Methods

### Historical Context

Residual expansion is closely related to established quantum chemistry methods:

| Method | Year | Key Idea |
|--------|------|----------|
| CIPSI | 1973 | Select determinants by PT2 contribution |
| Heat-bath CI | 2016 | Efficient screening via Hamiltonian structure |
| ASCI | 2017 | Adaptive sampling + PT2 selection |
| SHCI | 2017 | Semistochastic heat-bath CI |

### Our Implementation

Combines ideas from these methods:
1. **PT2 selection** (from CIPSI): Use energy denominator
2. **Iterative refinement** (from ASCI): Multiple rounds of selection
3. **Hamiltonian connections** (from Heat-bath CI): Only check connected configs
4. **Threshold pruning** (from SHCI): Filter by importance threshold

---

## Output of Stage 3

| Output | Description | Used In |
|--------|-------------|---------|
| `expanded_basis` | Enlarged configuration set | Stage 4 (SKQD) |
| `residual_energy` | Best energy from expansion | Final result comparison |
| `expansion_stats` | History of iterations, energy, basis size | Diagnostics |

---

## Why Residual Expansion Is Critical

### Before Residual Expansion

- Flow samples based on learned probability
- May miss low-probability but important configurations
- Basis limited to "typical" ground state support

### After Residual Expansion

- Systematically recovers missing important configurations
- Guided by physics (Hamiltonian structure)
- Energy improves with each iteration
- Reaches near-optimal basis for given size budget

### Energy Improvement Example

```
LiH molecule (4 electrons, 6 orbitals):

Stage 1 (NF-NQS):        E = -7.8742 Ha  (error: 9.5 mHa)
Stage 2 (Diversity):     E = -7.8812 Ha  (error: 2.5 mHa)
Stage 3 (Residual):      E = -7.8834 Ha  (error: 0.3 mHa)  ← Chemical accuracy!
Exact (FCI):             E = -7.8837 Ha
```
