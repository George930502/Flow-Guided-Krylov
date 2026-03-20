# Stage 2: Diversity-Aware Basis Extraction

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 2: DIVERSITY-AWARE BASIS EXTRACTION                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Input: accumulated_basis from Stage 1 (e.g., 4096 configurations)        │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Step 1: REMOVE DUPLICATES                                          │  │
│   │          4096 raw → 3500 unique                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                              ↓                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Step 2: BUCKET BY EXCITATION RANK                                  │  │
│   │          Rank 0 (HF): 1      Rank 1 (singles): 120                 │  │
│   │          Rank 2 (doubles): 2100   Rank 3 (triples): 980            │  │
│   │          Rank 4+: 299                                               │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                              ↓                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Step 3: ALLOCATE BUDGET PER RANK                                   │  │
│   │          Rank 0: 5%    Rank 1: 25%    Rank 2: 50%                  │  │
│   │          Rank 3: 15%   Rank 4+: 5%                                  │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                              ↓                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Step 4: DPP-INSPIRED SELECTION within each bucket                  │  │
│   │          Maximize: importance × diversity                           │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                              ↓                                              │
│   Output: selected_basis (e.g., 2048 diverse configurations)               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Why Diversity Selection?

### Problem: Redundant Configurations

Stage 1 samples configurations based on NQS probability. This leads to:
- Many near-duplicate configurations (differ by 1-2 orbitals)
- Over-representation of high-probability regions
- Under-representation of important but lower-probability configurations

### Solution: Stratified Diverse Sampling

Instead of keeping all accumulated configurations:
1. Group by physical meaning (excitation rank)
2. Select diverse representatives from each group
3. Ensure coverage of different correlation effects

---

## Technique 1: Excitation Rank Classification

**Purpose:** Classify configurations by their "distance" from the reference (Hartree-Fock) state.

### Definition

Excitation rank = number of electron-hole pairs relative to reference:
$$\text{rank}(x) = \frac{1}{2} \sum_i |x_i - x_i^{HF}|$$

| Rank | Name | Physical Meaning |
|------|------|------------------|
| 0 | Reference | Hartree-Fock state (mean-field) |
| 1 | Singles | One electron excited (orbital relaxation) |
| 2 | Doubles | Two electrons excited (dynamic correlation) |
| 3 | Triples | Three electrons excited (higher correlation) |
| 4+ | Higher | Four+ electrons excited (static correlation) |

### Implementation

```python
def compute_excitation_rank(config: torch.Tensor, reference: torch.Tensor) -> int:
    """
    Compute excitation rank relative to reference (HF state).

    Each excitation involves one hole (electron removed) and one particle (electron added).
    Total bit differences / 2 = number of excitations.
    """
    diff = (config != reference).sum().item()
    return diff // 2  # Divide by 2: each excitation = 2 bit flips
```

### Example

```
HF state (reference): [1,1,1,0,0 | 1,1,1,0,0]  (3α + 3β in first 3 orbitals)

Rank 0: [1,1,1,0,0 | 1,1,1,0,0]  (identical to HF)
Rank 1: [1,1,0,1,0 | 1,1,1,0,0]  (one α electron moved: orbital 3→4)
Rank 2: [1,0,1,1,0 | 1,1,0,0,1]  (two electrons moved)
Rank 3: [0,1,1,1,0 | 1,0,1,0,1]  (three electrons moved)
```

---

## Technique 2: Excitation Bucketing

**Purpose:** Organize configurations into buckets by excitation rank for stratified selection.

### Implementation

```python
class ExcitationBucketer:
    """Groups configurations by excitation rank."""

    def __init__(self, reference: torch.Tensor, n_orbitals: int):
        self.reference = reference
        self.n_orbitals = n_orbitals
        self.buckets: Dict[int, List[torch.Tensor]] = defaultdict(list)

    def add_configs(self, configs: torch.Tensor):
        """Add configurations to appropriate buckets."""
        for i in range(len(configs)):
            rank = compute_excitation_rank(configs[i], self.reference)
            self.buckets[rank].append(configs[i])

    def get_bucket_sizes(self) -> Dict[int, int]:
        """Get size of each bucket."""
        return {rank: len(configs) for rank, configs in self.buckets.items()}
```

### Typical Distribution

For a well-trained flow on a molecular system:

| Rank | Expected % | Why |
|------|------------|-----|
| 0 | ~0.1% | Only 1 HF configuration |
| 1 | ~5-10% | Single excitations (n_occ × n_virt) |
| 2 | ~50-60% | Double excitations dominate correlation |
| 3 | ~20-30% | Triple excitations (diminishing importance) |
| 4+ | ~5-10% | Higher excitations (small contribution) |

---

## Technique 3: Budget Allocation

**Purpose:** Allocate selection budget based on physical importance of each excitation class.

### Default Budget Fractions

```python
@dataclass
class DiversityConfig:
    max_configs: int = 2048          # Total budget

    # Excitation rank budgets (as fractions)
    rank_0_fraction: float = 0.05    # HF and near-HF
    rank_1_fraction: float = 0.25    # Single excitations
    rank_2_fraction: float = 0.50    # Double excitations (MOST IMPORTANT)
    rank_3_fraction: float = 0.15    # Triple excitations
    rank_4_plus_fraction: float = 0.05  # Higher excitations
```

### Why Emphasize Doubles?

From quantum chemistry theory:
1. **Brillouin's theorem:** Singles don't directly lower HF energy
2. **MP2 theory:** First correlation correction comes from doubles
3. **Coupled Cluster:** CCSD (doubles) captures ~95% of correlation energy
4. **CI expansion:** Doubles have largest coefficients after HF

### Budget Computation

```python
def _compute_bucket_budgets(self) -> Dict[int, int]:
    """Compute number of configs to select from each rank."""
    cfg = self.config
    max_configs = cfg.max_configs

    budgets = {
        0: int(max_configs * cfg.rank_0_fraction),   # e.g., 102
        1: int(max_configs * cfg.rank_1_fraction),   # e.g., 512
        2: int(max_configs * cfg.rank_2_fraction),   # e.g., 1024
        3: int(max_configs * cfg.rank_3_fraction),   # e.g., 307
    }

    # Remaining goes to rank 4+
    used = sum(budgets.values())
    budgets[4] = max_configs - used  # e.g., 103

    return budgets
```

---

## Technique 4: Hamming Distance Diversity

**Purpose:** Measure how "different" two configurations are.

### Definition

Hamming distance = number of bit positions that differ:
$$d_H(x, y) = \sum_i \mathbf{1}[x_i \neq y_i]$$

### Implementation

```python
def compute_hamming_distance(config1: torch.Tensor, config2: torch.Tensor) -> int:
    """Compute Hamming distance between two configurations."""
    return (config1 != config2).sum().item()

def compute_hamming_distance_matrix(configs: torch.Tensor) -> torch.Tensor:
    """Compute pairwise Hamming distance matrix."""
    n = len(configs)
    # Broadcast for vectorized computation
    c1 = configs.unsqueeze(1)  # (n, 1, sites)
    c2 = configs.unsqueeze(0)  # (1, n, sites)

    # Hamming distance = number of differing positions
    distances = (c1 != c2).sum(dim=-1)  # (n, n)

    return distances
```

### Example Distance Matrix

```
Configs:  A=[1,1,0,0]  B=[1,0,1,0]  C=[0,0,1,1]  D=[1,1,0,1]

         A    B    C    D
    A [  0    2    4    1  ]
    B [  2    0    2    3  ]
    C [  4    2    0    3  ]
    D [  1    3    3    0  ]

A and D are most similar (distance 1)
A and C are most different (distance 4)
```

---

## Technique 5: Importance Weighting

**Purpose:** Prioritize configurations that are more important for the ground state.

### Weight Sources

```python
def _compute_importance_weights(self, unique_configs, nqs_probs, local_energies, inverse):
    weights = torch.ones(n_unique, device=device)

    # 1. NQS probability weighting (higher |ψ|² = more important)
    if self.config.use_nqs_importance and nqs_probs is not None:
        unique_probs = aggregate_probs_for_unique(nqs_probs, inverse)
        weights = weights * (unique_probs + 1e-10)

    # 2. Energy-based weighting (lower E_loc = more important)
    if self.config.use_energy_importance and local_energies is not None:
        unique_energies = aggregate_energies_for_unique(local_energies, inverse)

        # Shift to positive range
        e_min = unique_energies.min()
        e_shifted = unique_energies - e_min + 1.0

        # Inverse energy weighting (lower is better)
        energy_weights = 1.0 / e_shifted
        weights = weights * energy_weights

    return weights
```

### Interpretation

| Weight Factor | Effect |
|---------------|--------|
| High NQS prob | Configuration appears often in ground state |
| Low local energy | Configuration contributes to lowering total energy |
| Combined | Balance sampling frequency with energy importance |

---

## Technique 6: DPP-Inspired Greedy Selection

**Purpose:** Select diverse subset that balances importance and coverage.

### Background: Determinantal Point Processes (DPP)

DPPs are probabilistic models that naturally encourage diversity:
- Probability of selecting subset S ∝ det(L_S)
- Off-diagonal elements encode "repulsion" between similar items
- Exact DPP sampling is O(n³), too expensive for large sets

### Greedy Approximation

Instead of exact DPP, use greedy selection that approximates DPP behavior:

```python
def _dpp_select(self, configs, weights, n_select):
    """
    DPP-inspired greedy selection for diversity.

    Algorithm:
    1. Start with highest-weight config
    2. Iteratively add config that maximizes: weight × min_distance_to_selected
    """
    n = len(configs)

    # Precompute distance matrix
    distances = compute_hamming_distance_matrix(configs).float()

    # Greedy selection
    selected = []
    remaining = set(range(n))

    # Start with highest weight configuration
    first = weights.argmax().item()
    selected.append(first)
    remaining.remove(first)

    while len(selected) < n_select and remaining:
        best_score = -float('inf')
        best_idx = None

        for idx in remaining:
            # Minimum distance to any already-selected config
            min_dist = distances[idx, selected].min().item()

            # Skip if too close to existing selection
            if min_dist < self.config.min_hamming_distance:  # e.g., 2
                continue

            # Score = weight × distance^scale
            # Higher weight + farther from selected = better
            score = weights[idx].item() * (min_dist ** self.config.dpp_kernel_scale)

            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            # All remaining are too close; pick by weight only
            remaining_list = list(remaining)
            remaining_weights = weights[remaining_list]
            best_local = remaining_weights.argmax().item()
            best_idx = remaining_list[best_local]

        selected.append(best_idx)
        remaining.remove(best_idx)

    return torch.tensor(selected, device=configs.device)
```

### Selection Criteria

The score for adding configuration x to selected set S:
$$\text{score}(x) = w(x) \cdot \left(\min_{s \in S} d_H(x, s)\right)^{\alpha}$$

where:
- $w(x)$ = importance weight
- $d_H(x, s)$ = Hamming distance to nearest selected config
- $\alpha$ = kernel scale (typically 0.5)

### Why This Works

| Component | Effect |
|-----------|--------|
| Weight term | Prefers important configurations |
| Distance term | Prefers configurations far from already-selected |
| Min distance | Ensures coverage of configuration space |
| Minimum distance threshold | Prevents selecting near-duplicates |

---

## Technique 7: Stratified Selection Pipeline

**Purpose:** Combine all techniques into a coherent selection algorithm.

### Full Selection Process

```python
def select(self, configs, nqs_probs=None, local_energies=None):
    """
    Select diverse subset of configurations.

    Returns:
        selected: (n_selected, n_sites) selected configurations
        stats: dictionary with selection statistics
    """
    # Step 1: Remove duplicates
    unique_configs, inverse = torch.unique(configs, dim=0, return_inverse=True)
    n_unique = len(unique_configs)

    # Step 2: Bucket by excitation rank
    self.bucketer = ExcitationBucketer(self.reference, self.n_orbitals)
    self.bucketer.add_configs(unique_configs)

    # Step 3: Compute importance weights
    if nqs_probs is not None or local_energies is not None:
        weights = self._compute_importance_weights(
            unique_configs, nqs_probs, local_energies, inverse
        )
    else:
        weights = torch.ones(n_unique)

    # Step 4: Compute budget per rank
    budget = self._compute_bucket_budgets()

    # Step 5: Select from each bucket
    selected_indices = []
    bucket_stats = {}

    for rank in sorted(self.bucketer.buckets.keys()):
        bucket_configs = self.bucketer.get_bucket(rank)
        if not bucket_configs:
            continue

        n_bucket = len(bucket_configs)
        n_select = budget.get(rank, budget.get(4, 0))  # Rank 4+ uses same budget

        if n_select == 0:
            continue

        # Get indices and weights for this bucket
        bucket_tensor = torch.stack(bucket_configs)
        bucket_indices = self._find_indices(unique_configs, bucket_tensor)
        bucket_weights = weights[bucket_indices]

        # DPP selection within bucket
        if self.config.use_dpp_selection and n_bucket > n_select:
            selected = self._dpp_select(bucket_tensor, bucket_weights, n_select)
        else:
            # Simple weighted selection (top-k by weight)
            selected = self._weighted_select(bucket_indices, bucket_weights,
                                             min(n_select, n_bucket))

        selected_indices.extend(selected.tolist())
        bucket_stats[f'rank_{rank}'] = {
            'available': n_bucket,
            'selected': len(selected),
        }

    # Step 6: Collect selected configurations
    selected = unique_configs[selected_indices]

    stats = {
        'n_input': len(configs),
        'n_unique': n_unique,
        'n_selected': len(selected),
        'bucket_stats': bucket_stats,
    }

    return selected, stats
```

---

## Configuration Parameters

```python
@dataclass
class DiversityConfig:
    """Configuration for diversity-aware selection."""

    # Maximum configurations to select
    max_configs: int = 2048

    # Excitation rank budgets (as fractions, should sum to ~1.0)
    rank_0_fraction: float = 0.05    # HF reference
    rank_1_fraction: float = 0.25    # Singles
    rank_2_fraction: float = 0.50    # Doubles (most important!)
    rank_3_fraction: float = 0.15    # Triples
    rank_4_plus_fraction: float = 0.05  # Higher

    # Diversity parameters
    min_hamming_distance: int = 2    # Minimum distance between selected configs
    use_dpp_selection: bool = True   # Use DPP-inspired selection
    dpp_kernel_scale: float = 0.5    # Exponent for distance in score

    # Importance weighting
    use_nqs_importance: bool = True  # Weight by NQS probability
    use_energy_importance: bool = True  # Weight by local energy
```

---

## Example Output

```
Stage 2: Diversity-Aware Basis Extraction
============================================================
Raw accumulated basis: 4096 configs
All configurations satisfy particle conservation

Bucket distribution:
  Rank 0: 1 available, 1 selected
  Rank 1: 156 available, 102 selected
  Rank 2: 2847 available, 1024 selected
  Rank 3: 892 available, 307 selected
  Rank 4+: 104 available, 103 selected

Selected 1537 diverse configs from 4000 unique

Selection stats:
  n_input: 4096
  n_unique: 4000
  n_selected: 1537
  Mean Hamming distance: 8.4
  Min Hamming distance: 2
```

---

## Output of Stage 2

| Output | Description | Used In |
|--------|-------------|---------|
| `selected_basis` | Diverse configuration set | Stage 3, 4 |
| `diversity_stats` | Bucket distribution, selection counts | Diagnostics |
| `nf_basis_size` | Number of selected configurations | Summary |

---

## Why Diversity Matters

### Without Diversity Selection

```
Input: 4096 configs (many near-duplicates)
→ Redundant basis vectors
→ Ill-conditioned Hamiltonian matrix
→ Numerical instability in diagonalization
→ Wasted computation on similar configs
```

### With Diversity Selection

```
Input: 4096 configs
→ Select 2048 diverse representatives
→ Good coverage of configuration space
→ Well-conditioned matrices
→ Efficient use of computational budget
```

### Physical Interpretation

Diverse selection ensures:
1. **Rank coverage:** All excitation types represented
2. **Spatial coverage:** Configurations spread across Hilbert space
3. **Importance coverage:** High-weight configs prioritized
4. **Numerical stability:** Distinct basis vectors
