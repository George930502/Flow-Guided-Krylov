# Stage 1: NF-NQS Co-Training

## Overview Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        STAGE 1: NF-NQS CO-TRAINING                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────────────┐         ┌──────────────────────┐                │
│   │  Normalizing Flow    │ samples │  Neural Quantum      │                │
│   │  (Particle-Conserving)├────────►  State (NQS)         │                │
│   │                      │         │                      │                │
│   │  Learns: WHERE to    │         │  Learns: AMPLITUDE   │                │
│   │  sample (support)    │◄────────┤  of wavefunction     │                │
│   └──────────────────────┘ teacher └──────────────────────┘                │
│            │                                  │                             │
│            │ samples configs                  │ computes |ψ(x)|²            │
│            ▼                                  ▼                             │
│   ┌─────────────────────────────────────────────────────────┐              │
│   │              Hamiltonian H                               │              │
│   │  Computes: E_loc(x) = ⟨x|H|ψ⟩/⟨x|ψ⟩ (local energy)      │              │
│   └─────────────────────────────────────────────────────────┘              │
│                              │                                              │
│                              ▼                                              │
│   ┌─────────────────────────────────────────────────────────┐              │
│   │         MIXED-OBJECTIVE LOSS FUNCTION                    │              │
│   │  L = w_t·L_teacher + w_p·L_physics - w_e·H(flow)        │              │
│   └─────────────────────────────────────────────────────────┘              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Technique 1: Particle-Conserving Flow (Gumbel-Top-K Sampling)

**Purpose:** Generate only physically valid configurations with correct electron count.

**Problem with Standard Flows:**
```
Standard NF: Sample each of 2n qubits independently
  → Could output [1,1,1,1,0,0,0,0] (4 electrons)
  → But system requires exactly 5 alpha + 5 beta electrons
  → ~95% of samples are INVALID (wrong electron count)
```

**Solution - Gumbel-Top-K:**
```python
# Instead of sampling each qubit independently,
# sample WHICH orbitals are occupied:

alpha_logits = self.alpha_scorer(context=None, batch_size)  # Score each orbital
alpha_config = self.gumbel_topk(alpha_logits, n_alpha)      # Select exactly n_alpha
```

### Gumbel-Top-K Algorithm

```
Input: logits (n_orbitals,), k (number to select), temperature T
Output: one_hot mask with exactly k ones

1. Sample Gumbel noise:
   g_i = -log(-log(U_i)), where U_i ~ Uniform(0,1)

2. Perturb logits:
   z_i = (logits_i + g_i) / T

3. Select top-k indices:
   indices = argsort(z)[-k:]

4. Create binary mask:
   one_hot = zeros(n_orbitals)
   one_hot[indices] = 1

Return: one_hot  ← EXACTLY k ones guaranteed
```

**Implementation (from `particle_conserving_flow.py`):**
```python
class GumbelTopK(nn.Module):
    def forward(self, logits, k, hard=True):
        # Add Gumbel noise for stochastic selection
        gumbel_noise = -torch.log(-torch.log(
            torch.rand_like(logits).clamp(min=1e-10)
        ))
        perturbed_logits = (logits + gumbel_noise) / self.temperature

        # Get top-k indices (ALWAYS returns exactly k)
        _, top_indices = torch.topk(perturbed_logits, k, dim=-1)

        # Create one-hot encoding
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, top_indices, 1.0)  # Exactly k ones

        if hard:
            # Straight-through estimator for gradients
            soft = F.softmax(perturbed_logits, dim=-1)
            soft_topk = soft * one_hot
            return one_hot - soft_topk.detach() + soft_topk
        else:
            return F.softmax(perturbed_logits / self.temperature, dim=-1)
```

**Why it guarantees particle conservation:**
- `torch.topk()` is a deterministic operation that always returns exactly k indices
- No probabilistic approximation—hard constraint enforced architecturally
- Works for both alpha and beta spin channels independently

### Alpha-Beta Correlation

```python
# Alpha channel: unconditional scoring
alpha_logits = self.alpha_scorer(context=None, batch_size=batch_size)
alpha_config = self.gumbel_topk(alpha_logits, self.n_alpha, hard=hard)

# Beta channel: conditioned on alpha occupation
alpha_context = self.alpha_to_beta(alpha_config)  # Encode alpha pattern
beta_input = torch.cat([zeros, alpha_context], dim=-1)
beta_logits = self.beta_conditioned_scorer(beta_input)
beta_config = self.gumbel_topk(beta_logits, self.n_beta, hard=hard)

# Combine: [alpha_orbitals | beta_orbitals]
configs = torch.cat([alpha_config, beta_config], dim=-1)
```

**Physical meaning:** Beta electrons "see" where alpha electrons are, enabling correlation effects like electron pairing.

---

## Technique 2: Temperature Annealing

**Purpose:** Transition from exploration (diverse sampling) to exploitation (focused sampling).

**Implementation:**
```python
# Temperature schedule: T(epoch) = T_init + progress × (T_final - T_init)
progress = min(1.0, epoch / temperature_decay_epochs)
temperature = initial_temperature + progress * (final_temperature - initial_temperature)
# Typically: 1.0 → 0.1 over 200 epochs

self.flow.set_temperature(temperature)
```

**Effect on Gumbel-Top-K:**
```python
perturbed_logits = (logits + gumbel_noise) / temperature
```

| Temperature | Behavior | Use Case |
|-------------|----------|----------|
| T = 1.0 (high) | High stochasticity, explores many configurations | Early training: discover state space |
| T = 0.5 (medium) | Balanced exploration/exploitation | Mid training: refine distribution |
| T = 0.1 (low) | Near-deterministic, focuses on highest-scored orbitals | Late training: concentrate on ground state |

**Mathematical intuition:**
- High T: Gumbel noise dominates → nearly uniform random selection
- Low T: Logits dominate → deterministic selection of highest-scored orbitals

---

## Technique 3: Neural Quantum State (NQS)

**Purpose:** Learn the ground state wavefunction amplitude |ψ(x)|.

### Architecture

```python
class DenseNQS:
    """
    Input:  configuration x ∈ {0,1}^n  (which orbitals are occupied)
    Output: log|ψ(x)|                   (log amplitude for numerical stability)
    """

    amplitude_net = Sequential(
        Linear(num_sites, 512), ReLU(),
        Linear(512, 512), ReLU(),
        Linear(512, 512), ReLU(),
        Linear(512, 512), ReLU(),
        Linear(512, 1),
        Tanh()  # Bounded output ∈ [-1, 1]
    )

    log_amp_scale = Parameter(1.0)  # Learnable scaling
```

### Key Design Choices

| Choice | Reason |
|--------|--------|
| **Log-amplitude output** | Avoids numerical overflow for small amplitudes |
| **Tanh final activation** | Bounds output to [-1, 1] for training stability |
| **Learnable scale** | Adjusts effective output range during training |
| **Deep network (4+ layers)** | Captures complex correlations in ground state |

### Probability Computation

```python
# NQS outputs log|ψ(x)|
log_amp = self.nqs.log_amplitude(configs)

# Born rule: P(x) = |ψ(x)|² = exp(2·log|ψ(x)|)
nqs_probs = torch.exp(2 * log_amp)

# Normalize over sampled configurations
nqs_probs = nqs_probs / nqs_probs.sum()
```

### Configuration Encoding

```python
def encode_configuration(self, x):
    """Convert binary occupation to float tensor."""
    return x.float()  # {0,1}^n → [0.0, 1.0]^n
```

---

## Technique 4: Local Energy Computation

**Purpose:** Compute the energy contribution of each configuration for importance weighting.

### Definition

The local energy at configuration x is:
$$E_{loc}(x) = \frac{\langle x|H|\psi\rangle}{\langle x|\psi\rangle}$$

Expanding in the computational basis:
$$E_{loc}(x) = H_{xx} + \sum_{x' \neq x} H_{xx'} \frac{\psi(x')}{\psi(x)}$$

where:
- $H_{xx}$ = diagonal element (one-body + two-body terms for config x)
- $H_{xx'}$ = off-diagonal elements (hopping/exchange terms)
- $\psi(x')/\psi(x)$ = amplitude ratio from NQS

### Implementation

```python
def _compute_local_energies(self, configs: torch.Tensor) -> torch.Tensor:
    n_configs = len(configs)
    local_energies = torch.zeros(n_configs, device=self.device)

    with torch.no_grad():
        # Diagonal elements: ⟨x|H|x⟩
        diag = self.hamiltonian.diagonal_elements_batch(configs)

        # NQS log-amplitudes for all configs
        log_psi = self.nqs.log_amplitude(configs.float())

        for i in range(n_configs):
            # Off-diagonal connections: which x' have ⟨x|H|x'⟩ ≠ 0
            connected, elements = self.hamiltonian.get_connections(configs[i])

            e_loc = diag[i]
            if len(connected) > 0:
                # Compute ψ(x')/ψ(x) = exp(log ψ(x') - log ψ(x))
                log_psi_connected = self.nqs.log_amplitude(connected.float())
                ratio = torch.exp(log_psi_connected - log_psi[i])

                # Sum: Σ H_xx' · ψ(x')/ψ(x)
                e_loc = e_loc + (elements * ratio).sum()

            local_energies[i] = e_loc.real

    return local_energies
```

### Physical Interpretation

| Property | Meaning |
|----------|---------|
| $\langle E_{loc} \rangle = E$ | Expected local energy equals variational energy |
| $\text{Var}(E_{loc}) = 0$ | Zero variance iff ψ is exact eigenstate |
| Low $E_{loc}(x)$ | Configuration x is important for ground state |

---

## Technique 5: Mixed-Objective Flow Loss

**Purpose:** Train the flow with three complementary signals to learn the ground state support.

### Loss Function

$$\mathcal{L}_{flow} = w_t \cdot L_{teacher} + w_p \cdot L_{physics} - w_e \cdot H(flow)$$

Typical weights: $w_t = 0.5$, $w_p = 0.4$, $w_e = 0.1$

### 5.1 Teacher Loss (KL Divergence)

**Goal:** Flow distribution should match NQS probability distribution.

$$L_{teacher} = D_{KL}(p_{NQS} \| p_{flow}) = \sum_x p_{NQS}(x) \log \frac{p_{NQS}(x)}{p_{flow}(x)}$$

```python
# Flow should match NQS probability distribution
teacher_loss = -torch.sum(nqs_probs.detach() * log_flow_probs)
```

**Interpretation:**
- If NQS assigns high probability to config x, flow should also sample x frequently
- Gradient flows to increase p_flow(x) where p_NQS(x) is high

### 5.2 Physics Loss (Energy Importance)

**Goal:** Flow should preferentially sample low-energy configurations.

$$L_{physics} = \mathbb{E}_{x \sim p_{flow}}[E_{loc}(x) - \bar{E}]$$

```python
# Use energy baseline for variance reduction
energy_deviation = local_energies - energy.detach()

# Physics loss = expected energy under flow distribution
physics_loss = (flow_probs * energy_deviation.detach()).sum()
```

**Interpretation:**
- Penalizes flow for assigning probability to high-energy configurations
- Baseline subtraction ($\bar{E}$) reduces gradient variance

### 5.3 Entropy Bonus (Exploration)

**Goal:** Prevent mode collapse by maintaining sampling diversity.

$$H(flow) = -\sum_x p_{flow}(x) \log p_{flow}(x)$$

```python
# Entropy = negative expected log probability
entropy = -torch.sum(flow_probs * log_flow_probs)
```

**Interpretation:**
- Maximizing entropy encourages spread-out distribution
- Prevents flow from collapsing to single configuration
- Balances exploitation (focus) with exploration (diversity)

### Combined Loss Implementation

```python
def _compute_flow_loss(self, ...):
    # Get flow probabilities
    flow_probs = self.flow.estimate_discrete_prob(unique_configs)
    flow_probs = flow_probs / (flow_probs.sum() + 1e-10)
    log_flow_probs = torch.log(flow_probs + 1e-10)

    # === Teacher Loss ===
    teacher_loss = -torch.sum(nqs_probs.detach() * log_flow_probs)

    # === Physics Loss ===
    energy_deviation = local_energies - energy.detach()
    physics_loss = (flow_probs * energy_deviation.detach()).sum()

    # === Entropy Bonus ===
    entropy = -torch.sum(flow_probs * log_flow_probs)

    # Combined loss (scaled by energy magnitude for stability)
    total_loss = (
        config.teacher_weight * teacher_loss +
        config.physics_weight * physics_loss -
        config.entropy_weight * entropy
    ) / (torch.abs(energy.detach()) + 1.0)

    return total_loss, {'teacher': teacher_loss, 'physics': physics_loss, 'entropy': entropy}
```

---

## Technique 6: NQS Energy Minimization (REINFORCE)

**Purpose:** Train NQS to represent the ground state by minimizing variational energy.

### Variational Principle

The variational energy is:
$$E[\psi] = \frac{\langle \psi | H | \psi \rangle}{\langle \psi | \psi \rangle} = \sum_x p(x) E_{loc}(x)$$

where $p(x) = |\psi(x)|^2 / \sum_{x'} |\psi(x')|^2$

### REINFORCE Gradient Estimator

The gradient of the energy with respect to NQS parameters θ:
$$\frac{\partial E}{\partial \theta} = 2 \cdot \text{Re}\left[ \left\langle (E_{loc} - \langle E \rangle) \frac{\partial \log \psi}{\partial \theta} \right\rangle \right]$$

### Implementation

```python
def _compute_nqs_loss(self, configs, probs, local_energies):
    # Compute log probabilities with gradients
    log_amp = self.nqs.log_amplitude(configs.float())
    log_probs = 2 * log_amp  # log|ψ|² = 2·log|ψ|

    # Baseline subtraction for variance reduction
    energy = (local_energies.detach() * probs.detach()).sum()
    centered_energies = local_energies.detach() - energy

    # Policy gradient loss (REINFORCE)
    loss = (centered_energies * log_probs * probs.detach()).sum()

    return loss
```

### Why Baseline Subtraction Works

Without baseline: high variance gradients due to E_loc fluctuations
With baseline:
- $\mathbb{E}[(\text{E}_{loc} - \bar{E}) \cdot \nabla \log \psi] = \mathbb{E}[E_{loc} \cdot \nabla \log \psi]$ (unbiased)
- But $\text{Var}[(E_{loc} - \bar{E}) \cdot \nabla \log \psi] \ll \text{Var}[E_{loc} \cdot \nabla \log \psi]$ (lower variance)

---

## Technique 7: Accumulated Basis Tracking

**Purpose:** Build a growing database of important configurations discovered during training.

### Why Accumulate?

- Training samples are transient (used once then discarded)
- Want to keep track of ALL good configurations discovered
- This basis feeds into Stage 2 (diversity selection) and Stage 3 (residual expansion)

### Implementation

```python
def _update_accumulated_basis(self, new_configs: torch.Tensor):
    if self.accumulated_basis is None:
        self.accumulated_basis = new_configs.clone()
    else:
        # Combine with existing basis
        combined = torch.cat([self.accumulated_basis, new_configs], dim=0)
        # Keep only unique configurations
        self.accumulated_basis = torch.unique(combined, dim=0)

    # Prune if exceeds maximum size
    max_size = self.config.max_accumulated_basis
    if len(self.accumulated_basis) > max_size:
        # Random pruning (could also use importance-based pruning)
        indices = torch.randperm(len(self.accumulated_basis))[:max_size]
        self.accumulated_basis = self.accumulated_basis[indices]
```

### Periodic Energy Evaluation

```python
def _compute_accumulated_energy(self) -> float:
    """Compute ground state energy in accumulated basis via exact diagonalization."""
    if self.accumulated_basis is None:
        return float('inf')

    with torch.no_grad():
        # Build Hamiltonian matrix in this subspace
        H_matrix = self.hamiltonian.matrix_elements(
            self.accumulated_basis, self.accumulated_basis
        )
        H_np = H_matrix.cpu().numpy()

        # Diagonalize to get ground state
        eigenvalues, _ = np.linalg.eigh(H_np)
        return float(eigenvalues[0])
```

**Use:** Tracked every N epochs to monitor convergence in subspace energy.

---

## Technique 8: Optimization Strategies

### 8.1 AdamW Optimizer

```python
flow_optimizer = torch.optim.AdamW(
    flow.parameters(),
    lr=5e-4,           # Learning rate
    weight_decay=1e-5  # L2 regularization (decoupled)
)
nqs_optimizer = torch.optim.AdamW(
    nqs.parameters(),
    lr=1e-3,           # NQS learns faster than flow
    weight_decay=1e-5
)
```

**Why AdamW over Adam:**
- Decoupled weight decay (not scaled by gradient magnitude)
- Better generalization, especially for neural networks
- Prevents overfitting to specific configurations

### 8.2 Cosine Annealing Learning Rate Schedule

```python
flow_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    flow_optimizer,
    T_max=num_epochs,  # Full cycle over training
    eta_min=1e-6       # Minimum learning rate
)
```

**Schedule:**
$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{t}{T_{max}}\pi\right)\right)$$

**Benefits:**
- Smooth decay avoids sudden learning rate drops
- Final low learning rate enables fine-tuning
- Often outperforms step decay

### 8.3 Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=1.0)
torch.nn.utils.clip_grad_norm_(nqs.parameters(), max_norm=1.0)
```

**Purpose:**
- Prevents exploding gradients from outlier configurations
- Stabilizes training when local energies have high variance
- max_norm=1.0 is a common conservative choice

---

## Technique 9: Exponential Moving Average (EMA) Tracking

**Purpose:** Smooth energy tracking for stable convergence monitoring.

```python
if self.energy_ema is None:
    self.energy_ema = current_energy
else:
    self.energy_ema = ema_decay * self.energy_ema + (1 - ema_decay) * current_energy
    # ema_decay = 0.95 → slow, stable tracking
```

**Properties:**
- Filters out epoch-to-epoch fluctuations
- Decay = 0.95 means ~20 epochs half-life
- Used for convergence detection, not training

---

## Technique 10: Convergence Detection

**Purpose:** Early stopping when flow has focused on ground state support.

### Metric: Unique Ratio

```python
unique_ratio = n_unique_configs / n_total_samples
```

| Unique Ratio | Interpretation |
|--------------|----------------|
| 0.8 - 1.0 | Flow still exploring, many distinct samples |
| 0.4 - 0.6 | Flow focusing, some repeated samples |
| 0.1 - 0.2 | Flow concentrated, most samples are repeats → **CONVERGED** |

### Convergence Check

```python
if epoch >= min_epochs:  # Don't stop too early
    if unique_ratio < convergence_threshold:  # e.g., 0.20
        print(f"Converged: flow focused on {n_unique} configurations")
        break
```

**Why this works:**
- Ground state has limited support (few important configurations)
- When flow learns this support, it repeatedly samples the same configs
- Low unique ratio indicates flow has "found" the ground state support

---

## Summary: Technique Interactions

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     TRAINING LOOP FLOW                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. SAMPLE from Flow (Gumbel-Top-K, temperature-annealed)              │
│         ↓                                                               │
│  2. EVALUATE NQS probabilities |ψ(x)|²                                 │
│         ↓                                                               │
│  3. COMPUTE local energies E_loc(x) = ⟨x|H|ψ⟩/⟨x|ψ⟩                   │
│         ↓                                                               │
│  4. UPDATE accumulated basis (unique configs)                          │
│         ↓                                                               │
│  5. COMPUTE Flow loss (teacher + physics - entropy)                    │
│         ↓                                                               │
│  6. COMPUTE NQS loss (REINFORCE with baseline)                         │
│         ↓                                                               │
│  7. BACKWARD pass + gradient clipping                                  │
│         ↓                                                               │
│  8. OPTIMIZER step (AdamW) + scheduler step (cosine)                   │
│         ↓                                                               │
│  9. CHECK convergence (unique ratio)                                   │
│         ↓                                                               │
│  10. REPEAT until converged or max_epochs                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Output of Stage 1

After training completes, Stage 1 produces:

| Output | Description | Used In |
|--------|-------------|---------|
| `accumulated_basis` | Tensor of discovered configurations | Stage 2, 3 |
| `nf_nqs_energy` | Best energy estimate from training | Comparison |
| `trained_flow` | Flow model (frozen after Stage 1) | Optional sampling |
| `trained_nqs` | NQS model (for amplitude evaluation) | Local energy |
| `training_history` | Loss curves, energies, unique ratios | Diagnostics |
