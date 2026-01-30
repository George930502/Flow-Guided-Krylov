# Module: Molecular Hamiltonian

## Overview

The Molecular Hamiltonian module provides the physical foundation for quantum chemistry calculations. It constructs the second-quantized Hamiltonian from molecular integrals and implements efficient operations for energy computation.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       MOLECULAR HAMILTONIAN MODULE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Input: Molecular geometry + basis set                                    │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  PySCF INTEGRATION                                                  │  │
│   │  • Build molecule from geometry                                     │  │
│   │  • Run Hartree-Fock calculation                                     │  │
│   │  • Extract one-body (h_pq) and two-body (g_pqrs) integrals         │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                              ↓                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  JORDAN-WIGNER TRANSFORMATION                                       │  │
│   │  • Map fermionic operators to qubit operators                       │  │
│   │  • Handle spin-orbital indexing                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                              ↓                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  HAMILTONIAN OPERATIONS                                             │  │
│   │  • Diagonal elements ⟨x|H|x⟩                                        │  │
│   │  • Off-diagonal connections ⟨x|H|y⟩                                 │  │
│   │  • Matrix element computation                                       │  │
│   │  • FCI energy calculation                                           │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Second Quantization Background

### Fermionic Operators

Creation and annihilation operators for electrons:
- $a_p^\dagger$: Create electron in spin-orbital p
- $a_p$: Annihilate electron in spin-orbital p

Anticommutation relations:
$$\{a_p, a_q^\dagger\} = \delta_{pq}, \quad \{a_p, a_q\} = 0, \quad \{a_p^\dagger, a_q^\dagger\} = 0$$

### Molecular Electronic Hamiltonian

$$H = \sum_{pq} h_{pq} a_p^\dagger a_q + \frac{1}{2} \sum_{pqrs} g_{pqrs} a_p^\dagger a_q^\dagger a_s a_r + E_{nuc}$$

where:
- $h_{pq}$ = one-body integrals (kinetic energy + nuclear attraction)
- $g_{pqrs}$ = two-body integrals (electron-electron repulsion)
- $E_{nuc}$ = nuclear repulsion energy

---

## Jordan-Wigner Transformation

### Purpose

Map fermionic operators to qubit (Pauli) operators:
$$a_p^\dagger \rightarrow \frac{1}{2}(X_p - iY_p) \prod_{q<p} Z_q$$

### Spin-Orbital Indexing

For a system with n spatial orbitals:
- Spin-orbitals 0 to n-1: Alpha spin (↑)
- Spin-orbitals n to 2n-1: Beta spin (↓)

```python
def _spin_orbital_index(self, spatial_orbital: int, spin: str) -> int:
    """Convert spatial orbital + spin to spin-orbital index."""
    if spin == 'alpha':
        return spatial_orbital
    else:  # beta
        return spatial_orbital + self.n_orbitals
```

### Configuration Representation

A configuration is a binary string of length 2n:
```
Config: [1, 1, 0, 0, 1, 1, 0, 0]
         ├───────┤  ├───────┤
          Alpha      Beta

Meaning: Electrons in orbitals 0,1 (alpha) and 0,1 (beta)
```

---

## Key Operations

### Diagonal Element Computation

$$\langle x|H|x\rangle = \sum_{p \in \text{occ}} h_{pp} + \frac{1}{2}\sum_{p,q \in \text{occ}} (g_{ppqq} - g_{pqqp}) + E_{nuc}$$

```python
def diagonal_element(self, config: torch.Tensor) -> torch.Tensor:
    """
    Compute diagonal Hamiltonian element ⟨x|H|x⟩.

    For occupied orbitals, sums:
    - One-body terms: h_pp (kinetic + nuclear)
    - Two-body terms: g_ppqq - g_pqqp (Coulomb - exchange)
    - Nuclear repulsion: E_nuc
    """
    # Find occupied spin-orbitals
    occupied = config.nonzero(as_tuple=True)[0]

    energy = self.nuclear_repulsion

    # One-body contribution
    for p in occupied:
        energy = energy + self.h1[p, p]

    # Two-body contribution (Coulomb - exchange)
    for i, p in enumerate(occupied):
        for q in occupied[i+1:]:
            # Coulomb: g_ppqq
            energy = energy + self.h2[p, p, q, q]
            # Exchange: -g_pqqp (only for same spin)
            if self._same_spin(p, q):
                energy = energy - self.h2[p, q, q, p]

    return energy
```

### Off-Diagonal Connections

For fermionic Hamiltonians, ⟨x|H|y⟩ ≠ 0 only when x and y differ by at most 2 spin-orbitals (single or double excitation).

```python
def get_connections(self, config: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find all configurations connected to input via Hamiltonian.

    Returns configurations that differ by:
    - Single excitation: one electron moved (h_pq term)
    - Double excitation: two electrons moved (g_pqrs term)
    """
    connected = []
    elements = []

    occupied = config.nonzero(as_tuple=True)[0].tolist()
    unoccupied = (config == 0).nonzero(as_tuple=True)[0].tolist()

    # Single excitations: p (occupied) → q (unoccupied)
    for p in occupied:
        for q in unoccupied:
            if self._same_spin_idx(p, q):  # Must conserve spin
                # Compute matrix element
                h_elem = self._single_excitation_element(config, p, q)
                if abs(h_elem) > 1e-12:
                    new_config = config.clone()
                    new_config[p] = 0
                    new_config[q] = 1
                    connected.append(new_config)
                    elements.append(h_elem)

    # Double excitations: (p,q) → (r,s)
    for i, p in enumerate(occupied):
        for q in occupied[i+1:]:
            for j, r in enumerate(unoccupied):
                for s in unoccupied[j+1:]:
                    if self._valid_double_excitation(p, q, r, s):
                        h_elem = self._double_excitation_element(config, p, q, r, s)
                        if abs(h_elem) > 1e-12:
                            new_config = config.clone()
                            new_config[p] = 0
                            new_config[q] = 0
                            new_config[r] = 1
                            new_config[s] = 1
                            connected.append(new_config)
                            elements.append(h_elem)

    if connected:
        return torch.stack(connected), torch.tensor(elements)
    else:
        return torch.empty(0, len(config)), torch.empty(0)
```

### Fermionic Sign (Parity)

Electron exchanges introduce sign changes due to antisymmetry:

```python
def _compute_parity(self, config: torch.Tensor, p: int, q: int) -> int:
    """
    Compute fermionic parity for moving electron from p to q.

    Sign = (-1)^(number of electrons between p and q)
    """
    if p < q:
        between = config[p+1:q].sum()
    else:
        between = config[q+1:p].sum()

    return (-1) ** int(between.item())
```

### Jordan-Wigner Sign for Double Excitations

**Critical:** For double excitations a⁺ₚ a⁺ᵣ aₛ aᵧ, operators are applied **RIGHT-TO-LEFT** in second quantization. This means:

1. **a_q first** (annihilate q) - operates on original configuration
2. **a_s second** (annihilate s) - configuration now has q removed
3. **a⁺_r third** (create r) - configuration has q,s removed
4. **a⁺_p fourth** (create p) - configuration has q,s removed, r added

Each JW string counts occupied sites to the LEFT, accounting for prior modifications:

```python
def _jw_sign_double_np(self, config, p, r, q, s):
    """JW sign for a+_p a+_r a_s a_q applied RIGHT-TO-LEFT."""
    total_count = 0

    # 1. a_q (first): count on original config
    total_count += config[:q].sum()

    # 2. a_s (second): q has been removed
    count_s = config[:s].sum()
    if q < s:
        count_s -= 1  # q was occupied, now empty
    total_count += count_s

    # 3. a+_r (third): q,s removed
    count_r = config[:r].sum()
    if q < r: count_r -= 1
    if s < r: count_r -= 1
    total_count += count_r

    # 4. a+_p (fourth): q,s removed, r added
    count_p = config[:p].sum()
    if q < p: count_p -= 1
    if s < p: count_p -= 1
    if r < p: count_p += 1  # r now occupied
    total_count += count_p

    return (-1) ** int(total_count)
```

**Why This Matters:** Incorrect operator ordering violates the variational principle (E_computed < E_exact), producing unphysical energies below the ground state.

---

## FCI Energy Calculation

### Full Configuration Interaction

FCI provides the exact ground state within the given basis set by diagonalizing H in the complete N-electron space.

### Efficient Subspace Diagonalization

```python
def fci_energy(self) -> float:
    """
    Compute FCI energy by diagonalizing in particle-conserving subspace.

    Much faster than full Hilbert space:
    - NH3 (16 qubits): 3,136 × 3,136 instead of 65,536 × 65,536
    """
    from itertools import combinations
    from scipy.sparse.linalg import eigsh

    n_orb = self.n_orbitals
    n_alpha = self.n_alpha
    n_beta = self.n_beta

    # Generate all valid determinants
    alpha_configs = list(combinations(range(n_orb), n_alpha))
    beta_configs = list(combinations(range(n_orb), n_beta))

    n_det = len(alpha_configs) * len(beta_configs)
    print(f"FCI: {n_det:,} determinants")

    # Build configuration list
    configs = []
    for alpha_occ in alpha_configs:
        for beta_occ in beta_configs:
            config = torch.zeros(self.num_sites, dtype=torch.long)
            for i in alpha_occ:
                config[i] = 1
            for i in beta_occ:
                config[i + n_orb] = 1
            configs.append(config)

    configs = torch.stack(configs)

    # Build Hamiltonian matrix in this subspace
    H_fci = self.matrix_elements(configs, configs)

    # Diagonalize
    if n_det < 1000:
        eigenvalues, _ = np.linalg.eigh(H_fci.cpu().numpy())
    else:
        H_sparse = scipy.sparse.csr_matrix(H_fci.cpu().numpy())
        eigenvalues, _ = eigsh(H_sparse, k=1, which='SA')

    return float(eigenvalues[0].real)
```

---

## PySCF Integration

### Computing Molecular Integrals

```python
def compute_molecular_integrals(geometry, basis="sto-3g"):
    """
    Compute one-body and two-body integrals using PySCF.

    Returns:
        h1: (2n, 2n) one-body integrals in spin-orbital basis
        h2: (2n, 2n, 2n, 2n) two-body integrals in spin-orbital basis
        nuclear_repulsion: float
        n_electrons: int
        n_orbitals: int
    """
    from pyscf import gto, scf, ao2mo

    # Build molecule
    mol = gto.Mole()
    mol.atom = geometry
    mol.basis = basis
    mol.build()

    # Run Hartree-Fock
    mf = scf.RHF(mol)
    mf.kernel()

    # Extract integrals in MO basis
    n_orb = mol.nao
    h1_ao = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    h1_mo = mf.mo_coeff.T @ h1_ao @ mf.mo_coeff

    # Two-electron integrals
    eri_ao = mol.intor('int2e')
    eri_mo = ao2mo.kernel(mol, mf.mo_coeff)
    eri_mo = ao2mo.restore(1, eri_mo, n_orb)  # (n,n,n,n)

    # Convert to spin-orbital basis
    h1_spin = np.zeros((2*n_orb, 2*n_orb))
    h1_spin[:n_orb, :n_orb] = h1_mo  # Alpha-alpha
    h1_spin[n_orb:, n_orb:] = h1_mo  # Beta-beta

    # Two-body: need to handle spin properly
    h2_spin = np.zeros((2*n_orb, 2*n_orb, 2*n_orb, 2*n_orb))
    # ... (spin-block assignment)

    return {
        'h1': h1_spin,
        'h2': h2_spin,
        'nuclear_repulsion': mol.energy_nuc(),
        'n_electrons': mol.nelectron,
        'n_orbitals': n_orb,
        'n_alpha': mol.nelectron // 2,
        'n_beta': mol.nelectron // 2,
    }
```

---

## Pre-defined Molecules

### H₂ (Hydrogen Molecule)

```python
def create_h2_hamiltonian(bond_length=0.74):
    """
    Create H2 Hamiltonian.

    - 2 electrons, 2 orbitals → 4 qubits
    - Valid configurations: C(2,1) × C(2,1) = 4
    - FCI energy (STO-3G, 0.74 Å): -1.1373 Ha
    """
    geometry = [
        ("H", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.0, bond_length)),
    ]
    integrals = compute_molecular_integrals(geometry, basis="sto-3g")
    return MolecularHamiltonian(**integrals)
```

### LiH (Lithium Hydride)

```python
def create_lih_hamiltonian(bond_length=1.6):
    """
    Create LiH Hamiltonian.

    - 4 electrons, 6 orbitals → 12 qubits
    - Valid configurations: C(6,2) × C(6,2) = 225
    - FCI energy (STO-3G, 1.6 Å): -7.8837 Ha
    """
    geometry = [
        ("Li", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.0, bond_length)),
    ]
    integrals = compute_molecular_integrals(geometry, basis="sto-3g")
    return MolecularHamiltonian(**integrals)
```

### H₂O (Water)

```python
def create_h2o_hamiltonian():
    """
    Create H2O Hamiltonian.

    - 10 electrons, 7 orbitals → 14 qubits
    - Valid configurations: C(7,5) × C(7,5) = 441
    - FCI energy (STO-3G): -75.0129 Ha
    """
    angle_rad = np.radians(104.5)
    oh_length = 0.96
    geometry = [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (oh_length, 0.0, 0.0)),
        ("H", (oh_length * np.cos(angle_rad), oh_length * np.sin(angle_rad), 0.0)),
    ]
    integrals = compute_molecular_integrals(geometry, basis="sto-3g")
    return MolecularHamiltonian(**integrals)
```

### Larger Molecules

| Molecule | Electrons | Orbitals | Qubits | Valid Configs | Use Case |
|----------|-----------|----------|--------|---------------|----------|
| BeH₂ | 6 | 7 | 14 | 1,225 | Small test |
| NH₃ | 10 | 8 | 16 | 3,136 | Medium test |
| N₂ | 14 | 10 | 20 | 14,400 | Large test |
| CH₄ | 10 | 9 | 18 | 15,876 | Large test |

---

## Hartree-Fock Reference State

### Definition

The Hartree-Fock state is the best single-determinant approximation:
$$|\text{HF}\rangle = a_{n_\alpha-1}^\dagger \cdots a_1^\dagger a_0^\dagger \cdot a_{n+n_\beta-1}^\dagger \cdots a_{n+1}^\dagger a_n^\dagger |0\rangle$$

Electrons fill the lowest-energy orbitals.

### Implementation

```python
def get_hf_state(self) -> torch.Tensor:
    """
    Get Hartree-Fock reference state configuration.

    Fills lowest n_alpha alpha orbitals and lowest n_beta beta orbitals.
    """
    config = torch.zeros(self.num_sites, dtype=torch.long, device=self.device)

    # Fill alpha orbitals (0 to n_alpha-1)
    for i in range(self.n_alpha):
        config[i] = 1

    # Fill beta orbitals (n_orb to n_orb + n_beta - 1)
    for i in range(self.n_beta):
        config[self.n_orbitals + i] = 1

    return config
```

### Example

For LiH with 4 electrons (2α + 2β) in 6 orbitals:
```
HF state: [1, 1, 0, 0, 0, 0,  1, 1, 0, 0, 0, 0]
           ├──────────────┤  ├──────────────┤
           Alpha orbitals    Beta orbitals
           (0,1 occupied)    (0,1 occupied)
```

---

## Performance Considerations

### Sparse vs Dense Operations

| Operation | Dense | Sparse | Recommendation |
|-----------|-------|--------|----------------|
| Small basis (<500) | O(n³) | O(nnz × k) | Dense |
| Large basis (>500) | O(n³) | O(nnz × k) | Sparse |
| Matrix construction | O(n²) | O(nnz) | Sparse |

### Memory Scaling

| System | Full Hilbert | Subspace | Memory (float64) |
|--------|--------------|----------|------------------|
| H₂ | 16 | 4 | ~128 B |
| LiH | 4,096 | 225 | ~400 KB |
| H₂O | 16,384 | 441 | ~1.5 MB |
| NH₃ | 65,536 | 3,136 | ~75 MB |
| N₂ | 1,048,576 | 14,400 | ~1.6 GB |

---

## Summary

The Molecular Hamiltonian module provides:

1. **Second-quantized Hamiltonian construction** from molecular integrals
2. **Jordan-Wigner transformation** for qubit representation
3. **Efficient diagonal/off-diagonal** element computation
4. **FCI energy calculation** in particle-conserving subspace
5. **PySCF integration** for molecular integral computation
6. **Pre-defined molecules** for benchmarking

This module is the physical foundation upon which all pipeline stages operate.
