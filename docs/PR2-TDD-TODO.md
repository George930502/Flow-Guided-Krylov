# PR #2: SKQD Integration — TDD TODO List

> **Branch**: `feat/skqd-integration` (from `feat/gpu-diag-skqd-fixes`)
> **Strategy**: Lightweight Krylov expansion module + new HI+NQS+SKQD method
> **不 port 整個 2159 行 SKQD**，而是新增精簡的 connection expansion + post-Krylov merge

---

## Phase 2a: Hamiltonian Connection Expansion（~4h）

### 目標
新增 `src/utils/krylov_expand.py` — 從 basis 出發，用 Hamiltonian connections 擴展 basis。

### Red
- [ ] `tests/test_krylov_expand.py`
  - `test_h2_expansion_finds_all_configs()`
  - `test_lih_expansion_discovers_new_configs()`
  - `test_expansion_respects_max_new()`
  - `test_expansion_dedup_against_existing()`
  - `test_expansion_preserves_particle_number()`
  - `test_hf_seed_expands_to_singles_doubles()`

### Green
- [ ] `src/utils/krylov_expand.py`
  - `expand_basis_via_connections(basis, hamiltonian, max_new, n_ref)`

### Refactor
- [ ] Add H-coupling ranking for expansion priority

---

## Phase 2b: HI+NQS+SKQD Method（~4h）

### 目標
新增 `src/methods/hi_nqs_skqd.py` — HI+NQS+SKQD with post-Krylov merge。

### Red
- [ ] `tests/test_hi_nqs_skqd.py`
  - `test_h2_energy_exact()`
  - `test_lih_energy_below_hf()`
  - `test_krylov_expansion_improves_energy()`
  - `test_post_merge_includes_both_nqs_and_krylov()`
  - `test_eigvec_weights_used()`
  - `test_convergence()`

### Green
- [ ] `src/methods/hi_nqs_skqd.py`
  - Post-Krylov merge: SKQD from HF → expand → merge with NQS → diag

---

## Phase 2c: E_SCI Evaluation（~1h）

- [ ] Add E_SCI metric to SolverResult metadata
- [ ] Compare E_SCI vs E_NQS in benchmark
