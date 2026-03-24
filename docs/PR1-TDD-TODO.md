# PR #1: GPU Diag + Bug Fixes + Vectorize — TDD TODO List

> **Target repo**: `George930502/Flow-Guided-Krylov` branch `hi-nqs-sqd`
> **Working branch**: `feat/gpu-diag-skqd-fixes` (from `hi-nqs-sqd`)
> **TDD 原則**: Red → Green → Refactor，每步都先寫失敗的測試

---

## Phase 0: 環境準備

### 0.1 建立工作 branch
- [ ] `git checkout -b feat/gpu-diag-skqd-fixes`

### 0.2 建立 venv + 安裝依賴
- [ ] `python3 -m venv .venv && source .venv/bin/activate`
- [ ] `pip install -e ".[dev,cuda]"` (包含 pytest, cupy)
- [ ] `pip install qiskit-addon-sqd` (optional, for comparison tests)
- [ ] 驗證: `python -c "import torch; print(torch.__version__)"`

### 0.3 建立測試基礎設施
- [ ] 建立 `tests/` 目錄
- [ ] 建立 `tests/conftest.py` — 共用 fixtures (small molecular systems)
- [ ] 建立 `pyproject.toml` pytest config section
- [ ] 驗證: `pytest --co` 可以收集到測試

---

## Phase 1a: Vectorize Format Conversion（~2h）

### 目標
將 `_configs_to_ibm_format()` 和 `_ibm_format_to_configs()` 從 Python double-loop 改為 numpy vectorized。

### Red: 寫失敗的測試
- [ ] `tests/test_format_conversion.py`
  ```
  class TestConfigsToIBMFormat:
      test_single_config_h2()           # 最小系統 (2 orbitals)
      test_batch_configs_lih()          # 多 configs (6 orbitals)
      test_roundtrip_identity()         # to_ibm -> from_ibm == original
      test_matches_original_python()    # 新版 output == 舊版 output
      test_large_batch_performance()    # 10K configs < 0.1s (舊版 ~1s)

  class TestIBMFormatToConfigs:
      test_single_config_h2()
      test_batch_configs_lih()
      test_roundtrip_identity()
      test_matches_original_python()
  ```

### Green: 最小實作
- [ ] 在 `src/methods/hi_nqs_sqd.py` 中，重寫 `_configs_to_ibm_format()`:
  ```python
  # OLD (lines 351-364): double Python loop
  # NEW: numpy flip
  bs[:, :n_orb] = np.flip(configs_np[:, :n_orb], axis=1)
  bs[:, n_orb:] = np.flip(configs_np[:, n_orb:], axis=1)
  ```
- [ ] 重寫 `_ibm_format_to_configs()` (lines 367-375) 同理
- [ ] 驗證: `pytest tests/test_format_conversion.py -v` 全 PASS

### Refactor
- [ ] 將 vectorized 版本抽成 standalone utility function（方便其他 methods 使用）
- [ ] 加 docstring + type hints

---

## Phase 1b: Vectorize Dedup（~1h）

### 目標
將 cumulative basis dedup 從 Python `set(tuple(row.tolist()))` 改為 numpy/torch vectorized。

### Red: 寫失敗的測試
- [ ] `tests/test_dedup.py`
  ```
  class TestVectorizedDedup:
      test_basic_dedup()                # 去除重複 configs
      test_no_duplicates_unchanged()    # 無重複時不變
      test_preserves_order()            # 保持原始順序（重要！）
      test_large_batch_performance()    # 50K configs < 0.05s
      test_matches_original_python()    # 新版 == 舊版結果
  ```

### Green: 最小實作
- [ ] 新增 `_vectorized_dedup(existing_bs, new_bs)` 函數
  ```python
  # 方案: 用 numpy structured array + np.unique
  combined = np.vstack([existing_bs, new_bs])
  _, idx = np.unique(combined.view(np.void(combined.itemsize * combined.shape[1])),
                     return_index=True)
  # 保留 new_bs 中不在 existing_bs 的 rows
  ```
- [ ] 替換 `hi_nqs_sqd.py` lines 153-158 的 Python loop
- [ ] 驗證: `pytest tests/test_dedup.py -v` 全 PASS

### Refactor
- [ ] 提取為 utility function

---

## Phase 1c: SKQD Bug Fixes（~2h）

### 目標
修復他們 `krylov/skqd.py` 中已知的 5 個 bugs。

### Red: 寫失敗的測試
- [ ] `tests/test_skqd_fixes.py`
  ```
  class TestOOMGuards:
      test_max_full_subspace_size_blocks_large_system()  # >15K configs 不枚舉
      test_sparse_threshold_3000()                       # n>=3000 用 sparse
      test_dense_fallback_refused_above_5000()           # n>5000 不 fallback dense

  class TestGetCombinedBasis:
      test_includes_zero_amplitude_configs()  # Krylov-expanded configs 不被丟棄
      test_nf_guided_basis_preserved()        # _nf_guided_basis 完整保留

  class TestRegularizationShift:
      test_regularization_subtracted()        # eigenvalue 扣除 reg shift

  class TestSparseEigsh:
      test_sparse_eigsh_at_3000_configs()     # threshold=3000 觸發 sparse
      test_matches_dense_result()             # sparse 結果 ≈ dense 結果
  ```

### Green: 最小實作
- [ ] 在 `krylov/skqd.py` 加 `MAX_FULL_SUBSPACE_SIZE = 15000`
- [ ] 修改 `_setup_particle_conserving_subspace()`: 加 size guard
- [ ] 修改 sparse threshold 從 100 → 3000
- [ ] 修改 `compute_ground_state_energy()`: 扣除 regularization shift
- [ ] 修改 `get_combined_basis()`: 使用 `_nf_guided_basis` 如果存在
- [ ] 驗證: `pytest tests/test_skqd_fixes.py -v` 全 PASS

### Refactor
- [ ] 加 `_rank_and_truncate_basis()` for essential config 保護
- [ ] 加 warning log 當 OOM guard 觸發時

---

## Phase 1d: GPU Diag Adapter（~4h）

### 目標
建立 `solve_fermion` 的 GPU drop-in replacement。

### Red: 寫失敗的測試
- [ ] `tests/test_gpu_diag.py`
  ```
  class TestGPUSolveFermion:
      test_h2_energy_matches_exact()         # H2 (4Q): 精確 FCI
      test_lih_energy_matches_solve_fermion() # LiH (12Q): 對比 IBM 結果
      test_h2o_energy_matches_solve_fermion() # H2O (14Q): 對比 IBM 結果
      test_returns_correct_types()            # energy=float, v0=ndarray, occ=ndarray
      test_occupancies_sum_to_nelec()         # sum(occ) = n_alpha + n_beta
      test_occupancies_match_solve_fermion()  # occ ≈ IBM's occ (tolerance 1e-6)
      test_sparse_path_above_3000()           # >3000 configs 用 sparse eigsh
      test_dense_path_below_3000()            # <3000 configs 用 torch.linalg.eigh
      test_oom_guard_at_10000()               # >10000 configs 不做 dense
      test_full_basis_better_than_batch()     # E(full) <= min(E(batch_i))
      test_gpu_if_available()                 # CUDA 可用時在 GPU 上跑
      test_cpu_fallback()                     # 無 GPU 時 fallback CPU

  class TestOrbitalOccupancies:
      test_hf_state_correct_occupancies()     # HF state: occ[i]=1 for occupied
      test_occupancies_from_eigenvector()     # occ = sum |c_i|^2 * n_p(x_i)
      test_matches_sqd_solver_computation()   # 對比 SQDSolver._compute_orbital_occupancies
  ```

### Green: 最小實作
- [ ] 新增 `src/utils/gpu_diag.py`
  ```python
  def gpu_solve_fermion(configs, hamiltonian, max_dense=10000):
      """Drop-in replacement for qiskit solve_fermion.

      Args:
          configs: torch.Tensor (n_configs, 2*n_orb) — our format (NOT IBM format)
          hamiltonian: MolecularHamiltonian
          max_dense: int — above this, use sparse eigsh

      Returns:
          (energy, eigenvector, occupancies) matching solve_fermion interface
      """
  ```
- [ ] 實作 sparse/dense dispatch
- [ ] 實作 `_compute_occupancies_from_eigvec(configs, v0)`
- [ ] 驗證: `pytest tests/test_gpu_diag.py -v` 全 PASS

### Refactor
- [ ] 加 CuPy GPU sparse eigsh path (if cupy available)
- [ ] 加 memory estimation + warning
- [ ] 加 timing instrumentation

---

## Phase 1e: 整合到 HI-NQS-SQD（~3h）

### 目標
在 `hi_nqs_sqd.py` 中加 `use_gpu_diag` 選項。

### Red: 寫失敗的測試
- [ ] `tests/test_hi_nqs_integration.py`
  ```
  class TestHINQSSKQDIntegration:
      test_gpu_diag_h2_matches_solve_fermion()  # H2: gpu_diag ≈ solve_fermion
      test_gpu_diag_lih_converges()              # LiH: 收斂 + 正確能量
      test_no_batching_with_gpu_diag()           # gpu_diag 模式不做 5-batch
      test_full_basis_diag()                     # 全 basis 單次對角化
      test_eigvec_weights_used_for_nqs()         # |c_i|^2 作為 NQS weights
      test_occupancies_feed_config_recovery()    # occ 正確傳遞給 recover_configurations
      test_fallback_to_solve_fermion()           # use_gpu_diag=False 走原路
      test_convergence_not_degraded()            # 收斂速度 >= 原版

  class TestEndToEnd:
      test_h2_full_pipeline()                    # H2 E2E: < 0.1 mHa error
      test_lih_full_pipeline()                   # LiH E2E: < 0.5 mHa error
  ```

### Green: 最小實作
- [ ] `HINQSSQDConfig` 加 `use_gpu_diag: bool = True` 欄位
- [ ] `HINQSSQDConfig` 加 `use_eigvec_weights: bool = True` 欄位
- [ ] 修改 `run_hi_nqs_sqd()` lines 197-241:
  ```python
  if cfg.use_gpu_diag:
      # 單次全 basis 對角化（不做 5-batch）
      configs_tensor = _ibm_format_to_configs(cumulative_bs, n_orb, n_qubits)
      e0, v0, occ = gpu_solve_fermion(configs_tensor, hamiltonian)
      e0 = e0 + nuclear_repulsion  # 加核排斥
  else:
      # 原始 5-batch solve_fermion path
      ...
  ```
- [ ] 修改 `_update_nqs_from_sqd()` lines 307-311:
  ```python
  if cfg.use_eigvec_weights and eigvec is not None:
      weights = torch.from_numpy(eigvec ** 2).float().to(device)
      weights = weights / weights.sum()
  else:
      # 原始 softmax(-advantage) weights
      ...
  ```
- [ ] 驗證: `pytest tests/test_hi_nqs_integration.py -v` 全 PASS

### Refactor
- [ ] 清理 unused imports (如果 use_gpu_diag=True 不需要 qiskit)
- [ ] 加 timing log 對比 solve_fermion vs gpu_diag
- [ ] 更新 docstring

---

## Phase 1f: Benchmark + 文件（~1h）

### 目標
驗證加速效果，寫 PR description。

- [ ] 建立 `scripts/benchmark_gpu_diag.py`
  - 對比 solve_fermion vs gpu_diag on LiH/H2O/BeH2/NH3/CH4/N2
  - 輸出: energy diff, time diff, speedup ratio
- [ ] 執行 benchmark，記錄結果到 `results/gpu_diag_benchmark.json`
- [ ] 寫 PR description:
  - Summary: GPU-accelerated diag + SKQD bug fixes + vectorized format conversion
  - Benchmark results table
  - Breaking changes: None (新功能默認 on，可 fallback)

---

## 測試執行順序

```bash
# Phase 1a
pytest tests/test_format_conversion.py -v

# Phase 1b
pytest tests/test_dedup.py -v

# Phase 1c
pytest tests/test_skqd_fixes.py -v

# Phase 1d
pytest tests/test_gpu_diag.py -v

# Phase 1e
pytest tests/test_hi_nqs_integration.py -v

# 全部通過
pytest tests/ -v

# Benchmark
python scripts/benchmark_gpu_diag.py
```

---

## 依賴關係圖

```
Phase 0 (環境) ─┬─→ Phase 1a (format conversion)
                ├─→ Phase 1b (dedup)         ──→ Phase 1e (integration)
                ├─→ Phase 1c (SKQD fixes)    ──→ Phase 1e (integration)
                └─→ Phase 1d (GPU diag)      ──→ Phase 1e (integration)
                                                      │
                                                      ↓
                                               Phase 1f (benchmark + PR)
```

Phase 1a, 1b, 1c, 1d 可以**並行開發**（互不依賴）。
Phase 1e 需要等 1a-1d 全部完成。
Phase 1f 需要等 1e 完成。

---

## 預估總時間: ~13h

| Phase | 時間 | Red | Green | Refactor |
|-------|------|-----|-------|----------|
| 0 | 1h | — | setup | — |
| 1a | 2h | 30m | 1h | 30m |
| 1b | 1h | 20m | 30m | 10m |
| 1c | 2h | 30m | 1h | 30m |
| 1d | 4h | 1h | 2h | 1h |
| 1e | 3h | 1h | 1.5h | 30m |
| 1f | 1h | — | 1h | — |
