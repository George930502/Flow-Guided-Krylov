# HI+NQS Pipeline 優化報告

> **日期**: 2026-03-24
> **作者**: thc1006 (秀吉)
> **專案**: George930502/Flow-Guided-Krylov `hi-nqs-sqd` branch
> **硬體**: NVIDIA DGX Spark (GB10, 128GB UMA, CUDA 13.0)
> **PRs**: #3 (GPU diag), #4 (Krylov expansion), #5 (KV cache + 消融)
> **測試**: 91/91 通過

---

## 一、原始 HI+NQS+SQD 的成果與瓶頸

原始 HI+NQS+SQD 在 ≤30Q STO-3G 上表現優秀（數據來自 README.md）：

| 分子 | Q | err vs FCI (mHa) | Basis | Time |
|------|---|------------------|-------|------|
| LiH | 12 | 0.000 | 225 | 3.9s |
| H₂O | 14 | 0.000 | 441 | 1.7s |
| BeH₂ | 14 | 0.000 | 1,200 | 2.1s |
| NH₃ | 16 | **-0.126** ⚠️ | 3,074 | 5.5s |
| CH₄ | 18 | 0.000 | 11,129 | 54s |
| N₂ | 20 | 0.000 | 12,632 | 44s |
| C₂H₄ | 28 | -0.141 (vs SCI) | 10,000 | **485s** |

**三個瓶頸：**
1. IBM `solve_fermion`（CPU PySCF SCI）在 28Q 要 485s，40Q+ 不可行
2. NQS sampling 無 KV cache → O(n³) 複雜度
3. 無物理引導的 basis expansion

---

## 二、我們做的 5 項優化

| # | 優化 | PR | 說明 |
|---|------|-----|------|
| 1 | GPU 對角化 | #3 | `gpu_solve_fermion()` 取代 IBM `solve_fermion` |
| 2 | Krylov 擴展 | #4 | `expand_basis_via_connections()` 2-hop H-connection |
| 3 | KV Cache | #5 | Transformer sampling O(n³)→O(n²) |
| 4 | Vectorized 工具 | #3 | Format conversion + dedup 從 Python loop → numpy |
| 5 | SKQD Bug Fixes | #3 | OOM guard + sparse threshold + regularization |

---

## 三、實驗結果

### 實驗 1: GPU diag 精度驗證

**目的**：確認 `gpu_solve_fermion` 和 IBM `solve_fermion` 在相同 basis 上給出相同能量。

**數據來源**：`results/gpu_diag_benchmark.json`

| 分子 | Q | Full Basis | IBM Energy (Ha) | GPU Energy (Ha) | 能量差 (mHa) | IBM Time | GPU Time |
|------|---|-----------|-----------------|-----------------|-------------|----------|----------|
| H₂ | 4 | 4 | -1.13728383 | -1.13728383 | **0.000** | 0.004s | 0.025s |
| LiH | 12 | 225 | -7.88232438 | -7.88232438 | **0.000** | 0.032s | 0.024s |
| H₂O | 14 | 441 | -75.01315470 | -75.01315470 | **0.000** | 0.050s | 0.027s |
| BeH₂ | 14 | 1,225 | -15.59511756 | -15.59511756 | **0.000** | 0.049s | 0.102s |
| NH₃ | 16 | 3,136 | -55.5178**1606** | -55.5176**9008** | **0.126** | 0.043s | 0.348s |

**結論**：
- H₂ ~ BeH₂：兩者能量**完全一致**（差 0.000 mHa），確認 GPU diag 正確性
- NH₃：IBM 給出 -55.51782（比 FCI 低 0.126 mHa = **variational violation**），我們給出 -55.51769（= 精確 FCI）
- **NH₃ 的 -0.126 mHa anomaly 源自 IBM 的 `spin_sq=0` 自旋投影，不是我們的 bug**

### 實驗 2: 完整 E2E Pipeline（小系統 ≤20Q）

**目的**：驗證優化後的完整 NQS → Krylov → GPU diag → feedback 迴路在小系統上收斂。

**數據來源**：`results/benchmark_e2e_hi_nqs_skqd.json`

| 分子 | Q | FCI Energy (Ha) | 我們的 Energy (Ha) | err vs FCI (mHa) | Basis | Iters | Time | 收斂？ |
|------|---|----------------|-------------------|------------------|-------|-------|------|--------|
| H₂ | 4 | -1.13728383 | -1.13728383 | **0.00** | 4 | 4 | 0.9s | ✓ |
| LiH | 12 | -7.88232438 | -7.88232438 | **0.00** | 220 | 4 | 0.7s | ✓ |
| H₂O | 14 | -75.01315470 | -75.01315470 | **0.00** | 423 | 4 | 0.9s | ✓ |
| BeH₂ | 14 | -15.59511756 | -15.59511752 | **0.00** | 788 | 4 | 1.4s | ✓ |
| NH₃ | 16 | -55.51769008 | -55.51709769 | **0.59** | 1,916 | 6 | 9.4s | ✓ |
| N₂ | 20 | -107.65412245 | -107.65320057 | **0.92** | 4,385 | 8 | 37.6s | ✓ |

**結論**：
- **4/6 系統達到精確 FCI**（H₂、LiH、H₂O、BeH₂：0.00 mHa）
- **2/6 系統達到 sub-化學精度**（NH₃ 0.59 mHa、N₂ 0.92 mHa，均 < 1.6 mHa 門檻）
- **全部 6 系統收斂**
- NH₃ 的 0.59 mHa 是**正確的 variational 結果**（原版 -0.126 mHa 是 violation）

### 與原始 HI+NQS+SQD 對比

| 分子 | Q | 原始 err (mHa) | **我們 err (mHa)** | 原始 Time | **我們 Time** |
|------|---|---------------|-------------------|-----------|-------------|
| LiH | 12 | 0.000 | **0.000** | 3.9s | **0.7s (5.6x ↑)** |
| H₂O | 14 | 0.000 | **0.000** | 1.7s | **0.9s (1.9x ↑)** |
| BeH₂ | 14 | 0.000 | **0.000** | 2.1s | **1.4s (1.5x ↑)** |
| NH₃ | 16 | -0.126 ⚠️ | **0.59 ✓** | 5.5s | 9.4s |
| N₂ | 20 | 0.000 | **0.92** | 44s | **37.6s (1.2x ↑)** |

- 小系統（≤14Q）：精度一致，速度快 1.5-5.6x
- NH₃：我們的結果物理上更正確
- N₂：我們 0.92 mHa vs 原始 0.000 mHa（原始用更多 basis + IBM tensor-product expansion）

### 實驗 3: Krylov-only Scaling（20Q → 40Q，無 NQS）

**目的**：測試純 Krylov expansion（不訓練 NQS）能推到多大。

**數據來源**：`results/benchmark_40q_krylov.json`

| 系統 | Q | Best ΔE_HF (mHa) | Best Basis | Time |
|------|---|------------------|-----------|------|
| N₂ STO-3G | 20 | -156.22 | 501 | 0.1s |
| N₂ CAS(10,12) cc-pVDZ | 24 | -201.93 | 2,001 | 0.6s |
| N₂ CAS(10,15) cc-pVDZ | 30 | -186.80 | 3,001 | 0.8s |
| N₂ CAS(10,20) cc-pVDZ | 40 | **-253.31** | 5,001 | **1.8s** |

**結論**：**Krylov-only 在 40Q 只需 1.8 秒，達到 -253 mHa below HF。** 不需要任何 NQS 訓練。

### 實驗 4: 首次 40Q/52Q E2E（有 sparse H 之前）

**目的**：用完整 NQS+SKQD pipeline 首次觸及 40Q 和 52Q。

**數據來源**：`results/benchmark_40q_52q_e2e.json`

| 系統 | Q | ΔE_HF (mHa) | Basis | Iters | Time | 備註 |
|------|---|-------------|-------|-------|------|------|
| N₂ CAS(10,12) | 24 | -197.88 | 5,857 | 5 | 46.4s | 收斂 |
| N₂ CAS(10,15) | 30 | -185.29 | 10,642 | 15 | 51.3s | iter 3+ 被 dense H >10K guard 擋住 |
| N₂ CAS(10,20) | 40 | -250.99 | 15,000 | 15 | 59.9s | iter 1+ 被擋住 |
| N₂ CAS(10,26) | 52 | -291.09 | 15,000 | 10 | 76.1s | iter 1+ 被擋住 |

**發現瓶頸**：`matrix_elements_fast()` 在 >10K configs 時拒絕建 dense H。
→ 之後修復了 sparse H construction path（`get_sparse_matrix_elements()`）。

### 實驗 5: 消融實驗（sparse H 修復後）

**目的**：量化每個優化元件的貢獻。

**數據來源**：`results/ablation_40q_52q.json`

#### 5A: 方法分離（固定 N₂ 40Q）

| 方法 | ΔE_HF (mHa) | Basis | Time | 佔總能量改善 |
|------|-------------|-------|------|------------|
| **Krylov-only (2K configs)** | **-250.50** | 2,001 | **1.3s** | **99.8%** |
| Krylov-only (5K configs) | -253.31 | 5,001 | 2.1s | 101% |
| NQS-only (10 iters) | -112.67 | 10,735 | 270.2s | 44.9% |
| NQS+SKQD (11 iters) | -250.88 | 15,000 | 309.6s | 100% (baseline) |

**結論**：
- **Krylov expansion 獨自達到 NQS+SKQD 99.8% 的能量**，只用 2,001 configs 和 1.3 秒
- **NQS-only 只達到 44.9%**（-113 vs -251 mHa），在 40Q 的 2.4 億搜索空間中效率很低
- **NQS 的邊際貢獻 = 0.4 mHa（0.16%）**，代價是多花 308 秒（238x 時間增加）

#### 5B: Qubit Scaling（NQS+SKQD, 24Q → 52Q）

| 系統 | Q | Config Space | ΔE_HF (mHa) | Basis | Iters | Time |
|------|---|-------------|-------------|-------|-------|------|
| N₂ CAS(10,12) | 24 | 627K | -198.57 | 8,727 | 6 | 55.9s |
| N₂ CAS(10,15) | 30 | 9M | -185.42 | 12,285 | 8 | 102.7s |
| N₂ CAS(10,20) | 40 | 240M | -250.90 | 12,240 | 5 | 140.3s |
| N₂ CAS(10,26) | **52** | **4.33B** | **-291.01** | **13,886** | **4** | **194.8s** |

**結論**：
- **52Q 成功**：43 億 config space 中用 13,886 個（0.0003%）在 195 秒內得到 -291 mHa
- **Config space 增長 7,000x，時間只增長 3.5x**（sub-linear scaling）
- **大系統收斂更快**：52Q 只需 4 iter，24Q 需要 6 iter

#### 5C: Krylov 深度掃描（固定 40Q NQS+SKQD）

| krylov_max_new | ΔE_HF (mHa) | Basis | Iters | Time | 邊際改善 |
|----------------|-------------|-------|-------|------|---------|
| 500 | -240.23 | 8,627 | 7 | 188.3s | — |
| **1,000** | **-248.70** | **9,076** | **5** | **138.0s** | **+8.5 mHa** |
| 2,000 | -250.88 | 15,000 | 11 | 310.1s | +2.2 mHa |
| 5,000 | -253.67 | 15,000 | 15 | 421.2s | +2.8 mHa |

**結論**：
- **krylov=1000 是最佳性價比**：138s 達到 -249 mHa
- **邊際效益遞減**：前 1K configs 每個貢獻 0.008 mHa，後 4K 每個只貢獻 0.001 mHa
- krylov=2000+ 撞到 max_basis_size=15K 天花板

---

## 四、工程交付

| 交付 | 描述 |
|------|------|
| `src/utils/gpu_diag.py` | GPU solve_fermion 替代品（dense + sparse H path） |
| `src/utils/krylov_expand.py` | 2-hop Hamiltonian connection expansion |
| `src/utils/format_utils.py` | Vectorized IBM format conversion + dedup |
| `src/methods/hi_nqs_skqd.py` | 完整 HI+NQS+SKQD pipeline（NQS + Krylov + GPU diag + feedback） |
| `src/nqs/transformer.py` | KV-cached autoregressive sampling |
| `src/krylov/skqd.py` | OOM guard + sparse threshold + reg shift fix |
| `tests/` (8 files) | 91 自動化測試 |
| `scripts/` (4 files) | GPU diag / 40Q Krylov / E2E / 消融 benchmark |
| `results/` (5 files) | 全部實驗原始 JSON 數據 |

---

## 五、總結

### 一句話

> 我們把 HI+NQS 的對角化後端從 IBM CPU-only 的 `solve_fermion`（28Q 要 485s）替換成 GPU sparse eigsh + Krylov expansion，讓 **52Q 在 195 秒內完成**，**小系統精度完全保持**，並發現在弱相關 N₂ 上 Krylov-only 就足夠（1.3s / 40Q）。

### 數字摘要

| 指標 | 值 |
|------|---|
| 最大系統 | **52 qubit**（N₂ CAS(10,26) cc-pVDZ，43 億 config space） |
| 52Q 能量 | **-291.01 mHa** below HF |
| 52Q 時間 | **194.8 秒** |
| 40Q Krylov-only | **-253.31 mHa** / 5,001 configs / **1.8 秒** |
| 小系統精確 FCI | H₂, LiH, H₂O, BeH₂（4/6 系統 0.00 mHa） |
| NH₃ 修正 | -0.126 mHa violation → 0.59 mHa correct |
| NQS 貢獻（40Q N₂）| 0.4 mHa（0.16%），暗示 NQS 在弱相關系統無用 |
| Tests | **91/91 通過** |
| 新增程式碼 | ~4,000 行 |

### 限制

1. **僅測試 N₂（弱相關系統）**：強相關系統（Cr₂、[2Fe-2S]）可能是 NQS 有價值的場景
2. **2-hop expansion** 弱於完整 Krylov 時間演化（expm_multiply）
3. **max_basis_size=15K 天花板**：限制了深度 Krylov 的效益

### 下一步

1. 測試強相關系統（Cr₂、[2Fe-2S]）
2. 整合完整 Krylov 時間演化
3. 增大 max_basis_size（需要更好的 sparse diag）

---

## 附錄：全部數據檔案

| 實驗 | 數據路徑 |
|------|---------|
| GPU vs IBM 精度 | `results/gpu_diag_benchmark.json` |
| E2E 小系統 (≤20Q) | `results/benchmark_e2e_hi_nqs_skqd.json` |
| Krylov-only scaling | `results/benchmark_40q_krylov.json` |
| 首次 40Q/52Q E2E | `results/benchmark_40q_52q_e2e.json` |
| 消融實驗 (A/B/C) | `results/ablation_40q_52q.json` |
