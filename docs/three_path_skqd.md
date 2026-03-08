# SKQD 四路徑比較：Direct Diag / Path A / Path B / Path C 完整技術文件

## 目錄

1. [理論背景](#1-理論背景)
2. [四條路徑總覽](#2-四條路徑總覽)
3. [共享基礎設施](#3-共享基礎設施)
4. [Direct Diag：無 Krylov 直接對角化](#4-direct-diag無-krylov-直接對角化)
5. [Path C：精確 Lanczos 時間演化](#5-path-c精確-lanczos-時間演化)
6. [Path B：Trotterized State-Vector 模擬](#6-path-btrotterized-state-vector-模擬)
7. [Path A：CUDA-Q 量子電路](#7-path-acuda-q-量子電路)
8. [受控實驗設計（Controlled Experiment）](#8-受控實驗設計controlled-experiment)
9. [流程全圖](#9-流程全圖)
10. [基準測試結果](#10-基準測試結果)
11. [檔案對照表](#11-檔案對照表)

---

## 1. 理論背景

### 1.1 Sample-Based Krylov Quantum Diagonalization (SKQD)

SKQD 來自 Yu et al. (arXiv:2501.09702) 的演算法。其核心思想是：

1. 從一個參考態 $\lvert \psi_0\rangle$（通常是 Hartree-Fock 態）出發
2. 透過時間演化算子 $U = e^{-iH\Delta t}$ 反覆作用，生成 **Krylov 態序列**：
   $$\lvert \psi_k\rangle = U^k \lvert \psi_0\rangle = e^{-ikH\Delta t} \lvert \psi_0\rangle, \quad k = 0, 1, \dots, d-1$$
3. 對每個 Krylov 態 $\lvert \psi_k\rangle$ 進行計算基底（computational basis）量測取樣
4. 收集所有取樣到的 bitstring，累積形成一個 **子空間基底**（subspace basis）
5. 在此基底上投影 Hamiltonian：$H_{\text{eff}}[i,j] = \langle s_i \rvert H \lvert s_j \rangle$
6. 對投影後的有效 Hamiltonian 進行對角化，取最小特徵值作為基態能量估計

### 1.2 最佳時間步長（Theorem 3.1, Epperly et al.）

時間演化的步長 $\Delta t$ 決定了 Krylov 子空間的品質。根據 Epperly 定理：

$$\Delta t_{\text{optimal}} = \frac{\pi}{E_{\max} - E_{\min}}$$

其中 $E_{\max}$ 和 $E_{\min}$ 是 Hamiltonian 在粒子數守恆子空間中的最大和最小特徵值（即 **光譜範圍 spectral range**）。

此步長確保 Krylov 態之間有最大的正交性，從而讓子空間能最有效地覆蓋基態波函數。

實作位置：`src/krylov/spectral_utils.py` — `compute_optimal_dt()`

### 1.3 Jordan-Wigner 變換

量子電路需要 Pauli 表示的 Hamiltonian。**Jordan-Wigner 變換** 將二次量化的費米子算子轉換為 qubit 算子：

- **數目算子**：$a_p^\dagger a_p \rightarrow \frac{1}{2}(I - Z_p)$
- **跳躍算子** ($p < q$)：$a_p^\dagger a_q \rightarrow \frac{1}{4}(XX + YY + iXY - iYX) \cdot Z_{\text{chain}}$
  - 其中 $Z_{\text{chain}} = Z_{p+1} \cdots Z_{q-1}$ 是 Jordan-Wigner 弦（string）

雙體算子透過單體算子的乘積組合而成：

$$a_p^\dagger a_r^\dagger a_s a_q = (a_p^\dagger a_q)(a_r^\dagger a_s) - \delta_{qr}(a_p^\dagger a_s)$$

實作位置：`src/hamiltonians/pauli_mapping.py` — `one_body_op()`, `two_body_op()`, `molecular_hamiltonian_to_pauli()`

### 1.4 Suzuki-Trotter 分解

量子電路無法直接實現 $e^{-iH\Delta t}$（因為 $H = \sum_k c_k P_k$ 中各 Pauli 項不對易）。Trotter 分解提供了近似：

**一階 Trotter：**
$$e^{-iH\Delta t} \approx \prod_k e^{-ic_k \Delta t \cdot P_k}$$

**二階 Suzuki-Trotter（本專案預設）：**
$$S_2(\Delta t) = \prod_{k=1}^{L} e^{-ic_k \frac{\Delta t}{2} P_k} \cdot \prod_{k=L}^{1} e^{-ic_k \frac{\Delta t}{2} P_k}$$

對稱結構使得誤差為 $O(\Delta t^3)$，優於一階的 $O(\Delta t^2)$。

### 1.5 投影對角化

給定一組取樣到的 computational basis 態 $\{\lvert s_i\rangle\}_{i=1}^{N}$，投影 Hamiltonian 矩陣為：

$$H_{\text{eff}}[i,j] = \langle s_i \rvert H \lvert s_j \rangle$$

由於 computational basis 是正交歸一的，重疊矩陣 $S = I$，因此只需求解 **標準特徵值問題**（不需廣義特徵值問題）。

對角化使用的方法視矩陣大小而定：
- 小矩陣：`torch.linalg.eigh`（dense）
- 大矩陣（>10K）：`scipy.sparse.linalg.eigsh` 或 CuPy sparse eigensolver

---

## 2. 四條路徑總覽

| 屬性 | Direct Diag (No Krylov) | Path C (Exact Lanczos) | Path B (Trotterized State-Vector) | Path A (CUDA-Q Circuit) |
|------|------------------------|------------------------|-----------------------------------|------------------------|
| **子空間構建方法** | 組合列舉（HF+singles+doubles） | Krylov 時間演化取樣 | Krylov 時間演化取樣 | Krylov 時間演化取樣 |
| **時間演化方法** | 無（不使用時間演化） | 精確 $e^{-iHt}$（Lanczos） | 二階 Suzuki-Trotter（state-vector） | 二階 Suzuki-Trotter（量子電路） |
| **Trotter 誤差** | 無 | 無 | 有（$O(\Delta t^3)$） | 有（$O(\Delta t^3)$） |
| **取樣噪聲** | 無（確定性列舉） | 有限 shots | 有限 shots | 有限 shots |
| **量子硬體噪聲** | 無 | 無 | 無 | 無（模擬器） |
| **Hilbert 空間** | 粒子數守恆子空間 | 完整 $2^n$ | 完整 $2^n$ | 完整 $2^n$ |
| **初始態** | 不適用 | HF（Hartree-Fock） | HF（Hartree-Fock） | HF（Hartree-Fock） |
| **取樣方式** | 確定性列舉 | `torch.multinomial` | `torch.multinomial` | `cudaq.sample` |
| **RNG seed** | 不適用（無隨機性） | `seed + k + 1000` | `seed + k + 1000` | `seed + k`（CUDA-Q 內部） |
| **實作位置** | `pipeline.py:_generate_essential_configs` + `projected_hamiltonian.py` | `quantum_skqd.py:_sample_exact` | `quantum_skqd.py:_sample_classical_trotterized` | `quantum_skqd.py:_sample_cudaq` |
| **系統規模限制** | 組合爆炸（doubles 數量 $\propto n^4$） | 記憶體（Lanczos 向量） | phase_table（$n_{\text{terms}} \times 2^n$）；$2^n \leq 100{,}000$ | CUDA-Q 電路深度 |
| **依賴** | PySCF + SciPy | PyTorch | PyTorch | CUDA-Q（`cuda-quantum-cu12`） |

**Path A/B/C 的唯一變因**：三條 Krylov 路徑之間 **只有時間演化方法不同**，所有其他變因完全一致。

**Direct Diag 的角色**：作為 **非 Krylov 基線**（baseline），展示在不使用時間演化的情況下，僅靠組合列舉（HF + 單激發 + 雙激發）能達到的精度上界。當 Direct Diag 已達 FCI 精度時，Krylov 展開的額外貢獻為零；當系統較大、doubles 不足以覆蓋基態時，Krylov 展開的價值才顯現。

---

## 3. 共享基礎設施

以下元件為三條路徑共享，確保受控實驗的有效性。

### 3.1 Jordan-Wigner 轉換（共享）

所有路徑都使用相同的 Pauli 表示 Hamiltonian：

```
MolecularHamiltonian (PySCF integrals)
    ↓ molecular_hamiltonian_to_pauli()
    ↓ Jordan-Wigner transformation
(coefficients: List[float], pauli_words: List[str], constant: float)
```

此轉換在 `QuantumCircuitSKQD.from_molecular_hamiltonian()` 中執行一次，結果共享給所有後端。

### 3.2 初始態準備（共享）

所有路徑使用相同的 Hartree-Fock 初始態：

```python
# quantum_skqd.py:_get_initial_state_gpu()
hf = hamiltonian.get_hf_state()       # e.g., [1,1,0,0, 1,1,0,0] for LiH
idx = int("".join(str(b) for b in hf), 2)  # 轉為整數索引
psi = torch.zeros(2^n, dtype=complex128)
psi[idx] = 1.0                         # |HF⟩ 在 computational basis 中
```

### 3.3 投影對角化（共享）

取樣完成後，所有路徑使用相同的對角化流程：

1. 將 bitstring 轉為 basis tensor（`_basis_from_samples()`）
2. 建構投影 Hamiltonian $H_{\text{eff}}$
   - 若有分子 Hamiltonian 物件：使用 **Slater-Condon 規則**（`_diagonalize_slater_condon()`）
   - 否則（純 Pauli）：使用 **向量化 Pauli 矩陣元素計算**（`_diagonalize_pauli_gpu()`）
3. 對稱化：$H_{\text{eff}} \leftarrow \frac{1}{2}(H_{\text{eff}} + H_{\text{eff}}^T)$
4. `torch.linalg.eigh()` 取最小特徵值

### 3.4 累積基底策略（共享）

所有路徑使用 **累積取樣**（cumulative basis）：

```
k=0: {bitstrings from |ψ₀⟩}
k=1: {bitstrings from |ψ₀⟩} ∪ {bitstrings from |ψ₁⟩}
k=2: {bitstrings from |ψ₀⟩} ∪ {bitstrings from |ψ₁⟩} ∪ {bitstrings from |ψ₂⟩}
...
```

每增加一個 Krylov 維度，子空間基底只增不減，能量估計單調改善（或至少不變）。

### 3.5 超參數設定（共享）

| 參數 | 數值 | 來源 |
|------|------|------|
| `max_krylov_dim` | 15 | 論文 Fig. 1（Ising 模擬） |
| `num_trotter_steps` | 1 | 論文：single $S_2(\Delta t)$ per evolution |
| `trotter_order` | 2 | 論文 Section IV |
| `shots` | 100,000 | 論文 Section V |
| `dt` | $\pi / \text{spectral\_range}$ | Theorem 3.1 (Epperly) |
| `seed` | 42 | 可重現性 |

---

## 4. Direct Diag：無 Krylov 直接對角化

### 4.1 理論

Direct Diag 是最簡單的子空間對角化方法：**完全不使用時間演化**，而是透過組合列舉，確定性地生成所有「重要的」計算基底態（HF 態、單激發、雙激發），然後直接在此子空間上建構並對角化投影 Hamiltonian。

這是 pipeline 中 `skip_nf_training=True`（Direct-CI 模式）的核心流程。其物理依據是 **Slater-Condon 規則**：

- 只有 **單激發**（single excitation）和 **雙激發**（double excitation）與 HF 態之間有非零的 Hamiltonian 矩陣元素
- 基態波函數以 HF 態為主導，主要修正來自 doubles（佔 > 90% 的相關能量），其次是 singles
- 對於小系統（$\leq 18$ qubits），HF + singles + doubles 已涵蓋整個粒子數守恆子空間（即 Full CI）

因此，Direct Diag 在小系統上等價於 **精確 FCI**（Full Configuration Interaction），是無偏差的精確解。

### 4.2 Essential Configs 生成

Essential configs 的生成由 `FlowGuidedKrylovPipeline._generate_essential_configs()` 實現：

```
[MolecularHamiltonian]
     │
     ▼ 讀取 HF 態
┌──────────────────────────────────────────────────┐
│ hf_config = hamiltonian.get_hf_state()            │
│   e.g., [1,1,0,0, 1,1,0,0] for LiH (4e, 6 orbs) │
└──────────────────────┬───────────────────────────┘
                       │
     ▼ 列舉 singles (α→α, β→β)
┌──────────────────────────────────────────────────┐
│ for each occupied orbital i:                      │
│   for each unoccupied orbital a (same spin):     │
│     single = hf_config.copy()                    │
│     single[i] = 0, single[a] = 1                 │
│     → 產生一個單激發 config                       │
└──────────────────────┬───────────────────────────┘
                       │
     ▼ 列舉 doubles (ii→aa, ij→ab, 含跨 spin)
┌──────────────────────────────────────────────────┐
│ for each pair of occupied orbitals (i, j):        │
│   for each pair of unoccupied orbitals (a, b):   │
│     double = hf_config.copy()                    │
│     double[i]=0, double[j]=0                     │
│     double[a]=1, double[b]=1                     │
│     → 產生一個雙激發 config                       │
│     (需保持粒子數守恆: n_alpha, n_beta 不變)     │
└──────────────────────┬───────────────────────────┘
                       │
     ▼ 合併去重
┌──────────────────────────────────────────────────┐
│ essential_configs = {HF} ∪ {singles} ∪ {doubles}  │
│ 以 tensor 形式返回：shape (N, n_qubits)           │
└──────────────────────────────────────────────────┘
```

### 4.3 Essential Configs 數量分析

Singles 和 doubles 的數量由佔據/未佔據軌道數決定：

$$n_{\text{singles}} = n_{\alpha}^{\text{occ}} \times n_{\alpha}^{\text{virt}} + n_{\beta}^{\text{occ}} \times n_{\beta}^{\text{virt}}$$

$$n_{\text{doubles}} = \binom{n_{\alpha}^{\text{occ}}}{1}\binom{n_{\alpha}^{\text{virt}}}{1}\binom{n_{\beta}^{\text{occ}}}{1}\binom{n_{\beta}^{\text{virt}}}{1} + \binom{n_{\alpha}^{\text{occ}}}{2}\binom{n_{\alpha}^{\text{virt}}}{2} + \binom{n_{\beta}^{\text{occ}}}{2}\binom{n_{\beta}^{\text{virt}}}{2}$$

| 系統 | Qubits | 佔據 | 虛擬 | Singles | Doubles | Total Essential | Full CI Configs |
|------|--------|------|------|---------|---------|----------------|-----------------|
| H2   | 4      | 1+1  | 1+1  | 2       | 1       | 4              | 4               |
| LiH  | 12     | 2+2  | 4+4  | 16      | 72      | 89             | 225             |
| H2O  | 14     | 5+5  | 2+2  | 20      | 60      | 81             | 441             |
| BeH2 | 14     | 3+3  | 4+4  | 24      | 168     | 193            | 1,225           |
| NH3  | 16     | 5+5  | 3+3  | 30      | 135     | 166            | 3,136           |
| CH4  | 18     | 5+5  | 4+4  | 40      | 256     | 297            | 15,876          |
| N2   | 20     | 7+7  | 3+3  | 42      | 189     | 232            | 14,400          |

**觀察**：對於小系統（H2），essential configs = full CI configs（完全相同）。隨著系統增大，essential configs 僅佔 full CI 的一小部分（如 CH4：297/15,876 ≈ 1.9%），此時 Direct Diag 不再等於 FCI，需要 Krylov 展開或 NF 取樣來補充更高階激發。

### 4.4 實作流程

```
[MolecularHamiltonian (PySCF)]
     │
     ▼ _generate_essential_configs()
┌──────────────────────────────────────────────────┐
│ 確定性列舉 HF + singles + doubles                 │
│   → essential_configs: tensor (N, n_qubits)       │
│   無隨機性，無取樣，無時間演化                     │
└──────────────────────┬───────────────────────────┘
                       │
     ▼ （可選）diversity_selection()
┌──────────────────────────────────────────────────┐
│ DPP-greedy 多樣性篩選                             │
│   essential configs 受「必保」保護，不會被濾除     │
│   （小系統通常全數保留）                           │
└──────────────────────┬───────────────────────────┘
                       │
     ▼ 建構投影 Hamiltonian
┌──────────────────────────────────────────────────┐
│ ProjectedHamiltonian:                             │
│   H_eff[i,j] = ⟨sᵢ|H|sⱼ⟩                        │
│   使用 Slater-Condon 規則（二次量化）             │
│   不需 Jordan-Wigner / Pauli 表示                 │
└──────────────────────┬───────────────────────────┘
                       │
     ▼ 對角化
┌──────────────────────────────────────────────────┐
│ torch.linalg.eigh(H_eff)                         │
│   → E₀ = 最小特徵值                               │
│   → 加上 nuclear_repulsion_energy                 │
└──────────────────────────────────────────────────┘
```

### 4.5 與 Krylov 路徑的關鍵差異

| 維度 | Direct Diag | Krylov 路徑 (Path A/B/C) |
|------|-------------|--------------------------|
| 子空間構建 | 組合列舉（確定性） | 時間演化取樣（隨機性） |
| 基底來源 | 物理直覺（Slater-Condon） | 動力學探索（$e^{-iHt}$ 驅動） |
| 高階激發 | 不包含（僅到 doubles） | 可自然探索到 triples+ |
| 隨機性 | 無 | 有（shots 取樣） |
| 可重現性 | 完全確定性 | 依賴 seed |
| Hamiltonian 表示 | 二次量化（Slater-Condon） | Pauli 表示（Jordan-Wigner） |
| 計算成本 | $O(N^2)$ 矩陣元素 | $O(d \times \text{shots} \times N^2)$ |

### 4.6 適用場景

- **小系統（$\leq 18$ qubits）**：Direct Diag 通常已達 FCI，Krylov 無額外貢獻
- **中型系統（20-24 qubits）**：Direct Diag 給出 CISD 級精度，Krylov 可補充 triples/quadruples
- **大型系統（$\geq 26$ qubits）**：doubles 數量組合爆炸（$\propto n^4$），Direct Diag 本身的矩陣建構也成為瓶頸；此時需 NF-NQS 取樣

### 4.7 特點

- **零誤差來源**（小系統）：無取樣噪聲、無 Trotter 誤差、無子空間截斷——當 essential configs 涵蓋全 CI 空間時，結果 **精確等於 FCI**
- **確定性結果**：不依賴隨機種子，多次執行結果完全一致
- **最低計算成本**：不需要時間演化、不需要 Pauli 分解、不需要 shots 取樣
- **作為基線（baseline）**：量化 Krylov 展開帶來的額外精度收益——`Krylov 貢獻 = E(Krylov) - E(Direct Diag)`

---

## 5. Path C：精確 Lanczos 時間演化

### 5.1 理論

Path C 在完整的 $2^n$ Hilbert 空間中計算 **精確的** $e^{-iHt}\lvert \psi\rangle$，不使用 Trotter 分解。這是實驗的 **黃金標準參考**（gold standard reference）。

Lanczos 演算法將矩陣指數投影到一個小的 Krylov 子空間上：

1. 建構 Lanczos 基底 $\{v_0, v_1, \dots, v_{m-1}\}$，其中 $v_0 = \lvert \psi\rangle / \lVert \lvert \psi\rangle\rVert $
2. 在 Lanczos 基底中，$H$ 的投影為三對角矩陣 $T$（$m \times m$，$m \ll 2^n$）
3. 計算小矩陣 $e^{-itT}$（$m$ 通常 $\leq 30$，可 dense 對角化）
4. 投影回原空間：$e^{-iHt}\lvert \psi\rangle \approx \lVert \lvert \psi\rangle\rVert \cdot V \cdot e^{-itT} \cdot e_0$

### 5.2 實作流程

```
[初始態 |HF⟩] ─── 完整 2^n state-vector (complex128)
     │
     ▼ 對每個 k = 0, 1, ..., d-1:
┌─────────────────────────────────────────────┐
│ 1. psi = |HF⟩                               │
│ 2. for _ in range(k):                       │
│       psi = _lanczos_exact_evolution(psi, T) │   ← Lanczos H|ψ⟩ matvec
│       psi = psi / ||psi||                    │
│ 3. probs = |psi|²                           │
│ 4. indices = torch.multinomial(probs, shots)│   ← seed = 42 + k + 1000
│ 5. bitstrings = format(indices, binary)     │
└─────────────────────────────────────────────┘
     │
     ▼ 累積所有 bitstrings
┌─────────────────────────────────────────────┐
│ 投影對角化：                                 │
│   basis = union of all sampled bitstrings    │
│   H_eff[i,j] = ⟨sᵢ|H|sⱼ⟩                  │
│   E₀ = min eigenvalue of H_eff             │
└─────────────────────────────────────────────┘
```

### 5.3 Hamiltonian matvec 實作

Path C 的 Lanczos 需要反覆計算 $H\lvert \psi\rangle$。使用 **輕量級 Pauli mask**（`_precompute_pauli_masks_lightweight()`）：

- 僅儲存 $O(n_{\text{terms}})$ 的整數遮罩（flip mask、YZ mask），不儲存 $O(n_{\text{terms}} \times 2^n)$ 的 phase table
- 每個 Pauli term 的相位透過 **bit parity** 即時計算：
  $$\text{phase}(x) = i^{n_Y} \cdot (-1)^{\text{popcount}(x \wedge \text{yz\_mask})}$$
- 分塊處理（chunk_size 根據維度自適應）以控制 GPU 記憶體

這使得 Path C 能處理 $\geq 18$ qubit 的系統（$2^{18} = 262{,}144$ 維），而不像 Path B 那樣被 phase_table 的記憶體限制。

### 5.4 特點

- **零 Trotter 誤差**：結果等價於在 $2^n$ 空間中做精確對角化的 Krylov 方法
- **取樣誤差依然存在**：使用有限 shots 取樣，因此結果不完全等於 FCI
- **適用所有系統大小**：記憶體瓶頸在 Lanczos 向量（$O(2^n)$），而非 phase table

---

## 6. Path B：Trotterized State-Vector 模擬

### 6.1 理論

Path B 在 GPU 上使用 **state-vector 模擬** 實現 Trotterized 時間演化。這是量子電路（Path A）的 **精確經典模擬**——兩者具有完全相同的 Trotter 分解結構，但 Path B 用數值計算取代量子閘操作。

每個 Pauli 旋轉 $e^{-i\theta P_k}$ 利用 $P^2 = I$ 的性質解析求解：

$$e^{-i\theta P}\lvert \psi\rangle = \cos(\theta)\lvert \psi\rangle - i\sin(\theta) P\lvert \psi\rangle$$

其中 $P\lvert \psi\rangle$ 透過預計算的 **flip mask** 和 **phase table** 在 $O(2^n)$ 時間內完成。

### 6.2 預計算結構

在首次呼叫時，`_precompute_pauli_actions()` 建構：

**Flip mask**（$n_{\text{terms}}$ 個整數）：

每個 Pauli term 的 flip mask 記錄了 X 和 Y 算子的位置。對 basis state $\lvert x\rangle$：
$$P_k\lvert x\rangle = \text{phase}(x) \cdot \lvert x \oplus \text{flip\_mask}_k\rangle$$

**Phase table**（$n_{\text{terms}} \times 2^n$ complex128 張量）：

對每個 (term, state) 對，記錄複數相位。相位由 Z 和 Y 算子的貢獻決定：
- Z 在 bit=1 的位置：因子 $-1$（即 $i^2$）
- Y 在 bit=0 的位置：因子 $+i$（即 $i^1$）
- Y 在 bit=1 的位置：因子 $-i$（即 $i^3$）

所有相位累積 mod 4，然後映射到 $\{1, i, -1, -i\}$。

### 6.3 實作流程

```
[初始態 |HF⟩] ─── 完整 2^n state-vector (complex128)
     │
     ▼ 預計算（一次性）
┌──────────────────────────────────────────────────┐
│ _precompute_pauli_actions():                      │
│   flip_masks:   (n_terms,) int64 ── 每個 term 的  │
│                    bit flip 遮罩                   │
│   phase_tables: (n_terms, 2^n) complex128 ── 每個 │
│                    (term, state) 的複數相位        │
└──────────────────────────────────────────────────┘
     │
     ▼ 對每個 k = 0, 1, ..., d-1:
┌──────────────────────────────────────────────────┐
│ 1. psi = |HF⟩                                    │
│ 2. for _ in range(k):       ← 重複 k 次          │
│       _apply_trotter_step(psi):                  │
│         for step in range(num_trotter_steps):    │
│           Forward half-step (dt_scale=0.5):      │
│             for each term k=0..L-1:              │
│               θ = c_k · Δt · 0.5                 │
│               psi = cos(θ)·psi - i·sin(θ)·P_k·psi│
│           Backward half-step (dt_scale=0.5):     │
│             for each term k=L-1..0:              │
│               θ = c_k · Δt · 0.5                 │
│               psi = cos(θ)·psi - i·sin(θ)·P_k·psi│
│ 3. psi = psi / ||psi||                           │
│ 4. probs = |psi|²                                │
│ 5. indices = torch.multinomial(probs, shots)     │   ← seed = 42 + k + 1000
│ 6. bitstrings = format(indices, binary)          │
└──────────────────────────────────────────────────┘
     │
     ▼ 累積所有 bitstrings → 投影對角化（同 Path C）
```

### 6.4 計算複雜度

每個 Krylov step 的成本：
$$O(k \cdot n_{\text{trotter\_steps}} \cdot 2 \cdot n_{\text{terms}} \cdot 2^n)$$

例如 LiH（12 qubits, ~630 Pauli terms, 1 Trotter step, $2^{12} = 4{,}096$）：
$$\text{每個 } k: 630 \times 2 \times 4{,}096 \approx 5.2\text{M 浮點運算}$$

### 6.5 記憶體限制

Phase table 的大小為 $n_{\text{terms}} \times 2^n \times 16$ bytes（complex128）。這限制了 Path B 的最大系統大小：

| 系統 | Qubits | $2^n$ | Pauli terms | Phase table 大小 |
|------|--------|-------|-------------|-----------------|
| H2   | 4      | 16    | ~15         | 3.8 KB          |
| LiH  | 12     | 4,096 | ~630        | 39.3 MB         |
| H2O  | 14     | 16,384| ~1,500      | 374 MB          |
| BeH2 | 14     | 16,384| ~1,500      | 374 MB          |
| NH3  | 16     | 65,536| ~3,000      | 3.0 GB          |
| CH4  | 18     | 262K  | ~6,000      | 24 GB           |

**因此，Path B 在 $2^n > 100{,}000$（約 $\geq 18$ qubits）時自動跳過**，以避免 OOM。

### 6.6 特點

- **有 Trotter 誤差**：與 Path A 完全相同的 Trotter 結構，因此 Trotter 誤差一致
- **無量子硬體噪聲**：純數值計算，結果為 Path A 的「理想上界」
- **Path B vs Path C 差異 = 純 Trotter 誤差**：是量化 Trotter 分解影響的最佳方式

---

## 7. Path A：CUDA-Q 量子電路

### 7.1 理論

Path A 使用 NVIDIA CUDA-Q 框架，在 GPU 加速的量子模擬器上執行 **真正的量子電路**。這是最接近實際量子硬體執行的路徑。

電路結構：

```
|0⟩^⊗n ──[X gates: prepare |HF⟩]──[S₂(Δt)]^k──[Measure all]
```

其中 $[S_2(\Delta t)]$ 是二階 Suzuki-Trotter 電路，由一系列 `exp_pauli` 閘組成。

### 7.2 CUDA-Q Kernel 設計

```python
@cudaq.kernel
def krylov_circuit_hf(
    num_qubits: int,
    krylov_power: int,        # k: 重複次數
    trotter_steps: int,        # 每次演化的 Trotter sub-steps
    H_pauli_words: list[cudaq.pauli_word],  # Pauli 項
    angles: list[float],       # 預計算的旋轉角度
    occ_qubits: list[int],    # HF 態中被佔據的 qubit
):
    qubits = cudaq.qvector(num_qubits)

    # 準備 HF 初始態
    for oq in range(len(occ_qubits)):
        x(qubits[occ_qubits[oq]])

    # 重複 k 次 Trotterized 時間演化
    for _ in range(krylov_power):
        for _ in range(trotter_steps):
            for i in range(len(angles)):
                exp_pauli(angles[i], qubits, H_pauli_words[i])

    # 量測
    mz(qubits)
```

### 7.3 CUDA-Q JIT Bug 與 Workaround

**已知問題**：CUDA-Q 的 JIT 編譯器在 kernel 內部計算 `coeffs[i] * dt` 時會產生零值旋轉。

**解決方案**：在 kernel 外部預計算所有旋轉角度：

```python
# 在 _init_cudaq() 中預計算
# exp_pauli(angle, qubits, P) 實現 exp(i * angle * P)
# 我們要 exp(-i * coeff * dt * P), 所以 angle = -coeff * dt

if trotter_order == 2:
    half = [-c * dt / 2 for c in pauli_coefficients]
    angles = half + half[::-1]          # 正向半步 + 反向半步
    pauli_words = words + words[::-1]
else:
    angles = [-c * dt for c in pauli_coefficients]
```

### 7.4 實作流程

```
[CUDA-Q 初始化（一次性）]
┌──────────────────────────────────────────────┐
│ _init_cudaq():                                │
│   cudaq.set_target("nvidia", option="fp64")  │
│   預計算 exp_pauli 角度（避免 JIT bug）       │
│   快取 HF 佔據 qubit 列表                    │
│                                              │
│ _build_cudaq_kernels():                       │
│   編譯 krylov_circuit_hf kernel（一次）       │
│   編譯 krylov_circuit_neel kernel（一次）     │
└──────────────────────────────────────────────┘
     │
     ▼ 對每個 k = 0, 1, ..., d-1:
┌──────────────────────────────────────────────┐
│ _sample_cudaq(k):                             │
│   1. cudaq.set_random_seed(seed + k)         │
│   2. result = cudaq.sample(                   │
│        krylov_circuit_hf,                     │
│        n_qubits, k, trotter_steps,           │
│        pauli_words, angles, occ_qubits,      │
│        shots_count=100000                     │
│      )                                       │
│   3. return dict(result.items())  → bitstrings│
└──────────────────────────────────────────────┘
     │
     ▼ 累積所有 bitstrings → 投影對角化（同 Path C）
```

### 7.5 CUDA-Q 配置

| 設定 | 值 | 說明 |
|------|---|------|
| target | `"nvidia"` | NVIDIA GPU 模擬器 |
| option | `"fp64"` | 雙精度（化學精度所需） |
| seed | `42 + k` | 每個 Krylov step 不同的隨機種子 |
| shots | 100,000 | 每個 Krylov 態的取樣次數 |

### 7.6 特點

- **最接近真實量子硬體**：電路結構完全匹配硬體執行
- **CUDA-Q 內部 RNG**：使用 `cudaq.set_random_seed`，與 Path B/C 的 `torch.Generator` 不同
- **無 phase_table 記憶體限制**：CUDA-Q 模擬器內部管理狀態向量
- **依賴 CUDA-Q 安裝**：需要 `cuda-quantum-cu12` 套件，僅 Linux + NVIDIA GPU

---

## 8. 受控實驗設計（Controlled Experiment）

### 8.1 設計原則

嚴謹的消融實驗（ablation study）要求 **一次只改變一個變因**。本比較實驗的唯一變因是 **時間演化方法**。

### 8.2 控制變因清單

| 變因 | 是否控制 | 說明 |
|------|---------|------|
| Hamiltonian | 相同 | 同一個 `MolecularHamiltonian` 實例 |
| Pauli 分解 | 相同 | 同一次 Jordan-Wigner 轉換結果 |
| 初始態 | 相同 | 都是 HF 態 |
| 時間步長 $\Delta t$ | 相同 | 都用 $\pi / \text{spectral\_range}$ |
| Krylov 維度 | 相同 | 都是 15 |
| Trotter 階數 | 相同 | 都是二階（Path C 雖不使用 Trotter，但設定一致） |
| Shots 數 | 相同 | 都是 100,000 |
| 累積策略 | 相同 | 都是 cumulative union |
| Hilbert 空間 | 相同 | 都在完整 $2^n$ 空間中操作 |
| 投影對角化 | 相同 | 都用 Slater-Condon 規則 + `torch.linalg.eigh` |
| **時間演化方法** | **不同** | Path C: Lanczos, Path B: Trotter state-vector, Path A: Trotter circuit |

### 8.3 誤差分離分析

透過四條路徑的組合比較，可以精確分離各種誤差來源：

```
Direct Diag error = 子空間截斷誤差（僅 HF+singles+doubles）
   （小系統 = 0，因為 essential configs 涵蓋全 CI 空間）

Path C error = 取樣誤差 + 子空間截斷誤差
Path B error = 取樣誤差 + 子空間截斷誤差 + Trotter 誤差

→ Krylov 貢獻 = E(Direct Diag) - E(Path C)
   （Krylov 時間演化帶來的額外精度收益，對小系統 ≈ 0）

→ Trotter 效應 = |E(Path B) - E(Path C)|
   （純粹由 Trotter 分解引入的誤差）

Path A error = 取樣誤差 + 子空間截斷誤差 + Trotter 誤差 + 電路效應

→ 電路效應 = |E(Path A) - E(Path B)|
   （CUDA-Q 模擬器 vs 數值計算的差異，理論上應為零或極小）
```

### 8.4 化學精度標準

所有結果以 **化學精度（chemical accuracy）** 為判定標準：

$$\text{error} < 1.594 \text{ mHa} \approx 1 \text{ kcal/mol}$$

這是化學中公認的「足夠精確」門檻——低於此誤差的能量差異在化學反應預測中可忽略不計。

---

## 9. 流程全圖

```
                    ┌──────────────────────────────────┐
                    │    MolecularHamiltonian (PySCF)   │
                    │   h1e, h2e, nuclear_repulsion     │
                    └────────┬─────────────────┬───────┘
                             │                 │
              ┌──────────────▼──────┐          │
              │  Direct Diag         │          │
              │  (無 Krylov 基線)    │   molecular_hamiltonian_to_pauli()
              │                     │   (Jordan-Wigner transformation)
              │  _generate_         │          │
              │  essential_configs()│   ┌──────▼─────────────────────┐
              │  HF+singles+doubles │   │  Pauli Hamiltonian (共享)   │
              └──────────┬──────────┘   │  coefficients, pauli_words │
                         │              └──────────┬────────────────┘
                         │                         │
                         │              ┌──────────▼────────────────┐
                         │              │ compute_optimal_dt(H)     │
                         │              │ dt = π / (E_max - E_min)  │
                         │              └──────────┬────────────────┘
                         │                         │
                         │              ┌──────────▼────────────────┐
                         │              │ QuantumCircuitSKQD (共享)  │
                         │              │   n_qubits, dt, config    │
                         │              └──┬────────┬──────────┬───┘
                         │                 │        │          │
                         │        ┌────────▼──┐ ┌───▼────┐ ┌──▼──────────┐
                         │        │  Path C    │ │ Path B  │ │  Path A      │
                         │        │  "exact"   │ │"classic"│ │  "cudaq"     │
                         │        └────────┬───┘ └───┬────┘ └──┬──────────┘
                         │                 │         │         │
                         │        ┌────────▼──┐ ┌────▼────┐ ┌──▼──────────┐
                         │        │ Lanczos    │ │ Trotter  │ │ exp_pauli   │
                         │        │ e^{-iHt}   │ │ cos-isin │ │ CUDA-Q      │
                         │        └────────┬───┘ └────┬────┘ └──┬──────────┘
                         │                 │         │         │
                         │                 └────┬────┘─────────┘
                         │                      │
                         │           ┌──────────▼──────────────┐
                         │           │ 取樣 (shots = 100,000)   │
                         │           │  B/C: torch.multinomial  │
                         │           │  A: cudaq.sample          │
                         │           └──────────┬──────────────┘
                         │                      │
                         │           ┌──────────▼──────────────┐
                         │           │ 累積基底 (k=0..14)       │
                         │           │  cumulative union        │
                         │           └──────────┬──────────────┘
                         │                      │
                         └──────────┬───────────┘
                                    │
                    ┌───────────────▼───────────────────┐
                    │ 投影對角化                          │
                    │   H_eff[i,j] = ⟨sᵢ|H|sⱼ⟩          │
                    │   E₀ = min eig(H_eff)              │
                    │                                    │
                    │   Direct Diag: Slater-Condon 規則   │
                    │   Path A/B/C: Slater-Condon 或 Pauli│
                    └───────────────┬───────────────────┘
                                    │
                    ┌───────────────▼───────────────────┐
                    │ 結果比較                            │
                    │   error = |E₀ - E_FCI| × 1000      │
                    │   (mHa)                            │
                    └──────────────────────────────────┘
```

---

## 10. 基準測試結果

### 10.1 數值結果（7 分子系統, STO-3G 基組）

| 系統 | Qubits | Path C (mHa) | Path B (mHa) | Path A (mHa) | Trotter B-C (mHa) |
|------|--------|:------------:|:------------:|:------------:|:-----------------:|
| H2   | 4      | 0.0000       | 0.0000       | 0.0000       | 0.0000            |
| LiH  | 12     | 0.0074       | 0.0119       | 0.0099       | 0.0045            |
| H2O  | 14     | 0.0626       | 0.0478       | 0.0437       | 0.0148            |
| BeH2 | 14     | 0.0198       | 0.0160       | 0.0198       | 0.0038            |
| NH3  | 16     | 0.3234       | 0.4253       | 0.3599       | 0.1018            |
| CH4  | 18     | 0.5451       | skip         | 0.4986       | ---               |
| N2   | 20     | 1.1427       | skip         | 1.1003       | ---               |

- 所有 7 個系統在所有可用路徑上均通過化學精度 (< 1.594 mHa)
- CH4/N2 的 Path B 因 phase_table 記憶體限制（$2^n > 100{,}000$）而跳過
- Trotter 效應（B-C 差異）隨系統大小增加，但仍在 0.1 mHa 量級

### 10.2 觀察

1. **Trotter 誤差極小**：二階 Suzuki-Trotter 在 $\Delta t = \pi / \text{spectral\_range}$ 下引入的額外誤差約 0.01-0.1 mHa，遠低於化學精度門檻
2. **電路效應可忽略**：Path A vs Path B（或 Path C）的差異在 0.01 mHa 量級
3. **主要誤差來源是子空間截斷**：Path C 的誤差隨系統大小增長（0.0000 → 1.1427 mHa），說明有限 shots 下的子空間覆蓋率才是精度瓶頸

---

## 11. 檔案對照表

| 檔案 | 角色 |
|------|------|
| `src/pipeline.py` | **Direct Diag 實作**。`FlowGuidedKrylovPipeline._generate_essential_configs()` 生成 HF+singles+doubles；`PipelineConfig.skip_nf_training=True` 啟用 Direct-CI 模式 |
| `src/krylov/quantum_skqd.py` | **三路徑核心實作**。`QuantumCircuitSKQD` 類別同時實現 Path A (`_sample_cudaq`)、Path B (`_sample_classical_trotterized`)、Path C (`_sample_exact`) |
| `src/krylov/skqd.py` | **經典 SKQD**（pipeline 主路徑使用）。在粒子數守恆子空間中做精確 `gpu_expm_multiply`，不使用 Pauli 分解 |
| `src/hamiltonians/pauli_mapping.py` | Jordan-Wigner 轉換。`PauliSum` 代數、`molecular_hamiltonian_to_pauli()` |
| `src/krylov/spectral_utils.py` | `compute_optimal_dt()`：從光譜範圍計算最佳 $\Delta t$ |
| `src/utils/gpu_linalg.py` | GPU 加速線性代數：`gpu_eigh`, `gpu_eigsh`, `gpu_expm_multiply` |
| `examples/quantum_vs_classical_krylov.py` | **三路徑比較腳本**。入口點 `run_comparison()`，輸出 summary table |

---

## 附錄 A：Path C vs Pipeline 中的 Classical SKQD 差異

本文件描述的 Path C（`quantum_skqd.py:_sample_exact`）和 pipeline 使用的 Classical SKQD（`skqd.py:SampleBasedKrylovDiagonalization`）有以下區別：

| 屬性 | Path C (quantum_skqd.py) | Pipeline Classical SKQD (skqd.py) |
|------|-------------------------|-----------------------------------|
| 空間 | 完整 $2^n$ Hilbert space | 粒子數守恆子空間（$\ll 2^n$） |
| Hamiltonian 表示 | Pauli 字串 | Slater-Condon 規則 |
| 時間演化 | Lanczos matvec（Pauli masks） | `gpu_expm_multiply`（dense 子空間矩陣） |
| 用途 | 三路徑比較實驗 | Pipeline 生產路徑 |

Path C 故意在完整 $2^n$ 空間操作，以確保與 Path A/B 使用相同的 Hilbert 空間，維持受控實驗的公平性。Pipeline 的 Classical SKQD 則利用粒子數守恆子空間大幅加速，是生產環境的最佳選擇。
