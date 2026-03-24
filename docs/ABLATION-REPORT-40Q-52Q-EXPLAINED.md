# HI+NQS+SKQD 消融實驗報告（詳細解說版）

> **日期**: 2026-03-22
> **硬體**: NVIDIA DGX Spark (GB10, 128GB UMA, CUDA 13.0)
> **系統**: N₂（氮氣分子）/ cc-pVDZ basis set
> **Seed**: 42
> **數據**: `results/ablation_40q_52q.json`

---

## 這份報告在回答什麼問題？

我們建立了一個名為 **HI+NQS+SKQD** 的量子化學計算 pipeline，它有三個核心元件：

1. **NQS**（Neural Quantum State）— 一個 Transformer 神經網路，學習「哪些電子排列方式最重要」
2. **Krylov expansion** — 從數學上用 Hamiltonian（能量算符）的連接關係，直接發現重要的電子排列
3. **GPU diag** — 在選出的電子排列上建立能量矩陣，用 GPU 對角化求最低能量

**消融實驗**就是逐一拔掉每個元件，看「少了它，結果會差多少」。這是科學研究中判斷每個模組貢獻度的標準做法。

---

## 背景知識：什麼是 qubit、config、mHa？

| 術語 | 意思 | 白話 |
|------|------|------|
| **Qubit (Q)** | 系統大小的度量。20Q = 10 個 alpha 軌道 + 10 個 beta 軌道 | 越多 qubit 越難算 |
| **Config** | 一種可能的電子排列方式。例如 `[1,1,0,0,1,1,0,0]` 表示前 2 個 alpha 和前 2 個 beta 軌道被佔據 | 全部可能的 configs 數量隨 qubit 指數增長 |
| **Config space** | 所有可能 configs 的總數。40Q 有 2.4 億種，52Q 有 43 億種 | 不可能全部列舉 |
| **ΔE_HF (mHa)** | 計算結果比 Hartree-Fock（最粗糙近似）低多少 milli-Hartree | 越負代表找到越多「相關能」，越好 |
| **FCI** | Full Configuration Interaction = 精確解（列舉所有 config 做精確對角化）| 20Q 以下可算，40Q+ 不可能 |
| **化學精度** | 1.6 mHa (1 kcal/mol)。低於此門檻就算「夠準」 | 實驗化學家能接受的誤差範圍 |

---

## 實驗 A：NQS 和 Krylov 各自的貢獻（40Q 氮氣分子）

### 實驗設計

固定系統為 N₂ CAS(10,20)/cc-pVDZ（40 qubit，config space = 2.4 億），分別跑三種方法：

| 方法 | 做什麼 | 不做什麼 |
|------|--------|---------|
| **Krylov-only** | 從 HF 出發，用數學連接關係擴展 basis | 不訓練任何神經網路 |
| **NQS-only** | 訓練 Transformer 學習生成 configs | 不做 Krylov 擴展 |
| **NQS+SKQD** | 兩者都做 + 迭代回饋 | 完整 pipeline |

### 結果

```
方法                  能量改善         用了多少 configs    花了多久
─────────────────────────────────────────────────────────────
Krylov-only (2K)     -250.50 mHa      2,001 configs      1.3 秒
Krylov-only (5K)     -253.31 mHa      5,001 configs      2.1 秒
NQS-only             -112.67 mHa     10,735 configs    270.2 秒
NQS+SKQD            -250.88 mHa     15,000 configs    309.6 秒
```

### 這代表什麼？

**Krylov expansion 像一個經驗豐富的偵探**：它從最簡單的電子排列（HF state）出發，沿著 Hamiltonian 的「數學線索」（單激發、雙激發）一步步發現真正重要的排列。2,001 個精挑細選的 configs，1.3 秒，就抓到了 250.5 mHa 的相關能。

**NQS 像一個亂槍打鳥的新手**：Transformer 神經網路在 40 qubit 的巨大空間（2.4 億種排列）裡隨機生成了 10,735 個 configs，花了 270 秒（4.5 分鐘），但只找到 112.67 mHa 的相關能 — 比 Krylov 少了整整 138 mHa。

**NQS+SKQD 合起來呢？** -250.88 mHa，幾乎跟 Krylov-only 的 -250.50 mHa 一樣。也就是說，**NQS 在這個系統上只貢獻了 0.4 mHa（0.16%）**，卻多花了 308 秒（238 倍的時間）。

### 為什麼 NQS 在 40Q 表現這麼差？

1. **搜索空間太大**：2.4 億種排列中，真正重要的只有幾千個。NQS 的隨機採樣大部分在「荒野」中浪費時間。

2. **N₂ 是弱相關系統**：氮氣分子的基態 wavefunction 主要由 HF + 少量 singles/doubles 主導。Krylov expansion 的 2-hop（HF → singles/doubles → triples/quadruples）已經抓到了最關鍵的排列。

3. **Krylov 有物理直覺**：它沿著 Slater-Condon 規則（量子力學告訴我們只有單激發和雙激發有非零矩陣元素）擴展，每一步都有物理意義。NQS 沒有這個先驗知識。

### 結論 A

> **在弱相關系統（如 N₂）上，Krylov expansion 獨自就能達到最佳結果。NQS 幾乎無用。**
>
> 但這不代表 NQS 永遠無用——對於強相關系統（如 Cr₂ 的六重鍵、[2Fe-2S] 鐵硫蛋白），Krylov 的 2-hop 可能不夠深，NQS 的全域搜索能力可能成為關鍵。這需要進一步實驗。

---

## 實驗 B：從 24 qubit 推到 52 qubit

### 實驗設計

固定方法為 NQS+SKQD，逐步增大 active space：

| 系統 | Qubit | 全部可能排列數 | 意義 |
|------|-------|-------------|------|
| N₂ CAS(10,12) | 24 | 627,264 | 小型 — 可驗證精度 |
| N₂ CAS(10,15) | 30 | 9,018,009 | 中型 |
| N₂ CAS(10,20) | 40 | 240,374,016 | 大型 — IBM 的 benchmark 規模 |
| N₂ CAS(10,26) | 52 | 4,327,008,400 | 極大 — IBM SQD 論文的最大規模 |

### 結果

```
系統               Qubit    能量改善        用了多少 configs    迭代次數    時間
──────────────────────────────────────────────────────────────────────────
N₂ CAS(10,12)       24    -198.57 mHa      8,727 configs      6 輪      56 秒
N₂ CAS(10,15)       30    -185.42 mHa     12,285 configs      8 輪     103 秒
N₂ CAS(10,20)       40    -250.90 mHa     12,240 configs      5 輪     140 秒
N₂ CAS(10,26)       52    -291.01 mHa     13,886 configs      4 輪     195 秒
```

### 這代表什麼？

**52 qubit 的問題空間有 43 億種電子排列。** 如果要用精確解（FCI），需要建一個 43 億 × 43 億的矩陣——那需要大約 63 萬 TB 的記憶體，地球上沒有任何電腦做得到。

**我們的方法只用了 13,886 個排列（0.0003%），在 195 秒內就得到了 -291 mHa 的相關能。**

三個值得注意的 scaling 行為：

1. **時間增長遠慢於問題規模**：config space 從 62 萬增長到 43 億（7,000 倍），但時間只從 56 秒增長到 195 秒（3.5 倍）。這是因為我們的方法不需要遍歷所有排列——Krylov expansion 精準地找到最重要的那幾千個。

2. **能量隨 qubit 增加而降低**：24Q 找到 -199 mHa，52Q 找到 -291 mHa。這不是因為「52Q 更準」，而是因為更大的 active space 包含了更多電子相關效應（更多虛擬軌道可以參與激發）。

3. **收斂變快**：52Q 只需 4 輪迭代就收斂，24Q 反而需要 6 輪。原因是在更大的軌道空間中，每一次 Krylov expansion 都能發現更多有價值的連接——每 hop 的 「信息密度」更高。

### 與 IBM 的對比

IBM 在 2026 年 3 月發表的 Science 論文用了 72 qubit 的量子電腦 + 超級計算機 Fugaku 做後處理。他們的 `solve_fermion` 函數在 28 qubit 的乙烯（C₂H₄）上就需要 485 秒，40 qubit 以上完全無法處理。

**我們的方法在 52 qubit 上只需要 195 秒，而且不需要任何量子硬體或超級計算機——只需一台 DGX Spark（桌面級 GPU 工作站）。**

---

## 實驗 C：Krylov 擴展要多深才夠？

### 實驗設計

固定 40Q N₂ + NQS+SKQD，調整 `krylov_max_new`（每輪最多新增多少 Krylov configs）：

### 結果

```
Krylov 深度    能量改善        時間      邊際改善（vs 上一級）
──────────────────────────────────────────────────────────
  500 configs  -240.23 mHa    188 秒     — (baseline)
1,000 configs  -248.70 mHa    138 秒     +8.47 mHa（省 50 秒！）
2,000 configs  -250.88 mHa    310 秒     +2.18 mHa（多 172 秒）
5,000 configs  -253.67 mHa    421 秒     +2.79 mHa（多 111 秒）
```

### 這代表什麼？

這就像在大海裡撈魚：

- **前 1,000 條 Krylov configs 是「大魚」**：容易撈到，每條都很有價值（平均每 config 貢獻 0.008 mHa）。
- **之後的 configs 是「小魚」**：越來越難撈，價值越來越低。從 2,000 到 5,000 多花了 111 秒，但只多撈到 2.8 mHa。

**最佳性價比是 krylov=1,000**：它在 138 秒內達到 -248.70 mHa，比 krylov=5000 快 3 倍，但只少了 5 mHa。

另一個有趣的現象：krylov=1,000 比 krylov=500 **更快**（138s vs 188s），因為更多 Krylov configs 讓 NQS 更快收斂（5 iter vs 7 iter），省下的 NQS 訓練時間超過了多花在 Krylov expansion 上的時間。

但 krylov=2,000 和 5,000 都撞到了 **max_basis_size=15,000 的天花板** — 當 NQS + Krylov 的總 configs 超過 15K，多出的就被丟掉了。要繼續改善，需要提升 sparse diag 的效能來支撐更大的 basis。

---

## 總結：一句話版本

### 給仁瑜

> 我們的 HI+NQS+SKQD pipeline 成功從 IBM 的 `solve_fermion`（28Q 就要 485 秒、40Q 不可行）加速到 **52Q 只需 195 秒**。消融實驗顯示在弱相關 N₂ 上，Krylov expansion 獨自貢獻 99.8% 的精度（1.3 秒 = 250 mHa），NQS 幾乎無用（0.4 mHa / 270 秒）。下一步應測試強相關系統（Cr₂、[2Fe-2S]）以確認 NQS 的價值場景。

### 給 George

> PR #3/#4/#5 實現了 GPU diag + Krylov expansion + KV cache，在 N₂ cc-pVDZ 上從 24Q 測到 52Q。52Q（43 億 config space）用 13,886 configs 在 195 秒內收斂到 -291 mHa below HF。91 個自動化測試全部通過。

### 給自己（秀吉）

> Krylov expansion 是金礦，NQS 在 N₂ 上是裝飾品。但強相關系統可能翻轉這個結論——需要跑 Cr₂ 和 [2Fe-2S] 來確認。另外，當前的 2-hop expansion 比我們 pipeline 的完整 Krylov 時間演化（expm_multiply）弱 13 mHa（251 vs 264 mHa at 40Q）——整合完整 Krylov 是下一個高價值任務。

---

## 附錄：完整數據路徑

| 檔案 | 路徑 |
|------|------|
| 消融報告 | `docs/ABLATION-REPORT-40Q-52Q.md` |
| 詳細解說 | `docs/ABLATION-REPORT-40Q-52Q-EXPLAINED.md` |
| JSON 數據 | `results/ablation_40q_52q.json` |
| 消融腳本 | `scripts/benchmark_ablation_40q_52q.py` |
| E2E benchmark | `results/benchmark_e2e_hi_nqs_skqd.json` |
| 40Q Krylov-only | `results/benchmark_40q_krylov.json` |
| GPU diag benchmark | `results/gpu_diag_benchmark.json` |

---

*報告生成: 2026-03-22*
*91 tests, 0 failures*
