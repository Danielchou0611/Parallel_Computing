# HW 1-3 Benchmark Results & Optimization History

## Summary Comparison

| Run # | Timestamp | Target | Description | Total Time (s) | Status | Speedup vs Baseline |
| :---: | :---: | :---: | :--- | :---: | :---: | :---: |
| 0 | 2026-09-30 17:50 | hw1-3 | Sequential Baseline (Unparallelized) | ~160.0s (TLE on p03, p05, p07, p08, p10) | 5/10 AC (160 penalty) | 1.00x |
| 1 | 2026-09-30 18:05 | hw1-3 | OpenMP initial parallelization, fused energy reduction, eliminate 0.89 GB mat array | ~40.6s | Local pass | ~4.0x |
| 2 | 2026-09-30 18:23 | hw1-3 | Hoisted 1D row pointers, 2-level sphere bounding-box peeling (slice/row level) | ~36.2s | Local pass | ~4.4x |
| 3 | 2026-09-30 20:18 | hw1-3 | 64-byte aligned row strides, schedule(static, 4) interleaved chunks eliminating load imbalance | 34.48 | 10/10 AC | ~4.64x |
| 4 | 2026-09-30 21:55 | hw1-3 | mmap 2MB Huge Pages, Parallel First-Touch NUMA, and Dead Computation Elimination | 26.45 | 10/10 AC | ~6.05x |
| 5 | 2026-09-30 22:07 | hw1-3 | schedule(static, 1) fine-grained round-robin sphere load balancing (Rank 59) | 22.43 | 10/10 AC (Rank 59) | ~7.13x |
| 6 | 2026-09-30 22:41 | hw1-3 | Row-level sphere filtering and fine-tuned distribution (Rank 55) | 21.21 | 10/10 AC (Rank 55) | ~7.54x |
| 7 | 2026-10-01 10:48 | hw1-3 | Stored-a architecture full-memory optimization | 20.91 | 10/10 AC | ~7.65x |
| 8 | 2026-10-01 14:13 | hw1-3 | Direct interval reaction, dynamic stored-a tiling, zero-copy init | 10.18 (local) | 10/10 AC | — |
| 9 | 2026-10-01 19:04 | hw1-3 | CPU affinity auto-binding, BJ L2 cache tuning, field_preseed, omp simd | **18.87** | **10/10 AC (官方新高紀錄)** | **~8.48x** 🚀 |

---

## Detailed Records

### Run #3: 64-Byte Cache Alignment & schedule(static, 4) Load Balancing
- **Target**: `hw1-3`
- **Date**: 2026-09-30 20:18
- **Total Time**: **34.48s** (10/10 AC) — **First Official Board Submission!**
- **Speedup vs Baseline**: ~4.64x (Penalty 160.0s ➔ 34.48s)
- **Leaderboard**: http://140.112.91.83/leaderboard/hw1-3
- **Backup File**: `hw1-3_v3_34.48s.cpp`
- **Key Modifications**:
  1. **消滅負載不均（Eliminate Load Imbalance via schedule(static, 4)）**：原本粗暴按 $N/8$ 切分會讓有球體的切片集中在少數 2~3 條執行緒，其他執行緒閒置。改為每 4 片輪流分發（Interleaved Round-Robin），讓全體 8 核心始終保持 100% 運算負載。
  2. **64-Byte 快取記憶體行對齊（64-Byte Row Alignment）**：將行步長 $SJ$ 補齊至 8 個 double 的整數倍，消滅所有 AVX2 跨快取線（Split-Cache-Line）載入懲罰。
  3. **雙層邊界盒篩選（2-Level Bounding Box Peeling）**：超過 90% 的切片完全不呼叫 `reactive()`，純向量化流式計算。
  4. **消滅 `mat` 陣列（Eliminate mat Array）**：記憶體直接省下 0.89 GB。
  5. **各測資全數突破為 AC**：
     - `p03`: 2.99s TLE ➔ **1.17s AC** (時限 2.0s)
     - `p05`: 7.39s TLE ➔ **3.33s AC** (時限 5.0s)
     - `p07`: 10.19s TLE ➔ **5.14s AC** (時限 7.5s)
     - `p08`: 10.63s TLE ➔ **6.23s AC** (時限 7.5s)
     - `p10`: 14.70s TLE ➔ **8.73s AC** (時限 10.0s)

```
judging 10 case(s)
  TC   STAT     NEW BEST
  p02  AC      0.15 0.17 ↓
  p01  AC      0.18 0.25 ↓
  p03  AC      1.17    - ↓
  p04  AC      1.99 2.10 ↓
  p05  AC      3.33    - ↓
  p06  AC      2.78 2.87 ↓
  p07  AC      5.14    - ↓
  p08  AC      6.23    - ↓
  p09  AC      4.78 5.00 ↓
  p10  AC      8.73    - ↓
──────────────────────────
Total: 10/10  34.48  160 ↓
```

### Run #8: Direct Interval Reaction, 64B Pointer Assumptions, Dynamic Stored-A Tiling & Zero-Copy Init
- **Target**: `hw1-3`
- **Date**: 2026-10-01 14:13
- **Total Time (Local)**: **10.180s** (10/10 AC)
- **Backup File**: `hw1-3_v7_10.18s.cpp`
- **Key Modifications**:
  1. **直通區間無分支反應計算（Direct Interval Execution Without Branching）**：
     重構 `build_reaction_intervals` 的執行邏輯，徹底消滅原先 `klo .. khi` 中逐點判斷的 `while (interval < n_intervals && k > intervals[interval].hi)` 與三元運算子 `is_react ? react(r) : r`。直接按合併區間切分：
     - 非反應間隔：100% 透過 `#pragma GCC ivdep` 進行 SIMD 向量化計算。
     - 反應間隔：無條件直通呼叫 `react(r)`，完全消除分支預測錯誤與純量搜尋開銷。
  2. **消除重複記憶體歸零（Zero-Copy mmap Anonymous Init）**：
     Linux `mmap` 的匿名頁面核心本來即保證初始化為 `0.0`。移除了初始化階段對 `unew` 及 `u` 光暈周圍的多餘 `memset` 與重寫，節省了數十億位元組的記憶體快取與 TLB 負擔。
  3. **動態 J 步長分發（Dynamic BJ_STORED）**：
     針對 $N=128$（如 `p02`）將 `BJ_STORED` 自 32 改為 16，確保 8 核心環境下全數 8 執行緒皆獲分發（原本僅 4 塊導致 4 核心完全閒置），`p02` 耗時直接由 0.059s 縮短至 **0.038s**（約 35% 加速）。
  4. **快取行 64 位元組對齊提示（__builtin_assume_aligned）**：
     行指針全數加入 64-byte 對齊宣告，使編譯器得以自動合併記憶體運算元（Folding Memory Operands），大幅減輕暫存器壓力並消除未對齊載入開銷。
  5. **向量迴圈展開（#pragma GCC unroll 4）**：
     於核心 Stencil 向量化迴圈加入局部展開提示，提高指令級平行度（ILP）。
  6. **實測成效**：
     全測資 10/10 100% 通過驗證（能量誤差 $\le 10^{-10}$，採樣溫度誤差 $0.00$），本地總耗時突破至 **10.180s**！

---

### Run #9: CPU Affinity Auto-Binding, BJ L2 Cache Tuning, field_preseed & omp simd
- **Target**: `hw1-3`
- **Date**: 2026-10-01 19:04
- **Total Time**: **18.87s** (10/10 AC) — **突破 18 秒大關！官方記分板歷史新高紀錄！**
- **Speedup vs Baseline**: **~8.48x** 🚀 (Penalty 160.0s ➔ 18.87s)
- **Leaderboard**: http://140.112.91.83/leaderboard/hw1-3
- **Backup File**: `hw1-3_v8_18.87s.cpp`
- **Key Modifications**:
  1. **Slurm Affinity 自動檢測與執行緒精準綁定（sched_getaffinity + omp_set_num_threads）**：
     透過 `sched_getaffinity` 動態讀取 Slurm 分配的 CPU 遮罩核心數（2、4 或 8），自動呼叫 `omp_set_num_threads` 設定完全對齊的執行緒數。徹底根絕了在 2 核（`p01`, `p04`）與 4 核（`p06`）環境下 OpenMP 預設建立過多執行緒互相爭搶、劇烈 Context Switch 的瓶頸！
     - `p04`: 1.19s ➔ **1.17s**（創新低）
     - `p06`: 1.47s ➔ **1.38s**（大幅超前歷史最佳）
  2. **大尺寸工作集快取調校（BJ = (N >= 640) ? 8 : 16）**：
     針對 $N \ge 640$（`p08`, `p09`, `p10`），將 $J$ 切塊大小從 16 調小為 8，使每條執行緒的 3-plane slab 運算工作集精準常駐於 Xeon Silver 處理器的 **1MB L2 快取**內，徹底消滅大尺寸下的 Cache Thrashing：
     - `p08`: 3.38s ➔ **3.33s**（創新低）
     - `p05`: 2.50s ➔ **2.45s**（創新低）
     - `p07`: 3.48s ➔ **3.42s**（創新低）
  3. **PRNG seed_mix 預乘與常數折疊（field_preseed）**：
     將每個格點原本重複進行的 `seed * 0x9E3779B97F4A7C15ull` 提出至初始化與外層計算，在數十億次格點訪問中省下大量 64 位元整數乘法。
  4. **全迴圈 OpenMP SIMD 向量化與 SIMD Reduction**：
     非反應區域全面以 `#pragma omp simd` 取代純量提示，大幅提升向量化吞吐量。
- **Scoreboard Breakdown**:
  ```
  judging 10 case(s)
    TC   STAT     NEW  BEST
    p01  AC      0.15  0.15 ↑
    p02  AC      0.10  0.10 ↓
    p03  AC      0.99  0.99 ↓
    p04  AC      1.17  1.19 ↓
    p06  AC      1.38  1.47 ↓
    p05  AC      2.45  2.50 ↓
    p07  AC      3.42  3.48 ↓
    p08  AC      3.33  3.38 ↓
    p09  AC      1.64  1.61 ↑
    p10  AC      4.23  4.22 ↑
  ───────────────────────────
  Total: 10/10  18.87 19.09 ↓ (最佳總和進一步推進至 18.82s)
  ```


