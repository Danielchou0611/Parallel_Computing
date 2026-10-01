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
| 6 | 2026-09-30 22:41 | hw1-3 | Row-level sphere filtering and fine-tuned distribution (Rank 55) | **21.21** | **10/10 AC (Rank 55)** | **~7.54x** 🚀 |
| 7 | 2026-10-01 11:09 | hw1-3 | Low-memory j-tile reduced from 16 to 8 rows | ~12.57 (local) | 10/10 AC | — |
| 8 | 2026-10-01 14:13 | hw1-3 | Direct interval reaction, aligned pointer hints, dynamic stored-a tiling, zero-copy init | **10.18 (local, latest)** | **10/10 AC** | **~15.7x** 🚀 |

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

