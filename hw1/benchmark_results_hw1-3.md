# HW 1-3 Benchmark Results & Optimization History

## Summary Comparison

| Run # | Timestamp | Target | Description | Total Time (s) | Status | Speedup vs Baseline |
| :---: | :---: | :---: | :--- | :---: | :---: | :---: |
| 0 | 2026-09-30 17:50 | hw1-3 | Sequential Baseline (Unparallelized) | ~160.0s (TLE on p03, p05, p07, p08, p10) | 5/10 AC (160 penalty) | 1.00x |
| 1 | 2026-09-30 18:05 | hw1-3 | OpenMP initial parallelization, fused energy reduction, eliminate 0.89 GB mat array | ~40.6s | Local pass | ~4.0x |
| 2 | 2026-09-30 18:23 | hw1-3 | Hoisted 1D row pointers, 2-level sphere bounding-box peeling (slice/row level) | ~36.2s | Local pass | ~4.4x |
| 3 | 2026-09-30 20:18 | hw1-3 | 64-byte aligned row strides, schedule(static, 4) interleaved chunks eliminating load imbalance | **34.48** | **10/10 AC (All Green)** | **~4.64x** 🚀 |

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
