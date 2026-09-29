# HW 1-2 Benchmark Results & Optimization History

## Summary Comparison

| Run # | Timestamp | Target | Description | Total Time (s) | Status | Speedup vs Baseline |
| :---: | :---: | :---: | :--- | :---: | :---: | :---: |
| 0 | 2026-09-29 22:42 | hw1-2 | Sequential Baseline (Unparallelized) | ~40.0s | 16/16 AC | 1.00x |
| 1 | 2026-09-30 01:24 | hw1-2 | Initial OpenMP Parallelization (gaussianBlur row-wise, detectKeypoints fork-join, computeDescriptor, matchFeatures preallocated slots) | 8.89 | 16/16 AC | ~4.50x |
| 2 | 2026-09-30 01:50 | hw1-2 | Squared distance matching (eliminate sqrt), contiguous flatB descriptors, 4-way unroll distance, parallel dog & grayscale | 8.76 | 16/16 AC | **~4.57x** 🚀 |

---

## Detailed Records

### Run #2: Squared Distance Matching & Contiguous Descriptor Streaming
- **Target**: `hw1-2`
- **Date**: 2026-09-30 01:50
- **Total Time**: 8.76s (16/16 AC) — **New Best!**
- **Speedup vs Baseline**: ~4.57x (vs v1: 1.015x)
- **Leaderboard**: http://140.112.91.83/leaderboard/hw1-2
- **Key Modifications**:
  1. **平方距離比對（消滅 640 萬次 `std::sqrt()`）**：
     - 在比大小和次小值時使用 $d_{sq}$，比值檢驗改為平方比 $0.5625$，只在配對成功時開根號一次。
  2. **B 圖特徵平坦化連續記憶體（`flatB`）**：
     - 將 B 圖特徵點描述子連續排布，消除內層迴圈每次訪問 `std::vector` 的指標跳躍。
  3. **4-way 獨立累加器管線展開**：
     - 128 維距離計算以 4 條獨立管線累加，提升 CPU 運算吞吐量。
  4. **補齊盲區平行化**：
     - `toGrayscale`、`downsample2x`、`oct.dog` 矩陣運算加入 `#pragma omp parallel for`。

```
judging 16 case(s)
  TC   STAT     NEW BEST
  a01  AC      0.06 0.06 ↑
  a02  AC      0.09 0.12 ↓
  a03  AC      0.16 0.15 ↑
  a04  AC      0.25 0.24 ↑
  a05  AC      0.41 0.38 ↑
  a06  AC      0.41 0.41 ↑
  a07  AC      0.98 1.01 ↓
  b01  AC      0.05 0.05 ↓
  b02  AC      0.09 0.09 ↓
  a08  AC      1.50 1.61 ↓
  b03  AC      0.15 0.15 ↓
  b04  AC      0.24 0.24 ↓
  b05  AC      0.25 0.26 ↓
  b06  AC      0.95 0.98 ↓
  b07  AC      1.25 1.35 ↓
  b08  AC      1.91 1.78 ↑
──────────────────────────
Total: 16/16   8.76 8.89 ↓
```

---

### Run #1: Initial OpenMP Parallelization on All 4 Core Stages
- **Target**: `hw1-2`
- **Date**: 2026-09-30 01:24
- **Total Time**: 8.89s (16/16 AC) — **First Official Score on Leaderboard!**
- **Leaderboard**: http://140.112.91.83/leaderboard/hw1-2
- **Key Modifications**:
  1. **`computeDescriptor` & `assignOrientation` 特徵描述平行化**：
     - 使用 `#pragma omp parallel for` 平行計算各個獨立特徵點的方向與 128 維描述子。
  2. **`gaussianBlur` 高斯濾波列平行化**：
     - 水平（Pass 1）與垂直（Pass 2）可分離濾波的橫列迴圈加入 `#pragma omp parallel for`。
  3. **`detectKeypoints` Fork-Join 無鎖收集**：
     - 各執行緒擁有私有小容器 `thread_kps[tid]`，利用 `schedule(static)` 平行偵測後依序合併，100% 保持特徵點原始順序。
  4. **`matchFeatures` 預留格子免鎖比對**：
     - 預先配置長度為 $N$ 的 `match_per_kp` 陣列，各執行緒只寫入自己的專屬格子，避免 Race Condition 並確保比對輸出順序。
  5. **編譯器防護**：
     - 加入 `-ffp-contract=off` 消除 FMA 浮點誤差，確保輸出與測資 100% 精準吻合。

```
judging 16 case(s)
  TC   STAT     NEW
  a01  AC      0.06
  a02  AC      0.12
  a03  AC      0.15
  a04  AC      0.24
  a05  AC      0.38
  a06  AC      0.41
  a07  AC      1.01
  b01  AC      0.05
  b02  AC      0.09
  b03  AC      0.15
  a08  AC      1.61
  b04  AC      0.24
  b05  AC      0.26
  b06  AC      0.98
  b07  AC      1.35
  b08  AC      1.78
───────────────────
Total: 16/16   8.89
```
