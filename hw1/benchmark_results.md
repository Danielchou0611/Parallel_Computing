# Benchmark Results & Optimization History

## Summary Comparison

| Run # | Timestamp | Target | Description | Total Time (s) | Status | Speedup vs Baseline |
| :---: | :---: | :---: | :--- | :---: | :---: | :---: |
| 1 | 2026-09-28 14:10 | hw1-1 | Initial baseline (OpenMP naive on loops) | 35.01 | 8/8 AC | 1.00x |
| 2 | 2026-09-28 21:36 | hw1-1 | Single-pass RGB loop, eliminate temporary channel vectors, dynamic CPU affinity setting, schedule(guided) | 29.57 | 8/8 AC | 1.18x |
| 3 | 2026-09-28 21:59 | hw1-1 | 2D Integral Image (SAT) O(1) box sum, contiguous I/O buffers, fast uncompressed PNG write | 4.34 | 8/8 AC | 8.07x |
| 4 | 2026-09-29 00:30 | hw1-1 | Zero-copy 1D RGB buffer (no vector<vector>), interleaved SAT_RGB, and -march=native AVX2 vectorization | 2.57 | 8/8 AC | **13.62x** 🚀 |

---

## Detailed Records

### Run #4: Zero-copy 1D RGB Buffer & Interleaved SAT_RGB
- **Target**: `hw1-1`
- **Date**: 2026-09-29 00:30
- **Total Time**: 2.57s (8/8 AC) — **New Best!**
- **Speedup vs Baseline**: 13.62x (vs v2: 1.69x)
- **Leaderboard Rank**: Jumped from #53 to **#35**!
- **Key Modifications**:
  1. **徹底消滅二維 Vector（Zero-Copy 平坦化記憶體）**：
     - 將原本每像素 12 bytes 的 `struct RGB { int r, g, b; }` 二維 vector，全面改為緊湊的連續一維陣列 `uint8_t* raw_data`（每像素 3 bytes，僅 54 MB）。
     - 讀取 PNG 時啟用 `png_set_strip_alpha` 直接解碼為 3-byte RGB，寫入時也直接編碼，達成真正的零拷貝（Zero-Copy），消除數千萬次記憶體轉換與複製。
  2. **交錯式結構體 `SAT_RGB`（快取命中率提升）**：
     - 將原本分離的三個獨立 75 MB 陣列改為單一交錯結構 `struct SAT_RGB { uint32_t r, g, b; }`。
     - 單次 Cache Line 載入直接取得三通道累加和，大幅降低 L1/L2 快取缺失。
  3. **硬體 SIMD 向量化編譯旗標（`-march=native`）**：
     - 在 Makefile 中加入 `-march=native`，讓伺服器 Xeon 處理器自動啟用 AVX2 向量並行指令集。

```
judging 8 case(s)
  TC   STAT    NEW BEST
  t02  AC     0.04 0.06 ↓
  t01  AC     0.01 0.02 ↓
  t04  AC     0.20 0.30 ↓
  t03  AC     0.49 0.78 ↓
  t05  AC     0.38 0.63 ↓
  t06  AC     0.51 0.89 ↓
  t07  AC     0.45 0.79 ↓
  t08  AC     0.48 0.86 ↓
─────────────────────────
Total: 8/8    2.57 4.34 ↓
```

---

### Run #3: 2D Integral Image (SAT) & Fast I/O
- **Target**: `hw1-1`
- **Date**: 2026-09-28 21:59
- **Total Time**: 4.34s (8/8 AC)
- **Speedup vs Baseline**: 8.07x
- **Leaderboard Rank**: Jumped from #99 to #51!
- **Key Modifications**:
  1. 2D 積分圖（SAT）演算法：將 11x11 暴力迴圈改為 $O(1)$ 的 4 次查表。
  2. I/O 記憶體配置連續化：改為大塊連續記憶體。
  3. PNG 快速輸出：`png_set_compression_level(0)`。

```
judging 8 case(s)
  TC   STAT    NEW  BEST
  t02  AC     0.06  0.20 ↓
  t01  AC     0.02  0.06 ↓
  t04  AC     0.30  2.86 ↓
  t03  AC     0.78  2.60 ↓
  t05  AC     0.63  6.17 ↓
  t06  AC     0.89  5.86 ↓
  t07  AC     0.79  5.93 ↓
  t08  AC     0.86  5.89 ↓
──────────────────────────
Total: 8/8    4.34 29.57 ↓
```

---

### Run #2: Single-pass RGB & Memory Reduction
- **Target**: `hw1-1`
- **Date**: 2026-09-28 21:36
- **Total Time**: 29.57s (8/8 AC)
- **Speedup vs Baseline**: 1.18x
- **Key Modifications**:
  1. 移除中間 6 個暫存陣列。
  2. 三通道單次走訪。
  3. 動態 CPU 核心數偵測。
  4. OpenMP 排程優化 `schedule(guided)`。

```
judging 8 case(s)
  TC   STAT    NEW  BEST
  t01  AC     0.06  0.08 ↓
  t02  AC     0.20  0.25 ↓
  t04  AC     2.86  3.33 ↓
  t03  AC     2.60  3.28 ↓
  t06  AC     5.86  7.08 ↓
  t05  AC     6.17  7.04 ↓
  t07  AC     5.93  6.99 ↓
  t08  AC     5.89  6.96 ↓
──────────────────────────
Total: 8/8   29.57 35.01 ↓
```

---

### Run #1: Initial Baseline
- **Target**: `hw1-1`
- **Date**: 2026-09-28 14:10
- **Total Time**: 35.01s (8/8 AC)
- **Description**: Initial baseline with OpenMP parallel for on 2D `std::vector<std::vector<int>>` and channels processed separately.

```
judging 8 case(s)
  TC   STAT    NEW
  t01  AC     0.08
  t02  AC     0.25
  t04  AC     3.33
  t03  AC     3.28
  t05  AC     7.04
  t06  AC     7.08
  t07  AC     6.99
  t08  AC     6.96
──────────────────
Total: 8/8   35.01
```
