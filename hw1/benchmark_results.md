# Benchmark Results & Optimization History

## Summary Comparison

| Run # | Timestamp | Target | Description | Total Time (s) | Status | Speedup vs Baseline |
| :---: | :---: | :---: | :--- | :---: | :---: | :---: |
| 1 | 2026-09-28 14:10 | hw1-1 | Initial baseline (OpenMP naive on loops) | 35.01 | 8/8 AC | 1.00x |
| 2 | 2026-09-28 21:36 | hw1-1 | Single-pass RGB loop, eliminate temporary channel vectors, dynamic CPU affinity setting, schedule(guided) | 29.57 | 8/8 AC | 1.18x |
| 3 | 2026-09-28 21:59 | hw1-1 | 2D Integral Image (SAT) O(1) box sum, contiguous I/O buffers, fast uncompressed PNG write | 4.34 | 8/8 AC | **8.07x** 🚀 |

---

## Detailed Records

### Run #3: 2D Integral Image (SAT) & Fast I/O
- **Target**: `hw1-1`
- **Date**: 2026-09-28 21:59
- **Total Time**: 4.34s (8/8 AC) — **New Best!**
- **Speedup vs Baseline**: 8.07x (vs v1: 6.81x)
- **Leaderboard Rank**: Jumped from #99 to #51!
- **Key Modifications**:
  1. **2D 積分圖（Summed-Area Table / SAT）演算法突破**：
     - 將原本 11x11（121 次迴圈）或 5x5（25 次迴圈）的暴力邊界走訪，徹底替換為 2D 前綴和查表。
     - 任何像素的方塊加總降為 **$O(1)$ 的 4 次陣列查表**，大圖運算量暴減 95% 以上。
     - 單純濾波運算時間從 6 秒暴降至 0.22 秒以內。
  2. **I/O 記憶體配置連續化**：
     - 移除 `read_png_file` 與 `write_png_file` 原本對每一列 row 個別 `malloc/free` 的 3700 次破碎小塊配置，改為單一大塊連續記憶體配置。
     - 像素轉換迴圈加入 `#pragma omp parallel for schedule(static)` 進行平行編解碼。
  3. **PNG 快速無損輸出**：
     - 啟用 `png_set_compression_level(0)` 與 `PNG_FILTER_NONE`，將 PNG 寫入時間從 3.0s 降至 0.04s（像素 100% 精確吻合）。

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
  1. **移除中間暫存陣列**：刪除了原版的 `redChannel`, `greenChannel`, `blueChannel`, `kernelSizes`, `tempRed`, `tempGreen`, `tempBlue` 共 6 個大尺寸 `vector<vector<int>>`，大幅減少 heap 記憶體配置與釋放開銷。
  2. **三通道單次迴圈合併（Single-pass RGB）**：將原本分開執行的 R、G、B 通道濾波合併到同一組迴圈中，同時計算 `sumR`, `sumG`, `sumB`，減少 3 倍的迴圈次數與記憶體重複走訪。
  3. **動態 CPU 核心數偵測**：加入 `sched_getaffinity` 取得 Slurm 分配的實際核心數並傳給 `omp_set_num_threads`。
  4. **OpenMP 排程優化**：外層迴圈加入 `#pragma omp parallel for schedule(guided)`。

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
