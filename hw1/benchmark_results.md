# Benchmark Results & Optimization History

## Summary Comparison

| Run # | Timestamp | Target | Description | Total Time (s) | Status | Speedup vs Baseline |
| :---: | :---: | :---: | :--- | :---: | :---: | :---: |
| 1 | 2026-09-28 14:10 | hw1-1 | Initial baseline (OpenMP naive on loops) | 35.01 | 8/8 AC | 1.00x |
| 2 | 2026-09-28 21:36 | hw1-1 | Single-pass RGB loop, eliminate temporary channel vectors, dynamic CPU affinity setting, schedule(guided) | 29.57 | 8/8 AC | 1.18x |

---

## Detailed Records

### Run #2: Single-pass RGB & Memory Reduction
- **Target**: `hw1-1`
- **Date**: 2026-09-28 21:36
- **Total Time**: 29.57s (8/8 AC) — **New Best!**
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
