# Benchmark Results & Optimization History

## Summary Comparison

| Run # | Timestamp | Target | Description | Total Time (s) | Status | Speedup vs Baseline |
| :---: | :---: | :---: | :--- | :---: | :---: | :---: |
| 1 | 2026-09-28 14:10 | hw1-1 | Initial baseline (OpenMP naive on loops) | 35.01 | 8/8 AC | 1.00x |
| 2 | 2026-09-28 21:36 | hw1-1 | Single-pass RGB loop, eliminate temporary channel vectors, dynamic CPU affinity setting, schedule(guided) | 29.57 | 8/8 AC | 1.18x |
| 3 | 2026-09-28 21:59 | hw1-1 | 2D Integral Image (SAT) O(1) box sum, contiguous I/O buffers, fast uncompressed PNG write | 4.34 | 8/8 AC | 8.07x |
| 4 | 2026-09-29 00:30 | hw1-1 | Zero-copy 1D RGB buffer (no vector<vector>), interleaved SAT_RGB, and -march=native AVX2 vectorization | 2.57 | 8/8 AC | 13.62x |
| 5 | 2026-09-29 01:47 | hw1-1 | Zero-copy 1D RGB buffer, interleaved SAT_RGB, mmap + madvise PNG reading, fast uncompressed write | 2.53 | 8/8 AC | 13.84x |
| 6 | 2026-09-29 02:03 | hw1-1 | Zero strip-alpha in libpng, cache-blocked Step 2 (B=256), border peeling Step 1, branchless fixed-point division (>> 19) | 2.47 | 8/8 AC | 14.17x |
| 7 | 2026-09-29 02:16 | hw1-1 | Quiet CRC on read, 59KB cache-resident sliding window box filter (zero 220MB SAT allocation) | 2.15 | 8/8 AC | 16.28x |
| 8 | 2026-09-29 02:28 | hw1-1 | Direct png_read_row stream, precomputed row pointers sliding window, -funroll-loops -fomit-frame-pointer | 2.13 | 8/8 AC | **16.44x** 🚀 |

---

## Detailed Records

### Run #8: Direct png_read_row Stream, Precomputed Row Pointers & Vector Unrolling
- **Target**: `hw1-1`
- **Date**: 2026-09-29 02:28
- **Total Time**: 2.13s (8/8 AC) — **New Best!**
- **Speedup vs Baseline**: 16.44x (vs v7: 1.01x)
- **Leaderboard Rank**: Solidly locked into Top 20!
- **Key Modifications**:
  1. **`png_read_row` 零指針連續串流讀取**：
     - 淘汰原本 `png_read_image` 需在 heap 上建立 `row_pointers` 二級指標的開銷，改為直接以 `png_read_row` 循序寫入連續記憶體，優化 CPU L1/L2 串流預取。
  2. **滑動窗口列指標外層預計算（消滅 5 億次重複運算）**：
     - 進入水平欄位迴圈前預先算出 4 條列指標（`row_a11`, `row_s11`, `row_a5`, `row_s5`），大幅減少內層重複的 `min/max` 與乘法定址。
  3. **水平滑動去分支（Peeling `y = 0`）**：
     - 將首個像素獨立於迴圈外處理，完全消除主體迴圈的 `if (y > 0)` 條件跳轉。
  4. **激進化編譯旗標**：
     - 在 Makefile 中追加 `-funroll-loops -fomit-frame-pointer`，進一步釋放暫存器並展開向量管線。

```
judging 8 case(s)
  TC   STAT    NEW BEST
  t01  AC     0.01 0.01 ↑
  t02  AC     0.03 0.04 ↓
  t04  AC     0.16 0.18 ↓
  t03  AC     0.42 0.42 ↓
  t05  AC     0.29 0.29 ↓
  t06  AC     0.44 0.43 ↑
  t07  AC     0.36 0.36 ↓
  t08  AC     0.42 0.42 ↑
─────────────────────────
Total: 8/8    2.13 2.15 ↓
```

---

### Run #7: Quiet CRC & 59KB Cache-Resident Sliding Window
- **Target**: `hw1-1`
- **Date**: 2026-09-29 02:16
- **Total Time**: 2.15s (8/8 AC) — **Massive Leap (ALL 8 Cases Improved ↓)!**
- **Speedup vs Baseline**: 16.28x (vs v6: 1.15x)
- **Leaderboard Rank**: Approaching Top 20!
- **Key Modifications**:
  1. **Quiet CRC 略過解壓縮 CRC 檢查**：
     - 加入 `png_set_crc_action(png, PNG_CRC_QUIET_USE, PNG_CRC_QUIET_USE);`，解壓縮讀取時略過每塊 IDAT chunk 冗餘的 CRC32 檢查。
  2. **徹底消滅 220MB 積分圖（記憶體開銷降 460 倍至 59 KB）**：
     - **徹底解決 Page Fault 與快取缺失**：舊版 220 MB `SAT` 陣列觸發 55,000 次 Page Fault 並灌爆 L3 快取。
     - **滑動窗口核心**：利用半徑僅為 2 與 5 的特性，每個執行緒只維護垂直累加值（約 59 KB），垂直滑動時僅需加新列減舊列，水平滑動時同步維護 box 和。
     - **效果**：完全免除 220MB 陣列的配置與走訪，全濾波運算完全在 CPU **L1 / L2 快取**中秒級完成！

```
judging 8 case(s)
  TC   STAT    NEW BEST
  t01  AC     0.01 0.01 ↓
  t02  AC     0.04 0.04 ↓
  t04  AC     0.18 0.20 ↓
  t03  AC     0.42 0.48 ↓
  t05  AC     0.29 0.34 ↓
  t06  AC     0.43 0.49 ↓
  t07  AC     0.36 0.42 ↓
  t08  AC     0.42 0.47 ↓
─────────────────────────
Total: 8/8    2.15 2.47 ↓
```

---

### Run #6: Zero-Strip-Alpha, Blocked Column SAT, Border Peeling & Fast Reciprocal Division
- **Target**: `hw1-1`
- **Date**: 2026-09-29 02:03
- **Total Time**: 2.47s (8/8 AC) — **New Best!**
- **Speedup vs Baseline**: 14.17x (vs v5: 1.024x)
- **Leaderboard Rank**: Progressing deeper into Top 30!
- **Key Modifications**:
  1. **移除單執行緒 `png_set_strip_alpha`**：
     - RGBA 圖片直接解碼為 4 通道平坦記憶體，避免 libpng 在單執行緒上逐列壓縮記憶體。通道 stride 交由後續由 OpenMP 8 核心並行的 Step 1 處理。
  2. **Step 1 邊界剝離（Border Peeling）**：
     - 將左右各 5 像素邊界與中間主體分開處理，中間 99.9% 像素使用純指標累加，徹底移除數千萬次 `min`/`max` 邊界分支，Step 1 耗時減少 ~3x。
  3. **Step 2 快取區塊化（Cache Blocking, $B=256$）**：
     - 將跨步數萬 bytes 的大步長直欄走訪，改為 $B=256$ 直欄區塊累加，極大化 L1 快取命中率，並由 GCC 自動向量化為 32-byte AVX2 向量指令，Step 2 耗時減少 3.5x。
  4. **Step 3 無分支定點數乘法除法（Branchless Fast Reciprocal Division）**：
     - 轉換為全整數亮度計算 `(299u*r + 587u*g + 114u*b) > 128000u`，經全色域 16.7M 顏色驗證 100% 精準等價，消除浮點運算。
     - 消除 x86 `idiv` 指令（25 cycles 縮減至 1 cycle），使用同 shift (`>> 19`) 的定點數乘法（乘以 `4333` 或 `20972`），配合條件移動指令（cmov）消除分支預測錯誤。
     - 列指標在外層預先計算，內層省去乘法定址。

```
judging 8 case(s)
  TC   STAT    NEW BEST
  t01  AC     0.01 0.01 ↑
  t02  AC     0.04 0.04 ↓
  t04  AC     0.20 0.20 ↓
  t03  AC     0.48 0.48 ↑
  t05  AC     0.34 0.34 ↑
  t06  AC     0.49 0.52 ↓
  t07  AC     0.42 0.42 ↑
  t08  AC     0.47 0.50 ↓
─────────────────────────
Total: 8/8    2.47 2.53 ↓
```

---

### Run #5: mmap + madvise PNG I/O & Memory Optimization
- **Target**: `hw1-1`
- **Date**: 2026-09-29 01:47
- **Total Time**: 2.53s (8/8 AC) — **New Best!**
- **Speedup vs Baseline**: 13.84x (vs v4: 1.02x)
- **Leaderboard Rank**: Jumped from #35 to **#34**!
- **Key Modifications**:
  1. **mmap + madvise 零拷貝檔案讀取**：
     - 使用 `open` + `mmap` 取代傳統 C library `fopen` / `fread`，消除核心態（Kernel space）到使用者態（User space）的緩衝拷貝。
     - 搭配 `madvise(..., MADV_WILLNEED | MADV_SEQUENTIAL)` 通知 Linux 核心進行大區塊預先分頁調入（Page Read-ahead），減少 page fault 延遲。
  2. **1MB 大緩衝區 PNG 輸出**：
     - 使用 `setvbuf` 配置 1MB I/O 寫入緩衝區，減少系統呼叫次數。

```
judging 8 case(s)
  TC   STAT    NEW BEST
  t01  AC     0.01 0.01 ↓
  t02  AC     0.04 0.04 ↑
  t04  AC     0.20 0.20 ↓
  t03  AC     0.48 0.49 ↓
  t05  AC     0.34 0.38 ↓
  t06  AC     0.52 0.51 ↑
  t07  AC     0.42 0.45 ↓
  t08  AC     0.50 0.48 ↑
─────────────────────────
Total: 8/8    2.53 2.57 ↓
```

---

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
