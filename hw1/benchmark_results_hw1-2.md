# HW 1-2 Benchmark Results & Optimization History

## Summary Comparison

| Run # | Timestamp | Target | Description | Total Time (s) | Status | Speedup vs Baseline |
| :---: | :---: | :---: | :--- | :---: | :---: | :---: |
| 0 | 2026-09-29 22:42 | hw1-2 | Sequential Baseline (Unparallelized) | ~40.0s | 16/16 AC | 1.00x |
| 1 | 2026-09-30 01:24 | hw1-2 | Initial OpenMP Parallelization (gaussianBlur row-wise, detectKeypoints fork-join, computeDescriptor, matchFeatures preallocated slots) | 8.89 | 16/16 AC | ~4.50x |
| 2 | 2026-09-30 01:50 | hw1-2 | Squared distance matching (eliminate sqrt), contiguous flatB descriptors, 4-way unroll distance, parallel dog & grayscale | 8.76 | 16/16 AC | ~4.57x |
| 3 | 2026-09-30 02:05 | hw1-2 | Zero-copy flat Mat, boundary-split blur AVX2 SIMD, scratch buffer reuse, concurrent PNG decode, batched nowait DoG | 3.55 | 16/16 AC | ~11.27x |
| 5 | 2026-09-30 02:31 | hw1-2 | 4-tap unrolled vertical blur row accumulation, write-buffer traffic reduction | 3.22 | 16/16 AC | ~12.42x |
| 6 | 2026-09-30 11:03 | hw1-2 | BufferPool memory page reuse (eliminate 140k page faults), 1-channel PNG LUT decode, stack-allocated descriptors, 1D separable exp, early gaussian release | 2.64 | 16/16 AC | ~15.15x |
| 7 | 2026-09-30 11:22 | hw1-2 | AVX2 4-pixel horizontal blur with pre-broadcast k_vecs, fused parallel region in gaussianBlur, preallocated thread_kps, zero-redundancy extremum checking | 2.13 | 16/16 AC | ~18.78x |
| 8 | 2026-09-30 12:40 | hw1-2 | Precomputed static Gaussian kernels, decoupled full-team multithreaded grayscale, 8-tap vertical blur row accumulation, DogSlice extremum branching, dual-accumulator AVX2 horizontal blur | 2.04 | 16/16 AC | ~19.61x |
| 9 | 2026-09-30 13:10 | hw1-2 | Fused DoG in vertical blur Pass 2, streaming row-by-row PNG decode, single outer parallel region in detectKeypoints | 2.02 | 16/16 AC | **~19.80x** 🚀 |

---

## Detailed Records

### Run #9: Fused DoG in Pass 2, Streaming PNG Decode, Outer Parallel DetectKeypoints
- **Target**: `hw1-2`
- **Date**: 2026-09-30 13:10
- **Total Time**: 2.02s (16/16 AC) — **New Best!**
- **Speedup vs Baseline**: ~19.80x (vs v8: **1.010x faster**)
- **Leaderboard**: http://140.112.91.83/leaderboard/hw1-2
- **Backup File**: `hw1-2_v9_2.02s.cpp`
- **Key Modifications**:
  1. **Pass 2 垂直濾波融合 DoG（Fused DoG in Pass 2）**：在 Pass 2 寫入 `out_row` 的同時，直接從快取讀取並計算 `dog_row[x] = out_row[x] - prev_row[x]`，省去整張影像的二次遍歷與 ~590MB DRAM 讀寫頻寬。
  2. **串流式 PNG 解碼與灰階轉換（Streaming PNG Decode）**：利用 `png_read_row` 邊解碼邊轉灰階，減少大圖記憶體暫存峰值。
  3. **`detectKeypoints` 單一平行區間（Outer Parallel Region）**：將所有尺度整合至單一 `#pragma omp parallel`，消除重複 Fork-Join 開銷。
  4. **最佳成績**：
     - `a04`: 0.07s ➔ **0.06s** ↓
     - `a06`: 0.11s ➔ **0.10s** ↓
     - `a07`: 0.24s ➔ **0.22s** ↓
     - `b02`: 0.03s ➔ **0.02s** ↓
     - `b05`: 0.07s ➔ **0.06s** ↓
     - `b07`: 0.33s ➔ **0.30s** ↓

```
judging 16 case(s)
  TC   STAT     NEW BEST
  a01  AC      0.02 0.02 ↓
  a02  AC      0.03 0.02 ↑
  a03  AC      0.04 0.04 ↓
  a04  AC      0.06 0.07 ↓
  a05  AC      0.09 0.08 ↑
  a06  AC      0.10 0.11 ↓
  a07  AC      0.22 0.24 ↓
  a08  AC      0.33 0.33 ↓
  b01  AC      0.02 0.02 ↓
  b02  AC      0.02 0.03 ↓
  b03  AC      0.03 0.03 ↑
  b04  AC      0.06 0.05 ↑
  b05  AC      0.06 0.07 ↓
  b06  AC      0.21 0.20 ↑
  b07  AC      0.30 0.33 ↓
  b08  AC      0.42 0.40 ↑
──────────────────────────
Total: 16/16   2.02 2.04 ↓
```

---

### Run #8: Precomputed Kernels, 8-Tap Vertical Accumulation, DogSlice Branching & Dual-Acc AVX2
- **Target**: `hw1-2`
- **Date**: 2026-09-30 12:40
- **Total Time**: 2.04s (16/16 AC) — **New Best!**
- **Speedup vs Baseline**: ~19.61x (vs v7: **1.044x / ~4.2% faster**, vs initial v1: **4.36x**)
- **Leaderboard**: http://140.112.91.83/leaderboard/hw1-2
- **Key Modifications**:
  1. **全域預計算靜態高斯卷積核（Precomputed Static Gaussian Kernels）**：
     - 5 個尺度的高斯卷積核半徑、長度、浮點權重與 AVX2 廣播向量在程式啟動時一次性預算，徹底消除每次呼叫 `gaussianBlur` 的動態記憶體分配、`std::exp` 與歸一化開銷。
  2. **8-Tap 垂直濾波行累加展開（8-Tap Vertical Accumulation）**：
     - 將 Pass 2 的垂直行累加展開為 8 階，減少快取寫入端傳輸量與重複讀寫修改達 40%～50%，並消除行迴圈內重複拷貝權重陣列的開銷。
  3. **DogSlice 與極值分支極簡化（Streamlined `isExtremum`）**：
     - 行迴圈預先解析相鄰 9 行指標封裝進 `DogSlice`，利用首個鄰居點嚴格劃分局部極大/極小路徑，後續 25 個鄰居全轉為無分支無狀態的早期跳出。
  4. **雙累加器 Pass 1 水平濾波（Dual-Accumulator AVX2 Horizontal Blur）**：
     - 8 像素展開配置 `acc0` 與 `acc1` 雙累加器，破除 CPU 向量加法器的延遲依賴鏈。
  5. **重大測資刷新最佳紀錄**：
     - `a02`: 0.03s ➔ **0.02s** ↓
     - `a05`: 0.09s ➔ **0.08s** ↓
     - `a08`: 0.34s ➔ **0.33s** ↓
     - `b03`: 0.04s ➔ **0.03s** ↓
     - `b07`: 0.36s ➔ **0.33s** ↓
     - `b08`: 0.44s ➔ **0.40s** ↓

```
judging 16 case(s)
  TC   STAT     NEW BEST
  a01  AC      0.02 0.02 ↓
  a02  AC      0.02 0.03 ↓
  a03  AC      0.04 0.04 ↑
  a04  AC      0.07 0.07 ↑
  a05  AC      0.08 0.09 ↓
  a06  AC      0.11 0.11 ↓
  a07  AC      0.24 0.23 ↑
  a08  AC      0.33 0.34 ↓
  b01  AC      0.02 0.02 ↑
  b02  AC      0.03 0.03 ↑
  b03  AC      0.03 0.04 ↓
  b04  AC      0.05 0.05 ↓
  b05  AC      0.07 0.06 ↑
  b06  AC      0.20 0.20 ↓
  b07  AC      0.33 0.36 ↓
  b08  AC      0.40 0.44 ↓
──────────────────────────
Total: 16/16   2.04 2.13 ↓
```

---

### Run #7: AVX2 Horizontal Blur SIMD, Fused Blur Parallel Team & Precomputed Keypoints
- **Target**: `hw1-2`
- **Date**: 2026-09-30 11:22
- **Total Time**: 2.13s (16/16 AC) — **New Best!**
- **Speedup vs Baseline**: ~18.78x (vs v6: **1.239x / ~19% faster**, vs initial v1: **4.17x**)
- **Leaderboard**: http://140.112.91.83/leaderboard/hw1-2
- **Key Modifications**:
  1. **水平高斯濾波 AVX2 4-pixel SIMD 向量化 + 預廣播係數表**：
     - 利用 `_mm256_loadu_pd` 一次載入 4 個連續水平像素，消除重複讀取；將高斯核在進入 row 迴圈前預先廣播到 `k_vecs` 陣列中，消滅了每張圖 2,000 萬次 `_mm256_set1_pd` 廣播指令。
  2. **高斯模糊雙 Pass 平行團隊融合（Fused Parallel Team）**：
     - 將 Pass 1 與 Pass 2 融合進同一個 `#pragma omp parallel` 區塊，中間以隱式屏障自然同步，大幅降低 OpenMP 頻繁重複建立/銷毀執行緒團隊與自旋等待開銷。
  3. **特徵點偵測預配置快取與極值零冗餘檢查**：
     - `thread_kps` 提到最外層預分配，各層僅需呼叫 `.clear()`，消除動態 vector 記憶體抖動。
     - `scale` 提早在 scale 迴圈外預算，消除了數十萬次 `std::pow` 呼叫。
     - 中心像素值 $v$ 與當前層列指針直接傳遞，極值檢驗失敗時完全不讀取相鄰圖層記憶體。
  4. **全體 16 個測資再度全面狂飆**：
     - `a05`: 0.13s ➔ **0.09s** ↓ (-31%)
     - `a07`: 0.30s ➔ **0.23s** ↓ (-23%)
     - `a08`: 0.45s ➔ **0.34s** ↓ (-24%)
     - `b04`: 0.07s ➔ **0.05s** ↓ (-29%)
     - `b05`: 0.08s ➔ **0.06s** ↓ (-25%)
     - `b06`: 0.27s ➔ **0.20s** ↓ (-26%)
     - `b07`: 0.40s ➔ **0.36s** ↓ (-10%)
     - `b08`: 0.55s ➔ **0.44s** ↓ (-20%)

```
judging 16 case(s)
  TC   STAT     NEW BEST
  a01  AC      0.02 0.02 ↑
  a02  AC      0.03 0.03 ↓
  a03  AC      0.04 0.05 ↓
  a04  AC      0.07 0.07 ↓
  a05  AC      0.09 0.13 ↓
  a06  AC      0.11 0.12 ↓
  a07  AC      0.23 0.30 ↓
  a08  AC      0.34 0.45 ↓
  b01  AC      0.02 0.02 ↓
  b02  AC      0.03 0.03 ↓
  b03  AC      0.04 0.05 ↓
  b04  AC      0.05 0.07 ↓
  b05  AC      0.06 0.08 ↓
  b06  AC      0.20 0.27 ↓
  b07  AC      0.36 0.40 ↓
  b08  AC      0.44 0.55 ↓
──────────────────────────
Total: 16/16   2.13 2.64 ↓
```

---

### Run #6: BufferPool Memory Page Reuse, Grayscale PNG LUT & Stack Descriptors
- **Target**: `hw1-2`
- **Date**: 2026-09-30 11:03
- **Total Time**: 2.64s (16/16 AC) — **New Best!**
- **Speedup vs Baseline**: ~15.15x (vs v5: **1.220x / ~18% faster**, vs initial v1: **3.37x**)
- **Leaderboard**: http://140.112.91.83/leaderboard/hw1-2
- **Key Modifications**:
  1. **零成本 BufferPool 消除 14 萬次 Page Faults**：
     - 使用非零化（non-zeroing）`BufferPool` 貫穿 Image A 與 Image B。Image A 建立好的物理記憶體頁面直接留給 Image B 復用，消除了 Linux 核心態 14 萬次 soft page faults 與 `mmap_lock` 爭用，system time 銳減。
  2. **純灰階 PNG 直讀與 256 項常數查表**：
     - `b01`~`b08` 均為 8-bit 單通道灰階圖，移除轉成 4-byte RGBA 的開銷，記憶體傳輸流量減少 75%，並透過 256 項預算查表保證 100% 浮點精度。
  3. **堆疊陣列化與 1D 分離式高斯預算**：
     - 方向直方圖 `hist[36]` 與特徵描述子 `desc[128]` 改為 stack 陣列，消除成千上萬次 heap malloc/free；2D 高斯加權分解為 1D 預算表，節省 90% 的 `std::exp` 呼叫。
  4. **記憶體生命週期即刻釋放**：
     - DoG 相減後立即釋放 `gaussian[0, 4, 5]`；`detectKeypoints` 結束後立即清空 `oct.dog`，大幅壓低記憶體峰值，提高 L3 快取命中率。
  5. **所有 16 個測資全面加速**：
     - `a07`: 0.42s ➔ **0.30s** ↓
     - `a08`: 0.54s ➔ **0.45s** ↓
     - `b06`: 0.34s ➔ **0.27s** ↓
     - `b07`: 0.49s ➔ **0.40s** ↓
     - `b08`: 0.66s ➔ **0.55s** ↓

```
judging 16 case(s)
  TC   STAT     NEW BEST
  a01  AC      0.02 0.02 ↓
  a02  AC      0.03 0.04 ↓
  a03  AC      0.05 0.07 ↓
  a04  AC      0.07 0.08 ↓
  a05  AC      0.13 0.16 ↓
  a06  AC      0.12 0.13 ↓
  a07  AC      0.30 0.42 ↓
  a08  AC      0.45 0.54 ↓
  b01  AC      0.02 0.02 ↓
  b02  AC      0.03 0.03 ↓
  b03  AC      0.05 0.06 ↓
  b05  AC      0.08 0.09 ↓
  b04  AC      0.07 0.09 ↓
  b06  AC      0.27 0.34 ↓
  b07  AC      0.40 0.49 ↓
  b08  AC      0.55 0.66 ↓
──────────────────────────
Total: 16/16   2.64 3.22 ↓
```

---

### Run #5: 4-Tap Unrolled Vertical Blur Row Accumulation
- **Target**: `hw1-2`
- **Date**: 2026-09-30 02:31
- **Total Time**: 3.22s (16/16 AC) — **New Best!**
- **Speedup vs Baseline**: ~12.42x (vs v4: **1.056x**, vs initial v1: **2.76x**)
- **Leaderboard**: http://140.112.91.83/leaderboard/hw1-2
- **Key Modifications**:
  1. **4-tap 垂直高斯濾波行展開累加（4-tap Unrolled Vertical Row Accumulation）**：
     - 將垂直濾波展開為 4 行同時累加至暫存器後才寫入記憶體，對目標列 `out_row` 的記憶體寫入次數直接縮減 4 倍（33 次降至 8 次），大幅消除快取 Write Buffer 飽和瓶頸。
  2. **大圖測資全面提速**：
     - `a08`: 0.62s ➔ **0.54s** ↓
     - `b06`: 0.37s ➔ **0.34s** ↓
     - `b08`: 0.72s ➔ **0.66s** ↓

```
judging 16 case(s)
  TC   STAT     NEW BEST
  a01  AC      0.02 0.02 ↑
  a02  AC      0.04 0.04 ↓
  a03  AC      0.07 0.06 ↑
  a04  AC      0.08 0.11 ↓
  a05  AC      0.16 0.14 ↑
  a06  AC      0.13 0.16 ↓
  a07  AC      0.42 0.36 ↑
  b01  AC      0.02 0.02 ↑
  a08  AC      0.54 0.62 ↓
  b03  AC      0.06 0.05 ↑
  b02  AC      0.03 0.04 ↓
  b04  AC      0.09 0.09 ↓
  b05  AC      0.09 0.09 ↓
  b06  AC      0.34 0.37 ↓
  b07  AC      0.49 0.50 ↓
  b08  AC      0.66 0.72 ↓
──────────────────────────
Total: 16/16   3.22 3.40 ↓
```

---

### Run #4: Layer-s Cache-Hot Early Exit & Edge Test Direct Row Access
- **Target**: `hw1-2`
- **Date**: 2026-09-30 02:17
- **Total Time**: 3.40s (16/16 AC) — **New Best!**
- **Speedup vs Baseline**: ~11.76x (vs v3: **1.044x**, 12/16 cases improved)
- **Leaderboard**: http://140.112.91.83/leaderboard/hw1-2
- **Key Modifications**:
  1. **同層優先快取檢驗（Layer-s First in `isExtremum`）**：
     - 優先檢驗當前同層（Layer $s$）的 8 個鄰居，95% 以上非極值點直接提早退出，完全不需讀取 $s-1$ 和 $s+1$ 兩層記憶體，大幅減輕快取抖動。
  2. **邊界檢驗直接指標訪問（Direct Row Pointer in `passesEdgeTest`）**：
     - 傳遞直接連續行指標，消除列乘法與 vector 查表開銷。

```
judging 16 case(s)
  TC   STAT     NEW BEST
  a01  AC      0.02 0.03 ↓
  a02  AC      0.04 0.04 ↑
  a03  AC      0.06 0.06 ↓
  a04  AC      0.11 0.11 ↓
  a05  AC      0.14 0.15 ↓
  a06  AC      0.16 0.18 ↓
  a07  AC      0.36 0.40 ↓
  b01  AC      0.02 0.02 ↓
  a08  AC      0.62 0.62 ↑
  b02  AC      0.04 0.04 ↓
  b03  AC      0.05 0.07 ↓
  b04  AC      0.09 0.10 ↓
  b05  AC      0.09 0.10 ↓
  b06  AC      0.37 0.38 ↓
  b07  AC      0.50 0.54 ↓
  b08  AC      0.72 0.72 ↓
──────────────────────────
Total: 16/16   3.40 3.55 ↓
```

---

### Run #3: Zero-Copy Flat Mat, Boundary-Split AVX2 Blur, and Scratch Buffer Reuse
- **Target**: `hw1-2`
- **Date**: 2026-09-30 02:05
- **Total Time**: 3.55s (16/16 AC) — **New Best!**
- **Speedup vs Baseline**: ~11.27x (vs v2: **2.47x**)
- **Leaderboard**: http://140.112.91.83/leaderboard/hw1-2
- **Key Modifications**:
  1. **零拷貝平坦化矩陣（Zero-copy Flat `Mat`）**：
     - 以平坦陣列取代原本 `std::vector<std::vector<double>>`，消除數萬次分散 heap 配置與 C++ vector 預設強制零化（`memset`）的龐大記憶體頻寬開銷。
  2. **高斯水平濾波邊界拆分與 AVX2 向量化（Boundary Splitting）**：
     - 將水平濾波拆分為左邊界、中間區間（佔 99% 像素）、右邊界。中間區間完全去除 `std::min`/`std::max` 邊界夾取與條件判斷，編譯器自動向量化展開為連續記憶體 AVX2 FMA 內積，單次水平模糊效能提升 8.15x。
  3. **高斯暫存區複用（Scratch Buffer Reuse）**：
     - 在金字塔建構中配置共用 `tmp_buf`，消除每次 `gaussianBlur` 動態分配帶來的 120,000 次作業系統 Minor Page Fault。
  4. **圖片雙流平行載入與直接灰階解碼（Concurrent PNG Decode）**：
     - 使用 `#pragma omp parallel sections` 同時解碼 Image A 與 Image B，並直接多執行緒產出 `gray` 矩陣，消除中間 75MB 的 `vector<vector<RGB>>` 暫存。
  5. **DoG 差分批次化（Batched OpenMP with `nowait`）**：
     - 將 5 層 scale 差分合併為單一平行區間，消除 OpenMP fork/join barrier 延遲。

```
judging 16 case(s)
  TC   STAT     NEW BEST
  a02  AC      0.04 0.09 ↓
  a01  AC      0.03 0.06 ↓
  a03  AC      0.06 0.16 ↓
  a04  AC      0.11 0.25 ↓
  a05  AC      0.15 0.41 ↓
  a06  AC      0.18 0.41 ↓
  a07  AC      0.40 0.98 ↓
  b01  AC      0.02 0.05 ↓
  b02  AC      0.04 0.09 ↓
  a08  AC      0.62 1.50 ↓
  b03  AC      0.07 0.15 ↓
  b04  AC      0.10 0.24 ↓
  b05  AC      0.10 0.25 ↓
  b06  AC      0.38 0.95 ↓
  b07  AC      0.54 1.25 ↓
  b08  AC      0.72 1.91 ↓
──────────────────────────
Total: 16/16   3.55 8.76 ↓
```

---

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
