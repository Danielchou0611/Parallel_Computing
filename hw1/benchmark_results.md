# Benchmark Results & Optimization Master Log

本文件整合所有作業的優化歷史與評測數據，詳細歷史請見各自分冊：
- [HW 1-1 詳細優化紀錄](benchmark_results_hw1-1.md)
- [HW 1-2 詳細優化紀錄](benchmark_results_hw1-2.md)

---

## 🏆 HW 1-1: Adaptive Denoising Filter

- **當前最佳成績**: **`1.77s`** (8/8 AC) — **19.78x 加速比** 🚀
- **Leaderboard**: http://140.112.91.83/leaderboard/hw1-1
- **主要檔案**: `hw1-1.cpp` (備份: `hw1-1_v8_2.13s.cpp`, `hw1-1_v9_1.77s.cpp`)

| Run # | 日期時間 | 版本摘要 | 總秒數 (s) | 狀態 | 相比 Baseline 加速比 |
| :---: | :---: | :--- | :---: | :---: | :---: |
| 1 | 2026-09-28 14:10 | 原始 Baseline (迴圈直出 OpenMP) | 35.01 | 8/8 AC | 1.00x |
| 2 | 2026-09-28 21:36 | Single-pass RGB 走訪、排程 guided | 29.57 | 8/8 AC | 1.18x |
| 3 | 2026-09-28 21:59 | 2D 積分圖 (SAT) O(1) 查表、快速 PNG 輸出 | 4.34 | 8/8 AC | 8.07x |
| 4 | 2026-09-29 00:30 | 1D 平坦陣列 Zero-Copy、交錯結構體、-march=native | 2.57 | 8/8 AC | 13.62x |
| 5 | 2026-09-29 01:47 | mmap + madvise 系統級預讀取 | 2.53 | 8/8 AC | 13.84x |
| 6 | 2026-09-29 02:03 | 邊界剝離 (Border Peeling)、快取區塊化 (B=256)、無分支定點數乘除法 | 2.47 | 8/8 AC | 14.17x |
| 7 | 2026-09-29 02:16 | 59KB 快取常駐滑動窗口濾波、Quiet CRC 檢查 | 2.15 | 8/8 AC | 16.28x |
| 8 | 2026-09-29 02:28 | png_read_row 零指標串流、列指標外層預計算 | 2.13 | 8/8 AC | 16.44x |
| 9 | 2026-09-29 23:24 | 自製 PNG chunk parser、單次 IDAT 解壓、SWAR 還原、平行 stored deflate (8 核心 adler32/crc32) | **1.77** | 8/8 AC | **19.78x** 🏆 |

---

## 🎯 HW 1-2: Feature Detection & Matching (SIFT-style)

- **當前最佳成績**: **`2.64s`** (16/16 AC) — **突破 15.15x 加速比！** 🚀
- **Leaderboard**: http://140.112.91.83/leaderboard/hw1-2
- **主要檔案**: `hw1-2.cpp` (備份: `hw1-2_v1_8.89s.cpp`, `hw1-2_v2_8.76s.cpp`, `hw1-2_v3_3.55s.cpp`, `hw1-2_v4_3.40s.cpp`, `hw1-2_v5_3.22s.cpp`, `hw1-2_v6_2.64s.cpp`)

| Run # | 日期時間 | 版本摘要 | 總秒數 (s) | 狀態 | 相比 Baseline 加速比 |
| :---: | :---: | :--- | :---: | :---: | :---: |
| 0 | 2026-09-29 22:42 | 官方 Sequential Baseline (未平行化) | ~40.0 | 16/16 AC | 1.00x |
| 1 | 2026-09-30 01:24 | 初版四大核心 OpenMP 平行化 (gaussianBlur 列平行、detectKeypoints Fork-Join、描述子平行、matchFeatures 預留格子免鎖) | 8.89 | 16/16 AC | ~4.50x |
| 2 | 2026-09-30 01:50 | 平方距離比對 (消滅 sqrt)、B 圖 flatB 連續記憶體、4-way 展開、補齊 dog/grayscale 盲區平行化 | 8.76 | 16/16 AC | ~4.57x |
| 3 | 2026-09-30 02:05 | Zero-copy 平坦 Mat、高斯水平濾波邊界拆分 (AVX2 SIMD)、暫存區複用消滅 12 萬次 Page Fault、雙流 PNG 平行解碼、DoG nowait | 3.55 | 16/16 AC | ~11.27x |
| 4 | 2026-09-30 02:17 | 極值檢驗 Layer-s 優先快取預判 (提早跳出 95% 候選)、邊界檢驗直接連續行指標訪問 | 3.40 | 16/16 AC | ~11.76x |
| 5 | 2026-09-30 02:31 | 4-tap 垂直高斯濾波行展開累加 (大幅消除快取 Write Buffer 飽和) | 3.22 | 16/16 AC | ~12.42x |
| 6 | 2026-09-30 11:03 | BufferPool 記憶體跨圖完全復用 (消滅 14 萬次 Page Faults)、純灰階 PNG 直讀 + 256 項常數查表、描述子 Stack 陣列化 + 1D 分離式 Exp、無用 Gaussian 提早釋放 | **2.64** | 16/16 AC | **~15.15x** 🏆 |
