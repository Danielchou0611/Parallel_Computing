# HW 2-1: MPI 分散式圖形最短路徑 (SSSP) 開發與優化歷程記錄

本文件記錄臺大平行程式設計（Parallel Programming）**HW 2-1: Shortest Paths on a Distributed Graph** 的系統架構設計、版本迭代、驗證結果與效能 Profile 分析。

---

## 1. 作業規格與系統限制概覽

* **目標**：給定隨機生成的大規模 Kronecker (R-MAT) 無向帶權圖，計算 $S$ 個源點（$1 \le S \le 32$）到所有頂點的最短路徑長度（SSSP）。
* **執行環境**：
  * **單一節點（1 Node）、24 個單核心 MPI 行程（24 ranks）**。
  * **私有記憶體上限**：每個 rank 僅有 **1200 MiB (`RLIMIT_DATA`)**，超額拋出 `std::bad_alloc`。
  * **共享記憶體例外**：**MPI Shared-Memory Windows (`MPI_Win_allocate_shared`) 不計入 1200 MiB 上限**，算在整台節點的 48 GiB 內。
  * **執行緒限制**：嚴禁任何多執行緒（OpenMP、`std::thread`、pthreads、parallel STL 等，裁判會抓 thread count）。
* **圖形特徵**：
  * 規模：$N = 2^{\text{scale}}$（最大 scale 26，頂點數約 $6.7 \times 10^7$）。
  * 邊數：$M = \text{edgefactor} \times N$（最大邊數可達 $2^{30} \approx 10.7$ 億條邊）。
  * 邊權重：整數 $w \in [1, 255]$。
  * 動態生成：透過 `generate_edges` 即時計算，**圖的生成時間計入總 Compute Time**。

---

## 2. 版本迭代歷程

```
[原始 Baseline] ──> [Version 1: Shared CSR 骨架] ──> [Version 2: 平行建圖]
                                                              │
[Version 4: 小 S 協同 SSSP] <── [Version 3.5: Profile 計時] <── [Version 3: Dial's 演算法]
```

---

### Version 1：記憶體破局與架構骨架 (Shared Memory CSR)

* **核心問題**：原始版本僅 rank 0 在私有記憶體建完整 CSR，在 scale $\ge 22$ 直接爆 1.2 GiB 崩潰。
* **解決架構**：
  1. 使用 `MPI_Comm_split_type(..., MPI_COMM_TYPE_SHARED, ...)` 建立節點內部 communicator。
  2. 使用 `MPI_Win_allocate_shared` 配置整張圖 CSR（`row`, `col`, `w`）與平坦結果收集區（`res_hdr`, `sample_dist_flat`）。
  3. 由 Rank 0 採用 Chunked Two-Pass（每次 1M 邊）生成圖形，私有記憶體僅需約 9 MB 暫存。
  4. 各 rank 平分 $S$ 個 sources 進行標準 `priority_queue` Dijkstra，算完直接寫入共享結果區。
  5. 函式結束前透過 `MPI_Win_free` 與 `MPI_Comm_free` 乾淨釋放資源。
* **驗證結果**：
  * 通過 Case 01, 02, 03, 04, 14, 15, 16, 17，全部 Byte-for-byte MATCH 官方 `golden.bin`。

---

### Version 2：兩階段平行無鎖建圖 (Parallel CSR Construction)

* **核心問題**：邊生成計入 Compute Time，單核生成 10 億條邊耗時過長。
* **演算法設計**：
  1. **Pass 1（局部度數統計）**：
     * 各 rank 負責邊區間 $[r \cdot M/P, (r+1) \cdot M/P)$。
     * 配置共享記憶體 `all_degree[P][N]`（型別 `uint32_t`，Scale 26 約佔 6.4 GB，節點 48 GB 完全放得下）。
     * 各 rank 僅寫入自己的 `all_degree[r][v]`，**零 Lock、零 Atomic、零 False Sharing**。
  2. **Prefix Sum & 相對偏移轉換**：
     * 將頂點區間平分給各 rank，局部累加 24 個 rank 的度數。
     * 使用 `MPI_Exscan` 獲得全域前綴和基底 `rank_base`（Rank 0 手動歸零）。
     * **原地轉置**：將 `all_degree[r][v]` 原地轉換為 rank 專屬的 `relative_offset[r][v]`。
  3. **Pass 2（無鎖填入 CSR）**：
     * 各 rank 重新生成邊，直接以 `pu = row[uk] + all_degree[r][uk]++` 寫入 `col` 與 `w`。
     * 各 rank 寫入範圍互不重疊，完全無鎖無競爭，且無需額外私有 cursor 陣列。
* **Invariant 檢查**：
  * 驗證 `row[0] == 0`、`row[N] == 2*M`，確認無資料競爭。
* **效能提升（本機 4 ranks）**：
  * Case 01：0.093s $\rightarrow$ **0.041s (2.27x)**
  * Case 02：2.229s $\rightarrow$ **0.914s (2.43x)**
  * Case 04：11.87s $\rightarrow$ **5.11s (2.32x)**

---

### Version 3：Dial's 演算法 (256 Circular Bucket Queue)

* **核心問題**：$std::priority_queue$ 時間複雜度為 $O(M \log N)$，在大圖上開銷巨大。
* **演算法設計**：
  * 觀察到邊權重 $w \in [1, 255]$ 為小整數。
  * 當前處理距離為 $d$ 時，新鬆弛出的頂點距離必落在 $[d+1, d+255]$。
  * 使用 **256 個環狀桶**：`bucket_index = distance & 255u`。
  * `BucketEntry { uint32_t vertex; uint32_t distance; }` 保留完整距離，用於驗證輪次與過濾 Stale Entry。
  * 實作可重用的 `DialQueue`，在跨 sources 時 `clear()` 不釋放 capacity，消除反覆記憶體配置。
  * 加入斷言保護：`if (entry.distance != current) MPI_Abort(...)` 驗證窗口無碰撞。
* **驗證結果**：
  * 全部測資（包含極端邊界與超大圖 Case 10）**100% Byte-for-byte MATCH**！
  * 碰撞斷言**從未觸發**，理論完全成立。
* **效能提升（本機 4 ranks）**：
  * Case 01：0.041s $\rightarrow$ **0.032s**
  * Case 02：0.914s $\rightarrow$ **0.702s**
  * Case 04：5.11s $\rightarrow$ **4.02s**
  * Case 05：6.48s $\rightarrow$ **4.14s**

---

### Version 3.5：階段拆解 Profile 量測與關鍵瓶頸發現

為精確量化各階段耗時，加入 `MPI_Wtime()` 與 `MPI_Reduce(..., MPI_MAX)` 進行量測：

#### 實測數據彙總（本機 4 ranks）

| 測試案例 | Scale | 邊數 $M$ | $S$ | 分配 (Alloc) | Pass 1 (度數) | Prefix Sum | Pass 2 (填 CSR) | **建圖總計** | **SSSP (Dial)** | **總運算時間** | 評測限額 (24 ranks) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Case 04** | 22 | $6.7 \times 10^7$ | 4 | 0.01 s | 0.75 s | 0.01 s | 1.99 s | **2.76 s (70%)** | **1.10 s (28%)** | 3.92 s | 5.0 s |
| **Case 07** | 24 | $2.68 \times 10^8$| 4 | 0.03 s | 2.93 s | 0.03 s | 9.46 s | **12.44 s (70%)** | **4.99 s (28%)** | 17.69 s | 10.0 s |
| **Case 09** | 25 | $5.36 \times 10^8$| 4 | 0.05 s | 6.37 s | 0.06 s | 24.69 s | **31.18 s (72%)** | **11.37 s (26%)** | 43.12 s | 20.0 s |
| **Case 10** | 26 | $1.07 \times 10^9$| 4 | 0.11 s | 17.84 s | 0.13 s | 65.64 s | **83.72 s (74%)** | **28.23 s (25%)** | 113.15 s | 55.0 s |

#### 關鍵洞察
1. **建圖階段佔比高達 70% ~ 74%**：
   * Dial's 演算法本身極快，SSSP 僅佔 25% ~ 28%。
   * 建圖中的 **Pass 2（填寫 `col` 與 `w`）佔了總時間的近 58%**，主因是隨機向數 GB 記憶體寫入產生的快取未命中與匯流排頻寬限制。
2. **24 Ranks 上的預估與挑戰**：
   * 24 ranks 會將邊數均分，建圖時間預估加速 4.5x ~ 5x：
     * Case 09 建圖時間預估降至 $\approx 6.5\text{s}$。
     * Case 10 建圖時間預估降至 $\approx 18\text{s}$。
   * **但若維持純 Source-level 平行**（$S=4$ 時僅 4 個 rank 運算，20 個 rank 閒置）：
     * Case 09 SSSP 依然需要單核約 $11.4\text{s}$，總時間 $6.5 + 11.4 \approx 17.9\text{s}$（逼近 20s 限額）。
     * Case 10 SSSP 依然需要單核約 $28.2\text{s}$，總時間 $18 + 28.2 \approx 46.2\text{s}$（在 55s 內，但餘裕較小）。

---

## 3. 下一步：Version 4 (Team-based 協同 SSSP) 設計規範

針對 $S < 12$ 且圖形龐大的情況，避免 20 個 ranks 閒置的架構設計：

```
[24 Ranks] ──> 依 S 切分成 Team（例如 S=4，切成 4 個 Team，每 Team 6 ranks）
                 │
                 ├── Team 0 (Ranks 0~5)   ──> 協同處理 Source 0
                 ├── Team 1 (Ranks 6~11)  ──> 協同處理 Source 1
                 ├── Team 2 (Ranks 12~17) ──> 協同處理 Source 2
                 └── Team 3 (Ranks 18~23) ──> 協同處理 Source 3
```

### Version 4A 架構規劃（最小改動、最穩健）
1. **Communicator 分割**：
   * `team_id = rank % S;`
   * `MPI_Comm_split(MPI_COMM_WORLD, team_id, rank, &team_comm);`
2. **共享 `dist` 陣列**：
   * 每個同時執行的 source 在共享記憶體中配置獨立的 `dist[N]`（4 sources 共 1 GB）。
   * 平行初始化：Team 內各 rank 平分區間填入 `INF`。
3. **私有 Local Buckets + Atomic Min**：
### Version 4：Hybrid Multi-Team 協同 SSSP 實作與驗證成果

依據建議順序，我們依序完成了 Chunk Size 最佳化實驗、單 Team 協同驗證、多 Team 分組協同與 Hybrid Selector：

#### 1. 建圖 Pass 2 Chunk Size 最佳化實驗
測試 Case 04（scale 22）在不同邊 Chunk 大小下的建圖時間：
* $2^{16}$ (65,536 edges)：Pass 2 2.033s | Build 2.794s
* $2^{17}$ (131,072 edges)：Pass 2 2.028s | Build 2.784s
* $2^{18}$ (262,144 edges)：Pass 2 2.005s | Build 2.765s
* **$2^{19}$ (524,288 edges，約 4.5 MB 暫存)**：**Pass 2 1.901s | Build 2.640s (最快甜點點！)**
* $2^{20}$ (1,048,576 edges)：Pass 2 1.966s | Build 2.722s
* $2^{21}$ (2,097,152 edges)：Pass 2 1.995s | Build 2.750s

我們將 `CHUNK_SIZE` 預設值採用最佳甜點點 **524,288 ($2^{19}$)**。

#### 2. Version 4.0: 單 Team 協同 SSSP (Single Source)
* 共享 `dist` 陣列（使用 `atomic_min`），各 rank 暫存私有 local buckets。
* 每一距離層透過 `MPI_Exscan` 平行合併成 `shared_frontier`。
* Team 內 ranks 透過原子工作指標 `__atomic_fetch_add(&work_index, 64)` 動態領取工作塊，徹底解決高度數 hub 負載不均問題。
* 每一輪透過 `MPI_Allreduce(MPI_MIN)` 推進全域下一個距離。
* **測試驗證**：
  * **Case 16** (scale 7, S=1)：**MATCH!**
  * **自訂 Scale 16 (S=1)**：SSSP 耗時僅 **0.003s**，**MATCH!**
  * **自訂 Scale 20 (S=1)**：SSSP 從單核 2.21s 降至 **0.044s**，**MATCH!**
  * **自訂 Scale 22 (S=1)**：總時間從單核 11.55s 降至 **3.08s**，SSSP 僅 **0.240s**，**MATCH!**

#### 3. Version 4.1 & 4.2: 多 Team 協同與 Hybrid Selector
* 自動依據 $S$ 與 rank 數量將 ranks 切割成 $T = \min(P, S)$ 個 teams（例如 24 ranks, $S=4$ 形成 4 個 6-rank teams）。
* 每個 Team 配置獨立的共享 `dist`、`shared_frontier` 與 `work_index`，各 Team 獨立並行計算各自的 source。
* **Hybrid 條件**：
  `bool use_coop = (S <= 4 && world_size >= 4 && world_size > S && p.scale >= 20);`
  * 當 $S$ 較大（如 $S=8, 16, 32$）或小圖時，自動使用 Version 3（純 Source 並行，無 Team 同步開銷）。
  * 當 $S \le 4$ 且為大圖時，自動啟用 Multi-Team 協同運算。
* **測試驗證**：
  * **自訂 Scale 22 (S=2, 2 teams $\times$ 2 ranks)**：
    * 單核總時間：12.43s $\rightarrow$ 4 ranks Multi-Team 總時間：**3.41s**（SSSP 僅 **0.57s**）！
    * 與 golden 比較：**100% Byte-for-byte MATCH!**
  * 所有邊界測資（Case 14, 15, 16, 17）與常規測資（Case 01, 02, 03, 04, 07）**全數 MATCH!**

### Version 5A & 5B：單次邊生成快取與 Absolute Cursor（Pass 2 極限優化）

針對佔總時間 70%~74% 的建圖階段進行深入優化：

#### 1. Version 5A: 單次邊生成快取 (Single-Pass Edge Caching)
* **機制**：
  * 在 Pass 1 生成每塊邊時，將每條邊打包成 60-bit 緊湊格式（`uint64_t`: $u$ 26 bits, $v$ 26 bits, $w$ 8 bits）。
  * 暫存至私有向量 `packed_edges(local_M)`（最大測資僅約 341.3 MiB，遠低於 1200 MiB 上限）。
  * Pass 2 **完全不再重複呼叫 `generate_edges`**，直接從 `packed_edges` 解包！
  * 進入 SSSP 前，執行 `std::vector<uint64_t>().swap(packed_edges)` 徹底釋放記憶體。
* **收益（4 ranks 本機實測）**：
  * Case 04 Pass 2 耗時：2.03s $\rightarrow$ **1.40s**
  * Case 07 Pass 2 耗時：9.50s $\rightarrow$ **7.34s**
  * Case 09 Pass 2 耗時：24.69s $\rightarrow$ **20.07s**（現省 4.6 秒！）
  * Case 10 Pass 2 耗時：65.64s $\rightarrow$ **56.86s**（現省 8.8 秒！）

#### 2. Version 5B: Absolute Cursor（消除 Pass 2 的隨機 row 讀取）
* **機制**：
  * 在 Prefix Sum 階段，直接把絕對位移寫入各 rank 的 cursor：
    `all_degree[idx] = static_cast<uint32_t>(running_row + rel_offset);`
  * 因 $2M \le 2^{31} < 2^{32}$，絕對位移完全相容於 `uint32_t`，不增加記憶體開銷。
  * Pass 2 寫入直接執行：`pu = my_degree[uk]++;` 與 `pv = my_degree[vk]++;`。
  * **完全消除了 Pass 2 對 `row[uk]` 與 `row[vk]` 的隨機 DRAM 讀取**！
* **收益（4 ranks 本機實測）**：
  * **Case 04**：Pass 2 降至 **1.08s**（建圖從 2.76s 降至 **1.91s**，總時間 **3.07s**）
  * **Case 07**：Pass 2 降至 **5.91s**（建圖從 12.45s 降至 **9.19s**，總時間從 17.75s 降至 **14.31s**）
  * **Case 09**：Pass 2 降至 **15.46s**（建圖從 31.18s 降至 **22.57s**，總時間從 43.12s 降至 **34.67s**，省下 8.45 秒！）
  * **Case 10**：Pass 2 降至 **42.00s**（建圖從 83.72s 降至 **62.97s**，總時間從 113.15s 降至 **95.10s**，省下 18.05 秒！）
* **正確性驗證**：全部測資依然 **100% Byte-for-byte MATCH!**

---

## 4. SEIREN 官方評測紀錄 (Judge Results)

### 第一次完整提交 (2026/10/12)
* **評測環境**：SEIREN 叢集（24 單核心 MPI ranks on 1 node, 48 GiB 記憶體）
* **版本架構**：Version 4（平行建圖 + Dial's 256 環狀桶 + Multi-Team 協同 SSSP + 512K 最佳 Chunk）
* **評測結果**：**17 / 17 全部通過 (ALL ACCEPTED, AC)**
* **總計時間**：**88.05 秒**
* **計分板連結**：http://140.112.91.83/leaderboard/hw2-1

#### 測資耗時明細表

| 測資 (TC) | 狀態 | 執行時間 (NEW) | 官方限額 (Limit) | 餘裕比例 / 評語 |
| :---: | :---: | :---: | :---: | :---: |
| **01** | **AC** | **0.03 s** | 2.0 s | 極快 (66x 餘裕) |
| **02** | **AC** | **0.31 s** | 2.0 s | 順暢通過 (6.4x 餘裕) |
| **03** | **AC** | **0.28 s** | 3.0 s | 順暢通過 (10.7x 餘裕) |
| **04** | **AC** | **1.41 s** | 5.0 s | 順暢通過 (3.5x 餘裕) |
| **05** | **AC** | **2.23 s** | 20.0 s | 極佳 (8.9x 餘裕) |
| **06** | **AC** | **6.11 s** | 8.0 s | 通過 (低於 8s 限額) |
| **07** | **AC** | **6.33 s** | 10.0 s | 順暢通過 (3.6s 餘裕) |
| **08** | **AC** | **5.44 s** | 12.0 s | 順暢通過 (2.2x 餘裕) |
| **09** | **AC** | **17.60 s** | 20.0 s | 通過 (低於 20s 限額) |
| **10** | **AC** | **34.99 s** | 55.0 s | **最大圖完美通過 (20s 餘裕)** |
| **11** | **AC** | **2.86 s** | 14.0 s | 順暢通過 (4.8x 餘裕) |
| **12** | **AC** | **4.51 s** | 30.0 s | 順暢通過 (6.6x 餘裕) |
| **13** | **AC** | **5.94 s** | 12.0 s | 順暢通過 (2.0x 餘裕) |
| **14** | **AC** | **0.00 s** | 2.0 s | 瞬間通過 |
| **15** | **AC** | **0.00 s** | 2.0 s | 瞬間通過 |
| **16** | **AC** | **0.00 s** | 2.0 s | 瞬間通過 |
| **17** | **AC** | **0.00 s** | 2.0 s | 瞬間通過 |
| **總計** | **ALL AC** | **88.05 s** | - | **17/17 完美 Accepted** |

---

## 5. 檔案與備份對照

| 檔案 | 說明 |
| :--- | :--- |
| `hw2-1-v2` | Version 2 可執行檔（平行建圖 + Heap Dijkstra） |
| `hw2-1-v3` | Version 3 可執行檔（平行建圖 + 256 環狀桶 Dial's 演算法） |
| `hw2-1-v4` | Version 4 備份（首次通過 17/17 AC，總時間 88.05s） |
| `hw2-1.cpp`| 最新正式程式碼（Version 6：5-byte Packed CSR + 全域 Multi-Team + Absolute Cursor） |
| `DEVELOPMENT_LOG.md` | 本紀錄文件 |
| `BENCHMARK_RESULTS.md` | Server 評測紀錄專用檔 |

---

## 6. 進階優化與架構突破 (Version 5 ~ Version 6)

### 6.1 Version 5B：單次邊生成快取與 Absolute Cursor
1. **60-bit Packed Edge Cache (`uint64_t`)**：在 Pass 1 產生的 $u, v, w$ 打包存進每 rank 私有的 `packed_edges(local_M)`，消除 Pass 2 的整套 Kronecker 邊重算。
2. **Absolute Cursor**：將 `all_degree[r * N + v]` 在 Prefix 階段直接轉為全域絕對 offset，Pass 2 消除兩次隨機 `row[u]` 讀取。
3. **SEIREN 實測**：**87.61s**，Case 09 從 17.60s 降至 14.43s。

### 6.2 Version 6：5-byte 緊湊 CSR 鄰居與全域 Multi-Team (最新重大突破)
1. **5-byte 緊湊 CSR 鄰居結構 (`CSRNeighbor { uint32_t to; uint8_t w; }`)**：
   * 原本 `col` (8 GB) 與 `w` (2 GB) 分離，每次寫入需打到兩個不同的 DRAM 快取行。
   * 打包為 `__attribute__((packed))` 5-byte 結構後，記憶體總量精確維持 10 GB，但隨機快取缺失次數直接減半！
   * **Case 09 實測**：Pass 2 從 **12.03s 暴跌至 7.76s（加速 35.5%）**，總時間降至 **17.81s**！
2. **SSSP 同步開銷量測與 $\Delta$-Stepping 驗證**：
   * 實作了 100% 正確的 $\Delta$-Stepping 原型，比對 golden binary 完全 0 diff。
   * 量測顯示 Case 09 中 528 輪 Dial 的總同步時間（Collectives）僅 0.09s (<0.5%)，證明同步非大圖瓶頸。
3. **全域 Multi-Team 協同平行化 ($S < P$)**：
   * 原本僅 $S \le 4$ 協同，導致 $S=8$（Case 06, 13）時高達 16 個 rank 空轉。
   * 現改為所有 $S < P$ 自動分組，Case 06 啟動 8 個 3-rank 隊伍，SSSP 時間從 6.02s 暴降至 2.29s，總時間從 **9.38s 降至 4.94s（砍半）**！
4. **致命 Bug 修復**：
   * 修復了 `cooperative_dial` 中 `std::fill(dist, INF)` 與 rank 0 `dist[src]=0` 之間缺失 Barrier 的 Data Race 隱患。

### 6.3 Version 6.1：FastGenerator 內聯常數預計算 + 純淨 Dial 調優
1. **內聯常數預計算 FastGenerator**：
   * 消除官方生成器重複計算 80 億次常數 hash 的巨大開銷，Pass 1 速度再次提升。
2. **SSSP 調優為純淨 Dial ($\Delta=1$)**：
   * 確保零重複邊鬆弛，減少大圖數百萬次重複探索。
3. **SEIREN 實測**：**61.43s** $\rightarrow$ **60.31s**（Case 10 達到 24.28s，Case 09 達到 9.13s）。

---

## 7. Milestone 1：重複邊量測器 (Duplicate-Ratio Profiler) 實測與決策分析

### 7.1 三層量測架構與驗證
* **第 1 層 (Rank-Local Uniqueness)**：每 rank 抽樣 500K 條邊原地 sort+unique，量測 rank 內部重複（全域 Uniqueness 之數學上界）。
* **第 2 層 (Global Sampled Uniqueness)**：透過 `MPI_Gatherv` 收集所有 rank 未 deduplicate 的原始 sample keys（共 12M 條），由 rank 0 原地 sort+unique，精準量測跨 rank 重複。
* **第 3 層 (Full-Graph HyperLogLog 估計器)**：Pass 1 後每 rank 原地掃描 100% 邊，利用 SplitMix64 與 HLL ($p=18$, 262,144 暫存器，每 rank 僅 256 KiB 記憶體)，`MPI_Reduce(MPI_MAX)` 合併後以 $\pm 0.20\%$ 理論標準差估計全圖 Cardinality。

### 7.2 基準驗證 (Case 03: Scale 20, Edgefactor 16, Seed 103)
* **全量排序 Ground Truth**：
  * 總無自環邊數：16,776,066
  * 精確唯一邊數：**15,705,624 (93.6192%)**
  * 精確重複邊數：**1,071,592 (6.3808%)**
* **HLL 全圖估計值**：
  * 估計唯一邊數：**15,696,887 (93.5672%)**
  * 誤差分析：僅相差 8,737 條邊，**實測誤差僅 0.055%**（完全符合且遠優於 $\pm 0.20\%$ 理論誤差界限）。
* **跨 Rank 數不變性**：不論在 4 ranks 還是 8 ranks 執行，HLL 暫存器最大值合流結果完全一致，估計值皆精確為 15,696,887。

### 7.3 全測資 Profiler 實測數據對照表

| 測資 | 參數規模 (Scale, EF, Scramble) | 總非自環邊數 | Rank-Local Unique | Global Sampled Unique | HLL 全圖估計唯一率 | HLL 全圖重複邊數 (重複率) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Case 03** | Scale 20, EF 16, Scr 0 | 16,776,066 | 99.54% | 98.55% | **93.57%** | 1,079,179 (6.43%) |
| **Case 07** | Scale 24, EF 16, Scr 0 | 268,432,601 | 99.98% | 99.90% | **96.64%** | 9,015,269 (3.36%) |
| **Case 09** | Scale 25, EF 16, Scr 0 | 536,867,342 | 99.99% | 99.96% | **97.47%** | 13,573,271 (2.53%) |
| **Case 10** | Scale 26, EF 16, Scr 1 | 1,073,737,253 | 99.99% | 99.98% | **97.85%** | 23,073,943 (2.15%) |
| **Case 13** | Scale 20, EF 256, Scr 1 | 268,415,759 | 99.55% | 97.55% | **74.94%** | **67,262,682 (25.06%)** |

### 7.4 關鍵決策與架構結論
1. **大圖 (Case 07, 09, 10，佔總執行時間 65% 以上)**：
   * 全圖唯一率分別高達 **96.64%、97.47%、97.85%**，重複邊僅 2%~3%。
   * 若在大圖進行全圖邊重新分配、Pair-Owner Hash 分流與排序以消除重複邊，通訊與排序開銷將高達數秒，遠遠超過省下 2% CSR 記憶體的效益。
   * **決策：大圖堅決放棄 Global Dedup，直接推進 Vertex-Owner CSR Construction！**
2. **高密度圖 (Case 13, EF 256)**：
   * 重複率高達 **25.06%**（6,726 萬條重複邊）。
   * 但 Case 13 的 Scale 僅為 20（頂點數僅 100 萬），現有版本執行時間僅 3.30s，非全局瓶頸。
3. **驗證了 Rank-Local Uniqueness 盲點**：
   * Case 13 的 Rank-Local Unique 高達 99.55%，但全圖真實 Unique 僅 74.94%，證實若未匯集全圖或使用 HLL，必定會嚴重低估全域重複率。

---

## 8. Milestone 7.2A：Vertex-Owner Direct Scatter A/B 實測與決策分析

### 8.1 測試設定與隔離保證
* **環境**：相同 8-rank 節點環境，交錯（Interleaved）執行以消除系統干擾。
* **純淨 Release 模式**：
  * `ENABLE_DEBUG_INVARIANTS = 0`（關閉所有逐筆 assert）
  * `ENABLE_PROFILE = 0`（抹除每輪 `MPI_Wtime` 與 skew 統計）
  * 僅保留最外層 `t_pass2` 與 `main.cpp` 的 `Compute time`。
* **對照版本**：
  * **Baseline (`PASS2_MODE = 0`)**：8-bit Chunk-Local Radix 緊湊 CSR（官方最佳 60.31s baseline）。
  * **Vertex-Owner (`PASS2_MODE = 1`)**：每輪 256K 邊獨立 Mailbox 切片 + Owner 局部連續讀取與直寫。

### 8.2 實測數據對照表（三次執行取 Median）

| 測資 | 版本 | Run 1 (Pass 2 / Total) | Run 2 (Pass 2 / Total) | Run 3 (Pass 2 / Total) | **Median (Pass 2 / Total)** | 相對表現 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Case 09** (Scale 25) | **Baseline** | 2.11s / 12.76s | 2.50s / 15.21s | 2.49s / 15.31s | **2.49s / 15.21s** | 基準線 |
| | **Vertex-Owner** | 6.17s / 19.49s | 6.61s / 19.94s | 6.03s / 19.03s | **6.17s / 19.49s** | **Pass 2 慢 2.48x (+148%)**<br>Total 慢 1.28x (+28.2%) |
| **Case 10** (Scale 26) | **Baseline** | 6.42s / 39.91s | 6.04s / 39.02s | 6.44s / 38.95s | **6.42s / 39.02s** | 基準線 |
| | **Vertex-Owner** | 16.13s / 49.05s | 16.21s / 48.01s | 16.24s / 48.26s | **16.21s / 48.26s** | **Pass 2 慢 2.52x (+153%)**<br>Total 慢 1.24x (+23.7%) |

### 8.3 深度微架構效能歸因分析
1. **Mailbox Shuffle 與同步基礎開銷高達 2.65 秒**：
   * 即使通訊量極小，兩次解包（First Pass Count + Second Pass Pack）加上跨輪 `MPI_Allgather` 與 2 次 `MPI_Barrier`，在 Case 10（512 輪）累積了約 2.65s 的固有損耗，已佔 Baseline Pass 2 總時間（6.42s）的 41%！
2. **Owner 隨機直寫並未顯著提速（單寫需 11.86 秒）**：
   * Baseline 的 8-bit Radix 是在 private chunk 內將邊分成 256 桶，每次寫入整桶完全留在 L1/L2 快取行與 TLB 覆蓋範圍內。
   * Vertex-Owner 雖然將寫入限制在該 rank 的 $local\_V$ 頂點區間內，但 mailbox 中抵達的頂點仍是**完全隨機分佈**，每筆邊皆伴隨「讀 Mailbox + 讀寫 `local_cursor` + 寫 `adj`」3 次記憶體存取，在巨大頂點空間下依然造成劇烈隨機快取行衝突。

### 8.4 關鍵決策與終止準則觸發
* **決策規則回顧**：只有當 Case 09 與 Case 10 的 Pass 2 median 至少快 25%，且 Total compute 穩定下降時，才推進至 Grouped Write；若整體僅改善 5%～10% 或退步，則終止重構，保留 60.31s baseline。
* **最終決策與結論**：

```text
Vertex-Owner Direct Scatter rejected:
 
Case 09 Pass 2:
2.49s → 6.17s
 
Case 10 Pass 2:
6.42s → 16.21s
 
Root cause:
mailbox shuffle and additional memory traffic outweighed
the reduced owner-local CSR scatter range.
```

* **行動確認**：
  * Vertex-Owner Direct Scatter 的 Pass 2 表現比 Baseline 慢 2.5 倍，Total Compute 退步超過 23%。
  * Grouped Write 在此基礎上還需額外支付 Staging Buffer 分配、Histogram 與搬移複製成本，注定更慢。
  * **正式決策：依照既定準則，正式終止 Vertex-Owner 路線，回歸並鎖定 60.31 秒的最佳正式 Baseline（PASS2_MODE=0）！**

---

## 9. Edge Generator 微架構優化（Milestone G1 ～ G4）

在否決 Vertex-Owner 路線後，依規劃轉向**圖生成器（FastGenerator）**的向量化與微架構優化。此方向具備**零資料通訊、不改動 CSR 與 SSSP 架構、100% 逐邊驗證**等優勢。

### 9.1 Milestone G1：組件耗時微基準分解 (Component Breakdown)
透過獨立隔離測試（`test_gen_breakdown.cpp`，10,000,000 邊）精確量測生成管線各階段耗時：

| 生成管線階段 | Case 10 (Scale 26, Scr 1) 耗時與佔比 | Case 09 (Scale 25, Scr 0) 耗時與佔比 | 分析與優化價值 |
| :--- | :--- | :--- | :--- |
| **1. Initial Mix Hash** | 0.0060s (1.28%) | 0.0059s (1.50%) | 極小，乘加管線已被編譯器優化 |
| **2. Kronecker Descent** | **0.2963s (63.17%)** | **0.2877s (72.93%)** | **絕對最大瓶頸（佔比 63%～73%）**，26 層迴圈迭代為核心開銷 |
| **3. Scramble ID** | 0.0130s (2.77%) | 0.0000s (0.00%) | 佔比極小，3 次乘法與異或位移效率高 |
| **4. Weight Generation** | 0.0367s (7.83%) | 0.0374s (9.48%) | 次要瓶頸，需一次 `mix_hash` 與 `% 255` |
| **5. 60-bit Edge Packing** | 0.0059s (1.25%) | 0.0057s (1.46%) | 純暫存器位移拼裝，極快 |
| **6. Degree Update** | 0.1112s (23.70%) | 0.0577s (14.63%) | 次大開銷，源於全圖隨機記憶體寫入衝突 |

* **結論**：證實優化重心必須鎖定在 **Kronecker Descent**。

### 9.2 Milestone G2：純標量分支消除檢查 (Branchless Inspection)
反組譯 GCC `-O3 -march=native` 彙編碼：
- 編譯器已自動將 `r >= 195` 轉為 `cmp dil, 0xc2; seta al`。
- 複合條件轉為無分支的 `lea eax, [r-146]; cmp eax, 48; setbe al; cmp r, 242; seta r13b; or eax, r13d`。
- **結論**：GCC 已產生物理無分支指令，單 edge 標量手動重構無邊際效益。

### 9.3 Milestone G3：四邊獨立管線批次 (4-Way Independent Scalar Batching)
利用暫存器級指令並行（ILP），一次並行交錯執行 4 條獨立 edge：
- 隱藏 64-bit 乘法與位移的 3~4 cycle 運算延遲。
- **實測產出**：吞吐量由 25.60 M edges/s 提升至 33.19 M edges/s（**1.30x 加速**）。

### 9.4 Milestone G4：AVX2 四邊並行 8-Level 區塊下降 (AVX2 Block Descent)
針對 Kronecker Descent 設計專用 256-bit SIMD 演算法：
1. 一個 `__m256i` 向量同時打包 4 條 edge 的 64-bit 狀態（共 32 個 bytes）。
2. 將 8-bit unsigned 透過 XOR 0x80 映射至 signed 域。
3. 透過 `_mm256_cmpgt_epi8`、`_mm256_and_si256`、`_mm256_or_si256` 進行平行比較。
4. 呼叫單一 `_mm256_movemask_epi8`，在 **1 個 cycle 內直接抽取 4 條邊的 8 層位元遮罩**！
5. 26 層 Descent 僅需 4 個 8-level 區塊，每區塊僅需 7 條向量指令即可完成 4 條邊的判定。

#### 微基準與 Pass 1 實測效能：
- **純生成吞吐量（`test_batch_gen`）**：
  - Case 10：25.60 M/s $\to$ **89.11 M edges/s（3.48x 加速）**。
  - Case 09：28.44 M/s $\to$ **98.55 M edges/s（3.47x 加速）**。
- **單 Rank Pass 1 總耗時（含 Degree 更新與 Edge Packing）**：
  - Case 10：2.4858s $\to$ **1.1678s（2.13x 加速，節省 1.32 秒）**。
  - Case 09：1.0058s $\to$ **0.4310s（2.33x 加速，節省 0.57 秒）**。
- **Chunk 緩衝架構驗證**：
  - 測試了 `G4-Fused`（無緩衝直接寫入 Degree），發現每 4 條邊插入隨機 Degree 寫入會阻塞 CPU 暫存器管線（降速至 0.70x）；證實維持 **512K Chunk 局部緩衝**能讓 SIMD 生成器在 L1/L2 達到極速發揮。

### 9.5 MPI 8-Rank 完整運行與正確性驗證

| 測資 | 模式 | Pass 1 耗時 | Pass 2 耗時 | 圖構建總時間 | 總計算時間 (Compute Time) | 效益 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Case 07** | **Baseline** | 1.6061s | 0.9466s | 2.6174s | 5.6323s | 基準線 |
| (Scale 24) | **G4 (AVX2)** | **0.9871s** | 1.1232s | **2.1789s** | **5.7001s** | **Pass 1 快 1.63x (-0.62s)** |
| **Case 09** | **Baseline** | 3.8077s | 2.4289s | 6.3701s | 13.8073s | 基準線 |
| (Scale 25) | **G4 (AVX2)** | **2.3381s** | 2.3020s | **4.7735s** | **12.4441s** | **Pass 1 快 1.63x (-1.47s)**<br>**Total 節省 1.36s** |
| **Case 10** | **Baseline** | 11.4947s | 5.7136s | 17.5337s | 36.8334s | 基準線 |
| (Scale 26) | **G4 (AVX2)** | **6.7284s** | 5.7386s | **12.7915s** | **32.3679s** | **Pass 1 快 1.71x (-4.77s)**<br>**Total 節省 4.47s (-12.1%)** |

#### 正確性檢驗：
- 跨 Scales 1～26、Scramble 0/1、多種 Seed 進行完全比對，所有 $(u, v, w)$ 100% 精確相符。
- 針對全套官方測資執行完整驗證：**Case 01 至 Case 17 全部 100% Byte-for-Byte MATCH 官方 Golden Binary（17/17 AC）**！

### 9.6 SEIREN 官方 24-Rank Judge 最新里程碑成績 (56.40 秒)
- **評測結果**：**17 / 17 AC，總時間 56.40 秒**（前版 Baseline 為 60.31 秒，大幅推進 **-3.91 秒**）。
- **關鍵亮點**：
  - **Case 10 (Scale 26, Scr 1)**：24.31s $\to$ **21.90s（暴降 2.41 秒！）**
  - **Case 07 (Scale 24, Scr 0)**：4.96s $\to$ **4.27s（降低 0.69 秒）**
  - **Case 09 (Scale 25, Scr 0)**：9.13s $\to$ **9.26s（微幅波動 1.4%）**
- **歸檔狀態**：
  - 凍結為正式基準：`hw2-1-g4-ac-56.40.cpp`、`hw2-1-g4-ac-56.40`
  - Git Tag 鎖定：`hw2-1-g4-ac-56.40`



