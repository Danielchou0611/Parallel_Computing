# HW1-3 Optimization Summary & Progression Log

## Current Status (2026-10-01)
- **Official SEIREN Leaderboard Best**: **20.91s** (10/10 AC)
- **Local Testbench Total Time**: **13.579s** (10/10 AC)
- **Git Branch**: `hw1` (Head commit: `f3f2bd0`)
- **Numerical Accuracy**: 100% verified (energy error $\le 10^{-10}$, sample error $0.00$).

---

## Benchmark Progression on SEIREN Cluster

| Test Case | N / T | Cores | Original Best | Run (Dynamic J) | **Current Best (Run #8)** | Status | Notes |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **p01** | 64 / 100 | 2 | 0.17s | 0.15s | **0.15s ~ 0.17s** | AC | Small, low impact |
| **p02** | 128 / 60 | default | 0.09s | 0.13s | **0.10s** | AC | B=0, pure diffusion |
| **p03** | 192 / 50 | default | 1.02s | 1.02s | **1.02s ~ 1.10s** | AC | Spheres present |
| **p04** | 256 / 40 | 2 | 1.53s | 1.21s | **1.19s** | AC | **B=0, dropped from 1.53s to 1.19s!** |
| **p05** | 384 / 22 | default | 2.60s | 2.63s | **2.63s** | AC | theta=0.6 |
| **p06** | 512 / 8 | 4 | 1.65s | 1.76s | **1.69s** | AC | B=0, pure diffusion |
| **p07** | 512 / 13 | default | 3.67s | 3.66s | **3.65s** | AC | Spheres present |
| **p08** | 640 / 6 | default | 3.65s | 3.62s | **3.62s ~ 3.67s** | AC | Spheres present |
| **p09** | 700 / 8 | default | 1.89s | 1.99s | **1.98s** | AC | B=0, theta=0.757 |
| **p10** | 768 / 6 | default | 4.71s | 4.77s | **4.72s** | AC | Largest case |
| **TOTAL** | — | — | **20.99s** | **20.94s** | **20.91s** | **10/10 AC** | **New Personal Best on Leaderboard** |

---

## Key Architectural Insights for Future Sessions

1. **Why `p04` dropped from 1.53s -> 1.19s**:
   - $B=0$ with $BJ=32$ spatial cache-tiling keeps 32 rows ($\approx 65\text{ KB}$) warm inside L2 cache ($1\text{ MB}$), avoiding repeated DRAM fetches as slice $i$ advances.
2. **NUMA vs Dynamic Scheduling**:
   - On SEIREN (dual-socket Xeon), `schedule(dynamic, 1)` caused chunks to migrate across sockets between timesteps, incurring cross-socket UPI traffic. Static scheduling (`schedule(static)`) eliminated this migration.
3. **Initialization & Memory Footprint Overhead**:
   - For $N=768$, the three arrays ($u$, $unew$, $a$) allocate $>11\text{ GB}$.
   - Initialization alone requires $>900\text{M}$ calls to `field()`.
   - In subsequent sessions, optimizing initialization (e.g. eliminating zero-initialization of $unew$, parallel page fault pre-touching) can yield substantial additional seconds off the total run time.
