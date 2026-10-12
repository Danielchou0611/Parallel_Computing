#include <mpi.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdint>
#include <algorithm>
#include <iomanip>
#include <cassert>
#include <cstring>
#include <fstream>
#include <immintrin.h>
#include "graph_gen.hpp"

inline uint64_t mix_hash(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

struct FastGenG5 {
    uint64_t mixed_seed;
    uint64_t mixed_seed_w;
    uint64_t scramble_k1, scramble_k2, scramble_k3;
    uint64_t scramble_mask;
    int scramble_sh;
    int scale;
    bool scramble;

    void init(int scale_in, bool scramble_in, uint64_t seed) {
        scale = scale_in;
        scramble = scramble_in;
        mixed_seed = mix_hash(seed);
        mixed_seed_w = mix_hash(seed ^ 0x77e19470ULL);

        uint64_t s_seed = seed ^ 0x5c2a3b1eULL;
        scramble_mask = (uint64_t(1) << scale) - 1;
        scramble_sh = scale / 2 > 0 ? scale / 2 : 1;
        scramble_k1 = mix_hash(s_seed ^ 0x1111ULL) | 1;
        scramble_k2 = mix_hash(s_seed ^ 0x2222ULL) | 1;
        scramble_k3 = mix_hash(s_seed ^ 0x3333ULL) | 1;
    }

    inline uint32_t scramble_id(uint32_t v) const {
        uint64_t x = v;
        x = (x * scramble_k1) & scramble_mask;
        x ^= x >> scramble_sh;
        x = (x * scramble_k2) & scramble_mask;
        x ^= x >> scramble_sh;
        x = (x * scramble_k3) & scramble_mask;
        return (uint32_t)x;
    }

    // Scalar fallback
    inline void generate1(uint64_t i, uint32_t &u, uint32_t &v, uint8_t &w) const {
        uint64_t state = mix_hash(mixed_seed ^ (i * 0xd6e8feb86659fd93ULL));
        uint64_t bits = state;
        uint32_t a = 0, b = 0;

        for (int level = 0; level < scale; level++) {
            if (level && (level & 7) == 0) {
                state = mix_hash(state);
                bits = state;
            }
            unsigned r = (unsigned)(bits & 0xff);
            bits >>= 8;
            uint32_t bit_a = (r >= 195);
            uint32_t bit_b = ((r >= 146 && r < 195) || r >= 243);
            a |= bit_a << level;
            b |= bit_b << level;
        }

        if (scramble) {
            a = scramble_id(a);
            b = scramble_id(b);
        }
        u = a;
        v = b;

        uint64_t lo = std::min(a, b), hi = std::max(a, b);
        uint64_t pair_key = (hi << 32) | lo;
        uint64_t h = mix_hash(mixed_seed_w ^ (pair_key * 0xd6e8feb86659fd93ULL));
        w = static_cast<uint8_t>(1 + (h % 255));
    }

    // G4: 4-way AVX2
    inline void generate4_avx2(uint64_t i, uint32_t *u, uint32_t *v, uint8_t *w) const {
        uint64_t s0 = mix_hash(mixed_seed ^ ((i + 0) * 0xd6e8feb86659fd93ULL));
        uint64_t s1 = mix_hash(mixed_seed ^ ((i + 1) * 0xd6e8feb86659fd93ULL));
        uint64_t s2 = mix_hash(mixed_seed ^ ((i + 2) * 0xd6e8feb86659fd93ULL));
        uint64_t s3 = mix_hash(mixed_seed ^ ((i + 3) * 0xd6e8feb86659fd93ULL));

        uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        uint32_t b0 = 0, b1 = 0, b2 = 0, b3 = 0;

        const __m256i sign_flip = _mm256_set1_epi8((char)0x80);
        const __m256i thr_a     = _mm256_set1_epi8((char)(195 ^ 0x80) - 1);
        const __m256i thr_b_lo  = _mm256_set1_epi8((char)(146 ^ 0x80) - 1);
        const __m256i thr_b_hi  = _mm256_set1_epi8((char)(195 ^ 0x80));
        const __m256i thr_b_top = _mm256_set1_epi8((char)(243 ^ 0x80) - 1);

        int rem_levels = scale;
        int shift = 0;

        while (rem_levels > 0) {
            if (shift > 0) {
                s0 = mix_hash(s0);
                s1 = mix_hash(s1);
                s2 = mix_hash(s2);
                s3 = mix_hash(s3);
            }
            __m256i vec = _mm256_set_epi64x(s3, s2, s1, s0);
            __m256i vec_s = _mm256_xor_si256(vec, sign_flip);

            __m256i cmp_a = _mm256_cmpgt_epi8(vec_s, thr_a);
            uint32_t mask_a = (uint32_t)_mm256_movemask_epi8(cmp_a);

            __m256i ge_lo = _mm256_cmpgt_epi8(vec_s, thr_b_lo);
            __m256i lt_hi = _mm256_cmpgt_epi8(thr_b_hi, vec_s);
            __m256i mid   = _mm256_and_si256(ge_lo, lt_hi);
            __m256i top   = _mm256_cmpgt_epi8(vec_s, thr_b_top);
            __m256i cmp_b = _mm256_or_si256(mid, top);
            uint32_t mask_b = (uint32_t)_mm256_movemask_epi8(cmp_b);

            int chunk = std::min(rem_levels, 8);
            uint32_t chunk_mask = (1u << chunk) - 1u;

            a0 |= ((mask_a >> 0)  & chunk_mask) << shift;
            a1 |= ((mask_a >> 8)  & chunk_mask) << shift;
            a2 |= ((mask_a >> 16) & chunk_mask) << shift;
            a3 |= ((mask_a >> 24) & chunk_mask) << shift;

            b0 |= ((mask_b >> 0)  & chunk_mask) << shift;
            b1 |= ((mask_b >> 8)  & chunk_mask) << shift;
            b2 |= ((mask_b >> 16) & chunk_mask) << shift;
            b3 |= ((mask_b >> 24) & chunk_mask) << shift;

            shift += 8;
            rem_levels -= 8;
        }

        if (scramble) {
            a0 = scramble_id(a0); b0 = scramble_id(b0);
            a1 = scramble_id(a1); b1 = scramble_id(b1);
            a2 = scramble_id(a2); b2 = scramble_id(b2);
            a3 = scramble_id(a3); b3 = scramble_id(b3);
        }
        u[0] = a0; v[0] = b0;
        u[1] = a1; v[1] = b1;
        u[2] = a2; v[2] = b2;
        u[3] = a3; v[3] = b3;

        uint64_lo_calc(a0, b0, u, v, w, 0);
        uint64_lo_calc(a1, b1, u, v, w, 1);
        uint64_lo_calc(a2, b2, u, v, w, 2);
        uint64_lo_calc(a3, b3, u, v, w, 3);
    }

    inline void uint64_lo_calc(uint32_t a, uint32_t b, uint32_t *u, uint32_t *v, uint8_t *w, int idx) const {
        uint64_t lo = std::min(a, b), hi = std::max(a, b);
        uint64_t h = mix_hash(mixed_seed_w ^ (((hi << 32) | lo) * 0xd6e8feb86659fd93ULL));
        w[idx] = static_cast<uint8_t>(1 + (h % 255));
    }

    // G5: 8-way AVX2
    inline void generate8_avx2(uint64_t i, uint32_t *u, uint32_t *v, uint8_t *w) const {
        uint64_t s0 = mix_hash(mixed_seed ^ ((i + 0) * 0xd6e8feb86659fd93ULL));
        uint64_t s4 = mix_hash(mixed_seed ^ ((i + 4) * 0xd6e8feb86659fd93ULL));
        uint64_t s1 = mix_hash(mixed_seed ^ ((i + 1) * 0xd6e8feb86659fd93ULL));
        uint64_t s5 = mix_hash(mixed_seed ^ ((i + 5) * 0xd6e8feb86659fd93ULL));
        uint64_t s2 = mix_hash(mixed_seed ^ ((i + 2) * 0xd6e8feb86659fd93ULL));
        uint64_t s6 = mix_hash(mixed_seed ^ ((i + 6) * 0xd6e8feb86659fd93ULL));
        uint64_t s3 = mix_hash(mixed_seed ^ ((i + 3) * 0xd6e8feb86659fd93ULL));
        uint64_t s7 = mix_hash(mixed_seed ^ ((i + 7) * 0xd6e8feb86659fd93ULL));

        uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
        uint32_t b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0, b7 = 0;

        const __m256i sign_flip = _mm256_set1_epi8((char)0x80);
        const __m256i thr_a     = _mm256_set1_epi8((char)(195 ^ 0x80) - 1);
        const __m256i thr_b_lo  = _mm256_set1_epi8((char)(146 ^ 0x80) - 1);
        const __m256i thr_b_hi  = _mm256_set1_epi8((char)(195 ^ 0x80));
        const __m256i thr_b_top = _mm256_set1_epi8((char)(243 ^ 0x80) - 1);

        int rem_levels = scale;
        int shift = 0;

        while (rem_levels > 0) {
            if (shift > 0) {
                s0 = mix_hash(s0); s4 = mix_hash(s4);
                s1 = mix_hash(s1); s5 = mix_hash(s5);
                s2 = mix_hash(s2); s6 = mix_hash(s6);
                s3 = mix_hash(s3); s7 = mix_hash(s7);
            }
            __m256i vec_a = _mm256_set_epi64x(s3, s2, s1, s0);
            __m256i vec_b = _mm256_set_epi64x(s7, s6, s5, s4);

            __m256i vec_s_a = _mm256_xor_si256(vec_a, sign_flip);
            __m256i vec_s_b = _mm256_xor_si256(vec_b, sign_flip);

            // bit_a
            __m256i cmp_a_a = _mm256_cmpgt_epi8(vec_s_a, thr_a);
            __m256i cmp_a_b = _mm256_cmpgt_epi8(vec_s_b, thr_a);
            uint32_t mask_a_a = (uint32_t)_mm256_movemask_epi8(cmp_a_a);
            uint32_t mask_a_b = (uint32_t)_mm256_movemask_epi8(cmp_a_b);

            // bit_b
            __m256i ge_lo_a = _mm256_cmpgt_epi8(vec_s_a, thr_b_lo);
            __m256i ge_lo_b = _mm256_cmpgt_epi8(vec_s_b, thr_b_lo);
            __m256i lt_hi_a = _mm256_cmpgt_epi8(thr_b_hi, vec_s_a);
            __m256i lt_hi_b = _mm256_cmpgt_epi8(thr_b_hi, vec_s_b);
            __m256i mid_a   = _mm256_and_si256(ge_lo_a, lt_hi_a);
            __m256i mid_b   = _mm256_and_si256(ge_lo_b, lt_hi_b);
            __m256i top_a   = _mm256_cmpgt_epi8(vec_s_a, thr_b_top);
            __m256i top_b   = _mm256_cmpgt_epi8(vec_s_b, thr_b_top);
            __m256i cmp_b_a = _mm256_or_si256(mid_a, top_a);
            __m256i cmp_b_b = _mm256_or_si256(mid_b, top_b);
            uint32_t mask_b_a = (uint32_t)_mm256_movemask_epi8(cmp_b_a);
            uint32_t mask_b_b = (uint32_t)_mm256_movemask_epi8(cmp_b_b);

            int chunk = std::min(rem_levels, 8);
            uint32_t chunk_mask = (1u << chunk) - 1u;

            a0 |= ((mask_a_a >> 0)  & chunk_mask) << shift;
            a1 |= ((mask_a_a >> 8)  & chunk_mask) << shift;
            a2 |= ((mask_a_a >> 16) & chunk_mask) << shift;
            a3 |= ((mask_a_a >> 24) & chunk_mask) << shift;

            a4 |= ((mask_a_b >> 0)  & chunk_mask) << shift;
            a5 |= ((mask_a_b >> 8)  & chunk_mask) << shift;
            a6 |= ((mask_a_b >> 16) & chunk_mask) << shift;
            a7 |= ((mask_a_b >> 24) & chunk_mask) << shift;

            b0 |= ((mask_b_a >> 0)  & chunk_mask) << shift;
            b1 |= ((mask_b_a >> 8)  & chunk_mask) << shift;
            b2 |= ((mask_b_a >> 16) & chunk_mask) << shift;
            b3 |= ((mask_b_a >> 24) & chunk_mask) << shift;

            b4 |= ((mask_b_b >> 0)  & chunk_mask) << shift;
            b5 |= ((mask_b_b >> 8)  & chunk_mask) << shift;
            b6 |= ((mask_b_b >> 16) & chunk_mask) << shift;
            b7 |= ((mask_b_b >> 24) & chunk_mask) << shift;

            shift += 8;
            rem_levels -= 8;
        }

        if (scramble) {
            a0 = scramble_id(a0); b0 = scramble_id(b0);
            a4 = scramble_id(a4); b4 = scramble_id(b4);
            a1 = scramble_id(a1); b1 = scramble_id(b1);
            a5 = scramble_id(a5); b5 = scramble_id(b5);
            a2 = scramble_id(a2); b2 = scramble_id(b2);
            a6 = scramble_id(a6); b6 = scramble_id(b6);
            a3 = scramble_id(a3); b3 = scramble_id(b3);
            a7 = scramble_id(a7); b7 = scramble_id(b7);
        }

        u[0] = a0; v[0] = b0;
        u[1] = a1; v[1] = b1;
        u[2] = a2; v[2] = b2;
        u[3] = a3; v[3] = b3;
        u[4] = a4; v[4] = b4;
        u[5] = a5; v[5] = b5;
        u[6] = a6; v[6] = b6;
        u[7] = a7; v[7] = b7;

        uint64_lo_calc(a0, b0, u, v, w, 0);
        uint64_lo_calc(a1, b1, u, v, w, 1);
        uint64_lo_calc(a2, b2, u, v, w, 2);
        uint64_lo_calc(a3, b3, u, v, w, 3);
        uint64_lo_calc(a4, b4, u, v, w, 4);
        uint64_lo_calc(a5, b5, u, v, w, 5);
        uint64_lo_calc(a6, b6, u, v, w, 6);
        uint64_lo_calc(a7, b7, u, v, w, 7);
    }

    inline void generate_range_g4(uint64_t begin, uint64_t end, uint32_t *u, uint32_t *v, uint8_t *w) const {
        uint64_t i = begin;
        uint64_t idx = 0;
        for (; i + 4 <= end; i += 4, idx += 4) {
            generate4_avx2(i, &u[idx], &v[idx], &w[idx]);
        }
        for (; i < end; ++i, ++idx) {
            generate1(i, u[idx], v[idx], w[idx]);
        }
    }

    inline void generate_range_g5(uint64_t begin, uint64_t end, uint32_t *u, uint32_t *v, uint8_t *w) const {
        uint64_t i = begin;
        uint64_t idx = 0;
        for (; i + 8 <= end; i += 8, idx += 8) {
            generate8_avx2(i, &u[idx], &v[idx], &w[idx]);
        }
        for (; i + 4 <= end; i += 4, idx += 4) {
            generate4_avx2(i, &u[idx], &v[idx], &w[idx]);
        }
        for (; i < end; ++i, ++idx) {
            generate1(i, u[idx], v[idx], w[idx]);
        }
    }
};

static uint64_t get_vm_hwm_kb() {
    std::ifstream st("/proc/self/status");
    std::string line;
    while (std::getline(st, line)) {
        if (line.rfind("VmHWM:", 0) == 0) {
            uint64_t val = 0;
            std::sscanf(line.c_str(), "VmHWM: %llu kB", (unsigned long long*)&val);
            return val;
        }
    }
    return 0;
}

void run_test_case(int scale, int scramble, uint64_t seed, const std::string &case_label, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    GraphParams p;
    p.scale = scale;
    p.edgefactor = 16;
    p.seed = seed;
    p.scramble = scramble;
    p.nsample = 1000;

    const uint64_t N = num_vertices(p);
    const uint64_t M = num_edges(p);
    const uint64_t edge_begin = (M * static_cast<uint64_t>(rank)) / size;
    const uint64_t edge_end   = (M * static_cast<uint64_t>(rank + 1)) / size;
    const uint64_t local_M    = edge_end - edge_begin;
    const uint64_t CHUNK      = 524288;

    FastGenG5 gctx;
    gctx.init(scale, scramble, seed);

    // Shared memory allocation for all_degree (identical to hw2-1.cpp)
    MPI_Comm shm_comm;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shm_comm);
    int shm_rank, shm_size;
    MPI_Comm_rank(shm_comm, &shm_rank);
    MPI_Comm_size(shm_comm, &shm_size);

    MPI_Win win;
    uint32_t *all_degree = nullptr;
    size_t all_degree_bytes = static_cast<size_t>(shm_size) * N * sizeof(uint32_t);
    MPI_Win_allocate_shared(shm_rank == 0 ? all_degree_bytes : 0, sizeof(uint32_t),
                            MPI_INFO_NULL, shm_comm, &all_degree, &win);

    if (shm_rank != 0) {
        int disp_unit;
        MPI_Aint sz;
        MPI_Win_shared_query(win, 0, &sz, &disp_unit, &all_degree);
    }
    uint32_t *my_degree = all_degree + static_cast<uint64_t>(shm_rank) * N;

    // Local buffers
    std::vector<uint32_t> u(CHUNK), v(CHUNK);
    std::vector<uint8_t>  w_chunk(CHUNK);
    std::vector<uint64_t> packed_edges(local_M);

    // Verification reference buffers
    std::vector<uint64_t> ref_packed;
    std::vector<uint32_t> ref_degree;

    if (rank == 0) {
        std::cout << "=================================================================\n";
        std::cout << "  [MILESTONE G5.3] FULL PASS 1 A/B: " << case_label << "\n";
        std::cout << "  Scale=" << scale << ", Scramble=" << scramble << ", N=" << N 
                  << ", local_M=" << local_M << " edges per rank\n";
        std::cout << "=================================================================\n";
    }

    auto execute_pass1 = [&](int mode) -> double {
        std::memset(my_degree, 0, N * sizeof(uint32_t));
        MPI_Win_sync(win);
        MPI_Barrier(comm);
        MPI_Win_sync(win);

        double t0 = MPI_Wtime();

        for (uint64_t i = edge_begin; i < edge_end; i += CHUNK) {
            uint64_t chunk_len = std::min(CHUNK, edge_end - i);
            if (mode == 4) {
                gctx.generate_range_g4(i, i + chunk_len, u.data(), v.data(), w_chunk.data());
            } else {
                gctx.generate_range_g5(i, i + chunk_len, u.data(), v.data(), w_chunk.data());
            }

            uint64_t base_idx = i - edge_begin;
            for (uint64_t k = 0; k < chunk_len; k++) {
                uint32_t uk = u[k], vk = v[k];
                uint8_t wk  = w_chunk[k];

                my_degree[uk]++;
                my_degree[vk]++;

                packed_edges[base_idx + k] = (static_cast<uint64_t>(uk) << 34) |
                                             (static_cast<uint64_t>(vk) << 8) |
                                             static_cast<uint64_t>(wk);
            }
        }

        MPI_Win_sync(win);
        MPI_Barrier(comm);
        MPI_Win_sync(win);

        double local_time = MPI_Wtime() - t0;
        double max_time = 0;
        MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        return max_time;
    };

    // 1. Warm-up
    execute_pass1(4);
    execute_pass1(5);

    // 2. Correctness check
    execute_pass1(4);
    ref_packed = packed_edges;
    ref_degree.assign(my_degree, my_degree + N);

    execute_pass1(5);
    bool packed_ok = (packed_edges == ref_packed);
    bool degree_ok = (std::memcmp(my_degree, ref_degree.data(), N * sizeof(uint32_t)) == 0);

    uint64_t deg_sum = 0;
    for (uint64_t vert = 0; vert < N; ++vert) deg_sum += my_degree[vert];
    bool sum_ok = (deg_sum == 2 * local_M);

    int local_ok = (packed_ok && degree_ok && sum_ok) ? 1 : 0;
    int global_ok = 0;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, comm);

    if (rank == 0) {
        if (!global_ok) {
            std::cerr << "FATAL: Correctness check failed in Pass 1 for " << case_label << "!\n";
            MPI_Abort(comm, 1);
        } else {
            std::cout << ">> Correctness check: 100% MATCH (packed_edges, my_degree, deg_sum=2*M)!\n";
        }
    }

    // 3. Interleaved 5-Run Benchmark: G4 -> G5 -> G5 -> G4 -> G4 -> G5 -> G5 -> G4 -> G4 -> G5
    const int RUNS = 5;
    std::vector<double> g4_times, g5_times;

    // Pattern: 5 iterations of (G4, G5) or interleaved sequence
    std::vector<int> schedule = {4, 5, 5, 4, 4, 5, 5, 4, 4, 5};
    for (int mode : schedule) {
        double elapsed = execute_pass1(mode);
        if (mode == 4) g4_times.push_back(elapsed);
        else g5_times.push_back(elapsed);
    }

    // Capacity & memory reporting
    uint64_t vm_hwm = get_vm_hwm_kb();
    uint64_t max_hwm = 0;
    MPI_Reduce(&vm_hwm, &max_hwm, 1, MPI_UINT64_T, MPI_MAX, 0, comm);

    if (rank == 0) {
        std::vector<double> s_g4 = g4_times;
        std::vector<double> s_g5 = g5_times;
        std::sort(s_g4.begin(), s_g4.end());
        std::sort(s_g5.begin(), s_g5.end());

        double g4_med = s_g4[RUNS / 2];
        double g5_med = s_g5[RUNS / 2];
        double pct = (g4_med - g5_med) / g4_med * 100.0;

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  G4 Pass 1 (5 runs): ";
        for (double t : g4_times) std::cout << t << "s ";
        std::cout << "\n    Min: " << s_g4.front() << "s | Med: " << g4_med << "s | Max: " << s_g4.back() 
                  << "s | Range: " << (s_g4.back() - s_g4.front()) << "s\n";

        std::cout << "  G5 Pass 1 (5 runs): ";
        for (double t : g5_times) std::cout << t << "s ";
        std::cout << "\n    Min: " << s_g5.front() << "s | Med: " << g5_med << "s | Max: " << s_g5.back() 
                  << "s | Range: " << (s_g5.back() - s_g5.front()) << "s\n";

        std::cout << "  ---------------------------------------------------------------\n";
        std::cout << "  G5 Median Improvement: " << (pct >= 0 ? "+" : "") << pct << "% ("
                  << (g4_med - g5_med) << "s faster)\n";

        double packed_mb = (packed_edges.size() * sizeof(uint64_t)) / (1024.0 * 1024.0);
        double chunk_mb  = (CHUNK * (4 + 4 + 1)) / (1024.0 * 1024.0);
        double shm_deg_mb = (N * sizeof(uint32_t)) / (1024.0 * 1024.0);
        std::cout << "  Memory Footprint per rank:\n";
        std::cout << "    - packed_edges: " << packed_mb << " MB\n";
        std::cout << "    - chunk buffers: " << chunk_mb << " MB\n";
        std::cout << "    - my_degree (shared slice): " << shm_deg_mb << " MB\n";
        std::cout << "    - Peak Private Memory (VmHWM): " << (max_hwm / 1024.0) << " MB (limit 1200 MB)\n";

        if (pct >= 8.0) {
            std::cout << "  => [STATUS]: PASSED (>= 8.0% threshold)\n";
        } else if (pct >= 5.0) {
            std::cout << "  => [STATUS]: BORDERLINE (5.0% ~ 8.0%)\n";
        } else {
            std::cout << "  => [STATUS]: BELOW 5.0% THRESHOLD\n";
        }
        std::cout << "=================================================================\n\n";
    }

    MPI_Win_free(&win);
    MPI_Comm_free(&shm_comm);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    run_test_case(24, 0, 107, "Case 07 (Scale 24, Scr 0)", MPI_COMM_WORLD);
    run_test_case(25, 0, 109, "Case 09 (Scale 25, Scr 0)", MPI_COMM_WORLD);
    run_test_case(26, 1, 110, "Case 10 (Scale 26, Scr 1)", MPI_COMM_WORLD);

    MPI_Finalize();
    return 0;
}
