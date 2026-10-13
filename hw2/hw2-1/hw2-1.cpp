// HW 2-1: MPI shortest paths (Version 5B + Detailed SSSP & Collective Diagnostics + Configurable Edge Cache)
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <utility>
#include <vector>
#include <immintrin.h>
#include <numeric>

#include "graph_gen.hpp"

#ifndef CHUNK_SIZE
#define CHUNK_SIZE 524288 // 512K edges per chunk
#endif

#ifndef CACHE_EDGES
#define CACHE_EDGES 1 // 1: packed_edges cache, 0: on-the-fly recomputation
#endif

#ifndef DELTA_STEPPING
#define DELTA_STEPPING 1 // 1: Dial, >1: Delta-Stepping with bucket width DELTA_STEPPING
#endif

#ifndef CHUNK_MODE
#define CHUNK_MODE 0 // 0: Legacy (16..2048), 1: Fixed 64, 2: Team-aware Adaptive (16/32/64)
#endif

#ifndef SOURCE_TEAM_MODE
#define SOURCE_TEAM_MODE 3 // 0: max teams (up to S), 1: 2 teams, 2: 1 team, 3: hybrid
#endif

#ifndef PRINT_TEAM_SELECTION
#define PRINT_TEAM_SELECTION 0
#endif


#ifndef ENABLE_PROFILE
#define ENABLE_PROFILE 0 // 0: Release (no reductions/printfs), 1: Debug profiling
#endif

#ifndef ENABLE_DEDUP_PROFILE
#define ENABLE_DEDUP_PROFILE 0 // 0: Release (zero overhead), 1: Full duplicate profiler
#endif

#ifndef ENABLE_BOUNDARY_PROFILE
#define ENABLE_BOUNDARY_PROFILE 0 // 0: Release (zero overhead), 1: Boundary & owner profiler
#endif

#ifndef PASS2_MODE
#define PASS2_MODE 0 // 0: 8-bit radix baseline, 1: Vertex-Owner direct scatter
#endif

#ifndef ENABLE_DEBUG_INVARIANTS
#define ENABLE_DEBUG_INVARIANTS 0 // 0: Release, 1: Verify cursor and mailbox invariants
#endif

#ifndef PRINT_PASS2
#define PRINT_PASS2 0
#endif

#ifndef GEN_MODE
#define GEN_MODE 2 // 0: baseline 1-way scalar, 1: 4-way scalar batch (G3), 2: 4-way AVX2 block descent (G4), 3: 8-way AVX2 dual-track (G5)
#endif

namespace {

static inline uint64_t choose_work_chunk(
    uint64_t frontier_size,
    int team_size)
{
#if CHUNK_MODE == 1

    // Fixed 64
    return 64;

#elif CHUNK_MODE == 2

    // Team-aware conservative adaptive strategy.
    // Small teams need finer granularity because one high-degree hub
    // can dominate a large vertex chunk.
    if (team_size <= 3) {
        if (frontier_size <= 32768)
            return 16;

        return 32;
    }

    // Larger teams use 16 / 32 / 64.
    // Never exceed 64.
    if (frontier_size <= 2048)
        return 16;

    if (frontier_size <= 32768)
        return 32;

    return 64;

#else

    // Legacy strategy used by the official 56.40-second baseline.
    if (frontier_size <= 4096)
        return 16;

    uint64_t chunk =
        frontier_size /
        (static_cast<uint64_t>(team_size) * 128);

    if (chunk < 32)
        chunk = 32;

    if (chunk > 2048)
        chunk = 2048;

    return chunk;

#endif
}

const uint32_t INF = 0xffffffffu;

struct FlatResultHeader {
    int64_t n_reached;
    uint64_t checksum;
    uint64_t digest1;
    uint64_t digest2;
};

inline uint64_t align64(uint64_t bytes) {
    return (bytes + 63) & ~uint64_t(63);
}

inline bool atomic_min_u32(uint32_t *ptr, uint32_t value) {
    uint32_t old = __atomic_load_n(ptr, __ATOMIC_RELAXED);
    while (value < old) {
        if (__atomic_compare_exchange_n(ptr, &old, value, false,
                                        __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
            return true;
        }
    }
    return false;
}

inline uint64_t mix_hash(uint64_t x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

struct FastGenerator {
    uint64_t mixed_seed;
    uint64_t mixed_seed_w;
    uint64_t scramble_k1;
    uint64_t scramble_k2;
    uint64_t scramble_k3;
    uint64_t scramble_mask;
    int scramble_sh;
    int scale;
    bool scramble;

    void init(const GraphParams &p) {
        scale = p.scale;
        scramble = p.scramble;
        mixed_seed = mix_hash(p.seed);
        mixed_seed_w = mix_hash(p.seed ^ 0x77e19470ULL);

        uint64_t s_seed = p.seed ^ 0x5c2a3b1eULL;
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

    inline void generate(uint64_t i, uint32_t &u, uint32_t &v, uint8_t &w) const {
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

    inline void generate4_scalar(uint64_t i, uint32_t *u, uint32_t *v, uint8_t *w) const {
        uint64_t s0 = mix_hash(mixed_seed ^ ((i + 0) * 0xd6e8feb86659fd93ULL));
        uint64_t s1 = mix_hash(mixed_seed ^ ((i + 1) * 0xd6e8feb86659fd93ULL));
        uint64_t s2 = mix_hash(mixed_seed ^ ((i + 2) * 0xd6e8feb86659fd93ULL));
        uint64_t s3 = mix_hash(mixed_seed ^ ((i + 3) * 0xd6e8feb86659fd93ULL));

        uint64_t bits0 = s0, bits1 = s1, bits2 = s2, bits3 = s3;
        uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        uint32_t b0 = 0, b1 = 0, b2 = 0, b3 = 0;

        for (int level = 0; level < scale; level++) {
            if (level && (level & 7) == 0) {
                s0 = mix_hash(s0); bits0 = s0;
                s1 = mix_hash(s1); bits1 = s1;
                s2 = mix_hash(s2); bits2 = s2;
                s3 = mix_hash(s3); bits3 = s3;
            }
            unsigned r0 = (unsigned)(bits0 & 0xff); bits0 >>= 8;
            unsigned r1 = (unsigned)(bits1 & 0xff); bits1 >>= 8;
            unsigned r2 = (unsigned)(bits2 & 0xff); bits2 >>= 8;
            unsigned r3 = (unsigned)(bits3 & 0xff); bits3 >>= 8;

            uint32_t bit_a0 = (r0 >= 195);
            uint32_t bit_b0 = ((r0 >= 146 && r0 < 195) || r0 >= 243);
            a0 |= bit_a0 << level;
            b0 |= bit_b0 << level;

            uint32_t bit_a1 = (r1 >= 195);
            uint32_t bit_b1 = ((r1 >= 146 && r1 < 195) || r1 >= 243);
            a1 |= bit_a1 << level;
            b1 |= bit_b1 << level;

            uint32_t bit_a2 = (r2 >= 195);
            uint32_t bit_b2 = ((r2 >= 146 && r2 < 195) || r2 >= 243);
            a2 |= bit_a2 << level;
            b2 |= bit_b2 << level;

            uint32_t bit_a3 = (r3 >= 195);
            uint32_t bit_b3 = ((r3 >= 146 && r3 < 195) || r3 >= 243);
            a3 |= bit_a3 << level;
            b3 |= bit_b3 << level;
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

        uint64_t lo0 = std::min(a0, b0), hi0 = std::max(a0, b0);
        uint64_t lo1 = std::min(a1, b1), hi1 = std::max(a1, b1);
        uint64_t lo2 = std::min(a2, b2), hi2 = std::max(a2, b2);
        uint64_t lo3 = std::min(a3, b3), hi3 = std::max(a3, b3);

        uint64_t h0 = mix_hash(mixed_seed_w ^ (((hi0 << 32) | lo0) * 0xd6e8feb86659fd93ULL));
        uint64_t h1 = mix_hash(mixed_seed_w ^ (((hi1 << 32) | lo1) * 0xd6e8feb86659fd93ULL));
        uint64_t h2 = mix_hash(mixed_seed_w ^ (((hi2 << 32) | lo2) * 0xd6e8feb86659fd93ULL));
        uint64_t h3 = mix_hash(mixed_seed_w ^ (((hi3 << 32) | lo3) * 0xd6e8feb86659fd93ULL));

        w[0] = static_cast<uint8_t>(1 + (h0 % 255));
        w[1] = static_cast<uint8_t>(1 + (h1 % 255));
        w[2] = static_cast<uint8_t>(1 + (h2 % 255));
        w[3] = static_cast<uint8_t>(1 + (h3 % 255));
    }

    inline void generate4_avx2(uint64_t i, uint32_t *u, uint32_t *v, uint8_t *w) const {
        uint64_t s0 = mix_hash(mixed_seed ^ ((i + 0) * 0xd6e8feb86659fd93ULL));
        uint64_t s1 = mix_hash(mixed_seed ^ ((i + 1) * 0xd6e8feb86659fd93ULL));
        uint64_t s2 = mix_hash(mixed_seed ^ ((i + 2) * 0xd6e8feb86659fd93ULL));
        uint64_t s3 = mix_hash(mixed_seed ^ ((i + 3) * 0xd6e8feb86659fd93ULL));

        uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        uint32_t b0 = 0, b1 = 0, b2 = 0, b3 = 0;

        const __m256i sign_flip = _mm256_set1_epi8((char)0x80);
        const __m256i thr_a     = _mm256_set1_epi8((char)(195 ^ 0x80) - 1);  // 66
        const __m256i thr_b_lo  = _mm256_set1_epi8((char)(146 ^ 0x80) - 1);  // 17
        const __m256i thr_b_hi  = _mm256_set1_epi8((char)(195 ^ 0x80));      // 67
        const __m256i thr_b_top = _mm256_set1_epi8((char)(243 ^ 0x80) - 1);  // 114

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

        uint64_t lo0 = std::min(a0, b0), hi0 = std::max(a0, b0);
        uint64_t lo1 = std::min(a1, b1), hi1 = std::max(a1, b1);
        uint64_t lo2 = std::min(a2, b2), hi2 = std::max(a2, b2);
        uint64_t lo3 = std::min(a3, b3), hi3 = std::max(a3, b3);

        uint64_t h0 = mix_hash(mixed_seed_w ^ (((hi0 << 32) | lo0) * 0xd6e8feb86659fd93ULL));
        uint64_t h1 = mix_hash(mixed_seed_w ^ (((hi1 << 32) | lo1) * 0xd6e8feb86659fd93ULL));
        uint64_t h2 = mix_hash(mixed_seed_w ^ (((hi2 << 32) | lo2) * 0xd6e8feb86659fd93ULL));
        uint64_t h3 = mix_hash(mixed_seed_w ^ (((hi3 << 32) | lo3) * 0xd6e8feb86659fd93ULL));

        w[0] = static_cast<uint8_t>(1 + (h0 % 255));
        w[1] = static_cast<uint8_t>(1 + (h1 % 255));
        w[2] = static_cast<uint8_t>(1 + (h2 % 255));
        w[3] = static_cast<uint8_t>(1 + (h3 % 255));
    }

    // G5: 8-way AVX2 (Dual-Track Interleaved Execution)
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

        uint64_t lo0 = std::min(a0, b0), hi0 = std::max(a0, b0);
        uint64_t lo4 = std::min(a4, b4), hi4 = std::max(a4, b4);
        uint64_t lo1 = std::min(a1, b1), hi1 = std::max(a1, b1);
        uint64_t lo5 = std::min(a5, b5), hi5 = std::max(a5, b5);
        uint64_t lo2 = std::min(a2, b2), hi2 = std::max(a2, b2);
        uint64_t lo6 = std::min(a6, b6), hi6 = std::max(a6, b6);
        uint64_t lo3 = std::min(a3, b3), hi3 = std::max(a3, b3);
        uint64_t lo7 = std::min(a7, b7), hi7 = std::max(a7, b7);

        uint64_t h0 = mix_hash(mixed_seed_w ^ (((hi0 << 32) | lo0) * 0xd6e8feb86659fd93ULL));
        uint64_t h4 = mix_hash(mixed_seed_w ^ (((hi4 << 32) | lo4) * 0xd6e8feb86659fd93ULL));
        uint64_t h1 = mix_hash(mixed_seed_w ^ (((hi1 << 32) | lo1) * 0xd6e8feb86659fd93ULL));
        uint64_t h5 = mix_hash(mixed_seed_w ^ (((hi5 << 32) | lo5) * 0xd6e8feb86659fd93ULL));
        uint64_t h2 = mix_hash(mixed_seed_w ^ (((hi2 << 32) | lo2) * 0xd6e8feb86659fd93ULL));
        uint64_t h6 = mix_hash(mixed_seed_w ^ (((hi6 << 32) | lo6) * 0xd6e8feb86659fd93ULL));
        uint64_t h3 = mix_hash(mixed_seed_w ^ (((hi3 << 32) | lo3) * 0xd6e8feb86659fd93ULL));
        uint64_t h7 = mix_hash(mixed_seed_w ^ (((hi7 << 32) | lo7) * 0xd6e8feb86659fd93ULL));

        w[0] = static_cast<uint8_t>(1 + (h0 % 255));
        w[1] = static_cast<uint8_t>(1 + (h1 % 255));
        w[2] = static_cast<uint8_t>(1 + (h2 % 255));
        w[3] = static_cast<uint8_t>(1 + (h3 % 255));
        w[4] = static_cast<uint8_t>(1 + (h4 % 255));
        w[5] = static_cast<uint8_t>(1 + (h5 % 255));
        w[6] = static_cast<uint8_t>(1 + (h6 % 255));
        w[7] = static_cast<uint8_t>(1 + (h7 % 255));
    }
};

// Single-rank Dial's algorithm (Version 3)
struct __attribute__((packed)) CSRNeighbor {
    uint32_t to;
    uint8_t w;
};

struct BucketEntry {
    uint32_t vertex;
    uint32_t distance;
};

class DialQueue {
public:
    std::array<std::vector<BucketEntry>, 256> buckets;

    void clear() {
        for (auto &b : buckets) {
            b.clear();
        }
    }
};

void dial(const uint64_t *row, const CSRNeighbor *adj,
          uint64_t N, uint32_t src, std::vector<uint32_t> &dist,
          DialQueue &q)
{
    dist.assign(N, INF);
    uint64_t queued = 0;
    uint32_t current = 0;

    dist[src] = 0;
    q.buckets[0].push_back({src, 0});
    ++queued;

    while (queued != 0) {
        auto &bucket = q.buckets[current & 255u];
        if (bucket.empty()) {
            ++current;
            continue;
        }

        BucketEntry entry = bucket.back();
        bucket.pop_back();
        --queued;

        if (entry.distance != current) {
            std::fprintf(stderr, "bucket collision: current=%u entry.distance=%u\n",
                         current, entry.distance);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        const uint32_t x = entry.vertex;
        if (dist[x] != entry.distance) {
            continue; // stale entry superseded by shorter path
        }

        const uint32_t d = entry.distance;

        for (uint64_t e = row[x]; e < row[x + 1]; ++e) {
            const uint32_t y = adj[e].to;
            const uint32_t nd = d + static_cast<uint32_t>(adj[e].w);
            if (nd < dist[y]) {
                dist[y] = nd;
                q.buckets[nd & 255u].push_back({y, nd});
                ++queued;
            }
        }
    }
}

#if ENABLE_PROFILE
struct RoundSummary {
    uint32_t src_idx = 0;
    uint32_t round_idx = 0;
    uint32_t distance = 0;
    uint64_t frontier_size = 0;
    uint64_t total_degree = 0;
    uint64_t max_degree = 0;
    double avg_degree = 0.0;
    uint64_t max_scanned_edges = 0;
    uint64_t min_scanned_edges = 0;
    double avg_scanned_edges = 0.0;
    double edge_imbalance_ratio = 1.0;
    double max_relax_time = 0.0;
    double min_relax_time = 0.0;
    double avg_relax_time = 0.0;
    double time_imbalance_ratio = 1.0;
    double barrier_skew = 0.0;
    uint32_t hubs_gt_1k = 0;
    uint32_t hubs_gt_4k = 0;
    uint32_t hubs_gt_16k = 0;
    uint32_t hubs_gt_64k = 0;
    uint64_t edges_gt_1k = 0;
    uint64_t edges_gt_4k = 0;
    uint64_t edges_gt_16k = 0;
    uint64_t edges_gt_64k = 0;
};

struct PhaseTimers {
    double t_alloc = 0.0;
    // Pass 1
    double t_p1_gen = 0.0;
    double t_p1_store_deg = 0.0;
    double t_p1_sync = 0.0;
    double t_p1_total = 0.0;
    // Prefix
    double t_pref_agg = 0.0;
    double t_pref_scan = 0.0;
    double t_pref_cursor = 0.0;
    double t_pref_sync = 0.0;
    double t_pref_total = 0.0;
    // Pass 2
    double t_p2_count = 0.0;
    double t_p2_scatter = 0.0;
    double t_p2_fill = 0.0;
    double t_p2_sync = 0.0;
    double t_p2_total = 0.0;
    // SSSP
    double t_sssp_init = 0.0;
    double t_sssp_relax = 0.0;
    double t_sssp_coll = 0.0;
    double t_sssp_result = 0.0;
    double t_sssp_total = 0.0;
    // Copy
    double t_copy = 0.0;
    // Actual Run Total
    double t_run_total = 0.0;

    void to_array(double arr[22]) const {
        arr[0] = t_alloc;
        arr[1] = t_p1_total;
        arr[2] = t_p1_gen;
        arr[3] = t_p1_store_deg;
        arr[4] = t_p1_sync;
        arr[5] = t_pref_total;
        arr[6] = t_pref_agg;
        arr[7] = t_pref_scan;
        arr[8] = t_pref_cursor;
        arr[9] = t_pref_sync;
        arr[10] = t_p2_total;
        arr[11] = t_p2_count;
        arr[12] = t_p2_scatter;
        arr[13] = t_p2_fill;
        arr[14] = t_p2_sync;
        arr[15] = t_sssp_total;
        arr[16] = t_sssp_init;
        arr[17] = t_sssp_relax;
        arr[18] = t_sssp_coll;
        arr[19] = t_sssp_result;
        arr[20] = t_copy;
        arr[21] = t_run_total;
    }
};
#endif

struct SSSPStats {
    uint64_t rounds = 0;
    uint64_t total_frontier = 0;
    double collective_time = 0.0;
    double relax_time = 0.0;
    double init_time = 0.0;
    double result_time = 0.0;
#if ENABLE_PROFILE
    std::vector<RoundSummary> round_summaries;
#endif
};

// Cooperative Dial's algorithm for a team of ranks
void cooperative_dial(
    const uint64_t *row, const CSRNeighbor *adj,
    uint64_t N, uint32_t src,
    uint32_t *dist, uint32_t *shared_frontier,
    uint64_t *shared_work_index,
    MPI_Comm team_comm, int team_rank, int team_size, MPI_Win win,
    SSSPStats &stats)
{
#if ENABLE_PROFILE
    double t_init_0 = MPI_Wtime();
#endif
    // 1. Parallel initialize dist
    const uint64_t v_begin = (N * static_cast<uint64_t>(team_rank)) / team_size;
    const uint64_t v_end   = (N * static_cast<uint64_t>(team_rank + 1)) / team_size;
    std::fill(dist + v_begin, dist + v_end, INF);

    MPI_Win_sync(win);
    MPI_Barrier(team_comm);
    MPI_Win_sync(win);

    std::array<std::vector<uint32_t>, 256> local_buckets;

    if (team_rank == 0) {
        dist[src] = 0;
        local_buckets[0].push_back(src);
    }

    MPI_Win_sync(win);
    MPI_Barrier(team_comm);
    MPI_Win_sync(win);
#if ENABLE_PROFILE
    stats.init_time += (MPI_Wtime() - t_init_0);
#endif

    uint32_t current = 0;

    while (current != INF) {
#if ENABLE_PROFILE
        stats.rounds++;
        double tc0 = MPI_Wtime();
#endif
        auto &my_bucket = local_buckets[current & 255u];
        uint64_t local_count = my_bucket.size();

        // 2. Prefix sum to merge local buckets into shared_frontier
        uint64_t my_offset = 0;
        MPI_Exscan(&local_count, &my_offset, 1, MPI_UINT64_T, MPI_SUM, team_comm);
        if (team_rank == 0) my_offset = 0;

        uint64_t total_count = 0;
        MPI_Allreduce(&local_count, &total_count, 1, MPI_UINT64_T, MPI_SUM, team_comm);
#if ENABLE_PROFILE
        stats.collective_time += (MPI_Wtime() - tc0);
#endif

        if (total_count > 0) {
#if ENABLE_PROFILE
            stats.total_frontier += total_count;

            // Phase 1: Hub & Degree distribution of shared frontier
            uint64_t loc_tot_deg = 0;
            uint64_t loc_max_deg = 0;
            uint32_t loc_hubs[4] = {0, 0, 0, 0}; // >1k, >4k, >16k, >64k
            uint64_t loc_hub_edges[4] = {0, 0, 0, 0};

            for (uint32_t x : my_bucket) {
                if (dist[x] != current) continue;
                uint64_t deg = row[x + 1] - row[x];
                loc_tot_deg += deg;
                if (deg > loc_max_deg) loc_max_deg = deg;
                if (deg > 1024)  { loc_hubs[0]++; loc_hub_edges[0] += deg; }
                if (deg > 4096)  { loc_hubs[1]++; loc_hub_edges[1] += deg; }
                if (deg > 16384) { loc_hubs[2]++; loc_hub_edges[2] += deg; }
                if (deg > 65536) { loc_hubs[3]++; loc_hub_edges[3] += deg; }
            }

            uint64_t round_tot_deg = 0, round_max_deg = 0;
            uint32_t round_hubs[4] = {0, 0, 0, 0};
            uint64_t round_hub_edges[4] = {0, 0, 0, 0};

            MPI_Reduce(&loc_tot_deg, &round_tot_deg, 1, MPI_UINT64_T, MPI_SUM, 0, team_comm);
            MPI_Reduce(&loc_max_deg, &round_max_deg, 1, MPI_UINT64_T, MPI_MAX, 0, team_comm);
            MPI_Reduce(loc_hubs, round_hubs, 4, MPI_UINT32_T, MPI_SUM, 0, team_comm);
            MPI_Reduce(loc_hub_edges, round_hub_edges, 4, MPI_UINT64_T, MPI_SUM, 0, team_comm);
#endif

            if (local_count > 0) {
                std::memcpy(shared_frontier + my_offset, my_bucket.data(), local_count * sizeof(uint32_t));
                my_bucket.clear();
            }

            if (team_rank == 0) {
                *shared_work_index = 0;
            }

#if ENABLE_PROFILE
            double tc1 = MPI_Wtime();
#endif
            MPI_Win_sync(win);
            MPI_Barrier(team_comm);
            MPI_Win_sync(win);
#if ENABLE_PROFILE
            stats.collective_time += (MPI_Wtime() - tc1);

            uint64_t my_scanned_edges = 0;
            double tr0 = MPI_Wtime();
#endif
            // 3. Dynamic work chunk distribution across team ranks
            const uint64_t work_chunk = choose_work_chunk(total_count, team_size);
            while (true) {
                uint64_t begin = __atomic_fetch_add(shared_work_index, work_chunk, __ATOMIC_RELAXED);
                if (begin >= total_count) break;
                uint64_t end = std::min(begin + work_chunk, total_count);

                for (uint64_t i = begin; i < end; ++i) {
                    uint32_t x = shared_frontier[i];
                    if (dist[x] != current) continue; // stale

                    uint64_t rx_begin = row[x];
                    uint64_t rx_end   = row[x + 1];
#if ENABLE_PROFILE
                    my_scanned_edges += (rx_end - rx_begin);
#endif
                    for (uint64_t e = rx_begin; e < rx_end; ++e) {
                        uint32_t y = adj[e].to;
                        uint32_t nd = current + static_cast<uint32_t>(adj[e].w);
                        if (atomic_min_u32(&dist[y], nd)) {
                            local_buckets[nd & 255u].push_back(y);
                        }
                    }
                }
            }
#if ENABLE_PROFILE
            double my_relax_time = MPI_Wtime() - tr0;
            stats.relax_time += my_relax_time;

            std::vector<uint64_t> all_scanned(team_size);
            std::vector<double> all_times(team_size);
            MPI_Gather(&my_scanned_edges, 1, MPI_UINT64_T, all_scanned.data(), 1, MPI_UINT64_T, 0, team_comm);
            MPI_Gather(&my_relax_time, 1, MPI_DOUBLE, all_times.data(), 1, MPI_DOUBLE, 0, team_comm);

            if (team_rank == 0) {
                uint64_t max_e = 0, min_e = ~0ULL, sum_e = 0;
                double max_t = 0.0, min_t = 1e9, sum_t = 0.0;
                for (int r = 0; r < team_size; ++r) {
                    if (all_scanned[r] > max_e) max_e = all_scanned[r];
                    if (all_scanned[r] < min_e) min_e = all_scanned[r];
                    sum_e += all_scanned[r];

                    if (all_times[r] > max_t) max_t = all_times[r];
                    if (all_times[r] < min_t) min_t = all_times[r];
                    sum_t += all_times[r];
                }
                double avg_e = (team_size > 0) ? (static_cast<double>(sum_e) / team_size) : 0.0;
                double edge_imbalance = (avg_e > 0.0) ? (static_cast<double>(max_e) / avg_e) : 1.0;
                double avg_t = (team_size > 0) ? (sum_t / team_size) : 0.0;
                double time_imbalance = (avg_t > 0.0) ? (max_t / avg_t) : 1.0;
                double barrier_skew = (max_t >= min_t) ? (max_t - min_t) : 0.0;

                RoundSummary rinfo{};
                rinfo.src_idx = static_cast<uint32_t>(src);
                rinfo.round_idx = static_cast<uint32_t>(stats.rounds);
                rinfo.distance = current;
                rinfo.frontier_size = total_count;
                rinfo.total_degree = round_tot_deg;
                rinfo.max_degree = round_max_deg;
                rinfo.avg_degree = (total_count > 0) ? (static_cast<double>(round_tot_deg) / total_count) : 0.0;
                rinfo.max_scanned_edges = max_e;
                rinfo.min_scanned_edges = (min_e == ~0ULL) ? 0 : min_e;
                rinfo.avg_scanned_edges = avg_e;
                rinfo.edge_imbalance_ratio = edge_imbalance;
                rinfo.max_relax_time = max_t;
                rinfo.min_relax_time = (min_t >= 1e8) ? 0.0 : min_t;
                rinfo.avg_relax_time = avg_t;
                rinfo.time_imbalance_ratio = time_imbalance;
                rinfo.barrier_skew = barrier_skew;
                rinfo.hubs_gt_1k = round_hubs[0];
                rinfo.hubs_gt_4k = round_hubs[1];
                rinfo.hubs_gt_16k = round_hubs[2];
                rinfo.hubs_gt_64k = round_hubs[3];
                rinfo.edges_gt_1k = round_hub_edges[0];
                rinfo.edges_gt_4k = round_hub_edges[1];
                rinfo.edges_gt_16k = round_hub_edges[2];
                rinfo.edges_gt_64k = round_hub_edges[3];

                stats.round_summaries.push_back(rinfo);
            }

            double tc2 = MPI_Wtime();
#endif
            MPI_Win_sync(win);
            MPI_Barrier(team_comm);
            MPI_Win_sync(win);
#if ENABLE_PROFILE
            stats.collective_time += (MPI_Wtime() - tc2);
#endif
        } else {
            my_bucket.clear();
        }

        // 4. Find next distance across the team
        uint32_t local_next = INF;
        for (uint32_t delta = 1; delta <= 255; ++delta) {
            uint32_t cand = current + delta;
            if (!local_buckets[cand & 255u].empty()) {
                local_next = cand;
                break;
            }
        }

#if ENABLE_PROFILE
        double tc3 = MPI_Wtime();
#endif
        uint32_t team_next = INF;
        MPI_Allreduce(&local_next, &team_next, 1, MPI_UINT32_T, MPI_MIN, team_comm);
#if ENABLE_PROFILE
        stats.collective_time += (MPI_Wtime() - tc3);
#endif
        current = team_next;
    }
}

// Cooperative Delta-Stepping algorithm
void cooperative_delta_stepping(
    const uint64_t *row, const CSRNeighbor *adj,
    uint64_t N, uint32_t src,
    uint32_t *dist, uint32_t *shared_frontier,
    uint64_t *shared_work_index,
    MPI_Comm team_comm, int team_rank, int team_size, MPI_Win win,
    SSSPStats &stats, uint32_t delta = 2)
{
    const uint64_t v_begin = (N * static_cast<uint64_t>(team_rank)) / team_size;
    const uint64_t v_end   = (N * static_cast<uint64_t>(team_rank + 1)) / team_size;
    std::fill(dist + v_begin, dist + v_end, INF);

    MPI_Win_sync(win);
    MPI_Barrier(team_comm);
    MPI_Win_sync(win);

    std::vector<std::vector<uint32_t>> local_bins;
    if (team_rank == 0) {
        dist[src] = 0;
        local_bins.resize(1);
        local_bins[0].push_back(src);
    }

    MPI_Win_sync(win);
    MPI_Barrier(team_comm);
    MPI_Win_sync(win);

    uint32_t curr_bin = 0;

    while (curr_bin != INF) {
#if ENABLE_PROFILE
        stats.rounds++;
        double tc0 = MPI_Wtime();
#endif
        uint64_t local_count = (curr_bin < local_bins.size()) ? local_bins[curr_bin].size() : 0;

        uint64_t my_offset = 0;
        MPI_Exscan(&local_count, &my_offset, 1, MPI_UINT64_T, MPI_SUM, team_comm);
        if (team_rank == 0) my_offset = 0;

        uint64_t total_count = 0;
        MPI_Allreduce(&local_count, &total_count, 1, MPI_UINT64_T, MPI_SUM, team_comm);
#if ENABLE_PROFILE
        stats.collective_time += (MPI_Wtime() - tc0);
#endif

        if (total_count > 0) {
#if ENABLE_PROFILE
            stats.total_frontier += total_count;
#endif

            if (local_count > 0) {
                std::memcpy(shared_frontier + my_offset, local_bins[curr_bin].data(), local_count * sizeof(uint32_t));
                local_bins[curr_bin].clear();
            }

            if (team_rank == 0) {
                *shared_work_index = 0;
            }

#if ENABLE_PROFILE
            double tc1 = MPI_Wtime();
#endif
            MPI_Win_sync(win);
            MPI_Barrier(team_comm);
            MPI_Win_sync(win);
#if ENABLE_PROFILE
            stats.collective_time += (MPI_Wtime() - tc1);

            double tr0 = MPI_Wtime();
#endif
            const uint64_t work_chunk = choose_work_chunk(total_count, team_size);
            while (true) {
                uint64_t begin = __atomic_fetch_add(shared_work_index, work_chunk, __ATOMIC_RELAXED);
                if (begin >= total_count) break;
                uint64_t end = std::min(begin + work_chunk, total_count);

                for (uint64_t i = begin; i < end; ++i) {
                    uint32_t u = shared_frontier[i];
                    uint32_t du = dist[u];
                    if (du < curr_bin * delta) continue;

                    for (uint64_t e = row[u]; e < row[u + 1]; ++e) {
                        uint32_t v = adj[e].to;
                        uint32_t nd = du + static_cast<uint32_t>(adj[e].w);
                        if (atomic_min_u32(&dist[v], nd)) {
                            uint32_t dest_bin = nd / delta;
                            if (dest_bin >= local_bins.size()) {
                                local_bins.resize(dest_bin + 1);
                            }
                            local_bins[dest_bin].push_back(v);
                        }
                    }
                }
            }
#if ENABLE_PROFILE
            stats.relax_time += (MPI_Wtime() - tr0);

            double tc2 = MPI_Wtime();
#endif
            MPI_Win_sync(win);
            MPI_Barrier(team_comm);
            MPI_Win_sync(win);
#if ENABLE_PROFILE
            stats.collective_time += (MPI_Wtime() - tc2);
#endif
        } else {
            // Find next non-empty bin
            uint32_t local_next = INF;
            for (size_t i = curr_bin + 1; i < local_bins.size(); ++i) {
                if (!local_bins[i].empty()) {
                    local_next = static_cast<uint32_t>(i);
                    break;
                }
            }

#if ENABLE_PROFILE
            double tc3 = MPI_Wtime();
#endif
            uint32_t team_next = INF;
            MPI_Allreduce(&local_next, &team_next, 1, MPI_UINT32_T, MPI_MIN, team_comm);
#if ENABLE_PROFILE
            stats.collective_time += (MPI_Wtime() - tc3);
#endif
            curr_bin = team_next;
        }
    }
}

}  // namespace

void run(const GraphParams &p, const std::vector<uint32_t> &sources, std::vector<SourceResult> &results)
{
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

#if ENABLE_PROFILE
    MPI_Barrier(MPI_COMM_WORLD);
    double t_run_start = MPI_Wtime();
    double t0 = t_run_start;
    PhaseTimers timers;
#endif

    // 1. Create shared communicator for processes on the same node
    MPI_Comm shm_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &shm_comm);
    int shm_rank, shm_size;
    MPI_Comm_rank(shm_comm, &shm_rank);
    MPI_Comm_size(shm_comm, &shm_size);

    const uint64_t N = num_vertices(p);
    const uint64_t M = num_edges(p);
    const size_t S = sources.size();

    // Cooperative team configuration (when sufficient ranks exceed sources)
    bool use_coop = (world_size >= 4 && world_size > static_cast<int>(S) && p.scale >= 20);
    int team_count = 1;
    if (use_coop) {
#if SOURCE_TEAM_MODE == 1
        // Mode 1: two teams.
        team_count = std::min<int>(world_size, 2);
#elif SOURCE_TEAM_MODE == 2
        // Mode 2: one team.
        team_count = 1;
#elif SOURCE_TEAM_MODE == 3
        // Hybrid mode.
        const int source_count = static_cast<int>(S);
        if (source_count <= 4 && M <= (uint64_t(1) << 26)) {
            // Small graph with few sources:
            // keep maximum source-level parallelism.
            team_count = std::min<int>(world_size, source_count);
        } else {
            // Medium/large graph or more sources:
            // use two teams to balance source concurrency,
            // memory bandwidth, team size, and collective cost.
            team_count = std::min<int>(world_size, 2);
        }
#else
        // Mode 0: maximum source-level concurrency.
        team_count = std::min<int>(world_size, static_cast<int>(S));
#endif
    }

#if PRINT_TEAM_SELECTION
    if (world_rank == 0) {
        std::printf(
            "[Team Selection] "
            "scale=%d edgefactor=%d "
            "S=%zu M=%llu "
            "use_coop=%d teams=%d\n",
            p.scale,
            p.edgefactor,
            S,
            static_cast<unsigned long long>(M),
            use_coop ? 1 : 0,
            team_count
        );
    }
#endif

    // 2. Compute memory layout for Shared-Memory Window
    uint64_t row_bytes     = align64((N + 1) * sizeof(uint64_t));
    uint64_t adj_bytes     = align64((2 * M) * sizeof(CSRNeighbor));
    uint64_t hdr_bytes     = align64(S * sizeof(FlatResultHeader));
    uint64_t sample_bytes  = align64((uint64_t)S * p.nsample * sizeof(int32_t));
    uint64_t all_deg_bytes = align64((uint64_t)shm_size * N * sizeof(uint32_t));

    // Cooperative mode buffers (up to team_count teams)
    uint64_t coop_dist_bytes     = use_coop ? align64((uint64_t)team_count * N * sizeof(uint32_t)) : 0;
    uint64_t coop_frontier_bytes = use_coop ? align64((uint64_t)team_count * N * sizeof(uint32_t)) : 0;
    uint64_t coop_ctrl_bytes     = use_coop ? align64((uint64_t)team_count * 64) : 0;

    // Window Buffer Reuse: all_degree (used only during graph construction)
    // shares space with coop_dist and coop_frontier (used only during SSSP)
    uint64_t coop_needed_bytes   = coop_dist_bytes + coop_frontier_bytes;
    uint64_t scratch_bytes       = std::max(all_deg_bytes, coop_needed_bytes);

    uint64_t total_bytes = row_bytes + adj_bytes + hdr_bytes + sample_bytes + scratch_bytes + coop_ctrl_bytes;

    MPI_Win win;
    void *baseptr = nullptr;
    MPI_Aint alloc_size = (shm_rank == 0) ? static_cast<MPI_Aint>(total_bytes) : 0;
    MPI_Win_allocate_shared(alloc_size, 1, MPI_INFO_NULL, shm_comm, &baseptr, &win);

    if (shm_rank != 0) {
        MPI_Aint sz;
        int disp;
        MPI_Win_shared_query(win, 0, &sz, &disp, &baseptr);
    }

    // Partition the shared memory window into typed pointers
    char *ptr = static_cast<char *>(baseptr);
    uint64_t *row = reinterpret_cast<uint64_t *>(ptr);
    ptr += row_bytes;
    CSRNeighbor *adj = reinterpret_cast<CSRNeighbor *>(ptr);
    ptr += adj_bytes;
    FlatResultHeader *res_hdr = reinterpret_cast<FlatResultHeader *>(ptr);
    ptr += hdr_bytes;
    int32_t *sample_dist_flat = reinterpret_cast<int32_t *>(ptr);
    ptr += sample_bytes;

    char *scratch_base = ptr;
    ptr += scratch_bytes;

    uint64_t *coop_work_index = reinterpret_cast<uint64_t *>(ptr);
    ptr += coop_ctrl_bytes;

    uint32_t *all_degree = reinterpret_cast<uint32_t *>(scratch_base);
    uint32_t *coop_dist = reinterpret_cast<uint32_t *>(scratch_base);
    uint32_t *coop_frontier = reinterpret_cast<uint32_t *>(scratch_base + coop_dist_bytes);

    // Each rank owns its distinct slice of all_degree: [shm_rank * N, (shm_rank + 1) * N)
    uint32_t *my_degree = all_degree + static_cast<uint64_t>(shm_rank) * N;
    std::memset(my_degree, 0, N * sizeof(uint32_t));

    MPI_Win_sync(win);
    MPI_Barrier(shm_comm);
    MPI_Win_sync(win);

#if ENABLE_PROFILE
    timers.t_alloc = MPI_Wtime() - t0;
    double t1 = MPI_Wtime();
#endif

    // 3. Parallel Graph Construction
    const uint64_t edge_begin = (M * static_cast<uint64_t>(shm_rank)) / shm_size;
    const uint64_t edge_end   = (M * static_cast<uint64_t>(shm_rank + 1)) / shm_size;
    const uint64_t local_M    = edge_end - edge_begin;

    const uint64_t CHUNK = CHUNK_SIZE;
    std::vector<uint32_t> u(CHUNK), v(CHUNK);
    std::vector<uint8_t> w_chunk(CHUNK);

#if CACHE_EDGES
    std::vector<uint64_t> packed_edges(local_M);
#endif

    FastGenerator gctx;
    gctx.init(p);

    // Pass 1: count degrees (and optionally cache packed edges)
    for (uint64_t i = edge_begin; i < edge_end; i += CHUNK) {
        uint64_t chunk_len = std::min(CHUNK, edge_end - i);
#if ENABLE_PROFILE
        double tg0 = MPI_Wtime();
#endif
#if GEN_MODE == 1
        uint64_t k = 0;
        for (; k + 4 <= chunk_len; k += 4) {
            gctx.generate4_scalar(i + k, &u[k], &v[k], &w_chunk[k]);
        }
        for (; k < chunk_len; ++k) {
            gctx.generate(i + k, u[k], v[k], w_chunk[k]);
        }
#elif GEN_MODE == 2
        uint64_t k = 0;
        for (; k + 4 <= chunk_len; k += 4) {
            gctx.generate4_avx2(i + k, &u[k], &v[k], &w_chunk[k]);
        }
        for (; k < chunk_len; ++k) {
            gctx.generate(i + k, u[k], v[k], w_chunk[k]);
        }
#elif GEN_MODE == 3
        uint64_t k = 0;
        for (; k + 8 <= chunk_len; k += 8) {
            gctx.generate8_avx2(i + k, &u[k], &v[k], &w_chunk[k]);
        }
        for (; k + 4 <= chunk_len; k += 4) {
            gctx.generate4_avx2(i + k, &u[k], &v[k], &w_chunk[k]);
        }
        for (; k < chunk_len; ++k) {
            gctx.generate(i + k, u[k], v[k], w_chunk[k]);
        }
#else
        for (uint64_t k = 0; k < chunk_len; k++) {
            gctx.generate(i + k, u[k], v[k], w_chunk[k]);
        }
#endif
#if ENABLE_PROFILE
        timers.t_p1_gen += (MPI_Wtime() - tg0);
        double ts0 = MPI_Wtime();
#endif
        uint64_t base_idx = i - edge_begin;
        for (uint64_t k = 0; k < chunk_len; k++) {
            uint32_t uk = u[k], vk = v[k];
            uint8_t wk  = w_chunk[k];

            my_degree[uk]++;
            my_degree[vk]++;

#if CACHE_EDGES
            packed_edges[base_idx + k] = (static_cast<uint64_t>(uk) << 34) |
                                         (static_cast<uint64_t>(vk) << 8) |
                                         static_cast<uint64_t>(wk);
#endif
        }
#if ENABLE_PROFILE
        timers.t_p1_store_deg += (MPI_Wtime() - ts0);
#endif
    }

#if CACHE_EDGES
    std::vector<uint32_t>().swap(u);
    std::vector<uint32_t>().swap(v);
    std::vector<uint8_t>().swap(w_chunk);
#endif

#if ENABLE_PROFILE
    double t_p1_sync0 = MPI_Wtime();
#endif
    MPI_Win_sync(win);
    MPI_Barrier(shm_comm);
    MPI_Win_sync(win);

#if ENABLE_PROFILE
    timers.t_p1_sync = MPI_Wtime() - t_p1_sync0;
    timers.t_p1_total = MPI_Wtime() - t1;
    double t_pass1 = timers.t_p1_total;
#endif

#if ENABLE_DEDUP_PROFILE
    {
        // 1. Layered deterministic strided sampling across all ranks
        const uint64_t sample_per_rank = std::min<uint64_t>(local_M, 500000ULL);
        std::vector<uint64_t> local_sample_keys;
        local_sample_keys.reserve(sample_per_rank);

        uint64_t local_self_loops = 0;

        for (uint64_t k = 0; k < sample_per_rank; ++k) {
            uint64_t idx = static_cast<uint64_t>(
                (static_cast<unsigned __int128>(k) * local_M) / sample_per_rank
            );
            uint64_t packed = packed_edges[idx];
            uint32_t uk = static_cast<uint32_t>(packed >> 34);
            uint32_t vk = static_cast<uint32_t>((packed >> 8) & 0x3ffffffu);

            if (uk == vk) {
                local_self_loops++;
            } else {
                uint32_t lo = std::min(uk, vk);
                uint32_t hi = std::max(uk, vk);
                local_sample_keys.push_back((static_cast<uint64_t>(lo) << 32) | hi);
            }
        }

        // Rank-local sort + unique
        std::vector<uint64_t> local_sorted = local_sample_keys;
        std::sort(local_sorted.begin(), local_sorted.end());
        auto uend = std::unique(local_sorted.begin(), local_sorted.end());
        uint64_t local_unique = uend - local_sorted.begin();
        uint64_t local_nonloops = local_sample_keys.size();
        uint64_t local_dups = local_nonloops - local_unique;
        double local_unique_ratio = (local_nonloops > 0) ? (double)local_unique / local_nonloops : 1.0;
        std::vector<uint64_t>().swap(local_sorted);

        // Gather all non-loop sample keys to rank 0 for Global Sample Sort + Unique
        int sendcount = static_cast<int>(local_nonloops);
        std::vector<int> recvcounts(world_size, 0);
        MPI_Gather(&sendcount, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        std::vector<int> displs(world_size, 0);
        int total_gathered_count = 0;
        if (world_rank == 0) {
            for (int r = 0; r < world_size; ++r) {
                displs[r] = total_gathered_count;
                total_gathered_count += recvcounts[r];
            }
        }

        std::vector<uint64_t> all_sample_keys;
        if (world_rank == 0) {
            all_sample_keys.resize(total_gathered_count);
        }

        MPI_Gatherv(local_sample_keys.data(), sendcount, MPI_UINT64_T,
                    all_sample_keys.data(), recvcounts.data(), displs.data(), MPI_UINT64_T,
                    0, MPI_COMM_WORLD);

        std::vector<uint64_t>().swap(local_sample_keys); // Free local buffer

        uint64_t global_sample_unique = 0;
        uint64_t global_sample_dups = 0;
        double global_sample_unique_pct = 0.0;
        double global_sample_dup_pct = 0.0;

        if (world_rank == 0) {
            std::sort(all_sample_keys.begin(), all_sample_keys.end());
            auto g_last = std::unique(all_sample_keys.begin(), all_sample_keys.end());
            global_sample_unique = g_last - all_sample_keys.begin();
            global_sample_dups = total_gathered_count - global_sample_unique;
            if (total_gathered_count > 0) {
                global_sample_unique_pct = 100.0 * (double)global_sample_unique / total_gathered_count;
                global_sample_dup_pct = 100.0 * (double)global_sample_dups / total_gathered_count;
            }
            std::vector<uint64_t>().swap(all_sample_keys); // Free gathered buffer
        }

        // Rank-local reductions
        uint64_t tot_sampled = 0, tot_self_loops = 0, tot_nonloops = 0;
        uint64_t tot_local_unique = 0, tot_local_dups = 0;
        double min_local_ratio = 1.0, max_local_ratio = 0.0;

        MPI_Reduce(&sample_per_rank, &tot_sampled, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_self_loops, &tot_self_loops, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_nonloops, &tot_nonloops, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_unique, &tot_local_unique, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_dups, &tot_local_dups, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_unique_ratio, &min_local_ratio, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_unique_ratio, &max_local_ratio, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        // 2. Full-Graph HyperLogLog (HLL) Cardinality Estimator (262,144 registers, p=18)
        constexpr int HLL_P = 18;
        constexpr uint32_t HLL_M_REG = 1u << HLL_P; // 262144
        std::vector<uint8_t> hll_local(HLL_M_REG, 0);

        uint64_t local_hll_nonloops = 0;
        for (uint64_t k = 0; k < local_M; ++k) {
            uint64_t packed = packed_edges[k];
            uint32_t uk = static_cast<uint32_t>(packed >> 34);
            uint32_t vk = static_cast<uint32_t>((packed >> 8) & 0x3ffffffu);
            if (uk == vk) continue;
            local_hll_nonloops++;

            uint32_t lo = std::min(uk, vk);
            uint32_t hi = std::max(uk, vk);
            uint64_t key = (static_cast<uint64_t>(lo) << 32) | hi;
            uint64_t h = mix_hash(key ^ 0x9e3779b97f4a7c15ULL);

            uint32_t j = h >> (64 - HLL_P);
            uint64_t w = h & ((1ULL << (64 - HLL_P)) - 1);
            uint8_t rho = (w == 0) ? (64 - HLL_P + 1) : static_cast<uint8_t>(__builtin_clzll(w) - HLL_P + 1);
            hll_local[j] = std::max(hll_local[j], rho);
        }

        std::vector<uint8_t> hll_global;
        if (world_rank == 0) hll_global.resize(HLL_M_REG);

        MPI_Reduce(hll_local.data(), hll_global.data(), HLL_M_REG, MPI_UINT8_T, MPI_MAX, 0, MPI_COMM_WORLD);
        std::vector<uint8_t>().swap(hll_local);

        uint64_t tot_hll_nonloops = 0;
        MPI_Reduce(&local_hll_nonloops, &tot_hll_nonloops, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

        if (world_rank == 0) {
            // Compute HLL estimate
            double harmonic_sum = 0.0;
            int zero_regs = 0;
            for (uint32_t j = 0; j < HLL_M_REG; ++j) {
                harmonic_sum += std::pow(2.0, -static_cast<double>(hll_global[j]));
                if (hll_global[j] == 0) zero_regs++;
            }
            double alpha_m = 0.72134;
            double hll_est = alpha_m * static_cast<double>(HLL_M_REG) * static_cast<double>(HLL_M_REG) / harmonic_sum;
            if (hll_est <= 2.5 * HLL_M_REG && zero_regs > 0) {
                hll_est = static_cast<double>(HLL_M_REG) * std::log(static_cast<double>(HLL_M_REG) / zero_regs);
            }
            double hll_unique_pct = (tot_hll_nonloops > 0) ? (100.0 * hll_est / tot_hll_nonloops) : 0.0;
            double hll_dup_pct = std::max(0.0, 100.0 - hll_unique_pct);

            double self_pct = (tot_sampled > 0) ? (100.0 * (double)tot_self_loops / tot_sampled) : 0.0;
            double local_unique_pct = (tot_nonloops > 0) ? (100.0 * (double)tot_local_unique / tot_nonloops) : 0.0;
            double local_dup_pct = (tot_nonloops > 0) ? (100.0 * (double)tot_local_dups / tot_nonloops) : 0.0;

            std::printf("\n=================================================================================\n");
            std::printf("  [DEDUP PROFILER: Scale %d, Edgefactor %d, Scramble %d]\n", p.scale, p.edgefactor, p.scramble);
            std::printf("---------------------------------------------------------------------------------\n");
            std::printf("  [Sampled Level: %llu edges total (%llu/rank)]\n",
                        (unsigned long long)tot_sampled, (unsigned long long)sample_per_rank);
            std::printf("    Self-Loops:                  %llu (%.4f%%)\n",
                        (unsigned long long)tot_self_loops, self_pct);
            std::printf("    Weighted Rank-Local Unique:  %llu (%.4f%% of non-loops)\n",
                        (unsigned long long)tot_local_unique, local_unique_pct);
            std::printf("    Weighted Rank-Local Dup:     %llu (%.4f%% of non-loops)\n",
                        (unsigned long long)tot_local_dups, local_dup_pct);
            std::printf("    Min/Max Per-Rank Unique:     [Min: %.4f%%, Max: %.4f%%]\n",
                        min_local_ratio * 100.0, max_local_ratio * 100.0);
            std::printf("    Global Sampled Unique:       %llu (%.4f%% of sample)\n",
                        (unsigned long long)global_sample_unique, global_sample_unique_pct);
            std::printf("    Global Sampled Duplicate:    %llu (%.4f%% of sample)\n",
                        (unsigned long long)global_sample_dups, global_sample_dup_pct);
            std::printf("---------------------------------------------------------------------------------\n");
            std::printf("  [Full-Graph HLL Estimate: 100%% of %llu non-loop edges scanned]\n",
                        (unsigned long long)tot_hll_nonloops);
            std::printf("    Estimated Global Unique:     %.0f (%.4f%% of total)\n",
                        hll_est, hll_unique_pct);
            std::printf("    Estimated Global Duplicate:  %.0f (%.4f%% of total)\n",
                        tot_hll_nonloops - hll_est, hll_dup_pct);
            std::printf("    HLL Theoretical Std Error:   ±0.20%%\n");
            std::printf("=================================================================================\n\n");
            std::fflush(stdout);

            std::vector<uint8_t>().swap(hll_global);
        }
    }
#endif

#if ENABLE_PROFILE
    double t_pref_0 = MPI_Wtime();
    double t2 = t_pref_0;
    double tagg0 = MPI_Wtime();
#endif

    // Parallel Prefix Sum & Rank-Relative Offset calculation (Absolute Cursor)
    const uint64_t v_begin = (N * static_cast<uint64_t>(shm_rank)) / shm_size;
    const uint64_t v_end   = (N * static_cast<uint64_t>(shm_rank + 1)) / shm_size;

    uint64_t local_adjacency_count = 0;
    for (uint64_t vert = v_begin; vert < v_end; vert++) {
        uint32_t deg_v = 0;
        for (int r = 0; r < shm_size; r++) {
            deg_v += all_degree[static_cast<uint64_t>(r) * N + vert];
        }
        local_adjacency_count += deg_v;
    }
#if ENABLE_PROFILE
    timers.t_pref_agg = MPI_Wtime() - tagg0;
    double tscan0 = MPI_Wtime();
#endif

    uint64_t rank_base = 0;
    MPI_Exscan(&local_adjacency_count, &rank_base, 1, MPI_UINT64_T, MPI_SUM, shm_comm);
    if (shm_rank == 0) rank_base = 0;
#if ENABLE_PROFILE
    timers.t_pref_scan = MPI_Wtime() - tscan0;
    double tcur0 = MPI_Wtime();
#endif

    uint64_t running_row = rank_base;
    for (uint64_t vert = v_begin; vert < v_end; vert++) {
        row[vert] = running_row;
        uint32_t rel_offset = 0;
        for (int r = 0; r < shm_size; r++) {
            uint64_t idx = static_cast<uint64_t>(r) * N + vert;
            uint32_t d = all_degree[idx];
            all_degree[idx] = static_cast<uint32_t>(running_row + rel_offset);
            rel_offset += d;
        }
        running_row += rel_offset;
    }

    if (v_end == N) {
        row[N] = running_row;
    }
#if ENABLE_PROFILE
    timers.t_pref_cursor = MPI_Wtime() - tcur0;
    double t_pref_s0 = MPI_Wtime();
#endif

    MPI_Win_sync(win);
    MPI_Barrier(shm_comm);
    MPI_Win_sync(win);

#if ENABLE_PROFILE
    timers.t_pref_sync = MPI_Wtime() - t_pref_s0;
    timers.t_pref_total = MPI_Wtime() - t_pref_0;
#endif

    if (shm_rank == 0) {
        if (row[0] != 0 || row[N] != 2 * M) {
            std::cerr << "Assertion failed: row[0]=" << row[0]
                      << ", row[N]=" << row[N] << ", expected 2*M=" << 2 * M << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

#if ENABLE_BOUNDARY_PROFILE
    if (world_rank == 0) {
        int P = world_size;
        uint64_t total_adj = row[N];
        uint64_t avg_adj_target = total_adj / P;

        std::vector<uint64_t> boundary(P + 1);
        boundary[0] = 0;
        boundary[P] = N;

        for (int r = 1; r < P; ++r) {
            uint64_t target = (total_adj * static_cast<uint64_t>(r)) / P;
            auto it = std::lower_bound(row, row + N + 1, target);
            uint64_t v = it - row;
            if (v > 0 && v <= N) {
                if (std::abs(static_cast<int64_t>(row[v]) - static_cast<int64_t>(target)) >
                    std::abs(static_cast<int64_t>(row[v - 1]) - static_cast<int64_t>(target))) {
                    v = v - 1;
                }
            }
            if (v < boundary[r - 1]) v = boundary[r - 1];
            if (v > N) v = N;
            boundary[r] = v;
        }

        std::vector<uint64_t> owner_V(P);
        std::vector<uint64_t> owner_adj(P);
        uint64_t max_owner_adj = 0;
        uint64_t min_nonempty_adj = UINT64_MAX;
        uint64_t max_owner_V = 0;
        int empty_owners = 0;

        for (int r = 0; r < P; ++r) {
            owner_V[r] = boundary[r + 1] - boundary[r];
            owner_adj[r] = row[boundary[r + 1]] - row[boundary[r]];
            if (owner_adj[r] > max_owner_adj) max_owner_adj = owner_adj[r];
            if (owner_adj[r] > 0 && owner_adj[r] < min_nonempty_adj) min_nonempty_adj = owner_adj[r];
            if (owner_V[r] > max_owner_V) max_owner_V = owner_V[r];
            if (owner_V[r] == 0) empty_owners++;
        }
        if (min_nonempty_adj == UINT64_MAX) min_nonempty_adj = 0;

        uint64_t max_vertex_degree = 0;
        uint64_t max_deg_vertex = 0;
        for (uint64_t v = 0; v < N; ++v) {
            uint64_t deg = row[v + 1] - row[v];
            if (deg > max_vertex_degree) {
                max_vertex_degree = deg;
                max_deg_vertex = v;
            }
        }

        constexpr int COARSE_BITS = 14;
        const int shift = (p.scale > COARSE_BITS) ? (p.scale - COARSE_BITS) : 0;
        const uint64_t num_blocks = (N + (1ULL << shift) - 1) >> shift;
        std::vector<uint32_t> coarse_owner(num_blocks);

        for (uint64_t b = 0; b < num_blocks; ++b) {
            uint64_t v_start = b << shift;
            if (v_start >= N) v_start = N - 1;
            auto it = std::upper_bound(boundary.begin(), boundary.end(), v_start);
            int o = static_cast<int>(it - boundary.begin()) - 1;
            if (o < 0) o = 0;
            if (o >= P) o = P - 1;
            coarse_owner[b] = static_cast<uint32_t>(o);
        }

        uint64_t test_verts = std::min<uint64_t>(N, 10000000ULL);
        uint64_t test_stride = (N > test_verts) ? (N / test_verts) : 1;
        uint64_t total_steps = 0;
        uint32_t max_steps = 0;
        uint64_t tested_count = 0;
        bool lookup_all_correct = true;

        for (uint64_t v = 0; v < N; v += test_stride) {
            tested_count++;
            uint64_t b = v >> shift;
            int owner = coarse_owner[b];
            int initial_owner = owner;

            while (owner > 0 && v < boundary[owner]) owner--;
            while (owner + 1 < P && v >= boundary[owner + 1]) owner++;

            uint32_t steps = std::abs(owner - initial_owner);
            total_steps += steps;
            if (steps > max_steps) max_steps = steps;

            if (v < boundary[owner] || v >= boundary[owner + 1]) {
                lookup_all_correct = false;
            }
        }

        int duplicate_boundaries = 0;
        for (int r = 1; r < P; ++r) {
            if (boundary[r] == boundary[r - 1]) duplicate_boundaries++;
        }

        uint64_t max_private_mem_bytes = 0;
        for (int r = 0; r < P; ++r) {
            uint64_t cursor_bytes = owner_V[r] * sizeof(uint32_t);
            uint64_t packed_bytes = ((M + P - 1) / P) * sizeof(uint64_t);
            uint64_t worst_staging = 96ULL * 1024 * 1024; // 96 MiB conservative staging peak
            uint64_t total_est = cursor_bytes + packed_bytes + worst_staging;
            if (total_est > max_private_mem_bytes) max_private_mem_bytes = total_est;
        }

        double avg_steps = (tested_count > 0) ? (static_cast<double>(total_steps) / tested_count) : 0.0;
        double max_imbalance = (avg_adj_target > 0) ? (static_cast<double>(max_owner_adj) / avg_adj_target) : 1.0;

        std::printf("\n=================================================================================\n");
        std::printf("  [MILESTONE 7.1: BOUNDARY & OWNER PROFILER (Scale %d, EF %d, Scr %d, P=%d)]\n",
                    p.scale, p.edgefactor, p.scramble, P);
        std::printf("---------------------------------------------------------------------------------\n");
        std::printf("  Total Vertices N:             %llu\n", (unsigned long long)N);
        std::printf("  Total Adjacencies (2*M):      %llu\n", (unsigned long long)total_adj);
        std::printf("  Average Target per Owner:     %llu\n", (unsigned long long)avg_adj_target);
        std::printf("---------------------------------------------------------------------------------\n");
        std::printf("  [Super-Hub & Degree Analysis]\n");
        std::printf("  Maximum Vertex Degree:        %llu (at vertex %llu)\n",
                    (unsigned long long)max_vertex_degree, (unsigned long long)max_deg_vertex);
        if (max_vertex_degree > avg_adj_target) {
            std::printf("  WARNING: Super-Hub degree EXCEEDS average owner target by %.2fx!\n",
                        (double)max_vertex_degree / avg_adj_target);
        } else {
            std::printf("  Super-Hub degree vs Target:   %.2f%% of target (Fits inside single owner)\n",
                        100.0 * max_vertex_degree / avg_adj_target);
        }
        std::printf("---------------------------------------------------------------------------------\n");
        std::printf("  [Owner Load Balance Analysis]\n");
        std::printf("  Max Owner Adjacency:          %llu (%.4fx avg target)\n",
                    (unsigned long long)max_owner_adj, max_imbalance);
        std::printf("  Min Non-Empty Adjacency:      %llu (%.4fx avg target)\n",
                    (unsigned long long)min_nonempty_adj,
                    (avg_adj_target > 0 ? (double)min_nonempty_adj / avg_adj_target : 0.0));
        std::printf("  Empty Owners Count:           %d / %d\n", empty_owners, P);
        std::printf("  Duplicate Boundaries Count:   %d / %d\n", duplicate_boundaries, P);
        std::printf("  Max Owner Vertex Count (local_V): %llu (Avg: %llu, %.2fx avg)\n",
                    (unsigned long long)max_owner_V, (unsigned long long)(N / P),
                    (double)max_owner_V / (N / P));
        std::printf("  Max Private Memory Estimate:  %.2f MiB (Budget: 1200 MiB, Margin: %.2f MiB)\n",
                    (double)max_private_mem_bytes / (1024.0 * 1024.0),
                    1200.0 - (double)max_private_mem_bytes / (1024.0 * 1024.0));
        std::printf("---------------------------------------------------------------------------------\n");
        std::printf("  [Coarse Lookup (Shift=%d, %llu Blocks, %llu Checked)]\n",
                    shift, (unsigned long long)num_blocks, (unsigned long long)tested_count);
        std::printf("  Lookup Verification:          %s\n", lookup_all_correct ? "100% PASSED" : "FAILED!");
        std::printf("  Correction Steps:             Avg = %.4f, Max = %u\n", avg_steps, max_steps);
        std::printf("=================================================================================\n\n");
        std::fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
#endif

#if ENABLE_PROFILE
    double t_prefix = timers.t_pref_total;
    double t3 = MPI_Wtime();
    double t_p2_0 = t3;
#endif

    // Pass 2: fill adj (lock-free) with Absolute Cursor
#if PASS2_MODE == 0
#if CACHE_EDGES
    if (p.scale >= 24) {
#ifndef RADIX_BITS
#define RADIX_BITS 8
#endif
        constexpr uint32_t NUM_BUCKETS = 1u << RADIX_BITS;
        const int bucket_shift = (p.scale > RADIX_BITS) ? (p.scale - RADIX_BITS) : 0;
        const uint64_t RADIX_CHUNK = 8388608; // 8M edges per chunk (128 MB working buffer)
        std::vector<uint64_t> bucketed_edges;
        std::vector<uint32_t> bcount(NUM_BUCKETS);
        std::vector<uint32_t> boffset(NUM_BUCKETS);
        std::vector<uint32_t> bcurr(NUM_BUCKETS);

        for (uint64_t chunk_start = 0; chunk_start < local_M; chunk_start += RADIX_CHUNK) {
            uint64_t chunk_len = std::min(RADIX_CHUNK, local_M - chunk_start);
            uint64_t total_dir = chunk_len * 2;
            if (bucketed_edges.size() < total_dir) bucketed_edges.resize(total_dir);

#if ENABLE_PROFILE
            double tpc0 = MPI_Wtime();
#endif
            std::fill(bcount.begin(), bcount.end(), 0);
            for (uint64_t k = 0; k < chunk_len; ++k) {
                uint64_t packed = packed_edges[chunk_start + k];
                uint32_t uk = static_cast<uint32_t>(packed >> 34);
                uint32_t vk = static_cast<uint32_t>((packed >> 8) & 0x3ffffffu);
                bcount[uk >> bucket_shift]++;
                bcount[vk >> bucket_shift]++;
            }

            boffset[0] = 0;
            for (uint32_t b = 1; b < NUM_BUCKETS; ++b) boffset[b] = boffset[b - 1] + bcount[b - 1];
            std::memcpy(bcurr.data(), boffset.data(), NUM_BUCKETS * sizeof(uint32_t));
#if ENABLE_PROFILE
            timers.t_p2_count += (MPI_Wtime() - tpc0);
            double tps0 = MPI_Wtime();
#endif

            for (uint64_t k = 0; k < chunk_len; ++k) {
                uint64_t packed = packed_edges[chunk_start + k];
                uint32_t uk = static_cast<uint32_t>(packed >> 34);
                uint32_t vk = static_cast<uint32_t>((packed >> 8) & 0x3ffffffu);
                uint8_t  wk = static_cast<uint8_t>(packed);

                bucketed_edges[bcurr[uk >> bucket_shift]++] = (static_cast<uint64_t>(uk) << 34) |
                                                              (static_cast<uint64_t>(vk) << 8) | wk;
                bucketed_edges[bcurr[vk >> bucket_shift]++] = (static_cast<uint64_t>(vk) << 34) |
                                                              (static_cast<uint64_t>(uk) << 8) | wk;
            }
#if ENABLE_PROFILE
            timers.t_p2_scatter += (MPI_Wtime() - tps0);
            double tpf0 = MPI_Wtime();
#endif

            for (uint32_t b = 0; b < NUM_BUCKETS; ++b) {
                uint32_t b_start = boffset[b];
                uint32_t b_end   = (b == NUM_BUCKETS - 1) ? static_cast<uint32_t>(total_dir) : boffset[b + 1];
                for (uint32_t i = b_start; i < b_end; ++i) {
                    uint64_t item = bucketed_edges[i];
                    uint32_t u = static_cast<uint32_t>(item >> 34);
                    uint32_t v = static_cast<uint32_t>((item >> 8) & 0x3ffffffu);
                    uint8_t  w = static_cast<uint8_t>(item);

                    uint64_t pu = my_degree[u]++;
                    adj[pu] = {v, w};
                }
            }
#if ENABLE_PROFILE
            timers.t_p2_fill += (MPI_Wtime() - tpf0);
#endif
        }
    } else {
#if ENABLE_PROFILE
        double tpf0 = MPI_Wtime();
#endif
        for (uint64_t k = 0; k < local_M; k++) {
            uint64_t packed = packed_edges[k];
            uint32_t uk = static_cast<uint32_t>(packed >> 34);
            uint32_t vk = static_cast<uint32_t>((packed >> 8) & 0x3ffffffu);
            uint8_t wk  = static_cast<uint8_t>(packed);

            uint64_t pu = my_degree[uk]++;
            adj[pu] = {vk, wk};

            uint64_t pv = my_degree[vk]++;
            adj[pv] = {uk, wk};
        }
#if ENABLE_PROFILE
        timers.t_p2_fill = MPI_Wtime() - tpf0;
#endif
    }
#else
    for (uint64_t i = edge_begin; i < edge_end; ++i) {
        uint32_t uk, vk;
        uint8_t wk;
        gctx.generate(i, uk, vk, wk);

        uint64_t pu = my_degree[uk]++;
        adj[pu] = {vk, wk};

        uint64_t pv = my_degree[vk]++;
        adj[pv] = {uk, wk};
    }
#endif

#if ENABLE_PROFILE
    double t_p2_sync0 = MPI_Wtime();
#endif
    MPI_Win_sync(win);
    MPI_Barrier(shm_comm);
    MPI_Win_sync(win);
#if ENABLE_PROFILE
    timers.t_p2_sync = MPI_Wtime() - t_p2_sync0;
    timers.t_p2_total = MPI_Wtime() - t_p2_0;
#endif

#if CACHE_EDGES
    // Free packed_edges buffer before SSSP
    std::vector<uint64_t>().swap(packed_edges);
#endif

#elif PASS2_MODE == 1
    // Milestone 7.2A: Vertex-Owner Direct Scatter Pass 2
    int P = world_size;
    uint64_t total_adj = row[N];

    // 1. Degree-balanced boundaries
    std::vector<uint64_t> boundary(P + 1);
    boundary[0] = 0;
    boundary[P] = N;

    for (int r = 1; r < P; ++r) {
        uint64_t target = (total_adj * static_cast<uint64_t>(r)) / P;
        auto it = std::lower_bound(row, row + N + 1, target);
        uint64_t v = it - row;
        if (v > 0 && v <= N) {
            if (std::abs(static_cast<int64_t>(row[v]) - static_cast<int64_t>(target)) >
                std::abs(static_cast<int64_t>(row[v - 1]) - static_cast<int64_t>(target))) {
                v = v - 1;
            }
        }
        if (v < boundary[r - 1]) v = boundary[r - 1];
        if (v > N) v = N;
        boundary[r] = v;
    }

    // 2. Coarse Owner Lookup Table
    constexpr int COARSE_BITS = 14;
    const int shift = (p.scale > COARSE_BITS) ? (p.scale - COARSE_BITS) : 0;
    const uint64_t num_blocks = (N + (1ULL << shift) - 1) >> shift;
    std::vector<uint32_t> coarse_owner(num_blocks);

    for (uint64_t b = 0; b < num_blocks; ++b) {
        uint64_t v_start = b << shift;
        if (v_start >= N) v_start = N - 1;
        auto it = std::upper_bound(boundary.begin(), boundary.end(), v_start);
        int o = static_cast<int>(it - boundary.begin()) - 1;
        if (o < 0) o = 0;
        if (o >= P) o = P - 1;
        coarse_owner[b] = static_cast<uint32_t>(o);
    }

    auto get_owner = [&](uint32_t v) -> int {
        int o = coarse_owner[v >> shift];
        while (o > 0 && v < boundary[o]) o--;
        while (o + 1 < P && v >= boundary[o + 1]) o++;
        return o;
    };

    // 3. Initialize Cross-Round Local Cursor
    uint64_t local_V = boundary[world_rank + 1] - boundary[world_rank];
    std::vector<uint32_t> local_cursor(local_V);
    for (uint64_t lv = 0; lv < local_V; ++lv) {
        local_cursor[lv] = static_cast<uint32_t>(row[boundary[world_rank] + lv]);
    }

    // 4. Allocate Independent Mailbox Window
    constexpr uint64_t CHUNK_UNDIRECTED = 262144; // 256K edges per chunk
    const uint64_t mailbox_capacity = static_cast<uint64_t>(P) * 2 * CHUNK_UNDIRECTED;

    MPI_Win mailbox_win;
    uint64_t *mailbox = nullptr;
    MPI_Aint mailbox_bytes = (shm_rank == 0) ? static_cast<MPI_Aint>(mailbox_capacity * sizeof(uint64_t)) : 0;
    MPI_Win_allocate_shared(mailbox_bytes, sizeof(uint64_t), MPI_INFO_NULL, shm_comm, &mailbox, &mailbox_win);
    if (shm_rank != 0) {
        int disp_unit = 0;
        MPI_Aint sz = 0;
        MPI_Win_shared_query(mailbox_win, 0, &sz, &disp_unit, &mailbox);
    }

    // 5. Unified Round Calculation
    uint64_t max_local_M = 0;
    MPI_Allreduce(&local_M, &max_local_M, 1, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
    uint64_t total_rounds = (max_local_M + CHUNK_UNDIRECTED - 1) / CHUNK_UNDIRECTED;

    // Profiling variables
    double t_count = 0.0;
    double t_allgather = 0.0;
    double t_offsets = 0.0;
    double t_pack = 0.0;
    double t_prod_sync = 0.0;
    double t_owner_scatter = 0.0;
    double t_owner_sync = 0.0;

    double max_round_skew = 0.0;
    double sum_round_skew = 0.0;
    uint64_t worst_round_idx = 0;
    uint64_t max_owner_total_overall = 0;

    std::vector<uint64_t> my_counts(P);
    std::vector<uint64_t> global_counts(static_cast<size_t>(P) * P);
    std::vector<uint64_t> slice_offset(static_cast<size_t>(P) * P);
    std::vector<uint64_t> slice_cur(P);
    std::vector<uint64_t> owner_base(P);
    std::vector<uint64_t> owner_total(P);

    for (uint64_t rnd = 0; rnd < total_rounds; ++rnd) {
        uint64_t chunk_start = rnd * CHUNK_UNDIRECTED;
        uint64_t chunk_len = (chunk_start < local_M) ? std::min(CHUNK_UNDIRECTED, local_M - chunk_start) : 0;

        // Step 1: Producer Count Pass
#if ENABLE_PROFILE
        double t_c0 = MPI_Wtime();
#endif
        std::fill(my_counts.begin(), my_counts.end(), 0ULL);
        for (uint64_t k = 0; k < chunk_len; ++k) {
            uint64_t p_edge = packed_edges[chunk_start + k];
            uint32_t uk = static_cast<uint32_t>(p_edge >> 34);
            uint32_t vk = static_cast<uint32_t>((p_edge >> 8) & 0x3ffffffu);
            my_counts[get_owner(uk)]++;
            my_counts[get_owner(vk)]++;
        }
#if ENABLE_PROFILE
        t_count += (MPI_Wtime() - t_c0);
        double t_g0 = MPI_Wtime();
#endif

        // Step 2: Collective Exchange of Counts
        MPI_Allgather(my_counts.data(), P, MPI_UINT64_T,
                      global_counts.data(), P, MPI_UINT64_T, MPI_COMM_WORLD);
#if ENABLE_PROFILE
        t_allgather += (MPI_Wtime() - t_g0);
        double t_off0 = MPI_Wtime();
#endif

        // Step 3: Compute Owner-Major Offsets
        uint64_t cur_pos = 0;
        uint64_t cur_round_max_owner = 0;
        for (int o = 0; o < P; ++o) {
            owner_base[o] = cur_pos;
            uint64_t o_cur = cur_pos;
            for (int p_idx = 0; p_idx < P; ++p_idx) {
                slice_offset[p_idx * P + o] = o_cur;
                o_cur += global_counts[p_idx * P + o];
            }
            owner_total[o] = o_cur - cur_pos;
#if ENABLE_PROFILE
            if (owner_total[o] > cur_round_max_owner) cur_round_max_owner = owner_total[o];
#endif
            cur_pos = o_cur;
        }
#if ENABLE_PROFILE
        t_offsets += (MPI_Wtime() - t_off0);

        if (world_rank == 0) {
            uint64_t tot_records = cur_pos;
            double avg_owner = static_cast<double>(tot_records) / P;
            double skew = (avg_owner > 0) ? (static_cast<double>(cur_round_max_owner) / avg_owner) : 1.0;
            sum_round_skew += skew;
            if (skew > max_round_skew) {
                max_round_skew = skew;
                worst_round_idx = rnd;
            }
            if (cur_round_max_owner > max_owner_total_overall) {
                max_owner_total_overall = cur_round_max_owner;
            }
        }
#endif

#if ENABLE_DEBUG_INVARIANTS
        if (cur_pos > mailbox_capacity) {
            std::cerr << "Mailbox capacity overflow: " << cur_pos << " > " << mailbox_capacity << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
#endif

        // Step 4: Producer Packing Pass
#if ENABLE_PROFILE
        double t_p0 = MPI_Wtime();
#endif
        for (int o = 0; o < P; ++o) {
            slice_cur[o] = slice_offset[world_rank * P + o];
        }
        for (uint64_t k = 0; k < chunk_len; ++k) {
            uint64_t p_edge = packed_edges[chunk_start + k];
            uint32_t uk = static_cast<uint32_t>(p_edge >> 34);
            uint32_t vk = static_cast<uint32_t>((p_edge >> 8) & 0x3ffffffu);
            uint8_t  wk = static_cast<uint8_t>(p_edge);

            int o_u = get_owner(uk);
            int o_v = get_owner(vk);

            uint64_t rec_u = (static_cast<uint64_t>(uk) << 34) | (static_cast<uint64_t>(vk) << 8) | wk;
            uint64_t rec_v = (static_cast<uint64_t>(vk) << 34) | (static_cast<uint64_t>(uk) << 8) | wk;

            mailbox[slice_cur[o_u]++] = rec_u;
            mailbox[slice_cur[o_v]++] = rec_v;
        }
#if ENABLE_PROFILE
        t_pack += (MPI_Wtime() - t_p0);
#endif

#if ENABLE_DEBUG_INVARIANTS
        for (int o = 0; o < P; ++o) {
            if (slice_cur[o] != slice_offset[world_rank * P + o] + my_counts[o]) {
                std::cerr << "Producer slice mismatch for rank " << world_rank << " owner " << o << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
#endif

        // Step 5: Producer Sync Wait
#if ENABLE_PROFILE
        double t_ps0 = MPI_Wtime();
#endif
        MPI_Win_sync(mailbox_win);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Win_sync(mailbox_win);
#if ENABLE_PROFILE
        t_prod_sync += (MPI_Wtime() - t_ps0);
#endif

        // Step 6: Owner Sequential Read & CSR Direct Scatter
#if ENABLE_PROFILE
        double t_s0 = MPI_Wtime();
#endif
        uint64_t o_start = owner_base[world_rank];
        uint64_t o_cnt   = owner_total[world_rank];
        for (uint64_t i = 0; i < o_cnt; ++i) {
            uint64_t rec = mailbox[o_start + i];
            uint32_t src = static_cast<uint32_t>(rec >> 34);
            uint32_t dst = static_cast<uint32_t>((rec >> 8) & 0x3ffffffu);
            uint8_t  w   = static_cast<uint8_t>(rec);

#if ENABLE_DEBUG_INVARIANTS
            if (src < boundary[world_rank] || src >= boundary[world_rank + 1]) {
                std::cerr << "Rank " << world_rank << " received invalid src " << src
                          << " outside [" << boundary[world_rank] << ", " << boundary[world_rank + 1] << ")\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
#endif

            uint64_t local_src = src - boundary[world_rank];
            uint32_t pos = local_cursor[local_src]++;
            adj[pos] = {dst, w};
        }
#if ENABLE_PROFILE
        t_owner_scatter += (MPI_Wtime() - t_s0);
#endif

        // Step 7: Owner Sync Wait
#if ENABLE_PROFILE
        double t_os0 = MPI_Wtime();
#endif
        MPI_Win_sync(win);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Win_sync(mailbox_win);
#if ENABLE_PROFILE
        t_owner_sync += (MPI_Wtime() - t_os0);
#endif
    }

    // Free Mailbox Window
    MPI_Win_free(&mailbox_win);

#if CACHE_EDGES
    // Free packed_edges buffer before SSSP
    std::vector<uint64_t>().swap(packed_edges);
#endif

#if ENABLE_DEBUG_INVARIANTS
    uint64_t mismatch_count = 0;
    for (uint64_t lv = 0; lv < local_V; ++lv) {
        uint64_t v = boundary[world_rank] + lv;
        if (local_cursor[lv] != static_cast<uint32_t>(row[v + 1])) {
            mismatch_count++;
        }
    }
    uint64_t tot_mismatches = 0;
    MPI_Reduce(&mismatch_count, &tot_mismatches, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    if (world_rank == 0) {
        if (tot_mismatches > 0) {
            std::cerr << "FATAL: Total cursor mismatches = " << tot_mismatches << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        } else {
            std::printf("  [INVARIANT CHECK: 100%% of vertices CSR cursor perfectly matches row[v+1]!]\n");
        }
    }
#endif

#if ENABLE_PROFILE
    // Pass 2 timing breakdown
    double times_p2[7] = {t_count, t_allgather, t_offsets, t_pack, t_prod_sync, t_owner_scatter, t_owner_sync};
    double max_times_p2[7] = {0};
    MPI_Reduce(times_p2, max_times_p2, 7, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        double avg_skew = (total_rounds > 0) ? (sum_round_skew / total_rounds) : 1.0;
        std::printf("\n=================================================================================\n");
        std::printf("  [MILESTONE 7.2A: VERTEX-OWNER DIRECT SCATTER (Rounds: %llu, P: %d)]\n",
                    (unsigned long long)total_rounds, P);
        std::printf("---------------------------------------------------------------------------------\n");
        std::printf("  [Timing Breakdown (MPI_MAX across ranks)]\n");
        std::printf("  Producer Count:               %.4f s\n", max_times_p2[0]);
        std::printf("  Counts Allgather:             %.4f s\n", max_times_p2[1]);
        std::printf("  Offset Calculation:           %.4f s\n", max_times_p2[2]);
        std::printf("  Producer Pack:                %.4f s\n", max_times_p2[3]);
        std::printf("  Producer Sync Wait:           %.4f s\n", max_times_p2[4]);
        std::printf("  Owner Direct Scatter:         %.4f s\n", max_times_p2[5]);
        std::printf("  Owner Sync Wait:              %.4f s\n", max_times_p2[6]);
        std::printf("  Sum of Phase Maxima:          %.4f s\n",
                    max_times_p2[0] + max_times_p2[1] + max_times_p2[2] + max_times_p2[3] +
                    max_times_p2[4] + max_times_p2[5] + max_times_p2[6]);
        std::printf("---------------------------------------------------------------------------------\n");
        std::printf("  [Mailbox Skew Diagnostics]\n");
        std::printf("  Average Owner Skew per Round: %.4fx\n", avg_skew);
        std::printf("  Maximum Owner Skew across Rounds: %.4fx (Worst Round: %llu)\n",
                    max_round_skew, (unsigned long long)worst_round_idx);
        std::printf("  Maximum Owner Total in a Round:   %llu records\n",
                    (unsigned long long)max_owner_total_overall);
        std::printf("=================================================================================\n\n");
        std::fflush(stdout);
    }
#endif

#endif

#if PRINT_PASS2
    double t_pass2_print = MPI_Wtime() - t3;
    if (world_rank == 0) std::printf("Pass 2 time: %.4f s\n", t_pass2_print);
#endif
#if ENABLE_PROFILE
    double t4 = MPI_Wtime();
#endif

    // 4. SSSP Execution
    const std::vector<uint32_t> samples = sample_vertices(p);
    SSSPStats sssp_stats{};

    if (use_coop) {
        // Multi-Team Cooperative Mode
        int team_id = world_rank % team_count;
        MPI_Comm team_comm;
        MPI_Comm_split(MPI_COMM_WORLD, team_id, world_rank, &team_comm);
        int team_rank, team_size;
        MPI_Comm_rank(team_comm, &team_rank);
        MPI_Comm_size(team_comm, &team_size);

        uint32_t *t_dist = coop_dist + static_cast<uint64_t>(team_id) * N;
        uint32_t *t_frontier = coop_frontier + static_cast<uint64_t>(team_id) * N;
        uint64_t *t_work_idx = coop_work_index + (static_cast<uint64_t>(team_id) * 8);

        const uint64_t t_v_begin = (N * static_cast<uint64_t>(team_rank)) / team_size;
        const uint64_t t_v_end   = (N * static_cast<uint64_t>(team_rank + 1)) / team_size;

        for (size_t s = team_id; s < S; s += team_count) {
#if ENABLE_PROFILE
            double t_src_0 = MPI_Wtime();
#endif
#if DELTA_STEPPING > 1
            cooperative_delta_stepping(row, adj, N, sources[s], t_dist, t_frontier,
                                       t_work_idx, team_comm, team_rank, team_size, win, sssp_stats, DELTA_STEPPING);
#else
            cooperative_dial(row, adj, N, sources[s], t_dist, t_frontier,
                             t_work_idx, team_comm, team_rank, team_size, win, sssp_stats);
#endif
#if ENABLE_PROFILE
            double t_src_elapsed = MPI_Wtime() - t_src_0;
            double max_src_time = 0;
            MPI_Reduce(&t_src_elapsed, &max_src_time, 1, MPI_DOUBLE, MPI_MAX, 0, team_comm);
            if (team_rank == 0) {
                std::printf("[Team %d Source %zu (v=%u)] SSSP Time: %.4fs\n",
                            team_id, s, sources[s], max_src_time);
            }
#endif

            // Parallel compute summary statistics for this source
#if ENABLE_PROFILE
            double t_res_0 = MPI_Wtime();
#endif
            int64_t local_reached = 0;
            uint64_t local_checksum = 0;
            uint64_t local_digest1 = 0;
            uint64_t local_digest2 = 0;

            for (uint64_t vert = t_v_begin; vert < t_v_end; vert++) {
                if (t_dist[vert] != INF) {
                    local_reached++;
                    local_checksum += static_cast<uint64_t>(t_dist[vert]) * (vert + 1);
                    local_digest1 += distance_digest(static_cast<uint32_t>(vert), t_dist[vert], 0);
                    local_digest2 += distance_digest(static_cast<uint32_t>(vert), t_dist[vert], 1);
                }
            }

            int64_t tot_reached = 0;
            uint64_t tot_checksum = 0;
            uint64_t tot_digest1 = 0;
            uint64_t tot_digest2 = 0;

            MPI_Reduce(&local_reached, &tot_reached, 1, MPI_INT64_T, MPI_SUM, 0, team_comm);
            MPI_Reduce(&local_checksum, &tot_checksum, 1, MPI_UINT64_T, MPI_SUM, 0, team_comm);
            MPI_Reduce(&local_digest1, &tot_digest1, 1, MPI_UINT64_T, MPI_SUM, 0, team_comm);
            MPI_Reduce(&local_digest2, &tot_digest2, 1, MPI_UINT64_T, MPI_SUM, 0, team_comm);

            if (team_rank == 0) {
                res_hdr[s].n_reached = tot_reached;
                res_hdr[s].checksum  = tot_checksum;
                res_hdr[s].digest1   = tot_digest1;
                res_hdr[s].digest2   = tot_digest2;

                int32_t *my_sample = sample_dist_flat + s * p.nsample;
                for (int k = 0; k < p.nsample; k++) {
                    my_sample[k] = (t_dist[samples[k]] == INF) ? -1 : static_cast<int32_t>(t_dist[samples[k]]);
                }
            }

            MPI_Win_sync(win);
            MPI_Barrier(team_comm);
            MPI_Win_sync(win);
#if ENABLE_PROFILE
            sssp_stats.result_time += (MPI_Wtime() - t_res_0);
#endif
        }

        MPI_Comm_free(&team_comm);
    } else {
        // Standard Version 3 mode: source-level distribution
        std::vector<uint32_t> dist;
        DialQueue dial_queue;

        for (size_t s = world_rank; s < S; s += world_size) {
            dial_queue.clear();
#if ENABLE_PROFILE
            double t_d0 = MPI_Wtime();
#endif
            dial(row, adj, N, sources[s], dist, dial_queue);
#if ENABLE_PROFILE
            sssp_stats.relax_time += (MPI_Wtime() - t_d0);
            double t_res_0 = MPI_Wtime();
#endif
            FlatResultHeader hdr{};
            hdr.n_reached = 0;
            hdr.checksum = 0;
            hdr.digest1 = 0;
            hdr.digest2 = 0;

            for (uint64_t vert = 0; vert < N; vert++) {
                if (dist[vert] != INF) {
                    hdr.n_reached++;
                    hdr.checksum += static_cast<uint64_t>(dist[vert]) * (vert + 1);
                    hdr.digest1 += distance_digest(static_cast<uint32_t>(vert), dist[vert], 0);
                    hdr.digest2 += distance_digest(static_cast<uint32_t>(vert), dist[vert], 1);
                }
            }

            res_hdr[s] = hdr;

            int32_t *my_sample = sample_dist_flat + s * p.nsample;
            for (int k = 0; k < p.nsample; k++) {
                my_sample[k] = (dist[samples[k]] == INF) ? -1 : static_cast<int32_t>(dist[samples[k]]);
            }
#if ENABLE_PROFILE
            sssp_stats.result_time += (MPI_Wtime() - t_res_0);
#endif
        }
    }

    MPI_Win_sync(win);
    MPI_Barrier(shm_comm);
    MPI_Win_sync(win);

#if ENABLE_PROFILE
    timers.t_sssp_total = MPI_Wtime() - t4;
    timers.t_sssp_init = sssp_stats.init_time;
    timers.t_sssp_relax = sssp_stats.relax_time;
    timers.t_sssp_coll = sssp_stats.collective_time;
    timers.t_sssp_result = sssp_stats.result_time;
    double t5 = MPI_Wtime();
#endif

    // 5. Rank 0 copies shared results to output vector
    if (world_rank == 0) {
        for (size_t s = 0; s < S; s++) {
            results[s].n_reached = res_hdr[s].n_reached;
            results[s].checksum  = res_hdr[s].checksum;
            results[s].digest1   = res_hdr[s].digest1;
            results[s].digest2   = res_hdr[s].digest2;

            int32_t *src_sample = sample_dist_flat + s * p.nsample;
            for (int k = 0; k < p.nsample; k++) {
                results[s].sample_dist[k] = src_sample[k];
            }
        }
    }

#if ENABLE_PROFILE
    timers.t_copy = MPI_Wtime() - t5;
    MPI_Barrier(MPI_COMM_WORLD);
    timers.t_run_total = MPI_Wtime() - t_run_start;

    // Collect RoundSummary from remote teams to rank 0
    if (use_coop) {
        if (world_rank != 0 && world_rank < team_count) {
            uint64_t n_rounds = sssp_stats.round_summaries.size();
            MPI_Send(&n_rounds, 1, MPI_UINT64_T, 0, 0x1234, MPI_COMM_WORLD);
            if (n_rounds > 0) {
                MPI_Send(sssp_stats.round_summaries.data(),
                         n_rounds * sizeof(RoundSummary),
                         MPI_BYTE, 0, 0x1235, MPI_COMM_WORLD);
            }
        } else if (world_rank == 0) {
            for (int r = 1; r < team_count; ++r) {
                uint64_t n_rounds = 0;
                MPI_Recv(&n_rounds, 1, MPI_UINT64_T, r, 0x1234, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (n_rounds > 0) {
                    std::vector<RoundSummary> r_data(n_rounds);
                    MPI_Recv(r_data.data(),
                             n_rounds * sizeof(RoundSummary),
                             MPI_BYTE, r, 0x1235, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    sssp_stats.round_summaries.insert(
                        sssp_stats.round_summaries.end(),
                        r_data.begin(), r_data.end());
                }
            }
        }
    }

    constexpr int NUM_TIMERS = 22;
    static const char *TIMER_NAMES[NUM_TIMERS] = {
        "1. Shared Alloc",
        "2. Pass 1 Total",
        "  - Edge Generation",
        "  - Store Packed & Deg",
        "  - Pass 1 Barrier/Sync",
        "3. Prefix Sum Total",
        "  - Degree Aggregation",
        "  - Exscan (Row Base)",
        "  - Absolute Cursor Calc",
        "  - Prefix Barrier/Sync",
        "4. Pass 2 Total",
        "  - Radix Bucket Count",
        "  - Radix Scatter",
        "  - CSR Scatter Fill",
        "  - Pass 2 Barrier/Sync",
        "5. SSSP Total",
        "  - Dist Init & Reset",
        "  - Adjacency Scan & Relax",
        "  - Collectives & Barrier",
        "  - Result Aggregation",
        "6. Shared Output Copy",
        "Actual Run Total"
    };

    double local_arr[NUM_TIMERS];
    timers.to_array(local_arr);

    double max_arr[NUM_TIMERS] = {0};
    double min_arr[NUM_TIMERS] = {0};
    double sum_arr[NUM_TIMERS] = {0};

    MPI_Reduce(local_arr, max_arr, NUM_TIMERS, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_arr, min_arr, NUM_TIMERS, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_arr, sum_arr, NUM_TIMERS, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        double actual_max_total = max_arr[21];
        double phase_max_sum = max_arr[0] + max_arr[1] + max_arr[5] + max_arr[10] + max_arr[15] + max_arr[20];
        double build_max_total = max_arr[1] + max_arr[5] + max_arr[10];
        double sssp_max_total = max_arr[15];
        double build_pct = (actual_max_total > 1e-9) ? (build_max_total / actual_max_total * 100.0) : 0.0;
        double sssp_pct = (actual_max_total > 1e-9) ? (sssp_max_total / actual_max_total * 100.0) : 0.0;

        std::printf("\n========================================================================================\n");
        std::printf("                     PHASE 0: 24-RANK ACCURATE PROFILE BREAKDOWN                       \n");
        std::printf("========================================================================================\n");
        std::printf("%-30s | %9s | %9s | %9s | %8s | %10s\n",
                    "Phase Name", "Min (s)", "Max (s)", "Avg (s)", "Max/Avg", "% Actual Tot");
        std::printf("-------------------------------+-----------+-----------+-----------+----------+-----------\n");
        for (int i = 0; i < NUM_TIMERS - 1; ++i) {
            double avg = sum_arr[i] / world_size;
            double ratio = (avg > 1e-9) ? (max_arr[i] / avg) : 1.0;
            double pct = (actual_max_total > 1e-9) ? (max_arr[i] / actual_max_total * 100.0) : 0.0;
            std::printf("%-30s | %9.4f | %9.4f | %9.4f | %7.2fx | %9.2f%%\n",
                        TIMER_NAMES[i], min_arr[i], max_arr[i], avg, ratio, pct);
            if (i == 0 || i == 4 || i == 9 || i == 14 || i == 19 || i == 20) {
                std::printf("-------------------------------+-----------+-----------+-----------+----------+-----------\n");
            }
        }
        double avg_tot = sum_arr[21] / world_size;
        double ratio_tot = (avg_tot > 1e-9) ? (actual_max_total / avg_tot) : 1.0;
        std::printf("%-30s | %9.4f | %9.4f | %9.4f | %7.2fx | %9.2f%%\n",
                    "ACTUAL RUN TOTAL", min_arr[21], actual_max_total, avg_tot, ratio_tot, 100.0);
        std::printf("========================================================================================\n");
        std::printf("Phase-Max Sum: %.4f s | Actual Max Total: %.4f s | Phase Sync Skew: %.4f s\n",
                    phase_max_sum, actual_max_total, phase_max_sum - actual_max_total);
        std::printf("Build Total (Pass1+Prefix+Pass2): %.4f s (%.2f%% of Actual Total)\n",
                    build_max_total, build_pct);
        std::printf("SSSP Total (Init+Relax+Coll+Res):  %.4f s (%.2f%% of Actual Total)\n",
                    sssp_max_total, sssp_pct);
        std::printf("========================================================================================\n\n");

        // PHASE 1: Degree-Aware Work-Imbalance Profiler
        if (!sssp_stats.round_summaries.empty()) {
            auto &rounds = sssp_stats.round_summaries;
            std::vector<double> edge_imb_vec;
            std::vector<double> time_imb_vec;
            std::vector<double> skew_vec;
            uint64_t total_frontier_verts = 0;
            uint64_t total_frontier_deg = 0;
            uint64_t global_max_deg = 0;
            uint32_t tot_hubs[4] = {0, 0, 0, 0};
            uint64_t tot_hub_edges[4] = {0, 0, 0, 0};

            for (const auto &r : rounds) {
                if (r.frontier_size == 0) continue;
                edge_imb_vec.push_back(r.edge_imbalance_ratio);
                time_imb_vec.push_back(r.time_imbalance_ratio);
                skew_vec.push_back(r.barrier_skew);
                total_frontier_verts += r.frontier_size;
                total_frontier_deg += r.total_degree;
                if (r.max_degree > global_max_deg) global_max_deg = r.max_degree;
                tot_hubs[0] += r.hubs_gt_1k;
                tot_hubs[1] += r.hubs_gt_4k;
                tot_hubs[2] += r.hubs_gt_16k;
                tot_hubs[3] += r.hubs_gt_64k;
                tot_hub_edges[0] += r.edges_gt_1k;
                tot_hub_edges[1] += r.edges_gt_4k;
                tot_hub_edges[2] += r.edges_gt_16k;
                tot_hub_edges[3] += r.edges_gt_64k;
            }

            std::sort(edge_imb_vec.begin(), edge_imb_vec.end());
            std::sort(time_imb_vec.begin(), time_imb_vec.end());
            std::sort(skew_vec.begin(), skew_vec.end());

            size_t valid_n = edge_imb_vec.size();
            double med_edge_imb = valid_n ? edge_imb_vec[valid_n / 2] : 1.0;
            double p95_edge_imb = valid_n ? edge_imb_vec[static_cast<size_t>(valid_n * 0.95)] : 1.0;
            double max_edge_imb = valid_n ? edge_imb_vec.back() : 1.0;

            double med_time_imb = valid_n ? time_imb_vec[valid_n / 2] : 1.0;
            double p95_time_imb = valid_n ? time_imb_vec[static_cast<size_t>(valid_n * 0.95)] : 1.0;
            double max_time_imb = valid_n ? time_imb_vec.back() : 1.0;

            double med_skew_ms = valid_n ? skew_vec[valid_n / 2] * 1000.0 : 0.0;
            double p95_skew_ms = valid_n ? skew_vec[static_cast<size_t>(valid_n * 0.95)] * 1000.0 : 0.0;
            double max_skew_ms = valid_n ? skew_vec.back() * 1000.0 : 0.0;

            std::printf("========================================================================================\n");
            std::printf("             PHASE 1: DEGREE-AWARE WORK-IMBALANCE PROFILER REPORT                      \n");
            std::printf("========================================================================================\n");
            std::printf("[Global Frontier & Hub Distribution]\n");
            std::printf("  Total Distance Rounds Analyzed: %zu\n", valid_n);
            std::printf("  Total Frontier Vertices:        %llu\n", (unsigned long long)total_frontier_verts);
            std::printf("  Total Frontier Degree (Edges):  %llu\n", (unsigned long long)total_frontier_deg);
            std::printf("  Max Vertex Degree Seen:         %llu\n", (unsigned long long)global_max_deg);
            double pct_deg = total_frontier_deg > 0 ? (100.0 / total_frontier_deg) : 0.0;
            std::printf("  Hub Vertices (> 1K):  %-8u (Workload: %12llu edges, %6.2f%% of total)\n",
                        tot_hubs[0], (unsigned long long)tot_hub_edges[0], tot_hub_edges[0] * pct_deg);
            std::printf("  Hub Vertices (> 4K):  %-8u (Workload: %12llu edges, %6.2f%% of total)\n",
                        tot_hubs[1], (unsigned long long)tot_hub_edges[1], tot_hub_edges[1] * pct_deg);
            std::printf("  Hub Vertices (>16K):  %-8u (Workload: %12llu edges, %6.2f%% of total)\n",
                        tot_hubs[2], (unsigned long long)tot_hub_edges[2], tot_hub_edges[2] * pct_deg);
            std::printf("  Hub Vertices (>64K):  %-8u (Workload: %12llu edges, %6.2f%% of total)\n",
                        tot_hubs[3], (unsigned long long)tot_hub_edges[3], tot_hub_edges[3] * pct_deg);
            std::printf("\n[Work Imbalance & Skew Across Rounds]\n");
            std::printf("  Edge Imbalance Ratio (Max/Avg): Median: %6.2fx | P95: %6.2fx | Max: %6.2fx\n",
                        med_edge_imb, p95_edge_imb, max_edge_imb);
            std::printf("  Time Imbalance Ratio (Max/Avg): Median: %6.2fx | P95: %6.2fx | Max: %6.2fx\n",
                        med_time_imb, p95_time_imb, max_time_imb);
            std::printf("  Barrier / Wait Skew:            Median: %6.2fms | P95: %6.2fms | Max: %6.2fms\n",
                        med_skew_ms, p95_skew_ms, max_skew_ms);

            // Heavy rounds analysis: filter rounds accounting for top 80% of total relaxation time
            std::vector<RoundSummary> time_sorted_rounds = rounds;
            std::sort(time_sorted_rounds.begin(), time_sorted_rounds.end(), [](const RoundSummary &a, const RoundSummary &b) {
                return a.max_relax_time > b.max_relax_time;
            });

            double total_relax_wall = 0.0;
            for (const auto &r : time_sorted_rounds) total_relax_wall += r.max_relax_time;

            double accum_wall = 0.0;
            std::vector<double> heavy_edge_imb;
            std::vector<double> heavy_time_imb;
            std::vector<double> heavy_skew;

            for (const auto &r : time_sorted_rounds) {
                if (r.frontier_size == 0) continue;
                accum_wall += r.max_relax_time;
                heavy_edge_imb.push_back(r.edge_imbalance_ratio);
                heavy_time_imb.push_back(r.time_imbalance_ratio);
                heavy_skew.push_back(r.barrier_skew);
                if (accum_wall >= 0.80 * total_relax_wall) break;
            }

            std::sort(heavy_edge_imb.begin(), heavy_edge_imb.end());
            std::sort(heavy_time_imb.begin(), heavy_time_imb.end());
            std::sort(heavy_skew.begin(), heavy_skew.end());

            size_t heavy_n = heavy_edge_imb.size();
            double h_med_edge = heavy_n ? heavy_edge_imb[heavy_n / 2] : 1.0;
            double h_p95_edge = heavy_n ? heavy_edge_imb[static_cast<size_t>(heavy_n * 0.95)] : 1.0;
            double h_max_edge = heavy_n ? heavy_edge_imb.back() : 1.0;

            double h_med_time = heavy_n ? heavy_time_imb[heavy_n / 2] : 1.0;
            double h_p95_time = heavy_n ? heavy_time_imb[static_cast<size_t>(heavy_n * 0.95)] : 1.0;
            double h_max_time = heavy_n ? heavy_time_imb.back() : 1.0;

            double h_med_skew_ms = heavy_n ? heavy_skew[heavy_n / 2] * 1000.0 : 0.0;
            double h_p95_skew_ms = heavy_n ? heavy_skew[static_cast<size_t>(heavy_n * 0.95)] * 1000.0 : 0.0;
            double h_max_skew_ms = heavy_n ? heavy_skew.back() * 1000.0 : 0.0;

            std::printf("\n[Heavy Rounds Work Imbalance (Top 80%% SSSP Relaxation Time, %zu Rounds)]\n", heavy_n);
            std::printf("  Edge Imbalance Ratio (Max/Avg): Median: %6.2fx | P95: %6.2fx | Max: %6.2fx\n",
                        h_med_edge, h_p95_edge, h_max_edge);
            std::printf("  Time Imbalance Ratio (Max/Avg): Median: %6.2fx | P95: %6.2fx | Max: %6.2fx\n",
                        h_med_time, h_p95_time, h_max_time);
            std::printf("  Barrier / Wait Skew:            Median: %6.2fms | P95: %6.2fms | Max: %6.2fms\n",
                        h_med_skew_ms, h_p95_skew_ms, h_max_skew_ms);

            std::printf("\n[Top 20 Most Time-Consuming Rounds (Heavy SSSP Rounds)]\n");
            std::printf("%-4s | %-4s | %-5s | %-10s | %-8s | %-10s | %-10s | %-9s | %-7s | %-9s | %-7s | %-8s\n",
                        "Rank", "Src", "Dist", "Frontier", "MaxDeg", "TotDeg", "MaxRankE", "AvgRankE", "EdgeImb", "MaxT(ms)", "TimeImb", "Skew(ms)");
            std::printf("-----+------+-------+------------+----------+------------+------------+-----------+---------+-----------+---------+---------\n");
            size_t show_heavy = std::min<size_t>(20, time_sorted_rounds.size());
            for (size_t i = 0; i < show_heavy; ++i) {
                const auto &r = time_sorted_rounds[i];
                std::printf("%-4zu | %-4u | %-5u | %-10llu | %-8llu | %-10llu | %-10llu | %-9.1f | %6.2fx  | %8.2f  | %6.2fx  | %7.2f\n",
                            i + 1, r.src_idx, r.distance,
                            (unsigned long long)r.frontier_size,
                            (unsigned long long)r.max_degree,
                            (unsigned long long)r.total_degree,
                            (unsigned long long)r.max_scanned_edges,
                            r.avg_scanned_edges,
                            r.edge_imbalance_ratio,
                            r.max_relax_time * 1000.0,
                            r.time_imbalance_ratio,
                            r.barrier_skew * 1000.0);
            }

            // Sort rounds by edge_imbalance_ratio descending
            std::vector<RoundSummary> sorted_rounds = rounds;
            std::sort(sorted_rounds.begin(), sorted_rounds.end(), [](const RoundSummary &a, const RoundSummary &b) {
                return a.edge_imbalance_ratio > b.edge_imbalance_ratio;
            });

            std::printf("\n[Top 20 Most Imbalanced Rounds (by Edge Imbalance)]\n");
            std::printf("%-4s | %-4s | %-5s | %-10s | %-8s | %-10s | %-10s | %-9s | %-7s | %-9s | %-7s | %-8s\n",
                        "Rank", "Src", "Dist", "Frontier", "MaxDeg", "TotDeg", "MaxRankE", "AvgRankE", "EdgeImb", "MaxT(ms)", "TimeImb", "Skew(ms)");
            std::printf("-----+------+-------+------------+----------+------------+------------+-----------+---------+-----------+---------+---------\n");
            size_t show_count = std::min<size_t>(20, sorted_rounds.size());
            for (size_t i = 0; i < show_count; ++i) {
                const auto &r = sorted_rounds[i];
                std::printf("%-4zu | %-4u | %-5u | %-10llu | %-8llu | %-10llu | %-10llu | %-9.1f | %6.2fx  | %8.2f  | %6.2fx  | %7.2f\n",
                            i + 1, r.src_idx, r.distance,
                            (unsigned long long)r.frontier_size,
                            (unsigned long long)r.max_degree,
                            (unsigned long long)r.total_degree,
                            (unsigned long long)r.max_scanned_edges,
                            r.avg_scanned_edges,
                            r.edge_imbalance_ratio,
                            r.max_relax_time * 1000.0,
                            r.time_imbalance_ratio,
                            r.barrier_skew * 1000.0);
            }
            std::printf("========================================================================================\n");

            // Evaluate Criteria A & B
            bool crit_a = (sssp_pct >= 40.0);
            bool crit_b = (h_p95_edge >= 2.0 && h_max_skew_ms >= 5.0);
            std::printf("[Phase 1 Continuation Criteria Evaluation]\n");
            std::printf("  Criteria A (SSSP >= 40%% of Actual Total): %s (SSSP is %.2f%% of total)\n",
                        crit_a ? "PASS" : "FAIL", sssp_pct);
            std::printf("  Criteria B (Heavy Rounds P95 Edge Imbalance >= 2.0x & Skew >= 5ms): %s (P95 is %.2fx, Max Skew is %.2fms)\n",
                        crit_b ? "PASS" : "FAIL", h_p95_edge, h_max_skew_ms);
            if (crit_a && crit_b) {
                std::printf("  Decision: Hub Splitting is JUSTIFIED and RECOMMENDED for next phase.\n");
            } else if (!crit_a) {
                std::printf("  Decision: SSSP does NOT dominate run time. Graph Build dominates (%.2f%%). Next phase must optimize Build.\n", build_pct);
            } else {
                std::printf("  Decision: Edge imbalance in heavy rounds is mild (P95=%.2fx). Fine-grained chunking or other SSSP/Build optimizations recommended.\n", h_p95_edge);
            }
            std::printf("========================================================================================\n\n");
        }
    }
#endif

    // 6. Clean up MPI resources
    MPI_Win_free(&win);
    MPI_Comm_free(&shm_comm);
}
