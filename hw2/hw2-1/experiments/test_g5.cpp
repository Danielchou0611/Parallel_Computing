#include <iostream>
#include <vector>
#include <chrono>
#include <cstdint>
#include <algorithm>
#include <iomanip>
#include <cassert>
#include <random>
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

    // G0: Scalar 1-way
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
        // Track A (0..3) and Track B (4..7) initial mix
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

    // 8-way -> 4-way -> scalar tail dispatcher
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

    // 4-way -> scalar tail dispatcher (G4)
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

    // Scalar baseline dispatcher (G0)
    inline void generate_range_g0(uint64_t begin, uint64_t end, uint32_t *u, uint32_t *v, uint8_t *w) const {
        uint64_t idx = 0;
        for (uint64_t i = begin; i < end; ++i, ++idx) {
            generate1(i, u[idx], v[idx], w[idx]);
        }
    }
};

// ====================================================================
// G5.1 Correctness Validation Suite
// ====================================================================
void run_g5_correctness_tests() {
    std::cout << "=================================================================\n";
    std::cout << "  [MILESTONE G5.1: EXTENSIVE BITWISE CORRECTNESS VALIDATION]\n";
    std::cout << "=================================================================\n";

    // 1. Critical scale boundaries and comprehensive scales
    std::vector<int> scales = {
        1, 2, 3, 4, 5, 6,
        7, 8, 9,           // Scale 8 boundary
        10, 14,
        15, 16, 17,        // Scale 16 boundary
        20, 22, 23,
        24, 25, 26         // Scale 24, 25, 26 boundary
    };

    std::vector<uint64_t> seeds = {
        0ULL, 1ULL, 110ULL, 0x123456789abcdef0ULL,
        0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL
    };

    uint64_t total_tests_checked = 0;

    for (int sc : scales) {
        for (int scramble : {0, 1}) {
            for (uint64_t seed : seeds) {
                FastGenG5 gen;
                gen.init(sc, scramble, seed);

                GraphParams p;
                p.scale = sc;
                p.edgefactor = 16;
                p.seed = seed;
                p.scramble = scramble;
                p.nsample = 1000;
                uint64_t M = num_edges(p);

                // A. Test arbitrary begin offsets and non-divisible range lengths
                std::vector<uint64_t> test_begins = {0, 1, 3, 7, 13, 27, 63, 101};
                std::vector<uint64_t> test_lens   = {1, 2, 3, 4, 5, 7, 8, 9, 11, 13, 15, 16, 17, 31, 33, 67, 127};

                for (uint64_t beg : test_begins) {
                    for (uint64_t len : test_lens) {
                        if (beg + len > M) continue;

                        std::vector<uint32_t> u0(len), v0(len); std::vector<uint8_t> w0(len);
                        std::vector<uint32_t> u4(len), v4(len); std::vector<uint8_t> w4(len);
                        std::vector<uint32_t> u5(len), v5(len); std::vector<uint8_t> w5(len);

                        gen.generate_range_g0(beg, beg + len, u0.data(), v0.data(), w0.data());
                        gen.generate_range_g4(beg, beg + len, u4.data(), v4.data(), w4.data());
                        gen.generate_range_g5(beg, beg + len, u5.data(), v5.data(), w5.data());

                        for (uint64_t k = 0; k < len; ++k) {
                            if (u0[k] != u4[k] || v0[k] != v4[k] || w0[k] != w4[k] ||
                                u0[k] != u5[k] || v0[k] != v5[k] || w0[k] != w5[k]) {
                                std::cerr << "FAIL on scale=" << sc << " scr=" << scramble << " seed=" << seed
                                          << " beg=" << beg << " k=" << k << " edge=" << (beg + k) << "\n"
                                          << "  G0: (" << u0[k] << ", " << v0[k] << ", " << (int)w0[k] << ")\n"
                                          << "  G4: (" << u4[k] << ", " << v4[k] << ", " << (int)w4[k] << ")\n"
                                          << "  G5: (" << u5[k] << ", " << v5[k] << ", " << (int)w5[k] << ")\n";
                                std::exit(1);
                            }
                        }
                        total_tests_checked += len;
                    }
                }

                // B. Edge index M-1 boundary test (last 16 edges)
                if (M >= 16) {
                    uint64_t beg = M - 16;
                    uint64_t len = 16;
                    std::vector<uint32_t> u0(len), v0(len); std::vector<uint8_t> w0(len);
                    std::vector<uint32_t> u5(len), v5(len); std::vector<uint8_t> w5(len);
                    gen.generate_range_g0(beg, beg + len, u0.data(), v0.data(), w0.data());
                    gen.generate_range_g5(beg, beg + len, u5.data(), v5.data(), w5.data());
                    for (uint64_t k = 0; k < len; ++k) {
                        assert(u0[k] == u5[k] && v0[k] == v5[k] && w0[k] == w5[k]);
                    }
                    total_tests_checked += len;
                }
            }
        }
    }

    std::cout << ">> G5.1 SUCCESS: Checked " << total_tests_checked 
              << " edges across all boundary scales, seeds, arbitrary begins, non-multiple lengths!\n"
              << ">> 100% BIT-FOR-BIT IDENTICAL: G0 == G4 == G5! <<\n\n";
}

// ====================================================================
// G5.2 Microbenchmark Suite (10M edges, warmup + 5 runs median)
// ====================================================================
void run_g5_microbenchmark() {
    std::cout << "=================================================================\n";
    std::cout << "  [MILESTONE G5.2: MICROBENCHMARK (10,000,000 EDGES, 5 RUNS MEDIAN)]\n";
    std::cout << "=================================================================\n";

    struct TestCase {
        std::string name;
        int scale;
        int scramble;
        uint64_t seed;
    };
    std::vector<TestCase> cases = {
        {"Case 09 (Scale 25, Scr 0)", 25, 0, 109},
        {"Case 10 (Scale 26, Scr 1)", 26, 1, 110}
    };

    const uint64_t N_EDGES = 10000000ULL;
    const int RUNS = 5;

    std::vector<uint32_t> u(N_EDGES), v(N_EDGES);
    std::vector<uint8_t>  w(N_EDGES);

    for (const auto &tc : cases) {
        FastGenG5 gen;
        gen.init(tc.scale, tc.scramble, tc.seed);

        std::cout << "--- " << tc.name << " ---\n";

        // Warmup runs
        gen.generate_range_g4(0, N_EDGES, u.data(), v.data(), w.data());
        gen.generate_range_g5(0, N_EDGES, u.data(), v.data(), w.data());

        // Benchmark G4 (5 runs)
        std::vector<double> g4_times(RUNS);
        for (int r = 0; r < RUNS; ++r) {
            auto t0 = std::chrono::high_resolution_clock::now();
            gen.generate_range_g4(0, N_EDGES, u.data(), v.data(), w.data());
            auto t1 = std::chrono::high_resolution_clock::now();
            g4_times[r] = std::chrono::duration<double>(t1 - t0).count();
        }
        std::sort(g4_times.begin(), g4_times.end());
        double g4_median = g4_times[RUNS / 2];

        // Benchmark G5 (5 runs)
        std::vector<double> g5_times(RUNS);
        for (int r = 0; r < RUNS; ++r) {
            auto t0 = std::chrono::high_resolution_clock::now();
            gen.generate_range_g5(0, N_EDGES, u.data(), v.data(), w.data());
            auto t1 = std::chrono::high_resolution_clock::now();
            g5_times[r] = std::chrono::duration<double>(t1 - t0).count();
        }
        std::sort(g5_times.begin(), g5_times.end());
        double g5_median = g5_times[RUNS / 2];

        double speedup = (g4_median - g5_median) / g4_median * 100.0;
        double ratio   = g4_median / g5_median;

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  G4 (4-way AVX2) Median: " << g4_median << " s (" 
                  << (N_EDGES / g4_median / 1e6) << " M edges/s)\n";
        std::cout << "  G5 (8-way AVX2) Median: " << g5_median << " s (" 
                  << (N_EDGES / g5_median / 1e6) << " M edges/s)\n";
        std::cout << "  Speedup: " << ratio << "x (" << (speedup >= 0 ? "+" : "") << speedup << "%)\n";

        if (speedup >= 15.0) {
            std::cout << "  => [THRESHOLD PASSED]: G5 improves by >= 15%!\n";
        } else {
            std::cout << "  => [THRESHOLD NOT MET]: G5 improvement is " << speedup << "% (< 15% required)\n";
        }
    }
}

int main() {
    run_g5_correctness_tests();
    run_g5_microbenchmark();
    return 0;
}
