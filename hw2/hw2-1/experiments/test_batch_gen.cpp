#include <iostream>
#include <vector>
#include <chrono>
#include <cstdint>
#include <algorithm>
#include <iomanip>
#include <cassert>
#include <immintrin.h>
#include "graph_gen.hpp"

inline uint64_t mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

struct FastGenBase {
    uint64_t mixed_seed;
    uint64_t mixed_seed_w;
    uint64_t scramble_k1, scramble_k2, scramble_k3;
    uint64_t scramble_mask;
    int scramble_sh;
    int scale;
    bool scramble;

    void init(const GraphParams &p) {
        scale = p.scale;
        scramble = p.scramble;
        mixed_seed = mix(p.seed);
        mixed_seed_w = mix(p.seed ^ 0x77e19470ULL);

        uint64_t s_seed = p.seed ^ 0x5c2a3b1eULL;
        scramble_mask = (uint64_t(1) << scale) - 1;
        scramble_sh = scale / 2 > 0 ? scale / 2 : 1;
        scramble_k1 = mix(s_seed ^ 0x1111ULL) | 1;
        scramble_k2 = mix(s_seed ^ 0x2222ULL) | 1;
        scramble_k3 = mix(s_seed ^ 0x3333ULL) | 1;
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

    // 1-way scalar baseline
    inline void generate1(uint64_t i, uint32_t &u, uint32_t &v, uint8_t &w) const {
        uint64_t state = mix(mixed_seed ^ (i * 0xd6e8feb86659fd93ULL));
        uint64_t bits = state;
        uint32_t a = 0, b = 0;

        for (int level = 0; level < scale; level++) {
            if (level && (level & 7) == 0) {
                state = mix(state);
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
        uint64_t h = mix(mixed_seed_w ^ (pair_key * 0xd6e8feb86659fd93ULL));
        w = static_cast<uint8_t>(1 + (h % 255));
    }

    // 4-way independent scalar (Milestone G3)
    inline void generate4_scalar(uint64_t i, uint32_t *u, uint32_t *v, uint8_t *w) const {
        uint64_t s0 = mix(mixed_seed ^ ((i + 0) * 0xd6e8feb86659fd93ULL));
        uint64_t s1 = mix(mixed_seed ^ ((i + 1) * 0xd6e8feb86659fd93ULL));
        uint64_t s2 = mix(mixed_seed ^ ((i + 2) * 0xd6e8feb86659fd93ULL));
        uint64_t s3 = mix(mixed_seed ^ ((i + 3) * 0xd6e8feb86659fd93ULL));

        uint64_t bits0 = s0, bits1 = s1, bits2 = s2, bits3 = s3;
        uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        uint32_t b0 = 0, b1 = 0, b2 = 0, b3 = 0;

        for (int level = 0; level < scale; level++) {
            if (level && (level & 7) == 0) {
                s0 = mix(s0); bits0 = s0;
                s1 = mix(s1); bits1 = s1;
                s2 = mix(s2); bits2 = s2;
                s3 = mix(s3); bits3 = s3;
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

        uint64_t h0 = mix(mixed_seed_w ^ (((hi0 << 32) | lo0) * 0xd6e8feb86659fd93ULL));
        uint64_t h1 = mix(mixed_seed_w ^ (((hi1 << 32) | lo1) * 0xd6e8feb86659fd93ULL));
        uint64_t h2 = mix(mixed_seed_w ^ (((hi2 << 32) | lo2) * 0xd6e8feb86659fd93ULL));
        uint64_t h3 = mix(mixed_seed_w ^ (((hi3 << 32) | lo3) * 0xd6e8feb86659fd93ULL));

        w[0] = static_cast<uint8_t>(1 + (h0 % 255));
        w[1] = static_cast<uint8_t>(1 + (h1 % 255));
        w[2] = static_cast<uint8_t>(1 + (h2 % 255));
        w[3] = static_cast<uint8_t>(1 + (h3 % 255));
    }

    // 4-way AVX2 Block Descent (Milestone G4)
    inline void generate4_avx2(uint64_t i, uint32_t *u, uint32_t *v, uint8_t *w) const {
        uint64_t s0 = mix(mixed_seed ^ ((i + 0) * 0xd6e8feb86659fd93ULL));
        uint64_t s1 = mix(mixed_seed ^ ((i + 1) * 0xd6e8feb86659fd93ULL));
        uint64_t s2 = mix(mixed_seed ^ ((i + 2) * 0xd6e8feb86659fd93ULL));
        uint64_t s3 = mix(mixed_seed ^ ((i + 3) * 0xd6e8feb86659fd93ULL));

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
                s0 = mix(s0);
                s1 = mix(s1);
                s2 = mix(s2);
                s3 = mix(s3);
            }
            __m256i vec = _mm256_set_epi64x(s3, s2, s1, s0);
            __m256i vec_s = _mm256_xor_si256(vec, sign_flip);

            // bit_a = (r >= 195) -> vec_s > 66
            __m256i cmp_a = _mm256_cmpgt_epi8(vec_s, thr_a);
            uint32_t mask_a = (uint32_t)_mm256_movemask_epi8(cmp_a);

            // bit_b = (r >= 146 && r < 195) || (r >= 243)
            //       = (vec_s > 17 && 67 > vec_s) || (vec_s > 114)
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

        uint64_t h0 = mix(mixed_seed_w ^ (((hi0 << 32) | lo0) * 0xd6e8feb86659fd93ULL));
        uint64_t h1 = mix(mixed_seed_w ^ (((hi1 << 32) | lo1) * 0xd6e8feb86659fd93ULL));
        uint64_t h2 = mix(mixed_seed_w ^ (((hi2 << 32) | lo2) * 0xd6e8feb86659fd93ULL));
        uint64_t h3 = mix(mixed_seed_w ^ (((hi3 << 32) | lo3) * 0xd6e8feb86659fd93ULL));

        w[0] = static_cast<uint8_t>(1 + (h0 % 255));
        w[1] = static_cast<uint8_t>(1 + (h1 % 255));
        w[2] = static_cast<uint8_t>(1 + (h2 % 255));
        w[3] = static_cast<uint8_t>(1 + (h3 % 255));
    }
};

void run_verification() {
    std::cout << "=================================================================\n";
    std::cout << "  RUNNING BITWISE VERIFICATION ACROSS SCALES 1..26 & SCRAMBLE 0/1\n";
    std::cout << "=================================================================\n";

    for (int scale = 1; scale <= 26; ++scale) {
        for (int scramble = 0; scramble <= 1; ++scramble) {
            GraphParams p;
            p.scale = scale;
            p.edgefactor = 16;
            p.seed = 100 + scale;
            p.scramble = scramble;
            p.nsample = 1000;

            FastGenBase gen;
            gen.init(p);

            // Test 1000 edges per configuration
            const uint64_t TEST_EDGES = 1000;
            for (uint64_t i = 0; i < TEST_EDGES; i += 4) {
                // Official reference
                uint32_t u_ref[4], v_ref[4];
                uint8_t  w_ref[4];
                for (int k = 0; k < 4; ++k) {
                    generate_edge(p, i + k, u_ref[k], v_ref[k], w_ref[k]);
                }

                // 1-way scalar
                uint32_t u_sc1[4], v_sc1[4];
                uint8_t  w_sc1[4];
                for (int k = 0; k < 4; ++k) {
                    gen.generate1(i + k, u_sc1[k], v_sc1[k], w_sc1[k]);
                }

                // 4-way scalar (Milestone G3)
                uint32_t u_sc4[4], v_sc4[4];
                uint8_t  w_sc4[4];
                gen.generate4_scalar(i, u_sc4, v_sc4, w_sc4);

                // 4-way AVX2 (Milestone G4)
                uint32_t u_avx[4], v_avx[4];
                uint8_t  w_avx[4];
                gen.generate4_avx2(i, u_avx, v_avx, w_avx);

                for (int k = 0; k < 4; ++k) {
                    if (u_ref[k] != u_sc1[k] || v_ref[k] != v_sc1[k] || w_ref[k] != w_sc1[k]) {
                        std::cerr << "FAIL 1-way! Scale=" << scale << " Scr=" << scramble << " i=" << (i+k) << "\n";
                        exit(1);
                    }
                    if (u_ref[k] != u_sc4[k] || v_ref[k] != v_sc4[k] || w_ref[k] != w_sc4[k]) {
                        std::cerr << "FAIL 4-way scalar! Scale=" << scale << " Scr=" << scramble << " i=" << (i+k) << "\n";
                        exit(1);
                    }
                    if (u_ref[k] != u_avx[k] || v_ref[k] != v_avx[k] || w_ref[k] != w_avx[k]) {
                        std::cerr << "FAIL 4-way AVX2! Scale=" << scale << " Scr=" << scramble << " i=" << (i+k)
                                  << " Ref=(" << u_ref[k] << "," << v_ref[k] << "," << (int)w_ref[k] << ") "
                                  << " AVX=(" << u_avx[k] << "," << v_avx[k] << "," << (int)w_avx[k] << ")\n";
                        exit(1);
                    }
                }
            }
        }
    }
    std::cout << ">> ALL SCALES 1..26 & SCRAMBLE 0/1: 100% BITWISE VERIFIED FOR SCALAR, G3, AND G4! <<\n\n";
}

void run_benchmark() {
    std::cout << "=================================================================\n";
    std::cout << "  BENCHMARKING GENERATOR THROUGHPUT (10,000,000 EDGES)\n";
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

    for (const auto &tc : cases) {
        GraphParams p;
        p.scale = tc.scale;
        p.edgefactor = 16;
        p.seed = tc.seed;
        p.scramble = tc.scramble;
        p.nsample = 1000;

        FastGenBase gen;
        gen.init(p);

        std::cout << "--- " << tc.name << " ---\n";

        // 1. Scalar 1-way
        uint64_t dummy1 = 0;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (uint64_t i = 0; i < N_EDGES; ++i) {
            uint32_t u, v; uint8_t w;
            gen.generate1(i, u, v, w);
            dummy1 += u + v + w;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double time1 = std::chrono::duration<double>(t1 - t0).count();

        // 2. Scalar 4-way (G3)
        uint64_t dummy4 = 0;
        alignas(64) uint32_t u4[4], v4[4];
        alignas(64) uint8_t  w4[4];
        auto t2 = std::chrono::high_resolution_clock::now();
        for (uint64_t i = 0; i < N_EDGES; i += 4) {
            gen.generate4_scalar(i, u4, v4, w4);
            dummy4 += u4[0] + u4[1] + u4[2] + u4[3] +
                      v4[0] + v4[1] + v4[2] + v4[3] +
                      w4[0] + w4[1] + w4[2] + w4[3];
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        double time4 = std::chrono::duration<double>(t3 - t2).count();

        // 3. AVX2 4-way (G4)
        uint64_t dummy_avx = 0;
        auto t4 = std::chrono::high_resolution_clock::now();
        for (uint64_t i = 0; i < N_EDGES; i += 4) {
            gen.generate4_avx2(i, u4, v4, w4);
            dummy_avx += u4[0] + u4[1] + u4[2] + u4[3] +
                         v4[0] + v4[1] + v4[2] + v4[3] +
                         w4[0] + w4[1] + w4[2] + w4[3];
        }
        auto t5 = std::chrono::high_resolution_clock::now();
        double time_avx = std::chrono::duration<double>(t5 - t4).count();

        assert(dummy1 == dummy4);
        assert(dummy1 == dummy_avx);

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  1. Scalar 1-way:         " << time1 << " s (" << (N_EDGES / time1 / 1e6) << " M edges/s) [1.00x]\n";
        std::cout << "  2. Scalar 4-way (G3):    " << time4 << " s (" << (N_EDGES / time4 / 1e6) << " M edges/s) [" << (time1 / time4) << "x]\n";
        std::cout << "  3. AVX2 4-way (G4):      " << time_avx << " s (" << (N_EDGES / time_avx / 1e6) << " M edges/s) [" << (time1 / time_avx) << "x]\n";
    }
}

int main() {
    run_verification();
    run_benchmark();
    return 0;
}
