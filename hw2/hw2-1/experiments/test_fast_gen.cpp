#include <cstdint>
#include <cstdio>
#include <chrono>
#include <algorithm>
#include "graph_gen.hpp"

namespace fast {

inline uint64_t mix(uint64_t x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

struct GeneratorContext {
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

    inline void generate(uint64_t i, uint32_t &u, uint32_t &v, uint8_t &w) const {
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
};

} // namespace fast

int main() {
    std::printf("Testing Fast Generator Equivalence...\n");

    GraphParams tests[] = {
        {16, 16, 101, 1, 1000},
        {20, 16, 102, 0, 1000},
        {22, 16, 104, 1, 1000},
        {23, 16, 106, 1, 1000},
        {25, 16, 109, 0, 1000},
        {26, 16, 110, 1, 1000},
    };

    const uint64_t COUNT = 2000000; // 2 million edges per test case

    for (const auto &p : tests) {
        fast::GeneratorContext ctx;
        ctx.init(p);

        // Verify correctness
        for (uint64_t i = 0; i < 100000; i++) {
            uint32_t u_ref, v_ref;
            uint8_t w_ref;
            generate_edge(p, i, u_ref, v_ref, w_ref);

            uint32_t u_fast, v_fast;
            uint8_t w_fast;
            ctx.generate(i, u_fast, v_fast, w_fast);

            if (u_ref != u_fast || v_ref != v_fast || w_ref != w_fast) {
                std::printf("FAILED on scale=%u scramble=%d i=%llu! Expected (%u,%u,%u), got (%u,%u,%u)\n",
                            p.scale, p.scramble, (unsigned long long)i, u_ref, v_ref, w_ref, u_fast, v_fast, w_fast);
                return 1;
            }
        }
        std::printf("Scale %2u (scramble=%d): 100K edges VERIFIED EXACT MATCH!\n", p.scale, p.scramble);

        // Benchmark speed on 2 million edges
        auto t0 = std::chrono::steady_clock::now();
        uint64_t check_ref = 0;
        for (uint64_t i = 0; i < COUNT; i++) {
            uint32_t u, v; uint8_t w;
            generate_edge(p, i, u, v, w);
            check_ref += u + v + w;
        }
        auto t1 = std::chrono::steady_clock::now();
        double time_ref = std::chrono::duration<double>(t1 - t0).count();

        auto t2 = std::chrono::steady_clock::now();
        uint64_t check_fast = 0;
        for (uint64_t i = 0; i < COUNT; i++) {
            uint32_t u, v; uint8_t w;
            ctx.generate(i, u, v, w);
            check_fast += u + v + w;
        }
        auto t3 = std::chrono::steady_clock::now();
        double time_fast = std::chrono::duration<double>(t3 - t2).count();

        if (check_ref != check_fast) {
            std::printf("Checksum mismatch!\n");
            return 1;
        }

        std::printf("  Ref: %.4fs (%.1f M edges/s) | Fast: %.4fs (%.1f M edges/s) | Speedup: %.2fx\n",
                    time_ref, COUNT / (time_ref * 1e6), time_fast, COUNT / (time_fast * 1e6), time_ref / time_fast);
    }

    std::printf("\nALL TESTS PASSED WITH 100%% EXACT BITWISE MATCH!\n");
    return 0;
}
