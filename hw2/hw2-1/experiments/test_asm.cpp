
#include <cstdint>

inline uint64_t mix_hash(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

void test_descent_scalar(uint64_t state, int scale, uint32_t &out_a, uint32_t &out_b) {
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
    out_a = a;
    out_b = b;
}
