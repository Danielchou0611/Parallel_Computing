#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>

inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

int main() {
    constexpr int HLL_P = 18;
    constexpr uint32_t HLL_M = 1u << HLL_P; // 262144
    std::vector<uint8_t> registers(HLL_M, 0);

    // Insert 1,000,000 distinct items
    for (uint64_t i = 0; i < 1000000; ++i) {
        uint64_t key = (i << 20) ^ (i * 1337);
        uint64_t h = splitmix64(key);
        uint32_t j = h >> (64 - HLL_P);
        uint64_t w = h & ((1ULL << (64 - HLL_P)) - 1);
        uint8_t rho = (w == 0) ? (64 - HLL_P + 1) : static_cast<uint8_t>(__builtin_clzll(w) - HLL_P + 1);
        registers[j] = std::max(registers[j], rho);
    }

    // Estimate
    double sum = 0.0;
    int zeros = 0;
    for (uint32_t j = 0; j < HLL_M; ++j) {
        sum += std::pow(2.0, -static_cast<double>(registers[j]));
        if (registers[j] == 0) zeros++;
    }
    double alpha = 0.72134; // for m >= 128
    double E = alpha * static_cast<double>(HLL_M) * static_cast<double>(HLL_M) / sum;

    if (E <= 2.5 * HLL_M && zeros > 0) {
        E = static_cast<double>(HLL_M) * std::log(static_cast<double>(HLL_M) / zeros);
    }

    std::cout << "True count: 1000000, Estimated: " << E 
              << ", Error: " << (100.0 * (E - 1000000) / 1000000) << "%" << std::endl;
    return 0;
}
