#include <iostream>
#include <chrono>
#include <cstdint>
#include "graph_gen.hpp"

int main() {
    GraphParams p;
    p.scale = 22;
    p.edgefactor = 16;
    p.seed = 100;
    p.scramble = 1;
    p.nsample = 0;

    uint64_t M = 10000000; // 10 million edges
    std::cout << "Testing generation of " << M << " edges..." << std::endl;

    auto t0 = std::chrono::high_resolution_clock::now();
    uint64_t dummy = 0;
    for (uint64_t i = 0; i < M; ++i) {
        uint32_t u, v;
        uint8_t w;
        generate_edge(p, i, u, v, w);
        dummy += u + v + w;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double dt = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "Time: " << dt << " s, Rate: " << (M / dt / 1e6) << " M edges/s per core (dummy=" << dummy << ")" << std::endl;
    return 0;
}
