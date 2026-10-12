#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include "graph_gen.hpp"

int main(int argc, char **argv) {
    if (argc < 5) {
        std::cerr << "Usage: ./test_dedup scale edgefactor scramble seed [sample_edges]" << std::endl;
        return 1;
    }
    GraphParams p;
    p.scale = std::stoi(argv[1]);
    p.edgefactor = std::stoi(argv[2]);
    p.scramble = std::stoi(argv[3]);
    p.seed = std::stoull(argv[4]);
    p.nsample = 0;

    uint64_t M = num_edges(p);
    uint64_t sample_M = (argc >= 6) ? std::stoull(argv[5]) : M;
    if (sample_M > M) sample_M = M;

    std::cout << "Testing scale=" << p.scale << " ef=" << p.edgefactor 
              << " scramble=" << p.scramble << " seed=" << p.seed << " total M=" << M 
              << " sample=" << sample_M << "..." << std::endl;

    std::vector<uint64_t> keys;
    keys.reserve(sample_M);
    uint64_t self_loops = 0;

    for (uint64_t i = 0; i < sample_M; ++i) {
        uint32_t u, v;
        uint8_t w;
        generate_edge(p, i, u, v, w);
        if (u == v) {
            self_loops++;
            continue;
        }
        uint32_t lo = std::min(u, v);
        uint32_t hi = std::max(u, v);
        keys.push_back((uint64_t(lo) << 32) | hi);
    }

    uint64_t non_self = keys.size();
    std::sort(keys.begin(), keys.end());
    auto last = std::unique(keys.begin(), keys.end());
    uint64_t unique_edges = last - keys.begin();

    std::cout << "  Generated:   " << sample_M << std::endl;
    std::cout << "  Self-loops:  " << self_loops << " (" << (100.0 * self_loops / sample_M) << "%)" << std::endl;
    std::cout << "  Non-self:    " << non_self << std::endl;
    std::cout << "  Unique:      " << unique_edges << " (" << (100.0 * unique_edges / sample_M) << "% of total, " 
              << (100.0 * unique_edges / non_self) << "% of non-self)" << std::endl;
    std::cout << "  Duplicates:  " << (sample_M - unique_edges) << " (" 
              << (100.0 * (sample_M - unique_edges) / sample_M) << "% redundant!)" << std::endl;
    return 0;
}
