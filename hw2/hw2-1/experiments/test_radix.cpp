#include <iostream>
#include <vector>
#include <chrono>
#include <cstdint>
#include <random>
#include <cstring>
#include <algorithm>

struct __attribute__((packed)) DirectedEdge {
    uint32_t u;
    uint32_t v;
    uint8_t w;
};

// 256-bucket (8-bit) Radix Sort
void radix_sort_u(std::vector<DirectedEdge> &src, std::vector<DirectedEdge> &dst, int scale) {
    size_t n = src.size();
    int passes = (scale + 7) / 8; // e.g. for scale 25, 4 passes (or 3 passes for top bits)

    uint32_t count[256];
    uint32_t offset[256];

    DirectedEdge *in = src.data();
    DirectedEdge *out = dst.data();

    for (int p = 0; p < passes; ++p) {
        int shift = p * 8;
        std::memset(count, 0, sizeof(count));

        for (size_t i = 0; i < n; ++i) {
            uint32_t bucket = (in[i].u >> shift) & 0xff;
            count[bucket]++;
        }

        offset[0] = 0;
        for (int b = 1; b < 256; ++b) {
            offset[b] = offset[b - 1] + count[b - 1];
        }

        for (size_t i = 0; i < n; ++i) {
            uint32_t bucket = (in[i].u >> shift) & 0xff;
            out[offset[bucket]++] = in[i];
        }

        std::swap(in, out);
    }
    if (in != src.data()) {
        std::memcpy(src.data(), dst.data(), n * sizeof(DirectedEdge));
    }
}

int main() {
    size_t N_edges = 44739242; // Case 09: 44.7 million directed edges per rank
    std::cout << "Allocating " << N_edges << " directed edges (" 
              << (N_edges * sizeof(DirectedEdge) / (1024*1024)) << " MB)..." << std::endl;

    std::vector<DirectedEdge> edges(N_edges);
    std::vector<DirectedEdge> temp(N_edges);

    std::mt19937_64 rng(42);
    for (size_t i = 0; i < N_edges; ++i) {
        edges[i].u = rng() & ((1 << 25) - 1);
        edges[i].v = rng() & ((1 << 25) - 1);
        edges[i].w = rng() & 0xff;
    }

    std::cout << "Starting Radix Sort (Scale 25, 4 passes of 8 bits)..." << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();
    radix_sort_u(edges, temp, 25);
    auto t1 = std::chrono::high_resolution_clock::now();
    double dt = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "Done in " << dt << " seconds! (" << (N_edges / dt / 1e6) << " M edges/s)" << std::endl;

    // Verify sort
    bool sorted = true;
    for (size_t i = 1; i < N_edges; ++i) {
        if (edges[i].u < edges[i-1].u) { sorted = false; break; }
    }
    std::cout << "Verification: " << (sorted ? "PASSED" : "FAILED") << std::endl;
    return 0;
}
