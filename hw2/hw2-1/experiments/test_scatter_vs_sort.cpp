#include <iostream>
#include <vector>
#include <chrono>
#include <cstdint>
#include <random>
#include <cstring>
#include <algorithm>

struct __attribute__((packed)) CSRNeighbor {
    uint32_t to;
    uint8_t w;
};

struct __attribute__((packed)) DirectedEdge {
    uint32_t u;
    uint32_t v;
    uint8_t w;
};

void radix_sort_u(DirectedEdge *src, DirectedEdge *dst, size_t n, int scale) {
    int passes = (scale + 7) / 8;
    uint32_t count[256];
    uint32_t offset[256];
    DirectedEdge *in = src;
    DirectedEdge *out = dst;

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
    if (in != src) {
        std::memcpy(src, dst, n * sizeof(DirectedEdge));
    }
}

int main() {
    size_t N_verts = 1 << 24; // 16.7M vertices
    size_t N_edges = 22369621; // 22.3M directed edges
    size_t ADJ_SIZE = 1 << 26; // 67M entries in adj (~335 MB)

    std::cout << "Benchmarking Random Scatter vs Sorted Stream on " << N_edges << " edges..." << std::endl;

    std::vector<DirectedEdge> random_edges(N_edges);
    std::mt19937 rng(42);
    for (size_t i = 0; i < N_edges; ++i) {
        random_edges[i].u = rng() % N_verts;
        random_edges[i].v = rng() % N_verts;
        random_edges[i].w = rng() & 0xff;
    }

    std::vector<uint32_t> my_degree(N_verts, 0);
    std::vector<CSRNeighbor> adj(ADJ_SIZE);

    // 1. Benchmark Random Scatter (CURRENT APPROACH)
    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N_edges; ++i) {
        uint32_t u = random_edges[i].u;
        uint64_t pu = my_degree[u]++;
        adj[pu & (ADJ_SIZE - 1)] = {random_edges[i].v, random_edges[i].w};
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double t_random = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "1. Current Random Scatter: " << t_random << " seconds" << std::endl;

    // Reset
    std::fill(my_degree.begin(), my_degree.end(), 0);

    // 2. Benchmark Radix Sort + Stream Scatter (NEW APPROACH)
    std::vector<DirectedEdge> temp_edges(N_edges);
    auto t2 = std::chrono::high_resolution_clock::now();
    radix_sort_u(random_edges.data(), temp_edges.data(), N_edges, 24);
    auto t3 = std::chrono::high_resolution_clock::now();
    double t_sort = std::chrono::duration<double>(t3 - t2).count();

    for (size_t i = 0; i < N_edges; ++i) {
        uint32_t u = random_edges[i].u;
        uint64_t pu = my_degree[u]++;
        adj[pu & (ADJ_SIZE - 1)] = {random_edges[i].v, random_edges[i].w};
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    double t_stream = std::chrono::duration<double>(t4 - t3).count();
    double t_total_new = t_sort + t_stream;

    std::cout << "2. New Approach: Sort (" << t_sort << "s) + Stream (" << t_stream 
              << "s) = Total " << t_total_new << " seconds" << std::endl;
    std::cout << "Speedup factor: " << (t_random / t_total_new) << "x faster!" << std::endl;

    return 0;
}
