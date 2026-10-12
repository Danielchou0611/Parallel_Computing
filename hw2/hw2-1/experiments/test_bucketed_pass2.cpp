#include <iostream>
#include <vector>
#include <chrono>
#include <cstdint>
#include <random>
#include <cstring>

struct __attribute__((packed)) CSRNeighbor {
    uint32_t to;
    uint8_t w;
};

struct __attribute__((packed)) DirectedEdge {
    uint32_t u;
    uint32_t v;
    uint8_t w;
};

int main() {
    size_t scale = 24;
    size_t N_verts = 1 << scale; // 16.7M vertices
    size_t N_edges = 22369621;   // 22.3M undirected edges = 44.7M directed edges
    size_t total_directed = N_edges * 2;
    size_t ADJ_SIZE = 1 << 26;   // 67M entries in adj

    std::cout << "Benchmarking 256-Bucket Pass 2 on " << total_directed << " directed edges..." << std::endl;

    // Generate random undirected edges
    std::vector<uint32_t> edge_u(N_edges);
    std::vector<uint32_t> edge_v(N_edges);
    std::vector<uint8_t>  edge_w(N_edges);
    std::mt19937 rng(42);
    for (size_t i = 0; i < N_edges; ++i) {
        edge_u[i] = rng() % N_verts;
        edge_v[i] = rng() % N_verts;
        edge_w[i] = rng() & 0xff;
    }

    std::vector<uint32_t> my_degree(N_verts, 0);
    std::vector<CSRNeighbor> adj(ADJ_SIZE);

    // 1. Current Approach: Random Scatter
    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N_edges; ++i) {
        uint32_t u = edge_u[i], v = edge_v[i];
        uint8_t w = edge_w[i];

        uint64_t pu = my_degree[u]++;
        adj[pu & (ADJ_SIZE - 1)] = {v, w};

        uint64_t pv = my_degree[v]++;
        adj[pv & (ADJ_SIZE - 1)] = {u, w};
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double t_random = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "1. Current Random Scatter: " << t_random << " seconds" << std::endl;

    // Reset
    std::fill(my_degree.begin(), my_degree.end(), 0);

    // 2. Bucketed Approach: 256 Buckets
    auto t2 = std::chrono::high_resolution_clock::now();
    int shift = scale - 8;
    uint32_t bcount[256] = {};
    for (size_t i = 0; i < N_edges; ++i) {
        bcount[edge_u[i] >> shift]++;
        bcount[edge_v[i] >> shift]++;
    }

    uint32_t boffset[256];
    boffset[0] = 0;
    for (int b = 1; b < 256; ++b) {
        boffset[b] = boffset[b - 1] + bcount[b - 1];
    }
    uint32_t bcurr[256];
    std::memcpy(bcurr, boffset, sizeof(boffset));

    std::vector<DirectedEdge> bucketed_edges(total_directed);
    for (size_t i = 0; i < N_edges; ++i) {
        uint32_t u = edge_u[i], v = edge_v[i];
        uint8_t w = edge_w[i];

        uint32_t bu = u >> shift;
        bucketed_edges[bcurr[bu]++] = {u, v, w};

        uint32_t bv = v >> shift;
        bucketed_edges[bcurr[bv]++] = {v, u, w};
    }

    // Now scatter bucket by bucket
    for (int b = 0; b < 256; ++b) {
        uint32_t start = boffset[b];
        uint32_t end   = (b == 255) ? total_directed : boffset[b + 1];
        for (uint32_t i = start; i < end; ++i) {
            uint32_t u = bucketed_edges[i].u;
            uint64_t pu = my_degree[u]++;
            adj[pu & (ADJ_SIZE - 1)] = {bucketed_edges[i].v, bucketed_edges[i].w};
        }
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double t_bucketed = std::chrono::duration<double>(t3 - t2).count();
    std::cout << "2. Bucketed Scatter (256 buckets): " << t_bucketed << " seconds" << std::endl;
    std::cout << "Ratio: " << (t_random / t_bucketed) << "x" << std::endl;

    return 0;
}
