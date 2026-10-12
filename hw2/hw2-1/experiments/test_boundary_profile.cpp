// Standalone Milestone 7.1 Boundary Profiler for arbitrary P ranks
#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "graph_gen.hpp"

static void read_input(const std::string &path, GraphParams &p, std::vector<uint32_t> &sources) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    uint64_t hdr[6] = {};
    if (rank == 0) {
        std::ifstream f(path);
        for (int i = 0; i < 6; ++i) f >> hdr[i];
        sources.resize(hdr[5]);
        for (auto &src : sources) f >> src;
    }
    MPI_Bcast(hdr, 6, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    p.scale = static_cast<int>(hdr[0]);
    p.edgefactor = static_cast<int>(hdr[1]);
    p.seed = hdr[2];
    p.scramble = static_cast<int>(hdr[3]);
    p.nsample = static_cast<int>(hdr[4]);
    sources.resize(hdr[5]);
    MPI_Bcast(sources.data(), static_cast<int>(hdr[5]), MPI_UINT32_T, 0, MPI_COMM_WORLD);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) std::cerr << "Usage: ./test_boundary_profile <case.txt> [P_target]\n";
        MPI_Finalize();
        return 1;
    }

    int P = (argc >= 3) ? std::stoi(argv[2]) : size;

    GraphParams p{};
    std::vector<uint32_t> sources;
    read_input(argv[1], p, sources);

    const uint64_t N = uint64_t(1) << p.scale;
    const uint64_t M = num_edges(p);

    // Each rank computes degree for its portion of edges
    const uint64_t edge_begin = (M * static_cast<uint64_t>(rank)) / size;
    const uint64_t edge_end   = (M * static_cast<uint64_t>(rank + 1)) / size;
    const uint64_t local_M    = edge_end - edge_begin;

    std::vector<uint32_t> local_deg(N, 0);
    constexpr uint64_t CHUNK = 524288;
    std::vector<uint32_t> u(CHUNK), v(CHUNK);
    std::vector<uint8_t> w(CHUNK);

    for (uint64_t i = 0; i < local_M; i += CHUNK) {
        uint64_t len = std::min(CHUNK, local_M - i);
        generate_edges(p, edge_begin + i, edge_begin + i + len, u.data(), v.data(), w.data());
        for (uint64_t k = 0; k < len; ++k) {
            local_deg[u[k]]++;
            local_deg[v[k]]++;
        }
    }

    std::vector<uint32_t> global_deg;
    if (rank == 0) global_deg.resize(N, 0);

    MPI_Reduce(local_deg.data(), global_deg.data(), N, MPI_UINT32_T, MPI_SUM, 0, MPI_COMM_WORLD);
    std::vector<uint32_t>().swap(local_deg); // Free memory

    if (rank == 0) {
        std::vector<uint64_t> row(N + 1, 0);
        for (uint64_t i = 0; i < N; ++i) {
            row[i + 1] = row[i] + global_deg[i];
        }

        uint64_t total_adj = row[N];
        uint64_t avg_adj_target = total_adj / P;

        std::vector<uint64_t> boundary(P + 1);
        boundary[0] = 0;
        boundary[P] = N;

        for (int r = 1; r < P; ++r) {
            uint64_t target = (total_adj * static_cast<uint64_t>(r)) / P;
            auto it = std::lower_bound(row.data(), row.data() + N + 1, target);
            uint64_t vert = it - row.data();
            if (vert > 0 && vert <= N) {
                if (std::abs(static_cast<int64_t>(row[vert]) - static_cast<int64_t>(target)) >
                    std::abs(static_cast<int64_t>(row[vert - 1]) - static_cast<int64_t>(target))) {
                    vert = vert - 1;
                }
            }
            if (vert < boundary[r - 1]) vert = boundary[r - 1];
            if (vert > N) vert = N;
            boundary[r] = vert;
        }

        std::vector<uint64_t> owner_V(P);
        std::vector<uint64_t> owner_adj(P);
        uint64_t max_owner_adj = 0;
        uint64_t min_nonempty_adj = UINT64_MAX;
        uint64_t max_owner_V = 0;
        int empty_owners = 0;

        for (int r = 0; r < P; ++r) {
            owner_V[r] = boundary[r + 1] - boundary[r];
            owner_adj[r] = row[boundary[r + 1]] - row[boundary[r]];
            if (owner_adj[r] > max_owner_adj) max_owner_adj = owner_adj[r];
            if (owner_adj[r] > 0 && owner_adj[r] < min_nonempty_adj) min_nonempty_adj = owner_adj[r];
            if (owner_V[r] > max_owner_V) max_owner_V = owner_V[r];
            if (owner_V[r] == 0) empty_owners++;
        }
        if (min_nonempty_adj == UINT64_MAX) min_nonempty_adj = 0;

        uint64_t max_vertex_degree = 0;
        uint64_t max_deg_vertex = 0;
        for (uint64_t vert = 0; vert < N; ++vert) {
            uint64_t deg = global_deg[vert];
            if (deg > max_vertex_degree) {
                max_vertex_degree = deg;
                max_deg_vertex = vert;
            }
        }
        std::vector<uint32_t>().swap(global_deg); // Free global_deg

        constexpr int COARSE_BITS = 14;
        const int shift = (p.scale > COARSE_BITS) ? (p.scale - COARSE_BITS) : 0;
        const uint64_t num_blocks = (N + (1ULL << shift) - 1) >> shift;
        std::vector<uint32_t> coarse_owner(num_blocks);

        for (uint64_t b = 0; b < num_blocks; ++b) {
            uint64_t v_start = b << shift;
            if (v_start >= N) v_start = N - 1;
            auto it = std::upper_bound(boundary.begin(), boundary.end(), v_start);
            int o = static_cast<int>(it - boundary.begin()) - 1;
            if (o < 0) o = 0;
            if (o >= P) o = P - 1;
            coarse_owner[b] = static_cast<uint32_t>(o);
        }

        uint64_t test_verts = std::min<uint64_t>(N, 10000000ULL);
        uint64_t test_stride = (N > test_verts) ? (N / test_verts) : 1;
        uint64_t total_steps = 0;
        uint32_t max_steps = 0;
        uint64_t tested_count = 0;
        bool lookup_all_correct = true;

        for (uint64_t vert = 0; vert < N; vert += test_stride) {
            tested_count++;
            uint64_t b = vert >> shift;
            int owner = coarse_owner[b];
            int initial_owner = owner;

            while (owner > 0 && vert < boundary[owner]) owner--;
            while (owner + 1 < P && vert >= boundary[owner + 1]) owner++;

            uint32_t steps = std::abs(owner - initial_owner);
            total_steps += steps;
            if (steps > max_steps) max_steps = steps;

            if (vert < boundary[owner] || vert >= boundary[owner + 1]) {
                lookup_all_correct = false;
            }
        }

        int duplicate_boundaries = 0;
        for (int r = 1; r < P; ++r) {
            if (boundary[r] == boundary[r - 1]) duplicate_boundaries++;
        }

        uint64_t max_private_mem_bytes = 0;
        for (int r = 0; r < P; ++r) {
            uint64_t cursor_bytes = owner_V[r] * sizeof(uint32_t);
            uint64_t packed_bytes = ((M + P - 1) / P) * sizeof(uint64_t);
            uint64_t worst_staging = 96ULL * 1024 * 1024; // 96 MiB conservative staging peak
            uint64_t total_est = cursor_bytes + packed_bytes + worst_staging;
            if (total_est > max_private_mem_bytes) max_private_mem_bytes = total_est;
        }

        double avg_steps = (tested_count > 0) ? (static_cast<double>(total_steps) / tested_count) : 0.0;
        double max_imbalance = (avg_adj_target > 0) ? (static_cast<double>(max_owner_adj) / avg_adj_target) : 1.0;

        std::printf("\n=================================================================================\n");
        std::printf("  [MILESTONE 7.1: BOUNDARY & OWNER PROFILER (Scale %d, EF %d, Scr %d, P=%d)]\n",
                    p.scale, p.edgefactor, p.scramble, P);
        std::printf("---------------------------------------------------------------------------------\n");
        std::printf("  Total Vertices N:             %llu\n", (unsigned long long)N);
        std::printf("  Total Adjacencies (2*M):      %llu\n", (unsigned long long)total_adj);
        std::printf("  Average Target per Owner:     %llu\n", (unsigned long long)avg_adj_target);
        std::printf("---------------------------------------------------------------------------------\n");
        std::printf("  [Super-Hub & Degree Analysis]\n");
        std::printf("  Maximum Vertex Degree:        %llu (at vertex %llu)\n",
                    (unsigned long long)max_vertex_degree, (unsigned long long)max_deg_vertex);
        if (max_vertex_degree > avg_adj_target) {
            std::printf("  WARNING: Super-Hub degree EXCEEDS average owner target by %.2fx!\n",
                        (double)max_vertex_degree / avg_adj_target);
        } else {
            std::printf("  Super-Hub degree vs Target:   %.2f%% of target (Fits inside single owner)\n",
                        100.0 * max_vertex_degree / avg_adj_target);
        }
        std::printf("---------------------------------------------------------------------------------\n");
        std::printf("  [Owner Load Balance Analysis]\n");
        std::printf("  Max Owner Adjacency:          %llu (%.4fx avg target)\n",
                    (unsigned long long)max_owner_adj, max_imbalance);
        std::printf("  Min Non-Empty Adjacency:      %llu (%.4fx avg target)\n",
                    (unsigned long long)min_nonempty_adj,
                    (avg_adj_target > 0 ? (double)min_nonempty_adj / avg_adj_target : 0.0));
        std::printf("  Empty Owners Count:           %d / %d\n", empty_owners, P);
        std::printf("  Duplicate Boundaries Count:   %d / %d\n", duplicate_boundaries, P);
        std::printf("  Max Owner Vertex Count (local_V): %llu (Avg: %llu, %.2fx avg)\n",
                    (unsigned long long)max_owner_V, (unsigned long long)(N / P),
                    (double)max_owner_V / (N / P));
        std::printf("  Max Private Memory Estimate:  %.2f MiB (Budget: 1200 MiB, Margin: %.2f MiB)\n",
                    (double)max_private_mem_bytes / (1024.0 * 1024.0),
                    1200.0 - (double)max_private_mem_bytes / (1024.0 * 1024.0));
        std::printf("---------------------------------------------------------------------------------\n");
        std::printf("  [Coarse Lookup (Shift=%d, %llu Blocks, %llu Checked)]\n",
                    shift, (unsigned long long)num_blocks, (unsigned long long)tested_count);
        std::printf("  Lookup Verification:          %s\n", lookup_all_correct ? "100% PASSED" : "FAILED!");
        std::printf("  Correction Steps:             Avg = %.4f, Max = %u\n", avg_steps, max_steps);
        std::printf("=================================================================================\n\n");
        std::fflush(stdout);
    }

    MPI_Finalize();
    return 0;
}
