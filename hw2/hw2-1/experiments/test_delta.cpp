#include <mpi.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

#include "graph_gen.hpp"

namespace {

const uint32_t INF = 0xffffffffu;

inline uint64_t align64(uint64_t bytes) {
    return (bytes + 63) & ~uint64_t(63);
}

inline bool atomic_min_u32(uint32_t *ptr, uint32_t value) {
    uint32_t old = __atomic_load_n(ptr, __ATOMIC_RELAXED);
    while (value < old) {
        if (__atomic_compare_exchange_n(ptr, &old, value, false,
                                        __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
            return true;
        }
    }
    return false;
}

// Single-rank Dial's algorithm (Baseline)
struct BucketEntry {
    uint32_t vertex;
    uint32_t distance;
};

void run_dial(const uint64_t *row, const uint32_t *col, const uint8_t *w,
              uint64_t N, uint32_t src, std::vector<uint32_t> &dist,
              uint64_t &rounds)
{
    dist.assign(N, INF);
    std::array<std::vector<BucketEntry>, 256> buckets;
    uint64_t queued = 0;
    uint32_t current = 0;
    rounds = 0;

    dist[src] = 0;
    buckets[0].push_back({src, 0});
    ++queued;

    while (queued != 0) {
        auto &bucket = buckets[current & 255u];
        if (bucket.empty()) {
            ++current;
            continue;
        }

        rounds++;
        BucketEntry entry = bucket.back();
        bucket.pop_back();
        --queued;

        const uint32_t x = entry.vertex;
        if (dist[x] != entry.distance) continue;

        const uint32_t d = entry.distance;
        for (uint64_t e = row[x]; e < row[x + 1]; ++e) {
            const uint32_t y = col[e];
            const uint32_t nd = d + static_cast<uint32_t>(w[e]);
            if (nd < dist[y]) {
                dist[y] = nd;
                buckets[nd & 255u].push_back({y, nd});
                ++queued;
            }
        }
    }
}

// Parallel Cooperative Delta-Stepping
void run_delta_stepping(
    const uint64_t *row, const uint32_t *col, const uint8_t *w,
    uint64_t N, uint32_t src,
    uint32_t *dist, uint32_t *shared_frontier,
    uint64_t *shared_work_index,
    MPI_Comm comm, int rank, int size, MPI_Win win,
    uint32_t delta, uint64_t &rounds, double &coll_time, double &relax_time)
{
    const uint64_t v_begin = (N * static_cast<uint64_t>(rank)) / size;
    const uint64_t v_end   = (N * static_cast<uint64_t>(rank + 1)) / size;
    std::fill(dist + v_begin, dist + v_end, INF);

    MPI_Win_sync(win);
    MPI_Barrier(comm);
    MPI_Win_sync(win);

    std::vector<std::vector<uint32_t>> local_bins;
    if (rank == 0) {
        dist[src] = 0;
        local_bins.resize(1);
        local_bins[0].push_back(src);
    }

    MPI_Win_sync(win);
    MPI_Barrier(comm);
    MPI_Win_sync(win);

    uint32_t curr_bin = 0;
    rounds = 0;
    coll_time = 0;
    relax_time = 0;

    while (curr_bin != INF) {
        rounds++;
        uint64_t local_count = (curr_bin < local_bins.size()) ? local_bins[curr_bin].size() : 0;

        double tc0 = MPI_Wtime();
        uint64_t my_offset = 0;
        MPI_Exscan(&local_count, &my_offset, 1, MPI_UINT64_T, MPI_SUM, comm);
        if (rank == 0) my_offset = 0;

        uint64_t total_count = 0;
        MPI_Allreduce(&local_count, &total_count, 1, MPI_UINT64_T, MPI_SUM, comm);
        coll_time += (MPI_Wtime() - tc0);

        if (total_count > N) std::printf("OVERFLOW: total_count=%llu > N=%llu\n", (unsigned long long)total_count, (unsigned long long)N); if (total_count > 0) {
            if (local_count > 0) {
                std::memcpy(shared_frontier + my_offset, local_bins[curr_bin].data(), local_count * sizeof(uint32_t));
                local_bins[curr_bin].clear();
            }

            if (rank == 0) {
                *shared_work_index = 0;
            }

            double tc1 = MPI_Wtime();
            MPI_Win_sync(win);
            MPI_Barrier(comm);
            MPI_Win_sync(win);
            coll_time += (MPI_Wtime() - tc1);

            double tr0 = MPI_Wtime();
            constexpr uint64_t WORK_CHUNK = 64;
            while (true) {
                uint64_t begin = __atomic_fetch_add(shared_work_index, WORK_CHUNK, __ATOMIC_RELAXED);
                if (begin >= total_count) break;
                uint64_t end = std::min(begin + WORK_CHUNK, total_count);

                for (uint64_t i = begin; i < end; ++i) {
                    uint32_t u = shared_frontier[i];
                    uint32_t du = dist[u];
                    if (du < curr_bin * delta) continue;

                    for (uint64_t e = row[u]; e < row[u + 1]; ++e) {
                        uint32_t v = col[e];
                        uint32_t nd = du + static_cast<uint32_t>(w[e]);
                        if (atomic_min_u32(&dist[v], nd)) {
                            uint32_t dest_bin = nd / delta;
                            if (dest_bin >= local_bins.size()) {
                                local_bins.resize(dest_bin + 1);
                            }
                            local_bins[dest_bin].push_back(v);
                        }
                    }
                }
            }
            relax_time += (MPI_Wtime() - tr0);

            double tc2 = MPI_Wtime();
            MPI_Win_sync(win);
            MPI_Barrier(comm);
            MPI_Win_sync(win);
            coll_time += (MPI_Wtime() - tc2);
        } else {
            // Find next non-empty bin
            uint32_t local_next = INF;
            for (size_t i = curr_bin + 1; i < local_bins.size(); ++i) {
                if (!local_bins[i].empty()) {
                    local_next = static_cast<uint32_t>(i);
                    break;
                }
            }

            double tc3 = MPI_Wtime();
            uint32_t team_next = INF;
            MPI_Allreduce(&local_next, &team_next, 1, MPI_UINT32_T, MPI_MIN, comm);
            coll_time += (MPI_Wtime() - tc3);
            curr_bin = team_next;
        }
    }
}

} // namespace

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const char *case_file = (argc > 1) ? argv[1] : "test_s22.txt";
    uint32_t test_delta = (argc > 2) ? std::atoi(argv[2]) : 32;

    FILE *f = std::fopen(case_file, "r");
    if (!f) {
        if (rank == 0) std::perror("fopen");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    GraphParams p{};
    if (std::fscanf(f, "%u %u %lu %d %lu", &p.scale, &p.edgefactor, &p.seed, &p.scramble, &p.nsample) != 5) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    size_t S = 0;
    if (std::fscanf(f, "%zu", &S) != 1) MPI_Abort(MPI_COMM_WORLD, 1);
    std::vector<uint32_t> sources(S);
    for (size_t i = 0; i < S; i++) {
        if (std::fscanf(f, "%u", &sources[i]) != 1) MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::fclose(f);

    const uint64_t N = num_vertices(p);
    const uint64_t M = num_edges(p);

    // Shared memory setup
    uint64_t row_bytes = align64((N + 1) * sizeof(uint64_t));
    uint64_t col_bytes = align64((2 * M) * sizeof(uint32_t));
    uint64_t w_bytes   = align64((2 * M) * sizeof(uint8_t));
    uint64_t deg_bytes = align64((uint64_t)size * N * sizeof(uint32_t));
    uint64_t dist_bytes = align64(N * sizeof(uint32_t));
    uint64_t front_bytes = align64(N * sizeof(uint32_t));
    uint64_t ctrl_bytes = align64(64);

    uint64_t total_bytes = row_bytes + col_bytes + w_bytes + deg_bytes + dist_bytes + front_bytes + ctrl_bytes;

    MPI_Win win;
    void *baseptr = nullptr;
    MPI_Aint alloc_sz = (rank == 0) ? total_bytes : 0;
    MPI_Win_allocate_shared(alloc_sz, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &baseptr, &win);

    if (rank != 0) {
        MPI_Aint sz;
        int disp;
        MPI_Win_shared_query(win, 0, &sz, &disp, &baseptr);
    }

    char *ptr = static_cast<char *>(baseptr);
    uint64_t *row = reinterpret_cast<uint64_t *>(ptr); ptr += row_bytes;
    uint32_t *col = reinterpret_cast<uint32_t *>(ptr); ptr += col_bytes;
    uint8_t *w    = reinterpret_cast<uint8_t *>(ptr);  ptr += w_bytes;
    uint32_t *all_degree = reinterpret_cast<uint32_t *>(ptr); ptr += deg_bytes;
    uint32_t *dist = reinterpret_cast<uint32_t *>(ptr); ptr += dist_bytes;
    uint32_t *shared_frontier = reinterpret_cast<uint32_t *>(ptr); ptr += front_bytes;
    uint64_t *shared_work_index = reinterpret_cast<uint64_t *>(ptr);

    uint32_t *my_degree = all_degree + static_cast<uint64_t>(rank) * N;
    std::memset(my_degree, 0, N * sizeof(uint32_t));

    MPI_Win_sync(win);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Win_sync(win);

    // Parallel build CSR
    const uint64_t edge_begin = (M * static_cast<uint64_t>(rank)) / size;
    const uint64_t edge_end   = (M * static_cast<uint64_t>(rank + 1)) / size;
    const uint64_t local_M    = edge_end - edge_begin;

    std::vector<uint64_t> packed_edges(local_M);
    const uint64_t CHUNK = 524288;
    std::vector<uint32_t> u_chunk(CHUNK), v_chunk(CHUNK);
    std::vector<uint8_t> w_chunk(CHUNK);

    for (uint64_t i = edge_begin; i < edge_end; i += CHUNK) {
        uint64_t chunk_len = std::min(CHUNK, edge_end - i);
        generate_edges(p, i, i + chunk_len, u_chunk.data(), v_chunk.data(), w_chunk.data());
        uint64_t base_idx = i - edge_begin;
        for (uint64_t k = 0; k < chunk_len; k++) {
            uint32_t uk = u_chunk[k], vk = v_chunk[k];
            uint8_t wk  = w_chunk[k];
            my_degree[uk]++;
            my_degree[vk]++;
            packed_edges[base_idx + k] = (static_cast<uint64_t>(uk) << 34) |
                                         (static_cast<uint64_t>(vk) << 8) |
                                         static_cast<uint64_t>(wk);
        }
    }

    MPI_Win_sync(win);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Win_sync(win);

    // Prefix sum
    const uint64_t v_begin = (N * static_cast<uint64_t>(rank)) / size;
    const uint64_t v_end   = (N * static_cast<uint64_t>(rank + 1)) / size;

    uint64_t local_adj = 0;
    for (uint64_t vert = v_begin; vert < v_end; vert++) {
        uint32_t d = 0;
        for (int r = 0; r < size; r++) d += all_degree[static_cast<uint64_t>(r) * N + vert];
        local_adj += d;
    }

    uint64_t rank_base = 0;
    MPI_Exscan(&local_adj, &rank_base, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) rank_base = 0;

    uint64_t running_row = rank_base;
    for (uint64_t vert = v_begin; vert < v_end; vert++) {
        row[vert] = running_row;
        uint32_t rel = 0;
        for (int r = 0; r < size; r++) {
            uint64_t idx = static_cast<uint64_t>(r) * N + vert;
            uint32_t d = all_degree[idx];
            all_degree[idx] = static_cast<uint32_t>(running_row + rel);
            rel += d;
        }
        running_row += rel;
    }
    if (v_end == N) row[N] = running_row;

    MPI_Win_sync(win);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Win_sync(win);

    // Pass 2: fill CSR
    for (uint64_t k = 0; k < local_M; k++) {
        uint64_t packed = packed_edges[k];
        uint32_t uk = static_cast<uint32_t>(packed >> 34);
        uint32_t vk = static_cast<uint32_t>((packed >> 8) & 0x3ffffffu);
        uint8_t wk  = static_cast<uint8_t>(packed);

        uint64_t pu = my_degree[uk]++;
        col[pu] = vk; w[pu] = wk;
        uint64_t pv = my_degree[vk]++;
        col[pv] = uk; w[pv] = wk;
    }
    std::vector<uint64_t>().swap(packed_edges);

    MPI_Win_sync(win);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Win_sync(win);

    if (rank == 0) std::printf("Graph CSR built successfully. Starting comparison...\n");

    uint32_t src0 = sources[0];

    // 1. Run Baseline Dial on rank 0
    std::vector<uint32_t> golden_dist;
    double t_dial = 0;
    uint64_t dial_rounds = 0;
    if (rank == 0) {
        double td0 = MPI_Wtime();
        run_dial(row, col, w, N, src0, golden_dist, dial_rounds);
        t_dial = MPI_Wtime() - td0;
        std::printf("Dial completed in %.4fs (rounds=%llu)\n", t_dial, (unsigned long long)dial_rounds);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // 2. Run Delta-Stepping across all ranks
    std::vector<uint32_t> deltas = {1, 2, 4, 8, 16, 32, 64};
    for (uint32_t d_val : deltas) {
        uint64_t ds_rounds = 0;
        double ds_coll = 0, ds_relax = 0;

        MPI_Barrier(MPI_COMM_WORLD);
        double tds0 = MPI_Wtime();
        run_delta_stepping(row, col, w, N, src0, dist, shared_frontier, shared_work_index,
                           MPI_COMM_WORLD, rank, size, win, d_val, ds_rounds, ds_coll, ds_relax);
        double t_ds = MPI_Wtime() - tds0;

        uint64_t max_rounds = 0;
        double max_time = 0, max_coll = 0, max_relax = 0;
        MPI_Reduce(&ds_rounds, &max_rounds, 1, MPI_UINT64_T, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&t_ds, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&ds_coll, &max_coll, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&ds_relax, &max_relax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            // Verify correctness against golden_dist
            size_t mismatches = 0;
            for (uint64_t v = 0; v < N; v++) {
                if (dist[v] != golden_dist[v]) {
                    if (mismatches < 5) {
                        std::printf("Mismatch at v=%llu: Delta=%u vs Dial=%u\n",
                                    (unsigned long long)v, dist[v], golden_dist[v]);
                    }
                    mismatches++;
                }
            }
            std::printf("[Delta=%2u] Time: %.4fs (Coll: %.4fs, Relax: %.4fs) | Rounds: %4llu | Mismatches: %zu\n",
                        d_val, max_time, max_coll, max_relax, (unsigned long long)max_rounds, mismatches);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Win_free(&win);
    MPI_Finalize();
    return 0;
}
