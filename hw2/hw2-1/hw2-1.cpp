// HW 2-1: MPI shortest paths (Version 5B + Detailed SSSP & Collective Diagnostics + Configurable Edge Cache)
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <utility>
#include <vector>

#include "graph_gen.hpp"

#ifndef CHUNK_SIZE
#define CHUNK_SIZE 524288 // 512K edges per chunk
#endif

#ifndef CACHE_EDGES
#define CACHE_EDGES 1 // 1: packed_edges cache, 0: on-the-fly recomputation
#endif

#ifndef DELTA_STEPPING
#define DELTA_STEPPING 1 // 1: Dial, >1: Delta-Stepping with bucket width DELTA_STEPPING
#endif

namespace {

const uint32_t INF = 0xffffffffu;

struct FlatResultHeader {
    int64_t n_reached;
    uint64_t checksum;
    uint64_t digest1;
    uint64_t digest2;
};

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

inline uint64_t mix_hash(uint64_t x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

struct FastGenerator {
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
        mixed_seed = mix_hash(p.seed);
        mixed_seed_w = mix_hash(p.seed ^ 0x77e19470ULL);

        uint64_t s_seed = p.seed ^ 0x5c2a3b1eULL;
        scramble_mask = (uint64_t(1) << scale) - 1;
        scramble_sh = scale / 2 > 0 ? scale / 2 : 1;
        scramble_k1 = mix_hash(s_seed ^ 0x1111ULL) | 1;
        scramble_k2 = mix_hash(s_seed ^ 0x2222ULL) | 1;
        scramble_k3 = mix_hash(s_seed ^ 0x3333ULL) | 1;
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
        uint64_t state = mix_hash(mixed_seed ^ (i * 0xd6e8feb86659fd93ULL));
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

        if (scramble) {
            a = scramble_id(a);
            b = scramble_id(b);
        }
        u = a;
        v = b;

        uint64_t lo = std::min(a, b), hi = std::max(a, b);
        uint64_t pair_key = (hi << 32) | lo;
        uint64_t h = mix_hash(mixed_seed_w ^ (pair_key * 0xd6e8feb86659fd93ULL));
        w = static_cast<uint8_t>(1 + (h % 255));
    }
};

// Single-rank Dial's algorithm (Version 3)
struct __attribute__((packed)) CSRNeighbor {
    uint32_t to;
    uint8_t w;
};

struct BucketEntry {
    uint32_t vertex;
    uint32_t distance;
};

class DialQueue {
public:
    std::array<std::vector<BucketEntry>, 256> buckets;

    void clear() {
        for (auto &b : buckets) {
            b.clear();
        }
    }
};

void dial(const uint64_t *row, const CSRNeighbor *adj,
          uint64_t N, uint32_t src, std::vector<uint32_t> &dist,
          DialQueue &q)
{
    dist.assign(N, INF);
    uint64_t queued = 0;
    uint32_t current = 0;

    dist[src] = 0;
    q.buckets[0].push_back({src, 0});
    ++queued;

    while (queued != 0) {
        auto &bucket = q.buckets[current & 255u];
        if (bucket.empty()) {
            ++current;
            continue;
        }

        BucketEntry entry = bucket.back();
        bucket.pop_back();
        --queued;

        if (entry.distance != current) {
            std::fprintf(stderr, "bucket collision: current=%u entry.distance=%u\n",
                         current, entry.distance);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        const uint32_t x = entry.vertex;
        if (dist[x] != entry.distance) {
            continue; // stale entry superseded by shorter path
        }

        const uint32_t d = entry.distance;

        for (uint64_t e = row[x]; e < row[x + 1]; ++e) {
            const uint32_t y = adj[e].to;
            const uint32_t nd = d + static_cast<uint32_t>(adj[e].w);
            if (nd < dist[y]) {
                dist[y] = nd;
                q.buckets[nd & 255u].push_back({y, nd});
                ++queued;
            }
        }
    }
}

struct SSSPStats {
    uint64_t rounds = 0;
    uint64_t total_frontier = 0;
    double collective_time = 0.0;
    double relax_time = 0.0;
};

// Cooperative Dial's algorithm for a team of ranks
void cooperative_dial(
    const uint64_t *row, const CSRNeighbor *adj,
    uint64_t N, uint32_t src,
    uint32_t *dist, uint32_t *shared_frontier,
    uint64_t *shared_work_index,
    MPI_Comm team_comm, int team_rank, int team_size, MPI_Win win,
    SSSPStats &stats)
{
    // 1. Parallel initialize dist
    const uint64_t v_begin = (N * static_cast<uint64_t>(team_rank)) / team_size;
    const uint64_t v_end   = (N * static_cast<uint64_t>(team_rank + 1)) / team_size;
    std::fill(dist + v_begin, dist + v_end, INF);

    MPI_Win_sync(win);
    MPI_Barrier(team_comm);
    MPI_Win_sync(win);

    std::array<std::vector<uint32_t>, 256> local_buckets;

    if (team_rank == 0) {
        dist[src] = 0;
        local_buckets[0].push_back(src);
    }

    MPI_Win_sync(win);
    MPI_Barrier(team_comm);
    MPI_Win_sync(win);

    uint32_t current = 0;

    while (current != INF) {
        stats.rounds++;
        auto &my_bucket = local_buckets[current & 255u];
        uint64_t local_count = my_bucket.size();

        double tc0 = MPI_Wtime();
        // 2. Prefix sum to merge local buckets into shared_frontier
        uint64_t my_offset = 0;
        MPI_Exscan(&local_count, &my_offset, 1, MPI_UINT64_T, MPI_SUM, team_comm);
        if (team_rank == 0) my_offset = 0;

        uint64_t total_count = 0;
        MPI_Allreduce(&local_count, &total_count, 1, MPI_UINT64_T, MPI_SUM, team_comm);
        stats.collective_time += (MPI_Wtime() - tc0);

        if (total_count > 0) {
            stats.total_frontier += total_count;

            if (local_count > 0) {
                std::memcpy(shared_frontier + my_offset, my_bucket.data(), local_count * sizeof(uint32_t));
                my_bucket.clear();
            }

            if (team_rank == 0) {
                *shared_work_index = 0;
            }

            double tc1 = MPI_Wtime();
            MPI_Win_sync(win);
            MPI_Barrier(team_comm);
            MPI_Win_sync(win);
            stats.collective_time += (MPI_Wtime() - tc1);

            double tr0 = MPI_Wtime();
            // 3. Dynamic work chunk distribution across team ranks
            constexpr uint64_t WORK_CHUNK = 64;
            while (true) {
                uint64_t begin = __atomic_fetch_add(shared_work_index, WORK_CHUNK, __ATOMIC_RELAXED);
                if (begin >= total_count) break;
                uint64_t end = std::min(begin + WORK_CHUNK, total_count);

                for (uint64_t i = begin; i < end; ++i) {
                    uint32_t x = shared_frontier[i];
                    if (dist[x] != current) continue; // stale

                    for (uint64_t e = row[x]; e < row[x + 1]; ++e) {
                        uint32_t y = adj[e].to;
                        uint32_t nd = current + static_cast<uint32_t>(adj[e].w);
                        if (atomic_min_u32(&dist[y], nd)) {
                            local_buckets[nd & 255u].push_back(y);
                        }
                    }
                }
            }
            stats.relax_time += (MPI_Wtime() - tr0);

            double tc2 = MPI_Wtime();
            MPI_Win_sync(win);
            MPI_Barrier(team_comm);
            MPI_Win_sync(win);
            stats.collective_time += (MPI_Wtime() - tc2);
        } else {
            my_bucket.clear();
        }

        // 4. Find next distance across the team
        uint32_t local_next = INF;
        for (uint32_t delta = 1; delta <= 255; ++delta) {
            uint32_t cand = current + delta;
            if (!local_buckets[cand & 255u].empty()) {
                local_next = cand;
                break;
            }
        }

        double tc3 = MPI_Wtime();
        uint32_t team_next = INF;
        MPI_Allreduce(&local_next, &team_next, 1, MPI_UINT32_T, MPI_MIN, team_comm);
        stats.collective_time += (MPI_Wtime() - tc3);
        current = team_next;
    }
}

// Cooperative Delta-Stepping algorithm
void cooperative_delta_stepping(
    const uint64_t *row, const CSRNeighbor *adj,
    uint64_t N, uint32_t src,
    uint32_t *dist, uint32_t *shared_frontier,
    uint64_t *shared_work_index,
    MPI_Comm team_comm, int team_rank, int team_size, MPI_Win win,
    SSSPStats &stats, uint32_t delta = 2)
{
    const uint64_t v_begin = (N * static_cast<uint64_t>(team_rank)) / team_size;
    const uint64_t v_end   = (N * static_cast<uint64_t>(team_rank + 1)) / team_size;
    std::fill(dist + v_begin, dist + v_end, INF);

    MPI_Win_sync(win);
    MPI_Barrier(team_comm);
    MPI_Win_sync(win);

    std::vector<std::vector<uint32_t>> local_bins;
    if (team_rank == 0) {
        dist[src] = 0;
        local_bins.resize(1);
        local_bins[0].push_back(src);
    }

    MPI_Win_sync(win);
    MPI_Barrier(team_comm);
    MPI_Win_sync(win);

    uint32_t curr_bin = 0;

    while (curr_bin != INF) {
        stats.rounds++;
        uint64_t local_count = (curr_bin < local_bins.size()) ? local_bins[curr_bin].size() : 0;

        double tc0 = MPI_Wtime();
        uint64_t my_offset = 0;
        MPI_Exscan(&local_count, &my_offset, 1, MPI_UINT64_T, MPI_SUM, team_comm);
        if (team_rank == 0) my_offset = 0;

        uint64_t total_count = 0;
        MPI_Allreduce(&local_count, &total_count, 1, MPI_UINT64_T, MPI_SUM, team_comm);
        stats.collective_time += (MPI_Wtime() - tc0);

        if (total_count > 0) {
            stats.total_frontier += total_count;

            if (local_count > 0) {
                std::memcpy(shared_frontier + my_offset, local_bins[curr_bin].data(), local_count * sizeof(uint32_t));
                local_bins[curr_bin].clear();
            }

            if (team_rank == 0) {
                *shared_work_index = 0;
            }

            double tc1 = MPI_Wtime();
            MPI_Win_sync(win);
            MPI_Barrier(team_comm);
            MPI_Win_sync(win);
            stats.collective_time += (MPI_Wtime() - tc1);

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
                        uint32_t v = adj[e].to;
                        uint32_t nd = du + static_cast<uint32_t>(adj[e].w);
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
            stats.relax_time += (MPI_Wtime() - tr0);

            double tc2 = MPI_Wtime();
            MPI_Win_sync(win);
            MPI_Barrier(team_comm);
            MPI_Win_sync(win);
            stats.collective_time += (MPI_Wtime() - tc2);
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
            MPI_Allreduce(&local_next, &team_next, 1, MPI_UINT32_T, MPI_MIN, team_comm);
            stats.collective_time += (MPI_Wtime() - tc3);
            curr_bin = team_next;
        }
    }
}

}  // namespace

void run(const GraphParams &p, const std::vector<uint32_t> &sources, std::vector<SourceResult> &results)
{
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    double t0 = MPI_Wtime();

    // 1. Create shared communicator for processes on the same node
    MPI_Comm shm_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &shm_comm);
    int shm_rank, shm_size;
    MPI_Comm_rank(shm_comm, &shm_rank);
    MPI_Comm_size(shm_comm, &shm_size);

    const uint64_t N = num_vertices(p);
    const uint64_t M = num_edges(p);
    const size_t S = sources.size();

    // Cooperative team configuration (when sufficient ranks exceed sources)
    bool use_coop = (world_size >= 4 && world_size > static_cast<int>(S) && p.scale >= 20);
    int team_count = use_coop ? std::min<int>(world_size, static_cast<int>(S)) : 1;

    // 2. Compute memory layout for Shared-Memory Window
    uint64_t row_bytes     = align64((N + 1) * sizeof(uint64_t));
    uint64_t adj_bytes     = align64((2 * M) * sizeof(CSRNeighbor));
    uint64_t hdr_bytes     = align64(S * sizeof(FlatResultHeader));
    uint64_t sample_bytes  = align64((uint64_t)S * p.nsample * sizeof(int32_t));
    uint64_t all_deg_bytes = align64((uint64_t)shm_size * N * sizeof(uint32_t));

    // Cooperative mode buffers (up to team_count teams)
    uint64_t coop_dist_bytes     = use_coop ? align64((uint64_t)team_count * N * sizeof(uint32_t)) : 0;
    uint64_t coop_frontier_bytes = use_coop ? align64((uint64_t)team_count * N * sizeof(uint32_t)) : 0;
    uint64_t coop_ctrl_bytes     = use_coop ? align64((uint64_t)team_count * 64) : 0;

    uint64_t total_bytes = row_bytes + adj_bytes + hdr_bytes + sample_bytes + all_deg_bytes
                         + coop_dist_bytes + coop_frontier_bytes + coop_ctrl_bytes;

    MPI_Win win;
    void *baseptr = nullptr;
    MPI_Aint alloc_size = (shm_rank == 0) ? static_cast<MPI_Aint>(total_bytes) : 0;
    MPI_Win_allocate_shared(alloc_size, 1, MPI_INFO_NULL, shm_comm, &baseptr, &win);

    if (shm_rank != 0) {
        MPI_Aint sz;
        int disp;
        MPI_Win_shared_query(win, 0, &sz, &disp, &baseptr);
    }

    // Partition the shared memory window into typed pointers
    char *ptr = static_cast<char *>(baseptr);
    uint64_t *row = reinterpret_cast<uint64_t *>(ptr);
    ptr += row_bytes;
    CSRNeighbor *adj = reinterpret_cast<CSRNeighbor *>(ptr);
    ptr += adj_bytes;
    FlatResultHeader *res_hdr = reinterpret_cast<FlatResultHeader *>(ptr);
    ptr += hdr_bytes;
    int32_t *sample_dist_flat = reinterpret_cast<int32_t *>(ptr);
    ptr += sample_bytes;
    uint32_t *all_degree = reinterpret_cast<uint32_t *>(ptr);
    ptr += all_deg_bytes;
    uint32_t *coop_dist = reinterpret_cast<uint32_t *>(ptr);
    ptr += coop_dist_bytes;
    uint32_t *coop_frontier = reinterpret_cast<uint32_t *>(ptr);
    ptr += coop_frontier_bytes;
    uint64_t *coop_work_index = reinterpret_cast<uint64_t *>(ptr);

    // Each rank owns its distinct slice of all_degree: [shm_rank * N, (shm_rank + 1) * N)
    uint32_t *my_degree = all_degree + static_cast<uint64_t>(shm_rank) * N;
    std::memset(my_degree, 0, N * sizeof(uint32_t));

    MPI_Win_sync(win);
    MPI_Barrier(shm_comm);
    MPI_Win_sync(win);

    double t_alloc = MPI_Wtime() - t0;
    double t1 = MPI_Wtime();

    // 3. Parallel Graph Construction
    const uint64_t edge_begin = (M * static_cast<uint64_t>(shm_rank)) / shm_size;
    const uint64_t edge_end   = (M * static_cast<uint64_t>(shm_rank + 1)) / shm_size;
    const uint64_t local_M    = edge_end - edge_begin;

    const uint64_t CHUNK = CHUNK_SIZE;
    std::vector<uint32_t> u(CHUNK), v(CHUNK);
    std::vector<uint8_t> w_chunk(CHUNK);

#if CACHE_EDGES
    std::vector<uint64_t> packed_edges(local_M);
#endif

    FastGenerator gctx;
    gctx.init(p);

    // Pass 1: count degrees (and optionally cache packed edges)
    for (uint64_t i = edge_begin; i < edge_end; i += CHUNK) {
        uint64_t chunk_len = std::min(CHUNK, edge_end - i);
        for (uint64_t k = 0; k < chunk_len; k++) {
            gctx.generate(i + k, u[k], v[k], w_chunk[k]);
        }
        uint64_t base_idx = i - edge_begin;
        for (uint64_t k = 0; k < chunk_len; k++) {
            uint32_t uk = u[k], vk = v[k];
            uint8_t wk  = w_chunk[k];

            my_degree[uk]++;
            my_degree[vk]++;

#if CACHE_EDGES
            packed_edges[base_idx + k] = (static_cast<uint64_t>(uk) << 34) |
                                         (static_cast<uint64_t>(vk) << 8) |
                                         static_cast<uint64_t>(wk);
#endif
        }
    }

#if CACHE_EDGES
    std::vector<uint32_t>().swap(u);
    std::vector<uint32_t>().swap(v);
    std::vector<uint8_t>().swap(w_chunk);
#endif

    MPI_Win_sync(win);
    MPI_Barrier(shm_comm);
    MPI_Win_sync(win);

    double t_pass1 = MPI_Wtime() - t1;
    double t2 = MPI_Wtime();

    // Parallel Prefix Sum & Rank-Relative Offset calculation (Absolute Cursor)
    const uint64_t v_begin = (N * static_cast<uint64_t>(shm_rank)) / shm_size;
    const uint64_t v_end   = (N * static_cast<uint64_t>(shm_rank + 1)) / shm_size;

    uint64_t local_adjacency_count = 0;
    for (uint64_t vert = v_begin; vert < v_end; vert++) {
        uint32_t deg_v = 0;
        for (int r = 0; r < shm_size; r++) {
            deg_v += all_degree[static_cast<uint64_t>(r) * N + vert];
        }
        local_adjacency_count += deg_v;
    }

    uint64_t rank_base = 0;
    MPI_Exscan(&local_adjacency_count, &rank_base, 1, MPI_UINT64_T, MPI_SUM, shm_comm);
    if (shm_rank == 0) rank_base = 0;

    uint64_t running_row = rank_base;
    for (uint64_t vert = v_begin; vert < v_end; vert++) {
        row[vert] = running_row;
        uint32_t rel_offset = 0;
        for (int r = 0; r < shm_size; r++) {
            uint64_t idx = static_cast<uint64_t>(r) * N + vert;
            uint32_t d = all_degree[idx];
            all_degree[idx] = static_cast<uint32_t>(running_row + rel_offset);
            rel_offset += d;
        }
        running_row += rel_offset;
    }

    if (v_end == N) {
        row[N] = running_row;
    }

    MPI_Win_sync(win);
    MPI_Barrier(shm_comm);
    MPI_Win_sync(win);

    if (shm_rank == 0) {
        if (row[0] != 0 || row[N] != 2 * M) {
            std::cerr << "Assertion failed: row[0]=" << row[0]
                      << ", row[N]=" << row[N] << ", expected 2*M=" << 2 * M << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    double t_prefix = MPI_Wtime() - t2;
    double t3 = MPI_Wtime();

    // Pass 2: fill adj (lock-free) with Absolute Cursor
#if CACHE_EDGES
    for (uint64_t k = 0; k < local_M; k++) {
        uint64_t packed = packed_edges[k];
        uint32_t uk = static_cast<uint32_t>(packed >> 34);
        uint32_t vk = static_cast<uint32_t>((packed >> 8) & 0x3ffffffu);
        uint8_t wk  = static_cast<uint8_t>(packed);

        uint64_t pu = my_degree[uk]++;
        adj[pu] = {vk, wk};

        uint64_t pv = my_degree[vk]++;
        adj[pv] = {uk, wk};
    }
#else
    for (uint64_t i = edge_begin; i < edge_end; ++i) {
        uint32_t uk, vk;
        uint8_t wk;
        gctx.generate(i, uk, vk, wk);

        uint64_t pu = my_degree[uk]++;
        adj[pu] = {vk, wk};

        uint64_t pv = my_degree[vk]++;
        adj[pv] = {uk, wk};
    }
#endif

    MPI_Win_sync(win);
    MPI_Barrier(shm_comm);
    MPI_Win_sync(win);

#if CACHE_EDGES
    // Free packed_edges buffer before SSSP
    std::vector<uint64_t>().swap(packed_edges);
#endif

    double t_pass2 = MPI_Wtime() - t3;
    double t4 = MPI_Wtime();

    // 4. SSSP Execution
    const std::vector<uint32_t> samples = sample_vertices(p);
    SSSPStats sssp_stats{};

    if (use_coop) {
        // Multi-Team Cooperative Mode
        int team_id = world_rank % team_count;
        MPI_Comm team_comm;
        MPI_Comm_split(MPI_COMM_WORLD, team_id, world_rank, &team_comm);
        int team_rank, team_size;
        MPI_Comm_rank(team_comm, &team_rank);
        MPI_Comm_size(team_comm, &team_size);

        uint32_t *t_dist = coop_dist + static_cast<uint64_t>(team_id) * N;
        uint32_t *t_frontier = coop_frontier + static_cast<uint64_t>(team_id) * N;
        uint64_t *t_work_idx = coop_work_index + (static_cast<uint64_t>(team_id) * 8);

        const uint64_t t_v_begin = (N * static_cast<uint64_t>(team_rank)) / team_size;
        const uint64_t t_v_end   = (N * static_cast<uint64_t>(team_rank + 1)) / team_size;

        for (size_t s = team_id; s < S; s += team_count) {
#if DELTA_STEPPING > 1
            cooperative_delta_stepping(row, adj, N, sources[s], t_dist, t_frontier,
                                       t_work_idx, team_comm, team_rank, team_size, win, sssp_stats, DELTA_STEPPING);
#else
            cooperative_dial(row, adj, N, sources[s], t_dist, t_frontier,
                             t_work_idx, team_comm, team_rank, team_size, win, sssp_stats);
#endif

            // Parallel compute summary statistics for this source
            int64_t local_reached = 0;
            uint64_t local_checksum = 0;
            uint64_t local_digest1 = 0;
            uint64_t local_digest2 = 0;

            for (uint64_t vert = t_v_begin; vert < t_v_end; vert++) {
                if (t_dist[vert] != INF) {
                    local_reached++;
                    local_checksum += static_cast<uint64_t>(t_dist[vert]) * (vert + 1);
                    local_digest1 += distance_digest(static_cast<uint32_t>(vert), t_dist[vert], 0);
                    local_digest2 += distance_digest(static_cast<uint32_t>(vert), t_dist[vert], 1);
                }
            }

            int64_t tot_reached = 0;
            uint64_t tot_checksum = 0;
            uint64_t tot_digest1 = 0;
            uint64_t tot_digest2 = 0;

            MPI_Reduce(&local_reached, &tot_reached, 1, MPI_INT64_T, MPI_SUM, 0, team_comm);
            MPI_Reduce(&local_checksum, &tot_checksum, 1, MPI_UINT64_T, MPI_SUM, 0, team_comm);
            MPI_Reduce(&local_digest1, &tot_digest1, 1, MPI_UINT64_T, MPI_SUM, 0, team_comm);
            MPI_Reduce(&local_digest2, &tot_digest2, 1, MPI_UINT64_T, MPI_SUM, 0, team_comm);

            if (team_rank == 0) {
                res_hdr[s].n_reached = tot_reached;
                res_hdr[s].checksum  = tot_checksum;
                res_hdr[s].digest1   = tot_digest1;
                res_hdr[s].digest2   = tot_digest2;

                int32_t *my_sample = sample_dist_flat + s * p.nsample;
                for (int k = 0; k < p.nsample; k++) {
                    my_sample[k] = (t_dist[samples[k]] == INF) ? -1 : static_cast<int32_t>(t_dist[samples[k]]);
                }
            }
        }

        MPI_Comm_free(&team_comm);
    } else {
        // Standard Version 3 mode: source-level distribution
        std::vector<uint32_t> dist;
        DialQueue dial_queue;

        for (size_t s = world_rank; s < S; s += world_size) {
            dial_queue.clear();
            dial(row, adj, N, sources[s], dist, dial_queue);

            FlatResultHeader hdr{};
            hdr.n_reached = 0;
            hdr.checksum = 0;
            hdr.digest1 = 0;
            hdr.digest2 = 0;

            for (uint64_t vert = 0; vert < N; vert++) {
                if (dist[vert] != INF) {
                    hdr.n_reached++;
                    hdr.checksum += static_cast<uint64_t>(dist[vert]) * (vert + 1);
                    hdr.digest1 += distance_digest(static_cast<uint32_t>(vert), dist[vert], 0);
                    hdr.digest2 += distance_digest(static_cast<uint32_t>(vert), dist[vert], 1);
                }
            }

            res_hdr[s] = hdr;

            int32_t *my_sample = sample_dist_flat + s * p.nsample;
            for (int k = 0; k < p.nsample; k++) {
                my_sample[k] = (dist[samples[k]] == INF) ? -1 : static_cast<int32_t>(dist[samples[k]]);
            }
        }
    }

    MPI_Win_sync(win);
    MPI_Barrier(shm_comm);
    MPI_Win_sync(win);

    double t_sssp = MPI_Wtime() - t4;
    double t5 = MPI_Wtime();

    // 5. Rank 0 copies shared results to output vector
    if (world_rank == 0) {
        for (size_t s = 0; s < S; s++) {
            results[s].n_reached = res_hdr[s].n_reached;
            results[s].checksum  = res_hdr[s].checksum;
            results[s].digest1   = res_hdr[s].digest1;
            results[s].digest2   = res_hdr[s].digest2;

            int32_t *src_sample = sample_dist_flat + s * p.nsample;
            for (int k = 0; k < p.nsample; k++) {
                results[s].sample_dist[k] = src_sample[k];
            }
        }
    }

    double t_copy = MPI_Wtime() - t5;

    double times[6] = {t_alloc, t_pass1, t_prefix, t_pass2, t_sssp, t_copy};
    double max_times[6] = {0};
    MPI_Reduce(times, max_times, 6, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Collect SSSP diagnostic stats
    uint64_t max_rounds = 0;
    double max_coll_time = 0;
    double max_relax_time = 0;
    uint64_t total_front = 0;
    MPI_Reduce(&sssp_stats.rounds, &max_rounds, 1, MPI_UINT64_T, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sssp_stats.collective_time, &max_coll_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sssp_stats.relax_time, &max_relax_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sssp_stats.total_frontier, &total_front, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        std::printf("[Profile] Alloc: %.4fs | Pass1: %.4fs | Prefix: %.4fs | Pass2: %.4fs | (Build: %.4fs) | SSSP: %.4fs | Copy: %.4fs\n",
                    max_times[0], max_times[1], max_times[2], max_times[3],
                    max_times[0] + max_times[1] + max_times[2] + max_times[3],
                    max_times[4], max_times[5]);
        if (use_coop) {
            std::printf("[SSSP Stats] Rounds: %llu | Collectives: %.4fs | Relax: %.4fs | TotalFrontier: %llu\n",
                        (unsigned long long)max_rounds, max_coll_time, max_relax_time, (unsigned long long)total_front);
        }
    }

    // 6. Clean up MPI resources
    MPI_Win_free(&win);
    MPI_Comm_free(&shm_comm);
}
