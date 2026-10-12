#include <mpi.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include "graph_gen.hpp"

struct __attribute__((packed)) CSRNeighbor {
    uint32_t to;
    uint8_t w;
};

inline uint64_t pack_directed(uint32_t u, uint32_t v, uint8_t w) {
    return (static_cast<uint64_t>(u) << 34) | (static_cast<uint64_t>(v) << 8) | static_cast<uint64_t>(w);
}

inline void unpack_directed(uint64_t packed, uint32_t &u, uint32_t &v, uint8_t &w) {
    u = static_cast<uint32_t>(packed >> 34);
    v = static_cast<uint32_t>((packed >> 8) & 0x3ffffffu);
    w = static_cast<uint8_t>(packed);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    GraphParams p;
    p.scale = (argc >= 2) ? std::stoi(argv[1]) : 22;
    p.edgefactor = (argc >= 3) ? std::stoi(argv[2]) : 16;
    p.seed = 104;
    p.scramble = 1;
    p.nsample = 0;

    uint64_t N = num_vertices(p);
    uint64_t M = num_edges(p);

    if (rank == 0) {
        std::cout << "Testing Vertex-Owner CSR on Scale " << p.scale << ", EF " << p.edgefactor 
                  << ", M=" << M << " with " << size << " ranks..." << std::endl;
    }

    // Vertex ownership ranges
    uint64_t v_start = (N * static_cast<uint64_t>(rank)) / size;
    uint64_t v_end   = (N * static_cast<uint64_t>(rank + 1)) / size;
    uint64_t v_len   = v_end - v_start;

    // Shared window allocation
    MPI_Comm shm_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shm_comm);

    // Memory layout:
    // row: (N + 1) * 8 bytes
    // adj: 2 * M * 5 bytes
    // mailbox: (2 * M) * 8 bytes (divided into size owner segments)
    // mailbox_heads: size * 64 bytes (aligned atomic counters)
    uint64_t row_bytes = (N + 1) * sizeof(uint64_t);
    uint64_t adj_bytes = 2 * M * sizeof(CSRNeighbor);
    uint64_t mb_bytes  = 2 * M * sizeof(uint64_t);
    uint64_t head_bytes = size * 64;

    uint64_t total_shm = row_bytes + adj_bytes + mb_bytes + head_bytes;

    MPI_Win win;
    void *baseptr = nullptr;
    MPI_Aint sz = (rank == 0) ? total_shm : 0;
    MPI_Win_allocate_shared(sz, 1, MPI_INFO_NULL, shm_comm, &baseptr, &win);
    if (rank != 0) {
        MPI_Aint s;
        int d;
        MPI_Win_shared_query(win, 0, &s, &d, &baseptr);
    }

    char *ptr = static_cast<char *>(baseptr);
    uint64_t *row = reinterpret_cast<uint64_t *>(ptr); ptr += row_bytes;
    CSRNeighbor *adj = reinterpret_cast<CSRNeighbor *>(ptr); ptr += adj_bytes;
    uint64_t *mailbox = reinterpret_cast<uint64_t *>(ptr); ptr += mb_bytes;
    uint64_t *mailbox_heads = reinterpret_cast<uint64_t *>(ptr); ptr += head_bytes;

    // Mailbox layout: each owner gets a segment of mailbox of capacity cap_per_owner
    uint64_t cap_per_owner = (2 * M + size - 1) / size + 2000000; // safety margin for skew
    uint64_t *my_mailbox = mailbox + static_cast<uint64_t>(rank) * cap_per_owner;
    uint64_t *my_head = mailbox_heads + rank * 8;
    *my_head = 0;

    MPI_Win_sync(win);
    MPI_Barrier(shm_comm);
    MPI_Win_sync(win);

    double t0 = MPI_Wtime();

    // PHASE 1: Generate edges and route to owner mailboxes
    const uint64_t edge_begin = (M * static_cast<uint64_t>(rank)) / size;
    const uint64_t edge_end   = (M * static_cast<uint64_t>(rank + 1)) / size;

    constexpr size_t BIN_CAP = 1024;
    std::vector<std::vector<uint64_t>> producer_bins(size);
    for (int o = 0; o < size; ++o) producer_bins[o].reserve(BIN_CAP);

    auto flush_bin = [&](int o) {
        if (producer_bins[o].empty()) return;
        uint64_t cnt = producer_bins[o].size();
        uint64_t *head_ptr = mailbox_heads + o * 8;
        uint64_t pos = __atomic_fetch_add(head_ptr, cnt, __ATOMIC_RELAXED);
        uint64_t *dest = mailbox + static_cast<uint64_t>(o) * cap_per_owner + pos;
        std::memcpy(dest, producer_bins[o].data(), cnt * sizeof(uint64_t));
        producer_bins[o].clear();
    };

    auto get_owner = [&](uint32_t v) -> int {
        return static_cast<int>((static_cast<uint64_t>(v) * size) / N);
    };

    for (uint64_t i = edge_begin; i < edge_end; ++i) {
        uint32_t u, v;
        uint8_t w;
        generate_edge(p, i, u, v, w);

        int ou = get_owner(u);
        producer_bins[ou].push_back(pack_directed(u, v, w));
        if (producer_bins[ou].size() >= BIN_CAP) flush_bin(ou);

        int ov = get_owner(v);
        producer_bins[ov].push_back(pack_directed(v, u, w));
        if (producer_bins[ov].size() >= BIN_CAP) flush_bin(ov);
    }
    for (int o = 0; o < size; ++o) flush_bin(o);

    MPI_Win_sync(win);
    MPI_Barrier(shm_comm);
    MPI_Win_sync(win);

    double t_shuffle = MPI_Wtime() - t0;
    double t1 = MPI_Wtime();

    // PHASE 2: Local owner degree counting and local row calculation
    uint64_t num_incoming = *my_head;
    std::vector<uint32_t> local_deg(v_len, 0);
    for (uint64_t i = 0; i < num_incoming; ++i) {
        uint32_t u, v;
        uint8_t w;
        unpack_directed(my_mailbox[i], u, v, w);
        local_deg[u - v_start]++;
    }

    // Local prefix sum to get local row slice length
    uint64_t my_slice_len = 0;
    for (uint64_t v = 0; v < v_len; ++v) {
        my_slice_len += local_deg[v];
    }

    // Exscan across ranks to get starting offset of this rank's slice in global adj
    uint64_t my_adj_start = 0;
    MPI_Exscan(&my_slice_len, &my_adj_start, 1, MPI_UINT64_T, MPI_SUM, shm_comm);
    if (rank == 0) my_adj_start = 0;

    // Fill local portion of row array
    uint64_t running = my_adj_start;
    std::vector<uint64_t> local_cursor(v_len);
    for (uint64_t v = 0; v < v_len; ++v) {
        row[v_start + v] = running;
        local_cursor[v] = running;
        running += local_deg[v];
    }
    if (rank == size - 1) {
        row[N] = running;
    }

    MPI_Win_sync(win);
    MPI_Barrier(shm_comm);
    MPI_Win_sync(win);

    double t_prefix = MPI_Wtime() - t1;
    double t2 = MPI_Wtime();

    // PHASE 3: Owner fills adj with streaming writes
    for (uint64_t i = 0; i < num_incoming; ++i) {
        uint32_t u, v;
        uint8_t w;
        unpack_directed(my_mailbox[i], u, v, w);
        uint64_t pos = local_cursor[u - v_start]++;
        adj[pos] = {v, w};
    }

    MPI_Win_sync(win);
    MPI_Barrier(shm_comm);
    MPI_Win_sync(win);

    double t_scatter = MPI_Wtime() - t2;
    double t_total = MPI_Wtime() - t0;

    if (rank == 0) {
        std::cout << "[Vertex-Owner] Shuffle: " << t_shuffle << "s | Prefix: " << t_prefix 
                  << "s | Scatter: " << t_scatter << "s | Total Build: " << t_total << "s" << std::endl;
        std::cout << "Verification: row[0]=" << row[0] << ", row[N]=" << row[N] << ", expected=" << 2*M << std::endl;
    }

    MPI_Win_free(&win);
    MPI_Comm_free(&shm_comm);
    MPI_Finalize();
    return 0;
}
