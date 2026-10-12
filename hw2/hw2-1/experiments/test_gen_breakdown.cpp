#include <iostream>
#include <vector>
#include <chrono>
#include <cstdint>
#include <algorithm>
#include <iomanip>

inline uint64_t mix_hash(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

struct FastGen {
    uint64_t mixed_seed;
    uint64_t mixed_seed_w;
    uint64_t scramble_k1, scramble_k2, scramble_k3;
    uint64_t scramble_mask;
    int scramble_sh;
    int scale;
    bool scramble;

    void init(int scale_in, bool scramble_in, uint64_t seed = 110) {
        scale = scale_in;
        scramble = scramble_in;
        mixed_seed = mix_hash(seed);
        mixed_seed_w = mix_hash(seed ^ 0x77e19470ULL);

        uint64_t s_seed = seed ^ 0x5c2a3b1eULL;
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
};

int main(int argc, char **argv) {
    int scale = (argc >= 2) ? std::stoi(argv[1]) : 26;
    bool scramble = (argc >= 3) ? (std::stoi(argv[2]) != 0) : true;
    uint64_t N_edges = (argc >= 4) ? std::stoull(argv[3]) : 10000000ULL; // 10M edges

    FastGen gen;
    gen.init(scale, scramble);

    std::cout << "=================================================================" << std::endl;
    std::cout << "  [MILESTONE G1: GENERATOR COMPONENT BREAKDOWN]" << std::endl;
    std::cout << "  Scale=" << scale << ", Scramble=" << scramble << ", Edges=" << N_edges << std::endl;
    std::cout << "=================================================================" << std::endl;

    // Buffer allocations for isolated stages
    std::vector<uint64_t> states(N_edges);
    std::vector<uint32_t> u_vec(N_edges), v_vec(N_edges);
    std::vector<uint8_t>  w_vec(N_edges);
    std::vector<uint64_t> packed(N_edges);
    uint64_t V_size = 1ULL << scale;
    std::vector<uint32_t> degrees(std::min<uint64_t>(V_size, 16777216ULL), 0); // 64 MB degree array

    // Phase 1: Pure Initial Mix Hash
    auto t0 = std::chrono::high_resolution_clock::now();
    uint64_t dummy = 0;
    for (uint64_t i = 0; i < N_edges; ++i) {
        states[i] = mix_hash(gen.mixed_seed ^ (i * 0xd6e8feb86659fd93ULL));
        dummy += states[i];
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double time_hash = std::chrono::duration<double>(t1 - t0).count();

    // Phase 2: Kronecker Descent only
    auto t2 = std::chrono::high_resolution_clock::now();
    for (uint64_t i = 0; i < N_edges; ++i) {
        uint64_t state = states[i];
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
        u_vec[i] = a;
        v_vec[i] = b;
        dummy += a + b;
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double time_descent = std::chrono::duration<double>(t3 - t2).count();

    // Phase 3: Scramble ID only
    double time_scramble = 0.0;
    if (scramble) {
        auto t4 = std::chrono::high_resolution_clock::now();
        for (uint64_t i = 0; i < N_edges; ++i) {
            u_vec[i] = gen.scramble_id(u_vec[i]);
            v_vec[i] = gen.scramble_id(v_vec[i]);
            dummy += u_vec[i] + v_vec[i];
        }
        auto t5 = std::chrono::high_resolution_clock::now();
        time_scramble = std::chrono::duration<double>(t5 - t4).count();
    }

    // Phase 4: Weight Generation
    auto t6 = std::chrono::high_resolution_clock::now();
    for (uint64_t i = 0; i < N_edges; ++i) {
        uint32_t a = u_vec[i], b = v_vec[i];
        uint64_t lo = std::min(a, b), hi = std::max(a, b);
        uint64_t pair_key = (hi << 32) | lo;
        uint64_t h = mix_hash(gen.mixed_seed_w ^ (pair_key * 0xd6e8feb86659fd93ULL));
        w_vec[i] = static_cast<uint8_t>(1 + (h % 255));
        dummy += w_vec[i];
    }
    auto t7 = std::chrono::high_resolution_clock::now();
    double time_weight = std::chrono::duration<double>(t7 - t6).count();

    // Phase 5: 60-bit Edge Packing
    auto t8 = std::chrono::high_resolution_clock::now();
    for (uint64_t i = 0; i < N_edges; ++i) {
        packed[i] = (static_cast<uint64_t>(u_vec[i]) << 34) |
                    (static_cast<uint64_t>(v_vec[i]) << 8) | w_vec[i];
        dummy += packed[i];
    }
    auto t9 = std::chrono::high_resolution_clock::now();
    double time_pack = std::chrono::duration<double>(t9 - t8).count();

    // Phase 6: Degree Update
    uint64_t deg_mask = degrees.size() - 1;
    auto t10 = std::chrono::high_resolution_clock::now();
    for (uint64_t i = 0; i < N_edges; ++i) {
        degrees[u_vec[i] & deg_mask]++;
        degrees[v_vec[i] & deg_mask]++;
    }
    auto t11 = std::chrono::high_resolution_clock::now();
    double time_degree = std::chrono::duration<double>(t11 - t10).count();

    // Integrated Full Pipeline (Current Pass 1 loop)
    std::fill(degrees.begin(), degrees.end(), 0);
    auto t12 = std::chrono::high_resolution_clock::now();
    for (uint64_t i = 0; i < N_edges; ++i) {
        uint64_t state = mix_hash(gen.mixed_seed ^ (i * 0xd6e8feb86659fd93ULL));
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
            a = gen.scramble_id(a);
            b = gen.scramble_id(b);
        }

        uint64_t lo = std::min(a, b), hi = std::max(a, b);
        uint64_t pair_key = (hi << 32) | lo;
        uint64_t h = mix_hash(gen.mixed_seed_w ^ (pair_key * 0xd6e8feb86659fd93ULL));
        uint8_t w = static_cast<uint8_t>(1 + (h % 255));

        packed[i] = (static_cast<uint64_t>(a) << 34) | (static_cast<uint64_t>(b) << 8) | w;
        degrees[a & deg_mask]++;
        degrees[b & deg_mask]++;
    }
    auto t13 = std::chrono::high_resolution_clock::now();
    double time_integrated = std::chrono::duration<double>(t13 - t12).count();

    double total_isolated = time_hash + time_descent + time_scramble + time_weight + time_pack + time_degree;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  1. Initial Mix Hash:      " << time_hash << " s (" << (100.0 * time_hash / total_isolated) << "%)\n";
    std::cout << "  2. Kronecker Descent:     " << time_descent << " s (" << (100.0 * time_descent / total_isolated) << "%)\n";
    std::cout << "  3. Scramble ID:           " << time_scramble << " s (" << (100.0 * time_scramble / total_isolated) << "%)\n";
    std::cout << "  4. Weight Generation:     " << time_weight << " s (" << (100.0 * time_weight / total_isolated) << "%)\n";
    std::cout << "  5. 60-bit Edge Packing:   " << time_pack << " s (" << (100.0 * time_pack / total_isolated) << "%)\n";
    std::cout << "  6. Degree Update:         " << time_degree << " s (" << (100.0 * time_degree / total_isolated) << "%)\n";
    std::cout << "-----------------------------------------------------------------\n";
    std::cout << "  Sum of Isolated Phases:   " << total_isolated << " s (" << (N_edges / total_isolated / 1e6) << " M edges/s)\n";
    std::cout << "  Current Integrated Loop:  " << time_integrated << " s (" << (N_edges / time_integrated / 1e6) << " M edges/s)\n";
    std::cout << "=================================================================\n";

    return (dummy == 0x12345678ULL) ? 1 : 0;
}
