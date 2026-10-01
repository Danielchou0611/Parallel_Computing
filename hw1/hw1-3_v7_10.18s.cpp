#include <iostream>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <sys/mman.h>
#ifdef __linux__
#include <sched.h>
#endif

// ========== START: DO NOT CHANGE BELOW ==========
static const double R = 0.5;    // the reaction's strength
static const int SUBSTEPS = 4;  // sub-steps of the reaction per time step
static const int NEWTON = 3;    // Newton iterations per sub-step

static double field(uint64_t seed, uint64_t index, int which) {
    uint64_t x = (index * 2 + which) ^ (seed * 0x9E3779B97F4A7C15ull);
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    x ^= x >> 31;
    return (x >> 11) * (1.0 / 9007199254740992.0);
}

struct Inclusion {
    double ci, cj, ck, r2;
};
static int inclusions(uint64_t seed, long N, Inclusion* out) {
    const int B = (int)(seed % 5);
    const uint64_t base = (uint64_t)N * N * N;
    for (int b = 0; b < B; b++) {
        out[b].ci = 1 + (0.2 + 0.6 * field(seed, base + 2 * b, 0)) * (N - 1);
        out[b].cj = 1 + (0.2 + 0.6 * field(seed, base + 2 * b, 1)) * (N - 1);
        out[b].ck = 1 + (0.2 + 0.6 * field(seed, base + 2 * b + 1, 0)) * (N - 1);
        const double r = (0.12 + 0.06 * field(seed, base + 2 * b + 1, 1)) * N;
        out[b].r2 = r * r;
    }
    return B;
}

static bool reactive(const Inclusion* inc, int B, long i, long j, long k) {
    for (int b = 0; b < B; b++) {
        const double di = i - inc[b].ci, dj = j - inc[b].cj, dk = k - inc[b].ck;
        if (di * di + dj * dj + dk * dk <= inc[b].r2) return true;
    }
    return false;
}

static double react(double r) {
    double x = r;
    for (int s = 0; s < SUBSTEPS; s++) {
        const double prev = x;
        for (int n = 0; n < NEWTON; n++) {
            const double e = exp(x);
            x -= (x + R * (e - 1.0) - prev) / (1.0 + R * e);
        }
    }
    return x;
}
// =========== END: DO NOT CHANGE ABOVE ===========

struct RowSphere {
    double ck;
    double rem_r2;
};

struct KInterval {
    long lo, hi;
};

static inline int build_reaction_intervals(const RowSphere* act, int n_act, long N,
                                           KInterval* intervals) {
    int n = 0;
    for (int b = 0; b < n_act; b++) {
        const double half = sqrt(act[b].rem_r2);
        long lo = (long)floor(act[b].ck - half);
        long hi = (long)ceil(act[b].ck + half);
        if (lo < 1) lo = 1;
        if (hi > N) hi = N;
        while (lo <= hi) {
            const double dk = lo - act[b].ck;
            if (dk * dk <= act[b].rem_r2) break;
            lo++;
        }
        while (hi >= lo) {
            const double dk = hi - act[b].ck;
            if (dk * dk <= act[b].rem_r2) break;
            hi--;
        }
        if (lo > hi) continue;

        int pos = n;
        while (pos > 0 && intervals[pos - 1].lo > lo) {
            intervals[pos] = intervals[pos - 1];
            pos--;
        }
        intervals[pos] = {lo, hi};
        n++;
    }

    int merged = 0;
    for (int i = 0; i < n; i++) {
        if (merged == 0 || intervals[i].lo > intervals[merged - 1].hi + 1) {
            intervals[merged++] = intervals[i];
        } else if (intervals[i].hi > intervals[merged - 1].hi) {
            intervals[merged - 1].hi = intervals[i].hi;
        }
    }
    return merged;
}

// ---------------------------------------------------------------- low memory slab
static void gen_a_slab(double* __restrict__ dst, long i, long j0, long j1,
                       long N, long SJ, uint64_t seed) {
    const long rows = j1 - j0 + 3;
    if (i < 1 || i > N) {
        memset(dst, 0, (size_t)rows * SJ * sizeof(double));
        return;
    }
    for (long t = 0; t < rows; t++) {
        const long j = j0 - 1 + t;
        double* row = dst + t * SJ;
        if (j < 1 || j > N) {
            memset(row, 0, (size_t)SJ * sizeof(double));
            continue;
        }
        row[0] = 0.0;
        row[N + 1] = 0.0;
        const uint64_t base = ((uint64_t)(i - 1) * N + (j - 1)) * N;
        #pragma GCC ivdep
        for (long k = 1; k <= N; k++) row[k] = field(seed, base + (k - 1), 1);
    }
}

template<bool NEED_ENERGY>
static double lowmem_tile(const double* __restrict__ cur_u,
                          double* __restrict__ cur_unew,
                          long j0, long j1, long N, long SI, long SJ,
                          uint64_t seed, const Inclusion* inc, int B,
                          const char* slice_has_sphere, double* slab) {
    const long rows = j1 - j0 + 3;
    const size_t plane = (size_t)rows * SJ;
    double* ap[3] = {slab, slab + plane, slab + 2 * plane};

    gen_a_slab(ap[0], 0, j0, j1, N, SJ, seed);
    gen_a_slab(ap[1], 1, j0, j1, N, SJ, seed);
    gen_a_slab(ap[2], 2, j0, j1, N, SJ, seed);

    double energy = 0.0;

    for (long i = 1; i <= N; i++) {
        const long i_SI = i * SI, im_SI = (i - 1) * SI, ip_SI = (i + 1) * SI;
        const bool sph = (B > 0) && slice_has_sphere[i];

        if (!sph) {
            for (long j = j0; j <= j1; j++) {
                const long t = j - j0 + 1;
                const long j_SJ = j * SJ, jm_SJ = (j - 1) * SJ, jp_SJ = (j + 1) * SJ;

                const double* __restrict__ uc  = (const double*)__builtin_assume_aligned(&cur_u[i_SI + j_SJ], 64);
                const double* __restrict__ uim = (const double*)__builtin_assume_aligned(&cur_u[im_SI + j_SJ], 64);
                const double* __restrict__ uip = (const double*)__builtin_assume_aligned(&cur_u[ip_SI + j_SJ], 64);
                const double* __restrict__ ujm = (const double*)__builtin_assume_aligned(&cur_u[i_SI + jm_SJ], 64);
                const double* __restrict__ ujp = (const double*)__builtin_assume_aligned(&cur_u[i_SI + jp_SJ], 64);

                const double* __restrict__ ac  = (const double*)__builtin_assume_aligned(ap[1] + t * SJ, 64);
                const double* __restrict__ aim = (const double*)__builtin_assume_aligned(ap[0] + t * SJ, 64);
                const double* __restrict__ aip = (const double*)__builtin_assume_aligned(ap[2] + t * SJ, 64);
                const double* __restrict__ ajm = (const double*)__builtin_assume_aligned(ap[1] + (t - 1) * SJ, 64);
                const double* __restrict__ ajp = (const double*)__builtin_assume_aligned(ap[1] + (t + 1) * SJ, 64);

                double* __restrict__ unew_row = (double*)__builtin_assume_aligned(&cur_unew[i_SI + j_SJ], 64);

                #pragma GCC ivdep
                #pragma GCC unroll 4
                for (long k = 1; k <= N; k++) {
                    const double up = uc[k], apv = ac[k];
                    const double flux = (apv + aim[k]) * (uim[k] - up)
                                      + (apv + aip[k]) * (uip[k] - up)
                                      + (apv + ajm[k]) * (ujm[k] - up)
                                      + (apv + ajp[k]) * (ujp[k] - up)
                                      + (apv + ac[k - 1]) * (uc[k - 1] - up)
                                      + (apv + ac[k + 1]) * (uc[k + 1] - up);
                    const double val = up + flux * (1.0 / 12.0);
                    unew_row[k] = val;
                    if (NEED_ENERGY) energy += val * val;
                }
            }
        } else {
            for (long j = j0; j <= j1; j++) {
                const long t = j - j0 + 1;
                const long j_SJ = j * SJ, jm_SJ = (j - 1) * SJ, jp_SJ = (j + 1) * SJ;

                const double* __restrict__ uc  = (const double*)__builtin_assume_aligned(&cur_u[i_SI + j_SJ], 64);
                const double* __restrict__ uim = (const double*)__builtin_assume_aligned(&cur_u[im_SI + j_SJ], 64);
                const double* __restrict__ uip = (const double*)__builtin_assume_aligned(&cur_u[ip_SI + j_SJ], 64);
                const double* __restrict__ ujm = (const double*)__builtin_assume_aligned(&cur_u[i_SI + jm_SJ], 64);
                const double* __restrict__ ujp = (const double*)__builtin_assume_aligned(&cur_u[i_SI + jp_SJ], 64);

                const double* __restrict__ ac  = (const double*)__builtin_assume_aligned(ap[1] + t * SJ, 64);
                const double* __restrict__ aim = (const double*)__builtin_assume_aligned(ap[0] + t * SJ, 64);
                const double* __restrict__ aip = (const double*)__builtin_assume_aligned(ap[2] + t * SJ, 64);
                const double* __restrict__ ajm = (const double*)__builtin_assume_aligned(ap[1] + (t - 1) * SJ, 64);
                const double* __restrict__ ajp = (const double*)__builtin_assume_aligned(ap[1] + (t + 1) * SJ, 64);

                double* __restrict__ unew_row = (double*)__builtin_assume_aligned(&cur_unew[i_SI + j_SJ], 64);

                RowSphere act[4];
                int n_act = 0;
                for (int b = 0; b < B; b++) {
                    const double di = i - inc[b].ci, dj = j - inc[b].cj;
                    const double d2 = di * di + dj * dj;
                    if (d2 <= inc[b].r2) {
                        act[n_act].ck = inc[b].ck;
                        act[n_act].rem_r2 = inc[b].r2 - d2;
                        n_act++;
                    }
                }

                if (n_act == 0) {
                    #pragma GCC ivdep
                    for (long k = 1; k <= N; k++) {
                        const double up = uc[k], apv = ac[k];
                        const double flux = (apv + aim[k]) * (uim[k] - up)
                                          + (apv + aip[k]) * (uip[k] - up)
                                          + (apv + ajm[k]) * (ujm[k] - up)
                                          + (apv + ajp[k]) * (ujp[k] - up)
                                          + (apv + ac[k - 1]) * (uc[k - 1] - up)
                                          + (apv + ac[k + 1]) * (uc[k + 1] - up);
                        const double val = up + flux * (1.0 / 12.0);
                        unew_row[k] = val;
                        if (NEED_ENERGY) energy += val * val;
                    }
                } else {
                    KInterval intervals[4];
                    const int n_intervals = build_reaction_intervals(act, n_act, N, intervals);
                    long cur_k = 1;
                    for (int iv = 0; iv < n_intervals; iv++) {
                        if (intervals[iv].lo > cur_k) {
                            #pragma GCC ivdep
                            #pragma GCC unroll 4
                            for (long k = cur_k; k < intervals[iv].lo; k++) {
                                const double up = uc[k], apv = ac[k];
                                const double flux = (apv + aim[k]) * (uim[k] - up)
                                                  + (apv + aip[k]) * (uip[k] - up)
                                                  + (apv + ajm[k]) * (ujm[k] - up)
                                                  + (apv + ajp[k]) * (ujp[k] - up)
                                                  + (apv + ac[k - 1]) * (uc[k - 1] - up)
                                                  + (apv + ac[k + 1]) * (uc[k + 1] - up);
                                const double val = up + flux * (1.0 / 12.0);
                                unew_row[k] = val;
                                if (NEED_ENERGY) energy += val * val;
                            }
                        }
                        for (long k = intervals[iv].lo; k <= intervals[iv].hi; k++) {
                            const double up = uc[k], apv = ac[k];
                            const double flux = (apv + aim[k]) * (uim[k] - up)
                                              + (apv + aip[k]) * (uip[k] - up)
                                              + (apv + ajm[k]) * (ujm[k] - up)
                                              + (apv + ajp[k]) * (ujp[k] - up)
                                              + (apv + ac[k - 1]) * (uc[k - 1] - up)
                                              + (apv + ac[k + 1]) * (uc[k + 1] - up);
                            const double r = up + flux * (1.0 / 12.0);
                            const double val = react(r);
                            unew_row[k] = val;
                            if (NEED_ENERGY) energy += val * val;
                        }
                        cur_k = intervals[iv].hi + 1;
                    }
                    if (cur_k <= N) {
                        #pragma GCC ivdep
                        #pragma GCC unroll 4
                        for (long k = cur_k; k <= N; k++) {
                            const double up = uc[k], apv = ac[k];
                            const double flux = (apv + aim[k]) * (uim[k] - up)
                                              + (apv + aip[k]) * (uip[k] - up)
                                              + (apv + ajm[k]) * (ujm[k] - up)
                                              + (apv + ajp[k]) * (ujp[k] - up)
                                              + (apv + ac[k - 1]) * (uc[k - 1] - up)
                                              + (apv + ac[k + 1]) * (uc[k + 1] - up);
                            const double val = up + flux * (1.0 / 12.0);
                            unew_row[k] = val;
                            if (NEED_ENERGY) energy += val * val;
                        }
                    }
                }
            }
        }

        double* tmp = ap[0]; ap[0] = ap[1]; ap[1] = ap[2]; ap[2] = tmp;
        gen_a_slab(ap[2], i + 2, j0, j1, N, SJ, seed);
    }
    return energy;
}

int main(int argc, char** argv) {
    if (argc != 6) return 1;

#ifdef __linux__
    if (getenv("OMP_NUM_THREADS") == nullptr) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        if (sched_getaffinity(0, sizeof(cpuset), &cpuset) == 0) {
            int ncpus = CPU_COUNT(&cpuset);
            if (ncpus < 1) ncpus = 1;
            if (ncpus > 8) ncpus = 8;
            omp_set_num_threads(ncpus);
        }
    }
#endif


    const long N = atol(argv[1]);
    const int T = atoi(argv[2]);
    const uint64_t seed = strtoull(argv[3], nullptr, 10);
    const double theta = atof(argv[4]);

    const long M = N + 2;
    const long SJ = (M + 7) & ~7L;
    const long SI = M * SJ;
    const size_t total_elements = (size_t)M * SI;
    const size_t total_bytes = total_elements * sizeof(double);

    const bool store_a = (N <= 384);

    double* u = (double*)mmap(nullptr, total_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    double* unew = (double*)mmap(nullptr, total_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    double* a = store_a ? (double*)mmap(nullptr, total_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)
                        : nullptr;
    if (u == MAP_FAILED || unew == MAP_FAILED || (store_a && a == MAP_FAILED)) {
        fprintf(stderr, "out of memory for N = %ld\n", N);
        return 1;
    }
    madvise(u, total_bytes, MADV_HUGEPAGE);
    madvise(unew, total_bytes, MADV_HUGEPAGE);
    if (store_a) madvise(a, total_bytes, MADV_HUGEPAGE);

    Inclusion inc[4];
    const int B = inclusions(seed, N, inc);
    double energy0 = 0.0;

    std::vector<char> slice_has_sphere(M, 0);
    if (B > 0) {
        for (long i = 1; i <= N; i++) {
            for (int b = 0; b < B; b++) {
                const double di = i - inc[b].ci;
                if (di * di <= inc[b].r2) {
                    slice_has_sphere[i] = 1;
                    break;
                }
            }
        }
    }

#pragma omp parallel for schedule(static) reduction(+:energy0)
    for (long i = 1; i <= N; i++) {
        const long i_SI = i * SI;
        for (long j = 1; j <= N; j++) {
            const long j_SJ = j * SJ;
            double* u_row = &u[i_SI + j_SJ];
            const uint64_t base_idx = ((uint64_t)(i - 1) * N + (j - 1)) * N;
            if (store_a) {
                double* a_row = &a[i_SI + j_SJ];
                #pragma GCC ivdep
                #pragma GCC unroll 4
                for (long k = 1; k <= N; k++) {
                    const uint64_t index = base_idx + (k - 1);
                    const double u_val = field(seed, index, 0);
                    u_row[k] = u_val;
                    a_row[k] = field(seed, index, 1);
                    energy0 += u_val * u_val;
                }
            } else {
                #pragma GCC ivdep
                #pragma GCC unroll 4
                for (long k = 1; k <= N; k++) {
                    const double u_val = field(seed, base_idx + (k - 1), 0);
                    u_row[k] = u_val;
                    energy0 += u_val * u_val;
                }
            }
        }
    }

    double energy = energy0;
    int steps = 0;
    const double threshold = theta * energy0;
    double step_energy = 0.0;
    const int BJ = 12;
    const int BJ_STORED = (N <= 128) ? 16 : ((N <= 256) ? 32 : 24);
    if (!store_a) {
        const long ntiles = (N + BJ - 1) / BJ;
        if (threshold <= 0.0) {
            double final_energy = 0.0;
#pragma omp parallel
            {
                std::vector<double> slab(3 * (size_t)(BJ + 2) * SJ);
                double* cur_u = u;
                double* cur_unew = unew;
                double thread_final_energy = 0.0;

                for (int s = 0; s < T; s++) {
                    const bool is_last = (s == T - 1);
                    if (is_last) {
#pragma omp for schedule(static, 1)
                        for (long jt = 0; jt < ntiles; jt++) {
                            const long j0 = 1 + jt * BJ;
                            const long j1 = std::min(N, j0 + BJ - 1);
                            thread_final_energy += lowmem_tile<true>(
                                cur_u, cur_unew, j0, j1, N, SI, SJ, seed,
                                inc, B, slice_has_sphere.data(), slab.data());
                        }
                    } else {
#pragma omp for schedule(static, 1)
                        for (long jt = 0; jt < ntiles; jt++) {
                            const long j0 = 1 + jt * BJ;
                            const long j1 = std::min(N, j0 + BJ - 1);
                            lowmem_tile<false>(
                                cur_u, cur_unew, j0, j1, N, SI, SJ, seed,
                                inc, B, slice_has_sphere.data(), slab.data());
                        }
                    }
                    std::swap(cur_u, cur_unew);
                }

#pragma omp atomic
                final_energy += thread_final_energy;
            }
            if (T % 2 == 1) std::swap(u, unew);
            steps = T;
            energy = (T == 0) ? energy0 : final_energy;
        } else {
#pragma omp parallel
            {
                std::vector<double> slab(3 * (size_t)(BJ + 2) * SJ);
                double* cur_u = u;
                double* cur_unew = unew;

                while (steps < T) {
                    double te = 0.0;
#pragma omp for schedule(static, 1) nowait
                    for (long jt = 0; jt < ntiles; jt++) {
                        const long j0 = 1 + jt * BJ;
                        const long j1 = std::min(N, j0 + BJ - 1);
                        te += lowmem_tile<true>(
                            cur_u, cur_unew, j0, j1, N, SI, SJ, seed,
                            inc, B, slice_has_sphere.data(), slab.data());
                    }
#pragma omp atomic
                    step_energy += te;

#pragma omp barrier
#pragma omp single
                    {
                        steps++;
                        energy = step_energy;
                        step_energy = 0.0;
                    }
                    std::swap(cur_u, cur_unew);
                    if (energy <= threshold) break;
                }
            }
            if (steps % 2 == 1) std::swap(u, unew);
        }
    } else if (threshold <= 0.0) {
        // High-efficiency stored_a for N <= 256 with threshold <= 0
        double final_energy = 0.0;
#pragma omp parallel
        {
            double* cur_u = u;
            double* cur_unew = unew;
            double thread_final_energy = 0.0;

            if (B == 0) {
                for (int s = 0; s < T; s++) {
                    const bool is_last = (s == T - 1);
#pragma omp for schedule(static)
                    for (long j_start = 1; j_start <= N; j_start += BJ_STORED) {
                        const long j_end = std::min(N, j_start + BJ_STORED - 1);
                        for (long i = 1; i <= N; i++) {
                            const long i_SI = i * SI;
                            const long im_SI = (i - 1) * SI;
                            const long ip_SI = (i + 1) * SI;

                            for (long j = j_start; j <= j_end; j++) {
                                const long j_SJ = j * SJ;
                                const long jm_SJ = (j - 1) * SJ;
                                const long jp_SJ = (j + 1) * SJ;

                                const double* __restrict__ uc  = (const double*)__builtin_assume_aligned(&cur_u[i_SI + j_SJ], 64);
                                const double* __restrict__ uim = (const double*)__builtin_assume_aligned(&cur_u[im_SI + j_SJ], 64);
                                const double* __restrict__ uip = (const double*)__builtin_assume_aligned(&cur_u[ip_SI + j_SJ], 64);
                                const double* __restrict__ ujm = (const double*)__builtin_assume_aligned(&cur_u[i_SI + jm_SJ], 64);
                                const double* __restrict__ ujp = (const double*)__builtin_assume_aligned(&cur_u[i_SI + jp_SJ], 64);

                                const double* __restrict__ ac  = (const double*)__builtin_assume_aligned(&a[i_SI + j_SJ], 64);
                                const double* __restrict__ aim = (const double*)__builtin_assume_aligned(&a[im_SI + j_SJ], 64);
                                const double* __restrict__ aip = (const double*)__builtin_assume_aligned(&a[ip_SI + j_SJ], 64);
                                const double* __restrict__ ajm = (const double*)__builtin_assume_aligned(&a[i_SI + jm_SJ], 64);
                                const double* __restrict__ ajp = (const double*)__builtin_assume_aligned(&a[i_SI + jp_SJ], 64);

                                double* __restrict__ unew_row = (double*)__builtin_assume_aligned(&cur_unew[i_SI + j_SJ], 64);

                                #pragma GCC ivdep
                                for (long k = 1; k <= N; k++) {
                                    const double up = uc[k], ap = ac[k];
                                    const double flux = (ap + aim[k]) * (uim[k] - up)
                                                      + (ap + aip[k]) * (uip[k] - up)
                                                      + (ap + ajm[k]) * (ujm[k] - up)
                                                      + (ap + ajp[k]) * (ujp[k] - up)
                                                      + (ap + ac[k - 1]) * (uc[k - 1] - up)
                                                      + (ap + ac[k + 1]) * (uc[k + 1] - up);
                                    const double val = up + flux * (1.0 / 12.0);
                                    unew_row[k] = val;
                                    if (is_last) thread_final_energy += val * val;
                                }
                            }
                        }
                    }
                    std::swap(cur_u, cur_unew);
                }
            } else {
                for (int s = 0; s < T; s++) {
                    const bool is_last = (s == T - 1);
#pragma omp for schedule(static, 1)
                    for (long i = 1; i <= N; i++) {
                        const long i_SI = i * SI;
                        const long im_SI = (i - 1) * SI;
                        const long ip_SI = (i + 1) * SI;

                        if (!slice_has_sphere[i]) {
                            for (long j = 1; j <= N; j++) {
                                const long j_SJ = j * SJ;
                                const long jm_SJ = (j - 1) * SJ;
                                const long jp_SJ = (j + 1) * SJ;

                                const double* __restrict__ uc  = (const double*)__builtin_assume_aligned(&cur_u[i_SI + j_SJ], 64);
                                const double* __restrict__ uim = (const double*)__builtin_assume_aligned(&cur_u[im_SI + j_SJ], 64);
                                const double* __restrict__ uip = (const double*)__builtin_assume_aligned(&cur_u[ip_SI + j_SJ], 64);
                                const double* __restrict__ ujm = (const double*)__builtin_assume_aligned(&cur_u[i_SI + jm_SJ], 64);
                                const double* __restrict__ ujp = (const double*)__builtin_assume_aligned(&cur_u[i_SI + jp_SJ], 64);

                                const double* __restrict__ ac  = (const double*)__builtin_assume_aligned(&a[i_SI + j_SJ], 64);
                                const double* __restrict__ aim = (const double*)__builtin_assume_aligned(&a[im_SI + j_SJ], 64);
                                const double* __restrict__ aip = (const double*)__builtin_assume_aligned(&a[ip_SI + j_SJ], 64);
                                const double* __restrict__ ajm = (const double*)__builtin_assume_aligned(&a[i_SI + jm_SJ], 64);
                                const double* __restrict__ ajp = (const double*)__builtin_assume_aligned(&a[i_SI + jp_SJ], 64);

                                double* __restrict__ unew_row = (double*)__builtin_assume_aligned(&cur_unew[i_SI + j_SJ], 64);

                                #pragma GCC ivdep
                                for (long k = 1; k <= N; k++) {
                                    const double up = uc[k], ap = ac[k];
                                    const double flux = (ap + aim[k]) * (uim[k] - up)
                                                      + (ap + aip[k]) * (uip[k] - up)
                                                      + (ap + ajm[k]) * (ujm[k] - up)
                                                      + (ap + ajp[k]) * (ujp[k] - up)
                                                      + (ap + ac[k - 1]) * (uc[k - 1] - up)
                                                      + (ap + ac[k + 1]) * (uc[k + 1] - up);
                                    const double val = up + flux * (1.0 / 12.0);
                                    unew_row[k] = val;
                                    if (is_last) thread_final_energy += val * val;
                                }
                            }
                        } else {
                            for (long j = 1; j <= N; j++) {
                                RowSphere act_spheres[4];
                                int n_act = 0;
                                for (int b = 0; b < B; b++) {
                                    const double di = i - inc[b].ci;
                                    const double dj = j - inc[b].cj;
                                    const double d2 = di * di + dj * dj;
                                    if (d2 <= inc[b].r2) {
                                        act_spheres[n_act].ck = inc[b].ck;
                                        act_spheres[n_act].rem_r2 = inc[b].r2 - d2;
                                        n_act++;
                                    }
                                }

                                const long j_SJ = j * SJ;
                                const long jm_SJ = (j - 1) * SJ;
                                const long jp_SJ = (j + 1) * SJ;

                                const double* __restrict__ uc  = (const double*)__builtin_assume_aligned(&cur_u[i_SI + j_SJ], 64);
                                const double* __restrict__ uim = (const double*)__builtin_assume_aligned(&cur_u[im_SI + j_SJ], 64);
                                const double* __restrict__ uip = (const double*)__builtin_assume_aligned(&cur_u[ip_SI + j_SJ], 64);
                                const double* __restrict__ ujm = (const double*)__builtin_assume_aligned(&cur_u[i_SI + jm_SJ], 64);
                                const double* __restrict__ ujp = (const double*)__builtin_assume_aligned(&cur_u[i_SI + jp_SJ], 64);

                                const double* __restrict__ ac  = (const double*)__builtin_assume_aligned(&a[i_SI + j_SJ], 64);
                                const double* __restrict__ aim = (const double*)__builtin_assume_aligned(&a[im_SI + j_SJ], 64);
                                const double* __restrict__ aip = (const double*)__builtin_assume_aligned(&a[ip_SI + j_SJ], 64);
                                const double* __restrict__ ajm = (const double*)__builtin_assume_aligned(&a[i_SI + jm_SJ], 64);
                                const double* __restrict__ ajp = (const double*)__builtin_assume_aligned(&a[i_SI + jp_SJ], 64);

                                double* __restrict__ unew_row = (double*)__builtin_assume_aligned(&cur_unew[i_SI + j_SJ], 64);

                                if (n_act == 0) {
                                    #pragma GCC ivdep
                                    for (long k = 1; k <= N; k++) {
                                        const double up = uc[k], ap = ac[k];
                                        const double flux = (ap + aim[k]) * (uim[k] - up)
                                                          + (ap + aip[k]) * (uip[k] - up)
                                                          + (ap + ajm[k]) * (ujm[k] - up)
                                                          + (ap + ajp[k]) * (ujp[k] - up)
                                                          + (ap + ac[k - 1]) * (uc[k - 1] - up)
                                                          + (ap + ac[k + 1]) * (uc[k + 1] - up);
                                        const double val = up + flux * (1.0 / 12.0);
                                        unew_row[k] = val;
                                        if (is_last) thread_final_energy += val * val;
                                    }
                                } else {
                                    KInterval intervals[4];
                                    const int n_intervals = build_reaction_intervals(
                                        act_spheres, n_act, N, intervals);
                                    long cur_k = 1;
                                    for (int iv = 0; iv < n_intervals; iv++) {
                                        if (intervals[iv].lo > cur_k) {
                                            #pragma GCC ivdep
                                            for (long k = cur_k; k < intervals[iv].lo; k++) {
                                                const double up = uc[k], ap = ac[k];
                                                const double flux = (ap + aim[k]) * (uim[k] - up)
                                                                  + (ap + aip[k]) * (uip[k] - up)
                                                                  + (ap + ajm[k]) * (ujm[k] - up)
                                                                  + (ap + ajp[k]) * (ujp[k] - up)
                                                                  + (ap + ac[k - 1]) * (uc[k - 1] - up)
                                                                  + (ap + ac[k + 1]) * (uc[k + 1] - up);
                                                const double val = up + flux * (1.0 / 12.0);
                                                unew_row[k] = val;
                                                if (is_last) thread_final_energy += val * val;
                                            }
                                        }
                                        for (long k = intervals[iv].lo; k <= intervals[iv].hi; k++) {
                                            const double up = uc[k], ap = ac[k];
                                            const double flux = (ap + aim[k]) * (uim[k] - up)
                                                              + (ap + aip[k]) * (uip[k] - up)
                                                              + (ap + ajm[k]) * (ujm[k] - up)
                                                              + (ap + ajp[k]) * (ujp[k] - up)
                                                              + (ap + ac[k - 1]) * (uc[k - 1] - up)
                                                              + (ap + ac[k + 1]) * (uc[k + 1] - up);
                                            const double r = up + flux * (1.0 / 12.0);
                                            const double val = react(r);
                                            unew_row[k] = val;
                                            if (is_last) thread_final_energy += val * val;
                                        }
                                        cur_k = intervals[iv].hi + 1;
                                    }
                                    if (cur_k <= N) {
                                        #pragma GCC ivdep
                                        for (long k = cur_k; k <= N; k++) {
                                            const double up = uc[k], ap = ac[k];
                                            const double flux = (ap + aim[k]) * (uim[k] - up)
                                                              + (ap + aip[k]) * (uip[k] - up)
                                                              + (ap + ajm[k]) * (ujm[k] - up)
                                                              + (ap + ajp[k]) * (ujp[k] - up)
                                                              + (ap + ac[k - 1]) * (uc[k - 1] - up)
                                                              + (ap + ac[k + 1]) * (uc[k + 1] - up);
                                            const double val = up + flux * (1.0 / 12.0);
                                            unew_row[k] = val;
                                            if (is_last) thread_final_energy += val * val;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    std::swap(cur_u, cur_unew);
                }
            }

            #pragma omp atomic
            final_energy += thread_final_energy;
        }
        if (T % 2 == 1) {
            std::swap(u, unew);
        }
        steps = T;
        energy = (T == 0) ? energy0 : final_energy;
    } else {
        // N <= 384 with threshold > 0 cases (e.g. p05)
        if (B == 0) {
#pragma omp parallel
            {
                double* cur_u = u;
                double* cur_unew = unew;

                while (steps < T) {
                    double thread_step_energy = 0.0;
#pragma omp for schedule(static)
                    for (long j_start = 1; j_start <= N; j_start += BJ_STORED) {
                        const long j_end = std::min(N, j_start + BJ_STORED - 1);
                        for (long i = 1; i <= N; i++) {
                            const long i_SI = i * SI;
                            const long im_SI = (i - 1) * SI;
                            const long ip_SI = (i + 1) * SI;

                            for (long j = j_start; j <= j_end; j++) {
                                const long j_SJ = j * SJ;
                                const long jm_SJ = (j - 1) * SJ;
                                const long jp_SJ = (j + 1) * SJ;

                                const double* __restrict__ uc  = (const double*)__builtin_assume_aligned(&cur_u[i_SI + j_SJ], 64);
                                const double* __restrict__ uim = (const double*)__builtin_assume_aligned(&cur_u[im_SI + j_SJ], 64);
                                const double* __restrict__ uip = (const double*)__builtin_assume_aligned(&cur_u[ip_SI + j_SJ], 64);
                                const double* __restrict__ ujm = (const double*)__builtin_assume_aligned(&cur_u[i_SI + jm_SJ], 64);
                                const double* __restrict__ ujp = (const double*)__builtin_assume_aligned(&cur_u[i_SI + jp_SJ], 64);

                                const double* __restrict__ ac  = (const double*)__builtin_assume_aligned(&a[i_SI + j_SJ], 64);
                                const double* __restrict__ aim = (const double*)__builtin_assume_aligned(&a[im_SI + j_SJ], 64);
                                const double* __restrict__ aip = (const double*)__builtin_assume_aligned(&a[ip_SI + j_SJ], 64);
                                const double* __restrict__ ajm = (const double*)__builtin_assume_aligned(&a[i_SI + jm_SJ], 64);
                                const double* __restrict__ ajp = (const double*)__builtin_assume_aligned(&a[i_SI + jp_SJ], 64);

                                double* __restrict__ unew_row = (double*)__builtin_assume_aligned(&cur_unew[i_SI + j_SJ], 64);

                                #pragma GCC ivdep
                                for (long k = 1; k <= N; k++) {
                                    const double up = uc[k], ap = ac[k];
                                    const double flux = (ap + aim[k]) * (uim[k] - up)
                                                      + (ap + aip[k]) * (uip[k] - up)
                                                      + (ap + ajm[k]) * (ujm[k] - up)
                                                      + (ap + ajp[k]) * (ujp[k] - up)
                                                      + (ap + ac[k - 1]) * (uc[k - 1] - up)
                                                      + (ap + ac[k + 1]) * (uc[k + 1] - up);
                                    const double val = up + flux * (1.0 / 12.0);
                                    unew_row[k] = val;
                                    thread_step_energy += val * val;
                                }
                            }
                        }
                    }

                    #pragma omp atomic
                    step_energy += thread_step_energy;

                    #pragma omp barrier
                    #pragma omp single
                    {
                        steps++;
                        energy = step_energy;
                        step_energy = 0.0;
                    }
                    std::swap(cur_u, cur_unew);
                    if (energy <= threshold) break;
                }
            }
            if (steps % 2 == 1) {
                std::swap(u, unew);
            }
        } else {
            // Sphere early-exit cases (p05)
#pragma omp parallel
            {
                while (steps < T) {
#pragma omp for schedule(static, 1) reduction(+:step_energy)
                    for (long i = 1; i <= N; i++) {
                        const long i_SI = i * SI;
                        const long im_SI = (i - 1) * SI;
                        const long ip_SI = (i + 1) * SI;

                        if (!slice_has_sphere[i]) {
                            for (long j = 1; j <= N; j++) {
                                const long j_SJ = j * SJ;
                                const long jm_SJ = (j - 1) * SJ;
                                const long jp_SJ = (j + 1) * SJ;

                                const double* __restrict__ uc  = &u[i_SI + j_SJ];
                                const double* __restrict__ uim = &u[im_SI + j_SJ];
                                const double* __restrict__ uip = &u[ip_SI + j_SJ];
                                const double* __restrict__ ujm = &u[i_SI + jm_SJ];
                                const double* __restrict__ ujp = &u[i_SI + jp_SJ];

                                const double* __restrict__ ac  = &a[i_SI + j_SJ];
                                const double* __restrict__ aim = &a[im_SI + j_SJ];
                                const double* __restrict__ aip = &a[ip_SI + j_SJ];
                                const double* __restrict__ ajm = &a[i_SI + jm_SJ];
                                const double* __restrict__ ajp = &a[i_SI + jp_SJ];

                                double* __restrict__ unew_row = &unew[i_SI + j_SJ];

                                #pragma GCC ivdep
                                for (long k = 1; k <= N; k++) {
                                    const double up = uc[k], ap = ac[k];
                                    const double flux = (ap + aim[k]) * (uim[k] - up)
                                                      + (ap + aip[k]) * (uip[k] - up)
                                                      + (ap + ajm[k]) * (ujm[k] - up)
                                                      + (ap + ajp[k]) * (ujp[k] - up)
                                                      + (ap + ac[k - 1]) * (uc[k - 1] - up)
                                                      + (ap + ac[k + 1]) * (uc[k + 1] - up);
                                    const double val = up + flux * (1.0 / 12.0);
                                    unew_row[k] = val;
                                    step_energy += val * val;
                                }
                            }
                        } else {
                            for (long j = 1; j <= N; j++) {
                                RowSphere act_spheres[4];
                                int n_act = 0;
                                for (int b = 0; b < B; b++) {
                                    const double di = i - inc[b].ci;
                                    const double dj = j - inc[b].cj;
                                    const double d2 = di * di + dj * dj;
                                    if (d2 <= inc[b].r2) {
                                        act_spheres[n_act].ck = inc[b].ck;
                                        act_spheres[n_act].rem_r2 = inc[b].r2 - d2;
                                        n_act++;
                                    }
                                }

                                const long j_SJ = j * SJ;
                                const long jm_SJ = (j - 1) * SJ;
                                const long jp_SJ = (j + 1) * SJ;

                                const double* __restrict__ uc  = &u[i_SI + j_SJ];
                                const double* __restrict__ uim = &u[im_SI + j_SJ];
                                const double* __restrict__ uip = &u[ip_SI + j_SJ];
                                const double* __restrict__ ujm = &u[i_SI + jm_SJ];
                                const double* __restrict__ ujp = &u[i_SI + jp_SJ];

                                const double* __restrict__ ac  = &a[i_SI + j_SJ];
                                const double* __restrict__ aim = &a[im_SI + j_SJ];
                                const double* __restrict__ aip = &a[ip_SI + j_SJ];
                                const double* __restrict__ ajm = &a[i_SI + jm_SJ];
                                const double* __restrict__ ajp = &a[i_SI + jp_SJ];

                                double* __restrict__ unew_row = &unew[i_SI + j_SJ];

                                if (n_act == 0) {
                                    #pragma GCC ivdep
                                    for (long k = 1; k <= N; k++) {
                                        const double up = uc[k], ap = ac[k];
                                        const double flux = (ap + aim[k]) * (uim[k] - up)
                                                          + (ap + aip[k]) * (uip[k] - up)
                                                          + (ap + ajm[k]) * (ujm[k] - up)
                                                          + (ap + ajp[k]) * (ujp[k] - up)
                                                          + (ap + ac[k - 1]) * (uc[k - 1] - up)
                                                          + (ap + ac[k + 1]) * (uc[k + 1] - up);
                                        const double val = up + flux * (1.0 / 12.0);
                                        unew_row[k] = val;
                                        step_energy += val * val;
                                    }
                                } else {
                                    KInterval intervals[4];
                                    const int n_intervals = build_reaction_intervals(
                                        act_spheres, n_act, N, intervals);
                                    long cur_k = 1;
                                    for (int iv = 0; iv < n_intervals; iv++) {
                                        if (intervals[iv].lo > cur_k) {
                                            #pragma GCC ivdep
                                            for (long k = cur_k; k < intervals[iv].lo; k++) {
                                                const double up = uc[k], ap = ac[k];
                                                const double flux = (ap + aim[k]) * (uim[k] - up)
                                                                  + (ap + aip[k]) * (uip[k] - up)
                                                                  + (ap + ajm[k]) * (ujm[k] - up)
                                                                  + (ap + ajp[k]) * (ujp[k] - up)
                                                                  + (ap + ac[k - 1]) * (uc[k - 1] - up)
                                                                  + (ap + ac[k + 1]) * (uc[k + 1] - up);
                                                const double val = up + flux * (1.0 / 12.0);
                                                unew_row[k] = val;
                                                step_energy += val * val;
                                            }
                                        }
                                        for (long k = intervals[iv].lo; k <= intervals[iv].hi; k++) {
                                            const double up = uc[k], ap = ac[k];
                                            const double flux = (ap + aim[k]) * (uim[k] - up)
                                                              + (ap + aip[k]) * (uip[k] - up)
                                                              + (ap + ajm[k]) * (ujm[k] - up)
                                                              + (ap + ajp[k]) * (ujp[k] - up)
                                                              + (ap + ac[k - 1]) * (uc[k - 1] - up)
                                                              + (ap + ac[k + 1]) * (uc[k + 1] - up);
                                            const double r = up + flux * (1.0 / 12.0);
                                            const double val = react(r);
                                            unew_row[k] = val;
                                            step_energy += val * val;
                                        }
                                        cur_k = intervals[iv].hi + 1;
                                    }
                                    if (cur_k <= N) {
                                        #pragma GCC ivdep
                                        for (long k = cur_k; k <= N; k++) {
                                            const double up = uc[k], ap = ac[k];
                                            const double flux = (ap + aim[k]) * (uim[k] - up)
                                                              + (ap + aip[k]) * (uip[k] - up)
                                                              + (ap + ajm[k]) * (ujm[k] - up)
                                                              + (ap + ajp[k]) * (ujp[k] - up)
                                                              + (ap + ac[k - 1]) * (uc[k - 1] - up)
                                                              + (ap + ac[k + 1]) * (uc[k + 1] - up);
                                            const double val = up + flux * (1.0 / 12.0);
                                            unew_row[k] = val;
                                            step_energy += val * val;
                                        }
                                    }
                                }
                            }
                        }
                    }

#pragma omp single
                    {
                        std::swap(u, unew);
                        steps++;
                        energy = step_energy;
                        step_energy = 0.0;
                    }

                    if (energy <= threshold) break;
                }
            }
        }
    }

    FILE* out = fopen(argv[5], "w");
    if (!out) return 1;
    char out_buf[1 << 20];
    setvbuf(out, out_buf, _IOFBF, sizeof(out_buf));
    fprintf(out, "%ld %d\n%.17g\n", N, steps, energy);
    for (int si = 0; si < 32; si++)
        for (int sj = 0; sj < 32; sj++)
            for (int sk = 0; sk < 32; sk++) {
                const long i = 1 + si * (N - 1) / 31, j = 1 + sj * (N - 1) / 31, k = 1 + sk * (N - 1) / 31;
                fprintf(out, "%.17g\n", u[i * SI + j * SJ + k]);
            }
    fclose(out);
    munmap(u, total_bytes);
    munmap(unew, total_bytes);
    if (store_a) munmap(a, total_bytes);
    return 0;
}
