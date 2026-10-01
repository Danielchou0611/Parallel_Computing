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

static inline __attribute__((always_inline))
double field_preseed(uint64_t seed_mix, uint64_t index, int which) {
    uint64_t x = (index * 2 + static_cast<uint64_t>(which)) ^ seed_mix;
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    x ^= x >> 31;
    return static_cast<double>(x >> 11) * (1.0 / 9007199254740992.0);
}

struct RowSphere {
    double ck;
    double rem_r2;
};

int main(int argc, char** argv) {
    if (argc != 6) return 1;
    const long N = atol(argv[1]);
    const int T = atoi(argv[2]);
    const uint64_t seed = strtoull(argv[3], nullptr, 10);
    const double theta = atof(argv[4]);
    const uint64_t seed_mix = seed * 0x9E3779B97F4A7C15ull;

    const long M = N + 2;
    const long SJ = (M + 7) & ~7L;
    const long SI = M * SJ;
    const size_t total_elements = (size_t)M * SI;
    const size_t total_bytes = total_elements * sizeof(double);

    double* u = (double*)mmap(nullptr, total_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    double* unew = (double*)mmap(nullptr, total_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    double* a = (double*)mmap(nullptr, total_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    madvise(u, total_bytes, MADV_HUGEPAGE);
    madvise(unew, total_bytes, MADV_HUGEPAGE);
    madvise(a, total_bytes, MADV_HUGEPAGE);

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

    memset(&u[0], 0, SI * sizeof(double));
    memset(&unew[0], 0, SI * sizeof(double));
    memset(&a[0], 0, SI * sizeof(double));
    memset(&u[(M - 1) * SI], 0, SI * sizeof(double));
    memset(&unew[(M - 1) * SI], 0, SI * sizeof(double));
    memset(&a[(M - 1) * SI], 0, SI * sizeof(double));

#pragma omp parallel for schedule(static, 1) reduction(+:energy0)
    for (long i = 1; i <= N; i++) {
        const long i_SI = i * SI;
        memset(&u[i_SI], 0, SJ * sizeof(double));
        memset(&unew[i_SI], 0, SJ * sizeof(double));
        memset(&a[i_SI], 0, SJ * sizeof(double));
        memset(&u[i_SI + (M - 1) * SJ], 0, SJ * sizeof(double));
        memset(&unew[i_SI + (M - 1) * SJ], 0, SJ * sizeof(double));
        memset(&a[i_SI + (M - 1) * SJ], 0, SJ * sizeof(double));

        for (long j = 1; j <= N; j++) {
            const long j_SJ = j * SJ;
            double* u_row = &u[i_SI + j_SJ];
            double* unew_row = &unew[i_SI + j_SJ];
            double* a_row = &a[i_SI + j_SJ];
            u_row[0] = 0.0; u_row[N + 1] = 0.0;
            unew_row[0] = 0.0; unew_row[N + 1] = 0.0;
            a_row[0] = 0.0; a_row[N + 1] = 0.0;

            const uint64_t base_idx = ((uint64_t)(i - 1) * N + (j - 1)) * N;
            for (long k = 1; k <= N; k++) {
                const uint64_t index = base_idx + (k - 1);
                const double u_val = field_preseed(seed_mix, index, 0);
                u_row[k] = u_val;
                a_row[k] = field_preseed(seed_mix, index, 1);
                energy0 += u_val * u_val;
            }
        }
    }

    double energy = energy0;
    int steps = 0;
    const double threshold = theta * energy0;
    double step_energy = 0.0;
    const int BJ = 32;

    if (threshold <= 0.0) {
        double final_energy = 0.0;
#pragma omp parallel
        {
            double* cur_u = u;
            double* cur_unew = unew;
            double thread_final_energy = 0.0;

            if (B == 0) {
                // Pure diffusion: Static J-cache-tiling with i sweeping
                for (int s = 0; s < T; s++) {
                    const bool is_last = (s == T - 1);
#pragma omp for schedule(static)
                    for (long j_start = 1; j_start <= N; j_start += BJ) {
                        const long j_end = std::min(N, j_start + BJ - 1);
                        for (long i = 1; i <= N; i++) {
                            const long i_SI = i * SI;
                            const long im_SI = (i - 1) * SI;
                            const long ip_SI = (i + 1) * SI;

                            for (long j = j_start; j <= j_end; j++) {
                                const long j_SJ = j * SJ;
                                const long jm_SJ = (j - 1) * SJ;
                                const long jp_SJ = (j + 1) * SJ;

                                const double* __restrict__ uc  = &cur_u[i_SI + j_SJ];
                                const double* __restrict__ uim = &cur_u[im_SI + j_SJ];
                                const double* __restrict__ uip = &cur_u[ip_SI + j_SJ];
                                const double* __restrict__ ujm = &cur_u[i_SI + jm_SJ];
                                const double* __restrict__ ujp = &cur_u[i_SI + jp_SJ];

                                const double* __restrict__ ac  = &a[i_SI + j_SJ];
                                const double* __restrict__ aim = &a[im_SI + j_SJ];
                                const double* __restrict__ aip = &a[ip_SI + j_SJ];
                                const double* __restrict__ ajm = &a[i_SI + jm_SJ];
                                const double* __restrict__ ajp = &a[i_SI + jp_SJ];

                                double* __restrict__ unew_row = &cur_unew[i_SI + j_SJ];

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
                // Sphere cases: schedule(static, 1) balanced cyclic distribution
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

                                const double* __restrict__ uc  = &cur_u[i_SI + j_SJ];
                                const double* __restrict__ uim = &cur_u[im_SI + j_SJ];
                                const double* __restrict__ uip = &cur_u[ip_SI + j_SJ];
                                const double* __restrict__ ujm = &cur_u[i_SI + jm_SJ];
                                const double* __restrict__ ujp = &cur_u[i_SI + jp_SJ];

                                const double* __restrict__ ac  = &a[i_SI + j_SJ];
                                const double* __restrict__ aim = &a[im_SI + j_SJ];
                                const double* __restrict__ aip = &a[ip_SI + j_SJ];
                                const double* __restrict__ ajm = &a[i_SI + jm_SJ];
                                const double* __restrict__ ajp = &a[i_SI + jp_SJ];

                                double* __restrict__ unew_row = &cur_unew[i_SI + j_SJ];

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

                                const double* __restrict__ uc  = &cur_u[i_SI + j_SJ];
                                const double* __restrict__ uim = &cur_u[im_SI + j_SJ];
                                const double* __restrict__ uip = &cur_u[ip_SI + j_SJ];
                                const double* __restrict__ ujm = &cur_u[i_SI + jm_SJ];
                                const double* __restrict__ ujp = &cur_u[i_SI + jp_SJ];

                                const double* __restrict__ ac  = &a[i_SI + j_SJ];
                                const double* __restrict__ aim = &a[im_SI + j_SJ];
                                const double* __restrict__ aip = &a[ip_SI + j_SJ];
                                const double* __restrict__ ajm = &a[i_SI + jm_SJ];
                                const double* __restrict__ ajp = &a[i_SI + jp_SJ];

                                double* __restrict__ unew_row = &cur_unew[i_SI + j_SJ];

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
                                    for (long k = 1; k <= N; k++) {
                                        const double up = uc[k], ap = ac[k];
                                        const double flux = (ap + aim[k]) * (uim[k] - up)
                                                          + (ap + aip[k]) * (uip[k] - up)
                                                          + (ap + ajm[k]) * (ujm[k] - up)
                                                          + (ap + ajp[k]) * (ujp[k] - up)
                                                          + (ap + ac[k - 1]) * (uc[k - 1] - up)
                                                          + (ap + ac[k + 1]) * (uc[k + 1] - up);
                                        const double r = up + flux * (1.0 / 12.0);
                                        bool is_react = false;
                                        for (int a_idx = 0; a_idx < n_act; a_idx++) {
                                            const double dk = k - act_spheres[a_idx].ck;
                                            if (dk * dk <= act_spheres[a_idx].rem_r2) {
                                                is_react = true;
                                                break;
                                            }
                                        }
                                        const double val = is_react ? react(r) : r;
                                        unew_row[k] = val;
                                        if (is_last) thread_final_energy += val * val;
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
        energy = final_energy;
    } else {
        // threshold > 0.0 cases
        if (B == 0) {
            // Pure diffusion early-exit with Static J-cache-tiling
#pragma omp parallel
            {
                double* cur_u = u;
                double* cur_unew = unew;

                while (steps < T) {
                    double thread_step_energy = 0.0;
#pragma omp for schedule(static)
                    for (long j_start = 1; j_start <= N; j_start += BJ) {
                        const long j_end = std::min(N, j_start + BJ - 1);
                        for (long i = 1; i <= N; i++) {
                            const long i_SI = i * SI;
                            const long im_SI = (i - 1) * SI;
                            const long ip_SI = (i + 1) * SI;

                            for (long j = j_start; j <= j_end; j++) {
                                const long j_SJ = j * SJ;
                                const long jm_SJ = (j - 1) * SJ;
                                const long jp_SJ = (j + 1) * SJ;

                                const double* __restrict__ uc  = &cur_u[i_SI + j_SJ];
                                const double* __restrict__ uim = &cur_u[im_SI + j_SJ];
                                const double* __restrict__ uip = &cur_u[ip_SI + j_SJ];
                                const double* __restrict__ ujm = &cur_u[i_SI + jm_SJ];
                                const double* __restrict__ ujp = &cur_u[i_SI + jp_SJ];

                                const double* __restrict__ ac  = &a[i_SI + j_SJ];
                                const double* __restrict__ aim = &a[im_SI + j_SJ];
                                const double* __restrict__ aip = &a[ip_SI + j_SJ];
                                const double* __restrict__ ajm = &a[i_SI + jm_SJ];
                                const double* __restrict__ ajp = &a[i_SI + jp_SJ];

                                double* __restrict__ unew_row = &cur_unew[i_SI + j_SJ];

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
                                    for (long k = 1; k <= N; k++) {
                                        const double up = uc[k], ap = ac[k];
                                        const double flux = (ap + aim[k]) * (uim[k] - up)
                                                          + (ap + aip[k]) * (uip[k] - up)
                                                          + (ap + ajm[k]) * (ujm[k] - up)
                                                          + (ap + ajp[k]) * (ujp[k] - up)
                                                          + (ap + ac[k - 1]) * (uc[k - 1] - up)
                                                          + (ap + ac[k + 1]) * (uc[k + 1] - up);
                                        const double r = up + flux * (1.0 / 12.0);
                                        bool is_react = false;
                                        for (int a_idx = 0; a_idx < n_act; a_idx++) {
                                            const double dk = k - act_spheres[a_idx].ck;
                                            if (dk * dk <= act_spheres[a_idx].rem_r2) {
                                                is_react = true;
                                                break;
                                            }
                                        }
                                        const double val = is_react ? react(r) : r;
                                        unew_row[k] = val;
                                        step_energy += val * val;
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
    munmap(a, total_bytes);
    return 0;
}
