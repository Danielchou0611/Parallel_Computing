#include <iostream>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
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

int main(int argc, char** argv) {
    if (argc != 6) return 1;
    const long N = atol(argv[1]);
    const int T = atoi(argv[2]);
    const uint64_t seed = strtoull(argv[3], nullptr, 10);
    const double theta = atof(argv[4]);

    const long M = N + 2;
    const long SJ = (M + 7) & ~7L;
    const long SI = M * SJ;
    const size_t total_elements = (size_t)M * SI;
    const size_t total_bytes = total_elements * sizeof(double);

    // Fast allocation using mmap with hugepage hint
    double* u = (double*)mmap(nullptr, total_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    double* unew = (double*)mmap(nullptr, total_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    double* a = (double*)mmap(nullptr, total_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    madvise(u, total_bytes, MADV_HUGEPAGE);
    madvise(unew, total_bytes, MADV_HUGEPAGE);
    madvise(a, total_bytes, MADV_HUGEPAGE);

    Inclusion inc[4];
    const int B = inclusions(seed, N, inc);
    double energy0 = 0.0;

    // Parallel first-touch: initialize halo and interior in parallel
#pragma omp parallel for schedule(static, 4) reduction(+:energy0)
    for (long i = 0; i < M; i++) {
        const long i_SI = i * SI;
        if (i == 0 || i == M - 1) {
            memset(&u[i_SI], 0, SI * sizeof(double));
            memset(&unew[i_SI], 0, SI * sizeof(double));
            memset(&a[i_SI], 0, SI * sizeof(double));
        } else {
            // Zero halos at j=0 and j=M-1
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
                    const double u_val = field(seed, index, 0);
                    u_row[k] = u_val;
                    a_row[k] = field(seed, index, 1);
                    energy0 += u_val * u_val;
                }
            }
        }
    }

    double energy = energy0;
    int steps = 0;
    const double threshold = theta * energy0;
    double step_energy = 0.0;

    if (threshold <= 0.0) {
#pragma omp parallel
        {
            for (int s = 0; s < T; s++) {
#pragma omp for schedule(static, 4)
                for (long i = 1; i <= N; i++) {
                    bool slice_has_sphere = false;
                    for (int b = 0; b < B; b++) {
                        const double di = i - inc[b].ci;
                        if (di * di <= inc[b].r2) {
                            slice_has_sphere = true;
                            break;
                        }
                    }

                    const long i_SI = i * SI;
                    const long im_SI = (i - 1) * SI;
                    const long ip_SI = (i + 1) * SI;

                    if (!slice_has_sphere) {
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
                                unew_row[k] = up + flux * (1.0 / 12.0);
                            }
                        }
                    } else {
                        for (long j = 1; j <= N; j++) {
                            bool row_has_sphere = false;
                            for (int b = 0; b < B; b++) {
                                const double di = i - inc[b].ci;
                                const double dj = j - inc[b].cj;
                                if (di * di + dj * dj <= inc[b].r2) {
                                    row_has_sphere = true;
                                    break;
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

                            if (!row_has_sphere) {
                                #pragma GCC ivdep
                                for (long k = 1; k <= N; k++) {
                                    const double up = uc[k], ap = ac[k];
                                    const double flux = (ap + aim[k]) * (uim[k] - up)
                                                      + (ap + aip[k]) * (uip[k] - up)
                                                      + (ap + ajm[k]) * (ujm[k] - up)
                                                      + (ap + ajp[k]) * (ujp[k] - up)
                                                      + (ap + ac[k - 1]) * (uc[k - 1] - up)
                                                      + (ap + ac[k + 1]) * (uc[k + 1] - up);
                                    unew_row[k] = up + flux * (1.0 / 12.0);
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
                                    unew_row[k] = reactive(inc, B, i, j, k) ? react(r) : r;
                                }
                            }
                        }
                    }
                }

#pragma omp single
                {
                    std::swap(u, unew);
                }
            }
        }
        steps = T;

        double final_energy = 0.0;
#pragma omp parallel for schedule(static, 4) reduction(+:final_energy)
        for (long i = 1; i <= N; i++) {
            const long i_SI = i * SI;
            for (long j = 1; j <= N; j++) {
                const double* __restrict__ u_row = &u[i_SI + j * SJ];
                for (long k = 1; k <= N; k++) {
                    final_energy += u_row[k] * u_row[k];
                }
            }
        }
        energy = final_energy;
    } else {
#pragma omp parallel
        {
            while (steps < T) {
#pragma omp for schedule(static, 4) reduction(+:step_energy)
                for (long i = 1; i <= N; i++) {
                    bool slice_has_sphere = false;
                    for (int b = 0; b < B; b++) {
                        const double di = i - inc[b].ci;
                        if (di * di <= inc[b].r2) {
                            slice_has_sphere = true;
                            break;
                        }
                    }

                    const long i_SI = i * SI;
                    const long im_SI = (i - 1) * SI;
                    const long ip_SI = (i + 1) * SI;

                    if (!slice_has_sphere) {
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
                                const double r = up + flux * (1.0 / 12.0);
                                unew_row[k] = r;
                                step_energy += r * r;
                            }
                        }
                    } else {
                        for (long j = 1; j <= N; j++) {
                            bool row_has_sphere = false;
                            for (int b = 0; b < B; b++) {
                                const double di = i - inc[b].ci;
                                const double dj = j - inc[b].cj;
                                if (di * di + dj * dj <= inc[b].r2) {
                                    row_has_sphere = true;
                                    break;
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

                            if (!row_has_sphere) {
                                #pragma GCC ivdep
                                for (long k = 1; k <= N; k++) {
                                    const double up = uc[k], ap = ac[k];
                                    const double flux = (ap + aim[k]) * (uim[k] - up)
                                                      + (ap + aip[k]) * (uip[k] - up)
                                                      + (ap + ajm[k]) * (ujm[k] - up)
                                                      + (ap + ajp[k]) * (ujp[k] - up)
                                                      + (ap + ac[k - 1]) * (uc[k - 1] - up)
                                                      + (ap + ac[k + 1]) * (uc[k + 1] - up);
                                    const double r = up + flux * (1.0 / 12.0);
                                    unew_row[k] = r;
                                    step_energy += r * r;
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
                                    const double val = reactive(inc, B, i, j, k) ? react(r) : r;
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

    FILE* out = fopen(argv[5], "w");
    if (!out) return 1;
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
