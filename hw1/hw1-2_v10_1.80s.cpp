#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <png.h>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <chrono>
// #include <pthread.h>
#include <memory>
#include <omp.h>
#include <cstring>
#include <immintrin.h>
#include <fcntl.h>
#include <unistd.h>

// ---------- shared PNG I/O ----------

struct Mat {
    int h = 0, w = 0;
    std::shared_ptr<double> buf;
    double* data = nullptr;

    Mat() = default;
    Mat(int height, int width) : h(height), w(width) {
        if (h > 0 && width > 0) {
            double* p = (double*)malloc(sizeof(double) * h * w);
            buf = std::shared_ptr<double>(p, free);
            data = p;
        }
    }
    Mat(int height, int width, double* external_ptr) : h(height), w(width), buf(nullptr), data(external_ptr) {}

    inline double* operator[](int y) { return data + y * w; }
    inline const double* operator[](int y) const { return data + y * w; }
    inline double* data_ptr() { return data; }
    inline const double* data_ptr() const { return data; }
};

struct BufferPool {
    struct Buffer {
        double* ptr = nullptr;
        size_t cap = 0;
        Buffer() = default;
        Buffer(size_t n) {
            if (posix_memalign((void**)&ptr, 64, n * sizeof(double)) != 0) {
                ptr = (double*)malloc(n * sizeof(double));
            }
            cap = n;
        }
        ~Buffer() { if (ptr) free(ptr); }
        Buffer(Buffer&& o) noexcept : ptr(o.ptr), cap(o.cap) { o.ptr = nullptr; o.cap = 0; }
        Buffer& operator=(Buffer&& o) noexcept {
            if (this != &o) {
                if (ptr) free(ptr);
                ptr = o.ptr; cap = o.cap;
                o.ptr = nullptr; o.cap = 0;
            }
            return *this;
        }
        Buffer(const Buffer&) = delete;
        Buffer& operator=(const Buffer&) = delete;

        void ensure(size_t n) {
            if (cap < n) {
                if (ptr) free(ptr);
                if (posix_memalign((void**)&ptr, 64, n * sizeof(double)) != 0) {
                    ptr = (double*)malloc(n * sizeof(double));
                }
                cap = n;
            }
        }
    };

    std::vector<Buffer> buffers;
    size_t next_idx = 0;

    Mat allocMat(int h, int w) {
        size_t needed = (size_t)h * w;
        if (next_idx < buffers.size()) {
            buffers[next_idx].ensure(needed);
            double* ptr = buffers[next_idx].ptr;
            next_idx++;
            return Mat(h, w, ptr);
        }
        buffers.emplace_back(needed);
        double* ptr = buffers.back().ptr;
        next_idx++;
        return Mat(h, w, ptr);
    }

    void reset() {
        next_idx = 0;
    }
};

static double s_gray_lut[256];
static bool s_lut_init = false;
static void init_lut() {
    if (!s_lut_init) {
        for (int i = 0; i < 256; i++) {
            s_gray_lut[i] = (0.299 * i + 0.587 * i + 0.114 * i) / 255.0;
        }
        s_lut_init = true;
    }
}

void read_png_to_gray(const char* file_name, Mat& gray, int& height, int& width) {
    FILE *fp = fopen(file_name, "rb");
    if (!fp) {
        std::cerr << "Error: Cannot open file " << file_name << std::endl;
        exit(EXIT_FAILURE);
    }
    posix_fadvise(fileno(fp), 0, 0, POSIX_FADV_SEQUENTIAL | POSIX_FADV_WILLNEED);
    setvbuf(fp, nullptr, _IOFBF, 2 * 1024 * 1024);

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    png_infop info = png_create_info_struct(png);
    if (setjmp(png_jmpbuf(png))) {
        std::cerr << "Error during PNG creation" << std::endl;
        exit(EXIT_FAILURE);
    }

    png_init_io(png, fp);
    png_read_info(png, info);

    width = png_get_image_width(png, info);
    height = png_get_image_height(png, info);
    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth = png_get_bit_depth(png, info);
    png_set_crc_action(png, PNG_CRC_QUIET_USE, PNG_CRC_QUIET_USE);
    png_set_compression_buffer_size(png, 2 * 1024 * 1024);

    if (bit_depth == 16) png_set_strip_16(png);
    if (color_type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);

    png_read_update_info(png, info);

    int channels = png_get_channels(png, info);
    size_t rowbytes = png_get_rowbytes(png, info);
    png_byte* row = (png_byte*)malloc(rowbytes);

    gray = Mat(height, width);
    if (channels == 1) {
        for (int y = 0; y < height; y++) {
            png_read_row(png, row, nullptr);
            double* gray_row = gray[y];
            #pragma GCC ivdep
            for (int x = 0; x < width; x++) {
                gray_row[x] = s_gray_lut[row[x]];
            }
        }
    } else if (channels == 3) {
        for (int y = 0; y < height; y++) {
            png_read_row(png, row, nullptr);
            double* gray_row = gray[y];
            #pragma GCC ivdep
            for (int x = 0; x < width; x++) {
                const png_byte* px = &row[x * 3];
                gray_row[x] = (0.299 * px[0] + 0.587 * px[1] + 0.114 * px[2]) / 255.0;
            }
        }
    } else {
        for (int y = 0; y < height; y++) {
            png_read_row(png, row, nullptr);
            double* gray_row = gray[y];
            #pragma GCC ivdep
            for (int x = 0; x < width; x++) {
                const png_byte* px = &row[x * channels];
                gray_row[x] = (0.299 * px[0] + 0.587 * px[1] + 0.114 * px[2]) / 255.0;
            }
        }
    }
    free(row);
    fclose(fp);
    png_destroy_read_struct(&png, &info, nullptr);
}

const int NUM_OCTAVES = 4;
const int S = 3;              // Lowe's scales-per-octave
const int NUM_SCALES = S + 3; // Gaussian images per octave
const double SIGMA0 = 1.6;
const double CONTRAST_THRESH = 0.03;
const double EDGE_THRESH_R = 10.0;

struct GaussianKernel {
    int radius = 0;
    int K = 0;
    double kernel[45];
    __m256d k_vecs[45];
};

static GaussianKernel s_kernels[NUM_SCALES];
static bool s_kernels_inited = false;

static void init_gaussian_kernels() {
    if (s_kernels_inited) return;
    double k = std::pow(2.0, 1.0 / S);
    for (int s = 1; s < NUM_SCALES; s++) {
        double sigma = SIGMA0 * std::pow(k, s);
        int radius = std::max(1, (int)std::ceil(3 * sigma));
        int K = 2 * radius + 1;
        s_kernels[s].radius = radius;
        s_kernels[s].K = K;

        double sum = 0.0;
        for (int i = -radius; i <= radius; i++) {
            double v = std::exp(-(i * i) / (2.0 * sigma * sigma));
            s_kernels[s].kernel[i + radius] = v;
            sum += v;
        }
        for (int i = 0; i < K; i++) {
            s_kernels[s].kernel[i] /= sum;
            s_kernels[s].k_vecs[i] = _mm256_set1_pd(s_kernels[s].kernel[i]);
        }
    }
    s_kernels_inited = true;
}

void downsample2x(const Mat& in, int height, int width, Mat& out) {
    int nh = height / 2, nw = width / 2;
    #pragma omp parallel for
    for (int y = 0; y < nh; y++) {
        const double* in_row = in[2 * y];
        double* out_row = out[y];
        #pragma GCC ivdep
        for (int x = 0; x < nw; x++)
            out_row[x] = in_row[2 * x];
    }
}

struct Octave {
    int height, width;
    std::vector<Mat> gaussian; // NUM_SCALES images
    std::vector<Mat> dog;      // NUM_SCALES - 1 images
};

template<int RADIUS, int K>
inline void runScale(int s, const double* in_mat, double* out_mat, const double* prev_g, double* out_dog, double* tmp_buf, int h, int w) {
    const GaussianKernel& gk = s_kernels[s];
    const double* k_ptr = gk.kernel;
    const __m256d* k_vecs = gk.k_vecs;

    // Pass 1: Horizontal blur with boundary splitting (AVX2 dual-accumulator 8-pixel middle)
    #pragma omp for schedule(static)
    for (int y = 0; y < h; y++) {
        const double* in_row = in_mat + y * w;
        double* tmp_row = tmp_buf + y * w;

        int left_end = std::min(RADIUS, w);
        for (int x = 0; x < left_end; x++) {
            double acc = 0.0;
            for (int i = -RADIUS; i <= RADIUS; i++) {
                int xx = std::min(std::max(x + i, 0), w - 1);
                acc += in_row[xx] * k_ptr[i + RADIUS];
            }
            tmp_row[x] = acc;
        }

        int right_start = std::max(RADIUS, w - RADIUS);
        int x = left_end;
        for (; x + 7 < right_start; x += 8) {
            __m256d acc0 = _mm256_setzero_pd();
            __m256d acc1 = _mm256_setzero_pd();
            const double* p = in_row + (x - RADIUS);
            #pragma GCC unroll 8
            for (int i = 0; i < K; i++) {
                __m256d kv = k_vecs[i];
                __m256d p0 = _mm256_loadu_pd(p + i);
                __m256d p1 = _mm256_loadu_pd(p + i + 4);
                acc0 = _mm256_add_pd(acc0, _mm256_mul_pd(p0, kv));
                acc1 = _mm256_add_pd(acc1, _mm256_mul_pd(p1, kv));
            }
            _mm256_storeu_pd(tmp_row + x, acc0);
            _mm256_storeu_pd(tmp_row + x + 4, acc1);
        }
        for (; x + 3 < right_start; x += 4) {
            __m256d acc = _mm256_setzero_pd();
            const double* p = in_row + (x - RADIUS);
            for (int i = 0; i < K; i++) {
                __m256d p_vec = _mm256_loadu_pd(p + i);
                acc = _mm256_add_pd(acc, _mm256_mul_pd(p_vec, k_vecs[i]));
            }
            _mm256_storeu_pd(tmp_row + x, acc);
        }
        for (; x < right_start; x++) {
            double acc = 0.0;
            const double* p = in_row + (x - RADIUS);
            for (int i = 0; i < K; i++) {
                acc += p[i] * k_ptr[i];
            }
            tmp_row[x] = acc;
        }

        for (int x = right_start; x < w; x++) {
            double acc = 0.0;
            for (int i = -RADIUS; i <= RADIUS; i++) {
                int xx = std::min(std::max(x + i, 0), w - 1);
                acc += in_row[xx] * k_ptr[i + RADIUS];
            }
            tmp_row[x] = acc;
        }
    }

    // Pass 2: Vertical blur (AVX2 dual-accumulator 8-pixel column-vectorized) + Fused DoG
    #pragma omp for schedule(static)
    for (int y = 0; y < h; y++) {
        double* out_row = out_mat + y * w;

        const double* rows[45];
        for (int i = -RADIUS; i <= RADIUS; i++) {
            int yy = std::min(std::max(y + i, 0), h - 1);
            rows[i + RADIUS] = tmp_buf + yy * w;
        }

        const double* prev_row = prev_g + y * w;
        double* dog_row = out_dog + y * w;

        int x = 0;
        for (; x + 7 < w; x += 8) {
            __m256d acc0 = _mm256_setzero_pd();
            __m256d acc1 = _mm256_setzero_pd();
            #pragma GCC unroll 8
            for (int i = 0; i < K; i++) {
                __m256d kv = k_vecs[i];
                __m256d r0 = _mm256_loadu_pd(rows[i] + x);
                __m256d r1 = _mm256_loadu_pd(rows[i] + x + 4);
                acc0 = _mm256_add_pd(acc0, _mm256_mul_pd(r0, kv));
                acc1 = _mm256_add_pd(acc1, _mm256_mul_pd(r1, kv));
            }
            _mm256_storeu_pd(out_row + x, acc0);
            _mm256_storeu_pd(out_row + x + 4, acc1);
            __m256d p0 = _mm256_loadu_pd(prev_row + x);
            __m256d p1 = _mm256_loadu_pd(prev_row + x + 4);
            _mm256_storeu_pd(dog_row + x, _mm256_sub_pd(acc0, p0));
            _mm256_storeu_pd(dog_row + x + 4, _mm256_sub_pd(acc1, p1));
        }
        for (; x + 3 < w; x += 4) {
            __m256d acc = _mm256_setzero_pd();
            for (int i = 0; i < K; i++) {
                __m256d r = _mm256_loadu_pd(rows[i] + x);
                acc = _mm256_add_pd(acc, _mm256_mul_pd(r, k_vecs[i]));
            }
            _mm256_storeu_pd(out_row + x, acc);
            __m256d p = _mm256_loadu_pd(prev_row + x);
            _mm256_storeu_pd(dog_row + x, _mm256_sub_pd(acc, p));
        }
        for (; x < w; x++) {
            double acc = 0.0;
            for (int i = 0; i < K; i++) {
                acc += rows[i][x] * k_ptr[i];
            }
            out_row[x] = acc;
            dog_row[x] = acc - prev_row[x];
        }
    }
}

std::vector<Octave> buildPyramid(const Mat& gray, int height, int width, BufferPool& pool) {
    init_gaussian_kernels();
    std::vector<Octave> octaves(NUM_OCTAVES);

    Mat tmp_mat = pool.allocMat(height, width);
    double* tmp_buf = tmp_mat.data_ptr();
    Mat base = gray;
    int h = height, w = width;
    for (int o = 0; o < NUM_OCTAVES; o++) {
        Octave& oct = octaves[o];
        oct.height = h;
        oct.width = w;
        oct.gaussian.resize(NUM_SCALES);
        oct.gaussian[0] = base;
        oct.dog.resize(NUM_SCALES - 1);
        for (int s = 0; s < NUM_SCALES - 1; s++) {
            oct.dog[s] = pool.allocMat(h, w);
        }
        for (int s = 1; s < NUM_SCALES; s++) {
            oct.gaussian[s] = pool.allocMat(h, w);
        }
        auto t_o0 = std::chrono::high_resolution_clock::now();

        #pragma omp parallel
        {
            runScale<7, 15>(1, base.data_ptr(), oct.gaussian[1].data_ptr(), oct.gaussian[0].data_ptr(), oct.dog[0].data_ptr(), tmp_buf, h, w);
            runScale<8, 17>(2, base.data_ptr(), oct.gaussian[2].data_ptr(), oct.gaussian[1].data_ptr(), oct.dog[1].data_ptr(), tmp_buf, h, w);
            runScale<10, 21>(3, base.data_ptr(), oct.gaussian[3].data_ptr(), oct.gaussian[2].data_ptr(), oct.dog[2].data_ptr(), tmp_buf, h, w);
            runScale<13, 27>(4, base.data_ptr(), oct.gaussian[4].data_ptr(), oct.gaussian[3].data_ptr(), oct.dog[3].data_ptr(), tmp_buf, h, w);
            runScale<16, 33>(5, base.data_ptr(), oct.gaussian[5].data_ptr(), oct.gaussian[4].data_ptr(), oct.dog[4].data_ptr(), tmp_buf, h, w);
        }
        auto t_o1 = std::chrono::high_resolution_clock::now();
        if (getenv("PROFILE")) {
            auto ms = [](std::chrono::high_resolution_clock::time_point a, std::chrono::high_resolution_clock::time_point b) {
                return std::chrono::duration<double, std::milli>(b - a).count();
            };
            std::cerr << "    oct " << o << " (" << w << "x" << h << ") blur+dog: " << ms(t_o0, t_o1) << " ms\n";
        }

        if (o + 1 < NUM_OCTAVES) {
            Mat down = pool.allocMat(h / 2, w / 2);
            downsample2x(oct.gaussian[S], h, w, down);
            base = down;
            h /= 2;
            w /= 2;
        }
        // Early release of unused Gaussians
        oct.gaussian[0] = Mat();
        oct.gaussian[4] = Mat();
        oct.gaussian[5] = Mat();
    }
    return octaves;
}

// ---------- keypoint detection, orientation, descriptor ----------

struct Keypoint {
    int octave, layer;   // DoG layer index (1 .. NUM_SCALES-3, inclusive both sides)
    int x, y;             // pixel coords within that octave's resolution
    double scale;         // sigma at this layer, in octave-local units
    double orientation;   // radians
    std::vector<double> descriptor;
};

struct DogSlice {
    const double* r0; const double* r1; const double* r2;
    const double* p0; const double* p1; const double* p2;
    const double* q0; const double* q1; const double* q2;
};

inline bool isExtremum(double v, const DogSlice& sl, int x) {
    double n0 = sl.r0[x - 1];
    if (n0 == v) return false;
    if (n0 < v) {
        // v can only be maximum
        if (sl.r0[x] >= v || sl.r0[x + 1] >= v) return false;
        if (sl.r1[x - 1] >= v || sl.r1[x + 1] >= v) return false;
        if (sl.r2[x - 1] >= v || sl.r2[x] >= v || sl.r2[x + 1] >= v) return false;

        // Check prev layer
        if (sl.p0[x - 1] >= v || sl.p0[x] >= v || sl.p0[x + 1] >= v) return false;
        if (sl.p1[x - 1] >= v || sl.p1[x] >= v || sl.p1[x + 1] >= v) return false;
        if (sl.p2[x - 1] >= v || sl.p2[x] >= v || sl.p2[x + 1] >= v) return false;

        // Check next layer
        if (sl.q0[x - 1] >= v || sl.q0[x] >= v || sl.q0[x + 1] >= v) return false;
        if (sl.q1[x - 1] >= v || sl.q1[x] >= v || sl.q1[x + 1] >= v) return false;
        if (sl.q2[x - 1] >= v || sl.q2[x] >= v || sl.q2[x + 1] >= v) return false;

        return true;
    } else {
        // v can only be minimum
        if (sl.r0[x] <= v || sl.r0[x + 1] <= v) return false;
        if (sl.r1[x - 1] <= v || sl.r1[x + 1] <= v) return false;
        if (sl.r2[x - 1] <= v || sl.r2[x] <= v || sl.r2[x + 1] <= v) return false;

        // Check prev layer
        if (sl.p0[x - 1] <= v || sl.p0[x] <= v || sl.p0[x + 1] <= v) return false;
        if (sl.p1[x - 1] <= v || sl.p1[x] <= v || sl.p1[x + 1] <= v) return false;
        if (sl.p2[x - 1] <= v || sl.p2[x] <= v || sl.p2[x + 1] <= v) return false;

        // Check next layer
        if (sl.q0[x - 1] <= v || sl.q0[x] <= v || sl.q0[x + 1] <= v) return false;
        if (sl.q1[x - 1] <= v || sl.q1[x] <= v || sl.q1[x + 1] <= v) return false;
        if (sl.q2[x - 1] <= v || sl.q2[x] <= v || sl.q2[x + 1] <= v) return false;

        return true;
    }
}

static const double EDGE_RATIO_THRESH = (EDGE_THRESH_R + 1.0) * (EDGE_THRESH_R + 1.0) / EDGE_THRESH_R;

inline bool passesEdgeTest(const double* ym1, const double* y0, const double* yp1, int x) {
    double dxx = y0[x + 1] + y0[x - 1] - 2 * y0[x];
    double dyy = yp1[x] + ym1[x] - 2 * y0[x];
    double dxy = (yp1[x + 1] - yp1[x - 1] - ym1[x + 1] + ym1[x]) / 4.0;
    double trace = dxx + dyy;
    double det = dxx * dyy - dxy * dxy;
    if (det <= 0) return false;
    return (trace * trace) < EDGE_RATIO_THRESH * det;
}

// Independent per (octave, layer, pixel) -- another natural parallelization
// target. Extrema in different octaves/layers never interact.
std::vector<Keypoint> detectKeypoints(const std::vector<Octave>& octaves) {
    std::vector<Keypoint> keypoints;
    int max_threads = omp_get_max_threads();
    std::vector<std::vector<Keypoint>> thread_kps(max_threads);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        for (int o = 0; o < NUM_OCTAVES; o++) {
            const Octave& oct = octaves[o];
            for (int s = 1; s < (int)oct.dog.size() - 1; s++) {
                thread_kps[tid].clear();
                #pragma omp barrier

                const Mat& d_prev = oct.dog[s - 1];
                const Mat& d_curr = oct.dog[s];
                const Mat& d_next = oct.dog[s + 1];
                double scale = SIGMA0 * std::pow(2.0, (double)s / S);

                #pragma omp for schedule(static)
                for (int y = 1; y < oct.height - 1; y++) {
                    DogSlice slice = {
                        d_curr[y - 1], d_curr[y], d_curr[y + 1],
                        d_prev[y - 1], d_prev[y], d_prev[y + 1],
                        d_next[y - 1], d_next[y], d_next[y + 1]
                    };
                    const double* d_ym1 = slice.r0;
                    const double* d_y0  = slice.r1;
                    const double* d_yp1 = slice.r2;
                    for (int x = 1; x < oct.width - 1; x++) {
                        double v = d_y0[x];
                        if (std::fabs(v) < CONTRAST_THRESH) continue;
                        if (!isExtremum(v, slice, x)) continue;
                        if (!passesEdgeTest(d_ym1, d_y0, d_yp1, x)) continue;

                        Keypoint kp;
                        kp.octave = o;
                        kp.layer = s;
                        kp.x = x;
                        kp.y = y;
                        kp.scale = scale;
                        thread_kps[tid].push_back(kp);
                    }
                }

                #pragma omp master
                {
                    for (int t = 0; t < max_threads; t++) {
                        keypoints.insert(keypoints.end(), thread_kps[t].begin(), thread_kps[t].end());
                    }
                }
                #pragma omp barrier
            }
        }
    }
    return keypoints;
}

// ponytail: single dominant orientation per keypoint (no histogram-peak
// splitting into multiple keypoints) -- simpler & still demonstrates the
// weighted-histogram parallelization pattern without duplicating keypoints.
void assignOrientation(Keypoint& kp, const Octave& oct) {
    const Mat& img = oct.gaussian[kp.layer];
    double sigma = 1.5 * kp.scale;
    int radius = (int)std::round(3 * sigma);

    const int NBINS = 36;
    double hist[NBINS] = {0.0};

    // Precompute 1D Gaussian weights for dx and dy
    double inv_2sigma2 = 1.0 / (2.0 * sigma * sigma);
    double exp_weight[65];
    for (int d = -radius; d <= radius; d++) {
        exp_weight[d + radius] = std::exp(-(d * d) * inv_2sigma2);
    }

    for (int dy = -radius; dy <= radius; dy++) {
        int y = kp.y + dy;
        if (y <= 0 || y >= oct.height - 1) continue;
        double wy = exp_weight[dy + radius];
        const double* img_y = img[y];
        const double* img_yp1 = img[y + 1];
        const double* img_ym1 = img[y - 1];

        for (int dx = -radius; dx <= radius; dx++) {
            int x = kp.x + dx;
            if (x <= 0 || x >= oct.width - 1) continue;

            double gx = img_y[x + 1] - img_y[x - 1];
            double gy = img_yp1[x] - img_ym1[x];
            double mag = std::sqrt(gx * gx + gy * gy);
            double angle = std::atan2(gy, gx); // (-pi, pi]

            double weight = wy * exp_weight[dx + radius];
            int bin = (int)std::round((angle + M_PI) / (2 * M_PI) * NBINS) % NBINS;
            hist[bin] += mag * weight;
        }
    }

    int best = 0;
    for (int b = 1; b < NBINS; b++)
        if (hist[b] > hist[best]) best = b;
    kp.orientation = best * (2 * M_PI / NBINS) - M_PI;
}

// Standard 4x4 cell x 8 orientation bin descriptor, sampled in a 16x16
// window rotated to the keypoint orientation. Each keypoint's descriptor is
// independent of every other's -- embarrassingly parallel across keypoints.
void computeDescriptor(Keypoint& kp, const Octave& oct) {
    const Mat& img = oct.gaussian[kp.layer];
    const int WINDOW = 16, CELLS = 4, BINS = 8;
    double cosA = std::cos(kp.orientation), sinA = std::sin(kp.orientation);

    double desc[128] = {0.0};

    for (int i = -WINDOW / 2; i < WINDOW / 2; i++) {
        for (int j = -WINDOW / 2; j < WINDOW / 2; j++) {
            // rotate sample offset into the keypoint's dominant orientation
            double rx = j * cosA - i * sinA;
            double ry = j * sinA + i * cosA;
            int x = kp.x + (int)std::round(rx);
            int y = kp.y + (int)std::round(ry);
            if (x <= 0 || x >= oct.width - 1 || y <= 0 || y >= oct.height - 1) continue;

            double gx = img[y][x + 1] - img[y][x - 1];
            double gy = img[y + 1][x] - img[y - 1][x];
            double mag = std::sqrt(gx * gx + gy * gy);
            double angle = std::atan2(gy, gx) - kp.orientation;
            while (angle < 0) angle += 2 * M_PI;
            while (angle >= 2 * M_PI) angle -= 2 * M_PI;

            double weight = std::exp(-(rx * rx + ry * ry) / (2 * (WINDOW / 2.0) * (WINDOW / 2.0)));

            int cellX = std::min(CELLS - 1, (i + WINDOW / 2) * CELLS / WINDOW);
            int cellY = std::min(CELLS - 1, (j + WINDOW / 2) * CELLS / WINDOW);
            int bin = std::min(BINS - 1, (int)(angle / (2 * M_PI) * BINS));

            desc[(cellY * CELLS + cellX) * BINS + bin] += mag * weight;
        }
    }

    double norm = 0.0;
    for (int k = 0; k < 128; k++) norm += desc[k] * desc[k];
    norm = std::sqrt(norm) + 1e-12;
    for (int k = 0; k < 128; k++) desc[k] = std::min(desc[k] / norm, 0.2); // clip large gradients

    norm = 0.0;
    for (int k = 0; k < 128; k++) norm += desc[k] * desc[k];
    norm = std::sqrt(norm) + 1e-12;
    for (int k = 0; k < 128; k++) desc[k] /= norm;

    kp.descriptor.assign(desc, desc + 128);
}

struct FeatureSet {
    std::vector<Keypoint> keypoints;
};

FeatureSet extractFeatures(const Mat& gray, int height, int width, BufferPool& pool) {
    auto t0 = std::chrono::high_resolution_clock::now();
    auto octaves = buildPyramid(gray, height, width, pool);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto keypoints = detectKeypoints(octaves);
    for (auto& oct : octaves) {
        oct.dog.clear();
        oct.dog.shrink_to_fit();
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (size_t i = 0; i < keypoints.size(); i++) {
        assignOrientation(keypoints[i], octaves[keypoints[i].octave]);
        computeDescriptor(keypoints[i], octaves[keypoints[i].octave]);
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    if (getenv("PROFILE")) {
        typedef std::chrono::high_resolution_clock::time_point TP;
        auto ms = [](TP a, TP b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        std::cerr << "  -- buildPyramid: " << ms(t0, t1) << " ms\n"
                  << "  -- detectKeypoints: " << ms(t1, t2) << " ms\n"
                  << "  -- orientation+desc: " << ms(t2, t3) << " ms\n";
    }
    FeatureSet fs;
    fs.keypoints = std::move(keypoints);
    return fs;
}

// image-space coordinates, for reporting (octave 0 = full resolution)
void imageCoords(const Keypoint& kp, double& ix, double& iy) {
    double scaleFactor = std::pow(2.0, kp.octave);
    ix = kp.x * scaleFactor;
    iy = kp.y * scaleFactor;
}

// ---------- feature matching ----------

struct Match {
    int idxA, idxB;
    double distance;
};

inline double descriptorDistSqEarly(const double* a, const double* b, double second_sq) {
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    for (int i = 0; i < 128; i += 32) {
        for (int k = i; k < i + 32; k += 8) {
            double d0 = a[k+0] - b[k+0];
            double d1 = a[k+1] - b[k+1];
            double d2 = a[k+2] - b[k+2];
            double d3 = a[k+3] - b[k+3];
            double d4 = a[k+4] - b[k+4];
            double d5 = a[k+5] - b[k+5];
            double d6 = a[k+6] - b[k+6];
            double d7 = a[k+7] - b[k+7];
            s0 += d0 * d0; s1 += d1 * d1; s2 += d2 * d2; s3 += d3 * d3;
            s0 += d4 * d4; s1 += d5 * d5; s2 += d6 * d6; s3 += d7 * d7;
        }
        if ((s0 + s1) + (s2 + s3) >= second_sq) return (s0 + s1) + (s2 + s3);
    }
    return (s0 + s1) + (s2 + s3);
}

// Brute-force nearest neighbor + Lowe's ratio test. Each query keypoint in A
// is matched independently against all of B -- parallelize over A.
const double RATIO_THRESH_SQ = 0.75 * 0.75; // 0.5625

std::vector<Match> matchFeatures(const FeatureSet& a, const FeatureSet& b) {
    const size_t numA = a.keypoints.size();
    const size_t numB = b.keypoints.size();
    std::vector<Match> match_per_kp(numA, {-1, -1, 0.0});

    // Flatten all descriptors of B into a single contiguous cache-friendly block
    std::vector<double> flatB(numB * 128);
    for (size_t j = 0; j < numB; j++) {
        memcpy(flatB.data() + j * 128, b.keypoints[j].descriptor.data(), 128 * sizeof(double));
    }
    const double* b_ptr = flatB.data();

    #pragma omp parallel for schedule(dynamic, 16)
    for (size_t i = 0; i < numA; i++) {
        const double* descA = a.keypoints[i].descriptor.data();
        double best_sq = 1e18, second_sq = 1e18;
        int bestIdx = -1;

        for (size_t j = 0; j < numB; j++) {
            const double* descB = b_ptr + j * 128;
            double d_sq = descriptorDistSqEarly(descA, descB, second_sq);
            if (d_sq < best_sq) {
                second_sq = best_sq;
                best_sq = d_sq;
                bestIdx = (int)j;
            } else if (d_sq < second_sq) {
                second_sq = d_sq;
            }
        }
        if (bestIdx >= 0 && best_sq < RATIO_THRESH_SQ * second_sq) {
            match_per_kp[i] = {(int)i, bestIdx, std::sqrt(best_sq)};
        }
    }

    std::vector<Match> matches;
    for (size_t i = 0; i < numA; i++) {
        if (match_per_kp[i].idxA != -1) {
            matches.push_back(match_per_kp[i]);
        }
    }
    return matches;
}

// ---------- output ----------

void writeOutput(const char* path, const FeatureSet& a, const FeatureSet& b,
                  const std::vector<Match>& matches) {
    std::ofstream out(path);
    char out_buf[1024 * 1024];
    out.rdbuf()->pubsetbuf(out_buf, sizeof(out_buf));

    out << "KEYPOINTS_A " << a.keypoints.size() << "\n";
    for (const auto& kp : a.keypoints) {
        double ix, iy;
        imageCoords(kp, ix, iy);
        out << ix << " " << iy << " " << kp.scale << " " << kp.orientation << "\n";
    }

    out << "KEYPOINTS_B " << b.keypoints.size() << "\n";
    for (const auto& kp : b.keypoints) {
        double ix, iy;
        imageCoords(kp, ix, iy);
        out << ix << " " << iy << " " << kp.scale << " " << kp.orientation << "\n";
    }

    out << "MATCHES " << matches.size() << "\n";
    for (const auto& m : matches)
        out << m.idxA << " " << m.idxB << " " << m.distance << "\n";
}

// ---------- driver ----------

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <imageA.png> <imageB.png> <output.txt>" << std::endl;
        return -1;
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    init_lut();
    init_gaussian_kernels();

    Mat grayA, grayB;
    int heightA = 0, widthA = 0;
    int heightB = 0, widthB = 0;

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            read_png_to_gray(argv[1], grayA, heightA, widthA);
        }
        #pragma omp section
        {
            read_png_to_gray(argv[2], grayB, heightB, widthB);
        }
    }
    auto t_read = std::chrono::high_resolution_clock::now();

    BufferPool pool;
    FeatureSet featuresA = extractFeatures(grayA, heightA, widthA, pool);
    grayA = Mat();
    pool.reset();
    auto t_featA = std::chrono::high_resolution_clock::now();

    FeatureSet featuresB = extractFeatures(grayB, heightB, widthB, pool);
    grayB = Mat();
    pool.reset();
    auto t_featB = std::chrono::high_resolution_clock::now();

    std::vector<Match> matches = matchFeatures(featuresA, featuresB);
    auto t_match = std::chrono::high_resolution_clock::now();

    writeOutput(argv[3], featuresA, featuresB, matches);
    auto t_write = std::chrono::high_resolution_clock::now();

    if (getenv("PROFILE")) {
        typedef std::chrono::high_resolution_clock::time_point TP;
        auto ms = [](TP a, TP b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        std::cerr << "Read PNG+Gray: " << ms(t0, t_read) << " ms\n"
                  << "Extract A:     " << ms(t_read, t_featA) << " ms (kps: " << featuresA.keypoints.size() << ")\n"
                  << "Extract B:     " << ms(t_featA, t_featB) << " ms (kps: " << featuresB.keypoints.size() << ")\n"
                  << "Match:         " << ms(t_featB, t_match) << " ms (matches: " << matches.size() << ")\n"
                  << "Write:         " << ms(t_match, t_write) << " ms\n"
                  << "Total:         " << ms(t0, t_write) << " ms\n";
    }

    return 0;
}
