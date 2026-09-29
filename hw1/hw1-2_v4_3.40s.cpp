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

// ---------- shared PNG I/O ----------

struct Mat {
    int h = 0, w = 0;
    std::shared_ptr<double> buf;
    double* data = nullptr;

    Mat() = default;
    Mat(int height, int width) : h(height), w(width) {
        if (h > 0 && w > 0) {
            double* p = (double*)malloc(sizeof(double) * h * w);
            buf = std::shared_ptr<double>(p, free);
            data = p;
        }
    }

    inline double* operator[](int y) { return data + y * w; }
    inline const double* operator[](int y) const { return data + y * w; }
    inline double* data_ptr() { return data; }
    inline const double* data_ptr() const { return data; }
};

void read_png_to_gray(const char* file_name, Mat& gray, int& height, int& width) {
    FILE *fp = fopen(file_name, "rb");
    if (!fp) {
        std::cerr << "Error: Cannot open file " << file_name << std::endl;
        exit(EXIT_FAILURE);
    }

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
    if (color_type == PNG_COLOR_TYPE_RGB || color_type == PNG_COLOR_TYPE_GRAY ||
        color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_filler(png, 0xFF, PNG_FILLER_AFTER);
    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);

    png_read_update_info(png, info);

    size_t rowbytes = png_get_rowbytes(png, info);
    std::vector<png_byte> raw_buf(height * rowbytes);
    std::vector<png_bytep> row_pointers(height);
    for (int y = 0; y < height; y++)
        row_pointers[y] = &raw_buf[y * rowbytes];

    png_read_image(png, row_pointers.data());
    fclose(fp);
    png_destroy_read_struct(&png, &info, nullptr);

    gray = Mat(height, width);
    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        const png_byte* row = &raw_buf[y * rowbytes];
        double* gray_row = gray[y];
        #pragma GCC ivdep
        for (int x = 0; x < width; x++) {
            const png_byte* px = &row[x * 4];
            gray_row[x] = (0.299 * px[0] + 0.587 * px[1] + 0.114 * px[2]) / 255.0;
        }
    }
}

// Separable Gaussian blur. Two independent passes (row-wise, then column-wise)
Mat gaussianBlur(const Mat& in, int height, int width, double sigma, double* tmp_buf) {
    int radius = std::max(1, (int)std::ceil(3 * sigma));
    std::vector<double> kernel(2 * radius + 1);
    double sum = 0.0;
    for (int i = -radius; i <= radius; i++) {
        double v = std::exp(-(i * i) / (2.0 * sigma * sigma));
        kernel[i + radius] = v;
        sum += v;
    }
    for (double& v : kernel) v /= sum;

    int K = 2 * radius + 1;
    const double* k_ptr = kernel.data();

    // Pass 1: Horizontal blur with boundary splitting (vectorized middle)
    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        const double* in_row = in[y];
        double* tmp_row = tmp_buf + y * width;

        int left_end = std::min(radius, width);
        for (int x = 0; x < left_end; x++) {
            double acc = 0.0;
            for (int i = -radius; i <= radius; i++) {
                int xx = std::min(std::max(x + i, 0), width - 1);
                acc += in_row[xx] * k_ptr[i + radius];
            }
            tmp_row[x] = acc;
        }

        int right_start = std::max(radius, width - radius);
        for (int x = left_end; x < right_start; x++) {
            double acc = 0.0;
            const double* p = in_row + (x - radius);
            #pragma GCC ivdep
            for (int i = 0; i < K; i++) {
                acc += p[i] * k_ptr[i];
            }
            tmp_row[x] = acc;
        }

        for (int x = right_start; x < width; x++) {
            double acc = 0.0;
            for (int i = -radius; i <= radius; i++) {
                int xx = std::min(std::max(x + i, 0), width - 1);
                acc += in_row[xx] * k_ptr[i + radius];
            }
            tmp_row[x] = acc;
        }
    }

    // Pass 2: Vertical blur (contiguous row accumulation)
    Mat out(height, width);
    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        double* out_row = out[y];
        int y0 = std::min(std::max(y - radius, 0), height - 1);
        const double* tmp_row0 = tmp_buf + y0 * width;
        double k0 = k_ptr[0];
        #pragma GCC ivdep
        for (int x = 0; x < width; x++) {
            out_row[x] = tmp_row0[x] * k0;
        }
        for (int i = -radius + 1; i <= radius; i++) {
            int yy = std::min(std::max(y + i, 0), height - 1);
            const double* tmp_row = tmp_buf + yy * width;
            double k = k_ptr[i + radius];
            #pragma GCC ivdep
            for (int x = 0; x < width; x++) {
                out_row[x] += tmp_row[x] * k;
            }
        }
    }
    return out;
}

Mat downsample2x(const Mat& in, int height, int width) {
    int nh = height / 2, nw = width / 2;
    Mat out(nh, nw);
    #pragma omp parallel for
    for (int y = 0; y < nh; y++) {
        const double* in_row = in[2 * y];
        double* out_row = out[y];
        #pragma GCC ivdep
        for (int x = 0; x < nw; x++)
            out_row[x] = in_row[2 * x];
    }
    return out;
}

const int NUM_OCTAVES = 4;
const int S = 3;              // Lowe's scales-per-octave
const int NUM_SCALES = S + 3; // Gaussian images per octave
const double SIGMA0 = 1.6;
const double CONTRAST_THRESH = 0.03;
const double EDGE_THRESH_R = 10.0;

struct Octave {
    int height, width;
    std::vector<Mat> gaussian; // NUM_SCALES images
    std::vector<Mat> dog;      // NUM_SCALES - 1 images
};

std::vector<Octave> buildPyramid(const Mat& gray, int height, int width) {
    std::vector<Octave> octaves(NUM_OCTAVES);
    double k = std::pow(2.0, 1.0 / S);

    std::vector<double> tmp_buf(height * width);
    Mat base = gray;
    int h = height, w = width;
    for (int o = 0; o < NUM_OCTAVES; o++) {
        Octave& oct = octaves[o];
        oct.height = h;
        oct.width = w;
        oct.gaussian.resize(NUM_SCALES);
        oct.gaussian[0] = base;
        auto t_o0 = std::chrono::high_resolution_clock::now();
        for (int s = 1; s < NUM_SCALES; s++) {
            double sigma = SIGMA0 * std::pow(k, s);
            oct.gaussian[s] = gaussianBlur(base, h, w, sigma, tmp_buf.data());
        }
        auto t_o1 = std::chrono::high_resolution_clock::now();
        oct.dog.resize(NUM_SCALES - 1);
        for (int s = 0; s < NUM_SCALES - 1; s++) {
            oct.dog[s] = Mat(h, w);
        }
        int total = h * w;
        #pragma omp parallel
        {
            for (int s = 0; s < NUM_SCALES - 1; s++) {
                const double* g_next = oct.gaussian[s + 1].data_ptr();
                const double* g_curr = oct.gaussian[s].data_ptr();
                double* dog_ptr = oct.dog[s].data_ptr();
                #pragma omp for nowait
                for (int i = 0; i < total; i++)
                    dog_ptr[i] = g_next[i] - g_curr[i];
            }
        }
        auto t_o2 = std::chrono::high_resolution_clock::now();
        if (getenv("PROFILE")) {
            auto ms = [](std::chrono::high_resolution_clock::time_point a, std::chrono::high_resolution_clock::time_point b) {
                return std::chrono::duration<double, std::milli>(b - a).count();
            };
            std::cerr << "    oct " << o << " (" << w << "x" << h << ") blur: " << ms(t_o0, t_o1) << " ms, dog: " << ms(t_o1, t_o2) << " ms\n";
        }

        if (o + 1 < NUM_OCTAVES) {
            base = downsample2x(oct.gaussian[S], h, w); // carry over scale = 2*sigma0
            h /= 2;
            w /= 2;
        }
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

inline bool isExtremum(const std::vector<Mat>& dog, int s, int x, int y) {
    double v = dog[s][y][x];
    bool isMax = true, isMin = true;

    // Check same layer first (cache hot, eliminates >95% of non-extrema immediately)
    const double* r0 = dog[s][y - 1];
    const double* r1 = dog[s][y];
    const double* r2 = dog[s][y + 1];

    double n;
    #define CHK(val) do { n = (val); if (n >= v) isMax = false; if (n <= v) isMin = false; if (!isMax && !isMin) return false; } while(0)
    CHK(r0[x - 1]); CHK(r0[x]); CHK(r0[x + 1]);
    CHK(r1[x - 1]);             CHK(r1[x + 1]);
    CHK(r2[x - 1]); CHK(r2[x]); CHK(r2[x + 1]);

    // Check layer s - 1
    const double* p0 = dog[s - 1][y - 1];
    const double* p1 = dog[s - 1][y];
    const double* p2 = dog[s - 1][y + 1];
    CHK(p0[x - 1]); CHK(p0[x]); CHK(p0[x + 1]);
    CHK(p1[x - 1]); CHK(p1[x]); CHK(p1[x + 1]);
    CHK(p2[x - 1]); CHK(p2[x]); CHK(p2[x + 1]);

    // Check layer s + 1
    const double* q0 = dog[s + 1][y - 1];
    const double* q1 = dog[s + 1][y];
    const double* q2 = dog[s + 1][y + 1];
    CHK(q0[x - 1]); CHK(q0[x]); CHK(q0[x + 1]);
    CHK(q1[x - 1]); CHK(q1[x]); CHK(q1[x + 1]);
    CHK(q2[x - 1]); CHK(q2[x]); CHK(q2[x + 1]);
    #undef CHK

    return isMax || isMin;
}

inline bool passesEdgeTest(const double* ym1, const double* y0, const double* yp1, int x) {
    double dxx = y0[x + 1] + y0[x - 1] - 2 * y0[x];
    double dyy = yp1[x] + ym1[x] - 2 * y0[x];
    double dxy = (yp1[x + 1] - yp1[x - 1] - ym1[x + 1] + ym1[x]) / 4.0;
    double trace = dxx + dyy;
    double det = dxx * dyy - dxy * dxy;
    if (det <= 0) return false;
    double ratio = (trace * trace) / det;
    return ratio < (EDGE_THRESH_R + 1) * (EDGE_THRESH_R + 1) / EDGE_THRESH_R;
}

// Independent per (octave, layer, pixel) -- another natural parallelization
// target. Extrema in different octaves/layers never interact.
std::vector<Keypoint> detectKeypoints(const std::vector<Octave>& octaves) {
    std::vector<Keypoint> keypoints;
    int max_threads = omp_get_max_threads();

    for (int o = 0; o < NUM_OCTAVES; o++) {
        const Octave& oct = octaves[o];
        for (int s = 1; s < (int)oct.dog.size() - 1; s++) {
            std::vector<std::vector<Keypoint>> thread_kps(max_threads);

            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                #pragma omp for schedule(static)
                for (int y = 1; y < oct.height - 1; y++) {
                    const double* d_ym1 = oct.dog[s][y - 1];
                    const double* d_y0  = oct.dog[s][y];
                    const double* d_yp1 = oct.dog[s][y + 1];
                    for (int x = 1; x < oct.width - 1; x++) {
                        if (std::fabs(d_y0[x]) < CONTRAST_THRESH) continue;
                        if (!isExtremum(oct.dog, s, x, y)) continue;
                        if (!passesEdgeTest(d_ym1, d_y0, d_yp1, x)) continue;

                        Keypoint kp;
                        kp.octave = o;
                        kp.layer = s;
                        kp.x = x;
                        kp.y = y;
                        kp.scale = SIGMA0 * std::pow(2.0, (double)s / S);
                        thread_kps[tid].push_back(kp);
                    }
                }
            }

            for (int t = 0; t < max_threads; t++) {
                keypoints.insert(keypoints.end(), thread_kps[t].begin(), thread_kps[t].end());
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
    std::vector<double> hist(NBINS, 0.0);

    for (int dy = -radius; dy <= radius; dy++) {
        int y = kp.y + dy;
        if (y <= 0 || y >= oct.height - 1) continue;
        for (int dx = -radius; dx <= radius; dx++) {
            int x = kp.x + dx;
            if (x <= 0 || x >= oct.width - 1) continue;

            double gx = img[y][x + 1] - img[y][x - 1];
            double gy = img[y + 1][x] - img[y - 1][x];
            double mag = std::sqrt(gx * gx + gy * gy);
            double angle = std::atan2(gy, gx); // (-pi, pi]

            double weight = std::exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
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

    std::vector<double> desc(CELLS * CELLS * BINS, 0.0);

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
    for (double v : desc) norm += v * v;
    norm = std::sqrt(norm) + 1e-12;
    for (double& v : desc) v = std::min(v / norm, 0.2); // clip large gradients (illumination robustness)

    norm = 0.0;
    for (double v : desc) norm += v * v;
    norm = std::sqrt(norm) + 1e-12;
    for (double& v : desc) v /= norm;

    kp.descriptor = desc;
}

struct FeatureSet {
    std::vector<Keypoint> keypoints;
};

FeatureSet extractFeatures(const Mat& gray, int height, int width) {
    auto t0 = std::chrono::high_resolution_clock::now();
    auto octaves = buildPyramid(gray, height, width);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto keypoints = detectKeypoints(octaves);
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

inline double descriptorDistSq(const double* a, const double* b) {
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    for (int i = 0; i < 128; i += 8) {
        double d0 = a[i+0] - b[i+0];
        double d1 = a[i+1] - b[i+1];
        double d2 = a[i+2] - b[i+2];
        double d3 = a[i+3] - b[i+3];
        double d4 = a[i+4] - b[i+4];
        double d5 = a[i+5] - b[i+5];
        double d6 = a[i+6] - b[i+6];
        double d7 = a[i+7] - b[i+7];
        s0 += d0 * d0;
        s1 += d1 * d1;
        s2 += d2 * d2;
        s3 += d3 * d3;
        s0 += d4 * d4;
        s1 += d5 * d5;
        s2 += d6 * d6;
        s3 += d7 * d7;
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
            double d_sq = descriptorDistSq(descA, descB);
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

    FeatureSet featuresA = extractFeatures(grayA, heightA, widthA);
    auto t_featA = std::chrono::high_resolution_clock::now();

    FeatureSet featuresB = extractFeatures(grayB, heightB, widthB);
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
