#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <png.h>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
// #include <pthread.h>
#include <omp.h>
#include <cstring>

// ---------- shared PNG I/O ----------

struct RGB {
    int r, g, b;
};

using Mat = std::vector<std::vector<double>>;

void read_png_file(const char* file_name, std::vector<std::vector<RGB>>& image) {
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

    int width = png_get_image_width(png, info);
    int height = png_get_image_height(png, info);
    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth = png_get_bit_depth(png, info);

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

    png_bytep* row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
    for (int y = 0; y < height; y++)
        row_pointers[y] = (png_byte*)malloc(png_get_rowbytes(png, info));
    png_read_image(png, row_pointers);
    fclose(fp);

    image.resize(height, std::vector<RGB>(width));
    for (int y = 0; y < height; y++) {
        png_bytep row = row_pointers[y];
        for (int x = 0; x < width; x++) {
            png_bytep px = &(row[x * 4]);
            image[y][x].r = px[0];
            image[y][x].g = px[1];
            image[y][x].b = px[2];
        }
        free(row_pointers[y]);
    }
    free(row_pointers);
    png_destroy_read_struct(&png, &info, nullptr);
}

// ---------- grayscale + Gaussian scale space ----------

Mat toGrayscale(const std::vector<std::vector<RGB>>& image, int height, int width) {
    Mat gray(height, std::vector<double>(width));
    #pragma omp parallel for
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++) {
            const RGB& p = image[y][x];
            gray[y][x] = (0.299 * p.r + 0.587 * p.g + 0.114 * p.b) / 255.0;
        }
    return gray;
}

Mat gaussianBlur(const Mat& in, int height, int width, double sigma) {
    int radius = std::max(1, (int)std::ceil(3 * sigma));
    std::vector<double> kernel(2 * radius + 1);
    double sum = 0.0;
    for (int i = -radius; i <= radius; i++) {
        double v = std::exp(-(i * i) / (2.0 * sigma * sigma));
        kernel[i + radius] = v;
        sum += v;
    }
    for (double& v : kernel) v /= sum;

    Mat tmp(height, std::vector<double>(width));
    #pragma omp parallel for
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++) {
            double acc = 0.0;
            for (int i = -radius; i <= radius; i++) {
                int xx = std::min(std::max(x + i, 0), width - 1);
                acc += in[y][xx] * kernel[i + radius];
            }
            tmp[y][x] = acc;
        }

    Mat out(height, std::vector<double>(width));
    #pragma omp parallel for
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++) {
            double acc = 0.0;
            for (int i = -radius; i <= radius; i++) {
                int yy = std::min(std::max(y + i, 0), height - 1);
                acc += tmp[yy][x] * kernel[i + radius];
            }
            out[y][x] = acc;
        }
    return out;
}

Mat downsample2x(const Mat& in, int height, int width) {
    int nh = height / 2, nw = width / 2;
    Mat out(nh, std::vector<double>(nw));
    #pragma omp parallel for
    for (int y = 0; y < nh; y++)
        for (int x = 0; x < nw; x++)
            out[y][x] = in[2 * y][2 * x];
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

    Mat base = gray;
    int h = height, w = width;
    for (int o = 0; o < NUM_OCTAVES; o++) {
        Octave& oct = octaves[o];
        oct.height = h;
        oct.width = w;
        oct.gaussian.resize(NUM_SCALES);
        oct.gaussian[0] = base;
        for (int s = 1; s < NUM_SCALES; s++) {
            double sigma = SIGMA0 * std::pow(k, s);
            oct.gaussian[s] = gaussianBlur(base, h, w, sigma);
        }
        oct.dog.resize(NUM_SCALES - 1);
        for (int s = 0; s < NUM_SCALES - 1; s++) {
            oct.dog[s] = Mat(h, std::vector<double>(w));
            #pragma omp parallel for
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    oct.dog[s][y][x] = oct.gaussian[s + 1][y][x] - oct.gaussian[s][y][x];
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

bool isExtremum(const std::vector<Mat>& dog, int s, int x, int y) {
    double v = dog[s][y][x];
    bool isMax = true, isMin = true;
    for (int ds = -1; ds <= 1 && (isMax || isMin); ds++)
        for (int dy = -1; dy <= 1 && (isMax || isMin); dy++)
            for (int dx = -1; dx <= 1 && (isMax || isMin); dx++) {
                if (ds == 0 && dy == 0 && dx == 0) continue;
                double n = dog[s + ds][y + dy][x + dx];
                if (n >= v) isMax = false;
                if (n <= v) isMin = false;
            }
    return isMax || isMin;
}

bool passesEdgeTest(const Mat& d, int x, int y) {
    double dxx = d[y][x + 1] + d[y][x - 1] - 2 * d[y][x];
    double dyy = d[y + 1][x] + d[y - 1][x] - 2 * d[y][x];
    double dxy = (d[y + 1][x + 1] - d[y + 1][x - 1] - d[y - 1][x + 1] + d[y - 1][x]) / 4.0;
    double trace = dxx + dyy;
    double det = dxx * dyy - dxy * dxy;
    if (det <= 0) return false;
    double ratio = (trace * trace) / det;
    return ratio < (EDGE_THRESH_R + 1) * (EDGE_THRESH_R + 1) / EDGE_THRESH_R;
}

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
                    for (int x = 1; x < oct.width - 1; x++) {
                        if (std::fabs(oct.dog[s][y][x]) < CONTRAST_THRESH) continue;
                        if (!isExtremum(oct.dog, s, x, y)) continue;
                        if (!passesEdgeTest(oct.dog[s], x, y)) continue;

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

void computeDescriptor(Keypoint& kp, const Octave& oct) {
    const Mat& img = oct.gaussian[kp.layer];
    const int WINDOW = 16, CELLS = 4, BINS = 8;
    double cosA = std::cos(kp.orientation), sinA = std::sin(kp.orientation);

    std::vector<double> desc(CELLS * CELLS * BINS, 0.0);

    for (int i = -WINDOW / 2; i < WINDOW / 2; i++) {
        for (int j = -WINDOW / 2; j < WINDOW / 2; j++) {
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
    for (double& v : desc) v = std::min(v / norm, 0.2);

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
    auto octaves = buildPyramid(gray, height, width);
    auto keypoints = detectKeypoints(octaves);
    #pragma omp parallel for
    for (size_t i = 0; i < keypoints.size(); i++) {
        assignOrientation(keypoints[i], octaves[keypoints[i].octave]);
        computeDescriptor(keypoints[i], octaves[keypoints[i].octave]);
    }
    FeatureSet fs;
    fs.keypoints = std::move(keypoints);
    return fs;
}

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

const double RATIO_THRESH_SQ = 0.75 * 0.75; // 0.5625

std::vector<Match> matchFeatures(const FeatureSet& a, const FeatureSet& b) {
    const size_t numA = a.keypoints.size();
    const size_t numB = b.keypoints.size();
    std::vector<Match> match_per_kp(numA, {-1, -1, 0.0});

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

    std::vector<std::vector<RGB>> imageA, imageB;
    read_png_file(argv[1], imageA);
    read_png_file(argv[2], imageB);

    int heightA = imageA.size(), widthA = imageA[0].size();
    int heightB = imageB.size(), widthB = imageB[0].size();

    Mat grayA = toGrayscale(imageA, heightA, widthA);
    Mat grayB = toGrayscale(imageB, heightB, widthB);

    FeatureSet featuresA = extractFeatures(grayA, heightA, widthA);
    FeatureSet featuresB = extractFeatures(grayB, heightB, widthB);

    std::vector<Match> matches = matchFeatures(featuresA, featuresB);

    writeOutput(argv[3], featuresA, featuresB, matches);

    return 0;
}
