#include <iostream>
#include <vector>
#include <algorithm>
#include <png.h>
#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include <omp.h>
#include <sched.h>
#include <cstdint>

// ---------- adaptive filtering ----------

struct RGB {
    int r, g, b;
};

inline double calculateLuminance(const RGB& pixel) {
    return 0.299 * pixel.r + 0.587 * pixel.g + 0.114 * pixel.b;
}

void adaptiveFilterRGB(
    const std::vector<std::vector<RGB>>& inputImage,
    std::vector<std::vector<RGB>>& outputImage,
    int height, 
    int width
) {
    int padH = height + 10;
    int padW = width + 10;
    int satH = padH + 1;
    int satW = padW + 1;

    // SAT buffers for R, G, B
    std::vector<uint32_t> satR(satH * satW, 0);
    std::vector<uint32_t> satG(satH * satW, 0);
    std::vector<uint32_t> satB(satH * satW, 0);

    // Step 1: Compute row prefix sums
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < padH; r++) {
        int orig_r = std::min(std::max(r - 5, 0), height - 1);
        const auto& row = inputImage[orig_r];

        uint32_t r_sum = 0, g_sum = 0, b_sum = 0;
        int row_offset = (r + 1) * satW;

        for (int c = 0; c < padW; c++) {
            int orig_c = std::min(std::max(c - 5, 0), width - 1);
            const RGB& p = row[orig_c];
            r_sum += p.r;
            g_sum += p.g;
            b_sum += p.b;
            satR[row_offset + (c + 1)] = r_sum;
            satG[row_offset + (c + 1)] = g_sum;
            satB[row_offset + (c + 1)] = b_sum;
        }
    }

    // Step 2: Compute column prefix sums (accumulate vertically)
    #pragma omp parallel for schedule(static)
    for (int c = 1; c <= padW; c++) {
        uint32_t r_col = 0, g_col = 0, b_col = 0;
        for (int r = 1; r <= padH; r++) {
            int idx = r * satW + c;
            r_col += satR[idx];
            g_col += satG[idx];
            b_col += satB[idx];
            satR[idx] = r_col;
            satG[idx] = g_col;
            satB[idx] = b_col;
        }
    }

    // Step 3: Query SAT for each pixel in parallel (O(1) box sum per pixel)
    #pragma omp parallel for schedule(static)
    for (int x = 0; x < height; x++) {
        const auto& in_row = inputImage[x];
        auto& out_row = outputImage[x];
        for (int y = 0; y < width; y++) {
            const RGB& cur = in_row[y];
            double brightness = calculateLuminance(cur);
            int radius = (brightness > 128) ? 5 : 2;
            int count = (2 * radius + 1) * (2 * radius + 1);

            int r1 = x + 5 - radius;
            int r2 = x + 6 + radius;
            int c1 = y + 5 - radius;
            int c2 = y + 6 + radius;

            int idx22 = r2 * satW + c2;
            int idx12 = r1 * satW + c2;
            int idx21 = r2 * satW + c1;
            int idx11 = r1 * satW + c1;

            uint32_t sumR = satR[idx22] - satR[idx12] - satR[idx21] + satR[idx11];
            uint32_t sumG = satG[idx22] - satG[idx12] - satG[idx21] + satG[idx11];
            uint32_t sumB = satB[idx22] - satB[idx12] - satB[idx21] + satB[idx11];

            out_row[y].r = sumR / count;
            out_row[y].g = sumG / count;
            out_row[y].b = sumB / count;
        }
    }
}

// ---------- shared PNG I/O ----------

void read_png_file(char* file_name, std::vector<std::vector<RGB>>& image) {
    FILE *fp = fopen(file_name, "rb");
    if (!fp) {
        std::cerr << "Error: Cannot open file " << file_name << std::endl;
        exit(EXIT_FAILURE);
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) {
        std::cerr << "Error: Cannot create PNG read structure" << std::endl;
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        std::cerr << "Error: Cannot create PNG info structure" << std::endl;
        png_destroy_read_struct(&png, nullptr, nullptr);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    if (setjmp(png_jmpbuf(png))) {
        std::cerr << "Error during PNG creation" << std::endl;
        png_destroy_read_struct(&png, &info, nullptr);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    png_init_io(png, fp);
    png_read_info(png, info);

    int width = png_get_image_width(png, info);
    int height = png_get_image_height(png, info);
    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth = png_get_bit_depth(png, info);

    if(bit_depth == 16)
        png_set_strip_16(png);

    if(color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png);

    if(color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png);

    if(png_get_valid(png, info, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png);

    if(color_type == PNG_COLOR_TYPE_RGB ||
       color_type == PNG_COLOR_TYPE_GRAY ||
       color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_filler(png, 0xFF, PNG_FILLER_AFTER);

    if(color_type == PNG_COLOR_TYPE_GRAY ||
       color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);

    png_read_update_info(png, info);

    size_t row_bytes = png_get_rowbytes(png, info);
    png_bytep* row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
    png_byte* raw_data = (png_byte*)malloc(row_bytes * height);
    for(int y = 0; y < height; y++) {
        row_pointers[y] = raw_data + y * row_bytes;
    }

    png_read_image(png, row_pointers);
    fclose(fp);

    image.resize(height, std::vector<RGB>(width));
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; y++) {
        png_bytep row = row_pointers[y];
        auto& img_row = image[y];
        for (int x = 0; x < width; x++) {
            png_bytep px = &(row[x * 4]);
            img_row[x].r = px[0];
            img_row[x].g = px[1];
            img_row[x].b = px[2];
        }
    }
    free(raw_data);
    free(row_pointers);

    png_destroy_read_struct(&png, &info, nullptr);
}

void write_png_file(char* file_name, const std::vector<std::vector<RGB>>& image) {
    int width = image[0].size();
    int height = image.size();

    FILE *fp = fopen(file_name, "wb");
    if (!fp) {
        std::cerr << "Error: Cannot open file " << file_name << std::endl;
        exit(EXIT_FAILURE);
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) {
        std::cerr << "Error: Cannot create PNG write structure" << std::endl;
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        std::cerr << "Error: Cannot create PNG info structure" << std::endl;
        png_destroy_write_struct(&png, nullptr);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    if (setjmp(png_jmpbuf(png))) {
        std::cerr << "Error during PNG creation" << std::endl;
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    png_init_io(png, fp);

    png_set_IHDR(
        png,
        info,
        width, height,
        8,
        PNG_COLOR_TYPE_RGB,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT
    );
    png_set_compression_level(png, 0);
    png_set_filter(png, 0, PNG_FILTER_NONE);
    png_write_info(png, info);

    size_t row_bytes = png_get_rowbytes(png, info);
    png_bytep* row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
    png_byte* raw_data = (png_byte*)malloc(row_bytes * height);
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; y++) {
        row_pointers[y] = raw_data + y * row_bytes;
        png_bytep row = row_pointers[y];
        const auto& img_row = image[y];
        for (int x = 0; x < width; x++) {
            row[x * 3] = img_row[x].r;
            row[x * 3 + 1] = img_row[x].g;
            row[x * 3 + 2] = img_row[x].b;
        }
    }

    png_write_image(png, row_pointers);
    png_write_end(png, nullptr);

    free(raw_data);
    free(row_pointers);

    png_destroy_write_struct(&png, &info);
    fclose(fp);
}

// ---------- driver ----------

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <inputfile.png> <outputfile.png>" << std::endl;
        return -1;
    }

    cpu_set_t cpuset;
    sched_getaffinity(0, sizeof(cpuset), &cpuset);
    int ncpus = CPU_COUNT(&cpuset);
    omp_set_num_threads(ncpus);

    char* input_file = argv[1];
    char* output_file = argv[2];

    std::vector<std::vector<RGB>> inputImage;
    read_png_file(input_file, inputImage);

    int height = inputImage.size();
    int width = inputImage[0].size();

    std::vector<std::vector<RGB>> outputImage(height, std::vector<RGB>(width));

    adaptiveFilterRGB(inputImage, outputImage, height, width);

    write_png_file(output_file, outputImage);

    return 0;
}