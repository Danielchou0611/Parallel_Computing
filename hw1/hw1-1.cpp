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
#include <cstring>

// ---------- adaptive filtering ----------

struct SAT_RGB {
    uint32_t r, g, b;
};

inline double calculateLuminance(uint8_t r, uint8_t g, uint8_t b) {
    return 0.299 * r + 0.587 * g + 0.114 * b;
}

void adaptiveFilterRGB(
    const uint8_t* in_data,
    uint8_t* out_data,
    int height, 
    int width
) {
    int padH = height + 10;
    int padW = width + 10;
    int satH = padH + 1;
    int satW = padW + 1;
    size_t sat_sz = (size_t)satH * satW;

    SAT_RGB* sat = (SAT_RGB*)malloc(sat_sz * sizeof(SAT_RGB));
    std::memset(sat, 0, satW * sizeof(SAT_RGB));

    // Step 1: Compute row prefix sums
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < padH; r++) {
        int orig_r = std::min(std::max(r - 5, 0), height - 1);
        const uint8_t* row = in_data + (size_t)orig_r * width * 3;

        uint32_t r_sum = 0, g_sum = 0, b_sum = 0;
        int row_offset = (r + 1) * satW;
        sat[row_offset] = {0, 0, 0};

        for (int c = 0; c < padW; c++) {
            int orig_c = std::min(std::max(c - 5, 0), width - 1);
            const uint8_t* p = row + orig_c * 3;
            r_sum += p[0];
            g_sum += p[1];
            b_sum += p[2];
            sat[row_offset + (c + 1)] = {r_sum, g_sum, b_sum};
        }
    }

    // Step 2: Compute column prefix sums
    #pragma omp parallel for schedule(static)
    for (int c = 1; c <= padW; c++) {
        uint32_t r_col = 0, g_col = 0, b_col = 0;
        for (int r = 1; r <= padH; r++) {
            int idx = r * satW + c;
            r_col += sat[idx].r;
            g_col += sat[idx].g;
            b_col += sat[idx].b;
            sat[idx] = {r_col, g_col, b_col};
        }
    }

    // Step 3: Query SAT for each pixel in parallel (O(1) box sum per pixel)
    #pragma omp parallel for schedule(static)
    for (int x = 0; x < height; x++) {
        const uint8_t* in_row = in_data + (size_t)x * width * 3;
        uint8_t* out_row = out_data + (size_t)x * width * 3;

        for (int y = 0; y < width; y++) {
            const uint8_t* cur = in_row + y * 3;
            double brightness = calculateLuminance(cur[0], cur[1], cur[2]);
            int radius = (brightness > 128) ? 5 : 2;
            int count = (2 * radius + 1) * (2 * radius + 1);

            int r1 = x + 5 - radius;
            int r2 = x + 6 + radius;
            int c1 = y + 5 - radius;
            int c2 = y + 6 + radius;

            const SAT_RGB& s22 = sat[r2 * satW + c2];
            const SAT_RGB& s12 = sat[r1 * satW + c2];
            const SAT_RGB& s21 = sat[r2 * satW + c1];
            const SAT_RGB& s11 = sat[r1 * satW + c1];

            out_row[y * 3 + 0] = (s22.r - s12.r - s21.r + s11.r) / count;
            out_row[y * 3 + 1] = (s22.g - s12.g - s21.g + s11.g) / count;
            out_row[y * 3 + 2] = (s22.b - s12.b - s21.b + s11.b) / count;
        }
    }

    free(sat);
}

// ---------- shared PNG I/O ----------

uint8_t* read_png_file(char* file_name, int& width, int& height) {
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

    width = png_get_image_width(png, info);
    height = png_get_image_height(png, info);
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

    if(color_type & PNG_COLOR_MASK_ALPHA)
        png_set_strip_alpha(png);

    if(color_type == PNG_COLOR_TYPE_GRAY ||
       color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);

    png_read_update_info(png, info);

    size_t row_bytes = (size_t)width * 3;
    uint8_t* raw_data = (uint8_t*)malloc(row_bytes * height);
    png_bytep* row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
    for(int y = 0; y < height; y++) {
        row_pointers[y] = raw_data + y * row_bytes;
    }

    png_read_image(png, row_pointers);
    fclose(fp);
    free(row_pointers);
    png_destroy_read_struct(&png, &info, nullptr);

    return raw_data;
}

void write_png_file(char* file_name, const uint8_t* image_data, int width, int height) {
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

    size_t row_bytes = (size_t)width * 3;
    png_bytep* row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
    for(int y = 0; y < height; y++) {
        row_pointers[y] = (png_bytep)(image_data + y * row_bytes);
    }

    png_write_image(png, row_pointers);
    png_write_end(png, nullptr);

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

    int width = 0, height = 0;
    uint8_t* in_data = read_png_file(input_file, width, height);
    uint8_t* out_data = (uint8_t*)malloc((size_t)width * height * 3);

    adaptiveFilterRGB(in_data, out_data, height, width);

    write_png_file(output_file, out_data, width, height);

    free(in_data);
    free(out_data);

    return 0;
}