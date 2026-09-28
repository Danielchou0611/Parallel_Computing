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
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// ---------- adaptive filtering ----------

struct SAT_RGB {
    uint32_t r, g, b;
};

void adaptiveFilterRGB(
    const uint8_t* in_data,
    uint8_t* out_data,
    int height, 
    int width,
    int channels
) {
    int padH = height + 10;
    int padW = width + 10;
    int satH = padH + 1;
    int satW = padW + 1;
    size_t sat_sz = (size_t)satH * satW;

    SAT_RGB* sat = (SAT_RGB*)malloc(sat_sz * sizeof(SAT_RGB));
    std::memset(sat, 0, satW * sizeof(SAT_RGB));

    // Step 1: Compute row prefix sums with border peeling
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < padH; r++) {
        int orig_r = std::min(std::max(r - 5, 0), height - 1);
        const uint8_t* row = in_data + (size_t)orig_r * width * channels;

        uint32_t r_sum = 0, g_sum = 0, b_sum = 0;
        int row_offset = (r + 1) * satW;
        sat[row_offset] = {0, 0, 0};

        // Left border (c = 0 .. 4) -> clamped to orig_c = 0
        const uint8_t* left_p = row;
        for (int c = 0; c < 5; c++) {
            r_sum += left_p[0];
            g_sum += left_p[1];
            b_sum += left_p[2];
            sat[row_offset + (c + 1)] = {r_sum, g_sum, b_sum};
        }

        // Center region (orig_c = 0 .. width - 1)
        const uint8_t* p = row;
        for (int orig_c = 0; orig_c < width; orig_c++) {
            r_sum += p[0];
            g_sum += p[1];
            b_sum += p[2];
            p += channels;
            sat[row_offset + (orig_c + 6)] = {r_sum, g_sum, b_sum};
        }

        // Right border (c = width + 5 .. width + 9) -> clamped to orig_c = width - 1
        const uint8_t* right_p = row + (size_t)(width - 1) * channels;
        for (int c = width + 5; c < padW; c++) {
            r_sum += right_p[0];
            g_sum += right_p[1];
            b_sum += right_p[2];
            sat[row_offset + (c + 1)] = {r_sum, g_sum, b_sum};
        }
    }

    // Step 2: Compute column prefix sums with cache blocking (B = 256)
    const int B = 256;
    #pragma omp parallel for schedule(static)
    for (int cb = 1; cb <= padW; cb += B) {
        int c_end = std::min(cb + B, padW + 1);
        uint32_t col_r[256], col_g[256], col_b[256];
        std::memset(col_r, 0, (c_end - cb) * sizeof(uint32_t));
        std::memset(col_g, 0, (c_end - cb) * sizeof(uint32_t));
        std::memset(col_b, 0, (c_end - cb) * sizeof(uint32_t));

        for (int r = 1; r <= padH; r++) {
            SAT_RGB* row_ptr = &sat[r * satW];
            #pragma GCC unroll 8
            for (int c = cb; c < c_end; c++) {
                int local_c = c - cb;
                col_r[local_c] += row_ptr[c].r;
                col_g[local_c] += row_ptr[c].g;
                col_b[local_c] += row_ptr[c].b;
                row_ptr[c] = {col_r[local_c], col_g[local_c], col_b[local_c]};
            }
        }
    }

    // Step 3: Query SAT for each pixel in parallel (O(1) box sum per pixel with fast reciprocal division)
    #pragma omp parallel for schedule(static)
    for (int x = 0; x < height; x++) {
        const uint8_t* in_row = in_data + (size_t)x * width * channels;
        uint8_t* out_row = out_data + (size_t)x * width * 3;

        const SAT_RGB* sat_r2_5 = sat + (size_t)(x + 11) * satW;
        const SAT_RGB* sat_r1_5 = sat + (size_t)x * satW;
        const SAT_RGB* sat_r2_2 = sat + (size_t)(x + 8) * satW;
        const SAT_RGB* sat_r1_2 = sat + (size_t)(x + 3) * satW;

        for (int y = 0; y < width; y++) {
            const uint8_t* cur = in_row + y * channels;
            // Exact integer luminance check: 299*r + 587*g + 114*b > 128000
            bool is_bright = (299u * cur[0] + 587u * cur[1] + 114u * cur[2]) > 128000u;

            const SAT_RGB* r2_ptr = is_bright ? sat_r2_5 : sat_r2_2;
            const SAT_RGB* r1_ptr = is_bright ? sat_r1_5 : sat_r1_2;
            int c1 = is_bright ? y : (y + 3);
            int c2 = is_bright ? (y + 11) : (y + 8);
            uint32_t mul = is_bright ? 4333u : 20972u; // >> 19 exact division by 121 or 25 without idiv

            const SAT_RGB& s22 = r2_ptr[c2];
            const SAT_RGB& s12 = r1_ptr[c2];
            const SAT_RGB& s21 = r2_ptr[c1];
            const SAT_RGB& s11 = r1_ptr[c1];

            out_row[y * 3 + 0] = ((s22.r - s12.r - s21.r + s11.r) * mul) >> 19;
            out_row[y * 3 + 1] = ((s22.g - s12.g - s21.g + s11.g) * mul) >> 19;
            out_row[y * 3 + 2] = ((s22.b - s12.b - s21.b + s11.b) * mul) >> 19;
        }
    }

    free(sat);
}

// ---------- shared PNG I/O (mmap accelerated) ----------

struct MemReader {
    const uint8_t* buf;
    size_t offset;
};

static void mem_read_fn(png_structp png, png_bytep data, png_size_t length) {
    MemReader* reader = (MemReader*)png_get_io_ptr(png);
    memcpy(data, reader->buf + reader->offset, length);
    reader->offset += length;
}

uint8_t* read_png_file(char* file_name, int& width, int& height, int& channels) {
    int fd = open(file_name, O_RDONLY);
    if (fd < 0) {
        std::cerr << "Error: Cannot open file " << file_name << std::endl;
        exit(EXIT_FAILURE);
    }
    struct stat st;
    fstat(fd, &st);
    size_t file_size = st.st_size;
    const uint8_t* mapped = (const uint8_t*)mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) {
        std::cerr << "Error: mmap failed on " << file_name << std::endl;
        exit(EXIT_FAILURE);
    }
    madvise((void*)mapped, file_size, MADV_WILLNEED | MADV_SEQUENTIAL);

    MemReader reader = {mapped, 0};

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) exit(EXIT_FAILURE);

    png_infop info = png_create_info_struct(png);
    if (!info) exit(EXIT_FAILURE);

    if (setjmp(png_jmpbuf(png))) {
        std::cerr << "Error during PNG creation" << std::endl;
        png_destroy_read_struct(&png, &info, nullptr);
        munmap((void*)mapped, file_size);
        exit(EXIT_FAILURE);
    }

    png_set_read_fn(png, &reader, mem_read_fn);
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

    if(color_type == PNG_COLOR_TYPE_GRAY ||
       color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);

    // Keep RGBA if present to avoid expensive single-threaded strip_alpha transformation
    png_read_update_info(png, info);

    channels = png_get_channels(png, info);
    size_t row_bytes = png_get_rowbytes(png, info);
    uint8_t* raw_data = (uint8_t*)malloc(row_bytes * height);
    png_bytep* row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
    for(int y = 0; y < height; y++) {
        row_pointers[y] = raw_data + y * row_bytes;
    }

    png_read_image(png, row_pointers);
    free(row_pointers);
    png_destroy_read_struct(&png, &info, nullptr);
    munmap((void*)mapped, file_size);

    return raw_data;
}

void write_png_file(char* file_name, const uint8_t* image_data, int width, int height) {
    FILE *fp = fopen(file_name, "wb");
    if (!fp) {
        std::cerr << "Error: Cannot open file " << file_name << std::endl;
        exit(EXIT_FAILURE);
    }

    char fbuf[1 << 20];
    setvbuf(fp, fbuf, _IOFBF, sizeof(fbuf));

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) {
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_write_struct(&png, nullptr);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    if (setjmp(png_jmpbuf(png))) {
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

    int width = 0, height = 0, channels = 3;
    uint8_t* in_data = read_png_file(input_file, width, height, channels);
    uint8_t* out_data = (uint8_t*)malloc((size_t)width * height * 3);

    adaptiveFilterRGB(in_data, out_data, height, width, channels);

    write_png_file(output_file, out_data, width, height);

    free(in_data);
    free(out_data);

    return 0;
}