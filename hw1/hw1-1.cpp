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

struct RGB {
    uint32_t r, g, b;
};

void sliding_box_filter(
    const uint8_t* in_data,
    uint8_t* out_data,
    int height,
    int width,
    int channels
) {
    auto get_pixel = [&](int r, int c) -> const uint8_t* {
        r = std::min(std::max(r, 0), height - 1);
        c = std::min(std::max(c, 0), width - 1);
        return in_data + ((size_t)r * width + c) * channels;
    };

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        int r_start = tid * height / nthreads;
        int r_end = (tid + 1) * height / nthreads;

        int pad_w = width + 12;
        std::vector<RGB> col11(pad_w, {0, 0, 0});
        std::vector<RGB> col5(pad_w, {0, 0, 0});

        for (int c = -5; c <= width + 5; c++) {
            int idx = c + 5;
            uint32_t r11 = 0, g11 = 0, b11 = 0;
            uint32_t r5 = 0, g5 = 0, b5 = 0;
            for (int dr = -5; dr <= 5; dr++) {
                const uint8_t* p = get_pixel(r_start + dr, c);
                r11 += p[0]; g11 += p[1]; b11 += p[2];
                if (dr >= -2 && dr <= 2) {
                    r5 += p[0]; g5 += p[1]; b5 += p[2];
                }
            }
            col11[idx] = {r11, g11, b11};
            col5[idx] = {r5, g5, b5};
        }

        for (int r = r_start; r < r_end; r++) {
            if (r > r_start) {
                int r_a11 = std::min(std::max(r + 5, 0), height - 1);
                int r_s11 = std::min(std::max(r - 6, 0), height - 1);
                int r_a5  = std::min(std::max(r + 2, 0), height - 1);
                int r_s5  = std::min(std::max(r - 3, 0), height - 1);

                const uint8_t* row_a11 = in_data + (size_t)r_a11 * width * channels;
                const uint8_t* row_s11 = in_data + (size_t)r_s11 * width * channels;
                const uint8_t* row_a5  = in_data + (size_t)r_a5  * width * channels;
                const uint8_t* row_s5  = in_data + (size_t)r_s5  * width * channels;

                // Left clamp c = -5 .. -1
                for (int c = -5; c < 0; c++) {
                    int idx = c + 5;
                    col11[idx].r += row_a11[0] - row_s11[0];
                    col11[idx].g += row_a11[1] - row_s11[1];
                    col11[idx].b += row_a11[2] - row_s11[2];

                    col5[idx].r += row_a5[0] - row_s5[0];
                    col5[idx].g += row_a5[1] - row_s5[1];
                    col5[idx].b += row_a5[2] - row_s5[2];
                }

                // Middle c = 0 .. width - 1
                for (int c = 0; c < width; c++) {
                    int idx = c + 5;
                    const uint8_t* p_a11 = row_a11 + c * channels;
                    const uint8_t* p_s11 = row_s11 + c * channels;
                    const uint8_t* p_a5  = row_a5  + c * channels;
                    const uint8_t* p_s5  = row_s5  + c * channels;

                    col11[idx].r += p_a11[0] - p_s11[0];
                    col11[idx].g += p_a11[1] - p_s11[1];
                    col11[idx].b += p_a11[2] - p_s11[2];

                    col5[idx].r += p_a5[0] - p_s5[0];
                    col5[idx].g += p_a5[1] - p_s5[1];
                    col5[idx].b += p_a5[2] - p_s5[2];
                }

                // Right clamp c = width .. width + 5
                const uint8_t* end_a11 = row_a11 + (size_t)(width - 1) * channels;
                const uint8_t* end_s11 = row_s11 + (size_t)(width - 1) * channels;
                const uint8_t* end_a5  = row_a5  + (size_t)(width - 1) * channels;
                const uint8_t* end_s5  = row_s5  + (size_t)(width - 1) * channels;
                for (int c = width; c <= width + 5; c++) {
                    int idx = c + 5;
                    col11[idx].r += end_a11[0] - end_s11[0];
                    col11[idx].g += end_a11[1] - end_s11[1];
                    col11[idx].b += end_a11[2] - end_s11[2];

                    col5[idx].r += end_a5[0] - end_s5[0];
                    col5[idx].g += end_a5[1] - end_s5[1];
                    col5[idx].b += end_a5[2] - end_s5[2];
                }
            }

            RGB box11 = {0, 0, 0};
            RGB box5 = {0, 0, 0};
            for (int dy = -5; dy <= 5; dy++) {
                int idx = dy + 5;
                box11.r += col11[idx].r; box11.g += col11[idx].g; box11.b += col11[idx].b;
                if (dy >= -2 && dy <= 2) {
                    box5.r += col5[idx].r; box5.g += col5[idx].g; box5.b += col5[idx].b;
                }
            }

            const uint8_t* in_row = in_data + (size_t)r * width * channels;
            uint8_t* out_row = out_data + (size_t)r * width * 3;

            // y = 0
            {
                const uint8_t* cur = in_row;
                bool is_bright = (299u * cur[0] + 587u * cur[1] + 114u * cur[2]) > 128000u;
                uint32_t mul = is_bright ? 4333u : 20972u;
                const RGB& b = is_bright ? box11 : box5;
                out_row[0] = (b.r * mul) >> 19;
                out_row[1] = (b.g * mul) >> 19;
                out_row[2] = (b.b * mul) >> 19;
            }

            // y = 1 .. width - 1
            for (int y = 1; y < width; y++) {
                int add11_idx = y + 10;
                int sub11_idx = y - 1;
                box11.r += col11[add11_idx].r - col11[sub11_idx].r;
                box11.g += col11[add11_idx].g - col11[sub11_idx].g;
                box11.b += col11[add11_idx].b - col11[sub11_idx].b;

                int add5_idx = y + 7;
                int sub5_idx = y + 2;
                box5.r += col5[add5_idx].r - col5[sub5_idx].r;
                box5.g += col5[add5_idx].g - col5[sub5_idx].g;
                box5.b += col5[add5_idx].b - col5[sub5_idx].b;

                const uint8_t* cur = in_row + y * channels;
                bool is_bright = (299u * cur[0] + 587u * cur[1] + 114u * cur[2]) > 128000u;
                uint32_t mul = is_bright ? 4333u : 20972u;
                const RGB& b = is_bright ? box11 : box5;

                out_row[y * 3 + 0] = (b.r * mul) >> 19;
                out_row[y * 3 + 1] = (b.g * mul) >> 19;
                out_row[y * 3 + 2] = (b.b * mul) >> 19;
            }
        }
    }
}

// ---------- shared PNG I/O (mmap accelerated + direct streaming) ----------

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

    // Skip redundant chunk CRC calculations during read
    png_set_crc_action(png, PNG_CRC_QUIET_USE, PNG_CRC_QUIET_USE);
    png_set_compression_buffer_size(png, 2 * 1024 * 1024);

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

    png_read_update_info(png, info);

    channels = png_get_channels(png, info);
    size_t row_bytes = png_get_rowbytes(png, info);
    uint8_t* raw_data = (uint8_t*)malloc(row_bytes * height);

    // Direct stream reading line by line (faster cold cache, no row_pointers allocation)
    for (int y = 0; y < height; y++) {
        png_read_row(png, raw_data + (size_t)y * row_bytes, nullptr);
    }

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

    sliding_box_filter(in_data, out_data, height, width, channels);

    write_png_file(output_file, out_data, width, height);

    free(in_data);
    free(out_data);

    return 0;
}