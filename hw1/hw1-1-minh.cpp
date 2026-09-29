// HW 1-1: Adaptive denoising filter
//
// The filter kernel itself is already a cache-resident sliding-window box sum and
// costs under 10% of the wall clock. The remaining time is libpng: zlib inflate on
// read and the PNG encoder on write are both single-threaded, so 7 of 8 cores idle
// through them. This version takes over both ends:
//
//   read   - parse the PNG chunks directly, inflate the IDAT stream in one shot
//            (libdeflate when available, zlib otherwise), unfilter in place
//   filter - unchanged sliding-window box sum, but writing straight into the final
//            PNG byte stream, so there is no separate output image to copy
//   write  - hand-built PNG: stored deflate blocks cut on row boundaries, one IDAT
//            chunk per thread, adler32/crc32 computed per segment in parallel
//
// Falls back to libpng for any input this fast path does not cover.

#include <iostream>
#include <vector>
#include <algorithm>
#include <png.h>
#include <zlib.h>
#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include <omp.h>
#include <cstdint>
#include <cstring>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#ifdef __linux__
#include <sched.h>
#endif

#ifdef USE_LIBDEFLATE
#include <libdeflate.h>
#endif

#ifdef HW1_TIMING
#define TIMER_DECL(name) std::chrono::high_resolution_clock::time_point name
#define TIMER_NOW(name) name = std::chrono::high_resolution_clock::now()
static double g_t_inflate = 0, g_t_unfilter = 0;
#define TIMER_ACC(var, a, b) \
    var += std::chrono::duration<double, std::milli>((b) - (a)).count()
#else
#define TIMER_DECL(name)
#define TIMER_NOW(name)
#define TIMER_ACC(var, a, b)
#endif

// Large working buffers come from mmap rather than malloc so the kernel can back
// them with huge pages: a 54 MB buffer is 13k faults at 4 KB but only 27 at 2 MB,
// and those faults are taken on the critical path the first time each page is
// touched. They are never unmapped - the process is about to exit and letting
// teardown happen in exit() keeps it off the measured path.
static void* alloc_big(size_t n) {
#ifdef __linux__
    void* p = mmap(nullptr, n, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p != MAP_FAILED) {
#ifdef MADV_HUGEPAGE
        madvise(p, n, MADV_HUGEPAGE);
#endif
        return p;
    }
#endif
    return malloc(n);
}

struct RGB {
    uint32_t r, g, b;
};

static volatile uint8_t g_prefault_sink;

// ---------------------------------------------------------------- PNG decode

static inline uint32_t be32(const uint8_t* p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

// Decoded image kept in PNG scanline form: every row is one filter byte followed
// by width*channels sample bytes. Keeping the filter byte in place means the
// inflate output needs no second buffer and no copy.
struct FastPng {
    uint8_t* buf = nullptr;   // owns height*stride bytes
    size_t stride = 0;        // 1 + width*channels
    int width = 0, height = 0, channels = 0;
};

static inline int paeth(int a, int b, int c) {
    int p = a + b - c;
    int pa = p > a ? p - a : a - p;
    int pb = p > b ? p - b : b - p;
    int pc = p > c ? p - c : c - p;
    if (pa <= pb && pa <= pc) return a;
    if (pb <= pc) return b;
    return c;
}

// Byte-wise addition of four packed bytes with the carries kept inside each
// byte: the low 7 bits are summed normally, and bit 7 is rebuilt from the carry
// out of bit 6 xored with both operands' top bits.
static inline uint32_t swar_add4(uint32_t a, uint32_t b) {
    return ((a & 0x7F7F7F7Fu) + (b & 0x7F7F7F7Fu)) ^ ((a ^ b) & 0x80808080u);
}

static inline uint32_t load32(const uint8_t* p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

static inline void store32(uint8_t* p, uint32_t v) {
    memcpy(p, &v, 4);
}

// Sub is a serial chain (x[i] += x[i-bpp]) so a compiler cannot vectorise it,
// but the chain is bpp bytes wide, which means one whole pixel can be carried
// in a register and added in a single SWAR step.
static void unfilter_sub4(uint8_t* x, size_t n) {
    uint32_t acc = load32(x);
    size_t i = 4;
    for (; i + 4 <= n; i += 4) {
        acc = swar_add4(load32(x + i), acc);
        store32(x + i, acc);
    }
    for (; i < n; i++) x[i] = (uint8_t)(x[i] + x[i - 4]);
}

static void unfilter_sub3(uint8_t* x, size_t n) {
    // Three bytes per step, reading four. The reconstructed pixel stays in a
    // register, so only the incoming pixel is loaded each round; the fourth byte
    // of the accumulator is garbage but SWAR never carries across a byte, so it
    // cannot corrupt the three bytes that get stored.
    size_t i = 3;
    uint32_t acc = load32(x);
    for (; i + 4 <= n; i += 3) {
        uint32_t sum = swar_add4(load32(x + i), acc);
        x[i]     = (uint8_t)sum;
        x[i + 1] = (uint8_t)(sum >> 8);
        x[i + 2] = (uint8_t)(sum >> 16);
        acc = sum;
    }
    for (; i < n; i++) x[i] = (uint8_t)(x[i] + x[i - 3]);
}

// In-place reconstruction of one scanline. `prev` is the already-reconstructed
// row above, or a zero row for row 0.
static void unfilter_row(uint8_t* cur, const uint8_t* prev, size_t n, int bpp) {
    uint8_t ft = cur[0];
    uint8_t* x = cur + 1;

    switch (ft) {
        case 0:  // None
            break;
        case 1:  // Sub
            if (bpp == 4 && n >= 4) unfilter_sub4(x, n);
            else if (bpp == 3 && n >= 3) unfilter_sub3(x, n);
            else for (size_t i = bpp; i < n; i++) x[i] = (uint8_t)(x[i] + x[i - bpp]);
            break;
        case 2:  // Up
            for (size_t i = 0; i < n; i++) x[i] = (uint8_t)(x[i] + prev[i]);
            break;
        case 3:  // Average
            for (size_t i = 0; i < (size_t)bpp && i < n; i++)
                x[i] = (uint8_t)(x[i] + (prev[i] >> 1));
            for (size_t i = bpp; i < n; i++)
                x[i] = (uint8_t)(x[i] + (((int)x[i - bpp] + (int)prev[i]) >> 1));
            break;
        case 4:  // Paeth
            for (size_t i = 0; i < (size_t)bpp && i < n; i++)
                x[i] = (uint8_t)(x[i] + prev[i]);
            for (size_t i = bpp; i < n; i++)
                x[i] = (uint8_t)(x[i] + paeth(x[i - bpp], prev[i], prev[i - bpp]));
            break;
        default:
            break;
    }
}

static void unfilter_image(FastPng& img) {
    const int bpp = img.channels;
    const size_t n = (size_t)img.width * img.channels;

    std::vector<uint8_t> zero(n, 0);
    const uint8_t* prev = zero.data();

    for (int r = 0; r < img.height; r++) {
        uint8_t* cur = img.buf + (size_t)r * img.stride;
        unfilter_row(cur, prev, n, bpp);
        prev = cur + 1;
    }
}

// Inflating the whole stream and then unfiltering it costs an extra round trip
// through memory for images far larger than L2. Interleaving the two at a chunk
// size that fits in cache keeps each scanline hot between the two passes.
static const size_t UNFILTER_CHUNK_BYTES = 128 * 1024;

// Returns false when the file uses a feature the fast path does not handle
// (interlacing, 16-bit samples, palette or grayscale); caller then uses libpng.
static bool fast_png_decode(const uint8_t* file, size_t file_size, FastPng& out) {
    static const uint8_t sig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    if (file_size < 8 + 25 || memcmp(file, sig, 8) != 0) return false;

    int width = 0, height = 0, channels = 0;
    bool have_ihdr = false;

    // First pass: read IHDR and total up the compressed payload so the IDAT
    // segments can be gathered without reallocating.
    struct Chunk { const uint8_t* data; uint32_t len; };
    std::vector<Chunk> idat;
    size_t idat_total = 0;

    size_t pos = 8;
    while (pos + 8 <= file_size) {
        uint32_t len = be32(file + pos);
        const uint8_t* type = file + pos + 4;
        const uint8_t* data = file + pos + 8;
        if (pos + 12 + (size_t)len > file_size) return false;

        if (memcmp(type, "IHDR", 4) == 0) {
            if (len != 13) return false;
            width = (int)be32(data);
            height = (int)be32(data + 4);
            int bit_depth = data[8];
            int color_type = data[9];
            int compression = data[10];
            int filter_method = data[11];
            int interlace = data[12];

            if (bit_depth != 8 || compression != 0 || filter_method != 0 || interlace != 0)
                return false;
            if (color_type == 2) channels = 3;
            else if (color_type == 6) channels = 4;
            else return false;

            if (width <= 0 || height <= 0) return false;
            have_ihdr = true;
        } else if (memcmp(type, "IDAT", 4) == 0) {
            idat.push_back({data, len});
            idat_total += len;
        } else if (memcmp(type, "IEND", 4) == 0) {
            break;
        }

        pos += 12 + (size_t)len;
    }

    if (!have_ihdr || idat.empty()) return false;

    out.width = width;
    out.height = height;
    out.channels = channels;
    out.stride = (size_t)1 + (size_t)width * channels;

    size_t raw_size = out.stride * (size_t)height;
    out.buf = (uint8_t*)alloc_big(raw_size);
    if (!out.buf) return false;

#ifdef USE_LIBDEFLATE
    // libdeflate is a one-shot API, so multiple IDAT chunks have to be joined
    // first. Most encoders emit one large chunk, in which case this is free.
    const uint8_t* zsrc;
    uint32_t zlen;
    std::vector<uint8_t> joined;
    if (idat.size() == 1) {
        zsrc = idat[0].data;
        zlen = idat[0].len;
    } else {
        joined.resize(idat_total);
        size_t off = 0;
        for (size_t i = 0; i < idat.size(); i++) {
            memcpy(joined.data() + off, idat[i].data, idat[i].len);
            off += idat[i].len;
        }
        zsrc = joined.data();
        zlen = (uint32_t)idat_total;
    }

    struct libdeflate_decompressor* dec = libdeflate_alloc_decompressor();
    if (!dec) { out.buf = nullptr; return false; }
    size_t actual = 0;
    enum libdeflate_result res =
        libdeflate_zlib_decompress(dec, zsrc, zlen, out.buf, raw_size, &actual);
    libdeflate_free_decompressor(dec);
    if (res != LIBDEFLATE_SUCCESS || actual != raw_size) {
        out.buf = nullptr;
        return false;
    }
#else
    z_stream zs;
    memset(&zs, 0, sizeof(zs));
    if (inflateInit(&zs) != Z_OK) { out.buf = nullptr; return false; }

    const int bpp = channels;
    const size_t n = (size_t)width * channels;
    const size_t chunk_rows =
        std::max((size_t)1, UNFILTER_CHUNK_BYTES / out.stride);

    std::vector<uint8_t> zero(n, 0);

    size_t idat_idx = 0;
    zs.next_in = (Bytef*)idat[0].data;
    zs.avail_in = idat[0].len;

    size_t total_out = 0;
    size_t rows_unfiltered = 0;
    bool stream_end = false;

    TIMER_DECL(ta); TIMER_DECL(tb); TIMER_DECL(tc);

    while (total_out < raw_size && !stream_end) {
        size_t target = std::min(raw_size, total_out + chunk_rows * out.stride);
        zs.next_out = out.buf + total_out;
        zs.avail_out = (uInt)(target - total_out);

        TIMER_NOW(ta);
        while (zs.avail_out > 0) {
            if (zs.avail_in == 0) {
                if (++idat_idx >= idat.size()) { stream_end = true; break; }
                zs.next_in = (Bytef*)idat[idat_idx].data;
                zs.avail_in = idat[idat_idx].len;
                continue;
            }
            int ret = inflate(&zs, Z_NO_FLUSH);
            if (ret == Z_STREAM_END) { stream_end = true; break; }
            if (ret != Z_OK) {
                inflateEnd(&zs);
                out.buf = nullptr;
                return false;
            }
        }

        TIMER_NOW(tb);
        TIMER_ACC(g_t_inflate, ta, tb);

        size_t produced = (size_t)zs.total_out;
        if (produced == total_out) break;  // no forward progress
        total_out = produced;

        // Unfilter only the scanlines that are now complete; they are still in cache.
        size_t rows_ready = total_out / out.stride;
        while (rows_unfiltered < rows_ready) {
            uint8_t* cur = out.buf + rows_unfiltered * out.stride;
            const uint8_t* prev =
                rows_unfiltered ? (cur - out.stride + 1) : zero.data();
            unfilter_row(cur, prev, n, bpp);
            rows_unfiltered++;
        }

        TIMER_NOW(tc);
        TIMER_ACC(g_t_unfilter, tb, tc);
    }
    inflateEnd(&zs);

    if (total_out != raw_size || rows_unfiltered != (size_t)height) {
        out.buf = nullptr;
        return false;
    }
    return true;
#endif

#ifdef USE_LIBDEFLATE
    unfilter_image(out);
    return true;
#endif
}

// ------------------------------------------------------- PNG encode (parallel)

// The output is written as stored (uncompressed) deflate blocks, which lets the
// whole byte stream be laid out before a single byte is produced. Blocks are cut
// on row boundaries so the filter can write each scanline straight to its final
// address, and the stream is split into one IDAT chunk per thread so every
// checksum is computed in parallel.
struct PngLayout {
    int width = 0, height = 0;
    size_t row_stride = 0;     // 1 filter byte + width*3
    int rows_per_block = 0;
    int nseg = 0;
    std::vector<int> seg_r0, seg_r1;
    std::vector<size_t> seg_chunk_off;    // offset of the chunk length field
    std::vector<size_t> seg_first_block;  // offset of this segment's first block header
    std::vector<size_t> seg_data_len;
    size_t adler_chunk_off = 0;
    size_t total_size = 0;
};

static const uint32_t MAX_STORED = 65535;

static bool plan_png(int width, int height, int nthreads, PngLayout& L) {
    L.width = width;
    L.height = height;
    L.row_stride = (size_t)1 + (size_t)width * 3;
    if (L.row_stride > MAX_STORED) return false;  // a row cannot span blocks here

    L.rows_per_block = (int)(MAX_STORED / L.row_stride);
    int total_blocks = (height + L.rows_per_block - 1) / L.rows_per_block;

    int nseg = nthreads;
    if (nseg > total_blocks) nseg = total_blocks;
    if (nseg < 1) nseg = 1;
    L.nseg = nseg;

    L.seg_r0.resize(nseg);
    L.seg_r1.resize(nseg);
    L.seg_chunk_off.resize(nseg);
    L.seg_first_block.resize(nseg);
    L.seg_data_len.resize(nseg);

    size_t off = 8 + 25;  // signature + IHDR

    for (int s = 0; s < nseg; s++) {
        int b0 = (int)((int64_t)s * total_blocks / nseg);
        int b1 = (int)((int64_t)(s + 1) * total_blocks / nseg);
        int r0 = b0 * L.rows_per_block;
        int r1 = std::min(b1 * L.rows_per_block, height);
        L.seg_r0[s] = r0;
        L.seg_r1[s] = r1;

        int nb = b1 - b0;
        size_t hdr = (s == 0) ? 2 : 0;  // zlib header lives at the very front
        size_t data_len = hdr + (size_t)nb * 5 + (size_t)(r1 - r0) * L.row_stride;

        L.seg_chunk_off[s] = off;
        L.seg_first_block[s] = off + 8 + hdr;
        L.seg_data_len[s] = data_len;
        off += 12 + data_len;
    }

    // adler32 of the uncompressed stream, carried in its own tiny IDAT chunk so
    // no segment has to wait for the others before computing its CRC.
    L.adler_chunk_off = off;
    off += 12 + 4;
    off += 12;  // IEND
    L.total_size = off;
    return true;
}

static inline void put_be32(uint8_t* p, uint32_t v) {
    p[0] = (uint8_t)(v >> 24);
    p[1] = (uint8_t)(v >> 16);
    p[2] = (uint8_t)(v >> 8);
    p[3] = (uint8_t)v;
}

static void write_chunk_header(uint8_t* p, size_t data_len, const char* type) {
    put_be32(p, (uint32_t)data_len);
    memcpy(p + 4, type, 4);
}

static void write_png_prologue(uint8_t* buf, const PngLayout& L) {
    static const uint8_t sig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    memcpy(buf, sig, 8);

    uint8_t* ihdr = buf + 8;
    write_chunk_header(ihdr, 13, "IHDR");
    uint8_t* d = ihdr + 8;
    put_be32(d, (uint32_t)L.width);
    put_be32(d + 4, (uint32_t)L.height);
    d[8] = 8;   // bit depth
    d[9] = 2;   // color type RGB
    d[10] = 0;  // deflate
    d[11] = 0;  // adaptive filtering
    d[12] = 0;  // no interlace
    put_be32(ihdr + 8 + 13, (uint32_t)crc32(0, ihdr + 4, 4 + 13));

    // zlib header: CM=8, CINFO=7, no preset dict, FCHECK chosen so 0x7801 % 31 == 0
    uint8_t* z = buf + L.seg_first_block[0] - 2;
    z[0] = 0x78;
    z[1] = 0x01;
}

static void write_png_epilogue(uint8_t* buf, const PngLayout& L, uint32_t adler) {
    uint8_t* a = buf + L.adler_chunk_off;
    write_chunk_header(a, 4, "IDAT");
    put_be32(a + 8, adler);
    put_be32(a + 12, (uint32_t)crc32(0, a + 4, 4 + 4));

    uint8_t* e = buf + L.adler_chunk_off + 12 + 4;
    write_chunk_header(e, 0, "IEND");
    put_be32(e + 8, (uint32_t)crc32(0, e + 4, 4));
}

// ---------------------------------------------------------------- box filter

// Each thread owns one layout segment, so its rows map to a contiguous run of
// stored blocks and every output address is reached by increment, never by a
// division. col11/col5 hold the vertical partial sums of the 11x11 and 5x5
// windows; they slide down by one row and across by one column, which keeps the
// working set at a few tens of KB regardless of image size.
static void filter_segment(
    const uint8_t* in_data,   // points at row 0's first sample (past the filter byte)
    size_t in_stride,
    uint8_t* png_buf,
    const PngLayout& L,
    int seg,
    int height,
    int width,
    int channels,
    uint32_t& seg_adler,
    uint32_t& seg_crc
) {
    const int r_start = L.seg_r0[seg];
    const int r_end = L.seg_r1[seg];

    auto get_pixel = [&](int r, int c) -> const uint8_t* {
        r = std::min(std::max(r, 0), height - 1);
        c = std::min(std::max(c, 0), width - 1);
        return in_data + (size_t)r * in_stride + (size_t)c * channels;
    };

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

    // Stored-block bookkeeping for this segment.
    const size_t block_span = 5 + (size_t)L.rows_per_block * L.row_stride;
    uint8_t* seg_base = png_buf + L.seg_first_block[seg];
    const bool is_last_seg = (seg == L.nseg - 1);

    uint32_t adler = 1;

    int rr = 0;  // row index inside the segment
    while (rr < r_end - r_start) {
        int k = rr / L.rows_per_block;
        int rows_here = std::min(L.rows_per_block, (r_end - r_start) - k * L.rows_per_block);
        uint8_t* blk = seg_base + (size_t)k * block_span;

        size_t payload = (size_t)rows_here * L.row_stride;
        bool final_block = is_last_seg && (rr + rows_here >= r_end - r_start);
        blk[0] = final_block ? 0x01 : 0x00;  // BFINAL | BTYPE=stored
        blk[1] = (uint8_t)(payload & 0xFF);
        blk[2] = (uint8_t)((payload >> 8) & 0xFF);
        blk[3] = (uint8_t)(~payload & 0xFF);
        blk[4] = (uint8_t)((~payload >> 8) & 0xFF);

        uint8_t* out_row = blk + 5;

        for (int i = 0; i < rows_here; i++, rr++) {
            int r = r_start + rr;

            if (r > r_start) {
                int r_a11 = std::min(std::max(r + 5, 0), height - 1);
                int r_s11 = std::min(std::max(r - 6, 0), height - 1);
                int r_a5  = std::min(std::max(r + 2, 0), height - 1);
                int r_s5  = std::min(std::max(r - 3, 0), height - 1);

                const uint8_t* row_a11 = in_data + (size_t)r_a11 * in_stride;
                const uint8_t* row_s11 = in_data + (size_t)r_s11 * in_stride;
                const uint8_t* row_a5  = in_data + (size_t)r_a5  * in_stride;
                const uint8_t* row_s5  = in_data + (size_t)r_s5  * in_stride;

                for (int c = -5; c < 0; c++) {
                    int idx = c + 5;
                    col11[idx].r += row_a11[0] - row_s11[0];
                    col11[idx].g += row_a11[1] - row_s11[1];
                    col11[idx].b += row_a11[2] - row_s11[2];
                    col5[idx].r += row_a5[0] - row_s5[0];
                    col5[idx].g += row_a5[1] - row_s5[1];
                    col5[idx].b += row_a5[2] - row_s5[2];
                }

                for (int c = 0; c < width; c++) {
                    int idx = c + 5;
                    const uint8_t* p_a11 = row_a11 + (size_t)c * channels;
                    const uint8_t* p_s11 = row_s11 + (size_t)c * channels;
                    const uint8_t* p_a5  = row_a5  + (size_t)c * channels;
                    const uint8_t* p_s5  = row_s5  + (size_t)c * channels;

                    col11[idx].r += p_a11[0] - p_s11[0];
                    col11[idx].g += p_a11[1] - p_s11[1];
                    col11[idx].b += p_a11[2] - p_s11[2];
                    col5[idx].r += p_a5[0] - p_s5[0];
                    col5[idx].g += p_a5[1] - p_s5[1];
                    col5[idx].b += p_a5[2] - p_s5[2];
                }

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

            const uint8_t* in_row = in_data + (size_t)r * in_stride;

            out_row[0] = 0;  // PNG filter type None
            uint8_t* px = out_row + 1;

            {
                const uint8_t* cur = in_row;
                bool is_bright = (299u * cur[0] + 587u * cur[1] + 114u * cur[2]) > 128000u;
                uint32_t mul = is_bright ? 4333u : 20972u;
                const RGB& b = is_bright ? box11 : box5;
                px[0] = (b.r * mul) >> 19;
                px[1] = (b.g * mul) >> 19;
                px[2] = (b.b * mul) >> 19;
            }

            for (int y = 1; y < width; y++) {
                box11.r += col11[y + 10].r - col11[y - 1].r;
                box11.g += col11[y + 10].g - col11[y - 1].g;
                box11.b += col11[y + 10].b - col11[y - 1].b;

                box5.r += col5[y + 7].r - col5[y + 2].r;
                box5.g += col5[y + 7].g - col5[y + 2].g;
                box5.b += col5[y + 7].b - col5[y + 2].b;

                const uint8_t* cur = in_row + (size_t)y * channels;
                bool is_bright = (299u * cur[0] + 587u * cur[1] + 114u * cur[2]) > 128000u;
                uint32_t mul = is_bright ? 4333u : 20972u;
                const RGB& b = is_bright ? box11 : box5;

                px[y * 3 + 0] = (b.r * mul) >> 19;
                px[y * 3 + 1] = (b.g * mul) >> 19;
                px[y * 3 + 2] = (b.b * mul) >> 19;
            }

            out_row += L.row_stride;
        }

        // The scanlines just written are still hot in L1/L2, so fold them into
        // the running adler32 now rather than in a second pass over the image.
        adler = (uint32_t)adler32(adler, blk + 5, (uInt)payload);
    }

    seg_adler = adler;

    uint8_t* chunk = png_buf + L.seg_chunk_off[seg];
    write_chunk_header(chunk, L.seg_data_len[seg], "IDAT");
    seg_crc = (uint32_t)crc32(0, chunk + 4, (uInt)(4 + L.seg_data_len[seg]));
    put_be32(chunk + 8 + L.seg_data_len[seg], seg_crc);
}

// ------------------------------------------------------------ libpng fallback

static uint8_t* libpng_read(const char* file_name, int& width, int& height,
                            int& channels, size_t& stride) {
    FILE* fp = fopen(file_name, "rb");
    if (!fp) {
        std::cerr << "Error: Cannot open file " << file_name << std::endl;
        exit(EXIT_FAILURE);
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) exit(EXIT_FAILURE);
    png_infop info = png_create_info_struct(png);
    if (!info) exit(EXIT_FAILURE);

    if (setjmp(png_jmpbuf(png))) {
        std::cerr << "Error during PNG read" << std::endl;
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

    if (bit_depth == 16) png_set_strip_16(png);
    if (color_type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);
    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);

    png_read_update_info(png, info);

    channels = png_get_channels(png, info);
    stride = png_get_rowbytes(png, info);
    uint8_t* data = (uint8_t*)alloc_big(stride * (size_t)height);

    // png_read_row reads one pass of an Adam7 image, not one reconstructed
    // scanline, so an interlaced file decoded that way comes out scrambled.
    // png_read_image handles both layouts; this path only runs for inputs the
    // fast decoder rejected, so the row-pointer array costs nothing that matters.
    std::vector<png_bytep> rows((size_t)height);
    for (int y = 0; y < height; y++) rows[y] = data + (size_t)y * stride;
    png_read_image(png, rows.data());

    png_destroy_read_struct(&png, &info, nullptr);
    fclose(fp);
    return data;
}

static void libpng_write(const char* file_name, const uint8_t* image_data,
                         int width, int height) {
    FILE* fp = fopen(file_name, "wb");
    if (!fp) {
        std::cerr << "Error: Cannot open file " << file_name << std::endl;
        exit(EXIT_FAILURE);
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) { fclose(fp); exit(EXIT_FAILURE); }
    png_infop info = png_create_info_struct(png);
    if (!info) { png_destroy_write_struct(&png, nullptr); fclose(fp); exit(EXIT_FAILURE); }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    png_init_io(png, fp);
    png_set_IHDR(png, info, width, height, 8, PNG_COLOR_TYPE_RGB,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_compression_level(png, 0);
    png_set_filter(png, 0, PNG_FILTER_NONE);
    png_write_info(png, info);

    size_t row_bytes = (size_t)width * 3;
    png_bytep* rows = (png_bytep*)malloc(sizeof(png_bytep) * height);
    for (int y = 0; y < height; y++)
        rows[y] = (png_bytep)(image_data + (size_t)y * row_bytes);

    png_write_image(png, rows);
    png_write_end(png, nullptr);

    free(rows);
    png_destroy_write_struct(&png, &info);
    fclose(fp);
}

// Slow path used only when a row does not fit in one stored block.
static void filter_to_plain(const uint8_t* in_data, size_t in_stride, uint8_t* out,
                            int height, int width, int channels, int nthreads) {
    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        int nt = omp_get_num_threads();
        int r_start = (int)((int64_t)tid * height / nt);
        int r_end = (int)((int64_t)(tid + 1) * height / nt);

        for (int r = r_start; r < r_end; r++) {
            uint8_t* out_row = out + (size_t)r * width * 3;
            for (int y = 0; y < width; y++) {
                const uint8_t* cur = in_data + (size_t)r * in_stride + (size_t)y * channels;
                bool is_bright = (299u * cur[0] + 587u * cur[1] + 114u * cur[2]) > 128000u;
                int rad = is_bright ? 5 : 2;
                uint32_t sr = 0, sg = 0, sb = 0;
                for (int i = -rad; i <= rad; i++) {
                    int rr = std::min(std::max(r + i, 0), height - 1);
                    const uint8_t* row = in_data + (size_t)rr * in_stride;
                    for (int j = -rad; j <= rad; j++) {
                        int cc = std::min(std::max(y + j, 0), width - 1);
                        const uint8_t* p = row + (size_t)cc * channels;
                        sr += p[0]; sg += p[1]; sb += p[2];
                    }
                }
                uint32_t k2 = (uint32_t)(2 * rad + 1) * (2 * rad + 1);
                out_row[y * 3 + 0] = (uint8_t)(sr / k2);
                out_row[y * 3 + 1] = (uint8_t)(sg / k2);
                out_row[y * 3 + 2] = (uint8_t)(sb / k2);
            }
        }
    }
}

// ---------------------------------------------------------------- driver

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <inputfile.png> <outputfile.png>" << std::endl;
        return -1;
    }

    int ncpus = 1;
#ifdef __linux__
    cpu_set_t cpuset;
    sched_getaffinity(0, sizeof(cpuset), &cpuset);
    ncpus = CPU_COUNT(&cpuset);
#else
    ncpus = (int)sysconf(_SC_NPROCESSORS_ONLN);
#endif
    if (ncpus < 1) ncpus = 1;
    omp_set_num_threads(ncpus);

    const char* input_file = argv[1];
    const char* output_file = argv[2];

    TIMER_DECL(T0); TIMER_DECL(T1); TIMER_DECL(T2); TIMER_DECL(T3);
    TIMER_NOW(T0);

    FastPng img;
    bool used_fast_decode = false;

    int fd = open(input_file, O_RDONLY);
    if (fd < 0) {
        std::cerr << "Error: Cannot open file " << input_file << std::endl;
        return EXIT_FAILURE;
    }
    struct stat st;
    fstat(fd, &st);
    size_t file_size = (size_t)st.st_size;
    const uint8_t* mapped = (const uint8_t*)mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);

    TIMER_DECL(TM);
    if (mapped != MAP_FAILED) {
        madvise((void*)mapped, file_size, MADV_WILLNEED);
        madvise((void*)mapped, file_size, MADV_SEQUENTIAL);

        // inflate walks the file strictly in order on one thread, so every page
        // fault it takes is serial latency. Touching one byte per page from all
        // cores first lets the faults (and any network round trips, if the file
        // lives on a shared filesystem) overlap instead of queueing behind it.
        // Only pays off when the pages are actually cold: on a warm page cache
        // it is a wasted pass, so it can be compiled out with -DNO_PREFAULT.
#ifndef NO_PREFAULT
        {
            const size_t page = 4096;
            const int64_t npages = (int64_t)((file_size + page - 1) / page);
            uint8_t sink = 0;
            #pragma omp parallel for schedule(static) reduction(| : sink)
            for (int64_t i = 0; i < npages; i++) {
                size_t off = (size_t)i * page;
                if (off < file_size) sink |= mapped[off];
            }
            g_prefault_sink = sink;  // keeps the loop from being optimised away
        }
#endif
        TIMER_NOW(TM);

        used_fast_decode = fast_png_decode(mapped, file_size, img);
        munmap((void*)mapped, file_size);
    } else {
        TIMER_NOW(TM);
    }

    if (!used_fast_decode) {
        // libpng hands back packed rows, so there is no leading filter byte.
        img.buf = libpng_read(input_file, img.width, img.height, img.channels, img.stride);
    }

    TIMER_NOW(T1);

    const int width = img.width, height = img.height, channels = img.channels;
    // Fast decode keeps the filter byte at the head of every row; libpng does not.
    const uint8_t* in_data = used_fast_decode ? img.buf + 1 : img.buf;
    const size_t in_stride = img.stride;

    PngLayout L;
    if (!plan_png(width, height, ncpus, L)) {
        // Rows too wide for a single stored block: fall back to libpng output.
        uint8_t* plain = (uint8_t*)malloc((size_t)width * height * 3);
        filter_to_plain(in_data, in_stride, plain, height, width, channels, ncpus);
        TIMER_NOW(T2);
        libpng_write(output_file, plain, width, height);
        TIMER_NOW(T3);
        return 0;
    }

    uint8_t* png_buf = (uint8_t*)alloc_big(L.total_size);
    if (!png_buf) {
        std::cerr << "Error: out of memory for output buffer" << std::endl;
        return EXIT_FAILURE;
    }
    write_png_prologue(png_buf, L);

    std::vector<uint32_t> seg_adler(L.nseg, 1), seg_crc(L.nseg, 0);

    #pragma omp parallel for schedule(static, 1) num_threads(L.nseg)
    for (int s = 0; s < L.nseg; s++) {
        filter_segment(in_data, in_stride, png_buf, L, s, height, width, channels,
                       seg_adler[s], seg_crc[s]);
    }

    // Segment checksums were computed independently; stitch them into the single
    // adler32 the zlib stream expects.
    uint32_t adler = 1;
    for (int s = 0; s < L.nseg; s++) {
        size_t len = (size_t)(L.seg_r1[s] - L.seg_r0[s]) * L.row_stride;
        adler = (uint32_t)adler32_combine(adler, seg_adler[s], (z_off_t)len);
    }
    write_png_epilogue(png_buf, L, adler);

    TIMER_NOW(T2);

    int ofd = open(output_file, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (ofd < 0) {
        std::cerr << "Error: Cannot open file " << output_file << std::endl;
        return EXIT_FAILURE;
    }
    size_t written = 0;
    while (written < L.total_size) {
        ssize_t n = write(ofd, png_buf + written, L.total_size - written);
        if (n <= 0) break;
        written += (size_t)n;
    }
    close(ofd);

    TIMER_NOW(T3);

#ifdef HW1_TIMING
    fprintf(stderr,
            "fault %6.1f | inflate %7.1f | unfilter %6.1f | READ %7.1f | "
            "filter+enc %6.1f | write %6.1f | TOTAL %7.1f ms | "
            "%dx%d ch=%d %s seg=%d in=%.1fMB out=%.1fMB\n",
            std::chrono::duration<double, std::milli>(TM - T0).count(),
            g_t_inflate, g_t_unfilter,
            std::chrono::duration<double, std::milli>(T1 - T0).count(),
            std::chrono::duration<double, std::milli>(T2 - T1).count(),
            std::chrono::duration<double, std::milli>(T3 - T2).count(),
            std::chrono::duration<double, std::milli>(T3 - T0).count(),
            width, height, channels,
            used_fast_decode ? "fastdec" : "libpng",
            L.nseg,
            file_size / 1048576.0, L.total_size / 1048576.0);
#endif

    return 0;
}
