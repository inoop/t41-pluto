/* Baseline sequential JPEG, 4:2:0, standard tables.  See plt_jpeg.h for why. */
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "plt_jpeg.h"

/* ---- the standard tables (ITU T.81 Annex K) -------------------------------
 *
 * Annex K's tables are only "examples", but every encoder uses them and every
 * decoder is tuned for them, so a debug dump has no reason to invent its own.
 */
static const uint8_t QT_LUMA[64] = {
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68,109,103, 77,
    24, 35, 55, 64, 81,104,113, 92,
    49, 64, 78, 87,103,121,120,101,
    72, 92, 95, 98,112,100,103, 99,
};
static const uint8_t QT_CHROMA[64] = {
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
};

/* ZIGZAG[i] is the natural-order index of the i-th coefficient in zigzag
 * order, which is both the order coefficients are coded in and the order a DQT
 * segment lists a quantization table. */
static const uint8_t ZIGZAG[64] = {
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
};

static const uint8_t DC_LUMA_BITS[16]   = {0,1,5,1,1,1,1,1,1,0,0,0,0,0,0,0};
static const uint8_t DC_CHROMA_BITS[16] = {0,3,1,1,1,1,1,1,1,1,1,0,0,0,0,0};
static const uint8_t DC_VALS[12] = {0,1,2,3,4,5,6,7,8,9,10,11};

static const uint8_t AC_LUMA_BITS[16] =
    {0,2,1,3,3,2,4,3,5,5,4,4,0,0,1,0x7d};
static const uint8_t AC_LUMA_VALS[162] = {
    0x01,0x02,0x03,0x00,0x04,0x11,0x05,0x12,0x21,0x31,0x41,0x06,0x13,0x51,0x61,0x07,
    0x22,0x71,0x14,0x32,0x81,0x91,0xa1,0x08,0x23,0x42,0xb1,0xc1,0x15,0x52,0xd1,0xf0,
    0x24,0x33,0x62,0x72,0x82,0x09,0x0a,0x16,0x17,0x18,0x19,0x1a,0x25,0x26,0x27,0x28,
    0x29,0x2a,0x34,0x35,0x36,0x37,0x38,0x39,0x3a,0x43,0x44,0x45,0x46,0x47,0x48,0x49,
    0x4a,0x53,0x54,0x55,0x56,0x57,0x58,0x59,0x5a,0x63,0x64,0x65,0x66,0x67,0x68,0x69,
    0x6a,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7a,0x83,0x84,0x85,0x86,0x87,0x88,0x89,
    0x8a,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9a,0xa2,0xa3,0xa4,0xa5,0xa6,0xa7,
    0xa8,0xa9,0xaa,0xb2,0xb3,0xb4,0xb5,0xb6,0xb7,0xb8,0xb9,0xba,0xc2,0xc3,0xc4,0xc5,
    0xc6,0xc7,0xc8,0xc9,0xca,0xd2,0xd3,0xd4,0xd5,0xd6,0xd7,0xd8,0xd9,0xda,0xe1,0xe2,
    0xe3,0xe4,0xe5,0xe6,0xe7,0xe8,0xe9,0xea,0xf1,0xf2,0xf3,0xf4,0xf5,0xf6,0xf7,0xf8,
    0xf9,0xfa,
};
static const uint8_t AC_CHROMA_BITS[16] =
    {0,2,1,2,4,4,3,4,7,5,4,4,0,1,2,0x77};
static const uint8_t AC_CHROMA_VALS[162] = {
    0x00,0x01,0x02,0x03,0x11,0x04,0x05,0x21,0x31,0x06,0x12,0x41,0x51,0x07,0x61,0x71,
    0x13,0x22,0x32,0x81,0x08,0x14,0x42,0x91,0xa1,0xb1,0xc1,0x09,0x23,0x33,0x52,0xf0,
    0x15,0x62,0x72,0xd1,0x0a,0x16,0x24,0x34,0xe1,0x25,0xf1,0x17,0x18,0x19,0x1a,0x26,
    0x27,0x28,0x29,0x2a,0x35,0x36,0x37,0x38,0x39,0x3a,0x43,0x44,0x45,0x46,0x47,0x48,
    0x49,0x4a,0x53,0x54,0x55,0x56,0x57,0x58,0x59,0x5a,0x63,0x64,0x65,0x66,0x67,0x68,
    0x69,0x6a,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7a,0x82,0x83,0x84,0x85,0x86,0x87,
    0x88,0x89,0x8a,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9a,0xa2,0xa3,0xa4,0xa5,
    0xa6,0xa7,0xa8,0xa9,0xaa,0xb2,0xb3,0xb4,0xb5,0xb6,0xb7,0xb8,0xb9,0xba,0xc2,0xc3,
    0xc4,0xc5,0xc6,0xc7,0xc8,0xc9,0xca,0xd2,0xd3,0xd4,0xd5,0xd6,0xd7,0xd8,0xd9,0xda,
    0xe2,0xe3,0xe4,0xe5,0xe6,0xe7,0xe8,0xe9,0xea,0xf2,0xf3,0xf4,0xf5,0xf6,0xf7,0xf8,
    0xf9,0xfa,
};

/* ---- bit output ----------------------------------------------------------- */

typedef struct {
    FILE    *f;
    uint32_t acc;      /* pending bits, right-aligned */
    int      n;        /* how many of them there are  */
} bitw;

/* A literal 0xFF in the entropy-coded stream has to be followed by a 0x00, or a
 * decoder reads it as the start of the next marker. */
static void put_bits(bitw *b, unsigned code, int len)
{
    b->acc = (b->acc << len) | (len ? (code & ((1u << len) - 1u)) : 0u);
    b->n  += len;
    while (b->n >= 8) {
        const int byte = (int)((b->acc >> (b->n - 8)) & 0xFFu);
        fputc(byte, b->f);
        if (byte == 0xFF) fputc(0x00, b->f);
        b->n -= 8;
    }
}

/* The stream is padded to a byte boundary with 1 bits, which no Huffman code
 * can be a prefix of. */
static void flush_bits(bitw *b)
{
    while (b->n) put_bits(b, 1, 1);
}

typedef struct { uint16_t code[256]; uint8_t size[256]; } huff;

/* Canonical Huffman: codes ascend within a length, and lengthening shifts left. */
static void build_huff(const uint8_t bits[16], const uint8_t *vals, huff *h)
{
    uint16_t c = 0;
    int k = 0;
    memset(h, 0, sizeof *h);
    for (int len = 1; len <= 16; len++) {
        for (int i = 0; i < bits[len - 1]; i++, k++) {
            h->code[vals[k]] = c++;
            h->size[vals[k]] = (uint8_t)len;
        }
        c = (uint16_t)(c << 1);
    }
}

static void put_huff(bitw *b, const huff *h, int sym)
{
    put_bits(b, h->code[sym], h->size[sym]);
}

/* ---- the transform -------------------------------------------------------- */

/* COS[u][x] = C(u)/2 * cos((2x+1)u*pi/16), so a plain double sum over x and y
 * is the whole forward DCT including its 1/4 C(u) C(v) scaling. */
static double COS[8][8];

static void init_cos(void)
{
    static int done;
    if (done) return;
    for (int u = 0; u < 8; u++)
        for (int x = 0; x < 8; x++)
            COS[u][x] = (u == 0 ? 0.353553390593273762 : 0.5)
                      * cos((2.0 * x + 1.0) * u * 3.14159265358979324 / 16.0);
    done = 1;
}

/* in[] is row-major, already level-shifted; out[] is indexed v*8+u. */
static void fdct(const int in[64], double out[64])
{
    double tmp[64];
    for (int y = 0; y < 8; y++)
        for (int u = 0; u < 8; u++) {
            double s = 0.0;
            for (int x = 0; x < 8; x++) s += in[y * 8 + x] * COS[u][x];
            tmp[y * 8 + u] = s;
        }
    for (int u = 0; u < 8; u++)
        for (int v = 0; v < 8; v++) {
            double s = 0.0;
            for (int y = 0; y < 8; y++) s += tmp[y * 8 + u] * COS[v][y];
            out[v * 8 + u] = s;
        }
}

/* The number of bits needed to hold |v|; JPEG codes that, then the value. */
static int category(int v)
{
    int a = v < 0 ? -v : v, n = 0;
    while (a) { a >>= 1; n++; }
    return n;
}

/* One block: DC as a difference from the previous block of the same component,
 * AC as (run of zeros, magnitude) pairs, with ZRL for runs past 15 and EOB for
 * an all-zero tail.  Returns this block's DC for the next call. */
static int encode_block(bitw *b, const int in[64], const uint8_t qt[64],
                        int prev_dc, const huff *dc, const huff *ac)
{
    double f[64];
    int zz[64];

    fdct(in, f);
    for (int i = 0; i < 64; i++) {
        const int nat = ZIGZAG[i];
        const double q = f[nat] / (double)qt[nat];
        zz[i] = (int)(q < 0.0 ? q - 0.5 : q + 0.5);
    }

    {
        const int diff = zz[0] - prev_dc;
        const int s = category(diff);
        put_huff(b, dc, s);
        if (s) put_bits(b, (unsigned)(diff < 0 ? diff + (1 << s) - 1 : diff), s);
    }

    {
        int run = 0;
        int last = 0;
        for (int i = 63; i > 0; i--) if (zz[i]) { last = i; break; }
        for (int i = 1; i <= last; i++) {
            if (zz[i] == 0) { run++; continue; }
            while (run > 15) { put_huff(b, ac, 0xF0); run -= 16; }   /* ZRL */
            {
                const int s = category(zz[i]);
                put_huff(b, ac, (run << 4) | s);
                put_bits(b, (unsigned)(zz[i] < 0 ? zz[i] + (1 << s) - 1 : zz[i]), s);
            }
            run = 0;
        }
        if (last < 63) put_huff(b, ac, 0x00);                        /* EOB */
    }
    return zz[0];
}

/* ---- sampling ------------------------------------------------------------- */

/* Blocks past the right or bottom edge repeat the last real pixel.  The decoder
 * crops to the true size, so what is in them only affects the edge block's
 * coefficients, not the picture. */
static int luma_at(const plt_pre_view_t *v, int x, int y)
{
    if (x >= v->w) x = v->w - 1;
    if (y >= v->h) y = v->h - 1;
    return v->nv12[(size_t)(v->y0 + y) * v->stride + v->x0 + x];
}

/* NV12's second plane is U,V interleaved at half resolution -- Cb then Cr, the
 * order JPEG wants.  plt_cam guarantees x0 and y0 are even, so the region's
 * chroma origin is just x0/2, y0/2 and no phase correction is needed. */
static void chroma_at(const plt_pre_view_t *v, int cx, int cy, int *cb, int *cr)
{
    const uint8_t *uv = v->nv12 + (size_t)v->stride * v->cap_h;
    const int mx = (v->w + 1) / 2 - 1, my = (v->h + 1) / 2 - 1;
    const uint8_t *row;

    if (cx > mx) cx = mx;
    if (cy > my) cy = my;
    row = uv + (size_t)(v->y0 / 2 + cy) * v->stride + v->x0 + 2 * cx;
    *cb = row[0];
    *cr = row[1];
}

/* ---- headers -------------------------------------------------------------- */

static void put_u16(FILE *f, unsigned v) { fputc((int)(v >> 8), f); fputc((int)(v & 0xFF), f); }
static void marker (FILE *f, int m)      { fputc(0xFF, f); fputc(m, f); }

static void scale_qt(const uint8_t base[64], int quality, uint8_t out[64])
{
    int s;
    if (quality < 1)   quality = 1;
    if (quality > 100) quality = 100;
    s = quality < 50 ? 5000 / quality : 200 - 2 * quality;
    for (int i = 0; i < 64; i++) {
        int q = ((int)base[i] * s + 50) / 100;
        out[i] = (uint8_t)(q < 1 ? 1 : q > 255 ? 255 : q);
    }
}

static void put_dht(FILE *f, int class_id, const uint8_t bits[16],
                    const uint8_t *vals, int nvals)
{
    marker(f, 0xC4);
    put_u16(f, 2 + 1 + 16 + (unsigned)nvals);
    fputc(class_id, f);
    fwrite(bits, 1, 16, f);
    fwrite(vals, 1, (size_t)nvals, f);
}

int plt_jpeg_write(const char *path, const plt_pre_view_t *v, int quality)
{
    uint8_t ql[64], qc[64];
    huff dcl, dcc, acl, acc;
    bitw b;
    FILE *f;
    int dc_y = 0, dc_cb = 0, dc_cr = 0;

    init_cos();
    scale_qt(QT_LUMA,   quality, ql);
    scale_qt(QT_CHROMA, quality, qc);
    build_huff(DC_LUMA_BITS,   DC_VALS,        &dcl);
    build_huff(DC_CHROMA_BITS, DC_VALS,        &dcc);
    build_huff(AC_LUMA_BITS,   AC_LUMA_VALS,   &acl);
    build_huff(AC_CHROMA_BITS, AC_CHROMA_VALS, &acc);

    if (!(f = fopen(path, "wb"))) return -1;

    marker(f, 0xD8);                                     /* SOI */

    marker(f, 0xE0);                                     /* APP0/JFIF */
    put_u16(f, 16);
    fwrite("JFIF\0", 1, 5, f);
    fputc(1, f); fputc(1, f);                            /* version 1.1 */
    fputc(0, f);                                         /* no density unit */
    put_u16(f, 1); put_u16(f, 1);
    fputc(0, f); fputc(0, f);                            /* no thumbnail */

    marker(f, 0xDB);                                     /* DQT, both tables */
    put_u16(f, 2 + 2 * 65);
    fputc(0x00, f);
    for (int i = 0; i < 64; i++) fputc(ql[ZIGZAG[i]], f);
    fputc(0x01, f);
    for (int i = 0; i < 64; i++) fputc(qc[ZIGZAG[i]], f);

    marker(f, 0xC0);                                     /* SOF0, baseline */
    put_u16(f, 8 + 3 * 3);
    fputc(8, f);
    put_u16(f, (unsigned)v->h);
    put_u16(f, (unsigned)v->w);
    fputc(3, f);
    fputc(1, f); fputc(0x22, f); fputc(0, f);            /* Y  2x2, table 0 */
    fputc(2, f); fputc(0x11, f); fputc(1, f);            /* Cb 1x1, table 1 */
    fputc(3, f); fputc(0x11, f); fputc(1, f);            /* Cr 1x1, table 1 */

    put_dht(f, 0x00, DC_LUMA_BITS,   DC_VALS,        12);
    put_dht(f, 0x10, AC_LUMA_BITS,   AC_LUMA_VALS,  162);
    put_dht(f, 0x01, DC_CHROMA_BITS, DC_VALS,        12);
    put_dht(f, 0x11, AC_CHROMA_BITS, AC_CHROMA_VALS,162);

    marker(f, 0xDA);                                     /* SOS */
    put_u16(f, 6 + 2 * 3);
    fputc(3, f);
    fputc(1, f); fputc(0x00, f);
    fputc(2, f); fputc(0x11, f);
    fputc(3, f); fputc(0x11, f);
    fputc(0, f); fputc(63, f); fputc(0, f);

    memset(&b, 0, sizeof b);
    b.f = f;

    /* One MCU is 16x16 of luma: four Y blocks, then one Cb and one Cr covering
     * the same ground at half resolution. */
    for (int my = 0; my < v->h; my += 16) {
        for (int mx = 0; mx < v->w; mx += 16) {
            int blk[64];

            for (int k = 0; k < 4; k++) {
                const int ox = mx + (k & 1) * 8, oy = my + (k >> 1) * 8;
                for (int y = 0; y < 8; y++)
                    for (int x = 0; x < 8; x++)
                        blk[y * 8 + x] = luma_at(v, ox + x, oy + y) - 128;
                dc_y = encode_block(&b, blk, ql, dc_y, &dcl, &acl);
            }

            {
                int cbb[64], crb[64];
                for (int y = 0; y < 8; y++)
                    for (int x = 0; x < 8; x++) {
                        int cb, cr;
                        chroma_at(v, mx / 2 + x, my / 2 + y, &cb, &cr);
                        cbb[y * 8 + x] = cb - 128;
                        crb[y * 8 + x] = cr - 128;
                    }
                dc_cb = encode_block(&b, cbb, qc, dc_cb, &dcc, &acc);
                dc_cr = encode_block(&b, crb, qc, dc_cr, &dcc, &acc);
            }
        }
    }

    flush_bits(&b);
    marker(f, 0xD9);                                     /* EOI */
    return fclose(f) ? -1 : 0;
}
