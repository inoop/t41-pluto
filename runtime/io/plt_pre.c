#include <math.h>
#include <stddef.h>

#include <string.h>

#include "plt_pre.h"

void plt_pre_build_lut(int8_t lut[256], float scale, int zero_point)
{
    for (int p = 0; p < 256; p++) {
        const double x = (double)p / 255.0 * 2.0 - 1.0;
        double q = rint(x / (double)scale) + (double)zero_point;
        if (q < -128.0) q = -128.0;
        if (q >  127.0) q =  127.0;
        lut[p] = (int8_t)q;
    }
}

void plt_pre_build_lut_linear(int8_t lut[256], float scale, int zero_point)
{
    for (int p = 0; p < 256; p++) {
        double q = rint((double)p / (double)scale) + (double)zero_point;
        if (q < -128.0) q = -128.0;
        if (q >  127.0) q =  127.0;
        lut[p] = (int8_t)q;
    }
}

/* BT.601 in fixed point.  The camera gives limited-range NV12; these are the
 * full-range coefficients the reference implementation uses, kept identical so
 * the two agree pixel for pixel. */
static void yuv_to_rgb(int y, int u, int v, int *r, int *g, int *b)
{
    const int cu = u - 128, cv = v - 128;
    int rr = y + ((1436 * cv) >> 10);
    int gg = y - ((352 * cu) >> 10) - ((731 * cv) >> 10);
    int bb = y + ((1815 * cu) >> 10);
    *r = rr < 0 ? 0 : rr > 255 ? 255 : rr;
    *g = gg < 0 ? 0 : gg > 255 ? 255 : gg;
    *b = bb < 0 ? 0 : bb > 255 ? 255 : bb;
}

/* The two loops below are the same walk over the region; only the store
 * differs.  `ur` indexes the UV row in captured-frame coordinates, so the
 * x0/y0 offset has to go through the chroma subsampling, not around it. */
void plt_pre_view_int8(const plt_pre_view_t *v, const int8_t lut[256], int8_t *chw)
{
    const uint8_t *uv = v->nv12 + (size_t)v->stride * v->cap_h;
    const int plane = v->w * v->h;

    for (int y = 0; y < v->h; y++) {
        const uint8_t *yr = v->nv12 + (size_t)(v->y0 + y) * v->stride + v->x0;
        const uint8_t *ur = uv + (size_t)((v->y0 + y) / 2) * v->stride + v->x0;
        for (int x = 0; x < v->w; x++) {
            int r, g, b;
            yuv_to_rgb(yr[x], ur[x & ~1], ur[(x & ~1) + 1], &r, &g, &b);
            chw[0 * plane + y * v->w + x] = lut[r];
            chw[1 * plane + y * v->w + x] = lut[g];
            chw[2 * plane + y * v->w + x] = lut[b];
        }
    }
}

void plt_pre_view_int8_bgr(const plt_pre_view_t *v, const int8_t lut[256], int8_t *chw)
{
    const uint8_t *uv = v->nv12 + (size_t)v->stride * v->cap_h;
    const int plane = v->w * v->h;

    for (int y = 0; y < v->h; y++) {
        const uint8_t *yr = v->nv12 + (size_t)(v->y0 + y) * v->stride + v->x0;
        const uint8_t *ur = uv + (size_t)((v->y0 + y) / 2) * v->stride + v->x0;
        for (int x = 0; x < v->w; x++) {
            int r, g, b;
            yuv_to_rgb(yr[x], ur[x & ~1], ur[(x & ~1) + 1], &r, &g, &b);
            chw[0 * plane + y * v->w + x] = lut[b];
            chw[1 * plane + y * v->w + x] = lut[g];
            chw[2 * plane + y * v->w + x] = lut[r];
        }
    }
}

void plt_pre_view_rgb(const plt_pre_view_t *v, uint8_t *rgb)
{
    const uint8_t *uv = v->nv12 + (size_t)v->stride * v->cap_h;

    for (int y = 0; y < v->h; y++) {
        const uint8_t *yr = v->nv12 + (size_t)(v->y0 + y) * v->stride + v->x0;
        const uint8_t *ur = uv + (size_t)((v->y0 + y) / 2) * v->stride + v->x0;
        for (int x = 0; x < v->w; x++) {
            int r, g, b;
            yuv_to_rgb(yr[x], ur[x & ~1], ur[(x & ~1) + 1], &r, &g, &b);
            rgb[(y * v->w + x) * 3 + 0] = (uint8_t)r;
            rgb[(y * v->w + x) * 3 + 1] = (uint8_t)g;
            rgb[(y * v->w + x) * 3 + 2] = (uint8_t)b;
        }
    }
}

void plt_pre_build_lut_fed(uint8_t lut[256], float scale, int zero_point, int linear)
{
    int8_t q[256];
    if (linear) plt_pre_build_lut_linear(q, scale, zero_point);
    else        plt_pre_build_lut(q, scale, zero_point);
    /* fold the zero-point removal the packer used to do per byte */
    for (int p = 0; p < 256; p++) lut[p] = (uint8_t)((int)q[p] - zero_point);
}

void plt_pre_view_ndhwc32(const plt_pre_view_t *v, const uint8_t lut[256],
                          volatile uint8_t *dst, uint32_t row_bytes, int bgr,
                          int dst_h, uint8_t pad)
{
    const uint8_t *uv = v->nv12 + (size_t)v->stride * v->cap_h;
    const int c0 = bgr ? 2 : 0, c2 = bgr ? 0 : 2;

    for (int y = 0; y < v->h; y++) {
        const uint8_t *yr = v->nv12 + (size_t)(v->y0 + y) * v->stride + v->x0;
        const uint8_t *ur = uv + (size_t)((v->y0 + y) / 2) * v->stride + v->x0;
        uint8_t *p = (uint8_t *)dst + (uint32_t)y * row_bytes;
        /* Chroma is subsampled 2x2, so a pixel pair shares one u,v -- and with
         * it the three chroma terms.  Two pixels per step computes them once. */
        for (int x = 0; x < v->w; x += 2, p += 64) {
            const int n = (x + 1 < v->w) ? 2 : 1;      /* an odd width ends short */
            const int cu = ur[x] - 128, cv = ur[x + 1] - 128;
            const int dr = (1436 * cv) >> 10;
            const int dg = -((352 * cu) >> 10) - ((731 * cv) >> 10);
            const int db = (1815 * cu) >> 10;
            for (int k = 0; k < n; k++) {
                const int yy = yr[x + k];
                int r = yy + dr, g = yy + dg, b = yy + db;
                r = r < 0 ? 0 : r > 255 ? 255 : r;
                g = g < 0 ? 0 : g > 255 ? 255 : g;
                b = b < 0 ? 0 : b > 255 ? 255 : b;
                uint8_t *cell = p + k * 32;
                cell[c0] = lut[r];
                cell[1]  = lut[g];
                cell[c2] = lut[b];
            }
        }
    }
    /* The letterbox margin.  Setting every lane of the cell is simplest and
     * harmless: only the first three are read, and the fourth is masked off by
     * the word load in the focus layer. */
    for (int y = v->h; y < dst_h; y++)
        memset((uint8_t *)dst + (uint32_t)y * row_bytes, pad, row_bytes);
}

void plt_pre_view_rgb_packed(const plt_pre_view_t *v, const uint8_t lut[256],
                             volatile uint8_t *dst, uint32_t row_bytes, int bgr,
                             int dst_h, uint8_t pad)
{
    const uint8_t *uv = v->nv12 + (size_t)v->stride * v->cap_h;
    const int c0 = bgr ? 2 : 0, c2 = bgr ? 0 : 2;

    for (int y = 0; y < v->h; y++) {
        const uint8_t *yr = v->nv12 + (size_t)(v->y0 + y) * v->stride + v->x0;
        const uint8_t *ur = uv + (size_t)((v->y0 + y) / 2) * v->stride + v->x0;
        uint8_t *p = (uint8_t *)dst + (uint32_t)y * row_bytes;
        for (int x = 0; x < v->w; x += 2, p += 6) {
            const int n = (x + 1 < v->w) ? 2 : 1;
            const int cu = ur[x] - 128, cv = ur[x + 1] - 128;
            const int dr = (1436 * cv) >> 10;
            const int dg = -((352 * cu) >> 10) - ((731 * cv) >> 10);
            const int db = (1815 * cu) >> 10;
            for (int k = 0; k < n; k++) {
                const int yy = yr[x + k];
                int r = yy + dr, g = yy + dg, b = yy + db;
                r = r < 0 ? 0 : r > 255 ? 255 : r;
                g = g < 0 ? 0 : g > 255 ? 255 : g;
                b = b < 0 ? 0 : b > 255 ? 255 : b;
                uint8_t *px = p + k * 3;
                px[c0] = lut[r];
                px[1]  = lut[g];
                px[c2] = lut[b];
            }
        }
    }
    for (int y = v->h; y < dst_h; y++)
        memset((uint8_t *)dst + (uint32_t)y * row_bytes, pad, row_bytes);
}
