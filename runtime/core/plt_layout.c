#include "core/plt_layout.h"

void plt_ndhwc32_from_planar(volatile uint8_t *dst, const int8_t *src,
                             int c, int h, int w, int zero_point)
{
    const int      pw    = plt_pad_w(w), ph = plt_pad_h(h);
    const uint32_t rowb  = plt_ndhwc32_row_bytes(w);
    const uint32_t plane = plt_ndhwc32_plane_bytes(h, w);
    const int      d     = plt_groups(c);

    for (int g = 0; g < d; g++)
        for (int y = 0; y < ph; y++)
            for (int x = 0; x < pw; x++) {
                volatile uint8_t *p = dst + g * plane + y * rowb + x * 32;
                for (int k = 0; k < 32; k++) {
                    const int ch = g * 32 + k;
                    p[k] = (ch < c && y < h && x < w)
                           ? (uint8_t)(src[(ch * h + y) * w + x] - zero_point)
                           : 0u;
                }
            }
}

void plt_ndhwc32_to_planar(int8_t *dst, const volatile uint8_t *src,
                           int c, int h, int w, int zero_point)
{
    const uint32_t rowb  = plt_ndhwc32_row_bytes(w);
    const uint32_t plane = plt_ndhwc32_plane_bytes(h, w);

    for (int ch = 0; ch < c; ch++) {
        const int g = ch / 32, k = ch % 32;
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                dst[(ch * h + y) * w + x] =
                    (int8_t)((int)src[g * plane + y * rowb + x * 32 + k] + zero_point);
    }
}
