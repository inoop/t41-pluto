/* NDHWC32: the layout the array reads and writes.
 *
 * A feature map is D = ceil(C/32) planes of PH x PW positions, each position 32
 * channels of one byte.  The array always works a whole 2-row by 4-pixel tile,
 * so a plane is padded up to those multiples; the padding is computed and
 * simply not copied out.
 *
 * This is a device-wide rule, not a property of any one executor, which is why
 * it lives here rather than next to the pointwise recipe.
 */
#ifndef PLT_CORE_LAYOUT_H
#define PLT_CORE_LAYOUT_H

#include <stdint.h>

#include "core/plt_tensor.h"

static inline int plt_pad_h(int h) { return (h + 1) & ~1; }
static inline int plt_pad_w(int w) { return (w + 3) & ~3; }
static inline int plt_groups(int c) { return (c + 31) / 32; }

static inline uint32_t plt_ndhwc32_row_bytes(int w)
{ return (uint32_t)plt_pad_w(w) * 32u; }

static inline uint32_t plt_ndhwc32_plane_bytes(int h, int w)
{ return (uint32_t)plt_pad_h(h) * plt_ndhwc32_row_bytes(w); }

static inline uint32_t plt_ndhwc32_bytes(int c, int h, int w)
{ return (uint32_t)plt_groups(c) * plt_ndhwc32_plane_bytes(h, w); }

/* Planar [C][H][W] int8 -> NDHWC32, zero-point removed.
 *
 * The array is fed u = x_q - x_zp, and its unsigned packed output IS the next
 * layer's u.  These two functions are the only place that convention lives, so
 * no kernel has to re-derive it.  Padding positions are written as zero, i.e.
 * as the zero point itself. */
void plt_ndhwc32_from_planar(volatile uint8_t *dst, const int8_t *src,
                             int c, int h, int w, int zero_point);

/* NDHWC32 -> planar [C][H][W] int8, zero point re-applied.  This is what the
 * MXU3 tail kernels want; the padding is dropped. */
void plt_ndhwc32_to_planar(int8_t *dst, const volatile uint8_t *src,
                           int c, int h, int w, int zero_point);

#endif /* PLT_CORE_LAYOUT_H */
