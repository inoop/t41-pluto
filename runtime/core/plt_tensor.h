/* Runtime tensors: what a layer's operands are, independent of any hardware.
 *
 * The on-disk records are in core/plt_model.h.  These are what the engine and
 * the executors pass around.  See docs/ARCHITECTURE.md 3.1 for the invariants;
 * the ones worth repeating here:
 *
 *   - `shape` is always the LOGICAL shape.  `c` is the true channel count, not
 *     the padded one; padding belongs to the layout, and core/plt_layout.h
 *     computes it.
 *   - `mem.bytes` is the ALLOCATED size, which for an NDHWC32 tensor does
 *     include that padding.
 *   - A tensor never owns memory.  It refers into an arena; the arena owns it.
 */
#ifndef PLT_CORE_TENSOR_H
#define PLT_CORE_TENSOR_H

#include <stdint.h>

#include "core/plt_model.h"     /* the dtype / format enums */
#include "hal/plt_mem.h"        /* plt_space_t, plt_mem_t */

typedef struct { int32_t n, c, h, w; } plt_shape_t;

/* An affine quantization: real = scale * (q - zero_point). */
typedef struct { float scale; int32_t zero_point; } plt_quant_t;

typedef struct {
    plt_shape_t shape;
    uint8_t     dtype;      /* enum plt_dtype  */
    uint8_t     layout;     /* enum plt_format */
    plt_quant_t q;
    plt_mem_t   mem;
    const char *name;
} plt_tensor_t;

#endif /* PLT_CORE_TENSOR_H */
