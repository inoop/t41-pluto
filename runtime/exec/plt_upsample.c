#include "exec/plt_upsample.h"
#include <stdlib.h>
#include <string.h>
#include "core/plt_layout.h"
#include "exec/plt_requant.h"
#include "hal/plt_dev.h"
#include "hal/plt_mxu3.h"

/* Nearest-neighbour upsample by params[0], then a requant.
 *
 * Three things this deliberately does not do.  It does not bounce through
 * malloc'd copies: the activation arena is cached memory, so input and output
 * are read and written in place.  It does not walk the OUTPUT computing
 * `oy/s, ox/s` per pixel -- it walks the input once and spreads.  And it does
 * not put every byte through a 256-entry table: the requant is one arithmetic
 * pass over the input row (plt_requant.h), which for these layers is the
 * identity and so costs nothing at all.
 *
 * The spread happens in place and BACKWARDS.  Cell ix lands at ix*s, which is
 * never before ix, so filling from the last cell down never overwrites a cell
 * it still has to read -- no second buffer.
 */
static int up_plan(plt_ctx_t *ctx, plt_node_t *node, plt_plan_t *plan)
{
    (void)ctx;
    const plt_tensor_t *o = &node->out;
    plan->out_pad_h = plt_pad_h(o->shape.h);
    plan->out_pad_w = plt_pad_w(o->shape.w);
    plan->out_bytes = plt_ndhwc32_bytes(o->shape.c, o->shape.h, o->shape.w);
    plan->scratch_bytes = 0;
    return 0;
}

/* Replicate each 32-byte cell `s` times, last cell first.
 *
 * A cell is one 32-byte vector lane, so doubling it is a load into both lanes
 * of a register and one 64-byte store -- four instructions where the word loop
 * needed twenty-four. */
static inline void up_spread(uint8_t *row, int cells, int s)
{
    if (s == 2 && cells > 0) {
        register const uint8_t *sp __asm__("t0") = row + (uint32_t)(cells - 1) * 32;
        register uint8_t       *dp __asm__("t1") = row + (uint32_t)(cells - 1) * 64;
        uint32_t n = (uint32_t)cells;
        __asm__ __volatile__(
            ".set push\n\t.set noreorder\n\t.set noat\n\t"
            "1:\n\t"
            PLT_M3_LAO(0, 0, 8, 0) PLT_M3_LAO(0, 1, 8, 0)   /* both lanes = cell */
            PLT_M3_SAO(0, 0, 9, 0) PLT_M3_SAO(0, 1, 9, 1)
            "addiu %[s], %[s], -32\n\t"
            "addiu %[d], %[d], -64\n\t"
            "addiu %[n], %[n], -1\n\t"
            "bnez %[n], 1b\n\t"
            "nop\n\t"
            ".set pop\n\t"
            : [s] "+r"(sp), [d] "+r"(dp), [n] "+r"(n) :: "memory");
        return;
    }
    for (int i = cells - 1; i >= 0; i--) {
        const uint32_t *sw = (const uint32_t *)(void *)(row + (uint32_t)i * 32);
        uint32_t w[8];
        for (int k = 0; k < 8; k++) w[k] = sw[k];
        for (int j = s - 1; j >= 0; j--) {
            uint32_t *dw = (uint32_t *)(void *)(row + (uint32_t)(i * s + j) * 32);
            for (int k = 0; k < 8; k++) dw[k] = w[k];
        }
    }
}

static int up_run(plt_ctx_t *ctx, const plt_node_t *node)
{
    const plt_tensor_t *in = &node->in[0], *out = &node->out;
    const int C = in->shape.c, H = in->shape.h, W = in->shape.w;
    const int OH = out->shape.h, OW = out->shape.w;
    const int s = node->rec->params[0] ? node->rec->params[0] : 2;
    const int D = plt_groups(C);

    const volatile uint8_t *src = plt_ctx_ptr(ctx, in->mem);
    volatile uint8_t *dst = plt_ctx_ptr(ctx, out->mem);
    const volatile uint8_t *tbl = plt_ctx_ptr(ctx, node->table);
    if (!src || !dst || !tbl) return plt_ctx_fail(ctx, "upsample: layer %u unmapped", node->rec->id);

    uint8_t lut[256]; for (int i = 0; i < 256; i++) lut[i] = tbl[i];
    plt_requant_t r; plt_requant_of(tbl, 1, 0, &r);
    int32_t cst[PLT_RQ_NCONST * 16] __attribute__((aligned(64)));
    plt_requant_consts(&r, cst);
    plt_requant_load(cst);

    const uint32_t irow = plt_ndhwc32_row_bytes(W), iplane = plt_ndhwc32_plane_bytes(H, W);
    const uint32_t orow = plt_ndhwc32_row_bytes(OW), oplane = plt_ndhwc32_plane_bytes(OH, OW);
    const int opad_h = plt_pad_h(OH), opad_w = plt_pad_w(OW);

    const uint8_t *ps = (const uint8_t *)src;
    uint8_t       *pd = (uint8_t *)dst;
    const uint32_t t0 = plt_prof_now(ctx);

    for (int g = 0; g < D; g++)
        for (int iy = 0; iy < H; iy++) {
            uint8_t *row = pd + g * oplane + (uint32_t)(iy * s) * orow;
            plt_requant_run(row, ps + g * iplane + (uint32_t)iy * irow,
                            (uint32_t)W * 32u, &r, lut);
            up_spread(row, W, s);
            /* padding cells and rows are read only by outputs that get dropped,
             * but zero them so the buffer is a function of the input alone */
            if (opad_w > OW)
                memset(row + (uint32_t)OW * 32, 0, (size_t)(opad_w - OW) * 32);
            for (int j = 1; j < s; j++)
                plt_copy_fast(row + (uint32_t)j * orow, row, (size_t)opad_w * 32);
            if (iy == H - 1)
                for (int oy = iy * s + s; oy < opad_h; oy++)
                    memset(pd + g * oplane + (uint32_t)oy * orow, 0, (size_t)opad_w * 32);
        }
    plt_prof_mark(ctx, &ctx->prof.tail_us, t0);
    return 0;
}

const plt_kernel_t plt_kernel_upsample = { PLT_OP_UPSAMPLE, PLT_EX_MXU_UPSAMPLE, "upsample", up_plan, up_run };
