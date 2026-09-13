#include <math.h>

#include "exec/plt_pool_mxu.h"

/* --- the executor ---------------------------------------------------------- */

#include <stdlib.h>

#include "core/plt_layout.h"
#include "hal/plt_dev.h"

static int pool_plan(plt_ctx_t *ctx, plt_node_t *node, plt_plan_t *plan)
{
    const plt_tensor_t *in = &node->in[0];

    if (node->rec->op != PLT_OP_AVGPOOL)
        return plt_ctx_fail(ctx, "pool_mxu: layer %u is not an average pool", node->rec->id);
    if (in->layout != PLT_FMT_NDHWC32)
        return plt_ctx_fail(ctx, "pool_mxu: layer %u input is not NDHWC32", node->rec->id);

    /* The planar bounce buffer, plus the packed result.  This kernel runs on
     * the CPU, so its scratch is host memory rather than arena space. */
    plan->scratch_bytes = 0;
    plan->out_bytes     = (uint32_t)node->out.shape.c;
    plan->out_pad_h     = 1;
    plan->out_pad_w     = 1;
    return 0;
}

static int pool_run(plt_ctx_t *ctx, const plt_node_t *node)
{
    const plt_tensor_t *in  = &node->in[0];
    const int C = in->shape.c, H = in->shape.h, W = in->shape.w;

    const volatile uint8_t *src = plt_ctx_ptr(ctx, in->mem);
    volatile uint8_t       *dst = plt_ctx_ptr(ctx, node->out.mem);
    if (!src || !dst) return plt_ctx_fail(ctx, "pool_mxu: layer %u operand unmapped",
                                          node->rec->id);

    const uint32_t t0 = plt_prof_now(ctx);

    /* In place: the activation arena is ordinary cached memory, so a bounce
     * copy and a planar transpose would only move the same bytes around.
     * out[c] = quantize(mean over HxW of (q - zp) * scale), with q the fed byte
     * converted back to int8. */
    const uint8_t *s8 = (const uint8_t *)(uintptr_t)src;
    const uint32_t rowb  = plt_ndhwc32_row_bytes(W);
    const uint32_t plane = plt_ndhwc32_plane_bytes(H, W);
    const int zp = in->q.zero_point, n = H * W;
    for (int c = 0; c < C; c++) {
        const uint32_t base = (uint32_t)(c / 32) * plane + (uint32_t)(c % 32);
        long sum = 0;
        for (int y = 0; y < H; y++) {
            const uint8_t *r = s8 + base + (uint32_t)y * rowb;
            for (int x = 0; x < W; x++)
                sum += (long)(int8_t)(uint8_t)(r[(uint32_t)x * 32u] + zp) - zp;
        }
        double mean = (double)sum * (double)in->q.scale / (double)n;
        double q = rint(mean / (double)node->out.q.scale) + node->out.q.zero_point;
        if (q < -128) q = -128;
        if (q > 127)  q = 127;
        dst[c] = (uint8_t)(int8_t)q;
    }
    plt_prof_mark(ctx, &ctx->prof.tail_us, t0);
    return 0;
}

const plt_kernel_t plt_kernel_pool_mxu = {
    PLT_OP_AVGPOOL, PLT_EX_MXU_POOL, "pool_mxu", pool_plan, pool_run
};
