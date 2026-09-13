/* SiLU as a byte table: the activation is a fixed function of one byte, so a
 * lookup is the whole kernel.  It runs on MXU3's `gshufvb` byte gather, the
 * same PLT_M3_LUT64 the convolutions fold into their tile loop.
 *
 * Rarely reached.  compile/model_build.py:_fuse_silu folds a SiLU into the
 * convolution that feeds it whenever that convolution's output has exactly one
 * consumer, which in YOLOX-Nano is all 104 of them -- the fused graph has no
 * SiLU layer left.  One whose input branches would land here. */
#include "exec/plt_silu.h"
#include <stdlib.h>
#include "core/plt_layout.h"
#include "hal/plt_dev.h"
#include "exec/plt_conv_nna.h"

static uint32_t buf_bytes(const plt_tensor_t *t)
{ return plt_ndhwc32_bytes(t->shape.c, t->shape.h, t->shape.w); }

static int silu_plan(plt_ctx_t *ctx, plt_node_t *node, plt_plan_t *plan)
{
    (void)ctx;
    const plt_tensor_t *o = &node->out;
    plan->out_pad_h = plt_pad_h(o->shape.h);
    plan->out_pad_w = plt_pad_w(o->shape.w);
    plan->out_bytes = buf_bytes(o);
    plan->scratch_bytes = 0;
    return 0;
}

static int silu_run(plt_ctx_t *ctx, const plt_node_t *node)
{
    const plt_tensor_t *in = &node->in[0];
    const uint32_t n = buf_bytes(in);
    const volatile uint8_t *src = plt_ctx_ptr(ctx, in->mem);
    volatile uint8_t *dst = plt_ctx_ptr(ctx, node->out.mem);
    const volatile uint8_t *tbl = plt_ctx_ptr(ctx, node->table);
    if (!src || !dst || !tbl) return plt_ctx_fail(ctx, "silu: layer %u operand unmapped", node->rec->id);

    uint8_t lut[256];
    for (int i = 0; i < 256; i++) lut[i] = tbl[i];
    plt_conv_lut_load(tbl);

    const uint32_t t0 = plt_prof_now(ctx);
    /* The activation arena is ordinary cached memory (plt_ctx_reserve_host is a
     * malloc), so there is no uncached window to bounce through: map the table
     * straight from src to dst, 64 bytes at a time.  An NDHWC32 buffer is a
     * whole number of 32-lane cells and 64-byte aligned, so the scalar tail is
     * dead code -- it is here because a wrong `n` should be slow, not wrong. */
    const uint8_t *s = (const uint8_t *)src;
    uint8_t       *d = (uint8_t *)dst;
    uint32_t i = 0;
    for (; i + 64 <= n; i += 64) {
        PLT_VLD64(29, s + i);
        __asm__ __volatile__(".set push\n.set noreorder\n"
            PLT_M3_LUT64(29, 30, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28)
            ".set pop\n" ::: "memory");
        PLT_VST64(30, d + i);
    }
    for (; i < n; i++) d[i] = lut[s[i]];
    plt_prof_mark(ctx, &ctx->prof.tail_us, t0);
    return 0;
}

const plt_kernel_t plt_kernel_silu = { PLT_OP_SILU, PLT_EX_MXU_SILU, "silu", silu_plan, silu_run };
