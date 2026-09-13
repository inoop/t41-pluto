#include "exec/plt_conv_std.h"
#include <stdlib.h>
#include "core/plt_layout.h"
#include "hal/plt_dev.h"

/* A general int8 convolution on the CPU: u = byte-128-zp, integer accumulate,
 * then the array's packed requant (floor((4*acc+B)/M)+128).  Byte-exact with
 * compile/detect_int._conv_bytes.
 *
 * The fallback for a convolution no NNA recipe serves: compile/ops/genconv.py
 * claims a general window with cin > 32, which plt_conv_dense.c cannot take
 * either once KH*KW*cin/32 passes the array's 32-group walk limit.
 *
 * MobileNetV1 and YOLOX-Nano never reach it -- their windows fit plt_conv_stem.c
 * -- and that is exactly how it shipped reading only input group 0: the one
 * assumption it makes about its input is the one its only caller breaks.  It is
 * slow, but it is the only thing standing behind the NNA, so it is worth being
 * right.  What covers it is the YOLOX-S per-layer gate (docs/BUILD.md): that
 * model is the first with windows this executor has to take, and it compares
 * every layer against the host simulator. */

static long floordiv(long long num, long den)      /* den > 0 */
{
    long long q = num / den;
    if (num % den != 0 && num < 0) q--;
    return (long)q;
}

static int std_plan(plt_ctx_t *ctx, plt_node_t *node, plt_plan_t *plan)
{
    (void)ctx;
    const plt_tensor_t *o = &node->out;
    plan->out_pad_h = plt_pad_h(o->shape.h);
    plan->out_pad_w = plt_pad_w(o->shape.w);
    plan->out_bytes = plt_ndhwc32_bytes(o->shape.c, o->shape.h, o->shape.w);
    plan->scratch_bytes = 0;
    return 0;
}

static int std_run(plt_ctx_t *ctx, const plt_node_t *node)
{
    const plt_layer_rec_t *r = node->rec;
    const plt_tensor_t *in = &node->in[0], *out = &node->out;
    const int Cin = in->shape.c, H = in->shape.h, W = in->shape.w;
    const int Cout = out->shape.c, OH = out->shape.h, OW = out->shape.w;
    const int kh = r->kh, kw = r->kw, sh = r->sh, sw = r->sw, pt = r->pt, pl = r->pl;
    const int zp = r->in_zp;

    const volatile uint8_t *src = plt_ctx_ptr(ctx, in->mem);
    volatile uint8_t *dst = plt_ctx_ptr(ctx, out->mem);
    const volatile int8_t *w = (const volatile int8_t *)plt_ctx_ptr(ctx, node->weights);
    const volatile int32_t *bm = (const volatile int32_t *)plt_ctx_ptr(ctx, node->table);
    if (!src || !dst || !w || !bm) return plt_ctx_fail(ctx, "conv_std: layer %u unmapped", node->rec->id);

    const uint32_t inb = plt_ndhwc32_bytes(Cin, H, W);
    const uint32_t irow = plt_ndhwc32_row_bytes(W);
    const uint32_t iplane = plt_ndhwc32_plane_bytes(H, W);
    const uint32_t orow = plt_ndhwc32_row_bytes(OW), oplane = plt_ndhwc32_plane_bytes(OH, OW);

    int8_t *ib = malloc(inb); int8_t *wr = malloc((size_t)Cout * Cin * kh * kw);
    int32_t *B = malloc((size_t)Cout * 4), *M = malloc((size_t)Cout * 4);
    uint8_t *ob = calloc(1, plt_ndhwc32_bytes(Cout, OH, OW));
    if (!ib || !wr || !B || !M || !ob) { free(ib); free(wr); free(B); free(M); free(ob);
        return plt_ctx_fail(ctx, "conv_std: oom"); }
    plt_copy_fast(ib, src, inb);
    for (int i = 0; i < Cout * Cin * kh * kw; i++) wr[i] = w[i];
    for (int i = 0; i < Cout; i++) { B[i] = bm[i]; M[i] = bm[Cout + i]; }

    const uint32_t t0 = plt_prof_now(ctx);
    for (int co = 0; co < Cout; co++) {
        const int8_t *wc = wr + (size_t)co * Cin * kh * kw;
        for (int oy = 0; oy < OH; oy++)
            for (int ox = 0; ox < OW; ox++) {
                long long acc = 0;
                for (int ky = 0; ky < kh; ky++) {
                    const int iy = oy * sh + ky - pt;
                    if (iy < 0 || iy >= H) continue;
                    for (int kx = 0; kx < kw; kx++) {
                        const int ix = ox * sw + kx - pl;
                        if (ix < 0 || ix >= W) continue;
                        const uint8_t *cell = (uint8_t *)ib + iy * irow + ix * 32;
                        for (int ci = 0; ci < Cin; ci++) {
                            /* NDHWC32: channel ci is lane ci%32 of PLANE ci/32,
                             * so a layer wider than one group reaches across
                             * planes -- which is every layer that gets here,
                             * since compile/ops/genconv.py claims Cin > 32. */
                            int u = (int)cell[(ci >> 5) * iplane + (ci & 31)] - 128 - zp;
                            acc += (long long)u * wc[(ci * kh + ky) * kw + kx];  /* [ci][ky][kx] */
                        }
                    }
                }
                long p = floordiv(4 * acc + B[co], M[co]) + 128;
                if (p < 0) p = 0; else if (p > 255) p = 255;
                ob[(co / 32) * oplane + oy * orow + ox * 32 + (co % 32)] = (uint8_t)p;
            }
    }
    plt_copy_fast(dst, ob, plt_ndhwc32_bytes(Cout, OH, OW));
    free(ib); free(wr); free(B); free(M); free(ob);
    plt_prof_mark(ctx, &ctx->prof.array_us, t0);
    return 0;
}

const plt_kernel_t plt_kernel_conv_std = { PLT_OP_CONV, PLT_EX_CPU_CONV, "conv_std", std_plan, std_run };
