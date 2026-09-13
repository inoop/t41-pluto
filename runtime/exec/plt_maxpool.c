#include "exec/plt_maxpool.h"
#include <stdlib.h>
#include <string.h>
#include "core/plt_layout.h"
#include "exec/plt_requant.h"
#include "hal/plt_dev.h"
#include "hal/plt_mxu3.h"

/* acc = max(acc, src) over `n` bytes (a multiple of 16), unsigned byte lanes.
 * A windowed max is the one part of this layer that vectorizes outright: it
 * stays in byte lanes, so there is no widen/narrow and no table -- 16 lanes
 * per instruction, and bit-exact by construction. */
static inline void m3_max_into(uint8_t *acc_, const uint8_t *src_, uint32_t n)
{
    uint32_t k = n >> 6;                       /* two 32-byte cells a pass */
    if (k) {
        register const uint8_t *src __asm__("t0") = src_;
        register uint8_t       *acc __asm__("t1") = acc_;
        __asm__ __volatile__(
            ".set push\n\t.set noreorder\n\t.set noat\n\t"
            "1:\n\t"
            PLT_M3_LAO(0, 0, 8, 0) PLT_M3_LAO(0, 1, 8, 1)   /* $t0 = src, $t1 = acc */
            PLT_M3_LAO(1, 0, 9, 0) PLT_M3_LAO(1, 1, 9, 1)
            PLT_M3_OP(PLT_M3_MAXUB, 0, 1, 2)
            PLT_M3_SAO(2, 0, 9, 0) PLT_M3_SAO(2, 1, 9, 1)
            "addiu %[s], %[s], 64\n\t"
            "addiu %[d], %[d], 64\n\t"
            "addiu %[k], %[k], -1\n\t"
            "bnez %[k], 1b\n\t"
            "nop\n\t"
            ".set pop\n\t"
            : [s] "+r"(src), [d] "+r"(acc), [k] "+r"(k) :: "memory");
    }
    if (n & 32) {                              /* an odd cell */
        const uint32_t o = (n >> 6) << 6;
        register const uint8_t *src __asm__("t0") = src_ + o;
        register uint8_t       *acc __asm__("t1") = acc_ + o;
        __asm__ __volatile__(
            ".set push\n\t.set noreorder\n\t.set noat\n\t"
            PLT_M3_LAO(0, 0, 8, 0)
            PLT_M3_LAO(1, 0, 9, 0)
            PLT_M3_OP(PLT_M3_MAXUB, 0, 1, 2)
            PLT_M3_SAO(2, 0, 9, 0)
            ".set pop\n\t"
            :: "r"(src), "r"(acc) : "memory");
    }
}

/* Per-lane windowed max (byte order is monotone in the real value), then a
 * requant LUT.  Out-of-range window positions take byte 0 (the minimum). */
static int mp_plan(plt_ctx_t *ctx, plt_node_t *node, plt_plan_t *plan)
{
    (void)ctx;
    const plt_tensor_t *in = &node->in[0], *o = &node->out;
    plan->out_pad_h = plt_pad_h(o->shape.h);
    plan->out_pad_w = plt_pad_w(o->shape.w);
    plan->out_bytes = plt_ndhwc32_bytes(o->shape.c, o->shape.h, o->shape.w);
    /* The separable path (stride 1) needs one group-plane to hold the
     * horizontal-max intermediate: input rows by output columns. */
    plan->scratch_bytes = plt_ndhwc32_plane_bytes(in->shape.h, o->shape.w)
                        + plt_ndhwc32_plane_bytes(o->shape.h, o->shape.w);
    return 0;
}

static int mp_run(plt_ctx_t *ctx, const plt_node_t *node)
{
    const plt_layer_rec_t *r = node->rec;
    const plt_tensor_t *in = &node->in[0], *out = &node->out;
    const int C = in->shape.c, H = in->shape.h, W = in->shape.w;
    const int OH = out->shape.h, OW = out->shape.w, D = plt_groups(C);
    const int kh = r->kh, kw = r->kw, sh = r->sh, sw = r->sw, pt = r->pt, pl = r->pl;

    const volatile uint8_t *src = plt_ctx_ptr(ctx, in->mem);
    volatile uint8_t *dst = plt_ctx_ptr(ctx, out->mem);
    const volatile uint8_t *tbl = plt_ctx_ptr(ctx, node->table);
    if (!src || !dst || !tbl) return plt_ctx_fail(ctx, "maxpool: layer %u unmapped", node->rec->id);
    uint8_t lut[256]; for (int i = 0; i < 256; i++) lut[i] = tbl[i];
    plt_requant_t rq; plt_requant_of(tbl, 1, 0, &rq);
    int32_t cst[PLT_RQ_NCONST * 16] __attribute__((aligned(64)));
    plt_requant_consts(&rq, cst);
    plt_requant_load(cst);

    const uint32_t irow = plt_ndhwc32_row_bytes(W), iplane = plt_ndhwc32_plane_bytes(H, W);
    const uint32_t orow = plt_ndhwc32_row_bytes(OW), oplane = plt_ndhwc32_plane_bytes(OH, OW);

    /* Cached arena: read src and write dst in place -- no bounce buffers. */
    const uint8_t *ps = (const uint8_t *)src;
    uint8_t       *pd = (uint8_t *)dst;
    const uint32_t t0 = plt_prof_now(ctx);

    /* A windowed max is separable (max is associative), so a stride-1 KxK pool
     * is a horizontal K-max then a vertical K-max -- 2K comparisons per output
     * instead of K*K.  Out-of-range positions contribute byte 0 (the minimum),
     * which the m[]=0 init already accounts for.  The LUT is monotone in the
     * real value, so it is applied once at the end. */
    uint8_t *hmax;
    const uint32_t hrow = plt_ndhwc32_row_bytes(OW);
    const uint32_t hplane = plt_ndhwc32_plane_bytes(H, OW);
    const uint32_t tplane = plt_ndhwc32_plane_bytes(OH, OW);
    if (sh == 1 && sw == 1 &&
        (hmax = plt_ctx_scratch(ctx, hplane + tplane)) != NULL) {
        uint8_t *tmp = hmax + hplane;
        /* Separable, and each pass is a span-wide unsigned byte max: for a given
         * tap the whole valid span shifts by whole cells (or whole rows), so one
         * vector pass covers every output position at once.  0 is the identity. */
        for (int g = 0; g < D; g++) {
            memset(hmax, 0, hplane);
            for (int y = 0; y < H; y++) {              /* horizontal */
                const uint8_t *sr = ps + g * iplane + y * irow;
                uint8_t *hr = hmax + y * hrow;
                for (int kx = 0; kx < kw; kx++) {
                    const int sft = kx - pl;
                    int lo = -sft < 0 ? 0 : -sft;
                    int hi = OW < W - sft ? OW : W - sft;
                    if (hi > lo)
                        m3_max_into(hr + lo * 32, sr + (lo + sft) * 32,
                                    (uint32_t)(hi - lo) * 32);
                }
            }
            memset(tmp, 0, tplane);
            for (int ky = 0; ky < kh; ky++) {          /* vertical: whole rows */
                const int sft = ky - pt;
                int lo = -sft < 0 ? 0 : -sft;
                int hi = OH < H - sft ? OH : H - sft;
                if (hi > lo)
                    m3_max_into(tmp + (uint32_t)lo * hrow,
                                hmax + (uint32_t)(lo + sft) * hrow,
                                (uint32_t)(hi - lo) * hrow);
            }
            /* The LUT is monotone in the real value, so it applies once at the
             * end -- and as arithmetic over the whole row, not per byte.  For a
             * scale-preserving pool (every one in YOLOX) it is the identity and
             * this is a copy. */
            for (int oy = 0; oy < OH; oy++)
                plt_requant_run(pd + g * oplane + oy * orow, tmp + oy * hrow,
                                (uint32_t)OW * 32u, &rq, lut);
        }
        plt_prof_mark(ctx, &ctx->prof.tail_us, t0);
        return 0;
    }

    for (int g = 0; g < D; g++)
        for (int oy = 0; oy < OH; oy++)
            for (int ox = 0; ox < OW; ox++) {
                uint8_t m[32]; for (int k = 0; k < 32; k++) m[k] = 0;
                for (int ky = 0; ky < kh; ky++) {
                    const int iy = oy * sh + ky - pt;
                    if (iy < 0 || iy >= H) continue;
                    for (int kx = 0; kx < kw; kx++) {
                        const int ix = ox * sw + kx - pl;
                        if (ix < 0 || ix >= W) continue;
                        const uint8_t *ic = ps + g * iplane + iy * irow + ix * 32;
                        for (int k = 0; k < 32; k++) if (ic[k] > m[k]) m[k] = ic[k];
                    }
                }
                uint8_t *oc = pd + g * oplane + oy * orow + ox * 32;
                for (int k = 0; k < 32; k++) oc[k] = plt_requant_byte(&rq, lut, m[k]);
            }
    plt_prof_mark(ctx, &ctx->prof.tail_us, t0);
    return 0;
}

const plt_kernel_t plt_kernel_maxpool = { PLT_OP_MAXPOOL, PLT_EX_MXU_MAXPOOL, "maxpool", mp_plan, mp_run };
