#include "exec/plt_concat.h"
#include <stdlib.h>
#include "core/plt_layout.h"
#include "exec/plt_requant.h"
#include "hal/plt_dev.h"

/* Channel concatenation: each input is requantized to the output scale and
 * copied into its own channel range.
 *
 * Two things used to make this the most expensive non-convolution layer.  Every
 * byte went through a 256-entry table, which cannot be vectorized; and an input
 * whose channel count is not a multiple of 32 lands at a lane offset inside the
 * output's cells, which turned the copy into a byte-by-byte scatter.  Layer 10
 * -- two 16-channel inputs meeting in one 32-lane group -- cost 8.2 ms on its
 * own, a third of the whole layer kind.
 *
 * Both go away at 16-byte granularity.  The requant is arithmetic
 * (plt_requant.h), and a 16-byte channel quad is exactly one vector lane, so
 * four cells' quads gather into one register, requantize together and scatter
 * back -- `laq` in, `saq` out, 64 payload bytes a pass whatever lane offset the
 * input lands on.  A whole aligned 32-lane group skips even that and runs as a
 * contiguous row.
 */
static int cc_plan(plt_ctx_t *ctx, plt_node_t *node, plt_plan_t *plan)
{
    (void)ctx;
    const plt_tensor_t *o = &node->out;
    plan->out_pad_h = plt_pad_h(o->shape.h);
    plan->out_pad_w = plt_pad_w(o->shape.w);
    plan->out_bytes = plt_ndhwc32_bytes(o->shape.c, o->shape.h, o->shape.w);
    plan->scratch_bytes = 0;
    return 0;
}

/* Move `cells` channel quads, one per 32-byte cell, requantizing on the way.
 * Source and destination each point at their own 16-byte lane within the first
 * cell; both advance by 32 per cell. */
/* An NDHWC16 source (`src_cell` 16) holds its quad at the start of a 16-byte
 * cell, so the four quads of a pass are adjacent rather than 32 bytes apart. */
#define CC_Q1 1
#define CC_Q2 2
#define CC_Q3 3
static void cc_quads16(uint8_t *dst_, const uint8_t *src_, int cells,
                       const plt_requant_t *r, const uint8_t lut[256])
{
    uint32_t n = (uint32_t)cells >> 2;
    if (n) {
        register const uint8_t *s __asm__("t0") = src_;
        register uint8_t       *d __asm__("t1") = dst_;
        if (r->kind == PLT_RQ_AFFINE)
            __asm__ __volatile__(".set push\n\t.set noreorder\n\t.set noat\n\t"
                "1:\n\t"
                PLT_M3_LAQ(0, 0, 8, 0) PLT_M3_LAQ(0, 1, 8, CC_Q1)
                PLT_M3_LAQ(0, 2, 8, CC_Q2) PLT_M3_LAQ(0, 3, 8, CC_Q3)
                PLT_RQ_APPLY64(0)
                PLT_M3_SAQ(0, 0, 9, 0) PLT_M3_SAQ(0, 1, 9, 2)
                PLT_M3_SAQ(0, 2, 9, 4) PLT_M3_SAQ(0, 3, 9, 6)
                "addiu %[s], %[s], 64\n\t" "addiu %[d], %[d], 128\n\t"
                "addiu %[n], %[n], -1\n\t" "bnez %[n], 1b\n\t" "nop\n\t"
                ".set pop\n\t"
                : [s] "+r"(s), [d] "+r"(d), [n] "+r"(n) :: "memory");
        else
            __asm__ __volatile__(".set push\n\t.set noreorder\n\t.set noat\n\t"
                "1:\n\t"
                PLT_M3_LAQ(0, 0, 8, 0) PLT_M3_LAQ(0, 1, 8, CC_Q1)
                PLT_M3_LAQ(0, 2, 8, CC_Q2) PLT_M3_LAQ(0, 3, 8, CC_Q3)
                PLT_M3_SAQ(0, 0, 9, 0) PLT_M3_SAQ(0, 1, 9, 2)
                PLT_M3_SAQ(0, 2, 9, 4) PLT_M3_SAQ(0, 3, 9, 6)
                "addiu %[s], %[s], 64\n\t" "addiu %[d], %[d], 128\n\t"
                "addiu %[n], %[n], -1\n\t" "bnez %[n], 1b\n\t" "nop\n\t"
                ".set pop\n\t"
                : [s] "+r"(s), [d] "+r"(d), [n] "+r"(n) :: "memory");
        src_ += (uint32_t)(cells & ~3) * 16; dst_ += (uint32_t)(cells & ~3) * 32;
    }
    for (int x = cells & ~3; x < cells; x++, src_ += 16, dst_ += 32)
        for (int k = 0; k < 16; k++) dst_[k] = plt_requant_byte(r, lut, src_[k]);
}

static void cc_quads(uint8_t *dst_, const uint8_t *src_, int cells,
                     const plt_requant_t *r, const uint8_t lut[256])
{
    uint32_t n = (uint32_t)cells >> 2;               /* four cells a pass */
    if (n) {
        register const uint8_t *s __asm__("t0") = src_;
        register uint8_t       *d __asm__("t1") = dst_;
        if (r->kind == PLT_RQ_AFFINE)
            __asm__ __volatile__(".set push\n\t.set noreorder\n\t.set noat\n\t"
                "1:\n\t"
                PLT_M3_LAQ(0, 0, 8, 0) PLT_M3_LAQ(0, 1, 8, 2)
                PLT_M3_LAQ(0, 2, 8, 4) PLT_M3_LAQ(0, 3, 8, 6)
                PLT_RQ_APPLY64(0)
                PLT_M3_SAQ(0, 0, 9, 0) PLT_M3_SAQ(0, 1, 9, 2)
                PLT_M3_SAQ(0, 2, 9, 4) PLT_M3_SAQ(0, 3, 9, 6)
                "addiu %[s], %[s], 128\n\t" "addiu %[d], %[d], 128\n\t"
                "addiu %[n], %[n], -1\n\t" "bnez %[n], 1b\n\t" "nop\n\t"
                ".set pop\n\t"
                : [s] "+r"(s), [d] "+r"(d), [n] "+r"(n) :: "memory");
        else
            __asm__ __volatile__(".set push\n\t.set noreorder\n\t.set noat\n\t"
                "1:\n\t"
                PLT_M3_LAQ(0, 0, 8, 0) PLT_M3_LAQ(0, 1, 8, 2)
                PLT_M3_LAQ(0, 2, 8, 4) PLT_M3_LAQ(0, 3, 8, 6)
                PLT_M3_SAQ(0, 0, 9, 0) PLT_M3_SAQ(0, 1, 9, 2)
                PLT_M3_SAQ(0, 2, 9, 4) PLT_M3_SAQ(0, 3, 9, 6)
                "addiu %[s], %[s], 128\n\t" "addiu %[d], %[d], 128\n\t"
                "addiu %[n], %[n], -1\n\t" "bnez %[n], 1b\n\t" "nop\n\t"
                ".set pop\n\t"
                : [s] "+r"(s), [d] "+r"(d), [n] "+r"(n) :: "memory");
        src_ += (uint32_t)(cells & ~3) * 32; dst_ += (uint32_t)(cells & ~3) * 32;
    }
    for (int x = cells & ~3; x < cells; x++, src_ += 32, dst_ += 32)
        for (int k = 0; k < 16; k++) dst_[k] = plt_requant_byte(r, lut, src_[k]);
}

static int cc_run(plt_ctx_t *ctx, const plt_node_t *node)
{
    const plt_tensor_t *out = &node->out;
    const int OH = out->shape.h, OW = out->shape.w, Cout = out->shape.c;
    volatile uint8_t *dst = plt_ctx_ptr(ctx, out->mem);
    const volatile uint8_t *tbl = plt_ctx_ptr(ctx, node->table);
    if (!dst || !tbl) return plt_ctx_fail(ctx, "concat: layer %u unmapped", node->rec->id);

    const uint32_t orow = plt_ndhwc32_row_bytes(OW), oplane = plt_ndhwc32_plane_bytes(OH, OW);

    /* The activation arena is cached memory, so both inputs and the output are
     * read/written in place -- no bounce buffers.  Padding lanes (a partial top
     * group when Cout is not a multiple of 32) are the only bytes no input
     * writes, so zero just those; every other lane is overwritten below. */
    uint8_t *pd = (uint8_t *)dst;
    const uint32_t t0 = plt_prof_now(ctx);
    if (Cout & 31) {
        const int og = Cout / 32;
        for (int y = 0; y < OH; y++)
            for (int x = 0; x < OW; x++)
                for (int ol = Cout & 31; ol < 32; ol++)
                    pd[og * oplane + y * orow + x * 32 + ol] = 0;
    }

    int off = 0;
    for (int s = 0; s < node->n_in; s++) {
        const plt_tensor_t *in = &node->in[s];
        const int Ci = in->shape.c, H = in->shape.h, W = in->shape.w;
        /* Written in place, already rescaled, by its producer: see
         * plt_node_t.in_view_mask. */
        if (node->in_view_mask & (1u << s)) { off += Ci; continue; }
        const uint8_t *src = (const uint8_t *)plt_ctx_ptr(ctx, in->mem);
        if (!src) return plt_ctx_fail(ctx, "concat: input %d unmapped", s);
        uint8_t lut[256]; for (int i = 0; i < 256; i++) lut[i] = tbl[s * 256 + i];
        plt_requant_t r; plt_requant_of(tbl, node->n_in, s, &r);
        int32_t cst[PLT_RQ_NCONST * 16] __attribute__((aligned(64)));
        plt_requant_consts(&r, cst);
    plt_requant_load(cst);

        const uint32_t irow = plt_ndhwc32_row_bytes(W), iplane = plt_ndhwc32_plane_bytes(H, W);

        if (in->layout == PLT_FMT_NDHWC16) {
            /* at most 16 channels, one quad a cell (plt_engine_load checked the
             * offset is whole quads and the requant has a vector form) */
            const int dj = off / 16, dg = dj / 2, dq = dj % 2;
            const uint32_t irow16 = (uint32_t)plt_pad_w(W) * 16u;
            for (int y = 0; y < OH; y++)
                cc_quads16(pd + dg * oplane + y * orow + dq * 16, src + (uint32_t)y * irow16,
                           OW, &r, lut);
            off += Ci;
            continue;
        }

        if ((off & 31) == 0 && (Ci & 31) == 0) {
            /* 32-aligned whole groups: input group g is output group off/32 + g
             * with the same lanes.  Concat only joins channels, so the two
             * planes have identical geometry and a whole GROUP is one
             * contiguous run -- worth saying, because a 13x13 row is six vector
             * passes and the per-row call overhead was most of the work on the
             * small layers.  It sweeps the padding columns too; those bytes
             * feed only outputs that get dropped. */
            const int og0 = off / 32;
            const int planewise = (H == OH && W == OW && iplane == oplane);
            for (int g = 0; g * 32 < Ci; g++) {
                if (planewise)
                    plt_requant_run(pd + (og0 + g) * oplane, src + g * iplane,
                                    oplane, &r, lut);
                else
                    for (int y = 0; y < OH; y++)
                        plt_requant_run(pd + (og0 + g) * oplane + y * orow,
                                        src + g * iplane + y * irow,
                                        (uint32_t)OW * 32u, &r, lut);
            }
        }
        else if ((off & 15) == 0 && (Ci & 15) == 0 && r.kind != PLT_RQ_TABLE) {
            /* a table has no vector form at all; that case takes the scalar
             * branch below rather than a half-correct fast path */
            /* 16-aligned: move one channel quad at a time, four cells a pass. */
            const int planewise = (H == OH && W == OW && iplane == oplane);
            for (int j = 0; j * 16 < Ci; j++) {
                const int sg = j / 2, sq = j % 2;
                const int dj = off / 16 + j, dg = dj / 2, dq = dj % 2;
                if (planewise)                       /* cells run on across rows */
                    cc_quads(pd + dg * oplane + dq * 16,
                             src + sg * iplane + sq * 16,
                             (int)(oplane / 32u), &r, lut);
                else
                    for (int y = 0; y < OH; y++)
                        cc_quads(pd + dg * oplane + y * orow + dq * 16,
                                 src + sg * iplane + y * irow + sq * 16,
                                 OW, &r, lut);
            }
        }
        else {
            for (int c = 0; c < Ci; c++) {
                const int ig = c / 32, il = c % 32, oc = off + c, og = oc / 32, ol = oc % 32;
                for (int y = 0; y < OH; y++)
                    for (int x = 0; x < OW; x++)
                        pd[og * oplane + y * orow + x * 32 + ol] =
                            plt_requant_byte(&r, lut, src[ig * iplane + y * irow + x * 32 + il]);
            }
        }
        off += Ci;
    }
    plt_prof_mark(ctx, &ctx->prof.tail_us, t0);
    return 0;
}

const plt_kernel_t plt_kernel_concat = { PLT_OP_CONCAT, PLT_EX_MXU_CONCAT, "concat", cc_plan, cc_run };
