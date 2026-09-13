/* A dense KxK convolution over 32 or more input channels, on the NNA.
 *
 * The array runs it as an ordinary 1x1 over KH*KW*Cin pseudo-channels --
 * plt_conv_pw_configure() is called unchanged; there is no dense-specific
 * configuration.  What this file owns is the FEED: the KH*KW taps are KH*KW
 * shifted views of one bordered input plane, so nothing is lowered into memory.
 *
 * It is plt_conv_dw.c's trick with an input-group loop around it, and
 * plt_conv_pw_tiles()'s weight passes underneath:
 *
 *     depthwise    d = KH*KW          one group's taps, Cout = 32, one pass
 *     THIS         d = KH*KW*Cin/32   every group's taps, Cout up to 512
 *
 * Pseudo-group p carries input group p / (KH*KW) at tap p % (KH*KW), which is
 * compile/layout.py:dense_tap -- group-major, so the taps of one group read
 * overlapping cache lines and only the step to the next group's plane is a cold
 * jump.  tests/dense_test.py checks that the packer agrees with this.
 *
 * Nothing reaches it today -- plt_conv_k3.c takes every pad-1 3x3 on the array's
 * native walk, which is faster and is not capped at Cin = 96.  This stays for
 * the windows that recipe is not written for (a 5x5, a 3x3 with other padding),
 * which are legal and would otherwise fall to the scalar CPU convolution.
 */
#include "exec/plt_conv_dense.h"
#include "exec/plt_conv_nna.h"
#include "exec/plt_conv_pw.h"

#include "core/plt_layout.h"
#include "hal/plt_dev.h"
#include "hal/plt_nna.h"
#include "hal/plt_mxu3.h"

#define CSTEP 128                      /* column-block step: 4 px x 32 ch x 8 b */

/* The 1x1 recipe walks at most 16 input groups per execution unit -- 32 in all,
 * which a KxK window reaches at KH*KW*Cin/32, so a 3x3 tops out at Cin = 96.
 *
 * Device-measured, not inferred: a pointwise layer widened with zero weights
 * (inert, so its output cannot change) is byte-exact at D = 32 and wrong at
 * D = 33.  The bound is the walk program's `load(n-2)` repeat field, which is
 * four bits wide.  compile/ops/dense.py:MAX_GROUPS carries the same number and
 * the reasoning; a wider layer never reaches this executor. */
#define DENSE_MAX_GROUPS 32

static int dense_check(plt_ctx_t *ctx, const plt_conv_geom_t *g)
{
    if (g->kh != g->kw || g->kh < 2)
        return plt_ctx_fail(ctx, "conv_dense: kernel %dx%d is not a square window",
                            g->kh, g->kw);
    if (g->cin % 32)
        return plt_ctx_fail(ctx, "conv_dense: %d input channels are not a multiple of 32",
                            g->cin);
    if (g->kh * g->kw * (g->cin / 32) > DENSE_MAX_GROUPS)
        return plt_ctx_fail(ctx, "conv_dense: %dx%d over %d channels needs %d tap groups, "
                            "more than the %d the array holds", g->kh, g->kw, g->cin,
                            g->kh * g->kw * (g->cin / 32), DENSE_MAX_GROUPS);
    if (g->in_bits != 8 || g->w_bits != 8 || g->out_bits != 8)
        return plt_ctx_fail(ctx, "conv_dense: %d/%d/%d bits, only 8/8/8 is implemented",
                            g->in_bits, g->w_bits, g->out_bits);
    return 0;
}

/* The 1x1 convolution the window becomes: one pseudo-channel per (input
 * channel, tap), the output channels rounded to a whole group. */
static void as_pointwise(const plt_conv_geom_t *g, plt_conv_geom_t *pw)
{
    *pw = *g;
    pw->cin    = g->kh * g->kw * g->cin;
    pw->cout   = (g->cout + 31) & ~31;
    pw->kh     = pw->kw = 1;
    pw->stride = 1;
    pw->pad_t  = pw->pad_l = 0;
    pw->in_h   = g->out_h;
    pw->in_w   = g->out_w;
}

/* Stride 1: every tap is a shifted pointer into the bordered input.
 *
 * This is plt_conv_pw_tiles() with the feed address changed and nothing else --
 * written out rather than parameterised, because that address is the whole
 * point of the file (the same choice plt_conv_dw.c makes).
 *
 * Pseudo-group g reads input group g/K2 at kernel row (g%K2)/kw and column
 * (g%K2)%kw.  Output pixel (oy,ox) wants input (oy+ky-1, ox+kx-1); the border
 * shifts the plane by one in each direction, so the -1s cancel.
 */
static void dense_tiles_stride1(const plt_conv_geom_t *pw, const plt_conv_bufs_t *b,
                                const volatile uint8_t *plane, uint32_t brow,
                                uint32_t bplane, int k2, int kw)
{
    const int d = pw->cin / 32, na = d / 2, dout = pw->cout / 32;
    const int ppw = plt_pad_w(pw->out_w), pph = plt_pad_h(pw->out_h);
    const uint32_t orow = plt_ndhwc32_row_bytes(pw->out_w);
    const uint32_t oplane = plt_ndhwc32_plane_bytes(pw->out_h, pw->out_w);
    const int cb_n = ppw / 4, rp_n = pph / 2;
    const int gpp = plt_conv_groups_per_pass(8, d);
    int prp = -1, pcb = -1, pg = -1;
    const int has_lut = b->act_lut != NULL;
    if (has_lut) plt_conv_lut_load(b->act_lut);

#define DN_STORE_PREV() do { if (pg >= 0) {                                     \
    volatile uint8_t *_t = b->out + pg * oplane + (2 * prp) * orow + pcb * CSTEP; \
    PLT_CONV_DRAIN(_t, orow, has_lut);                                          \
    } } while (0)

#define DN_TAP(tile, g) ((tile) + (uint32_t)((g) / k2) * bplane                 \
                                + (uint32_t)(((g) % k2) / kw) * brow            \
                                + (uint32_t)(((g) % k2) % kw) * 32u)

    for (int g0 = 0; g0 < dout; g0 += gpp) {
        const int gp = dout - g0 < gpp ? dout - g0 : gpp;

        PLT_NNA_FIELD(PLT_BANK_B, PLT_PW_W_ZERO, PLT_B_WSTREAM);
        for (int i = 0; i < gp * d * 4; i++)
            plt_nna_push_weight_slice(b->weights + (uint32_t)(g0 * d) * 1024 + i * 256);

        PLT_NNA_FIELD(PLT_BANK_B, PLT_PW_W_ZERO, PLT_B_TABLE_PTR);
        for (int j = 0; j < gp * 2; j++)
            plt_nna_push_table_half(b->table + g0 * 256 + j * 128);

        for (int rp = 0; rp < rp_n; rp++) {
            for (int cb = 0; cb < cb_n; cb++) {
                const volatile uint8_t *tile = plane + (uint32_t)(2 * rp) * brow
                                                     + (uint32_t)(4 * cb) * 32u;
                if (cb > 0) plt_nna_precommit();
                for (int g = 0; g < na; g++)
                    PLT_NNA_FEED_TILE(DN_TAP(tile, g), brow, PLT_PORT_ACT_A);
                if (cb > 0) plt_nna_pack();
                PLT_NNA_FIELD(PLT_BANK_A, PLT_PW_W_ZERO, PLT_A_MAC_A1);
                PLT_NNA_FIELD(PLT_BANK_A, PLT_PW_W_ZERO, PLT_A_MAC_A0);
                PLT_NNMAC(PLT_MAC_UNIT_A);
                for (int g = na; g < d; g++)
                    PLT_NNA_FEED_TILE(DN_TAP(tile, g), brow, PLT_PORT_ACT_B);
                PLT_NNMAC(PLT_MAC_UNIT_B);
                if (cb > 0) DN_STORE_PREV();
                prp = rp; pcb = cb; pg = g0;
                PLT_NNA_FIELD(PLT_BANK_B, PLT_PW_W_ZERO, PLT_B_COMMIT_WIN);

                /* The rest of this pass's output groups reuse the fed tile. */
                for (int g = 1; g < gp; g++) {
                    plt_nna_precommit(); plt_nna_pack();
                    PLT_NNMAC(PLT_MAC_UNIT_A); PLT_NNMAC(PLT_MAC_UNIT_B);
                    DN_STORE_PREV();
                    prp = rp; pcb = cb; pg = g0 + g;
                }
            }
            plt_nna_precommit(); plt_nna_pack();     /* the drain runs a pair behind */
            DN_STORE_PREV();
            pg = -1;
        }
    }
#undef DN_TAP
#undef DN_STORE_PREV
}

/* Stride 2: the four output pixels of a tile take every other input cell, so a
 * tap is a gather rather than a shifted pointer -- and the gather happens in the
 * PUSH REGISTER, exactly as plt_conv_dw.c's dw_tiles_stride2() does it.  A
 * 64-byte push is two output pixels and a tap is a whole 32-channel group, so
 * each push is two 32-byte lane loads at independent addresses.
 *
 * Only the base address differs from the depthwise case: it carries the input
 * group as well as the kernel row.  `kx` must stay a literal -- the assembler
 * evaluates the offsets -- so the three kernel columns are written out. */
#define DN2_PUSH(kx, h, jj)                                 \
    PLT_M3_LAO(0, 0, 8 + (h), 4 * (jj) + 0 + (kx))          \
    PLT_M3_LAO(0, 1, 8 + (h), 4 * (jj) + 2 + (kx))          \
    ".word %[nn]\n\t"

#define DN2_FEED_TILE(kx, rowp, brow, port) do {                               \
    register const uint8_t *_r0 __asm__("t0") = (rowp);                        \
    register const uint8_t *_r1 __asm__("t1") = (rowp) + 2 * (brow);           \
    __asm__ __volatile__(".set push\n\t.set noreorder\n\t"                   \
        DN2_PUSH(kx, 0, 0) DN2_PUSH(kx, 0, 1)                                  \
        DN2_PUSH(kx, 1, 0) DN2_PUSH(kx, 1, 1)                                  \
        ".set pop\n\t"                                                         \
        :: [nn] "i"(PLT_NN_WORD(2, ((port) & 0x1Fu) | (((port) >> 5) << 10))), \
           "r"(_r0), "r"(_r1) : "memory");                                     \
} while (0)

/* One input group's three kernel rows, nine pushes, all to the same unit.
 * The unit split falls on a group boundary -- dense_fused_stride2() only
 * accepts an even group count, which makes na = 9*D/2 a multiple of nine. */
#define DN2_GROUP(tile, gi, brow, bplane, port) do {                           \
    const uint8_t *_g = (const uint8_t *)(tile) + (uint32_t)(gi) * (bplane);   \
    for (int _ky = 0; _ky < 3; _ky++) {                                        \
        const uint8_t *_r = _g + (uint32_t)_ky * (brow);                       \
        DN2_FEED_TILE(0, _r, brow, port);                                      \
        DN2_FEED_TILE(1, _r, brow, port);                                      \
        DN2_FEED_TILE(2, _r, brow, port);                                      \
    }                                                                          \
} while (0)

static void dense_tiles_stride2(const plt_conv_geom_t *pw, const plt_conv_bufs_t *b,
                                const uint8_t *plane, uint32_t brow, uint32_t bplane)
{
    const int d = pw->cin / 32, dg = d / 9, dout = pw->cout / 32;
    const int ppw = plt_pad_w(pw->out_w), pph = plt_pad_h(pw->out_h);
    const uint32_t orow = plt_ndhwc32_row_bytes(pw->out_w);
    const uint32_t oplane = plt_ndhwc32_plane_bytes(pw->out_h, pw->out_w);
    const int gpp = plt_conv_groups_per_pass(8, d);
    int prp = -1, pcb = -1, pg = -1;
    const int has_lut = b->act_lut != NULL;
    if (has_lut) plt_conv_lut_load(b->act_lut);

#define DN2_STORE_PREV() do { if (pg >= 0) {                                    \
    volatile uint8_t *_t = b->out + pg * oplane + (2 * prp) * orow + pcb * CSTEP; \
    PLT_CONV_DRAIN(_t, orow, has_lut);                                          \
    } } while (0)

    for (int g0 = 0; g0 < dout; g0 += gpp) {
        const int gp = dout - g0 < gpp ? dout - g0 : gpp;

        PLT_NNA_FIELD(PLT_BANK_B, PLT_PW_W_ZERO, PLT_B_WSTREAM);
        for (int i = 0; i < gp * d * 4; i++)
            plt_nna_push_weight_slice(b->weights + (uint32_t)(g0 * d) * 1024 + i * 256);

        PLT_NNA_FIELD(PLT_BANK_B, PLT_PW_W_ZERO, PLT_B_TABLE_PTR);
        for (int j = 0; j < gp * 2; j++)
            plt_nna_push_table_half(b->table + g0 * 256 + j * 128);

        for (int rp = 0; rp < pph / 2; rp++) {
            /* output rows 2rp, 2rp+1 read input rows 4rp .. 4rp+3 (bordered). */
            const uint8_t *row0 = plane + (uint32_t)(4 * rp) * brow;
            for (int cb = 0; cb < ppw / 4; cb++) {
                const uint8_t *tile = row0 + (uint32_t)(8 * cb) * 32;
                if (cb > 0) plt_nna_precommit();
                for (int gi = 0; gi < dg / 2; gi++)
                    DN2_GROUP(tile, gi, brow, bplane, PLT_PORT_ACT_A);
                if (cb > 0) plt_nna_pack();
                PLT_NNA_FIELD(PLT_BANK_A, PLT_PW_W_ZERO, PLT_A_MAC_A1);
                PLT_NNA_FIELD(PLT_BANK_A, PLT_PW_W_ZERO, PLT_A_MAC_A0);
                PLT_NNMAC(PLT_MAC_UNIT_A);
                for (int gi = dg / 2; gi < dg; gi++)
                    DN2_GROUP(tile, gi, brow, bplane, PLT_PORT_ACT_B);
                PLT_NNMAC(PLT_MAC_UNIT_B);
                if (cb > 0) DN2_STORE_PREV();
                prp = rp; pcb = cb; pg = g0;
                PLT_NNA_FIELD(PLT_BANK_B, PLT_PW_W_ZERO, PLT_B_COMMIT_WIN);

                for (int g = 1; g < gp; g++) {
                    plt_nna_precommit(); plt_nna_pack();
                    PLT_NNMAC(PLT_MAC_UNIT_A); PLT_NNMAC(PLT_MAC_UNIT_B);
                    DN2_STORE_PREV();
                    prp = rp; pcb = cb; pg = g0 + g;
                }
            }
            plt_nna_precommit(); plt_nna_pack();
            DN2_STORE_PREV();
            pg = -1;
        }
    }
#undef DN2_STORE_PREV
}

/* The stride-2 gather is written out for a 3x3 window with one-pixel padding and
 * an even number of input groups (so the array's two execution units split on a
 * group boundary).  Every strided dense layer in YOLOX-S meets all three;
 * anything else is rejected in the compiler and runs on the CPU fallback. */
static int dense_fused_stride2(const plt_conv_geom_t *g)
{
    return g->stride == 2 && g->kh == 3 && g->kw == 3
        && g->pad_t == 1 && g->pad_l == 1 && (g->cin / 32) % 2 == 0;
}

static int dense_prep(plt_ctx_t *ctx, const plt_node_t *node, plt_conv_geom_t *g)
{
    if (plt_conv_geom_of(ctx, node, g) != 0) return -1;
    g->cin  = (g->cin + 31) & ~31;
    g->cout = (g->cout + 31) & ~31;
    return dense_check(ctx, g);
}

/* Every input group's plane, each in its own bordered buffer, laid out
 * contiguously so a pseudo-group's address is one multiply from its index. */
static uint32_t dense_scratch(const plt_conv_geom_t *g)
{
    return (uint32_t)(g->cin / 32) * plt_conv_border_bytes(g);
}

static int dense_plan(plt_ctx_t *ctx, plt_node_t *node, plt_plan_t *plan)
{
    plt_conv_geom_t g;

    if (dense_prep(ctx, node, &g) != 0) return -1;
    if (g.stride != 1 && !dense_fused_stride2(&g))
        return plt_ctx_fail(ctx, "conv_dense: stride %d with %dx%d pad %d,%d is not "
                            "a shape this feed is written for", g.stride, g.kh, g.kw,
                            g.pad_t, g.pad_l);

    plan->out_pad_h     = plt_pad_h(g.out_h);
    plan->out_pad_w     = plt_pad_w(g.out_w);
    plan->out_bytes     = plt_ndhwc32_bytes(g.cout, g.out_h, g.out_w);
    plan->scratch_bytes = dense_scratch(&g);
    return 0;
}

static int dense_run(plt_ctx_t *ctx, const plt_node_t *node)
{
    plt_conv_geom_t g, pw;
    plt_conv_bufs_t b;

    if (dense_prep(ctx, node, &g) != 0) return -1;
    if (plt_conv_bufs_of(ctx, node, &b) != 0) return -1;

    const uint32_t need = dense_scratch(&g);
    volatile uint8_t *col = plt_ctx_scratch(ctx, need);
    if (!col)
        return plt_ctx_fail(ctx, "conv_dense: no scratch holds the %u bytes the "
                            "bordered input needs", need);

    /* The bordered copy is also the staging the multi-pass layers want: it is
     * cached, and every weight pass re-reads it.  So unlike the 1x1 path there
     * is no separate plt_conv_stage_operands() step. */
    uint32_t t = plt_prof_now(ctx);
    {
        const uint32_t bplane   = plt_conv_border_bytes(&g);
        const uint32_t in_plane = plt_ndhwc32_plane_bytes(g.in_h, g.in_w);
        for (int gi = 0; gi < g.cin / 32; gi++)
            plt_conv_border_copy(&g, b.in + (uint32_t)gi * in_plane,
                                 col + (uint32_t)gi * bplane);
    }
    t = plt_prof_mark(ctx, &ctx->prof.assemble_us, t);

    as_pointwise(&g, &pw);
    if (!b.act_lut && node->rec->act == PLT_ACT_SILU)
        b.act_lut = b.table + (uint32_t)(pw.cout / 32) * 256u;

    plt_conv_pw_configure(&pw);
    if (g.stride == 1)
        dense_tiles_stride1(&pw, &b, col, plt_conv_border_row(&g),
                            plt_conv_border_bytes(&g), g.kh * g.kw, g.kw);
    else
        dense_tiles_stride2(&pw, &b, (const uint8_t *)col, plt_conv_border_row(&g),
                            plt_conv_border_bytes(&g));
    plt_prof_mark(ctx, &ctx->prof.array_us, t);
    return 0;
}

const plt_kernel_t plt_kernel_conv_dense = {
    PLT_OP_CONV, PLT_EX_NNA_DENSE, "conv_dense", dense_plan, dense_run
};
