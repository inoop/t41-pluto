/* 1x1 int8 convolution on the NNA.  The recipe, and the reasoning behind every
 * field, is in docs/NNA_POINTWISE.md. */
#include <string.h>

#include "exec/plt_conv_pw.h"
#include "exec/plt_conv_nna.h"
#include "core/plt_layout.h"
#include "hal/plt_dev.h"
#include "hal/plt_nna.h"

#define IN_BITS   8
#define CSTEP     128                /* column-block step: 4 px x 32 ch x 8 b */

/* Words of the operand file this executor keeps live for the whole run.  The
 * array reads the file during every nnmac, so what is in it is part of the
 * machine state, not a passing argument.  Declared in the header, because an
 * executor that writes its own tile loop over this configuration needs them. */
enum {
    W_ZERO       = PLT_PW_W_ZERO,
    W_UNIT_A_OFF = PLT_PW_W_UNIT_A_OFF,
    W_UNIT_B_OFF = PLT_PW_W_UNIT_B_OFF,
    W_MAC_A      = PLT_PW_W_MAC_A,
    W_MAC_B      = PLT_PW_W_MAC_B
};

/* One execution unit's operand walk over the `n` input groups it owns.  Steps
 * are in units of in_bits; the last one rewinds to the group it started from. */
static void build_unit_prog(plt_nna_prog_t *prog, int n)
{
    plt_nna_prog_clear(prog);
    if (n < 2) {
        plt_nna_prog_push(prog, PLT_PROG_STEP(1));
        plt_nna_prog_push(prog, PLT_PROG_STEP(IN_BITS - 1 - IN_BITS * n));
        return;
    }
    plt_nna_prog_push(prog, PLT_PROG_LOAD(3, n - 2));
    plt_nna_prog_push(prog, PLT_PROG_STEP(1));
    plt_nna_prog_push(prog, PLT_PROG_STEP(IN_BITS - 1));
    plt_nna_prog_push(prog, PLT_PROG_STEP(1));
    plt_nna_prog_push(prog, PLT_PROG_STEP(IN_BITS - 1 - IN_BITS * n));
}

void plt_conv_pw_configure(const plt_conv_geom_t *s)
{
    const int d = s->cin / 32, na = d / 2, nb = d - na;
    plt_nna_cfg_t   cfg;
    plt_nna_prog_t  prog_a, prog_b;
    plt_nna_words_t words;

    plt_nna_reset();

    plt_nna_cfg_init(&cfg);
    /* A 1x1 layer has no window, so the array walks a flat W-wide raster: both
     * the height and the width field carry W. */
    cfg.in_h              = s->out_w;
    cfg.in_w              = s->out_w;
    cfg.mode_code         = 2;
    cfg.mode_code_b       = 2;
    cfg.mac_pitch         = 4;
    cfg.mac_span          = 8;
    cfg.out_elem_class    = plt_nna_out_elem(8);
    cfg.tile_class        = plt_nna_tile_class(IN_BITS);
    cfg.tap_mask          = plt_nna_tap_mask(1);
    cfg.operand_precision = plt_nna_prec(8, IN_BITS);
    cfg.in_elem_class     = plt_nna_in_elem(IN_BITS);
    cfg.col_origin        = 0;
    cfg.row_origin        = 0;
    cfg.parity_tag        = d & 1;
    cfg.groups_unit_a     = na;
    cfg.groups_unit_b     = nb;
    cfg.edge_mode         = 0x12;
    cfg.mode_flags        = 2;
    cfg.pack_format       = 0;
    cfg.mac_balance       = 0;
    plt_nna_cfg_apply(&cfg);

    /* Each unit walks from its own program: unit A's at entry 0, unit B's at 8. */
    build_unit_prog(&prog_a, na);
    build_unit_prog(&prog_b, nb);
    plt_nna_prog_load(&prog_a, &prog_b);

    plt_nna_words_clear(&words);
    words.w[W_ZERO]       = 0;
    words.w[W_UNIT_A_OFF] = 0;
    words.w[W_UNIT_B_OFF] = 2 + IN_BITS * na;
    words.w[W_MAC_A]      = 0;
    words.w[W_MAC_B]      = 4 * IN_BITS * na + 8;

    plt_nna_arm();
    plt_nna_words_commit(&words);
    /* Both operand start offsets go in after the arm. */
    PLT_NNA_FIELD(PLT_BANK_B, W_UNIT_A_OFF, PLT_B_UNITA_START);
    PLT_NNA_FIELD(PLT_BANK_B, W_UNIT_B_OFF, PLT_B_UNITB_START);
}

void plt_conv_pw_tiles(const plt_conv_geom_t *s, const plt_conv_bufs_t *b)
{
    const int d    = s->cin / 32,  na   = d / 2;
    const int dout = s->cout / 32;
    const int pw   = plt_pad_w(s->out_w), ph = plt_pad_h(s->out_h);
    const int icell = b->in_cell ? b->in_cell : 32, ocell = b->out_cell ? b->out_cell : 32;
    const int rowb = pw * icell, plane = ph * rowb;
    const int orowb = pw * ocell, oplane = ph * orowb;
    /* Aliased partner groups all read plane 0: see plt_conv_bufs_t. */
    const int gstride = b->in_alias_groups ? 0 : plane;
    const int cb_n = pw / 4,  rp_n  = ph / 2;
    const int gpp  = plt_conv_groups_per_pass(8, d);

    volatile uint8_t *ow = b->weights, *ot = b->table;
    volatile uint8_t *oi = b->in,      *oo = b->out;

    /* The drain runs one MAC pair behind, so a store always writes the tile the
     * PREVIOUS MAC pair produced.  These carry that tile's coordinates. */
    int prp = -1, pcb = -1, pg = -1;
    /* A folded activation is a plain byte table; copy it out of the model image
     * once so the per-tile pass reads it from cache. */
    const int has_lut = b->act_lut != NULL;
    if (has_lut) plt_conv_lut_load(b->act_lut);
#define STORE_PREV() do { if (pg >= 0) {                                        \
    volatile uint8_t *_t = oo + pg * oplane + (2 * prp) * orowb + pcb * 4 * ocell; \
    if (ocell == 16) plt_conv_drain_tile16(_t, (uint32_t)orowb, has_lut, 0);    \
    else PLT_CONV_DRAIN(_t, (uint32_t)orowb, has_lut);                          \
    } } while (0)

    for (int g0 = 0; g0 < dout; g0 += gpp) {
        const int gp = dout - g0 < gpp ? dout - g0 : gpp;

        PLT_NNA_FIELD(PLT_BANK_B, W_ZERO, PLT_B_WSTREAM);       /* B.1c <- 0 */
        for (int i = 0; i < gp * d * 4; i++)
            plt_nna_push_weight_slice(ow + (uint32_t)(g0 * d) * 1024 + i * 256);

        PLT_NNA_FIELD(PLT_BANK_B, W_ZERO, PLT_B_TABLE_PTR);     /* B.1e <- 0 */
        for (int j = 0; j < gp * 2; j++)
            plt_nna_push_table_half(ot + g0 * 256 + j * 128);

        for (int rp = 0; rp < rp_n; rp++) {
            volatile uint8_t *row0 = oi + (2 * rp) * rowb;
#define PW_FEED(g, port) do {                                                   \
    if (icell == 16)                                                          \
        PLT_NNA_FEED_TILE16(row0 + (g) * gstride + cb * 64, (uint32_t)rowb, (port)); \
    else                                                                      \
        PLT_NNA_FEED_TILE(row0 + (g) * gstride + cb * CSTEP, (uint32_t)rowb, (port)); \
} while (0)
            for (int cb = 0; cb < cb_n; cb++) {
                if (cb > 0) plt_nna_precommit();
                for (int g = 0; g < na; g++)
                    PW_FEED(g, PLT_PORT_ACT_A);
                if (cb > 0) plt_nna_pack();
                PLT_NNA_FIELD(PLT_BANK_A, W_ZERO, PLT_A_MAC_A1);
                PLT_NNA_FIELD(PLT_BANK_A, W_ZERO, PLT_A_MAC_A0);
                PLT_NNMAC(PLT_MAC_UNIT_A);
                if (b->in_alias_groups)
                    /* The partner groups' weights are zero: push whatever the
                     * registers hold rather than load plane 0 again.  Only the
                     * operand count is observable. */
                    for (int i = 0; i < (d - na) * 4; i++) PLT_NNDWR(0, PLT_PORT_ACT_B);
                else
                    for (int g = na; g < d; g++)
                        PW_FEED(g, PLT_PORT_ACT_B);
                PLT_NNMAC(PLT_MAC_UNIT_B);
                if (cb > 0) STORE_PREV();
                prp = rp; pcb = cb; pg = g0;
                PLT_NNA_FIELD(PLT_BANK_B, W_ZERO, PLT_B_COMMIT_WIN);

                /* The remaining output groups of this pass reuse the tile that
                 * is already fed: only the weights advance. */
                for (int g = 1; g < gp; g++) {
                    plt_nna_precommit(); plt_nna_pack();
                    PLT_NNMAC(PLT_MAC_UNIT_A); PLT_NNMAC(PLT_MAC_UNIT_B);
                    STORE_PREV();
                    prp = rp; pcb = cb; pg = g0 + g;
                }
            }
            /* Flush the pipeline at the end of the row pair. */
            plt_nna_precommit(); plt_nna_pack();
            STORE_PREV();
            pg = -1;
        }
    }
#undef STORE_PREV
#undef PW_FEED
}

/* Deliberately NOT here: feeding a multi-pass layer from a pair-interleaved
 * copy of its input, as plt_conv_k3.c does.  Measured, it paid only on YOLOX-S
 * layer 44 (8 passes over 410 KB, -0.4 ms) and cost on every MobileNetV1 layer
 * it applied to -- the planes here are too small for the copy to earn back. */

/* --- the executor ---------------------------------------------------------- */

/* Resolve the geometry and round it to what the array processes: output
 * channels up to whole 32-lane groups, and input channels likewise but never
 * fewer than two groups (the recipe splits them across two execution units).
 * `*alias` is set when the input is a single real group whose partner must read
 * the same plane -- the compiler zero-pads the weights to match, so the partner
 * contributes nothing.  The logical (unpadded) channel counts stay on the
 * tensors; only the array sees the padded ones. */
static int pw_prep(plt_ctx_t *ctx, const plt_node_t *node, plt_conv_geom_t *g, int *alias)
{
    if (plt_conv_geom_of(ctx, node, g) != 0) return -1;
    *alias = g->cin <= 32;
    g->cin  = *alias ? 64 : ((g->cin + 31) & ~31);
    g->cout = (g->cout + 31) & ~31;
    return 0;
}

static int pw_check(plt_ctx_t *ctx, const plt_conv_geom_t *g)
{
    if (g->kh != 1 || g->kw != 1)
        return plt_ctx_fail(ctx, "conv_pw: kernel %dx%d is not 1x1", g->kh, g->kw);
    if (g->stride != 1)
        return plt_ctx_fail(ctx, "conv_pw: stride %d is not 1", g->stride);
    if (g->in_bits != 8 || g->w_bits != 8 || g->out_bits != 8)
        return plt_ctx_fail(ctx, "conv_pw: %d/%d/%d bits, only 8/8/8 is implemented",
                            g->in_bits, g->w_bits, g->out_bits);
    return 0;
}

static int pw_plan(plt_ctx_t *ctx, plt_node_t *node, plt_plan_t *plan)
{
    plt_conv_geom_t g;
    int alias;

    if (pw_prep(ctx, node, &g, &alias) != 0) return -1;
    if (pw_check(ctx, &g) != 0) return -1;

    (void)alias;
    plan->scratch_bytes = 0;
    plan->out_pad_h     = plt_pad_h(g.out_h);
    plan->out_pad_w     = plt_pad_w(g.out_w);
    plan->out_bytes     = plt_ndhwc32_bytes(g.cout, g.out_h, g.out_w);
    return 0;
}

static int pw_run(plt_ctx_t *ctx, const plt_node_t *node)
{
    plt_conv_geom_t g;
    plt_conv_bufs_t b;
    int alias;

    if (pw_prep(ctx, node, &g, &alias) != 0) return -1;
    if (pw_check(ctx, &g) != 0) return -1;
    if (plt_conv_bufs_of(ctx, node, &b) != 0) return -1;

    /* The input is fed where it lies, also on a multi-pass layer that re-feeds
     * it once per pass.  It used to be copied into scratch first, on the theory
     * that the copy would be cheaper to re-read; measured without the copy,
     * YOLOX-S is 1.8 ms faster and MobileNetV1 0.6 ms -- on this CPU every byte
     * moved costs memory bandwidth, and the copy is one more pass of it. */
    uint32_t t = plt_prof_now(ctx);
    if (alias) b.in_alias_groups = 1;
    if (!b.act_lut && node->rec->act == PLT_ACT_SILU)
        b.act_lut = b.table + (uint32_t)(g.cout / 32) * 256u;
    t = plt_prof_mark(ctx, &ctx->prof.assemble_us, t);

    plt_conv_pw_configure(&g);
    plt_conv_pw_tiles(&g, &b);
    plt_prof_mark(ctx, &ctx->prof.array_us, t);
    return 0;
}

const plt_kernel_t plt_kernel_conv_pw = {
    PLT_OP_CONV, PLT_EX_NNA_PW, "conv_pw", pw_plan, pw_run
};
