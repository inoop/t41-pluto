/* A dense 3x3 convolution on the array's NATIVE window walk.
 *
 * Every other convolution here is lowered to a 1x1 over pseudo-channels: the
 * nine taps become nine input groups and the array runs its 1x1 recipe over
 * them (plt_conv_stem.c, plt_conv_dw.c, plt_conv_dense.c).  That cannot reach a
 * wide layer.  The walk program holds at most 16 input groups per execution
 * unit -- device-measured, see docs/NNA_POINTWISE.md -- and a lowered 3x3 costs
 * KH*KW*Cin/32 of them, so it stops at Cin = 96.  YOLOX-S is mostly 128 and 256.
 *
 * Here the array steps the taps itself (`A.1e` = 0x1ff, `A.19`/`A.17` = the MAC
 * pitch and span of a 3x3), so the walk costs D = Cin/32 groups and the same
 * limit allows Cin up to 1024.  The recipe is T41_NNA_MANUAL.md 11.2 with the
 * 8-bit-input substitutions of 11.5.
 *
 * WHAT THIS DOES NOT DO YET.  The manual's version streams activations into an
 * ORAM ring and drains through ORAM staging, both driven by NNDMA, so that the
 * DMA of one row pair overlaps the arithmetic of the next.  None of that is
 * needed to get the right answer: the array never READS the activation store,
 * the CPU pushes every 64-byte chunk to it with `nndwr`.  So the feed here
 * comes from an ordinary cached bordered copy (plt_conv_border_copy, as the
 * depthwise and dense paths already use) and the drain goes straight to the
 * output tensor.  Same instructions to the array, same answer, no DMA.  The
 * ring is a prefetch device and belongs in a later pass, measured rather than
 * assumed.
 */
/* This file alone builds at -O2 (the tree is -O1).  Measured: the 3x3 tile loop
 * is 2.5 ms faster per YOLOX-S frame at -O2, while the depthwise and pointwise
 * executors and the tail layers come out slightly slower, so the whole tree
 * does not follow. */
#pragma GCC optimize("O2")

#include <string.h>

#include "exec/plt_conv_k3.h"
#include "exec/plt_conv_nna.h"
#include "exec/plt_conv_pw.h"
#include "exec/plt_conv_stem.h"

#include "core/plt_layout.h"
#include "hal/plt_dev.h"
#include "hal/plt_nna.h"

/* Words of the operand file this recipe keeps live.  w2 is the zero every
 * per-tile field write reads; w4 is the tap-slot step; w9 and w10 are the two
 * MAC phases' operand offsets, and are the only ones that move. */
enum { K3_W_ZERO = 2, K3_W_TAPS = 4, K3_W_MAC_A = 9, K3_W_MAC_B = 10 };

typedef struct {
    int s, K, walk, lead;        /* stride and the walk constants of 11.2.2 */
    int D, Dout, nA, nB;
    int CB, RP;                  /* column blocks, row pairs               */
    int cpr;                     /* 64-byte chunks per window row per group */
    int rowu;                    /* row operand units, in_bits*K*walk/8     */
    int b11;                     /* B.11 = nA * rowu                        */
    int tap_slots;               /* A.12/A.14 = D*9*w_bits/2                */
    int gpp, passes;             /* output groups per weight pass           */
    int in_bits;
} k3_geom_t;

static void k3_geom(const plt_conv_geom_t *g, k3_geom_t *k)
{
    k->in_bits = g->in_bits;
    k->s    = g->stride;
    k->K    = k->s == 1 ? 0x14 : 0x18;
    k->walk = k->s == 1 ? 4 : 5;
    k->lead = k->s == 1 ? 2 : 1;
    k->D    = g->cin / 32;
    k->Dout = g->cout / 32;
    k->nA   = k->D / 2;
    k->nB   = k->D - k->nA;
    k->CB   = plt_pad_w(g->out_w) / 4;
    k->RP   = plt_pad_h(g->out_h) / 2;
    /* A window row of one group is 4*s pixels; a chunk is 64 B, so 32 B per
     * pixel at 8-bit input gives 2*s chunks (manual 11.5's "column-block step
     * within a row" doubling). */
    k->cpr  = k->s * g->in_bits / 4;
    k->rowu = g->in_bits * k->K * k->walk / 8;
    k->b11  = k->nA * k->rowu;
    k->tap_slots = k->D * 9 * g->w_bits / 2;
    k->gpp  = PLT_NNA_TAP_SLOTS / k->tap_slots;
    if (k->gpp < 1) k->gpp = 1;
    if (k->gpp > k->Dout) k->gpp = k->Dout;
    k->passes = (k->Dout + k->gpp - 1) / k->gpp;
}

static int k3_check(plt_ctx_t *ctx, const plt_conv_geom_t *g)
{
    if (g->kh != 3 || g->kw != 3)
        return plt_ctx_fail(ctx, "conv_k3: kernel %dx%d is not 3x3", g->kh, g->kw);
    if (g->pad_t != 1 || g->pad_l != 1)
        return plt_ctx_fail(ctx, "conv_k3: pad %d,%d; the recipe is written for 1,1",
                            g->pad_t, g->pad_l);
    if (g->stride != 1 && g->stride != 2)
        return plt_ctx_fail(ctx, "conv_k3: stride %d is not 1 or 2", g->stride);
    if (g->cin % 32 || g->cin < 64)
        return plt_ctx_fail(ctx, "conv_k3: %d input channels; want a multiple of 32, "
                            "at least two groups", g->cin);
    if (g->cin / 32 > 32)
        return plt_ctx_fail(ctx, "conv_k3: %d input groups exceeds the 32 the walk "
                            "program holds", g->cin / 32);
    if (g->in_bits != 8 || g->w_bits != 8 || g->out_bits != 8)
        return plt_ctx_fail(ctx, "conv_k3: %d/%d/%d bits, only 8/8/8 is implemented",
                            g->in_bits, g->w_bits, g->out_bits);
    return 0;
}

/* --- the walk program (manual 11.2.4 step 2) -------------------------------
 *
 * Four entries per unit, for `n` groups on that unit:
 *
 *   loop(n)                                   0x7B00 | (n-1)
 *   step(walk-1, K/4)
 *   step(1, rowu - (K*walk/4 - K/4))
 *   move(s - n*rowu)
 *
 * Checked against the manual's own words: stride 1, two groups per unit gives
 * 0x7B01 0x1805 0x0819 0x03B1, and stride 2 gives the row step 0x824 and the
 * move 0x38A -- both reproduced exactly by these formulas at in_bits = 4. */
static void k3_unit_prog(plt_nna_prog_t *p, const k3_geom_t *k, int n)
{
    /* The step operands are K/4 and K*walk/4 whatever the input width -- only
     * `rowu` (and so B.11 and the final move) scales with in_bits.  Confirmed
     * against the working implementation in yolox-t41/src/nna/conv_i4_3x3.c:
     * `step_operand = mac_pitch/4`, `row_program_units = in_bits*mac_span/8`. */
    const int kq = k->K / 4, kwq = k->K * k->walk / 4;
    plt_nna_prog_clear(p);
    plt_nna_prog_push(p, PLT_PROG_LOAD(3, n - 1));
    plt_nna_prog_push(p, PLT_PROG_STEP2(k->walk - 1, kq));
    plt_nna_prog_push(p, PLT_PROG_STEP2(1, k->rowu - (kwq - kq)));
    plt_nna_prog_push(p, PLT_PROG_MOVE(k->s - n * k->rowu));
}

static void k3_configure(const plt_conv_geom_t *g, const k3_geom_t *k)
{
    plt_nna_cfg_t  cfg;
    plt_nna_prog_t pa, pb;

    plt_nna_reset();

    plt_nna_cfg_init(&cfg);
    cfg.parity_tag        = 3;
    cfg.in_elem_class     = plt_nna_in_elem(g->in_bits);
    cfg.mode_code         = 2;
    cfg.mode_code_b       = 2;
    cfg.in_h              = g->in_h;              /* written as H_in - 1 */
    cfg.in_w              = g->in_w;              /* written as W_in - 1 */
    cfg.pad_value         = (128 + g->in_zp) & 0xff;   /* the fed byte of u = 0 */
    cfg.mac_pitch         = k->K;
    cfg.mac_span          = k->K * k->walk;
    cfg.edge_mode         = 0x12;
    cfg.stride_code       = plt_nna_stride_code(k->s);
    cfg.operand_precision = plt_nna_prec(g->w_bits, g->in_bits);
    cfg.groups_unit_a     = k->nA;                /* written as nA - 1 */
    cfg.groups_unit_b     = k->nB;                /* written as nB - 1 */
    cfg.pack_format       = 0;
    cfg.out_elem_class    = plt_nna_out_elem(g->out_bits);
    cfg.mode_flags        = 2;
    cfg.readout_lag       = plt_nna_readout_lag(g->out_bits);
    cfg.lane_perm_row0    = plt_nna_lane_perm_row0(g->out_bits);
    cfg.lane_perm_row1    = plt_nna_lane_perm_row1(g->out_bits);
    cfg.unit_b_start      = k->b11;
    cfg.tile_class        = plt_nna_tile_class(g->in_bits);
    cfg.mac_balance       = k->s - 1;
    /* A.1e keeps its reset value 0x1ff -- every tap of a 3x3 enabled -- and
     * B.10 keeps its reset 0 (manual 15.1), so neither is written. */
    plt_nna_cfg_apply(&cfg);

    k3_unit_prog(&pa, k, k->nA);
    k3_unit_prog(&pb, k, k->nB);
    plt_nna_prog_load(&pa, &pb);

    plt_nna_arm();
}

/* --- the activation feed ---------------------------------------------------
 *
 * One unit's share of a column block is a flat list of 64-byte chunks:
 * group-major, the `walk` window rows next, the `cpr` chunks of a row innermost
 * (manual 11.2.5).  The routine pushes a prefix of that list after every MAC,
 * so the whole block has been fed by the time the block's tiles are done.
 *
 * Chunk i of any block sits at the same offset from the block's first chunk, so
 * the list is a table built once per layer and a run of pushes is an indexed
 * walk: `base + off[i]`.  (It used to be a cursor carried chunk to chunk, whose
 * column/row/group wraps mispredicted a branch every 2-5 pushes; before that,
 * three integer divisions per push.)
 */
typedef struct {
    const volatile uint8_t *plane;   /* bordered input, group 0, row -1/col -1;
                                        or, when `direct`, the input tensor     */
    uint32_t bplane, brow;
    int g0, n;                       /* this unit's first group and count      */
    int walk, cpr, s;
    int total;                       /* n * walk * cpr                         */
    int cur;                         /* chunks already pushed for this block   */

    /* Direct feed: read the input tensor itself, no bordered copy.  A block
     * whose window lies wholly inside the image is walked in place; a block
     * that touches the padding reads a copy gathered once per layer, into
     * `edge`, with exactly the bytes the bordered copy would have held there --
     * so the two paths feed identical operands.  Once per LAYER, not per
     * block visit: a multi-pass layer re-reads every block per pass. */
    int direct, in_h, in_w, CB;
    uint8_t pad;
    uint8_t *edge;                   /* edge block j at edge + j * total * 64   */
    const int16_t *edge_of;          /* block rp*CB+cb -> edge ordinal, or -1   */

    int cell;                        /* bytes a pixel: 32, or 16 (NDHWC16 input) */
    int32_t edge_offs[16 * 5 * 4];   /* chunk i of a gathered edge block        */
    const volatile uint8_t *base;    /* this block's chunk 0                    */
    /* A multi-pass layer's input, re-laid out pair-interleaved: row, then pixel
     * pair, then group, so group g of pixels (2p, 2p+1) of row y is the 64
     * bytes at y*prow + p*pst + g*64.  See k3_pair_bytes. */
    const volatile uint8_t *pairs;
    uint32_t prow, pst;
    const int32_t *off;              /* offs, or edge_offs for an edge block    */
    int32_t offs[16 * 5 * 4];
} k3_feed_t;

static void k3_feed_init(k3_feed_t *f, const k3_geom_t *k, const volatile uint8_t *plane,
                         uint32_t bplane, uint32_t brow, int g0, int n, int cell)
{
    f->cell = cell;
    f->plane = plane; f->bplane = bplane; f->brow = brow;
    f->g0 = g0; f->n = n;
    f->walk = k->walk; f->cpr = k->cpr; f->s = k->s;
    f->total = n * k->walk * k->cpr;
    f->cur = 0;
    f->direct = 0;
    /* A chunk is two pixels of 32 lanes, so columns step by 64 bytes, window
     * rows by a buffer row and groups by a plane. */
    int i = 0;
    for (int gi = 0; gi < n; gi++)
        for (int wr = 0; wr < k->walk; wr++)
            for (int c = 0; c < k->cpr; c++, i++) {
                f->offs[i] = (int32_t)((uint32_t)gi * bplane + (uint32_t)wr * brow
                                       + (uint32_t)c * 2u * (uint32_t)cell);
                f->edge_offs[i] = i * 2 * cell;
            }
    f->base = plane;
    f->off = f->offs;
    f->pairs = NULL;
}

static void k3_feed_pairs(k3_feed_t *f, const volatile uint8_t *pairs, uint32_t prow, uint32_t pst)
{
    f->pairs = pairs; f->prow = prow; f->pst = pst;
    int i = 0;
    for (int gi = 0; gi < f->n; gi++)
        for (int wr = 0; wr < f->walk; wr++)
            for (int c = 0; c < f->cpr; c++, i++)
                f->offs[i] = (int32_t)((uint32_t)wr * prow + (uint32_t)c * pst + (uint32_t)gi * 64u);
}

/* Does block (rp, cb)'s window reach outside the image?  Shared by the plan,
 * which sizes the edge store, and the run, which fills and reads it. */
static int k3_is_edge(const k3_geom_t *k, const plt_conv_geom_t *g, int rp, int cb)
{
    const int y0 = 2 * k->s * rp - 1, x0 = 4 * k->s * cb;
    return !(y0 >= 0 && y0 + k->walk <= g->in_h && x0 + 2 * k->cpr <= g->in_w);
}

static int k3_edge_blocks(const k3_geom_t *k, const plt_conv_geom_t *g)
{
    int n = 0;
    for (int rp = 0; rp < k->RP; rp++)
        for (int cb = 0; cb < k->CB; cb++) n += k3_is_edge(k, g, rp, cb);
    return n;
}

/* Gather one edge block of one unit, pixel by pixel, with the bordered copy's
 * rule: a pixel inside the image is the tensor's 32 bytes, anything else is the
 * pad byte. */
static void k3_feed_gather(const k3_feed_t *f, uint8_t *d, int rp, int cb)
{
    const int y0 = 2 * f->s * rp - 1, x0 = 4 * f->s * cb;
    const uint32_t padw = 0x01010101u * f->pad;
    for (int gi = 0; gi < f->n; gi++) {
        const uint8_t *pl = (const uint8_t *)(uintptr_t)f->plane
                          + (uint32_t)(f->g0 + gi) * f->bplane;
        for (int wr = 0; wr < f->walk; wr++) {
            const int y = y0 + wr;
            for (int x = x0; x < x0 + 2 * f->cpr; x++, d += f->cell) {
                if (y < 0 || y >= f->in_h || x < 0 || x >= f->in_w) {
                    for (int i = 0; i < f->cell; i += 4) memcpy(d + i, &padw, 4);
                } else {
                    memcpy(d, pl + (uint32_t)y * f->brow + (uint32_t)x * (uint32_t)f->cell,
                           (size_t)f->cell);
                }
            }
        }
    }
}

/* Point the unit at the block (rp, cb).  Input row `2s*rp - 1 + wr` and input
 * column `4s*cb + 2*c` become buffer row `2s*rp + wr` and buffer column
 * `4s*cb + 2*c + 1` of a bordered copy: it starts one row above and one column
 * left of the image, so the -1 of the padding cancels. */
static void k3_feed_at(k3_feed_t *f, int rp, int cb)
{
    f->cur = 0;
    if (f->direct) {
        const int j = f->edge_of[rp * f->CB + cb];
        if (j < 0) {
            if (f->pairs)
                f->base = f->pairs + (uint32_t)(2 * f->s * rp - 1) * f->prow
                        + (uint32_t)(2 * f->s * cb) * f->pst + (uint32_t)f->g0 * 64u;
            else
                f->base = f->plane + (uint32_t)f->g0 * f->bplane
                        + (uint32_t)(2 * f->s * rp - 1) * f->brow
                        + (uint32_t)(4 * f->s * cb) * (uint32_t)f->cell;
            f->off = f->offs;
        } else {
            f->base = f->edge + (uint32_t)j * (uint32_t)f->total * 2u * (uint32_t)f->cell;
            f->off = f->edge_offs;
        }
        return;
    }
    f->base = f->plane
            + (uint32_t)f->g0 * f->bplane
            + (uint32_t)(2 * f->s * rp) * f->brow
            + (uint32_t)(4 * f->s * cb + 1) * 32u;
    f->off = f->offs;
}

/* Push a run of chunks to one unit's activation port.
 *
 * The pushes rotate vr0..vr3 so that no push's load can overwrite a register an
 * earlier push has not yet read -- PLT_NNA_FEED_TILE's rule (plt_isa_nna.h),
 * applied to a run whose length is not known until it starts.  PLT_VLD64 takes
 * a plain pointer; the feed buffer is volatile only to stop the compiler caching
 * it across the pushes. */
#ifdef PLT_BENCH_K3_FIXSRC
static const uint8_t k3_bench_src[64] __attribute__((aligned(64)));
#endif

#define K3_NOCONV(vr)
/* The RGB graph input's byte conversion (plt_conv_border_copy_xor), done in
 * the push register instead of in a copy of the whole image: vr21 holds 64
 * bytes of 0x80 for the layer. */
#define K3_XOR80(vr) __asm__ __volatile__(".set push\n.set noreorder\n"        \
        PLT_M3_OP(PLT_M3_XORV, vr, 21, vr) ".set pop\n" ::: "memory");

#ifdef PLT_BENCH_K3_FIXSRC
#define K3_AT(f, i) ((void)(f), (void)(i), k3_bench_src)
#else
#define K3_AT(f, i) ((const uint8_t *)(uintptr_t)((f)->base + (f)->off[(i)]))
#endif
/* An NDHWC16 input's chunk is two 16-byte pixels: they go into quad lanes 0
 * and 2 of the push register, which are exactly the lanes that pixel's first 16
 * channels occupy in a 32-lane cell.  Lanes 1 and 3 keep whatever they held --
 * they feed channels 16-31, whose weights are zero (the input has at most 16). */
#define K3_LOAD16(vr, p) do {                                                 \
    register const uint8_t *_q __asm__("t0") = (p);                           \
    __asm__ __volatile__(".set push\n.set noreorder\n"                        \
        PLT_M3_LAQ(vr, 0, 8, 0) PLT_M3_LAQ(vr, 2, 8, 1) ".set pop\n"          \
        :: "r"(_q) : "memory");                                               \
} while (0)

#define K3_FEED_RUN_L(f, count, port, CONV, LOAD) do {                        \
    int _i = (f)->cur, _e = _i + (count);                                     \
    if (_e > (f)->total) _e = (f)->total;                                     \
    if (_i < _e) {                                                            \
        LOAD(0, K3_AT(f, _i)); CONV(0) PLT_NNDWR(0, (port)); _i++;            \
        while (_i + 4 <= _e) {                                                \
            LOAD(1, K3_AT(f, _i));     CONV(1) PLT_NNDWR(1, (port));          \
            LOAD(2, K3_AT(f, _i + 1)); CONV(2) PLT_NNDWR(2, (port));          \
            LOAD(3, K3_AT(f, _i + 2)); CONV(3) PLT_NNDWR(3, (port));          \
            LOAD(0, K3_AT(f, _i + 3)); CONV(0) PLT_NNDWR(0, (port));          \
            _i += 4;                                                          \
        }                                                                     \
        if (_i < _e) { LOAD(1, K3_AT(f, _i)); CONV(1) PLT_NNDWR(1, (port)); _i++; } \
        if (_i < _e) { LOAD(2, K3_AT(f, _i)); CONV(2) PLT_NNDWR(2, (port)); _i++; } \
        if (_i < _e) { LOAD(3, K3_AT(f, _i)); CONV(3) PLT_NNDWR(3, (port)); _i++; } \
        (f)->cur = _i;                                                        \
    }                                                                         \
} while (0)
#define K3_FEED_RUN(f, count, port, CONV) K3_FEED_RUN_L(f, count, port, CONV, PLT_VLD64)

/* An aliased partner unit's feed.  Its weights are all zero (see k3_prep), so
 * what it is fed cannot reach the output -- only HOW MANY operands it is fed
 * matters, because the feed is positional.  So it re-pushes whatever vr0 holds
 * instead of loading the plane a second time: the same instructions to the
 * array, half the memory traffic of an aliased layer. */
static void k3_feed_hollow(k3_feed_t *f, int count, int port_b)
{
    int n = count;
    if (n > f->total - f->cur) n = f->total - f->cur;
    f->cur += n > 0 ? n : 0;
    if (port_b) for (int i = 0; i < n; i++) PLT_NNDWR(0, PLT_PORT_ACT_B);
    else        for (int i = 0; i < n; i++) PLT_NNDWR(0, PLT_PORT_ACT_A);
}

static void k3_feed_a(k3_feed_t *f, int count) { K3_FEED_RUN(f, count, PLT_PORT_ACT_A, K3_NOCONV); }
static void k3_feed_b(k3_feed_t *f, int count) { K3_FEED_RUN(f, count, PLT_PORT_ACT_B, K3_NOCONV); }
static void k3_feed_a_x(k3_feed_t *f, int count) { K3_FEED_RUN(f, count, PLT_PORT_ACT_A, K3_XOR80); }
static void k3_feed_b_x(k3_feed_t *f, int count) { K3_FEED_RUN(f, count, PLT_PORT_ACT_B, K3_XOR80); }
static void k3_feed_a16(k3_feed_t *f, int count) { K3_FEED_RUN_L(f, count, PLT_PORT_ACT_A, K3_NOCONV, K3_LOAD16); }

/* The 3x3's packed drain at 8-bit output.
 *
 * Four readout registers, but NOT in the order plt_nna_drain_tile_fifo() uses
 * for the 1x1: here they alternate the tile's two ROWS first and its two column
 * halves second --
 *
 *     vr10 row 0, px 0-1     vr12 row 0, px 2-3
 *     vr11 row 1, px 0-1     vr13 row 1, px 2-3
 *
 * -- which is device-measured, not assumed: with the 1x1 order every tile came
 * out with its second half holding the next row's first half.  The manual's
 * 10 "vr10+vr11 to row r, vr12+vr13 to row r+1" describes the 1x1's packing. */
static inline void k3_drain_tile(volatile uint8_t *dst, uint32_t row_bytes)
{
    uint8_t *p = (uint8_t *)(uintptr_t)dst;
    PLT_DRAIN_FIFO(10); PLT_DRAIN_FIFO(11); PLT_DRAIN_FIFO(12); PLT_DRAIN_FIFO(13);
    PLT_VST64(10, p +                  0);
    PLT_VST64(11, p + row_bytes +      0);
    PLT_VST64(12, p +                 64);
    PLT_VST64(13, p + row_bytes +     64);
}

static inline void k3_drain_tile_lut(volatile uint8_t *dst, uint32_t row_bytes)
{
    uint8_t *p = (uint8_t *)(uintptr_t)dst;
    PLT_DRAIN_FIFO(10); PLT_DRAIN_FIFO(11); PLT_DRAIN_FIFO(12); PLT_DRAIN_FIFO(13);
    PLT_CONV_LUT_REG(10); PLT_CONV_LUT_REG(11); PLT_CONV_LUT_REG(12); PLT_CONV_LUT_REG(13);
    PLT_VST64(10, p +                  0);
    PLT_VST64(11, p + row_bytes +      0);
    PLT_VST64(12, p +                 64);
    PLT_VST64(13, p + row_bytes +     64);
}

/* Bytes the direct feed's edge store needs: an ordinal table over the blocks,
 * then every edge block's chunks for unit A and for unit B. */
static uint32_t k3_edge_bytes(const k3_geom_t *k, const plt_conv_geom_t *g)
{
    const uint32_t per = (uint32_t)(k->nA + k->nB) * (uint32_t)(k->walk * k->cpr) * 64u;
    return (uint32_t)(k->RP * k->CB) * sizeof(int16_t) + 64u
         + (uint32_t)k3_edge_blocks(k, g) * per;
}

/* Switch both cursors to the direct feed and fill the edge store. */
static void k3_feed_direct(k3_feed_t *fa, k3_feed_t *fb, const k3_geom_t *k,
                           const plt_conv_geom_t *g, uint8_t *store, uint8_t xor_mask)
{
    int16_t *edge_of = (int16_t *)(void *)store;
    uint8_t *data = store + (uint32_t)(k->RP * k->CB) * sizeof(int16_t);
    data += (64u - ((uintptr_t)data & 63u)) & 63u;
    const uint32_t bytes_a = (uint32_t)fa->total * 2u * (uint32_t)fa->cell;
    const uint32_t bytes_b = (uint32_t)fb->total * 2u * (uint32_t)fb->cell;
    int j = 0;

    for (int u = 0; u < 2; u++) {
        k3_feed_t *f = u ? fb : fa;
        f->direct = 1;
        f->in_h = g->in_h; f->in_w = g->in_w; f->CB = k->CB;
        /* Every push of an XOR feed is converted in the register, the edge
         * store's pads included -- so the pad goes in pre-converted. */
        f->pad = (uint8_t)((128 + g->in_zp) ^ xor_mask);
        f->edge_of = edge_of;
        f->edge = data;
    }
    const uint32_t nedge = (uint32_t)k3_edge_blocks(k, g);
    fb->edge = data + nedge * bytes_a;
    for (int rp = 0; rp < k->RP; rp++)
        for (int cb = 0; cb < k->CB; cb++) {
            if (!k3_is_edge(k, g, rp, cb)) { edge_of[rp * k->CB + cb] = -1; continue; }
            edge_of[rp * k->CB + cb] = (int16_t)j;
            k3_feed_gather(fa, fa->edge + (uint32_t)j * bytes_a, rp, cb);
            k3_feed_gather(fb, fb->edge + (uint32_t)j * bytes_b, rp, cb);
            j++;
        }
}

/* --- the tile loop (manual 11.2.5) ---------------------------------------- */

static void k3_tiles(const plt_conv_geom_t *g, const k3_geom_t *k,
                     const plt_conv_bufs_t *b, const volatile uint8_t *plane,
                     uint32_t bplane, uint32_t brow, uint8_t *edge_store, uint8_t xor_mask,
                     const volatile uint8_t *pairs, int cell)
{
#define feed_a(f, n) do { if (cell == 16) k3_feed_a16((f), (n));                     \
                          else if (xor_mask) k3_feed_a_x((f), (n)); else k3_feed_a((f), (n)); } while (0)
#define feed_b(f, n) do { if (hollow_b) k3_feed_hollow((f), (n), 1);                   \
                          else if (xor_mask) k3_feed_b_x((f), (n)); else k3_feed_b((f), (n)); } while (0)
    /* bplane 0 means both units read one plane: the partner is padding. */
    const int hollow_b = bplane == 0;
    if (xor_mask) {
        static const uint8_t top[64] __attribute__((aligned(64))) = { [0 ... 63] = 0x80 };
        PLT_VLD64(21, top);
    }
    const int ocell = b->out_cell ? b->out_cell : 32;
    const uint32_t orow   = (uint32_t)plt_pad_w(g->out_w) * (uint32_t)ocell;
    const uint32_t oplane = (uint32_t)plt_pad_h(g->out_h) * orow;
    const int blocks = k->CB;
    const int nblocks = k->RP * k->passes * k->CB;
    plt_nna_words_t w;
    k3_feed_t fa, fb;
    const int has_lut = b->act_lut != NULL;
    if (has_lut) plt_conv_lut_load(b->act_lut);

    /* The drain returns the tile committed one tile earlier, so a store always
     * writes the PREVIOUS tile's address; the first drain of the run is
     * discarded and one extra drain follows the loop (manual 11.2.5). */
    uint8_t sink[256];
    volatile uint8_t *prev = NULL;

    k3_feed_init(&fa, k, plane, bplane, brow, 0, k->nA, cell);
    k3_feed_init(&fb, k, plane, bplane, brow, k->nA, k->nB, cell);
    if (pairs) {
        const uint32_t pst = (uint32_t)k->D * 64u, prow = (uint32_t)(plt_pad_w(g->in_w) / 2) * pst;
        k3_feed_pairs(&fa, pairs, prow, pst);
        k3_feed_pairs(&fb, pairs, prow, pst);
    }
    if (edge_store) k3_feed_direct(&fa, &fb, k, g, edge_store, xor_mask);

    plt_nna_words_clear(&w);
    w.w[K3_W_ZERO]  = 0;
    w.w[K3_W_TAPS]  = (uint32_t)k->tap_slots;
    w.w[K3_W_MAC_A] = (uint32_t)(-1);
    w.w[K3_W_MAC_B] = (uint32_t)(-1 + 4 * k->b11);
    plt_nna_words_commit(&w);

    /* The requant table goes in once for the whole layer. */
    PLT_NNA_FIELD(PLT_BANK_B, K3_W_ZERO, PLT_B_TABLE_PTR);
    for (int j = 0; j < k->Dout * 2; j++)
        plt_nna_push_table_half(b->table + (uint32_t)j * 128);

#ifdef PLT_BENCH_SINK
#define K3_DRAIN_DST(p, r) ((void)(p), (void)(r), plt_bench_sink)
#define K3_DRAIN_ROW(r) 256
#else
#define K3_DRAIN_DST(p, r) (p)
#define K3_DRAIN_ROW(r) (r)
#endif
#define K3_STORE_PREV() do { if (prev) {                                      \
    if (ocell == 16) plt_conv_drain_tile16(prev, orow, has_lut, 1);           \
    else if (has_lut) k3_drain_tile_lut(K3_DRAIN_DST(prev, orow), K3_DRAIN_ROW(orow)); \
    else         k3_drain_tile(K3_DRAIN_DST(prev, orow), K3_DRAIN_ROW(orow));     \
} else { k3_drain_tile(sink, 128); } } while (0)

    /* The feed is one continuous stream of column blocks in (pass, row pair,
     * block) order, kept exactly `lead` blocks ahead of the MACs -- so at the
     * end of a row pair it is already feeding the next one, and at a pass
     * boundary it starts the whole feature map again for the next group of
     * output channels.  Block `f` of that stream is decoded back to the
     * (row pair, column block) whose rows it reads. */
#define K3_FED_RP(f) (((f) / k->CB) % k->RP)
#define K3_FED_CB(f) ((f) % k->CB)

    /* Pre-feed (manual 11.2.4 step 9): the first `lead` blocks go in before any
     * MAC, unit B's whole list first, then unit A's. */
    for (int f = 0; f < k->lead; f++) {
        const int frp = K3_FED_RP(f), fcb = K3_FED_CB(f);
        k3_feed_at(&fb, frp, fcb); feed_b(&fb, fb.total);
        k3_feed_at(&fa, frp, fcb); feed_a(&fa, fa.total);
    }

    /* Pass outermost, so a pass's weights are pushed ONCE.  They used to go in
     * inside the row-pair loop, which re-sent the whole stream RP times: 103 MB
     * of weight pushes per YOLOX-S frame for a 6.1 MB blob, more traffic than
     * the activations themselves.  Nothing about the array wanted that -- the
     * tap ring already survives the CB blocks of a row pair untouched, and this
     * only extends the same run to RP*CB blocks.  (plt_conv_pw.c and
     * plt_conv_dense.c always had the loops this way round.) */
    int t = 0;                       /* the column block being consumed */
    /* The fed block's (row pair, column block), advanced by counting rather
     * than decoded from `t + lead` with a divide and two modulos per block --
     * MIPS `div` is tens of cycles and unpipelined, and this runs RP*CB times
     * a pass. */
    int frp_n = K3_FED_RP(k->lead < nblocks ? k->lead : nblocks - 1);
    int fcb_n = K3_FED_CB(k->lead < nblocks ? k->lead : nblocks - 1);
    for (int pass = 0; pass < k->passes; pass++) {
        const int g0 = pass * k->gpp;
        const int gk = k->Dout - g0 < k->gpp ? k->Dout - g0 : k->gpp;
        const int per_tile_a = ((fa.total + gk - 1) / gk + 1) & ~1;
        const int per_tile_b = ((fb.total + gk - 1) / gk + 1) & ~1;

        /* Weights of this pass: the blob slice of (out-group, in-group) pairs
         * [g0*D, (g0+gk)*D), nine taps each, four 256-byte slices per tap.
         * Pushed from the model image, which is cached. */
        PLT_NNA_FIELD(PLT_BANK_B, K3_W_ZERO, PLT_B_WSTREAM);
        for (int i = 0; i < gk * k->D * 9 * 4; i++)
            plt_nna_push_weight_slice(b->weights
                + (uint32_t)(g0 * k->D) * 9u * 1024u + (uint32_t)i * 256u);

        for (int rp = 0; rp < k->RP; rp++) {
            /* Window origin for this row pair: column -1, row 2s*rp - 1. */
            w.w[0] = (uint32_t)(-1);
            w.w[1] = (uint32_t)(2 * k->s * rp - 1);
            plt_nna_words_commit(&w);
            PLT_NNA_FIELD(PLT_BANK_A, 0, PLT_A_PAD_ORG);
            PLT_NNA_FIELD(PLT_BANK_A, 1, PLT_A_ROW_ORG);

            for (int cb = 0; cb < blocks; cb++, t++) {
                /* The block this one's tiles feed, `lead` ahead in the stream.
                 * Past the end the feed keeps counting -- the array's operand
                 * positions have to advance whether or not anything reads them
                 * -- so the last block's rows are re-read rather than skipped. */
                const int frp = frp_n, fcb = fcb_n;
                if (t + 1 + k->lead < nblocks && ++fcb_n == k->CB) {
                    fcb_n = 0;
                    if (++frp_n == k->RP) frp_n = 0;
                }

                k3_feed_at(&fa, frp, fcb); k3_feed_at(&fb, frp, fcb);

                /* B.02, the commit window, steps by 4*gpp per pass. */
                w.w[3] = (uint32_t)(4 * k->gpp * pass);
                /* The two MAC words are set once and advance by 4*s per column
                 * block for the WHOLE layer -- `t`, not `cb`.  They do not
                 * restart at a row pair: the operand file is a continuous ring
                 * and the walk position runs through it.  (Restarting them made
                 * row pair 0 exact and every later one wrong.) */
                w.w[K3_W_MAC_A] = (uint32_t)(-1 + 4 * k->s * t);
                w.w[K3_W_MAC_B] = (uint32_t)(-1 + 4 * k->b11 + 4 * k->s * t);
                plt_nna_words_commit(&w);
                PLT_NNA_FIELD(PLT_BANK_B, 3, PLT_B_COMMIT_WIN);

                for (int p = 0; p < gk; p++) {
                    PLT_NNMAC(PLT_MAC_K3_A);
                    feed_b(&fb, per_tile_b);
                    PLT_NNMAC(PLT_MAC_K3_B);
                    feed_a(&fa, per_tile_a);

                    K3_STORE_PREV();
                    prev = b->out + (uint32_t)(g0 + p) * oplane
                         + (uint32_t)(2 * rp) * orow + (uint32_t)cb * 4u * (uint32_t)ocell;
                    PLT_NNCMD(PLT_CMD_K3_COMMIT);

                    PLT_NNA_FIELD(PLT_BANK_A, K3_W_TAPS, PLT_A_TAPSTEP2);
                    PLT_NNA_FIELD(PLT_BANK_A, K3_W_TAPS, PLT_A_TAPSTEP);
                }
                PLT_NNA_FIELD(PLT_BANK_A, K3_W_ZERO, PLT_A_MAC_A0);
                PLT_NNA_FIELD(PLT_BANK_A, K3_W_ZERO, PLT_A_MAC_A1);
                PLT_NNCMD(PLT_CMD_END_BLOCK);
            }
        }
    }
    K3_STORE_PREV();          /* the last tile */
#undef feed_a
#undef feed_b
#undef K3_STORE_PREV
#undef K3_FED_RP
#undef K3_FED_CB
}

/* --- the executor ---------------------------------------------------------- */

/* Round the geometry to what the array processes.  The walk splits the input
 * groups across two execution units, so a layer with one group or less is padded
 * to two and `*alias` is set: the partner group reads group 0's plane and the
 * compiler zero-padded its weights to match, so it contributes nothing.  This is
 * plt_conv_pw.c's pw_prep, for the same reason and with the same contract --
 * which is what lets the stem's 3x3s (Cin 3, 12 and 32) run here instead of
 * being lowered into pseudo-channels and assembled in DRAM. */
static int k3_prep(plt_ctx_t *ctx, const plt_node_t *node, plt_conv_geom_t *g,
                   int *alias, uint8_t *xor_mask)
{
    if (plt_conv_geom_of(ctx, node, g) != 0) return -1;
    /* An RGB graph input carries the two's-complement byte of q, not q + 128;
     * see plt_conv_border_copy_xor().  Decided on the REAL channel count, before
     * the padding below hides it. */
    if (xor_mask) *xor_mask = g->cin <= 3 ? PLT_STEM_FEED_XOR : 0x00;
    *alias  = g->cin <= 32;
    g->cin  = *alias ? 64 : ((g->cin + 31) & ~31);
    g->cout = (g->cout + 31) & ~31;
    return k3_check(ctx, g);
}

/* The feed reads the input tensor in place (k3_feed_direct), converting an RGB
 * graph input's bytes in the push register (K3_XOR80).  Building with
 * -DPLT_K3_BORDER_COPY restores the original bordered copy instead: the
 * reference path the device checksums (tools/deploy/bench.py --make-ref) were
 * recorded from, kept so they can be re-recorded. */
#ifdef PLT_K3_BORDER_COPY
#define K3_DIRECT_OK(xor_mask) ((void)(xor_mask), 0)
#else
#define K3_DIRECT_OK(xor_mask) ((void)(xor_mask), 1)
#endif

/* Bytes of the pair-interleaved copy a layer feeds from, or 0 for none.
 *
 * In NDHWC32 a block's chunks come from D planes x `walk` rows: 40 interleaved
 * streams for YOLOX-S layer 38, re-walked once per weight pass (16 of them)
 * over a tensor that lives in L2, at ~39 ns of stall a push.  Interleaved by
 * pixel pair the same chunks are `walk` sequential streams.  The copy is one
 * extra pass over the input, so it pays only for layers that re-walk it: from 3
 * passes, measured -1.9 ms of array time per YOLOX-S frame net of the copy (L38
 * -1.15 ms, L85 -0.4, L24 -0.2, L48/L98 -0.1 each).  An aliased layer has one
 * pass by construction. */
#define K3_PAIR_MIN_PASSES 3
static uint32_t k3_pair_bytes(const plt_conv_geom_t *g, const k3_geom_t *k, int alias)
{
    if (!alias && k->passes >= K3_PAIR_MIN_PASSES)
        return (uint32_t)g->in_h * (uint32_t)(plt_pad_w(g->in_w) / 2) * (uint32_t)k->D * 64u + 64u;
    return 0;
}

static uint32_t k3_scratch(const plt_conv_geom_t *g, int alias, uint8_t xor_mask)
{
    if (K3_DIRECT_OK(xor_mask)) {
        k3_geom_t k;
        k3_geom(g, &k);
        return k3_edge_bytes(&k, g) + k3_pair_bytes(g, &k, alias);
    }
    return (uint32_t)(alias ? 1 : g->cin / 32) * plt_conv_border_bytes(g);
}

static int k3_plan(plt_ctx_t *ctx, plt_node_t *node, plt_plan_t *plan)
{
    plt_conv_geom_t g;
    int alias;
    uint8_t xor_mask;
    if (k3_prep(ctx, node, &g, &alias, &xor_mask) != 0) return -1;
    plan->out_pad_h     = plt_pad_h(g.out_h);
    plan->out_pad_w     = plt_pad_w(g.out_w);
    plan->out_bytes     = plt_ndhwc32_bytes(g.cout, g.out_h, g.out_w);
    plan->scratch_bytes = k3_scratch(&g, alias, xor_mask);
    return 0;
}

static int k3_run(plt_ctx_t *ctx, const plt_node_t *node)
{
    plt_conv_geom_t g;
    plt_conv_bufs_t b;
    k3_geom_t k;
    int alias;
    uint8_t xor_mask;

    if (k3_prep(ctx, node, &g, &alias, &xor_mask) != 0) return -1;
    if (plt_conv_bufs_of(ctx, node, &b) != 0) return -1;
    k3_geom(&g, &k);

    if (b.act_lut == NULL && node->rec->act == PLT_ACT_SILU)
        b.act_lut = b.table + (uint32_t)(g.cout / 32) * 256u;

    uint32_t t = plt_prof_now(ctx);
    if (K3_DIRECT_OK(xor_mask)) {
        const uint32_t need = k3_scratch(&g, alias, xor_mask);
        uint8_t *store = plt_ctx_scratch(ctx, need);
        if (!store)
            return plt_ctx_fail(ctx, "conv_k3: no scratch holds the %u-byte edge store", need);
        const volatile uint8_t *pairs = NULL;
        if (k3_pair_bytes(&g, &k, alias)) {
            uint8_t *conv;
            conv = store + k3_edge_bytes(&k, &g);
            conv += (64u - ((uintptr_t)conv & 63u)) & 63u;
            pairs = conv;
            const uint32_t P = (uint32_t)(plt_pad_w(g.in_w) / 2);
            const uint32_t in_row = plt_ndhwc32_row_bytes(g.in_w);
            const uint32_t in_plane = plt_ndhwc32_plane_bytes(g.in_h, g.in_w);
            plt_conv_to_pairs(conv, b.in, k.D, g.in_h, (int)P, in_row, in_plane);
            t = plt_prof_mark(ctx, &ctx->prof.assemble_us, t);
        }
        k3_configure(&g, &k);
        /* An NDHWC16 input (a focus layer's, see plt_engine_load): 16-byte
         * cells, necessarily one aliased group. */
        const int cell = node->in[0].layout == PLT_FMT_NDHWC16 ? 16 : 32;
        if (cell == 16 && (!alias || pairs || xor_mask))
            return plt_ctx_fail(ctx, "conv_k3: an NDHWC16 input must be one plain group");
        k3_tiles(&g, &k, &b, b.in,
                 alias ? 0u : plt_ndhwc32_plane_bytes(g.in_h, g.in_w),
                 cell == 16 ? (uint32_t)plt_pad_w(g.in_w) * 16u : plt_ndhwc32_row_bytes(g.in_w),
                 store, xor_mask, pairs, cell);
        plt_prof_mark(ctx, &ctx->prof.array_us, t);
        return 0;
    }

    const uint32_t need = k3_scratch(&g, alias, xor_mask);
    volatile uint8_t *col = plt_ctx_scratch(ctx, need);
    if (!col)
        return plt_ctx_fail(ctx, "conv_k3: no scratch holds the %u bytes the bordered "
                            "input needs", need);
    {
        const uint32_t bplane   = plt_conv_border_bytes(&g);
        const uint32_t in_plane = plt_ndhwc32_plane_bytes(g.in_h, g.in_w);
        const int      nplanes  = alias ? 1 : g.cin / 32;
        for (int gi = 0; gi < nplanes; gi++)
            plt_conv_border_copy_xor(&g, b.in + (uint32_t)gi * in_plane,
                                     col + (uint32_t)gi * bplane, xor_mask);
    }
    t = plt_prof_mark(ctx, &ctx->prof.assemble_us, t);

    k3_configure(&g, &k);
    k3_tiles(&g, &k, &b, col, alias ? 0u : plt_conv_border_bytes(&g),
             plt_conv_border_row(&g), NULL, 0, NULL, 32);
    plt_prof_mark(ctx, &ctx->prof.array_us, t);
    return 0;
}

const plt_kernel_t plt_kernel_conv_k3 = {
    PLT_OP_CONV, PLT_EX_NNA_K3, "conv_k3", k3_plan, k3_run
};
