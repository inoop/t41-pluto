#include "exec/plt_conv_stem.h"
#include "exec/plt_conv_nna.h"
#include "exec/plt_conv_pw.h"
#include <stdlib.h>

#include "core/plt_layout.h"
#include "hal/plt_dev.h"
#include "hal/plt_nna.h"
#include "hal/plt_mxu3.h"

/* Move one 16-byte tap per step: `n` steps, source advancing by `sstep`, the
 * tap tensor by one cell (32).  This is the whole inner loop of the gather, so
 * it is written as one register-resident block -- a call per tap-pixel would
 * cost more in setup than the move itself. */
static inline void m3_gather16(uint8_t *d, const uint8_t *s, uint32_t sstep, uint32_t n)
{
    if (!n) return;
    __asm__ __volatile__(
        ".set push\n\t.set noreorder\n\t.set noat\n\t"
        "1:\n\t"
        "move $t0, %[s]\n\t" PLT_M3_LUQ(8, 0)
        "move $t0, %[d]\n\t" PLT_M3_SUQ(8, 0)
        "addu  %[s], %[s], %[ss]\n\t"
        "addiu %[d], %[d], 32\n\t"
        "addiu %[n], %[n], -1\n\t"
        "bnez  %[n], 1b\n\t"
        "nop\n\t"
        ".set pop\n\t"
        : [s] "+r"(s), [d] "+r"(d), [n] "+r"(n)
        : [ss] "r"(sstep) : "$t0", "memory");
}

/* A general convolution (>1 tap per position, up to 32 input channels) run on
 * the NNA as a 1x1 over assembled tap channels -- the recipe in
 * compile/layout.py:stem_im2col.  A KH x KW window over Cin channels is
 * Cin*KH*KW values, gathered into pseudo-channels and fed to the ordinary 1x1
 * engine (manual 8.6).
 *
 *   MobileNetV1's stem     3 -> 32, 3x3  ->  27 taps, one 32-lane group
 *   YOLOX-Nano's post-Focus 12 -> 16, 3x3 -> 108 taps, four groups
 *
 * The tap order is pluto's own, `t = (ky*KW + kx)*Cin + c`, and the packer,
 * this executor and the simulator all agree on it (compile/layout.py).
 *
 * Byte convention.  The array reads the assembled window byte UNSIGNED and the
 * requant folds `feed_offset = 128 + x_zp` out of the bias, so the fed byte
 * must be `b = q + 128` for a real tap and `128 + x_zp` for a border tap (the
 * zero-point in the fed domain).  A feature-map input already carries `b`, so
 * it is copied straight through; the RGB image is packed as the two's-
 * complement byte of `q`, which `^ 0x80` turns into `b`.  For the image
 * x_zp = 0, so its border value `128 + x_zp` is exactly 0x80 = PLT_STEM_FEED_XOR
 * -- the two cases meet.
 */

#define STEM_MAX_CIN 32     /* one input group: rows[] read plane 0 only */

/* Pseudo-channels this window assembles to, matching stem_pseudo_channels():
 * Cin*KH*KW rounded up to whole 32-lane groups, but never fewer than two (the
 * 1x1 recipe splits the input groups across the array's two execution units). */
/* Pseudo-channels per tap; must match compile/layout.py:stem_tap_stride.
 * A multi-group window pads each tap to 16 so the gather moves it as one
 * aligned vector load/store; those lanes carry zero weights. */
static int stem_stride(const plt_conv_geom_t *g)
{
    return g->cin * g->kh * g->kw <= 32 ? g->cin : (g->cin + 15) / 16 * 16;
}

static int stem_pseudo(const plt_conv_geom_t *g)
{
    const int taps = g->kh * g->kw * stem_stride(g);
    const int p = (taps + 31) & ~31;
    return p < 64 ? 64 : p;
}

static int stem_check(plt_ctx_t *ctx, const plt_conv_geom_t *g)
{
    if (g->cin > STEM_MAX_CIN)
        return plt_ctx_fail(ctx, "conv_stem: %d input channels; only up to %d fit in "
                            "one assembled group", g->cin, STEM_MAX_CIN);
    if (g->kh > 15)
        return plt_ctx_fail(ctx, "conv_stem: kernel height %d exceeds the staging cap", g->kh);
    if (g->in_bits != 8 || g->w_bits != 8 || g->out_bits != 8)
        return plt_ctx_fail(ctx, "conv_stem: %d/%d/%d bits, only 8/8/8 is implemented",
                            g->in_bits, g->w_bits, g->out_bits);
    return 0;
}

/* The geometry of the 1x1 convolution the assembled window is fed to.  Output
 * channels are rounded to a whole group (the compiler zero-pads to match), and
 * the input is the full set of pseudo-channels. */
static void as_pointwise(const plt_conv_geom_t *g, plt_conv_geom_t *pw)
{
    *pw = *g;
    pw->cin    = stem_pseudo(g);
    pw->cout   = (g->cout + 31) & ~31;
    pw->kh     = pw->kw = 1;
    pw->stride = 1;
    pw->pad_t  = pw->pad_l = 0;
    pw->in_h   = g->out_h;
    pw->in_w   = g->out_w;
}

/* Assemble the window: NDHWC32 image -> NDHWC32 tap planes.
 *
 * This is the path for a window whose taps are PACKED -- MobileNetV1's 3-channel
 * stem, whose 3-byte taps sit at lanes 0, 3, 6 ... and so cannot be moved a
 * vector lane at a time.  A 16-byte tap slot goes through stem_tiles_fused()
 * instead and never reaches memory.
 *
 * A pixel's taps come from KH*KW cells spread over KH input rows.  The input
 * rows are bounced through cached memory (`rows`) so the gather reads them
 * sequentially; the tap tensor `dst` is written directly, since it is cached
 * too.  im2col already replicates every input value KH*KW times, so the one
 * thing to avoid on top of that is touching bytes the array never reads: only
 * output rows [0,out_h), columns [0,out_w) and pseudo-channels [0,Cin*KH*KW)
 * matter -- padding rows/columns feed dropped outputs and the tail
 * pseudo-channels meet zero weights.  So each output pixel writes every one of
 * its KH*KW taps exactly once (a copy in range, the border `pad` out of range),
 * with no separate pre-fill and no staging bounce.
 *
 * The loop is tap-outer / column-inner: a tap's input row, its validity, and
 * its destination plane/lane/run split are all fixed for the whole output row,
 * so they hoist out of the column loop.  A tap's Cin channels are contiguous in
 * the source cell and land in one (or, across a plane boundary, two) contiguous
 * runs -- copied whole, not scattered byte by byte.
 */
static void im2col(const plt_conv_geom_t *g, uint8_t xor_mask, uint8_t pad,
                   const volatile uint8_t *src, volatile uint8_t *dst,
                   uint8_t *const rows[])
{
    const uint32_t in_row    = plt_ndhwc32_row_bytes(g->in_w);
    const uint32_t out_row   = plt_ndhwc32_row_bytes(g->out_w);
    const uint32_t out_plane = plt_ndhwc32_plane_bytes(g->out_h, g->out_w);
    const int kh = g->kh, kw = g->kw, cin = g->cin;
    const int stride = stem_stride(g);
    uint8_t *pd = (uint8_t *)dst;

    for (int oy = 0; oy < g->out_h; oy++) {
        for (int ky = 0; ky < kh; ky++) {
            const int iy = oy * g->stride + ky - g->pad_t;
            if (iy >= 0 && iy < g->in_h)
                plt_copy_fast(rows[ky], src + iy * in_row, in_row);
        }
        for (int ky = 0; ky < kh; ky++) {
            const int iy = oy * g->stride + ky - g->pad_t;
            const int vy = iy >= 0 && iy < g->in_h;
            const uint8_t *srow = rows[ky];
            for (int kx = 0; kx < kw; kx++) {
                const int base = (ky * kw + kx) * stride; /* first pseudo-channel */
                const int lane = base & 31;
                const int run0 = 32 - lane < cin ? 32 - lane : cin;
                const int run1 = cin - run0;             /* >0 iff the run crosses a plane */
                uint8_t *d0 = pd + (uint32_t)(base >> 5) * out_plane + oy * out_row + lane;
                uint8_t *d1 = pd + (uint32_t)((base >> 5) + 1) * out_plane + oy * out_row;
                /* Aligned tap, feature-map input: the valid columns form one
                 * run whose source and destination both advance by a fixed
                 * stride, so the whole run is one vector loop. */
                if (vy && xor_mask == 0 && (lane & 15) == 0 && run1 == 0 && cin <= 16) {
                    int lo = g->pad_l - kx, hi = g->in_w + g->pad_l - kx;
                    if (g->stride == 1) {
                        lo = lo < 0 ? 0 : lo;
                        hi = hi > g->out_w ? g->out_w : hi;
                        for (int ox = 0; ox < lo; ox++)
                            __builtin_memset(d0 + ox * 32, pad, (size_t)run0);
                        if (hi > lo)
                            m3_gather16(d0 + lo * 32,
                                        srow + (lo + kx - g->pad_l) * 32, 32,
                                        (uint32_t)(hi - lo));
                        for (int ox = hi; ox < g->out_w; ox++)
                            __builtin_memset(d0 + ox * 32, pad, (size_t)run0);
                        continue;
                    }
                }
                for (int ox = 0; ox < g->out_w; ox++) {
                    const int ix = ox * g->stride + kx - g->pad_l;
                    uint8_t *e0 = d0 + ox * 32, *e1 = d1 + ox * 32;
                    if (vy && ix >= 0 && ix < g->in_w) {
                        const uint8_t *px = srow + ix * 32;
                        if (xor_mask == 0) {
                            __builtin_memcpy(e0, px, (size_t)run0);
                            if (run1) __builtin_memcpy(e1, px + run0, (size_t)run1);
                        } else {
                            for (int j = 0; j < run0; j++) e0[j] = px[j] ^ xor_mask;
                            for (int j = 0; j < run1; j++) e1[j] = px[run0 + j] ^ xor_mask;
                        }
                    } else {
                        __builtin_memset(e0, pad, (size_t)run0);
                        if (run1) __builtin_memset(e1, pad, (size_t)run1);
                    }
                }
            }
        }
    }
}

/* --- the window assembled in the push register ------------------------------
 *
 * The tap tensor above exists only to hand the array bytes it could have been
 * handed directly.  The array is fed from a VECTOR REGISTER (nndwr pushes
 * register contents, not an address), and that register is addressable as four
 * 16-byte lanes -- PLT_VLD64 is already two `lao` lane loads of 32 bytes each.
 * A tap slot here is exactly 16 bytes, so one tap of one pixel is one lane, and
 * a 64-byte push is four `laq`s from four unrelated addresses:
 *
 *     laq vr0[0]  pixel ox,   tap 2g        laq vr0[2]  pixel ox+1, tap 2g
 *     laq vr0[1]  pixel ox,   tap 2g+1      laq vr0[3]  pixel ox+1, tap 2g+1
 *     nndwr vr0, port
 *
 * so the im2col never reaches memory.  Measured on the device: a lane-assembled
 * push costs 5.2 ns against 3.6 ns for a contiguous one, against ~280 ns per
 * push-equivalent of gather it replaces.
 *
 * All of it is static.  The gate below fixes a 3x3 stride-1 window with a
 * 16-byte tap slot, which is 9 taps in 5 input groups, so every lane's kernel
 * row and column -- and therefore every base register and every offset -- is a
 * compile-time constant.  The only thing the loop does is walk four row
 * pointers along the plane.
 *
 * The plane is bordered first (plt_conv_border_copy), which is what makes every
 * address valid: the -1s of the padding cancel against the one-cell margin, so
 * there is no bounds test anywhere in the feed.
 */

/* One lane of a push: output row `h` of the tile, column pair `jj`, pixel `e`
 * of that pair, kernel tap (ky,kx).
 *
 * Row base register  = $t0 + (h + ky)   -- the four row pointers, see below.
 * Offset, 16B units  = 2 * (2jj + e + kx), because an NDHWC32 cell is 32 bytes.
 *
 * Everything here is a literal the ASSEMBLER evaluates, so it may contain only
 * +, * and the bit operators -- no %, / or ?:, which is why the kernel taps are
 * written out at the call site rather than derived from a group number. */
#define ST_LANE(lane, h, ky, kx, jj, e) \
    PLT_M3_LAQ(0, lane, 8 + (h) + (ky), 2 * (2 * (jj) + (e) + (kx)))

/* One 64-byte push: two pixels, each contributing two taps of 16 lanes. */
#define ST_PUSH(kya, kxa, kyb, kxb, h, jj)      \
    ST_LANE(0, h, kya, kxa, jj, 0)              \
    ST_LANE(1, h, kyb, kxb, jj, 0)              \
    ST_LANE(2, h, kya, kxa, jj, 1)              \
    ST_LANE(3, h, kyb, kxb, jj, 1)              \
    ".word %[nn]\n\t"

/* The four pushes of one tile, in the order PLT_NNA_FEED_TILE uses: row 0
 * columns 0-1 and 2-3, then row 1.  One `sync` for the group, which is the
 * runtime's feed convention (hal/plt_isa_nna.h).
 *
 * The four row pointers are pinned to $t0..$t3 and rebuilt for every tile: a
 * local register variable is only guaranteed at the asm that names it, and
 * there is a drain call in the loop. */
#define ST_FEED_TILE(kya, kxa, kyb, kxb, rowp, brow, port) do {                \
    register const uint8_t *_r0 __asm__("t0") = (rowp);                        \
    register const uint8_t *_r1 __asm__("t1") = (rowp) + (brow);               \
    register const uint8_t *_r2 __asm__("t2") = (rowp) + 2 * (brow);           \
    register const uint8_t *_r3 __asm__("t3") = (rowp) + 3 * (brow);           \
    __asm__ __volatile__(".set push\n\t.set noreorder\n\t"                   \
        ST_PUSH(kya, kxa, kyb, kxb, 0, 0) ST_PUSH(kya, kxa, kyb, kxb, 0, 1)    \
        ST_PUSH(kya, kxa, kyb, kxb, 1, 0) ST_PUSH(kya, kxa, kyb, kxb, 1, 1)    \
        ".set pop\n\t"                                                         \
        :: [nn] "i"(PLT_NN_WORD(2, ((port) & 0x1Fu) | (((port) >> 5) << 10))), \
           "r"(_r0), "r"(_r1), "r"(_r2), "r"(_r3) : "memory");                 \
} while (0)

/* plt_conv_pw_tiles() with the feed replaced.  Written out rather than
 * parameterised, for the same reason dw_tiles_stride1() is: the feed line is
 * the whole point, and everything around it -- the MAC pair, the one-tile-
 * behind drain, the folded activation -- is deliberately identical. */
static void stem_tiles_fused(const plt_conv_geom_t *pw, const plt_conv_bufs_t *b,
                             const uint8_t *plane, uint32_t brow)
{
    const int ppw = plt_pad_w(pw->out_w), pph = plt_pad_h(pw->out_h);
    const uint32_t orow = plt_ndhwc32_row_bytes(pw->out_w);
    int prp = -1, pcb = -1, live = 0;
    const int has_lut = b->act_lut != NULL;
    if (has_lut) plt_conv_lut_load(b->act_lut);

#define ST_STORE_PREV() do { if (live) {                                       \
    volatile uint8_t *_t = b->out + (2 * prp) * orow + pcb * 128;              \
    PLT_CONV_DRAIN(_t, orow, has_lut);                                         \
    } } while (0)

    /* One output group, one weight pass: 5 input groups x 4 slices. */
    PLT_NNA_FIELD(PLT_BANK_B, PLT_PW_W_ZERO, PLT_B_WSTREAM);
    for (int i = 0; i < (pw->cin / 32) * 4; i++)
        plt_nna_push_weight_slice(b->weights + (uint32_t)i * 256);
    PLT_NNA_FIELD(PLT_BANK_B, PLT_PW_W_ZERO, PLT_B_TABLE_PTR);
    for (int j = 0; j < 2; j++)
        plt_nna_push_table_half(b->table + (uint32_t)j * 128);

    for (int rp = 0; rp < pph / 2; rp++) {
        /* Output rows 2rp and 2rp+1 read input rows 2rp-1 .. 2rp+2, which the
         * one-cell margin shifts to buffer rows 2rp .. 2rp+3. */
        const uint8_t *row0 = plane + (uint32_t)(2 * rp) * brow;
        for (int cb = 0; cb < ppw / 4; cb++) {
            const uint8_t *tile = row0 + (uint32_t)(4 * cb) * 32;
            if (cb > 0) plt_nna_precommit();
            /* group g carries taps 2g, 2g+1 with t = ky*3 + kx; group 4's
             * second slot has no tap and meets zero weights, so it repeats
             * tap 8 rather than reading a fifth row. */
            ST_FEED_TILE(0,0, 0,1, tile, brow, PLT_PORT_ACT_A);   /* taps 0,1 */
            ST_FEED_TILE(0,2, 1,0, tile, brow, PLT_PORT_ACT_A);   /* taps 2,3 */
            if (cb > 0) plt_nna_pack();
            PLT_NNA_FIELD(PLT_BANK_A, PLT_PW_W_ZERO, PLT_A_MAC_A1);
            PLT_NNA_FIELD(PLT_BANK_A, PLT_PW_W_ZERO, PLT_A_MAC_A0);
            PLT_NNMAC(PLT_MAC_UNIT_A);
            ST_FEED_TILE(1,1, 1,2, tile, brow, PLT_PORT_ACT_B);   /* taps 4,5 */
            ST_FEED_TILE(2,0, 2,1, tile, brow, PLT_PORT_ACT_B);   /* taps 6,7 */
            ST_FEED_TILE(2,2, 2,2, tile, brow, PLT_PORT_ACT_B);   /* tap  8   */
            PLT_NNMAC(PLT_MAC_UNIT_B);
            if (cb > 0) ST_STORE_PREV();
            prp = rp; pcb = cb; live = 1;
            PLT_NNA_FIELD(PLT_BANK_B, PLT_PW_W_ZERO, PLT_B_COMMIT_WIN);
        }
        plt_nna_precommit(); plt_nna_pack();
        ST_STORE_PREV();
        live = 0;
    }
#undef ST_STORE_PREV
}

/* Can the window be assembled in the push register instead of in memory?
 *
 * Needs a tap slot of exactly one 16-byte vector lane, and a 3x3 stride-1
 * window with one-pixel padding -- which is what makes every base register and
 * offset in ST_PUSH a constant, and what the group->tap mapping there assumes.
 * MobileNetV1's stem is deliberately excluded: its 3-byte taps sit at lanes
 * 0, 3, 6 ... and are not lane-addressable, so it keeps the memory im2col. */
static int stem_fused_ok(const plt_conv_geom_t *g)
{
    return g->kh == 3 && g->kw == 3 && g->stride == 1
        && g->pad_t == 1 && g->pad_l == 1
        && g->in_h == g->out_h && g->in_w == g->out_w
        && stem_stride(g) == 16 && stem_pseudo(g) == 160;
}

static int stem_plan(plt_ctx_t *ctx, plt_node_t *node, plt_plan_t *plan)
{
    plt_conv_geom_t g;

    if (plt_conv_geom_of(ctx, node, &g) != 0) return -1;
    if (stem_check(ctx, &g) != 0) return -1;

    const int taps    = g.kh * g.kw * stem_stride(&g);
    const int nplanes = (taps + 31) / 32;
    (void)nplanes;

    plan->out_pad_h     = plt_pad_h(g.out_h);
    plan->out_pad_w     = plt_pad_w(g.out_w);
    plan->out_bytes     = plt_ndhwc32_bytes(g.cout, g.out_h, g.out_w);
    /* Fused: one bordered copy of the input plane, the window assembled in the
     * push register.  Otherwise the assembled window, `nplanes` planes -- when
     * there is only one, a second partner group is needed by the recipe but
     * aliases this plane (see below), so a single plane is materialized. */
    plan->scratch_bytes = stem_fused_ok(&g)
        ? plt_conv_border_bytes(&g)
        : plt_ndhwc32_bytes(nplanes * 32, g.out_h, g.out_w);
    return 0;
}

static int stem_run(plt_ctx_t *ctx, const plt_node_t *node)
{
    plt_conv_geom_t g, pw;
    plt_conv_bufs_t b;

    if (plt_conv_geom_of(ctx, node, &g) != 0) return -1;
    if (stem_check(ctx, &g) != 0) return -1;
    if (plt_conv_bufs_of(ctx, node, &b) != 0) return -1;

    const int taps    = g.kh * g.kw * stem_stride(&g);
    const int nplanes = (taps + 31) / 32;
    const int dpw     = stem_pseudo(&g) / 32;
    const int alias   = nplanes < dpw;          /* one real plane, padded partner */
    const int fused   = stem_fused_ok(&g);

    const uint32_t need = fused ? plt_conv_border_bytes(&g)
                                : plt_ndhwc32_bytes(nplanes * 32, g.out_h, g.out_w);
    volatile uint8_t *col = plt_ctx_scratch(ctx, need);
    if (!col)
        return plt_ctx_fail(ctx, "conv_stem: no scratch holds the %u bytes the window "
                            "needs", need);

    /* Real feature-map bytes already carry b = q+128; the RGB image (Cin<=3) is
     * the two's-complement byte of q and needs ^0x80.  Border taps take the
     * zero point in the fed domain, 128 + x_zp (0x80 for the zp=0 image). */
    const uint8_t xor_mask = g.cin <= 3 ? PLT_STEM_FEED_XOR : 0x00;
    const uint8_t pad      = (uint8_t)(128 + g.in_zp);

    uint32_t t = plt_prof_now(ctx);
    if (fused) {
        /* One bordered copy of the input plane; the window itself is assembled
         * a lane at a time in the push register and never reaches memory. */
        plt_conv_border_copy(&g, b.in, col);
    } else {
        /* Cached staging: the KH input rows the gather reads sequentially. */
        const uint32_t in_row = plt_ndhwc32_row_bytes(g.in_w);
        uint8_t *pool = malloc((size_t)g.kh * in_row);
        if (!pool) return plt_ctx_fail(ctx, "conv_stem: out of memory for staging");
        uint8_t *rows[16];
        for (int ky = 0; ky < g.kh && ky < 16; ky++) rows[ky] = pool + (size_t)ky * in_row;
        im2col(&g, xor_mask, pad, b.in, col, rows);
        free(pool);
        b.in = col;
        b.in_alias_groups = alias;
    }
    t = plt_prof_mark(ctx, &ctx->prof.assemble_us, t);

    as_pointwise(&g, &pw);
    if (!b.act_lut && node->rec->act == PLT_ACT_SILU)
        b.act_lut = b.table + (uint32_t)(pw.cout / 32) * 256u;
    plt_conv_pw_configure(&pw);
    if (fused)
        stem_tiles_fused(&pw, &b, (const uint8_t *)col, plt_conv_border_row(&g));
    else
        plt_conv_pw_tiles(&pw, &b);
    plt_prof_mark(ctx, &ctx->prof.array_us, t);
    return 0;
}

const plt_kernel_t plt_kernel_conv_stem = {
    PLT_OP_CONV, PLT_EX_NNA_STD, "conv_stem", stem_plan, stem_run
};
