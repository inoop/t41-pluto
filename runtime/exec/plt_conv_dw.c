#include "exec/plt_conv_dw.h"
#include "hal/plt_nna.h"
#include "exec/plt_conv_nna.h"
#include "exec/plt_conv_pw.h"
#include <stdlib.h>
#include <string.h>

#include "core/plt_layout.h"
#include "hal/plt_dev.h"
#include "hal/plt_mxu3.h"

static int dw_check(plt_ctx_t *ctx, const plt_conv_geom_t *g)
{
    if (g->kh != 3 || g->kw != 3)
        return plt_ctx_fail(ctx, "conv_dw: kernel %dx%d is not 3x3", g->kh, g->kw);
    if (g->cin != g->cout)
        return plt_ctx_fail(ctx, "conv_dw: %d in, %d out; depthwise keeps its channels",
                            g->cin, g->cout);
    if (g->cin % 32)
        return plt_ctx_fail(ctx, "conv_dw: %d channels are not a multiple of 32", g->cin);
    if (g->in_bits != 8 || g->w_bits != 8 || g->out_bits != 8)
        return plt_ctx_fail(ctx, "conv_dw: %d/%d/%d bits, only 8/8/8 is implemented",
                            g->in_bits, g->w_bits, g->out_bits);
    return 0;
}

/* The pointwise convolution one channel group turns into. */
static void as_pointwise(const plt_conv_geom_t *g, plt_conv_geom_t *pw)
{
    *pw = *g;
    pw->cin    = g->kh * g->kw * 32;      /* 288 taps */
    pw->cout   = 32;
    pw->kh     = pw->kw = 1;
    pw->stride = 1;
    pw->pad_t  = pw->pad_l = 0;
    pw->in_h   = g->out_h;
    pw->in_w   = g->out_w;
}

/* Assemble one channel group's taps: tap t of channel ci becomes pseudo-channel
 * t*32 + ci at the output resolution.  Out-of-range taps take the zero-point-
 * removed zero, which is the byte 0.
 *
 * This is now only the fallback for a strided window whose padding is not the
 * usual one pixel: stride 1 reads the input directly (dw_tiles_stride1) and a
 * padded stride-2 window gathers into the push register (dw_tiles_stride2), so
 * neither materializes anything.  Kept because it is the one path that makes no
 * assumption about the geometry at all.
 *
 * The loop runs (output row, kernel row, kernel column) rather than the obvious
 * (kernel row, kernel column, output row): the three kernel columns of one
 * kernel row all read the SAME input row, so ordering it this way stages that
 * row from uncached memory once instead of three times.
 */
static void tap_im2col(const plt_conv_geom_t *g, const volatile uint8_t *src,
                       volatile uint8_t *dst, uint8_t *in_stage, uint8_t *out_stage)
{
    const uint32_t in_row    = plt_ndhwc32_row_bytes(g->in_w);
    const uint32_t out_row   = plt_ndhwc32_row_bytes(g->out_w);
    const uint32_t out_plane = plt_ndhwc32_plane_bytes(g->out_h, g->out_w);
    const int pw = plt_pad_w(g->out_w), ph = plt_pad_h(g->out_h);
    int lo[3], hi[3];
    const uint8_t pad = (uint8_t)(128 + g->in_zp);  /* border feeds u=0 */

    /* Per kernel column, the output columns whose source column is inside the
     * image: ix = ox*stride + kx - pad_l must land in [0, in_w). */
    for (int kx = 0; kx < g->kw; kx++) {
        int a = 0, b = g->out_w;
        while (a < b && a * g->stride + kx - g->pad_l < 0) a++;
        while (b > a && (b - 1) * g->stride + kx - g->pad_l >= g->in_w) b--;
        lo[kx] = a; hi[kx] = b;
    }

    for (int oy = 0; oy < ph; oy++)
        for (int ky = 0; ky < g->kh; ky++) {
            const int iy   = oy * g->stride + ky - g->pad_t;
            const int have = oy < g->out_h && iy >= 0 && iy < g->in_h;

            if (have) plt_copy_fast(in_stage, src + iy * in_row, in_row);

            for (int kx = 0; kx < g->kw; kx++) {
                volatile uint8_t *row = dst + (ky * g->kw + kx) * out_plane + oy * out_row;
                if (!have) { plt_fill_fast(row, pad, (size_t)pw * 32); continue; }

                plt_fill_fast(row, pad, (size_t)lo[kx] * 32);
                plt_fill_fast(row + (uint32_t)hi[kx] * 32, pad, (size_t)(pw - hi[kx]) * 32);
                for (int ox = lo[kx]; ox < hi[kx]; ox++)
                    memcpy(out_stage + (uint32_t)ox * 32,
                           in_stage + (uint32_t)(ox * g->stride + kx - g->pad_l) * 32, 32);
                plt_copy_fast(row + (uint32_t)lo[kx] * 32,
                              out_stage + (uint32_t)lo[kx] * 32,
                              (size_t)(hi[kx] - lo[kx]) * 32);
            }
        }
}

/* Stride 1 needs no tap tensor at all.
 *
 * The nine tap planes are nine SHIFTED VIEWS of the same input plane: tap
 * (ky,kx) at output pixel (oy,ox) reads input (oy+ky-1, ox+kx-1).  At stride 1
 * those are still four consecutive pixels, so the array can read them straight
 * out of the input -- the tap expansion becomes nine feed offsets rather than
 * nine planes the CPU has to build.  A 32-byte-aligned activation window is
 * enough for the array (device-checked), which is what makes the +-1 column
 * shift legal.
 *
 * The one thing it needs is a border: the walk reads rows -1..H and columns
 * -1..W+2, so the plane is copied once into a buffer with a zero margin.  One
 * plane copy per channel group instead of nine.
 */
/* The bordered plane this needs is shared with the fused stem feed:
 * plt_conv_border_copy() in plt_conv_nna.c. */

/* The tile loop, with the tap offsets folded into the feed address.  This is
 * plt_conv_pw_tiles() with one line changed -- written out rather than
 * parameterised, because that one line is the whole point. */
static void dw_tiles_stride1(const plt_conv_geom_t *g, const plt_conv_geom_t *pw,
                             const plt_conv_bufs_t *b, const volatile uint8_t *in, int kw,
                             int bordered, int icell, int ocell)
{
    /* The input is read in place: tap g of block (rp, cb) is the tile at input
     * row 2rp + ky - 1, column 4cb + kx - 1.  A block that reaches outside the
     * image reads a 4-row x 6-cell window gathered with the pad rule instead. */
    const uint32_t irow = (uint32_t)plt_pad_w(g->in_w) * (uint32_t)icell;
    const uint32_t erow = 8u * (uint32_t)icell;          /* the edge window's row */
    static uint8_t edge[4 * 256] __attribute__((aligned(64)));
    uint32_t toff_in[32], toff_edge[32];
    const uint32_t brow_b = bordered ? plt_conv_border_row(g) : irow;
    const int d    = pw->cin / 32, na = d / 2;
    const int ppw  = plt_pad_w(pw->out_w), pph = plt_pad_h(pw->out_h);
    const uint32_t orow = (uint32_t)plt_pad_w(pw->out_w) * (uint32_t)ocell;
    int prp = -1, pcb = -1, live = 0;
    if (b->act_lut) plt_conv_lut_load(b->act_lut);
    {
        for (int t = 0; t < d && t < 32; t++) {
            toff_in[t]   = (uint32_t)(t / kw) * brow_b + (uint32_t)(t % kw) * (uint32_t)icell;
            toff_edge[t] = (uint32_t)(t / kw) * erow   + (uint32_t)(t % kw) * (uint32_t)icell;
        }
    }

#define DW_STORE_PREV() do { if (live) {                                          \
    volatile uint8_t *_t = b->out + (2 * prp) * orow + pcb * 4 * ocell;          \
    if (ocell == 16) plt_conv_drain_tile16(_t, orow, b->act_lut != NULL, 0);      \
    else PLT_CONV_DRAIN(_t, orow, b->act_lut != NULL);                            \
    } } while (0)

    /* One output group, one weight pass: push the nine tiles and the table. */
    PLT_NNA_FIELD(PLT_BANK_B, PLT_PW_W_ZERO, PLT_B_WSTREAM);
    for (int i = 0; i < d * 4; i++)
        plt_nna_push_weight_slice(b->weights + (uint32_t)i * 256);
    PLT_NNA_FIELD(PLT_BANK_B, PLT_PW_W_ZERO, PLT_B_TABLE_PTR);
    for (int j = 0; j < 2; j++)
        plt_nna_push_table_half(b->table + (uint32_t)j * 128);

    for (int rp = 0; rp < pph / 2; rp++) {
        for (int cb = 0; cb < ppw / 4; cb++) {
            /* tap g = ky*kw + kx reads (2rp + ky - 1, 4cb + kx - 1). */
            /* tap g = ky*kw + kx sits ky rows and kx cells from tap 0; `toff`
             * holds those offsets for the buffer being read, so there is no
             * divide per tap. */
            const volatile uint8_t *tap0;
            const uint32_t *toff;
            uint32_t brow;
            if (bordered) {
                brow = brow_b;
                tap0 = in + (uint32_t)(2 * rp) * brow + (uint32_t)(4 * cb) * 32u;
                toff = toff_in;
            } else if (rp > 0 && cb > 0 && 2 * rp + 3 <= g->in_h && 4 * cb + 5 <= g->in_w) {
                tap0 = in + (uint32_t)(2 * rp - 1) * irow + (uint32_t)(4 * cb - 1) * (uint32_t)icell;
                brow = irow;
                toff = toff_in;
            } else {
                plt_conv_gather_window(g, in, edge, erow, 2 * rp - 1, 4 * cb - 1, 4, 6, icell);
                tap0 = edge;
                brow = erow;
                toff = toff_edge;
            }
#define DW_FEED(base, row, port) do {                                               \
    if (icell == 16) PLT_NNA_FEED_TILE16((base), (row), (port));                    \
    else             PLT_NNA_FEED_TILE((base), (row), (port));                      \
} while (0)
            if (cb > 0) plt_nna_precommit();
            for (int g = 0; g < na; g++)
                DW_FEED(tap0 + toff[g], brow, PLT_PORT_ACT_A);
            if (cb > 0) plt_nna_pack();
            PLT_NNA_FIELD(PLT_BANK_A, PLT_PW_W_ZERO, PLT_A_MAC_A1);
            PLT_NNA_FIELD(PLT_BANK_A, PLT_PW_W_ZERO, PLT_A_MAC_A0);
            PLT_NNMAC(PLT_MAC_UNIT_A);
            for (int g = na; g < d; g++)
                DW_FEED(tap0 + toff[g], brow, PLT_PORT_ACT_B);
            PLT_NNMAC(PLT_MAC_UNIT_B);
            if (cb > 0) DW_STORE_PREV();
            prp = rp; pcb = cb; live = 1;
            PLT_NNA_FIELD(PLT_BANK_B, PLT_PW_W_ZERO, PLT_B_COMMIT_WIN);
        }
        /* Flush the pipeline at the end of every row pair, as the pointwise
         * loop does: the drain runs one MAC pair behind. */
        plt_nna_precommit(); plt_nna_pack();
        DW_STORE_PREV();
        live = 0;
    }
#undef DW_STORE_PREV
#undef DW_FEED
}

/* Stride 2 needs no tap tensor either -- only a longer reach.
 *
 * At stride 1 a tap is a shifted POINTER, so the array reads four consecutive
 * cells and the whole feed is one address (dw_tiles_stride1 above).  At stride 2
 * the four output pixels of a tile take every other input cell, which is a
 * gather -- but the gather can happen in the PUSH REGISTER rather than in
 * memory.  A 64-byte push is two output pixels, and a depthwise tap is a whole
 * 32-channel group, so each push is two 32-byte lane loads at independent
 * addresses: `lao vr0[0]` and `lao vr0[1]`, then nndwr.
 *
 * Three instructions per push against three today, and tap_im2col disappears.
 */

/* One 64-byte push of tap column `kx`: output row half `h` of the tile, output
 * column pair `jj`.  Output pixel ox reads input cell 2*ox + kx (the -1 of the
 * padding cancels against the margin), so within a tile the two pixels of a
 * pair are 2 cells = 2 octets apart, and the pairs are 4 apart.
 *
 * The row base is chosen by the caller (it depends on the tap's kernel row), so
 * only `h` selects between the two registers here.  Literals only: the
 * assembler evaluates these. */
#define DW2_PUSH(kx, h, jj)                                 \
    PLT_M3_LAO(0, 0, 8 + (h), 4 * (jj) + 0 + (kx))          \
    PLT_M3_LAO(0, 1, 8 + (h), 4 * (jj) + 2 + (kx))          \
    ".word %[nn]\n\t"

/* The four pushes of one tile, in PLT_NNA_FEED_TILE's order: row 0 columns 0-1
 * and 2-3, then row 1.  `rowp` already points at this tap's kernel row. */
#define DW2_FEED_TILE(kx, rowp, brow, port) do {                               \
    register const uint8_t *_r0 __asm__("t0") = (rowp);                        \
    register const uint8_t *_r1 __asm__("t1") = (rowp) + 2 * (brow);           \
    __asm__ __volatile__(".set push\n\t.set noreorder\n\t"                   \
        DW2_PUSH(kx, 0, 0) DW2_PUSH(kx, 0, 1)                                  \
        DW2_PUSH(kx, 1, 0) DW2_PUSH(kx, 1, 1)                                  \
        ".set pop\n\t"                                                         \
        :: [nn] "i"(PLT_NN_WORD(2, ((port) & 0x1Fu) | (((port) >> 5) << 10))), \
           "r"(_r0), "r"(_r1) : "memory");                                     \
} while (0)

/* DW2_FEED_TILE over NDHWC16 cells: a 16-byte pixel into quad lanes 0 and 2,
 * addressed in 16-byte units (see PLT_NNA_FEED_TILE16). */
#define DW2_PUSH16(kx, h, jj)                               \
    PLT_M3_LAQ(0, 0, 8 + (h), 4 * (jj) + 0 + (kx))          \
    PLT_M3_LAQ(0, 2, 8 + (h), 4 * (jj) + 2 + (kx))          \
    ".word %[nn]\n\t"
#define DW2_FEED_TILE16(kx, rowp, brow, port) do {                             \
    register const uint8_t *_r0 __asm__("t0") = (rowp);                        \
    register const uint8_t *_r1 __asm__("t1") = (rowp) + 2 * (brow);           \
    __asm__ __volatile__(".set push\n\t.set noreorder\n\t"                   \
        DW2_PUSH16(kx, 0, 0) DW2_PUSH16(kx, 0, 1)                              \
        DW2_PUSH16(kx, 1, 0) DW2_PUSH16(kx, 1, 1)                              \
        ".set pop\n\t"                                                         \
        :: [nn] "i"(PLT_NN_WORD(2, ((port) & 0x1Fu) | (((port) >> 5) << 10))), \
           "r"(_r0), "r"(_r1) : "memory");                                     \
} while (0)

static void dw_tiles_stride2(const plt_conv_geom_t *g, const plt_conv_geom_t *pw,
                             const plt_conv_bufs_t *b, const uint8_t *in, int bordered,
                             int icell, int ocell)
{
    /* Read in place, as dw_tiles_stride1: block (rp, cb) reads input rows
     * 4rp - 1 .. 4rp + 3 and cells 8cb - 1 .. 8cb + 7, and a block reaching
     * outside the image reads that 5 x 9 window gathered with the pad rule. */
    const uint32_t irow = (uint32_t)plt_pad_w(g->in_w) * (uint32_t)icell;
    const uint32_t erow = 12u * (uint32_t)icell;         /* the edge window's row */
    const uint32_t brow_b = bordered ? plt_conv_border_row(g) : irow;
    static uint8_t edge[5 * 384] __attribute__((aligned(64)));
    /* 9 taps in 9 input groups; the recipe splits them d/2 on unit A, so the
     * call sites below send taps 0-3 to A and 4-8 to B. */
    const int d = pw->cin / 32;
    const int ppw = plt_pad_w(pw->out_w), pph = plt_pad_h(pw->out_h);
    const uint32_t orow = (uint32_t)plt_pad_w(pw->out_w) * (uint32_t)ocell;
    int prp = -1, pcb = -1, live = 0;
    if (b->act_lut) plt_conv_lut_load(b->act_lut);

#define DW2_STORE_PREV() do { if (live) {                                         \
    volatile uint8_t *_t = b->out + (2 * prp) * orow + pcb * 4 * ocell;          \
    if (ocell == 16) plt_conv_drain_tile16(_t, orow, b->act_lut != NULL, 0);      \
    else PLT_CONV_DRAIN(_t, orow, b->act_lut != NULL);                            \
    } } while (0)

/* tap g = ky*3 + kx: its kernel row picks the base, its kernel column the
 * lane offsets.  Both are literals at the call site. */
#define DW2_TAP(g, ky, kx, port) do {                                         \
    if (icell == 16) DW2_FEED_TILE16(kx, tile + (uint32_t)(ky) * brow, brow, port); \
    else             DW2_FEED_TILE(kx, tile + (uint32_t)(ky) * brow, brow, port);   \
} while (0)

    PLT_NNA_FIELD(PLT_BANK_B, PLT_PW_W_ZERO, PLT_B_WSTREAM);
    for (int i = 0; i < d * 4; i++)
        plt_nna_push_weight_slice(b->weights + (uint32_t)i * 256);
    PLT_NNA_FIELD(PLT_BANK_B, PLT_PW_W_ZERO, PLT_B_TABLE_PTR);
    for (int j = 0; j < 2; j++)
        plt_nna_push_table_half(b->table + (uint32_t)j * 128);

    for (int rp = 0; rp < pph / 2; rp++) {
        for (int cb = 0; cb < ppw / 4; cb++) {
            const uint8_t *tile;
            uint32_t brow;
            if (bordered) {
                brow = brow_b;
                tile = in + (uint32_t)(4 * rp) * brow + (uint32_t)(8 * cb) * 32u;
            } else if (rp > 0 && cb > 0 && 4 * rp + 4 <= g->in_h && 8 * cb + 8 <= g->in_w) {
                tile = in + (uint32_t)(4 * rp - 1) * irow + (uint32_t)(8 * cb - 1) * (uint32_t)icell;
                brow = irow;
            } else {
                plt_conv_gather_window(g, in, edge, erow, 4 * rp - 1, 8 * cb - 1, 5, 9, icell);
                tile = edge;
                brow = erow;
            }
            if (cb > 0) plt_nna_precommit();
            DW2_TAP(0, 0, 0, PLT_PORT_ACT_A); DW2_TAP(1, 0, 1, PLT_PORT_ACT_A);
            DW2_TAP(2, 0, 2, PLT_PORT_ACT_A); DW2_TAP(3, 1, 0, PLT_PORT_ACT_A);
            if (cb > 0) plt_nna_pack();
            PLT_NNA_FIELD(PLT_BANK_A, PLT_PW_W_ZERO, PLT_A_MAC_A1);
            PLT_NNA_FIELD(PLT_BANK_A, PLT_PW_W_ZERO, PLT_A_MAC_A0);
            PLT_NNMAC(PLT_MAC_UNIT_A);
            DW2_TAP(4, 1, 1, PLT_PORT_ACT_B); DW2_TAP(5, 1, 2, PLT_PORT_ACT_B);
            DW2_TAP(6, 2, 0, PLT_PORT_ACT_B); DW2_TAP(7, 2, 1, PLT_PORT_ACT_B);
            DW2_TAP(8, 2, 2, PLT_PORT_ACT_B);
            PLT_NNMAC(PLT_MAC_UNIT_B);
            if (cb > 0) DW2_STORE_PREV();
            prp = rp; pcb = cb; live = 1;
            PLT_NNA_FIELD(PLT_BANK_B, PLT_PW_W_ZERO, PLT_B_COMMIT_WIN);
        }
        plt_nna_precommit(); plt_nna_pack();
        DW2_STORE_PREV();
        live = 0;
    }
#undef DW2_TAP
#undef DW2_STORE_PREV
}

/* Can this depthwise be fed straight from a bordered copy?  The tap->base/offset
 * mapping in dw_tiles_stride2 is written out for a 3x3 stride-2 window with
 * one-pixel padding, which is every strided depthwise in the model. */
static int dw_fused_stride2(const plt_conv_geom_t *g)
{
    return g->stride == 2 && g->kh == 3 && g->kw == 3
        && g->pad_t == 1 && g->pad_l == 1;
}

/* Feed in place, or from a bordered copy?  In place saves the copy, but a block
 * touching the border is gathered cell by cell, and on a small plane most blocks
 * do: measured, a 14x14 layer is ~0.1 ms slower in place and a 28x28 one faster. */
#define DW_DIRECT_MIN_W 20

static int dw_direct(const plt_conv_geom_t *g)
{
    return g->in_w >= DW_DIRECT_MIN_W;
}

int plt_conv_dw_cell16_ok(const plt_node_t *node)
{
    plt_conv_geom_t g;
    const plt_layer_rec_t *r = node->rec;
    g.stride = r->sw ? r->sw : 1; g.kh = r->kh; g.kw = r->kw; g.pad_t = r->pt; g.pad_l = r->pl;
    g.in_w = node->in[0].shape.w;
    return node->in[0].shape.c <= 16 && r->kh == 3 && r->kw == 3
        && (g.stride == 1 || dw_fused_stride2(&g)) && dw_direct(&g);
}

static int dw_plan(plt_ctx_t *ctx, plt_node_t *node, plt_plan_t *plan)
{
    plt_conv_geom_t g;

    if (plt_conv_geom_of(ctx, node, &g) != 0) return -1;
    g.cin = (g.cin + 31) & ~31; g.cout = (g.cout + 31) & ~31;   /* whole 32-lane groups */
    if (dw_check(ctx, &g) != 0) return -1;

    plan->out_pad_h     = plt_pad_h(g.out_h);
    plan->out_pad_w     = plt_pad_w(g.out_w);
    plan->out_bytes     = plt_ndhwc32_bytes(g.cout, g.out_h, g.out_w);
    plan->scratch_bytes = (g.stride == 1 || dw_fused_stride2(&g))
        ? (dw_direct(&g) ? 0 : plt_conv_border_bytes(&g))         /* in place, or one bordered plane */
        : plt_ndhwc32_bytes(g.kh * g.kw * 32, g.out_h, g.out_w);  /* the tap tensor */
    return 0;
}

static int dw_run(plt_ctx_t *ctx, const plt_node_t *node)
{
    plt_conv_geom_t g, pw;
    plt_conv_bufs_t b;

    if (plt_conv_geom_of(ctx, node, &g) != 0) return -1;
    g.cin = (g.cin + 31) & ~31; g.cout = (g.cout + 31) & ~31;   /* whole 32-lane groups */
    if (dw_check(ctx, &g) != 0) return -1;
    if (plt_conv_bufs_of(ctx, node, &b) != 0) return -1;

    const int fused2 = dw_fused_stride2(&g);
    const int windowed = g.stride == 1 || fused2;
    const int direct = windowed && dw_direct(&g);
    const int icell = node->in[0].layout == PLT_FMT_NDHWC16 ? 16 : 32;
    const int ocell = node->out.layout   == PLT_FMT_NDHWC16 ? 16 : 32;
    if ((icell == 16 || ocell == 16) && (!direct || g.cin != 32))
        return plt_ctx_fail(ctx, "conv_dw: NDHWC16 needs one group fed in place");
    const uint32_t need = direct ? 0 : windowed ? plt_conv_border_bytes(&g)
        : plt_ndhwc32_bytes(g.kh * g.kw * 32, g.out_h, g.out_w);
    volatile uint8_t *col = direct ? NULL : plt_ctx_scratch(ctx, need);
    if (!direct && !col)
        return plt_ctx_fail(ctx, "conv_dw: no scratch holds the %u bytes the taps need",
                            need);

    as_pointwise(&g, &pw);

    uint8_t *stage = NULL;
    uint8_t *in_stage = NULL, *out_stage = NULL;
    if (g.stride != 1 && !fused2) {
        const uint32_t in_row  = plt_ndhwc32_row_bytes(g.in_w);
        const uint32_t out_row = plt_ndhwc32_row_bytes(g.out_w);
        stage = malloc((size_t)in_row + out_row);
        if (!stage) return plt_ctx_fail(ctx, "conv_dw: out of memory for staging");
        in_stage = stage; out_stage = stage + in_row;
    }

    const uint32_t in_plane  = plt_ndhwc32_plane_bytes(g.in_h,  g.in_w);
    const uint32_t out_plane = plt_ndhwc32_plane_bytes(g.out_h, g.out_w);
    const uint32_t taps      = (uint32_t)g.kh * g.kw;

    for (int grp = 0; grp < g.cin / 32; grp++) {
        plt_conv_bufs_t gb = b;   /* inherit in_alias_groups and friends */
        gb.weights = b.weights + (uint32_t)grp * taps * 1024u;
        gb.table   = b.table   + (uint32_t)grp * 256u;
        gb.in      = col;
        gb.out     = b.out     + (uint32_t)grp * out_plane;

        uint32_t t = plt_prof_now(ctx);
        if (windowed && !direct)
            plt_conv_border_copy(&g, b.in + (uint32_t)grp * in_plane, col);
        else if (!windowed)
            tap_im2col(&g, b.in + (uint32_t)grp * in_plane, col, in_stage, out_stage);
        t = plt_prof_mark(ctx, &ctx->prof.assemble_us, t);

        if (!gb.act_lut && node->rec->act == PLT_ACT_SILU)
            /* one table per channel group, then the folded activation table:
             * measure it from the BASE, past every group's requant table. */
            gb.act_lut = b.table + (uint32_t)(g.cin / 32) * 256u;
        plt_conv_pw_configure(&pw);
        if (g.stride == 1)
            dw_tiles_stride1(&g, &pw, &gb, direct ? b.in + (uint32_t)grp * in_plane : col,
                             g.kw, !direct, icell, ocell);
        else if (fused2)
            dw_tiles_stride2(&g, &pw, &gb, (const uint8_t *)(uintptr_t)
                             (direct ? b.in + (uint32_t)grp * in_plane : col), !direct,
                             icell, ocell);
        else
            plt_conv_pw_tiles(&pw, &gb);
        plt_prof_mark(ctx, &ctx->prof.array_us, t);
    }
    free(stage);
    return 0;
}

const plt_kernel_t plt_kernel_conv_dw = {
    PLT_OP_DWCONV, PLT_EX_NNA_DW, "conv_dw", dw_plan, dw_run
};
