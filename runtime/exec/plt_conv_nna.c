#include <string.h>

#include "exec/plt_conv_nna.h"
#include "core/plt_layout.h"
#include "hal/plt_dev.h"

#ifdef PLT_BENCH_SINK
uint8_t plt_bench_sink[512] __attribute__((aligned(64)));
#endif

int plt_conv_groups_per_pass(int w_bits, int in_groups)
{
    const int g = PLT_NNA_TAP_SLOTS / ((w_bits / 2) * in_groups);
    return g < 1 ? 1 : g;
}

int plt_conv_geom_of(plt_ctx_t *ctx, const plt_node_t *node, plt_conv_geom_t *geom)
{
    const plt_layer_rec_t *r = node->rec;

    if (node->n_in < 1) return plt_ctx_fail(ctx, "layer %u has no input", r->id);

    geom->cin    = node->in[0].shape.c;
    geom->cout   = node->out.shape.c;
    geom->in_h   = node->in[0].shape.h;
    geom->in_w   = node->in[0].shape.w;
    geom->out_h  = node->out.shape.h;
    geom->out_w  = node->out.shape.w;
    geom->kh     = r->kh;
    geom->kw     = r->kw;
    geom->stride = r->sw ? r->sw : 1;
    geom->pad_t  = r->pt;
    geom->pad_l  = r->pl;
    geom->in_bits  = r->in_bits;
    geom->w_bits   = r->w_bits;
    geom->out_bits = r->out_bits;
    geom->in_zp    = r->in_zp;

    /* Channel counts are logical here; each executor rounds them up to whole
     * 32-lane groups as the array needs (the compiler zero-pads the operands to
     * match).  So no multiple-of-32 check lives here. */
    if (r->sh != r->sw)
        return plt_ctx_fail(ctx, "layer %u: stride %ux%u is not square", r->id, r->sh, r->sw);
    return 0;
}

int plt_conv_bufs_of(plt_ctx_t *ctx, const plt_node_t *node, plt_conv_bufs_t *bufs)
{
    bufs->in_alias_groups = 0;
    bufs->act_lut = node->act_lut_override;
    bufs->in_cell  = node->in[0].layout == PLT_FMT_NDHWC16 ? 16 : 32;
    bufs->out_cell = node->out.layout   == PLT_FMT_NDHWC16 ? 16 : 32;
    bufs->weights = plt_ctx_ptr(ctx, node->weights);
    bufs->table   = plt_ctx_ptr(ctx, node->table);
    bufs->in      = plt_ctx_ptr(ctx, node->in[0].mem);
    bufs->out     = plt_ctx_ptr(ctx, node->out.mem);

    if (!bufs->weights || !bufs->table || !bufs->in || !bufs->out)
        return plt_ctx_fail(ctx, "layer %u: an operand is unmapped "
                            "(w=%p tbl=%p in=%p out=%p)", node->rec->id,
                            (void *)bufs->weights, (void *)bufs->table,
                            (void *)bufs->in, (void *)bufs->out);
    return 0;
}

static uint32_t round64(uint32_t v) { return (v + 63u) & ~63u; }

uint32_t plt_conv_stage_bytes(uint32_t in_bytes, uint32_t w_bytes, uint32_t tbl_bytes)
{
    return round64(in_bytes) + round64(w_bytes) + round64(tbl_bytes);
}

int plt_conv_stage_operands(plt_ctx_t *ctx, plt_conv_bufs_t *bufs,
                            uint32_t in_bytes, uint32_t w_bytes, uint32_t tbl_bytes)
{
    const uint32_t need = plt_conv_stage_bytes(in_bytes, w_bytes, tbl_bytes);
    uint8_t *p = plt_ctx_scratch(ctx, need);

    if (!p) return plt_ctx_fail(ctx, "no scratch to stage %u bytes of operands", need);

    /* A zero size means "leave this operand where it is" -- staging an operand
     * that is read only once is pure copying, so callers ask for only the ones
     * that are re-read. */
    if (in_bytes) {
        plt_copy_fast(p, bufs->in, in_bytes);
        bufs->in = p;
        p += round64(in_bytes);
    }
    if (w_bytes) {
        plt_copy_fast(p, bufs->weights, w_bytes);
        bufs->weights = p;
        p += round64(w_bytes);
    }
    if (tbl_bytes) {
        plt_copy_fast(p, bufs->table, tbl_bytes);
        bufs->table = p;
    }
    return 0;
}

/* --- the bordered plane (see the header) ----------------------------------- */

uint32_t plt_conv_border_row(const plt_conv_geom_t *g)
{ return ((uint32_t)(g->stride * plt_pad_w(g->out_w)) + PLT_CONV_BORDER_COLS) * 32u; }

uint32_t plt_conv_border_bytes(const plt_conv_geom_t *g)
{
    return ((uint32_t)(g->stride * plt_pad_h(g->out_h)) + PLT_CONV_BORDER_ROWS)
         * plt_conv_border_row(g);
}

/* Copy `n` bytes flipping every top bit -- the q-to-b conversion, see
 * plt_conv_border_copy_xor().  A word at a time; the run is a whole number of
 * 32-lane cells, so the tail is dead code. */
static void copy_xor80(volatile uint8_t *dst, const volatile uint8_t *src, size_t n)
{
    uint32_t       *d = (uint32_t *)(uintptr_t)dst;
    const uint32_t *s = (const uint32_t *)(uintptr_t)src;
    size_t w = n >> 2, i = 0;
    for (; i < w; i++) d[i] = s[i] ^ 0x80808080u;
    for (size_t j = w << 2; j < n; j++) dst[j] = src[j] ^ 0x80u;
}

void plt_conv_border_copy(const plt_conv_geom_t *g, const volatile uint8_t *src,
                          volatile uint8_t *dst)
{
    plt_conv_border_copy_xor(g, src, dst, 0x00);
}

/* Bordered copy, optionally converting the fed byte on the way through.
 *
 * The array reads the activation byte UNSIGNED and the requant folds
 * `feed_offset = 128 + x_zp` out of the bias, so a real tap must be fed
 * `b = q + 128`.  A feature map produced by the array already carries `b` and is
 * copied straight through (`xor_mask` 0).  The graph input for an RGB image is
 * packed as the two's-complement byte of `q` instead (compile/dump_run.py), and
 * `^ 0x80` is exactly that conversion -- `xor_mask` 0x80.
 *
 * plt_conv_stem.c states the same rule for the window it assembles; this is that
 * rule for the executors that feed the array straight from the input.  The
 * border itself is never converted: it is written as `128 + x_zp`, which is
 * already in the fed domain.
 */
void plt_conv_border_copy_xor(const plt_conv_geom_t *g, const volatile uint8_t *src,
                              volatile uint8_t *dst, uint8_t xor_mask)
{
    const uint32_t in_row = plt_ndhwc32_row_bytes(g->in_w);
    const uint32_t brow   = plt_conv_border_row(g);
    const uint32_t rows   = (uint32_t)(g->stride * plt_pad_h(g->out_h))
                          + PLT_CONV_BORDER_ROWS;
    const uint32_t kept   = 32u + (uint32_t)g->in_w * 32u;   /* left margin + data */
    const uint8_t pad = (uint8_t)(128 + g->in_zp);  /* border feeds u=0 */

    plt_fill_fast(dst, pad, brow);                           /* the row above */
    for (uint32_t y = 1; y < rows; y++) {
        volatile uint8_t *row = dst + y * brow;
        if ((int)y > g->in_h) { plt_fill_fast(row, pad, brow); continue; }
        plt_fill_fast(row, pad, 32);                         /* the left margin */
        if (xor_mask)
            copy_xor80(row + 32, src + (y - 1) * in_row, (size_t)g->in_w * 32);
        else
            plt_copy_fast(row + 32, src + (y - 1) * in_row, (size_t)g->in_w * 32);
        plt_fill_fast(row + kept, pad, brow - kept);         /* the right margin */
    }
}

void plt_conv_to_pairs(uint8_t *dst, const volatile uint8_t *src, int groups, int rows,
                       int pairs, uint32_t in_row, uint32_t in_plane)
{
    const uint32_t pst = (uint32_t)groups * 64u, prow = (uint32_t)pairs * pst;
    const uint8_t *s8 = (const uint8_t *)(uintptr_t)src;
    for (int g = 0; g < groups; g++)
        for (int y = 0; y < rows; y++) {
            const uint8_t *sp_ = s8 + (uint32_t)g * in_plane + (uint32_t)y * in_row;
            uint8_t *dp_ = dst + (uint32_t)y * prow + (uint32_t)g * 64u;
            register const uint8_t *sp __asm__("t0") = sp_;
            register uint8_t *dp __asm__("t1") = dp_;
            register uint32_t n __asm__("t2") = (uint32_t)pairs;
            register uint32_t step __asm__("t3") = pst;
            if (n) __asm__ __volatile__(".set push\n\t.set noreorder\n\t.set noat\n\t"
                "1:\n\t"
                PLT_M3_LAO(0, 0, 8, 0) PLT_M3_LAO(0, 1, 8, 1)
                PLT_M3_SAO(0, 0, 9, 0) PLT_M3_SAO(0, 1, 9, 1)
                "addiu %[s], %[s], 64\n\t"
                "addu %[d], %[d], %[st]\n\t"
                "addiu %[n], %[n], -1\n\t" "bnez %[n], 1b\n\t" "nop\n\t"
                ".set pop\n\t"
                : [s] "+r"(sp), [d] "+r"(dp), [n] "+r"(n), [st] "+r"(step) :: "memory");
        }
}

void plt_conv_gather_window(const plt_conv_geom_t *g, const volatile uint8_t *plane,
                            uint8_t *dst, uint32_t dst_row, int y0, int x0,
                            int rows, int cols, int cell)
{
    const uint32_t in_row = (uint32_t)plt_pad_w(g->in_w) * (uint32_t)cell;
    const uint32_t padw   = 0x01010101u * (uint8_t)(128 + g->in_zp);
    const uint8_t *pl = (const uint8_t *)(uintptr_t)plane;
    for (int r = 0; r < rows; r++) {
        const int y = y0 + r;
        uint8_t *d = dst + (uint32_t)r * dst_row;
        for (int c = 0; c < cols; c++, d += cell) {
            const int x = x0 + c;
            if (y < 0 || y >= g->in_h || x < 0 || x >= g->in_w)
                for (int i = 0; i < cell; i += 4) memcpy(d + i, &padw, 4);
            else
                memcpy(d, pl + (uint32_t)y * in_row + (uint32_t)x * (uint32_t)cell, (size_t)cell);
        }
    }
}

/* Park a folded activation table in the MXU3 register file for the tile loop.
 *
 * vr16..vr19 take the table's four 64-byte quarters, vr20..vr22 the three
 * constants PLT_M3_LUT64 needs (all-ones, 0x80, 0x01).  They stay there for the
 * whole layer: nothing else in the conv path writes vr16..vr30.
 */
void plt_conv_lut_load(const volatile uint8_t *lut)
{
    static const uint8_t ones[64] __attribute__((aligned(64))) = {
        [0 ... 63] = 0xFF };
    static const uint8_t top[64] __attribute__((aligned(64))) = {
        [0 ... 63] = 0x80 };
    static const uint8_t one[64] __attribute__((aligned(64))) = {
        [0 ... 63] = 0x01 };
    /* The caller's copy is a plain array, but not necessarily 64-byte aligned,
     * and PLT_VLD64 wants alignment. */
    static uint8_t staged[256] __attribute__((aligned(64)));
    for (int i = 0; i < 256; i++) staged[i] = lut[i];

    PLT_VLD64(16, staged +   0);
    PLT_VLD64(17, staged +  64);
    PLT_VLD64(18, staged + 128);
    PLT_VLD64(19, staged + 192);
    PLT_VLD64(20, ones);
    PLT_VLD64(21, top);
    PLT_VLD64(22, one);
}
