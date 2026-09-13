/* Types and small utilities shared by the NNA convolution executors.
 *
 * Shared TYPES and UTILITIES only -- no control flow.  Each convolution kind is
 * one written-out function that owns its own loop nest; what they have in
 * common is pushed DOWN into hal/plt_nna.h, not sideways into a driver.  The
 * reasoning is in docs/ARCHITECTURE.md 4.
 */
#ifndef PLT_EXEC_CONV_NNA_H
#define PLT_EXEC_CONV_NNA_H

#include <stdint.h>

#include "core/plt_ctx.h"
#include "core/plt_kernel.h"
#include "hal/plt_isa_mxu.h"
#include "hal/plt_mxu3.h"
#include "hal/plt_nna.h"

/* One convolution's shape, as the array sees it. */
typedef struct {
    int cin, cout;          /* channel counts, padded to multiples of 32 */
    int in_h, in_w;         /* input size                                */
    int out_h, out_w;       /* output size                               */
    int kh, kw;
    int stride;
    int pad_t, pad_l;
    int in_bits, w_bits, out_bits;
    int in_zp;              /* the value the array pads the border with  */
} plt_conv_geom_t;

/* A convolution's four operands, resolved to pointers into their windows. */
typedef struct {
    volatile uint8_t *weights, *table, *in, *out;

    /* Nonzero: every input group feeds from the SAME plane.
     *
     * The 1x1 recipe splits its input groups across the array's two execution
     * units, so a layer with one real group needs a partner.  The partner's
     * weights are all zero, so what it reads cannot affect the result -- which
     * means it does not need to exist.  Pointing it at the real plane saves
     * materializing and feeding a whole plane of padding.  Used by the stem and
     * by the one pointwise layer with 32 input channels. */
    int in_alias_groups;

    /* A SiLU folded into this convolution (compile/model_build.py:_fuse_silu),
     * or NULL.  The tile loop puts every drained byte through it while the tile
     * is still hot, which is the whole point: it replaces a separate full pass
     * over the output buffer. */
    const volatile uint8_t *act_lut;

    /* Bytes a pixel of the input and of the output: 32, or 16 for an NDHWC16
     * tensor (at most 16 channels; see plt_engine_load). */
    int in_cell, out_cell;
} plt_conv_bufs_t;

/* Put one drained tile (two rows of 128 bytes) through a folded activation
 * table, 64 bytes at a time on MXU3.
 *
 * The tile was just written, so it is hot; that is the entire reason folding
 * beats a separate pass over the whole buffer.  What makes the fold nearly free
 * is `gshufvb`, MXU3.1's data-dependent byte gather -- see PLT_M3_LUT64.  It
 * replaces 256 scalar byte lookups with 4 x 19 vector instructions: measured
 * 0.221 us against 0.907 us per tile on the device, and 96% of every model's
 * output tiles go through here.
 *
 * plt_conv_lut_load() must be called once, before the tile loop, to park the
 * table's four quarters and the three constants in vr16..vr22.  The tile pass
 * then owns vr23..vr30 as scratch, and touches nothing the feed (vr0..vr3),
 * the drain (vr10..vr13) or the config staging (vr31) uses.
 */
void plt_conv_lut_load(const volatile uint8_t *lut);

#define PLT_CONV_LUT_BLOCK(p) do {                                            \
    PLT_VLD64(29, (const uint8_t *)(uintptr_t)(p));                           \
    __asm__ __volatile__(".set push\n.set noreorder\n"                        \
        PLT_M3_LUT64(29, 30, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28) \
        ".set pop\n" ::: "memory");                                           \
    PLT_VST64(30, (uint8_t *)(uintptr_t)(p));                                 \
} while (0)

static inline void plt_conv_apply_lut(volatile uint8_t *tile, uint32_t row_bytes)
{
    uint8_t *p = (uint8_t *)(uintptr_t)tile;
    PLT_CONV_LUT_BLOCK(p +              0);
    PLT_CONV_LUT_BLOCK(p +             64);
    PLT_CONV_LUT_BLOCK(p + row_bytes +  0);
    PLT_CONV_LUT_BLOCK(p + row_bytes + 64);
}

/* Put a DRAIN register through the folded table in place, before it is ever
 * stored.  The drain has just filled vr10..vr13; running the table on those
 * registers directly means the tile is written once, already activated, instead
 * of stored raw, loaded back and stored again -- four loads and four stores
 * fewer per tile, and no store-to-load forwarding on memory written a moment
 * earlier.  PLT_M3_LUT64 reads its source only in its first three instructions,
 * so the source may be its own destination. */
#ifdef PLT_BENCH_NOLUT
#define PLT_CONV_LUT_REG(vr)
#else
#define PLT_CONV_LUT_REG(vr) __asm__ __volatile__(".set push\n.set noreorder\n"   \
        PLT_M3_LUT64(vr, vr, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28) \
        ".set pop\n" ::: "memory")
#endif

/* plt_nna_drain_tile_fifo() with the folded activation applied in-register. */
static inline void plt_conv_drain_tile_lut(volatile uint8_t *dst, uint32_t row_bytes)
{
    uint8_t *p = (uint8_t *)(uintptr_t)dst;
    PLT_DRAIN_FIFO(10); PLT_DRAIN_FIFO(11); PLT_DRAIN_FIFO(12); PLT_DRAIN_FIFO(13);
    PLT_CONV_LUT_REG(10); PLT_CONV_LUT_REG(11); PLT_CONV_LUT_REG(12); PLT_CONV_LUT_REG(13);
    PLT_VST64(10, p +             0);
    PLT_VST64(11, p +            64);
    PLT_VST64(12, p + row_bytes +  0);
    PLT_VST64(13, p + row_bytes + 64);
}

/* A drained tile stored as NDHWC16 cells: quad lanes 0 and 2 of each register
 * are the two pixels' first 16 channels.  `ro` orders the four registers as
 * the recipe drains them: 0 is the pointwise FIFO order (vr10/vr11 row 0,
 * vr12/vr13 row 1), 1 is the 3x3 order (vr10/vr12 row 0). */
#define PLT_CONV_STORE16(vr, dst) do {                                        \
    register uint8_t *_d __asm__("t0") = (uint8_t *)(uintptr_t)(dst);          \
    __asm__ __volatile__(".set push\n.set noreorder\n"                         \
        PLT_M3_SAQ(vr, 0, 8, 0) PLT_M3_SAQ(vr, 2, 8, 1) ".set pop\n"          \
        :: "r"(_d) : "memory");                                               \
} while (0)

static inline void plt_conv_drain_tile16(volatile uint8_t *dst, uint32_t row_bytes,
                                         int has_lut, int order3x3)
{
    PLT_DRAIN_FIFO(10); PLT_DRAIN_FIFO(11); PLT_DRAIN_FIFO(12); PLT_DRAIN_FIFO(13);
    if (has_lut) {
        PLT_CONV_LUT_REG(10); PLT_CONV_LUT_REG(11); PLT_CONV_LUT_REG(12); PLT_CONV_LUT_REG(13);
    }
    if (order3x3) {
        PLT_CONV_STORE16(10, dst + 0);         PLT_CONV_STORE16(12, dst + 32);
        PLT_CONV_STORE16(11, dst + row_bytes); PLT_CONV_STORE16(13, dst + row_bytes + 32);
    } else {
        PLT_CONV_STORE16(10, dst + 0);         PLT_CONV_STORE16(11, dst + 32);
        PLT_CONV_STORE16(12, dst + row_bytes); PLT_CONV_STORE16(13, dst + row_bytes + 32);
    }
}

/* One drained tile to memory, through the folded table when there is one. */
#ifdef PLT_BENCH_SINK
extern uint8_t plt_bench_sink[512];
#define PLT_CONV_DRAIN(dst, row_bytes, has_lut) do {                          \
    (void)(dst); (void)(row_bytes);                                           \
    if (has_lut) plt_conv_drain_tile_lut(plt_bench_sink, 256);                 \
    else         plt_nna_drain_tile_fifo(plt_bench_sink, 256);                 \
} while (0)
#else
#define PLT_CONV_DRAIN(dst, row_bytes, has_lut) do {                          \
    if (has_lut) plt_conv_drain_tile_lut((dst), (row_bytes));                  \
    else         plt_nna_drain_tile_fifo((dst), (row_bytes));                  \
} while (0)
#endif

/* A window walk with one-pixel padding, fed straight from the input.
 *
 * The taps of a KxK window are K*K SHIFTED VIEWS of the same plane, so the array
 * can read them out of the input itself rather than out of a lowered copy the
 * CPU builds -- provided every address the walk touches is valid.  Copying the
 * plane once into the middle of a bordered buffer is what makes that true, and
 * it is one plane copy instead of K*K planes of im2col.
 *
 * Used by the stride-1 depthwise tile loop and by the fused stem feed.
 */
#define PLT_CONV_BORDER_COLS 4    /* one on the left, three of slack on the right */
#define PLT_CONV_BORDER_ROWS 2    /* one above, one below                          */

/* The buffer covers the INPUT extent the walk reaches, which is `stride` input
 * cells per output cell, so a strided layer needs a correspondingly larger one.
 * At stride 1 these are the plain padded output dimensions plus the margin. */

uint32_t plt_conv_border_row  (const plt_conv_geom_t *g);
uint32_t plt_conv_border_bytes(const plt_conv_geom_t *g);

/* Copy one channel group's plane into the middle of such a buffer.  Only the
 * border is filled (with the input zero point in the fed domain): the interior
 * is about to be overwritten by the copy, and this runs once per channel group
 * on every layer that uses it, so a whole-buffer fill would double the traffic
 * for nothing. */
void plt_conv_border_copy_xor(const plt_conv_geom_t *g, const volatile uint8_t *src,
                              volatile uint8_t *dst, uint8_t xor_mask);
void plt_conv_border_copy(const plt_conv_geom_t *g, const volatile uint8_t *src,
                          volatile uint8_t *dst);

/* Gather a `rows` x `cols` window of one channel-group plane whose top-left
 * input cell is (y0, x0) into `dst`, `dst_row` bytes per row, with the bordered
 * copy's rule: a cell inside the image is the plane's 32 bytes, anything else is
 * the pad byte (128 + zp).  What lets a feed read its input in place and still
 * give edge tiles exactly the bytes a bordered copy would have held. */
void plt_conv_gather_window(const plt_conv_geom_t *g, const volatile uint8_t *plane,
                            uint8_t *dst, uint32_t dst_row, int y0, int x0,
                            int rows, int cols, int cell);

/* Copy `groups` NDHWC32 planes into the pair-interleaved layout a multi-pass
 * feed walks: group g of cells (2p, 2p+1) of row y lands at
 * dst + y * pairs*groups*64 + p * groups*64 + g * 64.  `rows` x `pairs` is
 * the extent copied; `in_row`/`in_plane` are the source's NDHWC32 strides. */
void plt_conv_to_pairs(uint8_t *dst, const volatile uint8_t *src, int groups, int rows,
                       int pairs, uint32_t in_row, uint32_t in_plane);

/* Derive the geometry from a node.  Fails if the node is not a convolution the
 * array can express -- an unpadded channel count, say. */
int plt_conv_geom_of(plt_ctx_t *ctx, const plt_node_t *node, plt_conv_geom_t *geom);

/* Resolve the node's four operands.  Fails if any of them is absent or lives in
 * a space this context does not map. */
int plt_conv_bufs_of(plt_ctx_t *ctx, const plt_node_t *node, plt_conv_bufs_t *bufs);

/* Bytes of scratch plt_conv_stage_operands() needs for these three operands. */
uint32_t plt_conv_stage_bytes(uint32_t in_bytes, uint32_t w_bytes, uint32_t tbl_bytes);

/* Copy a convolution's read-only operands into cached scratch and repoint
 * `bufs` at the copies.
 *
 * The array is fed from a vector register, so a feed source in a mapped window
 * costs an uncached load per push and buys nothing.  Staging turns those into
 * one bulk sequential read plus cached loads -- and a layer that runs several
 * weight passes re-feeds its activations from cache every pass.  The OUTPUT is
 * deliberately not staged: the drain writes memory the next layer reads. */
int plt_conv_stage_operands(plt_ctx_t *ctx, plt_conv_bufs_t *bufs,
                            uint32_t in_bytes, uint32_t w_bytes, uint32_t tbl_bytes);

/* How many output groups' weights fit in the array's tap slots at once, and so
 * how many weight passes a layer needs:  G = 288 / ((w_bits/2) * D). */
#define PLT_NNA_TAP_SLOTS 288
int plt_conv_groups_per_pass(int w_bits, int in_groups);

#endif /* PLT_EXEC_CONV_NNA_H */
