#include "exec/plt_focus.h"
#include <stdlib.h>
#include "core/plt_layout.h"
#include "exec/plt_requant.h"
#include "hal/plt_dev.h"
#include "hal/plt_isa_mxu.h"
#include "hal/plt_mxu3.h"

/* Space-to-depth: out channel (pos*Cin + c) at (oy,ox) = in channel c at
 * (2oy+dy, 2ox+dx); pos order tl,bl,tr,br == (0,0),(1,0),(0,1),(1,1). */
static const int DY[4] = { 0, 1, 0, 1 }, DX[4] = { 0, 0, 1, 1 };

static int focus_plan(plt_ctx_t *ctx, plt_node_t *node, plt_plan_t *plan)
{
    (void)ctx;
    const plt_tensor_t *o = &node->out;
    plan->out_pad_h = plt_pad_h(o->shape.h);
    plan->out_pad_w = plt_pad_w(o->shape.w);
    plan->out_bytes = plt_ndhwc32_bytes(o->shape.c, o->shape.h, o->shape.w);
    plan->scratch_bytes = 0;
    return 0;
}

/* The 3-channel case, which is what an RGB stem is, done a WORD at a time.
 *
 * There is nothing to vectorize here: the layer moves 3 bytes from each of four
 * source cells into 12 contiguous destination bytes, and no lane operation is
 * that fine.  What made it expensive was doing twelve single-byte stores per
 * output pixel, each with its own `oc/32, oc%32` address arithmetic.  The
 * twelve bytes are contiguous, so they are three word stores; the four sources
 * are four word loads whose top bytes are masked off.  Seventeen operations
 * instead of well over a hundred, which leaves the layer where it should be --
 * bound by reading 32 bytes of input cell to use 3 of them.
 *
 * Byte i of a cell is channel i, and the device is little-endian, so byte 0 of
 * a cell is the low byte of a word load.  The same assumption the word-wise
 * requant passes elsewhere in the runtime already make.
 */
static void focus_rgb(uint8_t *pd, const uint8_t *ps, int OH, int OW,
                      uint32_t irow, uint32_t orow)
{
    for (int oy = 0; oy < OH; oy++) {
        const uint8_t *ra = ps + (uint32_t)(2 * oy) * irow;
        const uint8_t *rb = ra + irow;
        uint8_t *o = pd + (uint32_t)oy * orow;
        int ox = 0;
        for (; ox + 1 < OW; ox += 2, ra += 128, rb += 128, o += 64) {
            const uint32_t a0 = *(const uint32_t *)(const void *)(ra);
            const uint32_t a1 = *(const uint32_t *)(const void *)(ra + 32);
            const uint32_t a2 = *(const uint32_t *)(const void *)(ra + 64);
            const uint32_t a3 = *(const uint32_t *)(const void *)(ra + 96);
            const uint32_t b0 = *(const uint32_t *)(const void *)(rb);
            const uint32_t b1 = *(const uint32_t *)(const void *)(rb + 32);
            const uint32_t b2 = *(const uint32_t *)(const void *)(rb + 64);
            const uint32_t b3 = *(const uint32_t *)(const void *)(rb + 96);
            uint32_t *w = (uint32_t *)(void *)o;
            w[0] = (a0 & 0x00FFFFFFu) | (b0 << 24);                    /* tl0-2 bl0   */
            w[1] = ((b0 >> 8) & 0x0000FFFFu) | ((a1 & 0xFFFFu) << 16); /* bl1-2 tr0-1 */
            w[2] = ((a1 >> 16) & 0xFFu) | ((b1 & 0x00FFFFFFu) << 8);   /* tr2  br0-2  */
            w[8] = (a2 & 0x00FFFFFFu) | (b2 << 24);
            w[9] = ((b2 >> 8) & 0x0000FFFFu) | ((a3 & 0xFFFFu) << 16);
            w[10] = ((a3 >> 16) & 0xFFu) | ((b3 & 0x00FFFFFFu) << 8);
        }
        for (; ox < OW; ox++, ra += 64, rb += 64, o += 32) {
            const uint32_t a0 = *(const uint32_t *)(const void *)(ra);
            const uint32_t a1 = *(const uint32_t *)(const void *)(ra + 32);
            const uint32_t b0 = *(const uint32_t *)(const void *)(rb);
            const uint32_t b1 = *(const uint32_t *)(const void *)(rb + 32);
            uint32_t *w = (uint32_t *)(void *)o;
            w[0] = (a0 & 0x00FFFFFFu) | (b0 << 24);
            w[1] = ((b0 >> 8) & 0x0000FFFFu) | ((a1 & 0xFFFFu) << 16);
            w[2] = ((a1 >> 16) & 0xFFu) | ((b1 & 0x00FFFFFFu) << 8);
        }
    }
}

/* The same layer over a PACKED input -- three bytes a pixel, row-major (see
 * plt_engine_load) -- two output cells at a time on MXU3.
 *
 * Two output cells need pixels 2ox..2ox+3 of input rows 2oy and 2oy+1: twelve
 * packed bytes of each row.  One 32-byte lane load takes them from each row
 * into the two halves of a register, one `gshufvb` with a constant index table
 * puts the 24 bytes where the two cells want them and zeroes the rest, and one
 * store writes both cells.  The input the loop touches is 1.2 MB instead of the
 * 13 MB of NDHWC32 cells, which is the whole point.
 *
 * Index table: cell bytes are tl0-2 bl0-2 tr0-2 br0-2 (pos order tl,bl,tr,br),
 * tl/tr from row 2oy (lane 0, bytes 0-5 and 6-11 for the second cell) and bl/br
 * from row 2oy+1 (lane 1, from byte 32); 0x80 zeroes a lane. */
static void focus_rgb_packed(uint8_t *pd, const uint8_t *ps, int OH, int OW,
                             uint32_t irow, uint32_t orow)
{
    static const uint8_t idx[64] __attribute__((aligned(64))) = {
         0,  1,  2, 32, 33, 34,  3,  4,  5, 35, 36, 37,
        [12 ... 31] = 0x80,
         6,  7,  8, 38, 39, 40,  9, 10, 11, 41, 42, 43,
        [44 ... 63] = 0x80,
    };
    PLT_VLD64(8, idx);
    const int pairs = OW / 2;
    for (int oy = 0; oy < OH; oy++) {
        const uint8_t *ra = ps + (uint32_t)(2 * oy) * irow;
        uint8_t *o = pd + (uint32_t)oy * orow;
        if (pairs > 0) {
            register const uint8_t *a __asm__("t0") = ra;
            register const uint8_t *b __asm__("t1") = ra + irow;
            register uint8_t       *d __asm__("t2") = o;
            register int            n __asm__("t3") = pairs;
            __asm__ __volatile__(".set push\n\t.set noreorder\n\t.set noat\n\t"
                "1:\n\t"
                PLT_M3_LAO(0, 0, 8, 0) PLT_M3_LAO(0, 1, 9, 0)
                PLT_M3_OP(PLT_M3_GSHUFVB, 8, 0, 1)
                PLT_M3_SAO(1, 0, 10, 0) PLT_M3_SAO(1, 1, 10, 1)
                "addiu %[a], %[a], 12\n\t" "addiu %[b], %[b], 12\n\t"
                "addiu %[d], %[d], 64\n\t"
                "addiu %[n], %[n], -1\n\t" "bnez %[n], 1b\n\t" "nop\n\t"
                ".set pop\n\t"
                : [a] "+r"(a), [b] "+r"(b), [d] "+r"(d), [n] "+r"(n) :: "memory");
        }
        if (OW & 1) {                       /* an odd width's last cell */
            const uint8_t *pa = ra + (uint32_t)(OW - 1) * 6u, *pb = pa + irow;
            uint8_t *c = o + (uint32_t)(OW - 1) * 32u;
            const uint8_t cell[12] = { pa[0], pa[1], pa[2], pb[0], pb[1], pb[2],
                                       pa[3], pa[4], pa[5], pb[3], pb[4], pb[5] };
            for (int i = 0; i < 12; i++) c[i] = cell[i];
            for (int i = 12; i < 32; i++) c[i] = 0;
        }
    }
}

/* The same over an NDHWC16 output (plt_engine_load picks it when every reader
 * knows 16-byte cells): four output cells per register, from 24 packed bytes
 * of each input row.  Half the output bytes of NDHWC32 cells, and the reader
 * feeds from half the lines. */
static void focus_rgb_packed16(uint8_t *pd, const uint8_t *ps, int OH, int OW,
                               uint32_t irow, uint32_t orow)
{
    static uint8_t idx[64] __attribute__((aligned(64)));
    for (int k = 0; k < 4; k++) {
        const uint8_t cell[16] = {
            6 * k, 6 * k + 1, 6 * k + 2, 32 + 6 * k, 33 + 6 * k, 34 + 6 * k,
            6 * k + 3, 6 * k + 4, 6 * k + 5, 35 + 6 * k, 36 + 6 * k, 37 + 6 * k,
            0x80, 0x80, 0x80, 0x80 };
        for (int i = 0; i < 16; i++) idx[16 * k + i] = cell[i];
    }
    PLT_VLD64(8, idx);
    const int quads = OW / 4;
    for (int oy = 0; oy < OH; oy++) {
        const uint8_t *ra = ps + (uint32_t)(2 * oy) * irow;
        uint8_t *o = pd + (uint32_t)oy * orow;
        if (quads > 0) {
            register const uint8_t *a __asm__("t0") = ra;
            register const uint8_t *b __asm__("t1") = ra + irow;
            register uint8_t       *d __asm__("t2") = o;
            register int            n __asm__("t3") = quads;
            __asm__ __volatile__(".set push\n\t.set noreorder\n\t.set noat\n\t"
                "1:\n\t"
                PLT_M3_LAO(0, 0, 8, 0) PLT_M3_LAO(0, 1, 9, 0)
                PLT_M3_OP(PLT_M3_GSHUFVB, 8, 0, 1)
                PLT_M3_SAO(1, 0, 10, 0) PLT_M3_SAO(1, 1, 10, 1)
                "addiu %[a], %[a], 24\n\t" "addiu %[b], %[b], 24\n\t"
                "addiu %[d], %[d], 64\n\t"
                "addiu %[n], %[n], -1\n\t" "bnez %[n], 1b\n\t" "nop\n\t"
                ".set pop\n\t"
                : [a] "+r"(a), [b] "+r"(b), [d] "+r"(d), [n] "+r"(n) :: "memory");
        }
        for (int ox = quads * 4; ox < OW; ox++) {     /* a width that is not 4k */
            const uint8_t *pa = ra + (uint32_t)ox * 6u, *pb = pa + irow;
            uint8_t *c = o + (uint32_t)ox * 16u;
            const uint8_t cell[12] = { pa[0], pa[1], pa[2], pb[0], pb[1], pb[2],
                                       pa[3], pa[4], pa[5], pb[3], pb[4], pb[5] };
            for (int i = 0; i < 12; i++) c[i] = cell[i];
            for (int i = 12; i < 16; i++) c[i] = 0;
        }
    }
}

static int focus_run(plt_ctx_t *ctx, const plt_node_t *node)
{
    const plt_tensor_t *in = &node->in[0], *out = &node->out;
    const int Cin = in->shape.c;
    const int OH = out->shape.h, OW = out->shape.w;

    const volatile uint8_t *src = plt_ctx_ptr(ctx, in->mem);
    volatile uint8_t *dst = plt_ctx_ptr(ctx, out->mem);
    const volatile uint8_t *tbl = plt_ctx_ptr(ctx, node->table);
    if (!src || !dst || !tbl) return plt_ctx_fail(ctx, "focus: layer %u unmapped", node->rec->id);
    uint8_t lut[256]; for (int i = 0; i < 256; i++) lut[i] = tbl[i];
    plt_requant_t r; plt_requant_of(tbl, 1, 0, &r);
    int32_t cst[PLT_RQ_NCONST * 16] __attribute__((aligned(64)));
    plt_requant_consts(&r, cst);
    plt_requant_load(cst);

    const uint32_t irow = plt_ndhwc32_row_bytes(in->shape.w);
    const uint32_t orow = plt_ndhwc32_row_bytes(OW), oplane = plt_ndhwc32_plane_bytes(OH, OW);

    /* Cached arena: read src and write dst in place -- no bounce buffers.  Cin<=32
     * so the input is one plane; padding output lanes (Cout not a multiple of 32)
     * are read only by zero weights downstream, so they are left untouched. */
    const uint8_t *ps = (const uint8_t *)src;
    uint8_t       *pd = (uint8_t *)dst;
    const uint32_t t0 = plt_prof_now(ctx);

    if (Cin == 3 && in->layout == PLT_FMT_NHWC && out->layout == PLT_FMT_NDHWC16) {
        if (r.kind != PLT_RQ_IDENTITY)
            return plt_ctx_fail(ctx, "focus: an NDHWC16 output carries no requant");
        focus_rgb_packed16(pd, ps, OH, OW, (uint32_t)in->shape.w * 3u,
                           (uint32_t)plt_pad_w(OW) * 16u);
    } else if (Cin == 3 && in->layout == PLT_FMT_NHWC) {
        focus_rgb_packed(pd, ps, OH, OW, (uint32_t)in->shape.w * 3u, orow);
        if (r.kind != PLT_RQ_IDENTITY)
            for (int oy = 0; oy < OH; oy++)
                plt_requant_run(pd + (uint32_t)oy * orow, pd + (uint32_t)oy * orow,
                                (uint32_t)OW * 32u, &r, lut);
    } else
    if (Cin == 3) {
        focus_rgb(pd, ps, OH, OW, irow, orow);
        /* The scale change, if there is one, is a second arithmetic pass over
         * what was just written -- it also touches the cells' unused lanes,
         * which nothing downstream reads (the stem's weights there are zero). */
        if (r.kind != PLT_RQ_IDENTITY)
            for (int oy = 0; oy < OH; oy++)
                plt_requant_run(pd + (uint32_t)oy * orow, pd + (uint32_t)oy * orow,
                                (uint32_t)OW * 32u, &r, lut);
    } else {
        for (int oy = 0; oy < OH; oy++)
            for (int ox = 0; ox < OW; ox++)
                for (int pos = 0; pos < 4; pos++) {
                    const int iy = 2 * oy + DY[pos], ix = 2 * ox + DX[pos];
                    const uint8_t *ic = ps + iy * irow + ix * 32;   /* group 0, c<32 */
                    for (int c = 0; c < Cin; c++) {
                        const int oc = pos * Cin + c;
                        pd[(oc / 32) * oplane + oy * orow + ox * 32 + (oc % 32)] =
                            plt_requant_byte(&r, lut, ic[c]);
                    }
                }
    }
    plt_prof_mark(ctx, &ctx->prof.tail_us, t0);
    return 0;
}

const plt_kernel_t plt_kernel_focus = { PLT_OP_FOCUS, PLT_EX_MXU_FOCUS, "focus", focus_plan, focus_run };
