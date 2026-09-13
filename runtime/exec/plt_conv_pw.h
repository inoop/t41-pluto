/* 1x1 int8 convolution on the NNA, stride 1, no padding.
 *
 * The recipe, and the reasoning behind every field, is in docs/NNA_POINTWISE.md.
 * All four operands are read and written with plain vector loads and stores, so
 * any mapped window will do -- ORAM for speed, the nmem pool when a layer's
 * operands do not fit on chip.  The output is the array's hardware-requantized
 * packed byte, p = y_q + 128.
 *
 * Operand layouts:
 *   weights  packed bit-plane tiles, 1024 B per (output group, input group),
 *            output-group major -- compile/layout.py:pack_pointwise_int8
 *   table    256 B per output group: int32 bias[32] then int32 mult[32]
 *   in, out  NDHWC32 planes PADDED to plt_pad_h() x plt_pad_w()
 * All four must be 64-byte aligned.
 *
 * Constraints: cin and cout are multiples of 32, and cin >= 64 -- a single
 * input group is served by a different recipe, so the compiler pads to two.
 */
#ifndef PLT_EXEC_CONV_PW_H
#define PLT_EXEC_CONV_PW_H

#include "core/plt_kernel.h"
#include "exec/plt_conv_nna.h"

extern const plt_kernel_t plt_kernel_conv_pw;

/* The recipe itself, for the one other executor that is a 1x1 convolution
 * underneath: the RGB stem assembles a 32-tap window and then runs exactly
 * this (exec/plt_conv_stem.c, manual 8.6).  Configure once per layer, then run
 * the tiles.  Anything else should go through plt_kernel_conv_pw. */
void plt_conv_pw_configure(const plt_conv_geom_t *geom);
void plt_conv_pw_tiles(const plt_conv_geom_t *geom, const plt_conv_bufs_t *bufs);

/* Which words plt_conv_pw_configure() leaves live in the array's operand file.
 * An executor that writes its own tile loop needs them, because the per-tile
 * field writes name a word rather than carrying a value. */
enum {
    PLT_PW_W_ZERO       = 1,   /* 0, for the per-tile field writes           */
    PLT_PW_W_UNIT_A_OFF = 2,   /* -> B.10, unit A's operand start             */
    PLT_PW_W_UNIT_B_OFF = 3,   /* -> B.11, unit B's                          */
    PLT_PW_W_MAC_A      = 4,   /* the operand nnmac 0x81 names                */
    PLT_PW_W_MAC_B      = 5    /* the operand nnmac 0xa3 names                */
};

#endif /* PLT_EXEC_CONV_PW_H */
