/* The 3x3 RGB stem on the NNA.
 *
 * The array has no 3-channel mode, but a 3x3 window over 3 colour channels is
 * 27 values, which fits inside ONE 32-channel group.  So this executor
 * assembles the window into 32 pseudo-channels and then runs the ordinary 1x1
 * recipe over it (manual 8.6) -- the same code path that is device-verified for
 * every pointwise layer, rather than a second convolution engine.
 *
 * The tap order is pluto's own and is shared with the compiler:
 *
 *     tap t = (ky*3 + kx)*3 + c,   t in 0..26;  taps 27..31 are zero
 *
 * (compile/layout.py:stem_tap).  The compiler packs the weights into that
 * order, so nothing else in the tree needs to know it.
 */
#ifndef PLT_EXEC_CONV_STEM_H
#define PLT_EXEC_CONV_STEM_H

#include "core/plt_kernel.h"

extern const plt_kernel_t plt_kernel_conv_stem;

/* The array reads an activation byte as UNSIGNED.  Interior layers have an
 * input zero point of -128, so u = x_q - x_zp already lands in [0,255]; the
 * image's zero point is 0, so its u is signed and must be shifted.  The
 * compiler folds the whole correction into the requant bias, exactly, in
 * integers (compile/ops/_conv.py:requant_bytes).  Shifting a byte by 128 is
 * flipping its top bit. */
#define PLT_STEM_FEED_XOR 0x80u

#endif /* PLT_EXEC_CONV_STEM_H */
