/* Depthwise 3x3 on the NNA.
 *
 * A depthwise convolution does not reduce across channels, so it does not fit
 * the array's 32-into-32 dot product directly.  Read one 32-channel group as a
 * 1x1 convolution over its nine taps, though, and it does: 9 taps x 32 channels
 * = 288 pseudo-channels into 32 outputs, with a block-diagonal weight matrix.
 * A depthwise layer is therefore C/32 independent pointwise convolutions over
 * an assembled tap tensor -- the same device-verified recipe again, with no
 * second convolution engine and no unproven hardware mode.
 *
 * The cost is honest and worth stating: the array computes 288 products per
 * output where 9 are wanted, so the depthwise layers run at 1/32 of the array's
 * peak.  Feeding the taps also costs `KH*KW` times the input bandwidth.  The
 * alternative -- the vendor's native 3x3 walk (manual 8.3) -- would avoid both,
 * and is the obvious performance work once the network runs end to end.
 *
 * The tap order is `t = ky*KW + kx`, shared with compile/layout.py's
 * depthwise_tap_order() and pack_depthwise_int8().
 */
#ifndef PLT_EXEC_CONV_DW_H
#define PLT_EXEC_CONV_DW_H

#include "core/plt_kernel.h"

extern const plt_kernel_t plt_kernel_conv_dw;

/* Can this depthwise layer read and write NDHWC16 cells (plt_engine_load)? */
int plt_conv_dw_cell16_ok(const plt_node_t *node);

#endif /* PLT_EXEC_CONV_DW_H */
