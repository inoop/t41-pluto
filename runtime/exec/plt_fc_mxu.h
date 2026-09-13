/* Int8 fully-connected.  Runs on the CPU/MXU3, not the array.  As with the
 * pool, the scalar code here is the reference the vector code must match.
 */
#ifndef PLT_EXEC_FC_MXU_H
#define PLT_EXEC_FC_MXU_H

#include <stdint.h>

#include "core/plt_kernel.h"

extern const plt_kernel_t plt_kernel_fc_mxu;

/* logits[co] = dequant( quantize( in_scale*w_scale[co] * (sum(in*w) - in_zp*sum(w) + bias) ) ).
 * w is [Cout][Cin] int8, row-major, matching ONNX Gemm transB=1. */
void plt_fc_int8(const int8_t *in, int Cin, const int8_t *w, const int32_t *bias,
                 const float *w_scale, int Cout, float in_scale, int in_zp,
                 float out_scale, int out_zp, float *logits);

#endif /* PLT_EXEC_FC_MXU_H */
