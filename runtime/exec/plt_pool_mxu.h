/* Global average pool.  Runs on the CPU, not the array: one int32 sum per
 * channel over the NDHWC32 tensor in place (plt_pool_mxu.c). */
#ifndef PLT_EXEC_POOL_MXU_H
#define PLT_EXEC_POOL_MXU_H

#include <stdint.h>

#include "core/plt_kernel.h"

extern const plt_kernel_t plt_kernel_pool_mxu;

#endif /* PLT_EXEC_POOL_MXU_H */
