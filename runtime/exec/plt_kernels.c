/* The executor table.  One (op, executor) pair maps to exactly one function;
 * adding a layer is one line here.  A static table, deliberately: no linker
 * sections, no constructors, and `grep` finds every executor in the build. */
#include <stddef.h>

#include "core/plt_kernel.h"
#include "exec/plt_conv_dense.h"
#include "exec/plt_conv_k3.h"
#include "exec/plt_conv_dw.h"
#include "exec/plt_conv_pw.h"
#include "exec/plt_conv_stem.h"
#include "exec/plt_fc_mxu.h"
#include "exec/plt_pool_mxu.h"
#include "exec/plt_silu.h"
#include "exec/plt_add.h"
#include "exec/plt_concat.h"
#include "exec/plt_upsample.h"
#include "exec/plt_maxpool.h"
#include "exec/plt_focus.h"
#include "exec/plt_conv_std.h"

static const plt_kernel_t *const KERNELS[] = {
    &plt_kernel_conv_pw,
    &plt_kernel_conv_dw,
    &plt_kernel_conv_dense,
    &plt_kernel_conv_k3,
    &plt_kernel_conv_stem,
    &plt_kernel_pool_mxu,
    &plt_kernel_fc_mxu,
    &plt_kernel_silu,
    &plt_kernel_add,
    &plt_kernel_concat,
    &plt_kernel_upsample,
    &plt_kernel_maxpool,
    &plt_kernel_focus,
    &plt_kernel_conv_std,
    NULL
};

const plt_kernel_t *const *plt_kernel_all(void)
{
    return KERNELS;
}

const plt_kernel_t *plt_kernel_find(uint8_t op, uint8_t exec)
{
    for (int i = 0; KERNELS[i]; i++)
        if (KERNELS[i]->op == op && KERNELS[i]->exec == exec) return KERNELS[i];
    return NULL;
}
