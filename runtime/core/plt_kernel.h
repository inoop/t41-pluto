/* The layer contract: what an executor is handed, and what it must provide.
 *
 * One (op, executor) pair maps to exactly one executor function.  Adding a
 * layer means writing that function and adding one line to the table in
 * exec/plt_kernels.c -- see docs/ADDING_A_LAYER.md.
 */
#ifndef PLT_CORE_KERNEL_H
#define PLT_CORE_KERNEL_H

#include "core/plt_ctx.h"
#include "core/plt_model.h"
#include "core/plt_tensor.h"

/* One layer with everything resolved: no id lookups, no blob arithmetic and no
 * format decoding left for the executor to do. */
typedef struct {
    const plt_layer_rec_t *rec;          /* the on-disk descriptor, for geometry */
    plt_tensor_t           in[PLT_MAX_IO];
    int                    n_in;
    plt_tensor_t           out;
    plt_mem_t              weights;      /* already resolved out of the blob     */
    plt_mem_t              table;        /* the requant table, likewise          */

    /* Set by the engine's memory planner, never by the model.
     *
     * act_lut_override: the byte table an NNA convolution applies to every
     * drained byte INSTEAD of its own folded activation (or instead of none).
     * The planner sets it when this layer's output is written straight into a
     * concat's output planes: the table is then the layer's activation composed
     * with the concat's rescale for that input, so the bytes land already in
     * the concat's quantization and the concat has nothing left to do for it.
     *
     * in_view_mask: bit s set means input s of this concat was written in place
     * by its producer (see above) and must not be copied. */
    const uint8_t         *act_lut_override;
    uint32_t               in_view_mask;
} plt_node_t;

/* What a node needs, before anything has been allocated. */
typedef struct {
    uint32_t scratch_bytes;              /* transient working memory       */
    uint32_t out_bytes;                  /* including any layout padding   */
    int      out_pad_h, out_pad_w;
} plt_plan_t;

typedef struct {
    uint8_t     op;                      /* enum plt_op   */
    uint8_t     exec;                    /* enum plt_exec */
    const char *name;

    /* Report what this node needs.  MUST NOT touch the array: the engine calls
     * every node's plan() before it runs any node's run(), which is what makes
     * a whole-graph arena layout possible. */
    int (*plan)(plt_ctx_t *ctx, plt_node_t *node, plt_plan_t *plan);

    /* Execute it.  0 on success, negative with ctx->err set on failure. */
    int (*run)(plt_ctx_t *ctx, const plt_node_t *node);
} plt_kernel_t;

/* The executor that claims this pair, or NULL if none does. */
const plt_kernel_t *plt_kernel_find(uint8_t op, uint8_t exec);

/* Every registered executor, for diagnostics.  Terminated by a NULL entry. */
const plt_kernel_t *const *plt_kernel_all(void);

#endif /* PLT_CORE_KERNEL_H */
