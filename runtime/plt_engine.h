/* The graph walker: turn a loaded .pluto model into a sequence of executor
 * calls, with the activations placed once in the nmem arena.
 *
 * The order is deliberate.  Every node is planned before any node runs, which
 * is what makes a whole-graph memory layout possible -- and it means a model
 * that cannot fit, or that names an executor this build does not have, fails at
 * load time with a message naming the layer, rather than part-way through an
 * inference.
 */
#ifndef PLT_ENGINE_H
#define PLT_ENGINE_H

#include <stdint.h>
#include <stdio.h>

#include "core/plt_ctx.h"
#include "core/plt_kernel.h"
#include "core/plt_model.h"

#define PLT_MAX_LAYERS 256

typedef struct {
    plt_ctx_t          *ctx;
    plt_model_t         model;
    int                 n;                        /* layers                   */
    plt_node_t          node[PLT_MAX_LAYERS];
    plt_plan_t          plan[PLT_MAX_LAYERS];
    const plt_kernel_t *kernel[PLT_MAX_LAYERS];
    uint64_t            chk[PLT_MAX_LAYERS];      /* FNV-1a of each output    */
    uint32_t            us[PLT_MAX_LAYERS];       /* microseconds per layer   */
    plt_prof_t          phase[PLT_MAX_LAYERS];    /* and where they went      */
    plt_mem_t           input;                    /* the graph's input tensor */
    /* The input is held packed, three bytes a pixel row-major, rather than as
     * NDHWC32 cells: see plt_engine_load.  plt_engine_input accepts either. */
    int                 input_compact;
    uint32_t            input_cells_bytes;        /* its NDHWC32 size          */
    uint32_t            input_packed_bytes;       /* h * w * 3                 */
    uint8_t            *luts;                     /* composed tables, 256 B each */
    uint8_t             composed[PLT_MAX_LAYERS]; /* output is in a concat's quantization */
} plt_engine_t;

/* Load the model, resolve every node, find every executor, plan, and place the
 * activations.  0 on success; ctx->err says why not. */
int plt_engine_load(plt_ctx_t *ctx, const char *model_path, plt_engine_t *eng);
void plt_engine_free(plt_engine_t *eng);

/* Write the graph's input.  `bytes` must match the input tensor's allocation. */
int plt_engine_input(plt_engine_t *eng, const void *data, uint32_t bytes);

/* Run layers [first, last] inclusive; -1 for either end means the whole graph.
 * Checksums and timings are filled in when the context asks for them; they are
 * diagnostics and never part of an inference. */
int plt_engine_run(plt_engine_t *eng, int first, int last);

/* FNV-1a over one tensor, for comparing a device run against the simulator. */
uint64_t plt_engine_checksum(const plt_engine_t *eng, int layer);

void plt_engine_report(const plt_engine_t *eng, FILE *out);

#endif /* PLT_ENGINE_H */
