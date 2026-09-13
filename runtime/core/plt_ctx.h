/* The runtime context: one object owns the device and the arenas, and one place
 * reports errors.
 *
 * The array is a single global machine, so the runtime is single-threaded by
 * construction and one process at a time holds the device.  What the context
 * buys is that no module keeps hidden state of its own -- everything an
 * executor needs is reachable from the context it was handed.
 */
#ifndef PLT_CORE_CTX_H
#define PLT_CORE_CTX_H

#include <stdint.h>

#include "core/plt_tensor.h"
#include "hal/plt_dev.h"
#include "hal/plt_mem.h"

/* Context flags. */
#define PLT_F_PROFILE  0x1    /* per-layer timing            */
#define PLT_F_CHECKSUM 0x2    /* per-layer output checksums  */
#define PLT_F_VERBOSE  0x4
#define PLT_F_NOREUSE  0x8    /* one activation buffer per layer */
#define PLT_F_LIVENESS 0x10   /* reuse buffers by liveness (a branchy graph, small arena) */
#define PLT_F_NOVIEWS  0x20   /* liveness without concat views or in-place writes */

/* Where a run's time goes.  The two halves are worth separating because they
 * have different fixes: `assemble` is the CPU rearranging a tensor before the
 * array can read it, and `array` is the CPU feeding the array and draining it.
 * Neither is "the NNA computing while the CPU waits" -- this runtime drives the
 * array synchronously, so the CPU is busy throughout. */
typedef struct {
    uint32_t assemble_us;   /* window assembly and operand padding */
    uint32_t array_us;      /* configure, feed, MAC, drain         */
    uint32_t tail_us;       /* the CPU kernels: pool and FC        */
} plt_prof_t;

typedef struct {
    plt_dev_t   dev;
    plt_arena_t oram;         /* the on-chip window     */
    plt_arena_t nmem;         /* the reserved DDR pool   */
    /* Two cached-memory bases, because a mapped window is uncached and an
     * uncached 64-byte load measured 194 ns against 2.8 ns:
     * anything only the CPU touches belongs here rather than in nmem, and in
     * this runtime that is everything, since only DMA reads a window directly.
     *
     *   host_base   the activation arena, owned by the context
     *   model_base  the loaded model image, borrowed; the weight and requant
     *               blobs are fed straight out of it with no copy
     */
    plt_arena_t host;
    uint8_t    *host_base;
    uint8_t    *model_base;
    uint8_t    *scratch;      /* shared working buffer; see below */
    uint32_t    scratch_bytes;
    plt_prof_t  prof;         /* accumulated only when PLT_F_PROFILE is set */
    int         flags;
    char        err[128];
} plt_ctx_t;

/* `scratch` is one buffer shared by every executor that asked for one, sized to
 * the largest request in the graph.  Its contents are undefined on entry to a
 * layer and meaningless after it returns, so nothing may carry state in it.
 *
 * It is ORDINARY CACHED MEMORY, not a mapped window, and that is the point.
 * The array is fed from a vector register: the CPU loads 64 bytes and pushes
 * them with `nndwr`, so the array never reads the feed source itself.  Sourcing
 * a feed from nmem or ORAM therefore buys nothing and costs an uncached load
 * per push -- about 224 ns each, which was most of the frame.  Measured: making
 * the depthwise feed source cached took the network from 155 ms to 106 ms.
 *
 * (ORAM was tried too, and is no faster than nmem for this: both are uncached
 * windows, so the cost is the access path, not the distance.) */

/* Open the device, map the windows, and start both arenas empty.
 * 0 on success; on failure ctx->err says why. */
int  plt_ctx_open(plt_ctx_t *ctx);

/* Back the activation arena with `bytes` of ordinary cached memory. */
int  plt_ctx_reserve_host(plt_ctx_t *ctx, uint32_t bytes);

void plt_ctx_close(plt_ctx_t *ctx);

/* The shared scratch buffer, or NULL if it is smaller than `bytes`. */
uint8_t *plt_ctx_scratch(const plt_ctx_t *ctx, uint32_t bytes);

/* Make the shared scratch at least `bytes` long.  The engine sizes it once from
 * the plans; a standalone tool can call this directly. */
int plt_ctx_reserve_scratch(plt_ctx_t *ctx, uint32_t bytes);

/* Resolve a buffer reference to a pointer into the right window.  NULL for an
 * absent ref or one whose space this context does not map. */
volatile uint8_t *plt_ctx_ptr(const plt_ctx_t *ctx, plt_mem_t mem);

/* Phase timing.  Both are no-ops unless PLT_F_PROFILE is set, so an executor can
 * bracket its phases unconditionally:
 *
 *     uint32_t t = plt_prof_now(ctx);
 *     assemble_the_window();
 *     t = plt_prof_mark(ctx, &ctx->prof.assemble_us, t);
 *     drive_the_array();
 *         plt_prof_mark(ctx, &ctx->prof.array_us, t);
 */
static inline uint32_t plt_prof_now(const plt_ctx_t *ctx)
{
    return (ctx->flags & PLT_F_PROFILE) ? plt_now_us() : 0;
}

static inline uint32_t plt_prof_mark(plt_ctx_t *ctx, uint32_t *bucket, uint32_t since)
{
    uint32_t now;
    if (!(ctx->flags & PLT_F_PROFILE)) return 0;
    now = plt_now_us();
    *bucket += now - since;
    return now;
}

/* Record a failure and return -1, so a caller can `return plt_ctx_fail(...)`. */
int plt_ctx_fail(plt_ctx_t *ctx, const char *fmt, ...);

#endif /* PLT_CORE_CTX_H */
