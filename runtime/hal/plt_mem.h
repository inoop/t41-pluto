/* A bump allocator over one mapped window.
 *
 * The runtime has no general heap on the device side: a graph's tensors are
 * placed once, in order, and freed all at once.  An arena hands back a
 * plt_mem_t rather than a pointer, so that a buffer's SPACE travels with it and
 * the caller cannot silently mix an ORAM offset with an nmem one.
 */
#ifndef PLT_HAL_MEM_H
#define PLT_HAL_MEM_H

#include <stdint.h>

/* Where a buffer lives.  Only DMA reads a mapped window directly, so anything
 * the CPU alone touches belongs in cached memory: an uncached 64-byte load
 * measured 194 ns against 2.8 ns. */
typedef enum {
    PLT_SPACE_HOST = 0,   /* ordinary cached memory the context owns   */
    PLT_SPACE_MODEL,      /* ordinary cached memory: the loaded model   */
    PLT_SPACE_NMEM,       /* the reserved DDR tensor pool, uncached     */
    PLT_SPACE_ORAM        /* on-chip operand SRAM, uncached            */
} plt_space_t;

/* A reference to a buffer: where it lives, not a pointer to it.  Resolve it
 * with plt_ctx_ptr().  bytes == 0 means "absent". */
typedef struct {
    plt_space_t space;
    uint32_t    off;
    uint32_t    bytes;
} plt_mem_t;

static inline plt_mem_t plt_mem_none(void)
{
    plt_mem_t m = { PLT_SPACE_HOST, 0, 0 };
    return m;
}

static inline int plt_mem_present(plt_mem_t m) { return m.bytes != 0; }

typedef struct {
    plt_space_t space;
    uint32_t    size;    /* bytes in the window */
    uint32_t    used;    /* bytes handed out    */
} plt_arena_t;

void plt_arena_init(plt_arena_t *a, plt_space_t space, uint32_t size);

/* Hand out `bytes`, aligned up to `align` (a power of two, at least 64 for
 * anything the array touches).  Returns an absent ref if the arena is full --
 * check with plt_mem_present(). */
plt_mem_t plt_arena_alloc(plt_arena_t *a, uint32_t bytes, uint32_t align);

/* Would this allocation fit?  For planning, before anything is handed out. */
int plt_arena_fits(const plt_arena_t *a, uint32_t bytes, uint32_t align);

void plt_arena_reset(plt_arena_t *a);

#endif /* PLT_HAL_MEM_H */
