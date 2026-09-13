#include "hal/plt_mem.h"

static uint32_t round_up(uint32_t value, uint32_t align)
{
    return (value + align - 1u) & ~(align - 1u);
}

void plt_arena_init(plt_arena_t *a, plt_space_t space, uint32_t size)
{
    a->space = space;
    a->size  = size;
    a->used  = 0;
}

int plt_arena_fits(const plt_arena_t *a, uint32_t bytes, uint32_t align)
{
    const uint32_t start = round_up(a->used, align);
    return (uint64_t)start + bytes <= a->size;
}

plt_mem_t plt_arena_alloc(plt_arena_t *a, uint32_t bytes, uint32_t align)
{
    const uint32_t start = round_up(a->used, align);
    plt_mem_t      m     = plt_mem_none();

    if ((uint64_t)start + bytes > a->size) return m;
    a->used  = start + bytes;
    m.space  = a->space;
    m.off    = start;
    m.bytes  = bytes;
    return m;
}

void plt_arena_reset(plt_arena_t *a)
{
    a->used = 0;
}
