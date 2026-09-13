#include "plt_dma.h"

void plt_dma_desc(plt_dev_t *d, unsigned index, uint32_t ddr_phys,
                  uint32_t oram_off, uint32_t bytes, int more)
{
    uint32_t lo = (ddr_phys & ~1u) | (more ? 1u : 0u);
    uint32_t hi = (((bytes / 64u) - 1u) << 20) | ((0x20000u + oram_off) / 64u);
    volatile uint32_t *slot = (volatile uint32_t *)(d->des + (size_t)index * 8u);
    slot[0] = lo;
    slot[1] = hi;
    __asm__ __volatile__("sync" ::: "memory");
}

void plt_dma_kick(plt_dev_t *d, uint32_t io_off, unsigned index)
{
    volatile uint32_t *reg = (volatile uint32_t *)(d->io + io_off);
    __asm__ __volatile__("sync" ::: "memory");
    (void)*(volatile uint32_t *)d->nmem;                 /* fence deposited data */
    *reg = index | 0x80000000u;
    __asm__ __volatile__("sync" ::: "memory");
    (void)*reg;                                          /* completion barrier   */
}

/* Blocking idle read: rdhwr $2, $rd (rd = 8/9/11 for read0/read1/write).
 * rd must be a compile-time immediate, so emit one fixed word per channel. */
#define PLT_RDHWR(rd) __asm__ __volatile__( \
    ".set push\n.set noreorder\nsync\n.word %0\n.set pop\n" \
    :: "i"(0x7c00003bu | (2u << 16) | ((rd) << 11)) : "memory")

void plt_dma_wait(plt_dev_t *d, uint32_t io_off)
{
    volatile uint32_t *reg = (volatile uint32_t *)(d->io + io_off);
    __asm__ __volatile__("sync" ::: "memory");
    (void)*reg;
    __asm__ __volatile__("sync" ::: "memory");
    if (io_off == PLT_IO_READ0)      PLT_RDHWR(8u);
    else if (io_off == PLT_IO_READ1) PLT_RDHWR(9u);
    else                             PLT_RDHWR(11u);
}
