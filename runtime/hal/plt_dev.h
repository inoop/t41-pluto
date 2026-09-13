/* Device layer: open /dev/soc-nna (enables COP2 + clocks on the calling core),
 * mmap the DESRAM / IO / ORAM windows and the nmem tensor pool, pin the CPU
 * (before open -- hazard 10.2), and cache-sync nmem buffers. Facts from
 * T41_NNA_MANUAL.md 6 and the T4x soc-nna driver. */
#ifndef PLT_HAL_DEV_H
#define PLT_HAL_DEV_H

#include <stdint.h>
#include <stddef.h>

#define PLT_NMEM_PHYS  0x07000000u
#define PLT_NMEM_SIZE  0x01000000u   /* 16 MB reserved pool (nmem= on cmdline) */
#define PLT_ORAM_PHYS  0x12620000u
#define PLT_ORAM_SIZE  0x00060000u   /* 384 KB usable */
#define PLT_DESRAM_PHYS 0x12500000u
#define PLT_DESRAM_SIZE 0x00004000u
#define PLT_IO_PHYS    0x12508000u
#define PLT_IO_SIZE    0x00001000u

#define PLT_DMA_TO_DEVICE   1
#define PLT_DMA_FROM_DEVICE 2

typedef struct {
    volatile uint8_t *nmem;   /* uncached /dev/mem view of the tensor pool  */
    volatile uint8_t *oram;   /* on-chip operand/staging SRAM (driver mmap) */
    volatile uint8_t *des;    /* DMA descriptor RAM (uncached /dev/mem)     */
    volatile uint8_t *io;     /* NNDMA kick/status registers                */
    int fd_nna;
    int fd_mem;
} plt_dev_t;

int  plt_dev_open(plt_dev_t *d);
void plt_dev_close(plt_dev_t *d);
int  plt_dev_cache_sync(plt_dev_t *d, uint32_t phys, uint32_t len, int dir);
void plt_word_copy(volatile void *dst, const void *src, size_t n);  /* 32-bit stores */

/* Window-to-window bulk copy.  Both windows are mapped UNCACHED, so moving them
 * a byte at a time costs one bus transaction per byte; this moves 64 bytes per
 * vector load/store and falls back to 32-bit words for the tail.  `n` and both
 * pointers should be 4-byte aligned. */
void plt_copy_fast(volatile void *dst, const volatile void *src, size_t n);

/* Fill a window region, 32 bits at a time. */
void plt_fill_fast(volatile void *dst, uint8_t value, size_t n);

/* Microseconds, for profiling only. */
uint32_t plt_now_us(void);

#endif /* PLT_HAL_DEV_H */
