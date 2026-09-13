/* NNDMA: descriptor-driven DDR<->ORAM movement. Descriptor is two 32-bit words
 * (manual 6.3):
 *   lo = ddr_physical | more
 *   hi = ((bytes/64)-1)<<20 | ((0x20000 + oram_offset)/64)
 * The ORAM field is formed by ADDITION, never OR. A channel is kicked by
 * writing (index | 0x80000000) to its IO register; completion is a blocking
 * rdhwr on the channel ($8 read0, $9 read1, $11 write). */
#ifndef PLT_HAL_DMA_H
#define PLT_HAL_DMA_H

#include <stdint.h>
#include "plt_dev.h"

#define PLT_IO_READ0 0x00u
#define PLT_IO_READ1 0x04u
#define PLT_IO_WRITE 0x10u

void plt_dma_desc(plt_dev_t *d, unsigned index, uint32_t ddr_phys,
                  uint32_t oram_off, uint32_t bytes, int more);
void plt_dma_kick(plt_dev_t *d, uint32_t io_off, unsigned index);
void plt_dma_wait(plt_dev_t *d, uint32_t io_off);

#endif /* PLT_HAL_DMA_H */
