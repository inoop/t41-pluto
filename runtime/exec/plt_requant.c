#include "exec/plt_requant.h"
#include <string.h>
#include "hal/plt_dev.h"

/* Requantize a contiguous run.  See plt_requant.h for why this is arithmetic
 * rather than a table: the vector unit cannot do a table at all. */
void plt_requant_load(const int32_t *cst)
{
    register const int32_t *pc __asm__("t4") = cst;
    __asm__ __volatile__(".set push\n\t.set noreorder\n\t"
        PLT_RQ_LOAD_CONSTS(12)
        ".set pop\n\t" : "+r"(pc) :: "memory");
}

void plt_requant_run(uint8_t *dst, const uint8_t *src, uint32_t n,
                     const plt_requant_t *r, const uint8_t lut[256])
{
    uint32_t i = 0;

    if (r->kind == PLT_RQ_IDENTITY) {         /* the layer is pure movement */
        if (dst != src) plt_copy_fast(dst, src, n);   /* 256 B a step */
        return;
    }

    if (r->kind == PLT_RQ_AFFINE && n >= 64) {
        uint32_t cnt = n >> 6;
        i = cnt << 6;
        {
            register const uint8_t *s __asm__("t0") = src;
            register uint8_t       *d __asm__("t1") = dst;
            __asm__ __volatile__(
                ".set push\n\t.set noreorder\n\t.set noat\n\t"
                "1:\n\t"
                PLT_M3_LUO(0, 0, 8) PLT_M3_LUO(0, 1, 8)     /* 64 bytes, t0 walks */
                PLT_RQ_APPLY64(0)
                PLT_M3_SAO(0, 0, 9, 0) PLT_M3_SAO(0, 1, 9, 1)
                "addiu %[d], %[d], 64\n\t"
                "addiu %[c], %[c], -1\n\t"
                "bnez %[c], 1b\n\t"
                "nop\n\t"
                ".set pop\n\t"
                : [s] "+r"(s), [d] "+r"(d), [c] "+r"(cnt) :: "memory");
        }
    }
    for (; i < n; i++) dst[i] = plt_requant_byte(r, lut, src[i]);
}
