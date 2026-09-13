/* MXU3 instruction encodings for the pluto runtime.
 *
 * MXU3 has 32 x 512-bit vector registers moved as two 256-bit halves. These
 * are the only MXU3 ops the HAL needs for NNA bring-up: a 64-byte aligned
 * load/store (base in $t0) and the config-readback half-store family
 * (SA0_VPR, manual 12.5 -- the sa_x form returns stale data and must not be
 * used for readback). Word encodings are hardware facts from the kernel's
 * mxuv3.c and T41_NNA_MANUAL.md 12.2/12.5. */
#ifndef PLT_HAL_ISA_MXU_H
#define PLT_HAL_ISA_MXU_H

#include <stdint.h>

/* Load 64 bytes at `src` into vector register `vr` (two 256-bit halves). */
#define PLT_VLD64(vr, src) do {                                             \
    register const void *_b __asm__("t0") = (src);                          \
    __asm__ __volatile__(".set push\n.set noreorder\n.word %1\n.word %2\n"  \
        ".set pop\n" :: "r"(_b),                                            \
        "i"(0x71001811u | (0u << 16) | (0u << 14) | ((vr) << 6)),           \
        "i"(0x71001811u | (1u << 16) | (1u << 14) | ((vr) << 6))            \
        : "memory"); } while (0)

/* Store vector register `vr` (64 bytes) to `dst` via the SA0_VPR family. */
#define PLT_VST64(vr, dst) do {                                             \
    register void *_b __asm__("t0") = (dst);                                \
    __asm__ __volatile__(".set push\n.set noreorder\n.word %1\n.word %2\n"  \
        ".set pop\n" :: "r"(_b),                                            \
        "i"(0x710000d5u | (0u << 16) | ((vr) << 11) | (0u << 9)),           \
        "i"(0x710000d5u | (1u << 16) | ((vr) << 11) | (1u << 9))            \
        : "memory"); } while (0)


/* --- the int8 dot-product core -------------------------------------------
 *
 * MXU3 has four 512-bit accumulators, vs0..vs3, each read as 16 x int32.
 * `s16macssb` multiplies two registers of 64 signed bytes and adds the WHOLE
 * 64-element sum into one accumulator word: a 64-element int8 dot product per
 * instruction.  It is present on both MXU3.0 and MXU3.1.  (So, it turns out,
 * is a data-dependent byte gather -- not 3.0's `gshufb`, which SIGILLs here,
 * but 3.1's `gshufvb`; see PLT_M3_GSHUFVB in plt_mxu3.h.)
 *
 * Field positions come from Ingenic's own binutils table (mxu3/mips-opc.c) and
 * the SLEIGH spec generated from it: VPR at bit 6 / 11 / 16, VSR at 6 / 11,
 * the accumulator word index at bits 2-5.  The two vector sources are
 * multiplied elementwise, so their order does not matter.
 */
#define PLT_MXU_EMIT(word) \
    __asm__ __volatile__(".set push\n.set noreorder\n.word %0\n.set pop\n" \
                         :: "i"(word) : "memory")

/** Zero accumulator `vs`. */
#define PLT_MXU_SUMZ(vs)          PLT_MXU_EMIT(0x4a60001cu | ((vs) << 6))

/** Copy accumulator `vs` into vector register `vr` (16 x int32). */
#define PLT_MXU_MFSUM(vr, vs)     PLT_MXU_EMIT(0x4a60000fu | ((vr) << 6) | ((vs) << 11))

/** vs[word] += sum over all 64 signed byte lanes of vra * vrb. */
#define PLT_MXU_MACSSB16(vs, word, vra, vrb) \
    PLT_MXU_EMIT(0x4bc00702u | ((word) << 2) | ((vs) << 6) | ((vrb) << 11) | ((vra) << 16))

#endif /* PLT_HAL_ISA_MXU_H */
