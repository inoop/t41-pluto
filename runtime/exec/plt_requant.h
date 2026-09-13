/* The scale change a data-moving layer applies, in a form the vector unit can
 * evaluate.
 *
 * concat, focus, upsample and maxpool all do the same thing to every byte they
 * touch: map it from the input's quantisation to the output's.  Expressed as a
 * 256-entry table that is one scalar load per byte, and MXU3 has no
 * data-dependent gather, so it cannot be vectorised -- it was the reason these
 * four layers ran at a fraction of copy speed.
 *
 * But the table is never arbitrary.  The compiler (compile/requant.py) emits a
 * DESCRIPTOR beside it saying which of three forms it takes:
 *
 *   IDENTITY  a no-op.  The layer is pure data movement; skip the pass entirely.
 *   AFFINE    out = clamp(((C*b + T0) >> K) + zo + 128, 0, 255), verified by the
 *             compiler against all 256 table entries.  64 bytes to a pass.
 *   TABLE     neither -- the scalar lookup is the only correct path.
 *
 * `T0` folds the input zero point and the rounding term into one accumulator
 * seed, so the per-lane work is a multiply, an add, a shift, an add and two
 * clamps.  The compiler's range check is exactly the one evaluated here:
 * |C|*255 + |C|*|off| + 2^(K-1) stays inside int32.
 *
 * Descriptors sit after ALL of a layer's tables, so `table + i*256` still
 * indexes the tables unchanged.
 */
#ifndef PLT_EXEC_REQUANT_H
#define PLT_EXEC_REQUANT_H

#include <stdint.h>

enum { PLT_RQ_TABLE = 0, PLT_RQ_IDENTITY = 1, PLT_RQ_AFFINE = 2 };

typedef struct { int32_t kind, c, k, off, zo; } plt_requant_t;

/** Descriptor `i` of the `n` a layer carries. */
static inline void plt_requant_of(const volatile uint8_t *table, int n, int i,
                                  plt_requant_t *r)
{
    const volatile int32_t *w =
        (const volatile int32_t *)(const volatile void *)
        (table + (uint32_t)n * 256u + (uint32_t)i * 24u);
    r->kind = w[0]; r->c = w[1]; r->k = w[2]; r->off = w[3]; r->zo = w[4];
}

/** The scalar form, for the tail of a vector loop and for PLT_RQ_TABLE. */
static inline uint8_t plt_requant_byte(const plt_requant_t *r, const uint8_t lut[256],
                                       uint8_t b)
{
    if (r->kind == PLT_RQ_IDENTITY) return b;
    if (r->kind != PLT_RQ_AFFINE)   return lut[b];
    int32_t q = ((r->c * ((int32_t)b - r->off) + ((int32_t)1 << (r->k - 1))) >> r->k) + r->zo;
    if (q < -128) q = -128; else if (q > 127) q = 127;
    return (uint8_t)(q + 128);
}

#include "hal/plt_mxu3.h"

/* Vector registers this owns.  Constants are splatted across all 16 int32
 * lanes, so each is a whole 64-byte register.
 *
 *   16..21  the constants   22,23  widen scratch
 *   24..27  the lanes, widened and then accumulated in place
 *   28,29   narrow scratch
 *
 * Deliberately clear of vr4..vr7, which plt_copy_fast uses: the constants stay
 * loaded across a whole layer, and the identity path copies, so overlapping
 * those two would quietly corrupt the next row's requant. */
#define PLT_RQ_Z    16
#define PLT_RQ_C    17
#define PLT_RQ_T0   18
#define PLT_RQ_ZO   19
#define PLT_RQ_HI   20
#define PLT_RQ_K    21
#define PLT_RQ_NCONST 6

/** Fill a [PLT_RQ_NCONST][16] int32 array for PLT_RQ_LOAD_CONSTS. */
static inline void plt_requant_consts(const plt_requant_t *r, int32_t *cst)
{
    const int32_t t0 = (int32_t)(-(long long)r->c * r->off
                                 + ((long long)1 << (r->k - 1)));
    const int32_t cv[PLT_RQ_NCONST] = { 0, r->c, t0, r->zo + 128, 255, r->k };
    for (int c = 0; c < PLT_RQ_NCONST; c++)
        for (int l = 0; l < 16; l++) cst[c * 16 + l] = cv[c];
}

/* Load them; `gpr` is a register NUMBER holding the array base, and is walked
 * forward 64 bytes per constant. */
#define PLT_RQ_LOAD_CONSTS(gpr)                                      \
    PLT_M3_LUO(PLT_RQ_Z,  0, gpr) PLT_M3_LUO(PLT_RQ_Z,  1, gpr)      \
    PLT_M3_LUO(PLT_RQ_C,  0, gpr) PLT_M3_LUO(PLT_RQ_C,  1, gpr)      \
    PLT_M3_LUO(PLT_RQ_T0, 0, gpr) PLT_M3_LUO(PLT_RQ_T0, 1, gpr)      \
    PLT_M3_LUO(PLT_RQ_ZO, 0, gpr) PLT_M3_LUO(PLT_RQ_ZO, 1, gpr)      \
    PLT_M3_LUO(PLT_RQ_HI, 0, gpr) PLT_M3_LUO(PLT_RQ_HI, 1, gpr)      \
    PLT_M3_LUO(PLT_RQ_K,  0, gpr) PLT_M3_LUO(PLT_RQ_K,  1, gpr)

/* One stage across all four lane registers before the next begins: each is an
 * independent chain, so front-loading them keeps the unit busy through the
 * result latency instead of stalling on it. */
#define PLT_RQ_ST(op, e, f, g)                                                \
    PLT_M3_OP(op, (e) + 0, f, (g) + 0) PLT_M3_OP(op, (e) + 1, f, (g) + 1)     \
    PLT_M3_OP(op, (e) + 2, f, (g) + 2) PLT_M3_OP(op, (e) + 3, f, (g) + 3)

/** Requantize the 64 bytes in register `v`, in place. */
#define PLT_RQ_APPLY64(v)                                             \
    PLT_M3_WIDEN4(v, PLT_RQ_Z, 22, 23, 24, 25, 26, 27)                \
    PLT_RQ_ST(PLT_M3_MULW,  24, PLT_RQ_C,  24)                        \
    PLT_RQ_ST(PLT_M3_ADDW,  24, PLT_RQ_T0, 24)                        \
    PLT_RQ_ST(PLT_M3_SRAW,  24, PLT_RQ_K,  24)                        \
    PLT_RQ_ST(PLT_M3_ADDW,  24, PLT_RQ_ZO, 24)                        \
    PLT_RQ_ST(PLT_M3_MAXSW, 24, PLT_RQ_Z,  24)                        \
    PLT_RQ_ST(PLT_M3_MINSW, 24, PLT_RQ_HI, 24)                        \
    PLT_M3_NARROW4(24, 25, 26, 27, 28, 29, v)


/* Load the constants into the vector registers.  They stay there for the whole
 * layer, so a caller with a row loop pays this ONCE rather than twelve loads
 * per row -- on a 52-cell row that overhead was almost half the work.
 *
 * plt_requant_run() below assumes they are live.  Nothing between the two may
 * write vr16..vr21; plt_copy_fast (vr4..vr7) and the NNA feed/drain deliberately
 * do not. */
void plt_requant_load(const int32_t *cst);

/** Requantize a contiguous run.  Constants must already be loaded. */
void plt_requant_run(uint8_t *dst, const uint8_t *src, uint32_t n,
                     const plt_requant_t *r, const uint8_t lut[256]);

#endif /* PLT_EXEC_REQUANT_H */
