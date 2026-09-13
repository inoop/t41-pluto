/* MXU3 SIMD encodings for the vector executors.
 *
 * MXU3 has 32 x 512-bit vector registers.  Every arithmetic instruction names
 * three of them -- e (bits 20-16), f (bits 15-11), g (bits 10-6) -- with
 * `g = op(e, f)`, whole register at a time.
 *
 * A register is ALSO addressable as lanes: 4 quads of 16 bytes or 2 octets of
 * 32.  `LUQ`/`SUQ` move lane 0 and POST-INCREMENT a GPR base by 16, so a run of
 * them walks an array; `LAQ`/`LAO` move a NAMED lane at `base + off*size` and
 * leave the base alone, so several of them assemble one register from
 * unrelated addresses.  The kernels below use only lane 0 of each register.
 * (Lane index at bits 15-13 for quads, 15-14 for octets; register at 10-6.)
 *
 * Op bases are hardware facts, taken from Ingenic's own binutils table
 * (references/ingenic-toolchain-src/mxu3) and checked on the device.
 *
 * Why these particular ops: getting a byte into an int32 lane and back permutes
 * lanes, but for an ELEMENTWISE operation the permutation does not matter as
 * long as the widen and the narrow are inverses.  This pair is (verified on
 * hardware):
 *
 *     lo = ilveb(v,0)   hi = ilvob(0,v)      byte  -> int16
 *     x0 = ilveh(lo,0)  x1 = ilvoh(0,lo)     int16 -> int32
 *     lo = ilveh(x0,x1)     v = ilveb(lo,hi)          and back
 */
#ifndef PLT_HAL_MXU3_H
#define PLT_HAL_MXU3_H

#define PLT_M3_S1(x) #x
#define PLT_M3_STR(x) PLT_M3_S1(x)
#define PLT_M3_W(enc) ".word " PLT_M3_STR(enc) "\n\t"

/* One 128-bit sub-register to/from memory; `gpr` post-increments by 16. */
#define PLT_M3_LUQ(gpr, sr) PLT_M3_W((0x1C << 26) | ((gpr) << 21) | (1 << 11) | ((sr) << 6) | 0x13)
#define PLT_M3_SUQ(gpr, sr) PLT_M3_W((0x1C << 26) | ((gpr) << 21) | ((sr) << 11) | 0x57)

/* One NAMED lane of a vector register, loaded from `base + off * size`.  The
 * other lanes and the base GPR are untouched, so four `LAQ`s from four
 * unrelated addresses assemble one 64-byte push register -- which is how a
 * convolution window reaches the array without ever being written to memory.
 *
 * `off` is a 5-bit signed index in units of the transfer size (quad 16 bytes,
 * octet 32), i.e. +-8 / +-16 NDHWC32 cells.  `gpr` is a register NUMBER, so the
 * caller must pin its pointers with `register ... __asm__("t0")`.
 *
 * These are not new ground: PLT_VLD64 in plt_isa_mxu.h is exactly
 * `lao vr[0],0(t0)` + `lao vr[1],1(t0)`.  The push register has always been
 * built from independently addressed lanes; these macros stop requiring the
 * lanes to be adjacent.  Encodings from docs/MXU3_OPCODE_TABLE.md (Ingenic's
 * own binutils table), device-checked by tools/mxuv3_ldst_offset_test.c. */
#define PLT_M3_LAQ(vr, lane, gpr, off) \
    PLT_M3_W(0x70000811 | ((gpr) << 21) | (((off) & 0x1f) << 16) | ((lane) << 13) | ((vr) << 6))
#define PLT_M3_LAO(vr, lane, gpr, off) \
    PLT_M3_W(0x70001811 | ((gpr) << 21) | (((off) & 0x1f) << 16) | ((lane) << 14) | ((vr) << 6))

/* g = op(e, f) on one sub-register. */
#define PLT_M3_OP(base, e, f, g) PLT_M3_W((base) | ((e) << 16) | ((f) << 11) | ((g) << 6))

/* The matching stores, and the post-incrementing 32-byte load.  `sao`/`saq`
 * take a lane index and a signed offset like the loads; `luo` walks a base GPR
 * forward by 32, which is how a run of constants is loaded. */
#define PLT_M3_SAO(vr, lane, gpr, off) \
    PLT_M3_W(0x700000d5 | ((gpr) << 21) | (((off) & 0x1f) << 16) | ((vr) << 11) | ((lane) << 9))
#define PLT_M3_SAQ(vr, lane, gpr, off) \
    PLT_M3_W(0x70000055 | ((gpr) << 21) | (((off) & 0x1f) << 16) | ((vr) << 11) | ((lane) << 8))
#define PLT_M3_LUO(vr, lane, gpr) \
    PLT_M3_W(0x70001813 | ((gpr) << 21) | ((lane) << 14) | ((vr) << 6))

#define PLT_M3_ILVEB 0x70400038   /* interleave even bytes      */
#define PLT_M3_ILVOB 0x72400038   /* interleave odd bytes       */
#define PLT_M3_ILVEH 0x70600038   /* interleave even halfwords  */
#define PLT_M3_ILVOH 0x72600038   /* interleave odd halfwords   */
#define PLT_M3_ADDW  0x4a800002   /* int32 add                  */
#define PLT_M3_SUBW  0x4a80000a   /* int32 subtract             */
#define PLT_M3_MULW  0x4a600022   /* int32 multiply             */
#define PLT_M3_SRAW  0x4a20003a   /* int32 arithmetic shift right, per lane */
#define PLT_M3_MINSW 0x4a000016   /* int32 signed min           */
#define PLT_M3_MAXSW 0x4a00001e   /* int32 signed max           */
#define PLT_M3_MAXUB 0x4a000008   /* uint8 max                  */
#define PLT_M3_ANDV  0x4a600002   /* bitwise and                */
#define PLT_M3_ORV   0x4a600004   /* bitwise or                 */
#define PLT_M3_XORV  0x4a600006   /* bitwise xor                */
#define PLT_M3_SLLB  0x4a200020   /* per-byte shift left        */

/* gshufvb: a data-dependent 64-lane BYTE GATHER, g[i] = f[e[i] & 63], except
 * that a lane whose index has bit 7 set comes out zero (bit 6 is ignored).
 * The x86 pshufb rule, and the reason a 256-entry table CAN be vectorized here.
 *
 * This is the 3.1-only opcode.  MXU3.0's `gshufb` (0x7020002b) is what earlier
 * comments in this tree meant when they said the part has no data-dependent
 * gather -- and it is genuinely absent: it SIGILLs on this chip.  gshufvb is
 * present, and is one of the two most-executed custom ops in libvenus.
 * Semantics device-probed 2026-09-11, see docs/T41_NNA_MANUAL.md 12.6. */
#define PLT_M3_GSHUFVB 0x7380002e

/* Widen sub-register `v` (16 bytes) into four int32 sub-registers w0..w3,
 * using `z` (zero) and two scratch sub-registers.  Elementwise only: the lane
 * order is scrambled and PLT_M3_NARROW4 is its exact inverse. */
#define PLT_M3_WIDEN4(v, z, t0, t1, w0, w1, w2, w3)  \
    PLT_M3_OP(PLT_M3_ILVEB, v,  z,  t0)              \
    PLT_M3_OP(PLT_M3_ILVOB, z,  v,  t1)              \
    PLT_M3_OP(PLT_M3_ILVEH, t0, z,  w0)              \
    PLT_M3_OP(PLT_M3_ILVOH, z,  t0, w1)              \
    PLT_M3_OP(PLT_M3_ILVEH, t1, z,  w2)              \
    PLT_M3_OP(PLT_M3_ILVOH, z,  t1, w3)

#define PLT_M3_NARROW4(w0, w1, w2, w3, t0, t1, v)    \
    PLT_M3_OP(PLT_M3_ILVEH, w0, w1, t0)              \
    PLT_M3_OP(PLT_M3_ILVEH, w2, w3, t1)              \
    PLT_M3_OP(PLT_M3_ILVEB, t0, t1, v)

/* --- a 256-entry byte table over 64 lanes ----------------------------------
 *
 * `gshufvb` reaches only 64 table bytes, so a 256-entry table is four gathers,
 * one per quarter, whose results are merged.  The quarter is the index's top two
 * bits, and both selections come nearly free from the gather itself:
 *
 *   bit 7: gshufvb zeroes a lane whose index has bit 7 set, so gathering the
 *          low half with the index and the high half with the index XOR 0x80
 *          leaves each lane in exactly one of the two -- an OR merges them.
 *   bit 6: gathering from an all-ones register with the index shifted left by
 *          one yields 0xFF exactly where bit 6 is clear, and `bselv` picks the
 *          even quarters' merge there and the odd quarters' elsewhere.
 *
 * gshufvb ignores index bit 6, so the gathers need no `& 0x3F` first.
 *
 * 10 instructions per 64 bytes.  (The first version took 19, with an explicit
 * bit-7 mask, a masked index and and/or merges; an andn version 14.)
 *
 * Registers: q0..q3 hold the table's four quarters, ff is 64 bytes of 0xFF,
 * c80 is 64 bytes of 0x80 and c01 is 64 bytes of 0x01; t0..t4 are scratch, t5 is
 * unused.  `src` is read only by the first four instructions, so it may be
 * `dst`.  All are register NUMBERS, so the caller owns the allocation.
 */
#define PLT_M3_ANDNV 0x4a600003   /* f & ~e (device-checked)    */
#define PLT_M3_BSELV 0x4a60000e   /* g = g ? f : e, bitwise     */

#define PLT_M3_LUT64(src, dst, q0, q1, q2, q3, ff, c80, c01, \
                     t0, t1, t2, t3, t4, t5)                                   \
    PLT_M3_OP(PLT_M3_SLLB,    src, c01, t0)   /* t0 = index << 1             */\
    PLT_M3_OP(PLT_M3_XORV,    src, c80, t1)   /* t1 = index ^ 0x80           */\
    PLT_M3_OP(PLT_M3_GSHUFVB, src, q0,  t2)                                    \
    PLT_M3_OP(PLT_M3_GSHUFVB, src, q1,  t3)                                    \
    PLT_M3_OP(PLT_M3_GSHUFVB, t1,  q2,  t4)                                    \
    PLT_M3_OP(PLT_M3_ORV,     t2,  t4,  t2)   /* t2 = quarters 0|2           */\
    PLT_M3_OP(PLT_M3_GSHUFVB, t1,  q3,  t4)                                    \
    PLT_M3_OP(PLT_M3_ORV,     t3,  t4,  t3)   /* t3 = quarters 1|3           */\
    PLT_M3_OP(PLT_M3_GSHUFVB, t0,  ff,  dst)  /* dst = 0xFF where bit 6 clear*/\
    PLT_M3_OP(PLT_M3_BSELV,   t3,  t2,  dst)

#endif /* PLT_HAL_MXU3_H */
