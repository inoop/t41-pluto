/* NNA (AIE) instruction encodings.
 *
 * Six coprocessor instructions in MIPS SPECIAL2 share one encoding
 *   word = 0x70000000 | (rs << 21) | ((payload & 0x3FFF) << 6) | 0x3A
 * with rs: nnrwr=0 nnrrd=1 nndwr=2 nndrd=3 nnmac=4 nncmd=5.  The vendor
 * precedes each with `sync`; this runtime does not -- see PLT_SYNC below.
 * Hardware facts from T41_NNA_MANUAL.md 11.
 *
 * This header is encodings only -- no policy, no derived values, no state.
 * The register map is in plt_nna_regs.h; the machine is in plt_nna.h.
 */
#ifndef PLT_HAL_ISA_NNA_H
#define PLT_HAL_ISA_NNA_H

#include <stdint.h>

#define PLT_NN_WORD(rs, payload) \
    (0x70000000u | ((rs) << 21) | (((payload) & 0x3FFFu) << 6) | 0x3Au)

#define PLT_NN_EMIT(rs, payload) do { __asm__ __volatile__( \
    ".set push\n.set noreorder\nsync\n.word %0\n.set pop\n"  \
    :: "i"(PLT_NN_WORD((rs), (payload))) : "memory"); } while (0)

#define PLT_NN_EMIT_NOSYNC(rs, payload) do { __asm__ __volatile__( \
    ".set push\n.set noreorder\n.word %0\n.set pop\n"  \
    :: "i"(PLT_NN_WORD((rs), (payload))) : "memory"); } while (0)

/* No barrier in front of the hot instructions.
 *
 * Ingenic's code puts a `sync` in front of every NNA instruction.  This runtime
 * used to as well, then dropped it from all but the first push of a run (worth
 * ~3 ms of an 80 ms frame), and now drops it everywhere a layer's tile loop
 * runs: field writes, pushes, MACs, drains and commands.  Measured together,
 * byte-exact on MobileNetV1 and YOLOX-Nano/S:
 *
 *     YOLOX-S 158.6 -> 152.8 ms, YOLOX-Nano 39.6 -> 38.7 ms, MobileNetV1 24.0 -> 23.8 ms
 *
 * (on top of 162.6 / 40.6 / 24.2 with the barrier still on the pushes and
 * drains).  A `sync` in isolation costs nothing measurable -- it is a nop-speed
 * instruction in a loop -- so what it cost here is the serialization it forces
 * each time there is vector work still in flight behind it.
 *
 * What the barrier would protect is ordering against the CPU's own memory
 * writes, and that ordering evidently holds without it: the 3x3 loop writes its
 * operand words to the stack and loads them into vr31 once per column block,
 * thousands of times per layer, and every layer stays exact.  Still, the engine
 * puts one PLT_SYNC after every layer, so no layer's last stores are ever in
 * flight when the next layer's CPU code, or a checksum, reads them. */
#define PLT_SYNC() do { __asm__ __volatile__("sync" ::: "memory"); } while (0)

#define PLT_NNRWR(payload) PLT_NN_EMIT_NOSYNC(0, (payload))   /* write a config field  */
#define PLT_NNRRD(payload) PLT_NN_EMIT(1, (payload))          /* read a config block   */
/* nndwr: payload encodes the source vr in [9:5] and the port in [4:0]|([14:10]<<5). */
#define PLT_NNDWR(vr, port) PLT_NN_EMIT_NOSYNC(2, ((((vr) & 0x1Fu) << 5) | ((port) & 0x1Fu) | (((port) >> 5) << 10)))
#define PLT_NNDWR_NEXT(vr, port) PLT_NNDWR((vr), (port))
#define PLT_NNDRD(payload) PLT_NN_EMIT_NOSYNC(3, (payload))   /* drain a readout bank  */
#define PLT_NNMAC(payload) PLT_NN_EMIT_NOSYNC(4, (payload))   /* multiply-accumulate   */
#define PLT_NNCMD(payload) PLT_NN_EMIT_NOSYNC(5, (payload))   /* array command         */

#endif /* PLT_HAL_ISA_NNA_H */
