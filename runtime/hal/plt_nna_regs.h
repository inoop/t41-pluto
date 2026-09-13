/* NNA register map: field numbers, ports, commands, program words, MAC and
 * drain payloads.  Names for numbers the manual fixes (T41_NNA_MANUAL.md 11-12)
 * -- no policy lives here.  The encodings that DERIVE a field's value from a
 * layer's properties are in plt_nna.h.
 */
#ifndef PLT_HAL_NNA_REGS_H
#define PLT_HAL_NNA_REGS_H

#include "plt_isa_nna.h"

/* --- commands (manual 11.4) --- */
#define PLT_CMD_RESET     0x00u
#define PLT_CMD_ARM       0x60u
#define PLT_CMD_COMMIT    0x83u   /* commit + clear the accumulators          */
#define PLT_CMD_PRECOMMIT 0x84u   /* opens the pair that 0x8b then closes     */
#define PLT_CMD_PACK      0x8Bu   /* commit the previous MAC pair, packed     */
#define PLT_CMD_K3_COMMIT 0x8Fu   /* 3x3 per-tile commit (manual 8.3)         */
#define PLT_CMD_QUAD_R23  0xA0u   /* precedes every rows-2-3 MAC (manual 8.4.1) */
#define PLT_CMD_END_BLOCK 0xC0u   /* closes a block                           */
#define PLT_CMD_NEXT_GRP  0xE0u   /* switch to the next output group          */

/* --- field addressing (manual 12.1) ---
 * payload = bank | (vr31 word << 5) | field.  A field takes its value from the
 * named word of the operand file, which is why writing one is two steps. */
#define PLT_BANK_A 0x400u
#define PLT_BANK_B 0x800u
#define PLT_FIELD(bank, word, field) ((bank) | (((word) & 0x1Fu) << 5) | ((field) & 0x1Fu))

/* --- bank A (manual 12.3) --- */
#define PLT_A_PARITY     0x01u   /* datapath/parity tag                          */
#define PLT_A_IN_ELEM    0x03u   /* input element width, bits/2-1                */
#define PLT_A_MODE       0x08u   /* mode code (paired with B.06)                 */
#define PLT_A_ROW_ORG    0x09u   /* signed input row the walk starts from        */
#define PLT_A_PAD_ORG    0x0Bu   /* signed input column, i.e. -pad               */
#define PLT_A_IN_H_M1    0x0Cu   /* input height - 1                             */
#define PLT_A_IN_W_M1    0x0Eu   /* input width - 1                              */
#define PLT_A_PAD_VAL    0x0Fu   /* the byte the border is padded with           */
#define PLT_A_MAC_A0     0x10u   /* unit-A MAC address (inert on this datapath)  */
#define PLT_A_MAC_A1     0x11u
#define PLT_A_TAPSTEP    0x12u   /* weight slot step per output group            */
#define PLT_A_TAPSTEP2   0x14u   /* written as a pair with A.12 (3x3)            */
#define PLT_A_MAC_SPAN   0x17u   /* MAC address span, in units of 4              */
#define PLT_A_EDGE       0x18u   /* row-edge / unit-window control               */
#define PLT_A_MAC_PITCH  0x19u   /* MAC address pitch, in units of 4             */
#define PLT_A_STRIDE     0x1Au   /* (stride_h-1) << 4 | (stride_w-1)             */
#define PLT_A_PREC       0x1Bu   /* (w_bits/2-1) << 4 | (in_bits/2-1)            */
#define PLT_A_UNITA_GRP  0x1Cu   /* input groups fed to unit A, minus 1          */
#define PLT_A_UNITB_GRP  0x1Du   /* input groups fed to unit B, minus 1          */
#define PLT_A_TAPMASK    0x1Eu   /* one enable bit per kernel tap                */

/* --- bank B (manual 12.3) --- */
#define PLT_B_COMMIT_WIN  0x02u  /* commit window                                */
#define PLT_B_PACKFMT     0x03u  /* pack / output format                         */
#define PLT_B_OUTELEM     0x04u  /* output element width, (bits-2)/2             */
#define PLT_B_MODEFLG     0x05u  /* mode flags                                   */
#define PLT_B_MODE_B      0x06u  /* mode code (paired with A.08)                 */
#define PLT_B_LAG         0x07u  /* packed readout lag, in 64-byte units         */
#define PLT_B_LANE0       0x08u  /* input lane permutation, tile row 0           */
#define PLT_B_LANE1       0x09u  /* input lane permutation, tile row 1           */
#define PLT_B_UNITA_START 0x10u  /* unit A's operand start offset                */
#define PLT_B_UNITB_START 0x11u  /* unit B's                                     */
#define PLT_B_TILECLS     0x15u  /* input tile size class, log2(tile_bytes/64)   */
#define PLT_B_PROG_PTR    0x18u  /* program push pointer                         */
#define PLT_B_PROG_PUSH   0x19u  /* push one program entry                       */
#define PLT_B_PROG_LEN    0x1Au  /* (len_b << 4) | len_a                         */
#define PLT_B_MAC_BAL     0x1Bu  /* MAC-stage half balance / reset guard         */
#define PLT_B_WSTREAM     0x1Cu  /* weight-stream offset                         */
#define PLT_B_TABLE_PTR   0x1Eu  /* requant-table push pointer                   */

/* Unit B's program occupies program-table entries 8 and up. */
#define PLT_PROG_SLOT_B 8

/* --- data ports (manual 11.6) --- */
#define PLT_PORT_W0    0x00u
#define PLT_PORT_W1    0x02u
#define PLT_PORT_W2    0x04u
#define PLT_PORT_W3    0x07u
#define PLT_PORT_ACT_A 0x20u
#define PLT_PORT_ACT_B 0x21u
#define PLT_PORT_TABLE 0x40u

/* --- program words (manual 12.4).  The step field is a SIGNED 10-bit value;
 * masking it any wider corrupts the walk. --- */
#define PLT_PROG_STEP(s)        (0x800u | ((uint32_t)(s) & 0x3FFu))
#define PLT_PROG_LOAD(cnt, byte)(0x7800u | (((uint32_t)(cnt) & 7u) << 8) | ((uint32_t)(byte) & 0xFFu))
#define PLT_PROG_LEN(la, lb)    ((((uint32_t)(lb)) << 4) | ((uint32_t)(la) & 0xFu))

/* The native 3x3 walk uses three more entry forms (manual 11.2.4).  Decoded
 * against the manual's own worked words: at stride 1 with two groups per unit
 * the four entries are 0x7B01 0x1805 0x0819 0x03B1, and at stride 2 the row
 * step is 0x824 and the move 0x38A.
 *   loop(n)      = PLT_PROG_LOAD(3, n-1)   -- note n-1, where the 1x1 uses n-2
 *   step(a, b)   = a << 11 | b
 *   move(x)      = x, ten-bit signed                                        */
#define PLT_PROG_STEP2(a, b)    ((((uint32_t)(a)) << 11) | ((uint32_t)(b) & 0x7FFu))
#define PLT_PROG_MOVE(x)        ((uint32_t)(x) & 0x3FFu)

/* --- MAC payloads (manual 11.5).  The payload names the vr31 word the array
 * samples as that MAC's operand. --- */
#define PLT_MAC_UNIT_A   0x81u   /* 1x1 int8: nnmac vw4,1                       */
#define PLT_MAC_UNIT_B   0xA3u   /* 1x1 int8: nnmac vw5,3                       */
#define PLT_MAC_QUAD_R01 0x20u   /* single input group (8.4.1), rows 0-1        */
#define PLT_MAC_QUAD_R23 0x41u   /* single input group (8.4.1), rows 2-3        */
/* 3x3 with D >= 2 (manual 11.2.5, 15.3): phase 1 `vw9,1` is unit A, groups
 * 0..nA-1 (port 0x20); phase 2 `vw10,2` is unit B, groups nA..D-1 (port 0x21).
 * That order is what makes w10 = -1 + 4*B.11 start past unit A's operands. */
#define PLT_MAC_K3_A     0x121u  /* vw9,1  unit A, groups 0..nA-1   */
#define PLT_MAC_K3_B     0x142u  /* vw10,2 unit B, groups nA..D-1   */

/* --- drains (manual 11.7): payload = (bank << 5) | dest_vr --- */
#define PLT_DRAIN_BANK0(R, vr) PLT_NNDRD(((((uint32_t)(R)) << 5) | ((vr) & 0x1Fu)))
#define PLT_DRAIN_FIFO(vr)     PLT_NNDRD(((0x20u << 5) | ((vr) & 0x1Fu)))
#define PLT_DRAIN_K3(row, vr)  PLT_NNDRD((((0x408u + (row)) << 5) | ((vr) & 0x1Fu)))

/* --- config readback blocks (manual 12.5): payload = (blk << 5) | dest_vr --- */
#define PLT_BLK_G 0x20u          /* block G = the bank-A file */

#endif /* PLT_HAL_NNA_REGS_H */
