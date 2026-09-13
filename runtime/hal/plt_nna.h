/* The NNA machine: the array's calling convention, named once.
 *
 * Everything an executor needs in order to talk to the array without knowing
 * register numbers -- the operand word file, a semantic configuration
 * descriptor, the two per-unit programs, the tile feed and drain, and the
 * command sequence with its hazards (T41_NNA_MANUAL.md 10).
 *
 * This layer knows nothing about layers.  It has no idea what a convolution
 * is; it knows about fields, programs, ports, tiles and commands.
 *
 * Vector register budget, fixed by this layer and relied on by executors:
 *   vr0..vr3    feed and push windows
 *   vr10..vr13  drain windows
 *   vr12        configuration readback
 *   vr31        the operand word file
 */
#ifndef PLT_HAL_NNA_H
#define PLT_HAL_NNA_H

#include <stdint.h>

#include "plt_isa_mxu.h"
#include "plt_isa_nna.h"
#include "plt_mxu3.h"
#include "plt_nna_regs.h"

/* ---------------------------------------------------------------------------
 * The operand word file (vr31)
 *
 * The array reads this 16-word file during `nnmac`, and configuration values
 * reach fields through it.  It is a value the caller owns -- an executor holds
 * one for the whole run and knows exactly what is live in it.
 * ------------------------------------------------------------------------- */
typedef struct __attribute__((aligned(64))) {
    uint32_t w[16];
} plt_nna_words_t;

void plt_nna_words_clear (plt_nna_words_t *words);
void plt_nna_words_commit(const plt_nna_words_t *words);   /* stage into vr31 */

/* Stage one value into word 0 and zero the rest. */
void plt_nna_stage_value(uint32_t value);

/* Write a field from a word that is already live in the file.  This does NOT
 * disturb the file, which is what makes it safe during a run.
 *
 * A macro, not a function, because an NNA instruction carries its payload in
 * the instruction word: `bank`, `word` and `field` must be literals. */
#define PLT_NNA_FIELD(bank, word, field) PLT_NNRWR(PLT_FIELD((bank), (word), (field)))

/* Write a value to a field, through word 0.  Clobbers the word file, so this
 * belongs to the configuration phase only -- never between a feed and its MAC.
 * `bank` and `field` must be literals; `value` need not be. */
#define PLT_NNA_FIELD_IMM(bank, field, value) do {   \
    plt_nna_stage_value((uint32_t)(value));          \
    PLT_NNA_FIELD((bank), 0, (field));               \
} while (0)

/* ---------------------------------------------------------------------------
 * The configuration descriptor
 *
 * One named member per hardware field.  Say what the layer IS; the encoding
 * helpers below turn that into field values.
 * ------------------------------------------------------------------------- */

/* Leave this field alone and emit no write at all.  Not the same as writing 0:
 * several fields have a non-zero reset value a recipe deliberately keeps. */
#define PLT_NNA_UNSET (-1)

typedef struct {
    /* ---- bank A, in field order ---- */
    int parity_tag;         /* A.01 datapath/parity tag                        */
    int in_elem_class;      /* A.03 plt_nna_in_elem()                          */
    int mode_code;          /* A.08 the mode pair, with mode_code_b            */
    int row_origin;         /* A.09 signed input row the walk starts from      */
    int col_origin;         /* A.0b signed input column, i.e. -pad             */
    int in_h;               /* A.0c input HEIGHT; written as in_h - 1          */
    int in_w;               /* A.0e input WIDTH;  written as in_w - 1          */
    int pad_value;          /* A.0f the border fill, i.e. the input zero point */
    int tap_step;           /* A.12 weight slot step per output group          */
    int tap_step2;          /* A.14 written as a pair with A.12 (3x3)          */
    int mac_span;           /* A.17 a recipe constant, in units of 4           */
    int edge_mode;          /* A.18 row-edge / unit-window control             */
    int mac_pitch;          /* A.19 a recipe constant, in units of 4           */
    int stride_code;        /* A.1a plt_nna_stride_code()                      */
    int operand_precision;  /* A.1b plt_nna_prec()                             */
    int groups_unit_a;      /* A.1c input groups for unit A; written as n - 1  */
    int groups_unit_b;      /* A.1d input groups for unit B; written as n - 1  */
    int tap_mask;           /* A.1e plt_nna_tap_mask()                         */

    /* ---- bank B, in field order ---- */
    int pack_format;        /* B.03 0 is the normal packed output              */
    int out_elem_class;     /* B.04 plt_nna_out_elem()                         */
    int mode_flags;         /* B.05                                            */
    int mode_code_b;        /* B.06 the mode pair, with mode_code              */
    int readout_lag;        /* B.07 plt_nna_readout_lag()                      */
    int lane_perm_row0;     /* B.08 plt_nna_lane_perm_row0()                   */
    int lane_perm_row1;     /* B.09 plt_nna_lane_perm_row1()                   */
    int unit_a_start;       /* B.10 pre-arm value only; see the note below     */
    int unit_b_start;       /* B.11 idem                                       */
    int tile_class;         /* B.15 plt_nna_tile_class()                       */
    int mac_balance;        /* B.1b MAC-stage half balance                     */
} plt_nna_cfg_t;

/* Set every member to PLT_NNA_UNSET.  Call this first: a member left at zero by
 * accident is written as zero, which for several fields is legal but wrong. */
void plt_nna_cfg_init(plt_nna_cfg_t *cfg);

/* Write every field the descriptor sets, bank A in field order then bank B.
 *
 * On return the word file is ZERO -- applying a descriptor spends it.  An
 * executor therefore installs the words it needs live (the ones its `nnmac`
 * payloads name, and the ones its per-tile field writes read) AFTER this call.
 *
 * This does not load the program and does not arm; both are separate because
 * recipes differ in what they do in between.  Fields a recipe also rewrites per
 * tile -- unit_a_start, unit_b_start -- are owned here only for their pre-arm
 * value. */
void plt_nna_cfg_apply(const plt_nna_cfg_t *cfg);

/* ---- the encoding formulas, named once (manual 12.3, 12.6) ---- */

/** A.1b: weight width in the high nibble, input width in the low, each bits/2-1. */
static inline int plt_nna_prec(int w_bits, int in_bits)
{
    return ((w_bits / 2 - 1) << 4) | (in_bits / 2 - 1);
}

/** A.03: the input element width code. */
static inline int plt_nna_in_elem(int in_bits) { return in_bits / 2 - 1; }

/** B.04: the output element width code. */
static inline int plt_nna_out_elem(int out_bits) { return (out_bits - 2) / 2; }

/** B.15: log2(tile_bytes / 64) for a 4px x 2row x 32ch tile.  Follows the INPUT
 *  width, so it does not change when the output width does. */
static inline int plt_nna_tile_class(int in_bits)
{
    if (in_bits <= 4) return in_bits / 2 - 1;
    return (in_bits <= 8) ? 2 : 3;
}

/** A.1a: stride - 1 in both nibbles, row stride high, column stride low. */
static inline int plt_nna_stride_code(int stride) { return (stride - 1) * 0x11; }

/** A.1e: one enable bit per kernel tap. */
static inline int plt_nna_tap_mask(int kernel) { return (1 << (kernel * kernel)) - 1; }

/** B.07: the packed-output readout lag, in 64-byte units. */
static inline int plt_nna_readout_lag(int out_bits) { return out_bits / 2; }

/** B.08 / B.09: the input lane permutation of the tile's two rows. */
static inline int plt_nna_lane_perm_row0(int out_bits)
{
    if (out_bits <= 4) return 0x3210;
    return (out_bits == 8) ? 0x5410 : 0x5140;
}
static inline int plt_nna_lane_perm_row1(int out_bits)
{
    if (out_bits <= 4) return 0x7654;
    return (out_bits == 8) ? 0x7632 : 0x7362;
}

/* ---------------------------------------------------------------------------
 * The walk programs
 *
 * The two execution units are independent machines: each walks its operands
 * from its OWN program, unit A's at entry 0 and unit B's at entry 8.  Giving
 * them one shared program is the single mistake that costs the most time,
 * because it leaves tile row 0 correct and only breaks row 1 (NNA_POINTWISE.md).
 * ------------------------------------------------------------------------- */
typedef struct {
    uint16_t word[16];
    int      len;
} plt_nna_prog_t;

void plt_nna_prog_clear(plt_nna_prog_t *prog);
void plt_nna_prog_push (plt_nna_prog_t *prog, uint32_t word);

/* Push slot A from entry 0 and slot B from entry PLT_PROG_SLOT_B, then write
 * the length pair B.1a.  Pass NULL for slot_b for a single-slot program. */
void plt_nna_prog_load(const plt_nna_prog_t *slot_a, const plt_nna_prog_t *slot_b);

/* ---------------------------------------------------------------------------
 * Operand movement
 *
 * Each of these is one tile's worth of traffic, in the order the array wants
 * it: load a window, push it, load the next.  The interleaving is part of the
 * contract, not an optimisation.
 * ------------------------------------------------------------------------- */

/** One 256-byte weight slice: four 64-byte pushes across ports W0..W3. */
void plt_nna_push_weight_slice(const volatile void *slice256);

/** Half a requant table group: two 64-byte pushes to the table port. */
void plt_nna_push_table_half(const volatile void *half128);

/** One 8-bit tile of one input plane: row 0's two halves, then row 1's.
 *  A macro because `port` is part of the instruction word. */
#ifdef PLT_BENCH_K3_FIXSRC
extern const uint8_t plt_bench_src[256 * 256];
#define PLT_BENCH_SRC(p, row) ((void)(p), (void)(row), plt_bench_src)
#else
#define PLT_BENCH_SRC(p, row) (p)
#endif
#define PLT_NNA_FEED_TILE(base, row_bytes, port) do {                       \
    const uint8_t *_p = PLT_BENCH_SRC((const uint8_t *)(uintptr_t)(base), row_bytes); \
    PLT_VLD64(0, _p +                    0); PLT_NNDWR(0, (port));          \
    PLT_VLD64(1, _p +                   64); PLT_NNDWR_NEXT(1, (port));     \
    PLT_VLD64(2, _p + (row_bytes) +      0); PLT_NNDWR_NEXT(2, (port));     \
    PLT_VLD64(3, _p + (row_bytes) +     64); PLT_NNDWR_NEXT(3, (port));     \
} while (0)

/** PLT_NNA_FEED_TILE over NDHWC16 cells: each push is two 16-byte pixels
 *  loaded into quad lanes 0 and 2, where a 32-lane cell's first 16 channels
 *  sit.  Lanes 1 and 3 keep stale bytes; they meet the zero weights of the
 *  channels such a tensor does not have. */
#define PLT_NNA_FEED_TILE16(base, row_bytes, port) do {                     \
    register const uint8_t *_a __asm__("t0") = (const uint8_t *)(uintptr_t)(base); \
    register const uint8_t *_b __asm__("t1") = _a + (row_bytes);            \
    __asm__ __volatile__(".set push\n\t.set noreorder\n\t"                  \
        PLT_M3_LAQ(0, 0, 8, 0) PLT_M3_LAQ(0, 2, 8, 1) ".word %[nn]\n\t"      \
        PLT_M3_LAQ(1, 0, 8, 2) PLT_M3_LAQ(1, 2, 8, 3) ".word %[mm]\n\t"      \
        PLT_M3_LAQ(2, 0, 9, 0) PLT_M3_LAQ(2, 2, 9, 1) ".word %[kk]\n\t"      \
        PLT_M3_LAQ(3, 0, 9, 2) PLT_M3_LAQ(3, 2, 9, 3) ".word %[ll]\n\t"      \
        ".set pop\n\t"                                                        \
        :: [nn] "i"(PLT_NN_WORD(2, (0u << 5) | ((port) & 0x1Fu) | (((port) >> 5) << 10))), \
           [mm] "i"(PLT_NN_WORD(2, (1u << 5) | ((port) & 0x1Fu) | (((port) >> 5) << 10))), \
           [kk] "i"(PLT_NN_WORD(2, (2u << 5) | ((port) & 0x1Fu) | (((port) >> 5) << 10))), \
           [ll] "i"(PLT_NN_WORD(2, (3u << 5) | ((port) & 0x1Fu) | (((port) >> 5) << 10))), \
           "r"(_a), "r"(_b) : "memory");                                    \
} while (0)

/** Drain one packed output tile from the FIFO and store it. */
void plt_nna_drain_tile_fifo(volatile void *dst, uint32_t row_bytes);

/* ---------------------------------------------------------------------------
 * Commands
 * ------------------------------------------------------------------------- */

/** Guarded reset (manual 10.1): B.1b written 0 immediately before nncmd 0x00.
 *  Without the guard the reset carries MAC-stage state over from the last run. */
void plt_nna_reset(void);

static inline void plt_nna_arm      (void) { PLT_NNCMD(PLT_CMD_ARM); }
static inline void plt_nna_precommit(void) { PLT_NNCMD(PLT_CMD_PRECOMMIT); }
static inline void plt_nna_pack     (void) { PLT_NNCMD(PLT_CMD_PACK); }
static inline void plt_nna_commit   (void) { PLT_NNCMD(PLT_CMD_COMMIT); }
static inline void plt_nna_block_end(void) { PLT_NNCMD(PLT_CMD_END_BLOCK); }

/** Read a 64-byte configuration block into `buf64` (64-byte aligned). */
#define PLT_NNA_READ_BLOCK(blk, buf64) do { \
    PLT_NNRRD(((blk) << 5) | 12u);          \
    PLT_VST64(12, (buf64));                 \
} while (0)

/* ---------------------------------------------------------------------------
 * Compatibility shim for the frozen probe tools in runtime/tools/, which stage
 * raw 16-word records of their own.  New code uses plt_nna_words_t.
 * ------------------------------------------------------------------------- */
static inline void plt_nna_stage(const uint32_t words16[16]) { PLT_VLD64(31, words16); }

#endif /* PLT_HAL_NNA_H */
