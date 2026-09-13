/* The NNA machine.  See plt_nna.h for the contract.
 *
 * Note on shape: every NNA instruction encodes its payload IN THE INSTRUCTION
 * WORD, so a payload must be a compile-time literal.  That is why anything
 * whose bank, field, word index or port varies at the call site is a macro in
 * plt_nna.h, and only the fixed-payload sequences are functions here. */
#include <string.h>

#include "hal/plt_nna.h"

#ifdef PLT_BENCH_K3_FIXSRC
const uint8_t plt_bench_src[256 * 256];
#endif

/* --- the operand word file ------------------------------------------------ */

void plt_nna_words_clear(plt_nna_words_t *words)
{
    memset(words, 0, sizeof *words);
}

void plt_nna_words_commit(const plt_nna_words_t *words)
{
    PLT_VLD64(31, words->w);
}

void plt_nna_stage_value(uint32_t value)
{
    plt_nna_words_t words;
    plt_nna_words_clear(&words);
    words.w[0] = value;
    plt_nna_words_commit(&words);
}

/* --- the configuration descriptor ----------------------------------------- */

void plt_nna_cfg_init(plt_nna_cfg_t *cfg)
{
    int *p = (int *)cfg;
    for (size_t i = 0; i < sizeof *cfg / sizeof(int); i++) p[i] = PLT_NNA_UNSET;
}

/* Fields the array wants as "count - 1", which the descriptor states as the
 * real count.  An unset member stays unset. */
static int minus_one(int value)
{
    return value == PLT_NNA_UNSET ? PLT_NNA_UNSET : value - 1;
}

/* Each list is one staged record: the word index a member's value goes into,
 * the field it is written to, and the member itself.  Staging and writing walk
 * the same list, so the two can never drift apart.  A record is 16 words, which
 * is why bank A takes two. */
#define PLT_CFG_BANK_A_1(X)                          \
    X( 0, PLT_A_PARITY,     parity_tag)              \
    X( 1, PLT_A_IN_ELEM,    in_elem_class)           \
    X( 2, PLT_A_MODE,       mode_code)               \
    X( 3, PLT_A_ROW_ORG,    row_origin)              \
    X( 4, PLT_A_PAD_ORG,    col_origin)              \
    X( 5, PLT_A_IN_H_M1,    in_h)                    \
    X( 6, PLT_A_IN_W_M1,    in_w)                    \
    X( 7, PLT_A_PAD_VAL,    pad_value)               \
    X( 8, PLT_A_TAPSTEP,    tap_step)                \
    X( 9, PLT_A_TAPSTEP2,   tap_step2)               \
    X(10, PLT_A_MAC_SPAN,   mac_span)                \
    X(11, PLT_A_EDGE,       edge_mode)               \
    X(12, PLT_A_MAC_PITCH,  mac_pitch)               \
    X(13, PLT_A_STRIDE,     stride_code)             \
    X(14, PLT_A_PREC,       operand_precision)       \
    X(15, PLT_A_UNITA_GRP,  groups_unit_a)

#define PLT_CFG_BANK_A_2(X)                          \
    X( 0, PLT_A_UNITB_GRP,  groups_unit_b)           \
    X( 1, PLT_A_TAPMASK,    tap_mask)

#define PLT_CFG_BANK_B(X)                            \
    X( 0, PLT_B_PACKFMT,     pack_format)            \
    X( 1, PLT_B_OUTELEM,     out_elem_class)         \
    X( 2, PLT_B_MODEFLG,     mode_flags)             \
    X( 3, PLT_B_MODE_B,      mode_code_b)            \
    X( 4, PLT_B_LAG,         readout_lag)            \
    X( 5, PLT_B_LANE0,       lane_perm_row0)         \
    X( 6, PLT_B_LANE1,       lane_perm_row1)         \
    X( 7, PLT_B_UNITA_START, unit_a_start)           \
    X( 8, PLT_B_UNITB_START, unit_b_start)           \
    X( 9, PLT_B_TILECLS,     tile_class)             \
    X(10, PLT_B_MAC_BAL,     mac_balance)

void plt_nna_cfg_apply(const plt_nna_cfg_t *cfg)
{
    plt_nna_cfg_t   e = *cfg;
    plt_nna_words_t rec;

    e.in_h          = minus_one(e.in_h);
    e.in_w          = minus_one(e.in_w);
    e.groups_unit_a = minus_one(e.groups_unit_a);
    e.groups_unit_b = minus_one(e.groups_unit_b);

#define STAGE(IDX, FIELD, MEMBER) rec.w[IDX] = (uint32_t)e.MEMBER;
#define WRITE_A(IDX, FIELD, MEMBER) \
    if (e.MEMBER != PLT_NNA_UNSET) PLT_NNA_FIELD(PLT_BANK_A, IDX, FIELD);
#define WRITE_B(IDX, FIELD, MEMBER) \
    if (e.MEMBER != PLT_NNA_UNSET) PLT_NNA_FIELD(PLT_BANK_B, IDX, FIELD);

    plt_nna_words_clear(&rec);
    PLT_CFG_BANK_A_1(STAGE)
    plt_nna_words_commit(&rec);
    PLT_CFG_BANK_A_1(WRITE_A)

    plt_nna_words_clear(&rec);
    PLT_CFG_BANK_A_2(STAGE)
    plt_nna_words_commit(&rec);
    PLT_CFG_BANK_A_2(WRITE_A)

    plt_nna_words_clear(&rec);
    PLT_CFG_BANK_B(STAGE)
    plt_nna_words_commit(&rec);
    PLT_CFG_BANK_B(WRITE_B)

#undef STAGE
#undef WRITE_A
#undef WRITE_B

    /* Leave the file zero, so nothing stale is live when the caller installs
     * the words it needs. */
    plt_nna_stage_value(0);
}

/* --- the walk programs ---------------------------------------------------- */

void plt_nna_prog_clear(plt_nna_prog_t *prog)
{
    memset(prog, 0, sizeof *prog);
}

void plt_nna_prog_push(plt_nna_prog_t *prog, uint32_t word)
{
    if (prog->len < (int)(sizeof prog->word / sizeof prog->word[0]))
        prog->word[prog->len++] = (uint16_t)word;
}

static void prog_push_slot(const plt_nna_prog_t *prog, unsigned entry)
{
    PLT_NNA_FIELD_IMM(PLT_BANK_B, PLT_B_PROG_PTR, entry);
    for (int i = 0; i < prog->len; i++)
        PLT_NNA_FIELD_IMM(PLT_BANK_B, PLT_B_PROG_PUSH, prog->word[i]);
}

void plt_nna_prog_load(const plt_nna_prog_t *slot_a, const plt_nna_prog_t *slot_b)
{
    int len_b = 0;

    prog_push_slot(slot_a, 0);
    if (slot_b) {
        prog_push_slot(slot_b, PLT_PROG_SLOT_B);
        len_b = slot_b->len;
    }
    PLT_NNA_FIELD_IMM(PLT_BANK_B, PLT_B_PROG_LEN, PLT_PROG_LEN(slot_a->len, len_b));
}

/* --- operand movement ----------------------------------------------------- */

void plt_nna_push_weight_slice(const volatile void *slice256)
{
    const uint8_t *p = (const uint8_t *)(uintptr_t)slice256;
    PLT_VLD64(0, p +   0); PLT_NNDWR(0, PLT_PORT_W0);
    PLT_VLD64(1, p +  64); PLT_NNDWR_NEXT(1, PLT_PORT_W1);
    PLT_VLD64(2, p + 128); PLT_NNDWR_NEXT(2, PLT_PORT_W2);
    PLT_VLD64(3, p + 192); PLT_NNDWR_NEXT(3, PLT_PORT_W3);
}

void plt_nna_push_table_half(const volatile void *half128)
{
    const uint8_t *p = (const uint8_t *)(uintptr_t)half128;
    PLT_VLD64(0, p +  0); PLT_NNDWR(0, PLT_PORT_TABLE);
    PLT_VLD64(1, p + 64); PLT_NNDWR_NEXT(1, PLT_PORT_TABLE);
}

void plt_nna_drain_tile_fifo(volatile void *dst, uint32_t row_bytes)
{
    uint8_t *p = (uint8_t *)(uintptr_t)dst;
    PLT_DRAIN_FIFO(10); PLT_DRAIN_FIFO(11); PLT_DRAIN_FIFO(12); PLT_DRAIN_FIFO(13);
    PLT_VST64(10, p +             0);
    PLT_VST64(11, p +            64);
    PLT_VST64(12, p + row_bytes +  0);
    PLT_VST64(13, p + row_bytes + 64);
}

/* --- commands ------------------------------------------------------------- */

void plt_nna_reset(void)
{
    plt_nna_stage_value(0);
    PLT_NNA_FIELD(PLT_BANK_B, 0, PLT_B_MAC_BAL);   /* B.1b <- w0 (= 0) */
    PLT_NNCMD(PLT_CMD_RESET);
}
