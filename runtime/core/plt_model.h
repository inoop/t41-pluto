/* The .pluto model container -- the C view of the format defined in
 * compile/format.py.  The _Static_asserts below fail the build if the two ever
 * drift.  Little-endian, fixed-size records; see docs/MODEL_FORMAT.md.
 *
 * These are the ON-DISK records, hence the _rec_t suffix: they are the file's
 * bytes, not the runtime's working types.  The runtime types -- plt_tensor_t
 * and friends -- live in core/plt_tensor.h. */
#ifndef PLT_CORE_MODEL_H
#define PLT_CORE_MODEL_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define PLT_MAGIC          0x4F544C50u   /* 'PLTO' little-endian */
#define PLT_VERSION_MAJOR  1
#define PLT_VERSION_MINOR  0
#define PLT_NAME_LEN       64
#define PLT_MAX_IO         4
#define PLT_UNUSED         0xFFFFFFFFu    /* unused id / absent offset */
#define PLT_BLOB_ALIGN     64

enum plt_dtype  { PLT_DT_INT8 = 0, PLT_DT_UINT8 = 1, PLT_DT_INT32 = 2, PLT_DT_FP32 = 3 };
enum plt_format { PLT_FMT_NHWC = 0, PLT_FMT_NDHWC32 = 1, PLT_FMT_NCHW = 2, PLT_FMT_VEC = 3,
                  /* runtime-only, never on disk: NDHWC32's row/plane layout with
                   * 16-byte cells, for a tensor of at most 16 channels whose
                   * producer and readers all know it (plt_engine_load) */
                  PLT_FMT_NDHWC16 = 4 };
enum plt_op     { PLT_OP_CONV = 0, PLT_OP_DWCONV = 1, PLT_OP_AVGPOOL = 2,
                  PLT_OP_MAXPOOL = 3, PLT_OP_FC = 4, PLT_OP_ADD = 5,
                  PLT_OP_CONCAT = 6, PLT_OP_UPSAMPLE = 7, PLT_OP_FOCUS = 8,
                  PLT_OP_SILU = 9 };
enum plt_exec   { PLT_EX_NNA_PW = 0, PLT_EX_NNA_DW = 1, PLT_EX_NNA_STD = 2,
                  PLT_EX_MXU_POOL = 3, PLT_EX_MXU_FC = 4, PLT_EX_MXU_SILU = 5,
                  PLT_EX_MXU_ADD = 6, PLT_EX_MXU_CONCAT = 7, PLT_EX_MXU_UPSAMPLE = 8,
                  PLT_EX_MXU_FOCUS = 9, PLT_EX_MXU_MAXPOOL = 10, PLT_EX_CPU_CONV = 11,
                  PLT_EX_NNA_DENSE = 12, PLT_EX_NNA_K3 = 13 };
enum plt_act    { PLT_ACT_NONE = 0, PLT_ACT_RELU = 1, PLT_ACT_RELU6 = 2,
                  PLT_ACT_SILU = 3 };

typedef struct {
    uint32_t magic;
    uint16_t version_major;
    uint16_t version_minor;
    uint32_t flags;
    uint32_t num_tensors;
    uint32_t num_layers;
    uint32_t num_inputs;
    uint32_t num_outputs;
    uint32_t input_ids[PLT_MAX_IO];
    uint32_t output_ids[PLT_MAX_IO];
    uint32_t tensors_off;
    uint32_t layers_off;
    uint32_t blob_off;
    uint32_t blob_size;
    uint32_t nmem_arena_hint;
    uint32_t reserved[4];
} plt_header_rec_t;

typedef struct {
    uint32_t id;
    char     name[PLT_NAME_LEN];
    uint8_t  dtype;
    uint8_t  format;
    uint8_t  ndims;
    uint8_t  _pad0;
    int32_t  shape[4];
    uint32_t scale_bits;      /* IEEE-754 f32 bit pattern */
    int32_t  zero_point;
    uint32_t data_off;        /* PLT_UNUSED if the tensor carries no data */
    uint32_t data_size;
    uint8_t  reserved[24];
} plt_tensor_rec_t;

typedef struct {
    uint32_t id;
    uint8_t  op;
    uint8_t  exec;
    uint8_t  act;
    uint8_t  _pad0;
    uint32_t input_ids[PLT_MAX_IO];
    uint32_t output_id;
    uint8_t  kh, kw, sh, sw, pt, pb, pl, pr;
    uint32_t groups;
    uint8_t  dh, dw;
    uint16_t _pad1;
    uint8_t  in_bits, w_bits, out_bits, _pad2;
    int32_t  in_zp;
    uint32_t weight_off, weight_size;
    uint32_t reqtbl_off, reqtbl_size;
    uint8_t  params[32];
    uint8_t  reserved[28];
} plt_layer_rec_t;

_Static_assert(sizeof(plt_header_rec_t) == 96,  "plt_header_rec_t must be 96 bytes");
_Static_assert(sizeof(plt_tensor_rec_t) == 128, "plt_tensor_rec_t must be 128 bytes");
_Static_assert(sizeof(plt_layer_rec_t)  == 128, "plt_layer_rec_t must be 128 bytes");

typedef struct {
    uint8_t            *data;    /* whole file image (owned)  */
    size_t              size;
    const plt_header_rec_t *header;
    const plt_tensor_rec_t *tensors; /* [header->num_tensors] */
    const plt_layer_rec_t  *layers;  /* [header->num_layers]  */
    const uint8_t      *blob;
} plt_model_t;

/* Returns 0 on success, negative on error (-1 I/O, -2 bad magic, -3 truncated). */
int  plt_model_load(const char *path, plt_model_t *m);
void plt_model_free(plt_model_t *m);
void plt_model_dump(const plt_model_t *m, FILE *out);


/* Look one record up by id; NULL if there is no such tensor. */
const plt_tensor_rec_t *plt_model_tensor(const plt_model_t *m, uint32_t id);

/* A blob region as raw bytes; NULL if the range is not inside the blob. */
const uint8_t *plt_model_blob(const plt_model_t *m, uint32_t off, uint32_t size);

#endif /* PLT_CORE_MODEL_H */
