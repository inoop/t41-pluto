"""The .pluto container binary layout — the single source of truth.

Mirrored on the C side by runtime/core/plt_model.h, which carries a _Static_assert on
every record size so the two never drift. Little-endian, fixed-size records.
See docs/MODEL_FORMAT.md for the human-readable spec.
"""
import struct
from enum import IntEnum

MAGIC = 0x4F544C50            # 'PLTO' as a little-endian u32
VERSION_MAJOR = 1
VERSION_MINOR = 0

NAME_LEN = 64
MAX_IO = 4
UNUSED = 0xFFFFFFFF           # sentinel: unused id / absent offset
BLOB_ALIGN = 64

# Record formats (spaces are ignored by struct; they group fields for reading).
HEADER_FMT = "<I HH I IIII 4I 4I IIIII 4I"
TENSOR_FMT = "<I 64s BBBB 4i I i II 24s"
LAYER_FMT  = "<I BBBB 4I I 8B I BBH BBBB i IIII 32s 28s"

HEADER_SIZE = struct.calcsize(HEADER_FMT)
TENSOR_SIZE = struct.calcsize(TENSOR_FMT)
LAYER_SIZE  = struct.calcsize(LAYER_FMT)

assert HEADER_SIZE == 96, HEADER_SIZE
assert TENSOR_SIZE == 128, TENSOR_SIZE
assert LAYER_SIZE == 128, LAYER_SIZE


class DType(IntEnum):
    INT8 = 0
    UINT8 = 1
    INT32 = 2
    FP32 = 3


class Format(IntEnum):
    NHWC = 0
    NDHWC32 = 1
    NCHW = 2
    VEC = 3


class Op(IntEnum):
    CONV = 0
    DWCONV = 1
    AVGPOOL = 2
    MAXPOOL = 3
    FC = 4
    ADD = 5
    CONCAT = 6
    UPSAMPLE = 7
    FOCUS = 8
    SILU = 9


class Exec(IntEnum):
    NNA_PW = 0
    NNA_DW = 1
    NNA_STD = 2
    MXU_POOL = 3
    MXU_FC = 4
    MXU_SILU = 5
    MXU_ADD = 6
    MXU_CONCAT = 7
    MXU_UPSAMPLE = 8
    MXU_FOCUS = 9
    MXU_MAXPOOL = 10
    CPU_CONV = 11
    NNA_DENSE = 12
    NNA_K3 = 13


class Act(IntEnum):
    NONE = 0
    RELU = 1
    RELU6 = 2
    SILU = 3


def f32_bits(x: float) -> int:
    """IEEE-754 single-precision bit pattern of x, as a u32."""
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]
