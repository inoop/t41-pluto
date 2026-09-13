"""SiLU activation x*sigmoid(x) as a 256-entry int8->int8 lookup table.

The input is a single int8 byte, so the table is the whole function: for every
possible input value there is one exact output value, computed once at compile
time from the input and output quantizations.  Runs as an elementwise MXU/scalar
pass over the tensor (runtime/exec/plt_silu.c), independent of NDHWC32 packing.

Representation: activations flow as the unsigned byte b = q + 128, where q is the
tensor's ORT int8 value; real = scale * ((b - 128) - zero_point).  See
docs/ADDING_A_LAYER.md and the pluto-yolox-nano-detection design notes.
"""
import numpy as np

from .. import format as fmt
from .. import onnx_import as oi

OP = fmt.Op.SILU
EXEC = fmt.Exec.MXU_SILU
NAME = "silu"


def match(L):
    return isinstance(L, oi.SiluLayer)


def build_lut(L):
    """uint8[256]: lut[b_in] = b_out, both in the q+128 byte representation."""
    b = np.arange(256)
    real = L.act_in.scale * ((b - 128) - L.act_in.zero_point)
    y = real / (1.0 + np.exp(-real))
    q = np.clip(np.rint(y / L.act_out.scale) + L.act_out.zero_point, -128, 127)
    return (q + 128).astype(np.uint8)


def apply_bytes(L, b_in):
    """Reference: the byte-domain SiLU, i.e. what the device executor computes."""
    return build_lut(L)[np.asarray(b_in, np.int64) & 0xFF]


def lower(L, blob):
    off, sz = blob.add(build_lut(L).tobytes())
    return dict(op=OP, exec=EXEC, act=fmt.Act.SILU,
                in_bits=8, w_bits=8, out_bits=8, in_zp=L.act_in.zero_point,
                reqtbl_off=off, reqtbl_size=sz)


def simulate(L, b_in):
    return apply_bytes(L, b_in)


def dump_blobs(L, act_in, out, outdir, idx):
    b_in = (act_in + 128).astype(np.uint8)
    b_out = (out + 128).astype(np.uint8)
    b_in.tofile(f"{outdir}/L{idx}_in.bin")
    b_out.tofile(f"{outdir}/L{idx}_gold.bin")
    build_lut(L).tofile(f"{outdir}/L{idx}_tbl.bin")
    return dict(N=int(np.asarray(act_in).size))
