"""A general (non-1x1, non-depthwise, >3 input channel) convolution, run on the
CPU as a straight integer conv.  YOLOX-Nano's only such layer is the 12->16 3x3
after Focus.  Raw int8 weights and a per-output-channel requant (bias,mult);
the executor computes u = byte-128-zp directly, so the table needs no feed
offset -- it is exactly compile/detect_int._conv_bytes in C.
"""
import numpy as np
from .. import format as fmt
from .. import onnx_import as oi
from .. import requant as rq
from . import _conv

OP = fmt.Op.CONV
EXEC = fmt.Exec.CPU_CONV
NAME = "genconv"


def match(L):
    # Fallback for a general conv too wide for the NNA stem recipe (Cin > 32,
    # so the assembled window would span more than one input group per tap).
    return isinstance(L, oi.ConvLayer) and L.kind == "stem" and L.cin > 32


def _table(L):
    B, M = rq.conv_table(L.w_scale, L.bias_q, L.act_in.scale,
                         L.act_out.scale, L.act_out.zero_point)
    return B.astype("<i4").tobytes() + M.astype("<i4").tobytes()


def lower(L, blob):
    woff, wsz = blob.add(np.ascontiguousarray(L.wq, np.int8).tobytes())
    roff, rsz = blob.add(_table(L))
    return dict(op=OP, exec=EXEC, act=fmt.Act.NONE,
                kernel=(L.kh, L.kw), stride=L.stride, pad=L.pad,
                groups=L.groups, dilation=L.dilation,
                in_bits=8, w_bits=8, out_bits=8, in_zp=L.act_in.zero_point,
                weight_off=woff, weight_size=wsz, reqtbl_off=roff, reqtbl_size=rsz)


def simulate(L, xq):
    return _conv.simulate(L, xq)


def dump_blobs(L, act_in, out, outdir, idx):
    u = (act_in - L.act_in.zero_point).astype("uint8")
    p = (out + 128).astype("uint8")
    _conv.dump_feature_blobs(outdir, idx, u, p)
    open(f"{outdir}/L{idx}_w.bin", "wb").write(np.ascontiguousarray(L.wq, np.int8).tobytes())
    open(f"{outdir}/L{idx}_tbl.bin", "wb").write(_table(L))
    return dict(Cin=L.cin, Cout=L.cout, H=act_in.shape[1], W=act_in.shape[2])
