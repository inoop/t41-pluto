"""Fully-connected -- runs on the CPU/MXU3 (runtime/exec/plt_fc_mxu.c).

Per-output-channel symmetric weights, so the scale cannot collapse into one
multiplier the way a convolution's can: the requant region is int32 bias[Cout]
followed by fp32 w_scale[Cout].  The output is dequantized fp32 logits.
"""
import numpy as np

from .. import format as fmt
from .. import onnx_import as oi

OP = fmt.Op.FC
EXEC = fmt.Exec.MXU_FC
NAME = "fc"


def match(L):
    return isinstance(L, oi.FCLayer)


def lower(L, blob):
    woff, wsz = blob.add(L.wq.astype(np.int8).tobytes())
    roff, rsz = blob.add(L.bias_q.astype("<i4").tobytes()
                         + L.w_scale.astype("<f4").tobytes())
    return dict(op=OP, exec=EXEC, act=fmt.Act.NONE,
                in_bits=8, w_bits=8, out_bits=32, in_zp=L.act_in.zero_point,
                weight_off=woff, weight_size=wsz, reqtbl_off=roff, reqtbl_size=rsz)


def simulate(L, xq):
    wsum = L.wq.astype(np.int64).sum(axis=1)
    acc = L.wq.astype(np.int64) @ xq.astype(np.int64)
    ACC = acc - L.act_in.zero_point * wsum + L.bias_q.astype(np.int64)
    real = L.act_in.scale * L.w_scale.astype(np.float64) * ACC
    ys, zy = L.act_out.scale, L.act_out.zero_point
    q = np.clip(np.rint(real / ys) + zy, -128, 127)
    return ((q - zy) * ys).astype(np.float32)


def dump_blobs(L, act_in, out, outdir, idx):
    act_in.astype("int8").tofile(f"{outdir}/L{idx}_in.bin")
    L.wq.astype("int8").tofile(f"{outdir}/L{idx}_w.bin")
    open(f"{outdir}/L{idx}_tbl.bin", "wb").write(
        L.bias_q.astype("<i4").tobytes() + L.w_scale.astype("<f4").tobytes())
    out.astype("float32").tofile(f"{outdir}/L{idx}_gold.bin")
    return dict(Cin=L.wq.shape[1], Cout=L.wq.shape[0])
