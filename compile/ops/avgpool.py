"""Global average pool -- runs on the CPU/MXU3 (runtime/exec/plt_pool_mxu.c).

Averaged in the real domain and requantized, which is what the MXU3 kernel does
and what ORT does; there is no NNA involvement.
"""
from .. import format as fmt
from .. import nnmath
from .. import onnx_import as oi

OP = fmt.Op.AVGPOOL
EXEC = fmt.Exec.MXU_POOL
NAME = "avgpool"


def match(L):
    return isinstance(L, oi.PoolLayer)


def lower(L, blob):
    return dict(op=OP, exec=EXEC, act=fmt.Act.NONE, in_zp=L.act_in.zero_point)


def simulate(L, xq):
    real = (xq.astype("float64") - L.act_in.zero_point) * L.act_in.scale
    return nnmath.quantize(real.mean(axis=(1, 2)), L.act_out.scale, L.act_out.zero_point)


def dump_blobs(L, act_in, out, outdir, idx):
    act_in.astype("int8").tofile(f"{outdir}/L{idx}_in.bin")
    out.astype("int8").tofile(f"{outdir}/L{idx}_gold.bin")
    return dict(C=act_in.shape[0], H=act_in.shape[1], W=act_in.shape[2])
