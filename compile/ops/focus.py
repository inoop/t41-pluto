"""Focus / space-to-depth stem: a 2x2 pixel block becomes 4x the channels,
[C,H,W] -> [4C, H/2, W/2], then a 256-entry requant (identity when scale
preserving).  Slice order matches YOLOX: (::2,::2),(1::2,::2),(::2,1::2),(1::2,1::2)."""
import numpy as np
from .. import format as fmt, nnmath as nm, requant as rq
from .. import onnx_import as oi

OP = fmt.Op.FOCUS
EXEC = fmt.Exec.MXU_FOCUS
NAME = "focus"


def match(L):
    return isinstance(L, oi.FocusLayer)


def _lut(L):
    return rq.act_table(L.act_in, L.act_out)


def apply_bytes(L, b_in):
    x = np.asarray(b_in, np.uint8)
    packed = np.concatenate([x[:, 0::2, 0::2], x[:, 1::2, 0::2],
                             x[:, 0::2, 1::2], x[:, 1::2, 1::2]], 0)
    return _lut(L)[packed.astype(np.int64)]


def lower(L, blob):
    off, sz = blob.add(rq.act_blob([(L.act_in, L.act_out)]))
    return dict(op=OP, exec=EXEC, in_zp=L.act_in.zero_point,
                reqtbl_off=off, reqtbl_size=sz)


def simulate(L, b_in):
    return apply_bytes(L, b_in)


def dump_blobs(L, act_in, out, outdir, idx):
    (act_in + 128).astype(np.uint8).tofile(f"{outdir}/L{idx}_in.bin")
    (out + 128).astype(np.uint8).tofile(f"{outdir}/L{idx}_gold.bin")
    _lut(L).tofile(f"{outdir}/L{idx}_tbl.bin")
    return dict()
