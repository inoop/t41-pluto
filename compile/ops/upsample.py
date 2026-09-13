"""Nearest-neighbour upsample by an integer factor, then a 256-entry requant
(identity when scale-preserving, which the FPN upsamples are)."""
import numpy as np
from .. import format as fmt, nnmath as nm, requant as rq
from .. import onnx_import as oi

OP = fmt.Op.UPSAMPLE
EXEC = fmt.Exec.MXU_UPSAMPLE
NAME = "upsample"


def match(L):
    return isinstance(L, oi.UpsampleLayer)


def _lut(L):
    return rq.act_table(L.act_in, L.act_out)


def apply_bytes(L, b_in):
    x = np.repeat(np.repeat(np.asarray(b_in, np.uint8), L.scale, 1), L.scale, 2)
    return _lut(L)[x.astype(np.int64)]


def lower(L, blob):
    off, sz = blob.add(rq.act_blob([(L.act_in, L.act_out)]))
    return dict(op=OP, exec=EXEC, in_zp=L.act_in.zero_point,
                reqtbl_off=off, reqtbl_size=sz, params=bytes([L.scale]))


def simulate(L, b_in):
    return apply_bytes(L, b_in)


def dump_blobs(L, act_in, out, outdir, idx):
    (act_in + 128).astype(np.uint8).tofile(f"{outdir}/L{idx}_in.bin")
    (out + 128).astype(np.uint8).tofile(f"{outdir}/L{idx}_gold.bin")
    _lut(L).tofile(f"{outdir}/L{idx}_tbl.bin")
    return dict(scale=L.scale)
