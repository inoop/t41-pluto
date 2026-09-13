"""Windowed max in the byte domain (byte order is monotonic in the real value,
so the max byte is the max value), then a 256-entry requant to the output scale
(identity when the pool is scale-preserving, which YOLOX's SPP pools are)."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from .. import format as fmt, nnmath as nm, requant as rq
from .. import onnx_import as oi

OP = fmt.Op.MAXPOOL
EXEC = fmt.Exec.MXU_MAXPOOL
NAME = "maxpool"


def match(L):
    return isinstance(L, oi.MaxPoolLayer)


def _lut(L):
    return rq.act_table(L.act_in, L.act_out)


def apply_bytes(L, b_in):
    x = np.asarray(b_in, np.uint8)
    xp = np.pad(x, ((0, 0), (L.pad[0], L.pad[1]), (L.pad[2], L.pad[3])))  # pad byte 0 = min real
    win = sliding_window_view(xp, (L.kh, L.kw), axis=(1, 2))[:, ::L.stride[0], ::L.stride[1]]
    return _lut(L)[win.max(axis=(3, 4)).astype(np.int64)]


def lower(L, blob):
    off, sz = blob.add(rq.act_blob([(L.act_in, L.act_out)]))
    return dict(op=OP, exec=EXEC, kernel=(L.kh, L.kw), stride=L.stride, pad=L.pad,
                in_zp=L.act_in.zero_point, reqtbl_off=off, reqtbl_size=sz)


def simulate(L, b_in):
    return apply_bytes(L, b_in)


def dump_blobs(L, act_in, out, outdir, idx):
    (act_in + 128).astype(np.uint8).tofile(f"{outdir}/L{idx}_in.bin")
    (out + 128).astype(np.uint8).tofile(f"{outdir}/L{idx}_gold.bin")
    _lut(L).tofile(f"{outdir}/L{idx}_tbl.bin")
    return dict(kh=L.kh, kw=L.kw)
