"""Channel concatenation; each input requantized to the output scale by its own
256-entry int8->int8 table, then copied into its channel range."""
import numpy as np
from .. import format as fmt, nnmath as nm, requant as rq
from .. import onnx_import as oi

OP = fmt.Op.CONCAT
EXEC = fmt.Exec.MXU_CONCAT
NAME = "concat"


def match(L):
    return isinstance(L, oi.ConcatLayer)


def _lut(q_in, q_out):
    return rq.act_table(q_in, q_out)


def apply_bytes(L, bins):
    outs = [_lut(qi, L.act_out)[np.asarray(b, np.int64) & 0xFF]
            for b, qi in zip(bins, L.act_in)]
    return np.concatenate(outs, axis=0)          # arrays are [C,H,W]


def lower(L, blob):
    off, sz = blob.add(rq.act_blob([(qi, L.act_out) for qi in L.act_in]))
    return dict(op=OP, exec=EXEC, in_zp=L.act_in[0].zero_point,
                reqtbl_off=off, reqtbl_size=sz)


def simulate(L, *bins):
    return apply_bytes(L, list(bins))


def dump_blobs(L, ins, out, outdir, idx):
    for j, a in enumerate(ins):
        (a + 128).astype(np.uint8).tofile(f"{outdir}/L{idx}_in{j}.bin")
    (out + 128).astype(np.uint8).tofile(f"{outdir}/L{idx}_gold.bin")
    open(f"{outdir}/L{idx}_tbl.bin", "wb").write(
        b"".join(_lut(qi, L.act_out).tobytes() for qi in L.act_in))
    return dict(N=len(ins))
