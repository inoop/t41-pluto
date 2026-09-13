"""Elementwise residual add of two int8 activations on independent scales.

Two f32 lookup tables map each input byte to its real contribution in units of
the output scale; the executor sums them, rounds once, and re-biases.  See
compile/ops/silu.py for the byte representation.
"""
import numpy as np
from .. import format as fmt, nnmath as nm
from .. import onnx_import as oi

OP = fmt.Op.ADD
EXEC = fmt.Exec.MXU_ADD
NAME = "add"


def match(L):
    return isinstance(L, oi.AddLayer)


def _tables(L):
    b = np.arange(256)
    fa = nm.to_real(b, L.act_in[0].scale, L.act_in[0].zero_point) / L.act_out.scale
    fb = nm.to_real(b, L.act_in[1].scale, L.act_in[1].zero_point) / L.act_out.scale
    return fa.astype("<f4"), fb.astype("<f4")


def _fixed(L):
    """The add as integer fixed point: acc = Ca*(a-128-za) + Cb*(b-128-zb),
    out = ((acc + 2**(K-1)) >> K) + zo, clamped.

    `K` is the smallest shift that reproduces the float reference EXACTLY for
    every one of the 65536 (a,b) byte pairs -- checked here, so the vectorized
    executor stays byte-identical to compile/ops/add.py:apply_bytes.  Without
    such a K the layer would have to keep the float path.
    """
    sa, za = L.act_in[0].scale, L.act_in[0].zero_point
    sb, zb = L.act_in[1].scale, L.act_in[1].zero_point
    so, zo = L.act_out.scale, L.act_out.zero_point
    b = np.arange(256)
    fa = (sa * ((b - 128) - za)) / so
    fb = (sb * ((b - 128) - zb)) / so
    A, B = b[:, None], b[None, :]
    ref = np.clip(np.rint(fa[A] + fb[B]) + zo, -128, 127).astype(np.int64)
    ap, bp = (b - 128 - za), (b - 128 - zb)
    for K in range(8, 31):
        Ca, Cb = round((sa / so) * (1 << K)), round((sb / so) * (1 << K))
        acc = Ca * ap[A] + Cb * bp[B]
        q = np.clip(((acc + (1 << (K - 1))) >> K) + zo, -128, 127).astype(np.int64)
        if not (q != ref).any():
            return int(Ca), int(Cb), int(K)
    raise ValueError("add: no exact fixed-point shift for this layer")


def apply_bytes(L, ba, bb):
    ra = nm.to_real(ba, L.act_in[0].scale, L.act_in[0].zero_point)
    rb = nm.to_real(bb, L.act_in[1].scale, L.act_in[1].zero_point)
    return nm.to_byte(ra + rb, L.act_out.scale, L.act_out.zero_point,
                      getattr(L.act_out, 'bits', 8))


def lower(L, blob):
    """The blob is six int32: Ca, Cb, K, and the three zero points.  The
    executor evaluates the add with integer vector arithmetic (MXU3), which the
    float tables could not express."""
    Ca, Cb, K = _fixed(L)
    p = np.array([Ca, Cb, K, L.act_in[0].zero_point, L.act_in[1].zero_point,
                  L.act_out.zero_point], "<i4")
    off, sz = blob.add(p.tobytes())
    return dict(op=OP, exec=EXEC, in_zp=L.act_in[0].zero_point,
                reqtbl_off=off, reqtbl_size=sz)


def simulate(L, ba, bb):
    return apply_bytes(L, ba, bb)


def dump_blobs(L, ins, out, outdir, idx):
    (ins[0] + 128).astype(np.uint8).tofile(f"{outdir}/L{idx}_a.bin")
    (ins[1] + 128).astype(np.uint8).tofile(f"{outdir}/L{idx}_b.bin")
    (out + 128).astype(np.uint8).tofile(f"{outdir}/L{idx}_gold.bin")
    fa, fb = _tables(L)
    open(f"{outdir}/L{idx}_tbl.bin", "wb").write(fa.tobytes() + fb.tobytes())
    return dict(N=int(np.asarray(out).size))
