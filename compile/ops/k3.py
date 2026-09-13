"""A dense 3x3 convolution run on the array's NATIVE window walk.

The other conv modules lower their window into pseudo-channels and let the
ordinary 1x1 recipe run it (compile/ops/stem.py, compile/ops/dense.py).  That
cannot reach a wide layer: the 1x1 walk program holds at most 16 input groups
per execution unit, and a lowered KxK window costs KH*KW*Cin/32 of them, so a
3x3 stops at Cin = 96 (measured -- see compile/ops/dense.py:MAX_GROUPS).

Here the window is not lowered at all.  The array steps the nine taps itself, so
the walk costs D = Cin/32 groups and the same limit allows Cin up to 1024 --
every dense 3x3 in YOLOX-S, whose widest is 256.  The recipe is
T41_NNA_MANUAL.md 11.2 with the 8-bit-input substitutions of 11.5.

That makes this the right path for the NARROW 3x3s too, not just the wide ones.
A layer with one input group or less is padded to two (the walk splits the groups
across two execution units) and the runtime aliases the partner's plane onto
group 0's, so the partner's zero weights contribute nothing -- the same trick
plt_conv_pw.c has always used for a single-group 1x1.  The point is what it
avoids: compile/ops/stem.py has to write a KH*KW*Cin-channel window to DRAM
first, and on YOLOX-S that assembly alone was 39 ms of a 245 ms frame.  So this
module is registered ahead of both stem and dense and claims every pad-1 3x3.

What changes from compile/ops/dense.py is only the weight blob: one 1,024-byte
bit-plane tile per (output group, input group, tap), out-group-major, with the
middle kernel row reversed (layout.pack_k3_int8).  The requant table is the same
256 bytes per output group, and the feed offset is the usual 128 + zp: the
array's `A.0f` is the PADDING fill value (manual 13), not a term it subtracts
from real activations, so the bias correction stays exactly as it is elsewhere.
"""
import numpy as np

from .. import format as fmt
from .. import layout
from .. import onnx_import as oi
from . import _conv

OP = fmt.Op.CONV
EXEC = fmt.Exec.NNA_K3
NAME = "k3"

# The walk holds 16 input groups per execution unit, 32 in all (the measurement
# is written up in docs/NNA_POINTWISE.md).  Here a group is a real input group,
# so this is Cin <= 1024 rather than the lowered path's Cin <= 96.
MAX_GROUPS = 32


def match(L):
    return (isinstance(L, oi.ConvLayer) and L.kind == "stem"
            and L.kh == 3 and L.kw == 3
            # Narrower than one group is fine: the blob pads to two groups and
            # the runtime aliases the partner's plane.  In between, the packer
            # has no way to place a part-group, so require whole ones.
            and (L.cin <= 32 or L.cin % 32 == 0)
            and L.stride in ((1, 1), (2, 2))
            and L.pad == (1, 1, 1, 1)
            and L.dilation == (1, 1)
            and L.cin // 32 <= MAX_GROUPS)


def pack_weights(L):
    return layout.pack_k3_int8(L.wq)


def lower(L, blob):
    # _conv.lower's default feed offset (128 + zp) and wsum over every tap are
    # exactly right here; nothing about the window changes the arithmetic.
    return _conv.lower(L, blob, OP, EXEC, pack_weights)


def simulate(L, xq):
    return _conv.simulate(L, xq)


def dump_blobs(L, act_in, out, outdir, idx):
    u = (act_in - L.act_in.zero_point).astype("uint8")
    p = (out + 128).astype("uint8")
    _conv.dump_feature_blobs(outdir, idx, u, p)
    open(f"{outdir}/L{idx}_w.bin", "wb").write(pack_weights(L))
    fo = 128 + L.act_in.zero_point
    wsum = np.asarray(L.wq, np.int64).reshape(L.wq.shape[0], -1).sum(axis=1)
    open(f"{outdir}/L{idx}_tbl.bin", "wb").write(_conv.requant_bytes(L, fo, wsum))
    return dict(Cin=L.cin, Cout=L.cout, H=out.shape[1], W=out.shape[2],
                InH=act_in.shape[1], InW=act_in.shape[2])
