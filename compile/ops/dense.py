"""A dense KxK convolution over 32 or more input channels, run on the NNA as a
1x1 over pseudo-channels.

This is the stem's reformulation (compile/ops/stem.py) carried to a window too
wide to assemble: a KH x KW window over Cin channels is KH*KW*Cin values, and
once they are laid out as pseudo-channels the array runs its ordinary 1x1
recipe over them.  Nothing new is asked of the array -- plt_conv_dense.c calls
plt_conv_pw_configure() unchanged.

What differs from the stem is that the window is never built in memory.  The
KH*KW taps are KH*KW SHIFTED VIEWS of one bordered input plane, so the runtime
feeds them straight from the input, one feed per (input group, tap).  That is
why the pseudo-channel order here is GROUP-MAJOR (layout.dense_tap): the taps of
one group read overlapping cache lines, and only the step to the next group's
plane is expensive.

No model here reaches it today: YOLOX-S's 28 dense 3x3s go to compile/ops/k3.py,
which is both faster and not capped at Cin = 96, and every 3x3 in YOLOX-Nano and
MobileNetV1 is depthwise or fits the stem.  It stays because a window k3 does not
serve -- a 5x5, say -- is legal, and this is the only thing that would put one on
the array at all.

tests/dense_test.py checks, on a real YOLOX-S layer, that the reformulation is
*identical* to the KxK convolution it replaces.
"""
import numpy as np

from .. import format as fmt
from .. import layout
from .. import onnx_import as oi
from . import _conv

OP = fmt.Op.CONV
EXEC = fmt.Exec.NNA_DENSE
NAME = "dense"

# The 1x1 recipe walks at most 16 input groups PER EXECUTION UNIT -- 32 in all.
#
# Device-measured, not inferred (2026-09-11).  A real 256->256 pointwise layer
# was widened with zero weights and zero activations, which cannot change its
# output, and run at a sweep of group counts: D = 8, 16, 24, 27, 28, 30, 32 are
# byte-exact and D = 33 (nA=16, nB=17) and D = 36 are not.  The limit is the
# walk program's repeat count (docs/NNA_POINTWISE.md "Program words",
# `load(n-2)`), whose field holds 4 bits, so n <= 16.
#
# It is not the tap-slot budget: 288 slots hold 72 groups of 8-bit weights, and
# a D=18 layer pushing all 288 slices is correct.
#
# A KxK window needs KH*KW*Cin/32 groups, so a 3x3 tops out at Cin = 96.  That
# is why compile/ops/k3.py exists: on the array's NATIVE window walk the taps
# cost nothing, the walk covers D groups rather than KH*KW*D, and a 3x3 reaches
# Cin = 1024.  k3 is tried first and takes every pad-1 3x3, so what is left for
# this module is the windows it does not serve -- a 5x5, or a 3x3 with padding
# the native recipe is not written for -- and genconv.py backs up both.
MAX_GROUPS = 32


def _groups(L):
    return L.kh * L.kw * (L.cin // 32)


def _feed_offset(L):
    return 128 + L.act_in.zero_point


def match(L):
    return (isinstance(L, oi.ConvLayer) and L.kind == "stem"
            and L.cin > 32 and L.cin % 32 == 0
            and L.dilation == (1, 1)
            and _groups(L) <= MAX_GROUPS)


def pack_weights(L):
    return layout.pack_dense_int8(L.wq)


def _wsum(L):
    """Per output channel, the sum over all tap columns -- what the feed offset
    multiplies into the accumulator."""
    return layout.dense_weight_matrix(L.wq).astype(np.int64).sum(axis=1)


def lower(L, blob):
    return _conv.lower(L, blob, OP, EXEC, pack_weights,
                       feed_offset=_feed_offset(L), wsum=_wsum(L))


def simulate(L, xq):
    return _conv.simulate(L, xq)


def dump_blobs(L, act_in, out, outdir, idx):
    """As the stem dumps them, with the same two views of the input.

      L<k>_col.bin  the lowered window -- feed this to the pointwise path to
                    test the array side on its own
      L<k>_in.bin   the plain feature map, what the executor actually gets
    """
    fo = _feed_offset(L)
    u = (act_in - L.act_in.zero_point).astype("int16")
    col = layout.dense_im2col(u, L.kh, L.kw, L.stride, L.pad) + fo   # fed bytes b

    layout.pack_ndhwc32(col.astype("uint8").astype("int8")) \
          .astype("uint8").tofile(f"{outdir}/L{idx}_col.bin")
    layout.pack_ndhwc32(u.astype("uint8").astype("int8")) \
          .astype("uint8").tofile(f"{outdir}/L{idx}_in.bin")
    layout.pack_ndhwc32((out + 128).astype("uint8").astype("int8")) \
          .astype("uint8").tofile(f"{outdir}/L{idx}_gold.bin")

    open(f"{outdir}/L{idx}_w.bin", "wb").write(pack_weights(L))
    open(f"{outdir}/L{idx}_tbl.bin", "wb").write(
        _conv.requant_bytes(L, fo, _wsum(L)))
    return dict(Cin=col.shape[0], Cout=L.cout, H=out.shape[1], W=out.shape[2],
                InH=act_in.shape[1], InW=act_in.shape[2])
