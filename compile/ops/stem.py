"""The 3x3 stem -- run on the NNA as a 1x1 convolution over 32 taps.

MobileNetV1's first layer is 3 -> 32 channels, 3x3, stride 2, pad 1.  The array
has no 3-channel mode, but a 3x3 window over 3 colour channels is 27 values,
which fits inside ONE 32-channel group.  So the window is assembled before the
push and the array runs an ordinary 1x1 convolution over 32 pseudo-channels
(manual 8.6) -- the recipe that is already device-verified.

The tap order is pluto's own, `t = (ky*3 + kx)*3 + c`, and lives in
compile/layout.py next to the im2col and the packer that both use it.  Input
groups are padded to two because the 1x1 recipe splits them across the array's
two execution units.

tests/stem_test.py checks, on the real layer, that this reformulation is
*identical* to the 3x3 convolution it replaces.

No model reaches it today.  compile/ops/k3.py is tried first and takes every
pad-1 3x3 including the narrow ones -- it pads Cin to two groups and aliases the
partner's plane, so it needs no assembled window at all, and on YOLOX-S the three
layers that used to come here cost 73.8 ms through this path against 35 through
that one, most of the difference being the window this module writes to DRAM.
What is left for this module is a window k3 cannot walk: a 5x5, or a 3x3 whose
padding is not 1.
"""
import numpy as np

from .. import format as fmt
from .. import layout
from .. import onnx_import as oi
from . import _conv

# The array reads the assembled window byte as UNSIGNED.  Feeding the raw
# activation byte b = q + 128 means the MAC sees an extra (128 + x_zp) per tap,
# a constant the requant bias absorbs (feed_offset = 128 + x_zp; see
# _conv.requant_bytes).  For MobileNetV1's image (x_zp = 0) that is 128; a
# post-Focus conv whose input is an ordinary asymmetric activation uses its own.
OP = fmt.Op.CONV
EXEC = fmt.Exec.NNA_STD
NAME = "stem"


def _feed_offset(L):
    return 128 + L.act_in.zero_point


def match(L):
    return isinstance(L, oi.ConvLayer) and L.kind == "stem" and L.cin <= 32


def pack_weights(L):
    return layout.pack_stem_int8(L.wq)


def _wsum(L):
    """Per output channel, the sum over all tap columns -- what the feed offset
    multiplies into the accumulator."""
    return layout.stem_weight_matrix(L.wq).astype(np.int64).sum(axis=1)


def lower(L, blob):
    return _conv.lower(L, blob, OP, EXEC, pack_weights,
                       feed_offset=_feed_offset(L), wsum=_wsum(L))


def simulate(L, xq):
    return _conv.simulate(L, xq)


def dump_blobs(L, act_in, out, outdir, idx):
    """Two input blobs, because the stem is brought up in two steps.

      L<k>_col.bin  the assembled 32-tap window, padded to two groups -- feed
                    this to the pointwise path to test the array side alone
      L<k>_in.bin   the raw 3-channel input, one group -- what plt_run_stem
                    gets, and what the engine will hand the executor
    """
    fo = _feed_offset(L)
    u = (act_in - L.act_in.zero_point).astype("int16")
    col = layout.stem_im2col(u, L.kh, L.kw, L.stride, L.pad) + fo   # fed bytes b

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
