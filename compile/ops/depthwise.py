"""Depthwise 3x3 -- the NNA depthwise executor (runtime/exec/plt_conv_dw.c).

Depthwise does not reduce across channels, so each (channel-group, tap) weight
tile is BLOCK-DIAGONAL: the layer becomes an ordinary 3x3 convolution the
array's tap-walk and MAC path already understand.  Wasteful -- 32x the MACs --
but it needs no undiscovered hardware mode.  See docs/ADDING_A_LAYER.md 2.

The packing is host-tested (tests/depthwise_pack_test.py); the executor is
Phase 6 and not yet on the device.
"""
from .. import format as fmt
from .. import layout
from .. import onnx_import as oi
from . import _conv

OP = fmt.Op.DWCONV
EXEC = fmt.Exec.NNA_DW
NAME = "depthwise"


def match(L):
    return isinstance(L, oi.ConvLayer) and L.kind == "depthwise"


def pack_weights(L):
    return layout.pack_depthwise_int8(L.wq)


def lower(L, blob):
    return _conv.lower(L, blob, OP, EXEC, pack_weights)


def simulate(L, xq):
    return _conv.simulate(L, xq)


def dump_blobs(L, act_in, out, outdir, idx):
    u = (act_in - L.act_in.zero_point).astype("uint8")
    p = (out + 128).astype("uint8")
    _conv.dump_feature_blobs(outdir, idx, u, p)
    open(f"{outdir}/L{idx}_w.bin", "wb").write(pack_weights(L))
    open(f"{outdir}/L{idx}_tbl.bin", "wb").write(_conv.requant_bytes(L))
    return dict(Cin=L.cin, Cout=L.cout, H=act_in.shape[1], W=act_in.shape[2])
