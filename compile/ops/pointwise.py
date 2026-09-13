"""1x1 convolution -- the NNA pointwise executor (runtime/exec/plt_conv_pw.c).

Device-verified: all 13 MobileNetV1 pointwise layers are byte-exact against
this module's simulate().  See docs/NNA_POINTWISE.md.
"""
from .. import format as fmt
from .. import layout
from .. import onnx_import as oi
from . import _conv

OP = fmt.Op.CONV
EXEC = fmt.Exec.NNA_PW
NAME = "pointwise"


def match(L):
    return isinstance(L, oi.ConvLayer) and L.kind == "pointwise"


def pack_weights(L):
    # The 1x1 recipe splits input groups across the array's two execution units,
    # so it needs at least two groups (>= 64 input channels); a smaller layer is
    # zero-padded up and its partner contributes nothing.  Wider layers pad only
    # to the next whole group.
    wq = L.wq.reshape(L.cout, L.cin, 1, 1)
    cin_p = max(64, layout.pad32(L.cin))
    if cin_p != L.cin:
        pad = layout.np.zeros((L.cout, cin_p - L.cin, 1, 1), wq.dtype)
        wq = layout.np.concatenate([wq, pad], axis=1)
    return layout.pack_pointwise_int8(wq)


def lower(L, blob):
    return _conv.lower(L, blob, OP, EXEC, pack_weights)


def simulate(L, xq):
    return _conv.simulate(L, xq)


def dump_blobs(L, act_in, out, outdir, idx):
    """in / gold / w / tbl, for driving one layer on the device by hand."""
    u = (act_in - L.act_in.zero_point).astype("uint8")
    p = (out + 128).astype("uint8")
    wq = L.wq.reshape(L.cout, L.cin, 1, 1)
    u, wq = _conv.pad_to_two_groups(u, wq)   # the same padding lower() applies

    _conv.dump_feature_blobs(outdir, idx, u, p)
    open(f"{outdir}/L{idx}_w.bin", "wb").write(layout.pack_pointwise_int8(wq))
    open(f"{outdir}/L{idx}_tbl.bin", "wb").write(_conv.requant_bytes(L))
    return dict(Cin=wq.shape[1], Cout=L.cout, H=act_in.shape[1], W=act_in.shape[2])
