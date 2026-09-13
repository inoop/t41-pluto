"""What the three convolution kinds have in common.

Each kind still gets its own module -- its own weight packing, its own executor
on the device -- but the lowering and the simulation of a quantized convolution
are identical once the packer is chosen.
"""
import numpy as np

from .. import format as fmt
from .. import layout, nnmath, requant as rq


def simulate(L, xq):
    """One quantized convolution: int accumulate, then the array's packed requant.

    Returns signed int8 with zero point -128, which is what the array's unsigned
    packed byte means once it is read as the next layer's input.
    """
    u = xq - L.act_in.zero_point
    acc = nnmath.conv_acc(u, L.wq, L.kh, L.kw, L.stride, L.pad,
                          depthwise=(L.kind == "depthwise"))
    B, M = rq.conv_table(L.w_scale, L.bias_q, L.act_in.scale,
                         L.act_out.scale, L.act_out.zero_point)
    nb = getattr(L, "out_bits", 8)
    return rq.apply_packed(acc, B, M, out_bits=nb) - (1 << (nb - 1))


def requant_bytes(L, feed_offset=0, wsum=None):
    """The requant table in the array's order: int32 bias[32] then int32 mult[32],
    256 bytes per output group.

    `feed_offset` covers layers whose activation byte is not `u = x_q - x_zp`
    itself.  The array reads that byte as UNSIGNED, so a layer whose input zero
    point is not -128 must be fed `u + offset` to keep it in [0,255].  The
    convolution is linear, so the whole correction folds into the bias:

        acc' = sum((u + offset) * w) = acc + offset * sum(w)

    and `4*acc + B == 4*acc' + (B - 4*offset*sum(w))`, exactly, in integers.
    """
    B, M = rq.conv_table(L.w_scale, L.bias_q, L.act_in.scale,
                         L.act_out.scale, L.act_out.zero_point)
    if feed_offset:
        B = B - 4 * feed_offset * np.asarray(wsum, np.int64)
    return layout.requant_table_bytes(B.astype("<i4"), M)


def lower(L, blob, op, exec_, pack_weights, feed_offset=None, wsum=None):
    """Add one convolution's constants to the blob and describe the layer.

    `pack_weights(L) -> bytes` is the kind's packer: the blob holds exactly the
    bytes the executor feeds to the array, not a convenience copy.

    The array is fed the raw activation byte b = q + 128, so its MAC sees
    u' = b = (q - x_zp) + (128 + x_zp); the extra (128 + x_zp) per input channel
    is a constant the requant bias absorbs (feed_offset below).  For a ReLU layer
    with x_zp = -128 the offset is 0 and nothing changes -- which is why
    MobileNetV1 is unaffected -- but an asymmetric layer (YOLOX) needs it.
    """
    if feed_offset is None:
        feed_offset = 128 + L.act_in.zero_point
        wsum = L.wq.reshape(L.wq.shape[0], -1).sum(axis=1)
    woff, wsz = blob.add(pack_weights(L))
    # A folded SiLU rides after the requant table, so the executor finds it at a
    # fixed offset from the table it already has (compile/model_build:_fuse_silu).
    tbl = requant_bytes(L, feed_offset, wsum)
    lut = getattr(L, "fused_silu_lut", None)
    act = fmt.Act.RELU if lut is None else fmt.Act.SILU
    if lut is not None:
        tbl = tbl + lut
    roff, rsz = blob.add(tbl)
    return dict(op=op, exec=exec_, act=act,
                kernel=(L.kh, L.kw), stride=L.stride, pad=L.pad,
                groups=L.groups, dilation=L.dilation,
                in_bits=getattr(L, "in_bits", 8),
                w_bits=getattr(L, "w_bits", 8),
                out_bits=getattr(L, "out_bits", 8),
                in_zp=L.act_in.zero_point,
                weight_off=woff, weight_size=wsz, reqtbl_off=roff, reqtbl_size=rsz)


def pad_to_two_groups(u, wq):
    """The array's 1x1 recipe splits the input groups across two execution units,
    so it needs at least two.  A 32-channel input is padded to 64 with zero
    channels and zero weights: the second group contributes nothing, at the cost
    of running it.  (The alternative is the array's separate single-input-group
    recipe -- manual 8.4.1 -- which pluto does not implement.)"""
    if u.shape[0] != 32:
        return u, wq
    return (np.concatenate([u, np.zeros_like(u)], axis=0),
            np.concatenate([wq, np.zeros_like(wq)], axis=1))


def dump_feature_blobs(outdir, idx, u, p):
    """The two feature-map blobs every conv device test needs."""
    layout.pack_ndhwc32(u.astype(np.int8)).astype(np.uint8).tofile(f"{outdir}/L{idx}_in.bin")
    layout.pack_ndhwc32(p.astype(np.int8)).astype(np.uint8).tofile(f"{outdir}/L{idx}_gold.bin")
