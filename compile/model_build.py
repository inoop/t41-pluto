"""Build a PlutoModel (and .pluto file) from the chain IR.

Every layer is lowered by its module in compile/ops/, which owns both the blob
bytes and the layer descriptor.  The blob therefore holds exactly what the
device executor reads -- packed weights in the array's layout, and the requant
table in the array's order -- not a convenience copy of the weights.

Tensor shapes are stored as [N, C, H, W] regardless of `format`; `format` says
only how those elements are laid out in memory.  A vector is [1, C, 1, 1].
"""
from . import format as fmt
from . import onnx_import as oi
from . import ops
from . import shapes
from .model_writer import PlutoTensor, PlutoLayer, PlutoModel


class _Blob:
    def __init__(self):
        self.buf = bytearray()

    def add(self, data: bytes, align=64):
        pad = (-len(self.buf)) % align
        self.buf.extend(b"\x00" * pad)
        off = len(self.buf)
        self.buf.extend(data)
        return off, len(data)


def _tensor_meta(L, tid):
    """(dtype, format, name) for a layer's output tensor.  The FC emits fp32
    logits; a pool emits a vector; every other activation is packed NDHWC32."""
    if isinstance(L, oi.FCLayer):
        return fmt.DType.FP32, fmt.Format.VEC, "logits"
    if isinstance(L, oi.PoolLayer):
        return fmt.DType.INT8, fmt.Format.VEC, f"act{tid}"
    return fmt.DType.INT8, fmt.Format.NDHWC32, f"act{tid}"


def _fuse_silu(model):
    """Fold a SiLU into the convolution that feeds it.

    The convolution already writes every output byte through the array's
    requant; a following SiLU is a second FULL PASS over that buffer -- a cold
    read and a write of every byte -- only to put each byte through a 256-entry
    table.  Applying the table to each tile as the convolution drains it, while
    the tile is still hot, removes the pass outright.  The table pass itself is
    vectorized too: MXU3.1's `gshufvb` IS a data-dependent byte gather, so the
    256 scalar lookups are 4 x 19 vector instructions (runtime/hal/plt_mxu3.h,
    PLT_M3_LUT64) -- 0.221 us against 0.907 us per tile, measured.

    This is expressed as a real property of the layer -- Act.SILU on the
    convolution, its table carried directly after the requant table -- so the
    .pluto says what the layer computes instead of the runtime inferring it.
    Only a convolution whose output has exactly ONE consumer can be folded; a
    branch still needs the unfused value.
    """
    from dataclasses import replace
    from .ops import silu as silu_op

    uses = {}
    for L in model.layers:
        for n in L.inputs:
            uses[n] = uses.get(n, 0) + 1
    for n in model.output_names:
        uses[n] = uses.get(n, 0) + 1

    consumers = {}
    for L in model.layers:
        for n in L.inputs:
            consumers.setdefault(n, []).append(L)

    dropped, out = set(), []
    for L in model.layers:
        if id(L) in dropped:
            continue
        if isinstance(L, oi.ConvLayer) and uses.get(L.output, 0) == 1:
            mod = ops.find(L)
            cs = consumers.get(L.output, [])
            if (mod is not None and mod.NAME in ("pointwise", "depthwise", "stem", "dense", "k3")
                    and len(cs) == 1 and isinstance(cs[0], oi.SiluLayer)):
                sl = cs[0]
                fused = replace(L, output=sl.output)
                fused.fused_silu_lut = silu_op.build_lut(sl).tobytes()
                dropped.add(id(sl))
                out.append(fused)
                continue
        out.append(L)
    model.layers = out
    return len(dropped)


def build(model: oi.Model) -> PlutoModel:
    blob = _Blob()
    _fuse_silu(model)
    shp, qnt = shapes.propagate(model, *model.input_shape)

    # id 0 is the graph input, handed to the runtime already in the array's
    # layout as the byte b = q + 128 (see compile/detect_int.py).
    ic, ih, iw = model.input_shape
    tensors = [PlutoTensor(0, "input", fmt.DType.INT8, fmt.Format.NDHWC32, 4,
                           (1, ic, ih, iw),
                           scale=model.input_quant.scale,
                           zero_point=model.input_quant.zero_point)]

    # Wire by logical tensor name: a branch (an input produced several layers
    # back, or read by more than one consumer) resolves through name2id.  A chain,
    # where every input is the previous output, yields exactly the sequential ids
    # the old chain builder assigned -- so MobileNetV1 is byte-identical.
    name2id = {model.input_name: 0}
    layers = []
    next_tid = 1

    for i, L in enumerate(model.layers):
        desc = ops.find_or_die(L).lower(L, blob)
        out = next_tid
        next_tid += 1
        name2id[L.output] = out
        in_ids = tuple(name2id[n] for n in L.inputs)

        c, hh, ww = shp[L.output]
        q = qnt[L.output]
        dtype, layout_fmt, name = _tensor_meta(L, out)
        tensors.append(PlutoTensor(out, name, dtype, layout_fmt, 4, (1, c, hh, ww),
                                   scale=q.scale, zero_point=q.zero_point))
        layers.append(PlutoLayer(i, inputs=in_ids, output=out, **desc))

    out_ids = tuple(name2id[n] for n in model.output_names) or (next_tid - 1,)
    return PlutoModel(tensors=tensors, layers=layers, inputs=(0,), outputs=out_ids,
                      blob=bytes(blob.buf), nmem_arena_hint=len(blob.buf))
