"""Import a QDQ int8 ONNX model into pluto's chain IR.

MobileNetV1's int8 graph is a straight chain of Conv / GlobalAveragePool /
Gemm ops wrapped in QuantizeLinear/DequantizeLinear fake-quant pairs (ReLU is
folded into the activation quant range, so there are no explicit Relu nodes).
This module harvests, per op, the int8 weights, per-output-channel weight
scales, int32 bias, and the per-tensor activation scale/zero-point that ORT
attached, and classifies each conv as stem / depthwise / pointwise.

The harvested scales are exactly those that produced the reference logits, so
a faithful integer datapath over this IR reproduces the ORT-int8 output.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import onnx
import onnx.numpy_helper as nph


@dataclass
class Quant:
    scale: float
    zero_point: int
    bits: int = 8             # 8 unless compile/quant4.py re-quantized this tensor


@dataclass
class ConvLayer:
    name: str
    kind: str                 # "stem" | "depthwise" | "pointwise"
    wq: np.ndarray            # int8 [Cout, Cin_per_group, KH, KW]
    w_scale: np.ndarray       # f32 [Cout]  (per output channel, symmetric)
    bias_q: np.ndarray        # int32 [Cout]
    groups: int
    stride: tuple
    pad: tuple                # (t, b, l, r)
    dilation: tuple
    act_in: Quant
    act_out: Quant
    cin: int
    cout: int
    kh: int
    kw: int
    inputs: tuple = ()        # logical source tensor names (filled by load())
    output: str = ""          # logical output tensor name


@dataclass
class PoolLayer:
    name: str
    kind: str                 # "global_avg"
    act_in: Quant
    act_out: Quant
    inputs: tuple = ()
    output: str = ""


@dataclass
class FCLayer:
    name: str
    wq: np.ndarray            # int8 [Cout, Cin]
    w_scale: np.ndarray       # f32 [Cout]
    bias_q: np.ndarray        # int32 [Cout]
    act_in: Quant
    act_out: Quant            # logits fake-quant
    inputs: tuple = ()
    output: str = ""


# --- detection ops (YOLOX-Nano) --------------------------------------------
# These are branchy, so they carry per-input Quants: an Add or Concat mixes
# activations quantized on different scales and must requant each to its own
# output scale.  Single-input ops keep one act_in.

@dataclass
class SiluLayer:
    """x * sigmoid(x), the YOLOX activation.  Bit-exact as a 256-entry int8->int8
    LUT built from (act_in, act_out) -- the conv output is int8, so the table is
    the whole function."""
    name: str
    act_in: Quant
    act_out: Quant
    inputs: tuple = ()
    output: str = ""


@dataclass
class AddLayer:
    """Elementwise residual add of two int8 activations on independent scales."""
    name: str
    act_in: tuple             # (Quant, Quant), one per input
    act_out: Quant
    inputs: tuple = ()
    output: str = ""


@dataclass
class ConcatLayer:
    """Channel concatenation; each input requantized to the output scale."""
    name: str
    axis: int
    act_in: tuple             # (Quant, ...), one per input
    act_out: Quant
    inputs: tuple = ()
    output: str = ""


@dataclass
class MaxPoolLayer:
    name: str
    kh: int
    kw: int
    stride: tuple
    pad: tuple                # (t, b, l, r)
    act_in: Quant
    act_out: Quant
    inputs: tuple = ()
    output: str = ""


@dataclass
class UpsampleLayer:
    """Nearest-neighbour resize by an integer factor (FPN 2x)."""
    name: str
    scale: int
    act_in: Quant
    act_out: Quant
    inputs: tuple = ()
    output: str = ""


@dataclass
class FocusLayer:
    """Space-to-depth stem: 2x2 pixel block -> channels, HxWx3 -> H/2 x W/2 x 12."""
    name: str
    act_in: Quant
    act_out: Quant
    inputs: tuple = ()
    output: str = ""


@dataclass
class Model:
    input_quant: Quant
    layers: List[object] = field(default_factory=list)
    input_name: str = ""                 # logical name of the graph input activation
    output_names: tuple = ()             # logical names of the graph outputs
    input_shape: tuple = (3, 224, 224)   # (C, H, W) of the graph input


def _attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            if a.ints:
                return list(a.ints)
            if a.type == onnx.AttributeProto.FLOAT:
                return a.f
            return a.i
    return default


class _Graph:
    def __init__(self, path):
        self.m = onnx.load(path)
        self.g = self.m.graph
        self.inits = {i.name: i for i in self.g.initializer}
        self.prod = {o: n for n in self.g.node for o in n.output}

    def const(self, name):
        if name in self.inits:
            return nph.to_array(self.inits[name])
        n = self.prod.get(name)
        if n is not None and n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    return nph.to_array(a.t)
        return None

    def dq(self, name):
        """(source, scale_array, zp_array) if `name` is a DequantizeLinear output."""
        n = self.prod.get(name)
        if n is None or n.op_type != "DequantizeLinear":
            return None
        z = self.const(n.input[2]) if len(n.input) > 2 else np.array(0, np.int8)
        return n.input[0], self.const(n.input[1]), z

    def out_quant(self, node):
        """The QuantizeLinear that consumes this node's output (scale, zp)."""
        for q in self.g.node:
            if q.op_type == "QuantizeLinear" and q.input[0] == node.output[0]:
                z = self.const(q.input[2]) if len(q.input) > 2 else np.array(0, np.int8)
                return Quant(float(self.const(q.input[1])), int(z))
        return None

    def act_source(self, name):
        """Logical producer of a compute node's int8 input `name`.

        A quantized edge is `<producer> -> QuantizeLinear -> DequantizeLinear ->
        <consumer>`; the fake-quant pair carries the scale but is not a layer.
        Skipping it collapses the QDQ graph to the graph of real operations, and
        the name returned -- the producer node's own output tensor -- is the
        stable id every consumer of that activation agrees on.  A graph input or
        initializer has no such pair and is returned unchanged.
        """
        # Pure layout ops carry no arithmetic and no layer of their own; pluto's
        # tensors are already logically [N,C,H,W], so a Flatten/Reshape/Squeeze
        # that only reshapes before an FC is transparent -- walk through it to
        # the activation that actually produced the values.
        SKIP = {"Flatten", "Reshape", "Squeeze", "Unsqueeze", "Identity"}
        n = self.prod.get(name)
        if n is not None and n.op_type == "DequantizeLinear":
            src = n.input[0]
            q = self.prod.get(src)
            if q is not None and q.op_type == "QuantizeLinear":
                src = q.input[0]
            p = self.prod.get(src)
            if p is not None and p.op_type in SKIP:
                return self.act_source(p.input[0])
            return src
        p = self.prod.get(name)
        if p is not None and p.op_type in SKIP:
            return self.act_source(p.input[0])
        return name

    def in_quant(self, name):
        """Quant on a compute node's int8 input `name` (its DequantizeLinear)."""
        d = self.dq(name)
        if d is None:
            return None
        return Quant(float(d[1]), int(d[2]))

    def tensor_quant(self, name):
        """Quant a tensor is stored at: the QuantizeLinear that consumes it."""
        for q in self.g.node:
            if q.op_type == "QuantizeLinear" and q.input[0] == name:
                z = self.const(q.input[2]) if len(q.input) > 2 else np.array(0, np.int8)
                return Quant(float(self.const(q.input[1])), int(z))
        return None

    def silu_source(self, mul):
        """If `mul` is the Mul of a SiLU (x * Sigmoid(x)), return (x_name, x_quant);
        else None.  The exporter emits SiLU as Sigmoid + Mul, so a Mul whose two
        operands are X and Sigmoid(X) is exactly that activation."""
        a, b = mul.input[0], mul.input[1]
        for sig_in, x_in in ((a, b), (b, a)):
            n = self.prod.get(self.act_source(sig_in))
            if n is not None and n.op_type == "Sigmoid":
                if self.act_source(n.input[0]) == self.act_source(x_in):
                    return self.act_source(x_in), self.in_quant(x_in)
        return None

    def is_focus(self, concat):
        """A Concat is the Focus stem iff every input is a Slice (of the image)."""
        for i in concat.input:
            n = self.prod.get(self.act_source(i))
            if n is None or n.op_type != "Slice":
                return False
        return True


def _classify(groups, cin, cout, kh, kw):
    if groups == 1 and kh == 1 and kw == 1:
        return "pointwise"
    if groups == cout and cin == cout:
        return "depthwise"
    return "stem"


def _maxpool(gr, node):
    k = _attr(node, "kernel_shape", [1, 1])
    s = _attr(node, "strides", [1, 1])
    p = _attr(node, "pads", [0, 0, 0, 0])              # ONNX: [t, l, b, r]
    return MaxPoolLayer(
        name=node.name, kh=k[0], kw=k[1], stride=tuple(s),
        pad=(p[0], p[2], p[1], p[3]),
        act_in=gr.in_quant(node.input[0]), act_out=gr.out_quant(node),
        inputs=(gr.act_source(node.input[0]),), output=node.output[0])


def _resize(gr, node):
    scale = 2
    for i in node.input[1:]:
        v = gr.const(i)
        if v is not None and v.size >= 4 and float(v.ravel()[-1]) > 1:
            scale = int(round(float(v.ravel()[-1])))
    return UpsampleLayer(
        name=node.name, scale=scale,
        act_in=gr.in_quant(node.input[0]), act_out=gr.out_quant(node),
        inputs=(gr.act_source(node.input[0]),), output=node.output[0])


def _terminal_is_classifier(gr):
    """The graph output comes straight off a Gemm/Conv (a classifier) rather than
    through a detector head's reshape/transpose tail."""
    name = gr.g.output[0].name
    seen = set()
    while name and name not in seen:
        seen.add(name)
        n = gr.prod.get(name)
        if n is None:
            return False
        if n.op_type in ("Gemm", "Conv"):
            return True
        if n.op_type in ("QuantizeLinear", "DequantizeLinear", "Reshape", "Flatten",
                         "Squeeze", "Transpose", "Identity"):
            name = n.input[0]
            continue
        return False
    return False


def _head_conv_outputs(gr):
    """Detector cut: the conv outputs feeding the head, found by walking back from
    the graph outputs and stopping at each Conv (so reg/obj/cls convs are the
    leaves).  Everything past them -- sigmoid, reshape, transpose, concat to
    [1,N,85] -- is done in the C decode tail, not compiled."""
    heads, seen = [], set()
    stack = [o.name for o in gr.g.output]
    while stack:
        t = stack.pop()
        if t in seen:
            continue
        seen.add(t)
        n = gr.prod.get(t)
        if n is None:
            continue
        if n.op_type == "Conv":
            if n.output[0] not in heads:
                heads.append(n.output[0])
            continue                        # do not descend past a conv
        for i in n.input:
            stack.append(i)
    return heads


def load(path: str) -> Model:
    gr = _Graph(path)
    model = Model(input_quant=Quant(1.0, 0))
    model.input_name = gr.g.input[0].name
    iq = gr.tensor_quant(model.input_name)
    if iq is not None:
        model.input_quant = iq
    dims = [d.dim_value for d in gr.g.input[0].type.tensor_type.shape.dim]
    if len(dims) == 4 and all(dims[1:]):
        model.input_shape = tuple(dims[1:])

    for node in gr.g.node:
        if node.op_type == "Conv":
            dqx = gr.dq(node.input[0])
            dqw = gr.dq(node.input[1])
            act_in = Quant(float(dqx[1]), int(dqx[2]))
            wq = gr.const(dqw[0]).astype(np.int8)
            w_scale = np.asarray(dqw[1], np.float32).ravel()
            bias_q = np.zeros(wq.shape[0], np.int64)
            if len(node.input) > 2:
                dqb = gr.dq(node.input[2])
                bias_q = gr.const(dqb[0]).astype(np.int64)
            groups = _attr(node, "group", 1)
            kshape = _attr(node, "kernel_shape", [wq.shape[2], wq.shape[3]])
            strides = _attr(node, "strides", [1, 1])
            pads = _attr(node, "pads", [0, 0, 0, 0])       # ONNX: [t, l, b, r]
            dil = _attr(node, "dilations", [1, 1])
            cout = wq.shape[0]
            # Depthwise has one input channel per group (wq.shape[1] == 1); a
            # 1-output conv also has groups == cout but is NOT depthwise, so the
            # per-group input count must be checked too.
            depthwise = groups == cout and wq.shape[1] == 1
            cin = groups if depthwise else wq.shape[1] * groups
            kind = _classify(groups, cin, cout, kshape[0], kshape[1])
            model.layers.append(ConvLayer(
                name=node.name, kind=kind, wq=wq, w_scale=w_scale, bias_q=bias_q,
                groups=groups, stride=tuple(strides),
                pad=(pads[0], pads[2], pads[1], pads[3]), dilation=tuple(dil),
                act_in=act_in, act_out=gr.out_quant(node),
                cin=cin, cout=cout, kh=kshape[0], kw=kshape[1],
                inputs=(gr.act_source(node.input[0]),), output=node.output[0]))

        elif node.op_type == "GlobalAveragePool":
            dqx = gr.dq(node.input[0])
            model.layers.append(PoolLayer(
                name=node.name, kind="global_avg",
                act_in=Quant(float(dqx[1]), int(dqx[2])),
                act_out=gr.out_quant(node),
                inputs=(gr.act_source(node.input[0]),), output=node.output[0]))

        elif node.op_type == "Gemm":
            dqx = gr.dq(node.input[0])
            dqw = gr.dq(node.input[1])
            wq = gr.const(dqw[0]).astype(np.int8)          # [Cout, Cin] (transB=1)
            w_scale = np.asarray(dqw[1], np.float32).ravel()
            dqb = gr.dq(node.input[2])
            bias_q = gr.const(dqb[0]).astype(np.int64)
            model.layers.append(FCLayer(
                name=node.name, wq=wq, w_scale=w_scale, bias_q=bias_q,
                act_in=Quant(float(dqx[1]), int(dqx[2])),
                act_out=gr.out_quant(node),
                inputs=(gr.act_source(node.input[0]),), output=node.output[0]))

        elif node.op_type == "Add":
            model.layers.append(AddLayer(
                name=node.name,
                act_in=(gr.in_quant(node.input[0]), gr.in_quant(node.input[1])),
                act_out=gr.out_quant(node),
                inputs=(gr.act_source(node.input[0]), gr.act_source(node.input[1])),
                output=node.output[0]))

        elif node.op_type == "MaxPool":
            model.layers.append(_maxpool(gr, node))

        elif node.op_type == "Resize":
            model.layers.append(_resize(gr, node))

        elif node.op_type == "Mul":
            silu = gr.silu_source(node)
            if silu is not None:
                model.layers.append(SiluLayer(
                    name=node.name, act_in=silu[1], act_out=gr.out_quant(node),
                    inputs=(silu[0],), output=node.output[0]))

        elif node.op_type == "Concat":
            if gr.is_focus(node):
                # Focus slices the raw image, so the slice-output quant IS the
                # image quant -- there is no QuantizeLinear on `images` itself.
                img_q = gr.in_quant(node.input[0])
                if img_q is not None:
                    model.input_quant = img_q
                model.layers.append(FocusLayer(
                    name=node.name, act_in=model.input_quant,
                    act_out=gr.out_quant(node),
                    inputs=(model.input_name,), output=node.output[0]))
            else:
                model.layers.append(ConcatLayer(
                    name=node.name, axis=_attr(node, "axis", 1),
                    act_in=tuple(gr.in_quant(i) for i in node.input),
                    act_out=gr.out_quant(node),
                    inputs=tuple(gr.act_source(i) for i in node.input),
                    output=node.output[0]))

    # Choose the graph outputs, then keep only their ancestors.  A classifier
    # ends at its Gemm; a detector is cut at the head convs (sigmoid + grid/stride
    # decode + NMS run in the C tail, not the compiled graph).  For a chain every
    # layer is an ancestor of the single output, so nothing is pruned.
    if _terminal_is_classifier(gr):
        outputs = (model.layers[-1].output,) if model.layers else ()
    else:
        outputs = tuple(_head_conv_outputs(gr))

    by_out = {L.output: L for L in model.layers}
    keep, stack = set(), list(outputs)
    while stack:
        o = stack.pop()
        if o in keep or o not in by_out:
            continue
        keep.add(o)
        stack.extend(by_out[o].inputs)
    model.layers = [L for L in model.layers if L.output in keep]
    model.output_names = tuple(outputs)
    return model
