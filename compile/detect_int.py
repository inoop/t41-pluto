"""Integer/byte-domain evaluation of the detection graph -- the exact arithmetic
the device executes.  Every activation is the unsigned byte b = q + 128; convs
feed u = (b-128) - z_in to the array's MAC (the feed-offset folds into the
requant bias on device) and emit the packed byte q_out + 128; the elementwise/
layout ops are the byte transforms in compile/ops/.  Checked against the
float-QDQ detect_ref boxes on the host, this proves the whole datapath before a
single line of device C.
"""
import numpy as np
from . import onnx_import as oi, nnmath, requant
from .ops import silu, add, concat, maxpool, upsample, focus


def _conv_bytes(b_in, L):
    u = np.asarray(b_in, np.int64) - 128 - L.act_in.zero_point          # = q - z_in
    acc = nnmath.conv_acc(u, L.wq, L.kh, L.kw, L.stride, L.pad,
                          depthwise=(L.kind == "depthwise"))
    B, M = requant.conv_table(L.w_scale, L.bias_q, L.act_in.scale,
                              L.act_out.scale, L.act_out.zero_point)
    # The array's packed requant emits q + 2^(n-1); re-centre on the byte domain
    # everything downstream speaks, b = q + 128.  n = 8 makes this the identity.
    nb = getattr(L, "out_bits", 8)
    p = requant.apply_packed(acc, B, M, out_bits=nb)
    return (p - (1 << (nb - 1)) + 128).astype(np.uint8)


def run_graph(model, image_chw):
    b0 = nnmath.to_byte(image_chw.astype(np.float64), model.input_quant.scale,
                        model.input_quant.zero_point)
    t = {model.input_name: b0}
    for L in model.layers:
        xi = [t[n] for n in L.inputs]
        if isinstance(L, oi.ConvLayer):     y = _conv_bytes(xi[0], L)
        elif isinstance(L, oi.SiluLayer):   y = silu.apply_bytes(L, xi[0])
        elif isinstance(L, oi.AddLayer):    y = add.apply_bytes(L, xi[0], xi[1])
        elif isinstance(L, oi.ConcatLayer): y = concat.apply_bytes(L, xi)
        elif isinstance(L, oi.MaxPoolLayer):y = maxpool.apply_bytes(L, xi[0])
        elif isinstance(L, oi.UpsampleLayer):y = upsample.apply_bytes(L, xi[0])
        elif isinstance(L, oi.FocusLayer):  y = focus.apply_bytes(L, xi[0])
        else: raise TypeError(type(L).__name__)
        t[L.output] = y
    return {n: t[n] for n in model.output_names}


def outputs_real(model, image_chw):
    """9 head outputs as real [C,H,W] (byte -> real via each output's quant)."""
    byname = {L.output: L for L in model.layers}
    outs = run_graph(model, image_chw)
    return {n: nnmath.to_real(outs[n], byname[n].act_out.scale,
                              byname[n].act_out.zero_point) for n in outs}


def run_all(model, image_chw):
    """Like run_graph but returns EVERY tensor's byte array, for per-layer debug."""
    import numpy as _np
    b0 = nnmath.to_byte(image_chw.astype(_np.float64), model.input_quant.scale,
                        model.input_quant.zero_point)
    t = {model.input_name: b0}
    for L in model.layers:
        xi = [t[n] for n in L.inputs]
        if isinstance(L, oi.ConvLayer):     y = _conv_bytes(xi[0], L)
        elif isinstance(L, oi.SiluLayer):   y = silu.apply_bytes(L, xi[0])
        elif isinstance(L, oi.AddLayer):    y = add.apply_bytes(L, xi[0], xi[1])
        elif isinstance(L, oi.ConcatLayer): y = concat.apply_bytes(L, xi)
        elif isinstance(L, oi.MaxPoolLayer):y = maxpool.apply_bytes(L, xi[0])
        elif isinstance(L, oi.UpsampleLayer):y = upsample.apply_bytes(L, xi[0])
        elif isinstance(L, oi.FocusLayer):  y = focus.apply_bytes(L, xi[0])
        else: raise TypeError(type(L).__name__)
        t[L.output] = y
    return t
