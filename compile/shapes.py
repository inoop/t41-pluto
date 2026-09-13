"""Per-tensor shape and quant propagation over the imported DAG.

model_build needs every tensor's [C,H,W] and (scale, zero_point) to lay out the
container's tensor table and size the runtime arena.  A chain carries one running
shape; a branchy graph must key it by tensor name.
"""
from . import onnx_import as oi


def _conv_hw(h, w, kh, kw, stride, pad):
    pt, pb, pl, pr = pad
    oh = (h + pt + pb - kh) // stride[0] + 1
    ow = (w + pl + pr - kw) // stride[1] + 1
    return oh, ow


def propagate(model, in_c=3, in_h=416, in_w=416):
    """Returns {name: (c, h, w)} and {name: Quant} for every tensor in the graph."""
    shp = {model.input_name: (in_c, in_h, in_w)}
    qnt = {model.input_name: model.input_quant}
    for L in model.layers:
        c, h, w = shp[L.inputs[0]]
        if isinstance(L, oi.ConvLayer):
            oh, ow = _conv_hw(h, w, L.kh, L.kw, L.stride, L.pad)
            out = (L.cout, oh, ow); q = L.act_out
        elif isinstance(L, oi.SiluLayer):
            out = (c, h, w); q = L.act_out
        elif isinstance(L, oi.AddLayer):
            out = (c, h, w); q = L.act_out
        elif isinstance(L, oi.ConcatLayer):
            out = (sum(shp[n][0] for n in L.inputs), h, w); q = L.act_out
        elif isinstance(L, oi.MaxPoolLayer):
            oh, ow = _conv_hw(h, w, L.kh, L.kw, L.stride, L.pad)
            out = (c, oh, ow); q = L.act_out
        elif isinstance(L, oi.UpsampleLayer):
            out = (c, h * L.scale, w * L.scale); q = L.act_out
        elif isinstance(L, oi.FocusLayer):
            out = (c * 4, h // 2, w // 2); q = L.act_out
        elif isinstance(L, oi.PoolLayer):
            out = (c, 1, 1); q = L.act_out
        elif isinstance(L, oi.FCLayer):
            out = (L.wq.shape[0], 1, 1); q = L.act_out
        else:
            raise TypeError(type(L).__name__)
        shp[L.output] = out
        qnt[L.output] = q
    return shp, qnt
