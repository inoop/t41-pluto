"""Float-QDQ reference for the detection graph: evaluate the imported IR in the
dequantized domain, then YOLOX decode + NMS.

This is NOT the bit-exact device golden (that is simulate.py's integer path, built
per-op during hardware bring-up).  It models the int8 QDQ semantics ORT uses --
every activation lives on its tensor's int8 grid, every op dequantizes, computes,
and re-quantizes -- so it reproduces the int8 ONNX the boxes were validated
against, across the branches and multi-input ops the chain simulator cannot walk.
Its job is to prove the imported graph (edges, cut, op set) is correct on the host.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from . import onnx_import as oi

STRIDES = (8, 16, 32)


def _fq(real, q):
    """Fake-quantize to a tensor's int8 grid: round onto the grid, read back."""
    qq = np.clip(np.rint(real / q.scale) + q.zero_point, -128, 127)
    return (qq - q.zero_point) * q.scale


def _conv(x, L):
    """Grouped/quantized conv in the real domain (dequantized weights + bias)."""
    wr = L.wq.astype(np.float64) * np.asarray(L.w_scale, np.float64)[:, None, None, None]
    br = L.bias_q.astype(np.float64) * (L.act_in.scale * np.asarray(L.w_scale, np.float64))
    Cin, H, W = x.shape
    g = L.groups
    xp = np.pad(x, ((0, 0), (L.pad[0], L.pad[1]), (L.pad[2], L.pad[3])))
    win = sliding_window_view(xp, (L.kh, L.kw), axis=(1, 2))[:, ::L.stride[0], ::L.stride[1]]
    # win: [Cin, OH, OW, kh, kw]
    OH, OW = win.shape[1], win.shape[2]
    cout = L.wq.shape[0]
    cin_g = Cin // g
    cout_g = cout // g
    out = np.empty((cout, OH, OW), np.float64)
    for gi in range(g):
        xs = win[gi * cin_g:(gi + 1) * cin_g]                       # [cin_g,OH,OW,kh,kw]
        ws = wr[gi * cout_g:(gi + 1) * cout_g]                       # [cout_g,cin_g,kh,kw]
        out[gi * cout_g:(gi + 1) * cout_g] = np.einsum("iyxhw,oihw->oyx", xs, ws)
    out += br[:, None, None]
    return _fq(out, L.act_out)


def _focus(x, L):
    # space-to-depth: [C,H,W] -> [4C, H/2, W/2], order matching YOLOX Focus:
    # cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]])
    tl = x[:, 0::2, 0::2]; bl = x[:, 1::2, 0::2]
    tr = x[:, 0::2, 1::2]; br = x[:, 1::2, 1::2]
    return _fq(np.concatenate([tl, bl, tr, br], 0), L.act_out)


def _maxpool(x, L):
    xp = np.pad(x, ((0, 0), (L.pad[0], L.pad[1]), (L.pad[2], L.pad[3])),
                constant_values=-np.inf)
    win = sliding_window_view(xp, (L.kh, L.kw), axis=(1, 2))[:, ::L.stride[0], ::L.stride[1]]
    return _fq(win.max(axis=(3, 4)), L.act_out)


def _upsample(x, L):
    return _fq(np.repeat(np.repeat(x, L.scale, 1), L.scale, 2), L.act_out)


def run_graph(model, image_chw):
    """image_chw: float [3,H,W] (YOLOX preproc). Returns {output_name: real [C,H,W]}."""
    t = {model.input_name: _fq(image_chw.astype(np.float64), model.input_quant)}
    for L in model.layers:
        xin = [t[n] for n in L.inputs]
        if isinstance(L, oi.ConvLayer):
            y = _conv(xin[0], L)
        elif isinstance(L, oi.SiluLayer):
            x = xin[0]; y = _fq(x * (1.0 / (1.0 + np.exp(-x))), L.act_out)
        elif isinstance(L, oi.AddLayer):
            y = _fq(xin[0] + xin[1], L.act_out)
        elif isinstance(L, oi.ConcatLayer):
            y = _fq(np.concatenate(xin, axis=L.axis - 1), L.act_out)   # our arrays are [C,H,W]
        elif isinstance(L, oi.MaxPoolLayer):
            y = _maxpool(xin[0], L)
        elif isinstance(L, oi.UpsampleLayer):
            y = _upsample(xin[0], L)
        elif isinstance(L, oi.FocusLayer):
            y = _focus(xin[0], L)
        else:
            raise TypeError(f"no ref for {type(L).__name__}")
        t[L.output] = y
    return {n: t[n] for n in model.output_names}


def decode(outs_by_name, model, conf=0.30):
    """9 head conv outputs -> (boxes xyxy, scores, classes) in letterboxed pixels."""
    # group outputs by scale via name suffix .0/.1/.2, assemble [reg,obj,cls]
    def pick(kind, s):
        for n in model.output_names:
            if f"{kind}_preds.{s}/" in n:
                return outs_by_name[n]
        raise KeyError(kind, s)
    allb, alls, allc = [], [], []
    for s, st in enumerate(STRIDES):
        reg = pick("reg", s); obj = pick("obj", s); cls = pick("cls", s)
        C, H, W = cls.shape
        yv, xv = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        grid = np.stack([xv, yv], 0).reshape(2, -1).astype(np.float64)
        reg = reg.reshape(4, -1); objp = obj.reshape(1, -1); clsp = cls.reshape(C, -1)
        cx = (reg[0] + grid[0]) * st; cy = (reg[1] + grid[1]) * st
        w = np.exp(reg[2]) * st; h = np.exp(reg[3]) * st
        objs = 1 / (1 + np.exp(-objp[0])); clss = 1 / (1 + np.exp(-clsp))
        sc = objs[None, :] * clss                                   # [C, HW]
        b = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)  # [HW,4]
        allb.append(b); alls.append(sc.T); allc.append(np.arange(C))
    boxes = np.concatenate(allb, 0)
    scores = np.concatenate(alls, 0)                                # [N, C]
    return boxes, scores


def nms(boxes, s, thr=0.45):
    x1, y1, x2, y2 = boxes.T; areas = (x2 - x1) * (y2 - y1)
    order = s.argsort()[::-1]; keep = []
    while order.size:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1); h = np.maximum(0, yy2 - yy1)
        iou = w * h / (areas[i] + areas[order[1:]] - w * h + 1e-9)
        order = order[1:][iou <= thr]
    return keep
