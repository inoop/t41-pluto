"""Re-quantize the IR to 4-bit weights and/or 4-bit activations.

Pure IR -> IR, applied between onnx_import.load() and model_build.build().
Nothing else in the compiler needs to know: the packers, the simulator and
detect_int all read `wq`, `w_scale`, `bias_q`, `act_in` and `act_out`, so
rewriting those fields is the whole change.

WHY 4-BIT AT ALL.  The frame is bound by the CPU pushing operands into the
array -- 110 MB per YOLOX-S frame at ~733 MB/s, against a 12 ms floor if the
pushes were free.  Four-bit cuts that twice over: 4-bit WEIGHTS halve the
tap-slot cost so `gpp` doubles and multi-pass layers stop re-feeding the whole
feature map, and 4-bit ACTIVATIONS halve the bytes per window row outright
(`cpr = s * in_bits / 4` in runtime/exec/plt_conv_k3.c).  110 MB -> ~45 MB.

WHAT THE HARDWARE ALLOWS, which is what makes this a real question:

  * ONE weight scale per output channel.  The array's requant is M = round(4/S)
    with S = x_scale * w_scale[co] / y_scale (requant.conv_table), one M per
    output channel, so a scale per 32 or 64 weights -- the thing that makes
    int4 work elsewhere -- has nowhere to live.  Per-channel symmetric is the
    ceiling here, and it is the pessimistic case.
  * SIXTEEN activation levels.  requant.apply_packed is
    clamp(floor((4*acc + B)/M) + 2^(n-1), 0, 2^n - 1), so at n = 4 the stored
    value q lands in [-8, 7].

Neither transform is an approximation of the hardware: both produce an IR the
existing exact-integer simulator runs as-is.  What they approximate is the
MODEL, and tests/eval_yolox.py measures by how much.
"""
import numpy as np

from . import onnx_import as oi

# Signed 4-bit, the grid the array's packed requant produces.
INT4_LO, INT4_HI = -8, 7


def _conv_like(L):
    return isinstance(L, (oi.ConvLayer, oi.FCLayer))


# --- weights ---------------------------------------------------------------

def _mse_scale(w_flat, denom, n_cand=20):
    """Per row, the clipping point that minimises squared quantization error.

    Absmax is the naive choice and it is a poor one at 4 bits: a single outlier
    weight stretches the grid so the other 99% land on two or three levels.
    Sweeping the clip down and keeping the best is the weight-side analogue of
    the percentile calibration the activations get, and it costs one pass.
    """
    amax = np.abs(w_flat).max(axis=1)
    best_s = amax / denom
    best_e = np.full(w_flat.shape[0], np.inf)
    for f in np.linspace(1.0, 0.3, n_cand):
        s = np.maximum(amax * f / denom, 1e-30)[:, None]
        q = np.clip(np.rint(w_flat / s), INT4_LO, INT4_HI)
        e = ((q * s - w_flat) ** 2).sum(axis=1)
        take = e < best_e
        best_e = np.where(take, e, best_e)
        best_s = np.where(take, s[:, 0], best_s)
    return best_s


def to_int4_weights(model, denom=7.0, mode="absmax"):
    """Re-quantize every convolution's and FC's weights to signed 4-bit.

    Per output channel o, the real weight is wq[o] * w_scale[o]; the int8 grid
    has 256 levels, fine enough that re-quantizing from it rather than from the
    original floats costs nothing next to 4-bit itself.

        new_scale[o] = max|w_real[o]| / denom
        new_wq[o]    = clip(rint(w_real[o] / new_scale[o]), -8, 7)

    `denom` = 7 keeps the grid symmetric (15 usable levels); `denom` = 8 uses all
    16 at the cost of asymmetry.  `mode` = "mse" replaces max|w| with a searched
    clipping point (see _mse_scale).  Which combination wins is an empirical
    question, and tests/eval_yolox.py answers it.

    THE BIAS MOVES TOO.  bias_q's implied scale is act_in.scale * w_scale[o], so
    changing w_scale silently corrupts every bias unless it is carried through
    the real domain.  That is the one step here that is easy to get wrong and
    impossible to notice by inspection, so tests/quant4_test.py checks it.
    """
    for L in model.layers:
        if not _conv_like(L):
            continue
        wq = np.asarray(L.wq, np.float64)
        ws = np.asarray(L.w_scale, np.float64)
        co = wq.shape[0]
        w_real = wq * ws.reshape((co,) + (1,) * (wq.ndim - 1))

        flat = w_real.reshape(co, -1)
        if mode == "mse":
            new_ws = _mse_scale(flat, float(denom))
        else:
            new_ws = np.abs(flat).max(axis=1) / float(denom)
        # An all-zero output channel has no scale; any positive value will do,
        # and keeping the old one keeps the requant table finite.
        new_ws = np.where(new_ws > 0, new_ws, ws)

        new_wq = np.clip(np.rint(w_real / new_ws.reshape((co,) + (1,) * (wq.ndim - 1))),
                         INT4_LO, INT4_HI)

        bias_real = np.asarray(L.bias_q, np.float64) * L.act_in.scale * ws
        new_bias = np.rint(bias_real / (L.act_in.scale * new_ws))

        L.wq = new_wq.astype(np.int8)
        L.w_scale = new_ws.astype(np.float32)
        L.bias_q = new_bias.astype(np.int64)
    return model


# --- activations -----------------------------------------------------------

def _quant_for(bounds, bits=4):
    """An ASYMMETRIC Quant covering [lo, hi] on a `bits`-wide grid.

    q runs over [-2^(b-1), 2^(b-1)-1] and real = scale * (q - zp), so

        scale = (hi - lo) / (2^b - 1)          zp = -2^(b-1) - lo/scale

    which is the same convention ORT's int8 quantization uses -- a ReLU tensor
    with lo = 0 comes out at zp = -128, and that is exactly what the int8 models
    in fixtures/ carry.

    Symmetric (zp = 0) was the first thing tried and it is badly wrong here: the
    head's logits live in roughly [-30, +3.6], so a grid centred on zero clamps
    the whole bulk onto one level and every head output goes constant.
    """
    lo, hi = float(bounds[0]), float(bounds[1])
    n = (1 << bits) - 1
    scale = (hi - lo) / n
    if not scale > 0:
        return oi.Quant(scale=1e-8, zero_point=0, bits=bits)
    zp = int(round(-(1 << (bits - 1)) - lo / scale))
    return oi.Quant(scale=scale, zero_point=zp, bits=bits)


def keep_8bit(model):
    """Tensors that stay 8-bit even in a 4-bit activation model.

    First and last layers at higher precision is standard practice, and here it
    is not a nicety: the stem reads the letterboxed IMAGE, whose range is the
    full 0..255, so a 16-level grid quantizes it in steps of 17 and the network
    is wrecked before the first convolution finishes -- measured, the relative
    error at layer 1 is already 10x.  The vendor's own 4-bit model does the same
    thing, with a dedicated 8-bit first layer (yolox-t41/src/nna/conv_first.c,
    "first_layer_i8").

    The nine head predictions are kept for the opposite reason: they are logits,
    read by a sigmoid, so a coarse grid there moves every score directly.  They
    are also the smallest tensors in the graph, so keeping them costs almost no
    bandwidth.
    """
    keep = {model.input_name} | set(model.output_names)
    for L in model.layers:
        if isinstance(L, oi.ConvLayer):
            keep.update(L.inputs)          # whatever the stem reads
            break
    return keep


def to_int4_activations(model, ranges, keep=None):
    """Re-quantize every intermediate activation to a 16-level grid.

    `ranges` maps tensor name -> [lo, hi] clipping bounds, from
    compile/calibrate.py.  A tensor with no entry, or one named in `keep`, holds
    its 8-bit quantization -- see keep_8bit() for which and why.

    A consumer's act_in must equal its producer's act_out (see
    shapes.propagate), so both sides are rewritten from the same table rather
    than recomputed independently.
    """
    keep = keep_8bit(model) if keep is None else keep
    qout = {name: _quant_for(r) for name, r in ranges.items() if name not in keep}

    for L in model.layers:
        if L.output in qout:
            L.act_out = qout[L.output]
        # Add and Concat carry a TUPLE of act_in, one per input, on independent
        # scales; every other layer carries a single Quant.  Overwriting the
        # tuple with one Quant is silent until the first residual block runs.
        if isinstance(L.act_in, (tuple, list)):
            L.act_in = tuple(qout.get(n, q) for n, q in zip(L.inputs, L.act_in))
        elif L.inputs and L.inputs[0] in qout:
            L.act_in = qout[L.inputs[0]]
    return model


# --- the entry point the CLI and the harness share -------------------------

def apply(model, w_bits=8, a_bits=8, denom=7.0, ranges=None, mode="absmax"):
    """Re-quantize `model` in place to the requested widths.

    8/8 is the IDENTITY and must stay so: it is what every existing gate
    compiles, and the two pinned .pluto md5s depend on it.
    """
    if w_bits not in (4, 8) or a_bits not in (4, 8):
        raise ValueError(f"only 8 and 4 bits are implemented, got w={w_bits} a={a_bits}")
    if w_bits == 4:
        to_int4_weights(model, denom=denom, mode=mode)
    if a_bits == 4:
        if not ranges:
            raise ValueError("4-bit activations need calibrated ranges "
                             "(python3 -m compile.calibrate)")
        to_int4_activations(model, ranges)

    if w_bits != 8 or a_bits != 8:
        # Stamp the widths on the layers so the container says what each layer
        # really is; _conv.lower reads them.  Only the graph input stays 8-bit,
        # so a layer that reads it keeps in_bits = 8.
        keep = keep_8bit(model) if a_bits == 4 else set()
        for L in model.layers:
            L.w_bits = w_bits
            L.out_bits = 8 if L.output in keep else a_bits
            L.in_bits = 8 if (L.inputs and L.inputs[0] in keep) else a_bits
    return model
