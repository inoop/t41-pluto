"""The integer arithmetic the hardware does, in NumPy.

Shared by the op modules in compile/ops/.  Keeping it here rather than inside
any one of them means the simulator and the device see the same rounding.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def quantize(x, scale, zp, lo=-128, hi=127):
    """Real -> quantized, round-half-away-from-zero then clamp."""
    return np.clip(np.rint(x / scale) + zp, lo, hi).astype(np.int32)


# Activations flow between ops as the unsigned byte b = q + 128, where q is the
# tensor's ORT int8 value.  These convert to/from the real domain; see the
# pluto-yolox-nano-detection design notes.

def to_real(byte, scale, zp):
    """byte (q + 128) -> real value scale * (q - zp)."""
    return scale * ((np.asarray(byte, np.int64) - 128) - zp)


def to_byte(real, scale, zp, bits=8):
    """real -> the byte b = q + 128, clamped to the `bits`-wide grid.

    The transport is always the same byte -- everything downstream reads
    b = q + 128 -- but a 4-bit tensor may only take the 16 values q in [-8, 7].
    Clamping here is what makes a 4-bit activation arm actually 4-bit: without
    it the convolutions would round to 16 levels and every other layer would
    quietly put the tensor back on the 256-level grid.
    """
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    q = np.clip(np.rint(real / scale) + zp, lo, hi)
    return (q + 128).astype(np.uint8)


def windows(x, kh, kw, stride, pad, pad_val):
    """[C,H,W] -> sliding windows [C,OH,OW,KH,KW], border filled with pad_val."""
    pt, pb, pl, pr = pad
    xp = np.pad(x, ((0, 0), (pt, pb), (pl, pr)), constant_values=pad_val)
    win = sliding_window_view(xp, (kh, kw), axis=(1, 2))
    return win[:, ::stride[0], ::stride[1], :, :]


def conv_acc(u, wq, kh, kw, stride, pad, depthwise=False):
    """Raw int accumulator, on the zero-point-removed feed u.

    The array is fed u = x_q - x_zp, so the border pads with 0 -- which is the
    real zero -- and there is no zero-point correction term in the MAC.

    The dense case goes through float64 BLAS rather than an int64 einsum, which
    is EXACT here and about fifty times faster -- the difference between 33 s and
    under a second for one 640x640 YOLOX-S image, which is what makes a sweep
    over a validation set possible at all.

    Why it is exact: every product is an integer, and the largest accumulator
    this hardware can reach is KH*KW*Cin * max|u| * max|w| = 2304 * 255 * 127 =
    7.5e7 for a 3x3 over 256 channels.  float64 represents every integer below
    2^53 = 9.0e15 exactly, so neither a product nor any partial sum can round.
    (float32 would NOT be safe -- its 2^24 is below the accumulator range.)
    """
    win = windows(u, kh, kw, stride, pad, 0)
    if depthwise:
        return np.einsum("cyxhw,chw->cyx", win.astype(np.int64),
                         wq[:, 0].astype(np.int64))
    c, oh, ow, _, _ = win.shape
    a = np.ascontiguousarray(win.transpose(1, 2, 0, 3, 4), np.float64) \
          .reshape(oh * ow, c * kh * kw)
    w = np.asarray(wq, np.float64).reshape(wq.shape[0], -1)
    return (a @ w.T).T.reshape(-1, oh, ow).astype(np.int64)
