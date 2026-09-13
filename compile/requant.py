"""Requantisation, in both directions the pipeline needs it.

CONVOLUTION (`conv_table`, `apply_packed`).  The array's packed requant is fixed
hardware:

    p = clamp(floor((4*acc + B) / M) + 2^(n-1), 0, 2^n - 1)

with `M` a DIVISOR, so a layer's real scale `S = x_scale*w_scale[co]/y_scale`
is expressed by inverting it: `M = round(4/S)`, `B = round(M*(S*bias_q + y_zp + 0.5))`.
The 4x numerator is where the array's slope resolution comes from, and is why
activations are fed zero-point-removed (docs/DESIGN.md).

ACTIVATION (`act_table`, `act_descriptor`).  A layer that only changes scale --
concat's inputs, focus, the FPN upsamples, the SPP pools -- applies

    out = clamp(rint(s_in * (b - 128 - zp_in) / s_out) + zp_out, -128, 127) + 128

which is AFFINE in b.  That matters because MXU3 has no data-dependent gather, so
a 256-entry lookup is one scalar load per byte and cannot be vectorised at all --
while the same function in fixed point,

    out = clamp(((C * (b - off) + 2^(K-1)) >> K) + zp_out, -128, 127) + 128

is int32 arithmetic, 64 bytes to a pass.  So the table is emitted alongside a
descriptor naming which form applies, and the COMPILER decides rather than
leaving the runtime to discover it:

    IDENTITY  the table is a no-op -- a scale-preserving concat input, the FPN
              upsamples, the SPP pools.  The layer is pure data movement.
    AFFINE    the fixed-point form reproduces the table for ALL 256 inputs
              (checked here, not assumed) and stays inside int32.
    TABLE     it does not, and the scalar lookup is the only correct path.

`K` is the smallest shift that works, which keeps `C` small and the
intermediates far from overflowing.
"""
import numpy as np

from . import nnmath as nm


def conv_table(w_scale, bias_q, in_scale, out_scale, out_zp):
    """Per-output-channel (B, M) for the array's packed requant.

    `M` is a divisor, so the real scale is inverted into it; the +0.5 in `B`
    turns the hardware's floor into round-to-nearest.
    """
    S = np.asarray(in_scale * np.asarray(w_scale, np.float64) / out_scale, np.float64)
    M = np.rint(4.0 / S).astype(np.int64)
    B = np.rint(M * (S * np.asarray(bias_q, np.float64) + out_zp + 0.5)).astype(np.int64)
    return B, M


def apply_packed(acc, B, M, out_bits=8):
    """The array's requant, exactly: floor division, then clamp to the byte grid.

    `acc` is [Cout, ...]; B and M broadcast over its leading axis.  The division
    must FLOOR (not truncate toward zero) -- the accumulator goes negative.
    """
    acc = np.asarray(acc, np.int64)
    shape = (-1,) + (1,) * (acc.ndim - 1)
    num = 4 * acc + np.asarray(B, np.int64).reshape(shape)
    p = num // np.asarray(M, np.int64).reshape(shape)          # floor
    return np.clip(p + (1 << (out_bits - 1)), 0, (1 << out_bits) - 1)


KIND_TABLE, KIND_IDENTITY, KIND_AFFINE = 0, 1, 2

DESC_WORDS = 6          # kind, C, K, off, zo, reserved
DESC_BYTES = DESC_WORDS * 4


def act_table(q_in, q_out):
    """The 256-entry byte table, b = q + 128 in and out."""
    b = np.arange(256)
    return nm.to_byte(nm.to_real(b, q_in.scale, q_in.zero_point),
                      q_out.scale, q_out.zero_point, getattr(q_out, "bits", 8))


def _affine(tbl, q_in, q_out):
    """Smallest (C, K) whose fixed-point form reproduces `tbl` exactly, or None.

    The runtime folds `off` and the rounding term into one accumulator seed, so
    the bound checked here is the one it actually evaluates:
    |C*b| + |C*off| + 2^(K-1) must stay inside int32.
    """
    off = 128 + q_in.zero_point
    zo = q_out.zero_point
    r = q_in.scale / q_out.scale
    b = np.arange(256, dtype=np.int64)
    for K in range(1, 31):
        base = int(round(r * (1 << K)))
        for C in (base, base - 1, base + 1):
            if C == 0:
                continue
            if 255 * abs(C) + abs(C) * abs(off) + (1 << (K - 1)) > 2**31 - 1:
                continue
            q = ((C * (b - off) + (1 << (K - 1))) >> K) + zo
            got = (np.clip(q, -128, 127) + 128).astype(np.uint8)
            if np.array_equal(got, tbl):
                return C, K, off, zo
    return None


def act_descriptor(q_in, q_out):
    """The 24-byte descriptor for this (input, output) quantisation pair."""
    tbl = act_table(q_in, q_out)
    if np.array_equal(tbl, np.arange(256, dtype=np.uint8)):
        return np.array([KIND_IDENTITY, 0, 0, 0, 0, 0], "<i4").tobytes()
    fit = _affine(tbl, q_in, q_out)
    if fit is None:
        return np.array([KIND_TABLE, 0, 0, 0, 0, 0], "<i4").tobytes()
    C, K, off, zo = fit
    return np.array([KIND_AFFINE, C, K, off, zo, 0], "<i4").tobytes()


def act_blob(pairs):
    """Tables for every (q_in, q_out) pair, then their descriptors.

    Descriptors go AFTER all the tables so a kernel that only knows about
    tables keeps indexing them as `table + i*256`, unchanged.
    """
    tbls = b"".join(act_table(qi, qo).tobytes() for qi, qo in pairs)
    descs = b"".join(act_descriptor(qi, qo) for qi, qo in pairs)
    return tbls + descs
