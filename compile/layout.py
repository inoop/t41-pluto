"""Tensor/weight layout transforms for the NNA.

Phase 1 implements and unit-tests the NDHWC32 feature packer (the format the
array reads). The weight super-tile and block-diagonal-depthwise packers are
added with the executors that consume them (Phases 3/5/6), where they can be
validated on the device.
"""
import numpy as np


def pack_ndhwc32(x_chw: np.ndarray) -> np.ndarray:
    """[C,H,W] int8 -> NDHWC32 [D,H,W,32] int8, D=ceil(C/32), channels zero-padded."""
    C, H, W = x_chw.shape
    D = (C + 31) // 32
    out = np.zeros((D, H, W, 32), dtype=np.int8)
    for d in range(D):
        c0 = d * 32
        c1 = min(c0 + 32, C)
        out[d, :, :, :c1 - c0] = np.transpose(x_chw[c0:c1], (1, 2, 0))
    return out


def unpack_ndhwc32(x_ndhwc32: np.ndarray, C: int) -> np.ndarray:
    """NDHWC32 [D,H,W,32] -> [C,H,W], dropping channel padding."""
    D, H, W, _ = x_ndhwc32.shape
    full = np.transpose(x_ndhwc32, (0, 3, 1, 2)).reshape(D * 32, H, W)
    return full[:C]


# --- NNA 8-bit pointwise weight packing (device-verified, session 2) ---
# The 1x1 int8 weight tile for one (input-group, output-group) 32x32 block is
# 1024 bytes = 4 slices of 256 B. The slices are four 2-bit weight bit-planes
# scaled 1,4,16,64; each byte packs 4 consecutive input channels in its four
# 2-bit fields. Position of the byte carrying weight[co][ci] (co,ci in 0..31):
#     slice s in 0..3, byte = port*64 + (co&7)*8 + (ci>>2), port = co>>3,
#     field = ci&3, plane bits = ((wq+128) >> (2*s)) & 3.
# The array reads the assembled unsigned byte as signed (offset 128), so it
# accumulates sum(u * wq). Verified byte-exact on the T41 (RESULT pw_acc match=1).

def pad32(c):
    """Round a channel count up to a whole number of 32-lane groups."""
    return (c + 31) // 32 * 32


def pack_pointwise_int8(wq_oihw: np.ndarray) -> bytes:
    """int8 weights [Cout, Cin, 1, 1] -> NNA 1x1 blob, output-group-major.

    Cin/Cout are padded up to a multiple of 32 with zero weights: a padded input
    channel is multiplied by the matching zero-padded activation lane and a padded
    output channel is simply not read by the next layer, so the padding is inert.
    Returns ceil(Cin/32)*ceil(Cout/32)*1024 bytes, tiles ordered (out_group, in_group)."""
    w = np.asarray(wq_oihw, np.int16).reshape(wq_oihw.shape[0], wq_oihw.shape[1])
    Cout, Cin = w.shape
    Coutp, Cinp = pad32(Cout), pad32(Cin)
    if (Coutp, Cinp) != (Cout, Cin):
        wp = np.zeros((Coutp, Cinp), np.int16)
        wp[:Cout, :Cin] = w
        w, Cout, Cin = wp, Coutp, Cinp
    D, Dout = Cin // 32, Cout // 32
    blob = bytearray(D * Dout * 1024)
    for go in range(Dout):
        for gi in range(D):
            tile = bytearray(1024)
            for col in range(32):          # output channel within group
                for cil in range(32):      # input channel within group
                    wass = int(w[go * 32 + col, gi * 32 + cil]) + 128   # 0..255
                    port, whi, wlo, f = col >> 3, col & 7, cil >> 2, cil & 3
                    for s in range(4):
                        pl = (wass >> (2 * s)) & 3
                        tile[s * 256 + port * 64 + whi * 8 + wlo] |= pl << (2 * f)
            base = (go * D + gi) * 1024
            blob[base:base + 1024] = tile
    return bytes(blob)


def requant_table_bytes(B: np.ndarray, M: np.ndarray) -> bytes:
    """Per output group: int32 bias[32] at +0x00, int32 mult[32] at +0x80 (256 B/group).

    Cout is padded to a multiple of 32; padded channels get bias 0 / mult 1 (an
    inert group the next layer does not read)."""
    Cout = B.shape[0]
    Coutp = pad32(Cout)
    if Coutp != Cout:
        Bp = np.zeros(Coutp, B.dtype); Bp[:Cout] = B; B = Bp
        Mp = np.ones(Coutp, M.dtype);  Mp[:Cout] = M; M = Mp
    out = bytearray()
    for g in range(Coutp // 32):
        out += B[g * 32:(g + 1) * 32].astype("<i4").tobytes()
        out += M[g * 32:(g + 1) * 32].astype("<i4").tobytes()
    return bytes(out)


def _pack_tile_bitplanes(tile_w, out):
    """Pack a 32x32 int16 weight tile into 1024 B of 4 x 2-bit planes (device layout)."""
    for col in range(32):              # output channel within the group
        for ci in range(32):           # input channel within the group
            wass = int(tile_w[col, ci]) + 128
            port, whi, wlo, f = col >> 3, col & 7, ci >> 2, ci & 3
            for s in range(4):
                out[s * 256 + port * 64 + whi * 8 + wlo] |= ((wass >> (2 * s)) & 3) << (2 * f)


def depthwise_tap_order(kh, kw):
    """Tap order of the depthwise blob: plain row-major, `t = ky*kw + kx`.

    The vendor's 3x3 walk takes the middle kernel row in reverse column order,
    but pluto assembles the taps itself (runtime/exec/plt_conv_dw.c) and so is
    free to pick the order.  Row-major keeps the packer, the executor and the
    simulator saying the same thing.
    """
    return [r * kw + c for r in range(kh) for c in range(kw)]


def pack_depthwise_int8(wq_c1hw: np.ndarray) -> bytes:
    """Depthwise int8 weights [C,1,KH,KW] -> NNA blob of BLOCK-DIAGONAL 32x32 tiles.

    Depthwise does not reduce across channels, so each (channel-group, tap) tile
    is diagonal: `W[co][ci] = w[g*32+co][tap]` when `co == ci`, else 0.

    Read as a 1x1 convolution this is one output group of 32 channels over
    `KH*KW` input groups of 32 -- 288 pseudo-channels for a 3x3 kernel, which is
    exactly what the pointwise recipe consumes, with the (out-group, in-group)
    tile order it expects.  So a depthwise layer is `C/32` independent 1x1
    convolutions over an assembled tap tensor, and needs no separate 3x3 engine
    (runtime/exec/plt_conv_dw.c).

    Blob order: (group, tap), 1024 B per tile, taps in `depthwise_tap_order`.
    """
    C, one, KH, KW = wq_c1hw.shape
    assert one == 1
    Cp = pad32(C)
    if Cp != C:
        w = np.zeros((Cp, 1, KH, KW), wq_c1hw.dtype); w[:C] = wq_c1hw; wq_c1hw = w; C = Cp
    G = C // 32
    taps = depthwise_tap_order(KH, KW)
    blob = bytearray(G * len(taps) * 1024)
    flat = wq_c1hw.reshape(C, KH * KW).astype(np.int16)
    for g in range(G):
        for ti, tap in enumerate(taps):
            tile = np.zeros((32, 32), np.int16)
            for i in range(32):
                tile[i, i] = flat[g * 32 + i, tap]       # diagonal
            base = (g * len(taps) + ti) * 1024
            view = bytearray(1024)
            _pack_tile_bitplanes(tile, view)
            blob[base:base + 1024] = view
    return bytes(blob)


# --- a general (grouped-by-1) convolution as a 1x1 over assembled taps -------
# The array has no small-channel kernel mode: an KH x KW window over Cin channels
# is Cin*KH*KW values, which pluto assembles into pseudo-channels and runs as an
# ordinary 1x1 convolution (manual 8.6) -- the device-verified recipe.  pluto
# picks the natural tap order:
#
#     tap t = (ky*KW + kx)*Cin + c
#
# The RGB stem (Cin=3, 3x3 -> 27 taps) is the special case that fits in a single
# 32-lane group; a post-Focus conv (Cin=12, 3x3 -> 108 taps) spans four.  Input
# groups are padded to at least two, which the 1x1 recipe requires.  Both the
# im2col and pack_stem_int8 use this, and nothing else needs to know it.

def stem_tap_stride(cin, kh, kw):
    """Pseudo-channels per tap.

    Packed: one pseudo-channel per input channel, no padding between taps.

    Packed (`cin`) for a window that already fits one 32-lane group, which is
    what MobileNetV1's 3x3x3 = 27 taps wants.

    A window spanning several groups instead pads each tap to a 16-BYTE-ALIGNED
    slot.  The runtime assembles such a window one tap at a time, and with the
    slot aligned the whole tap moves as a single aligned vector load/store that
    cannot straddle a cell boundary; packed taps land at lanes 0,12,24,4,...,
    where a 16-byte store would spill into the NEXT PIXEL's cell.  The lanes the
    padding wastes carry zero weights (stem_weight_matrix leaves them 0), so they
    cannot change the result -- they cost one extra input group on the array,
    which the faster gather more than repays.
    """
    return cin if cin * kh * kw <= 32 else (cin + 15) // 16 * 16


def stem_tap(ky, kx, c, cin, kw, kh=None):
    """Which pseudo-channel carries kernel tap (ky,kx) of input channel c."""
    stride = stem_tap_stride(cin, kh if kh is not None else kw, kw)
    return (ky * kw + kx) * stride + c


def stem_pseudo_channels(cin, kh, kw):
    """Number of pseudo-channels: KH*KW taps rounded up to >= two 32-lane groups."""
    return max(pad32(kh * kw * stem_tap_stride(cin, kh, kw)), 64)


def stem_im2col(u_chw, kh, kw, stride, pad):
    """[Cin,H,W] input -> [T,OH,OW] tap planes, T = stem_pseudo_channels.

    Out-of-range taps are written 0, which in the fed-byte domain the requant's
    feed offset already accounts for, so ordinary zero padding falls out.
    """
    C, H, W = u_chw.shape
    pt, pb, pl, pr = pad
    sh, sw = stride
    oh = (H + pt + pb - kh) // sh + 1
    ow = (W + pl + pr - kw) // sw + 1

    out = np.zeros((stem_pseudo_channels(C, kh, kw), oh, ow), u_chw.dtype)
    for ky in range(kh):
        for kx in range(kw):
            ys = np.arange(oh) * sh + ky - pt
            xs = np.arange(ow) * sw + kx - pl
            ok_y, ok_x = (ys >= 0) & (ys < H), (xs >= 0) & (xs < W)
            for c in range(C):
                plane = np.zeros((oh, ow), u_chw.dtype)
                plane[np.ix_(ok_y, ok_x)] = u_chw[np.ix_([c], ys[ok_y], xs[ok_x])][0]
                out[stem_tap(ky, kx, c, C, kw, kh)] = plane
    return out


def stem_weight_matrix(wq_oihw):
    """int8 [Cout,Cin,KH,KW] -> the [Cout,T] 1x1 weight matrix of the tap order."""
    Cout, C, KH, KW = wq_oihw.shape
    w = np.zeros((Cout, stem_pseudo_channels(C, KH, KW)), np.int16)
    for ky in range(KH):
        for kx in range(KW):
            for c in range(C):
                w[:, stem_tap(ky, kx, c, C, KW, KH)] = wq_oihw[:, c, ky, kx]
    return w


def pack_stem_int8(wq_oihw) -> bytes:
    """The conv's weights in the 1x1 blob layout over the assembled tap channels."""
    w = stem_weight_matrix(wq_oihw)
    return pack_pointwise_int8(w.reshape(w.shape[0], w.shape[1], 1, 1))


# --- a DENSE window, Cin a multiple of 32 -----------------------------------
#
# Same idea as the stem -- a KH x KW window run as a 1x1 over pseudo-channels --
# but ordered for a window that is never assembled in memory.  The runtime feeds
# the KH*KW taps as SHIFTED VIEWS of one bordered plane (plt_conv_dense.c, the
# trick plt_conv_dw.c already uses), so the layout is chosen to keep consecutive
# feeds close together:
#
#     pseudo-channel = ((c // 32)*KH*KW + ky*KW + kx)*32 + (c % 32)
#
# i.e. GROUP-MAJOR: all KH*KW taps of input group 0, then those of group 1.  The
# nine taps of a group read overlapping cache lines of the same plane, so those
# feeds are cheap; a jump to the next group's plane is the expensive one, and
# this order pays it once per group instead of once per tap.  Measured on nano:
# 0.073 us for a shifted-view feed against 0.377 us for a cross-plane one.
#
# The stem's 16-byte tap slot has no counterpart here: nothing is gathered into
# a register, so a tap is a whole aligned 32-lane group by construction.

def dense_tap(ky, kx, c, kh, kw):
    """Which pseudo-channel carries kernel tap (ky,kx) of input channel c."""
    return ((c // 32) * kh * kw + ky * kw + kx) * 32 + (c % 32)


def dense_pseudo_channels(cin, kh, kw):
    """Number of pseudo-channels.  Cin is a multiple of 32, so this is exact."""
    return (cin // 32) * kh * kw * 32


def dense_im2col(u_chw, kh, kw, stride, pad):
    """[Cin,H,W] input -> [T,OH,OW] tap planes in the dense order.

    The reference the device is checked against; the runtime never materializes
    this.  Out-of-range taps are 0, which the requant's feed offset accounts for.
    """
    C, H, W = u_chw.shape
    pt, pb, pl, pr = pad
    sh, sw = stride
    oh = (H + pt + pb - kh) // sh + 1
    ow = (W + pl + pr - kw) // sw + 1

    out = np.zeros((dense_pseudo_channels(C, kh, kw), oh, ow), u_chw.dtype)
    ys = np.arange(oh) * sh - pt
    xs = np.arange(ow) * sw - pl
    for ky in range(kh):
        yy = ys + ky
        ok_y = (yy >= 0) & (yy < H)
        for kx in range(kw):
            xx = xs + kx
            ok_x = (xx >= 0) & (xx < W)
            for c in range(C):
                plane = np.zeros((oh, ow), u_chw.dtype)
                plane[np.ix_(ok_y, ok_x)] = u_chw[np.ix_([c], yy[ok_y], xx[ok_x])][0]
                out[dense_tap(ky, kx, c, kh, kw)] = plane
    return out


def dense_weight_matrix(wq_oihw):
    """int8 [Cout,Cin,KH,KW] -> the [Cout,T] 1x1 weight matrix of the dense order."""
    Cout, C, KH, KW = wq_oihw.shape
    w = np.zeros((Cout, dense_pseudo_channels(C, KH, KW)), np.int16)
    for ky in range(KH):
        for kx in range(KW):
            for c in range(C):
                w[:, dense_tap(ky, kx, c, KH, KW)] = wq_oihw[:, c, ky, kx]
    return w


def pack_dense_int8(wq_oihw) -> bytes:
    """The conv's weights in the 1x1 blob layout over the dense tap channels."""
    w = dense_weight_matrix(wq_oihw)
    return pack_pointwise_int8(w.reshape(w.shape[0], w.shape[1], 1, 1))


# --- the array's NATIVE 3x3 walk ---------------------------------------------
#
# The window is not lowered at all here: the array itself steps the nine taps
# (T41_NNA_MANUAL.md 11.2), so the weight blob is one 32x32 bit-plane tile per
# (output group, input group, TAP) -- the same 1,024-byte tile the 1x1 recipe
# uses, just nine of them per group pair.
#
# Out-group-major, because the per-pass weight feed pushes the slice of
# (out-group, in-group) pairs `[k*gpp*D, (k*gpp + groups_k)*D)` (manual 11.2.5).
#
# The nine taps go in the order the array consumes them, which is row-major with
# THE MIDDLE ROW REVERSED (manual 11.2.4 step 6):
#
#     tap i  ->  row = i // 3,  col = 2 - i % 3 if row == 1 else i % 3
#
# Baking that here keeps the executor's weight feed a straight sequential push.

def k3_tap_position(i, kw=3):
    """Kernel (row, col) that tap slot `i` carries."""
    row, col = i // kw, i % kw
    return row, (kw - 1 - col) if row == 1 else col


def pack_k3_int8(wq_oihw) -> bytes:
    """int8 [Cout,Cin,3,3] -> the native-3x3 blob, Dout*D*9 tiles of 1,024 B."""
    Cout, Cin, KH, KW = wq_oihw.shape
    if (KH, KW) != (3, 3):
        raise ValueError(f"native 3x3 blob wants a 3x3 kernel, got {KH}x{KW}")
    w = np.asarray(wq_oihw, np.int16)
    Coutp = pad32(Cout)
    # The walk splits the input groups across two execution units, so a layer
    # narrower than two groups is padded to two.  The partner group's weights
    # are zero and the runtime aliases its plane onto group 0's, exactly as the
    # 1x1 path does for a single-group input (plt_conv_pw.c, pw_prep).
    Cinp = max(pad32(Cin), 64)
    if (Coutp, Cinp) != (Cout, Cin):
        wp = np.zeros((Coutp, Cinp, KH, KW), np.int16)
        wp[:Cout, :Cin] = w
        w = wp
    D, Dout = Cinp // 32, Coutp // 32

    out = bytearray()
    for go in range(Dout):
        for gi in range(D):
            for i in range(9):
                ky, kx = k3_tap_position(i, KW)
                tile = w[go * 32:(go + 1) * 32, gi * 32:(gi + 1) * 32, ky, kx]
                out += pack_pointwise_int8(tile.reshape(32, 32, 1, 1))
    return bytes(out)
