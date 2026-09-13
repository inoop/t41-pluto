"""Per-tensor activation ranges for 4-bit re-quantization.

A clipping range chosen for 256 levels is far too wide for 16: keeping the int8
range and simply coarsening the step wastes most of the grid on tail values that
almost never occur.  Re-deriving the range is what makes a 4-bit activation
measurement a fair test rather than a strawman, so this exists before any 4-bit
number is quoted.

EXACT, NOT SAMPLED.  Every activation in this graph is already an int8 byte, so
a 256-bin count per tensor is a COMPLETE description of its distribution -- no
histogram bin width to choose, no reservoir sampling, no approximation.  One
pass over the calibration set gives exact percentiles.

The calibration set is COCO128 (train2017), the same 128 images ORT calibrated
the int8 model on, and evaluation runs on val2017 -- so the two sets are
disjoint and the mAP is not reported on data the ranges were fitted to.

  python3 -m compile.calibrate --images <coco128/images/train2017> \\
      --onnx fixtures/yolox/yolox_s_int8.onnx --out build/ranges.json
"""
import argparse
import glob
import json
import os
import time

import numpy as np

from . import detect_int as di, preproc
from . import onnx_import as oi


def byte_histograms(model, image_paths, progress=True):
    """{tensor name: int64[256]} -- how often each byte value occurs."""
    size = model.input_shape[1]
    hist = {}
    t0 = time.time()
    for n, p in enumerate(image_paths, 1):
        chw, _ = preproc.letterbox_file(p, size)
        for name, b in di.run_all(model, chw).items():
            h = np.bincount(np.asarray(b, np.uint8).ravel(), minlength=256)
            if name in hist:
                hist[name] += h
            else:
                hist[name] = h
        if progress and (n % 10 == 0 or n == len(image_paths)):
            el = time.time() - t0
            print(f"  {n}/{len(image_paths)}  {el:.0f}s  "
                  f"eta {el / n * (len(image_paths) - n):.0f}s", flush=True)
    return hist


def mse_bounds_from(hist, quants, bits=4, n_cand=40):
    """Per tensor, the [lo, hi] that minimises the expected SQUARED error of a
    `bits`-wide grid -- fitted, not guessed at a percentile.

    Percentile clipping is the wrong criterion for these tensors and the
    difference is not marginal.  A layer-1 output here has std 2.5 but a range
    of [-57, +62]: the range is set by rare outliers, so even p99.9 leaves a
    step three times the standard deviation and the whole signal lands on one or
    two levels (79% relative error).  Choosing the clip by squared error instead
    gets the same tensor to about 15%, which an oracle sweep confirms is close to
    the best any 16-level grid can do.

    The histogram is exact -- 256 int8 levels with their counts -- so this is a
    closed-form search over the true distribution, not a sample.
    """
    n = (1 << bits) - 1
    half = 1 << (bits - 1)
    out = {}
    for name, h in hist.items():
        q8 = quants.get(name)
        if q8 is None:
            continue
        real = q8.scale * ((np.arange(256) - 128) - q8.zero_point)
        w = h.astype(np.float64)
        if w.sum() == 0:
            continue
        rmin, rmax = float(real.min()), float(real.max())
        best, best_b = np.inf, [rmin, rmax]
        for f in np.linspace(1.0, 0.03, n_cand):
            lo, hi = rmin * f, rmax * f
            if hi - lo <= 0:
                continue
            s = (hi - lo) / n
            zp = -half - lo / s
            qq = np.clip(np.rint(real / s) + zp, -half, half - 1)
            e = float((w * (s * (qq - zp) - real) ** 2).sum())
            if e < best:
                best, best_b = e, [lo, hi]
        out[name] = best_b
    return out


def bounds_from(hist, quants, percentile):
    """Per tensor, the SIGNED clipping bounds [lo, hi] at the given percentile.

    Signed, not a symmetric |max|.  A symmetric zero-point-0 grid is wrong for
    this network and wrong in a way that is invisible until the end: the head's
    logits live in roughly [-30, +3.6], so forcing the grid to be centred on
    zero clamps the entire bulk of the distribution onto the most negative level
    and every output tensor comes out constant.  The int8 quantization ORT
    produced is asymmetric for the same reason, and the array supports it -- the
    zero point folds into the requant bias, and window padding is the byte for
    q = zp, which is exactly real 0 whatever zp is.

    `quants` maps tensor name -> the int8 Quant it carries, which is what turns
    a byte back into the real value it stands for.
    """
    tail = (100.0 - percentile) / 200.0        # each side
    out = {}
    for name, h in hist.items():
        q = quants.get(name)
        if q is None:
            continue
        real = q.scale * ((np.arange(256) - 128) - q.zero_point)   # ascending
        cum = np.cumsum(h)
        total = cum[-1]
        if total == 0:
            continue
        lo = float(real[min(int(np.searchsorted(cum, tail * total)), 255)])
        hi = float(real[min(int(np.searchsorted(cum, (1.0 - tail) * total)), 255)])
        if hi <= lo:
            lo, hi = float(real[0]), float(real[-1])
        out[name] = [lo, hi]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="fixtures/yolox/yolox_s_int8.onnx")
    ap.add_argument("--images", required=True, help="calibration image directory")
    ap.add_argument("--limit", type=int, default=128)
    ap.add_argument("--percentiles", default="99.9,99.99,100")
    ap.add_argument("--out", required=True, help="output prefix; one JSON per percentile")
    ap.add_argument("--recalibrate", action="store_true",
                    help="ignore the cached histograms and walk the images again")
    args = ap.parse_args(argv)

    model = oi.load(args.onnx)
    paths = sorted(glob.glob(os.path.join(args.images, "*.jpg")))[:args.limit]
    if not paths:
        raise SystemExit(f"no .jpg under {args.images}")
    print(f"calibrating {os.path.basename(args.onnx)} on {len(paths)} images")

    cache = f"{args.out}.hist.npz"
    if os.path.exists(cache) and not args.recalibrate:
        z = np.load(cache)
        hist = {k: z[k] for k in z.files}
        print(f"reusing {cache} ({len(hist)} tensors)")
    else:
        hist = byte_histograms(model, paths)
        np.savez_compressed(cache, **hist)
        print(f"wrote {cache}")

    from . import shapes
    _, qnt = shapes.propagate(model, *model.input_shape)

    todo = [(f"p{float(p):g}", bounds_from(hist, qnt, float(p)))
            for p in args.percentiles.split(",")]
    todo.append(("mse", mse_bounds_from(hist, qnt, bits=4)))

    for name, b in todo:
        path = f"{args.out}.{name}.json"
        json.dump(b, open(path, "w"))
        # How much tighter than the int8 grid is this?  Near 1 means the int8
        # range was already tight and 4 bits will simply be coarse; well below 1
        # means clipping buys real resolution.
        tight = [(b[n][1] - b[n][0]) / (qnt[n].scale * 255)
                 for n in b if n in qnt and qnt[n].scale > 0]
        print(f"  {name}: {len(b)} tensors -> {path}  "
              f"median span/int8-span = {np.median(tight):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
