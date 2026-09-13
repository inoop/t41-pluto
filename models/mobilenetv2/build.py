"""Export, quantize, compile and evaluate ImageNet MobileNetV2 for the T41.

  python3 build.py [--calib 256] [--eval]

Downloads what it needs on first run, into data/ (not for git): Imagenette for
calibration, and torchvision's ImageNet weights.

  runs/mobilenetv2.onnx        float, from torchvision
  runs/mobilenetv2_int8.onnx   ONNX Runtime static QDQ int8
  runs/mobilenetv2.pluto       what the device runs
  runs/eval.json               top-1 on Imagenette val: float vs pluto int8

Quantization is the same recipe as models/patchcore/quantize.py: per-channel
symmetric weights, per-tensor asymmetric activations, calibrated on training
images, and the image input dictated symmetric (zero point 0) through
TensorQuantOverrides -- never edited afterwards, which breaks bias scales.
"""
import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time

import numpy as np
import onnx
import torch
import torchvision
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                      quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

import common as C

RUNS = os.path.join(C.HERE, "runs")
_M = None


def _init(path):
    global _M
    sys.path.insert(0, C.REPO)
    from compile import onnx_import as oi
    _M = oi.load(path)


def _one(item):
    p, _ = item
    return C.pluto_logits(_M, C.normalize(C.load(p)))


class Reader(CalibrationDataReader):
    def __init__(self, items):
        self.it = iter([{"image": C.normalize(C.load(p))[None]} for p, _ in items])

    def get_next(self):
        return next(self.it, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", type=int, default=256)
    ap.add_argument("--eval", action="store_true",
                    help="also measure top-1 on Imagenette val, float and pluto int8 "
                         "(slow: pluto's simulator runs in Python)")
    args = ap.parse_args()
    os.makedirs(RUNS, exist_ok=True)
    torch.set_num_threads(os.cpu_count())
    os.environ.setdefault("TORCH_HOME", os.path.join(C.DATA, "weights"))

    C.ensure_data()                  # Imagenette, for calibration (and --eval)
    weights = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1   # downloads itself
    for syn, (idx, name) in C.SYNSETS.items():
        assert weights.meta["categories"][idx] == name, (syn, idx, weights.meta["categories"][idx])
    net = torchvision.models.mobilenet_v2(weights=weights).eval()

    val = C.images("val")
    labels = np.array([y for _, y in val])

    if args.eval:                                   # float top-1
        t0 = time.time()
        preds = []
        with torch.no_grad():
            for i in range(0, len(val), 64):
                x = torch.from_numpy(np.stack([C.normalize(C.load(p)) for p, _ in val[i:i + 64]]))
                preds.append(net(x).argmax(1).numpy())
        float_top1 = float((np.concatenate(preds) == labels).mean())
        print(f"float top-1 on {len(val)} Imagenette val images: {float_top1:.4f} "
              f"({time.time()-t0:.0f}s)")

    fl = os.path.join(RUNS, "mobilenetv2.onnx")
    torch.onnx.export(net, torch.zeros(1, 3, C.SIZE, C.SIZE), fl, input_names=["image"],
                      output_names=["logits"], opset_version=13, dynamo=False)

    train = C.images("train")
    calib = train[:: max(1, len(train) // args.calib)][:args.calib]
    pre, q8 = os.path.join(RUNS, "pre.onnx"), os.path.join(RUNS, "mobilenetv2_int8.onnx")
    quant_pre_process(fl, pre)
    quantize_static(pre, q8, Reader(calib), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    extra_options={"ActivationSymmetric": False, "WeightSymmetric": True,
                                   "TensorQuantOverrides": {"image": [{"symmetric": True}]}})
    os.remove(pre)

    pl = os.path.join(RUNS, "mobilenetv2.pluto")
    subprocess.run([sys.executable, "-m", "compile.cli", q8, "-o", pl], cwd=C.REPO, check=True)

    sys.path.insert(0, C.REPO)
    from compile import onnx_import as oi
    iq = oi.load(q8).input_quant
    json.dump({"input_scale": iq.scale, "input_zero_point": iq.zero_point,
               "calibration_images": len(calib)},
              open(os.path.join(RUNS, "quant.json"), "w"), indent=1)
    if not args.eval:
        return

    t0 = time.time()
    with mp.Pool(os.cpu_count(), initializer=_init, initargs=(q8,)) as pool:
        logits = np.stack(pool.map(_one, val, chunksize=8))
    int8_pred = logits.argmax(1)
    int8_top1 = float((int8_pred == labels).mean())
    agree = float((int8_pred == np.concatenate(preds)).mean())
    print(f"pluto int8 top-1: {int8_top1:.4f}; agrees with float on {agree:.4f} of images "
          f"({time.time()-t0:.0f}s)")
    np.save(os.path.join(RUNS, "pluto_logits.npy"), logits.astype(np.float32))
    json.dump({"images": len(val), "float_top1": float_top1, "pluto_int8_top1": int8_top1,
               "int8_float_agreement": agree},
              open(os.path.join(RUNS, "eval.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
