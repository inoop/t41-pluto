"""Pack a JPEG as the device's input tensor, using YOLOX's own preprocessing.

  python3 -m compile.pack_input [out.bin] [image.jpg] [model.onnx]

Letterbox to the model's input size (aspect preserved, the remainder filled with
114), quantize with the input tensor's own scale and zero point, then lay it out
as NDHWC32 -- byte for byte what `plt run --input` expects.  This is the
reference the device is measured against, so it is also what plt_pre.c has to
reproduce live from the camera.
"""
import os
import sys

import cv2
import numpy as np

from . import nnmath, preproc
from . import onnx_import as oi

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_ROOT, "build/run/yolox_in.bin")
IMG = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_ROOT, "fixtures/dog.jpg")
ONNX = sys.argv[3] if len(sys.argv) > 3 else os.path.join(
    _ROOT, "fixtures/yolox/yolox_nano_int8.onnx")


m = oi.load(ONNX)
# the side the model wants -- 416 for YOLOX-Nano, 640 for YOLOX-S
chw, _ = preproc.letterbox(cv2.imread(IMG), m.input_shape[1])
b = nnmath.to_byte(chw.astype(np.float64), m.input_quant.scale, m.input_quant.zero_point)  # [3,H,W] uint8=q+128
C, H, W = b.shape
D = (C + 31) // 32
PH = (H + 1) & ~1; PW = (W + 3) & ~3
out = np.zeros((D, PH, PW, 32), np.uint8)
for d in range(D):
    c0 = d*32; c1 = min(c0+32, C)
    out[d, :H, :W, :c1-c0] = np.transpose(b[c0:c1], (1, 2, 0))
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
out.tofile(OUT)
print("wrote", OUT, out.size, "bytes  (input_quant", m.input_quant, ")")
