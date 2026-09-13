"""ImageNet MobileNetV2 for pluto: data, preprocessing, and pluto's arithmetic.

Weights: torchvision's MobileNet_V2_Weights.IMAGENET1K_V1 (the same backbone
models/patchcore uses), 71.88% top-1 on ImageNet as published by torchvision.

Evaluation data: Imagenette (fast.ai), ten easily separated ImageNet classes with
their original ImageNet labels.  There is no full ImageNet validation set on this
machine, so accuracy is top-1 over all 1000 classes on Imagenette's 3,925
validation images -- a real labelled accuracy, but on an easier subset: it is
not comparable to a full-ImageNet top-1.  What it does measure honestly is the
float vs pluto-int8 difference.
"""
import glob
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(HERE, "data")
SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

# Imagenette synset -> ImageNet-1k class index (checked against torchvision's
# category names in export.py).
SYNSETS = {
    "n01440764": (0, "tench"), "n02102040": (217, "English springer"),
    "n02979186": (482, "cassette player"), "n03000684": (491, "chain saw"),
    "n03028079": (497, "church"), "n03394916": (566, "French horn"),
    "n03417042": (569, "garbage truck"), "n03425413": (571, "gas pump"),
    "n03445777": (574, "golf ball"), "n03888257": (701, "parachute"),
}


IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"


def ensure_data():
    """Download and unpack Imagenette (342 MB) into data/ on first use.  Nothing
    under data/ belongs in git."""
    root = os.path.join(DATA, "imagenette2-320")
    if os.path.isdir(os.path.join(root, "val")) and os.path.isdir(os.path.join(root, "train")):
        return
    import tarfile
    import urllib.request
    os.makedirs(DATA, exist_ok=True)
    tgz = os.path.join(DATA, "imagenette2-320.tgz")
    print(f"downloading {IMAGENETTE_URL} ...", flush=True)
    urllib.request.urlretrieve(IMAGENETTE_URL, tgz + ".part")
    os.replace(tgz + ".part", tgz)
    with tarfile.open(tgz) as t:
        t.extractall(DATA)
    os.remove(tgz)


def images(split):
    """[(path, imagenet_index)] sorted, for 'train' or 'val'."""
    out = []
    for syn, (idx, _) in SYNSETS.items():
        for p in sorted(glob.glob(os.path.join(DATA, "imagenette2-320", split, syn, "*.JPEG"))):
            out.append((p, idx))
    return out


def load(path):
    """uint8 [224,224,3]: torchvision's evaluation transform -- resize the short
    side to 256 (bilinear), centre-crop 224."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = 256.0 / min(w, h)
    im = im.resize((max(SIZE, round(w * s)), max(SIZE, round(h * s))), Image.BILINEAR)
    w, h = im.size
    l, t = (w - SIZE) // 2, (h - SIZE) // 2
    return np.asarray(im.crop((l, t, l + SIZE, t + SIZE)))


def normalize(rgb):
    return ((rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)


# --- pluto's integer path -------------------------------------------------------
#
# The one place this project reaches into pluto's compiler: accuracy for the
# device has to be measured in the device's arithmetic (models/patchcore found
# ONNX Runtime's fused int8 kernels 60% away from it).  compile/simulate.py only
# walks chains and compile/detect_int.py has no pool/FC, so the two are joined
# here: the graph walker for the body, the classifier ops for the head.

def pluto_logits(model, x_chw):
    sys.path.insert(0, REPO)
    from compile import detect_int as di, onnx_import as oi, nnmath
    from compile.ops import avgpool, fc, add
    t = {model.input_name: nnmath.to_byte(x_chw.astype(np.float64), model.input_quant.scale,
                                          model.input_quant.zero_point)}
    y = None
    for L in model.layers:
        xi = [t[n] for n in L.inputs]
        if isinstance(L, oi.ConvLayer):
            y = di._conv_bytes(xi[0], L)
        elif isinstance(L, oi.AddLayer):
            y = add.apply_bytes(L, xi[0], xi[1])
        elif isinstance(L, oi.PoolLayer):
            q = avgpool.simulate(L, np.asarray(xi[0]).astype(np.int64) - 128)
            y = (np.asarray(q, np.int64) + 128).astype(np.uint8)
        elif isinstance(L, oi.FCLayer):
            return fc.simulate(L, np.asarray(xi[0]).astype(np.int64) - 128)
        else:
            raise TypeError(type(L).__name__)
        t[L.output] = y
    raise RuntimeError("model has no FC head")
