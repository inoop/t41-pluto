"""Per-layer reference checksums for the detection model.

  python3 -m compile.dump_yolox <outdir> [model.onnx]

`plt run --checksums` prints one line per layer, naming the OUTPUT TENSOR it
just wrote.  detect_int.run_all() computes every tensor of the same graph in
exact integer arithmetic on the host.  Matching them by tensor name rather than
by index sidesteps the fact that the device runs the fused graph (139 layers)
while the IR still has its SiLUs separate (243) -- the tensors themselves are
the same tensors either way.

This is the gate the four-detection summary is not: it compares all 139 layer
outputs byte for byte, which is what a rewritten data-moving kernel needs.
"""
import os
import sys

import numpy as np
import cv2

from . import detect_int, layout, model_build, nnmath, preproc
from . import onnx_import as oi
from .dump_run import fnv1a_words

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(_ROOT, "fixtures", "dog.jpg")
NANO = os.path.join(_ROOT, "fixtures/yolox/yolox_nano_int8.onnx")


def main(argv):
    outdir = argv[1] if len(argv) > 1 else "build/run"
    os.makedirs(outdir, exist_ok=True)
    onnx = argv[2] if len(argv) > 2 else NANO
    chw, _ = preproc.letterbox(cv2.imread(IMG), oi.load(onnx).input_shape[1])

    # the reference runs the UNFUSED graph, where each SiLU is still its own
    # layer and produces its own named tensor
    tensors = detect_int.run_all(oi.load(onnx), chw)

    # the device runs the FUSED graph and names tensors by position, so walk a
    # freshly built copy to learn which ONNX tensor each `actN` is
    fused = oi.load(onnx)
    model_build.build(fused)

    lines = []
    for i, L in enumerate(fused.layers):
        b = tensors.get(L.output)
        if b is None or not isinstance(b, np.ndarray) or b.ndim != 3:
            continue
        packed = layout.pack_ndhwc32(np.asarray(b, np.uint8).astype(np.int8))
        lines.append(f"act{i + 1} {fnv1a_words(packed.astype(np.uint8).tobytes()):016x}")
    with open(f"{outdir}/yolox_chk.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    # channel counts, so the checker can tell a real mismatch from a layer whose
    # top channel group is only partly used
    from . import shapes
    shp, _ = shapes.propagate(fused, *fused.input_shape)
    with open(f"{outdir}/yolox_tensors.txt", "w") as f:
        for i, L in enumerate(fused.layers):
            f.write(f"act{i + 1} {shp[L.output][0]}\n")
    print(f"wrote {outdir}/yolox_chk.txt: {len(lines)} tensors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
