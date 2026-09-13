"""Everything a device run of the whole model needs, and what to check it against.

  python3 -m compile.dump_run <outdir> [image_index]

  input.bin       the graph input in the array's layout: u = x_q - x_zp, NDHWC32
  checksums.txt   one line per layer, `L<i> <name> <fnv>` -- the FNV-1a that
                  plt_run --checksums must print, over the LOGICAL tensor
  logits.f32      the simulator's final logits

The checksum walks the same bytes the runtime does: for an NDHWC32 tensor the
D x H x W x 32 logical region without the tile padding, for a vector its C
elements.  Both hash 32-bit little-endian words, FNV-1a.
"""
import os
import sys

import numpy as np

from . import layout, nnmath, ops
from . import onnx_import as oi

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fnv1a_words(buf: bytes) -> int:
    h = 0xCBF29CE484222325
    for w in np.frombuffer(buf, "<u4"):
        h = ((h ^ int(w)) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def _tensor_bytes(L, a):
    """The layer output as the device holds it, logical region only."""
    if isinstance(L, oi.ConvLayer):
        return layout.pack_ndhwc32((a + 128).astype("uint8").astype("int8")).astype("uint8").tobytes()
    if isinstance(L, oi.PoolLayer):
        return a.astype("int8").tobytes()
    return a.astype("<f4").tobytes()          # FCLayer: dequantized logits


def main(argv):
    outdir = argv[1]
    which = int(argv[2]) if len(argv) > 2 else 0
    os.makedirs(outdir, exist_ok=True)

    model = oi.load(os.path.join(_ROOT, "fixtures", "mobilenetv1_int8.onnx"))
    x = np.fromfile(os.path.join(_ROOT, "fixtures", "val", f"inp_{which:02d}.f32"),
                    np.float32).reshape(3, 224, 224)

    q = model.input_quant
    xq = nnmath.quantize(x, q.scale, q.zero_point)
    u = (xq - model.layers[0].act_in.zero_point).astype("uint8")
    layout.pack_ndhwc32(u.astype("int8")).astype("uint8").tofile(f"{outdir}/input.bin")

    a = xq
    lines = []
    for i, L in enumerate(model.layers):
        a = ops.find_or_die(L).simulate(L, a)
        lines.append(f"L{i} {ops.find_or_die(L).NAME} {fnv1a_words(_tensor_bytes(L, a)):016x}")

    with open(f"{outdir}/checksums.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    a.astype("<f4").tofile(f"{outdir}/logits.f32")

    print("\n".join(lines))
    print(f"top1={int(np.argmax(a))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
