"""Dump one layer's device blobs + golden from the bit-accurate simulator.

  python3 -m compile.dump_layer <layer_index> <outdir>

Each op module decides what its layer needs; for a convolution that is

  L<k>_in.bin    uint8 activation as FED to the array: u = x_q - x_zp, NDHWC32
  L<k>_w.bin     the packed weights, exactly as the executor pushes them
  L<k>_tbl.bin   the requant table: per output group, int32 bias[32] at +0x00
                 then int32 mult[32] at +0x80
  L<k>_gold.bin  uint8 expected output p = y_q + 128, NDHWC32
  L<k>_meta.txt  geometry
"""
import os
import sys

import numpy as np

from . import onnx_import as oi
from . import ops, simulate as sim

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv):
    idx, outdir = int(argv[1]), argv[2]
    os.makedirs(outdir, exist_ok=True)

    model = oi.load(os.path.join(_ROOT, "fixtures", "mobilenetv1_int8.onnx"))
    x = np.fromfile(os.path.join(_ROOT, "fixtures", "val", "inp_00.f32"),
                    np.float32).reshape(3, 224, 224)

    act_in, L, out = sim.run_to(model, x, idx)
    mod = ops.find_or_die(L)
    meta = mod.dump_blobs(L, act_in, out, outdir, idx)

    text = " ".join(f"{k}={v}" for k, v in meta.items())
    with open(f"{outdir}/L{idx}_meta.txt", "w") as f:
        f.write(f"idx={idx} op={mod.NAME} {text}\n")
    print(f"L{idx}: {mod.NAME} {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
