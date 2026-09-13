"""Dump the classifier tail (GlobalAvgPool + FC) inputs/goldens from the simulator.

  python3 -m compile.dump_tail <outdir>

  tail_gapin.bin   int8  [C][H][W]   activation entering the pool (zp = gap_in_zp)
  tail_gapout.bin  int8  [C]         pooled+requantized activation (FC input)
  tail_fcw.bin     int8  [Cout][Cin] FC weights
  tail_fcb.bin     int32 [Cout]      FC bias
  tail_fcs.bin     f32   [Cout]      FC per-channel weight scale
  tail_logits.bin  f32   [Cout]      final dequantized logits (golden)
  tail_meta.txt    scalars
"""
import os
import sys

import numpy as np

from . import onnx_import as oi
from . import nnmath, ops

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv):
    outdir = argv[1]; os.makedirs(outdir, exist_ok=True)
    model = oi.load(os.path.join(ROOT, "fixtures", "mobilenetv1_int8.onnx"))
    x = np.fromfile(os.path.join(ROOT, "fixtures", "val", "inp_00.f32"), np.float32).reshape(3, 224, 224)
    q = model.input_quant
    a = nnmath.quantize(x, q.scale, q.zero_point)
    gap_in = gap_out = logits = None
    P = F = None
    for L in model.layers:
        if isinstance(L, oi.PoolLayer):
            P, gap_in = L, a.copy()
        elif isinstance(L, oi.FCLayer):
            F = L
        a = ops.find_or_die(L).simulate(L, a)
        if L is P:
            gap_out = a.copy()
        elif L is F:
            logits = a
    gap_in.astype(np.int8).tofile(f"{outdir}/tail_gapin.bin")
    gap_out.astype(np.int8).tofile(f"{outdir}/tail_gapout.bin")
    F.wq.astype(np.int8).tofile(f"{outdir}/tail_fcw.bin")
    F.bias_q.astype("<i4").tofile(f"{outdir}/tail_fcb.bin")
    F.w_scale.astype("<f4").tofile(f"{outdir}/tail_fcs.bin")
    logits.astype("<f4").tofile(f"{outdir}/tail_logits.bin")
    C, Hh, Ww = gap_in.shape
    with open(f"{outdir}/tail_meta.txt", "w") as f:
        f.write(f"C={C} H={Hh} W={Ww} gap_in_scale={P.act_in.scale!r} gap_in_zp={P.act_in.zero_point} "
                f"gap_out_scale={P.act_out.scale!r} gap_out_zp={P.act_out.zero_point} "
                f"fc_in_scale={F.act_in.scale!r} fc_in_zp={F.act_in.zero_point} "
                f"fc_out_scale={F.act_out.scale!r} fc_out_zp={F.act_out.zero_point} "
                f"fc_cout={F.wq.shape[0]} fc_cin={F.wq.shape[1]}\n")
    print(open(f"{outdir}/tail_meta.txt").read().strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
