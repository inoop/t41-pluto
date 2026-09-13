"""pluto.compile CLI: quantized-ONNX -> .pluto model.

  python3 -m compile.cli <int8_qdq.onnx> -o model.pluto
  python3 -m compile.cli <int8_qdq.onnx> -o model.pluto --w-bits 4

Four-bit re-quantizes the IR before the model is built (compile/quant4.py).
8/8 is the identity, so the default output is byte-for-byte what it always was.
"""
import argparse
import json

from . import onnx_import, model_build, quant4


def main():
    ap = argparse.ArgumentParser(description="Compile a QDQ int8 ONNX to a .pluto model")
    ap.add_argument("onnx")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--w-bits", type=int, default=8, choices=(4, 8),
                    help="weight width (default 8)")
    ap.add_argument("--a-bits", type=int, default=8, choices=(4, 8),
                    help="activation width (default 8); needs --ranges")
    ap.add_argument("--denom", type=float, default=7.0,
                    help="4-bit weight scale divisor: 7 symmetric, 8 uses all 16 levels")
    ap.add_argument("--wmode", default="absmax", choices=("absmax", "mse"),
                    help="4-bit weight scale: max|w| or an MSE-optimal clip search")
    ap.add_argument("--ranges", default=None,
                    help="calibrated activation ranges, from compile.calibrate")
    args = ap.parse_args()
    model = onnx_import.load(args.onnx)
    quant4.apply(model, w_bits=args.w_bits, a_bits=args.a_bits, denom=args.denom, mode=args.wmode,
                 ranges=json.load(open(args.ranges)) if args.ranges else None)
    pm = model_build.build(model)
    pm.write(args.out)
    print(f"wrote {args.out}: {len(pm.tensors)} tensors, {len(pm.layers)} layers, "
          f"blob {len(pm.blob)} bytes, bits w{args.w_bits}/a{args.a_bits}")


if __name__ == "__main__":
    main()
