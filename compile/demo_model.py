"""A tiny synthetic pluto model used by the Phase 0 round-trip test.

  python3 -m compile.demo_model write <path>   # serialize to a .pluto file
  python3 -m compile.demo_model dump            # print the text view to stdout
"""
import sys

from . import format as fmt
from .model_writer import PlutoTensor, PlutoLayer, PlutoModel


def build() -> PlutoModel:
    tensors = [
        PlutoTensor(0, "input", fmt.DType.INT8, fmt.Format.NHWC, 4,
                    (1, 224, 224, 3), scale=0.1, zero_point=-128),
        PlutoTensor(1, "stem_out", fmt.DType.INT8, fmt.Format.NDHWC32, 4,
                    (1, 112, 112, 32), scale=0.05, zero_point=0),
        PlutoTensor(2, "logits", fmt.DType.FP32, fmt.Format.VEC, 1,
                    (1000,), scale=1.0, zero_point=0),
    ]
    layers = [
        PlutoLayer(0, fmt.Op.CONV, fmt.Exec.NNA_STD, fmt.Act.RELU,
                   inputs=(0,), output=1, kernel=(3, 3), stride=(2, 2),
                   pad=(1, 1, 1, 1), groups=1, in_bits=8, w_bits=8, out_bits=8,
                   in_zp=-128, weight_off=0, weight_size=1024,
                   reqtbl_off=1024, reqtbl_size=512),
        PlutoLayer(1, fmt.Op.FC, fmt.Exec.MXU_FC, fmt.Act.NONE,
                   inputs=(1,), output=2, kernel=(1, 1), stride=(1, 1),
                   pad=(0, 0, 0, 0), groups=1, in_bits=8, w_bits=8, out_bits=32,
                   in_zp=0, weight_off=1536, weight_size=4096,
                   reqtbl_off=fmt.UNUSED, reqtbl_size=0),
    ]
    return PlutoModel(tensors=tensors, layers=layers, inputs=(0,), outputs=(2,),
                      blob=bytes(range(16)))


def main(argv):
    m = build()
    cmd = argv[1] if len(argv) > 1 else "dump"
    if cmd == "write":
        m.write(argv[2])
    elif cmd == "dump":
        sys.stdout.write(m.dump())
    else:
        sys.stderr.write("usage: demo_model {write <path>|dump}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
