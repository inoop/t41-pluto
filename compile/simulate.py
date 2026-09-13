"""Bit-accurate NumPy model of the pluto/NNA integer datapath.

Runs the chain IR from onnx_import, dispatching each layer through the op
registry in compile/ops/ -- the same table the compiler and the device-blob
dumper use.  This is the host golden the C runtime is checked against, and the
Phase 1 fidelity check compares it to the ORT-int8 reference logits.
"""
import numpy as np

from . import nnmath
from . import onnx_import as oi
from . import ops


def run(model: oi.Model, x_float: np.ndarray) -> np.ndarray:
    """x_float [3,H,W] -> logits [num_classes]."""
    q = model.input_quant
    a = nnmath.quantize(x_float, q.scale, q.zero_point)      # [3,H,W] int8
    for L in model.layers:
        a = ops.find_or_die(L).simulate(L, a)
    return a


def run_to(model: oi.Model, x_float: np.ndarray, idx: int):
    """Run the first `idx` layers.  Returns (activation entering layer idx, that
    layer, its output) -- what a device blob dump needs."""
    q = model.input_quant
    a = nnmath.quantize(x_float, q.scale, q.zero_point)
    for i, L in enumerate(model.layers):
        if i == idx:
            return a, L, ops.find_or_die(L).simulate(L, a)
        a = ops.find_or_die(L).simulate(L, a)
    raise IndexError(f"layer {idx} is past the end of the model")
