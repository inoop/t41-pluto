"""The op registry: one module per operation, mirroring runtime/exec/plt_kernels.c.

Each module exposes the same names -- OP, EXEC, NAME, match(), lower(),
simulate(), dump_blobs() -- so the compiler, the simulator and the device-blob
dumper all dispatch through one table instead of each re-deriving the layer
kind.  Adding an operation is one module plus one line here; the procedure is
in docs/ADDING_A_LAYER.md.
"""
from . import avgpool, dense, depthwise, fc, k3, pointwise, stem
from . import silu, add, concat, maxpool, upsample, focus, genconv

# Conv order matters.  `k3` first among the general windows: it uses the array's
# native walk and so takes the wide 3x3s that `dense` (which lowers the window
# into pseudo-channels) cannot.  `genconv` is the CPU fallback behind both.
REGISTRY = (pointwise, depthwise, k3, stem, dense, avgpool, fc,
            silu, add, concat, maxpool, upsample, focus, genconv)


def find(ir_layer):
    """The module that owns this IR layer, or None."""
    for mod in REGISTRY:
        if mod.match(ir_layer):
            return mod
    return None


def find_or_die(ir_layer):
    mod = find(ir_layer)
    if mod is None:
        raise ValueError(f"no op module claims {ir_layer!r}")
    return mod
