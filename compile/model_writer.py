"""Serialize a pluto model to the .pluto container, and dump a text view.

The .dump() text is byte-for-byte identical to what the C reader
(runtime/tools/plt_dump) prints, which is how the Phase 0 round-trip test
proves the writer and reader agree on the format.
"""
from dataclasses import dataclass, field
from typing import List, Sequence

from . import format as fmt
import struct


def _pad_ids(ids: Sequence[int], fill: int) -> list:
    ids = list(ids)
    return ids + [fill] * (fmt.MAX_IO - len(ids))


@dataclass
class PlutoTensor:
    id: int
    name: str
    dtype: int
    format: int
    ndims: int
    shape: Sequence[int]
    scale: float = 1.0
    zero_point: int = 0
    data_off: int = fmt.UNUSED
    data_size: int = 0

    def _shape4(self) -> list:
        s = list(self.shape)
        return s + [0] * (4 - len(s))

    def pack(self) -> bytes:
        s = self._shape4()
        name = self.name.encode("ascii")[: fmt.NAME_LEN]
        return struct.pack(
            fmt.TENSOR_FMT, int(self.id), name,
            int(self.dtype), int(self.format), int(self.ndims), 0,
            s[0], s[1], s[2], s[3],
            fmt.f32_bits(self.scale), int(self.zero_point),
            int(self.data_off), int(self.data_size), b"",
        )

    def dump(self, k: int) -> str:
        s = self._shape4()
        return (
            f"tensor {k} id={int(self.id)} name={self.name} "
            f"dtype={int(self.dtype)} format={int(self.format)} ndims={int(self.ndims)} "
            f"shape={s[0]},{s[1]},{s[2]},{s[3]} "
            f"scale={fmt.f32_bits(self.scale):08x} zero_point={int(self.zero_point)} "
            f"data_off={int(self.data_off)} data_size={int(self.data_size)}"
        )


@dataclass
class PlutoLayer:
    id: int
    op: int
    exec: int
    act: int = fmt.Act.NONE
    inputs: Sequence[int] = ()
    output: int = 0
    kernel: Sequence[int] = (0, 0)      # (kh, kw)
    stride: Sequence[int] = (1, 1)      # (sh, sw)
    pad: Sequence[int] = (0, 0, 0, 0)   # (t, b, l, r)
    groups: int = 1
    dilation: Sequence[int] = (1, 1)    # (dh, dw)
    in_bits: int = 8
    w_bits: int = 8
    out_bits: int = 8
    in_zp: int = 0
    weight_off: int = fmt.UNUSED
    weight_size: int = 0
    reqtbl_off: int = fmt.UNUSED
    reqtbl_size: int = 0
    params: bytes = b""

    def pack(self) -> bytes:
        ins = _pad_ids(self.inputs, fmt.UNUSED)
        params = (bytes(self.params) + b"\x00" * 32)[:32]
        return struct.pack(
            fmt.LAYER_FMT, int(self.id),
            int(self.op), int(self.exec), int(self.act), 0,
            ins[0], ins[1], ins[2], ins[3], int(self.output),
            self.kernel[0], self.kernel[1], self.stride[0], self.stride[1],
            self.pad[0], self.pad[1], self.pad[2], self.pad[3],
            int(self.groups), self.dilation[0], self.dilation[1], 0,
            int(self.in_bits), int(self.w_bits), int(self.out_bits), 0,
            int(self.in_zp),
            int(self.weight_off), int(self.weight_size),
            int(self.reqtbl_off), int(self.reqtbl_size),
            params, b"",
        )

    def dump(self, k: int) -> str:
        ins = _pad_ids(self.inputs, fmt.UNUSED)
        params = (bytes(self.params) + b"\x00" * 32)[:32]
        return (
            f"layer {k} id={int(self.id)} op={int(self.op)} exec={int(self.exec)} act={int(self.act)} "
            f"in={ins[0]},{ins[1]},{ins[2]},{ins[3]} out={int(self.output)} "
            f"kernel={self.kernel[0]}x{self.kernel[1]} stride={self.stride[0]}x{self.stride[1]} "
            f"pad={self.pad[0]},{self.pad[1]},{self.pad[2]},{self.pad[3]} "
            f"groups={int(self.groups)} dilation={self.dilation[0]}x{self.dilation[1]} "
            f"bits={int(self.in_bits)}/{int(self.w_bits)}/{int(self.out_bits)} in_zp={int(self.in_zp)} "
            f"weight={int(self.weight_off)}/{int(self.weight_size)} "
            f"reqtbl={int(self.reqtbl_off)}/{int(self.reqtbl_size)} "
            f"params={params.hex()}"
        )


@dataclass
class PlutoModel:
    tensors: List[PlutoTensor]
    layers: List[PlutoLayer]
    inputs: Sequence[int]
    outputs: Sequence[int]
    blob: bytes = b""
    flags: int = 0
    nmem_arena_hint: int = 0

    def _offsets(self):
        tensors_off = fmt.HEADER_SIZE
        layers_off = tensors_off + len(self.tensors) * fmt.TENSOR_SIZE
        blob_off = layers_off + len(self.layers) * fmt.LAYER_SIZE
        pad = (-blob_off) % fmt.BLOB_ALIGN
        blob_off += pad
        return tensors_off, layers_off, blob_off, pad

    def pack(self) -> bytes:
        tensors_off, layers_off, blob_off, pad = self._offsets()
        ins = _pad_ids(self.inputs, fmt.UNUSED)
        outs = _pad_ids(self.outputs, fmt.UNUSED)
        header = struct.pack(
            fmt.HEADER_FMT, fmt.MAGIC,
            fmt.VERSION_MAJOR, fmt.VERSION_MINOR, int(self.flags),
            len(self.tensors), len(self.layers), len(self.inputs), len(self.outputs),
            ins[0], ins[1], ins[2], ins[3], outs[0], outs[1], outs[2], outs[3],
            tensors_off, layers_off, blob_off, len(self.blob), int(self.nmem_arena_hint),
            0, 0, 0, 0,
        )
        body = b"".join(t.pack() for t in self.tensors)
        body += b"".join(l.pack() for l in self.layers)
        body += b"\x00" * pad
        body += bytes(self.blob)
        return header + body

    def write(self, path: str) -> None:
        with open(path, "wb") as f:
            f.write(self.pack())

    def dump(self) -> str:
        tensors_off, layers_off, blob_off, _ = self._offsets()
        ins = _pad_ids(self.inputs, fmt.UNUSED)
        outs = _pad_ids(self.outputs, fmt.UNUSED)
        lines = [
            f"header magic={fmt.MAGIC} version={fmt.VERSION_MAJOR}.{fmt.VERSION_MINOR} flags={int(self.flags)}",
            f"header counts tensors={len(self.tensors)} layers={len(self.layers)} "
            f"inputs={len(self.inputs)} outputs={len(self.outputs)}",
            f"header input_ids={ins[0]},{ins[1]},{ins[2]},{ins[3]}",
            f"header output_ids={outs[0]},{outs[1]},{outs[2]},{outs[3]}",
            f"header offsets tensors={tensors_off} layers={layers_off} blob={blob_off} "
            f"blob_size={len(self.blob)} arena={int(self.nmem_arena_hint)}",
        ]
        lines += [t.dump(k) for k, t in enumerate(self.tensors)]
        lines += [l.dump(k) for k, l in enumerate(self.layers)]
        return "\n".join(lines) + "\n"
