# Fixtures

Two quantized ONNX models and one test image. Everything else the repo needs is
derived from these.

| file | what it is |
|---|---|
| `mobilenetv1_int8.onnx` | MobileNetV1 224×224, per-tensor QDQ int8 |
| `yolox/yolox_nano_int8.onnx` | YOLOX-Nano 416×416, per-tensor QDQ int8, head outputs cut before decode |
| `dog.jpg` | the detection test image (bicycle, dog, car) |
| `val/inp_00.f32` | one preprocessed 224×224 classifier input, `[3,224,224]` fp32 |

## Why these are committed rather than downloaded

They are **outputs of a local pipeline**, not public artefacts, and every
byte-exactness gate in this repo is defined against these exact weights —
`make yoloxtests` compares all 139 layer outputs, `deploy-check` compares four
detections to four decimal places, `tests/model_end_to_end.sh` compares 29
layer checksums. Re-quantizing from upstream would produce a *different* model
and silently turn those gates into noise, so the models themselves are the
fixture.

## Where they came from

**YOLOX-Nano.** Weights `yolox_nano.pth` from Megvii's
[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) 0.1.1rc0 release.
Exported with their own `exps/default/yolox_nano.py`, with
`head.decode_in_inference = False` (the 9 head tensors are what the compiler
consumes; decode and NMS happen in C, `runtime/io/plt_detect.c`) and
`replace_module(nn.SiLU, ...)`. `torch.onnx.export(..., opset_version=13,
dynamo=False)` — torch 2.13's default dynamo exporter needs onnxscript, and the
legacy path avoids it.

**Quantization.** ONNX Runtime static QDQ int8, per-channel symmetric weights
and per-tensor asymmetric activations, calibrated on 128 COCO128 images. Run
`quant_pre_process()` first for shape inference.

**MobileNetV1.** A float ONNX quantized the same way. The float model and the
other 11 validation inputs are not committed; they were only used by a fidelity
report that is not part of this repo.

`val/inp_00.f32` is `(pixel/255)*2 - 1` in CHW order — `compile/dump_tail.py`
and `compile/dump_run.py` read it to build the device's MobileNet inputs.

## Rebuilding the device inputs

```sh
python3 -m compile.pack_input build/run/yolox_in.bin    # letterboxed dog.jpg
make blobs                                              # MobileNet model + inputs
```
