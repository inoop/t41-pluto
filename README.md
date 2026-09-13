# t41-pluto

The [Ingenic T41](https://en.ingenic.com.cn/products-detail/id-19.html) is a MIPS-based SoC used in several models of WiFi-enabled cameras, most notably the [Wyze Cam v4](https://www.wyze.com/products/wyze-cam), several Noorio cameras, and others. You can usually find a T41 device for under $20.

These little devices are suprisingly capable: dual-core MIPS processors with 512-bit SIMD extensions (MXU3.1), as well as hardware support for image manipulation like scaling and cropping, H264 encoding, etc. Most of this is well understood and documented.

The really interesting piece however, is the neural network accelerator (NNA), which is designed for real-time computer vision applications like object detection (cars, people, etc.). With some hand-waving, this is essentially a multiply-and-accumulate array that can carry out vector/matrix multiplications really fast. Since MAC is the single most computationally expensive part of 2D convolutions, this device can significantly speed up many covolutional neural networks (CNN).

Unfortunately we don't have access to any kind of low-level documentation that tells us how to use this device or even how it works. If we could figure out how to drive this accelerator, we could turn these cameras into low-cost CV AI development boards for fun and profit.

This repo is a product of a multi-week effort to reverse engineer the NNA hardware with the goal of running our own models on these cameras. The majority of the work was carried out using Claude, [Ghidra MCP](https://github.com/bethington/ghidra-mcp), and a lot of late-night prompting. The vendor ships a proprietary SDK with an ONNX compiler ('Magik') and runtime library ('venus'), the latter of which was the primary source of the analysis.

This has produced the following:

- a [hardware manual](docs/T41_NNA_MANUAL.md) that describes how the NNA works. This includes:
  - a high-level description of the hardware: tiling, configuration, 'walk' programming, re-quantization, etc.
  - the custom COP2 instructions that ferry data into- and out of- the array
  - a near-complete description of the various configuration registers
- a working model runtime ('Pluto') for running ONNX models on the device (vibe-coded by Claude)
  - an ONNX to .pluto compiler for 8-bit quantized models
  - a small on-device runtime (`plt`) that can run a model and serve output over RTSP
  - convolutions run on the NNA, other layers (activation, etc.) use optimized MXU3 kernels

I highly recommend reading the [manual](docs/T41_NNA_MANUAL.md) if you're interested in computer vision, TPUs, etc. It's an interesting design and I certainly learned a lot studying this device.

However, if you're just interested in running your own models on an Ingenic T41, you can feed it into your frontier model of choice and it should contain everything needed to reproduce the work in this repo.

# Performance Numbers

Here are some performance numbers for a small selection of pre-trained models that were converted to run on pluto. Note that these were quantized to 8-bit from FP32 originals, quantized aware training (QAT) would probably improve accuracy. With that said, YOLOX-S runs at about 8 fps which is probably acceptable for many applications.

| Model                   | Task              | Input   | Layers | Latency  | FPS | Notes                                                 |
| ----------------------- | ----------------- | ------- | -----: | -------: | --: | ----------------------------------------------------- |
| MobileNetV1             | classification    | 224×224 |     29 |  17.7 ms |  56 |                                                       |
| MobileNetV2             | classification    | 224×224 |     64 |  19.5 ms |  51 |                                                       |
| YOLOX-Nano              | detection (COCO)  | 416×416 |    139 |  28.9 ms |  35 |                                                       |
| YOLOX-S                 | detection (COCO)  | 640×640 |    109 | 121.3 ms | 8.2 | mAP@0.5:0.95 0.4216                                   |
| PatchCore (MobileNetV2) | anomaly detection | 224×224 |     52 |  17.7 ms |  56 | image AUROC: bottle 1.000, capsule 0.945, screw 0.820 |


# Usage

## Pluto Compiler

Pluto's compiler is a Python module run from the repository root. It takes an int8-quantized ONNX file and writes a .pluto file for the camera:

```
python3 -m compile.cli fixtures/mobilenetv1_int8.onnx -o ./mobilenetv1.pluto
```

## Quantization

The compiler input *must* be statically quantized QDQ ONNX. The compiler doesn't quantize; it reads scales and zero points already in the graph. A float model is rejected. So are "int8" downloads that use dynamic quantization or QOperator nodes (QLinearConv). The expected form is per-channel symmetric int8 weights and per-tensor int8 activations, each tensor wrapped in QuantizeLinear/DequantizeLinear pairs, with a calibrated range on every activation. That is what ONNX Runtime's static quantizer produces:

```
from onnxruntime.quantization import quantize_static, QuantFormat, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process

quant_pre_process("model.onnx", "model_pre.onnx")
quantize_static("model_pre.onnx", "model_int8.onnx", calibration_reader,
                quant_format=QuantFormat.QDQ, per_channel=True,
                activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                extra_options={"ActivationSymmetric": False, "WeightSymmetric": True,
                               "TensorQuantOverrides": {"image": [{"symmetric": True}]}})
```

- Calibration reader: it feeds a few hundred representative, preprocessed inputs (e.g. 64–256).
- Input override: replace image with your model's input name. It gives the image a zero point of 0, which all our models use.
- Set ranges up front: any range you want to control goes in TensorQuantOverrides. Never edit scales in the ONNX afterwards: bias scales are derived from input scales, so a later edit silently breaks the layer.
- Worked examples: models/mobilenetv2/build.py is a complete export-quantize-compile script for a classifier

## Running a model

On the camera, copy the file and run it with the runtime built by make (build/plt):

```
plt run model.pluto --input input.bin --profile
```

- --input takes the raw input tensor. Each pixel is a 32-byte cell with the quantized channel values in the first three bytes; rows are padded to a multiple of 4 pixels, height to an even number. models/mobilenetv2 and models/patchcore both have small packers.
- --repeat 15 gives warm timings.
- --topk 5 prints classifier labels; --yolox decodes detections.
- --camera --rtsp runs live.

## Supported Operations

The graph must use operations Pluto runs.

- 1×1 convolutions; 3×3 convolutions and 3×3 depthwise at stride 1 or 2 with padding 1; small dense kernels via a generic path
- ReLU and ReLU6 (folded into quantization)
- SiLU
- Add, Concat, max-pool, 2× nearest-neighbour Resize, global average pool, a final fully-connected (Gemm)
- YOLOX's Focus slicing

If you need support for additional layers, just ask your favorite AI model :)

# Reverse Engineering Notes

The NNA was a tough nut to crack, even with the power of a frontier AI model like Claude Opus 5. A combination of disassembly/decompilation of the vendor runtime (`libvenus.so`), as well as micro-benchmarks running on the device, were used to incrementally piece together and map our the hardware ISA, various register functionalties, encodings, and timings. In an ideal world I would have simply been able to tell Claude to "reverse engineer the NNA, make no mistakes" and let it run, but this did not work at all.

First of all, it's important to note that the The XBurst2 cores in the T41 are not just regular old MIPS, but have a 512-bit SIMD extension called MXU3 that the vendor code uses extensively. This means you cannot just drop the libvenus.so file in Ghidra and get a clean disassembly, let alone something that resembles C code.

To get around this I tracked down an Ingenic manual describing the MXU3's instruction set, and combining this with Ingenic's GCC tool chain source code, I asked Claude to write me a Ghidra [SLEIGH file](https://ghidra.re/ghidra_docs/languages/html/sleigh.html). It took several iterations to get right because Claude kept making mistakes either around p-code generation, or instruction stringification, requiring hand-holding in places. I then added an MCP server to let Claude talk to Ghidra directly to start the analysis process.

Unfortunately Ghidra doesn't play all that well with SIMD, and much of the vendor's kernels where heavily SIMD-optimized and make extensive use of loop unrolling. In addition, there are six special instructions in the MXU3 instruction space (COP3) whose exact function was unknown beyond the fact that they are used to read/write data to/from the array. 

Claude struggled to make sense of these listings of essentially thousands of lines of special instructions. Even with `/goal` it would just keep giving up on trying to reverse-engineer even a single kernel.

This exposes a real weakness in the current generation of coding assistants: they are not yet good at building mental models on their own. They reason for many turns and then forget what they discovered half an hour ago.

So I reframed the task. Instead of reverse engineering the code, I asked it to *write the missing NNA hardware manual*. I started with the NNA-related instructions: their bit encodings, whether their operands were immediates, the contexts they appeared in. That led to the hypothesis that the array contains several small memories with distinct roles (activations, weights, configuration, walk program).

With that in hand we could isolate functional regions in the code and compare kernels built for different cases. A 1x1 convolution kernel is structured differently from a 3x3 kernel, and both exist in 8-bit and 4-bit variants. The differences let Claude form hypotheses about how the hardware worked, which we then tested on the real thing.

I eventually settled on a workflow of handing Claude one narrow task, such as "determine what register B.10 is used for", letting it run, and having it update the manual once it found the answer. The vendor library and the hardware acted as oracles, so each hypothesis could be accepted or rejected. This works because Claude accumulates knowledge in the manual rather than rediscovering what each instruction and register does. Each task stays inside the "smart zone" of the context window, and the hardware validates the result.

A few times my meat brain had to step in. Early on, Claude's test programs would crash with SIGILL about half the time. I remembered that the T41 has two cores and that the NNA driver (/dev/nna_soc) enables COP2 instructions (MXU3 and NNA) on the calling core. Pinning the process to a single core fixed it.

# Next Steps

Pluto currently supports 8-bit activations and weights, but the hardware actually supports 4-bit and even 2-bit weights. Profiling shows that memory bandwidth and the CPU shoveling bytes into the NNA is currently the biggest bottleneck. The vendor ships a version of yolox-s that uses 4-bit weights and activations across (most of) the network, and also drops SiLU for ReLu. With this setup they are able to reach impressive speeds: the end-to-end latency for this model is just ~38ms.

However, just quantizing existing models to 4-bit kills accuracy, so re-training them using QAT is required to make them usable. I'm leaving this as an exercise to the reader.