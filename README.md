# t41-pluto

The [Ingenic T41](https://en.ingenic.com.cn/products-detail/id-19.html) is a MIPS-based SoC used in several models of WiFi-enabled cameras, most notably the [Wyze Cam v4](https://www.wyze.com/products/wyze-cam), several Noorio cameras, and others. You can usually find a T41 device for under $20.

These little devices are suprisingly capable: dual-core MIPS processors with 512-bit SIMD extensions (MXU3.1), as well as hardware support for image manipulation like scaling and cropping, H264 encoding, etc. Most of this is well understood and documented.

The really interesting piece however, is the neural network accelerator (NNA), which is designed for real-time computer vision applications like object detection (cars, people, etc.). With some hand-waving, this is essentially a multiply-and-accumulate array that can carry out vector/matrix multiplications really fast. Since MAC is the single most computationally expensive part of 2D convolutions, this device can significantly speed up many covolutional neural networks (CNN).

Unfortunately we don't have access to any kind of low-level documentation that tells us how to use this device or even how it works. If we could figure out how to drive this accelerator, we could turn these cameras into low-cost CV AI development boards for fun and profit.

This repo is a product of a multi-week effort to reverse engineer the NNA hardware with the goal of running our own models on this camera. The majority of the work was carried out using Claude, [Ghidra MCP](https://github.com/bethington/ghidra-mcp), and a lot of late-night prompting. The vendor ships a proprietary SDK with an ONNX compiler ('Magik') and runtime library ('venus'), the latter of which was the primary source of the analysis.

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
python3 -m compile.cli model_int8.onnx -o model.pluto
```

The input *must* be statically quantized QDQ ONNX. The compiler doesn't quantize; it reads scales and zero points already in the graph. A float model is rejected. So are "int8" downloads that use dynamic quantization or QOperator nodes (QLinearConv). The expected form is per-channel symmetric int8 weights and per-tensor int8 activations, each tensor wrapped in QuantizeLinear/DequantizeLinear pairs, with a calibrated range on every activation. That is what ONNX Runtime's static quantizer produces:

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
- Worked examples: models/mobilenetv2/build.py is a complete export-quantize-compile script for a classifier, and models/patchcore/quantize.py shows a custom output range.

The graph must use operations Pluto runs.

- Supported:
  - 1×1 convolutions; 3×3 convolutions and 3×3 depthwise at stride 1 or 2 with padding 1; small dense kernels via a generic path
  - ReLU and ReLU6 (folded into quantization)
  - SiLU
  - Add, Concat, max-pool, 2× nearest-neighbour Resize, global average pool, a final fully-connected (Gemm)
  - YOLOX's Focus slicing

## Running a model

On the camera, copy the file and run it with the runtime built by make (build/plt):

```
plt run model.pluto --input input.bin --profile
```

- --input takes the raw input tensor. Each pixel is a 32-byte cell with the quantized channel values in the first three bytes; rows are padded to a multiple of 4 pixels, height to an even number. models/mobilenetv2 and models/patchcore both have small packers.
- --repeat 15 gives warm timings.
- --topk 5 prints classifier labels; --yolox decodes detections.
- --camera --rtsp runs live.

# Reverse Engineering Notes

The NNA was a tough nut to crack, even with the power of a frontier AI model like Claude Opus 5. A combination of disassembly/decompilation of the vendor runtime (`libvenus.so`), as well as micro-benchmarks running on the device, were used to incrementally piece together and map our the hardware ISA, various register functionalties, encodings, and timings. In an ideal world I would have simply been able to tell Claude to "reverse engineer the NNA, make no mistakes" and let it run, but this did not work at all.

First of all, it's important to note that the The XBurst2 cores in the T41 are not just regular old MIPS, but have a 512-bit SIMD extension called MXU3 that the vendor code uses extensively. This means you cannot just drop the libvenus.so file in Ghidra and get a clean disassembly, let alone something that resembles C code.

To get around this I tracked down an Ingenic manual describing the MXU3's instruction set, and combining this with Ingenic's GCC tool chain source code, I asked Claude to write me a Ghidra [SLEIGH file](https://ghidra.re/ghidra_docs/languages/html/sleigh.html). It took several iterations to get right because Claude kept making mistakes either around p-code generation, or instruction stringification, requiring hand-holding in places. I then added an MCP server to let Claude talk to Ghidra directly to start the analysis process.

Unfortunately Ghidra doesn't play all that well with SIMD, and much of the vendor's kernels where heavily SIMD-optimized and make extensive use of loop unrolling. In addition, there are six special instructions in the MXU3 instruction space (COP3) whose exact function was unknown beyond the fact that they are used to read/write data to/from the array. 

Claude struggled to make sense of these listings of essentially thousands of lines of special instructions. Even with `/goal` it would just keep giving up on trying to reverse-engineer even a single kernel. This reveals a real problem with the current generation of coding assistants and frontier models, which is that they are not yet very good at building mental models automatically. The will reason for a large number of turns, and then eventually 'forget' what they discovered half an hour ago. 

To work around this I changed the task from reverse engineering the code to *writing the missing NNA hardware manual*. This let me break up the problem into a set of incremental, achievable steps, with a single memory at the center that Claude could update over time. For example, I started by analyzing the NNA-related instructions. What is their bit-encoding, are their values immediate, in what context do they appear, etc. This led to a hypothesis that there are several small memories inside the array for different functions (e.g. activations, weights, configuration, walk program).

This then let us isolate functional regions inside the code, and compare different kernels optimized for different use cases. For example, a kernel for a 1x1 convolution is structured differently from a 3x3 convolution, and 8-bit and 4-bit variants for these exist as well. Looking at code differences let Claude build hypotheses about how things worked, which could be tested against the real hardware.

I eventually settled on a workflow where I would hand Claude a specific task, e.g. "determine what register B.10 is used for", let it run for a while, and then have it update the hardware manual once it discovered the answer. The vendor library and the hardware acted as oracles in this flow, making it possible for Claude to form hypotheses and either accept or reject them. 

The reason that this works is that Claude 'learns' about the hardware over time and doesn't try to rediscover what each instruction does or what specific registers are for. Each task is constrained and typically fits in the 'smart zone' of the model's context window, and we have an oracle (i.e. the hardware) to validate our hypotheses.

There were a few times where my meat brain had to jump in to help. For example, in the early stages Claude would try to run things on the device and it would crash with `SIGILL` about 50% of the time. I remembered that there's two cores, and that the NNA driver (`/dev/nna_soc`) enables COP2 instructions (i.e. MXU3+NNA), and suggested it pin the process to a single core, which solved the problem. 

# Next Steps

Pluto currently supports 8-bit activations and weights, but the hardware actually supports 4-bit and even 2-bit weights. Profiling shows that memory bandwidth and the CPU shoveling bytes into the NNA is currently the biggest bottleneck. The vendor ships a version of yolox-s that uses 4-bit weights and activations across most of the network, and also drops SiLU for ReLu. With this setup they are able to reach impressive speeds: the end-to-end latency for this model is just ~38ms.

However, just quantizing existing models to 4-bit kills accuracy, so re-training them using QAT is required to make them usable. I'm leaving this as an exercise to the reader.