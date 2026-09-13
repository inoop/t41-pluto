# T41 Neural Network Accelerator — Programmer's Reference Manual

*Architecture, instruction set, configuration registers, data formats and programming procedures for the
neural network accelerator (NNA) of the Ingenic T41 system-on-chip.*

---

## About this document

This manual describes how to program the T41's neural network accelerator directly: what the hardware
computes, how it is configured, how operands reach it, and how results are read back. It is written for
engineers implementing inference kernels on the device.

**This is an independent reconstruction, not a vendor publication.** No Ingenic document describes the
accelerator's registers, instruction set or data path. The T41 SDK ships only the `soc-nna` kernel driver,
which handles memory and DMA plumbing; the *Venus Programming Manual* documents a framework API and an
operator capability envelope, not the hardware. Everything in this manual was established by analysing the
vendor inference library (`libvenus.m.so`) and by measurement on the device. [Appendix A](#appendix-a--sources)
records the sources; [Appendix C](#appendix-c--not-characterised) lists what has not been established.

**Validation.** All 83 convolutions of a production YOLOX-S network run on the accelerator from code written
against this manual, producing output byte-identical to the vendor library, using the model file's weights
and requantization tables unmodified. The complete 95-layer network runs end to end with identical layer
checksums and identical floating-point outputs.

### How to read this manual

The manual is organised so that each part depends only on the parts before it.

- **Part I — The machine** explains what the accelerator computes and how its work is organised: the
  memory hierarchy, the data formats, the tile, the two execution units, the ports, the walk program and the
  readout. It uses no register names beyond the `A.xx`/`B.xx` notation and can be read with no prior
  knowledge of the hardware.
- **Part II — Programming the NNA** gives the procedure: how configuration fields are written, the exact
  instruction order of one layer, memory and DMA, the operational hazards, and a complete worked example.
- **Part III — Layer recipes** gives one procedure per kind of layer (1×1, 3×3, single input group, 8-bit
  input, the RGB image layer, the floating-point heads, large-weight streaming and concatenation).
- **Part IV — Reference** is the instruction set, the configuration fields, the program table and the
  per-recipe field values.

Three reading paths:

| If you want to … | Read |
|---|---|
| understand the hardware | Part I, front to back |
| run one convolution as quickly as possible | [§1.2](#12-the-six-instructions), [§3.9](#39-the-instruction-order-of-one-layer), then [Chapter 5](#5-running-a-layer) and [Chapter 8](#8-worked-example) |
| look up a field, an instruction or a recipe constant | Part IV; [Appendix D](#appendix-d--register-number-cross-reference) for raw register numbers |

### Conventions

| Convention | Meaning |
|---|---|
| `A.xx`, `B.xx` | Configuration field `xx` (hexadecimal) in bank A or bank B ([§4.1](#41-banks-and-fields)) |
| `w0` … `w15` | The sixteen 32-bit word registers that occupy vector register `vr31` ([§4.2](#42-the-word-register-file)) |
| `nnrwr vwN,imm` | Instructions are written in the operand form the Ingenic assembler accepts ([§12.2](#122-operand-encoding)). Immediates are given in this instruction-field form throughout; the merged payload numbers that appear in vendor listings are converted in [Appendix D](#appendix-d--register-number-cross-reference) |
| `D`, `Dout` | Input and output channel-group counts, `Cin/32` and `Cout/32` ([§2.2](#22-channel-groups-of-32)) |
| `round64(x)` | `x` rounded up to a multiple of 64 |
| **H**, **M**, **L** | Confidence attached to a statement or table row. **H**: established by measurement on the device. **M**: consistent with device measurement but not isolated by a dedicated test. **L**: inferred from the vendor code or from a single observation |
| **Not characterised** | The behaviour has not been established; the statement is descriptive only. Every such item is collected in [Appendix C](#appendix-c--not-characterised) |
| *field*, *register* | A configuration *field* is one addressable quantity in bank A or B. The word *register* is reserved for the MXUv3 vector registers `vr0`–`vr31` and the word registers `w0`–`w15` |
| *recipe*, *executor* | A *recipe* is a procedure in this manual for running one kind of layer. An *executor* is the vendor library's implementation of the same; the mapping is in [Appendix B](#appendix-b--vendor-executor-map) |
| *unit A*, *unit B* | The two execution units. Unit A is the one fed on port `0x20`, whose group count is `A.1c` and whose start offset is `B.10`; unit B is fed on port `0x21`, with `A.1d` and `B.11` ([§3.3](#33-the-two-execution-units)) |
| *the NNA* | The accelerator. The vendor software calls it the AIE; Ingenic calls this generation NNA2 |
| *walk program*, *program table* | The *walk program* is the schedule the array follows during a MAC ([§3.5](#35-the-walk-program)); the *program table* is the sixteen-entry storage that holds it |

Numeric literals are hexadecimal when prefixed `0x`, decimal otherwise. Bit ranges are written `[msb:lsb]`.
Byte offsets into a structure are written `+0x00`.

`Cin` and `Cout` denote the **padded** channel counts a layer's weight blob carries, so `D` and `Dout` are
ordinary integer divisions in every recipe and code listing. The `ceil` form matters only when converting a
network's real channel counts, which need not be multiples of 32 ([§2.2](#22-channel-groups-of-32)).

The same immediate value can mean different things to different instructions: `0x20` is bank A to
`nnrwr`, the unit-A activation port to `nndwr`, and the packed readout bank to `nndrd`; `0x40` is bank B
to `nnrwr` and the table port to `nndwr`. Each instruction has its own immediate space
([§12.2](#122-operand-encoding)).

### Glossary

| Term | Meaning |
|---|---|
| **accumulator** | One of the 256 signed 32-bit sums the array keeps for the tile in progress |
| **arm** | The `nncmd 0x60` command that latches the static configuration and readies the accumulators |
| **bank (configuration)** | One of the two groups of configuration fields, A (geometry) and B (mode and activation) |
| **bank (readout)** | One of the two result stores: bank 0 (raw int32, random access) and bank 1 (packed, FIFO) |
| **chunk** | In the general int8 executor, the group of input planes one MAC phase reduces over |
| **column block** | Four consecutive output pixels — the width of a tile |
| **commit** | The `nncmd` command that ends a tile's accumulation, clears the accumulators and publishes the result |
| **descriptor** | One 8-byte NNDMA transfer record in DESRAM |
| **drain** | Reading a result out of a readout bank with `nndrd` |
| **executor** | The vendor library's procedure for one kind of layer ([Appendix B](#appendix-b--vendor-executor-map)) |
| **feed** | Pushing operands into the array with `nndwr` |
| **general int8 executor** | The vendor executor for 8-bit operands with floating-point output — the recipe of [§11.7](#117-floating-point-output-heads) |
| **halo** | The input rows and columns beyond a tile's own footprint that a 3×3 kernel needs — one on each side at stride 1 |
| **heads** | The network's final layers, which produce floating-point output ([§11.7](#117-floating-point-output-heads)) |
| **im2col** | Presenting a `K×K` convolution as `K²` shifted copies of the input, each convolved 1×1 |
| **image layer** | The first layer of a network, which reads a 3-channel image ([§11.6](#116-the-rgb-image-layer)) |
| **kick** | Starting an NNDMA descriptor chain by writing its index to the DMA engine ([§6.5](#65-kick-wait-and-fencing)) |
| **lead** (feed-ahead) | The number of column blocks the feed runs ahead of the MACs |
| **NDHWC32** | The channel-blocked tensor layout: planes of 32 channels, each plane a full `H × W` image |
| **pack stage** | The part of the readout path that applies the requantization table and packs a tile to 4- or 8-bit ([§3.8](#38-requantization-and-packing)) |
| **pass (weight pass)** | One resident set of weights; a layer whose weights do not fit at once is run in several passes |
| **phase (MAC phase)** | One `nnmac`: the walk of one execution unit over its operands for the current tile |
| **plane** | One group of 32 channels of a tensor, stored as a contiguous `H × W` image |
| **port** | One of the seven numbered inlets through which `nndwr` pushes 64 bytes into the array |
| **program entry** | One of the sixteen 15-bit words of the program table |
| **program slot** | Entries 0–7 (slot A) or 8–15 (slot B) of the program table, one per execution unit |
| **push slot** | The 64-byte position inside the array that the next `nndwr` to a port fills; it advances by one per push |
| **recipe** | A procedure in this manual for running one kind of layer (Part III) |
| **reduction round** | One (feed, MAC) round over one 32-channel input group |
| **reset** | The `nncmd 0x00` command; returns the configuration fields to their reset values and clears the word file, but does not fully re-initialise the MAC stage ([§5.1](#51-reset)) |
| **row-bank** | One of the 16 groups of 16 int32 accumulators that readout bank 0 returns per `nndrd` |
| **resident** | Held in ORAM for the whole layer, as opposed to streamed per pass |
| **ring (row ring)** | An ORAM buffer holding a sliding window of input rows for the 3×3 recipes |
| **row pair** | Two consecutive output rows — the height of a tile |
| **row quad** | Four consecutive output rows, produced by the single-input-group recipes in one pass of the two units |
| **staging row** | An ORAM row buffer that a tile's drained result is stored into before DMA to DDR |
| **tap** | One kernel position, `K²` per kernel; a *tap* of weights is the 32 × 32 matrix for that position |
| **tap slot** | One 32-input × 32-output weight matrix's worth of resident weight storage, as counted by the vendor planner |
| **tile** | 2 rows × 4 pixels × 32 output channels: the unit of work the array commits at once |
| **unit** | One of the two execution units, A and B |
| **vendor descriptor** | The per-layer record in the model file from which the vendor library plans a layer ([Appendix B](#appendix-b--vendor-executor-map)); distinct from an NNDMA descriptor |
| **walk** | The sequence of operand positions the array visits during one MAC phase |
| **weight slice** | 256 bytes of weights, pushed as four 64-byte writes to the four weight ports |
| **whole-input mode** | Keeping a layer's entire input resident in ORAM instead of a row ring |
| **window (activation window)** | The 64 bytes one activation push carries |
| **word register (VWR)** | One of the sixteen 32-bit words `w0`–`w15` that occupy `vr31` |

## Contents

**Part I — The machine**
- [1. What the NNA is](#1-what-the-nna-is)
- [2. Memory and data](#2-memory-and-data)
- [3. How work is organised](#3-how-work-is-organised)

**Part II — Programming the NNA**
- [4. The configuration model](#4-the-configuration-model)
- [5. Running a layer](#5-running-a-layer)
- [6. Memory and DMA](#6-memory-and-dma)
- [7. Operational hazards](#7-operational-hazards)
- [8. Worked example](#8-worked-example)

**Part III — Layer recipes**
- [9. Choosing a recipe](#9-choosing-a-recipe)
- [10. The recipe template](#10-the-recipe-template)
- [11. The recipes](#11-the-recipes)

**Part IV — Reference**
- [12. Instruction set](#12-instruction-set)
- [13. Configuration fields](#13-configuration-fields)
- [14. The program table](#14-the-program-table)
- [15. Field values by recipe](#15-field-values-by-recipe)
- [16. Runtime API](#16-runtime-api)

**Appendices**
- [Appendix A — Sources](#appendix-a--sources)
- [Appendix B — Vendor executor map](#appendix-b--vendor-executor-map)
- [Appendix C — Not characterised](#appendix-c--not-characterised)
- [Appendix D — Register-number cross-reference](#appendix-d--register-number-cross-reference)
- [Appendix E — Complete example listing](#appendix-e--complete-example-listing)


---

# Part I — The machine

Part I describes the accelerator as a machine: what one multiply-accumulate computes, where operands live,
how a layer is divided into tiles, how operands enter and results leave. It introduces every term the rest
of the manual uses. Nothing in it requires a register value; the configuration interface is Part II.

## 1. What the NNA is

### 1.1 The atomic operation

The NNA is a **low-precision multiply-accumulate (MAC) accelerator** with a fixed algorithm and a loadable walk
program, called the AIE inside the vendor software. Its primitive is a dot-product reduction over 32 input
channels, evaluated for 32 output channels at once; convolution, depthwise convolution and fully-connected
layers are all expressed as sequences of that one operation ([§2.4](#24-precision)).

Its software-visible behaviour is consistent with a **32 × 32 reduction and output structure**: 32
reduction lanes indexed by input channel, 32 output columns indexed by output channel. The physical
multiplier topology is not vendor-documented and is not needed to program the part. For each pixel's
32-channel activation vector `A`, against one 32 × 32 weight tap `W`, the array computes

```
acc[co] += sum over k = 0..31 of  A[k] * W[k][co]        for all 32 output columns at once
```

A reduction longer than 32 channels is streamed as successive (feed, MAC) rounds, each with its own
32 × 32 weight tap, all accumulating onto the same accumulators.

- **Element types.** Activations 4-bit or 8-bit; weights 2-, 4-, 6- or 8-bit; output 4-bit, 8-bit or float.
- **Accumulators.** Products accumulate into signed 32-bit accumulators. Whether they saturate or wrap on
  overflow is not characterised; production layers do not approach the limit.
- **Committed tile.** One commit covers **8 pixels × 32 output channels** — 2 image rows × 4 consecutive
  pixels — read out as 16 *row-banks*, groups of 16 int32 ([§3.1](#31-the-tile)).
- **Fused requantization.** The array can convert its own accumulators to packed 4- or 8-bit output using a
  per-channel table ([§3.8](#38-requantization-and-packing)), or hand back raw int32 for software to
  process.

Individual multipliers are not addressable: bytes are pushed into fixed **ports** and the hardware routes
them onto the array according to a small **walk program** loaded beforehand.

### 1.2 The six instructions

The array is driven by six coprocessor instructions and configured through about forty fields
([Chapter 13](#13-configuration-fields)). The instructions divide into configuration, data movement and
control:

```
nnrwr  vwN,field     configuration      write field <- word register N
nnrrd  vrN,block     configuration      read a 64-byte configuration block back
nndwr  vrN,port      data in            push 64 bytes into a port (positional)
nnmac  vwN,imm       control            run one MAC phase; accumulate
nncmd  imm           control            reset, arm, commit
nndrd  vrN,bank      data out           drain 64 bytes from a readout bank
```

The instructions carry no memory addresses. Data enters and leaves through the MXUv3 vector registers:
`nndwr` pushes the 64 bytes of a vector register, `nndrd` fills one. Where those bytes come from or go to —
ORAM, DDR or a shuffle network — is the CPU's business ([§2.1](#21-the-memory-hierarchy)).

### 1.3 NNA, MXUv3, NNDMA and the CPU

Three units besides the CPU are involved in running a layer. The CPU issues every instruction and is the
only agent that addresses all three. Confusing them is the most common source of error.

| Unit | What it is | Role in a convolution |
|---|---|---|
| **NNA (AIE)** | Low-precision MAC accelerator with its own operand memories, accumulators and sequencer | Performs the multiply-accumulate reductions |
| **MXUv3** | General-purpose 512-bit SIMD coprocessor, 32 vector registers `vr0`–`vr31` | Moves operands into and out of the array; performs floating-point requantization; builds activation windows for the image layer |
| **NNDMA** | Descriptor-driven DMA engine | Streams tensors between DDR and on-chip ORAM. Moves bytes; computes nothing |

The NNA and MXUv3 share an instruction encoding space — the six NNA instructions ride alongside the MXUv3
opcodes in the MIPS SPECIAL2 opcode space — and the NNA is configured through one MXUv3 register, `vr31`. They are otherwise
separate hardware. MXUv3 is documented separately in `MXUV3.md` and `MXU3_OPCODE_TABLE.md`.

One further block appears in the vendor software and plays no part in convolution: **AIP** at
`0x12b00000`, an image-geometry engine for resize and affine transforms. The accelerator's own internal
operand memories are introduced in [§2.1](#21-the-memory-hierarchy).

**Naming.** Ingenic calls this accelerator generation **NNA2**; the T40's is NNA1. Version strings inside the
camera libraries read `mips@NNA2` and `T41@NNA2`.

### 1.4 Scope

This manual establishes:

- the instruction set — six coprocessor instructions, their operand encoding, and the array's response to
  each ([Chapter 12](#12-instruction-set));
- the configuration interface — about forty fields addressed as `bank | source word | field`, each with a
  known location, width and reset value; the sixteen-entry program table, its entry format, its lengths and
  what a reset does to it; the `vr31` word file that the array reads directly during a MAC
  ([Chapter 4](#4-the-configuration-model), [Chapters 13](#13-configuration-fields)–[14](#14-the-program-table));
- the operand feed — weights, activations and the requantization table are pushed from vector registers
  into positional slots through three kinds of port ([§3.4](#34-ports-and-positional-feeds));
- the DMA engine — descriptor format, kick and wait protocol, ORAM plans, ring and whole-input schedules
  ([Chapter 6](#6-memory-and-dma));
- the data formats — NDHWC32 feature maps, the two-plane 4-bit weight layout, the requantization table
  formats ([Chapter 2](#2-memory-and-data), [§3.8](#38-requantization-and-packing));
- the readout — raw int32 row-banks, the packed FIFO, the floating-point path, and the exact fixed-point
  formula the array's own requantization stage applies ([§3.7](#37-accumulators-commit-and-readout),
  [§3.8](#38-requantization-and-packing)).

What it does not establish — among them the isolated effect of several `nncmd` values, the meaning of the
`nnmac` immediate, what a reset preserves in the MAC stage, strides above 2, and the absolute size of a
walk step — is collected in one place, [Appendix C](#appendix-c--not-characterised). In the body, such
items are marked **Not characterised**.

## 2. Memory and data

The array reads operands only from its own internal memories, and those memories are small: they hold the
weights of one pass and a sliding window of activations, not a layer. Everything else — the tensors, the
weights of a network, the staging of results — lives outside and is moved in and out by the CPU and the
DMA engine. This chapter describes that hierarchy and the formats data takes at each level.

### 2.1 The memory hierarchy

Five stores separate a tensor in DDR from the accumulators, and the result returns through the readout
banks along the same path.

```
   DDR (128 MB)   mem=72M@0x0             Linux
                  rmem=40M@0x4800000      media pipeline (/dev/rmem); not used by the NNA
                  nmem=16M@0x7000000      NNA tensor pool, parsed from /proc/cmdline
                  + a driver-allocated weight pool
        │
        │  NNDMA  (descriptors in DESRAM, 32 KB at 0x12500000)
        ▼
   ORAM (384 KB on-chip SRAM at 0x12620000)
        weights of the current pass · feature rows · requantization table · output staging rows
        │
        │  MXUv3 vector loads                      ▲  MXUv3 vector stores
        ▼                                          │
   vector registers  vr0 … vr30                    vector registers
        │                                          ▲
        │  nndwr vrN,port                          │  nndrd vrN,bank
        ▼                                          │
   ┌────────────────────────── the NNA ──────────────────────────┐
   │  ports 0,2,4,7 ──► weight memory  (WRAM)                     │
   │  ports 0x20/0x21 ► feature memory (FRAM)   ──► MAC array     │
   │  port 0x40 ──────► table memory   (BTRAM)      │             │
   │                                                ▼             │
   │                              256 int32 accumulators (1 tile) │
   │                                                │  commit     │
   │                                                ▼             │
   │              readout bank 0 (raw int32)   readout bank 1 (packed FIFO)
   └──────────────────────────────────────────────────────────────┘
```

**DDR.** The kernel command line carves physical memory into three regions. `mem=` is what Linux manages;
`rmem=` is reserved for the camera's media pipeline and is not used by the accelerator; `nmem=` is the
accelerator's tensor pool, which the runtime finds by parsing `/proc/cmdline`. The driver additionally
allocates a DDR pool for weights ([§6.1](#61-physical-memory-map)). DDR is cacheable, and the accelerator
has no coherency with the CPU caches: every transfer is bracketed by an explicit cache flush or invalidate
([§7.3](#73-cache-coherency)).

**ORAM** is the accelerator's working memory: 384 KB of on-chip SRAM from `0x12620000`. Weights, feature
planes, the requantization table and output staging rows all pass through it. Its size is what makes a
layer's *plan* — which operands are resident and which are streamed — the central question of every recipe
([§6.2](#62-oram)). Absolute ORAM addresses are never given to the array; only DMA descriptors and MXUv3
loads and stores use them.

**NNDMA** moves bytes between DDR and ORAM under the control of descriptors written into DESRAM. It is the
only path between the two; the CPU does not copy tensors itself ([§6.3](#63-descriptor-format)).

**Vector registers.** The array has no view of memory. Every operand it receives is the content of an
MXUv3 vector register at the moment of an `nndwr`, and every result it returns lands in one at an
`nndrd`. The CPU therefore loads 64-byte slices from ORAM into `vr0`–`vr30`, pushes them, drains results
into registers, and stores them back to ORAM staging rows. `vr31` is special: it is the file of sixteen
word registers through which the array is configured ([§4.2](#42-the-word-register-file)).

**The internal memories.** Behind the ports sit the array's own operand stores, known by name from the
driver's error codes: **WRAM** (weights), **FRAM** (features) and **BTRAM** (the requantization table).
Their sizes are **not characterised**; the association of each port with one memory is by name only [L].
What is established is their behaviour: they hold the weights of one pass and a window of activations, they
are filled positionally by the ports ([§3.4](#34-ports-and-positional-feeds)), and the weights stay
resident while activations stream past them.

**Accumulators and readout banks.** The 256 accumulators hold exactly one tile. A commit moves the tile
into the readout banks and clears the accumulators; bank 0 keeps the raw int32 values, bank 1 the
requantized, packed tile ([§3.7](#37-accumulators-commit-and-readout)).

The path back is the mirror image: `nndrd` fills a vector register, an MXUv3 store writes it to an ORAM
staging row, and NNDMA's write channel returns the row to DDR. On the floating-point path the MXUv3
requantizes the drained int32 values to fp32 in between ([§3.8](#38-requantization-and-packing)).

### 2.2 Channel groups of 32

Every tensor the array touches is organised in groups of 32 channels. A group is a *plane*: 32 channels of
one pixel are contiguous in memory, and the next 32 channels form a separate plane elsewhere. Two counts
follow and appear throughout this manual:

```
D    = Cin  / 32      input channel groups
Dout = Cout / 32      output channel groups
```

The group size is the array's own: one MAC reduces over 32 input lanes and produces 32 output columns
([§1.1](#11-the-atomic-operation)), so 32 channels is the unit in which it consumes and produces data. A
push is 64 bytes because that is the width of an MXUv3 vector register, and the number 32 fixes what a push
carries. At 4-bit precision the 32 channels of one pixel occupy 16 bytes, so a 64-byte push carries **four
pixels** of one plane — one tile's width. At 8-bit precision a pixel-group is 32 bytes and a push carries two
pixels.

Channel counts are padded to a multiple of 32 in both directions:

- **Input padding** costs nothing: padded activations are zero and the corresponding weight codes are 0, so
  they contribute nothing to the sum.
- **Output padding** produces real results in the padded lanes. The packed paths write them out and the
  consumer ignores them; the floating-point head path trims them during copy-out, writing only the real
  `cout` channels per pixel.

The image layer is the one case where the *input* channel count is far below 32. Three colour channels are
presented to the array as a 32-entry activation vector holding 27 real taps and 5 zero pads
([§11.6](#116-the-rgb-image-layer)).

Reduction over more than 32 input channels is performed by streaming: one (feed, MAC) round per input group,
each with its own 32 × 32 weight tap, all accumulating onto the same accumulators.

### 2.3 NDHWC32 feature maps

Feature maps use a channel-blocked layout. The dimensions, outermost first, are **N** (batch), **D** =
`ceil(C/32)` (channel groups), **H**, **W**, **C** (the innermost 32 channels, contiguous per pixel). A
pixel's first 32 channels are adjacent in memory; the next 32 form a separate plane.

For an `H × W × C` map with `bits`-wide elements:

| Quantity | Size |
|---|---|
| One pixel, one channel group | `32 · bits / 8` bytes — 16 B at 4-bit, 32 B at 8-bit |
| One row of one plane | `W · 32 · bits / 8` bytes |
| One plane | `H ·` row bytes |
| Whole tensor | `D ·` plane bytes, planes consecutive |

The runtime's tensor descriptor stores the dimensions as `+0x00 N`, `+0x04 D`, `+0x08 H`, `+0x0c W`
([Appendix B](#appendix-b--vendor-executor-map)).

**Row padding.** Some vendor planners round the row length up to a multiple of 64 (`round64`) and others use
the tensor's own unrounded stride. The two agree whenever `W` is a multiple of 4, which holds for every
layer observed. A generator should follow the recipe it implements: the 3×3 recipes round, the 1×1 and
single-input-group recipes take the tensor stride as it stands.

### 2.4 Precision

The vendor runtime accepts, per operator, the following bit widths. This is a framework-level statement, but
for convolution it is the only vendor-stated capability envelope, and it agrees with what the configuration
fields encode.

| Operator | Input feature | Weights | Output | Other limits |
|---|---|---|---|---|
| Convolution | 2 / 4 / 8-bit | 2 / 4 / 6 / 8-bit (2 / 4 only with 2-bit input) | 2 / 4 / 8-bit or float | any kernel size; **stride ≤ 2**; dilation supported; no batch for 2-bit input |
| DepthWiseConvolution | 2 / 4 / 8-bit | 2 / 4-bit (8-bit input: 2 / 4 / 6 / 8) | 2 / 4-bit (8-bit input: 8-bit) | any kernel; stride arbitrary for 2 / 4-bit input, ≤ 2 for 8-bit |
| FullConnected | 2 / 4 / 8-bit | 2 / 4 / 8-bit | 2 / 4 / 8-bit or float | batch only for 8-bit |
| Pooling / Add / Mul / UpSample | 2 / 4 / 8 / 10 / 12-bit | — | same | Concat and Normalize: 8 / 10 / 12-bit only |

How the envelope maps onto the hardware:

- The four weight widths 2 / 4 / 6 / 8 are exactly the four codes of the precision field `A.1b`
  (`bits/2 − 1`), and the 768-byte weight tap is the 6-bit option ([§2.5](#25-weights)).
- BatchNorm, BiasAdd and the activation functions are "integrated into convolution" — they are the fused
  requantization tail of [§3.8](#38-requantization-and-packing), not separate operators.
- "Stride ≤ 2" is consistent with the stride field `A.1a`, which vendor software only ever sets to 0 or
  `0x11`. Stride 2 is established on hardware; larger strides are not characterised.
- 10- and 12-bit feature formats exist in the runtime for pooling, add and concat, so a feature dtype code
  wider than 8-bit is legitimate even though no such layer has been observed.

### 2.5 Weights

Weights are staged into ORAM as one contiguous blob and pushed onto the four weight ports. Within one
(input group, output group) block the kernel taps are consecutive, each a contiguous 32 × 32 matrix in
which the input channel varies fastest (`m = cout · 32 + cin`). Quantization is per output channel,
symmetric, zero point 0, with `weight_scale[co] = amax / 128`, where `amax` is the largest absolute weight
of the channel.

| Weight width | Bytes per 32 × 32 tap | Layout |
|---|---|---|
| 4-bit | 512 | Two 2-bit planes; see below |
| 6-bit | 768 | Packed variant (vendor precision code 8) |
| 8-bit | 1,024 | One byte per weight, `cout · 32 + cin` order |

**The 4-bit two-plane layout.** A 32-input × 32-output tap occupies 512 bytes as two 2-bit planes:

- bytes 0–255 hold the **low** bit-pair of every weight;
- bytes 256–511 hold the **high** bit-pair;
- within each plane the bit-pair index is `m = cout · 32 + cin`, living in byte `m / 4` at bits
  `2·(m mod 4) + 1 : 2·(m mod 4)`;
- the weight is offset-binary: `w = (lo | hi << 2) − 8`, so the sixteen codes span −8 … +7.

The array multiplies with these values directly. The per-channel scale that converts codes to real numbers
is applied by the requantization stage, not in the MAC.

Where a tap has fewer than 32 real input channels, the unused positions carry code 0 (= −8) and are
multiplied by zero activations, contributing nothing.

**Tap order.** For a multi-tap kernel the taps of one (input group, output group) block are pushed in a
fixed order, and that order is not raster: **the middle kernel row is taken in reverse column order**,

```
tap index = 3 * row + (row == 1 ? 2 - col : col)
```

The order is dictated by the sequence in which the walk program presents input rows to the array; why the
middle row runs backwards is not characterised [L]. The blob is ordered output-group-major, so a pass
covering output groups `[k·gpp, (k+1)·gpp)` pushes the corresponding contiguous slice.

**Blobs are used verbatim.** Neither the weights nor the requantization table are repacked or rebuilt at
load time. A model file's weight and table bytes are staged into memory as they stand and DMA'd to ORAM
unchanged.

### 2.6 Output formats

The output format is selected by the readout path ([§3.8](#38-requantization-and-packing)):

| Output | Bytes per pixel per group | Produced by |
|---|---|---|
| Packed 4-bit | 16 | The array's own requantization stage |
| Packed 8-bit | 32 | The array's own requantization stage |
| fp32 NDHWC32 | 128 | Software requantization on MXUv3 |
| fp32 NHWC, real channel count | `4 · cout` | Software requantization plus a copy-out that trims padding |
| uint8 with zero point, or integer-rescaled int8/int16 | 32 | Other vendor readout paths, not required for convolution |

## 3. How work is organised

A convolution is a large matrix multiplication. For a 1×1 layer with `Cin` input and `Cout` output channels
over `H × W` pixels it is exactly `[H·W × Cin] · [Cin × Cout]`. A `K × K` layer is the same thing repeated
over `K²` spatial taps, accumulating. This chapter describes how the array divides that multiplication into
tiles, how the two execution units share a tile, how operands are fed, how the walk program steers a MAC
over them, and how the result comes out.

### 3.1 The tile

The unit of work is a **tile**:

```
2 image rows  ×  4 consecutive pixels  ×  32 output channels  =  256 int32 results
```

```
                 one column block (4 pixels)
              ┌──────┬──────┬──────┬──────┐
   row  r     │  p0  │  p1  │  p2  │  p3  │     each position: 32 output channels
              ├──────┼──────┼──────┼──────┤                  = 32 int32 accumulators
   row  r+1   │  p4  │  p5  │  p6  │  p7  │
              └──────┴──────┴──────┴──────┘
   8 positions × 32 channels = 256 accumulators

   read out as:  bank 0 — 16 row-banks of 16 int32
                          row-bank R = channels [16·(R mod 2), +16) of position R/2
                 bank 1 — the packed tile: two 64-byte reads at 4-bit output, four at 8-bit
```

One `nncmd` commit publishes exactly one tile. Software advances the tile coordinates; the array walks the
kernel taps and the input channel groups within a tile by itself, following the loaded walk program
([§3.5](#35-the-walk-program)). The tile's shape is fixed: no configuration field changes it. What the
input element width changes is only the tile's size in bytes — 128 bytes at 4-bit input, 256 at 8-bit —
which is what the tile-class field `B.15` encodes.

### 3.2 Marching a layer

A layer is covered as `ceil(H/2)` **row pairs** × `ceil(W/4)` **column blocks** × `Dout` output groups.
Column blocks and row pairs are software's loop; output groups within a block are a sequence of tiles that
reuse the same activation feed. The recipes name these loop variables `RP` (row pair), `CB` (column block)
and the constants `D`, `Dout`.

The single-input-group recipes produce four output rows per step of the outer loop — a **row quad** —
because their two MAC phases are given different row windows and each phase is committed as its own tile
([§3.3](#33-the-two-execution-units)).

### 3.3 The two execution units

The array has two execution units. In the general case **they split the input channel groups, not the
output rows or the pixels.** Unit A takes groups `0 … nA−1` and unit B takes `nA … D−1`, with

```
nA = D / 2        nB = D - nA
```

Each unit has:

| Per unit | Unit A | Unit B |
|---|---|---|
| Group count field | `A.1c` = `nA − 1` | `A.1d` = `nB − 1` |
| Activation port | `0x20` | `0x21` |
| Operand start offset | `B.10` | `B.11` |
| Program slot | entries 0 … 7 | entries 8 … 15 |
| MAC phase | the first `nnmac` of a tile | the second `nnmac` |

Both units contribute to the same tile: a tile's result is complete only after both MAC phases have run.
The program table holds one program per unit ([§3.5](#35-the-walk-program)), and `B.1a` carries both
lengths as `(len_B << 4) | len_A`.

The group-count fields are four bits wide, so each unit reduces over at most 16 input groups — 512 input
channels — per pass, and a layer with more than 32 input groups is run in several passes
([Chapter 9](#9-choosing-a-recipe)).

How the two units are used when a layer has a **single input group** is recipe-defined, because there is
nothing to split:

- the *single-input-group* recipe ([§11.3](#113-single-input-group-row-quad)) feeds one group on port
  `0x20` only and runs the two MAC phases over different input row windows, each phase committed as its
  own tile, so the first phase produces output rows `4q, 4q+1` and the second rows `4q+2, 4q+3` of a row
  quad;
- the 3×3 recipe's single-group variant ([§11.4](#114-single-input-group-two-slot)) gives each unit its own
  row window on its own port — unit A (`0x20`) rows `4q, 4q+1`, unit B (`0x21`) rows `4q+2, 4q+3` — with the
  same program in both slots;
- the image layer ([§11.6](#116-the-rgb-image-layer)) produces one output row per unit;
- the floating-point heads ([§11.7](#117-floating-point-output-heads)) split by the general rule, which
  with `D = 1` gives unit A no groups (`nA = D/2 = 0`); unit A is skipped and unit B carries the group.

The general int8 executor ([§11.7](#117-floating-point-output-heads)) does not split by port at all: both
units' operands enter through port `0x20`, and which unit a feed belongs to is selected by the MAC phase
that follows it and by the start offset written to `B.10` before it. Port and unit are therefore associated
by convention of the 4-bit recipes, not by hardware necessity.

### 3.4 Ports and positional feeds

Operands enter the array through **ports**. Vendor software uses seven.

| Port | Operand kind | One push carries |
|---|---|---|
| `0`, `2`, `4`, `7` | Weights | One quarter of a 256-byte weight slice, pushed in that port order |
| `0x20` | Activations, unit A | One 64-byte activation window |
| `0x21` | Activations, unit B | The same, for the second unit |
| `0x40` | Requantization table | 64 bytes of table |

`nndwr vrN,port` writes the 64 bytes of `vrN` into the named port. **No address travels with the
instruction.** Each port has an internal write pointer that advances one 64-byte slot per push; which
slots a MAC then reads is determined by the walk program. The destination is therefore *positional*, and the contract software must satisfy is the
**ordered sequence and count of pushes per port** — not a set of addresses.

The consequences are sharp:

- Dropping one push shifts and corrupts the entire remaining output, because every later push lands in the
  wrong slot.
- Feeding correct data in the wrong order produces garbage, not a permuted result.
- The bytes pushed are the *named register's*. They need not have been loaded from memory: the image layer
  pushes registers a shuffle network has just written ([§11.6](#116-the-rgb-image-layer)).
- The general-purpose base-register field of `nndwr` and `nnmac` ([§12.2](#122-operand-encoding)) is not
  read by the hardware; setting it to zero everywhere changes nothing.

### 3.5 The walk program

The walk program is the schedule the array follows while a MAC runs: it says where each successive
operand comes from. It is a 16-entry table, loaded before the arm and left alone thereafter.

| Field | Role |
|---|---|
| `B.18` | Push pointer, four bits, wraps at 16. Written directly to position the next push |
| `B.19` | Push. Each write stores a word at the pointer and increments it |
| `B.1a` | Program lengths, `(len_B << 4) \| len_A`, three bits each, 0 meaning 8 |

Slot A runs from entry 0 and slot B from entry 8; each slot holds up to eight entries. An entry is fifteen
bits — bit 15 of a pushed word is dropped:

```
entry = count[14:11] | operand[10:0]
```

`count` says what kind of entry it is:

- **Step** (`count` 1–14): *consume the operand window at the walk position, then advance the position
  by `step`* — `count` times over. `step` is the **signed 10-bit** value in bits `[9:0]`; bit 10 is ignored.
  The position indexes the windows that have been fed to the unit's port; the absolute size of one step is
  not characterised ([Chapter 14](#14-the-program-table)).
- **Move** (`count` = 0): advance once by `step`, consuming nothing. This is how a program repositions for
  the next output column without reading an operand it does not want.
- **Loop** (`count` = 15): repeat the entries between this one and `end_entry` (exclusive), `repeats` times.
  `operand = end_entry[10:8] | (repeats − 1)[7:0]`. `end_entry` is counted from the slot's first entry, so
  the same loop word serves slot A at entry 0 and slot B at entry 8 [L]. Loops do not nest.

A program is therefore a small loop over the walk, and the same schedule can be written several ways: one
step of count 3 is exactly three steps of count 1, as long as any enclosing loop's `end_entry` moves with
them. Both forms are equivalent ([Chapter 14](#14-the-program-table)).

**Mask a negative step into the operand field.** An unmasked negative integer sets bits `[14:11]` as well,
which turns the entry into a *loop* and derails the whole walk. Program words travel inside configuration
records, so neither disassembly nor instruction-stream comparison can detect that — only running the
layer can.

### 3.6 MAC phases and the walk position

`nnmac vwN,imm` runs one multiply-accumulate pass, walking the taps and channel groups the program
describes, and **adds** into the accumulators. Two MACs do not equal one; nothing but a commit clears the
accumulators.

The instruction names a `vr31` word register, and that register is not decoration. **It holds the operand
walk position for that MAC phase.** Every recipe initialises it before the arm, steps it in software as
the walk advances, and never commits it to a field with `nnrwr`. The word each recipe uses for each phase
is tabulated in [Chapter 15](#15-field-values-by-recipe).

Two regularities hold across all of them. Where the two units split input groups, the second phase's word is
the first's plus four times the second unit's start offset — `B.11`, or in the head recipe the value written
to `B.10` before the second phase — which counts in units of four walk steps. And a phase whose walk is performed in software steps its word, while a phase whose walk is in the
program leaves it alone.

The meaning of the 2-bit immediate is not characterised. In the general int8 family it is 1 on every inner chunk of a phase and 3
on the last; the 4-bit families do not follow that rule.

**The word file is live state.** Because the array samples `vr31` during a MAC, a word an `nnmac` names must
not be reused as scratch for anything else. Staging a configuration record into such a word produces a
byte-perfect instruction stream and a completely wrong tensor.

### 3.7 Accumulators, commit and readout

Accumulators are int32 and hold one tile: 8 pixels × 32 channels. A commit ends accumulation, clears the
accumulators, and publishes the tile to one or both readout banks.

| Bank | `nndrd` immediate | Contents | Access |
|---|---|---|---|
| 0 | `R`, 0 … 15 | Raw int32. Row-bank `R` holds 16 int32 = channels `[16·(R mod 2), +16)` of pixel `R/2` | Random-access, non-destructive; the same row-bank may be read repeatedly |
| 1 | `0x20` | The requantized, packed tile | FIFO: each read pops the next 64 bytes |

**Bank 1 runs behind the write point** by `B.07 × 64` bytes — one tile at the usual setting, where `B.07`
is the packed tile size in 64-byte units. A drain therefore returns the *previous* tile, and software must
either absorb that lag inside its loop or issue one discarded drain at the start of a run and one extra at
the end. Settings of `B.07` below 2 behave as 2 [L].

At 4-bit output one tile row (4 pixels × 32 channels) is one 64-byte read, so a tile is two reads; at 8-bit
output it is four.

### 3.8 Requantization and packing

Every convolution carries one 256-byte requantization table per 32-channel output group, stored in the model
file after the weights and consumed verbatim. Two table formats are used by the recipes in this manual,
selected by the readout path; a third, a 384-byte two-slope form, belongs to an alternative im2col
procedure ([§11.8](#118-large-weights-and-channel-concatenation)) and is described there. Which of the
first two applies is given by the layer's mode code in the vendor descriptor
([Appendix B](#appendix-b--vendor-executor-map)): 1 selects the floating-point path, 2 the array's own
packed path.

#### 3.8.1 Common structure

Both formats are 256 bytes per output channel group, holding two 32-entry arrays:

```
+0x00   int32 bias[32]        both formats
+0x80   scale[32]             fp32 in format 1, int32 in format 2
```

Entries are channel-linear within the group. Only the real `cout` entries are non-zero; padded output
channels hold zero. What the pack stage produces for a padded channel whose `mult` is 0 is not
characterised; the consumer ignores those lanes ([§2.2](#22-channel-groups-of-32)).

#### 3.8.2 The floating-point path

Used by the detection heads and the general int8 executor. The array commits raw int32 accumulators to
readout bank 0; software drains all sixteen row-banks and requantizes on MXUv3.

```
x = acc << shift                    shift: a per-layer constant from the vendor descriptor (1 on the heads)
x = x + bias[co]                    int32, table +0x00
y = (float) x
y = y * scale[co]                   fp32,  table +0x80
y = max(y, floor)                   -FLT_MAX for no activation clamp, 0 for a ReLU floor
```

The MXUv3 sequence is `sllw`, `addw`, `ffsiw`, `fmulw`, `fmaxw`, then a vector store. `scale[co]` is
`in_scale · weight_scale[co] / out_scale`. The result is IEEE single precision with one rounding, at the
multiply.

#### 3.8.3 The hardware packed path

Used by every 4-bit and 8-bit output layer. The table is pushed into the array through the table port and
the pack stage applies it when a tile is committed with `nncmd 0x8b` or `0x8f`. The second array is `int32
mult[32]`, holding small positive integers.

**Forward formula:**

```
packed = clamp( floor( (4 * acc + bias[co]) / mult[co] ) + 2^(out_bits-1),  0,  2^out_bits - 1 )
```

Three properties of this expression matter in practice:

- **`mult` is a divisor.** One output step is `mult / 4` accumulator counts. The word at `+0x80` is
  therefore the *reciprocal* of what the floating-point format stores at the same offset: format 1 holds
  `in_scale·w_scale/out_scale`, format 2 holds `4·out_scale/(in_scale·w_scale)`.
- **The division rounds toward −∞.** Floor, not truncation.
- **The offset is `2^(out_bits−1)`**: 8 at 4-bit output, 128 at 8-bit. With `bias = 0` an accumulator of 0
  maps to mid-scale.

The clamp rails are those of the packed type, and both are reachable.

The table holds one 256-byte block per output group. Which block the pack stage applies to a commit is
selected by `B.02`, in units of four 64-byte pushes; a recipe that leaves `B.02` at 0 and pushes only the
groups of the current pass relies on the stage advancing one block per commit [L].

Two pack-stage controls sit alongside the formula. `B.1e` shifts the table lookup, signed, with the two
16-channel halves of a pixel served separately — a half that lands outside its table saturates while the
other reads a neighbouring channel's entry. `B.05` bit 3 selects an alternative interpretation of the same
table. Not characterised: what that interpretation is. What is established: no layer observed sets the bit.

#### 3.8.4 Generating a table

To realise `out_q = clamp(round(acc · S + Z))` with `S = in_scale · w_scale[co] / out_scale` and output zero
point `Z`, invert the forward formula:

```
mult[co] = round( 4 / S )
         = round( 4 * out_scale / (in_scale * w_scale[co]) )

bias[co] = round( mult[co] * ( Z - 2^(out_bits-1) + real_bias[co] / out_scale )  +  mult[co] / 2 )
```

The trailing `mult[co] / 2` converts the hardware's floor into round-to-nearest. Choosing `S` and `Z` is a
quantization question; a table so constructed produces exactly the value the forward formula names.

Practical limits: `mult` must be positive, and the achievable scale resolution is `4/mult`, so very large
scales lose precision as `mult` approaches 1. Values observed in production models range from 38 to 688.

#### 3.8.5 Worked example — 4-bit output

Layer parameters: 4-bit activations, weights and output; output channel 4 of the first group has
`bias = −238` and `mult = 40`. One output step is `mult / 4 = 10` accumulator counts.

| Accumulator | `4·acc + bias` | `/ mult`, floored | `+ 8` | Packed |
|---|---|---|---|---|
| −11 | −282 | −8 | 0 | 0 |
| −10 | −278 | −7 | 1 | 1 |
| 0 | −238 | −6 | 2 | 2 |
| 20 | −158 | −4 | 4 | 4 |
| 55 | −18 | −1 | 7 | 7 |

The transfer function for this channel steps at accumulator −10, 0, 10, 20, 30, …, exactly `mult/4` apart,
which is the signature of the division.

#### 3.8.6 Worked example — 8-bit output

Layer parameters: 4-bit activations, 8-bit weights, 8-bit output. Samples from the array's own accumulators
and its own packed output in the same run:

| Accumulator | `bias` | `mult` | `floor((4·acc + bias)/mult) + 128` | Packed |
|---|---|---|---|---|
| −1,765 | −35,344 | 378 | 15 | 15 |
| −1,410 | −24,663 | 240 | 1 | 1 |
| 165 | −33,798 | 260 | 0 | 0 |
| −2,590 | −18,389 | 190 | −24 → clamped | 0 |

The bias magnitudes are larger here because `mult` is larger. The ratio `bias/mult` — the packed value an
accumulator of zero produces before the offset of 128 is added — lies between −94 and −130 in this layer,
placing the layer's zero in the lower part of the packed range.

#### 3.8.7 Interpreting a vendor table

For a table taken from a model file, `bias[co] / mult[co]` gives the output value, in packed units, that an
accumulator of zero produces, offset by `2^(out_bits−1)`. In production 4-bit layers this ratio sits between
−5.5 and −8, placing zero at or slightly above the bottom of the range: the output type is unsigned with
zero point 0, and the low clamp performs the layer's non-negativity. In such layers a large share of the
results sit on the low clamp.

### 3.9 The instruction order of one layer

The six instructions of [§1.2](#12-the-six-instructions) are issued in the following order within a layer.
One recipe pushes its weights before the arm, and whether the drain of the previous tile precedes or
follows the commit is fixed per recipe; otherwise the order is invariant. This sequence is the spine of
Part II; each step is expanded in [Chapter 5](#5-running-a-layer).

```
nnrwr B.1b ; nncmd 0x00              reset, guarded          (5.1)
nnrwr x N                            static configuration    (5.2)
nnrwr B.18 / B.19 / B.1a             walk program            (5.3)
nncmd 0x60                           arm                     (5.4)
nnrwr B.10 / B.11                    unit start offsets, after the arm
    nndwr port 0/2/4/7               weight preload          (5.5)
    nndwr port 0x40                  requantization table
    for each tile:
        nnrwr per-tile fields                                (5.2)
        nndwr port 0x20 / 0x21       activation feed         (5.6)
        nnmac                        MAC phases              (5.7)
        nndrd bank 1                 drain the previous tile (5.9)
        nnrwr B.02 ; nncmd 0x84/0x8b commit                  (5.8)
```

The array latches static configuration at the arm. Per-tile fields are read as the tile runs. An arm leaves
the walk program untouched; a reset keeps each entry's operand but rewrites its count to 1, so the program
is reloaded after every reset ([Chapter 14](#14-the-program-table)).

---
# Part II — Programming the NNA

Part II is the procedure. Chapter 4 gives the configuration model — how a field is addressed and written
and how the array reads the word file. Chapter 5 expands the instruction order of §3.9 into the steps of one
layer. Chapter 6 covers the memory map and the DMA engine that stage operands, Chapter 7 the hazards that
produce silent wrong results, and Chapter 8 runs one convolution end to end.

## 4. The configuration model

The NNA is configured through 16-bit fields, and every field value reaches the hardware by one path: the
value is placed in a word register and one `nnrwr` copies it into the field. This chapter defines that path —
how fields are named, where values are staged, which fields are read once and which while a tile runs, the
record idiom for writing them in bulk, and how committed state is read back. The field-by-field reference is
[§13](#13-configuration-fields); the procedure that uses this model is [§5](#5-running-a-layer).

### 4.1 Banks and fields

Configuration fields live in two banks:

| bank | contents | readback block (§4.5) |
|---|---|---|
| **A** | geometry: tensor dimensions, strides, coordinates, element widths | G |
| **B** | mode and activation: MAC mode, requantisation, program lengths, counters | H0 |

Each bank holds up to 32 fields, indexed 0…31. This manual names a field by bank and hexadecimal index:
`A.0c`, `B.15`.

`nnrwr vwN,imm` writes the low 16 bits of word register `wN` (§4.2) into the field named by `imm`. The
10-bit immediate is `bank | field`:

| immediate bits | meaning |
|---|---|
| bit 6 (`0x40`) | bank **B** |
| bit 5 (`0x20`) | bank **A** |
| bits [4:0] | the field index inside the bank |

So `A.0c` is written by `nnrwr vwN,0x2c` and `B.15` by `nnrwr vwN,0x55`, for any word `N`. The word is a
separate register operand, not part of the field's identity ([§12.2](#122-operand-encoding)).

Vendor listings show `nnrwr` with a single merged payload — instruction-word bits [20:6] — in which the same
information appears as `bank | (word << 5) | field`, with bit 11 selecting bank B, bit 10 bank A, bits [9:5]
the word number and bits [4:0] the field index. The two forms are the same bits: the instruction-field
immediate is the payload with the word bits removed (`0x40c`, `0x42c`, `0x56c` are all `A.0c`, fed from `w0`,
`w1` and `w11`). [Appendix D](#appendix-d--register-number-cross-reference) converts every payload number
that appears in vendor code to its field. The apparently distinct "register families" that different vendor
executors use are therefore the same fields staged from different words.

### 4.2 The word register file

`vr31` doubles as a file of sixteen 32-bit **word registers** `w0` … `w15` — the operand class the Ingenic
assembler calls VWR. Every configuration value passes through this file, and the NNA also reads it directly
while a MAC runs. The file is addressed in two ways:

- As `vr31`, by the ordinary vector loads and stores: `law vr31,(rs),k` loads one word into `wk`, and a
  64-byte record is loaded with one 64-byte vector load or as two 32-byte halves, the low half into
  `w0`–`w7` and the high half into `w8`–`w15`. The MXUv3 load and store forms are documented in `MXUV3.md`.
- As individual words, by the word-arithmetic instructions: `liwr wA,simm10` sets a word to a signed 10-bit
  immediate; `addiw wA,wB,simm8` computes `wA = wB + imm`; `addrw wA,wB` adds two words; `mfcpuw wA,rs` and
  `mtcpuw rt,wB` move a word to and from a general-purpose register.

Two NNA instructions name a word by number:

- `nnrwr vwN,imm` reads the low 16 bits of `wN` when it executes and commits them to a field (§4.1).
- `nnmac vwN,imm` names a word the array reads *live* during the MAC: it holds the walk position
  ([§3.6](#36-mac-phases-and-the-walk-position)), and its value at each moment of the pass is what the
  hardware uses. A word named by an `nnmac` is an operand of the MAC, not a staging slot.

`nncmd 0x00` clears the word file; `nncmd 0x60` (the arm) does not.

Because the array reads the file live, words that no `nnrwr` ever commits can still determine the result.
Each executor family keeps its own words stable for the whole run: the 4-bit 1×1 executor holds `w2`–`w5`
(`w4` and `w5` are the words its two `nnmac`s name; `w2` and `w3` hold the values it commits to `B.10` and
`B.11` after the arm), and the output is wrong in every pixel if any of the four is disturbed. The 4-bit 3×3
executor names `w9` and `w10` in its MACs, advances both with `addrw w9,w5 ; addrw w10,w5` after every
column block, and loads `w6`, `w13` and `w14` after the arm without ever committing them; the output is
wrong everywhere if they are not loaded.

Not characterised: which words each executor family's MACs read beyond the ones the `nnmac` operands name,
and what `w6`, `w13` and `w14` contribute in the 3×3 executor. What is established: the result depends on
them, and the safe rule is to load every word the vendor's prologue loads, step the ones it steps, and zero
the rest. [§7.9](#79-the-word-file-is-live-state) develops this rule.

### 4.3 Static and per-tile fields

Fields divide into two classes by *when* the hardware reads them.

- **Static fields** are latched at the arm (`nncmd 0x60`). They describe the layer as a whole — tensor
  geometry, element widths, MAC mode, requantisation parameters, program lengths — and are written once per
  layer, before the arm.
- **Per-tile fields** are read as a tile runs. They carry the position of the current pass — the signed
  input row and column coordinates `A.09` and `A.0b`, the start offsets `B.10` and `B.11` — and are rewritten
  between passes, after the arm and before each MAC.

[§13](#13-configuration-fields) identifies the class of every field. A per-tile field is typically stepped
rather than restaged: its word is advanced with `addiw` or `addrw` and the field rewritten from the same
word. The vendor idiom

```
addiw w4,w4,1         ; counter += 1
nnrwr vw4,0x50        ; B.10 <- w4
```

is exactly that; [§3.9](#39-the-instruction-order-of-one-layer) fixes where these writes fall relative to
the arm and the MACs.

### 4.4 Configuration records

Static fields are written in bulk. The idiom is to build a 64-byte **configuration record** of sixteen
32-bit words in memory, load it into `vr31` with one vector load, and then issue one `nnrwr` per field, each
naming the word that holds its value:

```c
static nna_config_t config;                   /* static storage, not an automatic */
nna_config_clear(&config);
config.word[NNA_W0] = height - 1;
config.word[NNA_W1] = width  - 1;
nna_config_stage(&config);                    /* one vector load into vr31       */
NNA_WRITE_FIELD(NNA_IN_H_M1, NNA_W0);         /* nnrwr vw0,0x2c  (A.0c)          */
NNA_WRITE_FIELD(NNA_IN_W_M1, NNA_W1);         /* nnrwr vw1,0x2e  (A.0e)          */
```

Only the low 16 bits of each word are transferred. A layer with more static fields than words stages several
records in turn, each followed by its burst of `nnrwr`. Four rules constrain the idiom.

- **Field immediates are instruction constants.** They cannot be computed at run time. A helper that writes an
  arbitrary field must be a macro or a `switch`, not a function taking a field number.
- **A record's zeroed words are as deliberate as its written ones.** The array reads the word file during a
  MAC (§4.2), so whatever a record leaves in `vr31` is live machine state.
- **Words a MAC names are reserved.** A word that appears as an `nnmac` operand — or that an executor keeps
  live for its MACs — must not be used to stage configuration once the layer is armed
  ([§3.6](#36-mac-phases-and-the-walk-position)). Records for static fields are staged before the arm, or in
  words the executor family does not use.
- **The record must be 64-byte aligned, and a local variable is not enough.** It is loaded with one 64-byte
  vector load. A compiler asked for an over-aligned automatic may place it in the frame without realigning
  the stack; give the record static storage.

### 4.5 Configuration readback

`nnrrd vrN,imm` copies one 64-byte block of committed state into vector register `vrN`. The 7-bit
immediate selects the block; the destination register is a separate operand. Four selectors return data:

| `nnrrd` immediate | block | contents |
|---|---|---|
| bit 5 set (`0x20`; bits [4:0] ignored) | **G** | bank-A field file |
| bit 6 set, bit 0 clear (`0x40`) | **H0** | bank-B field file |
| bit 6 set, bit 0 set (`0x41`) | **H1** | the program table ([§14](#14-the-program-table)) |
| bits 6 and 5 clear, bits [1:0] = 0 (`0x00`) | **L0** | status |
| otherwise | — | zeros |

In the merged-payload form of vendor listings the selector occupies payload bits [11:5] and the destination
register bits [4:0]; the instruction-field immediate is `payload >> 5`. G and H0 return the fields as
committed and are the reference for checking a configuration burst; H1 returns the program-table entries.

The status block L0 is read as sixteen words. Its `w0` bit 0 is set once any field has been written since
reset. The other non-zero words at reset are:

| L0 word | value at reset |
|---|---|
| `w2` | `0x000000ff` |
| `w4` | `0x00010000` |
| `w5` | `0x00000009` |
| `w6` | `0x00010000` |
| `w7` | `0x0000000a` |

Not characterised: the meaning of L0 words `w2`, `w4`, `w5`, `w6` and `w7`. What is established: the values
above are present at reset and `w0` bit 0 records that a write has occurred
([Appendix C](#appendix-c--not-characterised)).

**Reading the result.** A field write is visible in the very next `nnrrd`, before any arm. The vector
register an `nnrrd` fills must be stored to memory with the 32-byte half-store form (two stores; the
instruction family encoded `0x710000d5`, `SA0_VPR`), not the 64-byte full-store form (`0x7120d8d5`), which
returns stale data. A hang-safe minimal sequence is `sync ; nnrrd vr12,block ; sync ; store`.
## 5. Running a layer

This chapter expands the instruction order of [§3.9](#39-the-instruction-order-of-one-layer) step by step:
each section takes one step, gives the instructions it consists of, and states the constraints that bind it
to its neighbours. The recipes of Part III ([§11](#11-the-recipes)) are instances of this sequence, differing
only in how many times each step is performed and with what operands.

### 5.1 Reset

```
nnrwr B.1b, w      with w = 0
nncmd 0x00
```

`nncmd 0x00` returns every configuration field to its reset value, clears the `vr31` word file, and rewinds
the program table's push pointer to entry 0. It does not restore the program table to a fixed pattern: each
entry's count field is rewritten to 1 and its 11-bit operand is left unchanged, so the walk program must be
reloaded after every reset. MACs issued after a reset produce zeros until configuration, program and arm
have been redone.

**`B.1b` must be written immediately before the reset.** The reset does not fully re-initialise the MAC
stage: what the weight preload after it sees otherwise depends on the configuration in force before it. The
two instructions form a unit and belong at the start of every run. [§7.1](#71-the-b1b-reset-hazard)
describes the failure mode.

### 5.2 Field configuration

Values reach configuration fields through the `vr31` word file ([§4.2](#42-the-word-register-file)). The
idiom is to build a 64-byte record of sixteen 32-bit words, load it into `vr31` with one vector load, and
then issue one `nnrwr` per field naming the word that holds its value:

The record idiom — one vector load into `vr31`, one `nnrwr` per field naming its word — and the rules that
constrain it (field immediates are instruction constants; zeroed words are live state; words a MAC names
are reserved; the record must be 64-byte aligned in static storage) are given in
[§4.4](#44-configuration-records); the immediate encoding is [§4.1](#41-banks-and-fields).

Fields divide into those latched at the arm and those read as a tile runs; [§13](#13-configuration-fields)
identifies which is which. Per-tile fields are typically stepped with `addiw` or `addrw` on the word and
re-written, rather than restaged.

**Describing a layer instead of writing registers.** The record above is the mechanism; an implementation
does not have to spell it out for every field. The usual arrangement is a descriptor with one member per
field, initialised so that every field is left at its reset value, from which an apply step derives the
encodings and emits the writes. Two rules follow from the constraints above and hold for any such
implementation. The apply step owns the whole word file while it runs and leaves it in a known state, so
the words that must be live during the run are installed **after** it, never before. And it neither loads
the program nor arms, because the program's shape depends on the recipe and because one recipe pushes its
weights between the configuration and the arm.

### 5.3 Program loading

Load the walk program after the static fields and before the arm.

```
nnrwr B.18, w      w = 0            position at entry 0, slot A
nnrwr B.19, w      × len_A          push slot A's words
nnrwr B.18, w      w = 8            position at entry 8, slot B
nnrwr B.19, w      × len_B          push slot B's words
nnrwr B.1a, w      w = (len_B << 4) | len_A
```

Slot B must start at entry 8, and each length in `B.1a` must equal the number of words actually pushed into
that slot: a length is a real terminator, and declaring one entry more than was pushed corrupts every tile.
Each length field is three bits and encodes 1…8 with **0 meaning 8**, so a full eight-entry slot is written
as 0. Recipes that load only slot A write `len_B = 0`; the length pair is `(0 << 4) | len_A`. Not
characterised: how a slot declared 0 (which stores as 8) and never pushed is treated when the second unit
runs. What is established: every single-slot recipe writes `len_B = 0`, never pushes slot B, and is exact —
including one that runs the unit-B MAC. A program is required: without one the NNA runs and every output
pixel is wrong. The entry format is in [§14](#14-the-program-table); the program's role in the walk is
described in [§3.5](#35-the-walk-program).

### 5.4 Arm

```
nncmd 0x60
```

The arm latches static configuration and resets the accumulator and readout bank pointers. It leaves the
program table and its pointer untouched. Skipping it places results in the wrong half of the readout.

Two fields are written **after** the arm, not before: `B.10` and `B.11`, the operand start offsets of unit A
and unit B ([§3.3](#33-the-two-execution-units)). The words that the MAC phases name are also initialised at
this point ([§3.6](#36-mac-phases-and-the-walk-position)).

### 5.5 Weight preload

Weights are pushed 256 bytes at a time, as four 64-byte writes to ports `0`, `2`, `4` and `7` in that order
([§3.4](#34-ports-and-positional-feeds)):

```c
NNA_VLD64(NNA_VR0, slice +   0);  NNA_NNDWR(NNA_VR0, NNA_PORT_WEIGHT0);
NNA_VLD64(NNA_VR1, slice +  64);  NNA_NNDWR(NNA_VR1, NNA_PORT_WEIGHT1);
NNA_VLD64(NNA_VR2, slice + 128);  NNA_NNDWR(NNA_VR2, NNA_PORT_WEIGHT2);
NNA_VLD64(NNA_VR3, slice + 192);  NNA_NNDWR(NNA_VR3, NNA_PORT_WEIGHT3);
```

A 4-bit tap (512 bytes) is two such rounds; an 8-bit tap (1,024 bytes) is four. The weight DMA must have
completed before the first push — completed, not merely fenced
([§7.4](#74-fencing-and-waiting-are-different)).

For a multi-tap kernel the taps of one (input group, output group) block are pushed in the fixed, non-raster
tap order given in [§2.5](#25-weights). The blob is ordered output-group-major, so a pass covering output
groups `[k·gpp, (k+1)·gpp)` pushes the corresponding contiguous slice.

The requantization table is pushed through port `0x40`, 64 bytes per push, four pushes per output channel
group. The 3×3 recipe pushes the whole table once per layer and steps the commit window `B.02` across
passes; the 1×1 recipe pushes each pass's `groups × 4` entries at the start of the pass and keeps
`B.02 = 0`.

### 5.6 Activation feed

Activations are pushed as 64-byte windows to port `0x20` for unit A and `0x21` for unit B. What a window
contains depends on the geometry:

| Case | One 64-byte window |
|---|---|
| 4-bit input | 4 pixels × 32 channels of one row |
| 8-bit input | 2 pixels × 32 channels; a tile row is two windows and a tile four windows per plane |
| Image layer | 2 pixels × 32 assembled taps, built on MXUv3 |

**Lead.** The feed runs ahead of the MACs by a fixed number of column blocks, the *lead*. A recipe pre-feeds
`lead` column blocks before the first MAC and thereafter pushes one block's worth after each MAC, so the
NNA always has operands staged for the block it is about to compute. `lead` is 2 for a stride-1 3×3, 1 for
a padded stride-2 3×3, and 2 again at stride 2 when the left pad is 0. It also fixes the earliest point in
a row pair at which the next row pair's feature kick may be issued.

**Halo.** A 3×3 needs input rows and columns beyond the tile's own footprint, the *halo*. A tile of two
output rows at stride 1 spans four input rows: the two it is centred on plus one above and one below, and
over one column block the whole 4-row window of every input group is pushed. Horizontally the walk program
re-reads columns already fed, so the overlap between neighbouring column blocks is not pushed again within
a row pair. Vertically there is no such reuse: the input rows a row pair shares with the next are fed again
for that next row pair.

Where the number of pushes per block does not divide evenly among the tiles of that block, the recipe builds
a per-unit list of window addresses and consumes `ceil(len / groups)` entries after each MAC, rounded up to
an even number because the feed routine pushes two windows per call. This is what allows `D ≠ Dout`.

### 5.7 MAC

One `nnmac` per execution unit per tile:

```
nnmac vwA,imm      unit A's phase
nnmac vwB,imm      unit B's phase
```

The word registers and immediates per recipe are tabulated in [Chapter 15](#15-field-values-by-recipe).
MACs accumulate; nothing but a commit clears the accumulators.

Immediately before the unit-A MAC some recipes write `A.10` and `A.11`, the unit-A operand and accumulator
base. On descriptor-fed layers these are off the data path and any value, including none, gives identical
results; on pointer-fed layers they carry a per-tile descriptor word.

### 5.8 Commit

A commit ends accumulation, clears the accumulators, and publishes the tile
([§3.7](#37-accumulators-commit-and-readout)).

| Command | Effect |
|---|---|
| `nncmd 0x83` | Commit raw int32 to readout bank 0 |
| `nncmd 0x8b` | As `0x83`, and additionally requantize and pack the tile into readout bank 1 |
| `nncmd 0x8f` | As `0x8b`. Not characterised: whether readout bank 0 remains readable after `0x8b` as it does after `0x8f`. What is established: the 3×3 recipe reads bank 0 after `0x8f`; the floating-point heads read it after `0x83`; in the image-layer configuration, bank-0 reads after its `0x8b` return stale data |
| `nncmd 0x84` | Issued before `0x8b` in the 1×1, single-input-group and image recipes; its isolated effect is not characterised |

**Write `B.02` before the commit.** It selects the requantization window of the output group being
committed, in units of 4: `0` where a block commits a single group, `4 · g` per group, or
`4 · groups_per_pass · pass` per weight pass ([§3.8](#38-requantization-and-packing)). The write is
load-bearing even when the value is zero; omitting it leaves stale state in later tiles.

Some recipes close a column block with `nncmd 0xc0`, and the single-input-group recipe uses `nncmd 0xa0`
before its second phase and `nncmd 0xe0` at an output-group switch. These are reproduced positionally; their
isolated effects are not characterised ([Appendix C](#appendix-c--not-characterised)).

### 5.9 Drain

```
nndrd vrN,0x20      pop 64 bytes from the packed FIFO
nndrd vrN,R         read row-bank R of the raw int32 bank, non-destructively
```

At 4-bit output a tile is two packed reads (one per tile row); at 8-bit output it is four. Drained vectors
are stored to ORAM staging rows with MXUv3 vector stores, and a completed row pair is written back to DDR on
the write channel ([§6](#6-memory-and-dma)).

One adjustment is usually applied between drain and store: a per-nibble or per-byte unsigned maximum
against a clamp constant held in a vector register — a floor that is a no-op when the constant is zero, as
it is on most layers. Nothing else is applied; the packed values are final.

The drain returns the previous tile ([§3.7](#37-accumulators-commit-and-readout)). Two strategies absorb
the lag:

- **Store to the previous tile's address**, discarding the first drain of the run and issuing one extra
  drain after the last commit (the 3×3 recipe).
- **Withhold the first commit and carry the pending tile.** The first commit is delayed by one tile, so
  every drain lands in step with its own tile's addressing. The 1×1, single-input-group and head recipes do
  this.

### 5.10 Verifying a configuration

`nnrrd vrN,imm` reads a 64-byte configuration block back into a vector register. The immediate uses the
bank bits of `nnrwr`: bit 5 (`0x20`) bank A, bit 6 (`0x40`) bank B, bit 0 the program table within bank B.

The four blocks and their immediates are listed in [§4.5](#45-configuration-readback).

A field write is visible in the next `nnrrd`, before any arm, so a complete configuration can be checked
without moving data. Read the vector back to memory with the half-store form; the exact sequence, the block
layouts and the store encoding that must be used are given in [§4.5](#45-configuration-readback).
## 6. Memory and DMA

### 6.1 Physical memory map

| Region | Physical base | Size | Contents |
|---|---|---|---|
| CCU | `0x12200000` | — | XBurst2 cluster control block |
| CCU `MSCR` | `0x12200060` | 32 bit | Bits [12:10] select the L2 size (0 none, 1 = 128 KB, 2 = 256 KB, 3 = 512 KB, 4 = 1 MB); bit 5 enables the ORAM clock; bit 0 disables L2. Live value `0x02000420` |
| **DESRAM** | `0x12500000` | 32 KB aperture | DMA descriptors, 8 bytes each |
| **NNDMA IO** | `0x12508000` | 32 B | Channel kick and status registers |
| **ORAM** | `0x12620000` | 384 KB (`0x60000`) | On-chip operand, weight and staging memory |
| DDR weight pool | `0x03000000` | 8 MB | Driver-allocated through the `MALLOC` ioctl; lies inside the kernel's `mem=` window |
| `rmem` pool | `0x04800000` | 40 MB | Reserved for the camera's media pipeline, exposed as `/dev/rmem`; not used by the NNA |
| `nmem` pool | `0x07000000` | 16 MB | Reserved DDR for NNA tensors (`nmem=16M@0x7000000`) |
| DDR controller `DDRC_CCHC(7)` | `0x13012038` | 32 bit | Holds `0x88404002` after every driver `open`; a bus-port configuration, not a power control |
| AIP | `0x12b00000` | 64 KB | Image geometry engine; not part of convolution |
| FASTIO | `0x13f00000` | 255 B | A blocking read acts as an idle or completion barrier. Function not established; not required by any procedure here |

The kernel command line partitions DDR as `mem=72M@0x0 rmem=40M@0x4800000 nmem=16M@0x7000000`: the first
72 MB belong to Linux (and contain the weight pool), the next 40 MB to the media pipeline, and the last
16 MB to the NNA. The `nmem` base and size are read from `/proc/cmdline` at initialisation.

Cache attributes: DESRAM, the IO window and ORAM are uncached-accelerated; `nmem` and DDR are cacheable
write-back, so every transfer must be bracketed by a cache flush through the driver's `FLUSHCACHE` ioctl
([§6.8](#68-the-driver-interface), [§7.3](#73-cache-coherency)).

The ORAM base depends on the L2 size selected in `MSCR`: `base = 0x12600000 + L2_size`. A T41 unit that
boots with 128 KB of L2 places ORAM at `0x12620000` [L] (one board observed). The accelerator has no
IOMMU; physical addresses are obtained through an Ingenic-specific hardware register read (`rdhwr $4`),
enabled by the driver at `open`.

**The device has no interrupt.** The driver registers none and the device tree has no NNA node. Completion
is established by polling, through the blocking reads described in [§6.5](#65-kick-wait-and-fencing).

### 6.2 ORAM

ORAM is the accelerator's working memory. Weights, feature planes, the requantization table and output
staging rows all pass through it ([§2.1](#21-the-memory-hierarchy)).

**Size: 384 KB, from `0x12620000`.** ORAM is the part of the CPU cluster's L2 SRAM array not configured as
cache. The array is 512 KB and the boot loader configures 128 KB of it as L2, leaving 384 KB [L]. The
mapping window aliases with a period of `0x80000`: writes at offsets at or above `0x60000` reappear in the
L2 region, so the usable range is `0` … `0x5ffff` ([§7.6](#76-oram-aliasing)).

The `0xe0000` figure in the SDK header is not a T41 value; it is `1 MB − 128 KB`, the same rule applied to
a 1 MB array on a different part.

**ORAM is not a hard ceiling on layer size.** A layer whose resident plan does not fit is run by streaming
the weights per pass instead of keeping the whole blob resident ([§6.7](#67-weight-streaming)). A typical
resident plan, in allocation order, for the 3×3 recipe:

```
[ weights            D · Dout · 9 · 128 · w_bits ]
[ requantization     Dout · 256                  ]
[ feature ring       D groups × (3 + 3·stride) rows × row bytes   (3×3 recipe) ]
[ output staging     Dout planes × 2 buffers × 2 rows × staging row bytes ]
```

Absolute ORAM addresses are never given to the array. Only DMA descriptors and MXUv3 loads and stores use
them.

### 6.3 Descriptor format

Each descriptor is 64 bits, two 32-bit words; the length `len = bytes / 64 − 1` is 17 bits, split across
both words:

```
hi   [31:20]  len[11:0]
     [19:0]   oram  = (0x20000 + ORAM_offset) / 64
lo   [31:6]   ddr   = DDR physical address, 64-byte aligned
     [5:1]    len[16:12]
     [0]      more  = 1 if another descriptor follows in this chain
```

The closed form:

```
len = bytes / 64 - 1
lo  = ddr_physical | (len[16:12] << 1) | more
hi  = (len[11:0] << 20) | ((0x20000 + oram_offset) / 64)
```

Rules ([§7.5](#75-descriptor-construction)):

- The short form `lo = ddr_physical | more`, which omits the high length bits, is valid only for transfers
  of up to 256 KB (`len < 4096`).
- **Form the ORAM field by addition, not by OR.** `0x800 | (offset >> 6)` is correct only below `0x20000`;
  at or above 128 KB it aliases the destination back into the first 128 KB. The failure is silent — the
  configuration reads back correctly and small transfers land correctly — and corrupts every result.
- All addresses and lengths are in 64-byte units. The DDR address is physical and must lie in the `nmem`
  pool; an address from `dma_alloc_coherent` raises a bus error on this path.
- The length field is 17 bits in total, so one descriptor moves up to 8 MB (derived from the 17-bit length
  field; the largest transfer exercised is far smaller, 102,400 bytes).
- DESRAM must be cleared before a chain is written. The engine runs descriptors from the kicked index until
  it reaches one without the `more` bit.

### 6.4 Channels

| IO offset | Channel | Conventional use |
|---|---|---|
| `+0x00` | Read 0 | Weights and requantization table in the 1×1 and floating-point recipes; feature rows in the single-input-group and image recipes; the per-pass weight slice when the 3×3 recipe streams its weights ([§6.7](#67-weight-streaming)) |
| `+0x04` | Read 1 | Feature rows in the 3×3, 1×1 and floating-point recipes; the resident weight-and-table chain in the 3×3, single-input-group and image recipes |
| `+0x10` | Write | Finished output rows, ORAM staging to DDR |

The 3×3 recipe uses read 1 serially: its weight-and-table chain is kicked on read 1 while the array is
being configured, and its feature chains follow on the same channel once that chain has completed
([§5.5](#55-weight-preload), [§5.6](#56-activation-feed)).

The IO window holds eight registers. Registers 0, 1 and 4 are live buffer pointers and progress counters;
registers 2 (`0x00f1e3f1`) and 5 (`0x01f3ffea`) are fixed configuration latches written by the driver at
initialisation, not by the inference library.

**The two read channels are interchangeable as far as the array is concerned.** The assignment above is a
convention of each recipe, not a hardware role, and it is not consistent between recipes: the
single-input-group and image recipes use read 0 for features and read 1 for weights, the 1×1 and
floating-point recipes the reverse, and the 3×3 recipe puts both on read 1. Naming the channels "weight"
and "feature" therefore misleads; this manual calls them read 0 and read 1.

### 6.5 Kick, wait and fencing

A channel is started by writing `descriptor_index | 0x80000000` to its IO register. Bit 31 is the go bit.

The kick and wait sequences are:

```
kick:   rdhwr $N ; sync ; lw nmem[0] ; sw index|0x80000000, IO+off ; lw IO+off
wait:   sync ; lw IO+off ; sync ; rdhwr $N
```

`rdhwr $8`, `$9` and `$11` are the blocking idle reads for IO+0, IO+4 and IO+0x10 respectively. The
uncached load from `nmem` before a kick acts as a write fence for data just deposited there; the read-back
of the IO register after the kick is a completion barrier for the write itself. **Bundle descriptors into
one kick.** Several separate small kicks complete reliably only 25–50 % of the time; one kick per chain,
with all descriptors written first, is reliable.

Fencing and waiting are distinct operations and both are required ([§7.4](#74-fencing-and-waiting-are-different)).
The `sync` instructions order the CPU's accesses; the `rdhwr` read blocks until the channel is idle.
Omitting the wait before touching data a channel is still moving produces intermittent, position-dependent
corruption.

### 6.6 Ring buffers and whole-input mode

Feature data reaches ORAM under one of two schedules.

**Whole-input mode.** The entire input tensor is staged into ORAM once, before the layer's main loop, with
one descriptor per plane; there is no feature DMA afterwards. This is chosen when the tensor fits alongside
the weights and staging buffers.

**Ring mode.** ORAM holds a small number of rows per plane, and a chain per row pair refills them as the
computation advances, double-buffered by row-pair parity so that the DMA for the next pair runs while the
current pair is being consumed.

The 1×1 recipe selects whole-input mode when `H < 20` or `D · H < 300` or `Dout · D > 144`, and falls back to
a ring of 2 buffers × `D` planes × 2 rows otherwise. Both produce the same feed; only the per-plane stride
changes, from `2 · row_bytes` in the ring to `H_in · row_bytes` for a whole plane.

The 3×3 recipes always use a ring, because the kernel window needs rows above and below the output row
pair. The depth is recipe-specific: the 3×3 recipe uses `3 + 3 · stride` rows per group; the
single-input-group recipes use `7 · stride + K`. The ring slot for input row `r` is `r mod ring_rows`. The
rows above and below the current row pair — the vertical halo — are fed to the array again for the next
row pair, from the same ring slots ([§5.6](#56-activation-feed)). The extra depth beyond the window lets the
next strip's DMA be issued before the current strip is fully consumed; the kick must not be issued before
the feed has finished reading the rows it will overwrite.

### 6.7 Weight streaming

When a layer's weight blob alone exceeds ORAM, the resident plan is impossible and the layer is run with its
weights streamed per pass ([§11.8](#118-large-weights-and-channel-concatenation)):

1. The groups per pass are bounded by ORAM rather than only by the tap-slot budget:
   `groups_per_pass = min(0x120 / tap_slots, free_oram / per_group_bytes)`.
2. A double-buffered window in ORAM, where the resident weights would otherwise sit, holds one pass's
   slice. The requantization table is fetched once and stays resident.
3. Before each pass, that pass's slice is transferred from DDR into the window on read 0, through one
   rewritten descriptor, and the channel is waited on.
4. The slice is pushed onto the weight ports exactly as the resident path pushes the whole blob
   ([§5.5](#55-weight-preload)), the bank being addressed from zero every pass.

Everything else in the recipe — the feed schedule, the commit sequence, the drains, and `B.02` advancing by
`4 · groups_per_pass` per pass — is unchanged. A 589,824-byte weight blob runs this way against a 384 KB
ORAM.

### 6.8 The driver interface

The NNA is reached through `/dev/soc-nna` (character device 10,36); the ioctl magic is `'c'` and every
command is `_IOWR('c', n, int)`:

| ioctl | Number | Value | Purpose | Used |
|---|---|---|---|---|
| `MALLOC` | 0 | `0xC0046300` | Allocate from the DDR weight pool; returns `{vaddr, paddr, size}` | yes |
| `FREE` | 1 | `0xC0046301` | Release a `MALLOC` allocation | yes |
| `FLUSHCACHE` | 2 | `0xC0046302` | Cache maintenance for a DDR range: `{addr, len, dir}` | yes |
| `SETUP_DES` | 3 | `0xC0046303` | Write a descriptor through the driver | no — descriptors are written to DESRAM directly |
| `RDCH_START` | 4 | `0xC0046304` | Kick a read channel through the driver | no — kicks are direct IO writes |
| `WRCH_START` | 5 | `0xC0046305` | Kick the write channel through the driver | no — kicks are direct IO writes |
| `VERSION` | 6 | `0xC0046306` | Returns `0x41` for the T41 together with the `nmem` information | yes |

The `FLUSHCACHE` direction codes are `0` bidirectional, `1` to device (write back before a read channel
fetches from DDR) and `2` from device (invalidate after the write channel has deposited output). Every
transfer to or from cacheable memory is bracketed by this call ([§7.3](#73-cache-coherency)).

Opening the device enables the coprocessor and the `rdhwr` address translation for the calling core. The
hardware windows — DESRAM, the NNDMA IO registers and ORAM — are mapped uncached through `/dev/mem` opened
with `O_SYNC`, at the physical addresses of [§6.1](#61-physical-memory-map); the FASTIO and BSCALER windows
that the vendor mapping also establishes are not required by any procedure in this manual. Inference issues
only `FLUSHCACHE` and `VERSION`; descriptor construction and kicks are direct accesses into the mapped
windows.
## 7. Operational hazards

Each item below has produced a silent wrong result or a hang in practice. They are ordered by how difficult
they are to diagnose from the symptom; the mechanism behind each is described in the section cross-referenced.

### 7.1 The B.1b reset hazard

**`nncmd 0x00` does not fully re-initialise the MAC stage.** What the weight preload after a reset sees
depends on the configuration in force before it, unless field `B.1b` is written immediately beforehand ([§5.1](#51-reset)).

*Symptom.* Every output value saturated, and the result depends on which layer ran previously. The same
code is correct after one predecessor and wrong after another.

*Remedy.* Begin every run with

```
nnrwr B.1b, w      w = 0
nncmd 0x00
```

Writing `B.1b` *after* the reset does not cure it; the write must precede the reset. The behaviour is
consistent with the reset re-sampling the field into the MAC stage. [H]
Not characterised: whether writing another field before the reset would serve equally. What is established:
writing `B.1b = 0` immediately before `nncmd 0x00` is sufficient.

Vendor software does not encounter the hazard because, in the networks observed, every 8-bit-input layer
follows a layer that leaves `B.1b` at 0 and the 4-bit recipes are not sensitive to it.

### 7.2 Coprocessor context and thread pinning

The array is on coprocessor 2, which requires CP0 `Status.CU2`. The driver sets CU2 **only on the core that
called `open()`**, and does not save it in the thread context ([§6.8](#68-the-driver-interface)).

*Symptom.* SIGILL on a coprocessor instruction, intermittently, on roughly half of unpinned runs.

*Remedy.* Pin the process to one core **before** opening the device, and do not migrate.

*Note.* Coprocessor and ORAM state persist across processes; a fresh process sees the previous run's array
state. A test of whether an operation matters therefore overwrites the data that operation would have written
rather than omitting it: an omitted step leaves the stale state in place and the test appears to pass.

### 7.3 Cache coherency

`nmem` and DDR are cacheable write-back; ORAM, DESRAM and the IO window are uncached ([§2.1](#21-the-memory-hierarchy)).
The accelerator has no coherency with the CPU caches.

*Remedy.* Flush before the device reads, invalidate after it writes, using the driver's `FLUSHCACHE` ioctl
(direction 1 to device, 2 from device; [§6.8](#68-the-driver-interface)). Every transfer must be bracketed.

### 7.4 Fencing and waiting are different

A `sync` orders the CPU's own accesses. It does not report that a DMA channel has finished. The channel-idle
read (`rdhwr $8`, `$9`, `$11` for read 0, read 1 and write) blocks until the channel is idle ([§6.5](#65-kick-wait-and-fencing)).

*Symptom.* Intermittent corruption whose position depends on timing — commonly the first rows of a layer,
or the rows a strip DMA was still writing.

*Remedy.* Both. Fence around the kick; wait on the channel before touching the data it moves. In particular,
wait for the initial operand transfers before the first feed, even where vendor code appears not to: its own
timing may cover the race where a reimplementation's does not.

### 7.5 Descriptor construction

Two failure modes, both silent ([§6.3](#63-descriptor-format)):

- **ORAM field formed by OR.** `0x800 | (offset >> 6)` aliases every offset at or above `0x20000` back into
  the first 128 KB. Configuration readback and small transfers still look correct while every result is
  wrong. Form the field by addition: `(0x20000 + offset) / 64`.
- **DESRAM not cleared.** A chain runs from the kicked index until a descriptor without the `more` bit.
  Stale descriptors beyond the intended end continue the chain into arbitrary memory.

Also: several small kicks are unreliable (25–50 % completion) [M]. Write all descriptors, then issue one kick.

### 7.6 ORAM aliasing

The ORAM window aliases with a period of `0x80000` ([§6.2](#62-oram)). Offsets at or above `0x60000` are
the L2 cache region seen again, so a plan sized to the SDK header's 896 KB figure will silently overwrite
its own start. Usable ORAM is `0` … `0x5ffff`.

### 7.7 Readout lag and stale reads

The packed readout bank runs behind the write point ([§3.7](#37-accumulators-commit-and-readout)), so a drain
returns an earlier tile. A procedure that stores each drain to the current tile's address writes every result
one tile late.

*Remedy.* Either absorb the lag by storing to the previous tile's address, or discard the first drain of a
run and issue one extra at the end. Setting `B.07` below 2 does not remove the lag.

Bank 0 is normally readable immediately after a commit and is the reliable observation channel.
Not characterised: whether readout bank 0 is readable after every commit form. What is established: bank 0
holds the committed tile after `nncmd 0x8f` and `0x83`; in the image-layer configuration, bank-0 reads after
its `0x8b` commit return stale data and must not be used to observe results.

### 7.8 Do not interrupt a run

Interrupting the instruction stream mid-sequence — for example a watchdog killing the process between a MAC
and its drain — can wedge the accelerator. A wedged engine survives a soft reboot and requires a power
cycle. Wrap experimental runs in an alarm so they exit cleanly.

### 7.9 The word file is live state

The array samples the `vr31` word file during a MAC. Words that a MAC phase names hold that phase's operand
walk position, and words left over from a configuration record are part of the machine state.

*Symptom.* A byte-for-byte identical instruction stream and a completely wrong output tensor.

*Remedy.* Stage configuration records only before the arm, and only in words no MAC phase names
([§3.6](#36-mac-phases-and-the-walk-position)).

### 7.10 Errors invisible to static inspection

Two classes of defect cannot be found by disassembling or by comparing instruction streams, because the
values involved travel inside configuration records ([§4](#4-the-configuration-model)) rather than in instruction fields:

- wrong values in configuration or program words — a mis-masked program step, for example;
- anything depending on the residual contents of the word file.

Both require running the layer and comparing the output. An instruction-stream comparison is a useful first
gate but is not sufficient on its own.
## 8. Worked example

This chapter takes one layer from an unopened device to a verified output tensor, applying in order
[§4](#4-the-configuration-model), [§5](#5-running-a-layer), [§6](#6-memory-and-dma) and
[§7](#7-operational-hazards); each section says what is done and where the governing rule is stated. The
complete C listing is [Appendix E](#appendix-e--complete-example-listing); its constants are quoted here.

The example layer is a 1×1 convolution, `Cin = 64`, `Cout = 128`, `H = 12`, `W = 20`, stride 1, with 4-bit
activations, weights and output ([§11.1](#111-11-4-bit-operands)): `D = 2` input channel groups and
`Dout = 4` output channel groups ([§2.2](#22-channel-groups-of-32)), one input group per execution unit
([§3.3](#33-the-two-execution-units)); in tiles ([§3.1](#31-the-tile)) it is `COLUMN_BLOCKS = 5` by
`ROW_PAIRS = 6`. Height and width differ deliberately: several fields and strides come in height/width
pairs, and a square example would hide the distinction.

### 8.1 Open and map the device

Two constraints ([§7.2](#72-coprocessor-context-and-thread-pinning)) apply before any coprocessor
instruction is issued.

1. **Pin the process to one CPU core before opening the device.** The NNA is on coprocessor 2, which
   requires CP0 `Status.CU2`. The driver sets CU2 only on the core that called `open()`, and does not carry
   it in the thread context. On the two-core kernel a migration mid-run faults the next coprocessor
   instruction with SIGILL.
2. **Open `/dev/soc-nna` before mapping anything.** The open enables CU2 and the accelerator clocks.

```c
cpu_set_t set;
CPU_ZERO(&set);
CPU_SET(0, &set);
sched_setaffinity(0, sizeof set, &set);     /* before open(), not after */

int fd = open("/dev/soc-nna", O_RDWR);
```

Three physical windows are then mapped, all uncached ([§2.1](#21-the-memory-hierarchy)):

| Window | Physical base | Size | Contents |
|---|---|---|---|
| DESRAM | `0x12500000` | 32 KB | DMA descriptors |
| NNDMA IO | `0x12508000` | 32 B | DMA kick and status registers |
| ORAM | `0x12620000` | 384 KB | On-chip operand and staging memory |

Tensors live in the reserved `nmem` DDR pool, whose base and size come from the kernel command line
(`nmem=<size>@<addr>`). The DMA engine takes **physical** addresses in that pool; a `dma_alloc_coherent`
address raises a bus error on this path.

### 8.2 Allocate tensors

Four regions are needed in `nmem`, each 64-byte aligned:

| Region | Size for the example | Formula |
|---|---|---|
| Input tensor | 7,680 B | `D · H_in · round64(W_in · 32 · in_bits / 8)` |
| Weight blob | 4,096 B | `D · Dout · 128 · w_bits` for a 1×1 kernel |
| Requantization table | 1,024 B | `Dout · 256` |
| Output tensor | 15,360 B | `Dout · H · round64(W · 32 · out_bits / 8)` |

Feature maps use the NDHWC32 layout of [§2.3](#23-ndhwc32-feature-maps), one plane per channel group; an
input plane row is `ROW_BYTES = 320`, and so is an output row.

### 8.3 Prepare weights and the requantization table

Both are consumed as the model file stores them; nothing is repacked.

- **Weights** for 4-bit operands are two 2-bit planes per 32 × 32 tap, 512 bytes per tap
  ([§2.5](#25-weights)). For a 1×1 kernel there is one tap per (input group, output group) pair, and the
  blob is ordered output-group-major.
- **The requantization table** is 256 bytes per output channel group: `int32 bias[32]` at `+0x00` and
  `int32 mult[32]` at `+0x80`. [§3.8](#38-requantization-and-packing) gives the formulas for generating one.

### 8.4 Reset safely

A reset alone does not put the MAC stage into a known state ([§5.1](#51-reset)). `nncmd 0x00` returns the
configuration fields to their reset values, but the MAC stage retains configuration from whatever ran
before it unless field `B.1b` is written immediately beforehand. The first two instructions of every run
are therefore:

```c
nna_word_set(NNA_W0, 0);
NNA_WRITE_FIELD(NNA_MAC_BALANCE, NNA_W0);   /* nnrwr B.1b, w0 */
NNA_NNCMD(NNA_CMD_RESET);                   /* nncmd 0x00     */
```

This pair is `nna_reset()` in the companion API ([§16](#16-runtime-api)). [§7.1](#71-the-b1b-reset-hazard)
explains the failure it prevents: a completely saturated output tensor whose contents depend on which layer
ran previously.

### 8.5 Configure

Configuration reaches the NNA through the word file `w0…w15`, one `nnrwr` per field
([§5.2](#52-field-configuration)). The listing describes the layer in a descriptor and calls
`nna_config_apply()`, which derives the encodings ([§4](#4-the-configuration-model)). The fields divide
into those latched at the arm and those read per tile; for the example layer the static ones come out as:

| Field | Value | Meaning |
|---|---|---|
| `A.0c`, `A.0e` | 11, 19 | Input height − 1, input width − 1 |
| `A.08`, `B.06` | 2, 2 | Mode code pair, from ConvOp `+0x80` |
| `A.19`, `A.17` | 4, 8 | MAC address pitch and span (recipe constants) |
| `B.04` | 1 | Output element width class, `(out_bits − 2) / 2` |
| `B.15` | 1 | Input tile size class, `log2(tile_bytes / 64)` |
| `A.1e` | 1 | Kernel tap mask: one tap |
| `A.1b` | `0x11` | Operand precision: 4-bit weights, 4-bit input |
| `A.03` | 1 | Input element width, `in_bits / 2 − 1` |
| `A.09`, `A.0b` | 0, 0 | Input row and column origin; no padding |
| `A.1c`, `A.1d` | 0, 0 | Input channel groups per unit, minus one |
| `A.18` | `0x12` | Row-edge and unit-window control |
| `B.05`, `B.03`, `B.1b` | 2, 0, 0 | Mode flags, pack format, MAC balance |

Next comes the walk program ([§3.5](#35-the-walk-program), [§5.3](#53-program-loading)): two entries,
loaded into both slots with `NNA_PROG_LENGTHS(2, 2)` — consume one operand and step forward, then rewind.
`NNA_PROG_STEP` masks the negative rewind to the ten-bit step field so that it cannot spill into the count
bits and become an `NNA_PROG_LOOP` entry.

```c
static const uint16_t program[2] = { NNA_PROG_STEP(1), NNA_PROG_STEP(IN_BITS - 1 - IN_BITS * GROUPS_A) };
nna_program_load(program, 2, program, 2);   /* slot A, slot B */
```

Then the arm, and finally the two unit start offsets `B.10` and `B.11`, which must be written **after** the
arm ([§5](#5-running-a-layer)). The whole configuration can be read back and checked before any data moves
(§8.8).

> Not characterised: `A.0c`'s role in this recipe when H ≠ W. What is established: the vendor's 1×1
> procedure writes `W − 1` into both fields; the reference writes `H − 1` and `W − 1`; both are exact on
> every layer with H = W. See [Appendix C](#appendix-c--not-characterised).

### 8.6 DMA the operands

Descriptors are 8 bytes, written into DESRAM and chained. A channel is started by writing
`descriptor_index | 0x80000000` to its IO register, and runs from that index until a descriptor without the
`more` bit ([§6.3](#63-descriptor-format)).

```
lo = ddr_physical_address | more
hi = ((bytes / 64) - 1) << 20 | ((0x20000 + oram_offset) / 64)
```

The ORAM field must be formed by **addition**, not by OR. `0x800 | (offset >> 6)` aliases every offset at or
above 128 KB back into the first 128 KB, which corrupts results silently while configuration readback and
small transfers still look correct.

For this layer: chain 0 carries the table and the weights; a feature chain carries one descriptor per input
plane per row pair. In ORAM the whole input comes first, then the weights, the output staging area, and the
table. Clear DESRAM before writing descriptors, bundle a chain into one kick, and wait for the channel to go
idle before touching the data it moves. Several small kicks are unreliable; one kick per chain is not
([§6.5](#65-kick-wait-and-fencing)).

### 8.7 Feed, MAC, commit, drain

The per-tile sequence, once per output channel group `g` of a column block, follows the instruction order
of [§3.9](#39-the-instruction-order-of-one-layer):

```
for cb in 0 .. CB-1:                          # column blocks of four pixels
    if cb > 0: nncmd 0x84                     #   the previous block's last commit
    push unit A's windows                     nndwr vr0,0x20 ; nndwr vr1,0x20   (one pair per input group)
    if cb > 0: nncmd 0x8b                     #   ... completes here
    nnrwr A.11, w1 ; nnrwr A.10, w1           MAC address tags
    nnmac vw4,1                               unit A's MAC phase
    push unit B's windows                     nndwr vr0,0x21 ; nndwr vr1,0x21
    nnmac vw5,3                               unit B's MAC phase
    if cb > 0: drain and store tile (cb-1, Dout-1)
    nnrwr B.02, w1                            the committed group's requantization window
    for g in 1 .. Dout-1:                     # the other output groups reuse the window just fed
        nncmd 0x84 ; nncmd 0x8b               commit group g-1
        nnmac vw4,1 ; nnmac vw5,3             accumulate group g against the next weight tile
        drain and store tile (cb, g-1)

# after the last column block of the row pair
nncmd 0x84 ; nncmd 0x8b
drain and store tile (CB-1, Dout-1)           then write the row pair back
```

Three rules shape this loop, each stated in full elsewhere:

- **Feeds are positional** ([§3.4](#34-ports-and-positional-feeds), [§5.6](#56-activation-feed)): the
  ordered sequence and count of pushes is what matters, not any address.
- **Weights are pushed once per pass, activations once per column block** ([§5.5](#55-weight-preload)):
  all `Dout` output groups of a block reuse one activation window.
- **The packed drain lags by one tile** ([§3.7](#37-accumulators-commit-and-readout), [§5.9](#59-drain),
  [§7.7](#77-readout-lag-and-stale-reads)): every store handles the *previous* tile, and the row pair ends
  with one commit ([§5.8](#58-commit)) and one more store.

### 8.8 Verify the output

Three checks, in increasing cost:

1. **Configuration readback.** `nnrrd` reads the committed configuration into a vector register: block G is
   the bank-A field file, H0 bank B, H1 the program table. A field write is visible in the next `nnrrd`,
   before the arm. This catches field and program errors without running any data.
2. **DMA-only run.** Configure and issue every DMA, but feed nothing and run no MAC. Comparing ORAM against
   the source tensors isolates descriptor errors from NNA errors.
3. **Byte comparison against a reference.** Compare the output tensor byte for byte against a reference
   implementation or a vendor capture. On every layer measured, the result is deterministic and
   byte-for-byte reproducible; treat any difference from a reference as a defect.

Two classes of error are invisible to instruction-level inspection and require running on the device
([§7.10](#710-errors-invisible-to-static-inspection)): values inside configuration records — including
program words — and anything that depends on the residual state of the `vr31` word file.

### 8.9 The complete listing

[Appendix E](#appendix-e--complete-example-listing) contains the whole sequence of this chapter as
compilable C against the companion API ([§16](#16-runtime-api)), from `open()` to the byte comparison. Its
constants and its ORAM and DESRAM plans are the ones quoted above. Three details of it are easy to get wrong:

- The activation feed is **outside** the output-group loop: re-feeding per group pushes each window `Dout`
  times, and every operand after the first extra push lands in the wrong slot.
- Every `store_tile` call names the tile committed one step earlier, and the row pair ends with a commit
  followed by one more store; storing a tile immediately after committing it displaces the whole output.
- `w1` is set to zero once, after the arm, and every per-tile field write reads it from there. The word file
  is live state that the NNA samples: it is installed after `nna_config_apply()`, which owns all sixteen
  words while it runs, and nothing in the tile loop may overwrite those words.
# Part III — Layer recipes

Part III gives one complete procedure per kind of layer. Each recipe is an instance of the sequence in
[§3.9](#39-the-instruction-order-of-one-layer) and [Chapter 5](#5-running-a-layer), specialised to a
geometry: the ORAM plan, the DMA chains, the configuration values, the walk program, the feed schedule and
the tile loop. The recipes are written to a common template ([Chapter 10](#10-the-recipe-template)) so
that the same question is answered in the same place in each.

## 9. Choosing a recipe

A recipe is a complete programming procedure for one class of layer: the ORAM plan, the DMA chains, the
configuration, the walk program, the feed schedule and the tile loop.

| Geometry | Recipe |
|---|---|
| 1×1, `2 ≤ D ≤ 32`, 4-bit input | [§11.1](#111-11-4-bit-operands) |
| 1×1, 8-bit input | [§11.1](#111-11-4-bit-operands) with the substitutions in [§11.5](#115-8-bit-input-variants) |
| 3×3 pad 1, stride 1 or 2, `D ≥ 2`, `Dout ≤ 64` | [§11.2](#112-33-4-bit-operands) |
| 3×3 or 1×1, `D = 1` (`Cin ≤ 32`) | [§11.3](#113-single-input-group-row-quad) or [§11.4](#114-single-input-group-two-slot) |
| 3-channel image input, 3×3 stride 2 | [§11.6](#116-the-rgb-image-layer) |
| 1×1, 8-bit operands, floating-point output | [§11.7](#117-floating-point-output-heads) |
| Weight blob larger than ORAM | [§11.8](#118-large-weights-and-channel-concatenation) with the recipe its geometry selects |
| Several input tensors (channel concatenation) | [§11.8](#118-large-weights-and-channel-concatenation), on top of the geometry's recipe |

A layer's result does not depend on which recipe computes it: a geometry served by two recipes may use
either. This is established for one layer computed by two different procedures and assumed in general [L].

**Where the group limits come from.** The per-unit group-count fields `A.1c` and `A.1d` are four bits wide
([§3.3](#33-the-two-execution-units)), so each unit reduces over at most 16 input groups — 512 input
channels — per pass, and the two units together over 32. That is the origin of the 1×1 recipe's `D ≤ 32`
guard; a deeper layer is run in several passes. The input-size fields `A.0c` and `A.0e` are eleven bits,
so input height and width are at most 2048 (derived from the field widths, not measured).

Constants used throughout: `D = Cin/32`, `Dout = Cout/32`, `CB = ceil(W/4)` column blocks,
`RP = ceil(H/2)` row pairs, `round64(x)` rounds up to a multiple of 64.

## 10. The recipe template

Every recipe in [Chapter 11](#11-the-recipes) is written in six parts:

1. **Serves and guards** — the geometries the recipe handles and the conditions its planner checks.
2. **Geometry and derived sizes** — the quantities computed from the layer's shape: group counts, walk
   depth, row bytes, staging sizes, pass counts.
3. **ORAM plan and DMA chains** — the resident layout in ORAM and the descriptor chains that fill it.
4. **Prologue** — everything issued once per layer, in order: the guarded reset, the static fields, the
   walk program, the arm, the post-arm fields, the weight preload, the requantization table, and the
   pre-feed.
5. **Tile loop** — the per-tile sequence: per-tile fields, feed, MAC phases, drain, commit, DMA kicks and
   waits, and how the readout lag is absorbed.
6. **Variants and not-characterised items** — the recipe's stride and width variants, and every statement
   in the recipe that is marked **Not characterised**.

Three things are common to all recipes and are stated here once.

**The loop variables.** `D`, `Dout`, `CB` and `RP` are as defined in [Chapter 9](#9-choosing-a-recipe).
Software iterates row pairs (or row quads) outermost, column blocks within, and output groups innermost;
the array walks taps and input groups inside a tile ([§3.2](#32-marching-a-layer)).

**The packed drain routine.** At 4-bit output a tile of one output group is drained as two 64-byte reads,
one per tile row, clamped and stored into the ORAM staging rows:

```
nndrd vr10,0x20 ; nndrd vr11,0x20            bank 1: row r, then row r+1
maxu4bi vr10,vr10,vr29 ; maxu4bi vr11,vr11,vr29
store vr10 -> staging[parity][group][row 0] + cb*64
store vr11 -> staging[parity][group][row 1] + cb*64
```

The `maxu4bi` is a per-nibble unsigned maximum against a replicated clamp constant held in `vr29` — a ReLU
floor applied by software before the output DMA, and a no-op when the constant is 0, as it is on every 4-bit
layer. At 8-bit output the routine reads four registers, clamps with `maxub`, and stores 128 bytes per row
(`vr10`+`vr11` to row r, `vr12`+`vr13` to row r+1). Staging is double-buffered by row-pair parity so that
the write-back DMA of one row pair overlaps the computation of the next. The image layer drains into `vr22`
and `vr0` and stores 128 bytes per MAC phase per tile ([§11.6](#116-the-rgb-image-layer)); the
floating-point heads drain bank 0 instead ([§11.7](#117-floating-point-output-heads)).

**The readout lag.** Bank 1 returns the previous tile ([§3.7](#37-accumulators-commit-and-readout)).
Each recipe absorbs the lag in one of two ways, and says which: the 3×3 recipe discards the first drain of
a run into scratch, stores every later drain to the previous tile's address, and drains once more after
the loop; the 1×1, single-input-group and head recipes withhold the first commit of a row pair and carry
the last committed-but-undrained tile into the next column block or row pair.

## 11. The recipes

### 11.1 1×1, 4-bit operands

#### 11.1.1 Serves and guards

Serves 1×1 layers with `2 ≤ D ≤ 32`, stride 1 or 2, no padding, 4-bit input and weights, packed output. The vendor
descriptor fields that select this recipe are listed in [Appendix B](#appendix-b--vendor-executor-map).

**Constraints.** `D = Cin/32` input groups, summed over all inputs of the layer (a multi-input 1×1 is a channel
concat; see [§11.8](#118-large-weights-and-channel-concatenation)), **`2 ≤ D ≤ 32`** (a single input group is served
by another recipe, [§9](#9-choosing-a-recipe)), `Dout = Cout/32`, output `H×W` = input size (stride 1; strides ≤ 2
accepted, stride 2 uses the strided feed routines), pads must be 0, kernel 1×1. `in_bits`/`w_bits`/`out_bits` are 4
here; the requantization **mode** is `2` (hardware requant with the table in bank B). The output clamp byte is 0 on
every 4-bit layer.

#### 11.1.2 Geometry and derived sizes

`rowbytes = W·in_bits·32/8` (= the input tensor's dim-2 stride, W·16 for 4-bit), `out_rb = round64(W·32·out_bits/8)`
(DDR), `stg_rb = round64(((W+3)&~3)·32·out_bits/8)` (ORAM staging row), column blocks `CB = ceil(W/4)`, row pairs `RP
= ceil(H/2)`, units `nA = D/2` (groups `0..nA−1`, unit A, port `0x20`, MAC `nnmac vw4,1`), `nB = D − nA` (groups
`nA..D−1`, unit B, port `0x21`, MAC `nnmac vw5,3`), weight bytes per output group `wg = D·512·(w_bits/2)/2 = D·512`
(4-bit), table bytes per output group `tg = 256` (`tbl_dtype_bits·8`), **output groups per pass** `G = min(Dout,
0x120/((w_bits/2)·D), ORAM budget)` (tap-slot budget 288: `D = 16 → 9`, `D = 32 → 4`; `Cout = 512` with `Cin = 512` →
2 passes of 9+7), passes `P = ceil(Dout/G)`, the last pass has `Glast = Dout − (P−1)·G` groups. Column step `cstep =
stride_h·0x80·in_bits/8` (64 B = 4 px × 32 ch × 4 bit).

**Whole-input vs ring mode.** If `N == 1` and (`H < 20` or `D·H < 300` or `Dout·D > 144`), the planner tries to keep
the **whole input** in ORAM (`D·H_in·rowbytes`), one read kick before the pass loop and no feature DMA afterwards; if
that does not fit the free ORAM range it falls back to the **ring**: 2 buffers × D planes × 2 rows, one read kick per
row pair, double-buffered by row-pair parity. Both produce the same feed (the per-plane stride table just changes from
`2·rowbytes` to `H_in·rowbytes`). The ORAM budget then bounds `G`: `(free − 64 − tbl − input) / (2·2·stg_rb + wg)`.

#### 11.1.3 ORAM plan and DMA chains

**ORAM plan** ([§6.2](#62-oram)): `[input: ring or whole image][weights of one pass: G·wg][output staging: 2 parities
× G groups × 2 rows × stg_rb][requant table: Dout·tg]`. Absolute ORAM addresses are never given to the array; only the
DMA descriptors and the CPU loads and stores use them.

**DESRAM** ([§6.3](#63-descriptor-format)). Index 0 = table (`Dout·tg` bytes, `more`), index 1 = weights of the
current pass (`G·wg`); chain 0 is kicked on **IO+0** — again at the start of every later pass after rewriting index 1
to the next `G·wg` bytes of the blob. Feature chains from index 2: ring mode = per image `n`, per row pair `rp`: `D`
descriptors (one per plane: rows `2rp, 2rp+1` = `2·rowbytes` contiguous in DDR, clipped at the bottom; `more` on all
but the last plane) at index `2 + (n·RP+rp)·D`, into `ring[rp mod 2][plane]`; whole-input mode = `D` descriptors of
`H_in·rowbytes` at index 2. The output chains start at index **`2 + 2·D·RP·N`** in ring mode (the planner reserves two
descriptor slots per plane per row pair) and at **`2 + D`** in whole-input mode (right after the D whole-plane
descriptors): per pass `p`, image `n`, row pair `rp`: for each group `g` of the pass, for each of the (≤2) rows:
staging `[rp mod 2][g][row]` → DDR `out + (p·G+g)·H·out_rb + (2rp+row)·out_rb`, `more` on all but the last (`2·G`
descriptors per kick; step 8 for `G = 4`).

#### 11.1.4 Prologue

The prologue begins with the guarded reset, `nnrwr B.1b ← 0 ; nncmd 0x00` ([§5.1](#51-reset)), and then
proceeds as follows.

`nncmd 0x00` then, in this order, every value staged through vr31 words 11/12 with `law vr31,(sp),0xb/0xc`
([§4](#4-the-configuration-model)):
```
A.0c = W−1 (see 11.1.6)           A.0e = W−1        A.08 = B.06 = 2
A.19 = 4                          A.17 = 8          B.04 = {out 2-bit:0, 4-bit:1, else 3}
B.15 = in_bits/2−1 (4-bit) | 2 (8-bit) | 3 (16-bit) A.1e = 1   A.1b = (w_bits/2−1)<<4 | (in_bits/2−1)
A.03 = in_bits/2−1               A.0b = A.09 = 0   A.01 = D & 1    A.1c = nA−1   A.1d = nB−1
vr29 ← clamp byte (& ((1<<out_bits)−1)) replicated per out_bits lane, repiw
vr31 w2 = 0, w3 = 2 + 4·nA, w4 = 0, w5 = 16·nA + 8   (general: w3 = 2 + in_bits·nA, w5 = 4·in_bits·nA + 8, §11.5)
A.18 = 0x12   B.05 = 2   B.03 = 0   B.1b = 0
B.18 = 0 ; program A ; B.18 = 8 ; program B ; B.1a = lenB<<4 | lenA
nncmd 0x60 ; liwr w1,0 ; vr28 ← 1<<(out_bits−1) (repih; MXU-requant path only) ; B.10 = w2 ; B.11 = w3
```
`B.10 = 0` and `B.11 = 2 + in_bits·nA` are written once, after the arm. `w4 = 0` and `w5 = 4·in_bits·nA + 8` are the
MAC words of unit A and unit B; neither is stepped — the walk program does the walking ([§3.5](#35-the-walk-program),
[§3.6](#36-mac-phases-and-the-walk-position)). Each unit's program, for `groups` = its input groups
([§14](#14-the-program-table)): one group → `STEP(1) ; STEP(in_bits − 1 − in_bits·groups)` (length 2); two or more →
`LOOP(3, groups−1) ; STEP(1) ; STEP(in_bits − 1) ; STEP(1) ; STEP(in_bits − 1 − in_bits·groups)` (length 5). Every
entry consumes an operand, so the walk reads two windows per input group.

No `A.0f`, `A.12/A.14`, `B.07/B.08/B.09` are written by this recipe (`B.07` keeps its reset value). Per-tile fields
`A.10/A.11`, `B.02`, `B.1c`, `B.1e` are all written from `w1 = 0`; `B.02` stays 0 for every tile.

#### 11.1.5 Tile loop

**Pass loop** (`for p in 0..P−1`, `Gp` = groups of this pass; loop constants and the shared drain routine are in
[§10](#10-the-recipe-template)):
```
sync ; lw nmem[0] ; sync ; rdhwr $8              # chain 0 (weights+table of this pass) landed
nnrwr B.1c (0)
for i in 0..D·Gp·(w_bits/2)−1:                   # weight feed, 256 B per iteration, blob order
    law vr0 ← W[i·256 .. +64), vr1 ← +64 ; nndwr vr0,0 ; nndwr vr1,2
    law vr2 ← +128, vr3 ← +192       ; nndwr vr2,4 ; nndwr vr3,7
if p < P−1: DESRAM[1] ← next pass's G·wg weight bytes → ORAM weights ; kick IO+0 (index 0)
nnrwr B.1e (0) ; for j in 0..(Gp·tg/128)−1: law vr0 ← T[j·128], vr1 ← +64 ; nndwr vr0,0x40 ; nndwr vr1,0x40
        # Gp·4 pushes of 64 B: the table pushes cover only this pass's groups (T advances by Gp·256 B per pass)
for n in 0..N−1:
    ring: read kick (rp 0)
    for rp in 0..RP−1:
        read wait (sync ; lw IO+4 ; sync ; rdhwr $9)
        ring and rp < RP−1: read kick (rp+1)      # into the other parity buffer
        row0 = input buffer of rp (ring: parity slot; whole: image + 2rp·rowbytes), row1 = row0 + rowbytes
        for cb in 0..CB−1:
            if cb > 0: nncmd 0x84
            feedA(row0 + cb·cstep, row1 + cb·cstep): for g in 0..nA−1: law vr0 ← row0[g], vr1 ← row1[g] ; nndwr vr0,0x20 ; nndwr vr1,0x20   (plane stride = slot bytes)
            if cb > 0: nncmd 0x8b
            nnrwr A.11 ; nnrwr A.10 ; nnmac vw4,1                       # unit A, groups 0..nA−1
            feedB(...): groups nA..D−1 the same as nndwr vr0,0x21 ; nndwr vr1,0x21
            nnmac vw5,3                                                # unit B, groups nA..D−1
            if cb > 0: store(prev cb, group Gp−1)   # one-tile lag: the drain returns the previous MAC pair
            nnrwr B.02 (0)
            for g in 1..Gp−1:
                nncmd 0x84 ; nncmd 0x8b ; nnmac vw4,1 ; nnmac vw5,3 ; store(cb, g−1)
        nncmd 0x84 ; nncmd 0x8b ; write wait (sync ; lw IO+0x10 ; sync ; rdhwr $11)
        store(CB−1, Gp−1) ; write kick (rp)        # 2·Gp descriptors: staging[rp mod 2] → DDR rows 2rp,2rp+1
        swap input parity, swap staging parity
```
The requantization table is pushed once per weight pass, covering only that pass's output groups. Each MAC phase
follows immediately after its own unit's feed; commits are `nncmd 0x84 ; nncmd 0x8b` ([§5.8](#58-commit)). The feed
differs by input width and stride: at 8-bit input each plane contributes four windows per tile rather than two
([§11.5](#115-8-bit-input-variants)), and at stride 2 the window selects every other pixel.

`store(cb, g)` at 4-bit output is `nndrd vr10,0x20 ; nndrd vr11,0x20 ; maxu4bi vr10,vr10,vr29 ; maxu4bi vr11,vr11,vr29
; store vr10 → staging[rp mod 2][g][row 0] + cb·64 ; store vr11 → [row 1] + cb·64` — **a per-nibble unsigned maximum
against the clamp constant is applied by software before the output DMA**, a no-op when the clamp byte is 0, as it is
on every 4-bit layer. At 8-bit output the same store uses `maxub` and four registers ([§5.9](#59-drain)). Every MAC
pair therefore yields one output group of a 2-row × 4-pixel tile, the groups of a pass cycle through the weights
loaded in the array, and the activation window is pushed once per column block and reused for all `Gp` groups. The
readout lag is absorbed by skipping the commit and drain on the first column block of a row pair and flushing after
the last: the first drain of a row pair holds the previous column block's last group, so there is no junk drain (at
`cb = 0` no store precedes `B.02`).

#### 11.1.6 Variants and not-characterised items

**Variants.** 8-bit input is [§11.5](#115-8-bit-input-variants); stride 2 uses the strided feed routines with the same
tile loop; multi-input layers (channel concat over the summed `D`) are
[§11.8](#118-large-weights-and-channel-concatenation). Mode `1` (CPU requantization on the MXU) is not used by any
4-bit layer this recipe serves.

Not characterised: `A.0c`'s role in this recipe when H ≠ W. What is established: the vendor's procedure writes `W − 1`
into both `A.0c` and `A.0e`; a reference implementation writing `H − 1` and `W − 1` is exact on every layer with H =
W. `A.0c` is the input height minus one ([§13](#13-configuration-fields)) and the array masks coordinates beyond it,
so the two agree on square feature maps and remain safe wherever `W ≥ H` [L]; a layer with `H > W` needs `H − 1` in
`A.0c` [L].
### 11.2 3×3, 4-bit operands

#### 11.2.1 Serves and guards

Serves 3×3 pad-1 layers with `D ≥ 2` and `Dout ≤ 64`, stride 1 or 2, 4-bit activations, 4- or 8-bit weights
and 4- or 8-bit output, in one or several weight passes.
Stride 2 runs on the same routine (§11.2.6). A layer with a single input group is the two-slot recipe of
[§11.4](#114-single-input-group-two-slot). A plan must fit the
384 KB ORAM ([§6.2](#62-oram)); a layer whose weight blob does not fit ORAM alongside the rest of the plan is
run by weight streaming ([§6.7](#67-weight-streaming),
[§11.8](#118-large-weights-and-channel-concatenation)). Drain routine and shared constants:
[§10](#10-the-recipe-template).

#### 11.2.2 Geometry and derived sizes

**Inputs.** `D = Cin/32` input channel groups ([§2.2](#22-channel-groups-of-32)), `Dout = Cout/32` output groups, `H×W` = the **output** size, stride `s` (input size `s·H
× s·W`), 4-bit activations, weights of `w_bits` and output of `out_bits` (4 or 8), zero-point `zp`.

**Row bytes.** `nmem_rb = round64(W_in·16)` (DDR row bytes of one NDHWC32 input plane, [§2.3](#23-ndhwc32-feature-maps); 16 = 32 ch × 4-bit input / 8), `oram_rb =
round64(((W_in+3)&~3)·16)` (ORAM ring row bytes), `out_rb = round64(W·32·out_bits/8)` and `stg_rb =
round64(((W+3)&~3)·32·out_bits/8)` (DDR and staging output row bytes:
16 B per pixel at 4-bit output, 32 B at 8-bit; all four coincide on a 4-bit stride-1 layer).

**Blocks and tiles.** Column blocks `CB = ceil(W/4)` (a block = 4 output px = `4s` input px), row pairs `RP = ceil(H/2)`. A tile ([§3.1](#31-the-tile)) is 2 rows × 4 px ×
32 channels of one output group; a block yields one tile per output group of the pass. The input window of a
row pair is `walk` rows starting at input row `2s·rp − 1`,
`s` chunks of 64 B per row at column `cb·64s`.

**Unit split ([§3.3](#33-the-two-execution-units)).** The input groups are divided between the two execution units: unit A takes groups `0 … D/2−1` (`nA = D/2`, group
count `A.1c`, activation port `0x20`, MAC phase `nnmac vw9,1`); unit B takes groups `D/2 … D−1` (`nB = D −
D/2`, group count `A.1d`, port `0x21`, MAC phase `nnmac
vw10,2`). The two counts are equal for every even `D`.

**Walk constants.** `K = 0x14`, `walk = 4` at stride 1; `K = 0x18`, `walk = 2s + 2 − (s−1) = 5` at stride 2; `K·walk = 0x50 / 0x78`. `rowu = in_bits·K·walk/8` (40 at
stride 1, 60 at stride 2) is the row unit of the walk program; `row_operand_units = in_bits·K·walk/2` (160 /
240). `B.11 = (nA · row_operand_units) >> 2` (80 at stride 1
and 120 at stride 2 for `D = 4`).

**Weight passes.** `tap_slots = D·9·w_bits/2` (72 with 4-bit weights, 144 with 8-bit), `groups_per_pass = gpp = 0x120 / tap_slots` (4, or 2 with 8-bit weights), `passes =
ceil(Dout / gpp)`; the last pass has `Dout − (passes−1)·gpp` groups. A layer with `Dout > gpp` runs several
weight passes per row pair (§11.2.5).

**Feed lead.** `lead = 2` at stride 1, `1` for a padded stride-2 layer (`2` again when the left pad is 0).

#### 11.2.3 ORAM plan and DMA chains

**ORAM plan, in allocation order ([§6.2](#62-oram)).** `[weights: w_bytes = D·Dout·9·128·w_bits` (512 B per tap with 4-bit weights, 1,024 with 8-bit)`][requant table:
Dout·256` (the same 1 KB for 4 output groups whatever the output width)`][feature ring: D groups × (3 + 3s)
rows × oram_rb][output staging: Dout planes × 2 buffers × 2
rows × stg_rb]`, plus 256 B of scratch for the first drain. A plan that does not fit is not run by this recipe
(§11.2.1).

**DESRAM chains ([§6.3](#63-descriptor-format) format; the length has 5 extra high bits in `lo[5:1]`).** Index 0–1: weights → ORAM 0 and table → ORAM `tbl`, kicked
(channel register IO+4, [§6.4](#64-channels)) before the routine starts. Index 2…: one feature chain per kick:
kick 0 = input rows `0 … s+1`, kick k = the next `2s` rows
(stride 1: rows `2k+1, 2k+2`; clipped at `H_in`), each row of each group one descriptor of `nmem_rb` bytes
into ring slot `row mod (3+3s)` (group-major, rows inner, all
chained but the last). The plan reserves `D·max(s+2, 2s)·n_fkick` slots, `n_fkick = 1 + ceil((H_in − s −
2)/(2s))` (= RP for a padded layer), then the output chains: kick
j = output rows `2j, 2j+1`, per plane one descriptor per row from staging `[plane][j mod 2][row]` to DDR `out
+ plane·H·out_rb + row·out_rb` (8 descriptors per kick for
Dout = 4, hence the index step of 8).

**Ring depth ([§6.6](#66-ring-buffers-and-whole-input-mode)).** The feature ring is `3 + 3s` rows deep per input group (slot = input row mod `3+3s`: 6 at stride 1, 9 at
stride 2). The ring's extra depth (9 rows at stride 2 against the 5 the window needs) is what makes the early
kick of §11.2.5 safe: kick `rp+2` overwrites rows `4rp−1 …
4rp+2`, which the feed finished reading at block `CB−1`.

#### 11.2.4 Prologue

The prologue begins with the guarded reset, `nnrwr B.1b ← 0 ; nncmd 0x00` ([§5.1](#51-reset)), and then
proceeds as follows.

1. `nncmd 0x00` (reset, [§4](#4-the-configuration-model)). The static fields ([§13](#13-configuration-fields);
the values of this recipe in
   [§15](#15-field-values-by-recipe)) are written with `nnrwr`. Among them: `A.03 = in_bits/2 − 1` (= 1 for 4-bit input), `A.0c/A.0e = H_in − 1 / W_in − 1` (the input
   size), `A.0f = zp`, `A.1c = nA − 1`, `A.1d = nB − 1`, `A.19 = K`, `A.17 = K·walk`, `B.07 = 2` at 4-bit output and `4` at 8-bit (the packed tile size in 64-byte units), and
   `B.11 = (nA · row_operand_units) >> 2`, written once here. `B.10` is never written and keeps its reset value 0.
2. Walk program ([§3.5](#35-the-walk-program); the decode of the words is in [§14](#14-the-program-table)).
Unit A's four entries `loop(end 3, nA)`, `step(walk−1, K/4)`,
   `step(1, rowu − (K·walk/4 − K/4))`, `move(s − nA·rowu)` go into `B.19` (four writes); then `B.18 ← 8`; then unit B's four entries — the same two step entries with
   `loop(end 3, nB)` and `move(s − nB·rowu)` — into `B.19` again; then `B.1a ← 0x44`. At stride 1 with two groups per unit the words are `0x7B01 0x1805 0x0819 0x03B1`; at
   stride 2 the row step becomes `0x824` (`rowu = 60`) and the move `0x38a` for two groups — the row step is `K/4`, so `5`/`6` and the `1`/`s` in the last entry are the
   only stride-dependent parts.
3. `nncmd 0x60` (arm).
4. The vr31 words the loop uses (the array samples the word file during `nnmac`): `liwr w2,0`; `law w3 ← −1`;
`law w5 ← 4s`; `liwr w7,0`; `law w4 ← tap_slots` (72); `law
   w8 ← 4·gpp` (`(mode+2)·2·groups`, mode 0); `law w9 ← −1`; `law w10 ← −1 + 4·B.11` (319 at stride 1); `law w6 ← 2·K·s`; `law w13 ← −2·K·s`; `law w14 ← 4s
   − 2·K·s` (40 / −40 / −36 at stride 1).
5. Feature kick 0.
6. **Weight preload** ([§5.5](#55-weight-preload); single-pass layers only — otherwise the per-pass path of
§11.2.5): for each of the `D·Dout` (in-group, out-group)
   blocks of the blob (`9·128·w_bits` bytes: 4,608 with 4-bit weights, 9,216 with 8-bit), in blob order, 9 taps of `128·w_bits` B (512 / 1,024) with the middle kernel row
   taken in reverse column order (`tap = 3·row + (row==1 ? 2−col : col)`); each tap goes in `w_bits/2` slices of 256 B, each slice = four 64-byte pushes to the weight
   ports from `vr0/vr1`: `nndwr vr0,0 ; nndwr vr1,2 ; nndwr vr0,4 ; nndwr vr1,7` (the slice loop runs `w_bits/2` times; 1,152 pushes for `D = Dout = 4` with 4-bit
   weights, and per pass on a layer with 8-bit weights).
7. **Requant table** ([§3.8](#38-requantization-and-packing)): pushed once per layer for all output groups —
per output group two 128-byte halves, each half `nndwr
   vr0,0x40 ; nndwr vr1,0x40`; `4·Dout` pushes, 16 for the 1 KB table of `Dout = 4`. It is not re-pushed per pass.
8. Feature kick 1.
9. **Pre-feed** ([§5.6](#56-activation-feed)): the activation windows of the first `lead` column blocks are
pushed before any MAC — unit B's whole list first (port
   `0x21`), then unit A's (port `0x20`).

#### 11.2.5 Tile loop

**Feed step.** The activation feed is positional ([§3.4](#34-ports-and-positional-feeds)); the routine keeps it exactly `lead` column blocks ahead of the MACs
([§5.6](#56-activation-feed)). Feed step `p` (0..Dout−1) of block `B = rp·CB + cb` pushes, for unit B then
unit A, rows `r0 = 2·rp − 1 + 2·(p mod 2)` and `r0+1` of group
`nA + p/2` (unit B, `nndwr vr0,0x21 ; nndwr vr1,0x21`) / `p/2` (unit A, `nndwr vr0,0x20 ; nndwr vr1,0x20`),
each push one 64-byte chunk = 4 pixels at column `cb·4` from
ring slot `row mod (3+3s)` (row −1 is the top-pad slot; its contents are masked by the window origin `A.09`
and the pad value `A.0f`). Over the four steps of a block the
whole 4-row × 4-pixel window of every input group has been pushed. In the tile loop **the feed issued after a
MAC is the other unit's feed** (after unit A's `nnmac vw9,1`
unit B's share, after unit B's `nnmac vw10,2` unit A's share), always for the column block `lead` blocks
ahead. The pushes of a block are not really "steps": the routine
builds, per unit, a table of `n_unit · walk · s` ORAM-pointer increments (`n_unit` = `nA` or `nB`) (one entry per 64-byte chunk:
group-major, window rows inner, `s` chunks per row) and consumes
`per_tile = ceil(n_unit·walk·s / groups_this_pass)` entries after each `nnmac` (rounded up to an even number —
two chunks per push pair, an odd tail pushes only `vr0`),
clamped at the table end — so the tiles of a block take unequal shares when `D ≠ Dout`. With `D = Dout` this
is the two-rows-per-step picture above. Past the last block
of the image the feed keeps counting into ring slots nothing reads.

**Main loop (general schedule; a single-pass layer has `passes = 1`, `k = 0`).**
```
for rp in 0..RP-1:                              # output row pair
    for k in 0..passes-1:                       # weight pass
        if passes > 1: nnrwr B.1c (w2 = 0) ; re-push the weights of pass k ; liwr w7,0 (k = 0) or addrw w7,w8
        law w0 ← 2s·rp−1 ; nnrwr A.0b (w3 = −1) ; nnrwr A.09 (w0)       # window origin (col, row), right before block 0
        for cb in 0..CB-1:                      # column block of 4 output pixels
            if k == passes−1 and cb == CB−lead: feature kick (rp+2)   # stride 1: rows 2rp+5,2rp+6 → slots of rows 2rp−1,2rp, just read
            nnrwr B.02 (w7 = 4·k·gpp)
            for p in 0..groups_k-1:             # one tile = 2 rows × 4 px × 32 ch of output group k·gpp+p
                nnmac vw9,1  ; unit B's share of feed block B+lead  (nndwr vr0,0x21 ; nndwr vr1,0x21 …)
                nnmac vw10,2 ; unit A's share of feed block B+lead  (nndwr vr0,0x20 ; nndwr vr1,0x20 …)
                nndrd vr8,0x20 → sao staging[k·gpp+p][rp mod 2][row 0] + cb·16·out_bits   # the PREVIOUS tile (§7.7)
                nndrd vr9,0x20 → sao staging[...][row 1] + cb·16·out_bits
                nncmd 0x8f                      # commit (§5.8)
                if this drain completed row pair rp−1 (k == 0, cb == 0, p == 0, rp > 0): output kick (rp−1)
                nnrwr A.14, A.12 (w4 = tap_slots)
            nnrwr A.10, A.11 (w2 = 0) ; addrw w9,w5 ; addrw w10,w5 ; nncmd 0xc0     # w9, w10 += 4s
final: one more nndrd vr8,0x20 / nndrd vr9,0x20 + sao (the last tile) ; output kick (RP−1)
```
Per tile the MAC order is unit A (`nnmac vw9,1`, groups `0 … nA−1`) then unit B (`nnmac vw10,2`, groups `nA …
D−1`) ([§3.6](#36-mac-phases-and-the-walk-position)). `w9 =
−1` and `w10 = −1 + 4·B.11` at the start, both `+= 4s` per column block; `A.09 = 2s·rp − 1` advances by `2s`
per row pair, `A.0b = −1`.

**Readout lag ([§7.7](#77-readout-lag-and-stale-reads), [§5.9](#59-drain)).** A drain returns the tile committed one tile earlier: the first drain of a run returns
nothing useful and is discarded into scratch, every later drain is stored to the previous tile's staging
address, and one extra drain follows the loop. Tile `t` (linear
over row pairs, passes, blocks, tiles) lands in staging slot `[plane = k·gpp + p][rp mod 2][cb]`; the output
kick of row pair `rp−1` follows the drain that completes it.
At 8-bit output the drain is the four-register form of [§10](#10-the-recipe-template) (`nndrd
vr8/vr9/vr10/vr11,0x20`) and the staging column step is 128 B per block
(`16·out_bits`).

**Several weight passes (`Dout > gpp`).** The blocks run in the order **(row pair, pass, column block)**: every row pair is repeated once per pass, and each block of pass
`k` has `groups_k` tiles (output groups `k·gpp …`). Before block 0 of every (row pair, pass) the routine
writes `B.1c ← 0` (word `w2`, always zero — `B.1c` is not a
weight offset here), re-pushes the weights of the pass's groups from ORAM (the blob slice of (out-group,
in-group) pairs `[k·gpp·D, (k·gpp+groups_k)·D)`, same tap order
and ports as the single-pass preload — so the blob is out-group-major), then `A.0b/A.09` as usual (`A.09` is
the same for every pass of a row pair). The commit window
steps by `4·gpp` per pass: `B.02 = k · 4 · gpp` (`w7`, reset by `liwr` at pass 0 and advanced by
`addrw w7,w8`, `w8 = 4·gpp`). In a multi-pass layer the
prologue pushes only the requant table, and the first pass's weights arrive through this per-pass path at row
pair 0, after the pre-feed. The activation feed simply
continues: it stays `lead` blocks ahead in the same (row pair, pass, block) order, so at a pass boundary the
last blocks of pass `k` feed the first blocks of pass `k+1`
of the *same* row pair (the same ring rows again), and no re-pre-feed happens; the per-tile share of a fed
block is split by the *consuming* block's `groups_k`. The
feature kick of the next rows is issued in the **last** pass of a row pair, before block `CB − lead`.

#### 11.2.6 Variants and not-characterised items

**Stride 2.** The same routine with `s = 2`: `K = 0x18`, `walk = 5`, `K·walk = 0x78`; static fields `A.19 = 0x18`, `A.17 = 0x78`, `A.1a = 0x11`, `B.1b = 1`, `B.11 = nA ·
(in_bits·K·walk/2) / 4` (120 for `D = 4`), `A.0c/A.0e = H_in − 1 / W_in − 1`; program entries as in §11.2.4
with `rowu = 60` (row step → `0x824`, move → `0x38a` for two
groups); vr31 words `w5 = 4s`, `w6 = 2·K·s`, `w13 = −2·K·s`, `w14 = 4s − 2·K·s`, `w8 = 4·gpp`,
`w10 = 4·B.11 − 1`, `w3 = w9 = −1`, `w2 = w7 = 0`. The feed
lead is one block for a padded layer: one block is pre-fed before the first MAC, the feature kick of a row
pair is issued before column block `CB − 1`, and the
`A.09/A.0b` write comes right before block 0.

**8-bit weights and 8-bit output (4-bit input).** A 3×3 s1 layer with 8-bit weights (`w_bytes = Cin·Cout·9`) and 8-bit output runs on this recipe; what changes follows
from the bit widths alone:

- `tap_slots = D·9·8/2 = 144` → `groups_per_pass = 2` → two weight passes for `Dout = 4` (the multi-pass schedule unchanged: `B.1c ← 0`, the pass's 8 (out, in) pairs
  re-pushed, `B.02 = 0 / 8`, table pushed once).
- Weight blob: 9,216 B per (out, in) pair, 1,024 B per tap, four 256-B slices per tap (1,152 pushes per pass).
- Config: `A.1b = 0x31`, `A.12/A.14 = 144`, `B.04 = 3`, `B.07 = 4`, `B.08/B.09 = 0x5410/0x7632`; **unchanged** because they depend on the input width only: `A.03 = 1`,
  `A.08/B.06 = 2`, `A.19/A.17 = 0x14/0x50`, `B.11 = 80`, `B.15 = 1`, the program words `0x7B01 0x1805 0x0819 0x03B1`, `B.1a = 0x44`, and all vr31 loop words.
- Output side: `stg_rb`/`out_rb` double (640 B at W = 20), the staging column step is 128 B per block (`16·out_bits`), a staging plane is `2 bufs × 2 rows × stg_rb`, and
  each tile is drained with the four-register routine (`nndrd vr8,0x20 ; nndrd vr9,0x20 ; nndrd vr10,0x20 ; nndrd vr11,0x20`). Feed side (4-bit input): ring, kicks, lead
  and feed tables all as at 4-bit.

**Not characterised** (collected in [Appendix C](#appendix-c--not-characterised)):

- Not characterised: whether `A.03` influences this recipe at all. What is established: `A.03 = in_bits/2 − 1` (1 for 4-bit input) at stride 1, the vendor writes 0 at
  stride 2, and both values produce identical output.
- Not characterised: the role of `B.1c` in this recipe, which writes it 0. What is established: the 1×1 recipe uses `B.1c` as the weight-stream offset ([§13.4](#134-per-tile-fields)); here it is written 0 (from `w2`) before every (row pair, pass) of a multi-pass layer and never
  any other value; it is not a weight offset.
- Not characterised: the effect of `nncmd 0xc0` at the end of a column block [L]. What is established: it is issued after the `w9/w10` step of every column block, and the
  recipe is complete with it in place.


### 11.3 Single input group, row quad

#### 11.3.1 Serves and guards

Where `Cin ≤ 32` there is one input channel group ([§2.2](#22-channel-groups-of-32)) and nothing to split between
the two execution units ([§3.3](#33-the-two-execution-units)). Both units are instead given the same group with
different **row** windows, so each produces a different pair of output rows and a block covers a **row quad** of four
output rows. Two recipes exist for this case and both are exact; this one serves 1×1 and 3×3 kernels at stride 1 and 2
(the two-slot alternative is [§11.4](#114-single-input-group-two-slot); the choice is [§9](#9-choosing-a-recipe)).
Per output group the array runs two MAC phases, `nnmac vw1,0` for output rows `4q, 4q+1` and `nnmac vw2,1` for rows
`4q+2, 4q+3`, both fed on port `0x20`; the walk program is a single slot and slot B (port `0x21`) is never pushed. [H]

#### 11.3.2 Geometry and derived sizes

A block is a row quad `q` (output rows `4q … 4q+3`) × a column block `cb` (4 output pixels); a column *chunk* is
4 input pixels = 64 B of one ring row. `in_rb` = the input tensor's DDR row stride (`Win·32·in_bits/8`, no rounding —
the ring row *is* the DDR row) and `stg_rb = round64(((Wout+3)&~3)·out_bits·4)`. The input ring holds `7s + K` rows —
for K = 3 the same formula as the `3 + 7s` of [§11.4](#114-single-input-group-two-slot). The window of quad `q` is the
`3s + K` ring rows starting at ring row `4s·q` (row `4s·q − pad` of the image; rows outside the image are masked by
`A.09/A.0c`, not zero-filled). The feed runs ahead of the MACs: `lead = s + pad` chunks are pushed at the start of
every quad, block `cb ≥ 1` pushes chunks `s·(cb − 1) + lead … s·cb + lead − 1`, and the last block pushes chunk index
`chunks` — **one chunk past the right edge** (the data is masked by `A.0e`; the vendor does not clip). [H]

#### 11.3.3 ORAM plan and DMA chains

**ORAM plan** (one 64-aligned allocation, [§6.2](#62-oram)): `[ring: (7s + K) rows × in_rb][weights:
K²·128·w_bits·Dout][staging: 2 buffers × 4·Dout rows × stg_rb][table: 256·Dout]`.

**DMA chains** ([§6](#6-memory-and-dma)). This recipe swaps the read channels relative to the 1×1 and head recipes:
descriptor 0 = weights (more), 1 = table (last), kicked once on **read channel 1** at descriptor index 0 (kick word
`0x80000000`, after the channel-idle read). Feature rows go through **read channel 0** and are always kicked at
descriptor index 2, whose slots are **rewritten before every kick**: the first kick is one descriptor of
`3s + K − pad` contiguous rows (5 / 8) into ring rows `pad …`; every later kick (issued right after the first `nnmac`
of each quad) is up to `4s` single-row descriptors (rows `loaded … loaded + 4s − 1` into ring rows
`(row + pad) mod ring_rows`, `more` on all but the last) until the input is exhausted
([§6.6](#66-ring-buffers-and-whole-input-mode)). Output descriptors start at index `2 + (3s + K)` (the maximum
feature descriptor count), 4·Dout of them per quad in the order (row 0: group 0, 1, …; row 1: …), staging buffer
`q mod 2`, each `out_rb` bytes, and are written after the `rdhwr $11` fence ([§6.5](#65-kick-wait-and-fencing)) and
kicked on the write channel. [H]

**Weights and table** ([§2.5](#25-weights), [§5.3](#53-program-loading)): once, before the first feature kick: for
each output group, the 9 taps in the zigzag order (kernel row 1 reversed), each `w_bits/2` slices of 256 B, each
slice `vr0 ← T, vr1 ← T+64 → nndwr vr0,0 ; nndwr vr1,2` and `vr2 ← T+128, vr3 ← T+192 → nndwr vr2,4 ; nndwr vr3,7`
(`vr0..vr3` on the weight ports `0/2/4/7` of [§3.4](#34-ports-and-positional-feeds), 512 / 1,024 B per tap); then
`B.1e = 0` and the table as `2·Dout` pairs `vr0 ← T, vr1 ← T+64 → nndwr vr0,0x40 ; nndwr vr1,0x40` (128-byte steps). [H]

#### 11.3.4 Prologue

The prologue begins with the guarded reset, `nnrwr B.1b ← 0 ; nncmd 0x00` ([§5.1](#51-reset)), and then
proceeds as follows.

**Configuration** ([§13](#13-configuration-fields), [§15](#15-field-values-by-recipe)): `A.0f = zp & 0xf` (4-bit
input; `& 0xff` at 8 bit), `A.1a = 0` for 1×1 else `(s_row − 1) << 4 | (s_col − 1)` (0 / 0x11), `B.04 = 1` (4-bit
out) / 3 (8-bit), `B.15 = 1` (4-bit in) / 2, `A.1b = (w_bits/2 − 1) << 4 | (in_bits/2 − 1)` (0x11 / 0x31),
**`A.19 = 0x14`, `A.17 = 0xc8`** — constants of the recipe, not of the layer: `A.19` (20) is the row pitch and
`A.17` (200) `= A.19 × (7·stride + K)` = 20 × 10 at stride 1 with K = 3, where the two-unit 3×3 recipes write `K` /
`Kwalk` instead — `A.08 = B.06 = 2` (3 when the layer's operator code `≠ 2`), `A.0e = Win − 1`, `A.0c = Hin − 1`
(input size), **`A.1e` = the tap mask** (bit `kh·3 + kw` set for every tap that dilation keeps: `0x1ff` for K = 3,
`0x1` for K = 1), `B.05 = 2`, `B.03 = 0` (from the layer's operator code: 0 / 1 / 2 / 100 for codes 2 / 6 / 0xb /
other), `B.1b = 0` (also at stride 2 — the `s − 1` of [§15](#15-field-values-by-recipe) belongs to the two-unit
recipes), then the walk program ([§3.5](#35-the-walk-program), [§14](#14-the-program-table)): **3×3 stride 1:
`0x1805 0x80a 0x1805 0xbd9`, `B.1a = 4`; stride 2: `0x2805 0x2005 0xbd4`, `B.1a = 3`; 1×1: `0x805 0x814 0x805 0xbe3`,
`B.1a = 4`**, a single slot in every case, `nncmd 0x60`, `B.1c = 0`. Fields the two-unit 3×3 recipes write and this
one does not: `A.01 A.18 A.03 A.1c A.1d B.07 B.08 B.09 B.11`. New here: `A.1e`, `A.19/A.17` as constants. Before the
arm the recipe also loads **`vr29` = the input zero-point byte replicated**, the clamp constant of the drain routine. [H]

**The vr31 word file at `nnmac` time** ([§3.6](#36-mac-phases-and-the-walk-position)): `w0` = the tap mask (the last
A value), **`w1 = 4s·cb − pad`** (the block's first input column; `addiw w1,w1,4s` per column block), **`w2 = w1 +
100`**, `w6 = 4s·q − pad` (the `A.09` value), `w7 = −pad` (`A.0b`), `w8 = 0` (`A.10/A.11`), `w9 = K²·w_bits/2` (tap
slots, the `A.12` step), `w10 = 4·g` (the `B.02` value of the group being committed; `addrw w10,w11` after each
group's first phase), `w11 = 4`, `w14 = B.1a`, all others 0 (never written by the vendor; left at zero). [H]

#### 11.3.5 Tile loop

Each chunk `c` is pushed with a fixed load/push pattern ([§5.6](#56-activation-feed)): ring rows are loaded 64 B at
a time into `vr0..vr3` with `lao` and pushed with `nndwr`; every push in this recipe goes to port `0x20` and port
`0x21` is unused. Key: `vrN <- r` = `lao vrN` of the 64-byte chunk of ring row `r` at column `c·64`; `P400 / P420 /
P440 / P460` = `nndwr vr0,0x20 / nndwr vr1,0x20 / nndwr vr2,0x20 / nndwr vr3,0x20`. A `P440 P460` following a
single `vr0 <- r4` load means `vr2`/`vr3` are re-pushed with their current content — a push takes the register's
bytes as they are ([§3.4](#34-ports-and-positional-feeds)) — so rows shared by the two phases are pushed twice and the
second push lands in a different positional slot: six ring rows become eight pushes at stride 1, nine become ten at
stride 2. [H]

```
3x3 s1 (6 rows, 8 pushes): vr0<-r0 vr1<-r1 P400 P420 | vr2<-r2 vr3<-r3 P440 P460 | vr0<-r4 P440 P460 | vr1<-r5 P400 P420
3x3 s2 (9 rows, 10 pushes): vr0<-r0 vr1<-r1 P400 P420 | vr2<-r2 vr3<-r3 P440 P460 | vr0<-r4 P400 P400 | vr2<-r5 vr3<-r6 P440 P460 | vr0<-r7 vr1<-r8 P400 P420
1x1   (4 rows, 4 pushes):  vr0<-r0 vr1<-r1 P400 P420 | vr2<-r2 vr3<-r3 P440 P460
```

Per block and group the array runs two phases, and a phase's result is drained after the *next* `nnmac`; the commit
is `nncmd 0x84 ; nncmd 0x8b` ([§5.8](#58-commit)) and the readout model is
[§3.7](#37-accumulators-commit-and-readout). `B.10 = 0` is written at the top of every row quad. [H]

```
quad q:   B.10 = 0 ; wait read channel 0 ; feed chunks 0..lead-1 ; [q > 0: drain rows 2-3 of (q-1, last group, CB-1) ; output kick q-1]
          A.09 = 4s*q - pad ; A.0b = -pad ; w1 = -pad, w2 = w1 + 100
block cb: [cb > 0: feed its chunks ; w1 += 4s, w2 += 4s]   A.11 = 0 ; A.10 = 0 ; nnmac vw1,0        (group 0, rows 0-1)
          [cb > 0: drain rows 2-3 of (cb-1, last group)]  [cb == 0: rewrite + kick the next 4s feature rows]
          w10 = 0 ; B.02 = w10 ; nncmd 0x84 ; nncmd 0x8b ; nncmd 0xa0 ; nnmac vw2,1                  (group 0, rows 2-3)
          drain rows 0-1 of (cb, 0) ; B.02 ; 0x84 ; 0x8b
          for g = 1..Dout-1: nncmd 0xe0 ; A.12 = w9 ; nnmac vw1,0 ; drain rows 2-3 of (cb, g-1) ; w10 = 4g ; B.02 ; 0x84 ; 0x8b
                             nncmd 0xa0 ; nnmac vw2,1 ; drain rows 0-1 of (cb, g) ; B.02 ; 0x84 ; 0x8b
          nncmd 0xc0
end:      drain rows 2-3 of the last block ; output kick of the last quad
```

`nnmac vw1,0` is the phase that produces the quad's rows 0–1, `nnmac vw2,1` rows 2–3 (each from 4 of the fed rows at
stride 1); `nncmd 0xa0` precedes every `vw2,1`, `nncmd 0xe0` switches to the next output group (with `A.12 = tap
slots`: the weight slot base advances by one group), `nncmd 0xc0` closes the block. The readout lag is absorbed by
carrying the last committed-but-undrained phase forward: the last group's second phase of block `cb` is drained after
the first `nnmac` of block `cb + 1`, or — at the end of the quad — after the lead feed of quad `q + 1`, followed by
that quad's output kick; no junk drain is issued, and the final drain and output kick follow the loop. `B.02 = 4·g`
is the requant-table window of the group being committed (256 B = 4 pushes per group,
[§3.8](#38-requantization-and-packing)). [H]

The drain is the shared routine of [§10](#10-the-recipe-template) and [§5.9](#59-drain); recipe-specific are the
clamp constant and the staging target. 4-bit: `nndrd vr10,0x20 ; nndrd vr11,0x20` (the packed FIFO),
`maxu4bi vr10/vr11, vr29` — a per-nibble unsigned max with the replicated zero point, i.e. a ReLU floor that is a
no-op at zp = 0 —, `sao vr10 → row r, vr11 → row r+1`, 64 B each. 8-bit: four registers, `maxub`, `vr10+vr11 →
row r` (128 B), `vr12+vr13 → row r+1`. Both store into staging `(q mod 2, group, row)` at column `cb · 16 · out_bits`. [H]

#### 11.3.6 Variants and not-characterised items

- Variants: K = 1 (`A.1e = 0x1`, `A.1a = 0`, 4-push feed) and K = 3 at stride 1 (`A.1e = 0x1ff`, `B.1a = 4`,
  8-push feed) or stride 2 (`B.1a = 3`, 10-push feed); 4-bit or 8-bit input (`A.0f`, `A.1b`, `B.15`, the drain
  routine — [§11.5](#115-8-bit-input-variants)) and 4-bit or 8-bit output (`B.04`, `stg_rb`).
- Not characterised: how the unpushed slot B (declared 0, which stores as 8) is treated when the second phase runs.
  What is established: this recipe writes `len_B = 0`, never pushes slot B, and is exact.
- Not characterised: what `A.19 = 20` and `A.17 = 200` mean to the array beyond `A.17 = A.19 × (7·stride + K)` at
  stride 1, K = 3, and why they stay fixed otherwise. What is established: every variant writes them and is exact.
### 11.4 Single input group, two slot

#### 11.4.1 Serves and guards

An alternative recipe for a 3×3 kernel with a single input group (`D = 1`). It loads the same walk program
into both program slots and drives the two execution units ([§3.3](#33-the-two-execution-units)) with the
commit pair `nncmd 0xaf` / `nncmd 0xeb`, so that one pass over the input ring produces four output rows.
Guards, in manual terms: `D = 1`; `Dout ≥ 2`; `Hout % 4 == 0` (the row quad); the planner's `D = 1` size gate
on `Dout · 9 · bits`; and a single weight pass, `Dout · tap_slots ≤ 0x120` with `tap_slots = 9 · w_bits / 2`
per output group (36 at 8-bit).
The stride `s` is equal in both axes and at most 2, and `Hout` is even at stride 2.
The vendor planner applies further descriptor-level guards, listed in [Appendix
B](#appendix-b--vendor-executor-map). Choosing between this recipe and
[§11.3](#113-single-input-group-row-quad) is covered in [§9](#9-choosing-a-recipe).

#### 11.4.2 Geometry and derived sizes

Planner records for `D = 1` [H]: `rk = 2`, so an input feature kick moves `2 · rk · s = 4s` rows; the ring has
`2 · (1 + 4s) + 1 − s = 3 + 7s` rows per group (10 at stride 1, 17 at stride 2); the first kick brings `3s +
2` rows; the output staging buffers hold `2 · rk = 4` rows each and an output kick writes 4 rows.

A block is one **row quad** `q` (output rows `4q … 4q+3`) × one column block `cb` (4 output pixels). The one
input group is fed to both execution units with different row windows: **unit A** takes window rows `r0 … r0 +
walk − 1` with `r0 = 4 · s · q − 1` and produces output rows `4q, 4q+1`; **unit B** takes rows `r0 + 2s …` and
produces output rows `4q+2, 4q+3`. Ring slot = row mod ring rows; row −1 maps to slot `ring_rows − 1` and is
masked by the coordinate registers ([§6.6](#66-ring-buffers-and-whole-input-mode)). A block has `2 · Dout`
tiles, ordered (group 0: rows 0–1, rows 2–3), (group 1: …).

Feed tables: per unit and block, `walk · s` 64-byte chunks (row-major, `s` chunks per row, ports alternating).
`lead` = 2 blocks at stride 1 / 1 at stride 2, as in [§11.2](#112-33-4-bit-operands). Tile `k` of a
block pushes share `k` of each unit's table; shares are `roundeven(ceil(walk · s / Dout))` chunks (example,
walk 5, stride 2, `Dout = 2`: 6 + 4).

#### 11.4.3 ORAM plan and DMA chains

The ORAM ([§6.2](#62-oram)) holds the input ring of `3 + 7s` rows for the single group, `Dout` staging planes
of two 4-row buffers each (`q mod 2`), and a scratch area for the first drains. The staging row pitch is
`stg_rb = round64(((W + 3) & ~3) · 32)`. Staging placement alternates by tile index parity within a block: an even
tile (unit A) lands in rows `4q, 4q+1` of the quad's buffer, an odd tile (unit B) in rows `4q+2, 4q+3`, at
column `cb · 128` of plane `k`.
Feature chains: kick 0 and kick 1 in the prologue, then one kick `fk[2 + q]` per row quad, the last being
`fk[n − 1]`; each feature kick reserves `D · max(3s + 2, 4s)` descriptor slots.
Output chains: one kick per row quad, issued after the store of the quad's last tile (two row quads behind the compute, as the 3×3 recipe's output kicks, [§11.2](#112-33-4-bit-operands)).

#### 11.4.4 Prologue

The prologue begins with the guarded reset, `nnrwr B.1b ← 0 ; nncmd 0x00` ([§5.1](#51-reset)), and then
proceeds as follows.

Configuration is written in the order `A.01 A.18 A.0c A.0e A.1b B.05 B.03 A.08 A.03 A.19 A.17 A.1a B.06 B.04
B.07 B.08 B.09 B.11 B.15 B.1b A.0f` with the formulas of [§15](#15-field-values-by-recipe), except: there are
**no `A.1c/A.1d`** (no unit split), and **`B.11 = row_operand_units / 4`** `= in_bits · Kwalk / 2 / 4` (60 at stride
2: one group's window, the unit-B start offset). `B.07` is handled as in [§11.2](#112-33-4-bit-operands).

Walk program ([§3.5](#35-the-walk-program), [§5.3](#53-program-loading)): two words per slot, `[(walk − 1) <<
11 | op, 0x800 | ((s − (Kwalk/4 − op)) & 0x3ff)]` with `op` = 5 / 6 by stride; the group-count word `0x7800 |
…` and the group-step word of the `D ≥ 2` program are absent.
The pair is pushed to slot A, `B.18 = 8`, the same pair to slot B, `B.1a = 0x22`. Example (walk 5, stride 2):
`0x2006 0xbea`.

Then `nncmd 0x60`; feature kick 0 (chain address with bit 31 set, written to the NNDMA kick register); the
weights once (`Dout` pairs × 9 taps × `w_bits / 2` slices × 4 ports, middle kernel row reversed, exactly as
[§11.3](#113-single-input-group-row-quad)); the table (`Dout · 2` pairs of `nndwr vr0,0x40` / `nndwr vr1,0x40`
pushes, 128-byte steps); feature kick 1;
then `lead` whole blocks of activation feed ([§5.6](#56-activation-feed)), unit A then unit B. Weights, table,
program and `A.14/A.12` never change after the prologue; the row coordinate is the only per-quad write.

#### 11.4.5 Tile loop

Per tile the phase order is unit A, then unit B. The feed issued after each MAC is the *other* unit's share,
one block ahead ([§3.4](#34-ports-and-positional-feeds)): unit A's feed goes to port `0x20` (`nndwr
vr0/vr1,0x20`), unit B's to port `0x21` (`nndwr vr0/vr1,0x21`).

```
[cb == 0]            nnrwr A.0b = −1 ; A.09 = 4·s·q − 1           (row coordinate per quad)
[cb == CB − lead]    feature kick fk[2 + q]                        (the last kick is fk[n−1])
                     nnrwr B.02 = 0
for k in 0..Dout−1:
    nnmac vw9,0  ; unit-B feed, share k of block n+lead ; drain (tile t−lag) ; nncmd 0xaf   (unit A: rows 4q, 4q+1)
    nnmac vw10,0 ; unit-A feed, share k of block n+lead ; drain (tile t−lag) ; nncmd 0xeb   (unit B: rows 4q+2, 4q+3)
    nnrwr A.14 = tap_slots ; A.12 = tap_slots
nnrwr A.10 = 0 ; A.11 = 0 ; nncmd 0xc0
```

Drains ([§5.9](#59-drain)) are the 8-bit four-register drain of [§11.3](#113-single-input-group-row-quad) into
staging `plane k, buffer q mod 2, rows 2u, 2u+1, column cb · 128` (`u` = 0 for unit A, 1 for unit B).
The drain-lag strategy is that of [§11.2](#112-33-4-bit-operands): the first `lag` drains go to scratch, each
later drain stores the tile `lag` behind through the store-deferred routine, and one extra drain after the
loop flushes the last tile; the output kick of quad `q` follows the store of its last tile
([§5.8](#58-commit), [§3.7](#37-accumulators-commit-and-readout)).

#### 11.4.6 Variants and not-characterised items

- Variant: the `D ≥ 2` recipe uses `nnmac vw9,1` / `nnmac vw10,2` and the single commit `nncmd 0x8f` in place of this recipe's pair codes.
- Not characterised: the meaning of `nnmac vw9,0` / `nnmac vw10,0` and `nncmd 0xaf` / `nncmd 0xeb` beyond "unit A / unit B phase, commit". What is established: `nnmac vw9,0` + `nncmd 0xaf` produces output rows `4q, 4q+1` from window rows `r0 …`, and `nnmac vw10,0` + `nncmd 0xeb` produces rows `4q+2, 4q+3` from `r0 + 2s …`, with the same two program entries in both slots. See [Appendix C](#appendix-c--not-characterised).
- Not characterised: why `B.1a = 0x22` selects both slots. What is established: the pair loads with `B.18 = 8` between the two pushes and `B.1a = 0x22` after the second [L].
### 11.5 8-bit input variants

#### 11.5.1 Serves and guards

An 8-bit input tensor does not require a different recipe. The 1×1 procedure of
[§11.1](#111-11-4-bit-operands) serves it with the substitutions in §11.5.2; the 3×3 procedure of
[§11.2](#112-33-4-bit-operands), the single-input-group variant and the image variant take 8-bit input the
same way. The array reads an 8-bit activation stream as elements twice as wide; the recipe's structure is
unchanged. [H]

#### 11.5.2 Geometry and derived sizes

Everything not listed here is the 4-bit recipe verbatim: plan order, DESRAM layout, pass split, the
`nnmac vw4,1 / vw5,3` pair, the commit pair and the drain lag.

| Item | 4-bit input | 8-bit input | Rule |
|---|---|---|---|
| DDR and ORAM input row bytes | `W·16` | `W·32` | the tensor's dim-2 stride ([§2.3](#23-ndhwc32-feature-maps)) |
| Column-block step within a row | 64 | 128 | `stride · 0x80 · in_bits / 8` |
| Windows per plane per tile | 2: `vr0 ← row0`, `vr1 ← row1` | **4**: `vr0 ← row0`, `vr1 ← row0+64`, `vr2 ← row1`, `vr3 ← row1+64` | a row pair is `in_bits/2` pushes per plane |
| `A.03` | 1 | 3 | `in_bits/2 − 1` |
| `A.1b` | `0x11` | `0x13` | `(w_bits/2−1)<<4 \| (in_bits/2−1)` |
| `B.15` | 1 | 2 | input tile class ([§13](#13-configuration-fields)) |
| `w3` (→ `B.11`) | `2 + 4·nA` | `2 + 8·nA` | `2 + in_bits·nA` |
| `w5` (unit B's MAC word) | `16·nA + 8` | `32·nA + 8` | `4·in_bits·nA + 8` |
| Walk-program words, `n ≥ 2` groups | `0x7b00\|(n−2) 0x801 0x803 0x801 0x800\|(3−4n)` | `… 0x807 … 0x800\|(7−8n)` | step `0x800\|(in_bits−1)`, rewind `0x800\|(in_bits−1−in_bits·n)`; `n < 2`: `0x801, rewind` (both widths) |
| Whole-input ring rows | `D·4·W·16` | `D·4·W·32` | the plan formulas take the row bytes |

The walk program's step and rewind operands are in units of `in_bits` per input group (4 at 4-bit, 8 at
8-bit — the same unit as `B.11 = 2 + in_bits·nA`), which is why the 4-bit values are `3` and `3 − 4n` and the
8-bit ones `7` and `7 − 8n`. [H]

#### 11.5.2a The 3×3 at 8-bit input: three things this table does not cover

Both device-measured on 2026-09-11, bringing up [§11.2](#112-33-4-bit-operands) at 8-bit input, 8-bit weights
and 8-bit output (`runtime/exec/plt_conv_k3.c`). Each was wrong first and the failure is recorded because the
symptom identifies it.

**What scales with `in_bits`, and what does not.** Only `row_program_units` = `in_bits·K·walk/8` scales, and
with it `B.11` = `nA · row_program_units` and the program's last entry `move(s − n·row_program_units)`. The
walk program's *step* operands stay `K/4` and `K·walk/4`, and the per-column-block advance of `w9`/`w10`
stays `4s`, at both widths. Scaling the block advance to `8s` left only the first block of a row correct;
scaling the step operands broke the first block too. [H]

**`w9` and `w10` run for the whole layer.** They are initialised once in the prologue and advance `4s` per
column block from there — they do **not** restart at a row pair or a pass. Restarting them per row pair gives
a first row pair that is byte-exact and every later one wrong, which is a useful signature: it means the
operand walk, not the feed. [H]

**The packed drain order differs from the 1×1's.** At 8-bit output the four readout registers of a 3×3 tile
alternate the tile's two ROWS first and its two column halves second:

| register | tile row | tile pixels |
|---|---|---|
| `vr10` | 0 | 0–1 |
| `vr11` | 1 | 0–1 |
| `vr12` | 0 | 2–3 |
| `vr13` | 1 | 2–3 |

[§10](#10-the-recipe-template)'s "`vr10`+`vr11` to row *r*, `vr12`+`vr13` to row *r+1*" is the 1×1's packing
and is exact there. Using it for a 3×3 puts the next row's first half where the current row's second half
belongs — every tile half right, in a regular pattern. [H]

#### 11.5.3 ORAM plan and DMA chains

Unchanged except through the row bytes: every plan formula that takes `W·16` takes `W·32`, and the
column-block step within a row doubles (table above).

#### 11.5.4 Prologue

Field differences: `A.03`, `A.1b`, `B.15`, `B.11` (`w3`), unit B's MAC word (`w5`) and the walk-program
step/rewind words — all in the table above; per-recipe values in [§15](#15-field-values-by-recipe).

#### 11.5.5 Tile loop

Each push is one 64-byte window on the unit's activation port (`nndwr vr0..vr3,0x20` for unit A,
`nndwr vr0..vr3,0x21` for unit B; [§5.6](#56-activation-feed)). At 4 bits a tile row is one window, at 8 bits
two, so a plane's row pair is `in_bits/2` pushes: four windows per plane per tile instead of two. The
8-bit-output store drains `vr10`–`vr13`, applies `maxub`, and writes rows `d`, `d + 64`, `d + stg_rb`,
`d + stg_rb + 64` ([§2.6](#26-output-formats), [§3.8](#38-requantization-and-packing)). [H]

#### 11.5.6 Variants and not-characterised items

- 8-bit input with the 3×3, single-input-group and image procedures: the same substitution table applies. [H]
- Not characterised: input widths other than 4 and 8. What is established: the rules above are written in
  terms of `in_bits` and hold at both 4 and 8 ([Appendix C](#appendix-c--not-characterised)).
### 11.6 The RGB image layer

#### 11.6.1 Serves and guards

The first layer of a network takes a 3-channel image and does not fit the array's channel-group model
([§2.2](#22-channel-groups-of-32)). It is run as a **1×1 convolution over 32 "channels" of
8-bit activations**, with the 3×3 window over three colour channels assembled on MXUv3 before each push. To the NNA
this is the single-input-group family ([Chapter 9](#9-choosing-a-recipe)): `A.1c = 0` (one input group
after the im2col; the 27 taps + 5 pads form one 32-channel vector, so the 3×3 runs as a 1×1), `A.1e` = tap mask
`0x1`; the two MAC phases `nnmac vw1,0` and `nnmac vw2,1`; `nncmd 0x84 ; nncmd 0x8b` commits; bank-1 packed drains
`nndrd vr,0x20`; hardware requant with the 256-B `[32×int32 bias][32×int32 mult]` table
([§3.8](#38-requantization-and-packing)). This recipe produces ONE output row per MAC phase.

Guards: 3-channel 8-bit input stored as 4 bytes per pixel (B G R x, with the fourth byte 0 — the feed block relies on
it); 3×3, stride 2, pad 1; 4-bit weights; 4-bit packed output, 32 output channels; square input.

#### 11.6.2 Geometry and derived sizes

For a 640×640 image: `A.0c = A.0e = 639` (input size − 1); one image row is 640 px × 4 B = 2560 B; the output is
320×320 × 32 ch × 4 bit, one output row = 5120 B (`0x1400`); 160 output row pairs, 40 tiles of 8 output pixels per
row, 128 B of packed output per row per tile.

The window is built **on the MXU**: per output pixel a 32-byte activation vector holding the 27 real taps and 5 zero
pads, in this fixed order (`t` = byte, `dy/dx` relative to the pixel's centre input row/column `2·oy / 2·ox`):

| t | tap | t | tap | t | tap | t | tap |
|---|---|---|---|---|---|---|---|
| 0–2 | (−1,−1) B G R | 8–10 | (−1,+1) B G R | 16–18 | (+1,−1) B G R | 24–26 | (+1,+1) B G R |
| 3 | (0,−1) B | 11 | (0,−1) R | 19 | pad 0 | 27 | pad 0 |
| 4–6 | (−1,0) B G R | 12–14 | (0,0) B G R | 20–22 | (+1,0) B G R | 28–30 | (0,+1) B G R |
| 7 | (0,−1) G | 15 | pad 0 | 23 | pad 0 | 31 | pad 0 |

The 512-B weight blob is the matching 32-tap × 32-cout 4-bit tile ([§2.5](#25-weights)): **two 2-bit planes** —
bit-pair `m = cout·32 + t` of the low plane (bytes 0–255) and of the high plane (bytes 256–511), weight =
`(lo | hi << 2) − 8` [H]. The pad taps carry code 0 (= −8) and multiply zero activations. The blob is fed verbatim (two
rounds of `lao` 128 B → `nndwr vr22,0 ; nndwr vr0,2 ; nndwr vr29,4 ; nndwr vr30,7`, i.e. row-group 0, column groups
0–3; [§3.4](#34-ports-and-positional-feeds)).

#### 11.6.3 ORAM plan and DMA chains

ORAM footprint 133,888 B ([§6.2](#62-oram)). Image rows go on **read chain 0**, one 2560-B row per descriptor,
into a 40-slot ring (`slot = (16 + row) mod 40`, slot stride `0xB00`, data landing at `slot·0xB00 + 0x40`;
[§6.6](#66-ring-buffers-and-whole-input-mode)); the table (descriptor 20, chained)
and weights (21) go on read channel 1 to ORAM `0x1ba00` / `0x1b800`; the two staging rows (`0x1bb00` for output
row `2r`, `0x1cf00` for `2r+1`, double-buffered by row-pair parity at a distance of `0x2800`) go out on the write
chain (descriptors 22/23) after each row pair. Rows 0–3 are kicked first, then 4–23, then 20 rows after row pairs 1, 6,
11, … . The recipe waits for both initial kicks before the first pre-feed
([§6.5](#65-kick-wait-and-fencing)); without the first wait, output rows 0–2 — exactly the outputs
that read image rows 0–3 — come out wrong.

**Window rows and pointers.** Output row pair `rp` reads input rows `4rp−1 … 4rp+3` (five rows, `luo vr1..vr5`); rows
{−1, 0, +1} of the pair are convolved into output row `2rp`, rows {+1, +2, +3} into `2rp+1`. Real rows are read at
`slot·0xB00 + 0x3c`, i.e. one pixel before the data, so the left pad pixel is the zeroed 4-byte prefix; the top pad
row (row −1) is read at slot 15's base, zeroed ORAM. Padding is therefore zero memory (pad value 0), and the ring is
zeroed once. The column walk is the `luo` base auto-advance (+32 B per lane, +64 B per row per tile). A pre-feed before
every row pair builds tile 0's window; inside the loop each tile builds the *next* tile's window (one-tile pipeline), so
tile `cb` loads at `+64·(cb+1)` and its `law [14]` fills read `+64·(cb+2)`. The MAC column word `vw1 = 64·cb`
(`vw2 = vw1 + 4`) steps with `addrw` ([§3.6](#36-mac-phases-and-the-walk-position)).

#### 11.6.4 Prologue

Fields are named as in [Chapter 13](#13-configuration-fields). The sequence: `B.1b = 0` immediately
before `nncmd 0x00` ([§7.1](#71-the-b1b-reset-hazard), [§5.1](#51-reset)); `B.03 = 0`; DMA
(weights, table, first image rows); `A.0c = A.0e = 639`, `A.08 = B.06 = 2`, `A.1c = 0`, `A.19 = 8`, `A.17 = 16`,
`A.09 = A.0b = 0` (the walk starts at (0, 0); padding comes from the zeroed ring prefix, not from the array),
`A.18 = 0x12`, `A.1e = 1` (**one tap**), `A.1b = 0x13` (4-bit weights, 8-bit input), `B.04 = 1` (4-bit output),
`B.15 = 2` (8-bit input class, 256-byte input tiles), `A.01 = 3`; weight feed; `B.1b = 1`, `B.05 = 2`; program
`B.19 ← 0x802, 0x80e` — a single slot of two entries, `B.1a = 2` (steps +2 and +14;
[§5.3](#53-program-loading), [§3.5](#35-the-walk-program)); `nncmd 0x60`; table feed
`nndwr vr22,0x40 ; nndwr vr0,0x40` ×2 (bank B = the requant table, 256 B). `B.07` is not written. Per tile
`A.11 = A.10 = 0` and `B.02 = 0`; per row pair `B.10 = 0` before the pre-feed. `B.1b = 1` is the MAC-stage
half-balance control of [§13.3](#133-static-fields), written for stride 2 as the
3×3 stride-2 recipe does; the other recipes write 0.

#### 11.6.5 Tile loop

Each row pair opens with `B.10 = 0` and a pre-feed that builds tile 0's window. Per tile, in order: `nnmac vw1,0`;
`luo` the next tile's five rows; `B.02 = 0` (one weight pass, so the commit window is 0); `nncmd 0x84`;
`A.11 = A.10 = 0`; `nncmd 0x8b` ([§5.8](#58-commit)); `nnmac vw2,1`; the feed block for the first output row of
the *next* tile; `nndrd vr22,0x20 ; nndrd vr0,0x20`; `nncmd 0x8b`; floor and store; `nndrd vr22,0x20 ;
nndrd vr0,0x20`; push; the feed block for the second output row; floor and store; push; `addrw`. The first phase,
`nnmac vw1,0`, and the second, `nnmac vw2,1`, each produce one output row of the pair; the drain returns them into
`vr22` and `vr0`, which are stored to the staging rows for output rows `2r` and `2r+1` respectively.

**The feed block** (per output row of a tile, official names; the MXUv3 operations are documented in
`MXUV3_ISA_HARDWARE_RE.md`):
```
bshri vr19,vr1,8 ; bshri vr20,vr2,8 ; bshri vr21,vr3,8        # rows shifted two pixels left (byte shift by 8)
law vr19[14],0(row0+64) ; law vr20[14],0(row1+64) ; law vr21[14],0(row2+64)   # word 14 <- the tile's 16th pixel
[cmvw vr1,vr30,vr14,0 ; cmvw vr2,vr30,vr11,0 ; cmvw vr19,vr30,vr14,0 ; cmvw vr20,vr30,vr11,0]   # top-edge mask, rp 0 only
andv vr19,vr19,vr6 ; andv vr21,vr21,vr6 ; andv vr1,vr1,vr6 ; andv vr22,vr3,vr6      # vr6 = ff ff ff 00 per quad
ilveq vr15,vr1,vr22 ; ilveq vr16,vr19,vr21 ; ilvoq vr18,vr22,vr1 ; ilvoq vr19,vr21,vr19
gshufwb_1 vr22,vr8,vr2 ; gshufwb_1 vr0,vr8,vr20 ; gshufwb_1 vr29,vr7,vr2 ; gshufwb_1 vr30,vr7,vr20
orv vr22,vr15,vr22 ; orv vr0,vr16,vr0 ; orv vr29,vr18,vr29 ; orv vr30,vr19,vr30
ilveo vr15,vr22,vr0 ; ilveo vr16,vr29,vr30 ; ilvoo vr18,vr0,vr22 ; ilvoo vr19,vr30,vr29
nndwr vr15,0x20 ; nndwr vr16,0x20 ; nndwr vr18,0x20 ; nndwr vr19,0x20     # pixels 0-1, 2-3, 4-5, 6-7 (32 B each)
```
All activation pushes go to port `0x20` ([§5.6](#56-activation-feed)). `ilveq/ilvoq` put rows −1 and +1
(every 4th pixel zeroed) into alternating quads; `gshufwb_1` with a static 128-byte table (`vr8` = words 0–15, `vr7` =
16–31) spreads row 0's B, G, R bytes into the zeroed 4th bytes of row −1's three pixels and copies the centre pixels
whole; `orv` merges (this relies on the image's 4th byte being 0); `ilveo/ilvoo` order the 8 pixels. Constants:
`vr6 = ff ff ff 00` per quad, `vr11 = 0xffffffff`, `vr14 = 0`, `vr30 = repiw(pad word) = 0`, `vr13 = 0` (the `maxu4bi`
floor). The bracketed `cmvw` line (`cmvw vd,vs,vt,0` moves words where `vt` is 0) replaces the row −1 vectors by the pad
value on row pair 0 only; with pad 0 and zeroed ORAM the edge masks are no-ops.

**Drain, pack, store** ([§5.9](#59-drain), [§3.7](#37-accumulators-commit-and-readout)).
Per tile: `nndrd vr22,0x20 ; nndrd vr0,0x20` twice (bank 1, R = 0, 64 B = 4 pixels × 16 B each), `maxu4bi vr,vr13`
(floor 0, a no-op), `sao vr22[0]/[1]` → staging for row `2r`, `sao vr0[0]/[1]` → staging for row `2r+1`, 128 B per
row per tile. No software requant: the two-slope MXU requant block never executes for this layer. After the last tile
the two staging rows are written back (descriptors 22/23) and the next row pair's `B.10 = 0` and pre-feed follow.

#### 11.6.6 Variants and not-characterised items

Variants: only the 640×640 BGRx, stride-2, 32-cout geometry is characterised; other input sizes change `A.0c/A.0e`,
the ring schedule and the staging offsets. A bottom-edge variant of the `cmvw` mask never runs at 640 rows.

Not characterised: why the walk program is a single slot of two entries (`0x802, 0x80e`, `B.1a = 2`) rather than a
per-tile step. What is established: the reference implementation loads exactly those two entries and the layer's
output is correct with them ([Chapter 14](#14-the-program-table)).

Not characterised: what `B.1b = 1` selects in the MAC stage. What is established: it is written for stride 2 here and
in the 3×3 stride-2 recipe, 0 elsewhere, and it must be re-written to 0 before the next `nncmd 0x00`.

Not characterised: which MAC phase produces which of the two output rows. What is established [L]: the first drain
register `vr22` always holds row `2r` and `vr0` row `2r+1`; the vendor's naming of the two units is inconsistent, so
the recipe is written by MAC word and register.
### 11.7 Floating-point output heads

#### 11.7.1 Serves and guards

Serves 1×1 layers with 8-bit input and 8-bit weights padded to 32 output channels, whose output is fp32 in NHWC
layout with the real channel count (commonly 1, 4 or 6). The NNA commits raw int32 accumulators; requantization
and the store are performed on MXUv3 with vector stores, and there is no output DMA [H]. The recipe uses the
general int8 MAC phases `nnmac vw2,1` / `vw2,3` and `nnmac vw14,1` / `vw14,3`, commits with `nncmd 0x83`, and
takes readout path 1 ([§3.7](#37-accumulators-commit-and-readout), [§3.8](#38-requantization-and-packing)).

Guards: kernel 1×1, no padding, strides below 3, 8-bit operands on both sides; the heads use no activation, so the
ReLU floor is `−FLT_MAX` [H]. The output tensor's NHWC layout flag selects this recipe: `Dout = ceil(cout/32)` with
the **real** `cout`, shape (1, H, W, cout). `D = Cin/32`; unit A gets `nA = D/2` groups, unit B `nB = D − nA`
([§3.3](#33-the-two-execution-units)). With `D = 1`, `nA = 0`: unit A is skipped entirely — no `B.10 = 0`, no
`A.1c/A.1d` write, no `nnmac vw2,…` — and unit B carries the single group [H].

Both units are fed through port `0x20` (`nndwr vr0..vr3,0x20`); the stream is not split by port. The unit is
selected by the MAC phase and by the start offset written to `B.10` before each unit's feed — `B.10 = 0` before
unit A's planes, `B.10 = 8·(nA+1)` before unit B's. `B.11` is never written [H].

#### 11.7.2 Geometry and derived sizes

The 8-bit input row is `rb = W·32` bytes. A tile is 4 pixels × 2 rows, so a row pair holds `W/4` tiles
([§3.1](#31-the-tile)). A staging row is `round64(W·32·4)` bytes: W pixels of 32 fp32 channels, 128 bytes per pixel. A
unit's groups are walked in chunks of at most 16: `cA`, `cB` are the group counts of a full chunk of unit A and
unit B, `lA`, `lB` those of each unit's last chunk [H]. Whole-input mode holds iff
`0x60000 − 0x40 − 256 − D·H·rb ≥ 0` and one group's weights plus the staging still fit — 20² and 40² inputs are
whole, 80² is a ring of two row pairs (`D·4·rb` bytes) [H].

#### 11.7.3 ORAM plan and DMA chains

**ORAM** (one 64-aligned block, [§6.2](#62-oram)): `[input][weights D·1024][staging: 2 parities × 2 rows][table 256]`.

**Descriptors.** Index 0/1 = chain 0 (table → `o_tbl`, weights → `o_w`; kicked in the planner). Input descriptors
are **one row each**, `2·D` per row pair, plane-major, row inner: whole mode rewrites index 2.. for every pair (rows
land at `pl·H·rb + row·rb`), ring mode pre-writes all pairs at `2 + 2·D·rp` (rows at
`(rp%2)·D·2·rb + pl·2·rb + r·rb`). Output descriptors follow at `2 + 2·D·RP` (a 32-channel float row of `W·128`
bytes per output row from the staging parity) — written, never kicked. Kick = `0x80000000 | index` on IO+4
(features) / IO+0 (chain 0); whole mode kicks pair 0 in the planner, ring mode after `nncmd 0x60`; at the start
of every row pair the recipe waits and kicks the next pair (lead 1) ([§6.5](#65-kick-wait-and-fencing)) [H].

#### 11.7.4 Prologue

The prologue begins with the guarded reset, `nnrwr B.1b ← 0 ; nncmd 0x00` ([§5.1](#51-reset)), and then
proceeds as follows.

**Config**, in order ([Chapter 13](#13-configuration-fields), [Chapter 15](#15-field-values-by-recipe)): `A.0c = H−1`, `A.0e = W−1`,
`A.08 = 2`, `B.06 = 2`, `A.19 = 4`, `A.17 = 8`, `B.04 = 3`, `B.15 = 2`, `A.1e = 1`, `A.1b = 0x33`, `A.03 = 3`,
`A.0b = 0`, `A.09 = 0`, `A.01 = 3·(D odd)` (`(D%2) | (D%2)<<1`), `law vr20` ← the ReLU floor (`−FLT_MAX` =
`0xff7fffff` on the heads, else 0), `A.18 = 0x12`, `B.05 = 2`, `law vr30` ← the accumulator shift (1),
`B.1b = 0`, `B.19 = 0x801`, `B.19 = 0x807` (a single slot of two entries shared by both units: step one window, advance seven; no rewind words, [§3.5](#35-the-walk-program)), `B.1a = 2`, `nncmd 0x60`. No `B.03`, no `B.07`, no `B.11` [H].

Then the unit words: `w3 = 32·cA`, `w5 = 32·lA`, `w7 = cA−1`, `w9 = lA−1` for unit A (written only when
`nA > 0`); `w4 = 32·cB`, `w6 = 32·lB`, `w8 = cB−1`, `w10 = lB−1` for unit B; `w13 = 8·(nA+1)` (unit B's weight
bank, written to `B.10` before its feed; unit A's is `B.10 ← w1 = 0`); `w14 = 32·(nA+1)` (reloaded before every
unit-B MAC phase). `B.1c ← w1 (0)`, then the weights: `D·4` rounds of four 64-byte windows,
`nndwr vr0,0 ; vr1,2 ; vr2,4 ; vr3,7`, exactly as [§2.5](#25-weights) and [§5.3](#53-program-loading) [H].

#### 11.7.5 Tile loop

Per row pair, `W/4` column blocks `cb`. For each block: `B.10 = 0` (only when `nA > 0`); feed unit A's planes (per
plane `vr0..vr3` ← row 0 lo/hi, row 1 lo/hi at `row0 + pl·slot + cb·128`, four pushes `nndwr vrN,0x20`); if not
the first block of the pair: reload the table (`lao vr22..vr25` ← `o_tbl+0x00/0x40/0x80/0xc0`, because the requant
chain shares these registers) and `nncmd 0x83` (commit + clear, [§5.8](#58-commit)) for the **previous** tile;
`A.11 = A.10 = 0`; unit A's MAC phase; `B.10 ← w13`; feed unit B's planes through the same port; unit B's MAC
phase; then **drain the previous tile** ([§5.9](#59-drain)). After the last block: table, `nncmd 0x83`, drain, copy
out [H].

**MAC phases** ([§3.6](#36-mac-phases-and-the-walk-position)). `A.1c` and `A.1d` are rewritten before each phase and
mean *chunk sizes* here, not per-unit group counts: unit A writes `A.1c ← w7` (`= cA−1`), `A.1d ← w9` (`= lA−1`),
sets `w2 = 0`, and issues `nnmac vw2,1` for every inner chunk with `w2 += w3` after each, then `nnmac vw2,3` for
the last chunk. Unit B loads `w14 = 32·(nA+1)`, writes `A.1c ← w8` (`= cB−1`), `A.1d ← w10` (`= lB−1`), and issues
`nnmac vw14,1` per inner chunk with `w14 += w4`, then `nnmac vw14,3`. A unit with a single chunk issues only the
`,3` form [H]. The vr31 file at every `nnmac` is therefore
`(0, 0, w2, 32·cA, 32·cB, 32·lA, 32·lB, cA−1, cB−1, lA−1, lB−1, 2, 2, 8·(nA+1), w14, 0)`, where `cA`, `cB` are
the per-unit chunk sizes and `lA`, `lB` the last-chunk sizes; `w14 = 0` before the very first MAC and the last
unit-B value afterwards (it is never reset before a unit-A MAC) [H].

**Drain and requant** ([§3.8](#38-requantization-and-packing)). Sixteen row-banks of readout bank 0 per tile:
R = 0–7 = row 0, R = 8–15 = row 1, two banks per pixel (channels 0–15 and 16–31), each an int32 × 16-lane vector:

```
nndrd vr10..vr17, R                       R = 0..7                     (then 8..15)
sllw  vrN, vrN, vr30                      x = acc << shift             (shift = 1)
addw  vrN, vrN, vr22 | vr23               x += bias[ch]                (int32, table +0x00: vr22 = channels 0-15, vr23 = 16-31)
ffsiw vrN, vrN                            y = float(x)
fmulw vrN, vrN, vr24 | vr25               y *= scale[ch]               (fp32, table +0x80)
fmaxw vrN, vrN, vr20                      y = max(y, floor)            (-FLT_MAX: no ReLU)
sao   vr10..vr17 → staging row + cb·512   (128 B per pixel)
```

So the table of path 1 ([§2.6](#26-output-formats)) is exactly `int32 bias[32]` at 0 and `fp32 scale[32]` at 0x80
(only `cout` entries non-zero), and the requant is `((acc << 1) + bias) · scale` in fp32 with one rounding at the
multiply: the accumulator is the plain int32 dot product of the 8-bit operands [H].

**Copy-out**: per pixel the `cout` floats are moved as full 32-channel groups (four 32-byte `lao/sao` window-3
pairs, next group at staging `+2·stg_rb`) plus pieces derived from `bits = (cout%32)·32`: `bits>>8` × 32 B,
`(bits%256)>>7` × 16 B (window 1 = `laq/saq`), `(bits%128)>>6` × 8 B (window 0 = `lad/sad`), `(bits%64)>>5` × 4 B
(`law/saw` word 0) — cout 6 = 16 + 8, cout 4 = 16, cout 1 = 4. A final write-channel wait ends the procedure (no
output DMA was kicked) [H].

#### 11.7.6 Variants and not-characterised items

Variants: `D = 1` (unit B alone, 11.7.1); whole-input versus ring input (11.7.2); `cout` of 1, 4 or 6 (11.7.5).

- Not characterised: whether the NNA accepts a program of more than one slot for this recipe. What is
  established: a single slot of two entries with `B.1a = 2` is sufficient for every head geometry.
### 11.8 Large weights and channel concatenation

Two modifiers apply on top of a geometry's recipe ([§11.1](#111-11-4-bit-operands), [§11.2](#112-33-4-bit-operands))
without changing its program words, feed or drains: per-pass weight streaming when the blob does not fit
ORAM, and gathering the input planes from several tensors when the input is a channel concatenation.

#### 11.8.1 Serves and guards

- **Large weights.** Applied when the weight blob alone exceeds ORAM ([§6.2](#62-oram)), so the resident
  plan of [§5.5](#55-weight-preload) is impossible; the weights are streamed per pass ([§6.7](#67-weight-streaming)). [H]
- **Channel concatenation.** Applied when a layer has several input tensors. A concat is neither an operator
  nor a data movement: the layer has `D = Σ D_j` input planes ([§2.2](#22-channel-groups-of-32)) whose DDR
  addresses come from several tensors, which need not be adjacent in memory. [H] All multi-input layers of
  the reference network are 1×1 with `Cin ≥ 64`, so the modifier is established on the [§11.1](#111-11-4-bit-operands) recipe.
- **Guard.** The per-plane table of §11.8.3 is carved from the CPU scratch pool named by the layer
  descriptor; the pool-size check that fails returns error 8. [H]

#### 11.8.2 Geometry and derived sizes

Neither modifier changes a geometry field. Streaming derives only the pass count and per-pass blob size
from the ORAM budget ([§6.7](#67-weight-streaming)); concatenation feeds the summed `D` into the pass split
and program words, which know nothing about the concat. [H]

**Padding is implicit** in the im2col variant of §11.8.6: it rejects any explicit pad — the layer
descriptor's pad vector `[top, bottom, left, right]` must be all zero, the dilation vector `[1, 1]`, each
stride below 3, and the kernel size is read from the descriptor's kernel field. A "same"-padded stride-2
3×3 (40 → 20) derives its padding from the kernel, not from a stored pad; the model format carries no
separate pad operator. This is why the recipe accepts a 40×40 input and still produces 20×20. [H]

#### 11.8.3 ORAM plan and DMA chains

Before planning ORAM the recipe carves a **per-plane DDR address table** of `D` words from the CPU
scratch pool and fills it by walking the input vector in model order. From the runtime's tensor descriptor
it reads `t->shape[1]`, the plane count `D_t` (shape struct `N, D, H, W`, [§2.3](#23-ndhwc32-feature-maps));
`t->shape[2]`, the height `H`; `tensor_stride(t, dim 2)`, the byte stride of one plane row;
`t->MBuffer->data`, the base of the buffer holding the tensor; and `t->byte_offset`, the tensor's offset
within that buffer [M] — the last two together are the tensor's own DDR base.

```
plane = 0
for each input tensor t:                       # the layer's input vector, in model order
    D_t    = t->shape[1]                       # its 32-channel group count
    stride = tensor_stride(t, dim 2) · t->shape[2]     # its own plane stride  (row bytes × H)
    base   = t->MBuffer->data + t->byte_offset         # its own DDR base
    for j in 0 … D_t−1:  table[plane++] = base + j·stride
```

Every feature descriptor ([§6](#6-memory-and-dma)) then takes its DDR address from the table instead of
one base: ring mode `lo = table[pl] + 2·rp·rowbytes | more`, whole-input mode `lo = table[pl] + const | more`,
with `more` set on all but the last plane (`pl != D−1`). The ORAM side is unchanged — plane `pl` still lands
at `o_in + pl·slot` — so **only the DDR address of each plane differs from the single-input case**. [H]

A second table of `D` words follows and holds the **feed's per-plane ORAM stride**, passed to the feed
leaves as an argument. It is filled with the uniform ORAM slot, except when the tensor descriptor's
strided-view flag is set, in which case it takes the DDR deltas `table[pl+1] − table[pl]`. A
grouped-convolution path (group divisor above 1) divides `D` by the group count and shifts the table by the
descriptor's rebase count of planes. [M] With streamed weights the weight chain is rebuilt per pass; the
feature chain is unaffected.

#### 11.8.4 Prologue

As the base recipe ([§10](#10-the-recipe-template)), except that the resident preload of [§5.5](#55-weight-preload)
becomes the first pass's weight DMA, and the two tables are built before the ORAM plan.

#### 11.8.5 Tile loop

Unchanged. Streamed weights wrap the tile loop in the pass loop of [§11.2](#112-33-4-bit-operands) — one
weight DMA and one commit per pass ([§5](#5-running-a-layer)). Concatenation adds nothing: feed, drains and
requantization ([§3.8](#38-requantization-and-packing)) see the summed `D` and the ORAM slots as for a single input.

#### 11.8.6 Variants and not-characterised items

**Alternative im2col procedure (large-weight 3×3).** A second vendor procedure takes a different approach
to the kernel: instead of walking a spatial window it presents the 3×3 convolution as **im2col** — nine
spatially shifted views of the one input plane — and accumulates a 1×1 convolution per view. The weight
blob is then nine concatenated 1×1 weight sets ([§2.5](#25-weights)). It uses the general int8 MAC phases
and the MXUv3 requantization path of [§11.7](#117-floating-point-output-heads), extended with a packed
narrow output:

```
sllw  ,vr30      acc << shift
addw  ,vr22|23   + int32 bias
cltzw            sign of the biased accumulator
ffsiw            -> fp32
fmulw ,vr24|25   x positive scale        fmulw ,vr26|27   x negative scale
bselv            sign selects which product
ftsiw            -> int32
+ vr21           output zero point
satsuw4bi        saturate to unsigned [0,15]
gt_x             nibble-pack network
maxu4bi ,vr20    floor clamp
sa               16 B per 32-channel pixel
```

Its requantization table is a 384-byte-per-tile structure — int32 bias at `+0x00`, fp32 positive scale at
`+0x80`, fp32 negative scale at `+0x100` — expanded from the model file's 256-byte blob. The two-slope form
implements a leaky activation. The procedure has been read but not implemented; the reference
implementation runs such layers on the [§11.2](#112-33-4-bit-operands) recipe with streamed weights, which
gives the same output because both compute the same convolution. [M]

- Not characterised: the two-slope table and its expansion from the 256-byte blob (read, not implemented).
  What is established: the sequence above and the three table offsets, from the procedure's code.
- Not characterised: the strided-view and grouped-convolution paths of the second table. What is
  established: the flag is clear and the divisor is 1 on every layer of the reference network.
- Not characterised: the descriptor members `t->MBuffer->data` and `t->byte_offset` [L] as names. What is
  established: the shape struct `N, D, H, W`. See [Appendix B](#appendix-b--vendor-executor-map),
  [Appendix C](#appendix-c--not-characterised).
# Part IV — Reference

Part IV is the reference: the six instructions and their encodings, every configuration field with its
location, width, reset value and meaning, the program table, the field values each recipe writes, and the
companion runtime library.

## 12. Instruction set

### 12.1 The six instructions

The NNA is driven by **six** coprocessor instructions, all in the MIPS **SPECIAL2** space (opcode `0x1C`,
so every word starts `0x70…`) with function field `0x3A`. They share one 32-bit encoding:

```
word = 0x70000000 | (rs << 21) | ((payload & 0x7FFF) << 6) | 0x3A
      ├─op 0x1C──┤ ├─selector─┤ ├──── payload ────┤        ├─fn 0x3A─┤
        31:26        rs 25:21        20:6                     5:0
```

The **`rs` field (bits 25:21)** selects the instruction; the instruction payload is 15 bits, word bits
[20:6]. The Ingenic ISA manual (section 3.16, [Appendix A](#appendix-a--sources)) lists the same six with `rs` =
0…5 in the order `nnrwr`, `nnrrd`, `nndwr`, `nndrd`, `nnmac`, `nncmd`, and has no general-purpose register
operand. The instructions are emitted as raw `.word`s (the stock assembler does not know the mnemonics),
each preceded by a `sync`.

### 12.2 Operand encoding

The assembler does not see one payload but a register and an immediate. The split differs between the
instruction groups:

| instruction | binutils operands | register field | immediate |
|---|---|---|---|
| `nnrwr vwN,imm`, `nnmac vwN,imm` | `-h,-M` | `vr31` word `N` in word bits 15:11 | 10 bits: word bits 10:6 = imm[4:0], bits 20:16 = imm[9:5] |
| `nndwr vrN,imm` | `-f,-M` | vector register `N` in word bits 15:11 (the 64 bytes pushed) | same 10-bit split |
| `nndrd vrN,imm`, `nnrrd vrN,imm` | `-g,-N` | vector register `N` in word bits **10:6** (the destination) | 10 bits in word bits 20:11 |
| `nncmd imm` | `-L` | — | 15 bits in word bits 20:6 |

Vendor listings and older tools print a *merged payload* `p` = word bits [20:6] instead of the two operands.
For `nnrwr`, `nnmac` and `nndwr` the register is `p[9:5]` and the immediate is `p[4:0] | p[14:10] << 5`; for
`nndrd` and `nnrrd` the register is `p[4:0]` and the immediate is `p >> 5`. For `nnrwr` the immediate so
recovered is the `bank | field` form of §12.8: bit 5 = bank A, bit 6 = bank B, bits [4:0] = field index.
The full merged-to-operand table is [Appendix D](#appendix-d--register-number-cross-reference); this manual
uses instruction-field numbering everywhere else.

| rs | mnemonic | word template | payload meaning |
|----|----------|---------------|-----------------|
| `0` | `nnrwr` | `0x7000003A \| p<<6` | write a configuration field from a `vr31` word ([§4.1](#41-banks-and-fields)) |
| `1` | `nnrrd` | `0x7020003A \| p<<6` | read a 64-byte configuration block into a vector register ([§4.5](#45-configuration-readback)) |
| `2` | `nndwr` | `0x7040003A \| p<<6` | push one 64-byte operand vector into the port selected by the immediate |
| `3` | `nndrd` | `0x7060003A \| p<<6` | drain one readout row-bank into a vector register |
| `4` | `nnmac` | `0x7080003A \| p<<6` | run one multiply-accumulate pass |
| `5` | `nncmd` | `0x70A0003A \| p<<6` | array command (reset / arm / commit) |

### 12.3 Immediates versus data

The immediate — configuration field, port, readout row or array command — is baked into the instruction
word; it cannot be computed at run time. The *data* always travels in a register named by the same word:

- `nnrwr` writes the low 16 bits of the named word register `w0…w15` of `vr31` ([§4.2](#42-the-word-register-file));
- `nndwr` pushes the 64 bytes currently held by the named vector register;
- `nndrd` and `nnrrd` deliver 64 bytes into the named vector register;
- `nnmac` samples the named word register during the pass ([§3.6](#36-mac-phases-and-the-walk-position)).

The general-purpose register operands that vendor listings show next to `nndwr` and `nnmac` (`$t0`, `$v0`,
`$v1`) are not read by the hardware and are not operands. The MXU loads `law/lad/laq/lao` that precede a push
are ordinary memory instructions; the NNA sees only the vector register a subsequent `nndwr` names.

### 12.4 nncmd — array commands

| payload | word | effect |
|---------|------|--------|
| `0x00` | `0x70A0003A` | **reset**: wipes array state and the `vr31` word file, returns every field of [Chapter 13](#13-configuration-fields) to its reset value, and rewinds the program-table push pointer to 0, rewriting each entry's count to 1 and leaving its operand ([Chapter 14](#14-the-program-table)); MACs after it produce zeros until configuration, program and arm are redone. **It does not fully re-initialise the MAC stage**: what the weight preload after the reset sees depends on the configuration in force before it, unless `B.1b` is written immediately before — every run therefore starts `nnrwr B.1b ← 0 ; nncmd 0x00` ([§7.1](#71-the-b1b-reset-hazard), [§5.1](#51-reset)) |
| `0x60` | `0x70A0183A` | **arm**: latch the configuration, reset the accumulator/readout bank pointer; leaves the program table untouched |
| `0x83` | `0x70A020FA` | **commit (raw)**: end accumulation, clear the accumulators, publish int32 to readout bank 0 — the floating-point heads' only commit, once per tile ([§5.8](#58-commit)) |
| `0x8b` | `0x70A022FA` | **commit + pack**: as `0x83`, plus a hardware requant of the tile into readout bank 1 (the FIFO) |
| `0x8f` | `0x70A023FA` | **commit + pack** (`0x8b \| 4`): as `0x8b`. Not characterised: whether readout bank 0 remains readable after `0x8b` as it does after `0x8f`. What is established: the 3×3 recipe reads bank 0 after `0x8f`; the floating-point heads read it after `0x83` |
| `0x84` | `0x70A0213A` | issued before the `0x8b` that ends a tile in the 1×1, single-input-group and image recipes; isolated effect not characterised |
| `0xa0` | `0x70A0283A` | phase marker in the single-input-group recipe: precedes every second-phase `nnmac vw2,1`; isolated effect not characterised |
| `0xc0` | `0x70A0303A` | block marker: issued after each column block by the 3×3 recipes, once all `Dout` groups of the block are committed, and by the single-input-group recipe; isolated effect not characterised |
| `0xe0` | `0x70A0383A` | group marker in the single-input-group recipe: the switch to the next output group's `nnmac vw1,0`, written together with `A.12`; isolated effect not characterised |
| `0xaf` / `0xeb` | `0x70A02BFA` / `0x70A03AFA` | the two commits of the 3×3 single-group variant, after its `nnmac vw9,0` / `vw10,0` phases; isolated effect not characterised |

**How the payloads decompose (an observation, not a decode).** Written in binary, the eleven commands the
recipes issue fall into a regular shape: bit 7 is set on everything that acts on a tile, bits [6:5] hold a
2-bit selector, and the operation is in the low nibble.

| bit 7 | bits [6:5] | low nibble | payload | issued by |
|---|---|---|---|---|
| 0 | 0 | 0 | `0x00` | reset |
| 0 | 3 | 0 | `0x60` | arm |
| 1 | 0 | 3 / 4 / b / f | `0x83` `0x84` `0x8b` `0x8f` | the commit family |
| 1 | 1 | 0 / f | `0xa0` / `0xaf` | the single-input-group recipe before its second phase; the single-group 3×3 variant's first commit |
| 1 | 2 | 0 | `0xc0` | end of a column block, the 3×3 recipes and the single-input-group recipe |
| 1 | 3 | 0 / b | `0xe0` / `0xeb` | the output-group switch; the single-group 3×3 variant's second commit |

Where bits [6:5] are non-zero the value tracks the phase, the output-group switch or the end of a column
block, which would fit a unit or bank selector — but none of that has been characterised
([Appendix C](#appendix-c--not-characterised)); the decomposition only holds the eleven numbers as one family.

### 12.5 nnmac — multiply-accumulate

`nnmac vwN,imm` accumulates `acc[co] += Σ_k feed[k] · W[k][co]` for one pass, stepping the taps and channel
groups that the walk program describes ([§3.5](#35-the-walk-program)). The instruction names a word register
of `vr31` and a small immediate. In the general int8 family the immediate is 1 on every inner chunk of a
phase and 3 on its last (`nnmac vw2,1` / `vw2,3`), a rule the 4-bit recipes' codes do not follow.

| instruction | recipe | note |
|---|---|---|
| `vw2,1` / `vw2,3` (`0x41` / `0x43`) | general int8, and the floating-point heads | phase A = unit A's input groups, in chunks of ≤ 16: `,1` for every inner chunk, `,3` for the last |
| `vw14,1` / `vw14,3` (`0x1c1` / `0x1c3`) | general int8 | phase B = unit B's groups, inner / last; the heads run one `vw2,3` and one `vw14,3` per 4×2-pixel tile, then `nncmd 0x83` |
| `vw4,1` / `vw5,3` (`0x81` / `0xa3`) | 4-bit 1×1 | unit A (input groups `0..nA−1`) / unit B (`nA..D−1`), one pair per output group |
| `vw9,1` / `vw10,2` (`0x121` / `0x142`) | 4-bit 3×3, several input groups | unit A / unit B, one pair per output group of a tile |
| `vw9,0` / `vw10,0` (`0x120` / `0x140`) | 3×3 single-group variant | unit A / unit B, the two row-pair phases of a block, followed by `nncmd 0xaf` / `0xeb` |
| `vw1,0` / `vw2,1` (`0x20` / `0x41`) | single-input-group recipe, the image-layer recipe, the `Cin = 4` specialist | the two phases of a block, unit A then unit B: output rows 0–1 and 2–3 of a row quad in the single-input-group recipe, the two rows of a row pair in the image layer; each is followed by `nncmd 0x84, 0x8b` |

The named word register holds that phase's operand walk position: it is the word the recipe initialises
before the arm and steps in software as the walk advances, and no `nnrwr` ever reads it
([§3.6](#36-mac-phases-and-the-walk-position)). The word each recipe names is tabulated in
[Chapter 15](#15-field-values-by-recipe). Two regularities hold: where a recipe splits its input groups
between the two units, the second phase's word is the first's plus `4 · B.11` (the `B.10`/`B.11` start offset
in quarter units); and a phase whose walk is carried by the pushed program (the 1×1) never steps its word.

Not characterised: whether the array writes the word back, and the meaning of the immediate beyond its use as
inner/last marker. What is established: the array reads the word — the instruction is listed in the Ingenic
ISA manual section 3.16 as `NNA_mac(VWR[vwrp], imm)`, passing the word's value in.

### 12.6 nndwr — push an operand vector

`nndwr vrN,port` writes the 64 bytes of `vrN` into the array. The 10-bit immediate selects the **port**, of
which the recipes use exactly seven, and the destination slot inside the array is **positional**: each port
has an internal write pointer that advances by one 64-byte slot per push and is driven by the walk program
([§3.4](#34-ports-and-positional-feeds), [§3.5](#35-the-walk-program)). No address travels with the
instruction.

| port | bank | what a push carries | who pushes it |
|---|---|---|---|
| `0`, `2`, `4`, `7` | weights | one quarter of a 256-byte weight slice, the four ports in that order; a 4-bit 32×32 tap is two rounds, an 8-bit tap four | every recipe; usually `vr0..vr3`, the 3×3 recipe alternates `vr0/vr1` |
| `0x20` | activations, unit A | one 64-byte window of activations: 32 channels × 2 int8 pixels (general int8), 4 pixels × 32 channels × 4 bit of one input row (4-bit recipes), one 64-byte half of an 8-bit row, or 2 pixels × 32 assembled taps (image layer) | every recipe; from `vr0..vr3`, or `vr15/16/18/19` in the shuffle-fed recipes |
| `0x21` | activations, unit B | the same for the second execution unit | the 1×1 and 3×3 recipes (unit B) |
| `0x40` | requant table | 64 bytes of the requant table; a 32-channel group's 256-byte entry is four pushes | the packed-output recipes, once per pass or once per layer |

**The data path.** The pushed bytes are the named register's current content, sampled at the push. In most
recipes that register was loaded by the `law`/`lao` immediately before; in the image-layer recipe and the
`Cin = 4` specialist it holds the result of a byte-shuffle network with no load in between — the bytes need
not have come from memory, and re-pushing a register re-pushes whatever it holds at that moment. The
general-purpose register operands are not read (§12.3).

**Consequence for feeding.** What must be reproduced is the **exact ordered sequence and count** of pushes
per port, not a set of per-slot addresses: removing one push mis-slots every push after it, and a feed with
the right data in the wrong order drains zeros or garbage. The sequences are given per recipe in
[Chapter 11](#11-the-recipes): the 3×3 recipe keeps its feed two column blocks ahead of the MACs at stride 1
and one at stride 2; the 1×1 pushes one window per column block and reuses it for every output group of the
pass; the single-input-group recipe pushes 4–10 windows per column chunk, some rows twice — `nndwr vr2,0x20 ;
nndwr vr3,0x20` re-pushes rows 2 and 3, shared by its two phases, and at stride 2 `nndwr vr0,0x20 ; nndwr
vr0,0x20` pushes row 4 twice; the image layer pushes four assembled 32-tap vectors per output row.

### 12.7 nndrd — drain results

`nndrd vrN,imm` delivers 64 bytes from one of the two readout banks into `vrN`. The immediate is
`bank << 5 | R`:

- **`imm = R` (0…15), bank 0 — the raw int32 accumulators.** Row-bank `R` holds 16 int32 = channels
  `[16·(R mod 2), +16)` of pixel `⌊R/2⌋` of the committed tile. The read is random-access and
  non-destructive: the same row-bank can be read again and again after a `nncmd 0x83`/`0x8b`/`0x8f`, and after
  after `0x83` and `0x8f` it holds the tile just committed (after `0x8b`: not characterised, [§5.8](#58-commit)) — which is what makes bank 0 the measurement channel
  for the requant arithmetic (the image-layer recipe is the one mode in which bank 0 returns stale data). The
  general int8 recipe reads `nndrd vr(27+i), i + 4g`; the heads read all sixteen (`nndrd vr10..vr17`).
- **`imm = 0x20`, bank 1 — the packed output FIFO.** Each read pops the next 64 bytes of the
  hardware-requantized tile; bank 1 is always read with `R = 0`. At 4-bit output a tile row (4 pixels × 32
  channels) is one read, at 8-bit two; a 2-row tile is therefore two or four reads: the 3×3 recipe's
  `nndrd vr8,0x20 ; vr9,0x20` (rows 0/1; `vr10/vr11` added at 8 bits), the 1×1 and single-input-group
  recipes' `vr10,0x20 ; vr11,0x20` (`..vr13` at 8 bits), the image layer's and the `Cin = 4` specialist's
  `vr22,0x20 ; vr0,0x20`. The FIFO lags the write point, so a drain returns an earlier tile than the one just
  committed; the lag and the junk drain that precedes the first real one are described in
  [§3.7](#37-accumulators-commit-and-readout) and [§5.9](#59-drain).

### 12.8 nnrwr — write a configuration field

`nnrwr vwN,imm` copies the low 16 bits of word register `wN` into the configuration field the immediate
names. The immediate is 10 bits in the `bank | field` form: bit 5 selects bank A, bit 6 bank B, bits [4:0]
the field index within the bank. The word register is a separate operand and not part of the field's
identity — the same field fed from a different word is the same field. Field addressing is in
[§4.1](#41-banks-and-fields), the word register file in [§4.2](#42-the-word-register-file), the fields
themselves in [Chapter 13](#13-configuration-fields).

A configuration burst is therefore: stage values into word registers, then issue one `nnrwr` per field. The
vendor idiom `law w11 ; law w12 ; nnrwr X,w11 ; nnrwr Y,w12` is exactly that, and the per-tile idiom
`addiw w4,w4,1 ; nnrwr B.10,w4` steps a per-tile field from its word and rewrites it.

### 12.9 gshufvb — the MXU3.1 byte gather

Not an NNA instruction, but the one MXU3 op the executors depend on beyond load/store, and worth recording
here because the rest of this tree said for a long time that it did not exist.

MXU3.0 has `gshufb` (`0x7020002b`). **It is not on this chip** — it raises SIGILL. MXU3.1 replaces it with
`gshufvb`, encoded

```
gshufvb  vd, vtab, vidx      0x7380002e | vd<<6 | vtab<<11 | vidx<<16
```

which is a data-dependent gather over all 64 byte lanes of a vector register:

```
for i in 0..63:
    vd[i] = (vidx[i] & 0x80) ? 0 : vtab[vidx[i] & 0x3F]
```

Three properties, all device-probed (2026-09-11):

* **Bit 6 of the index is ignored.** Index 64 reads table byte 0; index 100 reads table byte 36.
* **Bit 7 zeroes the lane**, the x86 `pshufb` rule. Index 128 and index 255 both give 0, even though their
  low six bits are in range.
* **Lanes gather across the whole 64 bytes**, not within 32-byte halves: lane 3 reading index 63 returns
  table byte 63.

The zeroing is not a nuisance, it is a free predicate: gathering from an all-ones register yields `0xFF`
exactly where bit 7 of the index is clear, which is how `PLT_M3_LUT64`
(`runtime/hal/plt_mxu3.h`) builds a **256**-entry byte table out of a gather that only reaches 64 — four
gathers on the low six bits, selected by two masks the gather itself produces. 19 instructions per 64 bytes
against roughly 320 scalar, measured 0.221 µs against 0.907 µs for a 256-byte convolution output tile.

`gshufvb` is also, per `thingino-accel/docs/MXUV3_ISA_HARDWARE_RE.md`, one of the two most-executed custom
ops in libvenus — SPECIAL2 `func=0x2e` — so the vendor's own runtime leans on it heavily.
## 13. Configuration fields

### 13.1 Field addressing

A configuration field is named `A.xx` or `B.xx`: bank A holds the geometry of the layer, bank B the modes,
the activation setup and the program table ([§4.1](#41-banks-and-fields)). A field is written with
`nnrwr vwN,imm`, where `vwN` is the word register that holds the value ([§4.2](#42-the-word-register-file))
and `imm` is the bank bit or-ed with the field index: bit 5 (`0x20`) selects bank A, bit 6 (`0x40`) bank B,
so `A.19` is `nnrwr vwN,0x39` and `B.10` is `nnrwr vwN,0x50`. The programming model is Chapter 4
([§4](#4-the-configuration-model)); the raw payload numbers that appear in vendor code are listed in
[Appendix D](#appendix-d--register-number-cross-reference).

### 13.2 Storage and widths

The committed configuration reads back in 64-byte blocks: block **G** is the bank-A field file, **H0** the
bank-B field file, **H1** the program table and **L0** a status word ([§13.5](#135-readback-blocks)). Field
`f` reads back at word `f>>1`, halfword `f&1` of its bank's block (halfword 0 is bits [15:0], halfword 1
bits [31:16]), so `A.0c` is `G.w6[15:0]` and `B.11` is `H0.w8[31:16]`. The width column is the number of
readback bits the field occupies; **write-only** means the write changes no readback block (these fields
carry addresses and counts). Reset values are the state after `nncmd 0x00`. Field indices not listed have
no known writer.

**Bank A (block G)**

| field | readback location | width | reset | reference name |
|---|---|---|---|---|
| `A.01` | `G.w0[17:16]` | 2 | 0 | `NNA_INERT_PARITY_TAG` |
| `A.03` | `G.w1[19:16]` | 4 | 1 | `NNA_IN_ELEM_CLASS` |
| `A.08` | `G.w4[1:0]` | 2 | 0 | `NNA_PRECISION_A` |
| `A.09` | `G.w4[27:16]` | 12 | `0xfff` | `NNA_ROW_COORD` |
| `A.0b` | `G.w5[27:16]` | 12 | `0xfff` | `NNA_COL_COORD` |
| `A.0c` | `G.w6[10:0]` | 11 | `0x7ff` | `NNA_IN_H_M1` |
| `A.0e` | `G.w7[10:0]` | 11 | `0x7ff` | `NNA_IN_W_M1` |
| `A.0f` | `G.w7[27:16]` | 12 | 0 | `NNA_PAD_VALUE` |
| `A.10` | write-only | — | — | `NNA_MAC_BASE_A0` |
| `A.11` | write-only | — | — | `NNA_MAC_BASE_A1` |
| `A.12` | write-only | — | — | `NNA_PLANE_STRIDE_HI` |
| `A.14` | write-only | — | — | `NNA_PLANE_STRIDE` |
| `A.17` | `G.w11[24:18]` | 7 | `0x14` | `NNA_MAC_SPAN` |
| `A.18` | `G.w12[4:0]` | ≤5 | `0x12` | `NNA_EDGE_MODE` |
| `A.19` | `G.w12[21:18]` | 4 | 5 | `NNA_MAC_PITCH` |
| `A.1a` | `G.w13[4:0]` | ≤5 | 0 | `NNA_STRIDE_CODE` |
| `A.1b` | `G.w13[23:16]` | 8 | 0 | `NNA_OPERAND_PRECISION` |
| `A.1c` | `G.w14[3:0]` | 4 | 0 | `NNA_UNIT_A_GROUPS_M1` |
| `A.1d` | `G.w14[19:16]` | 4 | 0 | `NNA_UNIT_B_GROUPS_M1` |
| `A.1e` | `G.w15[8:0]` | 9 | `0x1ff` | `NNA_TAP_MASK` |
| `A.1f` | `G.w15[23:16]` | 8 | `0xff` | `NNA_PIXEL_MASK` |

`G.w9[16]` reads 1 at reset and has no writer.

**Storage granularity.** `A.17` and `A.19` are the only two fields whose readback bits do not start at a
halfword boundary (`G.w11[24:18]`, `G.w12[21:18]`). Both are stored **in units of 4**: a write keeps bits
[8:2] of the value for `A.17` and bits [5:2] for `A.19`, so `A.17` spans 0…`0x1fc` and `A.19` 0…`0x3c` in
steps of 4, and the two low bits of a written value are discarded. `A.18` stores bits {4, 1, 0} only; bits
[3:2] are not implemented. No write to any field bleeds into a neighbouring field. In written units the
reset state is `A.17 = 0x50`, `A.19 = 0x14`, `A.18 = 0x12`; the reset column above gives the same state as
stored field bits.

**Bank B (block H0)**

| field | readback location | width | reset | reference name |
|---|---|---|---|---|
| `B.01` | write-only | — | — | `NNA_UNKNOWN_B01` |
| `B.02` | `H0.w1[7:0]` | 8 | 0 | `NNA_COMMIT_WINDOW` |
| `B.03` | `H0.w1[17:16]` | 2 | 0 | `NNA_PACK_FORMAT` |
| `B.04` | `H0.w2[2:0]` | 3 | 0 | `NNA_OUT_ELEM_CLASS` |
| `B.05` | `H0.w2[19:16]` | 4 | 0 | `NNA_MODE_FLAGS` |
| `B.06` | `H0.w3[1:0]` | 2 | 0 | `NNA_PRECISION_B` |
| `B.07` | `H0.w3[20:16]` | 5 | 0 | `NNA_READOUT_LAG` |
| `B.08` | `H0.w4[15:0]` | 16 | `0x3210` | `NNA_LANE_PERM_ROW0` |
| `B.09` | `H0.w4[31:16]` | 16 | `0x7654` | `NNA_LANE_PERM_ROW1` |
| `B.10` | `H0.w8[9:0]` | 10 | 0 | `NNA_UNIT_A_START` |
| `B.11` | `H0.w8[25:16]` | 10 | 0 | `NNA_UNIT_B_START` |
| `B.15` | `H0.w10[17:16]` | 2 | 0 | `NNA_TILE_CLASS` |
| `B.18` | `H0.w12[3:0]` (push pointer) | 4 | 0 | `NNA_PROGRAM_PTR` |
| `B.19` | push into the program table ([§14](#14-the-program-table)) | — | — | `NNA_PROGRAM_PUSH` |
| `B.1a` | `H0.w13[6:0]` | ≤7 | 0 | `NNA_PROGRAM_LENGTHS` |
| `B.1b` | `H0.w13[17:16]` | 2 | 0 | `NNA_MAC_BALANCE` |
| `B.1c` | write-only | — | — | `NNA_WEIGHT_STREAM_OFF` |
| `B.1e` | `H0.w15[7:0]` | 8 | 0 | `NNA_TABLE_OFF` |

`B.08`/`B.09` reset to `0x3210`/`0x7654`, i.e. `H0.w4 = 0x76543210`: eight 4-bit lane slots in identity
order.

### 13.3 Static fields

The tables in this section and the next state what each field means and what an implementation writes to
it. The confidence column uses the H/M/L scale of the Conventions section; the per-recipe values are in
[§15](#15-field-values-by-recipe). A static field is written once per layer, before the arm; the exceptions
(`B.11`, and `B.10` in [§13.4](#134-per-tile-fields)) are noted in their rows.

| field | bits | meaning | conf |
|---|---|---|---|
| `A.0c` | 11 | Input height − 1. Coordinates beyond it are outside the image. Equals the output height at stride 1. 11 bits, so the input height is at most 2048 (derived) | H |
| `A.0e` | 11 | Input width − 1, with the same masking rule. 11 bits, so the input width is at most 2048 (derived) | H |
| `A.0f` | 12 | Padding fill value, i.e. the input zero-point, masked to the input element width (`& 0xf` at 4-bit input, `& 0xff` otherwise) | H |
| `A.1b` | 8 | Operand precision: high nibble the weight width, low nibble the input width, each as `bits/2 − 1`. `0x11` = 4-bit/4-bit, `0x33` = 8/8, `0x31` = 8-bit weights with 4-bit input, `0x13` = the reverse | H |
| `A.08`, `B.06` | 2, 2 | A mode code pair, always written together and always equal: 2 when the vendor's ConvOp word `+0x80` is 2, otherwise 3. **Not** the operand width — the 8-bit-input recipes write 2 as well. Not characterised: what the code selects. What is established: every layer of the reference network writes 2, and raising it to 3 on a 4-bit layer zeroes the output | H |
| `A.03` | 4 | Input element width code, `in_bits/2 − 1` (0 = 2-bit, 1 = 4-bit, 3 = 8-bit). MAC stage: a wrong value makes the array read the operand stream as wider elements. The vendor writes 0 at stride 2, and both values produce identical output on 4-bit input | H |
| `B.04` | 3 | Output element width code, `(out_bits − 2)/2` for 2-, 4- and 8-bit output (the vendor's 1×1 writes 3 for any other width) | H |
| `B.15` | 2 | Input tile size class, `log2(tile_bytes/64)`: 1 = 128 B (8 px × 32 ch × 4-bit), 2 = 256 B, 3 = 512 B. Derived from the **input** width, so it does not follow the output width | H |
| `A.1a` | ≤5 | Stride − 1: low nibble the column stride, high nibble the row stride. 0 at stride 1, `0x11` at stride 2 | H |
| `A.19`, `A.17` | 4, 7 | `A.19` is the MAC-stage row pitch — one walk step unit is `A.19/4`; it is a software-chosen constant, not a hardware limit, and the vendor's values 20 (stride 1) and 24 (stride 2) are choices. `A.17` is the pack-stage span, `A.19 × walk`; it affects only the packed output, never the accumulators. Both are stored in units of 4 (storage note above) | H |
| `A.1c`, `A.1d` | 4, 4 | Input channel groups per execution unit, minus one: `A.1c = D/2 − 1` for unit A, `A.1d = D − D/2 − 1` for unit B ([§3.3](#33-the-two-execution-units)). 4 bits, so at most 16 groups (512 input channels) per unit per pass — the origin of the 1×1 recipe's `D ≤ 32` guard (derived). The general int8 executor rewrites them per MAC phase as chunk sizes | H |
| `A.18` | 5 | Row-edge and unit-window control. Bit 4: unit B's column window aligned with unit A's (1) or shifted one 4-pixel block left (0). Bits [1:0]: right-edge mode — 0 and 1 none, 2 a padding edge at the last column, 3 a row cut after 12 columns. Bits [3:2] are not implemented | H |
| `A.1e` | 9 | Kernel tap enable mask: bit `k` enables tap `k` of the program's tap sequence. Reset `0x1ff` = all nine. MAC stage — a cleared bit removes that tap's products from every pixel. The 1×1, single-input-group, image and head recipes write it once before the arm; the 3×3 recipe leaves it at its reset value | H |
| `B.08`, `B.09` | 16, 16 | Input lane permutations, four bits per lane: `B.08` for the tile's first row, `B.09` for its second. Identity is `0x3210` / `0x7654`; the 3×3 recipe writes `0x5410` / `0x7632` at 8-bit and `0x5140` / `0x7362` at 6-bit output | H |
| `B.05` | 4 | Mode flags. Bit 0: accumulator preload from ORAM, inferred; the ORAM address it reads is not characterised. Bit 1: no observed effect; set by every vendor recipe. Bit 2: unsigned (offset-binary) output — every value gains `2^(out_bits−1)`, accumulators untouched. Bit 3: alternative table interpretation; no layer observed uses it. **Not characterised** | M |
| `B.03` | 2 | Pack/output format. 0 and 3 both give the normal packed output; 1 and 2 select formats no observed layer uses. Pack stage only | H |
| `B.1b` | 2 | MAC-stage half-balance. 0 accumulates the two 16-channel halves equally; each nonzero value unbalances them. The reading that value 1 scales the low half by 2⁄3 and the high half by 4⁄3 is [L] (fitted to one tile's accumulators). This is also the field that must be written immediately before `nncmd 0x00` | H |
| `A.01` | 2 | A static datapath/parity tag. Not characterised: its consumer. What is established: no effect at any of its four values on the layers of the reference network; the vendor writes a literal 3 | M |
| `A.1f` | 8 | Per-pixel enable mask of a tile, bit `p` for pixel `p = 4·row + col` ([§3.1](#31-the-tile)). Reset `0xff`; disabled pixels emit bias-only values | H |
| `B.07` | 5 | Packed readout lag in 64-byte units, normally the packed tile size (128 B → 2, 256 B → 4); 512 B → 8 is the natural extension but no such layer exists. Each drain returns the block `B.07 × 64` bytes behind the write point ([§3.7](#37-accumulators-commit-and-readout)); values below 2 behave as 2 [L] | H |
| `B.11` | 10 | Unit B's operand start offset. `B.11` is written once per layer — before the arm by the 3×3 recipe, after it by the 1×1 —; `B.10` also per row quad / row pair / column block by some recipes ([§13.4](#134-per-tile-fields)). The floating-point-head recipe writes `B.10` for both units and never `B.11` | H |
| `B.18`, `B.19`, `B.1a` | 4, —, ≤7 | Program pointer, program push, and the program lengths `(len_B << 4) \| len_A` ([§14](#14-the-program-table)) | H |
| `B.01` | w/o | Not characterised: its function. What is established: one vendor site writes the value 8, and injecting 8 or 1 before the arm changes nothing | M |

### 13.4 Per-tile fields

Per-tile fields are written inside the compute loop, between tiles or between commits.

| field | bits | meaning | conf |
|---|---|---|---|
| `A.09` | 12 | Signed input row coordinate of the current pass, `−pad_top + k·stride_h` | H |
| `A.0b` | 12 | Signed input column coordinate, `−pad_left + k·stride_w·n`; the column walk itself is in the program ([§3.5](#35-the-walk-program)) | H |
| `B.10` | 10 | Unit A's operand start offset. Written after the arm, and rewritten per row quad / row pair / column block by the single-input-group, image and floating-point-head recipes; the last writes `B.10` for both units and never `B.11` | H |
| `A.12`, `A.14` | w/o | Pack-stage per-plane readout stride, in accumulator slots: `D · 9 · w_bits / 2`. The stage reads a 288-slot (`0x120`) block per tile and this stride divides it into 32-channel output planes; the fixed 288-slot block is [L] (a vendor planner constant, not a measured memory size). `A.14` positions planes 1–3, `A.12` the upper pair; plane 0 sits at the fixed base. 0 selects the default. Accumulators are never affected — a wrong value misplaces planes in the packed output ([§3.8](#38-requantization-and-packing)) | H |
| `B.02` | 8 | Window index of the output group being committed, in units of 4: `4 · g` for group `g` of a pass. Written before each commit; the write itself is load-bearing | H |
| `A.10`, `A.11` | w/o | Unit A's MAC operand and accumulator base, the counterpart of `B.10`/`B.11`. Written as a pair around unit A's `nnmac`. Off the datapath of the descriptor-fed recipes: any value, or no write at all, leaves the output unchanged | H |
| `B.1c` | w/o | Weight-stream address offset in 256-byte units, applied per tile. 4 shifts the array's weight reads by one 1 KB super-tile. MAC stage | H |
| `B.1e` | 8 | Signed requantization-table offset for the pack stage. Accumulators untouched | H |

### 13.5 Readback blocks

Confidence letters follow the H/M/L legend of the Conventions section. `nnrrd vrN,imm` reads one 64-byte
block of committed state into vector register `N`; the block is selected by the immediate, and a field
write is visible in the very next `nnrrd`, before any arm. The read sequence and the store form that returns
current data are in [§4.5](#45-configuration-readback).

| block | `nnrrd` immediate | contents |
|---|---|---|
| **G** | bit 5 set (`0x20`; bits [4:0] ignored) | bank-A field file |
| **H0** | bit 6 set, bit 0 clear (`0x40`) | bank-B field file |
| **H1** | bit 6 set, bit 0 set (`0x41`) | the program table ([§14](#14-the-program-table)) |
| **L0** | bits 6 and 5 clear, bits [1:0] = 0 (`0x00`) | status: `w0` bit 0 = "written since reset". Not characterised: the remaining words. What is established: `w2 = 0xff`, `w4 = 0x00010000`, `w5 = 9`, `w6 = 0x00010000`, `w7 = 0xa` at reset |
| — | any other value | zeros |
## 14. The program table

The program table stores the [walk program](#35-the-walk-program): the schedule a MAC follows, saying where
each successive operand is taken from. The table holds sixteen entries of fifteen bits in two slots of
eight; slot A begins at entry 0 and is executed by unit A, slot B begins at entry 8 and is executed by
unit B ([§3.3](#33-the-two-execution-units)).

### 14.1 Transport

`B.19` is not a register but a **push**: each `nnrwr B.19` stores the selected word register into
`entry[ptr]` and increments `ptr`, the pointer readable at `H0.w12[3:0]`; `nnrwr B.18` sets `ptr`
directly ([§5.3](#53-program-loading)). The pointer is **four bits and wraps** — `B.18 ← 16` positions at
entry 0, `B.18 ← 31` at entry 15, and a push into entry 15 is followed by one into entry 0. There is no
slot boundary in the pointer; slot A and slot B are a convention of the length field and of where each
unit reads its program from. The sixteen entries are visible in block `H1` as its first sixteen halfwords.

An entry holds **fifteen bits**: bit 15 of a pushed word is dropped. A walking-bit push reads back
`0001 0002 … 4000 0000`, and a program word with bit 15 set (`0x9805` in place of `0x1805`) runs
identically to the word without it.

### 14.2 Entry format

```
entry = count[14:11] | operand[10:0]
```

| `count` | kind | meaning |
|---|---|---|
| 0 | **move** | advance the walk position by `step`, consuming no operand |
| 1…14 | **step** | consume the operand at the walk position and advance by `step` — `count` times over |
| 15 | **loop** | repeat entries `(this + 1) … (end_entry − 1)`, `repeats` times |

The operand field is 11 bits `[10:0]`. For a step or a move, the step value is the signed 10-bit field
`[9:0]`; bit 10 is ignored. For a loop, `operand = end_entry[10:8] | (repeats − 1)[7:0]`, where
`end_entry` is counted from the slot's first entry (0 for slot A, 8 for slot B), so the same loop word
serves both slots [L] — the constructor masks it to three bits. `repeats` = 1 makes a loop entry a
no-op, which is why the 1×1 recipe's two-group program opens with an apparently dead word. Loops do
**not** nest: a loop entry inside another loop's body does not multiply.

A negative step must be masked to ten bits before it is placed in the entry: an unmasked negative integer
sets bits `[14:11]` as well, which turns the entry into a loop and derails the whole walk. Program words are
data in a configuration record, so an instruction-stream comparison cannot expose such a mistake.

### 14.3 Lengths

`B.1a` carries two **three-bit** fields, `len_A` in bits `[2:0]` and `len_B` in bits `[6:4]`; everything
else is dropped (`0x88` reads back as `0x00`, `0xff` as `0x77`). Each encodes **1…8, with 0 meaning 8**: an
eight-entry program declared `0x88` runs correctly, while a six-entry program declared `0x00` does not. So
a slot really does hold eight entries, and for a slot that is pushed and executed a length is a hard
terminator — declaring five when four were pushed is wrong everywhere.

Recipes that load only slot A write `len_B = 0`. Not characterised: how a slot declared 0 and never pushed
is treated when the second unit runs. What is established: every single-slot recipe writes `len_B = 0`,
never pushes slot B, and is exact — including one that runs the unit-B MAC.

### 14.4 Reset

`nncmd 0x00` ([§5.1](#51-reset)) rewinds the pointer to 0 and rewrites every entry's **count field to 1**,
leaving the 11-bit operand untouched: `1234 → 0a34`, `7fff → 0fff`, `0000 → 0800`, and `0805` unchanged.
There is no fixed reset pattern — the `0x0800` seen throughout the table after a cold reset is count 1 over
operands that happened to be zero. `nncmd 0x60` (the arm) leaves the pointer and the entries alone.

### 14.5 A program, decoded

A 3×3 layer at stride 1, `D = 4`, with two input groups per unit pushes `0x7b01 0x1805 0x0819 0x03b1` into
both slots, with `B.1a = 0x44`:

| entry | value | decode | role |
|---|---|---|---|
| 0 | `0x7b01` | loop to entry 3, twice | once per input group of this unit |
| 1 | `0x1805` | 3 operands, `+5` each | the window's first three rows: a row costs `A.19 / 4` |
| 2 | `0x0819` | 1 operand, `+25` | the group's last row, then on to the next group |
| 3 | `0x03b1` | move `−79` | rewind the sweep and advance to the next output column |

The arithmetic closes exactly. A window row costs `A.19 / 4`, so a group's `walk` rows cost
`walk · A.19 / 4 = A.17 / 4`; entry 2's step tops the group up to `row_program_units`; and entry 3's
`stride − groups · row_program_units` returns the walk to where the next column's begins. **Summed over the
program, `max(count, 1) · step` is the stride.** At stride 2 the same formulas give `0x7b01 0x2006 0x0824
0x038a` ([15](#15-field-values-by-recipe)).

`A.19` is a software-chosen row pitch, not a hardware constant: the layer is exact at `A.19` = 16, 20, 24
and 28 when every derived quantity (`A.17`, `B.11`, the per-tile words and the program's step, top-up and
rewind) moves with it. One program step unit is a quarter of `A.19`.

Not characterised: the absolute size of one program step unit in bytes of ORAM, and the geometric identity
of the walk's starting position, which the `vr31` word an `nnmac` names supplies
([§3.6](#36-mac-phases-and-the-walk-position)). Everything above is relative to that origin.

The other recipes use the same format with a different net advance, their column movement being in software
rather than in the program: the 1×1's walk returns to its start (net 0, `2n` operands for `n` input groups),
and the `Cin ≤ 32` executor's programs net `+1` at either stride ([15](#15-field-values-by-recipe)).

### 14.6 Constructors

| Constructor | Expansion |
|---|---|
| `NNA_PROG_WORD(count, operand)` | `((count & 0xf) << 11) \| (operand & 0x7ff)` |
| `NNA_PROG_STEP_N(count, step)` | `NNA_PROG_WORD(count, step & 0x3ff)` — the ten-bit mask keeps a negative rewind inside the field |
| `NNA_PROG_STEP(step)` | `NNA_PROG_STEP_N(1, step)` — one operand, then a step |
| `NNA_PROG_MOVE(step)` | `NNA_PROG_STEP_N(0, step)` — advance without consuming |
| `NNA_PROG_LOOP(end_entry, repeats)` | `NNA_PROG_WORD(15, ((end_entry & 7) << 8) \| ((repeats − 1) & 0xff))` |
| `NNA_PROG_LENGTHS(len_a, len_b)` | `((len_b & 7) << 4) \| (len_a & 7)`, each 1…8 with 0 meaning 8 |

Slot A begins at entry 0 and slot B at entry 8 (`NNA_PROGRAM_SLOT_B`), eight entries each. The 3×3 program
of [§14.5](#145-a-program-decoded), written with the constructors for `nA` groups on unit A and `nB` on unit B
(the same `end_entry` of 3 serves both slots, [§14.2](#142-entry-format)):

```
step_operand = A.19 / 4
walk_step    = NNA_PROG_STEP_N(walk − 1, step_operand)
row_step     = NNA_PROG_STEP(row_program_units − (A.17 / 4 − step_operand))
slot_a = { NNA_PROG_LOOP(3, nA), walk_step, row_step, NNA_PROG_MOVE(stride − nA · row_program_units) }
slot_b = { NNA_PROG_LOOP(3, nB), walk_step, row_step, NNA_PROG_MOVE(stride − nB · row_program_units) }
B.1a   = NNA_PROG_LENGTHS(4, 4)
```

In the listing above `row_program_units` is the size of one input group in walk-step units — the quantity
entry 2 tops the group up to ([§14.5](#145-a-program-decoded)).
## 15. Field values by recipe

This chapter cross-tabulates the recipes of [Chapter 11](#11-the-recipes): for each configuration field
([§13](#13-configuration-fields)), the value or formula that each recipe writes, and for each recipe the `nnmac`
word registers of its two MAC phases ([§3.6](#36-mac-phases-and-the-walk-position)). A recipe section is
authoritative where a cell abbreviates it. Symbols: `D = Cin/32` input groups, `Dout = Cout/32`, `nA = D/2` and
`nB = D − nA` (unit A / unit B, [§3.3](#33-the-two-execution-units)), `s` = stride, `in_bits` / `w_bits` /
`out_bits`, `H_in × W_in` the input size, `H × W` the output size, `zp` the input zero-point, `pad` the left/top
pad; 3×3 walk constants `K = 0x14`, `walk = 4` at stride 1 and `K = 0x18`, `walk = 5` at stride 2
([§11.2](#112-33-4-bit-operands)); `gpp` output groups per weight pass, `k` the pass, `g` the output group of a
pass, `rp` the row pair, `q` the row quad, `cb` the column block; `cA`, `lA`, `cB`, `lB` the head recipe's chunk
sizes ([§11.7](#117-floating-point-output-heads)). "not written" = the field keeps its reset value
([§13](#13-configuration-fields)); "base" in the 8-bit column = the value of the recipe it modifies
([§11.5](#115-8-bit-input-variants)); "—" = not confirmed by a recipe section (listed at the end).

### 15.1 Static fields by recipe

Written once per layer, before the arm unless the cell says otherwise ([§5](#5-running-a-layer)).

| field | 1×1 4-bit [§11.1](#111-11-4-bit-operands) | 3×3 4-bit [§11.2](#112-33-4-bit-operands) | row quad [§11.3](#113-single-input-group-row-quad) | two slot [§11.4](#114-single-input-group-two-slot) | 8-bit [§11.5](#115-8-bit-input-variants) | image [§11.6](#116-the-rgb-image-layer) | fp heads [§11.7](#117-floating-point-output-heads) |
|---|---|---|---|---|---|---|---|
| `A.01` | `D & 1` | 3 | not written | 3 | base | 3 | `3·(D mod 2)` |
| `A.18` | `0x12` | `0x12` | not written | `0x12` | base | `0x12` | `0x12` |
| `A.0c` / `A.0e` | `W−1` / `W−1` (`H−1` in `A.0c` is also exact on square maps) | `H_in−1` / `W_in−1` | `H_in−1` / `W_in−1` | `H_in−1` / `W_in−1` | base | `H_in−1` / `W_in−1` | `H−1` / `W−1` |
| `A.0f` | not written | `zp` | `zp & 0xf` (4-bit input), `zp & 0xff` (8-bit) | `zp` | base | — | — |
| `A.1b` | `(w_bits/2−1)<<4 \| (in_bits/2−1)` = `0x11` | same formula: `0x11` / `0x31` (8-bit weights) | same: `0x11` / `0x31` | same | `0x13` (4-bit weights, 8-bit input) | `0x13` | `0x33` |
| `A.03` | `in_bits/2 − 1` | `in_bits/2 − 1` (the vendor writes 0 at stride 2; both exact) | not written | `in_bits/2 − 1` | 3 | — | 3 |
| `A.08` = `B.06` | 2 | 2 | 2 (3 when the operator code ≠ 2) | 2 | base | 2 | 2 |
| `A.19` | 4 | `K` (`0x14` / `0x18`) | `0x14` (recipe constant) | `K` | base | 8 | 4 |
| `A.17` | 8 | `K·walk` (`0x50` / `0x78`) | `0xc8` (recipe constant) | `K·walk` | base | 16 | 8 |
| `A.1a` | not written | `(s−1)·0x11` | 0 for 1×1, else `(s−1)<<4 \| (s−1)` | `(s−1)·0x11` | base | — | — |
| `A.1c` / `A.1d` | `nA−1` / `nB−1` | `nA−1` / `nB−1` | not written | not written (no unit split) | base | `A.1c = 0`; `A.1d` not written | per MAC phase, as chunk sizes (§15.2) |
| `A.1e` | 1 | `0x1ff` (not written, reset) | tap mask: `0x1ff` (K = 3), `0x1` (K = 1) | `0x1ff` (not written, reset) | base | 1 | 1 |
| `B.03` | 0 | 0 (from the requantization mode code) | 0 (from the operator code) | 0 | base | 0 | not written |
| `B.04` | `(out_bits−2)/2` | `(out_bits−2)/2` | 1 (4-bit output) / 3 (8-bit) | `(out_bits−2)/2` | base | 1 | 3 |
| `B.05` | 2 | 2 | 2 | 2 | base | 2 | 2 |
| `B.07` | not written | `out_bits/2` (2 at 4-bit output, 4 at 8-bit) | not written | `out_bits/2` | base | not written | not written |
| `B.08` / `B.09` | not written | `0x3210` / `0x7654` (≤4-bit output), `0x5410` / `0x7632` (8-bit), `0x5140` / `0x7362` (6-bit) | not written | as the 3×3 | base | — | — |
| `B.15` | `in_bits/2 − 1` (4-bit), 2 (8-bit), 3 (16-bit) | same | 1 (4-bit input) / 2 (8-bit) | same | 2 | 2 | 2 |
| `B.1b` | 0 | `s − 1` | 0 (also at stride 2) | `s − 1` | base | 1 (stride 2); 0 immediately before `nncmd 0x00` | 0 |
| `B.10` | 0, after the arm | not written (reset 0) | per row quad (§15.2) | not written | base | per row pair (§15.2) | per unit and block (§15.2) |
| `B.11` | `2 + in_bits·nA`, after the arm | `(nA · in_bits·K·walk/2) >> 2`, once | not written | `(in_bits·K·walk/2) / 4` (one group) | `2 + in_bits·nA` = `2 + 8·nA` (1×1) | — | not written |
| program, `B.1a` | two slots (`B.18 = 8` between), `(len_B<<4) \| len_A`, each length 2 (one group) or 5 | two slots of four entries, `0x44` | one slot: 4 (3×3 stride 1, 1×1) / 3 (3×3 stride 2) | the same pair in both slots, `0x22` | base | one slot of two entries, 2 | one slot of two entries, 2 |
| `B.1c` | 0 before each pass's weight feed | 0 before every (row pair, pass) of a multi-pass layer | 0 after the arm | not written | base | — | 0 before the weight feed |
| `B.1e` | 0 before each pass's table | — | 0 before the table | — | base | — | — |

### 15.2 Per-tile fields by recipe

Written inside the tile loop, or once in the prologue where the cell says "prologue".

| field | 1×1 4-bit | 3×3 4-bit | row quad | two slot | 8-bit | image | fp heads |
|---|---|---|---|---|---|---|---|
| `A.09` / `A.0b` | 0 / 0 (prologue) | `2s·rp − 1` / −1, before block 0 of every (row pair, pass) | `4s·q − pad` / `−pad`, per row quad | `4s·q − 1` / −1, per row quad | base | 0 / 0 (prologue) | 0 / 0 (prologue) |
| `A.10`, `A.11` | 0 before every unit-A MAC phase | 0 after the block's tiles, before `nncmd 0xc0` | 0 per block, before its first MAC phase | 0 after the block's tiles, before `nncmd 0xc0` | base | 0 per tile | 0 per column block |
| `B.02` | 0 per column block | `4·gpp·k` per column block (pass `k`) | `4·g` before each group's commit | 0 per block | base | 0 per tile | — |
| `A.14`, `A.12` | not written | `D·9·w_bits/2` after every tile | `A.12 = K²·w_bits/2` at every group switch (`nncmd 0xe0`); `A.14` not written | `9·w_bits/2` after every tile | base | — | — |
| `B.10` | static (§15.1) | not written | 0 at the top of every row quad | not written | base | 0 at the top of every row pair | 0 before unit A's feed (only when `nA > 0`), `8·(nA+1)` before unit B's feed, per column block |
| `A.1c` / `A.1d` | static | static | not written | not written | base | static | unit A: `cA−1` / `lA−1`; unit B: `cB−1` / `lB−1`, rewritten before each unit's phase |
| `B.1c` | 0 per pass (before the weight feed) | 0 per (row pair, pass) of a multi-pass layer | prologue only | not written | base | — | prologue only |

### 15.3 MAC-phase word registers by recipe

Each recipe names two `vr31` words in its `nnmac` instructions, one per MAC phase; the array samples the word
file at every `nnmac` ([§3.6](#36-mac-phases-and-the-walk-position)). Where the words are stepped between
blocks, the additions are the operand walk of the two units — no `nnrwr` reads them. Instruction-field form;
the payload is in parentheses.

| Recipe | Phase 1 | Phase 2 | The named words, and how they are initialised and stepped |
|---|---|---|---|
| 1×1 4-bit ([§11.1](#111-11-4-bit-operands)); 8-bit input ([§11.5](#115-8-bit-input-variants)) | `vw4,1` (`0x81`), unit A | `vw5,3` (`0xa3`), unit B | `w4 = 0`, `w5 = 4·in_bits·nA + 8` (`16·nA + 8` at 4-bit, `32·nA + 8` at 8-bit); neither is stepped — the walk program does the walking |
| 3×3, `D ≥ 2` ([§11.2](#112-33-4-bit-operands)) | `vw9,1` (`0x121`), unit A | `vw10,2` (`0x142`), unit B | `w9 = −1`, `w10 = −1 + 4·B.11`; both `+= 4s` (`addrw` with `w5 = 4s`) after every column block, before `nncmd 0xc0` |
| single input group, two slot ([§11.4](#114-single-input-group-two-slot)) | `vw9,0` (`0x120`), unit A: output rows `4q, 4q+1` | `vw10,0` (`0x140`), unit B: rows `4q+2, 4q+3` | — |
| single input group, row quad ([§11.3](#113-single-input-group-row-quad)) | `vw1,0` (`0x20`), output rows `4q, 4q+1` | `vw2,1` (`0x41`), rows `4q+2, 4q+3` | `w1 = 4s·cb − pad`, `w2 = w1 + 100`; both `+= 4s` per column block |
| RGB image layer ([§11.6](#116-the-rgb-image-layer)) | `vw1,0` (`0x20`), one output row | `vw2,1` (`0x41`), the other row of the pair | `w1 = 64·cb`, `w2 = w1 + 4`; stepped by `addrw` per tile |
| floating-point heads ([§11.7](#117-floating-point-output-heads)) | `vw2,1` (`0x41`) per inner chunk, `vw2,3` (`0x43`) for the last chunk, unit A | `vw14,1` (`0x1c1`) per inner chunk, `vw14,3` (`0x1c3`) for the last chunk, unit B | `w2 = 0` before unit A's phase, `w2 += w3` (`= 32·cA`) after every chunk; `w14 = 32·(nA+1)` before unit B's phase, `w14 += w4` (`= 32·cB`) after every chunk; a unit with one chunk issues only the `,3` form; with `D = 1` unit A is skipped |

**Unconfirmed.** Cells marked "—" are not stated by the recipe section concerned: [§11.6](#116-the-rgb-image-layer)
gives its prologue as a sequence in which `A.03`, `A.0f`, `A.1a`, `B.08`/`B.09`, `B.11`, `A.12`/`A.14`, `B.1c`
and `B.1e` do not appear; [§11.7](#117-floating-point-output-heads) likewise for `A.0f`, `A.1a`, `B.08`/`B.09`,
`B.02`, `A.12`/`A.14` and `B.1e`; [§11.2](#112-33-4-bit-operands) does not mention `B.1e`, nor
[§11.4](#114-single-input-group-two-slot) `B.1e`; and [§11.4](#114-single-input-group-two-slot) does not state the
initial values or the step of `w9`/`w10`. Whether an unlisted field is left at reset is not established; see
[Appendix C](#appendix-c--not-characterised).
## 16. Runtime API

This chapter describes the companion implementation: one C library written against this manual. Its
names (`yx_dev`, `nna_config_apply`, `yx_conv_*`, and the rest) are those of that library, not of the
hardware, which is the subject of Chapters 1–15. The library exposes the accelerator as six layers, and
code should be written against the highest one that expresses what it needs.

| Layer | Header | Contents |
|---|---|---|
| Convolution | `conv_i4_1x1.h`, `conv_i4_3x3.h`, and one further header per entry point of §16.3 | One entry point per recipe of [§11](#11-the-recipes) |
| Operations | `nna/nna.h` | Reset, arm, configuration records, program push, feeds, commits, drains |
| Registers | `nna/nna_regs.h` | Named fields, ports, commands, program-word constructors |
| Instructions | `nna/nna_isa.h`, `mxu3.h` | The six instructions and the vector load/store primitives |
| DMA | `nna/nndma.h` | Descriptors, kicks, waits |
| Device | `device.h` | Open, map, cache maintenance |

**Naming rule.** An identifier is an uppercase macro only when one of its arguments *is* an instruction
immediate — a field, a port, a MAC phase, a word index — which the assembler requires as a compile-time
constant. Everything else is a `static inline` function with its immediates fixed inside. There are no
exceptions, so an uppercase name in this API always signals a compile-time constraint.

### 16.1 Device

```c
typedef struct { uint8_t *nmem, *oram, *des, *io; int fd_mem, fd_nna, fd_lock; } yx_dev;

int  yx_dev_open (yx_dev* d);   /* pins CPU 0, opens /dev/soc-nna, maps the windows */
void yx_dev_close(yx_dev* d);
int  yx_cache_sync(yx_dev* d, uint32_t phys, uint32_t len, int dir);   /* 1 = to device, 2 = from device */
```

`yx_dev_open` performs the pinning and ordering constraints of [§8](#8-worked-example) and leaves the
windows mapped: `nmem` is the tensor pool; `oram`, `des` and `io` are the accelerator's own windows.

```c
#define YX_NMEM_PHYS 0x07000000u    /* physical base of the tensor pool */
#define YX_ORAM_PHYS 0x12620000u
#define YX_ORAM_SIZE 0x00060000u    /* 384 KB */
```

Descriptors take physical addresses: `YX_NMEM_PHYS + offset` for a tensor at `offset` in the pool.

### 16.2 Geometry and buffers

A convolution is described by its shape and by where its operands live.

```c
typedef struct {
    int cin, cout;      /* channel counts; padded to a multiple of 32   */
    int H, W;           /* OUTPUT height and width                      */
    int in_zp;          /* input zero point, and the border pad value    */
    int stride;         /* 1 or 2                                        */
    int in_bits;        /* 4 or 8                                        */
    int w_bits;         /* 4 or 8                                        */
    int out_bits;       /* 4, 8, or 32 for an fp32 head                  */
    int out_ch;         /* heads only: the real channel count            */
} yx_conv_geom;

typedef struct {
    uint32_t in_off, w_off, tbl_off, out_off;   /* offsets into the nmem pool */
    const yx_in_part* in;                        /* optional: several inputs   */
    int      n_in;
} yx_conv_bufs;
```

`H` and `W` are the **output** dimensions. The input size follows from the stride, and it is the input size
that the `A.0c` and `A.0e` fields carry. For a channel concatenation, `in` lists the parts:

```c
typedef struct {
    uint32_t off;       /* nmem offset of this tensor's first plane */
    uint32_t stride;    /* bytes between its planes                 */
    int      planes;    /* how many 32-channel planes it contributes */
} yx_in_part;
```

The parts need not be adjacent and need not be in plane order ([§11](#11-the-recipes)).

### 16.3 Running a convolution

```c
int yx_conv_i4_1x1 (yx_dev*, const yx_conv_geom*, const yx_conv_bufs*, int mode);
int yx_conv_i4_3x3 (yx_dev*, const yx_conv_geom*, const yx_conv_bufs*, int mode);
int yx_conv_i4_gen (yx_dev*, const yx_conv_geom*, int kernel, const yx_conv_bufs*, int mode);   /* row quad, §11.3 */
int yx_conv_i8of   (yx_dev*, const yx_conv_geom*, const yx_conv_bufs*, int mode);
int yx_conv_first  (yx_dev*, const yx_conv_geom*, const yx_conv_bufs*, int mode);
```

`mode` selects how far the call goes, which is the basis of the three checks of [§8.8](#88-verify-the-output):

| Mode | Behaviour |
|---|---|
| `YX_CONV_MODE_FULL` | Run the layer |
| `YX_CONV_MODE_CFGONLY` | Reset, configure, load the program, arm, then stop — so the configuration can be read back |
| `YX_CONV_MODE_DMAONLY` | Configure and issue every DMA, but feed nothing and run no MAC |

A complete call:

```c
yx_conv_geom g = { .cin = 128, .cout = 128, .H = 20, .W = 20,
                   .stride = 1, .in_bits = 4, .w_bits = 4, .out_bits = 4 };
yx_conv_bufs b = { .in_off = IN, .w_off = W, .tbl_off = TBL, .out_off = OUT };

if (yx_conv_i4_3x3(&dev, &g, &b, YX_CONV_MODE_FULL) != 0) { /* handle */ }
```

### 16.4 Operation layer

This layer is the one to use when writing a new recipe.

```c
/* configuration: describe the layer, then apply it */
nna_layer_config_t config;                        /* one member per field      */
void nna_config_init (nna_layer_config_t*);       /* every member unset        */
void nna_config_apply(const nna_layer_config_t*); /* write what is set         */
void nna_program_load(const uint16_t* slot_a, int len_a,
                      const uint16_t* slot_b, int len_b);

/* configuration: the word file underneath it */
nna_config_t record;                       /* 16 words, 64-byte aligned; declare it static */
void nna_config_clear(nna_config_t*);
void nna_config_stage(const nna_config_t*);       /* one vector load into vr31 */
void nna_word_set(int word, uint32_t value);      /* set one word register     */
NNA_WRITE_FIELD(FIELD, WORD)                      /* nnrwr FIELD, wWORD        */
NNA_WRITE_FIELD_VALUE(FIELD, WORD, VALUE)         /* stage then write          */
NNA_PUSH_PROGRAM_WORD(WORD)                       /* nnrwr B.19, wWORD         */

/* control */
void nna_reset(void);                      /* B.1b <- 0 ; nncmd 0x00 */
void nna_arm(void);                        /* nncmd 0x60             */
void nna_precommit(void);                  /* nncmd 0x84             */
void nna_commit_raw(void);                 /* nncmd 0x83             */
void nna_commit_packed(void);              /* nncmd 0x8b             */
void nna_commit_packed_keep(void);         /* nncmd 0x8f             */
void nna_block_end(void);                  /* nncmd 0xc0             */

/* operands */
void nna_push_weight_slice(const void* oram_slice);        /* 256 B over four ports */
void nna_push_requant_half(const void* oram_table);        /* 128 B over the table port */
void nna_feed_window_pair_unit_a(const void* w0, const void* w1);
void nna_feed_window_pair_unit_b(const void* w0, const void* w1);

/* execute and read back */
NNA_RUN_MAC(WORD, PHASE)                   /* nnmac vwWORD, PHASE    */
NNA_DRAIN_PACKED(VR)                       /* nndrd vrVR, 0x20       */
NNA_DRAIN_ACCUMULATOR(VR, ROW)             /* nndrd vrVR, ROW        */
```

`nna_reset` includes the `B.1b` write; there is no unguarded reset in this API.

`nna_config_apply` writes one field per member that is set, deriving the encodings from the layer's
dimensions and dtypes; a member left at `NNA_CFG_UNSET` produces no instruction, which is how a recipe
leaves a field at its reset value. It neither arms nor loads the program, and it owns all sixteen word
registers while it runs, leaving them zero — so the words a recipe needs live are installed afterwards
([§4](#4-the-configuration-model)). The encoding helpers `nna_operand_precision`, `nna_in_elem_class`,
`nna_out_elem_class`, `nna_tile_class`, `nna_stride_code`, `nna_tap_mask`, `nna_readout_lag` and
`nna_lane_perm_row0/1` are the formulas of [§13](#13-configuration-fields), named once.

### 16.5 DMA layer

```c
void nndma_descriptor(yx_dev*, uint32_t index, uint32_t ddr_phys,
                      uint32_t oram_offset, uint32_t bytes, int more);
void nndma_kick_read0(yx_dev*, uint32_t index);
void nndma_kick_read1(yx_dev*, uint32_t index);
void nndma_kick_write(yx_dev*, uint32_t index);
void nndma_wait_read0(yx_dev*);
void nndma_wait_read1(yx_dev*);
void nndma_wait_write(yx_dev*);
void nndma_finish_write(void);
```

`nndma_descriptor` performs the descriptor address arithmetic of [§6](#6-memory-and-dma), including
forming the ORAM field by addition. The channels are named read0, read1 and write rather than by cargo,
because the assignment differs between recipes.

### 16.6 Instruction layer

```c
NNA_NNRWR(WORD, FIELD)      NNA_NNDWR(VR, PORT)     NNA_NNMAC(WORD, PHASE)
NNA_NNRRD(VR, BLOCK)        NNA_NNDRD(VR, BANK)     NNA_NNCMD(CMD)
NNA_VLD64(VR, PTR)          NNA_VST64(VR, PTR)      NNA_WORD_LOAD(WORD, PTR)
NNA_WORD_IMMEDIATE(WORD, IMM)                       NNA_WORD_ADD(DST, SRC)
NNA_FENCE()
```

Every macro emits its instruction ([§12](#12-instruction-set)) behind a fence. Register operands are the
enumerations `enum nna_vector_register` (`NNA_VR0` … `NNA_VR31`) and `enum nna_word_register` (`NNA_W0` … `NNA_W15`).

### 16.7 When to use the instruction layer

Code reaches past the operation layer only for:

- **a feed pattern no operation expresses** — for example the image layer, which pushes registers a shuffle
  network has just written rather than windows loaded from ORAM;
- **an explicit vector register requirement**, where the recipe's drain and store must use the registers the
  procedure specifies;
- **diagnostics**, such as reading raw accumulator row-banks alongside a normal run.

If a recipe spells out vector register numbers or field writes for any other reason, either the operation
layer is missing an operation or the recipe is doing something genuinely unusual; the first is fixed by
adding the operation, the second by documenting the step in the recipe.
# Appendices

## Appendix A — Sources

No vendor document describes the accelerator itself. The material in this manual rests on the following.

**The instruction set is vendor-documented.** Ingenic's *XBurst2 ISA MXU3 Programming Manual*
(`XBurst2_ISA_MXU3_PM_1.0.pdf`, revision 1.0, 191 pages, published on Ingenic's FTP server under
`/SOC/CPU/`) gives, in its section 3.16 "Neural Networks Accelerate", the encoding and operand types of all six
NNA instructions, and in §2.2 the vector register file. It states the operation of `nnmac` as
`NNA_mac(VWR[vwrp], imm)`, which is the source for the statement that the instruction reads the named word
register ([§3.6](#36-mac-phases-and-the-walk-position)). It says nothing about what lies behind the
coprocessor interface; it defers that to an "NNA manual" that is not publicly available.

**The vendor inference library.** The build every function address in
[Appendix B](#appendix-b--vendor-executor-map) refers to is `libvenus.m.so` version 0.0.3.ALPHA, built
2022-05-18 with GCC 7.2.0 r5.1.1 and glibc 2.29, byte-identical to the copy shipped in the Magik toolkit's
`InferenceKit/nna2/mips720-glibc229/lib/uclibc/`. The application-facing `libvenus.so` on the camera is a
newer 0.1.7.3.ALPHA (2023-06-20, r5.1.5) with the same executors. Static analysis used a disassembler
language extension generated from Ingenic's own binutils opcode tables, so that the NNA and MXUv3
instructions appear under their official mnemonics.

**Measurement on the device.** Field meanings, instruction semantics, the readout behaviour and the
requantization arithmetic were established on a T41 camera board (`Ingenic_T110_T41ZN_001`) by perturbing
one quantity at a time in a working run and by reading internal state back through the configuration
readback path. The confidence letters and the **Not characterised** markers throughout the manual record
what that established and what it did not ([Appendix C](#appendix-c--not-characterised)).

**Vendor material that exists.** The T4x SDK's `soc-nna` kernel driver (DMA and memory plumbing only;
[§6.8](#68-the-driver-interface)), the T41 kernel base map and clock-controller header, u-boot's L2 sizing
code, the *XBurst2 CPU Core Programming Manual* (the CCU chapter and the L2 cache section), the *Venus
Programming Manual* (a framework API and the operator envelope of [§2.4](#24-precision)), and the Magik
toolkit headers. None describes the array.

**What is inferred rather than documented.** The 32-wide reduction and output structure of
[§1.1](#11-the-atomic-operation) comes from the observable operation — one `nnmac` produces 32 output
channels and one committed tile reads back as 16 row-banks of 16 int32 — not from a vendor statement.

**Related documents** in the same directory:

| Document | Contents |
|---|---|
| `PLATFORM.md` | The measured physical memory map and the `/dev/soc-nna` device model — the authority for every address in this manual |
| `BOARD_PERIPHERALS.md` | The camera board outside the NNA: GPIO, motors, audio, device nodes, the DDR split |
| `MXUV3.md`, `MXU3_OPCODE_TABLE.md`, `MXUV3_ISA_HARDWARE_RE.md` | The MXUv3 vector coprocessor: instruction set, opcode table, and the operations used by the image-layer feed |
| `T41_OPERAND_DATA_LAYOUT.md` | The canonical operand data layouts |
| `T41_NNCMD_EXPERIMENTS.md` | The `nncmd` semantics |
| `MAGIK_PACKING_QUANT.md` | The model file's weight packing and quantization |
| `T41_DRIVER_INTERFACE.md`, `NNA_CU2_PINNING.md` | The kernel driver ABI and the coprocessor-enable pinning requirement |

The remaining per-topic notes in the directory (`t41_nna_architecture.md`, `T41_NNA.md`,
`AIE_FEATURE_SET.md`, `T41_NNA_ISA.md`, `T41_NNA_CONFIG_REGISTERS.md`, `T41_OPERAND_BANK_FEED.md`,
`T41_NNA_DESCRIPTOR_DERIVATION.md`, `AIE_CONV_ALGORITHM.md`, `T41_CONV_EXECUTOR_PROGRAM.md`,
`T41_GENERIC_CONV2D_DESIGN.md`, `T41_CONV_LAYER_STRUCT.md`) are the working notes this manual consolidates;
where they disagree with this manual, this manual is current.
## Appendix B — Vendor executor map

The NNA does not know which procedure drives it, so this appendix documents how the vendor library
(`libvenus`) organises its own code; it is not a constraint on new software. It serves two purposes:
locating the vendor code behind a recipe in [Chapter 11](#11-the-recipes), and predicting which vendor
procedure a given layer geometry takes when comparing against vendor output. It is the one place in this
manual where vendor function addresses (`FUN_xxxxxxxx`) appear.

### B.1 Executors and the recipes derived from them

`libvenus` has some sixty convolution executors and picks one by geometry and dtype: the kernel setup
`FUN_001b8624` tries planner candidates in a fixed order (B.2), each of which either plans and runs the
layer or refuses it with an error code. Any executor whose guards a layer passes computes the same result.
Seven executors are exercised by a YOLOX-S network; the table lists those seven, plus the never-reached
`D = 1` member of the 3×3 family, together with the recipe in Chapter 11 that each corresponds to.

The 4-bit 3×3 executor's entry guard rejects every 4-bit-input tensor whose weights exceed ORAM, so such a
layer (layer 34 of YOLOX-S) runs on the im2col executor `FUN_0039b9d0`; the two executors compute the same
convolution, and the streaming form of the 3×3 recipe gives the same output.

| vendor executor (`libvenus.so`) | geometry it serves | YOLOX-S layers | recipe |
|---|---|---|---|
| `FUN_0026e540` (`.m` build `FUN_00104cd0`), the `base_i4` 3×3 core | 3×3 pad 1, stride 1 or 2, `Cin ≥ 64`, `Dout ≤ 64`, 4- or 8-bit weights and output, one or several weight passes | 27 | [3×3, 4-bit operands](#112-33-4-bit-operands) |
| `FUN_0039b9d0`, the im2col executor | as above with weights larger than ORAM, the weight blob streamed per pass | 1 (layer 34) | [large weights](#118-large-weights-and-channel-concatenation) |
| `FUN_003ac1bc`, the `base_i4` 1×1 executor | 1×1, `2 ≤ Cin/32 ≤ 32`, 4-bit input; also its 8-bit-input arm | 40 + 2 (layers 2, 6) | [1×1, 4-bit operands](#111-11-4-bit-operands); [8-bit variants](#115-8-bit-input-variants) |
| `FUN_002bf4b8` and its 1×1 sibling `FUN_002d06b0`, the single-input-group executors | `Cin ≤ 32`, 1×1 or 3×3, stride 1 or 2 | 3 (layers 1, 3, 4) | [row quad](#113-single-input-group-row-quad) |
| `FUN_00270df0`, the `D = 1` member of the 3×3 family (never reached on the camera) | `Cin ≤ 32`, 3×3 | (layer 1, alternative) | [two slot](#114-single-input-group-two-slot) |
| `FUN_003a50e0`, the `base_i8of` head executor | 1×1, 8-bit input and weights, fp32 NHWC output with the real channel count | 9 heads | [floating-point heads](#117-floating-point-output-heads) |
| `FUN_00369db4`, `first_layer_i8` | 3-channel 8-bit image, 3×3 stride 2, the window built on the MXU | 1 (layer 0) | [the RGB image layer](#116-the-rgb-image-layer) |

Two more executors are characterised but have no recipe of their own: `FUN_001c6750`, the int8 `Cin = 4`
pointwise specialist, whose shuffle feed is the image layer's; and `FUN_001f4f60`, the general `Cin ≥ 32`
int8 1×1 executor of the MobileNet path, whose configuration, weight stream and readout are understood
statically and are the source of B.3.

---

### B.2 Dispatch

The kernel setup tries planner candidates in a fixed order and stops at the first that returns success. Its
choice of candidate sequence is made on the **first input tensor's `D`**: when that is 1 it tries
`FUN_002d06b0`, then `FUN_002bf4b8`, then `FUN_0039b9d0`; otherwise `FUN_003ac1bc` (the 1×1 executor) first,
then `FUN_0039b9d0`, `FUN_0019e890` (the 3×3 planner) and `FUN_00243920`.

`FUN_0019e890` is both the 3×3 planner and the dispatcher for that family. It checks the operator
(`K ∈ {1,3}`, equal strides ≤ 2, pads ≤ 1, `Dout ≤ 64`), allocates ORAM, writes the descriptor chains, then
tries executors from a table:

| `K = 3` | First choice | Fallback |
|---|---|---|
| `D ≥ 2, Dout ≥ 2` | `FUN_0026e540` | `FUN_00263cc0` |
| `D ≥ 2, Dout = 1` | `FUN_00276500` | `FUN_00269340` |
| `D = 1, Dout ≥ 2` | `FUN_00270df0` | `FUN_00266a40` |
| `D = 1, Dout = 1` | `FUN_002788e0` | `FUN_0026bea0` |

The stride is not part of the choice. `FUN_00243920` is not a candidate for any 4-bit-input layer: its entry
guard requires an 8-bit input tensor.

The `D = 1` executors classify a layer by output width against its per-tile work and accept only one class,
so several 4-bit-output single-input-group layers are refused by both candidates of their row and fall
through to `FUN_002d06b0` and `FUN_002bf4b8`. A further CPU-side check on a scratch-pool size can reject a
layer after configuration has already been written; that is why some executors appear in a trace for a few
instructions and then exit.

### B.3 Byte-count formulas of the general int8 executor

Read from the code of `FUN_001f4f60` (M). The formulas of the 4-bit recipes are in
[Chapter 11](#11-the-recipes).

| Quantity | Formula |
|---|---|
| Feature bytes per descriptor | `round64(Win · Cin) + 64`, the `+64` being the convolution-window halo |
| Feature row stride | `round64(Cin · 32)` |
| Output tile bytes | `round64(Cout · 32)`, `Cout` padded to 32 |
| Weight super-tile | 1,024 (int8 32 × 32) or 768 (dtype code 8) |
| Requantization table | 256 per commit |
| Output-pixel MAC count | `ceil(Hout · Wout / 8)` |
| Feature rows per pass | `min(5, rows that fit in ORAM)` — hard cap 5 |

### B.4 The ConvOp descriptor

A convolution is described to an executor by a **ConvOp** object. The two executor families read partly
different structures: the first table is the general `Cin ≥ 32` int8 executor's view (offsets verified in
two sibling executors unless noted); the second lists the fields the 4-bit executors read.

| offset | field |
|--------|-------|
| `+0xc8` | activation-mode selector (`== 2` is special) |
| `+0xe8` | weight-arena pointer (must be non-null, else the executor bails) |
| `+0xec` | **requant left-shift** (byte); if `0xff`, defaults to 2 when `+0xc8 == 2`, else 5 |
| `+0xf0` | **ReLU/floor selector** (u16); `!= 0xffff` → floor 0 (ReLU); `== 0xffff` → floor −FLT_MAX (no ReLU) |
| `+0xf8` | mode word (low 3 bits checked) |
| `+0x118` | **kernel-size vector** (both elements must be 1 — this executor is 1×1 only) |
| `+0x124` | **stride vector** (each ≤ 2): `[0]` = column stride, `[1]` = row stride |
| `+0x15c` | split/group divisor |
| `+0x160` | row-pointer-array rebase |
| `+0x174` | stride assert (`{1}` or 4–5) in this executor; the 4-bit executors read it as the activation kind (`(x−2) <u 2` → `B.05` bit 3; `0xd` → float constants for the MXU requant path, [1×1 recipe](#111-11-4-bit-operands)) |
| `+0x17c` | mode word (low 3 bits must be 0) |

Fields the 4-bit executors read ([Chapter 11](#11-the-recipes); `FUN_001b8624` copies them from the layer
object):

| offset | field |
|--------|-------|
| `+0x80` | precision selector: 2 → `A.08 = B.06 = 2` (else 3); must not be 1 |
| `+0x9c` | input zero-point / pad value → `A.0f` (masked to the input width) |
| `+0xa0` | pointer to the CPU-side scratch pool (word 0 = its size in KB); the planners carve their kick tables from it ([single input group](#113-single-input-group-row-quad)) |
| `+0xa4` | requant-mode code (u16; `& 0xff`): 2 → `B.03 = 0`, 6 → 1, 0xb → 2 |
| `+0xa8` | zero-point byte (masked to the input width): the 3×3 planner requires it 0; the 1×1 and single-input-group store leaves use it as the clamp floor (`vr29`), 0 on every YOLOX-S layer |
| `+0xb0` | `& 7` = mode: 2 = hardware requant with the table in unit B (every YOLOX-S layer), 1 = MXU requant, 3 = the float/zero-point path |
| `+0xc0` | group-split flag (0 on YOLOX-S) |
| `+0xdc` | pointer to an optional image-preprocessing record (`FUN_00369db4`: flags word, then per-channel float mean at `+0x24..` and scale at `+0x28..0x30`, from which the pad byte per channel is quantized and clamped to 0..255); 0 on layer 0, so the default `0x80` triple is used and that loop is skipped |
| `+0xe0`, `+0xe4` | first input row, input rows (`A.0c = +0xe4 − +0xe0 − 1`) |
| `+0x104` | `Cout` |
| `+0x10c` | pad vector `[top, bottom, left, right]` |
| `+0x118` / `+0x124` / `+0x130` | kernel `[w, h]`, strides `[col, row]` (bytes read), dilation `[w, h]` |
| `+0x174` | activation kind (−1, 0, 1, 4, 5 are the plain int paths; 0xd → float constants; [1×1 recipe](#111-11-4-bit-operands)) |
| `+0x180` | 0 on YOLOX-S (a flag the `D = 1` paths test) |

The four geometry members at `+0x10c`, `+0x118`, `+0x124` and `+0x130` are **`std::vector<int>`**, not
inline arrays: they sit exactly twelve bytes apart — the libstdc++ `{begin, end, end_of_storage}` triple —
and the code dereferences them as such, e.g. `*(*(op + 0x124) + 4)` for the row stride. An inline-array
reading would have `+0x10c`'s four elements overlapping `+0x118`.

Tensor lists hang off the op's argument. The **shape struct** of a tensor is `+0x00 N, +0x04 D (channel
groups), +0x08 H, +0x0c W`; the 3×3 executor's `A.0c = *(+8) − 1`, `A.0e = *(+0xc) − 1` are taken from
the **input** tensor's shape ([Chapter 13](#13-configuration-fields)). On stride-1 layers the input and
output sizes coincide, so those layers cannot distinguish the two readings.

An **operand/quant bundle** hangs off the op too, carrying the DDR base pointers and strides for the
weight, bias and requant-table DMAs (`+0x20`/`+0x38` and `+0x4c`/`+0x64` are the two base/stride pairs
that feed the weight+bias descriptor; `+0x40` and `+0x6c` are dtype enums). Not characterised: which of
the two pairs is the weight and which the bias. What is established: both feed the weight+bias descriptor.
The `Cin = 4` executor's structure additionally exposes the activation zero-points at `+0x184..0x188`
(five zero-point bytes; `+0x184` is the signed input zero-point used in the `acc − in_zp·wsum` correction).

---

### B.5 Vendor program tables

The program each vendor executor pushes, by executor address in the `.m` build (`push` = `nnrwr B.19`,
`LEN` = `nnrwr B.1a`; the program-table format is in [Chapter 14](#14-the-program-table)). Single-slot
programs:

| executor(s) | pushes | LEN |
|---|---|---|
| `000e14e0`, `00209990` | `0x800, 0x801` | 2 |
| `000e38b0 000ea380 00143aa0 0017b6b0 00186ad0 0018f5c0 00197b20 001da490 001e2b10 001eba30 001ed660 00225800 002287a0` | `0x801` | 1 |
| `001fb150 00204b60` / `001fdb90` | `0x806` / `0x804` | 1 |
| `001119c0` | `0x803 0x803 0x803 0x80f` | 4 |
| `00115040` | `0x803 ×4, 0x800, 0x812` | 6 |
| `00129570` / `0012ecc0` | `0x806 ×3, 0x81e` / `0x803 ×3, 0x827` | 4 |
| `0012c330` / `00133640` | `0x805 ×4, 0x800, 0x81e` / `0x805 ×4, 0x800, 0x850` | 6 |
| `001f8a30` | `0x804 ×5, 0x81c` | 6 |
| `00221620` (slot 0) | `0x806 ×4, 0x800, 0xbe9` | 6 |
| `00109900` | `0x1805 0xbd3 0x2805 0x806` | 4 |
| the single-input-group executors ([row quad](#113-single-input-group-row-quad), [two slot](#114-single-input-group-two-slot)) | `0x805 0x814 0x805 0xbe3` (1×1) / `0x2805 0x2005 0xbd4` (3×3 stride 2) / `0x1805 0x80a 0x1805 0xbd9` (3×3 stride 1) | 4 / 3 / 4 |
| `00142630` | `0x800 0x800` | 2 |
| `00149350 001a3270` / `001c6750` / `0016d020` | `0x807 0xbfb` / `0x802 0x80e` / `0x802, f(bits)` | 2 |
| `001f4f60 001ef2c0` | `0x801, 0x800 \| ((2·D−1)·(S>>2) & 0x3ff)` (= `0x807` int8) | 2 |
| `0020b5f0 00210c50` / `0020c960` / `00212450` | `0x801 0xbff` / `0x804 0xbfd` / `0x801 0x81f` | 2 |
| `00217300` / `0021a3e0` | `0x800, f(bits)` / `f(shape)` | 2 / 1 |

Two-slot programs (`B.18 ← 8` between the halves): `000fa5c0 000ffbe0 00104cd0 0010ca90` (the 4-bit
3×3 family: two 4-word programs, `0x7b01 0x1805 0x819 0x3b1` in both slots at stride 1,
`0x7b01 0x2006 0x824 0x38a` at stride 2 with two groups per unit, [3×3 recipe](#112-33-4-bit-operands)),
`000fd2d0 001026f0 001074f0 0010ee40` (3 + 3), and the camera build's 4-bit executor.

The `D = 1` member of the 3×3 family (`FUN_00270df0`) loads the same two entries into both slots
(`B.1a = 0x22`; for example `0x2006 0x0bea` at walk 5, stride 2) — the form of
[§11.4](#114-single-input-group-two-slot).

### B.6 Planner guards of the single-input-group 3×3 executor

The vendor executor behind [§11.4](#114-single-input-group-two-slot) accepts a layer only when all of the
following hold. The symbols are the planner's own: `p2` is the second descriptor record, `walk` the input
rows one row pair consumes, `Kwalk` the kernel size times the walk depth, and `class` the executor class code.

| Guard | Meaning |
|---|---|
| `class = 1` | the 4-bit, `D = 1` class only |
| strides equal and ≤ 2; `Hout` even at stride 2 | |
| `walk + p2[0x54] < 0x1e` | walk depth plus the record's row allowance below 30 |
| `Kwalk · 2 · in_bits/2 · 8 < 0x8001` | one group's walk fits 32 KB of operand |
| `2 · Dout ≥ class` | |
| `Kwalk < 0x1fd` | |
| record 1 `+0x5c` = `+0x60` = 0 | |
| `Hout % 4 == 0` | a whole number of row quads |
| `Dout · 9 · w_bits < 0x242` | the `D = 1` size gate |
| `Dout · tap_slots ≤ 0x120`, `tap_slots = 9 · w_bits / 2` | a single weight pass (36 slots per output group at 8-bit) |

Record 1 `+0x2c` is the constant 1, not a stride. A layer failing the pool check after these guards is
rejected with error 8.
## Appendix C — Not characterised

Every statement in the body marked **Not characterised** is collected here, in document order. The first
column names the section that carries the statement; the third records what *is* established, so that each
item is a ready-made experiment.

| Section | Not characterised | What is established |
|---|---|---|
| 1.1 The atomic operation | Whether they saturate or wrap on overflow is not characterised | — |
| 2.1 The memory hierarchy | Their sizes are **not characterised** | — |
| 2.4 Precision | larger strides are not characterised. | — |
| 2.5 Weights | why the middle row runs backwards is not characterised. | — |
| 3.5 The walk program | the absolute size of one step is not characterised | — |
| 3.6 MAC phases and the walk position | The meaning of the 2-bit immediate is not characterised. | — |
| 3.8.1 Common structure | What the pack stage produces for a padded channel whose `mult` is 0 is not characterised | — |
| 3.8.3 The hardware packed path | what that interpretation is. | no layer observed sets the bit. |
| 4.2 The word register file | which words each executor family's MACs read beyond the ones the `nnmac` operands name, and what `w6`, `w13` and `w14` contribute in the 3×3 executor. | the result depends on them, and the safe rule is to load every word the vendor's prologue loads, step the ones it steps, and zero the rest. §7.9 develops this rule. |
| 4.5 Configuration readback | the meaning of L0 words `w2`, `w4`, `w5`, `w6` and `w7`. | the values above are present at reset and `w0` bit 0 records that a write has occurred (Appendix C). |
| 5.3 Program loading | how a slot declared 0 (which stores as 8) and never pushed is treated when the second unit runs. | every single-slot recipe writes `len_B = 0`, never pushes slot B, and is exact — including one that runs the unit-B MAC. |
| 5.8 Commit | whether readout bank 0 remains readable after `0x8b` as it does after `0x8f`. | the 3×3 recipe reads bank 0 after `0x8f`; the floating-point heads read it after `0x83`; in the image-layer configuration, bank-0 reads after its `0x8b` return stale data |
| 5.8 Commit | their isolated effects are not characterised | — |
| 7.1 The B.1b reset hazard | whether writing another field before the reset would serve equally. | writing `B.1b = 0` immediately before `nncmd 0x00` is sufficient. |
| 7.7 Readout lag and stale reads | whether readout bank 0 is readable after every commit form. | bank 0 holds the committed tile after `nncmd 0x8f` and `0x83`; in the image-layer configuration, bank-0 reads after its `0x8b` commit return stale data and must not be used to observe results. |
| 8.5 Configure | `A.0c`'s role in this recipe when H ≠ W. | the vendor's 1×1 procedure writes `W − 1` into both fields; the reference writes `H − 1` and `W − 1`; both are exact on every layer with H = W. |
| 11.2.6 Variants and not-characterised items | the effect of `nncmd 0xc0` at the end of a column block. | it is issued after the `w9/w10` step of every column block, and the recipe is complete with it in place. |
| 11.2.6 Variants and not-characterised items | the role of `B.1c` in this recipe, which writes it 0. | the 1×1 recipe uses `B.1c` as the weight-stream offset (§13.4); here it is written 0 (from `w2`) before every (row pair, pass) of a multi-pass layer and never any other value; it is not a weight offset. |
| 11.2.6 Variants and not-characterised items | whether `A.03` influences this recipe at all. | `A.03 = in_bits/2 − 1` (1 for 4-bit input) at stride 1, the vendor writes 0 at stride 2, and both values produce identical output. |
| 11.3.6 Variants and not-characterised items | how the unpushed slot B (declared 0, which stores as 8) is treated when the second phase runs. | this recipe writes `len_B = 0`, never pushes slot B, and is exact. |
| 11.3.6 Variants and not-characterised items | what `A.19 = 20` and `A.17 = 200` mean to the array beyond `A.17 = A.19 × (7·stride + K)` at stride 1, K = 3, and why they stay fixed otherwise. | every variant writes them and is exact. |
| 11.4.6 Variants and not-characterised items | the meaning of `nnmac vw9,0` / `nnmac vw10,0` and `nncmd 0xaf` / `nncmd 0xeb` beyond "unit A / unit B phase, commit". | `nnmac vw9,0` + `nncmd 0xaf` produces output rows `4q, 4q+1` from window rows `r0 …`, and `nnmac vw10,0` + `nncmd 0xeb` produces rows `4q+2, 4q+3` from `r0 + 2s …`, with the same two program entries in both slots. |
| 11.4.6 Variants and not-characterised items | why `B.1a = 0x22` selects both slots. | the pair loads with `B.18 = 8` between the two pushes and `B.1a = 0x22` after the second. |
| 11.5.6 Variants and not-characterised items | input widths other than 4 and 8. | the rules above are written in terms of `in_bits` and hold at both 4 and 8 (Appendix C). |
| 11.6.6 Variants and not-characterised items | why the walk program is a single slot of two entries (`0x802, 0x80e`, `B.1a = 2`) rather than a per-tile step. | the reference implementation loads exactly those two entries and the layer's output is correct with them (Chapter 14). |
| 11.6.6 Variants and not-characterised items | what `B.1b = 1` selects in the MAC stage. | it is written for stride 2 here and in the 3×3 stride-2 recipe, 0 elsewhere, and it must be re-written to 0 before the next `nncmd 0x00`. |
| 11.6.6 Variants and not-characterised items | which MAC phase produces which of the two output rows. | the first drain register `vr22` always holds row `2r` and `vr0` row `2r+1`; the vendor's naming of the two units is inconsistent, so the recipe is written by MAC word and register. |
| 11.7.6 Variants and not-characterised items | whether the NNA accepts a program of more than one slot for this recipe. | a single slot of two entries with `B.1a = 2` is sufficient for every head geometry. |
| 11.8.6 Variants and not-characterised items | the descriptor members `t->MBuffer->data` and `t->byte_offset` as names. | the shape struct `N, D, H, W`. |
| 11.8.6 Variants and not-characterised items | the strided-view and grouped-convolution paths of the second table. | the flag is clear and the divisor is 1 on every layer of the reference network. |
| 11.8.6 Variants and not-characterised items | the two-slope table and its expansion from the 256-byte blob (read, not implemented). | the sequence above and the three table offsets, from the procedure's code. |
| 12.5 nnmac — multiply-accumulate | whether the array writes the word back, and the meaning of the immediate beyond its use as inner/last marker. | the array reads the word — the instruction is listed in the Ingenic ISA manual section 3.16 as `NNA_mac(VWR[vwrp], imm)`, passing the word's value in. |
| 12.7 nndrd — drain results | after `0x8b`: not characterised, [§5.8] | — |
| 13.3 Static fields | its consumer. | no effect at any of its four values on the layers of the reference network; the vendor writes a literal 3 |
| 13.3 Static fields | its function. | one vendor site writes the value 8, and injecting 8 or 1 before the arm changes nothing |
| 13.3 Static fields | what the code selects. | every layer of the reference network writes 2, and raising it to 3 on a 4-bit layer zeroes the output |
| 13.5 Readback blocks | the remaining words. | `w2 = 0xff`, `w4 = 0x00010000`, `w5 = 9`, `w6 = 0x00010000`, `w7 = 0xa` at reset |
| 14.3 Lengths | how a slot declared 0 and never pushed is treated when the second unit runs. | every single-slot recipe writes `len_B = 0`, never pushes slot B, and is exact — including one that runs the unit-B MAC. |
| 14.5 A program, decoded | the absolute size of one program step unit in bytes of ORAM, and the geometric identity of the walk's starting position, which the `vr31` word an `nnmac` names supplies (§3.6). Everything above is relative to that origin. | — |
| B.4 The ConvOp descriptor | which of the two pairs is the weight and which the bias. | both feed the weight+bias descriptor. |
## Appendix D — Register-number cross-reference

Vendor listings and traces print every NNA instruction as one merged 15-bit payload `p` (word bits [20:6]);
this manual prints the instruction-field form from [§12.2](#122-operand-encoding). This appendix is the
conversion between the two, and the complete list of payload numbers the vendor library uses.

### D.1 Payload and instruction-field forms

The split depends on the instruction, because the register and immediate fields sit in different word bits
([§12.2](#122-operand-encoding)):

| instruction | register | immediate | reading of the merged payload |
|---|---|---|---|
| `nnrwr vwN,imm` | `N = p[9:5]` (word of the word-register file, [§4.2](#42-the-word-register-file)) | `p[4:0] \| (p[14:10] << 5)` = `bank \| field` | `bank(bit 11 = B, bit 10 = A) \| (word << 5) \| field` |
| `nnmac vwN,imm` | `N = p[9:5]` | `p[4:0] \| (p[14:10] << 5)` | `(word << 5) \| code` |
| `nndwr vrN,imm` | `N = p[9:5]` (the vector register pushed) | `p[4:0] \| (p[14:10] << 5)` = port | `(port[6:5] << 10) \| (vr << 5) \| port[4:0]` |
| `nndrd vrN,imm`, `nnrrd vrN,imm` | `N = p[4:0]` (the destination) | `p >> 5` = `bank << 5 \| row` | `(imm << 5) \| vr` |

For `nnrwr` the immediate is the field address of [§4.1](#41-banks-and-fields): bit 5 set = bank A (`0x20`),
bit 6 set = bank B (`0x40`), bits [4:0] = field index. `A.xx` is therefore immediate `0x20 | xx` and `B.xx`
is `0x40 | xx`; in the merged form bank A shows as `0x4nn` and bank B as `0x8nn`, with the word number in
bits [9:5]. Bit 14 of `p` (immediate bit 9) is zero in every payload the library uses.

Worked examples:

- `0x40c`: `p[4:0] = 0x0c`, `p[14:10] = 1`, `p[9:5] = 0` → `nnrwr vw0,0x2c` — field `A.0c` written from word 0.
- `0x5e0`: `p[4:0] = 0`, `p[14:10] = 1` → port `0x20`; `p[9:5] = 15` → `nndwr vr15,0x20` — vector register 15 pushed to port ACT_A.
- `0x408` as a drain: `p[4:0] = 8`, `p >> 5 = 0x20` → `nndrd vr8,0x20` — one 64-byte pop of the output FIFO (bank 1, row 0) into vr8.

The same number means different things to different instructions. `0x408` is `A.08` from `vw0` under `nnrwr`
but the FIFO pop above under `nndrd`; `0x400` is `nndwr vr0,0x20` (activation push) and `nndrd vr0,0x20`
(FIFO pop). Likewise the immediate `0x20` is bank A for `nnrwr`, port ACT_A for `nndwr` and the FIFO bank for
`nndrd`, and `0x40` is bank B for `nnrwr` and the requant-table port for `nndwr`.

### D.2 Configuration fields by vendor register number

The 321 distinct `nnrwr` payloads in the vendor library, one row per field ([Chapter 13](#13-configuration-fields)).
The word column is `p[9:5]` of each number in order: `A.0c` ← `40c 42c 44c 4ec 56c 5cc` is `A.0c` written from
`vw0 vw1 vw2 vw7 vw11 vw14`. Readback locations are in [§4.5](#45-configuration-readback).

| field | vendor payload numbers | source word `vwN` for each |
|---|---|---|
| `A.01` | `401 421 441 561` | 0 1 2 11 |
| `A.03` | `403 423 443 563` | 0 1 2 11 |
| `A.08` | `408 428 448 468 568 5a8 5c8` | 0 1 2 3 11 13 14 |
| `A.09` | `409 429 469 489 4a9 4c9 4e9 569 589 5a9 5c9` | 0 1 3 4 5 6 7 11 12 13 14 |
| `A.0b` | `40b 42b 44b 46b 48b 4ab 4cb 4eb 50b 52b 56b 58b 5cb 5eb` | 0 1 2 3 4 5 6 7 8 9 11 12 14 15 |
| `A.0c` | `40c 42c 44c 4ec 56c 5cc` | 0 1 2 7 11 14 |
| `A.0e` | `40e 42e 46e 4ee 58e 5ce` | 0 1 3 7 12 14 |
| `A.0f` | `40f 42f 44f 56f 5af 5ef` | 0 1 2 11 13 15 |
| `A.10` | `410 430 450 470 490 4b0 4d0 4f0 510 530 550 570 590 5b0 5d0` | 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 |
| `A.11` | `411 431 451 471 491 4b1 4d1 4f1 511 531 551 571 591 5d1` | 0 1 2 3 4 5 6 7 8 9 10 11 12 14 |
| `A.12` | `492 4b2 4f2 512 532 592` | 4 5 7 8 9 12 |
| `A.14` | `494` | 4 |
| `A.17` | `417 437 457 477 597 5d7` | 0 1 2 3 12 14 |
| `A.18` | `418 438 458 478 498 578 5d8` | 0 1 2 3 4 11 14 |
| `A.19` | `419 439 459 579 5d9 5f9` | 0 1 2 11 14 15 |
| `A.1a` | `41a 43a 45a 47a 49a 57a 59a 5da` | 0 1 2 3 4 11 12 14 |
| `A.1b` | `41b 43b 45b 4fb 51b 59b 5bb 5fb` | 0 1 2 7 8 12 13 15 |
| `A.1c` | `41c 43c 45c 49c 4bc 4fc 51c 53c 55c 57c 59c 5bc 5dc` | 0 1 2 4 5 7 8 9 10 11 12 13 14 |
| `A.1d` | `41d 43d 45d 4dd 4fd 51d 53d 55d 57d 59d 5bd 5dd 5fd` | 0 1 2 6 7 8 9 10 11 12 13 14 15 |
| `A.1e` | `41e 43e 45e 47e 49e 4be 4de 4fe 51e 53e 55e 57e 59e` | 0 1 2 3 4 5 6 7 8 9 10 11 12 |
| `A.1f` | `41f 43f 45f 47f 49f` | 0 1 2 3 4 |
| `B.01` | `801` | 0 |
| `B.02` | `802 822 882 8a2 8c2 8e2 902 922 942 962 982 9c2 9e2` | 0 1 4 5 6 7 8 9 10 11 12 14 15 |
| `B.03` | `803 843 883 963 9c3` | 0 2 4 11 14 |
| `B.04` | `804 824 844 864 8c4 8e4 904 964 9a4 9c4` | 0 1 2 3 6 7 8 11 13 14 |
| `B.05` | `805 825 865 985 9a5 9c5` | 0 1 3 12 13 14 |
| `B.06` | `806 826 846 866 886 8a6 8c6 8e6 986 9a6 9c6` | 0 1 2 3 4 5 6 7 12 13 14 |
| `B.07` | `8c7 8e7 907 927` | 6 7 8 9 |
| `B.08` | `848 908 928 948 9a8` | 2 8 9 10 13 |
| `B.09` | `869 929 949 969 9c9` | 3 9 10 11 14 |
| `B.10` | `810 830 850 870 890 8b0 8d0 8f0 910 930 950 990 9b0 9d0 9f0` | 0 1 2 3 4 5 6 7 8 9 10 12 13 14 15 |
| `B.11` | `811 831 871 891 8b1 8d1 911 931 951 971 991 9d1` | 0 1 3 4 5 6 8 9 10 11 12 14 |
| `B.15` | `815 835 855 975 995 9b5` | 0 1 2 11 12 13 |
| `B.18` | `818 898 8d8 9d8` | 0 4 6 14 |
| `B.19` | `819 839 859 879 899 8b9 939 979 9b9 9d9` | 0 1 2 3 4 5 9 11 13 14 |
| `B.1a` | `81a 8fa 97a 9ba 9da` | 0 7 11 13 14 |
| `B.1b` | `81b 83b 89b 97b 99b 9bb 9db` | 0 1 4 11 12 13 14 |
| `B.1c` | `81c 83c 85c 87c 89c 8bc 8dc 8fc 91c 93c 95c 97c 99c 9bc 9dc` | 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 |
| `B.1e` | `81e 83e 87e 89e 8be 8de 8fe 91e 93e 95e 97e 99e 9de 9fe` | 0 1 3 4 5 6 7 8 9 10 11 12 14 15 |

Every one of the 321 numbers converts under the D.1 rule to `bank | field` matching its row; no payload with
bit 14 set, and no field index outside the table, occurs in the library. `B.19` is the program-table push
and `B.18` its pointer; `A.10`, `A.11`, `A.12`, `A.14`, `B.01` and `B.1c` are write-only (no readback bits).

### D.3 Ports, drains and MAC codes

Payload numbers for the operand ports ([§3.4](#34-ports-and-positional-feeds)), the drains
([§3.7](#37-accumulators-commit-and-readout)) and the MAC pass codes, converted under the D.1 rule. The
apparent variety of "activation ports" and "table ports" across vendor executors is one port each with
different source registers.

| payload | instruction form | meaning |
|---|---|---|
| `0x00 / 0x22 / 0x44 / 0x67` | `nndwr vr0,0 / vr1,2 / vr2,4 / vr3,7` | weight ports WEIGHT0…WEIGHT3 from vr0…vr3 |
| `0x04 / 0x27` | `nndwr vr0,4 / vr1,7` | weight ports WEIGHT2 / WEIGHT3 from vr0 / vr1 |
| `0x400 / 0x420 / 0x440 / 0x460` | `nndwr vr0,0x20 / vr1,0x20 / vr2,0x20 / vr3,0x20` | activation port ACT_A from vr0…vr3 |
| `0x401 / 0x421` | `nndwr vr0,0x21 / vr1,0x21` | activation port ACT_B from vr0 / vr1 |
| `0x5e0 / 0x600 / 0x640 / 0x660` | `nndwr vr15,0x20 / vr16,0x20 / vr18,0x20 / vr19,0x20` | ACT_A from vr15 / vr16 / vr18 / vr19 |
| `0x540 / 0x560` | `nndwr vr10,0x20 / vr11,0x20` | ACT_A from vr10 / vr11 |
| `0x800 / 0x820` | `nndwr vr0,0x40 / vr1,0x40` | requant-table port TABLE from vr0 / vr1 |
| `0x940 / 0x960` | `nndwr vr10,0x40 / vr11,0x40` | TABLE from vr10 / vr11 |
| `0x408 / 0x409` | `nndrd vr8,0x20 / vr9,0x20` | output-FIFO pop (bank 1, row 0) into vr8 / vr9 |
| `0x40a … 0x40d` | `nndrd vr10,0x20 … vr13,0x20` | FIFO pop into vr10…vr13 |
| `0x416 / 0x400 / 0x401 / 0x402` | `nndrd vr22,0x20 / vr0,0x20 / vr1,0x20 / vr2,0x20` | FIFO pop into vr22 / vr0 / vr1 / vr2 |
| `0x1b + 0x21·i + 0x80·g` | `nndrd vr(27+i),i + 4g` | raw accumulator row-bank `i + 4g` (bank 0) into vr27+i |
| `0x81 / 0xa3` | `nnmac vw4,1 / vw5,3` | MAC pass, code 1 from vw4 / code 3 from vw5 |
| `0x121 / 0x142` | `nnmac vw9,1 / vw10,2` | MAC pass, code 1 from vw9 / code 2 from vw10 |
| `0x120 / 0x140` | `nnmac vw9,0 / vw10,0` | MAC pass, code 0 from vw9 / vw10 |
| `0x20 / 0x41` | `nnmac vw1,0 / vw2,1` | MAC pass, code 0 from vw1 / code 1 from vw2 |
| `0x41 / 0x43` | `nnmac vw2,1 / vw2,3` | MAC pass, code 1 / code 3 from vw2 |
| `0x1c1 / 0x1c3` | `nnmac vw14,1 / vw14,3` | MAC pass, code 1 / code 3 from vw14 |

The port and bank constants behind these forms: WEIGHT0…WEIGHT3 = `0x00 / 0x02 / 0x04 / 0x07`, ACT_A = `0x20`,
ACT_B = `0x21`, TABLE = `0x40`; drain bank 1 (the packed output FIFO, always read with row 0) = `0x20`, drain
bank 0 (raw int32 accumulators) = row number `R` with no bank bits, so its immediate is `R` itself.
## Appendix E — Complete example listing

The listing below is the complete program described in [Chapter 8](#8-worked-example): the single-pass,
whole-input case of the 1×1 recipe ([§11.1](#111-11-4-bit-operands)) for a 64→128-channel, 12×20, 4-bit
layer, written against the companion implementation ([Chapter 16](#16-runtime-api)). All four output
groups are resident at once and the input fits in ORAM, so there is no pass loop and no input ring.

```c
#include "device.h"
#include "nna/nna.h"        /* operations: reset, arm, feeds, MAC, commit, drain  */
#include "nna/nna_regs.h"   /* field, port, command and program names             */
#include "nna/nndma.h"      /* descriptors, kicks, waits                          */

#define CIN            64
#define COUT           128
#define H              12                       /* output rows; stride 1, so also input rows       */
#define W              20                       /* output columns                                  */
#define IN_BITS        4
#define D              (CIN / 32)               /* input channel groups                    -> 2    */
#define DOUT           (COUT / 32)              /* output channel groups                   -> 4    */
#define GROUPS_A       (D / 2)                  /* input groups fed to unit A              -> 1    */
#define GROUPS_B       (D - GROUPS_A)           /* ... and to unit B                       -> 1    */
#define ROW_BYTES      (W * 32 * IN_BITS / 8)   /* one input plane row                     -> 320  */
#define OUT_RB         YX_ROUND_UP_64(W * 32 * 4 / 8)   /* DDR output row                  -> 320  */
#define STG_RB         OUT_RB                   /* ORAM staging row (W is a multiple of 4)         */
#define COLUMN_BLOCKS  (W / 4)                  /* four pixels per tile                    -> 5    */
#define ROW_PAIRS      (H / 2)                  /* two rows per tile                       -> 6    */

/* ORAM plan: the whole input, the weights, the output staging, the requantization table. */
#define O_IN     0x0000u
#define O_W      (O_IN  + D * H * ROW_BYTES)
#define O_STG    (O_W   + DOUT * D * 512u)
#define O_TBL    (O_STG + 2u * DOUT * 2u * STG_RB)

/* DESRAM: table, weights, one descriptor per input plane, then one output chain per row pair. */
#define DESC_TABLE     0u
#define DESC_WEIGHTS   1u
#define DESC_INPUT     2u
#define DESC_OUTPUT    (DESC_INPUT + D)

static void configure(void)
{
    static const uint16_t program[2] =                  /* one input group per unit: step, then rewind  */
    {
        NNA_PROG_STEP(1),
        NNA_PROG_STEP(IN_BITS - 1 - IN_BITS * GROUPS_A)
    };
    nna_layer_config_t config;
    static nna_config_t record;                         /* static: see the note after this listing */
    uint32_t clamp_word[8];
    int lane;

    nna_reset();                                        /* nnrwr B.1b,0 ; nncmd 0x00 */

    /* vr29 is the per-nibble output clamp the store applies after every drain -- zero on this layer. It is
     * MXU state, not array state: the array reads only vr31. */
    for (lane = 0; lane < 8; lane++)
    {
        clamp_word[lane] = 0u;
    }
    MXU3_BROADCAST_WORD(NNA_VR29, clamp_word);

    /* Describe the layer. Every member left unset keeps its reset value and emits no write at all. */
    nna_config_init(&config);
    config.in_h              = H;                       /* the input HEIGHT ... */
    config.in_w              = W;                       /* ... and the width, separately */
    config.in_elem_class     = nna_in_elem_class(IN_BITS);
    config.mode_code         = 2;
    config.row_origin        = 0;                       /* no padding: the walk starts at (0, 0) */
    config.col_origin        = 0;
    config.mac_span          = 8;                       /* constants of the recipe, not of the layer */
    config.mac_pitch         = 4;
    config.edge_mode         = 0x12;
    config.operand_precision = nna_operand_precision(4, IN_BITS);
    config.groups_unit_a     = GROUPS_A;
    config.groups_unit_b     = GROUPS_B;
    config.tap_mask          = nna_tap_mask(1);         /* a 1x1 kernel has one tap */
    config.pack_format       = 0;
    config.out_elem_class    = nna_out_elem_class(4);
    config.mode_flags        = 2;
    config.tile_class        = nna_tile_class(IN_BITS);
    config.mac_balance       = 0;
    config.parity_tag        = D & 1;
    nna_config_apply(&config);

    /* Both units carry one input group, so both slots get the same two-entry walk. Two or more groups per
     * unit need the five-entry form of §11.1.4. */
    nna_program_load(program, 2, program, 2);

    nna_arm();

    /*
     * The word file the tile loop lives on. It is installed AFTER the configuration, which owns all
     * sixteen words while it runs and leaves them zero: w1 is the zero every per-tile field write reads,
     * w4 and w5 are the operands the two MACs name, and w2 and w3 feed the two unit start offsets, which
     * this recipe writes after the arm.
     */
    nna_config_clear(&record);
    record.word[NNA_W1] = 0u;
    record.word[NNA_W2] = 0u;
    record.word[NNA_W3] = 2 + IN_BITS * GROUPS_A;
    record.word[NNA_W4] = 0u;
    record.word[NNA_W5] = 4 * IN_BITS * GROUPS_A + 8;
    nna_config_stage(&record);

    NNA_WRITE_FIELD(NNA_UNIT_A_START, NNA_W2);
    NNA_WRITE_FIELD(NNA_UNIT_B_START, NNA_W3);
}

/** One tile's activation windows: one 64-byte pair per input plane, in plane order. */
static void feed_unit_a(const uint8_t* row0, const uint8_t* row1)
{
    int plane;

    for (plane = 0; plane < GROUPS_A; plane++)
    {
        uint32_t offset = (uint32_t)plane * H * ROW_BYTES;

        nna_feed_window_pair_unit_a(row0 + offset, row1 + offset);
    }
}

static void feed_unit_b(const uint8_t* row0, const uint8_t* row1)
{
    int plane;

    for (plane = GROUPS_A; plane < D; plane++)
    {
        uint32_t offset = (uint32_t)plane * H * ROW_BYTES;

        nna_feed_window_pair_unit_b(row0 + offset, row1 + offset);
    }
}

/**
 * Drain one committed tile into the staging area. Because the packed FIFO trails the write point by one
 * tile, the caller always passes the coordinates of the tile committed one step earlier.
 */
static void store_tile(yx_dev* dev, int parity, int group, int column_block)
{
    uint8_t* dst = (uint8_t*)dev->oram + O_STG
                 + (uint32_t)parity * (DOUT * 2u * STG_RB)
                 + (uint32_t)group * (2u * STG_RB)
                 + (uint32_t)column_block * 64u;

    NNA_DRAIN_PACKED(NNA_VR10);                     /* output row 0 of the tile */
    NNA_DRAIN_PACKED(NNA_VR11);                     /* output row 1            */
    MXU3_CLAMP_NIBBLE(NNA_VR10, NNA_VR29);
    MXU3_CLAMP_NIBBLE(NNA_VR11, NNA_VR29);
    NNA_VST64(NNA_VR10, dst);
    NNA_VST64(NNA_VR11, dst + STG_RB);
}

int run_layer(yx_dev* dev, const yx_conv_bufs* buf)
{
    const uint8_t* oram = (const uint8_t*)dev->oram;
    int row_pair;
    int column_block;
    int group;
    int plane;
    int row;
    uint32_t descriptor;

    /* Chain 0: the requantization table, then the weights. */
    nndma_descriptor(dev, DESC_TABLE, YX_NMEM_PHYS + buf->tbl_off, O_TBL, DOUT * 256u, 1);
    nndma_descriptor(dev, DESC_WEIGHTS, YX_NMEM_PHYS + buf->w_off, O_W, DOUT * D * 512u, 0);
    nndma_kick_read0(dev, DESC_TABLE);

    /* Feature chain: one descriptor per input plane, the whole image. */
    for (plane = 0; plane < D; plane++)
    {
        nndma_descriptor(dev, DESC_INPUT + plane,
                         YX_NMEM_PHYS + buf->in_off + (uint32_t)plane * H * ROW_BYTES,
                         O_IN + (uint32_t)plane * H * ROW_BYTES,
                         H * ROW_BYTES, plane != D - 1);
    }
    nndma_kick_read1(dev, DESC_INPUT);

    /* Output chains: per row pair, one descriptor per output group per row, staging -> DDR. */
    descriptor = DESC_OUTPUT;
    for (row_pair = 0; row_pair < ROW_PAIRS; row_pair++)
    {
        for (group = 0; group < DOUT; group++)
        {
            for (row = 0; row < 2; row++, descriptor++)
            {
                int last = (group == DOUT - 1) && (row == 1);

                nndma_descriptor(dev, descriptor,
                                 YX_NMEM_PHYS + buf->out_off
                                     + (uint32_t)group * H * OUT_RB
                                     + (uint32_t)(2 * row_pair + row) * OUT_RB,
                                 O_STG + (uint32_t)(row_pair % 2) * (DOUT * 2u * STG_RB)
                                     + (uint32_t)group * (2u * STG_RB)
                                     + (uint32_t)row * STG_RB,
                                 OUT_RB, !last);
            }
        }
    }

    configure();

    nndma_wait_read0(dev);                              /* weights and table have landed */
    NNA_WRITE_FIELD(NNA_WEIGHT_STREAM_OFF, NNA_W1);     /* w1 = 0: start of the weight bank */
    for (group = 0; group < D * DOUT * 2; group++)      /* 256 B per push round, four ports */
    {
        nna_push_weight_slice(oram + O_W + (uint32_t)group * 256u);
    }
    NNA_WRITE_FIELD(NNA_TABLE_OFF,         NNA_W1);
    for (group = 0; group < DOUT * 2; group++)          /* 128 B per push round */
    {
        nna_push_requant_half(oram + O_TBL + (uint32_t)group * 128u);
    }

    nndma_wait_read1(dev);                              /* the input image has landed */

    for (row_pair = 0; row_pair < ROW_PAIRS; row_pair++)
    {
        const uint8_t* row0 = oram + O_IN + 2u * (uint32_t)row_pair * ROW_BYTES;
        const uint8_t* row1 = row0 + ROW_BYTES;
        int parity = row_pair % 2;

        for (column_block = 0; column_block < COLUMN_BLOCKS; column_block++)
        {
            uint32_t column = (uint32_t)column_block * 64u;     /* 4 px x 32 ch x 4 bit */

            /* The previous block's last commit straddles this block's unit-A feed. */
            if (column_block > 0)
            {
                nna_precommit();                        /* nncmd 0x84 */
            }
            feed_unit_a(row0 + column, row1 + column);
            if (column_block > 0)
            {
                nna_commit_packed();                    /* nncmd 0x8b */
            }

            NNA_WRITE_FIELD(NNA_MAC_BASE_A1, NNA_W1);   /* the per-tile address tags, both 0 */
            NNA_WRITE_FIELD(NNA_MAC_BASE_A0, NNA_W1);
            NNA_RUN_MAC(NNA_W4, 1);                     /* nnmac vw4,1 -- unit A */

            feed_unit_b(row0 + column, row1 + column);
            NNA_RUN_MAC(NNA_W5, 3);                     /* nnmac vw5,3 -- unit B */

            /* The commit above published the previous block's last group. */
            if (column_block > 0)
            {
                store_tile(dev, parity, DOUT - 1, column_block - 1);
            }
            NNA_WRITE_FIELD(NNA_COMMIT_WINDOW, NNA_W1);

            /* The remaining output groups reuse the activation window already in the array. */
            for (group = 1; group < DOUT; group++)
            {
                nna_precommit();
                nna_commit_packed();
                NNA_RUN_MAC(NNA_W4, 1);
                NNA_RUN_MAC(NNA_W5, 3);
                store_tile(dev, parity, group - 1, column_block);
            }
        }

        /* Tail of the row pair: commit the last tile, drain it, and write the pair back. */
        nna_precommit();
        nna_commit_packed();
        nndma_wait_write(dev);
        store_tile(dev, parity, DOUT - 1, COLUMN_BLOCKS - 1);
        nndma_kick_write(dev, DESC_OUTPUT + (uint32_t)row_pair * 2u * DOUT);
    }

    nndma_wait_write(dev);
    return 0;
}
```
