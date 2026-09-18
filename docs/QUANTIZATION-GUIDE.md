# Making models for this runtime

A guide to designing, training and quantizing models so they run well on the AMD XDNA1 (Phoenix) NPU through an
`.ignite` container. The README walks one model from Hugging Face to a running container; this page is about
making a *good* one, and about the choices that decide whether a model belongs on this hardware at all.

Everything here was measured on the test PC unless it is marked DERIVED. The engine and the compiler live in
[ignite-xdna](https://github.com/jdominick05/ignite-xdna); paths like `results/model_zoo/` refer to that checkout.

The one sentence that governs the rest:

> **This engine is dataflow-bound, not math-bound. The model that uses the hardware best is the one that moves
> the fewest activation bytes per frame, not the one with the fewest multiply-accumulates.**

Three measurements say so. With every weight operation switched off, YOLOv8n's dispatch still takes **5.37 ms of
7.39 ms** (`results/model_zoo/dispatch_floor_yolov8n_full.json`). SESR M7 sits on a **2.53 ms** non-compute floor
while moving about **67 MB of activations per frame**. And on a pure matrix multiply the int8 array reaches
**46.53 MACs/cycle/core out of 256 — 18.2 %** of its ceiling. Arithmetic is the resource you have spare. Bytes
are the resource you are spending.

And the runtime cannot spend fewer of them for you. Keeping activations or weights on the chip so they are
fetched from DDR less often was built twice, byte-exact both times, and was **3.40 ms and 3.63 ms slower** on
YOLOv8s. Taking the MemTile out of the path would save nothing either: a MemTile hop costs no measurable time
per byte. Cutting the bytes is the model's job ([§1](#what-the-runtime-cannot-take-off-the-bill)).

---

## 1. What a frame actually pays for

A container keeps one flat workspace in DDR with a region per tensor. Every layer reads its input from DDR
through the MemTile into the cores and writes its output back to DDR, where the next layer reads it again.
Nothing stays on chip between layers, so a layer's cost is almost entirely the bytes it drags across that
boundary.

The geometry is the same for every model:

| Quantity | Value |
|---|---|
| Output tile | 5 rows x 20 columns |
| Output blocks per tile | 4 blocks of 8 channels |
| Activation packet | 6,400 B, whatever the layer |
| Joined drain object | 12,800 B (four 3,200 B core outputs) |
| Cores | 4 columns x 4 rows |
| DMA repeat limit | 64 per shim task |

Three multipliers follow from that geometry, and they are what you design against:

**Fills run about 4x the input tensor.** Each core's 6,400 B packet carries its own halo rows, so neighbouring
tiles re-read the pixels they share. Seven of SESR's nine layers fetch 4,326,400 B to consume a 1,065,024 B
input; the other two fetch twice that, for the reason below and for a residual operand.

**A packet is 6,400 B whatever the kernel touches, and the kernel size decides how much of it is slack.**

| Convolution | Input channels per packet | Share of each packet plane the kernel never reads |
|---|---:|---:|
| 3x3 stride 1 | 32 | 23 % |
| 3x3 stride 2 | 8 | 42 % |
| 5x5 stride 1 | 8 | 73 % |

A 5x5 convolution at stride 1 has to split its input one 8-channel block per packet, because its 25 taps of
256 weights fill 6,400 of the 9,216 bytes a weight packet carries. On a 32-channel input it therefore fetches
four times the packets of a 3x3, and three quarters of each is never read. SESR's 5x5 tail fetches 8,652,800 B
where its 3x3 layers fetch 4,326,400. The slack cannot be trimmed at run time: a transfer's length is the packet
size, and a smaller packet multiplies packets faster than it saves bytes (DERIVED: at 4,032 B, YOLOv8n would save
0.38 ms of transport and spend 0.69 ms issuing the extra tasks).

**Drains pad to four blocks.** A layer with 16 channels is two real blocks padded to four, so **half of its
drained bytes are junk planes**. Removing them inside the kernel was tried and abandoned: the wider accumulator
spilled a 1.4 KB stack frame into a 2 KB core stack and hung the device.

Measured per-frame traffic — the number to compare a design against:

| Model | Layers | DMA tasks | Activation fills | Drains | Weights* |
|---|---:|---:|---:|---:|---:|
| SESR M7 (256x256, 16 ch) | 9 | 1,007 | 710 tasks / 47,590,400 B | 261 / 19,468,800 B | 36 / 6,403,072 B |
| YOLOv8n (640x640) | 66 | 2,972 | 2,146 / 83,072,000 B | 523 / 18,112,000 B | 303 / 25,792,256 B |
| YOLOv8s (640x640) | 66 | 7,143 | 5,857 / 237,670,400 B | 897 / 27,814,400 B | 389 / 83,296,768 B |

\* The weight column is a floor: the stream report counts a single-chunk layer's weight item as zero bytes.
Activation and drain bytes are exact.

Read the SESR row again: nine layers, no depth, no width — and the worst traffic per unit of work in the zoo. It
is also the one model class where AMD's stack still wins outright, 3.65 ms against 6.67 ms
([measurements](PERFORMANCE.md#yolov8s-and-sesr-m7-against-amds-stack)). Nine layers at 256x256x16 is the shape
to avoid.

### What the runtime cannot take off the bill

YOLOv8s is the other model AMD's stack wins, narrowly: **16.95 ms against 17.24 ms** in the same sitting
([measurements](PERFORMANCE.md#yolov8s-and-sesr-m7-against-amds-stack)). The gap is inside the NPU stage — AMD's
`session.run` took 13.14 ms against Ignition's 16.70 ms dispatch — and Ignition's host work wins most of it back.
Its 237.7 MB of fills per frame are the price, and that is now a **known limitation of this runtime**, not an
open item. Every way of moving fewer bytes without changing the model was tried in ignite-xdna, and none
survived:

| Tried in the runtime | Result |
|---|---|
| Hold activations in the MemTile, so one fetch serves several output groups | Byte-exact, **3.40 ms slower** on YOLOv8s |
| Hold each layer's weights in the MemTile instead of re-sending them every round | Byte-exact, **3.63 ms slower** on YOLOv8s and 1.16 ms slower on YOLOv8n |
| Route activations around the MemTile | Nothing to gain: a MemTile hop measured under 0.001 ms per MB against a direct path |
| Trim each packet to what the kernel reads | Not possible without shrinking every packet, which costs more in tasks than it saves (DERIVED) |
| Compress weights in the MemTile's hardware codec | Works on this part, but stalls on every data pattern tried except a ramp; parked |
| Pass halo rows between neighbouring cores | Reaches one halo row of two, and needs a kernel change; not built |

The same effort spent on the model moves further: halving the input side quarters the fills ([§3](#3-shape-rules-that-cost-nothing-to-follow)).
The measurements are in ignite-xdna's
[BENCHMARKS.md](https://github.com/jdominick05/ignite-xdna/blob/main/docs/BENCHMARKS.md#memtile-residency-does-not-pay-on-the-graph-engine-and-the-yolov8s-gap-is-a-known-limitation-2026-09-16-desktop-2).

---

## 2. Stay inside the op vocabulary

The engine has four opcodes — NOP, CONV, MAXPOOL, RESIDUAL — and the compiler accepts only what it can map onto
them.

**Compiles today**

- Convolutions: **1x1 stride 1 pad 0**, **3x3 stride 1 or 2 pad 1**, **5x5 stride 1 pad 2**.
- 2x nearest-neighbour upsample, fused into the consumer's input.
- Max pool (the 5x5 SPPF pool; pooling forces a halo of 2).
- Residual add, including the per-operand left shifts that wider models need, and a HardSwish applied after the add
  (built for a split convolution; no model uses it yet, and it stays in the program for one that does).
- Concat and slice, ReLU as an integer epilogue, and the HardSigmoid-and-multiply form the quantizer emits in
  place of SiLU. That form is not free: on YOLO-World v2 the swap alone costs 11 points of mAP in FP32 (41.5 to 30.5 %
  on the first 500 COCO images). `ignite-compile --silu-sigmoid` (ignite-xdna 0.3.1) replaces it on the NPU with a
  four-line integer sigmoid, exact against an integer reference model: on all 5,000 COCO val2017 images YOLOv8n goes
  from 27.10 to 34.12 mAP@50-95, YOLOv8s from 37.21 to 42.37 and YOLOv8n-pose from 32.71 to 44.16 OKS, for 2–4 % more
  NPU dispatch. It needs no requantization, and the compiler refuses it for a model with CPU steps
  ([measurements](PERFORMANCE.md#ignition-vs-amds-ryzen-ai-stack)).
- Convolution biases in int8, as XINT8 writes them, or in int32 at the input scale times the weight scale, which the
  engine's 32-bit accumulator takes unchanged. Needs ignite-xdna `80ca69e` or later, in no release yet: older versions
  truncate an int32 bias to int8 without an error.

**Does not compile**

- Softmax, and a general activation-by-activation multiply.
- MatMul and Gemm in mid-graph, GlobalAveragePool anywhere but a terminal classification chain, and
  therefore attention too. One exception, since ignite-xdna `bd87558`: a **terminal** chain —
  GlobalAveragePool, optionally a quantizer's `Mul`, Flatten/Reshape, then Gemm or MatMul feeding a Q/DQ
  straight to the graph output (or a bare terminal Gemm) — lowers the Gemm as one 1x1 convolution on the
  NPU, with no new core opcode. The 2,048-feature, 1,000-class head measures 1.38 ms per frame on silicon
  ([measurements](PERFORMANCE.md#classification-head-against-amds-stack)); one head per model, and it must
  be the graph's output.
- **A global average pool is matched, never computed.** The head's Gemm becomes a convolution; the average
  does not run on the device at all. Since ignite-xdna `3f940b4` the compiler carves the pooling span into a
  host region automatically, so a whole classifier compiles and is correct — but it declares **one host
  segment**, and a container with any host segment is a hybrid, not a network wholly on the NPU. Only a
  model whose input already holds pooled features reaches 0 host segments. Build expecting the host step.
- **Every stored map must be at least one 20-pixel tile, which decides your input size.** A /32 backbone at
  224 ends at 7x7 and fails with `a 14-pixel map is smaller than one 20-pixel tile` (measured on
  `yolov8n-cls` at 224); at 640 the same graph compiles and verifies 28/28 layers exact on silicon. That is
  8x the arithmetic of a 224 classifier — a classifier here is priced at 640, not at the resolution its
  accuracy was reported at.
- **Classification is where the op set bites hardest.** Of 15 XINT8 classifiers put through the compiler,
  **one** reached a schedulable graph. ResNet50, wide_resnet50_2, wide_resnet101_2, resnext50_32x4d and
  densenet121 all stop at a **3x3 stride-2 `MaxPool`** the core does not implement (it takes only the SPPF
  5x5 pad-2 form) — the stem pool every ResNet-family zoo model begins with, before pooling, grouping or
  the head is ever reached. `regnetx_002` stops at grouped convolution (group 3, not depthwise). So do not
  plan a ResNet50 classifier around the head lowering: the head is the part that works. Sweep and refusals
  verbatim: ignite-xdna `results/aie/model_zoo_classifier_compile_20260918T142628Z.log`.
- Dilation other than 1, and 1x1 convolutions at stride 2.
- Bilinear upsampling. Nearest is free; bilinear leaves the engine.

**Depthwise and grouped convolutions compile, but avoid them anyway.** They lower as dense diagonal weights: the
DMA traffic is identical to a full convolution of the same shape, and the extra multiply-accumulates land on
zeros. Measured on YOLO11n's six depthwise head convolutions, the expansion changed nothing — 3,360 DMA tasks and
98,176,000 B of fills before and after. A MobileNet-style backbone therefore costs what a dense one costs while
computing far less with it.

**The escape hatch, and its price.** A region of the graph can be declared a host segment and run on ONNX
Runtime's CPU provider between two NPU dispatches; SESR's `DepthToSpace` tail works the same way. It is how
YOLO11n keeps its attention core — two matrix multiplies and a softmax — at **0.71-0.73 ms per frame** in the default
power mode (0.55-0.57 ms in `performance`, whose CPU threads never sleep) and one
ONNX Runtime call. Keep the region as small as the unsupported ops allow: with YOLO11n's whole C2PSA block on the CPU,
convolutions included, the same step took 1.95-1.98 ms. The CPU step follows the power mode only with ignite-xdna
`fbd53f5` or later, which is in no release yet. Use it for a tail or a single exotic block, never for something in the
middle of a hot loop.

Several regions cost more, and are sometimes still the only route. YOLO-World v2 runs its four text attention blocks as
host segments in the middle of the neck, each with its projection convolution: 9.650 ms of a 48.227 ms profiled frame
(ignite-xdna `results/aie/yolow_gptq/`).
- **Region rules:** regions may share constants (its text guide), and a region's input may be a Concat view over
  several tensors.
- **Constants at run time:** a region's constants can be replaced at run time, which is how one YOLO-World container
  takes any class names (`EngineSession.set_host_constants`).
- **Version:** these rules need ignite-xdna `206cfec` or later, in no release yet.

---

## 3. Shape rules that cost nothing to follow

**Channels in multiples of 32.** Padding 16 channels to 32 is free: those junk planes are already being written
and drained, so the capacity comes out of bytes you are paying for anyway.

**Spatial sizes that divide 20 and 5.** A map the tile does not divide uses overlapping edge tiles that recompute
and rewrite the shared pixels.

**Never go below the tile.** Maps narrower than 20 px waste lanes outright. This is exactly why ResNet50 has no
container: 28 of its 53 convolutions run on maps smaller than the tile.

**Downsample early, then go deep and wide.** Early full-resolution layers dominate everything: YOLOv8n's first
convolution alone is 6,553,600 B of fills and 3,276,800 B of drains per frame. Halving a map's side quarters its
traffic; doubling its channels costs weights and arithmetic you have to spare.

**Keep one consumer kind per tensor.** A tensor feeding both a convolution and a pool needs two different halo
fill values, 128 and 0, and the compiler refuses it.

---

## 4. Quantizing, in the order that works

Export, cut, calibrate, compile. ignite-xdna's `pipelines/yolov8n/` is the reference implementation, and the same
four steps serve every model in the zoo.

**1. Export at opset 17 and cut the head.** `1_export.py` writes the ONNX graph and `1b_cut_head.py` removes the
decode tail, so the trunk ends at the raw head tensors and decode runs in native C on the host at 0.044 ms,
against 0.28 ms for the numpy version. Opset 17 is not a preference: it is what the quantizer and the vendor
runtime both accept.

**2. Quantize plain XINT8 first, always.** `3b_quantize_cut.py` produces the format the engine expects:
power-of-two scales, weights symmetric and clipped to [-127, 127], activations uint8 with zero point 128.
Measure that before reaching for anything cleverer.

**3. Check for shift-cut collapse before believing any accuracy number.** RegNetX-002 lowered cleanly — 324 of
326 nodes on the NPU at 2.45 ms — and scored **0.50 % top-1**, because a clamp (131 to -108) distorted the scale
grid by 3.25 x 10^32. The design-time defence is keeping per-channel weight and batch-norm ranges comparable, and
running cross-layer equalization before quantizing.

**...and for a convolution whose output is a small difference of large terms.** Any rounding error is large against
such an output. In YOLO-World v2's four C2fAttn output convolutions, the part reading the ordinary branches and the
part reading the attention have RMS 22.4 and 22.3 while their sum has 1.34.
- **The collapse:** plain XINT8 scores 2.1 % mAP (first 500 images). The CPU collapses too, so it is the quantized
  model, not the runtime.
- **What does not help:** splitting each convolution so each part gets its own weight scale buys nothing, and neither
  does keeping the halves' outputs in float.
- **What does:** nearest rounding and XINT8's int8 bias each swamp the output. GPTQ rounding at the same power-of-two
  scale, with an int32 bias, recovers 24.7 % on the first 300 images, against 24.5 % with those convolutions in FP32
  (ignite-xdna `pipelines/yolow/3c_gptq_cv2.py`, `results/aie/yolow_gptq/`). Measure each suspect layer's output
  against FP32 before blaming the attention or the precision.

**4. Reach for AdaRound only where it has paid.** It recovers rounding error, not a collapsed distribution:

| Model | Plain XINT8 | AdaRound | Verdict |
|---|---:|---:|---|
| ResNet50 (top-1) | 71.70 % | **79.80 %** | Worth it; nearly all of the 8.4-point loss returned |
| YOLOv6n (mAP@50-95) | 22.92 | **33.57** | Worth it |
| YOLOv8n-pose (OKS mAP@50-95) | 32.64 | **34.32** | Worth it |
| YOLOv8 detection | — | ~unchanged | Not worth the hours |
| MobileViT-XXS (top-1) | 0.00 % | 0.00 % | Cannot rescue a collapse |

It also hits a RAM wall at 640x640 — resolution-specific, not width-specific — and needs a GPU there.

**5. Do not over-calibrate.** Past roughly 32-64 images the operating point stops moving at these widths, tested
twice and null both times: YOLOv8m read -0.11 mAP going from 64 images to 200, YOLOv8l +0.22 going from 32 to
200. The cost is real — 128 images takes about 5 minutes and 13.5 GB of temporary disk, and YOLOv8l at 200 images
took 2,953 s and 106 GB of spool.

**6. But do care what is in the calibration set.** The count is a null; the content is not. A YOLOv8n container
calibrated on coco128 finds 7 objects on `bus.jpg` where the benchmarked container finds 5. Calibrate on images
that look like the deployment, and evaluate on the full set before quoting accuracy.

**7. Treat input resolution as an accuracy knob, not only a latency knob.** ResNet50's plain-XINT8 accuracy peaked
at 256x256, not at the 224x224 everyone assumes.

---

## 5. If you are training from scratch

Design the quantization problem away instead of recovering from it: quantization-aware training with
**power-of-two clamping enforced in the forward pass**, **ReLU or HardSwish activations only** so the compiler
never has to rewrite one, **channel counts padded to multiples of 16, 32 or 64** to match the vector lanes, and
**RepVGG-style reparameterization** so inference collapses to plain convolutions. The bar worth aiming at is
100 % of nodes on the NPU at roughly no loss under *plain* XINT8, with AdaRound left in the drawer.

Two failure modes are worth designing out from the start, because both were measured here and one is invisible
until you reach silicon:

- **Multi-branch gating that multiplies branches at different scales.** BiSeNetV2 read 59.47 % pixel accuracy
  under CPU simulation and **15.33 % on the physical device**. Fixed-point elementwise multiplication across
  disparate branch scales attenuates minority classes, and simulation will not warn you.
- **Softmax in the trunk.** MobileViT-XXS survives INT8 at **0.00 % top-1**, and the engine has no softmax op in
  any case.

---

## 6. Precision: what this silicon offers

| Precision | Status on Phoenix (AIE2) | When to use it |
|---|---|---|
| **INT8 (XINT8)** | The default path, 256 MACs/cycle | Everything, unless it collapses |
| **INT16 activations** | Native, but 64 MACs/cycle; measured at **0.93x int8** end to end (0.68-1.04x) | Models that collapse at 8 bits: +48.2 dB SQNR. The vendor runtime path is deadlocked — opset 17 forces quantization nodes it rejects, opset 21 its parser rejects |
| **BF16** | Real kernels; GEMM is the project's best kernel result (1.33x over CPU at large shapes), standalone activation kernels lose end to end | Fused into a larger kernel, never as its own dispatch |
| **W4A8 / INT4** | **Not available.** AIE2 has no int4 multiply-accumulate at all; that is Strix-class silicon | Not an option on this part |

A 7 % throughput cost for a model class that otherwise scores zero is a good trade. The same 7 % on a model that
already works is not.

---

## 7. Where the balance sits

Spend accuracy budget on **width and depth at low spatial resolution**. Spend latency budget on **resolution only
where the task genuinely needs it**.

| Trade | What it costs on this hardware |
|---|---|
| Halve the spatial resolution | About 4x less activation traffic — the dominant cost |
| Double the channels at the same resolution | Weight traffic (about 0.26 ms on SESR, DERIVED) and arithmetic you have spare |
| Pad 16 channels to 32 | **Free** — the junk planes are drained either way |
| Add a layer at low resolution | Cheap: one more pass over a small map |
| Add a layer at full resolution | Expensive: fills run about 4x the tensor |
| Use a 5x5 convolution at stride 1 | One packet per 8 input channels instead of 32: up to 4x a 3x3's fills, 73 % of each plane never read |
| Move a block to the host | About 0.5 ms per frame and one ONNX Runtime call for YOLO11n's attention core, 9.650 ms for YOLO-World v2's four attention blocks, and the container stops being pure NPU |

---

## 8. Validate in this order

1. **Layer exactness.** ignite-xdna's `tools/verify_engine_container.py` compares every layer and head against
   ONNX Runtime's CPU provider on the same quantized model; YOLOv8n reads 66 of 66 exact. A model that is not
   layer-exact is a compiler bug, not a quantization result, and the two must never be diagnosed together.
2. **Accuracy on the full dataset**, never on one image. The pipelines' `5_eval_map.py` and its pose equivalent
   run all 5,000 COCO images; single-image impressions have been wrong here before.
3. **Latency in one quiet sitting**, with the NPU idle before each run (`xrt-smi examine -r aie-partitions`) and
   runs interleaved against whatever you are comparing with. Latency drifts between sessions on a shared machine
   with no code change, so only within-sitting comparisons mean anything. And measure a byte saving rather than
   pricing it: two runtime changes whose saved bytes predicted gains of 0.75 and 0.73 ms on YOLOv8s measured
   3.40 and 3.63 ms slower.

---

## 9. Checklist before you train

- [ ] Every op in the trunk is on the compiles-today list; exotic blocks are isolated at the tail.
- [ ] Channel counts are multiples of 32.
- [ ] No feature map is narrower than 20 px, and spatial sizes divide 20 and 5 where they can.
- [ ] Resolution drops early; depth and width come after the first downsample.
- [ ] No depthwise or grouped convolutions in the hot path.
- [ ] 5x5 convolutions at stride 1 only where a 3x3 cannot do the job, and never on a wide full-resolution input.
- [ ] No softmax, no cross-branch multiply at mismatched scales, no bilinear upsample.
- [ ] Activations are ReLU or HardSwish, and clamping is power-of-two in the forward pass.
- [ ] Calibration data looks like the deployment, at 32-64 images.
- [ ] An exactness gate is wired up before any accuracy claim.

Related reading: [what each model costs here](PERFORMANCE.md), and, in ignite-xdna,
[MODEL_ZOO_BENCHMARKS.md](https://github.com/jdominick05/ignite-xdna/blob/main/docs/MODEL_ZOO_BENCHMARKS.md) for
the per-model traffic,
[DECISIONS.md](https://github.com/jdominick05/ignite-xdna/blob/main/docs/DECISIONS.md) for why the engine is
shaped this way, and
[BENCHMARKS.md](https://github.com/jdominick05/ignite-xdna/blob/main/docs/BENCHMARKS.md) for the full accuracy
record.
