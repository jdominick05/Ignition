# Ignition TODO

State of `main` on 2026-09-17 (version 0.3.3). Measured figures name the commit whose message records the run;
Ignition does not track benchmark logs (`results/` is gitignored). How the pipeline works is in the
[performance notes](docs/PERFORMANCE.md#how-a-frame-runs).

## Completed

- [x] **Native NPU path** (`ee66fbd`): `YOLOPipeline` serves an `.ignite` container through ignite-xdna's
  `YoloPipeline` on NPU Device 0, with boxes from the NPU's detect heads and no ONNX Runtime `run()` calls.
- [x] **Camera and shutdown** (`ee66fbd`): timed opens across DirectShow, Media Foundation and OpenCV's default
  backend; q, ESC, closing the window, Ctrl+C and Ctrl+Break release the camera, window and NPU and exit 0.
- [x] **Live detection on silicon** (`ee66fbd`, `8ffa482`): 7.82–8.03 ms mean G2G at 5.0–5.9 objects per
  frame, resident memory −1.15 to +0.02 MB over 500 frames, and `bus.jpg` boxes matching ONNX Runtime (mIoU
  0.977). Every run is in the [performance notes](docs/PERFORMANCE.md#yolov8n-through-live_ignitionpy).
- [x] **CLI on `.ignite`** (`ee66fbd`): `ignition detect`, including `--stream --benchmark`.
- [x] **Consumer README** (`d42d33e`, `e3a6fcf`): same-sitting comparison with AMD's stack and a compatibility
  matrix.
- [x] **pip install** (`d6da435`, ignite-xdna `e73d4d8`): Ignition's wheel and sdist install on their own; the
  `npu` extra with ignite-xdna's wheel adds the NPU path.
- [x] **v0.3.1 release** (tag at `7d65d06`): both wheels, the sdist and SHA-256 sums on GitHub and GitLab, with
  [notes](docs/releases/v0.3.1.md); every file was downloaded back from both hosts and matched the sums.
- [x] **Honest backends:** an `.onnx` model runs on ONNX Runtime's CPU execution provider with no NPU call, and
  says so; `ignition devices` prints only what `pyxrt` reports (name, BDF, XRT and NPU driver versions).
- [x] **Camera rate:** the webcam's auto exposure sets its rate, not the backend or pixel format: 30 distinct fps
  in a bright room, 15 in a dim one. `live_ignition.py` reports the read and distinct rate and takes
  `--camera-backend`; `--exposure-priority off` holds 30 fps in dim light, with a darker image and fewer detections.
- [x] **Install from a release:** the [usage notes](docs/USAGE.md#other-ways-to-install) give the release page's pip commands beside the
  editable checkouts, and both commands work from wheels downloaded from the v0.3.1 release into fresh
  environments.
- [x] **Python coverage:** the CPU install from source passes on Python 3.10, 3.11, 3.12 and 3.13. The NPU path
  needs the Python `pyxrt` links (3.13 with XRT 2.21.0), and the README says so.
- [x] **Post-processing in native code** (ignite-xdna `e2867b6`): DFL decode and per-class NMS for the NPU
  heads run in native C (`pipelines/decode_native.c`), with detections identical to the numpy and OpenCV path
  over 6,000 synthetic trials, 312 recorded and 340 live frames. Live decode and NMS medians fell from
  0.28–0.29 ms to 0.044–0.045 ms with about 5 objects in view. The README's AMD comparison and re-check rows were
  re-run on it.
- [x] **Model zoo engine on ignite-xdna `main`** (ignite-xdna `6bd2718`): the YOLOv8s and SESR M7 engine work is
  merged with the native decode. Containers rebuilt from the merge were exact on every layer on Device 0, the
  YOLOv8n container compiled before the model zoo still runs on it within 0.07 ms of the rebuilt one, YOLOv8s
  decodes natively (0.041 ms against 0.316 ms), and Ignition's `model-zoo` branch ran YOLOv8n, YOLOv8s and
  SESR M7 at 7.68, 17.15 and 6.64 ms mean glass-to-glass (ignite-xdna `results/aie/model_zoo_main_phoenix_20260914T2209Z.log`).
- [x] **Model zoo in `live_ignition.py`:** `--task` serves detection, classification and super-resolution models,
  reading the task from the container manifest or the ONNX outputs by default, and `--json` writes each run's
  summary. With ignite-xdna `68c2fea` on `bus.jpg`, YOLOv8s ran at 17.22 ms and SESR M7 at 6.59 ms mean
  glass-to-glass on the NPU, and ResNet50 classified on the CPU at 30.49 ms ([performance notes](docs/PERFORMANCE.md#other-models)).
- [x] **Suite runner in Ignition:** `ignition suite MODEL...` runs each model in its own `ignition.live` process
  and writes one JSON record and log per model plus an index, refusing an existing output directory. The app
  moved into the package as `ignition.live`; `live_ignition.py` launches it from a checkout
  ([usage notes](docs/USAGE.md#6-compare-models-in-one-command)).
- [x] **ignite-xdna's suites on `ignition suite`** (ignite-xdna `373d897`): `tools/model_zoo_bench.py` hands its
  presets to `ignition suite`, writes each run's records into a new `results/model_zoo/<suite>_<UTC time>/`
  directory, and rewrites its JSON and `docs/MODEL_ZOO_BENCHMARKS.md`'s tables only when every run is clean. The
  tables there were not re-run.
- [x] **YOLO11n with its attention block** (ignite-xdna `0be9132`): the C2PSA-ablated model detects nothing, so the
  graph engine runs stock YOLO11n instead. Its convolutions run in two NPU dispatches and its attention block on
  ONNX Runtime's CPU provider between them, with all 84 layers exact on Device 0. In one sitting on `bus.jpg`,
  `live_ignition.py` took 10.996 and 11.048 ms mean glass-to-glass (NPU dispatch 8.45 ms, attention block
  1.69 ms, reported as the `host` stage). AMD's stack took 34.492 and 34.366 ms, and ONNX Runtime's CPU provider
  31.328 ms (ignite-xdna `results/aie/yolo11n_hybrid_phoenix_20260915T0216Z.log`, [performance notes](docs/PERFORMANCE.md#yolo11n-with-its-attention-block)).
- [x] **YOLO11n's attention-block convolutions on the NPU** (ignite-xdna `d223e7b`): the CPU step keeps only the attention
  core (two matrix multiplies and a softmax); the block's seven convolutions and residual adds run on the NPU, all 91
  layers exact on Device 0. In one sitting, 500 frames each, the attention-core container took 10.411 and 10.115 ms
  mean glass-to-glass (CPU step 0.533 and 0.505 ms) against 11.109 and 10.987 ms with the whole block on the CPU and
  36.361 and 34.549 ms on AMD's stack (ignite-xdna `results/aie/yolo11n_attention_core_phoenix_20260915T1522Z.log`).
- [x] **YOLOv8n-pose on the NPU** (ignite-xdna `c762c0e`): pose containers (nine heads: box, person score and 17
  keypoints) run all 75 layers on the NPU, exact on Device 0, and `live_ignition.py` draws each person's skeleton.
  On all 5,000 COCO val2017 images the container scored 32.77 OKS mAP@50-95; on the numpy letterbox its detections
  equal ONNX Runtime CPU's byte for byte (32.71), against 32.64 recorded for AMD's stack. In one sitting, 500 frames
  of `bus.jpg` each, it took 8.318 and 8.339 ms from frame to people against 12.056 and 12.089 ms on AMD's stack,
  and 8.360 and 8.481 ms through `live_ignition.py` (ignite-xdna `results/aie/yolov8n_pose_phoenix_20260915T1919Z.log`).
- [x] **One-step installer** (recorded in the commit that adds this entry): `install.ps1`, pasted into PowerShell,
  installs or updates the NPU path. It checks the NPU driver (and installs AMD's production driver with
  `-InstallDriver`), installs Python 3.13 and Git through winget when missing, installs the XRT SDK 2.21.75, clones
  both projects, and builds one environment with mlir-aie 1.4.2 and llvm-aie. Installed from local branches into a
  scratch folder, its printed instructions compiled YOLOv8n in 13.2 s and ran `bus.jpg` at 7.886 ms mean
  glass-to-glass, 5 detections per frame; a second run updated the checkouts in place.
- [x] **README walkthrough and model tools** (recorded in the commit that adds this entry): the README takes a user
  from `install.ps1` to a running container in 224 words of prose, held to 250 by `.github/checks/readme_words.py`;
  measurements moved to `docs/PERFORMANCE.md` and reference to `docs/USAGE.md`. `install.ps1` also builds a Python
  3.12 environment with AMD Quark 0.11.2, Ultralytics 8.4.153 and the Hugging Face CLI. Typed into one PowerShell
  window on a scratch install, the README's commands downloaded YOLOv8n from Hugging Face, exported and cut it,
  calibrated it on coco128 (304 s, 13.5 GB of temporary disk), compiled it in 13.0 s and ran `bus.jpg` at 7.730 ms
  mean glass-to-glass; the container matched its model on 66/66 layers.
- [x] **Badges against AMD's stack** (recorded in the commit that adds this entry): every number badge compares
  Ignition with AMD's stack and reads `docs/PERFORMANCE.md`, including the models Ignition loses, which mark where it
  needs work. In one sitting AMD's stack took 16.954 and 16.958 ms on YOLOv8s against Ignition's 17.240 and 17.265 ms,
  and 3.654 and 3.632 ms on SESR M7 against 6.671 and 6.662 ms (ignite-xdna
  `results/aie/yolov8s_sesr_vs_amd_phoenix_20260915T2146Z.log`).
- [x] **Energy per frame against AMD's stack, and power modes** (recorded in the commit that adds this entry): the
  first energy comparison found every NPU run spinning all 16 hardware threads between frames, because the setting
  meant to stop it never reached MSVC's OpenMP runtime. ignite-xdna fixed that and added host-sized power modes, which
  `live_ignition.py --power-mode` and `ignition suite --power-mode` select; `--max-fps` paces the loop like a camera. On
  YOLOv8n at 30 frames per second the default `balanced` mode spent 196.9 and 178.7 mJ per frame and `efficiency`
  170.3 and 159.1 mJ, against 217.6 and 200.4 mJ on AMD's stack and 1,384.8 and 1,371.8 mJ for the old spinning
  behaviour, now `performance` (ignite-xdna `results/aie/energy_power_modes_paced30_yolov8n_phoenix_20260916T1702Z.log`,
  [performance notes](docs/PERFORMANCE.md#energy-per-frame-against-amds-stack)).
- [x] **v0.3.2 release** (tag at `7ce27fd`): the Ignition 0.3.2 wheel and sdist, the ignite-xdna 0.3.0 wheel and
  SHA-256 sums on GitHub and GitLab, with [notes](docs/releases/v0.3.2.md). Before tagging, the wheel, the sdist and the
  two-wheel `[npu]` install each passed in a fresh environment, the last loading ignite-xdna's native ingress kernel;
  after publishing, every file was downloaded back from both hosts and matched the sums.
- [x] **Every AMD comparison in the default power mode, energy on every model, and the NPU's own power mode**
  (recorded in the commit that adds this entry). The latency tables and badges were re-measured with `balanced`
  instead of spinning threads. YOLOv8n took 8.42 / 8.39 ms against AMD's 10.39 / 10.34, and YOLOv8s 18.05 / 17.99
  against 16.74 / 16.56 (ignite-xdna `results/aie/latency_balanced_default_phoenix_20260916T1745Z.log`). Energy per
  frame was measured on YOLOv8s, SESR M7, YOLO11n and YOLOv8n-pose. That found YOLO11n's CPU step spinning its own
  ONNX Runtime threads in every mode; ignite-xdna `fbd53f5` puts it under the power mode, and YOLO11n then spent
  166.3 / 164.9 mJ per frame against AMD's 1,387.2 / 1,405.8 (ignite-xdna
  `results/aie/energy_power_modes_models_phoenix_20260916T2012Z.log`). Switching the NPU to `powersaver` saved 14 %
  per frame at 30 fps for twice the latency, and nothing flat out (ignite-xdna
  `results/aie/energy_npu_pmode_yolov8n_phoenix_20260916T1931Z.log`).
- [x] **The sigmoid SiLU in the documented containers, and every AMD comparison re-measured on it** (recorded in the
  commit that adds this entry). The README and usage notes compile YOLOv8n, YOLOv8s and YOLOv8n-pose with
  `--silu-sigmoid`, and the `npu` extra requires ignite-xdna 0.3.1, which has it, the in-place head readback and SESR
  M7's native resize and image output. In one sitting on `bus.jpg`, with a control compiled without the flag beside each
  container: YOLOv8n 8.50 / 8.53 ms against AMD's 10.66 / 10.42 (control 8.26 / 8.29), YOLOv8n-pose 9.28 / 9.35 against
  12.07 / 11.97 (control 8.99 / 9.00), YOLOv8s 18.21 / 18.18 against 16.79 / 16.75 (control 17.87 / 17.80), SESR M7
  4.78 / 4.78 against 4.35 / 4.35, YOLO11n 10.46 / 10.50 against 36.77 / 38.30 (ignite-xdna `results/aie/release_033/latency_release033_phoenix_20260917T1538Z.log`). On all
  5,000 COCO val2017 images the three flagged containers score 34.12 / 42.37 / 44.16 against AMD's 26.68 / 37.31 /
  32.64. An energy sitting on the new containers was discarded (below, §6).

## Active

### 1. Model zoo in Ignition

- [x] **YOLO26n (v0.3.4).** The newest YOLO family, and the one AMD's stack handles worst: its Vitis AI EP
  places 12 of 1,526 nodes where the engine runs 107 of 107 layers. 11.03 / 11.01 ms against AMD's
  37.44 / 37.67 (**3.41x**) and 32.55 mAP@50-95 against 23.64, in one sitting
  ([measurements](docs/PERFORMANCE.md#yolo26n-with-its-dfl-free-head)). Ignition needed no code change:
  YOLO26 dropped DFL and the container declares reg_max, which the decode follows. Needs ignite-xdna
  0.3.2, since reg_max became a container property after 0.3.1 was cut. Its own float export scores
  39.66, so the win is against AMD on identical quantized weights, not against float.

ignite-xdna compiles YOLOv8s and SESR M7 to `.ignite` and runs them on Device 0 (merge `6bd2718`), and Ignition's
`live_ignition.py` serves them (`--task`, `--json`). The table below is the pre-merge record; current figures are in the
[performance notes](docs/PERFORMANCE.md#yolov8s-and-sesr-m7-against-amds-stack).

ignite-xdna's `tools/model_zoo_bench.py` drives Ignition for the suites. Measured 2026-09-14 on Ignition's
`model-zoo` branch before the merge, logs in ignite-xdna `results/model_zoo/`, G2G means:

| Model | ONNX Runtime CPU, 300 frames | NPU through Ignition |
|---|---:|---:|
| YOLOv8s | 83.66 ms | 17.73 ms |
| yolo11n_no_c2psa | 35.79 ms | — |
| SESR M7 | 15.58 ms | 6.57 ms |
| ResNet50 | 34.24 ms | — |

On the merge, through the same Ignition branch: YOLOv8s 17.15 ms and SESR M7 6.64 ms, in a separate sitting
(ignite-xdna `results/aie/model_zoo_main_phoenix_20260914T2209Z.log`).

- [x] The `npu` extra requires ignite-xdna 0.3.1 (ignite-xdna `f98cab2`), the first release with `--silu-sigmoid`, the
  NPU power governor and power-mode ONNX Runtime threads for CPU steps. 0.3.0 (`554ab57`) was the first with
  `pipelines/sr_pipeline.py`, `pipelines/pose_pipeline.py` and host segments, which Ignition's super-resolution, pose
  and YOLO11n paths import.
- [ ] YOLO11n's attention core, two matrix multiplies and a softmax, runs on the CPU (0.74–0.75 ms per frame in the
  default power mode, 0.53 ms in `performance`, with ignite-xdna 0.3.1);
  the engine has no softmax or activation-by-activation multiply, so running it on the NPU is ignite-xdna work.
  The block cannot be dropped: the C2PSA-ablated `yolo11n_no_c2psa` finds nothing on `bus.jpg`
  (highest class score 0.02 in FP32 and 0.06 in XINT8), and ignite-xdna measured mAP@50-95 0.19 for it on AMD's
  Vitis AI EP against 38.72 for stock FP32 on the CPU (ignite-xdna `docs/BENCHMARKS.md`, its YOLOv11n section).
- [x] Measure YOLO11n's COCO accuracy through the container. Compiled with `--silu-sigmoid` the attention-core
  container scores **34.63 mAP@50-95** on all 5,000 COCO val2017 images, against 25.80 without the flag and 25.82
  for AMD's stack on the same harness (float 38.72). See [YOLO11n with its attention block](docs/PERFORMANCE.md#yolo11n-with-its-attention-block).
- [ ] YOLOv8n-pose decodes its keypoints in numpy: decode and NMS take 0.36 ms per frame against 0.13–0.14 ms for
  YOLOv8n's native decode. A container from the AdaRound pose model (34.32 OKS mAP@50-95 on AMD's stack) is not
  built or checked.
- [ ] ResNet50 classification runs on the CPU in this app. The engine half moved twice since this item was
  written: ignite-xdna `bd87558` lowered the terminal head (1.378–1.386 ms on silicon where AMD's stack crashes
  at placement, [measurements](docs/PERFORMANCE.md#classification-head-against-amds-stack)), and `3f940b4` put a
  **whole classifier** on the device — `yolov8n-cls` at 640, 28/28 layers exact, its pooling carried by one host
  segment, logits equal to ONNX Runtime on the same frame. `pipelines/vision.py` still raises
  `NotImplementedError` for a `.ignite` classify container. Open: wire `--task classify` to ignite-xdna's
  `ClassificationPipeline`. Deliberately not urgent: of fifteen XINT8 classifiers put through the compiler,
  **one** reaches a schedulable graph — the ResNet family stops at a 3x3 stride-2 stem `MaxPool` the core does not
  implement — so wiring buys one model at 640 rather than a zoo, and the unlock is engine-side (stem pooling and
  grouped convolution), not here. The accuracy question is also still open: the 2026-09-17 sitting fed a constant
  zero-point vector and the 2026-09-18 classifier has no measured ImageNet top-1.
- [ ] SESR M7 dispatch is 4.18–4.19 ms against a 1.5 ms target, with a 2.53 ms non-compute floor (§4). Its host
  stages are native since ignite-xdna 0.3.1 (0.57 ms against 2.84 ms for AMD's arm), so the 0.42–0.43 ms it trails AMD's
  stack by is all NPU stage. Its native resize is within one code of OpenCV's but changes the output image (41.42–42.19 dB
  PSNR against the OpenCV path); no SESR quality has been measured through it.

### 2. Release and distribution

- [ ] ignite-xdna is on no package index, so the `npu` extra resolves only with its wheel beside Ignition's.
  Decide whether to publish both to PyPI.
- [ ] No `.ignite` container ships, so the NPU path still needs the mlir-aie toolchain. Decide whether a
  prebuilt `yolov8n_full.ignite` can be attached to a release. The licence check found:
  - **A container carries the model's trained weights.** Its manifest lists `engine.xclbin` (188,542 B, the NPU
    program built from ignite-xdna's `kernels/aie2/conv_engine/engine.cc`), `insts.bin` (430,180 B) and
    `wpackets.bin` (8,164,864 B, YOLOv8n's quantized weights) (ignite-xdna
    `results/model_zoo/manifest_yolov8n_full.json`).
  - **No file records the weights' licence.** ignite-xdna's `pipelines/yolov8n/1_export.py` loads `yolov8n.pt`
    through `ultralytics`, which downloads it. The `ultralytics` 8.4.142 package installed in the export
    environment here declares AGPL-3.0 and offers an Enterprise licence covering "Ultralytics software and AI
    models" in products that bypass the AGPL's requirements. Ignition and ignite-xdna are AGPL-3.0, but neither
    states the weights' terms, and ignite-xdna's README says upstream model artifacts "are not redistributed here".
  - **ignite-xdna's file headers disagree with its licence.** Its LICENSE and README say AGPL-3.0, while 44 files
    under `src/`, `tools/`, `npu/` and `kernels/` carry Apache-2.0 SPDX headers, 2 carry MIT, and `engine.cc`
    carries none.
  - **Portability is unrecorded.** The manifest pins kernel and xclbin hashes but no NPU driver or XRT version,
    and the only verified setup is the one in the usage notes' [Compatibility](docs/USAGE.md#compatibility) table.
  - **Split container resolution:** Splitting into an engine execution container (`.ignite-exec`, containing `engine.xclbin` and `insts.bin`) and a separate weights artifact (`wpackets.bin` or `.weights`) decouples bytecode from model weights. Precompiled engine bytecode can ship directly under Apache-2.0 in the release wheel, while weights are downloaded or supplied by the user under upstream terms without licence entanglement or toolchain dependencies.
- [ ] The published v0.2.0 notes still quote the figures v0.3.1 corrects. Decide whether to edit them.

### 3. Hardware coverage

Only Phoenix on Windows 11 is verified ([usage notes](docs/USAGE.md#compatibility)).

- [ ] Run the `.ignite` path on a Hawk Point NPU (same XDNA1 generation) and record it before calling it
  supported.
- [ ] **Blocked on hardware:** measure the power modes on a host with a different core count before calling their
  sizing verified. Only the 8-core, 16-thread Ryzen 7 8700G has run them. A Hawk Point laptop such as the Ryzen 5
  8645HS (6 cores, 12 threads) would cover this and the item above in one sitting.

### 4. NPU dispatch time (ignite-xdna)

AMD's NPU stage is about 1.1 ms faster than Ignition's in the default power mode, 0.9 ms in `performance`, about
0.26 ms of it the `--silu-sigmoid` option, on the same model and image in the [performance notes](docs/PERFORMANCE.md#ignition-vs-amds-ryzen-ai-stack)' same-sitting
comparison (0.7 ms with spinning threads in `ffdefad`). Ignition's NPU figure includes the head readback, which the
default mode slows by 0.14 ms. Most of Ignition's dispatch is activations moving between host memory and the NPU:
5.37 ms of a 7.39 ms dispatch with every weight operation switched off (ignite-xdna
`results/model_zoo/dispatch_floor_yolov8n_full.json`).

The gap is wider on other models. In one sitting in the default mode AMD's `session.run` took 12.84–12.89 ms on
YOLOv8s against Ignition's 17.11–17.12 ms dispatch with `--silu-sigmoid` (16.70–16.79 ms without), and 1.51 ms on SESR
M7 against 4.18–4.19 ms, so AMD's stack is faster end to end on both: 16.75–16.79 against 18.18–18.21 ms (17.80–17.87
without the flag), and 4.35 against 4.78 ms ([performance notes](docs/PERFORMANCE.md#yolov8s-and-sesr-m7-against-amds-stack)).
With spinning threads on 2026-09-15 the YOLOv8s gap was 0.30 ms; in the default mode without the flag it was 1.36 ms on
2026-09-16 and 1.04–1.08 ms on 2026-09-17, and how much of that is the mode rather than the installation was not separated.

- [x] **Keep data on the NPU between layers: tried in ignite-xdna, and closed as a known limitation on YOLOv8s
  (2026-09-16).** The target was Ignition's NPU stage no slower than AMD's in a same-sitting run. Holding activations
  in the MemTile (an activation ring) and holding weights there (a resident weight buffer) were both built, both
  byte-exact, and 3.40 and 3.63 ms slower on YOLOv8s. A MemTile hop costs no measurable time per byte, so routing
  around it gains nothing, and trimming packets, fuller fill tasks, hardware compression and core-to-core halo
  exchange are unavailable, parked or need a kernel change (ignite-xdna `078769a`,
  `results/aie/weight_buffer_phoenix_20260916T1508Z.log`, `results/aie/memtile_hop_phoenix_20260916T1523Z.log`).
  The traffic that remains is cut by model shape ([model guide](docs/QUANTIZATION-GUIDE.md#what-the-runtime-cannot-take-off-the-bill)).
  SESR M7's dispatch target in §1 is not closed by this.

### 5. Compile time (ignite-xdna)

`ignite-compile` runs its stages one after another, each on one core. One YOLOv8n compile on the 8-core,
16-thread Ryzen 7 8700G, run in the environment `install.ps1` builds and measured inside a job object on 2026-09-15,
took 13.7 s wall-clock and 15.25 s of CPU across 70 processes: 1.11 cores on average, one thread above half a core
for 94 % of the time, and a peak of 5.5 cores only in its last half second. In order, the stages were Python
lowering and scheduling (0–1.9 s), `clang++` compiling `engine.cc` (1.9–4.9 s, 3.1 s CPU), and `aiecc` on the
1 MB `design.mlir` (5.3–13.5 s, 8.6 s CPU on one busy thread).

- [ ] Make `ignite-compile` detect the host's cores and use them:
  - compile `engine.cc` while lowering runs, or reuse its object when `kernel_sha256` and the compile flags are
    unchanged;
  - find which of `aiecc`'s steps take its 8 s, and run the ones that can be split in parallel.
- **Done when:** a compile keeps more than one core busy for most of its wall-clock time, is faster than a serial
  compile of the same model on the same host, and writes `insts.bin` and `wpackets.bin` byte-identical to it.

### 6. Energy and power modes

Power modes size the host's worker threads to its own cores: `performance` spins every logical processor,
`balanced` (the default) sleeps one per physical core, `efficiency` sleeps a quarter of the physical cores. All of
the models have been measured, on the 8-core, 16-thread Ryzen 7 8700G only ([performance notes](docs/PERFORMANCE.md#energy-per-frame-against-amds-stack)).

- [x] Re-measure the latency comparisons against AMD's stack in the `balanced` default. Done for every table and
  badge: `balanced` costs YOLOv8n 0.43 ms against `performance`, all of it host work around an unchanged dispatch.
- [x] Measure energy per frame at 30 fps and at full speed on YOLOv8s, SESR M7, YOLOv8n-pose and YOLO11n. Ignition
  spends less on YOLO11n (8.4 times) and YOLOv8n-pose (19 % flat out, 27 % at 30 fps). AMD's stack spends less
  flat out on YOLOv8s and SESR M7, and at 30 fps the stacks do not separate on either. Two sittings were discarded
  for a 5.5 W idle offset; ignite-xdna's energy tool now flags one.
- [x] Measure what the NPU's own power mode buys before deciding whether `efficiency` should switch it.
  `xrt-smi configure --pmode powersaver` saved 14 % energy per frame at 30 fps for twice the latency, and nothing
  flat out. Ignition then ran slower than AMD's stack in the same mode, and the setting slows every NPU application.
- [x] Decide whether any power mode should switch the NPU's device-wide mode. The maintainer's decision
  (2026-09-16): yes. Nothing switches it yet.
- [x] Design and build that switch (recorded in the commit that adds this entry; ignite-xdna's
  `pipelines/npu_power.py`, first released in ignite-xdna 0.3.1). With `--power-mode efficiency` and paced frames, `--npu-power auto`
  (the default) lowers the device to `powersaver` at the end of warm-up if the predicted frame fits 80 % of its period,
  the device reads `default` and no other process uses the NPU. It restores `default` on slow frames, on another
  process's context and on exit, and a lease file covers a killed run. At 30 fps it saved 20 % and 28 % energy per frame in two sittings on
  YOLOv8n and 27 % on YOLO11n, for about twice the G2G ([performance notes](docs/PERFORMANCE.md#efficiency-with-the-npu-power-switch)).
- [ ] Measure the switch on a webcam with `--fresh`, where the frame period comes from the camera's measured rate;
  only `--max-fps` has been measured.
- [ ] Price the NPU's `balanced` device mode as an intermediate step for models that do not fit `powersaver` (YOLOv8s at
  30 fps), which the switch does not use.
- [x] ignite-xdna `fbd53f5` puts a container's CPU step (YOLO11n's attention core) under the power mode. It ships in
  ignite-xdna 0.3.1, which the `npu` extra requires since v0.3.3; 0.3.0 still spins those threads in every mode.
- [ ] Measure energy per frame on the `--silu-sigmoid` containers on a clean host. Every energy figure in the
  performance notes predates them. The v0.3.3 attempt at 30 fps was discarded: a desktop application held one core,
  idle baselines sat at 36.7–40.6 W against about 35 W clean, and AMD's YOLOv8n arm read 222.42 and 315.98 mJ in its
  two runs (ignite-xdna `results/aie/release_033/`).
- [ ] Find out why SESR M7's frame does not spin in `performance` (9.7–10.0 % CPU, like the sleeping modes).
- [x] `--power-mode` needs an ignite-xdna with `pipelines/power.py`: the `npu` extra now requires ignite-xdna 0.3.0,
  which has it. An older ignite-xdna ignores the mode and prints no power line.
- The 6-core measurement is blocked on hardware and lives in §3 with the Hawk Point run.

### 7. Models AMD's stack cannot run usefully on XDNA1 (frontier)

**The goal:** more models that are *usable* on a Phoenix NPU through Ignition than through AMD's stack. Usable means
layer-exact on the NPU, accurate against FP32, and faster than ONNX Runtime's CPU provider on the same host. Every
claim is measured beside AMD's stack in one sitting, as the badges already are.

**The gap to aim at**, from measurements already in ignite-xdna's `docs/BENCHMARKS.md`:
- **XINT8 collapse.** AMD's quantization collapses some models: MobileViT-XXS at 0.00 % top-1, DenseNet-121 and
  ResNeXt-50 at 0.10 %, YOLO-World v2 at 1.8 % mAP against 37.0 % in FP32. The CPU collapses the same way on the same
  quantized models, so the answer is precision, not placement. (YOLO-World v2's turned out to be rounding, not a lack of
  bits: four convolutions whose output is a small difference of large terms, recovered at 8 bits with GPTQ rounding and
  int32 biases in ignite-xdna `80ca69e`.)
- **INT16 activations.** The silicon runs them at 0.93 times int8's speed, but AMD's toolchain cannot express them
  on XDNA1: at opset 17 its runtime rejects the quantization nodes, and at opset 21 its parser rejects the model.
- **Placement.** Some models barely reach the NPU: YOLO11n gets 6 of 1,300 nodes and YOLO-World v2 48 of 1,081,
  which leaves them slower than the CPU alone. ignite-xdna's graph engine runs both with only their attention on the CPU
  (YOLO-World v2: 63 of its 67 convolutions on the NPU).
- **Newer releases.** In this project's testing, Ryzen AI 1.8 installed without the Phoenix xclbin and rejected the
  driver's XDNA1 xclbins, so XDNA1 users stay on 1.7.1 ([amd/RyzenAI-SW#400](https://github.com/amd/RyzenAI-SW/issues/400)).
- **SiLU's HardSigmoid form.** AMD's XINT8 replaces every SiLU's sigmoid with a HardSigmoid. On the first 500 COCO
  images, the form alone is 64.8-74.8 % of the XINT8 accuracy loss of YOLOv8n, YOLOv8s and YOLOv8n-pose. The graph
  engine now has a four-line sigmoid instead, as an option (Kernel decisions, below).
- **The class that fits.** The engine's tile is 20 px, which is the deepest map of a network at a 640 input
  (stride 32). Dense-prediction models at 640 fit: detection, segmentation, pose, depth, matting and
  super-resolution. Classifiers at 224 shrink to 7 px and do not.

**Compiler and tooling (no kernel change):**
- [ ] A compatibility matrix in the performance notes: per model, layer-exactness, accuracy and latency against the
  CPU, with AMD's placement, accuracy and latency beside them, and a "usable models" badge against AMD.
- [ ] `ignition check model.onnx`: which nodes would run on the NPU, which on the CPU, and which are refused with the
  rule that refuses them, plus a predicted frame time, all before a compile.
- [ ] Automatic partitioning. Today CPU segments are named by hand (`--host-region`). The compiler should find the
  largest NPU-ready subgraphs and decide with the dispatch cost model (about 0.25 ms fixed per dispatch plus transport)
  where a CPU segment pays.
- [ ] Exact graph rewrites:
  - **Hypothesis, to prove offline first:** a 1x1 stride-2 convolution is a 3x3 stride-2 pad-1 convolution with only
    its centre weight, which reads the same pixels at the same output size. It would unlock ResNet-style downsample
    shortcuts.
  - Channel counts padded to multiples of 32.
  - Classifier tails (global average pool, Gemm) as CPU segments.
- [ ] **Split containers and composable pipelines (with ignite-xdna Milestone 5.6):**
  - **Motivation:** An `.ignite` container historically bundled manifest, microcode, and stationary weights into a single binary. Splitting decouples microcode from weights and enables composable vision pipelines and early-exit cascades in Ignition.
  - **Measured Silicon Reality (Phoenix 8700G, 2026-09-20, Desktop 2):**
    - Context recreation between stages costs **29.63 ms** (FATAL); all stages must execute within a single persistent `InferenceSession`.
    - Intra-context NPU dispatch gap between linked segments costs only **6.6 µs**.
    - Decoupled weight sidecars (`.weights`) incur **0.000 ms runtime dispatch penalty** and compress container file sizes by **92.4% on YOLOv8n** (8.84 MB -> 674 KB) and **95.9% on YOLOv8s** (30.76 MB -> 1.26 MB) with bit-exact silicon execution.
    - Early-exit cascades via `max_segments=1` (cutting at layer 10) were recorded as a **73% latency reduction** (2.055 ms vs 7.556 ms). **Withdrawn:** the cut point has no detect heads, so it is a per-segment cost, not a faster detector ([why](docs/PERFORMANCE.md#yolov8s-and-sesr-m7-against-amds-stack)).
  - **Engine Capabilities (landed on `split-container-sizing`):**
    - `ignite-compile --split-layer <IDX>` cuts execution into contiguous NPU segments (`insts_0.bin`, `insts_1.bin`) dispatched sequentially.
    - `ignite-compile --decouple-weights` emits slimmed `.ignite` plus companion `.weights` sidecar.
    - `serializer.py:decouple_container_weights()` enables in-place stripping of existing containers in ~15 ms without re-compiling.
    - `InferenceSession` and `EngineSession` auto-resolve `.weights` sidecars and support `max_segments` early exit.
  - **Consumer Integration Next Steps:**
    - Support `--weights <file.weights>` in `live_ignition.py` and auto-discover `.weights` companion sidecars alongside `.ignite` containers.
    - Surface early-exit cascading in `AsyncYOLOPipeline` to enable high-throughput (>400 FPS) background rejection before full head evaluation.
    - Composable multi-task pipelines: dynamically assemble pipelines sharing a single backbone activation buffer (e.g. Detect + Pose) with zero duplicate computation.
- [x] **Mixed precision through CPU segments, YOLO-World v2** (ignite-xdna `9096925`, `80ca69e`). The collapse is four
  C2fAttn output convolutions, not the attention. With those four as FP32 CPU steps a container scored 24.5 % mAP on the
  first 300 COCO val2017 images, identical to ONNX Runtime; AMD's stack runs that model at the same accuracy with 110
  nodes on the NPU and 96.34 ms per image in its eval. Requantized with GPTQ rounding and int32 biases, the four run on
  the NPU instead: 24.7 %, exact on every layer, and a profiled frame of 48.0-48.2 ms against 53.0-53.5 ms for the
  FP32-step container in the same sitting. The profile quantizes the input and dequantizes the heads in numpy, and it is
  not a sitting against AMD's stack (ignite-xdna `results/aie/yolow_gptq/`).
  **Against AMD's stack in one sitting** (ignite-xdna `1c9364e`, `460324f`; `5_eval_map.py` per-image inference on the
  first 300 images, interleaved):
  - the container 47.33 and 47.51 ms at 24.7 %;
  - AMD's stack on the FP32-step model, its best usable one, 95.91 and 95.89 ms at 24.5 %: so 2.0 times AMD at equal
    accuracy;
  - but DirectML on the iGPU runs the FP32 model at 46.22 ms at 43.0 %, with every node on DirectML;
  - the CPU runs it at 67.75 ms at 43.0 %.

  No README row or badge until the container clearly beats the iGPU on speed or energy (ignite-xdna
  `results/aie/yolow_sitting_vs_amd/`).

  **Frame to detections and energy per frame** (ignite-xdna `a62450d`..`469c9d5`: a bit-exact int8 decode, then
  `pipelines/yolow/4b_g2g.py` on `bus.jpg`, 500 frames per run, interleaved):
  - **Latency, 80 classes:** container 39.912 / 39.853 ms, AMD's stack 109.729 / 109.376 ms (2.75 times), iGPU FP32
    52.415 ms, CPU FP32 81.593 ms.
  - **Latency, five names:** container 32.595 ms, iGPU 28.444 ms.
  - **Energy per frame, flat out** (`tools/energy_sitting.py`, own idle): container 1040.36 / 908.91 mJ, AMD's stack
    4483.41 / 4544.51 mJ, iGPU 1201.12 / 1336.00 mJ, CPU 3610.19 / 3580.88 mJ.
  - **Energy per frame at 5 fps:** 1428.13, 5337.04, 842.48 and 4214.34 mJ.

  So against AMD's stack the container is both faster and cheaper at equal accuracy. It beats the iGPU flat out with
  80 classes, but not with five classes or at 5 fps, and stays 18.3 points less accurate: no README row or badge yet
  (ignite-xdna `results/aie/yolow_g2g_sitting/`, `results/aie/yolow_energy/`).
- [ ] Mixed precision for MobileViT-XXS: not tried. At its 224 px input the maps fall below the 20 px tile.
- [ ] Per-channel weight scales, which AMD's stack rejects. **Answered (ignite-xdna `engine.cc`):** a weight packet
  carries one output shift for its 32 output channels and a per-channel int32 bias, so per-channel weight scales are a
  change to the one engine program. On YOLO-World v2's collapsing convolutions, GPTQ at one scale per tensor was ahead
  of per-channel scales on two of four layers and within 4 dB on the other two, with no kernel change (ignite-xdna
  `results/aie/yolow_gptq/cv2_cancellation.log`).
- [ ] **Open-vocabulary detection in Ignition.** ignite-xdna `6a39780`..`a78a500`: one YOLO-World v2 container takes
  any class names at run time, because the vocabulary enters only its CPU attention steps and the host decode
  (`YoloWorldPipeline.set_classes`, 90.1 ms for six names; 70/70 layers exact with another vocabulary). Renaming 23
  COCO categories to synonyms cost the quantized model 22 % of their mAP against 11 % in FP32. To do, in order:
  - [x] a faster decode and native ingress in the timed path (ignite-xdna `a62450d`: bit-exact int8 decode, 40.1 ms
    glass-to-glass with 80 classes);
  - [x] energy per frame against AMD's stack, the CPU and the iGPU (above);
  - [x] the NPU's power-saving mode at 5 fps, where the iGPU spends less (ignite-xdna `c714629`). The container in
    `efficiency` with the NPU in `powersaver` spends 1086.27 / 1060.09 mJ per frame against 1364.78 / 1489.51 mJ in its
    defaults (20-29 % less against each arm's own idle, at most 19 % against the median idle). That costs 1.71 times
    the frame time: 69.1-69.3 against 40.5 ms. The iGPU still spends 651.69 / 684.28 mJ, though its paced frames take
    79.6 ms. No power-mode default changes;
  - the accuracy gap to FP32 (24.7 against 43.0 %), which decides whether the iGPU comparison can ever be like for
    like;
  - `--task world --classes` in `live_ignition.py` and `ignition suite`;
  - a sitting through Ignition.

**Kernel decisions (the one-program rule: the maintainer decides, with measured sizing):**
- [x] **A better SiLU than AMD's HardSigmoid form** (ignite-xdna `94305cf`..`4b238a7`, approved by the maintainer).
  `ignite-compile --silu-sigmoid` gives every SiLU a four-line sigmoid, which the core program applies after the
  convolution passes. Every layer is exact on the NPU against an integer reference model. Containers compiled without
  the flag are byte-identical to before and no slower.
  - **Accuracy, all 5,000 COCO val2017 images** (mAP; pose OKS): AMD's stack 26.68 / 37.31 / 32.64, today's containers
    27.10 / 37.21 / 32.71, sigmoid containers 34.12 / 42.37 / 44.16 for YOLOv8n / YOLOv8s / YOLOv8n-pose.
  - **Glass-to-glass, one sitting** (ms, means of two runs of `bus.jpg`): AMD's stack 10.741 / 16.986 / 12.340, today's
    containers 8.419 / 18.008 / 9.006, sigmoid containers 8.723 / 18.427 / 9.324. The flag costs 2.2-3.9 % of the NPU
    dispatch.
  - **Energy at 30 fps:** YOLOv8n does not separate from AMD's stack. YOLOv8n-pose spends less than AMD's stack and
    YOLOv8s more, with or without the flag.
  - **It stays a compiler flag.** YOLO11n and YOLO-World v2 cannot use it (their CPU segments), and SESR M7 has no SiLU.

  Decided by the maintainer (2026-09-17): Ignition's documented YOLOv8n, YOLOv8s and YOLOv8n-pose containers are
  compiled with the flag, released in v0.3.3. Re-measured in the release sitting against same-commit controls, the flag
  costs 0.24 / 0.25 ms on YOLOv8n, 0.34 / 0.39 ms on YOLOv8s and 0.30 / 0.35 ms on YOLOv8n-pose (ignite-xdna `results/aie/release_033/latency_release033_phoenix_20260917T1538Z.log`).
  Evidence for the accuracy: ignite-xdna `results/aie/silu_sigmoid_vs_amd/`.
- [ ] Maps smaller than the 20 px tile: classifiers, and 28 of ResNet50's 53 convolutions.
- [ ] An INT16-activation kernel, for models that collapse at 8 bits.
- [ ] Dilated and transposed convolutions, for segmentation and depth decoders.
- [ ] Attention on the NPU with the BF16 GEMM kernel. Low priority: YOLO11n's attention core is under a millisecond
  on the CPU.

**App tasks built from the above:** instance segmentation (YOLOv8n-seg: its mask assembly is one small matrix
multiply on the CPU, and the rest is convolutions, like pose), depth (FastDepth, MiDaS small), matting (MODNet),
semantic segmentation (BiSeNetV2), oriented boxes, and open-vocabulary detection (YOLO-World v2, item above).

**Split and Modular Containers (Decoupled Weights & Subgraph Cascades):**
- [x] **Decoupled stationary weights (`.weights` sidecar):** `serializer.py` and `ignite-compile --decouple-weights` strips weights from `.ignite` containers, shrinking distribution size by **92.3%–95.9%** across the entire YOLOv8 family (0.68–8.67 MB containers) with 0.000 ms steady-state dispatch penalty and bit-exact outputs. (An earlier draft read 90.5%–96.8%, wider than any model measured.)
- [ ] **Early-exit cascades (`max_segments=N`):** multi-segment NPU execution works on all six YOLOv8 variants, but the early-exit *speedup* is **withdrawn**. This entry read "cutting latency uniformly by 72.1%–75.0% ... and beating AMD's monolithic passes by 2.40x to 5.23x"; both compared a partial dispatch against AMD's complete pass, and a partial dispatch has no detect heads, so it returns no detections at all ([why](docs/PERFORMANCE.md#yolov8-family-split-containers-against-amds-stack)). A real early exit needs a decision rule on the backbone's feature map; neither that rule nor its cost nor its accuracy exists yet.
- [x] **Zero-copy stage chaining (`InferenceSession.compose`):** Reuses a single shared DDR workspace (`bo_ws`) across modular subgraphs without memory bloat or host bus transfers.

**Not pursued:** general transformers on XDNA1. Every CPU island in the middle of a block pays the dispatch floor.

## Needs a decision: duplicated history on `main`

`main` holds two copies of the same 11 commits (`a7efb00` … `965579d` and `e08f23c` … `c16ceee`), joined by
`10f22f4` (`git merge -s ours --allow-unrelated-histories`); tags `v0.1.0` and `v0.2.0` point into the second.
Removing them means rewriting `main` and both tags on both remotes, a force-push this workflow does not allow,
so they stay unless the maintainer decides otherwise. Rebasing local `main` alone would only make it diverge
from both remotes.
