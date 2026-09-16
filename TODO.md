# Ignition TODO

State of `main` on 2026-09-16 (version 0.3.2). Measured figures name the commit whose message records the run;
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

## Active

### 1. Model zoo in Ignition

ignite-xdna's `main` compiles YOLOv8s and SESR M7 to `.ignite` and runs them on Device 0 (merge `6bd2718`), and
Ignition's `live_ignition.py` serves them (`--task`, `--json`); neither project has released it.

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

- [x] The `npu` extra requires ignite-xdna 0.3.0 (ignite-xdna `554ab57`), the first ignite-xdna release with
  `pipelines/sr_pipeline.py`, `pipelines/pose_pipeline.py` and host segments, which Ignition's super-resolution, pose
  and YOLO11n paths import.
- [ ] YOLO11n's attention core, two matrix multiplies and a softmax, runs on the CPU (0.51–0.53 ms per frame);
  the engine has no softmax or activation-by-activation multiply, so running it on the NPU is ignite-xdna work.
  The block cannot be dropped: the C2PSA-ablated `yolo11n_no_c2psa` finds nothing on `bus.jpg`
  (highest class score 0.02 in FP32 and 0.06 in XINT8), and ignite-xdna measured mAP@50-95 0.19 for it on AMD's
  Vitis AI EP against 38.72 for stock FP32 on the CPU (ignite-xdna `docs/BENCHMARKS.md`, its YOLOv11n section).
- [ ] Measure YOLO11n's COCO accuracy through the container. On `bus.jpg` it finds 6 objects where ONNX Runtime
  on Ignition's numpy letterbox finds 7, because the two inputs differ by one code in 15% of pixel values.
- [ ] YOLOv8n-pose decodes its keypoints in numpy: decode and NMS take 0.29–0.32 ms per frame against 0.03 ms for
  YOLOv8n's native decode. A container from the AdaRound pose model (34.32 OKS mAP@50-95 on AMD's stack) is not
  built or checked.
- [ ] ResNet50 has no `.ignite` lowering yet.
- [ ] SESR M7 dispatch is 4.25 ms against a 1.5 ms target, with a 2.53 ms non-compute floor (§4).

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
- [ ] The published v0.2.0 notes still quote the figures v0.3.1 corrects. Decide whether to edit them.

### 3. Hardware coverage

Only Phoenix on Windows 11 is verified ([usage notes](docs/USAGE.md#compatibility)).

- [ ] Run the `.ignite` path on a Hawk Point NPU (same XDNA1 generation) and record it before calling it
  supported.

### 4. NPU dispatch time (ignite-xdna)

AMD's NPU stage is about 0.7 ms faster than Ignition's on the same model and image in the
[performance notes](docs/PERFORMANCE.md#ignition-vs-amds-ryzen-ai-stack)' same-sitting comparison (0.8 ms in `d42d33e`). Most of
Ignition's dispatch is activations moving between host memory and the NPU: 5.37 ms of a 7.39 ms dispatch with
every weight operation switched off (ignite-xdna `results/model_zoo/dispatch_floor_yolov8n_full.json`).

The gap is wider on other models. In one sitting AMD's `session.run` took 13.14–13.17 ms on YOLOv8s against Ignition's
16.70–16.74 ms dispatch, and 1.46–1.47 ms on SESR M7 against 4.39–4.41 ms, so AMD's stack is faster end to end on
both: 16.95–16.96 against 17.24–17.27 ms, and 3.63–3.65 against 6.66–6.67 ms ([performance notes](docs/PERFORMANCE.md#yolov8s-and-sesr-m7-against-amds-stack)).

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
`balanced` (the default) sleeps one per physical core, `efficiency` sleeps a quarter of the physical cores. Only
YOLOv8n on the 8-core, 16-thread Ryzen 7 8700G has been measured ([performance notes](docs/PERFORMANCE.md#energy-per-frame-against-amds-stack)).

- [ ] Re-measure the latency comparisons against AMD's stack in the `balanced` default: every latency badge and table
  was measured with spinning threads, which is `performance` now, and `balanced` gave up about 0.46 ms per frame on
  YOLOv8n.
- [ ] Measure energy per frame at 30 fps and at full speed on YOLOv8s, SESR M7, YOLOv8n-pose and YOLO11n.
- [ ] Measure the modes on a 6-core part (with the Hawk Point run in §3) before calling their sizing verified.
- [ ] Decide whether `efficiency` should also switch the NPU's own power mode (`xrt-smi configure --pmode`, 0.8 GHz in
  `powersaver`). It is device-wide, so it would slow every other NPU application until put back.
- [x] `--power-mode` needs an ignite-xdna with `pipelines/power.py`: the `npu` extra now requires ignite-xdna 0.3.0,
  which has it. An older ignite-xdna ignores the mode and prints no power line.

## Needs a decision: duplicated history on `main`

`main` holds two copies of the same 11 commits (`a7efb00` … `965579d` and `e08f23c` … `c16ceee`), joined by
`10f22f4` (`git merge -s ours --allow-unrelated-histories`); tags `v0.1.0` and `v0.2.0` point into the second.
Removing them means rewriting `main` and both tags on both remotes, a force-push this workflow does not allow,
so they stay unless the maintainer decides otherwise. Rebasing local `main` alone would only make it diverge
from both remotes.
