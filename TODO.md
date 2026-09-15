# Ignition TODO

State of `main` on 2026-09-14 (version 0.3.1). Measured figures name the commit whose message records the run;
Ignition does not track benchmark logs (`results/` is gitignored). How the pipeline works is in the
[README](README.md#how-a-frame-runs).

## Completed

- [x] **Native NPU path** (`ee66fbd`): `YOLOPipeline` serves an `.ignite` container through ignite-xdna's
  `YoloPipeline` on NPU Device 0, with boxes from the NPU's detect heads and no ONNX Runtime `run()` calls.
- [x] **Camera and shutdown** (`ee66fbd`): timed opens across DirectShow, Media Foundation and OpenCV's default
  backend; q, ESC, closing the window, Ctrl+C and Ctrl+Break release the camera, window and NPU and exit 0.
- [x] **Live detection on silicon** (`ee66fbd`, `8ffa482`): 7.82–8.03 ms mean G2G at 5.0–5.9 objects per
  frame, resident memory −1.15 to +0.02 MB over 500 frames, and `bus.jpg` boxes matching ONNX Runtime (mIoU
  0.977). Every run is in the [README](README.md#performance).
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
- [x] **Install from a release:** the README's Install section gives the release page's pip commands beside the
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
  glass-to-glass on the NPU, and ResNet50 classified on the CPU at 30.49 ms ([README](README.md#other-models)).
- [x] **Suite runner in Ignition:** `ignition suite MODEL...` runs each model in its own `ignition.live` process
  and writes one JSON record and log per model plus an index, refusing an existing output directory. The app
  moved into the package as `ignition.live`; `live_ignition.py` launches it from a checkout
  ([README](README.md#6-compare-models-in-one-command)).
- [x] **ignite-xdna's suites on `ignition suite`** (ignite-xdna `373d897`): `tools/model_zoo_bench.py` hands its
  presets to `ignition suite`, writes each run's records into a new `results/model_zoo/<suite>_<UTC time>/`
  directory, and rewrites its JSON and `docs/MODEL_ZOO_BENCHMARKS.md`'s tables only when every run is clean. The
  tables there were not re-run.
- [x] **YOLO11n with its attention block** (ignite-xdna `0be9132`): the C2PSA-ablated model detects nothing, so the
  graph engine runs stock YOLO11n instead. Its convolutions run in two NPU dispatches and its attention block on
  ONNX Runtime's CPU provider between them, with all 84 layers exact on Device 0. In one sitting on `bus.jpg`,
  `live_ignition.py` took 10.996 and 11.048 ms mean glass-to-glass (NPU dispatch 8.45 ms, attention block
  1.69 ms, reported as the `host` stage). AMD's stack took 34.492 and 34.366 ms, and ONNX Runtime's CPU provider
  31.328 ms (ignite-xdna `results/aie/yolo11n_hybrid_phoenix_20260915T0216Z.log`, [README](README.md#yolo11n-with-its-attention-block)).
- [x] **YOLO11n's attention-block convolutions on the NPU** (ignite-xdna `d223e7b`): the CPU step keeps only the attention
  core (two matrix multiplies and a softmax); the block's seven convolutions and residual adds run on the NPU, all 91
  layers exact on Device 0. In one sitting, 500 frames each, the attention-core container took 10.411 and 10.115 ms
  mean glass-to-glass (CPU step 0.533 and 0.505 ms) against 11.109 and 10.987 ms with the whole block on the CPU and
  36.361 and 34.549 ms on AMD's stack (ignite-xdna `results/aie/yolo11n_attention_core_phoenix_20260915T1522Z.log`).

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

- [ ] Raise the `npu` extra's minimum to the first ignite-xdna release that ships `pipelines/sr_pipeline.py`,
  which Ignition's super-resolution path imports; no ignite-xdna release has it yet.
- [ ] YOLO11n on the NPU needs ignite-xdna from source at `0be9132` or later, and `d223e7b` or later to keep only the
  attention core on the CPU; no ignite-xdna release has host segments. Raise the `npu` extra's minimum when one
  does, with the super-resolution item above.
- [ ] YOLO11n's attention core, two matrix multiplies and a softmax, runs on the CPU (0.51–0.53 ms per frame);
  the engine has no softmax or activation-by-activation multiply, so running it on the NPU is ignite-xdna work.
  The block cannot be dropped: the C2PSA-ablated `yolo11n_no_c2psa` finds nothing on `bus.jpg`
  (highest class score 0.02 in FP32 and 0.06 in XINT8), and ignite-xdna measured mAP@50-95 0.19 for it on AMD's
  Vitis AI EP against 38.72 for stock FP32 on the CPU (ignite-xdna `docs/BENCHMARKS.md`, its YOLOv11n section).
- [ ] Measure YOLO11n's COCO accuracy through the container. On `bus.jpg` it finds 6 objects where ONNX Runtime
  on Ignition's numpy letterbox finds 7, because the two inputs differ by one code in 15% of pixel values.
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
    and the only verified setup is the one in the README's [Compatibility](README.md#compatibility) table.
- [ ] The published v0.2.0 notes still quote the figures v0.3.1 corrects. Decide whether to edit them.

### 3. Hardware coverage

Only Phoenix on Windows 11 is verified ([README](README.md#compatibility)).

- [ ] Run the `.ignite` path on a Hawk Point NPU (same XDNA1 generation) and record it before calling it
  supported.

### 4. NPU dispatch time (ignite-xdna)

AMD's NPU stage is about 0.7 ms faster than Ignition's on the same model and image in the
[README](README.md#ignition-vs-amds-ryzen-ai-stack)'s same-sitting comparison (0.8 ms in `d42d33e`). Most of
Ignition's dispatch is activations moving between host memory and the NPU: 5.37 ms of a 7.39 ms dispatch with
every weight operation switched off (ignite-xdna `results/model_zoo/dispatch_floor_yolov8n_full.json`).

- [ ] Keep activations on the NPU between layers in ignite-xdna (SESR M7's target needs the same change),
  then re-run the AMD comparison in one sitting.
- **Done when:** Ignition's NPU stage is no slower than AMD's in a same-sitting run.

## Needs a decision: duplicated history on `main`

`main` holds two copies of the same 11 commits (`a7efb00` … `965579d` and `e08f23c` … `c16ceee`), joined by
`10f22f4` (`git merge -s ours --allow-unrelated-histories`); tags `v0.1.0` and `v0.2.0` point into the second.
Removing them means rewriting `main` and both tags on both remotes, a force-push this workflow does not allow,
so they stay unless the maintainer decides otherwise. Rebasing local `main` alone would only make it diverge
from both remotes.
