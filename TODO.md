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

## Active

### 1. Model zoo in Ignition

ignite-xdna's `main` compiles YOLOv8s and SESR M7 to `.ignite` and runs them on Device 0 (merge `6bd2718`). The
Ignition side is still a branch that no longer fast-forwards onto `main`:

- **Ignition `model-zoo`** (`3cf2c49`, forked from `e3a6fcf`) makes `live_ignition.py` task-aware (`--task`,
  `--json`), with classification and super-resolution pipelines.

ignite-xdna's `tools/model_zoo_bench.py` drives Ignition for the suites. Measured 2026-09-14 on the branch
before the merge, logs in ignite-xdna `results/model_zoo/`, G2G means:

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
- [ ] Replay Ignition `model-zoo` onto `main`, land it, and document `--task` and `--json` in the README.
- [ ] Move the suite runner into Ignition as one command that writes a JSON record per model.
- [ ] yolo11n_no_c2psa finds nothing on `bus.jpg` (C2PSA removed), so check it detects before lowering it.
  ResNet50 has no `.ignite` lowering yet.
- [ ] SESR M7 dispatch is 4.25 ms against a 1.5 ms target, with a 2.53 ms non-compute floor (§4).

### 2. Release and distribution

- [ ] ignite-xdna is on no package index, so the `npu` extra resolves only with its wheel beside Ignition's.
  Decide whether to publish both to PyPI.
- [ ] No `.ignite` container ships, so the NPU path still needs the mlir-aie toolchain. Decide whether a
  prebuilt `yolov8n_full.ignite` can be attached to a release, after checking the model's licence.
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
