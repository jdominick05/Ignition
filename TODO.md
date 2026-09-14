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
- [x] **Consumer README** (`13157f8`, `6100486`): same-sitting comparison with AMD's stack and a compatibility
  matrix.
- [x] **pip install** (`60a7be1`, ignite-xdna `389f0a7`): Ignition's wheel and sdist install on their own; the
  `npu` extra with ignite-xdna's wheel adds the NPU path.
- [x] **v0.3.1 release** (tag at `ea5f86e`): both wheels, the sdist and SHA-256 sums on GitHub and GitLab, with
  [notes](docs/releases/v0.3.1.md); every file was downloaded back from both hosts and matched the sums.

## Active

### 1. Camera rate: 30 fps from the webcam

The test webcam delivers 15 fps at 640×480 through DirectShow whatever rate or FOURCC is requested, even in a
lit room (`8ffa482`); Media Foundation delivered 30 fps when both were probed on 2026-09-14. `ThreadedCamera`
requests no format at all.

- [ ] Log the negotiated `CAP_PROP_FOURCC`, `CAP_PROP_FPS` and `CAP_PROP_AUTO_EXPOSURE` on the `[camera]` line.
- [ ] Try MJPG at 30 fps on DirectShow (FOURCC, then size, then rate, auto-exposure priority off), judged by the
  measured frame arrival rate, not the property read-back.
- [ ] If DirectShow stays at 15 fps, prefer Media Foundation when it opens within the timeout, or add a
  `--backend dshow|msmf|any` flag.
- **Done when:** `live_ignition.py --headless --frames 300 --fresh` reports about 30 FPS with 310 distinct
  source frames.

### 2. Post-processing in native code

DFL decode and NMS take 0.27–0.33 ms per frame with 5–6 objects in view and 0.05 ms on an empty scene
(`8ffa482`). The code is ignite-xdna's `YoloDecoder.postprocess` (`pipelines/yolo_pipeline.py`): numpy DFL
decode, then `cv2.dnn.NMSBoxesBatched` over lists built with `.tolist()`.

- [ ] Move DFL anchor expansion (for anchors that survive the class-max prune) and greedy per-class NMS into
  native C in ignite-xdna, built and loaded like `pipelines/preprocess_simd.c` (OpenMP C compiled on first use,
  loaded through ctypes, no AVX2 intrinsics today). Use AVX2 only where it measurably helps.
- [ ] Before switching, require boxes identical to the numpy path on `bus.jpg` and on recorded camera frames.
- **Target:** at or under 0.05 ms with objects in view, about 0.25 ms back per frame (0.30 − 0.05).

### 3. Model zoo in Ignition

Two unlanded branches hold this work, and neither fast-forwards onto its `main` any more:

- **ignite-xdna `worktree-model-zoo`** (`d828678`, `6e704ce`, `ffbc77e`, forked from `397da63`) compiles
  YOLOv8s and SESR M7 to `.ignite` and runs them on Device 0.
- **Ignition `model-zoo`** (`8a1248a`, forked from `6100486`) makes `live_ignition.py` task-aware (`--task`,
  `--json`), with classification and super-resolution pipelines.

ignite-xdna's `tools/model_zoo_bench.py` drives Ignition for the suites. Measured 2026-09-14, logs in
ignite-xdna `results/model_zoo/` on that branch, G2G means:

| Model | ONNX Runtime CPU, 300 frames | NPU through Ignition |
|---|---:|---:|
| YOLOv8s | 83.66 ms | 17.73 ms |
| yolo11n_no_c2psa | 35.79 ms | — |
| SESR M7 | 15.58 ms | 6.57 ms |
| ResNet50 | 34.24 ms | — |

- [ ] Land ignite-xdna `worktree-model-zoo` on its `main` (Ignition's super-resolution path imports
  `pipelines/sr_pipeline.py`, which only that branch has), then raise the `npu` extra's minimum to the first
  ignite-xdna version that ships it.
- [ ] Replay Ignition `model-zoo` onto `main`, land it, and document `--task` and `--json` in the README.
- [ ] Move the suite runner into Ignition as one command that writes a JSON record per model.
- [ ] yolo11n_no_c2psa finds nothing on `bus.jpg` (C2PSA removed), so check it detects before lowering it.
  ResNet50 has no `.ignite` lowering yet.
- [ ] SESR M7 dispatch is 4.25 ms against a 1.5 ms target, with a 2.53 ms non-compute floor (§7).

### 4. Honest backends

- [ ] For an `.onnx` model with `backend="xdna1"` (the default of `ignition.compile` and `ignition detect`),
  `YOLOPipeline` loads ignite-xdna's layer backend and calls it once per frame. It discards the output,
  swallows any exception, and takes boxes from ONNX Runtime. Drop the call, or time it as a separate
  stage, and state that this path runs on the CPU.
- [ ] `ignition devices` prints a fixed device name, core count, tile clock, MemTile SRAM and INT8 TOPS. Print
  only what `pyxrt` reports.

### 5. Release and distribution

- [ ] Add installing from a release page to the README's Install section, beside the editable checkouts.
- [ ] ignite-xdna is on no package index, so the `npu` extra resolves only with its wheel beside Ignition's.
  Decide whether to publish both to PyPI.
- [ ] No `.ignite` container ships, so the NPU path still needs the mlir-aie toolchain. Decide whether a
  prebuilt `yolov8n_full.ignite` can be attached to a release, after checking the model's licence.
- [ ] The published v0.2.0 notes still quote the figures v0.3.1 corrects. Decide whether to edit them.
- **Done when:** the README's install commands, copied from a release page, work in a fresh environment.

### 6. Hardware and Python coverage

Only Phoenix on Windows 11 with Python 3.13 is verified ([README](README.md#compatibility)).

- [ ] Run the `.ignite` path on a Hawk Point NPU (same XDNA1 generation) and record it before calling it
  supported.
- [ ] The package declares Python 3.10 or later, but the XRT SDK's `pyxrt` is built for 3.13. Test the CPU
  install on 3.10–3.12 and state that the NPU path needs the Python `pyxrt` was built for.

### 7. NPU dispatch time (ignite-xdna)

AMD's NPU stage is about 0.8 ms faster than Ignition's on the same model and image (`13157f8`). Most of
Ignition's dispatch is activations moving between host memory and the NPU: 5.37 ms of a 7.39 ms dispatch with
every weight operation switched off (ignite-xdna `results/model_zoo/dispatch_floor_yolov8n_full.json`).

- [ ] Keep activations on the NPU between layers in ignite-xdna (SESR M7's target needs the same change),
  then re-run the AMD comparison in one sitting.
- **Done when:** Ignition's NPU stage is no slower than AMD's in a same-sitting run.

## Needs a decision: duplicated history on `main`

`main` holds two copies of the same 11 commits (`a7efb00` … `965579d` and `e08f23c` … `5465db7`), joined by
`971d540` (`git merge -s ours --allow-unrelated-histories`); tags `v0.1.0` and `v0.2.0` point into the second.
Removing them means rewriting `main` and both tags on both remotes, a force-push this workflow does not allow,
so they stay unless the maintainer decides otherwise. Rebasing local `main` alone would only make it diverge
from both remotes.
