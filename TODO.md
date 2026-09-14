# Ignition TODO

State of `main` at `971d540` (2026-09-14). Measured figures name the commit whose message records the run;
Ignition does not track benchmark logs (`results/` is gitignored). How the pipeline works is in the
[README](README.md#how-a-frame-runs).

## Completed: native NPU integration (ignite-xdna `v0.1.0-phoenix-npu`)

- [x] **NPU runtime integration** (`ee66fbd`). `src/ignition/pipelines/yolo.py` recognises an `.ignite`
  container by its suffix or `IGNT` header and serves it through ignite-xdna's `YoloPipeline` on NPU
  Device 0, with boxes decoded from the NPU's detect heads. ONNX Runtime is not called on this path
  (0 `run()` calls in every recorded run). `.onnx` models keep the ONNX Runtime path. The pipeline is a
  context manager, and a finalizer releases the NPU hardware context if `close()` is never called.
- [x] **Camera capture and shutdown** (`ee66fbd`). `ThreadedCamera` in `live_ignition.py` tries
  DirectShow, then Media Foundation, then OpenCV's default backend. Each open runs in its own thread and
  is abandoned after `--open-timeout`. The capture thread publishes frames with sequence numbers and
  skips empty reads. q, ESC, closing the window, Ctrl+C and Ctrl+Break all release the camera, the
  window and the NPU context, then exit 0.
- [x] **Live multi-object detection on silicon** (`8ffa482`). Headless 300-frame runs on a lit webcam
  averaged 7.82 and 7.87 ms G2G with 5.44 and 5.27 objects per frame. The windowed run averaged 7.98 ms
  over 714 frames at 5.74 objects per frame. The 500-frame run averaged 8.02 ms and the `--fresh` run
  8.03 ms. A re-check on 2026-09-14 averaged 7.89 ms with 5.43 objects per frame.
- [x] **Stability over 500+ frames** (`ee66fbd`, `8ffa482`). Resident memory changed by −1.12 MB over
  500 lit webcam frames, −1.15 MB over 500 dark ones and +0.02 MB over 500 frames of a still image.
  Every run exited 0, and afterwards `xrt-smi` reported no hardware contexts running.
- [x] **Box parity** (`ee66fbd`). On `bus.jpg` the NPU heads give 5 detections (4 person, 1 bus), the
  same set as the ONNX Runtime path, with a class-matched mIoU of 0.977.
- [x] **CLI on `.ignite`** (`ee66fbd`). `ignition detect` and `ignition detect --stream --benchmark` run
  the container. On 2026-09-14 a 100-frame benchmark on `bus.jpg` sustained 123.47 FPS with a median of
  7.98 ms.

## Active

### 1. Camera rate: 30 fps from the webcam

On Desktop 2 the 640×480 webcam delivers 15 fps through DirectShow, whatever frame rate or FOURCC is
requested, while Media Foundation delivered 30 fps when both were probed on 2026-09-14. The `--fresh` run in a lit room (`8ffa482`) ran at
15 FPS, so dim-light exposure is not the whole cause. `ThreadedCamera` currently requests no format at
all.

- [ ] Log the negotiated `CAP_PROP_FOURCC`, `CAP_PROP_FPS` and `CAP_PROP_AUTO_EXPOSURE` on the
  `[camera]` line.
- [ ] Try MJPG at 30 fps on DirectShow: request FOURCC first, then size, then rate, with auto-exposure
  priority off. Judge the result by the measured frame arrival rate, not by the property read-back.
- [ ] If DirectShow stays at 15 fps, prefer Media Foundation when it opens within the timeout, or add a
  `--backend dshow|msmf|any` flag.
- **Done when:** `live_ignition.py --headless --frames 300 --fresh` reports about 30 FPS with 310 distinct
  source frames.

### 2. Post-processing in native code

DFL decode and NMS take 0.27–0.33 ms per frame with 5–6 objects in view, against 0.05 ms on an empty
scene (`8ffa482`). That difference is the latency cost of a busy scene. The code is ignite-xdna's
`YoloDecoder.postprocess` in `pipelines/yolo_pipeline.py`: numpy DFL decode, then
`cv2.dnn.NMSBoxesBatched` over Python lists built with `.tolist()`. Ignition's wrapper adds about 8 µs.

- [ ] Move DFL anchor expansion, for the anchors that survive the class-max prune, and greedy per-class
  NMS into AVX2 C in ignite-xdna. Build and load it like `pipelines/preprocess_simd.c` (ctypes, `-mavx2`).
- [ ] Before switching, require boxes identical to the numpy path on `bus.jpg` and on recorded camera
  frames.
- **Target:** post-processing at or under 0.05 ms with objects in view, about 0.25 ms back per frame (derived: 0.30 − 0.05).

### 3. Model zoo in Ignition

Two unmerged branches already hold this work:

- **ignite-xdna `worktree-model-zoo`** (`d828678`, `6e704ce`, `ffbc77e`) compiles YOLOv8s and SESR M7 to
  `.ignite` and runs them on Device 0.
- **Ignition `model-zoo`** (`91ace94`) makes `live_ignition.py` task-aware (`--task`, `--json`), with
  classification and super-resolution pipelines.

ignite-xdna's `tools/model_zoo_bench.py` drives Ignition for the suites. Measured 2026-09-14, logs in
ignite-xdna `results/model_zoo/` on that branch, G2G means:

| Model | ONNX Runtime CPU, 300 frames | NPU through Ignition |
|---|---:|---:|
| YOLOv8s | 83.66 ms | 17.73 ms |
| yolo11n_no_c2psa | 35.79 ms | — |
| SESR M7 | 15.58 ms | 6.57 ms |
| ResNet50 | 34.24 ms | — |

- [ ] Land ignite-xdna `worktree-model-zoo` on its `main`. Ignition's native super-resolution path
  imports `pipelines/sr_pipeline.py`, which only that branch has.
- [ ] Fast-forward Ignition `main` to `model-zoo`, then document `--task` and `--json` in the README.
- [ ] Move the suite runner into Ignition as one command that writes a JSON record per model.
- [ ] yolo11n_no_c2psa finds nothing on `bus.jpg` (C2PSA removed), so check it detects before lowering it.
  ResNet50 has no `.ignite` lowering yet.
- [ ] SESR M7 dispatch is 4.25 ms against a 1.5 ms target. The non-compute floor is 2.53 ms because
  activations return to host memory between layers. Reaching the target needs on-chip activation
  hand-off in ignite-xdna.

### 4. Honest backends

- [ ] For an `.onnx` model with `backend="xdna1"` (the default of `ignition.compile` and `ignition detect`),
  `YOLOPipeline` loads ignite-xdna's layer backend and calls it once per frame. It discards the output,
  swallows any exception, and takes boxes from ONNX Runtime. Drop the call, or time it as a separate
  stage, and state that this path runs on the CPU.
- [ ] `ignition devices` prints fixed values for tile clock, MemTile SRAM and INT8 TOPS. Print only what
  `pyxrt` reports.

## Needs a decision: duplicated history on `main`

`main` contains two copies of the same 11 commits: `a7efb00` … `965579d`, and `e08f23c` … `5465db7`.
Tags `v0.1.0` and `v0.2.0` point into the second copy. `971d540` joined the two histories with
`git merge -s ours --allow-unrelated-histories` and left the tree unchanged. Both origin (GitLab) and
github already have that `main`.

Removing the copies means rewriting `main` on both remotes and moving both tags. That is a force-push,
which this repository's workflow does not allow. The duplicates stay unless the maintainer explicitly
decides otherwise. Rebasing local `main` alone would not remove them from the remotes; it would only make
local `main` diverge from both.
