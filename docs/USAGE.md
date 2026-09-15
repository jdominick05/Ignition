# Using Ignition

Reference for what the README's walkthrough leaves out. Install with the README's one-step installer first.

## Compatibility

| | Status | Details |
|---|---|---|
| **AMD Phoenix NPU (XDNA1)** | ✅ Verified | Ryzen 7 8700G, NPU `[003d:00:01.1]` |
| **AMD Hawk Point NPU (XDNA1)** | ⚠️ Untested | Same XDNA1 4×4 NPU generation as Phoenix; the `.ignite` path has not been run on one |
| **Strix-class NPUs (AIE2P)** | ❌ Not supported | Different NPU architecture; `.ignite` containers target XDNA1 |
| **Windows 11** | ✅ Verified | Windows 11 Pro, build 26200.9445 |
| **Linux** | ❌ Not supported | Ignition uses the Windows XRT stack |
| **NPU driver and XRT** | ✅ Verified | NPU driver 32.0.20101.3760, firmware 1.5.5.391, XRT 2.21.0 |
| **Python** | ✅ 3.10–3.13 (CPU), 3.13 (NPU) | The CPU path installs and runs on 3.10, 3.11, 3.12 and 3.13. The NPU path needs the Python the XRT SDK's `pyxrt` was built for, 3.13 with XRT 2.21.0; on 3.12 it stops at `DLL load failed while importing pyxrt` |
| **YOLOv8n detection on the NPU** | ✅ Verified | 640×640 input, AMD Quark XINT8, compiled to `build/yolov8n_full.ignite` by ignite-xdna |
| **Other models on the NPU** | ⚠️ From source only | YOLOv8s detection, SESR M7 super-resolution and YOLOv8n-pose pose estimation run on the NPU through `live_ignition.py` with ignite-xdna built from its `main`, and so does YOLO11n detection with its attention matrix multiplies and softmax on ONNX Runtime's CPU provider; none is in a release of either project ([how](#5-run-other-models)) |
| **Other ONNX models** | ✅ CPU only | ONNX Runtime's CPU execution provider; `live_ignition.py` detects, classifies, estimates poses or upscales according to the model's outputs |
| **Webcams** | ✅ Verified | USB webcams through DirectShow, then Media Foundation; tested at 640×480 |
| **Video files and images** | ✅ Verified | Anything OpenCV opens, letterboxed to 640×640; tested with 640×480 and 810×1080 frames |
| **Interfaces** | ✅ Verified | Webcam app `live_ignition.py`, Python API, `ignition` command line |
| **AMD Ryzen AI Software** | Not needed at runtime | The model input is AMD Quark's XINT8 ONNX; turning it into a container needs ignite-xdna and the mlir-aie toolchain |

## Other ways to install

**From a release.** Download the wheels from the [v0.3.1 release](https://github.com/jdominick05/Ignition/releases/tag/v0.3.1) into one folder and run pip in that folder; GitLab's release carries the same files. For the NPU path:

```bash
pip install ./ignite_xdna-0.2.0-py3-none-any.whl "./ignition_ai-0.3.1-py3-none-any.whl[npu]"
```

For `.onnx` models on the CPU only:

```bash
pip install ./ignition_ai-0.3.1-py3-none-any.whl
```

The wheels carry the Python API and the `ignition` command, and the container is still built in an ignite-xdna checkout. v0.3.1 predates the backend, camera, native-decode, model-zoo and suite changes listed in [TODO.md](../TODO.md)'s completed work: its wheels have no webcam app, which the package now holds as `ignition.live` and a checkout's `live_ignition.py` launches.

**From source**, which the steps below use:

```bash
git clone https://github.com/jdominick05/ignite-xdna.git
git clone https://github.com/jdominick05/Ignition.git
pip install -e ignite-xdna
pip install -e Ignition
cd Ignition
```

### 1. See it work

```bash
python live_ignition.py
```

A window opens on webcam 0 with boxes, labels, scores and the per-frame latency. Press q or ESC to quit.

### 2. Measure it on your machine

```bash
python live_ignition.py --headless --frames 300
```

This command's output on the test machine (lit room, webcam 0):

```text
[summary] stop: frame limit | 310 frames processed (10 warm-up, 300 timed) | 70 distinct source frames | camera 0 via DSHOW 640x480
[summary] camera: 71 frames in 2.4 s = 29.78 fps read, 0 repeated the previous frame, 29.78 distinct fps
[summary] G2G mean 7.485 ms | P50 7.483 | P95 7.637 | P99 7.794 | max 7.942 | over the last 300 timed frames; cold first frame 11.35 ms
[summary] stage means (ms): preprocess 0.133 | NPU forward 7.317 (dispatch 7.122, readback 0.195) | decode+NMS 0.030
[summary] boxes from NPU heads on 300/300 timed frames | 4.62 detections per frame
[summary] RSS 211.7 MB at the first timed frame, 211.7 MB at the end (+0.00 MB over 300 frames)
```

More ways to run it:

```bash
python live_ignition.py --headless --frames 300 --fresh   # wait for each new camera frame (camera-paced)
python live_ignition.py --headless --frames 300 --fresh --exposure-priority off   # hold 30 fps in dim light: darker image, fewer detections
python live_ignition.py --model ../ignite-xdna/build/yolov8n_full.ignite --source examples/assets/bus.jpg --headless --frames 300
python live_ignition.py --model ../ignite-xdna/models/yolov8n_cut_xint8.onnx --source examples/assets/bus.jpg --headless --frames 300   # CPU path
python live_ignition.py --headless --frames 300 --json run.json   # also write the summary as JSON
```

| Flag | Meaning |
|---|---|
| `--model` | An `.ignite` container (NPU) or `.onnx` model (CPU). Defaults to `../ignite-xdna/build/yolov8n_full.ignite`. |
| `--task` | What the model computes. `auto` (the default) reads it from the container's manifest or the ONNX model's outputs; `detect`, `classify`, `super_resolution` or `pose` sets it. |
| `--source` | Webcam index (`0`, `1`, …), video file or image (default `0`). |
| `--headless` | No window; prints a progress line every 100 frames. |
| `--frames N` | Stop after N timed frames and print the summary (default 0: run until stopped). |
| `--warmup N` | Untimed frames before the timed ones (default 10). |
| `--fresh` | Wait for a new camera frame before each inference instead of reusing the newest one. |
| `--conf`, `--iou` | Detection confidence and NMS IoU thresholds (defaults 0.25 and 0.45). |
| `--open-timeout` | Seconds allowed for each camera backend to open (default 8). |
| `--camera-backend` | `auto` (DirectShow, then Media Foundation, then OpenCV's default; the default), `dshow`, `msmf` or `any`. |
| `--exposure-priority` | `keep` (default) leaves the webcam's setting. `off` holds the frame rate in dim light at the cost of a darker image; `on` lets auto exposure lower it. DirectShow only. The camera keeps this setting across programs, so the previous value is put back on exit; a killed process leaves it changed. |
| `--json PATH` | Also write the run's summary to PATH as one JSON object: glass-to-glass percentiles, stage means, resident memory, how many frames the NPU produced, the task's result (detections per frame, top-1 class counts or output shape), the camera's measured rate and the host's library versions. |

### 3. Use it from Python

```python
import ignition

with ignition.compile("../ignite-xdna/build/yolov8n_full.ignite", pipeline="yolo") as pipe:
    result = pipe.predict("examples/assets/bus.jpg")
    print(result.summary())  # detections plus preprocess, NPU dispatch, head readback and NMS times
```

- **`pipe.predict`** takes a path, a PIL image or a BGR numpy array.
- **`pipe.stream(frames)`** yields one result per frame from an iterable.

### 4. Use it from the command line

```bash
ignition devices
ignition detect ../ignite-xdna/build/yolov8n_full.ignite --input examples/assets/bus.jpg --output detections.jpg
ignition detect ../ignite-xdna/build/yolov8n_full.ignite --input examples/assets/bus.jpg --stream --benchmark --frames 100
```

On the test machine, the 100-frame benchmark sustained 128.78 frames per second with a median of 7.70 ms.

### 5. Run other models

`live_ignition.py` works out what a model computes from the container's manifest or the ONNX model's outputs, and `--task` overrides it:

- **Detection:** YOLO models (YOLOv8n, YOLOv8s, YOLO11n), drawn as boxes.
- **Classification:** one `(1, N)` output, such as ResNet50, shown as the top 5. Preprocessing follows a timm `preprocess_config.json` beside the model, or ImageNet defaults. It runs on the CPU only.
- **Super-resolution:** one image output a whole multiple of the input size, such as SESR M7 (256×256 to 512×512), shown as the upscaled image.
- **Pose estimation:** a head-cut YOLOv8-pose model (nine outputs, three with 51 keypoint channels), drawn as each person's box and 17-keypoint skeleton.

YOLOv8s, SESR M7, YOLO11n and YOLOv8n-pose run on the NPU with ignite-xdna built from its `main`. No ignite-xdna release has its super-resolution or pose pipeline yet; without them a SESR or pose model stops at load with an `ImportError` that says so, and a pose `.onnx` model on the CPU needs the pose pipeline too, for its decode. Build the containers in the ignite-xdna checkout with the toolchain YOLOv8n needs (ignite-xdna's [model zoo notes](https://github.com/jdominick05/ignite-xdna/blob/main/docs/MODEL_ZOO_BENCHMARKS.md) record how these were built and checked):

```bash
ignite-compile --engine graph --input models/yolov8s_cut_xint8.onnx --output build/yolov8s.ignite
ignite-compile --engine graph --input models/sesr_m7_xint8.onnx --output build/sesr_m7.ignite
ignite-compile --engine graph --input models/yolo11n_cut_xint8.onnx --output build/yolo11n.ignite --host-region "/model.10/m/m.0/attn/qkv/conv/Conv=/model.10/m/m.0/attn/Reshape_1"
ignite-compile --engine graph --input models/yolov8n-pose_cut_xint8.onnx --output build/yolov8n_pose.ignite
```

`yolov8n-pose_cut_xint8.onnx` comes from ignite-xdna's `pipelines/yolov8n-pose/` export and quantization scripts; neither project ships it.

YOLO11n's attention core, the two matrix multiplies and softmax inside its C2PSA block, has no NPU lowering. `--host-region FROM=TO` keeps only the nodes between the qkv convolution's output and the reshape after the second matrix multiply on the CPU; every convolution, the block's included, runs on the NPU. The container runs two NPU dispatches with one ONNX Runtime call between them, and the `[summary]` stage line reports that call as `host`. In Git Bash, set `MSYS_NO_PATHCONV=1` for that command, or Git Bash rewrites the node names as file paths. `yolo11n_cut_xint8.onnx` comes from ignite-xdna's `pipelines/yolov11/` export and quantization scripts; neither project ships it.

Then, from Ignition:

```bash
python live_ignition.py --model ../ignite-xdna/build/yolov8s.ignite --source examples/assets/bus.jpg   # window with boxes
python live_ignition.py --model ../ignite-xdna/build/yolo11n.ignite --source examples/assets/bus.jpg   # window with boxes
python live_ignition.py --model ../ignite-xdna/build/yolov8n_pose.ignite --source examples/assets/bus.jpg   # window with skeletons
python live_ignition.py --model ../ignite-xdna/build/sesr_m7.ignite --source examples/assets/bus.jpg --headless --frames 300 --json sesr_m7.json
python live_ignition.py --model ../ignite-xdna/models/resnet50_xint8_c64.onnx --source examples/assets/bus.jpg   # top 5, CPU
```

### 6. Compare models in one command

```bash
ignition suite ../ignite-xdna/build/yolov8n_full.ignite ../ignite-xdna/models/yolov8n_cut_xint8.onnx --out results/suite-yolov8n
```

- **One process per model:** each model runs through the app in its own process (`python -m ignition.live --headless`), on `examples/assets/bus.jpg` unless `--source` names another image, video or webcam, with `--warmup` (default 10) untimed and `--frames` (default 300) timed frames.
- **One record per model:** `--out` gets `<model>.json`, which holds the app's `--json` summary plus the command, its return code, the Ignition and ignite-xdna versions with `git describe --dirty` of their checkouts, and, for an `.ignite` container, what `xrt-smi` reported before and after the run. A `<model>.log` and an `index.json` sit beside them. Checkout paths, the output directory and the Windows profile directory appear as labels.
- **Nothing overwritten:** an existing `--out` directory is refused. The command exits 1 if a run fails, records no timed frames, or starts while another hardware context is on the NPU, which would make its latency contention.

## Limitations

- **One NPU model in a release.** YOLOv8n is the only model the released packages run on the NPU. YOLOv8s, SESR M7 and YOLO11n need ignite-xdna built from its `main`, ResNet50 classification has no `.ignite` lowering, and other ONNX models run on the CPU.
- **A build step.** The `.ignite` container is built from AMD Quark's quantized model with ignite-xdna and the mlir-aie toolchain; it is not a pip install.
- **AMD's stack is faster on some models.** On YOLOv8s (16.95 against 17.24 ms) and SESR M7 (3.63 against 6.66 ms) its NPU stage outruns the engine, which moves activations and weights to the NPU every frame ([measurements](PERFORMANCE.md#yolov8s-and-sesr-m7-against-amds-stack)).
- **Narrow hardware support.** Only Phoenix has been verified. Hawk Point is untested, and Strix-class NPUs and Linux are not supported.
- **YOLO11n's attention core runs on the CPU.** Its matrix multiplies and softmax took 0.51–0.53 ms of a 10.1–10.4 ms frame. No accuracy has been measured for the container, and its borderline boxes change with one-code differences in preprocessing: on `bus.jpg` it finds 6 objects where AMD's stack finds 7.
- **Dim light halves the webcam's frame rate.** The test webcam's auto exposure drops to 15 fps in a dim room; `--exposure-priority off` holds 30 fps with a darker image and fewer detections.
Open work is tracked in [TODO.md](../TODO.md).

## Project layout

```
Ignition/
├── .github/                   # README badges: badges.toml, badges.py, and the workflow that checks them
├── install.ps1                # one-step PowerShell installer and updater (see Install)
├── live_ignition.py           # launches ignition.live from this checkout
├── src/ignition/
│   ├── __init__.py            # compile(), devices()
│   ├── live.py                # webcam / video / image app: detect, classify or upscale; percentiles, --json
│   ├── suite.py               # ignition suite: one ignition.live process and JSON record per model
│   ├── pipelines/yolo.py      # YOLOPipeline: .ignite on the NPU via ignite-xdna, .onnx on ONNX Runtime
│   ├── pipelines/vision.py    # task inference, ClassificationPipeline, PosePipeline, SuperResolutionPipeline
│   ├── pipelines/streaming.py # 3-stage asynchronous runner for .onnx models
│   ├── backends/              # xdna1 layer backend, ONNX Runtime CPU backend
│   ├── model.py, devices.py   # Model API, NPU discovery
│   └── cli/main.py            # ignition devices | run | benchmark | detect | suite
├── examples/                  # yolo_vision_demo.py, quickstart.py, assets/bus.jpg
├── TODO.md                    # completed milestones and open work
└── pyproject.toml             # package ignition-ai, console script `ignition`
```
