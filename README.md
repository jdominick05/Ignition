``` 
 (                                  
 )\ )                )              
(()/((  (      (  ( /((             
 /(_))\))( (   )\ )\())\  (   (     
(_))((_))\ )\ |(_|_))((_) )\  )\ )  
|_ _|(()(_)(_/((_) |_ (_)((_)_(_/(  
 | |/ _` | ' \)) |  _|| / _ \ ' \)) 
|___\__, |_||_||_|\__||_\___/_||_|  
    |___/                               
```

**Real-time object detection on the AMD Ryzen™ AI NPU: faster end to end than AMD's own runtime, from a fraction of the install size.**

![NPU: AMD Phoenix XDNA1](https://img.shields.io/badge/NPU-AMD%20Phoenix%20XDNA1-ed1c24)
![YOLOv8n: about 8 ms per frame](https://img.shields.io/badge/YOLOv8n-~8%20ms%20per%20frame-brightgreen)
![Verified on Ryzen 7 8700G](https://img.shields.io/badge/verified-Ryzen%207%208700G-blue)
![License: AGPL-3.0-or-later](https://img.shields.io/badge/license-AGPL--3.0--or--later-blue)

Ignition runs YOLOv8n object detection on the NPU built into AMD Ryzen AI processors. All 66 layers of the network run on the NPU, not your CPU. Point it at a webcam and each captured frame becomes labelled boxes in about 8 milliseconds.

- **About 25% lower end-to-end latency than AMD's stack.** The same model and image took 7.75–7.78 ms per frame through Ignition and 10.33–10.37 ms through AMD's ONNX Runtime Vitis AI execution provider, measured back to back on the same machine.
- **Inference about 4× faster than on the CPU.** The network itself took 7.4 ms on the NPU and 28.4 ms on ONNX Runtime's CPU path, for the same model and image. End to end, a frame took 7.75–7.78 ms against 32.1 ms.
- **Lighter to install and run.** Resident memory was 192.5–192.7 MB against AMD's 305.4–306.3 MB (37% less), and the runtime install is about 318 MB against 4.7 GB (93% smaller).
- **Same answers.** On the reference image both stacks find the same five objects, and their boxes overlap 97% on average (mean IoU 0.968).
- **A working app included.** `python live_ignition.py` opens your webcam with boxes, labels, confidence scores and a live latency readout.
- **YOLO11n with its attention block, from source.** In one sitting stock YOLO11n took 11.00–11.05 ms per frame through Ignition and 34.37–34.49 ms through AMD's stack. Its convolutions run on the NPU and its attention block on the CPU between two NPU dispatches ([details](#yolo11n-with-its-attention-block)).

## Ignition vs AMD's Ryzen AI stack

The same model (`yolov8n_cut_xint8.onnx`, AMD Quark XINT8) ran on the same image (`examples/assets/bus.jpg`) on a Ryzen 7 8700G. Each run had 50 warm-up and 500 timed frames. Runs alternated AMD, Ignition, AMD, Ignition, with the NPU checked idle before each, and both runs are shown in every cell. Ignition ran on ignite-xdna `5688f6d`.

| | **Ignition** | **AMD Ryzen AI Software 1.7.1**<br/>ONNX Runtime + Vitis AI EP |
|---|---|---|
| **End-to-end latency, mean** | **7.78 / 7.75 ms** | 10.37 / 10.33 ms |
| End-to-end latency, 95th percentile | **8.02 / 7.96 ms** | 10.93 / 10.98 ms |
| ↳ image preparation (letterbox, quantize) | **0.31 / 0.31 ms**, native code into the NPU's buffer | 1.72 / 1.73 ms |
| ↳ NPU | 7.43 / 7.41 ms | **6.71 / 6.70 ms** |
| ↳ box decode and NMS | **0.033 / 0.031 ms**, native code | 1.94 / 1.90 ms |
| **Process memory (RSS)** | **192.7 / 192.5 MB** | 305.4 / 306.3 MB |
| **Runtime install on disk** | **≈318 MB** | ≈4,704 MB |
| Detections on `bus.jpg` | 4 people, 1 bus | the same 5 objects (box IoU 0.93–0.99) |
| Where the network runs | all 66 layers on the NPU | 922 of 929 graph nodes on the NPU; 7 quantize/dequantize nodes on the CPU |
| Models it accelerates | YOLOv8n in a release; YOLOv8s, SESR M7 and YOLO11n (attention block on the CPU) with ignite-xdna from source; other `.onnx` models run on the CPU | any ONNX model its compiler accepts |
| Before the first run | a compiled `.ignite` container (a one-time build with ignite-xdna) | none: the model compiles on its first session and is cached |

**What the comparison does and does not show:**
- **Where Ignition's lead comes from.** It is entirely in host-side code. AMD's runtime executes the network and leaves image preparation and box decoding to your application, so the AMD column uses Ignition's reference numpy versions of those steps. Ignition ships them as native code. Hand-written native pre- and post-processing on AMD's stack would narrow the gap.
- **Where AMD is ahead.** AMD's NPU stage itself is about 0.7 ms faster than Ignition's. AMD's stack also runs a much wider range of models, and its current release targets newer NPUs.
- **What the disk figures count.** AMD's covers the Ryzen AI 1.7.1 install (4,072 MB), its ONNX Runtime with the Vitis AI EP (615 MB), `voe` (3 MB) and the compile cache (14 MB). Ignition's covers the XRT SDK with `pyxrt` (262 MB), ONNX Runtime for the CPU fallback (44 MB), ignite-xdna (3 MB), the `.ignite` container (8 MB) and Ignition itself (66 KB). Neither counts Python, numpy, OpenCV or the NPU driver. Building a container needs the mlir-aie toolchain, which is not counted.
- **Why the AMD column uses 1.7.1.** In this project's testing, Ryzen AI Software 1.8.0 installed without the Phoenix/Hawk Point xclbin that its own documentation calls for, and rejected the driver's XDNA1 xclbins, so NPU inference on these chips stays on 1.7.1. That looks like a packaging bug, reported upstream as [amd/RyzenAI-SW#400](https://github.com/amd/RyzenAI-SW/issues/400) ([ignite-xdna decision record](https://github.com/jdominick05/ignite-xdna/blob/main/docs/DECISIONS.md)).

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
| **Other models on the NPU** | ⚠️ From source only | YOLOv8s detection and SESR M7 super-resolution run on the NPU through `live_ignition.py` with ignite-xdna built from its `main`, and so does YOLO11n detection with its attention block on ONNX Runtime's CPU provider; none is in a release of either project ([how](#5-run-other-models)) |
| **Other ONNX models** | ✅ CPU only | ONNX Runtime's CPU execution provider; `live_ignition.py` detects, classifies or upscales according to the model's outputs |
| **Webcams** | ✅ Verified | USB webcams through DirectShow, then Media Foundation; tested at 640×480 |
| **Video files and images** | ✅ Verified | Anything OpenCV opens, letterboxed to 640×640; tested with 640×480 and 810×1080 frames |
| **Interfaces** | ✅ Verified | Webcam app `live_ignition.py`, Python API, `ignition` command line |
| **AMD Ryzen AI Software** | Not needed at runtime | The model input is AMD Quark's XINT8 ONNX; turning it into a container needs ignite-xdna and the mlir-aie toolchain |

## Quick start

### What you need

- **Hardware:** a Windows 11 PC with an AMD Phoenix NPU and AMD's NPU driver.
- **Python:** the version the XRT SDK's `pyxrt` was built for, 3.13 with XRT 2.21.0; `pyxrt` does not load on others. The CPU path on its own runs on 3.10 to 3.13.
- **Runtime and model:** [ignite-xdna](https://github.com/jdominick05/ignite-xdna) checked out next to Ignition, with the YOLOv8n container built there by `ignite-compile --engine graph --output build/yolov8n_full.ignite` (needs the mlir-aie IRON toolchain; see ignite-xdna's README).

### Install

**From a release.** Download the wheels from the [v0.3.1 release](https://github.com/jdominick05/Ignition/releases/tag/v0.3.1) into one folder and run pip in that folder; GitLab's release carries the same files. For the NPU path:

```bash
pip install ./ignite_xdna-0.2.0-py3-none-any.whl "./ignition_ai-0.3.1-py3-none-any.whl[npu]"
```

For `.onnx` models on the CPU only:

```bash
pip install ./ignition_ai-0.3.1-py3-none-any.whl
```

The wheels carry the Python API and the `ignition` command, and the container is still built in an ignite-xdna checkout. v0.3.1 predates the backend, camera, native-decode, model-zoo and suite changes listed in [TODO.md](TODO.md)'s completed work: its wheels have no webcam app, which the package now holds as `ignition.live` and a checkout's `live_ignition.py` launches.

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
| `--task` | What the model computes. `auto` (the default) reads it from the container's manifest or the ONNX model's outputs; `detect`, `classify` or `super_resolution` sets it. |
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

YOLOv8s, SESR M7 and YOLO11n run on the NPU with ignite-xdna built from its `main`. No ignite-xdna release has its super-resolution pipeline yet; without it a SESR container stops at load with an `ImportError` that says so. Build the containers in the ignite-xdna checkout with the toolchain YOLOv8n needs (ignite-xdna's [model zoo notes](https://github.com/jdominick05/ignite-xdna/blob/main/docs/MODEL_ZOO_BENCHMARKS.md) record how these were built and checked):

```bash
ignite-compile --engine graph --input models/yolov8s_cut_xint8.onnx --output build/yolov8s.ignite
ignite-compile --engine graph --input models/sesr_m7_xint8.onnx --output build/sesr_m7.ignite
ignite-compile --engine graph --input models/yolo11n_cut_xint8.onnx --output build/yolo11n.ignite --host-region /model.10/
```

YOLO11n's attention block (C2PSA, the nodes named `/model.10/`) has no NPU lowering, so `--host-region` keeps it on the CPU: the container runs two NPU dispatches with one ONNX Runtime call between them, and the `[summary]` stage line reports that call as `host`. In Git Bash, set `MSYS_NO_PATHCONV=1` for that command, or Git Bash rewrites `/model.10/` as a file path. `yolo11n_cut_xint8.onnx` comes from ignite-xdna's `pipelines/yolov11/` export and quantization scripts; neither project ships it.

Then, from Ignition:

```bash
python live_ignition.py --model ../ignite-xdna/build/yolov8s.ignite --source examples/assets/bus.jpg   # window with boxes
python live_ignition.py --model ../ignite-xdna/build/yolo11n.ignite --source examples/assets/bus.jpg   # window with boxes
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

## Performance

The table's figures come from YOLOv8n on NPU Device 0 of a Ryzen 7 8700G; [other models](#other-models) follow it:
- **Setup:** Windows 11, `build/yolov8n_full.ignite`, webcam 0 through DirectShow at 640×480, 10 warm-up frames per run unless the command sets `--warmup`. Rows with a commit ran ignite-xdna `v0.1.0-phoenix-npu`, which decodes boxes in numpy and runs NMS through OpenCV; *re-check* rows ran ignite-xdna `5688f6d`, which does both in one native C pass.
- **Records:** each row names the commit whose message records the run. *Re-check* rows were measured on 2026-09-14 and are recorded in the commit that added them to this README.

| `live_ignition.py` run | Scene | Timed frames | Objects / frame | Mean | 99th pct | Record |
|---|---|---:|---:|---:|---:|---|
| `--headless --frames 300` | webcam, dark room | 300 | 0 | 7.58 ms | 7.91 ms | `ee66fbd` |
| `--headless --frames 300` | webcam, lit room | 300 | 5.44 | 7.82 ms | 8.18 ms | `8ffa482` |
| `--headless --frames 300` | webcam, lit room | 300 | 5.27 | 7.87 ms | 8.58 ms | `8ffa482` |
| `--headless --frames 300` | webcam, lit room | 300 | 5.43 | 7.89 ms | 8.39 ms | `0f374e5` |
| `--headless --frames 300` | webcam, lit room | 300 | 4.62 | 7.49 ms | 7.79 ms | re-check |
| `--headless --frames 500` | webcam, lit room | 500 | 5.90 | 8.02 ms | 8.55 ms | `8ffa482` |
| `--headless --frames 300 --fresh` | webcam, every frame new | 300 | 5.00 | 8.03 ms | 8.51 ms | `8ffa482` |
| window, closed with q | webcam, lit room | 714 | 5.74 | 7.98 ms | 8.52 ms | `8ffa482` |
| `--source` a 640×480 crop of `bus.jpg`, `--headless --frames 500` | still image | 500 | 4.00 | 7.86 ms | 8.39 ms | `ee66fbd` |
| `--source examples/assets/bus.jpg --headless --frames 300` | still image, 810×1080 | 300 | 5.00 | 8.06 ms | 8.67 ms | `0f374e5` |
| `--source examples/assets/bus.jpg --headless --warmup 50 --frames 500` | still image, 810×1080 | 500 | 5.00 | 7.78 ms | 8.28 ms | re-check |
| `--source examples/assets/bus.jpg --headless --warmup 50 --frames 500` | still image, 810×1080 | 500 | 5.00 | 7.75 ms | 8.10 ms | re-check |

- **Frame rate:** Ignition's processing loop ran at 122 to 129 frames per second with the numpy decode and 133 to 134 with the native one, so the camera sets the live rate, and the camera's auto exposure sets that. The test webcam, a Logitech C920, delivered 30 distinct frames per second in a bright room and 15 in a dim one, whatever rate or pixel format was requested. In the dim room `--exposure-priority off` held 30, with 2.43 detections per frame against 5.55–6.06 at 15. Media Foundation reads at 30 fps partly by repeating frames, so the `[summary] camera:` line counts the distinct ones. With `--fresh`, each new frame became detections 8.10 ms after it arrived.
- **Decode and NMS:** in native code they took 0.030 ms with 4.62 objects per frame on the webcam and 0.031–0.033 ms with 5 on `bus.jpg`. The numpy and OpenCV version behind the rows with a commit took 0.25–0.33 ms with 5–6 objects in view and 0.05 ms on an empty scene.
- **Long runs:** resident memory did not grow: −1.12 MB over 500 webcam frames in a lit room, −1.15 MB over 500 in a dark one, +0.02 MB over 500 frames of a still image.
- **Clean exit:** every run exited cleanly and released the NPU; afterwards `xrt-smi` reported no hardware contexts running.

### Other models

Measured on 2026-09-14 through `live_ignition.py` with ignite-xdna `68c2fea` and containers built from it, on `examples/assets/bus.jpg` with 10 warm-up and 300 timed frames, one run each; recorded in the commit that added this section.

| Model | Task | Runs on | Mean | 99th pct | Output |
|---|---|---|---:|---:|---|
| YOLOv8s | detect | NPU | 17.22 ms | 17.62 ms | 6 objects |
| SESR M7 | super_resolution | NPU | 6.59 ms | 7.36 ms | 512×512 image |
| SESR M7 | super_resolution | CPU (ONNX Runtime) | 13.99 ms | 18.68 ms | 512×512 image |
| ResNet50 | classify | CPU (ONNX Runtime) | 30.49 ms | 40.09 ms | top-1 ImageNet class 654 (minibus), p = 0.58 |

- **Where the time goes:** YOLOv8s spent 16.66 ms in NPU dispatch and 0.040 ms in native decode and NMS. SESR M7 spent 4.33 ms in NPU dispatch and 1.91 ms in host post-processing of its output.
- **On the webcam:** SESR M7 took 6.51 ms per frame on webcam 0 at 640×480.

### YOLO11n with its attention block

Measured on 2026-09-15 (UTC) in one sitting with ignite-xdna `0be9132` and a container built from it, on `examples/assets/bus.jpg` with 50 warm-up and 500 timed frames per run. Runs alternated AMD's stack and Ignition, then ONNX Runtime's CPU provider, and `xrt-smi` reported no hardware contexts before and after every run. AMD's runs used the Vitis AI loop of Ignition's `benchmarks/benchmark_yolo_vitisai.py`, with Ignition's numpy letterbox and decode. The record is ignite-xdna's `results/aie/yolo11n_hybrid_phoenix_20260915T0216Z.log`.

| Run | Stack | Mean | 99th pct | Where the time goes | Objects |
|---|---|---:|---:|---|---:|
| 1 | AMD Ryzen AI Software 1.7.1 (ONNX Runtime + Vitis AI EP) | 34.49 ms | 37.60 ms | `session.run` 30.91 ms | 7 |
| 2 | Ignition, `yolo11n.ignite` | **11.00 ms** | 11.29 ms | NPU dispatch 8.45 ms, attention block on the CPU 1.69 ms | 6 |
| 3 | AMD Ryzen AI Software 1.7.1 (ONNX Runtime + Vitis AI EP) | 34.37 ms | 37.49 ms | `session.run` 30.74 ms | 7 |
| 4 | Ignition, `yolo11n.ignite` | **11.05 ms** | 11.58 ms | NPU dispatch 8.44 ms, attention block on the CPU 1.69 ms | 6 |
| 5 | ONNX Runtime CPU, through `live_ignition.py` | 31.33 ms | 37.33 ms | `session.run` 27.82 ms | 7 |

- **Why AMD's stack is slow here:** its Vitis AI EP rejects the attention block's 4-D matrix multiplies and places 6 of the model's 1,300 graph nodes on the NPU ([ignite-xdna's notes](https://github.com/jdominick05/ignite-xdna/blob/main/docs/BENCHMARKS.md#stock-yolo11n-with-its-c2psa-attention-block-npu-segments-around-a-host-step-2026-09-15-desktop-2)). In this sitting it was slower than ONNX Runtime's CPU provider alone.
- **Why 6 objects against 7:** Ignition's boxes equal ONNX Runtime's CPU decode of the same NPU input (IoU 1.0). The runs differ in their input. AMD's stack and the CPU run use Ignition's numpy letterbox, and Ignition's NPU path uses its native ingress. Both place the image identically, but 15% of the pixel codes differ by one. On the numpy letterbox, ONNX Runtime's CPU provider finds the classes AMD's stack reported: five people, a bus and a handbag. On the native input it finds four people, a bus and a train. No accuracy was measured.
- **Memory:** resident memory was 203.8 and 204.3 MB for Ignition, unchanged over each run, against 343.1 and 342.8 MB for AMD's stack.

## How a frame runs

```mermaid
flowchart LR
    SRC["webcam, video or image<br/>ThreadedCamera"] --> P["YOLOPipeline"]
    P -- ".ignite" --> ING["native C ingress into the<br/>mapped input buffer"]
    ING --> NPU["NPU Device 0<br/>66 layers on 16 AIE2 cores"]
    NPU --> RB["P3/P4/P5 head readback"]
    P -- ".onnx" --> ORT["letterbox and<br/>ONNX Runtime on the CPU"]
    RB --> POST["DFL decode and<br/>per-class NMS"]
    ORT --> POST
    POST --> DET["detections"]
```

Latency is measured glass to glass: from the moment the loop takes a frame to the moment its detections exist. Drawing and display come after it.

| Stage (native path) | What happens | Mean per frame |
|---|---|---:|
| Ingress | One native C pass (OpenMP) letterboxes, resizes and quantizes the frame straight into the NPU's mapped input buffer | 0.133 ms |
| NPU dispatch | One run of ignite-xdna's graph-engine program computes all 66 layers on the 16 AIE2 cores | 7.122 ms |
| Head readback | The P3, P4 and P5 box and class tensors are read from the output buffer | 0.195 ms |
| Decode and NMS | One native C pass decodes DFL boxes for the anchors that clear the confidence threshold and runs per-class non-maximum suppression on the host | 0.030 ms |

These stage times come from the webcam re-check above: 640×480 frames with 4.62 objects per frame. A larger source costs more at ingress; the 810×1080 `bus.jpg` in the AMD comparison took 0.31 ms.
- **What the NPU time is spent on:** activations move between host memory and the NPU between layers, so most of the dispatch is data movement. A copy of the container with every weight operation switched off still took 5.37 ms of a 7.39 ms dispatch (ignite-xdna `results/model_zoo/dispatch_floor_yolov8n_full.json`).
- **Host segments:** a YOLO11n container runs its attention block on ONNX Runtime's CPU provider between its two NPU dispatches, reading and writing the NPU's workspace buffer, and reports that step as `host`. It calls ONNX Runtime once per frame; a YOLOv8n container never does.
- **The CPU fallback:** an `.onnx` model runs the same steps on ONNX Runtime's CPU execution provider instead.
- **Other tasks:** the diagram shows detection. `SuperResolutionPipeline` runs an `.ignite` container through ignite-xdna's super-resolution pipeline on the NPU, or an `.onnx` model on ONNX Runtime; `ClassificationPipeline` runs on ONNX Runtime only.
- **The camera:** a capture thread (`ThreadedCamera`) owns the webcam so sensor I/O never stalls inference. It tries DirectShow, then Media Foundation, abandons a backend that does not open within `--open-timeout`, and skips empty frames. It counts the frames it reads and the ones that repeat the previous frame. q, ESC, closing the window, Ctrl+C and Ctrl+Break all release the camera and the NPU.

## Limitations

- **One NPU model in a release.** YOLOv8n is the only model the released packages run on the NPU. YOLOv8s, SESR M7 and YOLO11n need ignite-xdna built from its `main`, ResNet50 classification has no `.ignite` lowering, and other ONNX models run on the CPU.
- **A build step.** The `.ignite` container is built from AMD Quark's quantized model with ignite-xdna and the mlir-aie toolchain; it is not a pip install.
- **Narrow hardware support.** Only Phoenix has been verified. Hawk Point is untested, and Strix-class NPUs and Linux are not supported.
- **YOLO11n's attention block runs on the CPU.** It took 1.69 ms of an 11.0 ms frame. No accuracy has been measured for the container, and its borderline boxes change with one-code differences in preprocessing: on `bus.jpg` it finds 6 objects where AMD's stack finds 7.
- **Dim light halves the webcam's frame rate.** The test webcam's auto exposure drops to 15 fps in a dim room; `--exposure-priority off` holds 30 fps with a darker image and fewer detections.
Open work is tracked in [TODO.md](TODO.md).

## Project layout

```
Ignition/
├── live_ignition.py           # launches ignition.live from this checkout
├── src/ignition/
│   ├── __init__.py            # compile(), devices()
│   ├── live.py                # webcam / video / image app: detect, classify or upscale; percentiles, --json
│   ├── suite.py               # ignition suite: one ignition.live process and JSON record per model
│   ├── pipelines/yolo.py      # YOLOPipeline: .ignite on the NPU via ignite-xdna, .onnx on ONNX Runtime
│   ├── pipelines/vision.py    # task inference, ClassificationPipeline, SuperResolutionPipeline
│   ├── pipelines/streaming.py # 3-stage asynchronous runner for .onnx models
│   ├── backends/              # xdna1 layer backend, ONNX Runtime CPU backend
│   ├── model.py, devices.py   # Model API, NPU discovery
│   └── cli/main.py            # ignition devices | run | benchmark | detect | suite
├── examples/                  # yolo_vision_demo.py, quickstart.py, assets/bus.jpg
├── TODO.md                    # completed milestones and open work
└── pyproject.toml             # package ignition-ai, console script `ignition`
```

## License

Ignition is licensed under the [GNU Affero General Public License v3.0 or later](LICENSE).
