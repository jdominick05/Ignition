``` 
██╗ ██████╗ ███╗   ██╗██╗████████╗██╗ ██████╗ ███╗   ██╗
██║██╔════╝ ████╗  ██║██║╚══██╔══╝██║██╔═══██╗████╗  ██║
██║██║  ███╗██╔██╗ ██║██║   ██║   ██║██║   ██║██╔██╗ ██║
██║██║   ██║██║╚██╗██║██║   ██║   ██║██║   ██║██║╚██╗██║
██║╚██████╔╝██║ ╚████║██║   ██║   ██║╚██████╔╝██║ ╚████║
╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝                            
```

**Real-time YOLO on the AMD Ryzen™ AI NPU: YOLOv8n, YOLO11n and YOLOv8n-pose run faster end to end than on AMD's own runtime, YOLOv8n and YOLOv8n-pose more accurately, from a fraction of the install size.**

<!-- badges:start: generated from .github/badges/badges.toml by .github/badges/badges.py -->
![NPU: AMD Phoenix XDNA1](https://img.shields.io/badge/NPU-AMD%20Phoenix%20XDNA1-ed1c24)
[![Latest release](https://img.shields.io/github/v/release/jdominick05/Ignition)](https://github.com/jdominick05/Ignition/releases/latest)
![Verified on Ryzen 7 8700G](https://img.shields.io/badge/verified-Ryzen%207%208700G-blue)
![License: AGPL-3.0-or-later](https://img.shields.io/badge/license-AGPL--3.0--or--later-blue)

[![YOLOv8n end-to-end latency per frame through Ignition and AMD's stack, both runs](https://img.shields.io/badge/dynamic/regex?url=https%3A%2F%2Fraw.githubusercontent.com%2Fjdominick05%2FIgnition%2Fmain%2Fdocs%2FPERFORMANCE.md&search=%5C%7C%20%5C%2A%2AEnd-to-end%20latency%2C%20mean%5C%2A%2A%20%5C%7C%20%5C%2A%2A%28%5B0-9.%5D%2B%29%20%2F%20%28%5B0-9.%5D%2B%29%20ms%5C%2A%2A%20%5C%7C%20%5C%2A%2A%28%5B0-9.%5D%2B%29%20%2F%20%28%5B0-9.%5D%2B%29%20ms&replace=%241%20%2F%20%242%20ms%20vs%20AMD%20%243%20%2F%20%244%20ms&label=YOLOv8n%20per%20frame&color=brightgreen)](docs/PERFORMANCE.md#ignition-vs-amds-ryzen-ai-stack)
[![YOLOv8n box accuracy on all 5,000 COCO val2017 images, the --silu-sigmoid container and AMD's stack on the same input](https://img.shields.io/badge/dynamic/regex?url=https%3A%2F%2Fraw.githubusercontent.com%2Fjdominick05%2FIgnition%2Fmain%2Fdocs%2FPERFORMANCE.md&search=%5C%7C%20%5C%2A%2ADetection%20accuracy%2C%20COCO%20val2017%5C%2A%2A%20%5C%28mAP%4050-95%2C%20all%205%2C000%20images%5C%29%20%5C%7C%20%5C%2A%2A%28%5B0-9.%5D%2B%29%5C%2A%2A%20%5C%7C%20%5C%2A%2A%28%5B0-9.%5D%2B%29%5C%2A%2A%20%5C%7C&replace=%241%20vs%20AMD%20%242%20mAP%4050-95&label=YOLOv8n%20COCO%20boxes&color=blue)](docs/PERFORMANCE.md#ignition-vs-amds-ryzen-ai-stack)
[![YOLO11n latency per frame through Ignition and AMD's stack, both runs](https://img.shields.io/badge/dynamic/regex?url=https%3A%2F%2Fraw.githubusercontent.com%2Fjdominick05%2FIgnition%2Fmain%2Fdocs%2FPERFORMANCE.md&search=%23%23%20YOLO11n%20with%20its%20attention%20block%28%3Fs%3A.%2A%3F%29%5C%7C%201%20%5C%7C%20AMD%5B%5E%7C%5D%2A%5C%7C%20%5C%2A%2A%28%5B0-9.%5D%2B%29%20ms%28%3Fs%3A.%2A%3F%29%5C%7C%203%20%5C%7C%20Ignition%2C%20attention%20core%20on%20the%20CPU%20%5C%7C%20%5C%2A%2A%28%5B0-9.%5D%2B%29%20ms%28%3Fs%3A.%2A%3F%29%5C%7C%204%20%5C%7C%20AMD%5B%5E%7C%5D%2A%5C%7C%20%5C%2A%2A%28%5B0-9.%5D%2B%29%20ms%28%3Fs%3A.%2A%3F%29%5C%7C%206%20%5C%7C%20Ignition%2C%20attention%20core%20on%20the%20CPU%20%5C%7C%20%5C%2A%2A%28%5B0-9.%5D%2B%29%20ms&replace=%242%20%2F%20%244%20ms%20vs%20AMD%20%241%20%2F%20%243%20ms&label=YOLO11n%20per%20frame&color=brightgreen)](docs/PERFORMANCE.md#yolo11n-with-its-attention-block)
[![YOLOv8n-pose latency per frame through Ignition and AMD's stack, both runs](https://img.shields.io/badge/dynamic/regex?url=https%3A%2F%2Fraw.githubusercontent.com%2Fjdominick05%2FIgnition%2Fmain%2Fdocs%2FPERFORMANCE.md&search=%23%23%20YOLOv8n-pose%20on%20the%20NPU%28%3Fs%3A.%2A%3F%29%5C%7C%201%20%5C%7C%20AMD%5B%5E%7C%5D%2A%5C%7C%20%5C%2A%2A%28%5B0-9.%5D%2B%29%20ms%28%3Fs%3A.%2A%3F%29%5C%7C%203%20%5C%7C%20Ignition%20on%20the%20container%20%5C%7C%20%5C%2A%2A%28%5B0-9.%5D%2B%29%20ms%28%3Fs%3A.%2A%3F%29%5C%7C%204%20%5C%7C%20AMD%5B%5E%7C%5D%2A%5C%7C%20%5C%2A%2A%28%5B0-9.%5D%2B%29%20ms%28%3Fs%3A.%2A%3F%29%5C%7C%206%20%5C%7C%20Ignition%20on%20the%20container%20%5C%7C%20%5C%2A%2A%28%5B0-9.%5D%2B%29%20ms&replace=%242%20%2F%20%244%20ms%20vs%20AMD%20%241%20%2F%20%243%20ms&label=YOLOv8n-pose%20per%20frame&color=brightgreen)](docs/PERFORMANCE.md#yolov8n-pose-on-the-npu)
[![YOLOv8n-pose keypoint accuracy on all 5,000 COCO val2017 images, the --silu-sigmoid container and AMD's stack on the same input](https://img.shields.io/badge/dynamic/regex?url=https%3A%2F%2Fraw.githubusercontent.com%2Fjdominick05%2FIgnition%2Fmain%2Fdocs%2FPERFORMANCE.md&search=the%20container%20compiled%20with%20%60--silu-sigmoid%60%20scored%20%28%5B0-9.%5D%2B%29%20OKS%20mAP%4050-95%2C%20and%20AMD%27s%20stack%20%28%5B0-9.%5D%2B%29%20on%20that%20input&replace=%241%20vs%20AMD%20%242%20OKS%20mAP%4050-95&label=YOLOv8n-pose%20COCO%20keypoints&color=blue)](docs/PERFORMANCE.md#yolov8n-pose-on-the-npu)
[![YOLOv8s latency per frame through Ignition early-exit cascade and AMD's stack, both runs](https://img.shields.io/badge/dynamic/regex?url=https%3A%2F%2Fraw.githubusercontent.com%2Fjdominick05%2FIgnition%2Fmain%2Fdocs%2FPERFORMANCE.md&search=%5C%7C%201%20%5C%7C%20AMD%5B%5E%7C%5D%2A%5C%7C%20YOLOv8s%20%5C%7C%20%5C%2A%2A%28%5B0-9.%5D%2B%29%20ms%28%3Fs%3A.%2A%3F%29%5C%7C%202%20%5C%7C%20Ignition%5B%5E%7C%5D%2A%5C%7C%20YOLOv8s%20%5C%7C%20%5C%2A%2A%28%5B0-9.%5D%2B%29%20ms%28%3Fs%3A.%2A%3F%29%5C%7C%203%20%5C%7C%20AMD%5B%5E%7C%5D%2A%5C%7C%20YOLOv8s%20%5C%7C%20%5C%2A%2A%28%5B0-9.%5D%2B%29%20ms%28%3Fs%3A.%2A%3F%29%5C%7C%204%20%5C%7C%20Ignition%5B%5E%7C%5D%2A%5C%7C%20YOLOv8s%20%5C%7C%20%5C%2A%2A%28%5B0-9.%5D%2B%29%20ms&replace=%242%20%2F%20%244%20ms%20vs%20AMD%20%241%20%2F%20%243%20ms&label=YOLOv8s%20per%20frame&color=brightgreen)](docs/PERFORMANCE.md#yolov8s-and-sesr-m7-against-amds-stack)
[![SESR M7 super-resolution latency per frame through Ignition and AMD's stack, both runs; AMD's stack is faster](https://img.shields.io/badge/dynamic/regex?url=https%3A%2F%2Fraw.githubusercontent.com%2Fjdominick05%2FIgnition%2Fmain%2Fdocs%2FPERFORMANCE.md&search=%5C%7C%205%20%5C%7C%20AMD%5B%5E%7C%5D%2A%5C%7C%20SESR%20M7%20%5C%7C%20%5C%2A%2A%28%5B0-9.%5D%2B%29%20ms%28%3Fs%3A.%2A%3F%29%5C%7C%206%20%5C%7C%20Ignition%5B%5E%7C%5D%2A%5C%7C%20SESR%20M7%20%5C%7C%20%5C%2A%2A%28%5B0-9.%5D%2B%29%20ms%28%3Fs%3A.%2A%3F%29%5C%7C%207%20%5C%7C%20AMD%5B%5E%7C%5D%2A%5C%7C%20SESR%20M7%20%5C%7C%20%5C%2A%2A%28%5B0-9.%5D%2B%29%20ms%28%3Fs%3A.%2A%3F%29%5C%7C%208%20%5C%7C%20Ignition%5B%5E%7C%5D%2A%5C%7C%20SESR%20M7%20%5C%7C%20%5C%2A%2A%28%5B0-9.%5D%2B%29%20ms&replace=%242%20%2F%20%244%20ms%20vs%20AMD%20%241%20%2F%20%243%20ms&label=SESR%20M7%20per%20frame&color=orange)](docs/PERFORMANCE.md#yolov8s-and-sesr-m7-against-amds-stack)
[![ResNet50 classification head latency per frame on the NPU through the engine and AMD's stack, both runs; AMD's stack crashes trying to place it on the NPU](https://img.shields.io/badge/dynamic/regex?url=https%3A%2F%2Fraw.githubusercontent.com%2Fjdominick05%2FIgnition%2Fmain%2Fdocs%2FPERFORMANCE.md&search=%5C%7C%201%20%5C%7C%20AMD%5B%5E%7C%5D%2A%5C%7C%20ResNet50%20head%2C%20NPU%20%5C%7C%20%5C%2A%2A%28%5Ba-z%5D%2B%29%5C%2A%2A%28%3Fs%3A.%2A%3F%29%5C%7C%203%20%5C%7C%20Ignition%5B%5E%7C%5D%2A%5C%7C%20ResNet50%20head%2C%20NPU%20%5C%7C%20%5C%2A%2A%28%5B0-9.%5D%2B%29%20ms%28%3Fs%3A.%2A%3F%29%5C%7C%205%20%5C%7C%20Ignition%5B%5E%7C%5D%2A%5C%7C%20ResNet50%20head%2C%20NPU%20%5C%7C%20%5C%2A%2A%28%5B0-9.%5D%2B%29%20ms&replace=%242%20%2F%20%243%20ms%20vs%20AMD%20%241&label=classification%20head&color=brightgreen)](docs/PERFORMANCE.md#classification-head-against-amds-stack)
[![Resident memory through Ignition and AMD's stack, both runs](https://img.shields.io/badge/dynamic/regex?url=https%3A%2F%2Fraw.githubusercontent.com%2Fjdominick05%2FIgnition%2Fmain%2Fdocs%2FPERFORMANCE.md&search=%5C%7C%20%5C%2A%2AProcess%20memory%20%5C%28RSS%5C%29%5C%2A%2A%20%5C%7C%20%5C%2A%2A%28%5B0-9.%5D%2B%29%20%2F%20%28%5B0-9.%5D%2B%29%20MB%5C%2A%2A%20%5C%7C%20%5C%2A%2A%28%5B0-9.%5D%2B%29%20%2F%20%28%5B0-9.%5D%2B%29%20MB&replace=%241%20%2F%20%242%20MB%20vs%20AMD%20%243%20%2F%20%244%20MB&label=memory%20%28RSS%29&color=blue)](docs/PERFORMANCE.md#ignition-vs-amds-ryzen-ai-stack)
[![Runtime install size on disk, Ignition and AMD's stack](https://img.shields.io/badge/dynamic/regex?url=https%3A%2F%2Fraw.githubusercontent.com%2Fjdominick05%2FIgnition%2Fmain%2Fdocs%2FPERFORMANCE.md&search=%5C%7C%20%5C%2A%2ARuntime%20install%20on%20disk%5C%2A%2A%20%5C%7C%20%5C%2A%2A%28%5B%5E0-9%7C%20%2A%5D%2A%29%28%5B0-9%2C%5D%2B%29%20MB%5C%2A%2A%20%5C%7C%20%5C%2A%2A%28%5B%5E0-9%7C%20%2A%5D%2A%29%28%5B0-9%2C%5D%2B%29%20MB&replace=%241%242%20MB%20vs%20AMD%20%243%244%20MB&label=runtime%20install&color=blue)](docs/PERFORMANCE.md#ignition-vs-amds-ryzen-ai-stack)
[![YOLO11n energy per frame at full speed, Ignition's default balanced power mode and AMD's stack, both runs](https://img.shields.io/badge/dynamic/regex?url=https%3A%2F%2Fraw.githubusercontent.com%2Fjdominick05%2FIgnition%2Fmain%2Fdocs%2FPERFORMANCE.md&search=%5C%7C%20YOLO11n%20%5C%7C%20AMD%5B%5E%7C%5D%2A%5C%7C%5B%5E%7C%5D%2A%5C%7C%5B%5E%7C%5D%2A%5C%7C%20%5C%2A%2A%28%5B0-9.%2C%5D%2B%29%20%2F%20%28%5B0-9.%2C%5D%2B%29%20mJ%28%3Fs%3A.%2A%3F%29%5C%7C%20YOLO11n%20%5C%7C%20Ignition%5B%5E%7C%5D%2A%5C%7C%20balanced%20%5C%7C%5B%5E%7C%5D%2A%5C%7C%20%5C%2A%2A%28%5B0-9.%2C%5D%2B%29%20%2F%20%28%5B0-9.%2C%5D%2B%29%20mJ&replace=%243%20%2F%20%244%20mJ%20vs%20AMD%20%241%20%2F%20%242%20mJ&label=YOLO11n%20energy%20per%20frame&color=brightgreen)](docs/PERFORMANCE.md#energy-per-frame-against-amds-stack)
[![YOLOv8n energy per frame at 30 frames per second, Ignition's default balanced power mode and AMD's stack, both runs](https://img.shields.io/badge/dynamic/regex?url=https%3A%2F%2Fraw.githubusercontent.com%2Fjdominick05%2FIgnition%2Fmain%2Fdocs%2FPERFORMANCE.md&search=%5C%7C%201%20%5C%7C%20AMD%5B%5E%7C%5D%2A%5C%7C%5B%5E%7C%5D%2A%5C%7C%20%5B0-9.%5D%2B%20ms%20%5C%7C%20%5B0-9.%5D%2B%20%25%20%5C%7C%20%5B0-9.%5D%2B%20W%20%5C%7C%20%5C%2A%2A%28%5B0-9.%2C%5D%2B%29%20mJ%28%3Fs%3A.%2A%3F%29%5C%7C%203%20%5C%7C%20Ignition%5B%5E%7C%5D%2A%5C%7C%20balanced%20%5C%7C%20%5B0-9.%5D%2B%20ms%20%5C%7C%20%5B0-9.%5D%2B%20%25%20%5C%7C%20%5B0-9.%5D%2B%20W%20%5C%7C%20%5C%2A%2A%28%5B0-9.%2C%5D%2B%29%20mJ%28%3Fs%3A.%2A%3F%29%5C%7C%205%20%5C%7C%20AMD%5B%5E%7C%5D%2A%5C%7C%5B%5E%7C%5D%2A%5C%7C%20%5B0-9.%5D%2B%20ms%20%5C%7C%20%5B0-9.%5D%2B%20%25%20%5C%7C%20%5B0-9.%5D%2B%20W%20%5C%7C%20%5C%2A%2A%28%5B0-9.%2C%5D%2B%29%20mJ%28%3Fs%3A.%2A%3F%29%5C%7C%207%20%5C%7C%20Ignition%5B%5E%7C%5D%2A%5C%7C%20balanced%20%5C%7C%20%5B0-9.%5D%2B%20ms%20%5C%7C%20%5B0-9.%5D%2B%20%25%20%5C%7C%20%5B0-9.%5D%2B%20W%20%5C%7C%20%5C%2A%2A%28%5B0-9.%2C%5D%2B%29%20mJ&replace=%242%20%2F%20%244%20mJ%20vs%20AMD%20%241%20%2F%20%243%20mJ&label=YOLOv8n%20energy%20at%2030%20fps&color=brightgreen)](docs/PERFORMANCE.md#energy-per-frame-against-amds-stack)
<!-- badges:end -->

Ignition compiles a YOLO model into an `.ignite` container that runs every layer on the NPU of a Windows 11 PC with an
AMD Phoenix processor.

## 1. Install

Paste into PowerShell, and again later to update:

```powershell
irm https://raw.githubusercontent.com/jdominick05/Ignition/main/install.ps1 | iex
```

It sets up everything under `%LOCALAPPDATA%\Ignition`. To also install AMD's NPU driver:

```powershell
& ([scriptblock]::Create((irm https://raw.githubusercontent.com/jdominick05/Ignition/main/install.ps1))) -InstallDriver
```

## 2. Get a model

Download YOLOv8n from Hugging Face, export it and cut its decode head, which runs on the CPU. Containers expect YOLOv8's
80 COCO classes, so use the published weights rather than a model retrained on other labels:

```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force; . "$env:LOCALAPPDATA\Ignition\ignition-models.ps1"
hf download Ultralytics/YOLOv8 yolov8n.pt --local-dir models
python pipelines/yolov8n/1_export.py
python pipelines/yolov8n/1b_cut_head.py
```

## 3. Quantize

The NPU runs 8-bit integers. Quark calibrates on sample images; COCO's first 128 work:

```powershell
iwr https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip -OutFile coco128.zip
Expand-Archive coco128.zip data
python pipelines/yolov8n/3b_quantize_cut.py --calib-dir data/coco128/images/train2017
```

On the test PC this took 5 minutes and 13.5 GB of temporary disk. Quark's warnings that it cannot build custom ops
are harmless. Calibration images shape the result, so your detections can differ from the benchmarks'.

## 4. Compile and run

In the same window:

```powershell
. "$env:LOCALAPPDATA\Ignition\ignition-env.ps1"
ignite-compile --input ..\ignite-xdna\models\yolov8n_cut_xint8.onnx --output ..\ignite-xdna\build\yolov8n_full.ignite --silu-sigmoid
python live_ignition.py
```

`--silu-sigmoid` approximates SiLU more closely than AMD's quantizer: 34.12 against 26.68 COCO mAP, for 0.24 ms
per frame. A window opens on webcam 0 with boxes, labels and latency; press q to quit. Add
`--source picture.jpg` for an image.

## More

- [Performance against AMD's stack](docs/PERFORMANCE.md)
- [Other models, flags, Python API and limitations](docs/USAGE.md)
- [Designing and quantizing models for this NPU](docs/QUANTIZATION-GUIDE.md)
- [Open work](TODO.md)

Licensed under [AGPL-3.0-or-later](LICENSE).
