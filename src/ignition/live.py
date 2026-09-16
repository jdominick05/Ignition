#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: AGPL-3.0-or-later
r"""Live inference on the AMD Phoenix NPU (XDNA1) or ONNX Runtime from a webcam, video or image.

    python live_ignition.py                                # YOLOv8n on webcam 0 in a window
    python live_ignition.py --headless --frames 300        # benchmark: G2G mean, P50, P95, P99
    python live_ignition.py --source clip.mp4 --model ..\ignite-xdna\models\yolov8n_cut_xint8.onnx
    python live_ignition.py --model ..\ignite-xdna\models\sesr_m7_xint8.onnx --headless --frames 300 --json sesr.json

Run it in the mlir-aie-iron conda environment, where pyxrt loads.

This module is the app. In a checkout, live_ignition.py at the root launches it on the checkout's
own src; an installed package runs it as `python -m ignition.live`. `ignition suite` runs it once
per model (one process each) and keeps each --json record.

The model picks the backend and the task. A bare-metal .ignite container runs the
whole network on NPU Device 0 through ignite-xdna; ONNX Runtime is not called. An
.onnx model runs on ONNX Runtime (CPU). The task comes from the container
manifest or the ONNX output shapes (--task overrides it):
  detect            YOLO (six head-cut outputs or one (1, 84, N) tensor): boxes, labels, scores
  classify          one (1, N) output, e.g. ResNet50: timm eval preprocessing, top-5
  pose              YOLOv8-pose (nine head-cut outputs): people, each with 17 keypoints
  super_resolution  one image output a whole multiple of the input size, e.g. SESR M7 (2x)
The default model is ignite-xdna's graph-engine container build\yolov8n_full.ignite
when it exists, otherwise build\yolov8n.ignite, which carries no detect heads and so
draws no boxes (a warning says so).

Glass-to-glass (G2G) is timed from the moment the loop takes a frame to the moment
its output exists: preprocessing, the network, and decoding (NMS, top-k, or the
upscaled image). Drawing and display come after it. A capture thread owns the camera
(ThreadedCamera), so USB sensor I/O never stalls the inference loop: the loop runs on
the newest frame, or with --fresh waits for each new one.

--json PATH writes the run's summary (latency percentiles, stage means, RSS drift,
host) as one JSON object, for tools that compare models.

--power-mode chooses how an .ignite container's host threads trade CPU power for speed,
sized to this machine's own cores: performance keeps them spinning on every logical
processor, balanced (the default) lets them sleep between frames with one per physical
core, and efficiency sleeps them with a quarter of the physical cores. --max-fps caps the
processing rate, waiting between frames the way a camera would.

Stop with q or ESC in the window, by closing the window, or with Ctrl+C /
Ctrl+Break. Every path releases the camera, closes the window and releases the
NPU hardware context, then exits 0.
"""
import os

# OpenCV reads this when video I/O initialises, so it must precede every cv2 import (including
# the one inside ignition). Without it the Media Foundation backend can take ~90 s to open.
# `python -m ignition.live` imports the ignition package, and so cv2, before this line runs:
# live_ignition.py and `ignition suite` set it before Python starts.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import argparse
import json
import logging
import platform
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# The checkout holding src/ignition/live.py, for default_model's sibling ignite-xdna fallback.
CHECKOUT = Path(__file__).resolve().parents[2]

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ignition.pipelines.vision import (  # noqa: E402
    TASK_CLASSIFY, TASK_DETECT, TASK_POSE, TASK_SUPER_RESOLUTION, TASKS, create_pipeline, draw_poses, infer_task)
from ignition.pipelines.yolo import draw_detections, is_ignite_container  # noqa: E402

try:
    import psutil
except ImportError:  # the ironenv venv has no psutil; rss_mb falls back to the Win32 API
    psutil = None

WINDOW_NAME = "Ignition - AMD Phoenix NPU"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
Frame = Tuple[Optional[np.ndarray], int, float]
CAMERA_BACKENDS = ("DSHOW", "MSMF", "ANY")


def _frame_sample(frame: np.ndarray) -> bytes:
    """Every 16th pixel of every 16th row: sensor noise makes a new frame differ from the last one here."""
    return frame[::16, ::16].tobytes()


def _fourcc_text(value: float) -> str:
    code = int(value)
    text = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))
    return text if code > 0 and text.isprintable() and text.strip() else str(code)


def _set_exposure_priority(index: int, value: int, log: Callable[[str], None]) -> Optional[int]:
    """Sets DirectShow camera ``index``'s exposure auto priority; returns the value to put back, or None."""
    try:
        from ignition._camera_controls import ExposurePriority
        with ExposurePriority(index) as control:
            previous = control.get()
            control.set(value)
            effect = "frame rate held in dim light" if value == 0 else "auto exposure may lower the frame rate"
            log(f"[camera] {control.name}: exposure auto priority {previous} -> {control.get()} for this run "
                f"({effect}); the camera keeps this setting until it is put back on exit")
        return previous
    except Exception as exc:  # noqa: BLE001 - the control is optional: run at the camera's own setting
        log(f"[camera] exposure auto priority left unchanged: {type(exc).__name__}: {exc}")
        return None


# -- sources -----------------------------------------------------------------------
class ThreadedCamera:
    """Webcam reader on its own thread; the inference loop takes frames without blocking on USB I/O.

    ``cv2.VideoCapture.read`` blocks for a whole sensor period (33-66 ms). The
    capture thread absorbs that wait and publishes each frame with a sequence
    number and its arrival time. Backends are tried DirectShow first, then Media
    Foundation, then OpenCV's default. Each open runs in a helper thread and is
    abandoned after ``open_timeout_s``, because DirectShow can block forever on an
    IR sensor.

    It counts the frames it reads and the ones that repeat the previous frame
    (compared on a 1-in-16 pixel sample), because a backend's reported FPS is not
    what arrives: on the test webcam DirectShow reports FPS -1, and Media
    Foundation reads at 30 fps partly by returning repeated frames.

    With ``exposure_priority`` (0 or 1) it sets the camera's exposure auto priority
    through DirectShow before opening, and puts the previous value back in
    ``release()`` (or at once if the open fails or another backend opens the camera).
    """

    kind = "camera"

    def __init__(self, index: int, open_timeout_s: float = 8.0, log: Callable[[str], None] = print,
                 backends: Optional[Sequence[str]] = None, exposure_priority: Optional[int] = None):
        self.index = index
        self.finished = False
        self.read_failures = 0
        self._log = log
        self._priority_to_restore: Optional[int] = None
        if exposure_priority is not None:
            self._priority_to_restore = _set_exposure_priority(index, exposure_priority, log)
        try:
            self.backend, self._cap, first = self._open(index, open_timeout_s, log, backends)
        except Exception:
            self._restore_exposure_priority()
            raise
        if self._priority_to_restore is not None and self.backend != "DSHOW":
            log(f"[camera] exposure auto priority is set by DirectShow device index, but {self.backend} opened "
                "the camera: putting it back")
            self._restore_exposure_priority()
        self.label = f"camera {index} via {self.backend} {first.shape[1]}x{first.shape[0]}"
        self._cond = threading.Condition()
        self._frame: Optional[np.ndarray] = first
        self._seq = 1
        self._stamp = self._first_stamp = time.perf_counter()
        self._sample = _frame_sample(first)
        self.frames, self.repeats = 1, 0
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="ignition-camera", daemon=True)
        self._thread.start()

    @staticmethod
    def _backends(names: Optional[Sequence[str]] = None) -> List[Tuple[str, int]]:
        return [(name, int(getattr(cv2, f"CAP_{name}"))) for name in (names or CAMERA_BACKENDS)
                if hasattr(cv2, f"CAP_{name}")]

    @staticmethod
    def _open(index: int, timeout_s: float, log: Callable[[str], None], names: Optional[Sequence[str]] = None):
        attempts = []
        for name, api in ThreadedCamera._backends(names):
            box = {}

            def worker(box=box, api=api):
                try:
                    cap = cv2.VideoCapture(index, api)
                    box["cap"] = cap
                    if cap.isOpened():
                        ok, frame = cap.read()
                        if ok and frame is not None and frame.size > 0:
                            box["frame"] = frame
                except cv2.error as exc:
                    box["error"] = exc

            t0 = time.perf_counter()
            opener = threading.Thread(target=worker, name=f"ignition-camera-open-{name}", daemon=True)
            opener.start()
            opener.join(timeout_s)
            dt = time.perf_counter() - t0
            if opener.is_alive():
                outcome = f"no answer after {timeout_s:.0f} s, abandoned"
            elif "frame" in box:
                frame = box["frame"]
                cap = box["cap"]
                log(f"[camera] index {index} via {name}: opened, {frame.shape[1]}x{frame.shape[0]} ({dt:.2f} s); "
                    f"backend reports FOURCC {_fourcc_text(cap.get(cv2.CAP_PROP_FOURCC))}, "
                    f"FPS {cap.get(cv2.CAP_PROP_FPS):g}, AUTO_EXPOSURE {cap.get(cv2.CAP_PROP_AUTO_EXPOSURE):g} "
                    "(unverified: the [summary] camera line has the measured rate)")
                return name, cap, frame
            else:
                outcome = f"error: {box['error']}" if "error" in box else "no frame"
                if box.get("cap") is not None:
                    box["cap"].release()
            attempts.append(f"{name} {outcome}")
            log(f"[camera] index {index} via {name}: {outcome} ({dt:.2f} s)")
        raise RuntimeError(f"camera {index} delivered no frame ({'; '.join(attempts)})")

    def _loop(self) -> None:
        while self._running:
            try:
                ok, frame = self._cap.read()
            except cv2.error:
                ok, frame = False, None
            if not ok or frame is None or frame.size == 0:
                self.read_failures += 1
                time.sleep(0.005)
                continue
            stamp = time.perf_counter()
            sample = _frame_sample(frame)
            with self._cond:
                self._frame, self._stamp = frame, stamp
                self._seq += 1
                self.frames += 1
                self.repeats += sample == self._sample
                self._sample = sample
                self._cond.notify_all()

    def read(self, after_seq: int = 0, timeout_s: float = 1.0) -> Frame:
        """The newest frame as ``(frame, seq, arrival_time)``.

        Returns at once when a frame newer than ``after_seq`` exists (``after_seq=0``
        accepts any frame); otherwise waits up to ``timeout_s`` and returns
        ``(None, after_seq, 0.0)``. Each frame is a fresh array the capture thread
        never writes again, so the caller may keep it without copying.
        """
        with self._cond:
            if self._seq <= after_seq:
                self._cond.wait_for(lambda: self._seq > after_seq or not self._running, timeout_s)
            if self._seq <= after_seq or self._frame is None:
                return None, after_seq, 0.0
            return self._frame, self._seq, self._stamp

    def rates(self) -> Tuple[int, int, float]:
        """``(frames, repeats, span_s)``: frames read since the open, how many repeated the previous frame, and
        the seconds from the first frame to the latest."""
        with self._cond:
            return self.frames, self.repeats, self._stamp - self._first_stamp

    def _restore_exposure_priority(self) -> None:
        previous, self._priority_to_restore = self._priority_to_restore, None
        if previous is None:
            return
        try:
            from ignition._camera_controls import ExposurePriority
            with ExposurePriority(self.index) as control:
                control.set(previous)
                self._log(f"[camera] exposure auto priority put back to {control.get()}")
        except Exception as exc:  # noqa: BLE001 - report it; shutdown must continue
            self._log(f"[camera] could not put exposure auto priority back to {previous}: "
                      f"{type(exc).__name__}: {exc}")

    def release(self) -> None:
        self._running = False
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=2.0)
        self._cap.release()
        self._restore_exposure_priority()


class FileSource:
    """A video file (frames in order, then ``finished``) or a still image (the same frame each read)."""

    def __init__(self, path: Path):
        self.kind = "image" if path.suffix.lower() in IMAGE_SUFFIXES else "video"
        self.finished = False
        self.read_failures = 0
        self._seq = 0
        self._cap = None
        self._image = None
        if self.kind == "image":
            self._image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if self._image is None or self._image.size == 0:
                raise RuntimeError(f"OpenCV could not decode image {path}")
            self.label = f"image {path.name} {self._image.shape[1]}x{self._image.shape[0]}"
        else:
            self._cap = cv2.VideoCapture(str(path))
            if not self._cap.isOpened():
                raise RuntimeError(f"OpenCV could not open video {path}")
            self.label = f"video {path.name}"

    def read(self, after_seq: int = 0, timeout_s: float = 1.0) -> Frame:
        if self._image is not None:
            return self._image, 1, time.perf_counter()
        ok, frame = self._cap.read()
        if not ok or frame is None or frame.size == 0:
            self.finished = True
            return None, after_seq, 0.0
        self._seq += 1
        return frame, self._seq, time.perf_counter()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()


def open_source(spec: str, open_timeout_s: float, camera_backend: str = "auto", exposure_priority: str = "keep"):
    text = spec.strip()
    if text.isdigit():
        names = None if camera_backend == "auto" else [camera_backend.upper()]
        priority = {"keep": None, "off": 0, "on": 1}[exposure_priority]
        return ThreadedCamera(int(text), open_timeout_s, backends=names, exposure_priority=priority)
    path = Path(text)
    if not path.is_file():
        raise FileNotFoundError(f"{path} is neither a webcam index nor an existing file")
    return FileSource(path)


# -- helpers -----------------------------------------------------------------------
def default_model() -> Path:
    """ignite-xdna's graph-engine container if present, else its legacy container."""
    roots: List[Path] = []
    try:
        from ignite_xdna.runtime.driver import get_repo_root
        roots.append(get_repo_root())
    except Exception:  # noqa: BLE001 - ignite-xdna not importable: try the sibling checkout
        pass
    roots.append(CHECKOUT.parent / "ignite-xdna")
    for root in roots:
        for name in ("yolov8n_full.ignite", "yolov8n.ignite"):
            candidate = root / "build" / name
            if candidate.is_file():
                return candidate
    return roots[0] / "build" / "yolov8n.ignite"


def rss_mb() -> float:
    """Resident set of this process in MB (the working set on Windows)."""
    if psutil is not None:
        return psutil.Process().memory_info().rss / 2 ** 20
    if sys.platform == "win32":
        import ctypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        kernel32 = ctypes.WinDLL("kernel32")
        psapi = ctypes.WinDLL("psapi")
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p  # a 64-bit handle; the default int truncates
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_uint32]
        pmc = Counters()
        pmc.cb = ctypes.sizeof(Counters)
        if psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return pmc.WorkingSetSize / 2 ** 20
    return float("nan")


class Samples:
    """Fixed-capacity sample buffer (a ring once full), so a long live run never grows memory."""

    def __init__(self, capacity: int):
        self._buf = np.zeros(max(1, capacity), dtype=np.float64)
        self.count = 0

    def add(self, value: float) -> None:
        self._buf[self.count % self._buf.size] = value
        self.count += 1

    def values(self) -> np.ndarray:
        return self._buf[:min(self.count, self._buf.size)]

    def mean(self) -> float:
        v = self.values()
        return float(v.mean()) if v.size else float("nan")


class StopRequest:
    """Turns Ctrl+C (SIGINT) and Ctrl+Break (SIGBREAK) into a flag the loop checks once per frame."""

    def __init__(self):
        self.reason: Optional[str] = None

    @property
    def requested(self) -> bool:
        return self.reason is not None

    def _handle(self, signum, _frame) -> None:
        self.reason = signal.Signals(signum).name

    def install(self) -> None:
        if sys.platform == "win32":
            # A process started in a new process group (some IDEs, launchers and job runners do
            # this) inherits "ignore Ctrl+C" and never sees SIGINT; turn Ctrl+C handling back on.
            import ctypes
            ctypes.WinDLL("kernel32").SetConsoleCtrlHandler(None, False)
        for name in ("SIGINT", "SIGBREAK", "SIGTERM"):
            sig = getattr(signal, name, None)
            if sig is not None:
                signal.signal(sig, self._handle)


def output_source(task: str, result: Any) -> str:
    """Where the frame's output came from: ``npu``, ``onnxruntime`` or ``none``."""
    return (result.head_source if task == TASK_DETECT else result.source) or "none"


def output_count(task: str, result: Any) -> str:
    if task == TASK_DETECT:
        return f"{len(result.detections)} objects"
    if task == TASK_CLASSIFY:
        top = result.topk[0]
        return f"top-1 class {top.class_id} ({top.score:.2f})"
    if task == TASK_POSE:
        return f"{len(result.people)} people"
    return f"{result.image.shape[1]}x{result.image.shape[0]} output"


def render(frame: np.ndarray, display: Optional[np.ndarray], task: str, result: Any, fps: float,
           backend: str, source: str) -> np.ndarray:
    """The image to show: boxes or skeletons on the frame, the top-5 over the frame, or the upscaled frame; plus
    the HUD."""
    if task == TASK_SUPER_RESOLUTION:
        canvas = result.image.copy()
    else:
        if display is None or display.shape != frame.shape:
            display = np.empty_like(frame)
        np.copyto(display, frame)  # draw on a copy: the next loop may reuse this frame
        canvas = display
    t = result.timings_ms
    stage = "ORT" if output_source(task, result) == "onnxruntime" else "NPU"
    lines = [
        f"G2G {t.get('g2g_ms', t['total_ms']):5.2f} ms  {stage} {t.get('backbone_ms', 0.0):5.2f} ms  "
        f"{fps:5.1f} FPS  {output_count(task, result)}",
        f"{backend} | {task}: {output_source(task, result)} | {source}",
    ]
    if task == TASK_DETECT:
        draw_detections(canvas, result.detections, inplace=True)
    elif task == TASK_POSE:
        draw_poses(canvas, result.people, inplace=True)
    elif task == TASK_CLASSIFY:
        lines += [f"class {c.class_id:4d}  p={c.score:.3f}" for c in result.topk]
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 12 + 20 * len(lines)), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        cv2.putText(canvas, text, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0) if i == 0 else (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, "q / ESC: quit", (8, canvas.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


def host_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"hostname": platform.node(), "processor": platform.processor(),
                            "python": platform.python_version(), "numpy": np.__version__,
                            "opencv": cv2.__version__, "cpu_count": os.cpu_count()}
    try:
        import onnxruntime as ort
        info["onnxruntime"] = ort.__version__
    except ImportError:
        pass
    return info


# -- main --------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Live inference on the AMD Phoenix NPU (XDNA1) or ONNX Runtime.")
    ap.add_argument("--model", default=None,
                    help=r"an .ignite container (NPU) or .onnx model (ONNX Runtime); default "
                         r"..\ignite-xdna\build\yolov8n_full.ignite if present, else ..\ignite-xdna\build\yolov8n.ignite")
    ap.add_argument("--task", choices=("auto",) + TASKS, default="auto",
                    help="what the model computes (default auto: from the manifest or the ONNX outputs)")
    ap.add_argument("--source", default="0", help="webcam index (0, 1, ...) or a video or image file (default 0)")
    ap.add_argument("--headless", action="store_true", help="no window: run the loop and print progress lines")
    ap.add_argument("--frames", type=int, default=0,
                    help="stop after N timed frames and report G2G mean, P50, P95, P99 (default 0: until stopped)")
    ap.add_argument("--warmup", type=int, default=10, help="untimed frames before the timed ones (default 10)")
    ap.add_argument("--fresh", action="store_true",
                    help="wait for a new camera frame before each inference instead of reusing the newest one")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold for detections and people (default 0.25)")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold for detections and people (default 0.45)")
    ap.add_argument("--open-timeout", type=float, default=8.0, help="seconds allowed per camera backend open")
    ap.add_argument("--camera-backend", choices=["auto", "dshow", "msmf", "any"], default="auto",
                    help="OpenCV capture backend for a webcam; auto tries DirectShow, then Media Foundation, "
                         "then OpenCV's default (default auto)")
    ap.add_argument("--exposure-priority", choices=["keep", "off", "on"], default="keep",
                    help="the webcam's exposure auto priority for this run, put back on exit (DirectShow): off "
                         "holds the frame rate in dim light with a darker image, on lets auto exposure lower it "
                         "(default keep: leave the camera's setting)")
    ap.add_argument("--power-mode", choices=POWER_MODES, default=None,
                    help="for .ignite containers, how host threads trade CPU power for speed, sized to this machine: "
                         "performance spins them on every logical processor, balanced sleeps them between frames with "
                         "one per physical core, efficiency sleeps them with a quarter of the physical cores "
                         "(default: IGNITE_XDNA_POWER_MODE if set, else balanced)")
    ap.add_argument("--max-fps", type=float, default=0.0,
                    help="process at most this many frames per second, waiting between frames; the wait is not part "
                         "of G2G (default 0: as fast as the source and model allow)")
    ap.add_argument("--json", default=None, help="write the run summary to this JSON file")
    return ap.parse_args(argv)


POWER_MODES = ("efficiency", "balanced", "performance")


def select_power_mode(mode: Optional[str]) -> None:
    """Hand --power-mode to ignite-xdna. It must be in the environment before ignite_xdna is first imported, because
    importing it loads the native preprocessor, whose OpenMP runtime reads its settings once."""
    if mode:
        os.environ["IGNITE_XDNA_POWER_MODE"] = mode


def power_settings() -> Optional[Dict[str, Any]]:
    """The power mode ignite-xdna's native preprocessor loaded with; None before ignite-xdna had power modes."""
    try:
        from ignite_xdna.pipelines import power
    except ImportError:
        return None
    settings = power.current_settings()
    return None if settings is None else {**settings.as_dict(), "description": settings.describe()}


def main(argv=None) -> int:
    args = parse_args(argv)
    select_power_mode(args.power_mode)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
    stop = StopRequest()
    stop.install()

    model = Path(args.model) if args.model else default_model()
    if not model.is_file():
        print(f"[Ignition] model not found: {model}", flush=True)
        return 2
    native = is_ignite_container(model)
    print(f"[Ignition] loading {model} ({'bare-metal .ignite container' if native else 'ONNX model'})", flush=True)
    try:
        task = infer_task(model) if args.task == "auto" else args.task
        task, pipeline = create_pipeline(model, task=task, conf_thres=args.conf, iou_thres=args.iou, device_id=0)
    except Exception as exc:  # noqa: BLE001 - report and exit non-zero
        print(f"[Ignition] could not load {model}: {type(exc).__name__}: {exc}", flush=True)
        return 2
    backend = "NPU Device 0" if native else "ONNX Runtime CPU"
    print(f"[Ignition] task: {task}", flush=True)
    if not native:
        print("[Ignition] ONNX Runtime CPU backend active", flush=True)
    power = power_settings() if native else None
    if power:
        print(f"[Ignition] power mode {power['description']}", flush=True)

    try:
        source = open_source(args.source, args.open_timeout, args.camera_backend, args.exposure_priority)
    except Exception as exc:  # noqa: BLE001 - report and exit non-zero
        pipeline.close()
        print(f"[Ignition] could not open source {args.source!r}: {exc}", flush=True)
        return 2
    print(f"[Ignition] source: {source.label}", flush=True)

    capacity = args.frames if args.frames > 0 else 10_000
    g2g, pre, net, dispatch, host, readback, post, age = (Samples(capacity) for _ in range(8))
    frame_limit = args.warmup + args.frames if args.frames > 0 else 0
    processed = timed = unique = npu_frames = detections = people = 0
    top1_counts: Dict[int, int] = {}
    output_shape: Optional[List[int]] = None
    last_seq = 0
    cold_ms = float("nan")
    rss_first = rss_last = float("nan")
    fps, t_prev = 0.0, None
    display: Optional[np.ndarray] = None
    stop_reason = "frame limit"

    window = not args.headless
    if window:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    period = 1.0 / args.max_fps if args.max_fps > 0 else 0.0
    t_schedule: Optional[float] = None
    try:
        while True:
            if stop.requested:
                stop_reason = stop.reason
                break
            if frame_limit and processed >= frame_limit:
                break
            if period:
                # An absolute schedule: a late frame does not make the next one early, and the rate averages max_fps.
                if t_schedule is None:
                    t_schedule = time.perf_counter()
                delay = t_schedule + processed * period - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            frame, seq, arrival = source.read(after_seq=last_seq if args.fresh else 0, timeout_s=1.0)
            if frame is None or frame.size == 0:
                if source.finished:
                    stop_reason = "end of video"
                    break
                continue
            if seq != last_seq:
                unique += 1
                last_seq = seq

            result = pipeline.predict(frame, annotate=False) if task == TASK_DETECT else pipeline.predict(frame)
            t_done = time.perf_counter()
            t = result.timings_ms
            processed += 1
            if processed == 1:
                cold_ms = t["total_ms"]
            if processed > args.warmup:
                timed += 1
                if timed == 1:
                    rss_first = rss_mb()
                g2g.add(t.get("g2g_ms", t["total_ms"]))
                pre.add(t["preprocess_ms"])
                net.add(t["backbone_ms"])
                post.add(t["postprocess_ms"])
                if "dispatch_ms" in t:
                    dispatch.add(t["dispatch_ms"])
                    readback.add(t["readback_ms"])
                if "host_ms" in t:
                    host.add(t["host_ms"])
                if args.fresh and source.kind == "camera":
                    age.add((t_done - arrival) * 1000.0)
                npu_frames += output_source(task, result) == "npu"
                if task == TASK_DETECT:
                    detections += len(result.detections)
                elif task == TASK_CLASSIFY:
                    top1_counts[result.topk[0].class_id] = top1_counts.get(result.topk[0].class_id, 0) + 1
                elif task == TASK_POSE:
                    people += len(result.people)
                else:
                    output_shape = [int(v) for v in result.image.shape]

            if t_prev is not None:
                rate = 1.0 / max(t_done - t_prev, 1e-6)
                fps = rate if fps == 0.0 else 0.9 * fps + 0.1 * rate
            t_prev = t_done

            if window:
                canvas = render(frame, display, task, result, fps, backend, source.label)
                if task != TASK_SUPER_RESOLUTION:
                    display = canvas
                cv2.imshow(WINDOW_NAME, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    stop_reason = "q/ESC"
                    break
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    stop_reason = "window closed"
                    break
            elif processed % 100 == 0:
                print(f"[run] frame {processed} | G2G {t['total_ms']:.2f} ms | {fps:.0f} FPS | "
                      f"{output_count(task, result)} | {task}: {output_source(task, result)}", flush=True)
        rss_last = rss_mb()
    finally:
        source.release()
        if window:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        pipeline.close()
        print("[Ignition] shutdown: source released, window closed, NPU hardware context released", flush=True)

    print(f"[summary] stop: {stop_reason} | {processed} frames processed ({min(processed, args.warmup)} warm-up, "
          f"{timed} timed) | {unique} distinct source frames | {source.label}", flush=True)
    record: Dict[str, Any] = {
        "schema": 1, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(model), "model_name": model.name, "model_bytes": model.stat().st_size, "task": task,
        "backend": "npu" if native else "onnxruntime-cpu", "source": source.label, "source_kind": source.kind,
        "frames_requested": args.frames, "warmup": args.warmup, "frames_processed": processed, "frames_timed": timed,
        "stop_reason": stop_reason, "host": host_info(), "max_fps": args.max_fps, "power_mode": power,
    }
    if source.kind == "camera":
        frames, repeats, span = source.rates()
        if frames > 1 and span > 0:
            print(f"[summary] camera: {frames} frames in {span:.1f} s = {(frames - 1) / span:.2f} fps read, "
                  f"{repeats} repeated the previous frame, {(frames - 1 - repeats) / span:.2f} distinct fps",
                  flush=True)
            record["camera"] = {"frames_read": frames, "repeated_frames": repeats, "span_s": span,
                                "read_fps": (frames - 1) / span, "distinct_fps": (frames - 1 - repeats) / span}
    if timed:
        v = g2g.values()
        p50, p95, p99 = np.percentile(v, [50, 95, 99])
        print(f"[summary] G2G mean {v.mean():.3f} ms | P50 {p50:.3f} | P95 {p95:.3f} | P99 {p99:.3f} | "
              f"max {v.max():.3f} | over the last {v.size} timed frames; cold first frame {cold_ms:.2f} ms",
              flush=True)
        forward = "NPU + host forward" if host.count else "NPU forward"
        stages = f"preprocess {pre.mean():.3f} | {forward if native else 'ONNX Runtime'} {net.mean():.3f}"
        if dispatch.count:
            host_part = f", host {host.mean():.3f}" if host.count else ""
            stages += f" (dispatch {dispatch.mean():.3f}{host_part}, readback {readback.mean():.3f})"
        post_name = {TASK_DETECT: "decode+NMS", TASK_POSE: "decode+NMS",
                     TASK_CLASSIFY: "softmax+top-k"}.get(task, "image output")
        print(f"[summary] stage means (ms): {stages} | {post_name} {post.mean():.3f}", flush=True)
        if age.count:
            print(f"[summary] camera arrival to output mean {age.mean():.3f} ms", flush=True)
        if task == TASK_DETECT:
            origin = f"boxes from NPU heads on {npu_frames}/{timed} timed frames" if native else "boxes from ONNX Runtime"
            print(f"[summary] {origin} | {detections / timed:.2f} detections per frame", flush=True)
        elif task == TASK_CLASSIFY:
            top = max(top1_counts.items(), key=lambda kv: kv[1])
            print(f"[summary] top-1 class {top[0]} on {top[1]}/{timed} timed frames", flush=True)
        elif task == TASK_POSE:
            origin = (f"keypoints from NPU heads on {npu_frames}/{timed} timed frames" if native
                      else "keypoints from ONNX Runtime")
            print(f"[summary] {origin} | {people / timed:.2f} people per frame", flush=True)
        else:
            print(f"[summary] output image {output_shape} from {'the NPU' if native else 'ONNX Runtime'}", flush=True)
        print(f"[summary] RSS {rss_first:.1f} MB at the first timed frame, {rss_last:.1f} MB at the end "
              f"({rss_last - rss_first:+.2f} MB over {timed} frames)", flush=True)
        record.update({
            "g2g_ms": {"mean": float(v.mean()), "p50": float(p50), "p95": float(p95), "p99": float(p99),
                       "min": float(v.min()), "max": float(v.max()), "cold_first_frame": float(cold_ms)},
            "fps_from_mean": 1000.0 / float(v.mean()),
            "stages_ms": {"preprocess": pre.mean(), "network": net.mean(), "postprocess": post.mean(),
                          **({"dispatch": dispatch.mean(), "readback": readback.mean()} if dispatch.count else {}),
                          **({"host": host.mean()} if host.count else {})},
            "rss_mb": {"first_timed": rss_first, "end": rss_last, "drift": rss_last - rss_first},
            "npu_frames": npu_frames,
        })
        if task == TASK_DETECT:
            record["detections_per_frame"] = detections / timed
        elif task == TASK_CLASSIFY:
            record["top1_class_counts"] = {str(k): n for k, n in sorted(top1_counts.items())}
        elif task == TASK_POSE:
            record["people_per_frame"] = people / timed
        else:
            record["output_shape"] = output_shape
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"[summary] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
