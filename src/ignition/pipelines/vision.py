# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
src/ignition/pipelines/vision.py
Image classification, pose estimation and super-resolution pipelines, and the model-kind dispatch
that lets one camera loop (live_ignition.py) serve detection, classification, pose and
super-resolution models.

The model file selects the execution path, as for YOLOPipeline:
  * ``.onnx``: ONNX Runtime with the CPU execution provider.
  * ``.ignite``: an ignite-xdna container on NPU ``device_id``. The manifest's ``task`` names what
    the container computes; containers without the field are YOLO detectors.

Every result carries ``timings_ms`` with the same keys: ``preprocess_ms``, ``backbone_ms`` (the
network: ONNX Runtime ``run`` or the NPU forward pass), ``postprocess_ms``, ``g2g_ms`` and
``total_ms`` (frame in memory to finished output), plus ``dispatch_ms``/``readback_ms`` on the NPU.
"""

import json
import logging
import math
import time
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .yolo import (NATIVE_BACKENDS, YOLOPipeline, _release_native, is_ignite_container, load_bgr,
                   note_onnx_runs_on_cpu)

_log = logging.getLogger("ignition")

TASK_DETECT = "detect"
TASK_CLASSIFY = "classify"
TASK_SUPER_RESOLUTION = "super_resolution"
TASK_POSE = "pose"
TASKS = (TASK_DETECT, TASK_CLASSIFY, TASK_SUPER_RESOLUTION, TASK_POSE)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_INTERPOLATION = {"bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC, "nearest": cv2.INTER_NEAREST}
SR_PIXEL_MEAN = 128.0


def _ort_session(model_path: Path):
    import onnxruntime as ort
    if hasattr(ort, "set_default_logger_severity"):
        ort.set_default_logger_severity(3)
    options = ort.SessionOptions()
    options.log_severity_level = 3
    return ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])


def ignite_manifest(model_path: Union[str, Path]) -> Dict[str, Any]:
    """The JSON manifest of an ignite-xdna ``.ignite`` container."""
    from ignite_xdna.compiler.serializer import IgniteModelReader
    with IgniteModelReader(Path(model_path)) as reader:
        return dict(reader.manifest)


def _onnx_io_shapes(model_path: Path) -> Tuple[List[List[Any]], List[List[Any]]]:
    """Graph input and output shapes, read without building an inference session."""
    import onnx

    model = onnx.load(str(model_path), load_external_data=False)

    def dims(values):
        return [[d.dim_value if d.HasField("dim_value") else d.dim_param for d in v.type.tensor_type.shape.dim]
                for v in values]

    initializers = {t.name for t in model.graph.initializer}
    return dims([v for v in model.graph.input if v.name not in initializers]), dims(model.graph.output)


def infer_task(model_path: Union[str, Path]) -> str:
    """``detect``, ``classify``, ``pose`` or ``super_resolution``, from the container manifest or the ONNX
    output shapes.

    ONNX: six outputs (a head-cut YOLO) or one ``(1, 84, N)`` tensor is detection; nine outputs, three of
    them with 51 channels (a head-cut YOLOv8-pose: 17 keypoints x 3), are pose estimation; one 2-D output
    is classification; one 4-D output whose height is a whole multiple of the input height is
    super-resolution.
    """
    path = Path(model_path)
    if is_ignite_container(path):
        task = ignite_manifest(path).get("task", TASK_DETECT)
        if task not in TASKS:
            raise ValueError(f"{path} declares task {task!r}; Ignition serves {TASKS}")
        return task
    inputs, outputs = _onnx_io_shapes(path)
    if len(outputs) == 9 and sum(len(o) == 4 and o[1] == 51 for o in outputs) == 3:
        return TASK_POSE
    if len(outputs) == 6 or (len(outputs) == 1 and len(outputs[0]) == 3 and outputs[0][1] == 84):
        return TASK_DETECT
    if len(outputs) == 1 and len(outputs[0]) == 2:
        return TASK_CLASSIFY
    if len(outputs) == 1 and len(outputs[0]) == 4 and inputs and len(inputs[0]) == 4:
        h_in, h_out = inputs[0][2], outputs[0][2]
        if isinstance(h_in, int) and isinstance(h_out, int) and 0 < h_in < h_out and h_out % h_in == 0:
            return TASK_SUPER_RESOLUTION
    raise ValueError(f"cannot tell the task of {path} (inputs {inputs}, outputs {outputs}); name it explicitly")


# -- classification ------------------------------------------------------------------
@dataclass
class Classification:
    class_id: int
    score: float


@dataclass
class ClassificationResult:
    """Top-k classes (softmax probabilities, best first) and stage latencies."""
    topk: List[Classification]
    timings_ms: Dict[str, float]
    orig_shape: Tuple[int, int]
    source: str = "onnxruntime"


def load_preprocess_config(model_path: Path, input_hw: Tuple[int, int],
                           config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """timm data config: ``config_path``, else ``preprocess_config.json`` beside the model when its
    input size matches the model's, else ImageNet defaults at the model's input size."""
    candidates = [Path(config_path)] if config_path else [model_path.with_name("preprocess_config.json")]
    for candidate in candidates:
        if candidate.is_file():
            cfg = json.loads(candidate.read_text(encoding="utf-8"))
            if tuple(cfg.get("input_size", [0, 0, 0])[1:]) == tuple(input_hw):
                cfg["path"] = str(candidate)
                return cfg
            if config_path:
                raise ValueError(f"{candidate} is for input {cfg.get('input_size')}, the model takes {input_hw}")
    return {"input_size": [3, input_hw[0], input_hw[1]], "interpolation": "bicubic", "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD), "crop_pct": 0.875, "crop_mode": "center", "path": None}


def classification_preprocess(img_bgr: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    """timm eval transform: resize the shorter side to size / crop_pct, centre crop, normalize, NCHW float32."""
    _, size_h, size_w = (int(v) for v in cfg["input_size"])
    interp = _INTERPOLATION.get(str(cfg.get("interpolation", "bicubic")), cv2.INTER_CUBIC)
    scale_size = int(math.floor(size_h / float(cfg.get("crop_pct", 0.875))))
    h, w = img_bgr.shape[:2]
    if h <= w:
        new_h, new_w = scale_size, int(scale_size * w / h)
    else:
        new_h, new_w = int(scale_size * h / w), scale_size
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=interp)
    top, left = int(round((new_h - size_h) / 2.0)), int(round((new_w - size_w) / 2.0))
    rgb = cv2.cvtColor(resized[top:top + size_h, left:left + size_w], cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / np.float32(255.0)
    x -= np.asarray(cfg["mean"], dtype=np.float32)
    x /= np.asarray(cfg["std"], dtype=np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


class ClassificationPipeline:
    """ImageNet-style classifier on ONNX Runtime: timm eval preprocessing, softmax, top-k."""

    task = TASK_CLASSIFY

    def __init__(self, model_path: Union[str, Path], backend: str = "cpu", topk: int = 5,
                 preprocess_config: Optional[Union[str, Path]] = None, device_id: int = 0):
        self.model_path = Path(model_path)
        self.device_id = device_id
        self.is_native = is_ignite_container(self.model_path)
        if self.is_native:
            raise NotImplementedError(f"{self.model_path}: ignite-compile produces no classification containers; "
                                      f"run the .onnx model")
        note_onnx_runs_on_cpu(self.model_path, backend.lower())
        self.backend_name = "cpu"
        self.topk = int(topk)
        self.session = _ort_session(self.model_path)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.config = load_preprocess_config(self.model_path, tuple(inp.shape[2:]), preprocess_config)

    def predict(self, image: Union[str, Path, np.ndarray]) -> ClassificationResult:
        if self.session is None:
            raise RuntimeError("ClassificationPipeline is closed")
        img_bgr = load_bgr(image, copy=False)
        t0 = time.perf_counter()
        x = classification_preprocess(img_bgr, self.config)
        t1 = time.perf_counter()
        logits = self.session.run(None, {self.input_name: x})[0].reshape(-1).astype(np.float64)
        t2 = time.perf_counter()
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        k = min(self.topk, probs.size)
        idx = np.argpartition(-probs, k - 1)[:k]
        idx = idx[np.argsort(-probs[idx])]
        topk = [Classification(int(i), float(probs[i])) for i in idx]
        t3 = time.perf_counter()
        g2g = (t3 - t0) * 1000.0
        timings = {"preprocess_ms": (t1 - t0) * 1000.0, "backbone_ms": (t2 - t1) * 1000.0,
                   "postprocess_ms": (t3 - t2) * 1000.0, "g2g_ms": g2g, "total_ms": g2g}
        return ClassificationResult(topk, timings, (img_bgr.shape[0], img_bgr.shape[1]))

    def close(self) -> None:
        self.session = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# -- pose estimation -----------------------------------------------------------------
# COCO's 17-point skeleton as ultralytics draws it, 0-indexed: nose, eyes, ears, shoulders, elbows, wrists,
# hips, knees, ankles.
POSE_SKELETON = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6),
)


@dataclass
class Person:
    """One person in source pixels: box ``(x, y, w, h)``, score, and 17 COCO keypoints as float32 ``(x, y,
    visibility)`` rows."""
    x: float
    y: float
    w: float
    h: float
    score: float
    keypoints: np.ndarray


@dataclass
class PoseResult:
    """People (best first) and stage latencies.

    ``source`` is ``"onnxruntime"`` or ``"npu"``; ``pipeline_timings`` is ignite-xdna's per-frame
    timing record for NPU frames."""
    people: List[Person]
    timings_ms: Dict[str, float]
    orig_shape: Tuple[int, int]
    source: str = "onnxruntime"
    pipeline_timings: Optional[Any] = None


def draw_poses(img_bgr: np.ndarray, people: List[Person], kpt_conf: float = 0.5,
               color: Tuple[int, int, int] = (0, 230, 115), inplace: bool = False) -> np.ndarray:
    """Draws each person's box, score and skeleton, joining keypoints whose visibility is at least ``kpt_conf``
    (a copy unless ``inplace``)."""
    canvas = img_bgr if inplace else img_bgr.copy()
    for person in people:
        x1, y1 = int(round(person.x)), int(round(person.y))
        x2, y2 = int(round(person.x + person.w)), int(round(person.y + person.h))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"person {person.score:.2f}", (x1 + 3, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)
        kpts = person.keypoints
        visible = kpts[:, 2] >= kpt_conf
        for a, b in POSE_SKELETON:
            if visible[a] and visible[b]:
                cv2.line(canvas, (int(kpts[a, 0]), int(kpts[a, 1])), (int(kpts[b, 0]), int(kpts[b, 1])),
                         (255, 160, 0), 2, cv2.LINE_AA)
        for (px, py, _), shown in zip(kpts, visible):
            if shown:
                cv2.circle(canvas, (int(px), int(py)), 3, (0, 0, 255), -1, cv2.LINE_AA)
    return canvas


class PosePipeline:
    """YOLOv8-pose: people and 17 keypoints each.

    An ``.ignite`` container runs every layer on NPU ``device_id`` through ignite-xdna's pose pipeline; a
    head-cut ``.onnx`` model (nine outputs) runs on ONNX Runtime's CPU provider. Both decode the nine heads
    with ignite-xdna's ``PoseDecoder`` (DFL boxes, sigmoid score, keypoints, class-agnostic NMS), so the
    ignite-xdna package is needed for either.
    """

    task = TASK_POSE

    def __init__(self, model_path: Union[str, Path], backend: str = "cpu", conf_thres: float = 0.25,
                 iou_thres: float = 0.45, device_id: int = 0):
        self.model_path = Path(model_path)
        self.device_id = device_id
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.session = None
        self.native: Optional[Any] = None
        self.is_native = is_ignite_container(self.model_path)
        try:
            from ignite_xdna.pipelines import pose_pipeline
        except ImportError as exc:
            raise ImportError("pose models need ignite-xdna's pose_pipeline, which decodes the heads of "
                              ".ignite and .onnx models alike") from exc
        self._pose = pose_pipeline
        if self.is_native:
            if backend.lower() not in NATIVE_BACKENDS:
                raise ValueError(f"{self.model_path} is an .ignite container, which runs only on the NPU")
            native = pose_pipeline.PosePipeline(self.model_path, device_index=device_id, conf_thres=conf_thres,
                                                iou_thres=iou_thres)
            self.native = native
            self._native_finalizer = weakref.finalize(self, _release_native, native)
            self.backend_name = "xdna1"
            _log.info("[Ignition] Native XDNA1 NPU backend active (Device %d)", device_id)
        else:
            note_onnx_runs_on_cpu(self.model_path, backend.lower())
            self.backend_name = "cpu"
            self.session = _ort_session(self.model_path)
            self.input_name = self.session.get_inputs()[0].name
            shapes = [tuple(o.shape) for o in self.session.get_outputs()]
            branch = {64: 0, 1: 1, 51: 2}
            if len(shapes) != 9 or any(len(s) != 4 or s[1] not in branch or not isinstance(s[2], int) for s in shapes):
                raise ValueError(f"{self.model_path}: expected nine head-cut YOLOv8-pose outputs, got {shapes}")
            # ignite-xdna's head order: box, score, keypoints, each at strides 8, 16, 32 (largest grid first)
            self._order = sorted(range(9), key=lambda i: (branch[shapes[i][1]], -shapes[i][2]))
            self._decoder = pose_pipeline.PoseDecoder(imgsz=int(self.session.get_inputs()[0].shape[3]),
                                                      conf_thres=conf_thres, iou_thres=iou_thres)

    def predict(self, image: Union[str, Path, np.ndarray]) -> PoseResult:
        img_bgr = load_bgr(image, copy=False)
        shape = (img_bgr.shape[0], img_bgr.shape[1])
        if self.is_native:
            native = self.native
            if native is None:
                raise RuntimeError("PosePipeline is closed")
            t0 = time.perf_counter()
            found, hw = native.predict_sync(img_bgr, conf_thres=self.conf_thres, iou_thres=self.iou_thres)
            people = [Person(d.x0, d.y0, d.w, d.h, d.score, d.keypoints) for d in found]
            g2g = (time.perf_counter() - t0) * 1000.0
            timings = {"preprocess_ms": hw.preprocess_ms, "backbone_ms": hw.npu_forward_ms,
                       "postprocess_ms": hw.postprocess_ms, "g2g_ms": g2g, "total_ms": g2g,
                       "dispatch_ms": hw.dispatch_ms, "readback_ms": hw.readback_ms}
            return PoseResult(people, timings, shape, "npu", hw)
        if self.session is None:
            raise RuntimeError("PosePipeline is closed")
        t0 = time.perf_counter()
        x, pad, scale = self._pose.letterbox(img_bgr, self._decoder.imgsz)
        t1 = time.perf_counter()
        outs = self.session.run(None, {self.input_name: x})
        t2 = time.perf_counter()
        decoder = self._decoder
        found = decoder.postprocess(decoder.decode([outs[i] for i in self._order], None, self.conf_thres), pad,
                                    scale, self.conf_thres, self.iou_thres)
        people = [Person(d.x0, d.y0, d.w, d.h, d.score, d.keypoints) for d in found]
        t3 = time.perf_counter()
        g2g = (t3 - t0) * 1000.0
        timings = {"preprocess_ms": (t1 - t0) * 1000.0, "backbone_ms": (t2 - t1) * 1000.0,
                   "postprocess_ms": (t3 - t2) * 1000.0, "g2g_ms": g2g, "total_ms": g2g}
        return PoseResult(people, timings, shape)

    def close(self) -> None:
        """Releases the ONNX Runtime session or the NPU hardware context; safe to call again."""
        self.session = None
        if self.native is not None:
            self.native = None
            self._native_finalizer()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# -- super-resolution ----------------------------------------------------------------
@dataclass
class SuperResolutionResult:
    """The upscaled frame (BGR uint8, ``scale`` times the network input) and stage latencies.

    ``source`` is ``"onnxruntime"`` or ``"npu"``; ``pipeline_timings`` is ignite-xdna's per-frame
    timing record for NPU frames."""
    image: np.ndarray
    timings_ms: Dict[str, float]
    orig_shape: Tuple[int, int]
    source: str = "onnxruntime"
    pipeline_timings: Optional[Any] = None


def sr_preprocess(img_bgr: np.ndarray, input_hw: Tuple[int, int]) -> np.ndarray:
    """SESR input: bilinear resize to the network size, RGB, minus 128, NCHW float32."""
    h, w = input_hw
    if img_bgr.shape[:2] != (h, w):
        img_bgr = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    x = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    x -= np.float32(SR_PIXEL_MEAN)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def sr_postprocess(pred: np.ndarray) -> np.ndarray:
    """Network output (NCHW float32, centred on 128) -> BGR uint8 image."""
    y = np.squeeze(pred, axis=0).transpose(1, 2, 0) + np.float32(SR_PIXEL_MEAN)
    np.clip(y, 0.0, 255.0, out=y)
    return cv2.cvtColor(y.astype(np.uint8), cv2.COLOR_RGB2BGR)


class SuperResolutionPipeline:
    """Single-image super-resolution (SESR): ONNX Runtime for ``.onnx``, the NPU for ``.ignite``."""

    task = TASK_SUPER_RESOLUTION

    def __init__(self, model_path: Union[str, Path], backend: str = "cpu", device_id: int = 0):
        self.model_path = Path(model_path)
        self.device_id = device_id
        self.session = None
        self.native: Optional[Any] = None
        self.is_native = is_ignite_container(self.model_path)
        if self.is_native:
            if backend.lower() not in NATIVE_BACKENDS:
                raise ValueError(f"{self.model_path} is an .ignite container, which runs only on the NPU")
            try:
                from ignite_xdna.pipelines.sr_pipeline import SuperResolutionPipeline as NativeSR
            except ImportError as exc:
                raise ImportError(".ignite super-resolution containers need ignite-xdna's sr_pipeline") from exc
            native = NativeSR(self.model_path, device_index=device_id)
            self.native = native
            self._native_finalizer = weakref.finalize(self, _release_native, native)
            self.backend_name = "xdna1"
            self.input_hw = tuple(native.input_hw)
            self.scale = int(native.scale)
            _log.info("[Ignition] Native XDNA1 NPU backend active (Device %d)", device_id)
        else:
            note_onnx_runs_on_cpu(self.model_path, backend.lower())
            self.backend_name = "cpu"
            self.session = _ort_session(self.model_path)
            inp, out = self.session.get_inputs()[0], self.session.get_outputs()[0]
            self.input_name = inp.name
            self.input_hw = (int(inp.shape[2]), int(inp.shape[3]))
            self.scale = int(out.shape[2]) // self.input_hw[0]

    def predict(self, image: Union[str, Path, np.ndarray]) -> SuperResolutionResult:
        img_bgr = load_bgr(image, copy=False)
        shape = (img_bgr.shape[0], img_bgr.shape[1])
        if self.is_native:
            native = self.native
            if native is None:
                raise RuntimeError("SuperResolutionPipeline is closed")
            t0 = time.perf_counter()
            out, hw = native.predict_sync(img_bgr)
            g2g = (time.perf_counter() - t0) * 1000.0
            timings = {"preprocess_ms": hw.preprocess_ms, "backbone_ms": hw.npu_forward_ms,
                       "postprocess_ms": hw.postprocess_ms, "g2g_ms": g2g, "total_ms": g2g,
                       "dispatch_ms": hw.dispatch_ms, "readback_ms": hw.readback_ms}
            return SuperResolutionResult(out, timings, shape, "npu", hw)
        if self.session is None:
            raise RuntimeError("SuperResolutionPipeline is closed")
        t0 = time.perf_counter()
        x = sr_preprocess(img_bgr, self.input_hw)
        t1 = time.perf_counter()
        y = self.session.run(None, {self.input_name: x})[0]
        t2 = time.perf_counter()
        out = sr_postprocess(y)
        t3 = time.perf_counter()
        g2g = (t3 - t0) * 1000.0
        timings = {"preprocess_ms": (t1 - t0) * 1000.0, "backbone_ms": (t2 - t1) * 1000.0,
                   "postprocess_ms": (t3 - t2) * 1000.0, "g2g_ms": g2g, "total_ms": g2g}
        return SuperResolutionResult(out, timings, shape)

    def close(self) -> None:
        """Releases the ONNX Runtime session or the NPU hardware context; safe to call again."""
        self.session = None
        if self.native is not None:
            self.native = None
            self._native_finalizer()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def create_pipeline(model_path: Union[str, Path], task: Optional[str] = None, conf_thres: float = 0.25,
                    iou_thres: float = 0.45, device_id: int = 0) -> Tuple[str, Any]:
    """``(task, pipeline)`` for a model file: ``.ignite`` on the NPU, ``.onnx`` on ONNX Runtime CPU."""
    path = Path(model_path)
    task = task or infer_task(path)
    backend = "xdna1" if is_ignite_container(path) else "cpu"
    if task == TASK_DETECT:
        return task, YOLOPipeline(path, backend=backend, conf_thres=conf_thres, iou_thres=iou_thres,
                                  device_id=device_id)
    if task == TASK_CLASSIFY:
        return task, ClassificationPipeline(path, backend=backend, device_id=device_id)
    if task == TASK_SUPER_RESOLUTION:
        return task, SuperResolutionPipeline(path, backend=backend, device_id=device_id)
    if task == TASK_POSE:
        return task, PosePipeline(path, backend=backend, conf_thres=conf_thres, iou_thres=iou_thres,
                                  device_id=device_id)
    raise ValueError(f"unknown task {task!r}; expected one of {TASKS}")
