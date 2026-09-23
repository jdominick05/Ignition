# Copyright (C) 2026 The Ignition contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
src/ignition/pipelines/yolo.py
End-to-end YOLOv8n object detection pipeline for AMD Phoenix XDNA1 / AIE2 silicon.
Integrates letterboxing, INT8 Conv2D feature backbone execution, anchor/DFL decoding,
and per-class Non-Maximum Suppression (NMS).

The model file selects the execution path:
  * ``.onnx``: ONNX Runtime's CPU execution provider computes the head tensors; decode and
    NMS run here. The NPU is not used, whichever backend was requested.
  * ``.ignite`` (bare-metal ignite-xdna container, ``IGNT`` header magic): the whole
    network runs on the NPU through ``ignite_xdna.pipelines.yolo_pipeline.YoloPipeline``,
    one dispatch per frame with boxes decoded from the NPU's detect heads. ONNX
    Runtime is not called.
"""

import logging
import os
import weakref
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image

from ..model import Model
from ..backends.base import BenchmarkReport
from ..backends.cpu import CPUBackend

_log = logging.getLogger("ignition")

INPUT_SIZE = 640
STRIDES = (8, 16, 32)
REG_MAX = 16
NUM_CLASSES = 80

IGNITE_MAGIC = b"IGNT"
NATIVE_BACKENDS = ("xdna1", "npu", "aie2")

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


@dataclass
class Detection:
    """Represents a single detected bounding box and class prediction."""
    x: float
    y: float
    w: float
    h: float
    score: float
    class_id: int
    class_name: str

    @property
    def xyxy(self) -> Tuple[float, float, float, float]:
        """Returns [x1, y1, x2, y2] bounding box coordinates."""
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    @property
    def xywh(self) -> Tuple[float, float, float, float]:
        """Returns [x, y, w, h] bounding box coordinates."""
        return (self.x, self.y, self.w, self.h)


@dataclass
class YOLOResult:
    """Complete prediction result containing detections, annotated image, and timing metrics.

    ``head_source`` names where the boxes came from: ``"npu"`` (the NPU's detect
    heads), ``"none"`` (an ``.ignite`` container without detect heads) or
    ``"onnxruntime"``. ``pipeline_timings`` is ignite-xdna's ``PipelineTimings``
    for NPU frames.
    """
    detections: List[Detection]
    timings_ms: Dict[str, float]
    orig_shape: Tuple[int, int]
    image_annotated: Optional[np.ndarray] = None
    head_source: Optional[str] = None
    pipeline_timings: Optional[Any] = None

    def save(self, output_path: Union[str, Path]) -> Path:
        """Saves the annotated visual detection image to disk."""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        if self.image_annotated is not None:
            cv2.imwrite(str(out_p), self.image_annotated)
            return out_p
        raise RuntimeError("No annotated image present in result to save.")

    def summary(self) -> str:
        """Generates a formatted summary table of detections and pipeline latencies."""
        lines = [
            f"=== YOLOv8 Detection Result ===",
            f"Image Dimensions: {self.orig_shape[1]}x{self.orig_shape[0]} px",
            f"Objects Detected: {len(self.detections)}",
        ]
        for i, d in enumerate(self.detections, 1):
            x1, y1, x2, y2 = d.xyxy
            lines.append(f"  [{i:02d}] {d.class_name:<15} conf={d.score:.2f}  box=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
        lines.append(f"--- Pipeline Latency Breakdown ---")
        lines.append(f"  Preprocess (Letterbox + INT8): {self.timings_ms.get('preprocess_ms', 0.0):.2f} ms")
        lines.append(f"  Feature Backbone Inference:   {self.timings_ms.get('backbone_ms', 0.0):.2f} ms")
        if "dispatch_ms" in self.timings_ms:
            lines.append(f"    NPU Dispatch:                {self.timings_ms['dispatch_ms']:.2f} ms")
            if "host_ms" in self.timings_ms:
                lines.append(f"    Host Layers (ONNX Runtime):  {self.timings_ms['host_ms']:.2f} ms")
            lines.append(f"    Head Readback:               {self.timings_ms.get('readback_ms', 0.0):.2f} ms")
        lines.append(f"  Head Decode + Per-Class NMS:   {self.timings_ms.get('postprocess_ms', 0.0):.2f} ms")
        lines.append(f"  Total End-to-End Latency:      {self.timings_ms.get('total_ms', 0.0):.2f} ms")
        fps = 1000.0 / self.timings_ms.get('total_ms', 1.0)
        lines.append(f"  End-to-End Throughput:         {fps:.1f} FPS")
        return "\n".join(lines)


def is_ignite_container(model_path: Any) -> bool:
    """True for an ignite-xdna ``.ignite`` container: the suffix, or the ``IGNT`` header magic."""
    if not isinstance(model_path, (str, os.PathLike)):
        return False
    path = Path(model_path)
    if path.suffix.lower() == ".ignite":
        return True
    try:
        with path.open("rb") as f:
            return f.read(len(IGNITE_MAGIC)) == IGNITE_MAGIC
    except OSError:
        return False


def note_onnx_runs_on_cpu(model_path: Path, backend: str) -> None:
    """Warns when an NPU backend was requested for an ONNX model: it runs on ONNX Runtime's CPU provider."""
    if backend in NATIVE_BACKENDS:
        _log.warning("[Ignition] %s is an ONNX model: it runs on ONNX Runtime's CPU execution provider; "
                     "only an .ignite container runs on the NPU", model_path.name)


def load_bgr(image: Union[str, Path, np.ndarray, Image.Image], copy: bool = True) -> np.ndarray:
    """Returns the input as an HxWx3 BGR array; with ``copy=False`` an array is used as is."""
    if isinstance(image, (str, Path)):
        img_bgr = cv2.imread(str(image))
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read image at path: {image}")
        return img_bgr
    if isinstance(image, Image.Image):
        rgb = np.array(image.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if isinstance(image, np.ndarray):
        return image.copy() if copy else image
    raise TypeError(f"Unsupported image type: {type(image)}")


_anchor_cache: Dict[Tuple[int, Tuple[int, ...]], Tuple[np.ndarray, np.ndarray]] = {}


def anchors_and_strides(imgsz: int = INPUT_SIZE, strides: Tuple[int, ...] = STRIDES) -> Tuple[np.ndarray, np.ndarray]:
    """Generates anchor center coordinates (1,2,N) and stride vector (1,N)."""
    key = (imgsz, strides)
    if key in _anchor_cache:
        return _anchor_cache[key]
    pts, sts = [], []
    for s in strides:
        g = imgsz // s
        c = np.arange(g, dtype=np.float32) + 0.5
        yy, xx = np.meshgrid(c, c, indexing="ij")
        pts.append(np.stack([xx.ravel(), yy.ravel()], 0))
        sts.append(np.full(g * g, float(s), np.float32))
    out = (np.concatenate(pts, 1)[None], np.concatenate(sts)[None])
    _anchor_cache[key] = out
    return out


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    np.exp(x, out=x)
    x /= x.sum(axis=axis, keepdims=True)
    return x


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x, dtype=np.float32))


_bins_vec = np.arange(REG_MAX, dtype=np.float32).reshape(1, 1, REG_MAX, 1)


def decode_heads(
    outs: List[np.ndarray],
    imgsz: int = INPUT_SIZE,
    strides: Tuple[int, ...] = STRIDES,
    conf_thres: Optional[float] = None
) -> np.ndarray:
    """
    Decodes the 6 raw conv head outputs into (1, 84, N) [xywh, class_probs] tensor.
    Vectorized DFL (Distribution Focal Loss) expected values and anchor grid offsets in NumPy
    using strided views along the bin dimension to eliminate intermediate array allocations.
    """
    box_f, cls_f = outs[:3], outs[3:]
    nc = cls_f[0].shape[1]

    box = np.concatenate([b.reshape(1, 4 * REG_MAX, -1) for b in box_f], 2)
    cls = np.concatenate([c.reshape(1, nc, -1) for c in cls_f], 2)
    anc, st = anchors_and_strides(imgsz, strides)

    if conf_thres is not None:
        c = min(max(float(conf_thres), 1e-12), 1.0 - 1e-12)
        t = np.log(c / (1.0 - c))
        keep = np.flatnonzero(cls[0].max(0) > t)
        if keep.size == 0:
            return np.zeros((1, 4 + nc, 0), np.float32)
        box = box[:, :, keep]
        cls = cls[:, :, keep]
        anc = anc[:, :, keep]
        st = st[:, keep]

    box = box.astype(np.float32, copy=False)
    n = box.shape[2]

    # Vectorized DFL: strided view (1, 4, 16, n) without transposing
    v = box.reshape(1, 4, REG_MAX, n)
    d = _softmax(v.copy(), axis=2)
    ltrb = (d * _bins_vec).sum(axis=2)  # (1, 4, n) distances left, top, right, bottom

    x1y1 = anc - ltrb[:, 0:2]
    x2y2 = anc + ltrb[:, 2:4]
    cxcy = (x1y1 + x2y2) * 0.5
    wh = x2y2 - x1y1
    xywh = np.concatenate([cxcy, wh], 1) * st[:, None]

    return np.concatenate([xywh, _sigmoid(cls.astype(np.float32, copy=False))], 1)


def letterbox(
    img_bgr: np.ndarray,
    size: int = INPUT_SIZE,
    color: Tuple[int, int, int] = (114, 114, 114),
    canvas_buf: Optional[np.ndarray] = None,
    out_tensor: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Tuple[int, int], float]:
    """
    Resizes image preserving aspect ratio with symmetric stride-32 padding.
    Accelerated with in-place OpenCV sub-slice resize and direct in-place tensor
    packing into pinned buffers to eliminate intermediate memory allocations.
    Returns:
      x: (1, 3, size, size) float32 RGB array normalized to [0.0, 1.0]
      pad: (pad_top, pad_left)
      scale: scaling factor
    """
    h, w = img_bgr.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    pad_top = (size - nh) // 2
    pad_left = (size - nw) // 2

    canvas = canvas_buf if canvas_buf is not None else np.empty((size, size, 3), dtype=np.uint8)
    canvas.fill(color[0])
    sub = canvas[pad_top:pad_top + nh, pad_left:pad_left + nw]
    cv2.resize(img_bgr, (nw, nh), dst=sub, interpolation=cv2.INTER_LINEAR)

    if out_tensor is not None:
        np.multiply(canvas[:, :, 2], 1.0 / 255.0, out=out_tensor[0, 0], casting="unsafe")
        np.multiply(canvas[:, :, 1], 1.0 / 255.0, out=out_tensor[0, 1], casting="unsafe")
        np.multiply(canvas[:, :, 0], 1.0 / 255.0, out=out_tensor[0, 2], casting="unsafe")
        return out_tensor, (pad_top, pad_left), scale

    # Zero-allocation fallback
    blob = np.empty((1, 3, size, size), dtype=np.float32)
    np.multiply(canvas[:, :, 2], 1.0 / 255.0, out=blob[0, 0], casting="unsafe")
    np.multiply(canvas[:, :, 1], 1.0 / 255.0, out=blob[0, 1], casting="unsafe")
    np.multiply(canvas[:, :, 0], 1.0 / 255.0, out=blob[0, 2], casting="unsafe")
    return blob, (pad_top, pad_left), scale


def postprocess_detections(
    output: np.ndarray,
    pad: Tuple[int, int],
    scale: float,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 300,
) -> List[Detection]:
    """
    Converts (1, 84, N) box/score tensor into filtered Detection objects via per-class NMS.
    Maps box coordinates back to original unpadded image space.
    """
    if output.shape[-1] == 0:
        return []

    out = output[0].T  # (N, 84)
    boxes = out[:, :4].copy()
    scores_all = out[:, 4:]
    cls = scores_all.argmax(axis=1)
    scores = scores_all[np.arange(len(cls)), cls]

    keep = scores >= conf_thres
    boxes, scores, cls = boxes[keep], scores[keep], cls[keep]
    if len(boxes) == 0:
        return []

    if len(scores) > 10 * max_det:
        top = np.argpartition(-scores, 10 * max_det)[:10 * max_det]
        boxes, scores, cls = boxes[top], scores[top], cls[top]

    # Invert letterbox padding and scale
    boxes[:, 0] -= pad[1]
    boxes[:, 1] -= pad[0]
    boxes /= scale
    xywh = np.stack([
        boxes[:, 0] - boxes[:, 2] / 2,
        boxes[:, 1] - boxes[:, 3] / 2,
        boxes[:, 2],
        boxes[:, 3]
    ], axis=1)

    bl, sl = xywh.tolist(), scores.tolist()
    # Batched per-class NMS via cv2
    idx = cv2.dnn.NMSBoxesBatched(bl, sl, cls.tolist(), conf_thres, iou_thres)
    if len(idx) == 0:
        return []

    idx = np.array(idx).reshape(-1)
    if len(idx) > max_det:
        idx = idx[np.argsort(-scores[idx])[:max_det]]

    detections = []
    for i in idx:
        x, y, w, h = xywh[i]
        sc = float(scores[i])
        c_id = int(cls[i])
        c_name = COCO_CLASSES[c_id] if 0 <= c_id < len(COCO_CLASSES) else f"class_{c_id}"
        detections.append(Detection(
            x=float(x),
            y=float(y),
            w=float(w),
            h=float(h),
            score=sc,
            class_id=c_id,
            class_name=c_name,
        ))

    return detections


def draw_detections(
    img_bgr: np.ndarray,
    detections: List[Detection],
    box_color: Tuple[int, int, int] = (0, 230, 115),
    text_color: Tuple[int, int, int] = (0, 0, 0),
    inplace: bool = False,
) -> np.ndarray:
    """Draws styled bounding boxes and class score labels onto the BGR image (a copy unless ``inplace``)."""
    canvas = img_bgr if inplace else img_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det.xyxy]
        # Bounding box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

        # Label badge
        label = f"{det.class_name} {det.score:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        badge_y1 = max(y1 - th - baseline - 4, 0)
        badge_y2 = badge_y1 + th + baseline + 4
        badge_x2 = min(x1 + tw + 6, canvas.shape[1])

        cv2.rectangle(canvas, (x1, badge_y1), (badge_x2, badge_y2), box_color, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 3, badge_y2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            text_color,
            1,
            cv2.LINE_AA
        )
    return canvas


def _release_native(native: Any) -> None:
    """Closes an ignite-xdna pipeline: its XRT hardware context and buffer objects."""
    try:
        native.close()
    except Exception as exc:  # noqa: BLE001 - teardown must not raise
        _log.warning("[Ignition] closing the native NPU pipeline failed: %s", exc)


class YOLOPipeline:
    """
    End-to-End YOLOv8 Object Detection Pipeline.
    Manages image letterbox preprocessing, the network, and CPU anchor decoding with
    per-class Non-Maximum Suppression.

    An ``.ignite`` container is served natively by ignite-xdna on NPU ``device_id``
    (``is_native``); any other model file runs on ONNX Runtime's CPU execution provider,
    and ``backend_name`` is then ``"cpu"`` whichever backend was requested.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        backend: str = "xdna1",
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        device_id: int = 0,
        **backend_kwargs
    ):
        self.model_path = Path(model_path)
        self.backend_name = backend.lower()
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device_id = device_id
        self.ort_session = None
        self.native: Optional[Any] = None  # ignite_xdna YoloPipeline for .ignite containers
        self.head_status: Optional[Any] = None
        self.is_native = is_ignite_container(self.model_path)

        if self.is_native:
            self._init_native()
            return

        # 1. Initialize reference CPU execution session for head feature extraction
        import onnxruntime as ort
        if hasattr(ort, "set_default_logger_severity"):
            ort.set_default_logger_severity(3)
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3  # Error only to suppress benign graph initializer warnings
        self.ort_session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_names = [o.name for o in self.ort_session.get_outputs()]

        # 2. An ONNX model never reaches the NPU: only an .ignite container does
        note_onnx_runs_on_cpu(self.model_path, self.backend_name)
        self.backend_name = "cpu"

    def _init_native(self) -> None:
        """Opens an ``.ignite`` container on the NPU through ignite-xdna's YOLOv8n pipeline."""
        if self.backend_name not in NATIVE_BACKENDS:
            raise ValueError(f"{self.model_path} is an .ignite container, which runs only on the NPU "
                             f"(backend 'xdna1'), not on backend '{self.backend_name}'")
        if not self.model_path.is_file():
            raise FileNotFoundError(f".ignite container not found: {self.model_path}")
        try:
            from ignite_xdna.pipelines.yolo_pipeline import YoloPipeline
        except ImportError as exc:
            raise ImportError(".ignite containers need the ignite-xdna runtime "
                              "(pip install -e ../ignite-xdna)") from exc

        native = YoloPipeline(
            model_path_or_bundle=self.model_path,
            device_index=self.device_id,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
        )
        # Boxes come from the NPU's detect heads only. ignite-xdna also opens an ONNX
        # Runtime session over the cut model for oracle comparisons; drop it so no
        # frame can reach ONNX Runtime and its memory is returned.
        native._ort_cut_sess = None
        self.native = native
        self.backend_name = "xdna1"
        # Closes the hardware context if close() is never called (collection or interpreter exit).
        self._native_finalizer = weakref.finalize(self, _release_native, native)

        self.head_status = getattr(native.session, "head_status", None)
        _log.info("[Ignition] Native XDNA1 NPU backend active (Device %d)", self.device_id)
        if self.head_status is not None and not self.head_status.present:
            _log.warning(
                "[Ignition] %s carries no NPU detect heads (%s): frames will have no detections. "
                "Use the graph-engine container build/yolov8n_full.ignite (ignite-compile --engine graph).",
                self.model_path.name, self.head_status.reason)

    def predict(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        conf_thres: Optional[float] = None,
        iou_thres: Optional[float] = None,
        annotate: bool = True,
    ) -> YOLOResult:
        """
        Runs complete object detection pipeline on input image.
        Returns YOLOResult with detections, annotated image (unless ``annotate=False``),
        and detailed latency metrics.
        """
        conf = conf_thres if conf_thres is not None else self.conf_thres
        iou = iou_thres if iou_thres is not None else self.iou_thres

        if self.is_native:
            return self._predict_native(image, conf, iou, annotate)

        # 1. Load image to BGR numpy array
        img_bgr = load_bgr(image, copy=True)

        orig_shape = (img_bgr.shape[0], img_bgr.shape[1])

        # 2. Preprocess: Letterbox resize and RGB normalization
        t0 = time.perf_counter()
        inp_tensor, pad, scale = letterbox(img_bgr, size=INPUT_SIZE)
        t1 = time.perf_counter()
        preprocess_ms = (t1 - t0) * 1000.0

        # 3. Head tensors from ONNX Runtime on the CPU
        t2 = time.perf_counter()
        raw_outputs = self.ort_session.run(None, {self.input_name: inp_tensor})
        t3 = time.perf_counter()
        backbone_ms = (t3 - t2) * 1000.0

        # 4. Postprocess: Dynamic head decoding & per-class NMS
        t4 = time.perf_counter()
        if len(raw_outputs) == 6:
            # Cut-head model: 6 raw convolution heads -> NumPy DFL anchor decode
            decoded = decode_heads(raw_outputs, imgsz=INPUT_SIZE, conf_thres=conf)
        elif len(raw_outputs) == 1 and raw_outputs[0].ndim == 3 and raw_outputs[0].shape[1] == 84:
            # Full uncut model: already (1, 84, N)
            decoded = raw_outputs[0]
        else:
            raise ValueError(f"Unexpected model outputs format: {len(raw_outputs)} tensors")

        detections = postprocess_detections(
            decoded,
            pad=pad,
            scale=scale,
            conf_thres=conf,
            iou_thres=iou,
        )
        t5 = time.perf_counter()
        postprocess_ms = (t5 - t4) * 1000.0
        total_ms = (t5 - t0) * 1000.0

        # 5. Visual annotation
        annotated_img = draw_detections(img_bgr, detections) if annotate else None

        timings = {
            "preprocess_ms": preprocess_ms,
            "backbone_ms": backbone_ms,
            "postprocess_ms": postprocess_ms,
            "total_ms": total_ms,
        }

        return YOLOResult(
            detections=detections,
            timings_ms=timings,
            orig_shape=orig_shape,
            image_annotated=annotated_img,
            head_source="onnxruntime",
        )

    def _predict_native(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        conf: float,
        iou: float,
        annotate: bool,
    ) -> YOLOResult:
        """One NPU frame: ignite-xdna stages, dispatches, reads the heads back and decodes them.

        Glass-to-glass (``g2g_ms`` = ``total_ms``) runs from the frame in memory to the
        Detection list. A camera frame is used as is (no copy); ignite-xdna's native
        preprocessor writes the quantized plane straight into the NPU workspace buffer.
        """
        native = self.native
        if native is None:
            raise RuntimeError("YOLOPipeline is closed")
        img_bgr = load_bgr(image, copy=False)

        t0 = time.perf_counter()
        if native.conf_thres != conf or native.iou_thres != iou:
            native.conf_thres, native.iou_thres = conf, iou
        dets, hw = native.predict_sync(img_bgr, use_oracle_for_boxes=False)
        # ignite-xdna boxes are (x0, y0, w, h) in source pixels, the same layout as Detection.
        detections = [Detection(d.x0, d.y0, d.w, d.h, d.score, d.class_id, d.class_name) for d in dets]
        g2g_ms = (time.perf_counter() - t0) * 1000.0

        timings = {
            "preprocess_ms": hw.preprocess_ms,
            "backbone_ms": hw.npu_forward_ms,
            "postprocess_ms": hw.postprocess_ms,
            "g2g_ms": g2g_ms,
            "total_ms": g2g_ms,
        }
        dispatch_ms = getattr(native.session, "last_dispatch_ms", None)
        if dispatch_ms is not None:
            # A container with host segments (YOLO11's C2PSA block) runs them on ONNX Runtime's CPU
            # provider between NPU dispatches; ignite-xdna reports that time apart from the dispatch.
            host_ms = float(getattr(native.session, "last_host_ms", 0.0) or 0.0)
            timings["dispatch_ms"] = float(dispatch_ms)
            if not getattr(native.session, "single_dispatch", True):
                timings["host_ms"] = host_ms
            timings["readback_ms"] = max(hw.npu_forward_ms - float(dispatch_ms) - host_ms, 0.0)

        return YOLOResult(
            detections=detections,
            timings_ms=timings,
            orig_shape=(img_bgr.shape[0], img_bgr.shape[1]),
            image_annotated=draw_detections(img_bgr, detections) if annotate else None,
            head_source=hw.head_source,
            pipeline_timings=hw,
        )

    def stream(
        self,
        frame_iterator: Any,
        conf_thres: Optional[float] = None,
        iou_thres: Optional[float] = None,
        annotate: bool = False,
    ):
        """
        Streams frames using the high-performance 3-stage asynchronous pipelined runner.
        Overlaps preprocessing and postprocessing/NMS with physical silicon feature extraction.
        An ``.ignite`` container streams synchronously: one NPU dispatch per frame, in order.
        """
        if self.is_native:
            for item in frame_iterator:
                yield self.predict(item, conf_thres=conf_thres, iou_thres=iou_thres, annotate=annotate)
            return

        from .streaming import AsyncYOLOPipeline
        async_pipe = AsyncYOLOPipeline(
            model_path=self.model_path,
            backend=self.backend_name,
            conf_thres=conf_thres if conf_thres is not None else self.conf_thres,
            iou_thres=iou_thres if iou_thres is not None else self.iou_thres,
            device_id=self.device_id,
        )
        try:
            yield from async_pipe.stream(
                frame_iterator=frame_iterator,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                annotate=annotate,
            )
        finally:
            async_pipe.close()

    def benchmark(
        self,
        warmup: int = 20,
        iterations: int = 100
    ) -> BenchmarkReport:
        """Benchmarks sustained feature extraction throughput."""
        if self.is_native:
            # Whole-frame NPU latency (preprocess to detections) on a mid-grey 640x640 frame
            frame = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
            for _ in range(warmup):
                self.predict(frame, annotate=False)
            latencies = []
            t_start = time.perf_counter()
            for _ in range(iterations):
                latencies.append(self.predict(frame, annotate=False).timings_ms["total_ms"] * 1000.0)
            elapsed_s = time.perf_counter() - t_start
            lat = np.array(latencies)
            mean_us = float(np.mean(lat))
            return BenchmarkReport(
                backend_name="xdna1",
                iterations=iterations,
                mean_us=mean_us,
                median_us=float(np.median(lat)),
                min_us=float(np.min(lat)),
                p95_us=float(np.percentile(lat, 95)),
                fps=iterations / elapsed_s if elapsed_s > 0 else 0.0,
                intermediate_ddr_bytes=0,
                init_us=0.0,
            )

        # ONNX Runtime CPU benchmark
        dummy_in = np.zeros((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
        for _ in range(warmup):
            self.ort_session.run(None, {self.input_name: dummy_in})

        latencies = []
        for _ in range(iterations):
            ts = time.perf_counter()
            self.ort_session.run(None, {self.input_name: dummy_in})
            te = time.perf_counter()
            latencies.append((te - ts) * 1e6)

        lat = np.array(latencies)
        mean_us = float(np.mean(lat))
        return BenchmarkReport(
            backend_name="cpu",
            iterations=iterations,
            mean_us=mean_us,
            median_us=float(np.median(lat)),
            min_us=float(np.min(lat)),
            p95_us=float(np.percentile(lat, 95)),
            fps=1e6 / mean_us if mean_us > 0 else 0.0,
            intermediate_ddr_bytes=int(dummy_in.nbytes * 2),
            init_us=0.0,
        )

    def close(self):
        """Releases hardware resources (the NPU hardware context and buffer objects); safe to call again."""
        if self.native is not None:
            self.native = None
            self._native_finalizer()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
