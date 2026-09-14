# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
src/ignition/pipelines/streaming.py
3-stage asynchronous runner for an ONNX YOLOv8 model on ONNX Runtime's CPU execution provider.
Overlaps letterbox preprocessing, inference and head decode/NMS in three threads. The NPU is not
used: an .ignite container runs through YOLOPipeline instead.
"""

from typing import List, Tuple, Dict, Any, Optional, Union, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
import os
import time
import queue
import threading
import cv2
import numpy as np
from PIL import Image

from .yolo import (
    INPUT_SIZE,
    STRIDES,
    REG_MAX,
    Detection,
    YOLOResult,
    letterbox,
    decode_heads,
    postprocess_detections,
    draw_detections,
    is_ignite_container,
    note_onnx_runs_on_cpu,
)
from ..backends.base import BenchmarkReport


_SENTINEL = object()


@dataclass
class FrameToken:
    """
    Static pre-allocated buffer token for zero-allocation ring pipelining.
    Cycles ownership between Stage 1 (Preprocess), Stage 2 (NPU), and Stage 3 (NMS).
    """
    token_id: int
    canvas: np.ndarray = field(default_factory=lambda: np.empty((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8))
    inp_tensor: np.ndarray = field(default_factory=lambda: np.empty((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32))

    # Stage 1 metadata
    frame_idx: int = 0
    orig_shape: Tuple[int, int] = (0, 0)
    pad: Tuple[int, int] = (0, 0)
    scale: float = 1.0
    img_bgr: Optional[np.ndarray] = None
    t1_start: float = 0.0
    t1_end: float = 0.0

    # Stage 2 metadata
    raw_outputs: Optional[List[np.ndarray]] = None
    t2_start: float = 0.0
    t2_end: float = 0.0

    # Stage 3 metadata
    t3_start: float = 0.0
    t3_end: float = 0.0
    result: Optional[YOLOResult] = None

    def reset(self):
        """Resets dynamic per-frame fields while retaining pre-allocated buffers."""
        self.frame_idx = 0
        self.orig_shape = (0, 0)
        self.pad = (0, 0)
        self.scale = 1.0
        self.img_bgr = None
        self.t1_start = 0.0
        self.t1_end = 0.0
        self.raw_outputs = None
        self.t2_start = 0.0
        self.t2_end = 0.0
        self.t3_start = 0.0
        self.t3_end = 0.0
        self.result = None


class AsyncYOLOPipeline:
    """
    3-Stage Asynchronous Pipelined YOLOv8 Object Detection Engine.

    Stage 1: Producer / Preprocess Thread
      - Ingests raw frames from iterator/source.
      - Executes in-place OpenCV letterbox resize, float normalization, and quant layout
        directly into static pinned host memory.
      - Pushes to npu_queue (bounded maxsize=2).

    Stage 2: Inference Thread
      - Pulls preprocessed buffers from npu_queue.
      - Runs the model on ONNX Runtime's CPU execution provider.
      - Pushes to nms_queue (bounded maxsize=2).

    Stage 3: Consumer / Postprocess Thread
      - Pulls egress buffers from nms_queue.
      - Vectorized DFL softmax decode, coordinate un-letterbox, and batched NMS.
      - Emits completed YOLOResult and returns token to free pool.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        backend: str = "xdna1",
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        device_id: int = 0,
        ring_depth: int = 2,
        **backend_kwargs
    ):
        self.model_path = Path(model_path)
        self.backend_name = backend.lower()
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device_id = device_id
        self.ring_depth = max(2, ring_depth)
        self._closed = False

        if is_ignite_container(self.model_path):
            raise ValueError(
                f"{self.model_path} is a bare-metal .ignite container: it runs one synchronous NPU "
                "dispatch per frame through YOLOPipeline (ignition.compile(model, pipeline='yolo')), "
                "not through the ONNX Runtime asynchronous pipeline")

        # 1. Initialize ONNX Runtime session for head feature extraction
        import onnxruntime as ort
        if hasattr(ort, "set_default_logger_severity"):
            ort.set_default_logger_severity(3)
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3
        sess_options.intra_op_num_threads = min(os.cpu_count() or 12, 12)
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

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

        # 3. Static Token Pool & Bounded Intermediate Queues
        # Strictly bounded maxsize=2 to prevent host memory bloat and L3 cache thrashing
        self.free_tokens: queue.Queue = queue.Queue(maxsize=self.ring_depth)
        for i in range(self.ring_depth):
            self.free_tokens.put(FrameToken(token_id=i))

        self.npu_queue: queue.Queue = queue.Queue(maxsize=self.ring_depth)
        self.nms_queue: queue.Queue = queue.Queue(maxsize=self.ring_depth)
        self.out_queue: queue.Queue = queue.Queue(maxsize=self.ring_depth)

        self._stop_event = threading.Event()

    def _stage1_preprocess_worker(self, frame_iterator: Iterable[Any], annotate: bool):
        """Stage 1: Ingests frames and performs in-place letterboxing into pinned buffers."""
        try:
            for idx, item in enumerate(frame_iterator):
                if self._stop_event.is_set():
                    break

                # Acquire pre-allocated token from static pool (blocks if pipeline full)
                token: FrameToken = self.free_tokens.get()
                token.reset()
                token.frame_idx = idx
                token.t1_start = time.perf_counter()

                # Ingest image to BGR numpy array
                if isinstance(item, (str, Path)):
                    img_bgr = cv2.imread(str(item))
                    if img_bgr is None:
                        raise FileNotFoundError(f"Failed to read image at path: {item}")
                elif isinstance(item, Image.Image):
                    rgb = np.array(item.convert("RGB"))
                    img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                elif isinstance(item, np.ndarray):
                    img_bgr = item
                else:
                    raise TypeError(f"Unsupported frame type: {type(item)}")

                token.orig_shape = (img_bgr.shape[0], img_bgr.shape[1])
                if annotate:
                    token.img_bgr = img_bgr.copy()

                # In-place accelerated letterbox resize and normalization
                _, token.pad, token.scale = letterbox(
                    img_bgr,
                    size=INPUT_SIZE,
                    canvas_buf=token.canvas,
                    out_tensor=token.inp_tensor,
                )
                token.t1_end = time.perf_counter()

                # Forward to Stage 2
                self.npu_queue.put(token)

        except Exception as exc:
            self.out_queue.put(exc)
        finally:
            self.npu_queue.put(_SENTINEL)

    def _stage2_inference_worker(self):
        """Stage 2: Runs the model on ONNX Runtime's CPU execution provider."""
        try:
            while not self._stop_event.is_set():
                item = self.npu_queue.get()
                if item is _SENTINEL:
                    self.nms_queue.put(_SENTINEL)
                    break
                if isinstance(item, Exception):
                    self.nms_queue.put(item)
                    break

                token: FrameToken = item
                token.t2_start = time.perf_counter()

                raw_outputs = self.ort_session.run(None, {self.input_name: token.inp_tensor})

                token.t2_end = time.perf_counter()
                token.raw_outputs = raw_outputs

                # Forward to Stage 3
                self.nms_queue.put(token)

        except Exception as exc:
            self.out_queue.put(exc)
        finally:
            pass

    def _stage3_nms_worker(self, conf: float, iou: float, annotate: bool):
        """Stage 3: Vectorized DFL decode and per-class batched NMS."""
        try:
            while not self._stop_event.is_set():
                item = self.nms_queue.get()
                if item is _SENTINEL:
                    self.out_queue.put(_SENTINEL)
                    break
                if isinstance(item, Exception):
                    self.out_queue.put(item)
                    break

                token: FrameToken = item
                token.t3_start = time.perf_counter()

                # DFL Softmax Head Decode
                raw_outputs = token.raw_outputs
                if len(raw_outputs) == 6:
                    decoded = decode_heads(raw_outputs, imgsz=INPUT_SIZE, conf_thres=conf)
                elif len(raw_outputs) == 1 and raw_outputs[0].ndim == 3 and raw_outputs[0].shape[1] == 84:
                    decoded = raw_outputs[0]
                else:
                    raise ValueError(f"Unexpected model outputs format: {len(raw_outputs)} tensors")

                # Batched per-class NMS
                detections = postprocess_detections(
                    decoded,
                    pad=token.pad,
                    scale=token.scale,
                    conf_thres=conf,
                    iou_thres=iou,
                )
                token.t3_end = time.perf_counter()

                # Latency metrics
                pre_ms = (token.t1_end - token.t1_start) * 1000.0
                back_ms = (token.t2_end - token.t2_start) * 1000.0
                post_ms = (token.t3_end - token.t3_start) * 1000.0
                total_ms = (token.t3_end - token.t1_start) * 1000.0

                timings = {
                    "preprocess_ms": pre_ms,
                    "backbone_ms": back_ms,
                    "postprocess_ms": post_ms,
                    "total_ms": total_ms,
                }

                annotated = None
                if annotate and token.img_bgr is not None:
                    annotated = draw_detections(token.img_bgr, detections)

                res = YOLOResult(
                    detections=detections,
                    timings_ms=timings,
                    orig_shape=token.orig_shape,
                    image_annotated=annotated,
                    head_source="onnxruntime",
                )

                # Return token to pool for reuse
                self.free_tokens.put(token)

                # Emit result
                self.out_queue.put(res)

        except Exception as exc:
            self.out_queue.put(exc)
        finally:
            pass

    def stream(
        self,
        frame_iterator: Iterable[Any],
        conf_thres: Optional[float] = None,
        iou_thres: Optional[float] = None,
        annotate: bool = False,
    ) -> Iterator[YOLOResult]:
        """
        Asynchronously streams frames through 3-stage concurrency model.
        Yields completed YOLOResult objects in real time.
        """
        if self._closed:
            raise RuntimeError("Cannot invoke stream() on a closed AsyncYOLOPipeline")

        conf = conf_thres if conf_thres is not None else self.conf_thres
        iou = iou_thres if iou_thres is not None else self.iou_thres

        self._stop_event.clear()

        # Drain any residual items in queues
        for q in (self.npu_queue, self.nms_queue, self.out_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

        # Spawn worker threads
        t1 = threading.Thread(
            target=self._stage1_preprocess_worker,
            args=(frame_iterator, annotate),
            daemon=True,
            name="Ignition-Preprocess-Stage1"
        )
        t2 = threading.Thread(
            target=self._stage2_inference_worker,
            daemon=True,
            name="Ignition-Inference-Stage2"
        )
        t3 = threading.Thread(
            target=self._stage3_nms_worker,
            args=(conf, iou, annotate),
            daemon=True,
            name="Ignition-NMS-Stage3"
        )

        t1.start()
        t2.start()
        t3.start()

        try:
            while True:
                item = self.out_queue.get()
                if item is _SENTINEL:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            self._stop_event.set()
            t1.join(timeout=3.0)
            t2.join(timeout=3.0)
            t3.join(timeout=3.0)

    def predict(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        conf_thres: Optional[float] = None,
        iou_thres: Optional[float] = None,
    ) -> YOLOResult:
        """Runs pipelined prediction on a single image."""
        results = list(self.stream([image], conf_thres=conf_thres, iou_thres=iou_thres, annotate=True))
        if not results:
            raise RuntimeError("Pipeline failed to produce detection result.")
        return results[0]

    def benchmark(
        self,
        warmup: int = 15,
        iterations: int = 200,
        synthetic_shape: Tuple[int, int, int] = (720, 1280, 3)
    ) -> BenchmarkReport:
        """
        Profiles sustained pipelined throughput (FPS) and latency on the CPU.
        """
        # Generate synthetic frames
        warmup_frames = [np.random.randint(0, 256, synthetic_shape, dtype=np.uint8) for _ in range(warmup)]
        bench_frames = [np.random.randint(0, 256, synthetic_shape, dtype=np.uint8) for _ in range(iterations)]

        # Warmup
        for _ in self.stream(warmup_frames, annotate=False):
            pass

        # Timed sustained benchmark
        latencies_us = []
        t0 = time.perf_counter()
        for res in self.stream(bench_frames, annotate=False):
            latencies_us.append(res.timings_ms["total_ms"] * 1000.0)
        t1 = time.perf_counter()

        elapsed_s = t1 - t0
        sustained_fps = len(bench_frames) / elapsed_s if elapsed_s > 0 else 0.0

        lat_arr = np.array(latencies_us)
        return BenchmarkReport(
            backend_name=self.backend_name,
            iterations=iterations,
            mean_us=float(np.mean(lat_arr)),
            median_us=float(np.median(lat_arr)),
            min_us=float(np.min(lat_arr)),
            p95_us=float(np.percentile(lat_arr, 95)),
            fps=sustained_fps,
            intermediate_ddr_bytes=0,
            init_us=0.0,
        )

    def close(self):
        """Stops the worker threads; safe to call again."""
        if not self._closed:
            self._stop_event.set()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
