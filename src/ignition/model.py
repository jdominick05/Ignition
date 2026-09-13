# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
src/ignition/model.py
High-level model lifecycle container and runner abstractions for Ignition.
"""

from typing import Union, Optional, Any, Tuple
from pathlib import Path
import numpy as np

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

from .backends.base import BaseBackend, BenchmarkReport
from .backends.xdna1 import XDNA1Backend
from .backends.cpu import CPUBackend


class Model:
    """
    High-level model container representing a compiled neural network on Ignition.
    Manages input transformation, device dispatch, and performance profiling.
    """

    def __init__(
        self,
        model_path: Union[str, Path, Any],
        backend: Union[str, BaseBackend] = "xdna1",
        precision: str = "int8",
        **kwargs
    ):
        self.model_path = model_path
        self.precision = precision.lower()
        self.kwargs = kwargs

        # 1. Resolve backend
        if isinstance(backend, str):
            b_lower = backend.lower()
            if b_lower in ("xdna1", "npu", "aie2"):
                self.backend: BaseBackend = XDNA1Backend(
                    device_id=kwargs.get("device_id", 0),
                    num_cores=kwargs.get("num_cores", 16)
                )
            elif b_lower in ("cpu", "reference", "ort"):
                self.backend = CPUBackend()
            else:
                raise ValueError(f"Unknown backend: {backend}. Supported: 'xdna1', 'cpu'")
        elif isinstance(backend, BaseBackend):
            self.backend = backend
        else:
            raise TypeError(f"Invalid backend type: {type(backend)}")

        self.backend_name = backend.lower() if isinstance(backend, str) else getattr(backend, "name", "xdna1")

        # 2. Load model into backend
        self.backend.load(model_path, **kwargs)

    def _is_yolo_model(self) -> bool:
        """Determines if the loaded model is a YOLO architecture."""
        return "yolo" in str(self.model_path).lower()

    def predict(
        self,
        input_data: Union[str, Path, Any, np.ndarray],
        input_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Executes forward inference on input data.
        Automatically handles preprocessing and device execution.
        """
        # 1. Preprocess input
        tensor = self._preprocess(input_data, input_size=input_size)

        # 2. Dispatch to backend
        return self.backend.run(tensor)

    def stream(
        self,
        frame_iterator: Any,
        conf_thres: Optional[float] = None,
        iou_thres: Optional[float] = None,
        annotate: bool = False,
        **kwargs
    ):
        """
        Pipelined streaming inference over an input frame iterator.
        Yields predictions or YOLOResult in real time as each frame completes.
        """
        if self._is_yolo_model():
            from .pipelines.streaming import AsyncYOLOPipeline
            async_pipe = AsyncYOLOPipeline(
                model_path=self.model_path,
                backend=self.backend_name,
                conf_thres=conf_thres or 0.25,
                iou_thres=iou_thres or 0.45,
            )
            try:
                yield from async_pipe.stream(
                    frame_iterator, conf_thres=conf_thres, iou_thres=iou_thres, annotate=annotate
                )
            finally:
                async_pipe.close()
            return

        for item in frame_iterator:
            yield self.predict(item, **kwargs)

    def benchmark(
        self,
        input_data: Optional[Union[str, Path, Any, np.ndarray]] = None,
        warmup: int = 20,
        iterations: int = 500
    ) -> BenchmarkReport:
        """
        Profiles sustained execution latency percentiles and throughput on physical hardware.

        Args:
            input_data: Optional sample input data.
            warmup: Number of initial untimed executions.
            iterations: Number of timed executions.

        Returns:
            BenchmarkReport: Latency percentiles (min, median, p95), FPS, and DDR traffic metrics.
        """
        tensor = None
        if input_data is not None:
            tensor = self._preprocess(input_data)

        return self.backend.benchmark(
            input_tensor=tensor,
            warmup=warmup,
            iterations=iterations,
        )

    def _preprocess(
        self,
        data: Union[str, Path, Any, np.ndarray],
        input_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """Preprocesses inputs into clean tensor arrays."""
        # Check if file path
        if isinstance(data, (str, Path)):
            p = Path(data)
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
                if not _HAS_PIL:
                    raise ImportError("Pillow is required for image file inference. Run 'pip install pillow'.")
                data = Image.open(p)
            elif p.suffix.lower() == ".npy":
                data = np.load(str(p))

        # Check if PIL Image
        if _HAS_PIL and isinstance(data, Image.Image):
            img = data.convert("RGB")
            if input_size is not None:
                img = img.resize((input_size[1], input_size[0]))
            arr = np.array(img, dtype=np.float32)
            # Standard HWC -> CHW -> NCHW
            arr = np.transpose(arr, (2, 0, 1))
            arr = np.expand_dims(arr, axis=0)
            if self.precision == "int8":
                # Normalize and quantize to int8 [-128, 127]
                arr = np.clip(arr - 128.0, -128, 127).astype(np.int8)
            return arr

        # NumPy Array
        if isinstance(data, np.ndarray):
            arr = data
            if self.precision == "int8" and arr.dtype != np.int8:
                if np.issubdtype(arr.dtype, np.floating):
                    arr = np.clip(arr, -128, 127).astype(np.int8)
                else:
                    arr = arr.astype(np.int8)
            return arr

        raise TypeError(f"Unsupported input data type: {type(data)}")

    def close(self):
        """Releases underlying backend resources."""
        self.backend.teardown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class Runner:
    """Convenience runner for streaming batch inference pipelines."""

    def __init__(self, model: Model):
        self.model = model

    def __call__(self, *args, **kwargs) -> np.ndarray:
        return self.model.predict(*args, **kwargs)
