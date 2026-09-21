# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Ignition: High-performance ergonomic inference engine for AMD Ryzen AI XDNA1 NPUs.
"""

from typing import Union, Optional, List, Any
from pathlib import Path

from .model import Model, Runner
from .backends.base import BenchmarkReport, BaseBackend
from .devices import probe_devices, get_default_device, DeviceInfo
from .pipelines.yolo import YOLOPipeline, Detection, YOLOResult, is_ignite_container
from .pipelines.streaming import AsyncYOLOPipeline

__version__ = "0.3.4"


def compile(
    model: Union[str, Path, Any],
    backend: str = "xdna1",
    precision: str = "int8",
    pipeline: Optional[str] = None,
    **kwargs
) -> Union[Model, YOLOPipeline, AsyncYOLOPipeline]:
    """
    Compiles an ONNX model or partitioned graph for high-performance execution.

    Args:
        model: Path to ONNX model, ignite-xdna .ignite container, ONNX ModelProto, or PartitionedGraph.
        backend: Execution target ('xdna1' for AMD Phoenix NPU, 'cpu' for ORT CPU reference).
        precision: Arithmetic precision ('int8' stationary vector layout).
        pipeline: Optional specialized task pipeline ('yolo', 'async_yolo', 'segment', 'matte').
        **kwargs: Additional backend or pipeline configuration options.

    Returns:
        Model, YOLOPipeline, or AsyncYOLOPipeline.
    """
    if pipeline in ("yolo", "sync_yolo"):
        return YOLOPipeline(model, backend=backend, **kwargs)
    elif pipeline in ("segment", "matte"):
        from .pipelines.dense import DensePipeline
        from .pipelines.yolo import NATIVE_BACKENDS
        if is_ignite_container(model) and backend.lower() not in NATIVE_BACKENDS:
            raise ValueError("an .ignite container requires the NPU backend")
        return DensePipeline(model, pipeline, **kwargs)
    elif pipeline in ("async_yolo", "streaming_yolo", "stream_yolo", "streaming"):
        if is_ignite_container(model):
            # An .ignite container runs one synchronous NPU dispatch per frame; YOLOPipeline.stream()
            # yields its results in order without the ONNX Runtime stages of AsyncYOLOPipeline.
            return YOLOPipeline(model, backend=backend, **kwargs)
        return AsyncYOLOPipeline(model, backend=backend, **kwargs)
    return Model(model, backend=backend, precision=precision, **kwargs)


def devices() -> List[DeviceInfo]:
    """
    Probes system for available physical AMD Phoenix/Hawk Point XDNA1 NPUs.

    Returns:
        List[DeviceInfo]: Detected hardware accelerators and device attributes.
    """
    return probe_devices()


__all__ = [
    "compile",
    "Model",
    "Runner",
    "YOLOPipeline",
    "AsyncYOLOPipeline",
    "Detection",
    "YOLOResult",
    "is_ignite_container",
    "devices",
    "probe_devices",
    "get_default_device",
    "DeviceInfo",
    "BenchmarkReport",
    "BaseBackend",
    "__version__",
]
