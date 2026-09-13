# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Ignition: High-performance ergonomic inference engine for AMD Ryzen AI XDNA1 NPUs.
"""

from typing import Union, Optional, List, Any
from pathlib import Path

from .model import Model, Runner
from .backends.base import BenchmarkReport, BaseBackend
from .devices import probe_devices, get_default_device, DeviceInfo
from .pipelines.yolo import YOLOPipeline, Detection, YOLOResult

__version__ = "0.1.0"


def compile(
    model: Union[str, Path, Any],
    backend: str = "xdna1",
    precision: str = "int8",
    pipeline: Optional[str] = None,
    **kwargs
) -> Union[Model, YOLOPipeline]:
    """
    Compiles an ONNX model or partitioned graph for high-performance execution.

    Args:
        model: Path to ONNX model, ONNX ModelProto, or PartitionedGraph.
        backend: Execution target ('xdna1' for AMD Phoenix NPU, 'cpu' for ORT CPU reference).
        precision: Arithmetic precision ('int8' stationary vector layout).
        pipeline: Optional specialized task pipeline (e.g. 'yolo').
        **kwargs: Additional backend or pipeline configuration options.

    Returns:
        Model or specialized pipeline (e.g. YOLOPipeline).
    """
    if pipeline == "yolo":
        return YOLOPipeline(model, backend=backend, **kwargs)
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
    "Detection",
    "YOLOResult",
    "devices",
    "probe_devices",
    "get_default_device",
    "DeviceInfo",
    "BenchmarkReport",
    "BaseBackend",
    "__version__",
]
