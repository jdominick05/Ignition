# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
src/ignition/pipelines/__init__.py
End-to-end task-specific inference pipelines for computer vision and perception models.
"""

from .yolo import YOLOPipeline, Detection, YOLOResult
from .vision import (
    TASKS,
    Classification,
    ClassificationPipeline,
    ClassificationResult,
    SuperResolutionPipeline,
    SuperResolutionResult,
    create_pipeline,
    infer_task,
)

__all__ = [
    "YOLOPipeline", "Detection", "YOLOResult",
    "ClassificationPipeline", "Classification", "ClassificationResult",
    "SuperResolutionPipeline", "SuperResolutionResult",
    "TASKS", "create_pipeline", "infer_task",
]
