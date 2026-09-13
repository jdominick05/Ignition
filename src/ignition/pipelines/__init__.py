# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
src/ignition/pipelines/__init__.py
End-to-end task-specific inference pipelines for computer vision and perception models.
"""

from .yolo import YOLOPipeline, Detection, YOLOResult

__all__ = ["YOLOPipeline", "Detection", "YOLOResult"]
