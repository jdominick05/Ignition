# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
src/ignition/backends/__init__.py
Backend implementations for Ignition.
"""

from .base import BaseBackend, BenchmarkReport
from .xdna1 import XDNA1Backend
from .cpu import CPUBackend

__all__ = [
    "BaseBackend",
    "BenchmarkReport",
    "XDNA1Backend",
    "CPUBackend",
]
