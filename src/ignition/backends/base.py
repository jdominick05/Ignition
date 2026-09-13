# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
src/ignition/backends/base.py
Abstract base class and execution reports for Ignition inference backends.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union
from pathlib import Path
import numpy as np


@dataclass
class BenchmarkReport:
    """Benchmark performance report capturing sustained latency and throughput."""
    backend_name: str
    iterations: int
    mean_us: float
    median_us: float
    min_us: float
    p95_us: float
    fps: float
    intermediate_ddr_bytes: int = 0
    init_us: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend_name,
            "iterations": self.iterations,
            "mean_latency_us": self.mean_us,
            "median_latency_us": self.median_us,
            "min_latency_us": self.min_us,
            "p95_latency_us": self.p95_us,
            "throughput_fps": self.fps,
            "intermediate_ddr_bytes": self.intermediate_ddr_bytes,
            "init_latency_us": self.init_us,
        }

    def summary(self) -> str:
        return (
            f"=== Benchmark Report ({self.backend_name}) ===\n"
            f"  Iterations:          {self.iterations}\n"
            f"  Mean Latency:        {self.mean_us:.2f} μs\n"
            f"  Median Latency:      {self.median_us:.2f} μs\n"
            f"  Min Latency:         {self.min_us:.2f} μs\n"
            f"  P95 Latency:         {self.p95_us:.2f} μs\n"
            f"  Sustained FPS:       {self.fps:.1f} inferences/sec\n"
            f"  Intermediate DDR:    {self.intermediate_ddr_bytes} Bytes"
        )

    def __str__(self) -> str:
        return self.summary()


class BaseBackend(ABC):
    """Abstract interface for all hardware and reference inference backends."""

    def __init__(self, backend_name: str):
        self.backend_name = backend_name

    @abstractmethod
    def load(self, model_path: Union[str, Path, Any], **kwargs):
        """Loads and prepares the model for execution."""
        pass

    @abstractmethod
    def run(self, input_tensor: np.ndarray) -> np.ndarray:
        """Executes a single synchronous inference and returns unswizzled output."""
        pass

    @abstractmethod
    def benchmark(
        self,
        input_tensor: Optional[np.ndarray] = None,
        warmup: int = 20,
        iterations: int = 200
    ) -> BenchmarkReport:
        """Profiles sustained execution over warmup and benchmark iterations."""
        pass

    @abstractmethod
    def teardown(self):
        """Releases all allocated hardware buffers, contexts, and file handles."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.teardown()
