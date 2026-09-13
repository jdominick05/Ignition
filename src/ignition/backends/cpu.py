# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
src/ignition/backends/cpu.py
Reference CPU backend using ONNX Runtime CPUExecutionProvider for parity assertions and baselines.
"""

import time
from typing import Optional, Dict, Any, Union
from pathlib import Path
import numpy as np
import onnxruntime as ort

from .base import BaseBackend, BenchmarkReport


class CPUBackend(BaseBackend):
    """
    Reference inference backend executing via ONNX Runtime CPUExecutionProvider.
    Used for parity validation, accuracy baselines, and fallback execution.
    """

    def __init__(self):
        super().__init__(backend_name="cpu")
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.input_shape: Optional[list] = None
        self.input_type: Optional[str] = None
        self.output_names: list = []

    def load(self, model_path: Union[str, Path, Any], **kwargs):
        """Loads ONNX model into ONNX Runtime CPU session."""
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = kwargs.get("threads", 4)
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        if hasattr(model_path, "SerializeToString"):
            # ONNX ModelProto
            model_bytes = model_path.SerializeToString()
            self.session = ort.InferenceSession(
                model_bytes,
                sess_options=opts,
                providers=["CPUExecutionProvider"]
            )
        else:
            p = str(model_path)
            self.session = ort.InferenceSession(
                p,
                sess_options=opts,
                providers=["CPUExecutionProvider"]
            )

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_shape = inp.shape
        self.input_type = inp.type
        self.output_names = [out.name for out in self.session.get_outputs()]

    def run(self, input_tensor: np.ndarray) -> np.ndarray:
        """Executes reference CPU inference."""
        if self.session is None:
            raise RuntimeError("Model is not loaded. Call load() before run().")

        # Handle tensor shape adjustments
        cur_tensor = input_tensor
        if self.input_shape is not None:
            expected_shape = [d if isinstance(d, int) and d > 0 else 1 for d in self.input_shape]
            expected_size = int(np.prod(expected_shape))
            flat = cur_tensor.flatten()
            if flat.size >= expected_size:
                cur_tensor = flat[:expected_size].reshape(expected_shape)
            else:
                padded = np.zeros(expected_size, dtype=flat.dtype)
                padded[:flat.size] = flat
                cur_tensor = padded.reshape(expected_shape)

        # Match expected input type if needed
        if "float" in str(self.input_type) and cur_tensor.dtype != np.float32:
            cur_tensor = cur_tensor.astype(np.float32)
        elif "int8" in str(self.input_type) and cur_tensor.dtype != np.int8:
            cur_tensor = cur_tensor.astype(np.int8)

        outputs = self.session.run(self.output_names, {self.input_name: cur_tensor})
        return outputs[0]

    def benchmark(
        self,
        input_tensor: Optional[np.ndarray] = None,
        warmup: int = 20,
        iterations: int = 200
    ) -> BenchmarkReport:
        """Profiles sustained CPU execution."""
        if self.session is None:
            raise RuntimeError("Model is not loaded. Call load() before benchmark().")

        if input_tensor is None:
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in (self.input_shape or [1, 8, 32, 32])]
            dtype = np.float32 if "float" in str(self.input_type) else np.int8
            input_tensor = np.zeros(shape, dtype=dtype)

        # Warmup
        for _ in range(warmup):
            self.run(input_tensor)

        # Benchmark
        latencies_us = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            self.run(input_tensor)
            t1 = time.perf_counter()
            latencies_us.append((t1 - t0) * 1e6)

        mean_us = float(np.mean(latencies_us))
        median_us = float(np.median(latencies_us))
        min_us = float(np.min(latencies_us))
        p95_us = float(np.percentile(latencies_us, 95))
        fps = 1e6 / mean_us if mean_us > 0 else 0.0

        return BenchmarkReport(
            backend_name="cpu",
            iterations=iterations,
            mean_us=mean_us,
            median_us=median_us,
            min_us=min_us,
            p95_us=p95_us,
            fps=fps,
            intermediate_ddr_bytes=0,
            init_us=0.0,
        )

    def teardown(self):
        self.session = None
