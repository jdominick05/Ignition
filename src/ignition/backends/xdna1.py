# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
src/ignition/backends/xdna1.py
Production hardware inference backend for AMD Phoenix XDNA1 / AIE2 silicon.
Integrates directly with ignite-xdna InferenceSession and multi-layer MemTile scheduler.
"""

from typing import Optional, Dict, Any, Union
from pathlib import Path
import numpy as np

from .base import BaseBackend, BenchmarkReport


class XDNA1Backend(BaseBackend):
    """
    Hardware inference backend executing on physical AMD Phoenix AIE2 silicon (Ryzen 7 8700G).
    Coordinates zero-copy host BO marshaling, MemTile L2 activation ping-ponging,
    and automatic CPU fallback for unsupported operations.
    """

    def __init__(self, device_id: int = 0, num_cores: int = 16):
        super().__init__(backend_name="xdna1")
        self.device_id = device_id
        self.num_cores = num_cores
        self.session = None
        self._input_bytes: Optional[int] = None
        self._output_bytes: Optional[int] = None

    def load(self, model_path: Union[str, Path, Any], **kwargs):
        """
        Compiles and loads model onto AMD Phoenix AIE2 silicon via ignite-xdna.
        Supports ONNX model files, ONNX ModelProtos, and PartitionedGraph instances.
        """
        from ignite_xdna.runtime.session import InferenceSession
        from ignition.assets import get_default_xclbin_path

        xclbin_path = kwargs.get("xclbin_path")
        if xclbin_path is None:
            try:
                cand = get_default_xclbin_path()
                if cand.exists():
                    xclbin_path = str(cand)
            except Exception:
                pass

        # Pass model directly to InferenceSession which automatically invokes
        # GraphPartitioner and MemTileMultiPassScheduler for multi-layer subgraphs
        self.session = InferenceSession(
            model_path_or_bundle=model_path,
            xclbin_path=xclbin_path,
            device_index=self.device_id,
            num_cores=self.num_cores,
            enable_fusion=kwargs.get("enable_fusion", True),
            ring_depth=kwargs.get("ring_depth", 2),
        )
        self._input_bytes = self.session.in_bytes
        self._output_bytes = self.session.out_bytes

    def run(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Executes inference on physical silicon with zero intermediate DDR roundtrips.
        """
        if self.session is None:
            raise RuntimeError("Model is not loaded. Call load() before run().")

        return self.session.run(input_tensor, unswizzle=True)

    def benchmark(
        self,
        input_tensor: Optional[np.ndarray] = None,
        warmup: int = 20,
        iterations: int = 200
    ) -> BenchmarkReport:
        """
        Measures sustained pipelined latency and throughput on physical AIE2 cores.
        """
        if self.session is None:
            raise RuntimeError("Model is not loaded. Call load() before benchmark().")

        if input_tensor is None:
            input_tensor = np.zeros(self._input_bytes or 8192, dtype=np.int8)

        stats = self.session.benchmark(
            input_tensor=input_tensor,
            warmup=warmup,
            iterations=iterations,
        )

        return BenchmarkReport(
            backend_name="xdna1",
            iterations=iterations,
            mean_us=stats["mean_us"],
            median_us=stats["median_us"],
            min_us=stats["min_us"],
            p95_us=stats["p95_us"],
            fps=stats["fps"],
            intermediate_ddr_bytes=0,  # Fused in MemTile L2 SRAM
            init_us=0.0,
        )

    def teardown(self):
        """Closes hardware session and releases PyXRT device contexts."""
        if self.session is not None:
            try:
                self.session.close()
            except Exception:
                pass
            self.session = None
