#!/usr/bin/env python3
# Copyright (C) 2026 The Ignition contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
r"""Live inference on the AMD Phoenix NPU (XDNA1) or ONNX Runtime from a webcam, video or image.

    python live_ignition.py                                # YOLOv8n on webcam 0 in a window
    python live_ignition.py --headless --frames 300        # benchmark: G2G mean, P50, P95, P99
    python live_ignition.py --help                         # every flag

The app is the ignition.live module; its docstring says how latency is measured. This launcher runs
this checkout's own src, and sets OpenCV's Media Foundation option before anything imports cv2.
"""
import os

# OpenCV reads this when video I/O initialises, so it must precede every cv2 import (including the one
# inside ignition). Without it the Media Foundation backend can take ~90 s to open.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ignition.live import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
