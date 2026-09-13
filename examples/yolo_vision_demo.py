# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
examples/yolo_vision_demo.py
Standalone visual object detection demo using the Ignition high-performance engine
on AMD Phoenix XDNA1 / AIE2 silicon.
"""

import sys
import os
import argparse
from pathlib import Path
import urllib.request

import ignition


SAMPLE_IMAGE_URL = "https://ultralytics.com/images/bus.jpg"


def resolve_model_path(custom_path: str = None) -> Path:
    """Finds available YOLOv8n model checkpoint."""
    if custom_path and os.path.exists(custom_path):
        return Path(custom_path)

    candidates = [
        Path("models/yolov8n_cut_xint8.onnx"),
        Path(r"C:\Users\Ignis\PycharmProjects\ignite-xdna\models\yolov8n_cut_xint8.onnx"),
        Path(r"C:\Users\Ignis\PycharmProjects\ignite-xdna\models\yolov8n_xint8.onnx"),
        Path("models/yolov8n_xint8.onnx"),
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()

    raise FileNotFoundError(
        "Could not locate YOLOv8n INT8 model. Specify path with --model <path/to/yolov8n_cut_xint8.onnx>"
    )


def resolve_image_path(custom_path: str = None) -> Path:
    """Resolves demo image, downloading official test asset if needed."""
    if custom_path and os.path.exists(custom_path):
        return Path(custom_path)

    default_asset = Path(__file__).parent / "assets" / "bus.jpg"
    default_asset.parent.mkdir(parents=True, exist_ok=True)

    if not default_asset.exists():
        print(f"Downloading sample image from {SAMPLE_IMAGE_URL} -> {default_asset}...")
        try:
            urllib.request.urlretrieve(SAMPLE_IMAGE_URL, str(default_asset))
            print(f"Downloaded successfully ({default_asset.stat().st_size} bytes).")
        except Exception as e:
            print(f"Warning: Failed to download online asset ({e}). Using local placeholder if present.")

    if not default_asset.exists():
        raise FileNotFoundError(f"Input image not found: {default_asset}")

    return default_asset.resolve()


def main():
    parser = argparse.ArgumentParser(
        description="Ignition YOLOv8n Real-Time Vision Demo on AMD Phoenix Silicon"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to INT8 QDQ YOLOv8n ONNX model",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image (default: examples/assets/bus.jpg)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="examples/assets/yolo_output.jpg",
        help="Output path for annotated detection image",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="xdna1",
        choices=["xdna1", "cpu"],
        help="Execution target backend (xdna1 for Phoenix NPU, cpu for reference)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence score threshold (default: 0.25)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold (default: 0.45)",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  IGNITION HIGH-PERFORMANCE VISION ENGINE - AMD PHOENIX AIE2")
    print("=" * 65)

    # 1. Hardware Discovery
    devices = ignition.devices()
    if devices:
        dev = devices[0]
        print(f"Hardware Target:    {dev.name} ({dev.bdf})")
        print(f"AIE2 Compute Array: {dev.num_cores} Cores @ {dev.tile_clock_ghz:.2f} GHz")
        print(f"On-Chip L2 Memory:  {dev.memtile_sram_kb} KB MemTile SRAM")
        print(f"Peak Compute:       {dev.peak_int8_tops:.2f} INT8 TOPS")
    else:
        print("Hardware Target:    Standard Host Processor")

    # 2. Resolve Paths
    model_path = resolve_model_path(args.model)
    image_path = resolve_image_path(args.image)
    out_path = Path(args.output).resolve()

    print(f"Selected Model:     {model_path}")
    print(f"Input Asset:        {image_path}")
    print(f"Backend Target:     {args.backend.upper()}")
    print("-" * 65)

    # 3. Compile YOLOv8 Pipeline
    print(f"Compiling YOLOv8n pipeline on target '{args.backend}'...")
    pipeline = ignition.compile(
        model_path,
        backend=args.backend,
        pipeline="yolo",
        conf_thres=args.conf,
        iou_thres=args.iou,
    )

    try:
        # 4. Execute End-to-End Inference
        print(f"Executing detection pipeline on '{image_path.name}'...")
        result = pipeline.predict(image_path, conf_thres=args.conf, iou_thres=args.iou)

        # 5. Save and Display Results
        result.save(out_path)
        print()
        print(result.summary())
        print()
        print(f"Annotated visualization saved: {out_path}")
        print("=" * 65)

    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
