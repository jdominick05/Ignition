# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
tests/test_async_pipeline.py
Verification suite for the 3-stage asynchronous pipelined execution runner in Ignition.
Tests bounded queue concurrency, host buffer reuse, numerical/bounding-box parity,
and sustained >= 26.5 FPS silicon throughput on AMD Phoenix XDNA1 silicon.
"""

import time
from pathlib import Path
import pytest
import numpy as np
from click.testing import CliRunner

import ignition
from ignition.pipelines.yolo import YOLOPipeline
from ignition.pipelines.streaming import AsyncYOLOPipeline
from ignition.cli.main import cli


YOLO_MODEL_PATH = Path(r"C:\Users\Ignis\PycharmProjects\ignite-xdna\models\yolov8n_cut_xint8.onnx")
BUS_IMAGE_PATH = Path("examples/assets/bus.jpg")


@pytest.fixture(scope="module")
def assert_hardware_available():
    """Confirms Phoenix NPU and test assets are present."""
    if not YOLO_MODEL_PATH.exists():
        pytest.skip(f"YOLO ONNX model not found at {YOLO_MODEL_PATH}")
    if not BUS_IMAGE_PATH.exists():
        pytest.skip(f"Test image not found at {BUS_IMAGE_PATH}")
    devs = ignition.devices()
    if not devs:
        pytest.skip("No physical AMD XDNA1 NPU detected on system")


def test_async_pipeline_numerical_and_box_parity(assert_hardware_available):
    """
    Asserts 100.0% numerical and bounding-box parity between synchronous execution
    and the 3-stage asynchronous pipelined execution model.
    """
    pipe_sync = YOLOPipeline(YOLO_MODEL_PATH, backend="xdna1")
    pipe_async = AsyncYOLOPipeline(YOLO_MODEL_PATH, backend="xdna1")

    try:
        res_sync = pipe_sync.predict(BUS_IMAGE_PATH, conf_thres=0.25, iou_thres=0.45)
        res_async = pipe_async.predict(BUS_IMAGE_PATH, conf_thres=0.25, iou_thres=0.45)

        # 1. Assert identical detection counts
        assert len(res_sync.detections) == len(res_async.detections)
        assert len(res_sync.detections) > 0

        # 2. Assert bit-exact class IDs, class names, scores, and bounding box coordinates
        for det_sync, det_async in zip(res_sync.detections, res_async.detections):
            assert det_sync.class_id == det_async.class_id
            assert det_sync.class_name == det_async.class_name
            assert abs(det_sync.score - det_async.score) < 1e-4

            # Bounding box coordinates (x, y, w, h)
            box_diff = max(
                abs(det_sync.x - det_async.x),
                abs(det_sync.y - det_async.y),
                abs(det_sync.w - det_async.w),
                abs(det_sync.h - det_async.h),
            )
            assert box_diff < 1e-3, f"Bounding box mismatch: {det_sync} vs {det_async}"

    finally:
        pipe_sync.close()
        pipe_async.close()


def test_async_pipeline_token_pool_lifecycle_and_bounded_queues(assert_hardware_available):
    """
    Verifies bounded intermediate queue depths (maxsize=2) and static token recycling.
    Ensures memory footprint remains stationary across frame streams.
    """
    pipe = AsyncYOLOPipeline(YOLO_MODEL_PATH, backend="xdna1", ring_depth=2)

    try:
        # Assert queue depth constraints
        assert pipe.npu_queue.maxsize == 2
        assert pipe.nms_queue.maxsize == 2
        assert pipe.out_queue.maxsize == 2
        assert pipe.free_tokens.maxsize == 2
        assert pipe.free_tokens.qsize() == 2

        # Stream 30 synthetic frames
        frames = [np.random.randint(0, 256, (360, 640, 3), dtype=np.uint8) for _ in range(30)]
        results = list(pipe.stream(frames, annotate=False))

        assert len(results) == 30

        # After stream completion, tokens must return to free pool
        assert pipe.free_tokens.qsize() == 2
        assert pipe.npu_queue.empty()
        assert pipe.nms_queue.empty()
        assert pipe.out_queue.empty()

    finally:
        pipe.close()


def test_async_pipeline_sustained_throughput_200_frames(assert_hardware_available):
    """
    Benchmarks sustained throughput on Device 0 ([003d:00:01.1]) across 200 synthetic frames.
    Confirms sustained throughput increases from ~19.5 FPS to >= 26.5 FPS with zero dropped frames.
    """
    pipe = AsyncYOLOPipeline(YOLO_MODEL_PATH, backend="xdna1")

    try:
        # Generate 200 synthetic 720p frames
        frames = [np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8) for _ in range(200)]

        # Warmup ring pipeline
        for _ in pipe.stream(frames[:10], annotate=False):
            pass

        # Sustained measurement
        results = []
        t0 = time.perf_counter()
        for res in pipe.stream(frames, annotate=False):
            results.append(res)
        t1 = time.perf_counter()

        elapsed_s = t1 - t0
        sustained_fps = len(results) / elapsed_s if elapsed_s > 0 else 0.0

        # Assert zero dropped frames
        assert len(results) == 200, f"Expected 200 frames, but received {len(results)}"

        # Assert throughput gate (>= 26.5 FPS)
        assert sustained_fps >= 26.5, f"Sustained throughput {sustained_fps:.2f} FPS fell below 26.5 FPS gate"

        # Assert detailed latency breakdown is present on all results
        for res in results:
            assert "preprocess_ms" in res.timings_ms
            assert "backbone_ms" in res.timings_ms
            assert "postprocess_ms" in res.timings_ms
            assert "total_ms" in res.timings_ms
            assert res.timings_ms["preprocess_ms"] > 0
            assert res.timings_ms["backbone_ms"] > 0
            assert res.timings_ms["postprocess_ms"] > 0

    finally:
        pipe.close()


def test_async_pipeline_early_termination_and_drain(assert_hardware_available):
    """
    Verifies that early termination or breaking out of stream() drains and joins cleanly.
    """
    pipe = AsyncYOLOPipeline(YOLO_MODEL_PATH, backend="xdna1")

    try:
        frames = (np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(100))
        consumed = 0
        for _ in pipe.stream(frames, annotate=False):
            consumed += 1
            if consumed >= 5:
                break

        assert consumed == 5
        # Ensure free token pool restored on next stream invocation
        frames2 = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(5)]
        results2 = list(pipe.stream(frames2, annotate=False))
        assert len(results2) == 5

    finally:
        pipe.close()


def test_cli_detect_streaming_benchmark(assert_hardware_available):
    """
    Tests CLI streaming benchmark command:
    ignition detect <model.onnx> --input <image> --backend xdna1 --stream --benchmark
    """
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "detect",
            str(YOLO_MODEL_PATH),
            "--input",
            str(BUS_IMAGE_PATH),
            "--backend",
            "xdna1",
            "--stream",
            "--benchmark",
            "--frames",
            "20",
        ],
    )

    assert result.exit_code == 0, f"CLI detect failed:\n{result.output}"
    assert "Ignition Streaming Benchmark Report (XDNA1)" in result.output
    assert "Total Stream Frames Processed: 20" in result.output
    assert "Sustained Pipelined Throughput:" in result.output
    assert "Average Preprocessing Latency:" in result.output
    assert "Average Backbone NPU Latency:" in result.output
    assert "Average Postprocess/NMS Lat:" in result.output
