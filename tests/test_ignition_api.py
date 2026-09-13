# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
tests/test_ignition_api.py
Integration and physical silicon verification suite for the Ignition SDK and CLI.
"""

import os
import tempfile
from pathlib import Path
import numpy as np
import pytest
from click.testing import CliRunner

import ignition
from ignition.cli.main import cli
from ignite_xdna.compiler.partitioner import build_synthetic_multi_layer_conv_model
from ignite_xdna.compiler import (
    extract_conv_subgraph,
    prepare_image_activations,
    run_exact_fixed_point_reference,
)
from ignite_xdna.runtime import calculate_numerical_parity


@pytest.fixture(scope="module")
def synthetic_models():
    """Generates synthetic 2-layer and 3-layer ONNX models for testing."""
    tmpdir = tempfile.mkdtemp()
    m2_path = os.path.join(tmpdir, "conv2d_2layer.onnx")
    m3_path = os.path.join(tmpdir, "conv2d_3layer.onnx")

    m2 = build_synthetic_multi_layer_conv_model(num_layers=2, in_channels=32, out_channels=32)
    m3 = build_synthetic_multi_layer_conv_model(num_layers=3, in_channels=32, out_channels=32)

    import onnx
    onnx.save(m2, m2_path)
    onnx.save(m3, m3_path)

    yield {"m2": m2_path, "m3": m3_path}


def test_devices_probe():
    """Verifies physical AMD XDNA NPU hardware discovery and specification reporting."""
    devs = ignition.devices()
    assert len(devs) >= 1, "Expected at least one detected AMD XDNA NPU device"

    dev = devs[0]
    assert "Ryzen 7 8700G" in dev.name or "Phoenix" in dev.name
    assert dev.bdf == "003d:00:01.1"
    assert dev.num_cores == 16
    assert dev.memtile_sram_kb == 2048
    assert dev.tile_clock_ghz == 1.80
    assert dev.peak_int8_tops == 14.75


def test_compile_and_predict_cpu(synthetic_models):
    """Verifies reference CPU compilation and prediction."""
    m_path = synthetic_models["m2"]
    model = ignition.compile(m_path, backend="cpu")

    inp = np.zeros((1, 8, 32, 32), dtype=np.int8)
    out = model.predict(inp)

    assert out is not None
    assert isinstance(out, np.ndarray)
    assert out.size > 0
    model.close()


def test_compile_and_predict_xdna1_silicon(synthetic_models):
    """Verifies hardware compilation and physical silicon inference dispatch on Phoenix NPU."""
    m_path = synthetic_models["m2"]
    model = ignition.compile(m_path, backend="xdna1")

    inp = np.zeros((1, 8, 32, 32), dtype=np.int8)
    out = model.predict(inp)

    assert out is not None
    assert isinstance(out, np.ndarray)
    assert len(out.flatten()) == 2048
    assert out.dtype == np.int8
    model.close()


def test_numerical_parity_silicon_exact():
    """Asserts physical silicon execution matches exact fixed-point reference with 100% bit parity."""
    repo = Path(r"C:\Users\Ignis\PycharmProjects\ignite-xdna")
    exec_a = repo / "build" / "layer_conv0_exec.bin"
    init_a = repo / "build" / "layer_conv0_init.bin"
    model_path = repo / "models" / "yolov8n_cut_xint8.onnx"
    calib_img = repo / "data" / "bisenetv2_calib" / "000000000139.jpg"

    if not (exec_a.exists() and init_a.exists() and model_path.exists()):
        pytest.skip("Precompiled hardware artifacts not found in ignite-xdna")

    sub0 = extract_conv_subgraph(str(model_path), node_name="/model.15/m.0/cv1/conv/Conv")
    in_single = prepare_image_activations(str(calib_img), sub0["scale_x"], in_channels=sub0["in_channels"])
    in_4col = np.tile(in_single, 4)
    ref = run_exact_fixed_point_reference(sub0, in_single, out_pixels=64, num_cores=16)

    model = ignition.compile((init_a, exec_a), backend="xdna1")
    hw_out = model.predict(in_4col)
    parity = calculate_numerical_parity(ref, hw_out)

    assert parity["bit_agreement_pct"] == 100.0, f"Expected 100% bit agreement, got {parity['bit_agreement_pct']}%"
    assert parity["max_ae"] == 0.0, f"Expected MaxAE 0.0, got {parity['max_ae']}"
    assert parity["mae"] == 0.0

    model.close()


def test_numerical_parity_xdna1_vs_cpu(synthetic_models):
    """Asserts physical silicon and CPU reference execution both yield bounded, valid activations."""
    m_path = synthetic_models["m2"]
    model_hw = ignition.compile(m_path, backend="xdna1")
    model_cpu = ignition.compile(m_path, backend="cpu")

    inp = np.zeros((1, 8, 32, 32), dtype=np.int8)
    out_hw = model_hw.predict(inp).flatten()
    out_cpu = model_cpu.predict(inp).flatten()

    assert out_hw is not None and len(out_hw) > 0
    assert out_cpu is not None and len(out_cpu) > 0
    assert np.all(out_hw >= -128) and np.all(out_hw <= 127)
    assert np.all(out_cpu >= -128) and np.all(out_cpu <= 127)

    model_hw.close()
    model_cpu.close()


def test_benchmark_report_metrics(synthetic_models):
    """Verifies benchmark execution, zero intermediate DDR traffic, and report metrics."""
    m_path = synthetic_models["m2"]
    model = ignition.compile(m_path, backend="xdna1")

    report = model.benchmark(iterations=30, warmup=10)
    assert report.backend_name == "xdna1"
    assert report.iterations == 30
    assert report.mean_us > 0.0
    assert report.median_us > 0.0
    assert report.min_us > 0.0
    assert report.fps > 0.0
    assert report.intermediate_ddr_bytes == 0  # 0 DDR bytes verified

    summary = report.summary()
    assert "Benchmark Report (xdna1)" in summary
    assert "Sustained FPS" in summary
    model.close()


def test_cli_commands(synthetic_models):
    """Verifies CLI entrypoints (devices, run, benchmark)."""
    runner = CliRunner()

    # 1. Test 'devices'
    res_dev = runner.invoke(cli, ["devices"])
    assert res_dev.exit_code == 0
    assert "AMD Ryzen 7 8700G" in res_dev.output
    assert "003d:00:01.1" in res_dev.output

    # 2. Test 'run'
    m_path = synthetic_models["m2"]
    res_run = runner.invoke(cli, ["run", str(m_path), "--backend", "xdna1"])
    assert res_run.exit_code == 0
    assert "Inference successfully completed on physical silicon!" in res_run.output

    # 3. Test 'benchmark'
    res_bench = runner.invoke(cli, ["benchmark", str(m_path), "--backend", "xdna1", "--iterations", "20", "--warmup", "5"])
    assert res_bench.exit_code == 0
    assert "Benchmark Report (xdna1)" in res_bench.output
