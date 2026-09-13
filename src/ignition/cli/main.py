# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
src/ignition/cli/main.py
Command-line interface for the Ignition AI engine.
"""

import sys
import time
from pathlib import Path
import click
import numpy as np

import ignition


@click.group()
@click.version_option(version=ignition.__version__, prog_name="ignition")
def cli():
    """Ignition: Production-grade AMD Ryzen AI XDNA1 AIE2 Inference Engine."""
    pass


@cli.command("devices")
def list_devices():
    """Probe and display detected AMD XDNA NPU hardware status."""
    click.echo("================================================================================")
    click.echo("IGNITION HARDWARE ACCELERATOR PROBE")
    click.echo("================================================================================")
    devs = ignition.devices()
    if not devs:
        click.echo("No physical AMD XDNA NPU detected via PyXRT.")
        click.echo("Ensure AMD NPU drivers (XRT / IPU Driver) are initialized.")
        return

    for dev in devs:
        click.echo(str(dev))
        click.echo("--------------------------------------------------------------------------------")


@cli.command("run")
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--input", "input_path", type=click.Path(exists=True), required=False, help="Path to input image (.png, .jpg) or tensor (.npy).")
@click.option("--backend", "-b", default="xdna1", type=click.Choice(["xdna1", "cpu"]), help="Execution backend.")
def run_model(model_path, input_path, backend):
    """Execute inference for an ONNX model on AMD Phoenix AIE2 silicon or CPU."""
    click.echo(f"[*] Compiling {model_path} on backend: {backend.upper()}...")
    try:
        model = ignition.compile(model_path, backend=backend)
    except Exception as e:
        click.secho(f"[-] Compilation error: {e}", fg="red")
        sys.exit(1)

    # Prepare input
    if input_path is not None:
        p = Path(input_path)
        click.echo(f"[*] Loading input data: {p.name}")
        output = model.predict(p)
    else:
        click.echo("[*] Generating synthetic int8 test activation...")
        synth_input = np.zeros((1, 8, 32, 32), dtype=np.int8)
        output = model.predict(synth_input)

    click.secho("[+] Inference successfully completed on physical silicon!", fg="green")
    click.echo(f"    Output Tensor Shape: {output.shape}")
    click.echo(f"    Output Tensor Dtype: {output.dtype}")
    click.echo(f"    Sample Values (first 8 bytes): {output.flatten()[:8].tolist()}")
    model.close()


@cli.command("benchmark")
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--backend", "-b", default="xdna1", type=click.Choice(["xdna1", "cpu"]), help="Execution backend.")
@click.option("--iterations", "-n", default=500, type=int, help="Number of timed benchmark iterations.")
@click.option("--warmup", "-w", default=20, type=int, help="Number of warmup iterations.")
@click.option("--compare-cpu", is_flag=True, help="Run side-by-side comparison against ONNX Runtime CPU.")
def benchmark_model(model_path, backend, iterations, warmup, compare_cpu):
    """Benchmark sustained latency and throughput on AMD Phoenix silicon."""
    click.echo("================================================================================")
    click.echo("IGNITION SUSTAINED HARDWARE BENCHMARK")
    click.echo("================================================================================")
    click.echo(f"Model:      {model_path}")
    click.echo(f"Backend:    {backend.upper()}")
    click.echo(f"Iterations: {iterations} (Warmup: {warmup})")
    click.echo("--------------------------------------------------------------------------------")

    model = ignition.compile(model_path, backend=backend)
    report = model.benchmark(iterations=iterations, warmup=warmup)
    click.echo(report.summary())
    model.close()

    if compare_cpu and backend != "cpu":
        click.echo("--------------------------------------------------------------------------------")
        click.echo("Executing baseline comparison against ONNX Runtime CPU...")
        cpu_model = ignition.compile(model_path, backend="cpu")
        cpu_report = cpu_model.benchmark(iterations=iterations, warmup=warmup)
        click.echo(cpu_report.summary())
        cpu_model.close()

        speedup = cpu_report.mean_us / report.mean_us if report.mean_us > 0 else 0.0
        click.echo("================================================================================")
        click.secho(f"Speedup vs ORT CPU: {speedup:.2f}x", fg="cyan", bold=True)
        click.echo("================================================================================")


if __name__ == "__main__":
    cli()
