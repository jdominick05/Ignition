# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
src/ignition/cli/main.py
Command-line interface for the Ignition AI engine.
"""

import sys
import time
from typing import Optional, List
from pathlib import Path
import click
import cv2
import numpy as np

import ignition


@click.group()
@click.version_option(version=ignition.__version__, prog_name="ignition")
def cli():
    """Ignition: Production-grade AMD Ryzen AI XDNA1 AIE2 Inference Engine."""
    pass


@cli.command("devices")
@click.option("--restore-npu-power", is_flag=True,
              help="Put back an NPU power mode that an Ignition run, no longer running, left lowered; print the mode.")
def list_devices(restore_npu_power):
    """Probe and display detected AMD XDNA NPU hardware status."""
    if restore_npu_power:
        try:
            from ignite_xdna.pipelines import npu_power
        except ImportError:
            click.echo("This ignite-xdna has no NPU power-mode governor (pipelines/npu_power.py); nothing to restore.")
        else:
            click.echo(npu_power.recover_stale_lease() or "No NPU power lease to recover.")
            click.echo(f"NPU power mode: {npu_power.read_mode() or 'unknown (xrt-smi did not report it)'}")
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

    where = "on physical silicon" if backend == "xdna1" else "on the CPU"
    click.secho(f"[+] Inference successfully completed {where}!", fg="green")
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


def _load_frame_stream(input_path: Path, max_frames: Optional[int] = None):
    """Yields frames from an image file, image directory, or video."""
    video_exts = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if input_path.is_dir():
        files = sorted([p for p in input_path.iterdir() if p.suffix.lower() in image_exts])
        if not files:
            raise ValueError(f"No supported image files found in directory: {input_path}")
        if max_frames:
            files = files[:max_frames]
        for p in files:
            yield cv2.imread(str(p))
    elif input_path.suffix.lower() in video_exts:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {input_path}")
        count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
                count += 1
                if max_frames and count >= max_frames:
                    break
        finally:
            cap.release()
    elif input_path.suffix.lower() in image_exts:
        img = cv2.imread(str(input_path))
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {input_path}")
        if max_frames and max_frames > 1:
            for _ in range(max_frames):
                yield img
        else:
            yield img
    else:
        raise ValueError(f"Unsupported input format: {input_path}")


@cli.command("detect")
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--input", "-i", "input_path", type=click.Path(exists=True), required=True, help="Path to input image file, video, or directory.")
@click.option("--output", "-o", "output_path", default="yolo_output.jpg", type=str, help="Path to save annotated visual detection image.")
@click.option("--backend", "-b", default="xdna1", type=click.Choice(["xdna1", "cpu"]), help="Execution target backend.")
@click.option("--conf", "-c", default=0.25, type=float, help="Confidence score threshold (default: 0.25).")
@click.option("--iou", default=0.45, type=float, help="NMS IoU threshold (default: 0.45).")
@click.option("--stream", is_flag=True, default=False, help="Enable 3-stage asynchronous pipelined execution runner.")
@click.option("--benchmark", is_flag=True, default=False, help="Profile sustained throughput and latency metrics across stream frames.")
@click.option("--frames", default=100, type=int, help="Number of frames for benchmark mode (default: 100).")
def detect_objects(model_path, input_path, output_path, backend, conf, iou, stream, benchmark, frames):
    """Execute end-to-end YOLOv8 object detection on AMD Phoenix AIE2 silicon or CPU."""
    in_p = Path(input_path).resolve()
    target_pipeline = "async_yolo" if stream else "yolo"
    mode_str = "Async Pipelined" if stream else "Synchronous"
    click.echo(f"[*] Compiling YOLOv8 {mode_str} detection pipeline on target '{backend.upper()}'...")
    try:
        pipeline = ignition.compile(
            model_path,
            backend=backend,
            pipeline=target_pipeline,
            conf_thres=conf,
            iou_thres=iou,
        )
    except Exception as e:
        click.secho(f"[-] Pipeline compilation error: {e}", fg="red")
        sys.exit(1)

    device = "NPU" if getattr(pipeline, "is_native", False) else "CPU"
    runs_on = "NPU" if device == "NPU" else "CPU (ONNX Runtime)"
    click.echo(f"[*] Inference runs on the {runs_on}")

    try:
        if benchmark:
            click.echo(f"[*] Benchmarking sustained streaming throughput over {frames} frames...")
            frame_iter = _load_frame_stream(in_p, max_frames=frames)
            t_start = time.perf_counter()
            results = []
            if hasattr(pipeline, "stream"):
                stream_fn = pipeline.stream
            else:
                stream_fn = pipeline.predict
            for res in pipeline.stream(frame_iter, conf_thres=conf, iou_thres=iou, annotate=False):
                results.append(res)
            t_end = time.perf_counter()
            elapsed_s = t_end - t_start
            fps = len(results) / elapsed_s if elapsed_s > 0 else 0.0

            pre_lat = [r.timings_ms["preprocess_ms"] for r in results]
            back_lat = [r.timings_ms["backbone_ms"] for r in results]
            post_lat = [r.timings_ms["postprocess_ms"] for r in results]
            tot_lat = [r.timings_ms["total_ms"] for r in results]

            click.echo()
            click.echo("================================================================================")
            click.secho(f"Ignition Streaming Benchmark Report ({runs_on})", fg="cyan", bold=True)
            click.echo("================================================================================")
            click.echo(f"Total Stream Frames Processed: {len(results)}")
            click.echo(f"Total Stream Elapsed Time:     {elapsed_s * 1000.0:.2f} ms")
            click.secho(f"Sustained Pipelined Throughput: {fps:.2f} FPS", fg="green", bold=True)
            click.echo(f"Average Preprocessing Latency:  {np.mean(pre_lat):.2f} ms")
            click.echo(f"Average {device} Inference Latency:  {np.mean(back_lat):.2f} ms")
            click.echo(f"Average Postprocess/NMS Lat:   {np.mean(post_lat):.2f} ms")
            click.echo(f"Median End-to-End Latency:     {np.median(tot_lat):.2f} ms")
            click.echo(f"P95 End-to-End Latency:        {np.percentile(tot_lat, 95):.2f} ms")
            click.echo("================================================================================")
            return

        click.echo(f"[*] Executing detection on '{in_p.name}'...")
        if stream:
            frame_iter = _load_frame_stream(in_p, max_frames=1)
            results = list(pipeline.stream(frame_iter, conf_thres=conf, iou_thres=iou, annotate=True))
            if not results:
                raise RuntimeError("No results produced by streaming pipeline.")
            result = results[0]
        else:
            result = pipeline.predict(in_p, conf_thres=conf, iou_thres=iou)

        # Save output image
        out_p = Path(output_path).resolve()
        result.save(out_p)

        click.echo()
        click.echo(result.summary())
        click.echo()
        click.secho(f"[+] Detection visual output saved: {out_p}", fg="green", bold=True)
    finally:
        pipeline.close()


@cli.command("suite")
@click.argument("models", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--source", "-s", default=None,
              help="Image, video or webcam index every model runs on (default: examples/assets/bus.jpg of this checkout).")
@click.option("--frames", "-n", default=300, show_default=True, type=click.IntRange(min=1),
              help="Timed frames per model.")
@click.option("--warmup", "-w", default=10, show_default=True, type=click.IntRange(min=0),
              help="Untimed frames per model before the timed ones.")
@click.option("--out", "-o", "out_dir", default=None, type=click.Path(file_okay=False),
              help="New directory for the records (default: suite-<UTC time> in the current directory); "
                   "an existing directory is refused.")
@click.option("--timeout", default=1800.0, show_default=True, type=float, help="Seconds allowed per model process.")
@click.option("--power-mode", type=click.Choice(["efficiency", "balanced", "performance"]), default=None,
              help="Host-thread power mode for .ignite containers, passed to every run (default: balanced).")
def run_model_suite(models, source, frames, warmup, out_dir, timeout, power_mode):
    """Run each model in its own ignition.live process and write one JSON record per model."""
    from ignition.suite import default_source, format_table, run_suite

    src = source if source is not None else default_source()
    if src is None:
        click.secho("[-] No --source given, and there is no examples/assets/bus.jpg next to this package", fg="red")
        sys.exit(2)
    out = Path(out_dir) if out_dir else Path.cwd() / time.strftime("suite-%Y%m%dT%H%M%SZ", time.gmtime())
    click.echo("================================================================================")
    click.echo("IGNITION MODEL SUITE")
    click.echo("================================================================================")
    click.echo(f"Models:  {len(models)}, one process each")
    click.echo(f"Source:  {src}")
    click.echo(f"Frames:  {frames} timed after {warmup} warm-up")
    if power_mode:
        click.echo(f"Power:   {power_mode}")
    click.echo(f"Records: {out}")
    click.echo("--------------------------------------------------------------------------------")
    try:
        index = run_suite([Path(m) for m in models], out, str(src), frames, warmup, timeout, echo=click.echo,
                          power_mode=power_mode)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        click.secho(f"[-] {exc}", fg="red")
        sys.exit(2)
    click.echo("================================================================================")
    for line in format_table(index):
        click.echo(line)
    click.echo("================================================================================")
    if not index["all_ok"]:
        failed = ", ".join(r["model"] for r in index["records"] if r["problems"])
        click.secho(f"[-] Problems in {failed}; every record is in {out}", fg="red", bold=True)
        sys.exit(1)
    click.secho(f"[+] {len(index['records'])} records written to {out}", fg="green", bold=True)


if __name__ == "__main__":
    cli()

