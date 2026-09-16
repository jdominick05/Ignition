# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: AGPL-3.0-or-later
r"""
src/ignition/suite.py
Model suite runner behind ``ignition suite``.

Every model runs in its own ``python -m ignition.live --headless --frames N --json`` process, so each
number is a cold process with its own warm-up, and every model gets one JSON record and one log in a new
output directory, plus ``index.json`` listing them. An existing output directory is refused: a suite never
overwrites earlier records.

A record is ``ignition.live``'s ``--json`` summary plus what produced it:
  * the command, its return code and wall time;
  * the Ignition and ignite-xdna versions, and ``git describe --dirty`` of their checkouts ("-dirty" when
    tracked files differ from the commit, so a number is never credited to code it did not run);
  * for an ``.ignite`` container, what ``xrt-smi examine -r aie-partitions`` said before and after.
A hardware context present before an NPU run makes that latency contention, not a result: the record is
kept, lists the problem, and the suite reports failure. Checkout paths become ``<Ignition>`` and
``<ignite-xdna>``, and the Windows profile directory ``C:\Users\<user>``, in every record and log. Only
runs actually executed are written; nothing is estimated.
"""

import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

XRT_SMI = Path(r"C:\Windows\System32\AMD\xrt-smi.exe")
IDLE_TEXT = "No hardware contexts running"
_PROFILE = re.compile(r"C:[\\/]+Users[\\/]+(?!<user>)[^\\/\s\"']+", re.IGNORECASE)
_RESERVED_NAMES = {"index"}


class Scrubber:
    """Replaces checkout roots with labels, then the Windows profile directory with ``C:\\Users\\<user>``,
    in every string of a value (dicts and lists included)."""

    def __init__(self, roots: Sequence[Tuple[Path, str]] = ()):
        self._roots = []
        # Longest first, so a worktree inside a checkout gets its own label.
        for path, label in sorted(roots, key=lambda r: -len(str(r[0]))):
            parts = [re.escape(p) for p in re.split(r"[\\/]+", str(Path(path).resolve())) if p]
            self._roots.append((re.compile(r"[\\/]+".join(parts), re.IGNORECASE), label))

    def __call__(self, value: Any) -> Any:
        if isinstance(value, str):
            for pattern, label in self._roots:
                value = pattern.sub(lambda _m, text=label: text, value)
            return _PROFILE.sub(lambda m: "C:\\Users\\<user>" if "\\" in m.group(0) else "C:/Users/<user>", value)
        if isinstance(value, dict):
            return {k: self(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self(v) for v in value]
        return value


def checkout_root(path: Path) -> Optional[Path]:
    """The nearest directory above ``path`` holding ``.git`` (a directory, or a file in a worktree)."""
    for parent in Path(path).resolve().parents:
        if (parent / ".git").exists():
            return parent
    return None


def git_describe(root: Optional[Path]) -> str:
    """``git describe --always --dirty --abbrev=7`` of a checkout, or "" outside one."""
    if root is None:
        return ""
    try:
        res = subprocess.run(["git", "-C", str(root), "describe", "--always", "--dirty", "--abbrev=7"],
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return ""
    return res.stdout.strip()


def xrt_partitions() -> Optional[str]:
    """What ``xrt-smi examine -r aie-partitions`` prints, or None when xrt-smi is not installed."""
    if not XRT_SMI.is_file():
        return None
    try:
        res = subprocess.run([str(XRT_SMI), "examine", "-r", "aie-partitions"], capture_output=True, text=True,
                             timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"xrt-smi failed: {type(exc).__name__}: {exc}"
    return (res.stdout + res.stderr).strip()


def default_source() -> Optional[Path]:
    """``examples/assets/bus.jpg`` of the checkout this package runs from, if there is one."""
    candidate = Path(__file__).resolve().parents[2] / "examples" / "assets" / "bus.jpg"
    return candidate if candidate.is_file() else None


def record_names(models: Sequence[Path]) -> List[str]:
    """One file name per model: its stem with dots as underscores, numbered when two models share one."""
    names: List[str] = []
    for model in models:
        base = Path(model).stem.replace(".", "_")
        name, n = base, 1
        while name in names or name in _RESERVED_NAMES:
            n += 1
            name = f"{base}_{n}"
        names.append(name)
    return names


def _text(data: Any) -> str:
    if data is None:
        return ""
    return data.decode("utf-8", "replace") if isinstance(data, bytes) else str(data)


def _ms(value: Any) -> str:
    return f"{value:.3f} ms" if isinstance(value, (int, float)) and value == value else "n/a"


def run_suite(models: Sequence[Path], out_dir: Path, source: str, frames: int, warmup: int = 10,
              timeout_s: float = 1800.0, echo: Callable[[str], None] = print,
              power_mode: Optional[str] = None) -> Dict[str, Any]:
    """Runs every model in its own ``ignition.live`` process; writes ``<name>.json`` and ``<name>.log`` per
    model and ``index.json`` into ``out_dir``, which must not exist. Returns the index (``all_ok`` says
    whether every run exited 0 with timed frames and, on the NPU, started on an idle device)."""
    if frames < 1:
        raise ValueError("--frames must be at least 1: every suite run needs a frame limit")
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; a suite never overwrites earlier records")
    models = [Path(m).resolve() for m in models]
    missing = [str(m) for m in models if not m.is_file()]
    if missing:
        raise FileNotFoundError(f"model not found: {', '.join(missing)}")
    source = str(source).strip()
    if not source.isdigit():
        source_path = Path(source).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"{source} is neither a webcam index nor an existing file")
        source = str(source_path)

    import ignition
    from ignition.pipelines.yolo import is_ignite_container

    package_src = Path(ignition.__file__).resolve().parents[1]
    ignition_root = checkout_root(Path(ignition.__file__))
    provenance: Dict[str, Any] = {
        "ignition_version": ignition.__version__, "ignition_path": ignition.__file__,
        "ignition_commit": git_describe(ignition_root),
        "ignite_xdna_version": None, "ignite_xdna_path": None, "ignite_xdna_commit": "",
        "python": sys.executable, "hostname": socket.gethostname(),
    }
    roots = [(ignition_root, "<Ignition>")] if ignition_root else []
    try:
        import ignite_xdna
        xdna_root = checkout_root(Path(ignite_xdna.__file__))
        provenance.update({"ignite_xdna_version": getattr(ignite_xdna, "__version__", None),
                           "ignite_xdna_path": ignite_xdna.__file__, "ignite_xdna_commit": git_describe(xdna_root)})
        if xdna_root:
            roots.append((xdna_root, "<ignite-xdna>"))
    except ImportError:
        pass  # CPU-only install: .onnx models still run
    # The output directory appears in every command and log line; its path can name the user (a temp dir).
    roots.append((out_dir, "<out>"))
    scrub = Scrubber(roots)

    # The child imports this same ignition package, with OpenCV's Media Foundation option set before cv2 loads.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(package_src)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
    env["PYTHONIOENCODING"] = "utf-8"

    created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out_dir.mkdir(parents=True)
    source_label = f"webcam {source}" if source.isdigit() else Path(source).name
    rows: List[Dict[str, Any]] = []
    for model, name in zip(models, record_names(models)):
        native = is_ignite_container(model)
        live_json = out_dir / f".{name}.live.json"
        log_path = out_dir / f"{name}.log"
        cmd = [sys.executable, "-m", "ignition.live", "--model", str(model), "--source", source, "--headless",
               "--frames", str(frames), "--warmup", str(warmup), "--json", str(live_json)]
        if power_mode:
            cmd += ["--power-mode", power_mode]
        entry: Dict[str, Any] = {"suite_created_utc": created, "model_path": str(model), "log": log_path.name,
                                 "command": cmd, **provenance}
        problems: List[str] = []
        if native:
            before = xrt_partitions()
            entry["xrt_smi_before"] = before
            entry["npu_idle_before"] = None if before is None else IDLE_TEXT in before
            if entry["npu_idle_before"] is False:
                problems.append("a hardware context was present before the run: the latency is contention")
        echo(f"[suite] {model.name}: {frames} timed frames after {warmup} warm-up on the "
             f"{'NPU' if native else 'CPU'}, source {source_label}")
        t0 = time.perf_counter()
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                 timeout=timeout_s, env=env)
            rc, out, err = res.returncode, res.stdout, res.stderr
        except subprocess.TimeoutExpired as exc:
            rc, out, err = "timeout", _text(exc.stdout), _text(exc.stderr)
        entry["wall_s"] = round(time.perf_counter() - t0, 1)
        entry["returncode"] = rc
        log = "$ " + " ".join(cmd) + "\n" + out + (("\n[stderr]\n" + err) if err.strip() else "")
        log_path.write_text(scrub(log), encoding="utf-8", newline="\n")
        if live_json.is_file():
            entry.update(json.loads(live_json.read_text(encoding="utf-8")))
            live_json.unlink()
        if native:
            after = xrt_partitions()
            entry["xrt_smi_after"] = after
            entry["npu_idle_after"] = None if after is None else IDLE_TEXT in after
        if rc != 0:
            problems.append(f"return code {rc}")
        if "g2g_ms" not in entry:
            problems.append("no timed frames recorded")
        entry["problems"] = problems
        (out_dir / f"{name}.json").write_text(json.dumps(scrub(entry), indent=2) + "\n", encoding="utf-8",
                                               newline="\n")
        g2g = entry.get("g2g_ms", {})
        rows.append({"name": name, "model": model.name, "task": entry.get("task"), "backend": entry.get("backend"),
                     "frames_timed": entry.get("frames_timed", 0), "g2g_mean_ms": g2g.get("mean"),
                     "g2g_p99_ms": g2g.get("p99"), "returncode": rc, "problems": problems})
        echo(f"[suite]   rc {rc} | G2G mean {_ms(g2g.get('mean'))} | P99 {_ms(g2g.get('p99'))} | {entry['wall_s']} s"
             + (f" | {'; '.join(problems)}" if problems else ""))
    index = {"schema": 1, "created_utc": created, "source": source, "frames": frames, "warmup": warmup,
             **provenance, "all_ok": all(not r["problems"] for r in rows), "records": rows}
    (out_dir / "index.json").write_text(json.dumps(scrub(index), indent=2) + "\n", encoding="utf-8", newline="\n")
    return index


def format_table(index: Dict[str, Any]) -> List[str]:
    """The index as fixed-width lines: model, task, backend, timed frames, G2G mean and P99, return code."""
    lines = [f"{'Model':<34} {'Task':<17} {'Backend':<16} {'Frames':>6} {'G2G mean':>11} {'P99':>11}  rc"]
    for r in index["records"]:
        lines.append(f"{r['model']:<34} {str(r['task'] or 'n/a'):<17} {str(r['backend'] or 'n/a'):<16} "
                     f"{r['frames_timed']:>6} {_ms(r['g2g_mean_ms']):>11} {_ms(r['g2g_p99_ms']):>11}  {r['returncode']}")
    return lines
