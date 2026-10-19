# Experimental workspace agent

`ignition.experimental.HybridAgent` adds a bounded native Gemma tool loop to the
checkout-only [hybrid assistant](HYBRID-ASSISTANT.md). The main model runs on the
780M and the BF16 drafter runs on the Phoenix NPU. Tokenization and tools run on
the host; no CPU model inference is used.

Supply the same prepared assets, ROCm runtime, pinned local tokenizer snapshot
and Python environments as the hybrid assistant, plus an explicit workspace:

```python
from ignition.experimental import HybridAgent

with HybridAgent(study_root=study, artifacts=assets, llama_runtime=rocm_dlls,
                 tokenizer_python=tokenizer_python, snapshot=snapshot,
                 workspace=workspace, input_depth=2, vector_copy=True) as agent:
    answer = agent.run(task)
    print(answer)
```

The registered tools are `read_text(path, offset=0)`, `list_files(path)` and
`calculate(expression)`. File paths must resolve beneath the explicit workspace;
absolute paths, drive paths, traversal outside it and alternate data streams are
rejected. Text reads accept UTF-8 files up to 1 MiB and return pages of 1,024
characters with a next offset. Listings accept at most 100 entries; choose a
smaller directory otherwise. Arithmetic supports decimal literals, parentheses,
unary signs and `+ - * /` with 28-digit precision and bounded syntax. There are
no write, shell or network tools. These checks are application restrictions,
not an operating-system sandbox against concurrent filesystem changes.

The loop uses Gemma's native tool protocol with thinking disabled. It executes
only completed tool handoffs and serializes tool responses as JSON strings in
the model's chat template. Truncated output, malformed calls and mixed
text/tool output do not execute tools. Default budgets are four model turns,
eight tool calls, 64 output tokens per turn and a 2,560-token context with up
to 2,048 prompt tokens. Long conversations can reach that prompt bound.

Every generation checks CPU, GPU and NPU contention before admission. A busy
device causes cancellable retries for at most 60 seconds by default, preserving
results from earlier tools; foreign processes are never stopped. NPU contexts
are released before tools execute. `cancel()` requests a cooperative stop;
`reset()` clears conversation history while retaining loaded model weights.
Final answers are delivered after a completed model turn. Always close the
agent, preferably through its context manager.

The study's `tools/gemma4-agent.ps1` launcher requires `-Workspace` and supports
interactive `/reset`, `/quit`, one-shot `-Task`, or UTF-8 `-TaskFile`. Its Python
frontend, `tools/gemma4_agent.py`, lists explicit local asset paths with
`--help`. Optional one-shot `--report` requires a new path and may contain
private workspace text and prompt token IDs; review it before sharing.
The frontend opts into double-buffer transport and vector copies; the Python
API defaults remain one input buffer and scalar copies.

Optional `gpu_activity_helper=PATH` in Python, `--gpu-activity-helper PATH` in
the frontend or `-GpuActivityHelper PATH` in the launcher enables the
[bounded GPU activity experiment](HYBRID-ASSISTANT.md). It is disabled by
default, joins all activity before target verification and tools execute, and has no qualified energy
benefit. Its native read/calculate/answer check matches the prior qualified
turn tokens; that comparison reuses the registered reference.

A bounded read-file, calculate, answer task passed with separate GPU-control
token equality checks for every model turn. The actual terminal frontend also
completed that task and exited with an idle NPU witness. This does not establish
general agent accuracy, net acceleration, realtime performance or unattended
24/7 readiness. The supported vision API and released wheel remain separate.
