# Experimental Gemma assistant

The checkout-only `ignition.experimental.HybridAssistant` wraps the reusable
`npu/gemma4` backend in ignite-xdna. It targets Gemma 4 E2B-it on the Radeon 780M
and the E2B assistant's BF16 draft steps on the Phoenix NPU. This interface is
under development. A bounded repeated-request check has passed on the Phoenix
host, including cancellation, GPU control token equality and NPU release while
the persistent GPU worker remains alive. This is not a speed qualification.

For the separate owned-prefix optimization, import
`PrefixBoundAssistant` from `ignition.experimental.prefix_bound`. It accepts
the same paths and streaming methods, defaulting to `input_depth=2` and
`vector_copy=True`; other values for those two options are rejected. The
wrapper binds shared KV/rotary/length patches once per draft chain and replaces
all token/hidden upload payloads on later proposals. Each NPU dispatch still
reads back and verifies its entire input. Implementation sources are pinned in
local request diagnostics and must resolve to the selected study checkout.
The ordinary `HybridAssistant` remains the frozen control.

The study's terminal frontend accepts `--prefix-binding` alongside
`--input-depth 2 --vector-copy`. Its PowerShell launcher exposes the same option:

```powershell
.\tools\gemma4-chat.ps1 -InputDepth 2 -VectorCopy -PrefixBinding `
    -GpuActivityHelper "$env:USERPROFILE\gemma4-assistant\bin\gpu_activity.exe"
```

Run this from the study checkout. The helper is optional and adds checked GPU
work; its energy benefit remains unmeasured. Prefix binding has bounded packet
equality and cancellation/recovery checks in the parent study. It is an opt-in
checkout interface, with no qualified net acceleration or unattended-operation
claim. The pending package-energy protocol continues to use the control.

Use the Python 3.13 NPU environment and put this checkout's `src` first on
`PYTHONPATH`. Supply explicit paths to the study checkout, prepared local assets,
and the matching llama.cpp ROCm DLL directory:

```python
from ignition.experimental import HybridAssistant

with HybridAssistant(study_root=study, artifacts=assets,
                     llama_runtime=rocm_dlls) as assistant:
    stream = assistant.stream_tokens(prompt_ids, max_new_tokens=32, depth=2)
    try:
        for chunk in stream:
            print(chunk.text, end="", flush=True)
    finally:
        stream.close()
```

`prompt_ids` must use the pinned main model's tokenizer and chat template. The
current interface accepts 1–64 new tokens and draft depth 1–4. Its default
512-token context accepts up to 256 prompt IDs. Explicit `context_tokens=2560`
accepts up to 2,048 prompt IDs and requires the matching verified rotary table.
`max_prompt_tokens` exposes the selected prompt bound to a tokenizer client.
It uses greedy verification, FP32 target KV, disabled
flash attention and disabled ROCm fusion to match the instrumented GPU control.
These are experimental bounds; larger contexts and general agent workloads
remain unqualified.

Required assets include the selected mixed Q5 GGUF, exported BF16 assistant
weights and metadata, verified rotary/GELU tables, and the bounded GPU worker.
The backend validates asset identities; it does not download missing models.
Model weights stay loaded between requests. Every new request prefills its full
prompt and releases its NPU contexts after completion or cancellation. Serving
does not generate a second GPU-only reference answer.

Consume and close a stream on the same thread. `cancel()` may request a stop
from another thread between hardware operations; continue consuming or close the
stream on its owner thread to perform cleanup. Closing an unread stream releases
request admission without loading a model. Always close the assistant itself.

Before each request, admission checks foreign inference CPU activity, GPU engine
activity and the NPU idle witness. Small desktop GPU activity is recorded and
does not qualify the request for performance claims. Other workloads are never
stopped. The study's bounded research wrapper adds continued contention monitoring
for qualification runs. `last_result` is local diagnostics and may contain prompt
hashes and private generated token IDs; review it before sharing.
The per-request check uses the study's native `npu.contention` sampler: CPU and
GPU rate samples share a 500 ms interval, followed by a fresh XRT context check.
Missing CPU, GPU or NPU telemetry rejects admission. The launcher's PowerShell
snapshot remains separate; the runtime no longer starts PowerShell per request.

Realtime speed and unattended operation remain unverified. A bounded
[workspace tool loop](WORKSPACE-AGENT.md) is available separately; its limited
read-file and arithmetic task has passed local checks.
A terminal frontend is available in the study
checkout as `tools/gemma4_chat.py`; `--help` lists its explicit artifact, ROCm,
tokenizer Python and snapshot paths. It supports `/reset`, `/quit`, conversation
history and `--prompt` for one response. The tokenizer runs in its own Python
process, uses only local assets and performs no model inference. Transformers
may import Torch there; the NPU environment does not need it. Public prompt ID
equality, Unicode transport and overlong-prompt recovery have passed local checks.
An end-to-end text request also produced the expected accented text and emoji
and released the NPU on exit; foreign activity was observed during startup, so
that run is a functional check only.

On the current development host, from the ignite-xdna checkout:

```powershell
.\tools\gemma4-chat.ps1
# Or answer one prompt and exit:
.\tools\gemma4-chat.ps1 -Prompt 'Reply with exactly the word OK and nothing else.'
# Opt into the qualified double-buffer packet transport:
.\tools\gemma4-chat.ps1 -InputDepth 2
# Also opt into aligned arena vector copies:
.\tools\gemma4-chat.ps1 -InputDepth 2 -VectorCopy
# A larger conversation budget, with the prepared 2,560-position rotary table:
.\tools\gemma4-chat.ps1 -ContextTokens 2560 -InputDepth 2 -VectorCopy
# Large or multiline prompts can be read as UTF-8 text:
.\tools\gemma4-chat.ps1 -ContextTokens 2560 -InputDepth 2 -VectorCopy -PromptFile .\prompt.txt
```

The launcher exposes the artifact, Python, ROCm and Hub paths as parameters and
prints a contention snapshot before starting. Runtime admission repeats before
each generation. Conversation history is kept only for completed responses;
`/reset` starts a new conversation when the current context bound is reached.

The experimental Python wrapper also accepts `input_depth=2`; the default is
one input buffer. Both modes passed bounded token-equality and lifecycle checks.
The double-buffer mode preserves model precision and the same verification
rules. It is an experimental transport option, not a realtime performance claim.
`vector_copy=True` in Python, or `-VectorCopy` in the launcher, adds aligned arena
copy acceleration with scalar fallback for unaligned offsets and tails. It has
also passed bounded token-equality, cancellation and cleanup checks. This option
preserves BF16 values exactly and remains disabled by default.

The larger mode has passed bounded GPU-control token checks at prompt lengths
511, 513, 1,024 and 2,048, plus cancellation and a return to a short prompt in
the same loaded worker. The tokenizer uses the same selected prompt bound.
`tools/gemma4_chat.py --prompt-file FILE` reads one UTF-8 prompt without passing
its text through a long command line. Optional `--report FILE` stores local
diagnostics for one-shot requests, including the actual prompt-token hash;
review private output and paths before sharing. Compiler diagnostics are kept
in that diagnostic record rather than printed between answer fragments.

This interface is separate from the supported vision
`ignition.compile()` API. See [USAGE.md](USAGE.md) for that API.

`last_result.finish_reason` distinguishes an actual model end token (`eos`)
from an output budget stop (`length`). Gemma's native tool handoff is also an
end token; `eos` alone does not mean a final user-facing answer. A completed
request alone does not establish that the model ended its turn.

Local diagnostics also include `operation_ms`: host wall time spent in
admission, target preparation, target-state conversion, drafting, GPU
verification, rollback and rendering. These intervals exclude time suspended
at a yielded chunk; they do not add up to the whole request. NPU session
setup/cleanup, other Python bookkeeping and client delays are outside these
operation intervals. The study's request profiler reports the remaining time
as `unattributed_ms`. These diagnostics are not qualified performance results.

An optional `gpu_activity_helper=PATH` enables the study's bounded Radeon 780M
activity helper. It is disabled by default. The helper matches HIP device zero's
Windows adapter identity and performs checked integer shader work during each
draft chain, then stops and joins before target GPU verification. Each window
requests at most 100 ms, with only one
dispatch outstanding; joining includes finishing that dispatch and checking
its output. The helper is idle before the next streamed chunk or tool call,
and closes with the assistant. Model weights and numerical verification are
unchanged. Only the owned helper process is excluded from contention checks.

Both experimental Python wrappers accept this argument. The Python chat/agent
frontends expose `--gpu-activity-helper PATH`; their PowerShell launchers expose
`-GpuActivityHelper PATH`. Supply the explicitly built `gpu_activity.exe` from
the matching study checkout. This adds GPU work to reduce an observed idle-gap
latency penalty. Bounded repeated-request, active-cancellation/recovery and
native-tool checks pass, but energy savings, general workload acceleration and
24/7 behavior are unqualified. It does not change system power settings.
