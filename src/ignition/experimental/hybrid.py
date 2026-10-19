"""Persistent GPU/NPU Gemma assistant wrapper over an ignite-xdna checkout."""
# Copyright (C) 2026 The Ignition contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import importlib
from pathlib import Path
import sys


class HybridAssistant:
    """Experimental greedy E2B token streaming; requires the NPU Python env.

    Prompt IDs must come from the pinned model's chat tokenizer. Model inference
    runs on the GPU and NPU. Tokenization and control remain host operations.
    Consume/close streams on the thread that started consuming them; cancel()
    can request a stop from another thread between hardware submissions.
    """
    def __init__(self, *, study_root, artifacts, llama_runtime, collect_trace=False,input_depth=1,vector_copy=False,context_tokens=512,gpu_activity_helper=None):
        if sys.version_info[:2] != (3, 13):
            raise RuntimeError('The experimental Phoenix backend requires the Python 3.13 NPU environment')
        root=Path(study_root).resolve()
        if not (root/'npu/gemma4/runtime.py').is_file():
            raise FileNotFoundError('An ignite-xdna checkout containing npu/gemma4 is required')
        sys.path.insert(0,str(root))
        module=importlib.import_module('npu.gemma4.runtime')
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise RuntimeError('Gemma runtime imported from a different checkout')
        self._backend=module.PersistentAssistant(artifacts,llama_runtime,collect_trace=collect_trace,input_depth=input_depth,
                                                 vector_copy=vector_copy,context_tokens=context_tokens,gpu_activity_helper=gpu_activity_helper)

    @property
    def max_prompt_tokens(self):return self._backend.max_prompt_tokens

    def stream_tokens(self, prompt_tokens, *, max_new_tokens=64, depth=2):
        """Yield chunks with token_ids, text and finished; no reference replay."""
        return self._backend.generate(prompt_tokens,max_new_tokens=max_new_tokens,depth=depth)

    @property
    def last_result(self):
        """Local diagnostics; may contain private token IDs. Nothing is sent."""
        return self._backend.last_result

    def cancel(self):self._backend.cancel()
    def close(self):self._backend.close()
    def __enter__(self):return self
    def __exit__(self,*exc):self.close()
