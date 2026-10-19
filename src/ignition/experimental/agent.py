"""Checkout-only bounded workspace agent using Gemma's native tool protocol."""
# Copyright (C) 2026 The Ignition contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
from pathlib import Path

from .hybrid import HybridAssistant


class HybridAgent:
    def __init__(self,*,study_root,artifacts,llama_runtime,tokenizer_python,snapshot,workspace,
                 context_tokens=2560,input_depth=1,vector_copy=False,collect_trace=False,
                 max_rounds=4,max_tool_calls=8,max_hardware_wait=60,gpu_activity_helper=None):
        self.assistant=HybridAssistant(study_root=study_root,artifacts=artifacts,llama_runtime=llama_runtime,
            context_tokens=context_tokens,input_depth=input_depth,vector_copy=vector_copy,collect_trace=collect_trace,
            gpu_activity_helper=gpu_activity_helper)
        self.tokenizer=None;self.closed=False
        try:
            from npu.gemma4 import agent as backend
            from npu.gemma4.tokenizer import ChatTokenizer
            if not Path(backend.__file__).resolve().is_relative_to(Path(study_root).resolve()):
                raise RuntimeError('Agent imported from a different checkout')
            self.tokenizer=ChatTokenizer(tokenizer_python,snapshot,max_prompt_tokens=self.assistant.max_prompt_tokens)
            self._agent=backend.BoundedAgent(self.assistant,self.tokenizer,workspace,
                                            max_rounds=max_rounds,max_tool_calls=max_tool_calls,record_prompts=collect_trace,
                                            max_hardware_wait=max_hardware_wait)
        except BaseException:
            self.close();raise
    def run(self,prompt,*,on_event=None):
        if self.closed:raise RuntimeError('Agent is closed')
        return self._agent.run(prompt,on_event=on_event)
    @property
    def last_result(self):return self._agent.last_result
    def reset(self):self._agent.reset()
    def cancel(self):self._agent.cancel()
    def close(self):
        if self.closed:return
        try:self.assistant.close()
        finally:
            if self.tokenizer is not None:self.tokenizer.close()
            self.closed=True
    def __enter__(self):return self
    def __exit__(self,*exc):self.close()
