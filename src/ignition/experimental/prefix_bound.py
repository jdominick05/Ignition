"""Opt-in checkout-only assistant with owned NPU prefix binding."""
# Copyright (C) 2026 The Ignition contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import hashlib
import importlib
from pathlib import Path

from .hybrid import HybridAssistant


class PrefixBoundAssistant(HybridAssistant):
    """Experimental host optimization with the same greedy GPU/NPU model path.

    The measured configuration uses input depth two and vector copies. Every
    NPU input remains fully read back and verified. Energy and unattended
    operation are unqualified; the regular HybridAssistant remains the control.
    """
    def __init__(self, *, study_root, **kwargs):
        kwargs.setdefault('input_depth', 2)
        kwargs.setdefault('vector_copy', True)
        if kwargs['input_depth'] != 2 or kwargs['vector_copy'] is not True:
            raise ValueError('Prefix binding requires input_depth=2 and vector_copy=True')
        super().__init__(study_root=study_root, **kwargs)
        root = Path(study_root).resolve()
        try:
            modules = {
                'tools.gemma4_prefix_candidate': 'tools/gemma4_prefix_candidate.py',
                'kernels.transformer_bf16.prefix_cache': 'kernels/transformer_bf16/prefix_cache.py',
                'kernels.transformer_bf16.prefix_binding': 'kernels/transformer_bf16/prefix_binding.py',
                'kernels.transformer_bf16.state': 'kernels/transformer_bf16/state.py',
            }
            for name, relative in modules.items():
                loaded = importlib.import_module(name)
                if Path(loaded.__file__).resolve() != root/relative:
                    raise RuntimeError('Prefix implementation imported from another checkout')
            module = importlib.import_module('tools.gemma4_prefix_candidate')
            paths = [Path(__file__).resolve(), Path(__file__).with_name('hybrid.py'),
                     *(root/relative for relative in modules.values())]
            self._backend.sources.update({str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
            self._backend.drafter = module.PrefixCachedDraft.from_draft(self._backend.drafter)
            self._backend.drafter.bind_prefix = True
        except BaseException:
            self.close()
            raise
