# Copyright (C) 2026 The Ignition contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Dense model tasks using the study's canonical family transforms.

These research models require ignite-xdna's source checkout (the pinned family
preprocessing lives in npu/), including for the CPU reference path.
"""
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import time
import weakref

import numpy as np

from .yolo import load_bgr, is_ignite_container, _release_native


def family_module(task):
    # Load exactly the canonical file, including in an editable install launched
    # from the separate Ignition checkout. Do not mutate the caller's sys.path.
    from ignite_xdna.runtime.driver import get_repo_root
    name = 'bisenetv2' if task == 'segment' else 'modnet'
    path = get_repo_root() / 'npu' / f'{name}.py'
    if not path.is_file():
        raise RuntimeError(f'{task} requires the ignite-xdna research checkout preprocessing: {path}')
    spec = importlib.util.spec_from_file_location(f'_ignition_dense_{name}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class DenseResult:
    tensor: np.ndarray
    mask: np.ndarray | None
    alpha: np.ndarray | None
    timings_ms: dict
    orig_shape: tuple
    source: str


class DensePipeline:
    def __init__(self, model_path, task, device_id=0):
        from .vision import _ort_session
        if task not in ('segment','matte'):
            raise ValueError(f'unsupported dense task {task}')
        self.task = task
        self.family = family_module(task)
        self.model_path = Path(model_path)
        self.is_native = is_ignite_container(self.model_path)
        self.native = self.session = None
        self._closed = False
        if self.is_native:
            try:
                from ignite_xdna.runtime.dense_session import DenseTensorSession
            except ImportError as ex:
                # The hybrid engine path for these two tasks is in the ignite-xdna source checkout but
                # in no released wheel, so this used to surface as a bare ModuleNotFoundError from a
                # --task the CLI advertises.
                raise RuntimeError(
                    f"--task {task} on an .ignite container needs ignite_xdna.runtime.dense_session. "
                    "It landed in the ignite-xdna source checkout on 2026-09-21 but is in no released "
                    "wheel yet, so install ignite-xdna from source to use it. The .onnx CPU path for "
                    "these two tasks additionally needs that same checkout, whose npu/ transforms are "
                    "not packaged in the wheel. Both run slower than AMD's stack; see the segmentation "
                    "and matting section of docs/PERFORMANCE.md before building around them."
                ) from ex
            self.native = DenseTensorSession(self.model_path,device_index=device_id)
            self._finalizer = weakref.finalize(self,_release_native,self.native)
            shape = self.native.ignite_manifest['input_shape']
            self.hybrid = bool(self.native._host_steps)
            self.backend_name = 'xdna1-hybrid' if self.hybrid else 'xdna1'
            if self.native.task != task:
                actual = self.native.task
                self.close()
                raise ValueError(f'container task {actual} differs from {task}')
        else:
            self.session = _ort_session(self.model_path)
            inp = self.session.get_inputs()[0]
            shape,self.input_name = inp.shape,inp.name
            self.backend_name = 'cpu'
        if len(shape) != 4 or shape[:2] != [1,3] or shape[2] != shape[3]:
            self.close()
            raise ValueError(f'expected static square NCHW image input, got {shape}')
        self.input_size = int(shape[2])

    def predict(self,image):
        if self._closed:
            raise RuntimeError('DensePipeline is closed')
        img = load_bgr(image,copy=False)
        t0 = time.perf_counter()
        x,shape = self.family.preprocess(img,target_size=self.input_size)
        t1 = time.perf_counter()
        hw = {}
        if self.is_native:
            y,hw = self.native.run(x)
        else:
            y = self.session.run(None,{self.input_name:x})[0]
        t2 = time.perf_counter()
        mask = self.family.postprocess_mask(y,shape) if self.task == 'segment' else None
        alpha = self.family.postprocess_matte(y,shape) if self.task == 'matte' else None
        t3 = time.perf_counter()
        timings = dict(preprocess_ms=(t1-t0)*1e3,backbone_ms=(t2-t1)*1e3,
                       postprocess_ms=(t3-t2)*1e3,g2g_ms=(t3-t0)*1e3,total_ms=(t3-t0)*1e3)
        if hw:
            timings.update(dispatch_ms=hw['npu_ms'],host_ms=hw['host_ms'],
                           transfer_ms=hw['transfer_ms'],readback_ms=hw['readback_ms'])
        source = ('hybrid' if self.hybrid else 'npu') if self.is_native else 'onnxruntime'
        return DenseResult(y,mask,alpha,timings,shape,source)

    def visualize(self,image,result):
        if self.task == 'segment':
            return self.family.overlay_segmentation(image,result.mask)
        return np.clip(image.astype(np.float32)*result.alpha[:,:,None],0,255).astype(np.uint8)

    def close(self):
        self.session = None
        if self.native is not None:
            self._finalizer()
            self.native = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self,*exc):
        self.close()
