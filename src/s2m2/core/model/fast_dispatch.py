"""Optional fast eager path for S2M2.forward (inference.py: fp16 channels-last, custom CUDA kernels, CUTLASS / cuDNN
fp16-accumulate convs) that never gets in the way of export.

S2M2.forward calls fast_forward() first; it returns None -- and the plain reference forward runs -- whenever
  - an exporter / tracer / compiler is active (torch.onnx.export, torch.export, torch.jit.trace/script, torch.compile),
  - gradients are on or the model is in training mode,
  - the input is not a CUDA batch of 1 padded to /32,
  - the kernels are unavailable (prebuilt binaries missing and no nvcc, GPU older than sm_80, nvidia-cudnn-frontend not
    installed, ...) or S2M2_FAST=0.
So exported graphs (ONNX -> TensorRT, torch.export / AOTInductor) are exactly those of the reference model, and without
the kernels the portable reference forward runs. The fast path runs on a private prepared copy of the model (fp16 weights,
~half the fp32 weight size extra), rebuilt when the reference parameters change (e.g. load_state_dict).
"""

import copy
import logging
import os

import torch

log = logging.getLogger(__name__)
_STATE = {"ok": None}


def exporting():
    return (
        torch.jit.is_tracing()
        or torch.jit.is_scripting()
        or torch.onnx.is_in_onnx_export()
        or torch.compiler.is_compiling()
        or torch.compiler.is_exporting()
    )


def available():
    if _STATE["ok"] is None:
        ok = (
            os.environ.get("S2M2_FAST", "1") == "1"
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability() >= (8, 0)
        )
        if ok:
            try:
                from . import cudnn_conv

                if cudnn_conv.cudnn is None:
                    raise ImportError("nvidia-cudnn-frontend not installed")
                from .inference import cut, kern

                kern()
                cut()
            except (
                Exception
            ) as e:  # no binaries / no compiler / unsupported GPU: reference path, same outputs
                log.info("S2M2 fast path unavailable (%s); using the reference forward", e)
                ok = False
        _STATE["ok"] = ok
    return _STATE["ok"]


def _version(model):
    tensors = (*model.parameters(), *model.buffers())
    if any(t.is_inference() for t in tensors):
        return None
    return tuple((id(t), t._version, t.dtype, t.device) for t in tensors)


def fast_forward(model, img0, img1):
    if exporting() or model.training or torch.is_grad_enabled():
        return None
    if not (
        img0.is_cuda
        and img0.dim() == 4
        and img0.shape[0] == 1
        and img0.shape[-1] % 32 == 0
        and img0.shape[-2] % 32 == 0
        and img1.shape == img0.shape
    ):
        return None
    if not available():
        return None
    # the fast path manages its own precisions (fp16 + fp32 OT / state): no autocast, also not while preparing the copy
    with torch.autocast("cuda", enabled=False):
        st = model.__dict__.get(
            "_fast"
        )  # kept outside the module tree (not in state_dict / .to() / modules())
        ver = _version(model)
        if ver is None:
            return None  # inference tensors cannot provide a reliable mutation counter
        ver = (
            ver,
            model.refine_iter,
            model.output_upsample,
            model.disp_init.ot_iter,
            model.disp_init.use_positivity,
        )
        if st is None or st[1] != ver or st[2] != img0.device:
            from .inference import prepare_inference

            model.__dict__.pop("_fast", None)
            fm = copy.deepcopy(model)
            prepare_inference(fm, hacc=True, kernels=True)
            st = model.__dict__["_fast"] = (fm, ver, img0.device)
        return st[0].forward_fast(img0.float(), img1.float(), full_outputs=True)
