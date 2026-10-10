"""General fp16-accumulate convolutions through cuDNN graphs (copied from exp 003 IGEV-general, port of RAFT-Stereo exp 002 convs).

PyTorch always asks cuDNN for fp32 accumulation; on GeForce cards that runs the tensor cores at half rate
(RTX 5060 Ti: ~47 vs ~80 TFLOPS). Here every fp16 Conv2d/Conv3d (groups 1, dilation 1) runs as a plain cuDNN
conv_fprop graph with compute_data_type=HALF, NHWC / NDHWC in and out.

General = any input size, no per-size compilation or tuning: plans that need runtime compilation (NVRTC) are
deselected, the first precompiled plan in cuDNN's heuristic order is used (no timing), ~ms per new geometry.
No fused epilogue (those only exist as runtime-compiled kernels on sm_120): bias is added by PyTorch.
Engines 0 (reference, very slow), 8 (crashes on sm_120, exp 002 DESIGN_FLAWS D9) and 28 (legacy implicit_convolve_hhgemm,
5x slower than torch for S2M2 3x1 GRU convs) are skipped.
S2M2_CONV_VERIFY=1 checks every new plan against an fp32 reference (debug).
"""

import os

import torch
import torch.nn.functional as F


def _stream():
    """torch's current raw CUDA stream (torch.cuda.current_stream() costs ~3 us per call)."""
    return torch._C._cuda_getCurrentRawStream(torch._C._cuda_getDevice())


try:
    import cudnn
except ImportError:  # optional: pip install nvidia-cudnn-frontend
    cudnn = None

VERIFY = os.environ.get("S2M2_CONV_VERIFY") == "1"
_handle = None
_plans = {}
_ws = [None]
stats = {"cudnn": 0, "torch": 0, "built": 0}


def _get_handle():
    global _handle
    if _handle is None:
        _handle = cudnn.create_handle()
    return _handle


def _out_size(x, w, padding, stride):
    return [
        (x.shape[2 + i] + 2 * padding[i] - w.shape[2 + i]) // stride[i] + 1
        for i in range(w.dim() - 2)
    ]


def _cl_stride(shape):
    """channels-last strides for [N, C, *spatial]."""
    n, c, *sp = shape
    st = [0] * len(shape)
    acc = c
    st[1] = 1
    for i in range(len(sp) - 1, -1, -1):
        st[2 + i] = acc
        acc *= sp[i]
    st[0] = acc
    return st


def _build(x, w, padding, stride):
    H16 = cudnn.data_type.HALF
    nd = w.dim() - 2
    out_sp = _out_size(x, w, padding, stride)
    yshape = [x.shape[0], w.shape[0], *out_sp]
    h = _get_handle()
    try:
        g = cudnn.pygraph(
            io_data_type=H16, intermediate_data_type=H16, compute_data_type=H16, handle=h
        )
        X = g.tensor(name="X", dim=list(x.shape), stride=list(x.stride()), data_type=H16)
        W = g.tensor(name="W", dim=list(w.shape), stride=list(w.stride()), data_type=H16)
        Y = g.conv_fprop(
            image=X,
            weight=W,
            padding=list(padding),
            stride=list(stride),
            dilation=[1] * nd,
            compute_data_type=H16,
        )
        Y.set_output(True).set_data_type(H16).set_dim(yshape).set_stride(_cl_stride(yshape))
        g.validate()
        g.build_operation_graph()
        g.deselect_behavior_notes([cudnn.behavior_note.RUNTIME_COMPILATION])
        g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        g.check_support()
    except Exception:  # noqa: BLE001 - no half-compute graph for this geometry
        return None
    for i in range(g.get_execution_plan_count()):
        if g.get_plan_name_at_index(i).split("_")[0] in ("eng0", "eng8", "eng28"):
            continue
        try:
            g.build_plan_at_index(i)
        except Exception:  # noqa: BLE001
            continue
        if VERIFY and not _verify(g, X, W, Y, i, x, w, yshape, padding, stride):
            continue
        stats["built"] += 1
        return g, X, W, Y, i, g.get_workspace_size_plan_at_index(i), yshape
    return None


def _verify(g, X, W, Y, i, x, w, yshape, padding, stride):
    xr = (
        (torch.rand(x.shape, device=x.device) * 2 - 1).half().contiguous(memory_format=_mf(x.dim()))
    )
    y = torch.full(yshape, float("nan"), device=x.device, dtype=torch.half).contiguous(
        memory_format=_mf(x.dim())
    )
    ws = torch.empty(
        max(g.get_workspace_size_plan_at_index(i), 1), device=x.device, dtype=torch.uint8
    )
    g.execute_plan_at_index({X: xr, W: w, Y: y}, ws, i, handle=_get_handle())
    conv = F.conv2d if x.dim() == 4 else F.conv3d
    ref = conv(xr.float(), w.float(), None, stride, padding)
    dev = float((y.float() - ref).abs().max() / (ref.abs().max() + 1e-3))
    ok = dev == dev and dev < 0.05
    if not ok:
        print(
            f"[cudnn_conv] rejected {g.get_plan_name_at_index(i)} x{tuple(x.shape)} w{tuple(w.shape)}: {dev:.3g}"
        )
    return ok


def _mf(dim):
    return torch.channels_last if dim == 4 else torch.channels_last_3d


def conv_half_acc(x, w, b, padding, stride):
    """fp16 conv with fp16 accumulation; x any layout (made channels-last), w channels-last fp16. Returns channels-last."""
    mf = _mf(x.dim())
    if not x.is_contiguous(memory_format=mf):
        x = x.contiguous(memory_format=mf)
    key = (x.shape, w.shape, tuple(padding), tuple(stride), x.device.index)
    e = _plans.get(key, False)
    if e is False:
        e = _plans[key] = _build(x, w, padding, stride)
    if e is None:
        stats["torch"] += 1
        conv = F.conv2d if x.dim() == 4 else F.conv3d
        return conv(x, w, b, stride, padding).contiguous(memory_format=mf)
    stats["cudnn"] += 1
    g, X, W, Y, i, ws_size, yshape = e
    y = torch.empty(yshape, device=x.device, dtype=x.dtype, memory_format=mf)
    ws = _ws[0]
    if ws is None or ws.numel() < ws_size:
        ws = _ws[0] = torch.empty(max(ws_size, 1), device=x.device, dtype=torch.uint8)
    h = _get_handle()
    cudnn.set_stream(handle=h, stream=_stream())
    g.execute_plan_at_index({X: x, W: w, Y: y}, ws, i, handle=h)
    if b is not None:
        y.add_(b.view(1, -1, *([1] * (x.dim() - 2))))
    return y
