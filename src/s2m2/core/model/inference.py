"""Inference-only fast path for S2M2 (general: any input size, no per-size compile / autotune / cudnn.benchmark).

prepare_inference(model, hacc=True) then model.forward_fast(img0, img1) (images [1,3,H,W], values 0..255, padded /32).
  - pure fp16 weights + activations, channels_last everywhere: conv <-> transformer permutes become free views
    (upstream: NCHW tensors -> cuDNN nchw<->nhwc conversions + .contiguous() copies, ~10 % of the time)
  - attention: fused q|k|v projection, SDPA on strided views (no head transposes), residual folded into the output
    projection (addmm_), in-place FFN residual; the relative-PE attention of the feature pyramid no longer builds the
    [N, N, 32] PE tensor (2.1 GB at Midd-F): pe_sum = attention marginals over rows / columns x small 1D tables (exact)
  - Unet / MRT up path: 1x1 conv before bilinear x2 (linear ops commute; 4x fewer conv FLOPs)
  - initial disparity (optimal transport): rows are independent -> processed in row chunks (bounded memory), fp32
  - refinement: disparity / conf / occ kept fp32; cost-volume lookup = 1D linear interpolation by gather with fp32
    coordinates (upstream: grid_sample with fp16 coordinates, 0.5 px steps beyond x = 512)
  - CNN backbone one image at a time, GroupNorm as 2 in-place passes; full-res tail (upsampling) without 9x unfold
  - hacc: fp16-accumulate cuDNN convs (precompiled engines only, ~ms per new shape), cudnn_conv.py
"""

import math
import os
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attentions import BasicAttnBlock, ConvBlock2D, CrossAttn, GlobalAttnBlock, SelfAttn
from .feature_fusion import FeatureFusion
from .refinenet import ConvGRU, GlobalRefiner, LocalRefiner
from .stacked_MRT import MRT, StackedMRT
from .submodules import CNNEncoder, DispInit, UpsampleMask1x, UpsampleMask4x
from .unet import Unet
from .utils import custom_sinc

CL = torch.channels_last
_SDPA_ALWAYS = os.environ.get("S2M2_SDPA") == "1"
_ATT16 = os.environ.get("S2M2_ATT16", "1") == "1"
_CUT_CONV = (
    os.environ.get("S2M2_CUT_CONV", "1") == "1"
)  # CUTLASS fused-epilogue convs (rule: kxk C>=128, 1x1 C>=768)
_CUT_LIN = os.environ.get("S2M2_CUT_LIN", "1") == "1"  # CUTLASS for linears with a residual
_PAIR_ATTN = os.environ.get("S2M2_PAIR_ATTN", "1") == "1"
_RELPE_FLASH = (
    os.environ.get("S2M2_RELPE_FLASH", "1") == "1"
)  # rel-PE attention as one flash call with one-hot values  # row attention per image strip (no strip copies)   # explicit (head_dim > 256) attention GEMMs with fp16 accumulation
OT_BYTES = 512 * 2**20  # working-set budget of one optimal-transport row chunk
# on-the-fly cost volume (no [h, w, w] correlation volume): "1" always, "0" never, "auto": when the volume would exceed
# CV_OTF_BYTES (the lookup path then keeps the two LayerNormed feature maps, 2*h*w*C fp16, instead of h*w*w fp16)
_CV_OTF = os.environ.get("S2M2_CV_OTF", "auto")
CV_OTF_BYTES = int(float(os.environ.get("S2M2_CV_OTF_MB", "256")) * 2**20)
_INPLACE_ATTN = (
    os.environ.get("S2M2_INPLACE_ATTN", "1") == "1"
)  # row-attention blocks write their strips back in place
# low-memory mode for very large inputs (exact; slower GRU): ConvGRU in row strips instead of full-size fused buffers.
# "auto": when one image's 1/4-res C-channel map exceeds LOWMEM_MB (XL: > ~16 MP input)
_ATTN384 = (
    os.environ.get("S2M2_ATTN384", "1") == "1"
)  # fused flash-style kernel for head_dim 384 (XL level 0)
_LOWMEM = os.environ.get("S2M2_LOWMEM", "auto")
LOWMEM_BYTES = int(float(os.environ.get("S2M2_LOWMEM_MB", "768")) * 2**20)
_LM = [False]
GRU_STRIP_BYTES = 128 * 2**20
_INPLACE_CB = (
    os.environ.get("S2M2_INPLACE_CB", "1") == "1"
)  # Unet ConvBlock2D strips written back in place
_INPLACE_BB = (
    os.environ.get("S2M2_INPLACE_BB", "1") == "1"
)  # backbone conv2 block written back in place (exact)
CV_CHUNK_BYTES = (
    256 * 2**20
)  # correlation rows materialized at a time for the optimal transport (on-the-fly mode)
ATTN_BYTES = 256 * 2**20  # budget of one explicit (relative-PE) attention score chunk


def _cl(x):
    return x.contiguous(memory_format=CL)


# ------------------------------------------------------------------------------------------------ epilogues
_OPT = {"kern": False, "hacc": False, "cut": False}
ACT = {nn.GELU: 6, nn.ReLU: 1, nn.Sigmoid: 4}
_ACT_T = {
    0: lambda v: v,
    1: torch.relu_,
    3: torch.tanh_,
    4: torch.sigmoid_,
    6: torch.ops.aten.gelu_,
    7: lambda v: v.sigmoid_().clamp_(0.01, 0.99),
}
_K = [None]


def kern():
    if _K[0] is None:
        from .kernels import Kernels

        _K[0] = Kernels.get()
    return _K[0]


def _dense_rows(y, cl):
    """C if y is dense [P, C] rows (channels-last 4D if cl, else row-major), else None."""
    if y.dtype != torch.half:
        return None
    if cl:
        return y.shape[1] if y.dim() == 4 and y.is_contiguous(memory_format=CL) else None
    return y.shape[-1] if y.is_contiguous() else None


def epi(y, b=None, act=0, r=None, cl=True):
    """in place: y = act(y + b) (+ r). y channels-last 4D (cl) or row-major [..., C]; b fp32 [C] or None."""
    if b is None and act == 0 and r is None:
        return y
    C = _dense_rows(y, cl) if _OPT["kern"] else None
    if C is not None and (r is None or (r.shape == y.shape and _dense_rows(r, cl) == C)):
        P = y.numel() // C
        k = kern()
        if C % 8 == 0:
            k.launch(
                "bias_act8",
                k.linear(P * (C // 8)),
                y,
                0 if b is None else b,
                0 if r is None else r,
                P,
                C,
                act,
            )
        else:
            k.launch(
                "bias_act",
                k.linear(P * C),
                y,
                0 if b is None else b,
                0 if r is None else r,
                P,
                C,
                act,
            )
        return y
    if b is not None:
        y = y.add_(b.to(y.dtype).view((1, -1, 1, 1) if cl else (-1,)))
    y = _ACT_T[act](y)
    return y.add_(r) if r is not None else y


def conv_raw(m, x):
    """prepared Conv2d m without its bias, channels-last out."""
    if m._hacc:
        from .cudnn_conv import conv_half_acc

        return conv_half_acc(x, m.weight, None, list(m.padding), list(m.stride))
    return _cl(F.conv2d(x, m.weight, None, m.stride, m.padding))


def _cut_conv(w, b, x, padding, stride, act, r, out=None):
    """CUTLASS conv with the bias / act / residual epilogue fused (s2m2_cutlass.cu mode 0); None if not applicable.
    out: dense channels-last destination of the right shape (else allocated)."""
    x = _cl(x)
    N, C, H, W = x.shape
    K, _, R, S = w.shape
    Ho, Wo = (H + 2 * padding[0] - R) // stride + 1, (W + 2 * padding[1] - S) // stride + 1
    if r is not None and (
        tuple(r.shape) != (N, K, Ho, Wo) or not r.is_contiguous(memory_format=CL)
    ):
        return None
    if out is not None and (
        tuple(out.shape) != (N, K, Ho, Wo) or not out.is_contiguous(memory_format=CL)
    ):
        return None
    y = (
        out
        if out is not None
        else torch.empty((N, K, Ho, Wo), device=x.device, dtype=x.dtype, memory_format=CL)
    )
    from .cutlass_conv import epi_args

    if cut().conv(_tile(R, S), x, w, padding, stride, epi_args(0, act, b, r, K, y, K)):
        return None
    return y


def _tile(R, S):
    """fixed tile rule (bench/s2m2_cutlass_probe.py): 1x1 convs / linears 128x128x32 with 8 warps (cfg 8), spatial
    convs 128x128x32 with 4 warps (cfg 0, = cuDNN's precompiled tile)."""
    return 8 if R * S == 1 else 0


def conv_ep(m, x, act=0, r=None):
    """Conv2d m (bias kept aside as fp32 m._b32) + one epilogue pass: act(conv(x) + b) (+ r)."""
    if not hasattr(m, "_b32"):
        return epi(_cl(m(x)), None, act, r)
    if (getattr(m, "_cut", False) and (act or r is not None or m._b32 is not None)) or (
        r is not None and getattr(m, "_cut_res", False)
    ):
        y = _cut_conv(m.weight, m._b32, x, m.padding, m.stride[0], act, r)
        if y is not None:
            return y
    return epi(conv_raw(m, x), m._b32, act, r)


def _conv_forward(self, x):
    return conv_ep(self, x)


def lin(w, b, x, act=0, r=None, out=None):
    """x [..., C] row-major @ w[K, C]^T, then act(. + b) (+ r) in one pass. hacc: as a 1x1 fp16-accumulate cuDNN conv.
    out: optional dense [P, K] destination (CUTLASS path writes it directly, else copied)."""
    C = x.shape[-1]
    K = w.shape[0]
    x2 = x.reshape(-1, C)
    P = x2.shape[0]
    if out is not None:
        y = lin(w, b, x, act, r)
        if y.data_ptr() != out.data_ptr():
            out.copy_(y.reshape(out.shape))
        return out
    if (
        (r is not None or b is not None or act)
        and _CUT_LIN
        and _OPT["cut"]
        and C % 8 == 0
        and K % 8 == 0
        and K >= 128
        and x2.is_contiguous()
    ):
        r2 = None if r is None else r.reshape(-1, K)
        if (
            r2 is None or r2.is_contiguous()
        ):  # CUTLASS 1x1 conv with bias / act / residual in the epilogue (one pass)
            y = (
                _LIN_OUT[0]
                if _LIN_OUT[0] is not None
                else torch.empty((P, K), device=x.device, dtype=x.dtype)
            )
            from .cutlass_conv import epi_args

            x4 = torch.as_strided(x2, (1, C, 1, P), (P * C, 1, P * C, C))
            if not cut().conv(
                8, x4, w.view(K, C, 1, 1), (0, 0), 1, epi_args(0, act, b, r2, K, y, K)
            ):
                return y.view(*x.shape[:-1], K)
    if _OPT["hacc"]:
        from .cudnn_conv import conv_half_acc

        x4 = torch.as_strided(x2, (1, C, 1, P), (P * C, 1, P * C, C))
        y = (
            conv_half_acc(x4, w.view(K, C, 1, 1), None, [0, 0], [1, 1])
            .permute(0, 2, 3, 1)
            .reshape(P, K)
        )
    else:
        y = torch.mm(x2, w.t())
    y = epi(y, b, act, None if r is None else r.reshape(-1, K), cl=False)
    return y.view(*x.shape[:-1], K)


_LIN_OUT = [None]  # destination for the next CUTLASS residual linear (set by _ffn_ with out=)


def run_seq(seq, x, r=None):
    """nn.Sequential with conv + activation fused into one epilogue (kern) or in-place activations; r = residual
    added after the last module (only if it is a conv)."""
    mods = list(seq)
    i = 0
    while i < len(mods):
        m = mods[i]
        nxt = mods[i + 1] if i + 1 < len(mods) else None
        last = i + 1 == len(mods)
        if hasattr(m, "_b32") and type(nxt) in ACT:
            x = conv_ep(m, x, ACT[type(nxt)], r if i + 2 == len(mods) else None)
            if i + 2 == len(mods):
                r = None
            i += 2
            continue
        if hasattr(m, "_b32") and last and r is not None:
            x = conv_ep(m, x, 0, r)
            r = None
        elif type(m) in ACT:
            x = _ACT_T[ACT[type(m)]](x)
        else:
            x = m(x)
        i += 1
    return x if r is None else x.add_(r)


STRIP_BYTES = (
    int(os.environ.get("S2M2_STRIP_MB", 32)) * 2**20
)  # inputs of one row strip of a local (conv / row-attention) block


def strips(fn, xs, halo, per_image=False):
    """y = fn(*xs) for a block that is local in y (receptive field +-halo rows; halo 0 = row-wise), evaluated in row
    strips with halo so only one strip of the block's intermediates is alive. Exact: zero padding at a strip edge only
    touches halo rows, which are dropped. Small inputs: one call."""
    H, W = xs[0].shape[-2:]
    per_row = sum(x.shape[0] * x.shape[1] for x in xs) * W * xs[0].element_size()
    rows = max(8, STRIP_BYTES // per_row)
    if H <= rows + 2 * halo:
        return fn(*xs)
    N = xs[0].shape[0]
    # per_image (block independent per batch element): batch-1 row strips of channels-last tensors are dense views
    groups = [slice(i, i + 1) for i in range(N)] if per_image and N > 1 else [slice(0, N)]
    rows = rows * (N if per_image else 1)
    out = None
    for g in groups:
        for r0 in range(0, H, rows):
            r1 = min(H, r0 + rows)
            a, b = max(0, r0 - halo), min(H, r1 + halo)
            y = fn(*[x[g, :, a:b] for x in xs])
            if out is None:
                out = torch.empty(
                    (N, y.shape[1], H, W), device=y.device, dtype=y.dtype, memory_format=CL
                )
            out[g, :, r0:r1] = y[:, :, r0 - a : r0 - a + r1 - r0]
            del y
    return out


def strips_inplace(fn, x, halo):
    """x = fn(x) for a local block (receptive field +-halo rows, C_out == C_in) in row strips written back into x: the
    original halo rows above each strip are saved before they are overwritten (no second full-size map). Same strip
    inputs as strips() -> same result."""
    N, C, H, W = x.shape
    rows = max(8, STRIP_BYTES // (N * C * W * x.element_size()))
    if H <= rows + 2 * halo:
        return fn(x)
    prev = None  # original rows [r0 - halo, r0)
    for r0 in range(0, H, rows):
        r1 = min(H, r0 + rows)
        cur = x[:, :, r0 : min(H, r1 + halo)]
        y = fn(cur if prev is None else torch.cat([prev, cur], 2))
        top = 0 if prev is None else prev.shape[2]
        prev = x[:, :, max(r0, r1 - halo) : r1].clone() if r1 < H else None
        x[:, :, r0:r1] = y[:, :, top : top + r1 - r0]
        del y
    return x


def strips_into_inplace(fn, x, halo):
    """strips_into(fn, [x], halo, C) writing each image's result rows back into x (C_out == C_in): the original halo
    rows above a strip are saved before they are overwritten; fn reads its strip input before writing dst."""
    N, C, H, W = x.shape
    rows = max(8, STRIP_BYTES // (C * W * x.element_size())) * N
    if H <= rows + 2 * halo:
        rows = H
    for i in range(N):
        prev = None
        for r0 in range(0, H, rows):
            r1 = min(H, r0 + rows)
            cur = x[i : i + 1, :, r0 : min(H, r1 + halo)]
            inp = cur if prev is None else torch.cat([prev, cur], 2)
            top = 0 if prev is None else prev.shape[2]
            prev = x[i : i + 1, :, max(r0, r1 - halo) : r1].clone() if r1 < H else None
            fn([inp], top, r1 - r0, x[i : i + 1, :, r0:r1])
            del inp, cur
    return x


def strips_into(fn, xs, halo, cout):
    """per-image row strips (as strips(..., per_image=True)) for blocks that write their result rows straight into the
    output: fn(strip_xs, top, nrows, dst) writes rows [top, top + nrows) of the strip's result into dst (a dense
    channels-last [1, cout, nrows, W] view of the output). No per-strip output copy."""
    N, _, H, W = xs[0].shape
    per_row = sum(x.shape[1] for x in xs) * W * xs[0].element_size()
    rows = max(8, STRIP_BYTES // per_row) * N
    out = torch.empty((N, cout, H, W), device=xs[0].device, dtype=xs[0].dtype, memory_format=CL)
    for i in range(N):
        for r0 in range(0, H, rows if H > rows + 2 * halo else H):
            r1 = min(H, r0 + rows) if H > rows + 2 * halo else H
            a, b = max(0, r0 - halo), min(H, r1 + halo)
            fn([x[i : i + 1, :, a:b] for x in xs], r0 - a, r1 - r0, out[i : i + 1, :, r0:r1])
    return out


def up_conv(seq, x):
    """Sequential(Upsample(x2 bilinear), Conv1x1) evaluated as conv1x1 then upsample (exact up to rounding)."""
    y = _cl(seq[1](x))
    N, C, H, W = y.shape
    if _OPT["kern"] and C % 8 == 0:
        out = torch.empty((N, C, 2 * H, 2 * W), device=y.device, dtype=y.dtype, memory_format=CL)
        _up2x_into(y, out, C)
        return out
    return _cl(F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False))


def _up2x_into(y, dst, ys, oy0=0, nrows=None):
    """bilinear x2 of channels-last y into dst (pointer of a channels-last tensor or slot view, pixel stride ys);
    only output rows [oy0, oy0 + nrows) if given."""
    N, C, H, W = y.shape
    nrows = 2 * H if nrows is None else nrows
    k = kern()
    k.launch(
        "up2x_nhwc8", k.linear(N * nrows * 2 * W * (C // 8)), y, dst, N, H, W, C, ys, oy0, nrows
    )


def ff_up(ff, z_skip, seq, z_low):
    """FeatureFusion(z_skip, up_conv(seq, z_low)) without materializing the upsampled tensor or the concat copy: z_skip
    is copied into slot 0 of the concat buffer, the x2 upsampling kernel writes slot 1, the gate mix reads both slots."""
    y = _cl(seq[1](z_low))
    N, C, h, w = y.shape
    H, W = z_skip.shape[-2:]
    if not (
        _OPT["kern"]
        and ff.use_gate
        and C % 8 == 0
        and z_skip.shape[1] == C
        and (H, W) == (2 * h, 2 * w)
        and ff.feature_fusion[0].kernel_size == (1, 1)
    ):
        return ff(
            z_skip,
            _cl(F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False))
            if not _OPT["kern"]
            else up_conv(seq, z_low),
        )
    if not (getattr(ff.feature_fusion[2], "_cut", False)):
        zc = torch.empty((N, 2 * C, H, W), device=y.device, dtype=y.dtype, memory_format=CL)
        zc[:, :C] = z_skip
        _up2x_into(y, zc[:, C:], 2 * C)
        del y
        return _ff_cat_strips(ff, zc, C)
    # per-image row strips: the concat buffer exists only for one strip (memory), output rows written in place
    out = torch.empty((N, C, H, W), device=y.device, dtype=y.dtype, memory_format=CL)
    rows = max(8, STRIP_BYTES // (2 * C * W * y.element_size()))
    for i in range(N):
        for r0 in range(0, H, rows):
            r1 = min(H, r0 + rows)
            zc = torch.empty(
                (1, 2 * C, r1 - r0, W), device=y.device, dtype=y.dtype, memory_format=CL
            )
            zc[:, :C] = z_skip[i : i + 1, :, r0:r1]
            _up2x_into(y[i : i + 1], zc[:, C:], 2 * C, r0, r1 - r0)
            _ff_core_cat_into(ff, zc, out[i : i + 1, :, r0:r1])
            del zc
    return out


def ff_down(ff, z_skip, seq, z_hi):
    """FeatureFusion(z_skip, Sequential(AvgPool2d(2), Conv1x1)(z_hi)) with the 1x1 conv (CUTLASS, bias epilogue) writing
    straight into slot 1 of the concat buffer (no cat copy of the conv output)."""
    conv = seq[1]
    C = z_skip.shape[1]
    if not (
        _OPT["cut"]
        and ff.use_gate
        and ff.feature_fusion[0].kernel_size == (1, 1)
        and conv.weight.shape[0] == C
        and C % 8 == 0
        and conv.weight.shape[1] % 8 == 0
        and hasattr(conv, "_b32")
    ):
        return ff(z_skip, run_seq(seq, z_hi))
    p = _cl(seq[0](z_hi))
    N, _, H, W = p.shape
    if (H, W) != tuple(z_skip.shape[-2:]):
        return ff(z_skip, run_seq(seq, z_hi))
    zc = torch.empty((N, 2 * C, H, W), device=p.device, dtype=p.dtype, memory_format=CL)
    from .cutlass_conv import epi_args

    if cut().conv(
        8, p, conv.weight, (0, 0), 1, epi_args(0, 0, conv._b32, None, 0, zc[:, C:], 2 * C)
    ):
        return ff(z_skip, run_seq(seq, z_hi))
    del p
    zc[:, :C] = z_skip
    return _ff_cat_strips(ff, zc, C)


def _ff_cat_strips(ff, zc, C):
    if getattr(ff.feature_fusion[2], "_cut", False) and C % 8 == 0:
        return strips_into(lambda t, top, n, dst: _ff_core_cat_into(ff, t[0], dst), [zc], 0, C)
    return strips(lambda t: _ff_core_cat(ff, t), [zc], 0, per_image=True)


def _ff_core_cat_into(self, z, dst):
    """_ff_core_cat writing into dst: the output 1x1 conv (CUTLASS, raw) writes dst, the gate mix runs in place on it."""
    C = z.shape[1] // 2
    f = self.feature_fusion
    if not (getattr(f[2], "_cut", False) and C % 8 == 0):
        dst.copy_(_ff_core_cat(self, z))
        return
    g = self.feature_gate
    z = _cl(z)
    w = conv_ep(g[2], conv_ep(g[0], z, 6), 7)
    h = conv_ep(f[0], z, 6)
    from .cutlass_conv import epi_args

    if cut().conv(8, h, f[2].weight, (0, 0), 1, epi_args(0, 0, None, None, 0, dst, C)) or not (
        w.shape == dst.shape and _dense_rows(w, True) == C
    ):
        dst.copy_(_ff_core_cat(self, z))
        return
    del h
    P = dst.numel() // C
    k = kern()
    k.launch("gate_mix8s", k.linear(P * (C // 8)), dst, f[2]._b32, w, z, z[:, C:], P, C, 2 * C)


def _ff_core_cat(self, z):
    """_ff_core on the concat input z = [z0 | z1] (1x1 kernels: halo 0)."""
    z = _cl(z)
    C = z.shape[1] // 2
    g, f = self.feature_gate, self.feature_fusion
    w = conv_ep(g[2], conv_ep(g[0], z, 6), 7)
    out = conv_raw(f[2], conv_ep(f[0], z, 6))
    if (
        w.shape == out.shape
        and _dense_rows(w, True) == C
        and _dense_rows(out, True) == C
        and C % 8 == 0
    ):
        P = out.numel() // C
        k = kern()
        k.launch("gate_mix8s", k.linear(P * (C // 8)), out, f[2]._b32, w, z, z[:, C:], P, C, 2 * C)
        return out
    z0, z1 = z[:, :C], z[:, C:]
    out = epi(out, f[2]._b32)
    return out.add_(torch.sub(z0, z1).mul_(w)).add_(z1)


def _ln(x):
    C = x.shape[-1]
    if _OPT["kern"] and x.dtype == torch.half and x.is_contiguous() and C % 4 == 0 and C <= 1024:
        y = torch.empty_like(x)
        P = x.numel() // C
        k = kern()
        k.launch("ln_rows", (((P * 32 + 255) // 256, 1, 1), (256, 1, 1)), x, y, P, C, 1e-5)
        return y
    return F.layer_norm(x, x.shape[-1:])


# ------------------------------------------------------------------------------------------------ conv blocks
def _ff_forward(self, z0, z1):
    k = self.feature_fusion[0].kernel_size[0]
    return strips(lambda a, b: _ff_core(self, _cl(a), _cl(b)), [z0, z1], k // 2, per_image=True)


def _ff_core(self, z0, z1):
    z = torch.cat([z0, z1], dim=1)
    if not self.use_gate:
        return run_seq(self.feature_fusion, z)
    g, f = self.feature_gate, self.feature_fusion
    if _OPT["kern"]:
        w = conv_ep(g[2], conv_ep(g[0], z, 6), 7)
        out = conv_raw(f[2], conv_ep(f[0], z, 6))
        del z
        if (
            all(_dense_rows(t, True) == out.shape[1] and t.shape == out.shape for t in (w, z0, z1))
            and out.shape[1] % 8 == 0
        ):
            P = out.numel() // out.shape[1]
            k = kern()
            k.launch(
                "gate_mix8",
                k.linear(P * (out.shape[1] // 8)),
                out,
                f[2]._b32,
                w,
                z0,
                z1,
                P,
                out.shape[1],
            )
            return out
        out = epi(out, f[2]._b32)
    else:
        w = run_seq(g, z).clamp_(0.01, 0.99)
        out = run_seq(f, z)
        del z
    d = torch.sub(z0, z1).mul_(w)
    return out.add_(d).add_(z1)


def _convblock_forward(self, z):
    l1 = self.convs_1x[2]
    if (
        _OPT["cut"]
        and getattr(l1, "_cut_res", False)
        and hasattr(l1, "_b32")
        and _INPLACE_CB
        and l1.weight.shape[0] == z.shape[1]
        and z.is_contiguous(memory_format=CL)
    ):
        # in place (no call site reuses a ConvBlock2D input: Unet enc*/dec* get temporaries)
        return strips_into_inplace(
            lambda t, top, n, dst: _convblock_core_into(self, _cl(t[0]), top, n, dst), z, 2
        )
    if _OPT["cut"] and getattr(l1, "_cut_res", False) and hasattr(l1, "_b32"):
        return strips_into(
            lambda t, top, n, dst: _convblock_core_into(self, _cl(t[0]), top, n, dst),
            [z],
            2,
            l1.weight.shape[0],
        )
    return strips(lambda t: _convblock_core(self, _cl(t)), [z], 2, per_image=True)


def _convblock_core_into(self, z, top, n, dst):
    """3x3 path on the strip with halo; the pointwise 1x1 path only on the strip's own rows, its last conv adds the 3x3
    path as residual and writes the output rows (CUTLASS epilogue) straight into dst."""
    a = run_seq(self.convs, z)
    zi, ai = z[:, :, top : top + n], a[:, :, top : top + n]
    c0, c2 = self.convs_1x[0], self.convs_1x[2]
    h = conv_ep(c0, zi, ACT[type(self.convs_1x[1])])
    if _cut_conv(c2.weight, c2._b32, h, c2.padding, 1, 0, ai, out=dst) is None:
        dst.copy_(conv_ep(c2, h, 0, _cl(ai)))


def _convblock_core(self, z):
    a = run_seq(self.convs, z)
    return run_seq(self.convs_1x, z, r=a)


# ------------------------------------------------------------------------------------------------ attention
def _heads(t, n, heads, hd):
    """[..., heads*hd] (any batch dims flattened to n sequences) -> [n, heads, L, hd] strided view."""
    return t.reshape(n, -1, heads, hd).transpose(1, 2)


def _qkv(attn, zn):
    return lin(attn._wqkv, None, zn)  # v bias folded into attn._pbias (softmax rows sum to 1)


def sdpa(q, k, v, out=None):
    """flash attention needs head_dim <= 256; above (XL level 0: one 384-d head) PyTorch falls back to the sm80
    mem-efficient kernel, slower here than explicit fp16 matmul + softmax (called on row strips: bounded memory)."""
    if q.shape[-1] <= 256 or _SDPA_ALWAYS:
        o = F.scaled_dot_product_attention(q, k, v)
        if out is not None:
            out.copy_(o.transpose(1, 2).reshape(out.shape))
        return o
    if _ATTN384 and _OPT["kern"] and q.shape[-1] == 384 and q.dtype == torch.half and q.is_cuda:
        y = _attn384(q, k, v, out)
        if y is not False:
            return y
    mm = torch.backends.cuda.matmul
    prev = mm.allow_fp16_accumulation
    mm.allow_fp16_accumulation = (
        _OPT["hacc"] and _ATT16
    )  # fp16-accumulate cuBLAS GEMMs (2x tensor-core rate on GeForce)
    try:
        n, h, L, d = q.shape
        q3, k3, v3 = (
            t.reshape(n * h, t.shape[2], d) for t in (q, k, v)
        )  # views for the strided q|k|v slices
        a = torch.baddbmm(
            q3.new_empty(()).expand(n * h, L, k3.shape[1]),
            q3,
            k3.transpose(1, 2),
            beta=0,
            alpha=d**-0.5,
        ).softmax(-1)  # scale inside the GEMM: no extra pass over q
        if (
            out is not None and h == 1
        ):  # [n, L, d] rows of the caller's merged output buffer: no merge / cat copy
            torch.bmm(a, v3, out=out.view(n, L, d))
            return None
        o = torch.bmm(a, v3).view(n, h, L, d)
        if out is not None:
            out.copy_(o.transpose(1, 2).reshape(out.shape))
        return o
    finally:
        mm.allow_fp16_accumulation = prev


def _attn384(q, k, v, out=None):
    """attn_d384 kernel (s2m2_kernels.cu, flash-style, fp16-accumulate MMAs) on [n, H, L, 384] strided views sharing
    strides (q|k|v slices of one projection, or of two same-shaped ones). Writes out ([n*L, 384] rows, H == 1) or
    returns o [n, H, L, 384]; False if the layout is not supported."""
    n, H, L, d = q.shape
    st = q.stride()
    if (
        k.stride() != st
        or v.stride() != st
        or st[3] != 1
        or k.shape != q.shape
        or v.shape != q.shape
        or any((t.data_ptr() % 16) or (st[2] % 8) for t in (q, k, v))
    ):
        return False
    if out is not None and H == 1:
        o, ost, ret = out, (L * d, 0, d), None
    else:
        o = torch.empty((n, H, L, d), device=q.device, dtype=q.dtype)
        ost, ret = o.stride()[:3], o
    kern().launch(
        "attn_d384",
        (((L + 63) // 64, n * H, 1), (128, 1, 1)),
        q,
        k,
        v,
        o,
        st[0],
        st[1],
        st[2],
        *ost,
        H,
        L,
        float(d**-0.5),
        smem=2 * 32 * (384 + 8) * 2,
    )
    if out is not None and ret is not None:
        out.copy_(ret.transpose(1, 2).reshape(out.shape))
    return ret


def _merge(o):
    """[n, heads, L, hd] -> [n*L, heads*hd]."""
    n, h, L, d = o.shape
    return o.transpose(1, 2).reshape(n * L, h * d)


def _cross(attn, z, seq_shape):
    """symmetric cross attention between the two batch halves of z (NHWC or [B, N, C]), residual included,
    out of place (z may be a block input that is still needed). seq_shape = (sequences per half, length)."""
    zn = _ln(z)
    qkv = _qkv(attn, zn)
    del zn
    D = attn.num_heads * attn.head_dim
    B = z.shape[0] // 2
    q, k, v = qkv[..., :D], qkv[..., D : 2 * D], qkv[..., 2 * D :]
    n = seq_shape[0]

    def hv(t, s):
        return _heads(t[s], n, attn.num_heads, attn.head_dim)

    a, b = slice(0, B), slice(B, None)
    if (
        attn.head_dim > 256 and not _SDPA_ALWAYS
    ):  # explicit path writes both halves into one buffer (no cat)
        o = torch.empty((z.numel() // z.shape[-1], D), device=z.device, dtype=z.dtype)
        half = o.shape[0] // 2
        sdpa(hv(q, a), hv(k, b), hv(v, b), out=o[:half])
        sdpa(hv(q, b), hv(k, a), hv(v, a), out=o[half:])
        del qkv, q, k, v
        return lin(attn.proj.weight, attn._pbias, o, r=z).view(z.shape)
    xo = sdpa(hv(q, a), hv(k, b), hv(v, b))
    yo = sdpa(hv(q, b), hv(k, a), hv(v, a))
    del qkv, q, k, v
    o = torch.cat([_merge(xo), _merge(yo)], 0)
    del xo, yo
    return lin(attn.proj.weight, attn._pbias, o, r=z).view(z.shape)


def _self(attn, z, n, pe=None):
    """self attention over n sequences of z (flattened) + residual; returns the new z."""
    zn = _ln(z)
    qkv = _qkv(attn, zn)
    del zn
    D = attn.num_heads * attn.head_dim
    q, k, v = (
        _heads(qkv[..., i * D : (i + 1) * D], n, attn.num_heads, attn.head_dim) for i in range(3)
    )
    if pe is None:
        o = sdpa(q, k, v)
    else:
        o = _attn_rel_pe(attn, q, k, v, pe)
    del qkv, q, k, v
    return lin(attn.proj.weight, attn._pbias, _merge(o), r=z).view(z.shape)


def _attn_rel_pe(attn, q, k, v, pe):
    """softmax(q k^T * scale) v + pe_proj(sum_j attn_ij pe_ij), without the [N, N, 32] pe tensor:
    pe_ij = .5 [px(x_i - x_j) | py(y_i - y_j)], so sum_j attn_ij px(.) = sum_{x_j} (sum_{y_j} attn_ij) Tx[x_i, x_j]."""
    Tx, Ty, h, w = pe
    n, H, N, hd = q.shape
    d2 = (hd + w + h + 7) // 8 * 8
    if _OPT["kern"] and _RELPE_FLASH and d2 <= 256:
        return _attn_rel_pe_flash(attn, q, k, v, pe, d2)
    out = torch.empty((n, H, N, hd), device=q.device, dtype=q.dtype)
    per = max(1, ATTN_BYTES // (N * N * 2 * 3))
    qf, kf, vf, of = (t.reshape(n * H, N, hd) for t in (q, k, v, out))
    for s in range(0, n * H, per):
        e = min(n * H, s + per)
        a = torch.bmm(qf[s:e] * attn.scale, kf[s:e].transpose(1, 2)).softmax(-1)  # [c, N, N]
        o = torch.bmm(a, vf[s:e])
        # marginals of the attention over yj / xj (i = (yi, xi), j = (yj, xj)) as GEMVs with a ones vector (fp32
        # accumulation, contiguous reads; a strided sum over yj is ~10x slower)
        a3 = a.view(-1, h, w)
        mx = torch.matmul(a.new_ones((1, 1, h)), a3).view(e - s, h, w, w)  # [c, yi, xi, xj]
        my = torch.matmul(a3, a.new_ones((1, w, 1))).view(e - s, h, w, h)  # [c, yi, xi, yj]
        del a, a3
        pex = torch.einsum("cyax,axd->cyad", mx, Tx)
        pey = torch.einsum("cyaj,yjd->cyad", my, Ty)
        ps = torch.cat([pex, pey], -1).view(e - s, N, -1)
        of[s:e] = lin(attn.pe_proj.weight, attn.pe_proj._b32, ps, r=o)
    return out


def _attn_rel_pe_flash(attn, q, k, v, pe, d2):
    """_attn_rel_pe with ONE flash-attention call and no [N, N] score tensor: the attention marginals over y_j / x_j are
    attention outputs for one-hot values, so V' = [v | onehot(x_j) | onehot(y_j)] (q, k zero-padded to the same width;
    zeros do not change the scores) gives o, mx[i, x_j] = sum_yj a_ij and my[i, y_j] = sum_xj a_ij in one pass."""
    Tx, Ty, h, w = pe
    n, H, N, hd = q.shape
    dev, dt = q.device, q.dtype
    Q = torch.zeros((n, H, N, d2), device=dev, dtype=dt)
    K = torch.zeros((n, H, N, d2), device=dev, dtype=dt)
    V = torch.zeros((n, H, N, d2), device=dev, dtype=dt)
    Q[..., :hd] = q
    K[..., :hd] = k
    V[..., :hd] = v
    j = torch.arange(N, device=dev)
    oh = torch.zeros((N, w + h), device=dev, dtype=dt)
    oh[j, j % w] = 1
    oh[j, w + j // w] = 1
    V[..., hd : hd + w + h] = oh
    o2 = F.scaled_dot_product_attention(Q, K, V, scale=attn.scale)
    del Q, K, V
    c = n * H
    o = o2[..., :hd]
    mx = o2[..., hd : hd + w].reshape(c, h, w, w)  # [c, yi, xi, xj]
    my = o2[..., hd + w : hd + w + h].reshape(c, h, w, h)  # [c, yi, xi, yj]
    pex = torch.einsum("cyax,axd->cyad", mx, Tx)
    pey = torch.einsum("cyaj,yjd->cyad", my, Ty)
    ps = torch.cat([pex, pey], -1).view(c, N, -1)
    out = lin(attn.pe_proj.weight, attn.pe_proj._b32, ps, r=o.reshape(c, N, hd).contiguous())
    return out.view(n, H, N, hd)


def _pe_tables(h, w, pe_dim, dtype, device):
    """1D halves of utils.get_pe (same arithmetic in `dtype`): Tx[x_i, x_j] = .5 * pe_x[x_i - x_j + w - 1], Ty likewise."""

    def table(L_n):
        L = 2 * L_n + 1
        sig = 5 / pe_dim
        pos = torch.linspace(-3, 3, L, device=device).to(dtype).tanh()
        dim_t = torch.linspace(-1, 1, pe_dim // 2, device=device).to(dtype)
        p = custom_sinc((dim_t[None, :] - pos[:, None]) / sig)
        p = F.normalize(p, p=2, dim=-1)
        i = torch.arange(L_n, device=device)
        return (0.5 * p[(i[:, None] - i[None, :]) + L_n - 1]).to(dtype).contiguous()

    return table(w), table(h)


def _ffn_(ffn, z, out=None):
    """FFN block with residual; out: dense [P, C] destination (rows of the block's output strip: no strip copy)."""
    l1, l2 = ffn.ffn[0], ffn.ffn[2]
    hdn = lin(l1.weight, l1._b32, _ln(z), act=6)
    if out is None:
        return lin(l2.weight, l2._b32, hdn, r=z)
    _LIN_OUT[0] = out
    try:
        return lin(l2.weight, l2._b32, hdn, r=z, out=out)
    finally:
        _LIN_OUT[0] = None


def _attn_pair(attn, z0, z1, cross):
    """(cross or self) attention + residual on the two images' row strips z0, z1 [1, rows, W, C] (dense views), each
    processed on its own (no copy of the two strips into one batch). Cross: image 0 attends to image 1 and vice versa."""
    D = attn.num_heads * attn.head_dim
    qkv = [_qkv(attn, _ln(z)) for z in (z0, z1)]
    n = z0.shape[0] * z0.shape[1]

    def hv(t, i):
        return _heads(t[..., i * D : (i + 1) * D], n, attn.num_heads, attn.head_dim)

    res = []
    for a in (0, 1):
        b = 1 - a if cross else a
        q, k, v = hv(qkv[a], 0), hv(qkv[b], 1), hv(qkv[b], 2)
        if attn.head_dim > 256 and not _SDPA_ALWAYS:
            o = torch.empty((z0.numel() // z0.shape[-1], D), device=z0.device, dtype=z0.dtype)
            sdpa(q, k, v, out=o)
        else:
            o = _merge(sdpa(q, k, v))
        res.append(o)
    del qkv
    return [lin(attn.proj.weight, attn._pbias, o, r=z).view(z.shape) for o, z in zip(res, (z0, z1))]


def _basic_attn_pair_forward(self, z):
    """BasicAttnBlock on the 2-image channels-last z in row strips; per strip the two images' rows are dense views
    (inputs read in place) and the last FFN writes straight into the output rows (no strip copies)."""
    N, C, H, W = z.shape
    z = _cl(z)
    rows = max(8, STRIP_BYTES // (N * C * W * z.element_size()))
    # in place: a strip's output rows depend only on the same input rows, which are fully read (into new tensors) by the
    # first attention of the strip; no call site reuses the block input (forward_fast keeps its own copy of py4[0])
    out = z if _INPLACE_ATTN else torch.empty_like(z, memory_format=CL)
    for r0 in range(0, H, rows):
        r1 = min(H, r0 + rows)
        zs = [
            z[i : i + 1, :, r0:r1].permute(0, 2, 3, 1) for i in (0, 1)
        ]  # [1, rows, W, C] dense views
        zs = _attn_pair(self.cross_attn.attn, zs[0], zs[1], True)
        zs = [_ffn_(self.ffn_c, t) for t in zs]
        zs = _attn_pair(self.self_attn.attn, zs[0], zs[1], False)
        for i, t in enumerate(zs):
            dst = out[i : i + 1, :, r0:r1].permute(0, 2, 3, 1)
            assert dst.is_contiguous()
            _ffn_(self.ffn, t, out=dst.view(-1, C))
        del zs
    return out


def _basic_attn_forward(self, z, pe=None):
    """1D (row-wise) attention block on a channels-last [B, C, H, W] tensor, in row strips; returns channels-last."""
    N, C, H, W = z.shape
    if (
        N == 2
        and _OPT["kern"]
        and _PAIR_ATTN
        and H > max(8, STRIP_BYTES // (N * C * W * z.element_size()))
    ):
        return _basic_attn_pair_forward(
            self, z
        )  # several strips: avoid the strip copies (one strip: no copies anyway)
    return strips(lambda t: _basic_attn_core(self, t), [z], 0)


def _basic_attn_core(self, z):
    zh = z.permute(0, 2, 3, 1).contiguous()  # free view for channels-last input
    B2, H, W, C = zh.shape
    zh = _cross(
        self.cross_attn.attn, zh, (B2 // 2 * H, W)
    )  # out of place: the block input stays intact
    zh = _ffn_(self.ffn_c, zh)
    zh = _self(self.self_attn.attn, zh, B2 * H)
    zh = _ffn_(self.ffn, zh)
    return zh.permute(0, 3, 1, 2)


def _global_attn_forward(self, z, pe=None):
    zh = z.permute(0, 2, 3, 1).contiguous()
    B, H, W, C = zh.shape
    zh = zh.view(B, H * W, C)
    if self.cross_attn is not None:
        zh = _cross(self.cross_attn.attn, zh, (B // 2, H * W))
        zh = _ffn_(self.ffn_c, zh)
    zh = _self(self.self_attn.attn, zh, B, pe if self.self_attn.attn.use_pe else None)
    zh = _ffn_(self.ffn, zh)
    return zh.view(B, H, W, C).permute(0, 3, 1, 2)


# ------------------------------------------------------------------------------------------------ unet / MRT
def _unet_forward(self, z):
    pe = None
    if self.use_pe:
        H, W = z.shape[-2:]
        h, w = H // 8, W // 8
        Tx, Ty = _pe_tables(h, w, 32, z.dtype, z.device)
        pe = (Tx, Ty, h, w)
    z0 = self.enc0(z)
    z1 = self.enc1(run_seq(self.down_conv0, z0))
    z2 = self.enc2(run_seq(self.down_conv1, z1))
    z3 = run_seq(self.down_conv2, z2)
    for block in self.enc3s:
        z3 = block(z3, pe)
    for block in self.dec3s:
        z3 = block(z3, pe)
    z2n = self.dec2(ff_up(self.concat_conv2, z2, self.up_conv2, z3))
    del z2
    z1n = self.dec1(ff_up(self.concat_conv1, z1, self.up_conv1, z2n))
    del z1
    z0n = self.dec0(ff_up(self.concat_conv0, z0, self.up_conv0, z1n))
    return z0n, z1n, z2n, z3


def _mrt_forward(self, z0, z1=None, z2=None, z3=None):
    if isinstance(
        z0, list
    ):  # list input (emptied here): the caller holds no reference, inputs freed as soon as replaced
        zs = z0
        z0, z1, z2, z3 = zs
        zs.clear()
    z0 = self.enc_attn0(z0)
    z1 = self.enc_attn1(ff_down(self.down_concat1, z1, self.down_conv0, z0))
    z2 = self.enc_attn2(ff_down(self.down_concat2, z2, self.down_conv1, z1))
    z3 = ff_down(self.down_concat3, z3, self.down_conv2, z2)
    for block in self.enc_attn3s:
        z3 = block(z3)
    for block in self.dec_attn3s:
        z3 = block(z3)
    z2 = self.dec_attn2(ff_up(self.up_concat2, z2, self.up_conv2, z3))
    z1 = self.dec_attn1(ff_up(self.up_concat1, z1, self.up_conv1, z2))
    z0 = self.dec_attn0(ff_up(self.up_concat0, z0, self.up_conv0, z1))
    return z0, z1, z2, z3


def _stacked_forward(self, z0, z1=None, z2=None, z3=None):
    if isinstance(
        z0, list
    ):  # list input (emptied): no reference to the pyramid kept while the MRTs run
        zs = list(z0)
        z0.clear()
        for u in self.uformer_list:
            zs = list(u(zs))
        return zs[0]
    for u in self.uformer_list:
        z0, z1, z2, z3 = u(z0, z1, z2, z3)
    return z0


# ------------------------------------------------------------------------------------------------ backbone
def _group_norm_(x, gn):
    """GroupNorm in place on a channels-last tensor (2 passes: stats, x*s + t)."""
    N, C, H, W = x.shape
    G = gn.num_groups
    if (
        _OPT["kern"]
        and x.dtype == torch.half
        and x.is_contiguous(memory_format=CL)
        and (C // G) % 8 == 0
        and G <= 32
        and C // 8 <= 256
    ):
        k = kern()
        CV = C // 8
        # deterministic: per-block fp64 partials (fixed-order block reduction, no atomics), summed here in fixed order
        ppb = max(1, 256 // CV)
        nb = min((H * W + ppb - 1) // ppb, 1024)
        part = torch.empty((N, nb, G, 2), device=x.device, dtype=torch.float64)
        k.launch("gn_stats", ((nb, N, 1), (ppb * CV, 1, 1)), x, part, 0, H * W, C, G)
        acc = part.sum(dim=1)
        cnt = float(H * W * (C // G))
        mean = acc[..., 0] / cnt
        k.launch(
            "gn_stats",
            ((nb, N, 1), (ppb * CV, 1, 1)),
            x,
            part,
            mean.float().contiguous(),
            H * W,
            C,
            G,
        )
        var = part[..., 1].sum(dim=1) / cnt
        rstd = torch.rsqrt(var.float() + gn.eps)
        s = rstd.repeat_interleave(C // G, dim=1) * gn.weight.float()
        k.launch(
            "affine_nc8",
            k.linear(N * H * W * CV),
            x,
            s.contiguous(),
            gn.bias.float().contiguous(),
            mean.float().contiguous(),
            H * W,
            C,
            N,
            G,
        )
        return x
    xf = x.float().reshape(N, G, -1)
    variance, mean = torch.var_mean(xf, dim=-1, correction=0, keepdim=True)
    y = ((xf - mean) * torch.rsqrt(variance + gn.eps)).reshape_as(x)
    x.copy_(y * gn.weight.float().view(1, C, 1, 1) + gn.bias.float().view(1, C, 1, 1))
    return x


STEM_HALO = 8  # full-res rows: conv1_down 5x5/2 (2) + 3x3 at 1/2 (2), kept even


def _encoder_forward(self, x):
    """conv0 + conv1_down (full-res part, local) in row strips, GroupNorm on the whole 1/2-res map (2 passes),
    conv2 + residual in strips."""

    def stem(t):
        return run_seq(self.conv1_down, run_seq(self.conv0, t))

    H, W = x.shape[-2:]
    rows = max(
        64, (STRIP_BYTES // (self.norm1.num_channels * W)) // 2 * 2
    )  # C-ch half-res output per full-res row
    if H <= rows + 2 * STEM_HALO:
        x2 = stem(x)
    else:
        x2 = None
        for r0 in range(0, H, rows):
            r1 = min(H, r0 + rows)
            a, b = max(0, r0 - STEM_HALO), min(H, r1 + STEM_HALO)
            y = stem(x[:, :, a:b])
            if x2 is None:
                x2 = torch.empty(
                    (y.shape[0], y.shape[1], (H + 1) // 2, y.shape[3]),
                    device=y.device,
                    dtype=y.dtype,
                    memory_format=CL,
                )
            t = (r0 - a) // 2
            x2[:, :, r0 // 2 : (r1 + 1) // 2] = y[:, :, t : t + (r1 + 1) // 2 - r0 // 2]
            del y
    x2 = _group_norm_(x2, self.norm1)

    def blk(t):
        t = _cl(t)
        return run_seq(self.conv2, t, r=t)

    x2 = strips_inplace(blk, x2, 2) if _INPLACE_BB else strips(blk, [x2], 2)
    return self.conv2_down(x2), x2


# ------------------------------------------------------------------------------------------------ disparity init
def _dispinit_forward(self, feature):
    """optimal transport per image row (rows are independent problems) in row chunks, fp32 Sinkhorn on the fp16
    correlation (as upstream under autocast); soft-argmax window and disparity in fp32.
    Returns disparity, conf, occ [1,1,h,w] fp32 and the raw correlation [h, w, w] fp16."""
    assert feature.shape[0] == 2, "batch size 1 only"
    if _OPT["kern"] and _use_otf(*feature.shape[2:], feature.shape[1]):
        return _dispinit_otf(self, feature)
    f = F.layer_norm(
        feature.permute(0, 2, 3, 1).contiguous(),
        feature.shape[1:2],
        self.layer_norm.weight,
        self.layer_norm.bias,
        self.layer_norm.eps,
    )
    h, w = f.shape[1:3]
    cv = torch.bmm(f[0], f[1].transpose(1, 2))  # [h, w(left i), w(right j)]
    del f
    dev = cv.device
    if _OPT["kern"] and cv.is_contiguous():
        return _ot_kernels(self, cv, h, w)
    mask = (
        torch.triu(torch.ones((w, w), dtype=torch.bool, device=dev), diagonal=1)
        if self.use_positivity
        else None
    )
    marg = torch.cat([torch.ones([w], device=dev), torch.full([1], float(w), device=dev)]) / (2 * w)
    lm = marg.log()
    log_const = math.log(2 * w)
    disp = torch.empty((h, w), device=dev)
    conf = torch.empty((h, w), device=dev)
    occ = torch.empty((h, w), device=dev)
    xg = torch.arange(w, device=dev, dtype=torch.float32)
    rows = max(1, OT_BYTES // ((w + 1) ** 2 * 4 * 4))
    pl = 2
    for r0 in range(0, h, rows):
        r1 = min(h, r0 + rows)
        a = cv[r0:r1].float()
        if mask is not None:
            a.masked_fill_(mask, -1e4)
        a = F.pad(a, (0, 1, 0, 1))  # dustbins
        v = lm - torch.logsumexp(a, dim=1)
        u = lm - torch.logsumexp(a + v[:, None, :], dim=2)
        for _ in range(self.ot_iter - 1):
            v = lm - torch.logsumexp(a + u[:, :, None], dim=1)
            u = lm - torch.logsumexp(a + v[:, None, :], dim=2)
        a.add_(u[:, :, None]).add_(v[:, None, :])
        p = a[:, :-1, :-1].add_(log_const).exp_()  # [r, w, w] view
        if mask is not None:
            p.masked_fill_(mask, 0)
        am = p.argmax(dim=2, keepdim=True)
        c = torch.zeros((r1 - r0, w, 1), device=dev, dtype=p.dtype)
        cl = torch.zeros((r1 - r0, w, 1), device=dev, dtype=p.dtype)
        for k in range(2 * pl + 1):
            idx = am + (k - pl)
            ok = (idx >= 0) & (idx < w)
            wt = torch.gather(p, 2, idx.clamp(0, w - 1)).mul_(ok)
            c += wt
            cl += wt * idx
        eps = 1e-4
        disp[r0:r1] = (xg.to(p.dtype) - ((cl + eps) / (c + eps)).squeeze(2)).float()
        conf[r0:r1] = c.squeeze(2)
        occ[r0:r1] = p.sum(dim=2)
        del a, p
    return disp[None, None], conf[None, None], occ[None, None], cv


def _use_otf(h, w, C):
    if _CV_OTF == "1":
        return C in (64, 128, 192, 256, 384, 512)
    if _CV_OTF == "0":
        return False
    return C in (64, 128, 192, 256, 384, 512) and h * w * w * 2 > CV_OTF_BYTES


def _dispinit_otf(self, feature):
    """on-the-fly variant: the LayerNormed features and their correlation are materialized only CV_CHUNK_BYTES worth of
    rows at a time for the optimal transport (rows are independent; same LayerNorm, cuBLAS GEMM and OT kernels per
    chunk), and the refinement lookups recompute their few correlations from the raw transformer output (kept anyway
    for the refiners) + per-pixel LayerNorm statistics (OtfCostVolume)."""
    _, C, h, w = feature.shape
    x = feature.permute(0, 2, 3, 1)
    if not x.is_contiguous():
        x = x.contiguous()
    ln = self.layer_norm
    rows = max(1, min(h, CV_CHUNK_BYTES // (w * w * 2)))
    buf = torch.empty((rows, w, w), device=x.device, dtype=x.dtype)
    disp, conf, occ = (torch.empty((h, w), device=x.device) for _ in range(3))
    ms = torch.empty((2, h, w, 2), device=x.device)
    for r0 in range(0, h, rows):
        r1 = min(h, r0 + rows)
        xc = x[:, r0:r1]
        f = F.layer_norm(xc, (C,), ln.weight, ln.bias, ln.eps)
        var, mean = torch.var_mean(xc.float(), dim=-1, unbiased=False)
        ms[:, r0:r1, :, 0] = mean
        ms[:, r0:r1, :, 1] = torch.rsqrt(var + ln.eps)
        del var, mean
        cv = torch.bmm(f[0], f[1].transpose(1, 2), out=buf[: r1 - r0])
        del f
        _ot_kernels(self, cv, r1 - r0, w, (disp[r0:r1], conf[r0:r1], occ[r0:r1]))
    del buf
    return (
        disp[None, None],
        conf[None, None],
        occ[None, None],
        OtfCostVolume(x, ms, ln.weight.float().contiguous(), ln.bias.float().contiguous()),
    )


def _ot_kernels(self, cv, h, w, outs=None):
    """Sinkhorn + soft-argmax as 2 + 2 (ot_iter - 1) + 1 sync-free kernels on the fp16 correlation (s2m2_kernels.cu):
    same fp32 math as the chunked path, no [rows, w+1, w+1] fp32 tensors (~6 passes over 4x the data)."""
    dev = cv.device
    marg = torch.cat([torch.ones([1]), torch.full([1], float(w))]) / (
        2 * w
    )  # fp32, as the torch path
    lm_a, lm_d = (float(x) for x in marg.log())
    pos = 1 if self.use_positivity else 0
    u = torch.empty((h, w + 1), device=dev)
    v = torch.empty((h, w + 1), device=dev)
    k = kern()
    col = ((w + 1 + 31) // 32, h, 1), (32, 8, 1)
    nrow = h * (w + 1)
    row = ((nrow * 32 + 255) // 256, 1, 1), (256, 1, 1)
    k.launch("ot_col_lse", col, cv, 0, v, w, pos, lm_a, lm_d)
    k.launch("ot_row_lse", row, cv, v, u, h, w, pos, lm_a, lm_d)
    for _ in range(self.ot_iter - 1):
        k.launch("ot_col_lse", col, cv, u, v, w, pos, lm_a, lm_d)
        k.launch("ot_row_lse", row, cv, v, u, h, w, pos, lm_a, lm_d)
    disp, conf, occ = (
        outs if outs is not None else (torch.empty((h, w), device=dev) for _ in range(3))
    )
    k.launch(
        "ot_final",
        ((((h * w) * 32 + 255) // 256, 1, 1), (256, 1, 1)),
        cv,
        u,
        v,
        h,
        w,
        pos,
        float(math.log(2 * w)),
        disp,
        conf,
        occ,
    )
    return disp[None, None], conf[None, None], occ[None, None], cv


class OtfCostVolume:
    """FastCostVolume lookups (radius 4) without the [h, w, w] volume (cv_lookup_otf kernel): correlations recomputed
    from the raw features x [2, h, w, C] fp16 with LayerNorm statistics ms [2, h, w, 2] (mean, rstd) and affine g, b.
    Same positions / fp16 correlation values / interpolation as FastCostVolume."""

    def __init__(self, x, ms, g, b):
        self.x, self.ms, self.g, self.b = x, ms, g, b
        self.h, self.w, self.C = x.shape[1:]
        W1, W2 = self.w, self.w // 2
        self.s1, self.s2 = W1 / (W1 - 1), W2 / (W2 - 1)

    def __call__(self, disp):
        h, w = self.h, self.w
        d = disp.reshape(h * w).contiguous()
        o1 = torch.empty((h * w, 9), device=d.device, dtype=torch.half)
        o2 = torch.empty((h * w, 9), device=d.device, dtype=torch.half)
        k = kern()
        nb = min((h * w + 7) // 8, 65535 * 4)
        k.launch(
            "cv_lookup_otf",
            ((nb, 1, 1), (256, 1, 1)),
            self.x,
            self.ms,
            self.g,
            self.b,
            d,
            o1,
            o2,
            h,
            w,
            self.C,
            self.s1,
            0.5 * self.s2,
            self.s2,
        )

        def f(c):
            return c.view(1, h, w, 9).permute(0, 3, 1, 2)

        return f(o1), f(o2)


class FastCostVolume:
    """local correlation lookup (radius r) at full and half disparity resolution: 1D linear interpolation along the
    right-image axis with zero padding, fp32 coordinates. Sample position x * W / (W - 1) - 0.5 (W = cost-volume width
    of the level) == the lookup of the released weights (upstream 92319f1: x / (W - 1) normalisation sampled with
    align_corners=False); upstream 8f3bd78 samples at exactly x (+0.5 px, costs S Midd-H EPE +0.13)."""

    def __init__(self, cv, radius):
        h, w, w2 = cv.shape
        self.h, self.w = h, w
        self.cv = cv.reshape(h * w, w2)
        # level 2 (avg_pool1d(2) of cv) is not stored: its samples are read as pair means of cv (saves h*w*w/2 fp16,
        # 1/3 of the cost-volume memory; the pair mean is taken in fp32 instead of upstream's fp16 avg_pool)
        self.W2 = w2 // 2
        dx = torch.arange(-radius, radius + 1, device=cv.device, dtype=torch.float32)[None]
        base = torch.arange(w, device=cv.device, dtype=torch.float32).repeat(h)[:, None]
        W1, W2 = w2, w2 // 2
        self.s1, self.s2 = W1 / (W1 - 1), W2 / (W2 - 1)
        # level 1: (base - d + dx) * s1 - 0.5; level 2: (base / 2 - d / 2 + dx) * s2 - 0.5
        self.b1 = base * self.s1 - 0.5
        self.dx1 = dx * self.s1
        self.b2 = base * (0.5 * self.s2) - 0.5
        self.dx2 = dx * self.s2

    @staticmethod
    def _lerp(cv, x):
        W = cv.shape[1]
        x0 = x.floor()
        t = x - x0
        i0 = x0.long()
        v0 = (
            torch.gather(cv, 1, i0.clamp(0, W - 1)).float().masked_fill_((i0 < 0) | (i0 > W - 1), 0)
        )
        i1 = i0 + 1
        v1 = (
            torch.gather(cv, 1, i1.clamp(0, W - 1)).float().masked_fill_((i1 < 0) | (i1 > W - 1), 0)
        )
        return v1.sub_(v0).mul_(t).add_(v0)

    @staticmethod
    def _lerp_pooled(cv, x, W2):
        """_lerp on avg_pool1d(cv, 2) (width W2) without materializing it: level-2 sample i = mean(cv[2i], cv[2i+1])."""
        x0 = x.floor()
        t = x - x0
        i0 = x0.long()

        def g(i):
            ok = (i >= 0) & (i <= W2 - 1)
            j = 2 * i.clamp(0, W2 - 1)
            return (
                (torch.gather(cv, 1, j).float() + torch.gather(cv, 1, j + 1).float())
                .mul_(0.5)
                .masked_fill_(~ok, 0)
            )

        v0 = g(i0)
        return g(i0 + 1).sub_(v0).mul_(t).add_(v0)

    def __call__(self, disp):
        h, w = self.h, self.w
        d = disp.reshape(h * w, 1)
        c1 = self._lerp(self.cv, (self.b1 - d * self.s1) + self.dx1)
        c2 = self._lerp_pooled(self.cv, (self.b2 - d * (0.5 * self.s2)) + self.dx2, self.W2)
        k = c1.shape[1]

        def f(c):
            return c.half().view(1, h, w, k).permute(0, 3, 1, 2)

        return f(c1), f(c2)


# ------------------------------------------------------------------------------------------------ refiners
def _gru_forward(self, h, x):
    for zr, q in ((self.convzr1, self.convq1), (self.convzr2, self.convq2)):
        hx = torch.cat([h, x], dim=1)
        zr_ = zr(hx).sigmoid_()
        C = h.shape[1]
        z, r = zr_[:, :C], zr_[:, C:]
        hx[:, :C] = r * h
        qv = q(hx).tanh_()
        del hx
        h = torch.addcmul(h, z, qv.sub_(h))  # (1-z) h + z q
    return h


_CUT = [None]


def cut():
    if _CUT[0] is None:
        from .cutlass_conv import CutlassConvs

        _CUT[0] = CutlassConvs.get()
    return _CUT[0]


def _gru_forward_fused(self, h, x):
    """ConvGRU with the CUTLASS fused-epilogue convs (s2m2_cutlass.cu): z|r conv writes z and r*h (into the [r*h | x]
    buffer B) from registers, q conv writes h + z (tanh(q) - h) back into the [h | x] buffer A (and, after the second
    stage, into a dense hidden buffer for the update heads). No cat, no gate passes. The buffers persist across the
    refinement iterations (self._bufs, dropped by forward_fast): h stays in A, only x is copied per call."""
    N, C, H, W = h.shape
    if x.shape[1] != C or C % 8 or not h.is_cuda:
        return _gru_forward(self, h, x)
    bufs = getattr(self, "_bufs", None)
    if bufs is None or bufs[0].shape != (N, 2 * C, H, W):
        A = torch.empty((N, 2 * C, H, W), device=h.device, dtype=h.dtype, memory_format=CL)
        bufs = self._bufs = (A, torch.empty_like(h, memory_format=CL))
    A, Hd = bufs  # persistent; B and Z are per call (not held during the next refine_unet)
    B = torch.empty_like(A)
    Z = torch.empty_like(h, memory_format=CL)
    if (
        h.data_ptr() != Hd.data_ptr()
    ):  # not our own output of the previous iteration (whose h is already in A)
        A[:, :C] = h
    A[:, C:] = x
    B[:, C:] = x
    from .cutlass_conv import epi_args

    cc = cut()
    for s, (zr, q) in enumerate(((self.convzr1, self.convq1), (self.convzr2, self.convq2))):
        e1 = epi_args(1, 0, zr._b32, h=A, hs=2 * C, z=Z, zs=C, y=B, ys=2 * C, split=C)
        e2 = epi_args(
            2, 0, q._b32, h=A, hs=2 * C, z=Z, zs=C, y=A, ys=2 * C, y2=Hd if s else None, y2s=C
        )
        if cc.conv(0, A, zr.weight, list(zr.padding), 1, e1) or cc.conv(
            0, B, q.weight, list(q.padding), 1, e2
        ):
            raise RuntimeError("s2m2_cutlass: GRU conv not supported")
    return Hd


def _gru_strips(gru, h, x):
    """ConvGRU in row strips (halo 4: vertical 5x1 z|r conv, then q conv on r*h) -> new hidden; only one strip of the
    GRU buffers alive (low-memory mode). Exact: zero padding at a strip edge only reaches the dropped halo rows."""
    N, C, H, W = h.shape
    halo = 4
    rows = max(32, GRU_STRIP_BYTES // (2 * C * W * h.element_size()))
    if H <= rows + 2 * halo:
        return _gru_forward_fused(gru, h, x) if _OPT["cut"] else gru(h, x)
    out = torch.empty_like(h, memory_format=CL)
    for r0 in range(0, H, rows):
        r1 = min(H, r0 + rows)
        a, b = max(0, r0 - halo), min(H, r1 + halo)
        hs, xs = _cl(h[:, :, a:b]), _cl(x[:, :, a:b])
        y = _gru_forward_fused(gru, hs, xs) if _OPT["cut"] else gru(hs, xs)
        out[:, :, r0:r1] = y[:, :, r0 - a : r0 - a + r1 - r0]
        del y, hs, xs
    gru._bufs = None
    return out


def _global_refiner_forward(self, ctx, disp, conf):
    mask = (conf > 0.2).float()
    conf_logit = (mask * conf).logit(eps=1e-1)
    inp = torch.cat([(disp / 1e2 * mask).half(), conf_logit.half(), ctx], dim=1)
    feat = run_seq(self.init_feat, _cl(inp))
    refine = self.refine_unet(feat)[0]
    upd = self.out_feat(refine).float() * 1e2
    return mask * disp + (1 - mask) * upd


def _local_refiner_forward(self, hidden, ctx, disp, conf, occ, cv_fn):
    conf_logit = conf.logit(eps=1e-2)
    occ_logit = occ.logit(eps=1e-2)
    c1, c2 = cv_fn(disp)
    cf1 = run_seq(self.corr_feat1, c1 / 16)
    cf2 = run_seq(self.corr_feat2, c2 / 16)
    df = run_seq(self.disp_feat, _cl((disp / 1e2).half()))
    co = run_seq(self.conf_occ_feat, _cl(torch.cat([conf_logit, occ_logit], dim=1).half()))
    if _LM[0]:  # cat + conv in row strips: no full-size concat buffer
        k = max(m.kernel_size[0] for m in self.disp_corr_ctx_cat if isinstance(m, nn.Conv2d))
        x = strips(
            lambda *t: run_seq(self.disp_corr_ctx_cat, torch.cat([_cl(u) for u in t], dim=1)),
            [df, cf1, cf2, ctx, co],
            k // 2,
        )
    else:
        x = run_seq(self.disp_corr_ctx_cat, torch.cat([df, cf1, cf2, ctx, co], dim=1))
    del df, cf1, cf2, co
    refine = self.refine_unet(x)[0]
    del x
    if _LM[0]:
        hidden = _gru_strips(self.gru, hidden, refine)
    else:
        hidden = (
            _gru_forward_fused(self.gru, hidden, refine)
            if _OPT["cut"]
            else self.gru(hidden, refine)
        )
    du = run_seq(self.disp_update, hidden).float()
    cu = run_seq(self.conf_occ_update, hidden).float()
    conf = torch.sigmoid(cu[:, 0:1] + conf_logit)
    occ = torch.sigmoid(cu[:, 1:2] + occ_logit)
    return hidden, disp + du, conf, occ


# ------------------------------------------------------------------------------------------------ upsampling
def _convex_kernel(X, mask, f):
    b, c, h, w = X.shape
    if not (
        _OPT["kern"]
        and b == 1
        and X.dtype == torch.float32
        and X.is_contiguous()
        and mask.dtype == torch.half
        and mask.shape[1] == 9
        and mask.is_contiguous(memory_format=CL)
        and tuple(mask.shape[2:]) == (f * h, f * w)
    ):
        return None
    out = torch.empty((1, c, f * h, f * w), device=X.device)
    k = kern()
    k.launch("convex_up", k.linear(f * h * f * w), X, mask, out, c, h, w, f)
    return out


def _upsample4x_multi(X, mask):
    """convex 4x upsampling of X [1,c,h,w] fp32 with 9-neighbour weights mask [1,9,4h,4w] (softmax), no 9x unfold."""
    y = _convex_kernel(X, mask, 4)
    if y is not None:
        return y
    b, c, h, w = X.shape
    wts = mask.float().softmax(dim=1)
    Xp = F.pad(X, (1, 1, 1, 1), mode="replicate")
    out = torch.zeros((b, c, h, 4, w, 4), device=X.device)
    for k in range(9):
        i, j = divmod(k, 3)
        out.addcmul_(
            Xp[:, :, i : i + h, j : j + w][:, :, :, None, :, None],
            wts[:, k : k + 1].reshape(b, 1, h, 4, w, 4),
        )
    return out.view(b, c, 4 * h, 4 * w)


def _filter1x_multi(X, mask):
    y = _convex_kernel(X, mask, 1)
    if y is not None:
        return y
    b, c, H, W = X.shape
    wts = mask.float().softmax(dim=1)
    Xp = F.pad(X, (1, 1, 1, 1), mode="replicate")
    out = torch.zeros_like(X)
    for k in range(9):
        i, j = divmod(k, 3)
        out.addcmul_(Xp[:, :, i : i + H, j : j + W], wts[:, k : k + 1])
    return out


def _filter2x_multi(X, mask):
    """upstream upsample1x with output_upsample: 3x3 (replicate) neighbours of X [1,c,H,W] taken nearest-x2, weights =
    softmax over the 9 channels of bilinear-x2 (align_corners=False) mask [1,9,H,W] -> [1,c,2H,2W] fp32."""
    b, c, H, W = X.shape
    wts = F.interpolate(mask.float(), scale_factor=2, mode="bilinear", align_corners=False).softmax(
        dim=1
    )
    Xp = F.pad(X, (1, 1, 1, 1), mode="replicate")
    out = torch.zeros((b, c, H, 2, W, 2), device=X.device)
    for k in range(9):
        i, j = divmod(k, 3)
        out.addcmul_(
            Xp[:, :, i : i + H, j : j + W][:, :, :, None, :, None],
            wts[:, k : k + 1].reshape(b, 1, H, 2, W, 2),
        )
    return out.view(b, c, 2 * H, 2 * W)


def _upmask4_forward(self, fx, fy, fy_done=False):
    """fy_done: fy is conv_y(feature) already (computed early so the 1/2-res feature need not be kept)."""
    return run_seq(
        self.conv_concat, torch.cat([self.conv_x(fx), fy if fy_done else self.conv_y(fy)], dim=1)
    )


def _upmask1_forward(self, disp, rgb, ctx, ctx_done=False):
    cat = torch.cat(
        [
            run_seq(self.conv_disp, disp),
            run_seq(self.conv_rgb, rgb),
            ctx if ctx_done else self.conv_ctx(ctx),
        ],
        dim=1,
    )
    return run_seq(self.conv_concat, cat)


# ------------------------------------------------------------------------------------------------ model
def forward_fast(self, img0, img1, full_outputs=True):
    """Padded [1,3,H,W] RGB inputs, values 0..255; return fp32 (disp, occ, conf).
    With full_outputs=False, only disparity is computed and the other outputs are None.
    """
    if (
        img0.shape[0] != 1
        or img1.shape != img0.shape
        or img0.shape[1] != 3
        or img0.shape[-2] % 32
        or img0.shape[-1] % 32
    ):
        raise ValueError(
            f"forward_fast: expects two [1, 3, H, W] images with H, W divisible by 32 (image_pad(x, 32)); "
            f"got {tuple(img0.shape)} / {tuple(img1.shape)} (batch > 1: call once per pair)"
        )
    C = self.feature_channels
    _LM[0] = _LOWMEM == "1" or (
        _LOWMEM == "auto" and C * (img0.shape[-2] // 4) * (img0.shape[-1] // 4) * 2 > LOWMEM_BYTES
    )
    n0 = _cl(((img0 / 255.0 - 0.5) * 2).half())
    n1 = _cl(((img1 / 255.0 - 0.5) * 2).half())
    f4_0, f2_0 = self.cnn_backbone(n0)
    # the 1/2-res left feature is only used by two convs of the upsampling heads (conv_y 3x3 -> 64 ch, conv_ctx 2x2/2 ->
    # 16 ch at full res): for C > 128 apply them now and drop the C-channel map (XL 12 MP: 2.4 GB held -> 0.8 GB). Exact.
    early = f2_0.shape[1] > 128
    if early:
        f2_0 = (
            _cl(self.upsample_mask_4x_refine.conv_y(f2_0)),
            _cl(self.upsample_mask_1x.conv_ctx(f2_0)),
        )
    if (
        early and _LM[0]
    ):  # low-memory: park the two upsampling-head maps in pinned host memory until the end
        f2_0 = tuple(
            torch.empty(t.permute(0, 2, 3, 1).shape, dtype=t.dtype, pin_memory=True).copy_(
                t.permute(0, 2, 3, 1), non_blocking=True
            )
            for t in f2_0
        )
    f4_1, _ = self.cnn_backbone(n1)
    del n1, _
    feature_4x = torch.cat([f4_0, f4_1], dim=0)
    del f4_0, f4_1
    py4, py8, py16, py32 = self.feat_pyramid(feature_4x)
    del feature_4x
    if (
        _INPLACE_ATTN
    ):  # the transformer overwrites its inputs: keep image 0 of py4 (feature fusion) as a copy
        py0 = py4[0:1].clone(memory_format=CL)
        pl = [py4, py8, py16, py32]
        del py4, py8, py16, py32
        tr4 = self.transformer(pl)
    else:
        tr4 = self.transformer(py4, py8, py16, py32)
        del py8, py16, py32
        py0 = py4[0:1]
    disp, conf, occ, cv = self.disp_init(tr4)
    tr0 = tr4[0:1]
    disp = self.global_refiner(tr0, disp, conf)
    if self.use_positivity:
        disp = disp.clamp(min=0)
    fus = self.feat_fusion_layer(tr0, py0)
    del tr4, tr0, py0
    ctx0 = run_seq(self.ctx_feat, fus)
    del fus
    hidden = torch.tanh(ctx0)
    w = ctx0.shape[-1]
    coords = torch.arange(w, device=disp.device, dtype=torch.float32).view(1, 1, 1, w)
    if os.environ.get("S2M2_CV") == "grid" and not isinstance(
        cv, OtfCostVolume
    ):  # ablation: upstream(-E) lookup via grid_sample (fp32 grid -> fp32 cv)
        from .submodules import CostVolume

        cvu = CostVolume(
            cv[None].float(), coords.reshape(1, 1, w, 1).repeat(1, cv.shape[0], 1, 1), radius=4
        )

        def cv_fn(d):
            return tuple(_cl(c.half()) for c in cvu(d))
    elif isinstance(cv, OtfCostVolume):
        cv_fn = cv
    else:
        cv_fn = FastCostVolume(cv, radius=4)
    del cv
    for _ in range(self.refine_iter):
        hidden, disp, conf, occ = self.refiner(hidden, ctx0, disp, conf, occ, cv_fn)
        if self.use_positivity:
            disp = disp.clamp(min=0)
        occ = occ * (coords - disp >= 0)
    del cv_fn
    self.refiner.gru._bufs = None
    if early and f2_0[0].device.type == "cpu":
        f2_0 = tuple(t.to(n0.device, non_blocking=True).permute(0, 3, 1, 2) for t in f2_0)
    mask4 = (
        self.upsample_mask_4x_refine(hidden, f2_0[0], True)
        if early
        else self.upsample_mask_4x_refine(hidden, f2_0)
    )
    X = torch.cat([disp * 4, occ, conf], 1) if full_outputs else disp * 4
    up = _upsample4x_multi(X, mask4)
    del mask4
    fw = (
        self.upsample_mask_1x(_cl(up[:, 0:1].half()), n0, f2_0[1], True)
        if early
        else self.upsample_mask_1x(_cl(up[:, 0:1].half()), n0, f2_0)
    )
    del f2_0
    if (
        self.output_upsample
    ):  # upstream upsample1x(output_upsample=True): 2x output, disparity in output pixels
        up = _filter2x_multi(up, fw)
        up[:, 0:1] *= 2
    else:
        up = _filter1x_multi(up, fw)
    if full_outputs:
        return up[:, 0:1], up[:, 1:2], up[:, 2:3]
    return up, None, None


class _PSConv(nn.Module):
    """ConvTranspose2d with kernel == stride, no padding, as a 1x1 conv to (ky, kx, Cout) channels + one NHWC pixel
    shuffle copy (exact: every output pixel gets exactly one tap). cuDNN's transposed conv here: ~10x slower."""

    def __init__(self, t):
        super().__init__()
        ci, co, kh, kw = t.weight.shape
        self.k, self.co = (kh, kw), co
        self.conv = nn.Conv2d(ci, kh * kw * co, 1).to(t.weight)
        self.conv.weight.data = (
            t.weight.data.permute(2, 3, 1, 0).reshape(kh * kw * co, ci, 1, 1).contiguous()
        )
        b = (
            t.bias.data
            if t.bias is not None
            else torch.zeros(co, dtype=t.weight.dtype, device=t.weight.device)
        )
        self.conv.bias.data = b.repeat(kh * kw)

    def forward(self, x):
        y = self.conv(_cl(x))  # channels-last [N, kh*kw*co, H, W]
        N, _, H, W = y.shape
        kh, kw = self.k
        t = y.permute(0, 2, 3, 1).view(N, H, W, kh, kw, self.co).permute(0, 1, 3, 2, 4, 5)
        return t.reshape(N, H * kh, W * kw, self.co).permute(0, 3, 1, 2)  # channels-last view


def _tconv_as_conv(t):
    """stride-1 ConvTranspose2d == Conv2d with the flipped, transposed kernel and padding k - 1 - p (exact)."""
    ci, co, kh, kw = t.weight.shape
    c = nn.Conv2d(
        ci,
        co,
        (kh, kw),
        padding=(kh - 1 - t.padding[0], kw - 1 - t.padding[1]),
        bias=t.bias is not None,
    ).to(t.weight)
    c.weight.data = t.weight.data.flip(2, 3).transpose(0, 1).contiguous()
    if t.bias is not None:
        c.bias.data = t.bias.data.clone()
    return c


def _replace_tconvs(model):
    for parent in list(model.modules()):
        for name, t in list(parent.named_children()):
            if (
                type(t) is not nn.ConvTranspose2d
                or t.groups != 1
                or any(d != 1 for d in t.dilation)
                or any(t.output_padding)
            ):
                continue
            if tuple(t.stride) == tuple(t.kernel_size) and not any(t.padding):
                setattr(parent, name, _PSConv(t))
            elif all(s == 1 for s in t.stride) and all(
                p <= k - 1 for p, k in zip(t.padding, t.kernel_size)
            ):
                setattr(parent, name, _tconv_as_conv(t))


def _pool2x_forward(self, x):
    N, C, H, W = x.shape
    if not (
        x.dtype == torch.half
        and C % 8 == 0
        and x.is_contiguous(memory_format=CL)
        and H >= 2
        and W >= 2
    ):
        return _cl(F.avg_pool2d(x, 2))
    y = torch.empty((N, C, H // 2, W // 2), device=x.device, dtype=x.dtype, memory_format=CL)
    k = kern()
    k.launch("pool2x_nhwc8", k.linear(N * (H // 2) * (W // 2) * (C // 8)), x, y, N, H, W, C)
    return y


def _bind(m, fn):
    m.forward = types.MethodType(fn, m)


def prepare_inference(model, hacc=True, kernels=True, cutlass=None):
    """In place: fp16 + channels_last weights, fused projections, fast forwards. Returns #convs on fp16-acc cuDNN.
    kernels: conv / linear bias + activation (+ residual) as one custom epilogue kernel (s2m2_kernels.cu, compiled once)."""
    if cutlass is None:
        cutlass = hacc and os.environ.get("S2M2_CUTLASS", "1") == "1"
    _OPT.update(kern=kernels, hacc=hacc, cut=cutlass)
    model.half().to(memory_format=CL)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            m._b32 = None if m.bias is None else m.bias.float().contiguous()
        if isinstance(m, (SelfAttn, CrossAttn)):
            m._wqkv = torch.cat([m.q.weight, m.k.weight, m.v.weight], 0).contiguous()
            # attn @ (v + b_v) = attn @ v + b_v  ->  proj(o + b_v) = proj(o) + W_proj b_v: no bias pass on q|k|v
            m._pbias = (m.proj.weight.float() @ m.v.bias.float()).contiguous()
        elif isinstance(m, ConvGRU):
            for s in ("1", "2"):
                cz, cr = getattr(m, "convz" + s), getattr(m, "convr" + s)
                c = nn.Conv2d(
                    cz.in_channels, 2 * cz.out_channels, cz.kernel_size, padding=cz.padding
                ).to(cz.weight)
                c.weight.data = _cl(torch.cat([cz.weight.data, cr.weight.data], 0))
                c.bias.data = torch.cat([cz.bias.data, cr.bias.data], 0)
                setattr(m, "convzr" + s, c)
            _bind(m, _gru_forward)
        elif isinstance(m, FeatureFusion):
            _bind(m, _ff_forward)
        elif isinstance(m, ConvBlock2D):
            _bind(m, _convblock_forward)
        elif isinstance(m, BasicAttnBlock):
            _bind(m, _basic_attn_forward)
        elif isinstance(m, GlobalAttnBlock):
            _bind(m, _global_attn_forward)
        elif isinstance(m, Unet):
            _bind(m, _unet_forward)
        elif isinstance(m, MRT):
            _bind(m, _mrt_forward)
        elif isinstance(m, StackedMRT):
            _bind(m, _stacked_forward)
        elif isinstance(m, CNNEncoder):
            _bind(m, _encoder_forward)
        elif isinstance(m, DispInit):
            _bind(m, _dispinit_forward)
        elif isinstance(m, GlobalRefiner):
            _bind(m, _global_refiner_forward)
        elif isinstance(m, LocalRefiner):
            _bind(m, _local_refiner_forward)
        elif isinstance(m, UpsampleMask4x):
            _bind(m, _upmask4_forward)
        elif isinstance(m, UpsampleMask1x):
            _bind(m, _upmask1_forward)
        elif (
            type(m) is nn.AvgPool2d
            and kernels
            and m.kernel_size in (2, (2, 2))
            and m.stride in (2, (2, 2), None)
            and m.padding in (0, (0, 0))
            and not m.ceil_mode
        ):
            _bind(m, _pool2x_forward)
    for m in list(model.modules()):
        if isinstance(m, ConvGRU):
            for s in ("1", "2"):
                delattr(m, "convz" + s)
                delattr(m, "convr" + s)
    model.forward_fast = types.MethodType(forward_fast, model)
    if os.environ.get("S2M2_TCONV", "1") == "1":
        _replace_tconvs(model)
    n = 0
    for m in model.modules():
        if (
            type(m) is nn.Conv2d
            and m.groups == 1
            and all(d == 1 for d in m.dilation)
            and m.padding_mode == "zeros"
            and not isinstance(m.padding, str)
        ):
            m.weight.data = _cl(m.weight.data)
            m._b32 = None if m.bias is None else m.bias.float().contiguous()
            m._hacc = hacc
            K, C, R, S = m.weight.shape
            # fixed shape-class rule from bench/s2m2_cutlass_probe.py (no per-size choice): CUTLASS + fused epilogue
            # beats cuDNN + separate pass for spatial convs with >= 256 in/out channels and for 1x1 convs (8-warp tile)
            m._cut = (
                cutlass
                and _CUT_CONV
                and C % 8 == 0
                and K % 8 == 0
                and m.stride[0] == m.stride[1]
                and K >= 128
                and ((R * S > 1 and C >= 256 and K >= 256) or R * S == 1)
            )
            # 1x1 convs with a residual: CUTLASS + (bias, residual) epilogue beats cuDNN + a 3-operand pass
            m._cut_res = (
                cutlass
                and _CUT_CONV
                and C % 8 == 0
                and K % 8 == 0
                and K >= 128
                and R * S == 1
                and m.stride[0] == 1
            )
            m.forward = types.MethodType(_conv_forward, m)
            n += 1
    if kernels:
        kern()
    if cutlass:
        cut()
    return n
