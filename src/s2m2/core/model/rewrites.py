"""Inference/export rewrites: commute pointwise convolutions before upsampling and
replace supported transposed convolutions with ordinary convolutions. Plain PyTorch.
"""

import torch.nn as nn


class PSConv(nn.Module):
    """ConvTranspose2d with kernel == stride, no padding == 1x1 conv to kh*kw*co channels + pixel shuffle."""

    def __init__(self, t):
        super().__init__()
        ci, co, kh, kw = t.weight.shape
        self.k, self.co = (kh, kw), co
        self.conv = nn.Conv2d(ci, kh * kw * co, 1).to(t.weight)
        self.conv.weight.data = (
            t.weight.data.permute(2, 3, 1, 0).reshape(kh * kw * co, ci, 1, 1).contiguous()
        )
        b = t.bias.data if t.bias is not None else t.weight.new_zeros(co)
        self.conv.bias.data = b.repeat(kh * kw)

    def forward(self, x):
        y = self.conv(x)
        N, _, H, W = y.shape
        kh, kw = self.k
        t = y.reshape(N, kh, kw, self.co, H, W).permute(0, 3, 4, 1, 5, 2)  # N, co, H, kh, W, kw
        return t.reshape(N, self.co, H * kh, W * kw)


def tconv_as_conv(t):
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


def prepare(model, upconv=True, tconv=True):
    """Apply supported algebraic rewrites after loading weights; return replacement counts."""
    n = dict(upconv=0, tconv=0)
    for parent in list(model.modules()):
        for name, c in list(parent.named_children()):
            if (
                upconv
                and isinstance(c, nn.Sequential)
                and len(c) == 2
                and isinstance(c[0], nn.Upsample)
                and isinstance(c[1], nn.Conv2d)
                and c[1].kernel_size == (1, 1)
                and c[1].stride == (1, 1)
                and not any(c[1].padding)
            ):
                setattr(parent, name, nn.Sequential(c[1], c[0]))
                n["upconv"] += 1
            elif (
                tconv
                and type(c) is nn.ConvTranspose2d
                and c.groups == 1
                and c.dilation == (1, 1)
                and not any(c.output_padding)
            ):
                if tuple(c.stride) == tuple(c.kernel_size) and not any(c.padding):
                    setattr(parent, name, PSConv(c))
                    n["tconv"] += 1
                elif all(s == 1 for s in c.stride) and all(
                    p < k for p, k in zip(c.padding, c.kernel_size)
                ):
                    setattr(parent, name, tconv_as_conv(c))
                    n["tconv"] += 1
    return n
