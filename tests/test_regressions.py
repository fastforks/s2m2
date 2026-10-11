import copy

import pytest
import torch
from s2m2.core.model.rewrites import prepare
from s2m2.core.model.submodules import CostVolume


@pytest.mark.parametrize("width", [4, 8])
def test_cost_pool_matches_upstream(width):
    cv = torch.randn(1, 2, width, width)
    coords = torch.arange(width).reshape(1, 1, width, 1).expand(1, 2, width, 1)
    got = CostVolume(cv, coords, radius=4).cv_2x
    expected = torch.nn.functional.avg_pool2d(cv.reshape(2 * width, 1, 1, width), (1, 2))
    torch.testing.assert_close(got, expected.reshape(2, 1, width, width // 2))


@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("stride", [1, 3])
@pytest.mark.parametrize("bias", [True, False])
def test_rewrite_preserves_transposed_conv(dilation, stride, bias):
    model = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(2, 3, 3, stride=stride, dilation=dilation, bias=bias)
    ).eval()
    ref = copy.deepcopy(model)
    prepare(model)
    x = torch.randn(1, 2, 5, 7)
    torch.testing.assert_close(model(x), ref(x))


@pytest.mark.parametrize("stride,padding", [(1, 0), (2, 0), (1, 1)])
def test_upsample_commutation_requires_pointwise_same_grid(stride, padding):
    model = torch.nn.Sequential(
        torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            torch.nn.Conv2d(2, 3, 1, stride=stride, padding=padding),
        )
    ).eval()
    reference = copy.deepcopy(model)
    prepare(model)
    x = torch.randn(1, 2, 5, 7)
    torch.testing.assert_close(model(x), reference(x))


def test_transposed_conv_crop_is_preserved():
    model = torch.nn.Sequential(torch.nn.ConvTranspose2d(2, 3, 3, padding=3)).eval()
    reference = copy.deepcopy(model)
    prepare(model)
    x = torch.randn(1, 2, 7, 9)
    torch.testing.assert_close(model(x), reference(x))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("offset", [1000.0, 60000.0])
def test_custom_group_norm_large_mean(offset, monkeypatch):
    from s2m2.core.model import inference

    monkeypatch.setitem(inference._OPT, "kern", True)
    x = (
        (offset + 0.1 * torch.randn(1, 64, 32, 32, device="cuda"))
        .half()
        .contiguous(memory_format=torch.channels_last)
    )
    norm = torch.nn.GroupNorm(8, 64).cuda().half()
    expected = (
        torch.nn.functional.group_norm(
            x.cpu().double(), 8, norm.weight.cpu().double(), norm.bias.cpu().double(), norm.eps
        )
        .cuda()
        .half()
    )
    with torch.no_grad():
        got = inference._group_norm_(x.clone(), norm)
    torch.testing.assert_close(got, expected, atol=0.008, rtol=0.002)
