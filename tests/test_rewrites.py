import copy

import pytest
import torch
import torch.nn as nn
from s2m2.core.model.rewrites import prepare


@pytest.mark.parametrize("kernel,stride,padding", [(2, 2, 0), (4, 4, 0), (3, 1, 1)])
def test_transposed_conv_rewrite_is_exact(kernel, stride, padding):
    model = nn.Sequential(nn.ConvTranspose2d(3, 5, kernel, stride=stride, padding=padding)).eval()
    ref = copy.deepcopy(model)
    assert prepare(model)["tconv"] == 1
    x = torch.randn(2, 3, 6, 7)
    torch.testing.assert_close(model(x), ref(x))


def test_conv_moves_before_bilinear_upsampling():
    model = nn.Sequential(nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), nn.Conv2d(4, 3, 1))).eval()
    ref = copy.deepcopy(model)
    assert prepare(model)["upconv"] == 1
    assert isinstance(model[0][0], nn.Conv2d)
    x = torch.randn(1, 4, 5, 6)
    torch.testing.assert_close(model(x), ref(x))
