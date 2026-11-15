"""The gate's reliability floor (FUSION.GATE_R_FLOOR): default behaviour unchanged, larger floor only down-weights."""

import math

import torch

from rgfusion.gated_fusion import GatedModalityAttention, stream_bias


def test_default_floor_is_unchanged():
    r = torch.tensor([[1e-9, 0.5, 1.0]])
    assert torch.allclose(stream_bias(r), stream_bias(r, r_floor=1e-4))
    assert abs(float(stream_bias(r)[0, 0, 0]) - math.log(1e-4)) < 1e-5          # r near 0 is clamped to 1e-4
    assert abs(float(stream_bias(r)[0, 0, 1]) - math.log(0.5)) < 1e-6


def test_larger_floor_raises_only_the_small_values():
    r = torch.tensor([[0.0, 0.1, 0.8]])
    b = stream_bias(r, r_floor=0.3)[0, 0]
    assert abs(float(b[0]) - math.log(0.3)) < 1e-6 and abs(float(b[1]) - math.log(0.3)) < 1e-6   # both lifted to the floor
    assert abs(float(b[2]) - math.log(0.8)) < 1e-6                                              # above the floor: untouched


def test_floor_keeps_a_distrusted_stream_in_the_softmax():
    torch.manual_seed(0)
    att = GatedModalityAttention(dim=16, num_heads=2, num_layers=1)
    tokens = torch.randn(1, 5, 2, 16)
    pos = torch.rand(5, 2)

    def lidar_weight(r_lidar, floor):
        r = torch.tensor([[r_lidar, 1.0]])
        _, w = att(tokens, stream_bias(r, r_floor=floor).expand(1, 5, 2), pos)
        return float(w[..., 0].mean())

    assert lidar_weight(0.0, 1e-4) < 1e-3                   # default: the lidar is removed
    assert lidar_weight(0.0, 0.3) > 0.1                     # floor 0.3: it is still heard
    assert lidar_weight(0.0, 0.3) < lidar_weight(1.0, 0.3)  # but still attended less than a trusted lidar
