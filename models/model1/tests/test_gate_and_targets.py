"""Gated attention, aux targets and the reliability target."""

import torch

from rgfusion.aux_heads import ReliabilityTarget, car_heatmap, focal_loss_per_sample
from rgfusion.gated_fusion import GatedModalityAttention, stream_bias

torch.manual_seed(0)
PC_RANGE = [0, -40, -3, 70.4, 40, 1]


def _fusion():
    f = GatedModalityAttention(dim=32, num_heads=4, num_layers=2).eval()
    tokens = torch.randn(2, 50, 2, 32)
    pos = torch.rand(50, 2)
    return f, tokens, pos


def test_weights_sum_to_one():
    f, tokens, pos = _fusion()
    _, w = f(tokens, stream_bias(torch.full((2, 2), 0.9)), pos)
    torch.testing.assert_close(w.sum(-1), torch.ones(2, 50))


def test_low_reliability_camera_is_ignored():
    f, tokens, pos = _fusion()
    r = torch.tensor([[0.95, 1e-6], [0.95, 1e-6]])
    out_a, w = f(tokens, stream_bias(r), pos)
    other = tokens.clone()
    other[:, :, 1] = torch.randn_like(other[:, :, 1]) * 10        # change only the camera tokens
    out_b, _ = f(other, stream_bias(r), pos)
    assert w[..., 1].max() < 1e-3
    torch.testing.assert_close(out_a, out_b, atol=1e-2, rtol=1e-3)


def test_invalid_camera_cells_are_ignored():
    f, tokens, pos = _fusion()
    valid = torch.ones(2, 50, 2, dtype=torch.bool)
    valid[:, :25, 1] = False
    _, w = f(tokens, stream_bias(torch.full((2, 2), 0.9), valid), pos)
    assert w[:, :25, 1].max() < 1e-3
    assert w[:, 25:, 1].min() > 1e-3


def test_heatmap_peaks_at_car_centres_only():
    boxes = torch.zeros(1, 3, 8)
    boxes[0, 0] = torch.tensor([20.2, 0.2, -1, 3.9, 1.6, 1.5, 0, 1])   # Car
    boxes[0, 1] = torch.tensor([30.2, 5.2, -1, 0.8, 0.6, 1.7, 0, 2])   # Pedestrian: ignored
    # row 2 is padding (class 0)
    hm = car_heatmap(boxes, PC_RANGE, (200, 176))
    assert hm.shape == (1, 1, 200, 176)
    assert (hm == 1).sum() == 1
    assert hm[0, 0, 100, 50] == 1                                     # y=0.2 -> row 100, x=20.2 -> col 50


def test_focal_loss_lower_for_correct_prediction():
    target = torch.zeros(1, 1, 20, 20)
    target[0, 0, 10, 10] = 1
    good = torch.full_like(target, -6.0)
    good[0, 0, 10, 10] = 6.0
    bad = torch.full_like(target, -6.0)
    bad[0, 0, 3, 3] = 6.0
    assert focal_loss_per_sample(good, target) < focal_loss_per_sample(bad, target)


def test_reliability_target():
    t = ReliabilityTarget(2, momentum=1.0).train()
    keep = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    t(torch.tensor([[1.0, 2.0], [1.0, 2.0]]), keep)                  # tau = (1, 2)
    t.eval()
    r = t(torch.tensor([[1.0, 2.0], [2.0, 4.0], [0.5, 1.0]]), torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 0.0]]))
    torch.testing.assert_close(r[0], torch.tensor([1.0, 1.0]))      # typical -> 1
    torch.testing.assert_close(r[1], torch.exp(torch.tensor([-1.0, -1.0])))  # twice as bad -> e^-1
    assert r[2, 0] == 1.0 and r[2, 1] == 0.0                        # dropped stream -> 0
