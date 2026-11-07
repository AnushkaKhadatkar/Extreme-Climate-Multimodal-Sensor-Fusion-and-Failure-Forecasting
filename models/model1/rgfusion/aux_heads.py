"""Per-stream auxiliary BEV heads and the reliability target they define.

Each stream alone predicts a BEV heatmap of Car centres from its own tokens.
Its per-sample focal loss L_m says how well that stream explains the scene on
its own. The reliability target is

    r*_m = keep_m * min(1, exp(1 - L_m / tau_m))

  keep_m : fraction of the stream that survived modality dropout (1 when clean,
           0 when zeroed or replaced by noise, in between for a partial mask)
  tau_m  : running mean of L_m over clean samples (EMA), i.e. what is typical
           for that stream. A typical or better frame gets 1; a frame twice as
           bad gets exp(-1) = 0.37.

tau is per stream, so a camera that is always worse than lidar at localising
Cars in BEV is not marked unreliable for that alone; only frames worse than
that stream's own norm are.
"""

import math

import torch
import torch.nn as nn

CAR_CLASS = 1  # gt_boxes[..., 7] is the 1-based index into CLASS_NAMES ['Car', ...]


class BEVHeatmapHead(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )
        nn.init.constant_(self.net[-1].bias, -math.log((1 - 0.01) / 0.01))  # prior p = 0.01

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def car_heatmap(gt_boxes: torch.Tensor, pc_range, grid_hw) -> torch.Tensor:
    """gt_boxes (B, G, 8) [x, y, z, dx, dy, dz, heading, class] -> (B, 1, H, W) Gaussian targets.
    Each Car's centre cell is exactly 1; padding rows (class 0) are ignored."""
    B = gt_boxes.shape[0]
    H, W = grid_hw
    cell_x = (pc_range[3] - pc_range[0]) / W
    cell_y = (pc_range[4] - pc_range[1]) / H
    target = torch.zeros(B, 1, H, W, device=gt_boxes.device)
    rows = torch.arange(H, device=gt_boxes.device).view(H, 1).float()
    cols = torch.arange(W, device=gt_boxes.device).view(1, W).float()
    for b in range(B):
        boxes = gt_boxes[b]
        boxes = boxes[boxes[:, 7].round() == CAR_CLASS]
        if len(boxes) == 0:
            continue
        c = torch.floor((boxes[:, 0] - pc_range[0]) / cell_x)
        r = torch.floor((boxes[:, 1] - pc_range[1]) / cell_y)
        inside = (c >= 0) & (c < W) & (r >= 0) & (r < H)
        c, r, boxes = c[inside], r[inside], boxes[inside]
        if len(c) == 0:
            continue
        sigma = torch.clamp(torch.minimum(boxes[:, 3], boxes[:, 4]) / cell_x / 4.0, min=1.0)
        d2 = (rows.unsqueeze(0) - r.view(-1, 1, 1)) ** 2 + (cols.unsqueeze(0) - c.view(-1, 1, 1)) ** 2
        heat = torch.exp(-d2 / (2 * sigma.view(-1, 1, 1) ** 2)).amax(dim=0)
        target[b, 0] = heat
    return target


def focal_loss_per_sample(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """CenterNet focal loss, one value per sample: logits/target (B, 1, H, W) -> (B,)."""
    p = torch.sigmoid(logits.float()).clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    neg = 1.0 - pos
    pos_loss = torch.log(p) * (1 - p) ** 2 * pos
    neg_loss = torch.log(1 - p) * p ** 2 * (1 - target) ** 4 * neg
    num_pos = pos.flatten(1).sum(1).clamp(min=1.0)
    return -(pos_loss.flatten(1).sum(1) + neg_loss.flatten(1).sum(1)) / num_pos


class ReliabilityTarget(nn.Module):
    """r*_m = keep_m * min(1, exp(1 - L_m / tau_m)), tau_m = EMA of L_m over clean samples."""

    def __init__(self, num_streams: int = 2, momentum: float = 0.01):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("tau", torch.full((num_streams,), -1.0))  # -1: not initialised yet

    @torch.no_grad()
    def forward(self, losses: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        """losses, keep: (B, M) -> r* (B, M)."""
        losses = losses.float()
        if self.training:
            clean = keep >= 0.999
            for m in range(losses.shape[1]):
                if clean[:, m].any():
                    batch_mean = losses[clean[:, m], m].mean()
                    if self.tau[m] < 0:
                        self.tau[m] = batch_mean
                    else:
                        self.tau[m] = (1 - self.momentum) * self.tau[m] + self.momentum * batch_mean
        tau = self.tau.clamp(min=1e-6).unsqueeze(0)
        quality = torch.exp(1.0 - losses / tau).clamp(max=1.0)
        quality = torch.where(self.tau.unsqueeze(0) < 0, torch.ones_like(quality), quality)
        return keep.float() * quality
