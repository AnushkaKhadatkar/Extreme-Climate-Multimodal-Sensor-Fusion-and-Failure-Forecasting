"""Per-stream reliability scores r_m in (0, 1), estimated from each stream's own input.

No corruption labels exist at inference (context.md section 6): each estimator
sees only its own stream, i.e. pooled features plus a few cheap input statistics.

  lidar : log point counts in 7 range bins (0-10 ... 60-70 m), mean intensity
  camera: mean, std, gradient energy, saturated and dark pixel fractions

Inputs are detached: the estimators observe the shared features but do not
shape them, and their outputs are trained only against the reliability target
(see aux_heads.reliability_target). The fusion uses log r as a fixed bias, so a
reported r is what the estimator believes, not something the detection loss
tuned for convenience.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

RANGE_BINS = torch.tensor([0, 10, 20, 30, 40, 50, 60, 70.4])


def lidar_stats(points: torch.Tensor, batch_size: int) -> torch.Tensor:
    """points (N, 5) = [batch_idx, x, y, z, intensity(0-255)] -> (B, 8)."""
    b = points[:, 0].long()
    rng = torch.linalg.norm(points[:, 1:3], dim=1)
    edges = RANGE_BINS.to(points.device)
    bins = torch.bucketize(rng, edges[1:-1])                     # 0..6
    counts = torch.zeros(batch_size * 7, device=points.device)
    counts.index_add_(0, b * 7 + bins, torch.ones_like(rng))
    counts = torch.log1p(counts.view(batch_size, 7)) / 10.0
    inten = torch.zeros(batch_size, device=points.device).index_add_(0, b, points[:, 4])
    n = torch.zeros(batch_size, device=points.device).index_add_(0, b, torch.ones_like(rng))
    mean_int = (inten / n.clamp(min=1) / 255.0).unsqueeze(1)
    return torch.cat([counts, mean_int], dim=1)


def camera_stats(images: torch.Tensor) -> torch.Tensor:
    """images (B, 3, H, W) in [0, 1] -> (B, 5)."""
    grey = images.mean(dim=1, keepdim=True)
    gx = grey[..., :, 1:] - grey[..., :, :-1]
    gy = grey[..., 1:, :] - grey[..., :-1, :]
    grad = gx.abs().mean(dim=(1, 2, 3)) + gy.abs().mean(dim=(1, 2, 3))
    flat = grey.flatten(1)
    return torch.stack([
        flat.mean(1), flat.std(1), grad * 10.0,
        (flat > 0.98).float().mean(1), (flat < 0.02).float().mean(1),
    ], dim=1)


class ReliabilityEstimator(nn.Module):
    """Pooled features (avg + max) + input statistics -> one logit per frame."""

    def __init__(self, feat_channels: int, n_stats: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * feat_channels + n_stats, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        nn.init.constant_(self.mlp[-1].bias, 2.0)  # start near r = 0.88: trust streams by default

    def forward(self, feats: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        feats = feats.detach().float()
        pooled = torch.cat([F.adaptive_avg_pool2d(feats, 1).flatten(1),
                            F.adaptive_max_pool2d(feats, 1).flatten(1)], dim=1)
        return self.mlp(torch.cat([pooled, stats.detach().float()], dim=1)).squeeze(1)  # logit (B,)
