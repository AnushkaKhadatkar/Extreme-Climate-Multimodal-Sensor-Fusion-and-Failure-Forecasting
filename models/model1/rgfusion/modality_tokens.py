"""Camera tokens on the lidar BEV grid.

For every BEV cell, K reference points (the cell centre at K heights) are mapped
back to the original lidar frame (aug_tracking.undo_world_aug), projected into
the image with the per-frame calibration (trans_lidar_to_cam = R0_rect @
Tr_velo_to_cam, trans_cam_to_img = P2), and the image feature map is sampled
there. The camera token is the mean over the samples that land inside the image
in front of the camera. Cells with no valid sample get a learned "no-view" token
and are flagged invalid, so the fusion can ignore the camera there.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aug_tracking import undo_world_aug


def bev_cell_centers(pc_range, grid_hw, device) -> torch.Tensor:
    """(H*W, 2) xy centres, row-major over (y, x), matching OpenPCDet's (B, C, ny, nx) BEV maps."""
    H, W = grid_hw
    x0, y0, _, x1, y1, _ = pc_range
    xs = x0 + (torch.arange(W, device=device, dtype=torch.float32) + 0.5) * (x1 - x0) / W
    ys = y0 + (torch.arange(H, device=device, dtype=torch.float32) + 0.5) * (y1 - y0) / H
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)


def project_to_image(xyz_lidar: torch.Tensor, lidar_to_cam: torch.Tensor, cam_to_img: torch.Tensor):
    """xyz_lidar (B, N, 3); lidar_to_cam (B, 4, 4); cam_to_img (B, 3, 4).
    Returns uv (B, N, 2) in full-resolution pixels and depth (B, N)."""
    ones = torch.ones_like(xyz_lidar[..., :1])
    cam = torch.cat([xyz_lidar, ones], dim=-1) @ lidar_to_cam.transpose(1, 2)   # (B, N, 4)
    img = cam @ cam_to_img.transpose(1, 2)                                      # (B, N, 3)
    depth = img[..., 2]
    uv = img[..., :2] / depth.clamp(min=1e-3).unsqueeze(-1)
    return uv, cam[..., 2]


class CameraToBEV(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pc_range, grid_hw, sample_heights):
        super().__init__()
        self.pc_range = list(pc_range)
        self.grid_hw = tuple(grid_hw)
        self.register_buffer("heights", torch.tensor(sample_heights, dtype=torch.float32), persistent=False)
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )
        self.no_view = nn.Parameter(torch.zeros(out_channels))

    def forward(self, img_feats: torch.Tensor, batch_dict: dict, image_hw):
        """img_feats (B, C, h, w) from the image as loaded (full image_hw = (H_img, W_img)).
        Returns cam_bev (B, D, H, W) and valid (B, H, W) bool."""
        B = img_feats.shape[0]
        H, W = self.grid_hw
        K = self.heights.numel()
        centers = bev_cell_centers(self.pc_range, self.grid_hw, img_feats.device)          # (HW, 2)
        pts = torch.cat([
            centers.unsqueeze(1).expand(-1, K, -1),
            self.heights.view(1, K, 1).expand(H * W, -1, -1),
        ], dim=-1).reshape(1, H * W * K, 3).expand(B, -1, -1)                               # (B, HWK, 3)

        pts = undo_world_aug(pts, batch_dict)
        uv, depth = project_to_image(pts, batch_dict["trans_lidar_to_cam"].float(),
                                     batch_dict["trans_cam_to_img"].float())
        h_img, w_img = image_hw
        valid = (depth > 0.5) & (uv[..., 0] >= 0) & (uv[..., 0] < w_img) & (uv[..., 1] >= 0) & (uv[..., 1] < h_img)

        # The feature map covers the whole (resized) image, so normalise by the full image size.
        grid = torch.stack([uv[..., 0] / w_img * 2 - 1, uv[..., 1] / h_img * 2 - 1], dim=-1)
        grid = grid.view(B, H * W * K, 1, 2).to(img_feats.dtype)
        sampled = F.grid_sample(img_feats, grid, mode="bilinear", align_corners=False)      # (B, C, HWK, 1)
        sampled = sampled.view(B, -1, H * W, K)
        v = valid.view(B, 1, H * W, K).to(sampled.dtype)
        n_valid = v.sum(dim=-1)                                                              # (B, 1, HW)
        mean = (sampled * v).sum(dim=-1) / n_valid.clamp(min=1)                              # (B, C, HW)

        tokens = self.proj(mean.view(B, -1, H, W))                                           # (B, D, H, W)
        cell_valid = (n_valid.view(B, H, W) > 0)
        tokens = torch.where(cell_valid.unsqueeze(1), tokens,
                             self.no_view.view(1, -1, 1, 1).to(tokens.dtype).expand_as(tokens))
        return tokens, cell_valid
