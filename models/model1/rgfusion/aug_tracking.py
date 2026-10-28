"""Undo OpenPCDet's world augmentation so BEV points can be projected into the image.

Training applies, in this order (stf_multimodal_dataset.yaml):
    random_world_flip (x axis)  ->  y := -y                     records flip_x
    random_world_rotation       ->  rotate about z by +theta     records noise_rot
    random_world_scaling        ->  xyz := s * xyz               records noise_scale
The image is not changed, so a point in the augmented frame must be mapped back
to the original lidar frame before Tr_velo_to_cam / P2 are applied.

Why not OpenPCDet's own ``lidar_aug_matrix``: DatasetTemplate.set_lidar_aug_matrix
only applies the flip when ``flip_y`` is present. With ALONG_AXIS_LIST ['x'] only
``flip_x`` exists, so that matrix silently drops the flip and every flipped frame
would project onto the wrong side of the image. This module reads the three
recorded values directly. Keys are absent at test time, which means identity.
"""

import torch


def undo_world_aug(xyz: torch.Tensor, batch_dict: dict) -> torch.Tensor:
    """xyz: (B, N, 3) in the augmented frame -> (B, N, 3) in the original lidar frame."""
    B = xyz.shape[0]
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]

    scale = batch_dict.get("noise_scale")
    if scale is not None:
        s = torch.as_tensor(scale, dtype=xyz.dtype, device=xyz.device).reshape(B, 1)
        x, y, z = x / s, y / s, z / s

    rot = batch_dict.get("noise_rot")
    if rot is not None:
        t = torch.as_tensor(rot, dtype=xyz.dtype, device=xyz.device).reshape(B, 1)
        c, s = torch.cos(t), torch.sin(t)
        # forward was x' = x c - y s, y' = x s + y c (common_utils.rotate_points_along_z)
        x, y = x * c + y * s, -x * s + y * c

    flip = batch_dict.get("flip_x")
    if flip is not None:
        f = torch.as_tensor(flip, device=xyz.device).reshape(B, 1).to(torch.bool)
        y = torch.where(f, -y, y)

    return torch.stack([x, y, z], dim=-1)
