"""Undoing OpenPCDet's world augmentation, and lidar -> image projection."""

import numpy as np
import torch

from pcdet.datasets.augmentor import augmentor_utils
from rgfusion.aug_tracking import undo_world_aug
from rgfusion.modality_tokens import bev_cell_centers, project_to_image

# Synthetic KITTI-style calibration: lidar x forward, y left, z up -> camera x right, y down, z forward.
V2R = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
P2 = np.array([[1000, 0, 960, 0], [0, 1000, 512, 0], [0, 0, 1, 0]], dtype=np.float32)


def test_undo_matches_openpcdet_augmentation():
    rng = np.random.default_rng(0)
    pts = rng.uniform([0, -40, -3], [70, 40, 1], size=(500, 3)).astype(np.float32)
    boxes = np.zeros((1, 7), dtype=np.float32)
    aug = np.concatenate([pts, np.zeros((500, 1), np.float32)], axis=1)
    boxes, aug, flip = augmentor_utils.random_flip_along_x(boxes, aug, return_flip=True, enable=True)
    boxes, aug, rot = augmentor_utils.global_rotation(boxes, aug, [-0.78, 0.78], return_rot=True, noise_rotation=0.6)
    boxes, aug, scale = augmentor_utils.global_scaling(boxes, aug, [0.95, 1.05], return_scale=True)
    batch = {"flip_x": np.array([flip]), "noise_rot": np.array([rot]), "noise_scale": np.array([scale])}
    back = undo_world_aug(torch.from_numpy(aug[None, :, :3]), batch)[0].numpy()
    np.testing.assert_allclose(back, pts, atol=1e-4)


def test_undo_is_identity_without_augmentation_keys():
    xyz = torch.randn(2, 10, 3)
    assert torch.equal(undo_world_aug(xyz, {}), xyz)


def test_projection_of_known_point():
    # 20 m ahead, 2 m left, on the camera's height -> left of the principal point, same row.
    xyz = torch.tensor([[[20.0, 2.0, 0.0]]])
    uv, depth = project_to_image(xyz, torch.from_numpy(V2R)[None], torch.from_numpy(P2)[None])
    np.testing.assert_allclose(uv[0, 0].numpy(), [960 - 1000 * 2 / 20, 512], atol=1e-3)
    assert abs(depth.item() - 20.0) < 1e-4


def test_bev_cell_centers_are_row_major_over_y_then_x():
    c = bev_cell_centers([0, -40, -3, 70.4, 40, 1], (200, 176), torch.device("cpu"))
    assert c.shape == (200 * 176, 2)
    np.testing.assert_allclose(c[0].numpy(), [0.2, -39.8], atol=1e-4)      # first row, first column
    np.testing.assert_allclose(c[1].numpy(), [0.6, -39.8], atol=1e-4)      # x varies fastest
    np.testing.assert_allclose(c[176].numpy(), [0.2, -39.4], atol=1e-4)    # next row = next y
