"""Controlled sensor failures applied at EVALUATION time through the M1_EVAL_CORRUPTION hook.

    M1_EVAL_CORRUPTION=eval_corruptions:cam_dark  python openpcdet_entry.py test ...

Each function is fn(data_dict, rng) -> data_dict, called on every test sample before voxelisation
(rgfusion/register.py), seeded by the frame id so the same frame gets the same corruption on every run.
Never applied in training (context.md section 6).

This is a first, deliberately simple probe of the question "does the model cope when one or both sensors
fail", not the corruption engine of context.md section 2. In particular these are SYNTHETIC failures, not
physical simulations of icing, snow or fog, and they must be labelled synthetic in any output. Two of them
(cam_zeros, cam_occlusion, lidar_sparse, lidar_sector) are the same kind of perturbation as the training-time
modality dropout (rgfusion/modality_dropout.py), so they test failures the model has seen the shape of;
cam_dark and cam_blur were never seen in training and are closer to real camera degradation.

    camera:  cam_zeros      black image (dead camera)
             cam_occlusion  grey rectangle over 60% of the image (lens blocked)
             cam_dark       brightness x 0.1 (night / underexposure)
             cam_blur       strong Gaussian blur, sigma 25 px (smeared lens)
    lidar:   lidar_sparse   keep 5% of the points (heavy backscatter / dropout)
             lidar_sector   remove a 60% azimuth sector (blocked dome)
    both:    both_dark_sparse   cam_dark + lidar_sparse (two modalities degraded)
"""

import cv2
import numpy as np

from rgfusion.modality_dropout import _drop_camera, _drop_lidar

CONDITIONS = ["cam_zeros", "cam_occlusion", "cam_dark", "cam_blur", "lidar_sparse", "lidar_sector", "both_dark_sparse",
              "cam_haze", "cam_motion_blur", "cam_color_cast"]

# The last three are HELD-OUT camera failures: the M5p / M4bp variants (photometric dropout: dark + isotropic blur) never
# train on them, so they measure whether the camera reliability score and the model generalise beyond the failures
# they were trained on.
#   cam_haze        washed-out, low-contrast image (img * 0.4 + 0.5): a fog-like lens, SYNTHETIC
#   cam_motion_blur horizontal motion blur, 61 px kernel
#   cam_color_cast  strong orange cast (channels x [1.0, 0.6, 0.3])


def cam_zeros(data_dict, rng):
    if "images" in data_dict:
        data_dict["images"] = np.zeros_like(data_dict["images"])
    return data_dict


def cam_occlusion(data_dict, rng):
    if "images" in data_dict:
        data_dict["images"], _ = _drop_camera(data_dict["images"], "mask", {"MASK_FRACTION": [0.6, 0.6]}, rng)
    return data_dict


def cam_dark(data_dict, rng):
    if "images" in data_dict:
        data_dict["images"] = (data_dict["images"] * 0.1).astype(data_dict["images"].dtype)
    return data_dict


def cam_blur(data_dict, rng):
    if "images" in data_dict:
        img = np.ascontiguousarray(data_dict["images"], dtype=np.float32)
        data_dict["images"] = cv2.GaussianBlur(img, (0, 0), sigmaX=25.0)
    return data_dict


def cam_haze(data_dict, rng):
    if "images" in data_dict:
        data_dict["images"] = np.clip(data_dict["images"] * 0.4 + 0.5, 0.0, 1.0).astype(data_dict["images"].dtype)
    return data_dict


def cam_motion_blur(data_dict, rng):
    if "images" in data_dict:
        kernel = np.full((1, 61), 1.0 / 61, dtype=np.float32)
        img = np.ascontiguousarray(data_dict["images"], dtype=np.float32)
        data_dict["images"] = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT)
    return data_dict


def cam_color_cast(data_dict, rng):
    if "images" in data_dict:
        gain = np.array([1.0, 0.6, 0.3], dtype=np.float32)
        data_dict["images"] = np.clip(data_dict["images"] * gain, 0.0, 1.0).astype(data_dict["images"].dtype)
    return data_dict


def lidar_sparse(data_dict, rng):
    data_dict["points"], _ = _drop_lidar(data_dict["points"], "sparse", {"SPARSE_KEEP": 0.05}, rng)
    return data_dict


def lidar_sector(data_dict, rng):
    data_dict["points"], _ = _drop_lidar(data_dict["points"], "sector", {"MASK_FRACTION": [0.6, 0.6]}, rng)
    return data_dict


def both_dark_sparse(data_dict, rng):
    return lidar_sparse(cam_dark(data_dict, rng), rng)
