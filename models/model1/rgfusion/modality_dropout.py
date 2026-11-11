"""Train-time modality dropout: the only "degradation" Model 1 ever sees in training.

context.md section 6: training uses clear weather only and weather corruption is
applied at evaluation time only. So the perturbations here are deliberately
generic and NOT shaped like fog, snow or icing:

    lidar  : sparse  (keep a tiny random fraction of points)
             noise   (replace every point with a uniform random point)
             sector  (remove a contiguous azimuth sector)
    camera : zeros   (black image)
             noise   (uniform noise image)
             mask    (grey rectangle over part of the image)

At most one stream is perturbed per sample, so there is always one usable
stream. Each sample records how much of each stream survived
(``m1_keep_lidar``, ``m1_keep_cam`` in [0, 1]); the reliability target is
scaled by it (see reliability.py).

Registered as an OpenPCDet DataAugmentor method by register.py and enabled by a
``model1_modality_dropout`` entry in the model config's AUG_CONFIG_LIST. Runs in
training only (OpenPCDet never augments at test time).
"""

from functools import partial

import numpy as np

LIDAR_MODES = ("sparse", "noise", "sector")
CAMERA_MODES = ("zeros", "noise", "mask")


def _drop_lidar(points: np.ndarray, mode: str, cfg, rng) -> tuple[np.ndarray, float]:
    n = len(points)
    if mode == "sparse":
        keep = max(int(n * cfg.get("SPARSE_KEEP", 0.02)), min(n, 50))  # never empty: spconv needs voxels
        idx = rng.choice(n, size=keep, replace=False)
        return points[idx], keep / max(n, 1)
    if mode == "noise":
        lo = np.array(cfg.get("NOISE_RANGE_MIN", [0.0, -40.0, -3.0]), dtype=np.float32)
        hi = np.array(cfg.get("NOISE_RANGE_MAX", [70.4, 40.0, 1.0]), dtype=np.float32)
        out = points.copy()
        out[:, :3] = rng.uniform(lo, hi, size=(n, 3)).astype(np.float32)
        out[:, 3] = rng.uniform(0.0, 255.0, size=n).astype(np.float32)  # STF intensity is 0-255 (D5)
        return out, 0.0
    if mode == "sector":
        frac = rng.uniform(*cfg.get("MASK_FRACTION", [0.4, 0.8]))
        az = np.arctan2(points[:, 1], points[:, 0])
        lo_q = rng.uniform(0.0, 1.0 - frac)
        a0, a1 = np.quantile(az, [lo_q, lo_q + frac])
        keep_mask = (az < a0) | (az > a1)
        if keep_mask.sum() < 50:
            keep_mask[rng.choice(n, size=min(n, 50), replace=False)] = True
        return points[keep_mask], float(keep_mask.mean())
    raise ValueError(f"unknown lidar mode {mode!r}")


def _drop_camera(image: np.ndarray, mode: str, cfg, rng) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    if mode == "zeros":
        return np.zeros_like(image), 0.0
    if mode == "noise":
        return rng.uniform(0.0, 1.0, size=image.shape).astype(image.dtype), 0.0
    if mode == "mask":
        frac = rng.uniform(*cfg.get("MASK_FRACTION", [0.4, 0.8]))
        aspect = rng.uniform(0.5, 2.0)
        mh = int(min(h, np.sqrt(frac * h * w / aspect)))
        mw = int(min(w, frac * h * w / max(mh, 1)))
        y0 = rng.integers(0, h - mh + 1)
        x0 = rng.integers(0, w - mw + 1)
        out = image.copy()
        out[y0:y0 + mh, x0:x0 + mw] = 0.5
        return out, 1.0 - (mh * mw) / (h * w)
    raise ValueError(f"unknown camera mode {mode!r}")


def apply_modality_dropout(data_dict: dict, cfg, rng: np.random.Generator) -> dict:
    """Pure function behind the augmentor (separate so tests can call it directly)."""
    p_lidar = float(cfg.get("P_LIDAR", 0.25))
    p_cam = float(cfg.get("P_CAMERA", 0.25))
    assert p_lidar + p_cam <= 1.0, "P_LIDAR + P_CAMERA must be <= 1 (at most one stream is dropped)"
    keep_lidar, keep_cam = 1.0, 1.0
    u = rng.uniform()
    if u < p_lidar:
        mode = cfg.get("LIDAR_MODES", LIDAR_MODES)[rng.integers(len(cfg.get("LIDAR_MODES", LIDAR_MODES)))]
        data_dict["points"], keep_lidar = _drop_lidar(data_dict["points"], mode, cfg, rng)
    elif u < p_lidar + p_cam and "images" in data_dict:
        mode = cfg.get("CAMERA_MODES", CAMERA_MODES)[rng.integers(len(cfg.get("CAMERA_MODES", CAMERA_MODES)))]
        data_dict["images"], keep_cam = _drop_camera(data_dict["images"], mode, cfg, rng)
    data_dict["m1_keep_lidar"] = np.float32(keep_lidar)
    data_dict["m1_keep_cam"] = np.float32(keep_cam)
    return data_dict


def model1_modality_dropout(self, data_dict=None, config=None):
    """OpenPCDet DataAugmentor method (bound by register.py)."""
    if data_dict is None:
        return partial(self.model1_modality_dropout, config=config)
    # np.random is reseeded per dataloader worker by OpenPCDet; draw a generator from it
    # so --fix_random_seed keeps this reproducible.
    rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))
    return apply_modality_dropout(data_dict, config, rng)
