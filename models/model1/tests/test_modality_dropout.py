"""Train-time modality dropout: rates, exclusivity, keep fractions, never-empty lidar."""

import numpy as np
from easydict import EasyDict

from rgfusion.modality_dropout import apply_modality_dropout

CFG = EasyDict(P_LIDAR=0.25, P_CAMERA=0.25, LIDAR_MODES=["sparse", "noise", "sector"],
               CAMERA_MODES=["zeros", "noise", "mask"], SPARSE_KEEP=0.02, MASK_FRACTION=[0.4, 0.8])


def sample(rng):
    pts = rng.uniform([1, -20, -2, 0], [60, 20, 0.5, 255], size=(4000, 4)).astype(np.float32)
    img = rng.uniform(0, 1, size=(64, 96, 3)).astype(np.float32)
    return {"points": pts, "images": img}


def test_rates_and_never_both():
    rng = np.random.default_rng(0)
    n = 4000
    drop_l = drop_c = 0
    for _ in range(n):
        d = apply_modality_dropout(sample(rng), CFG, rng)
        dl, dc = d["m1_keep_lidar"] < 1, d["m1_keep_cam"] < 1
        assert not (dl and dc), "both streams perturbed in one sample"
        drop_l += dl
        drop_c += dc
    assert abs(drop_l / n - 0.25) < 0.03
    assert abs(drop_c / n - 0.25) < 0.03


def test_each_mode_and_keep_fraction():
    rng = np.random.default_rng(1)
    for mode in ["sparse", "noise", "sector"]:
        cfg = EasyDict({**CFG, "P_LIDAR": 1.0, "P_CAMERA": 0.0, "LIDAR_MODES": [mode]})
        d = apply_modality_dropout(sample(rng), cfg, rng)
        assert len(d["points"]) >= 50, f"{mode}: spconv needs at least some points"
        assert 0.0 <= d["m1_keep_lidar"] < 1.0
        assert d["m1_keep_cam"] == 1.0
    for mode, expect_zero_keep in [("zeros", True), ("noise", True), ("mask", False)]:
        cfg = EasyDict({**CFG, "P_LIDAR": 0.0, "P_CAMERA": 1.0, "CAMERA_MODES": [mode]})
        d = apply_modality_dropout(sample(rng), cfg, rng)
        assert d["images"].shape == (64, 96, 3)
        assert (d["m1_keep_cam"] == 0.0) == expect_zero_keep
        assert d["m1_keep_lidar"] == 1.0


def test_clean_sample_untouched():
    rng = np.random.default_rng(2)
    s = sample(rng)
    pts, img = s["points"].copy(), s["images"].copy()
    d = apply_modality_dropout(s, EasyDict({**CFG, "P_LIDAR": 0.0, "P_CAMERA": 0.0}), rng)
    assert np.array_equal(d["points"], pts) and np.array_equal(d["images"], img)
    assert d["m1_keep_lidar"] == 1.0 and d["m1_keep_cam"] == 1.0
