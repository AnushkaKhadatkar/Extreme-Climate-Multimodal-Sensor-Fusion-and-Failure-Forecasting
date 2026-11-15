"""The evaluation-time sensor failures: what each one does, and that they are deterministic (CPU only)."""

import numpy as np
import pytest

import eval_corruptions as ec


def sample(seed=0):
    rng = np.random.default_rng(seed)
    n = 4000
    pts = np.stack([rng.uniform(1, 70, n), rng.uniform(-30, 30, n), rng.uniform(-2, 1, n), rng.uniform(0, 255, n)], 1)
    return {"frame_id": "000001", "points": pts.astype(np.float32),
            "images": rng.uniform(0.2, 0.9, (64, 96, 3)).astype(np.float32)}


def run(name, seed=0):
    return getattr(ec, name)(sample(), np.random.default_rng(seed))


@pytest.mark.parametrize("name", ec.CONDITIONS)
def test_shapes_dtypes_and_range_are_preserved(name):
    out = run(name)
    assert out["images"].shape == (64, 96, 3) and out["images"].dtype == np.float32
    assert out["images"].min() >= 0.0 and out["images"].max() <= 1.0
    assert out["points"].ndim == 2 and out["points"].shape[1] == 4 and len(out["points"]) >= 50  # spconv needs voxels


@pytest.mark.parametrize("name", ec.CONDITIONS)
def test_deterministic_for_a_given_seed(name):
    a, b = run(name, seed=7), run(name, seed=7)
    assert np.array_equal(a["points"], b["points"]) and np.array_equal(a["images"], b["images"])


def test_camera_failures():
    assert run("cam_zeros")["images"].max() == 0.0
    assert np.allclose(run("cam_dark")["images"], sample()["images"] * 0.1)
    occ = run("cam_occlusion")["images"]
    assert 0.5 < (occ == 0.5).all(axis=2).mean() < 0.7                       # about 60% grey
    blur, clean = run("cam_blur")["images"], sample()["images"]
    assert blur.std() < clean.std() * 0.3                                    # most of the contrast is gone
    for name in ("cam_zeros", "cam_dark", "cam_blur", "cam_occlusion"):      # camera failures leave the lidar alone
        assert np.array_equal(run(name)["points"], sample()["points"])


def test_lidar_failures():
    clean = sample()
    assert len(run("lidar_sparse")["points"]) == pytest.approx(0.05 * len(clean["points"]), abs=5)
    kept = len(run("lidar_sector")["points"]) / len(clean["points"])
    assert 0.3 < kept < 0.5                                                  # a 60% sector is gone
    for name in ("lidar_sparse", "lidar_sector"):                            # lidar failures leave the camera alone
        assert np.array_equal(run(name)["images"], clean["images"])


def test_both_degrades_both():
    out, clean = run("both_dark_sparse"), sample()
    assert np.allclose(out["images"], clean["images"] * 0.1) and len(out["points"]) < 0.1 * len(clean["points"])


def test_no_camera_is_tolerated():
    d = sample()
    d.pop("images")
    for name in ("cam_zeros", "cam_dark", "cam_blur", "cam_occlusion"):
        assert "images" not in getattr(ec, name)(dict(d), np.random.default_rng(0))
