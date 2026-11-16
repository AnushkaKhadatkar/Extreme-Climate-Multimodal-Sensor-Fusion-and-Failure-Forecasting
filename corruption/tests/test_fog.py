"""corruption/fog.py: severity mapping and the apply() contract. CPU only; needs the vendored LiDAR_fog_sim and
its LUTs (both committed), but no OpenPCDet/model dependency."""

import numpy as np
import pytest

from corruption.base import rng_for_frame
from corruption.fog import ALPHAS_WITH_LUT, apply, severity_to_alpha


def synthetic_points(seed=0, n=2000):
    rng = np.random.default_rng(seed)
    pts = np.zeros((n, 4), dtype=np.float32)
    pts[:, 0] = rng.uniform(2, 60, n)
    pts[:, 1] = rng.uniform(-20, 20, n)
    pts[:, 2] = rng.uniform(-2, 1, n)
    pts[:, 3] = rng.uniform(0, 255, n)
    return pts


def test_severity_maps_onto_shipped_alphas_only():
    for s in (0.0, 0.1, 0.25, 0.5, 0.73, 1.0):
        assert severity_to_alpha(s) in ALPHAS_WITH_LUT


def test_severity_extremes_hit_the_lightest_and_densest_shipped_alpha():
    assert severity_to_alpha(0.0) == min(ALPHAS_WITH_LUT)
    assert severity_to_alpha(1.0) == max(ALPHAS_WITH_LUT)


def test_severity_is_monotonic_in_alpha():
    alphas = [severity_to_alpha(s) for s in np.linspace(0, 1, 11)]
    assert alphas == sorted(alphas)


@pytest.mark.parametrize("severity", [-0.1, 1.1])
def test_severity_out_of_range_raises(severity):
    rng = rng_for_frame("000001", "fog", 0.5)
    with pytest.raises(ValueError):
        apply({"points": synthetic_points()}, severity, rng)


def test_apply_returns_float32_points_same_shape():
    pts = synthetic_points()
    out = apply({"points": pts.copy()}, 0.5, rng_for_frame("000001", "fog", 0.5))
    assert out["points"].dtype == np.float32
    assert out["points"].shape == pts.shape


def test_apply_does_not_touch_images():
    pts = synthetic_points()
    img = np.full((8, 8, 3), 0.5, dtype=np.float32)
    data = {"points": pts.copy(), "images": img.copy()}
    out = apply(data, 0.5, rng_for_frame("000001", "fog", 0.5))
    assert np.array_equal(out["images"], img)


def test_apply_is_deterministic_for_the_same_frame_seed_and_severity():
    """The regression test for the real bug found and fixed: fog_simulation.py's noise draws from a module-level
    RNG seeded once at import time, so without the reseed in fog.apply() this would be flaky depending on call
    order. Call apply() with something else in between to catch exactly that failure mode."""
    pts = synthetic_points()
    rng_a = rng_for_frame("000042", "fog", 0.7)
    out_a = apply({"points": pts.copy()}, 0.7, rng_a)

    # Interleave an unrelated call (different frame, different severity) to perturb the shared module RNG's
    # state, the way a real batch of frames processed in sequence would.
    apply({"points": synthetic_points(seed=99)}, 0.3, rng_for_frame("999999", "fog", 0.3))

    rng_b = rng_for_frame("000042", "fog", 0.7)
    out_b = apply({"points": pts.copy()}, 0.7, rng_b)

    assert np.array_equal(out_a["points"], out_b["points"])


def test_apply_differs_for_different_frames_at_the_same_severity():
    """Determinism per-frame should not mean every frame gets an identical corruption draw."""
    pts = synthetic_points()
    out_a = apply({"points": pts.copy()}, 0.7, rng_for_frame("000001", "fog", 0.7))
    out_b = apply({"points": pts.copy()}, 0.7, rng_for_frame("000002", "fog", 0.7))
    assert not np.array_equal(out_a["points"], out_b["points"])


def test_severity_zero_vs_one_differ_in_attenuation():
    """Higher severity (alpha) should attenuate intensity more, on average, for identical input points and a
    hard-target-only comparison (soft=False avoids the backscatter relocation noise dominating the comparison)."""
    pts = synthetic_points()
    rng = rng_for_frame("000001", "fog", 0.0)
    low = apply({"points": pts.copy()}, 0.0, rng, soft=False)
    rng = rng_for_frame("000001", "fog", 1.0)
    high = apply({"points": pts.copy()}, 1.0, rng, soft=False)
    assert high["points"][:, 3].mean() < low["points"][:, 3].mean()


def test_apply_mutates_and_returns_the_same_dict():
    data = {"points": synthetic_points()}
    out = apply(data, 0.5, rng_for_frame("000001", "fog", 0.5))
    assert out is data
