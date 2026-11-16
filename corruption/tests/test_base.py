"""corruption/base.py: the seeding helpers every channel depends on. CPU only, no model/OpenPCDet dependency."""

import numpy as np
import pytest

from corruption.base import check_severity, rng_for_frame, seed_for_frame


def test_seed_is_deterministic_for_the_same_triple():
    a = seed_for_frame("000123", "fog", 0.5)
    b = seed_for_frame("000123", "fog", 0.5)
    assert a == b


def test_seed_differs_across_frame_channel_and_severity():
    base = seed_for_frame("000123", "fog", 0.5)
    assert seed_for_frame("000124", "fog", 0.5) != base          # different frame
    assert seed_for_frame("000123", "icing", 0.5) != base        # different channel
    assert seed_for_frame("000123", "fog", 0.6) != base          # different severity


def test_rng_for_frame_is_usable_and_deterministic():
    a = rng_for_frame("000123", "fog", 0.5).integers(0, 10**9)
    b = rng_for_frame("000123", "fog", 0.5).integers(0, 10**9)
    assert a == b


@pytest.mark.parametrize("severity", [0.0, 0.5, 1.0])
def test_check_severity_accepts_the_valid_range(severity):
    check_severity(severity)  # must not raise


@pytest.mark.parametrize("severity", [-0.001, 1.001, -5.0, 5.0, float("nan")])
def test_check_severity_rejects_outside_the_valid_range(severity):
    with pytest.raises(ValueError):
        check_severity(severity)


def test_rng_for_frame_returns_a_real_generator():
    rng = rng_for_frame("000123", "fog", 0.5)
    assert isinstance(rng, np.random.Generator)
