"""openpcdet_entry_seed.py patches set_random_seed to use M1_SEED instead of OpenPCDet's hardcoded 666.
CPU-only: imports the module (which does the patching at import time) and checks the patched function's behaviour,
without running any training."""

import importlib
import os
import sys

import pytest


@pytest.fixture
def seeded_module(monkeypatch):
    monkeypatch.setenv("M1_SEED", "777")
    sys.modules.pop("openpcdet_entry_seed", None)
    sys.modules.pop("openpcdet_entry", None)
    mod = importlib.import_module("openpcdet_entry_seed")
    yield mod
    sys.modules.pop("openpcdet_entry_seed", None)
    sys.modules.pop("openpcdet_entry", None)


def test_missing_m1_seed_fails_loudly(monkeypatch):
    monkeypatch.delenv("M1_SEED", raising=False)
    sys.modules.pop("openpcdet_entry_seed", None)
    with pytest.raises(KeyError):
        importlib.import_module("openpcdet_entry_seed")


def test_patches_the_hardcoded_666(seeded_module):
    import pcdet.utils.common_utils as common_utils
    import numpy as np

    # Directly verify the seed actually applied is M1_SEED (777), not 666, by reproducing what set_random_seed(777) does.
    common_utils.set_random_seed(666)
    a = np.random.randint(0, 10**9)
    np.random.seed(777)
    b = np.random.randint(0, 10**9)
    assert a == b


def test_local_rank_offset_preserved(seeded_module):
    import pcdet.utils.common_utils as common_utils
    import numpy as np
    common_utils.set_random_seed(667)                  # 666 + LOCAL_RANK(1)
    a = np.random.randint(0, 10**9)
    np.random.seed(778)                                # M1_SEED(777) + offset(1)
    b = np.random.randint(0, 10**9)
    assert a == b


def test_importing_twice_in_one_process_does_not_double_patch(monkeypatch):
    """Simulates train.py then test.py sharing a long-lived interpreter (not the case here, since each docker run
    is a fresh process, but a second import must still be a safe no-op, and this is what caught the original bug)."""
    import pcdet.utils.common_utils as common_utils
    import numpy as np

    monkeypatch.setenv("M1_SEED", "777")
    for _ in range(2):
        sys.modules.pop("openpcdet_entry_seed", None)
        sys.modules.pop("openpcdet_entry", None)
        importlib.import_module("openpcdet_entry_seed")

    common_utils.set_random_seed(666)
    a = np.random.randint(0, 10**9)
    np.random.seed(777)
    b = np.random.randint(0, 10**9)
    assert a == b                                      # still 777, not 777 applied to an already-patched 666

    sys.modules.pop("openpcdet_entry_seed", None)
    sys.modules.pop("openpcdet_entry", None)
