#!/usr/bin/env python3
"""Like openpcdet_entry.py, but the seed --fix_random_seed applies is M1_SEED (env var), not OpenPCDet's
hardcoded 666.

tools/train.py always calls common_utils.set_random_seed(666 + cfg.LOCAL_RANK) when --fix_random_seed is passed
(pcdet/utils/common_utils.py; no --seed argument exists), and also passes seed=666 to build_dataloader. This patches
pcdet.utils.common_utils.set_random_seed so any call with 666 uses M1_SEED instead, LOCAL_RANK offsets preserved, and
every other call (there are none elsewhere in train.py) is untouched; build_dataloader's seed=666 is approximated
the same way. Needed for the extra-seed runs of M5p / M4bp (docs/RUN_LOG.md): one trained model per seed, so the
gate-vs-no-gate difference can be judged against run-to-run variance, not guessed from a single run.

Usage: same as openpcdet_entry.py, plus M1_SEED set in the environment.
    M1_SEED=777 python openpcdet_entry_seed.py train --cfg_file ... --extra_tag ... --fix_random_seed ...
"""

import os
import sys

SEED = int(os.environ["M1_SEED"])  # fail loudly if the caller forgot to set it

import pcdet.utils.common_utils as common_utils  # noqa: E402

# Idempotent: if this module is imported twice in one process (e.g. two test cases, or train.py then test.py in the
# same interpreter), re-patching an already-patched function would double-apply the offset logic using 777 as the
# new "666", corrupting the seed silently. Patch the original exactly once and mark it so a second import is a no-op.
if not getattr(common_utils.set_random_seed, "_m1_seed_patched", False):
    _orig_set_seed = common_utils.set_random_seed

    def _patched_set_seed(seed):
        offset = seed - 666 if seed >= 666 else 0  # preserve a LOCAL_RANK offset if one was added
        _orig_set_seed(SEED + offset)

    _patched_set_seed._m1_seed_patched = True
    common_utils.set_random_seed = _patched_set_seed

import openpcdet_entry  # noqa: E402  (does the rest: registers Model1Fusion, runs train.py/test.py as __main__)

if __name__ == "__main__":
    openpcdet_entry.main()
