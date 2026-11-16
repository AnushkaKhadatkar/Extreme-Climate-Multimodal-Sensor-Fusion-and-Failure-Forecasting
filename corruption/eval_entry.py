#!/usr/bin/env python3
"""Run any OpenPCDet detector's tools/test.py with a corruption channel applied to every evaluated frame, WITHOUT
modifying that detector's own code or config. Generalises models/model1/rgfusion/register.py's
_install_eval_corruption pattern (same monkeypatch target, same env-var-driven hook) to work for the baseline too,
since the baseline has no hook of its own and context.md section 10 says not to modify it beyond what running it
on our splits needs -- this file lives outside baseline/ entirely, so baseline/ stays untouched.

Usage (same argument style as baseline/scripts/04_train_and_eval.sh's own test.py calls):

    CORRUPTION_CHANNEL=corruption.fog:apply CORRUPTION_SEVERITY=1.0 \
    python corruption/eval_entry.py --cfg_file cfgs/kitti_models/pv_rcnn_stf.yaml --ckpt <path> \
        --set DATA_CONFIG.DATA_SPLIT.test test_clear DATA_CONFIG.INFO_PATH.test kitti_infos_test_clear.pkl

Env:
    CORRUPTION_CHANNEL   module:function, e.g. "corruption.fog:apply" (required)
    CORRUPTION_SEVERITY  float in [0, 1] (required)
    OPENPCDET_ROOT        default /opt/OpenPCDet

Every argument after this script's own name is passed to OpenPCDet's tools/test.py unchanged -- this script does
not parse or understand them, so the whole override mechanism (--cfg_file, --ckpt, --set DATA_CONFIG...) keeps
working exactly as 04_train_and_eval.sh already uses it.

Determinism: corruption.base.rng_for_frame(frame_id, channel_name, severity) is used for every frame, so re-running
this exact (cfg, ckpt, split, channel, severity) gives the identical corrupted evaluation set every time -- the
same guarantee models/model1's M1_EVAL_CORRUPTION hook gives Model 1, now available to any OpenPCDet detector.
"""

import importlib
import os
import runpy
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))  # so `corruption.base` imports regardless of caller's cwd

from corruption.base import rng_for_frame  # noqa: E402


def _patch_torch_load_weights_only() -> None:
    """PyTorch >= 2.6 defaults torch.load's weights_only to True, which rejects OpenPCDet's own checkpoints
    (they pickle numpy scalars). The baseline's own Docker image (baseline/docker/Dockerfile) was built with an
    older torch and has never hit this; stf-openpcdet:pinned was rebuilt for the RTX 5090 with a newer torch and
    DOES hit it (confirmed: UnpicklingError loading checkpoint_epoch_67.pth with this exact message). The same
    fix already landed in models/model1/docker/Dockerfile (source-patches detector3d_template.py's 4 torch.load
    call sites) because Model 1's image is built FROM stf-openpcdet:pinned and hit the identical error first.
    This script does NOT touch baseline/docker/ (that is Rohil's owned file per BASELINE_TASKS.md section 2, and
    a Dockerfile change needs a rebuild + his sign-off, not a quick unilateral edit mid-validation-run) -- instead
    this monkeypatches torch.load's default here, scoped to this process only, so the fix lands where this script
    already has licence to patch (same pattern as the corruption-eval hook itself) without touching anything
    outside corruption/. Trusting weights_only=False here is the same judgement call already made and tested for
    Model 1's image: these are our own checkpoints, trained by this project, not arbitrary downloaded files."""
    import functools
    import torch
    original_load = torch.load
    torch.load = functools.wraps(original_load)(lambda *a, **k: original_load(*a, **{**k, "weights_only": False}))


def _install_hook(channel_spec: str, severity: float) -> None:
    module_name, fn_name = channel_spec.split(":")
    channel_fn = getattr(importlib.import_module(module_name), fn_name)
    channel_name = module_name.rsplit(".", 1)[-1]  # "corruption.fog" -> "fog", matches corruption/base.py's naming

    from pcdet.datasets.dataset import DatasetTemplate
    original_prepare_data = DatasetTemplate.prepare_data

    def prepare_data(self, data_dict):
        if not self.training:
            rng = rng_for_frame(str(data_dict.get("frame_id", "")), channel_name, severity)
            data_dict = channel_fn(data_dict, severity, rng)
        return original_prepare_data(self, data_dict)

    DatasetTemplate.prepare_data = prepare_data


def main() -> None:
    channel_spec = os.environ.get("CORRUPTION_CHANNEL")
    severity_str = os.environ.get("CORRUPTION_SEVERITY")
    if not channel_spec or severity_str is None:
        sys.exit("CORRUPTION_CHANNEL and CORRUPTION_SEVERITY must both be set in the environment. " + __doc__)
    _patch_torch_load_weights_only()
    _install_hook(channel_spec, float(severity_str))

    tools = Path(os.environ.get("OPENPCDET_ROOT", "/opt/OpenPCDet")) / "tools"
    os.chdir(tools)
    sys.path.insert(0, str(tools))  # test.py imports sibling modules relative to tools/
    sys.argv = [str(tools / "test.py")] + sys.argv[1:]
    runpy.run_path(str(tools / "test.py"), run_name="__main__")


if __name__ == "__main__":
    main()
