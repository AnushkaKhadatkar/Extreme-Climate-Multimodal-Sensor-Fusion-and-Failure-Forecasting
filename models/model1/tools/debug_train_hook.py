#!/usr/bin/env python3
"""Run OpenPCDet's real tools/train.py for Model 1 with a per-iteration debug print.

Use it to find out where a training run goes non-finite, without changing any Model 1
code. It prints, for the first M1_DEBUG_ITERS iterations (default 14):
  - the loss, the two auxiliary losses and the number of points per sample,
  - the gradient norm after unscaling (what OpenPCDet clips),
and, at the first non-finite loss, which weights and which batch tensors are non-finite.
Every line starts with "M1DBG" so it can be grepped out of the training output.

Arguments are passed to train.py unchanged, exactly like openpcdet_entry.py:

    python debug_train_hook.py --cfg_file cfgs/model1/model1_lidar_rgb.yaml --extra_tag dbg \
        --epochs 3 --set DATA_CONFIG.INFO_PATH.train "['kitti_infos_train200.pkl']"

It was written for the M3 NaN investigation (docs/RUN_LOG.md, 2026-11-13). It exits with
os._exit after the last debug iteration, which leaves DataLoader workers running for a
moment: kill them (pkill -f debug_train_hook) if a pipe into grep does not close.

Note for the investigation: `--set OPTIMIZATION.USE_AMP False` does NOT turn mixed
precision off, because train.py reads USE_AMP before it applies --set. Edit the yaml.
"""

import os
import runpy
import sys
from pathlib import Path

import torch

MODEL1_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL1_DIR))

import rgfusion.register  # noqa: E402,F401  (side effect: registers Model1Fusion)
from rgfusion.detector import Model1Fusion  # noqa: E402

N_ITERS = int(os.environ.get("M1_DEBUG_ITERS", "14"))
TOOLS = Path(os.environ.get("OPENPCDET_ROOT", "/opt/OpenPCDet")) / "tools"
os.chdir(TOOLS)
sys.path.insert(0, str(TOOLS))

import train_utils.train_utils as tu  # noqa: E402  (needs TOOLS on sys.path)

_orig_clip = tu.clip_grad_norm_
_steps = [0]


def _clip(params, max_norm, *a, **k):
    norm = _orig_clip(list(params), max_norm, *a, **k)
    _steps[0] += 1
    if _steps[0] <= N_ITERS:
        print(f"M1DBG step{_steps[0]} gradnorm={float(norm):.2f}", flush=True)
    return norm


tu.clip_grad_norm_ = _clip

_orig_loss = Model1Fusion.get_training_loss
_iters = [0]
_reported = [False]


def _loss(self, batch_dict):
    loss, tb, disp = _orig_loss(self, batch_dict)
    _iters[0] += 1
    if _iters[0] <= N_ITERS:
        npts = [int(p) for p in torch.bincount(batch_dict["points"][:, 0].long())]
        print(f"M1DBG it{_iters[0]} loss={loss.item():.3f} aux_lidar={tb.get('m1_aux_lidar', float('nan')):.2f} "
              f"aux_cam={tb.get('m1_aux_cam', float('nan')):.2f} npts={npts}", flush=True)
    if not torch.isfinite(loss) and not _reported[0]:
        _reported[0] = True
        bad_w = [n for n, p in self.named_parameters() if not torch.isfinite(p).all()]
        print(f"M1DBG FIRST NON-FINITE LOSS at it{_iters[0]}: {len(bad_w)} non-finite weights, e.g. {bad_w[:5]}; "
              f"frames={list(batch_dict['frame_id'])}", flush=True)
        for key in ("points", "voxels", "voxel_features", "voxel_coords", "voxel_num_points"):
            v = batch_dict.get(key)
            if torch.is_tensor(v):
                print(f"M1DBG   {key}: shape {tuple(v.shape)} finite={bool(torch.isfinite(v.float()).all())} "
                      f"min={float(v.float().min()):.2f} max={float(v.float().max()):.2f}", flush=True)
    if _iters[0] >= N_ITERS:
        print("M1DBG done", flush=True)
        os._exit(0)
    return loss, tb, disp


Model1Fusion.get_training_loss = _loss

if __name__ == "__main__":
    sys.argv = [str(TOOLS / "train.py")] + sys.argv[1:]
    runpy.run_path(str(TOOLS / "train.py"), run_name="__main__")
