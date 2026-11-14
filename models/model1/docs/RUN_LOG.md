# Model 1 run log

A dated record of what was actually run, with the command, the config, the outcome and
where the raw log lives. The raw logs are **not** committed (`CLAUDE.md`: never commit
logs; `models/*/output/` is gitignored). They are kept in `run_logs/<date>_<name>/`
next to the repo checkout, each with a `MANIFEST.sha256` so a log can later be shown to be
unchanged. Add a new entry here every time a run starts or finishes, and keep failed
runs: a failed run is evidence too.

Numbers in this file are from the runs listed. Anything marked "estimate" is not measured.

---

## 2026-11-13: data build and environment on the RTX 5090 box

Machine: Windows 11, NVIDIA GeForce RTX 5090 (32 GB, driver 610.60), Docker 29.9.3.

| Step | Result |
| --- | --- |
| Baseline `kitti_format` from the shared Drive (3 zip parts) | 7,829 / 7,829 each of `velodyne`, `calib`, `label_2`. Its `image_2` is the baseline's blank placeholders (3,506 files, 5,796 bytes each): not usable by Model 1. |
| Raw `cam_stereo_left` archive from Drive (`z01`..`z25` + `.zip`) | 26 / 26 parts, 36.68 GB, every part the exact expected size |
| `calib_cam_stereo_left.json` | not in the camera archive: regenerated from `tonemap_stats.json` (`data_prep/00_make_calib_from_stats.py`). P equals P2 in 7,829 / 7,829 baseline calib files. Width/height (1920x1024) are inferred, not read from the original file. |
| `01_extract_camera.py` | 7,829 TIFFs: 3,469 train, 781 val, 1,847 test_clear, 946 test_light_fog, 786 test_dense_fog (all match) |
| `02_build_image_2.py build` (in Docker) | 7,829 / 7,829 PNGs, 0 errors, 15.35 GB |
| `03_verify_images.py` | ALL CHECKS PASSED. Median lidar-image shift (+0, +0) px over 91 frames (limit +/-8). Same as the (0, 0) px recorded on the data branch. |
| Docker volume `stf_kitti_mm` | 7,829 each of velodyne, calib, label_2 and real image_2 PNGs |
| Infos (`03_create_infos.py`, via the runner) | train, val, test_clear, test_light_fog, test_dense_fog, trainval, dbinfos_train generated |
| Unit tests (`run_model1_docker.ps1`, default command) | **20 passed** (38 s) on the 5090, including the full OpenPCDet training step and inference for both configs on synthetic frames |

Environment notes:

- The Drive tar `stf-openpcdet-pinned.tar` was built on a 4090 (PyTorch 2.0.1 + CUDA 11.8, kernels up
  to sm_90). On the 5090 (sm_120) it fails with "no kernel image is available for execution on the
  device". The image was rebuilt from `baseline/docker/Dockerfile` (CUDA 12.8, torch 2.7.1+cu128,
  spconv-cu120): `stf-openpcdet:pinned`, then `stf-model1:pinned` on top. This is a different
  environment from the one the baseline numbers came from (4090, torch 2.0.1). Record that when
  comparing, even though the pinned OpenPCDet commit is the same (233f849).
- `run_model1_docker.ps1` crashed on Windows PowerShell 5.1 at its first `docker volume inspect` and
  regenerated the infos on every run. Both fixed in this branch (`Test-Docker`, last-line check).

Raw record: `run_logs/2026-11-13_M3_nan_investigation/env/environment.txt`.

---

## 2026-11-13: M3 (200-frame slice) run 1, and the NaN loss

Goal of M3 (README): overfit 200 real frames, watch the gate scores, time an epoch.
Data: the first 200 frames of `kitti_infos_train.pkl` (`kitti_infos_train200.pkl`, made inside the
volume). 50 iterations per epoch at batch 4. Config `cfgs/model1/model1_lidar_rgb.yaml` as committed
(`USE_AMP: True`, LR 3e-3 one-cycle, grad-norm clip 10).

**Result: failed. The loss goes NaN within the first 50 iterations.**

| Run (extra_tag) | Setup | Outcome |
| --- | --- | --- |
| `m3_overfit` | 30 epochs, `--fix_random_seed`, config as committed | iter 1 loss 32.76, NaN by iter 50 |
| `m3_noamp` | 3 epochs, `--set OPTIMIZATION.USE_AMP False` | iter 1 loss 32.74, NaN by iter 50 |
| `m3_rand1`, `m3_rand2` | 3 epochs, no fixed seed | iter 1 loss 27.78 / 31.39, NaN by iter 50 both |
| `m3_dbg`, `m3_dbg2` | real `train.py` plus a per-iteration print hook | NaN first appears at iteration 9 to 11 |
| `m3_cfgnoamp` | 3 epochs, **`USE_AMP: False` edited in the yaml** | **finite: loss 50.0, 5.8, 5.3, 5.1, 4.9, 4.6; 0 NaN messages** |
| (replica, not through `train.py`) | plain fp32 loop, same data, 150 iterations | finite: loss about 14 down to about 4 |
| `hooktest` | `tools/debug_train_hook.py`, config as committed (AMP on), 14 iterations | no NaN in 14 iterations; iteration-1 loss only 9.4; gradient norm `inf` at 4 of the first 5 steps, then 200 to 650 |

What the evidence shows, and what it does not:

- `--set OPTIMIZATION.USE_AMP False` **does not turn mixed precision off.** `tools/train.py` reads
  `args.use_amp = ... cfg.OPTIMIZATION.get('USE_AMP')` at line 60, before `cfg_from_list` applies
  `--set` at lines 62-63. So `m3_noamp` was an AMP-on run, and "NaN persists without AMP" was wrong.
- With `USE_AMP: False` in the yaml, the same 200 frames train without NaN. Mixed precision is the main
  suspect. This is one 3-epoch run, so it is strong evidence of association, not proof of cause.
- The failure is not seed-only (two unfixed-seed runs also failed) but is run-dependent: `hooktest`
  survived 14 iterations. The initial lidar auxiliary loss varies a lot between runs (about 5 to 74 at
  iteration 1), and the gradient norms under AMP are large and often `inf`.
- The NaN shows up first in `spatial_features`, the output of the lidar 3D backbone, with finite
  inputs: it is in the weights/gradients, not in the data. At the first non-finite loss, 164 weights
  were already non-finite.
- The "NaN or Inf found in input tensor" lines are tensorboardX complaining about the NaN loss. They
  are a symptom.

Open decision (needs Anushka, who owns `rgfusion/` and the configs): either train in fp32
(`USE_AMP: False`, a recorded deviation from `context.md` section 8 "mixed precision throughout"), or
fix the auxiliary heads' initialisation and loss scale so that fp16 is safe. Neither is applied yet.

Raw logs: `run_logs/2026-11-13_M3_nan_investigation/logs/` (training logs, the hook output) and
`scripts/` (the exact scripts used). Timing: on these 200-frame runs an epoch of 50 iterations took about
15 s after a 30 s first-iteration warm-up. That is not a full-data epoch time (estimate only).

---

## Status after these runs

M3 is **not** complete. Not yet measured: a clean overfit curve, the gate scores `r_lidar` / `r_cam`
over a healthy run, a full-data epoch time, GPU memory at batch 4 on the full set. M4 and M5 have not
been started.
