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

## 2026-11-13 to 2026-11-14: M3 follow-up and two more environment fixes

- `m3_fp32_200f_30ep` (200 frames, 30 epochs, `USE_AMP: False` in a scratch copy of the yaml, seed 666):
  finite throughout, epoch-average loss about 4.8 down to about 3.2, 0 NaN messages.
- `m3_fp32_full_1ep` (full 3,469 frames, 1 epoch, same setup): **one epoch = 4 min 51 s** of training
  (868 iterations, about 0.34 s each), loss average 4.82, 0 NaN. GPU memory about 13 GB at batch 4.
- Both runs then crashed in the evaluation that `train.py` runs after training, exposing two problems in
  the rebuilt 5090 image, both fixed in `models/model1/docker/Dockerfile` (not in `baseline/docker/`):
  1. `torch.load` default `weights_only=True` (PyTorch >= 2.6) rejects OpenPCDet checkpoints:
     `UnpicklingError: Weights only load failed` at `detector3d_template.py:367`. Patched with
     `weights_only=False` at the 4 call sites (the build asserts the count).
  2. `nvJitLinkError ERROR_INTERNAL ... may need newer version of nvJitLink library` in the AP scoring
     (numba-cuda IoU kernels): the image had `nvidia-nvjitlink-cu12` 12.8.61 (torch's pin) next to
     `nvidia-cuda-nvcc-cu12` 12.9.86, while `baseline/docker/requirements.lock.txt` pins nvjitlink
     12.9.86. Installed 12.9.86 (`--no-deps`). torch still runs on the GPU afterwards. The same mismatch
     exists in a fresh build of the baseline image: tell Rohil.
- After both fixes: single-checkpoint evaluation of the 200-frame model on val (781 frames) completed and
  printed an AP table (3D AP, moderate, about 28; a 200-frame model, so a pipeline check, not a result).
- `tools/eval_calibration.py` crashed on real results (`ValueError: all the input arrays must have same
  number of dimensions`): frames with no detections store `name` as an empty float array, so
  `name == "Car"` was a scalar. Fixed, with a regression test (6 tests in that file pass).
- Archive: `run_logs/2026-11-13_M3_nan_investigation/`.

M3 goals met: loss falls over a healthy run, an epoch is timed, and the evaluation path works end to end.

---

## 2026-11-13 to 2026-11-14: M5, the full Model 1 (lidar + RGB), fp32

**Result: completed, exit 0, no NaN anywhere. Single run, single seed.**

Setup: `models/model1/tools/train_and_eval.sh` in `stf-model1:pinned`, config
`cfgs/model1/model1_lidar_rgb.yaml` with **`USE_AMP: False` set by `sed` in the deployed copy** (the committed
yaml still says `USE_AMP: True`; a recorded deviation from `context.md` section 8), `--fix_random_seed`
(666), batch 4, 80 epochs, everything else as committed. Repo HEAD at launch: `5159767` plus uncommitted
Dockerfile and `eval_calibration.py` fixes (above). Started 2026-11-13 21:22 local; training ended
03:32 (6 h 10 min); best-on-val over all 80 checkpoints done 04:40; test splits, ECE and reliability done
04:45. Epoch-average training loss 5.6 (epoch 1) to 2.11 (epoch 80).

Best-on-val epoch (clear val, Car 3D AP R40 IoU 0.5 moderate): **epoch 63**. Test splits are never used to
choose.

Car 3D AP, R40, IoU 0.5 (frames: 1,847 / 946 / 786, as required):

| Split | Model 1 easy / mod / hard | Our baseline reproduction | Paper (PV-RCNN †, alpha = 0) |
| --- | --- | --- | --- |
| Clear | 81.39 / **77.70** / 72.72 | 78.99 / 76.59 / 71.64 | 79.61 / 77.05 / 71.03 |
| Light fog | 78.61 / **76.89** / 74.31 | 68.07 / 68.26 / 66.49 | 69.55 / 70.17 / 68.44 |
| Dense fog | 68.13 / **67.85** / 65.58 | 38.23 / 39.49 / 38.75 | 45.03 / 46.00 / 45.08 |

Moderate AP vs our baseline reproduction: clear +1.1, light fog +8.6, dense fog +28.4. (The "GATE PASS"
line the script prints is the baseline's reproduction gate and means nothing for Model 1.) The paper's
moderate > easy inversion in fog does not appear for Model 1 (76.89 < 78.61, 67.85 < 68.13).

Calibration (`tools/eval_calibration.py`; Car, every detection with score >= 0.1, correct = TP at 3D IoU
0.5, 15 equal-width bins):

| Split | ECE | D-ECE 0-20 m | 20-40 m | 40-70.4 m | mean confidence / accuracy |
| --- | --- | --- | --- | --- | --- |
| Clear | 0.162 | 0.196 | 0.184 | 0.138 | 0.470 / 0.596 |
| Light fog | 0.171 | 0.196 | 0.198 | 0.150 | 0.459 / 0.588 |
| Dense fog | 0.150 | 0.126 | 0.151 | 0.156 | 0.330 / 0.357 |

Reliability (`tools/summarize_reliability.py`, medians per split):

| Split | frames | r_lidar | r_cam | mean camera attention weight |
| --- | --- | --- | --- | --- |
| Clear | 1,847 | 0.797 | 0.832 | 0.521 |
| Light fog | 946 | 0.726 | 0.832 | 0.547 |
| Dense fog | 786 | 0.420 | 0.832 | 0.627 |

How to read this (hedged on purpose):

- **The AP gain in fog is not yet attributable.** Model 1 differs from the baseline in the lidar trunk
  (SECOND-style, not PV-RCNN), no `gt_sampling`, an added camera stream, fp32, and a different software
  stack (RTX 5090, torch 2.7.1 vs the baseline's 4090, torch 2.0.1). Only the lidar-only ablation (M4, same
  trunk and head, camera off) can say how much is the camera and how much is the trunk. One seed.
- **Calibration is poor and nothing to compare against yet.** ECE is about 0.15-0.17 on every split, far
  above the 0.05 target (which is defined with two modalities degraded and is not measured here: that needs
  the corruption engine). Scores are under-confident on clear and light fog (mean confidence below
  accuracy). The baseline's ECE has not been measured, so there is no calibration comparison. Per
  `context.md` section 1, an AP gain without a calibration gain is not progress on the thesis; this is such
  a case so far.
- **Reliability moves the way the thesis hopes, with caveats.** The median lidar reliability falls from
  0.797 (clear) to 0.726 (light fog) to 0.420 (dense fog) in fog never seen in training, and the mean camera
  attention weight rises from 0.521 to 0.627. The camera reliability does not change (0.832 in all three
  splits). These are medians of one run; no check yet that the score tracks real degradation per frame.

Evidence: `run_logs/2026-11-14_M5_fp32_seed666/` (`MANIFEST.sha256`; container log, `train.log`,
`eval_all_val.log` for all 80 epochs, per-split eval logs, ECE json, reliability jsonl, the three
`result.pkl`, best (63) and last (80) checkpoints, TensorBoard events, the scripts used).

---

## Status after these runs

M3 is complete in the sense of its goals (see above). **M5 is complete as a first run** (one seed, fp32).
Not done: M4 (lidar-only ablation, needed to attribute the AP gain), the baseline's ECE on the same definition
(run `eval_calibration.py` on the baseline's `result.pkl`), ECE with two modalities degraded (needs the
corruption engine), an fp16-safe fix of the auxiliary heads, M6 (writeup). Not yet committed: the Dockerfile
fixes, the `eval_calibration.py` fix and test, and this entry.
