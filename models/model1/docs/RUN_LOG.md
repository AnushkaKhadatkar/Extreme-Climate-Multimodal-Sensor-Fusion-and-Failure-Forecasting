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
- **Reliability: the lidar side moves the way the thesis hopes; the camera side does not work.** The median
  lidar reliability falls from 0.797 (clear) to 0.726 (light fog) to 0.420 (dense fog) in fog never seen in
  training, with a wide spread (std 0.18 / 0.26 / 0.29; in dense fog the 10th-90th percentile range is roughly
  0.01 to 0.78 and some frames are at 0), and the mean camera attention weight rises from 0.521 to 0.627.
  **The camera reliability estimator is effectively constant**: its median equals its maximum, 0.8323, in all
  three splits (std 0.010 / 0.004 / 0.028; only 11 / 5 / 31 distinct values over 1,847 / 946 / 786 frames),
  so it is not tracking camera health per frame. This is the gate collapse that `DESIGN.md` section 7 says to
  watch for, on the camera side. Any gating effect in M5 therefore comes through `r_lidar` (and the learned
  attention), not through a per-frame trust in the camera. (An earlier version of this entry said only that the
  camera score "does not change"; the per-frame statistics above are the accurate statement.) One run; no
  check yet that `r_lidar` tracks real degradation per frame.

Evidence: `run_logs/2026-11-14_M5_fp32_seed666/` (`MANIFEST.sha256`; container log, `train.log`,
`eval_all_val.log` for all 80 epochs, per-split eval logs, ECE json, reliability jsonl, the three
`result.pkl`, best (63) and last (80) checkpoints, TensorBoard events, the scripts used).

---

## 2026-11-14: M4, the lidar-only ablation (camera off), fp32

**Result: completed, exit 0, no NaN. Single run, single seed.**

Setup: same container, script and recorded deviation as M5, with `cfgs/model1/model1_lidar_only.yaml`
(`FUSION.USE_CAMERA: False`; same lidar trunk, same head, same augmentation, same schedule and seed 666),
`USE_AMP: False` set by `sed` in the deployed copy. Started 07:33 local, training ended 09:35 (2 h 02 min,
about 1.5 min per epoch), everything done 09:59. Best-on-val epoch: **71**. Frames evaluated 1,847 / 946 / 786.
No reliability report (there is no camera stream, so no `r_lidar` / `r_cam`).

Car 3D AP, R40, IoU 0.5, moderate (easy / moderate / hard in brackets):

| Split | M4 lidar-only | M5 lidar + RGB | M5 minus M4 | Our baseline repro | Paper |
| --- | --- | --- | --- | --- | --- |
| Clear | **74.57** (78.65 / 74.57 / 69.49) | 77.70 | +3.1 | 76.59 | 77.05 |
| Light fog | **70.97** (70.71 / 70.97 / 69.09) | 76.89 | +5.9 | 68.26 | 70.17 |
| Dense fog | **44.09** (43.39 / 44.09 / 43.82) | 67.85 | **+23.8** | 39.49 | 46.00 |

(The "GATE ... FAIL" the script prints for M4 is the baseline's reproduction band, 75.05-79.05, applied to a
different model: not meaningful. M4's clear moderate is 2.0 below our baseline reproduction and the
moderate > easy inversion in both fog splits does appear for M4.)

Calibration (same definition as M5):

| Split | M4 ECE | M5 ECE | M5 minus M4 | M4 D-ECE 0-20 / 20-40 / 40-70.4 m | M4 mean confidence / accuracy |
| --- | --- | --- | --- | --- | --- |
| Clear | 0.168 | 0.162 | -0.006 | 0.189 / 0.203 / 0.140 | 0.419 / 0.553 |
| Light fog | 0.180 | 0.171 | -0.009 | 0.191 / 0.202 / 0.166 | 0.400 / 0.527 |
| Dense fog | 0.108 | 0.150 | **+0.042** | 0.126 / 0.142 / 0.083 | 0.278 / 0.320 |

How to read this (hedged on purpose):

- **Most of M5's fog gain over the baseline is not the lidar trunk.** The lidar-only model with the same
  trunk, head, schedule and seed lands near the baseline and the paper in fog (light 70.97, dense 44.09; the
  paper has 70.17 and 46.00), only +2.7 / +4.6 above our baseline reproduction. Adding the camera stream
  (and the gating) accounts for the rest in this run: +5.9 in light fog and +23.8 in dense fog. One seed, one
  run each, fp32 on a new software stack: the size of the effect is large, the exact numbers are not
  established.
- **This does not yet show that the reliability gate matters.** M4 versus M5 changes two things at once: a
  camera stream and the gated fusion. A camera-without-gate ablation (naive fusion, same camera features, no
  reliability weighting) is needed to separate them. Not run.
- **No calibration gain to go with the AP gain.** ECE is about the same in clear and light fog (M5 lower by
  0.006 and 0.009, well inside what one seed can move) and **worse** in dense fog (0.150 against 0.108). Both
  models are under-confident (mean confidence below accuracy). Per `context.md` section 1, the AP gain in
  fog is not a thesis win by itself. (The lower dense-fog D-ECE for M4 comes with far fewer, low-score
  detections: 1,948 against 2,360.)
- **Not checked:** whether the camera stream is using cues specific to the STF dense-fog split rather than
  fog robustness in general. The test splits were never used for model selection.

Evidence: `run_logs/2026-11-14_M4_lidar_only_fp32_seed666/` (same contents as the M5 archive, with
`MANIFEST.sha256`).

---

## 2026-11-14: the baseline's ECE, measured for the first time

The baseline's three test-split `result.pkl` files (PV-RCNN, alpha = 0, evaluated 2026-10-28) were shared on
Google Drive. They are the baseline's: the 3D AP in their eval logs equals `baseline/README.md` on every split
(78.99 / 76.59 / 71.64, 68.07 / 68.26 / 66.49, 38.23 / 39.49 / 38.75) and the frame counts are 1,847 / 946 / 786.
ECE and D-ECE computed with the same tool, the same definition and the same ground truth (`kitti_infos_<split>.pkl`
from the `stf_kitti_mm` volume) as the Model 1 runs.

| Split | Baseline ECE | M4 ECE | M5 ECE | Baseline D-ECE 0-20 / 20-40 / 40-70.4 m | Baseline mean confidence / accuracy |
| --- | --- | --- | --- | --- | --- |
| Clear | 0.182 | 0.168 | 0.162 | 0.069 / 0.147 / 0.221 | 0.558 / 0.385 |
| Light fog | 0.200 | 0.180 | 0.171 | 0.100 / 0.173 / 0.237 | 0.503 / 0.309 |
| Dense fog | 0.263 | 0.108 | 0.150 | 0.204 / 0.268 / 0.280 | 0.426 / 0.163 |

(Detections kept: 11,930 / 5,728 / 3,984 for the baseline against 7,587 / 3,104 / 2,360 for M5.)

How to read this (hedged on purpose):

- **Model 1 (both M4 and M5) is better calibrated than the baseline, most of all in dense fog**: M5 is 0.020 lower in
  clear weather, 0.029 lower in light fog and 0.113 lower in dense fog; M4 is 0.155 lower in dense fog. This corrects
  what the M4 and M5 entries said (they had no baseline ECE to compare with).
- **The baseline is over-confident and gets worse in fog**: its mean confidence is far above its accuracy, 0.558
  against 0.385 in clear weather and 0.426 against 0.163 in dense fog (confidence about 2.6 times accuracy): the
  "confidently wrong" behaviour the project is about. Model 1 errs the other way (mean confidence below accuracy,
  slightly under-confident). The baseline is well calibrated at close range in clear weather (D-ECE 0.069 at 0-20 m)
  and bad far away, where weather bites.
- **The calibration gain is not from the camera.** The lidar-only M4 is as well calibrated as M5 (clear, light fog)
  or better (dense fog: 0.108 against 0.150), so the improvement over the baseline comes with the different trunk,
  head and loss, not with the camera stream or the gate. Cross-detector ECE comparisons are also affected by how many
  low-score detections each detector emits, so treat the size of the gap with care. One seed per model.
- **Still far from the thesis target.** Every ECE here is 0.11-0.26, against a target of 0.05 that is defined with
  two sensors degraded, which is not measured (needs the corruption engine).

Evidence: `run_logs/2026-11-14_baseline_ece/` (`PROVENANCE.txt` with the Drive links, the three `result.pkl` and logs,
the ECE json, `MANIFEST.sha256`).

---

## 2026-11-14 to 2026-11-15: M4b, camera without the reliability gate, fp32

**Result: completed, exit 0, no NaN. Single run, single seed.**

Setup: `cfgs/model1/model1_lidar_rgb_nogate.yaml` = the M5 config plus `FUSION.GATE_ENABLED: False` (the reliability
scores are still estimated, trained and logged, but enter the attention bias as r = 1; invalid-pixel masking,
auxiliary heads, reliability loss and modality dropout unchanged), `USE_AMP: False` by `sed` in the deployed copy,
seed 666, batch 4, 80 epochs, same container recipe as M5 and M4. Started 2026-11-14 10:33 local, training ended
about 16:55 (6 h 22 min), everything done 18:13. Best-on-val epoch **72**. Frames 1,847 / 946 / 786. Final epoch-average
training loss 2.11, the same as M5's.

Moderate Car 3D AP (R40, IoU 0.5), easy / hard in brackets for M4b:

| Split | M4 no camera | **M4b camera, no gate** | M5 camera + gate | Camera adds (M4b - M4) | Gate adds (M5 - M4b) |
| --- | --- | --- | --- | --- | --- |
| Clear | 74.57 | **77.55** (81.21 / 72.58) | 77.70 | +2.98 | +0.15 |
| Light fog | 70.97 | **76.32** (76.86 / 72.53) | 76.89 | +5.35 | +0.57 |
| Dense fog | 44.09 | **68.29** (68.47 / 65.69) | 67.85 | **+24.20** | -0.44 |

ECE and D-ECE (same definition as before):

| Split | M4 ECE | **M4b ECE** | M5 ECE | Camera adds (M4b - M4) | Gate adds (M5 - M4b) | M4b D-ECE 0-20 / 20-40 / 40-70.4 m | M4b mean confidence / accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Clear | 0.168 | **0.153** | 0.162 | -0.015 | +0.009 | 0.161 / 0.175 / 0.141 | 0.465 / 0.571 |
| Light fog | 0.180 | **0.159** | 0.171 | -0.021 | +0.012 | 0.177 / 0.177 / 0.143 | 0.456 / 0.552 |
| Dense fog | 0.108 | **0.137** | 0.150 | +0.029 | +0.013 | 0.108 / 0.145 / 0.140 | 0.351 / 0.358 |

(Lower ECE is better, so a positive "adds" is worse calibration.)

What the M4b estimators believe, per frame (it does **not** drive attention in M4b):

| Split | r_lidar median (std) | r_cam median (std; distinct values) | camera attention weight (M4b / M5) |
| --- | --- | --- | --- |
| Clear | 0.818 (0.068) | 0.837 (0.019; 33) | 0.498 / 0.521 |
| Light fog | 0.816 (0.187) | 0.837 (0.007; 24) | 0.509 / 0.547 |
| Dense fog | 0.427 (0.283) | 0.837 (0.018; 40) | 0.575 / 0.627 |

How to read this (hedged on purpose):

- **The camera stream, not the gate, accounts for the gain.** M4b scores essentially the same AP as M5 on every split
  (differences of +0.15, +0.57 and -0.44 AP, well inside what one seed can move), while adding the camera over the
  lidar-only M4 is worth +3 AP in clear weather, +5 in light fog and **+24 in dense fog**. In this run the reliability
  gate adds nothing measurable to detection accuracy beyond having the camera.
- **The gate does not help calibration either: it is slightly worse.** M4b is better calibrated than M5 on all three
  splits (by 0.009 to 0.013 ECE; small, one seed). The camera itself improves ECE in clear and light fog (-0.015,
  -0.021) and worsens it in dense fog (+0.029, where the lidar-only M4 was best at 0.108). So there is **no
  calibration gain from the gate, and the AP gain from the camera comes with a mixed calibration effect**.
- **The reliability estimators still respond to fog without being used.** In M4b, where r never reaches the
  attention, the lidar reliability still falls in dense fog (median 0.43 against 0.82 in clear weather; M5: 0.42
  against 0.80), because the estimator is trained on its auxiliary target regardless. The model can flag lidar
  degradation as a by-product, but nothing here shows that acting on it (gating) helps. Camera attention also rises
  in dense fog without the gate (0.498 to 0.575), so the attention shift is learned, not gate-driven.
- **The camera reliability estimator is constant in M4b as well** (median = max = 0.837, std 0.007-0.019), the
  same collapse seen in M5, so in neither run could the gate act on camera health.
- **What this does not show.** It does not show that a reliability gate cannot help. STF fog degrades the lidar but
  mostly leaves the camera usable, and the gate's purpose is to react to a sensor that fails: injected sensor failures
  (the corruption engine, not built) are the fair test, and the camera reliability estimator would have to be fixed
  first. One seed per model; fp32 on a new software stack.

Evidence: `run_logs/2026-11-14_M4b_nogate_fp32_seed666/` (same contents as the M4 and M5 archives, with
`MANIFEST.sha256`).

---

## 2026-11-14: sensor-failure probe v1 (exploratory): does the camera score react, and does the gate help when a sensor fails?

Why: in fog the camera score sat at 0.83 and the gate added nothing, but fog does not break the camera, so it could
not show whether the camera score works or whether a gate helps. This probe injects synthetic failures at evaluation
time (`tools/eval_corruptions.py`, through the `M1_EVAL_CORRUPTION` hook; 18 CPU tests) and re-evaluates the three
trained models' best checkpoints (M4 epoch 71, M4b 72, M5 63) with no retraining (`tools/run_corruption_probe.sh`,
`tools/summarize_corruption_probe.py`). **Clear val split (781 frames), not the test splits, so the test splits were not
used to explore; synthetic failures, not physical fog/icing; one seed per model.** The failures: `cam_zeros` (black
image), `cam_occlusion` (60% grey patch), `cam_dark` (brightness x 0.1), `cam_blur` (Gaussian sigma 25 px),
`lidar_sparse` (5% of points), `lidar_sector` (60% azimuth sector removed), `both_dark_sparse`. zeros/occlusion/sparse/
sector are the same kind of perturbation as the training-time dropout; dark and blur were never seen in training.

Moderate Car 3D AP and ECE (- = a camera failure cannot change the lidar-only M4):

| Condition | M4 AP | M4b AP | M5 AP | M4 ECE | M4b ECE | M5 ECE |
| --- | --- | --- | --- | --- | --- | --- |
| clean | 75.47 | 75.45 | 75.84 | 0.184 | 0.155 | 0.166 |
| cam_zeros | - | 75.06 | 75.05 | - | 0.165 | 0.164 |
| cam_occlusion | - | 74.78 | 75.50 | - | 0.166 | 0.169 |
| cam_dark | - | 70.31 | 73.09 | - | 0.190 | 0.199 |
| cam_blur | - | 47.37 | 44.22 | - | 0.126 | 0.197 |
| lidar_sparse | 16.80 | 44.63 | 39.97 | 0.159 | 0.126 | 0.128 |
| lidar_sector | 21.21 | 30.96 | 30.79 | 0.147 | 0.122 | 0.114 |
| both_dark_sparse | - | 40.72 | 35.78 | - | 0.123 | 0.124 |

What the reliability estimators believe (median r_lidar / median r_cam / mean camera attention weight; in M4b r does
not drive attention):

| Condition | M4b r_lidar | M4b r_cam | M4b cam weight | M5 r_lidar | M5 r_cam | M5 cam weight |
| --- | --- | --- | --- | --- | --- | --- |
| clean | 0.818 | 0.837 | 0.484 | 0.797 | 0.832 | 0.509 |
| cam_zeros | 0.818 | 0.014 | 0.142 | 0.797 | 0.000 | 0.012 |
| cam_occlusion | 0.818 | 0.106 | 0.241 | 0.797 | 0.124 | 0.143 |
| cam_dark | 0.818 | 0.837 | 0.460 | 0.797 | 0.832 | 0.495 |
| cam_blur | 0.818 | 0.837 | 0.433 | 0.797 | 0.832 | 0.419 |
| lidar_sparse | 0.061 | 0.837 | 0.628 | 0.000 | 0.832 | 0.890 |
| lidar_sector | 0.230 | 0.837 | 0.635 | 0.553 | 0.832 | 0.636 |
| both_dark_sparse | 0.061 | 0.837 | 0.603 | 0.000 | 0.832 | 0.888 |

How to read this (hedged on purpose):

- **This corrects the earlier statement that the camera reliability estimator is "constant / collapsed".** It sits at
  its clean-image ceiling (about 0.84), as the target construction implies (a normal image has an expected target of
  about 0.84), and it **does react** to the failures it was trained on: r_cam falls to about 0.0 for a dead camera and
  about 0.1 for a 60% occlusion. It is **blind** to a darkened or blurred camera (r_cam unchanged at 0.83-0.84), which
  are degradations it never saw, and real STF fog does not move it either. So in fog it could not have acted, and
  under dark or blur it cannot act.
- **The lidar estimator works for lidar failures**: r_lidar falls to 0.06 (M4b) / 0.00 (M5) for a 5%-sparse lidar and to
  0.23 / 0.55 for a removed sector, as it did in dense fog (0.43).
- **The camera rescues a failed lidar, which is the point of fusion**: with 5% of the lidar points the lidar-only M4
  collapses to 16.8 AP, while M4b and M5 hold 44.6 and 40.0; with a 60% sector gone, 21.2 against 31.0 and 30.8.
- **No advantage of the gate, and possibly a cost.** A dead or occluded camera costs M4b and M5 about the same (75.1
  and 75.1; 74.8 and 75.5). For a failed lidar the gate correctly drives r_lidar to about 0 and pushes the camera
  attention to 0.89, but M5 scores lower than M4b (39.97 against 44.63; with both sensors degraded 35.78 against
  40.72): setting r to about 0 excludes the lidar stream entirely (bias log 1e-4), although a 5%-sparse lidar still
  carries information that the ungated model uses. Cautions: these gaps (4.7 and 4.9 AP) are the size of the
  between-model variation seen elsewhere (the dark camera gives M5 +2.8 AP although r_cam does not move, so that gain
  is not a gate effect), so with one seed each this is **no evidence the gate helps and weak evidence it hurts**, not
  a measured harm.
- **Neither model is protected from a silent camera failure.** A blurred camera drops AP from 75 to 47 (M4b) and 44
  (M5), and r_cam does not notice: the "confidently wrong, silent about it" case the thesis targets, not solved by
  this gate. ECE does not rise there (0.126 and 0.197), so the failure is not visible in ECE either.
- **Calibration under failure** is similar with and without the gate (differences of 0.001 to 0.03, both directions),
  and nowhere near 0.05.
- **Implications to decide (protocol):** the camera estimator was trained only on generic dropout (zeros, noise,
  mask). Making it notice darkness or blur would mean adding such degradations to training; DESIGN.md section 4 says
  that needing shaped perturbations is a finding to report, not a silent change. A softer floor on r (so a distrusted
  stream is down-weighted and not removed) is a separate, cheap thing to try. Neither is done.

Evidence: `run_logs/2026-11-14_sensor_failure_probe_v1/` (probe logs, ECE json, reliability jsonl, `summary.md`,
`MANIFEST.sha256`).

---

## Status after these runs

M3, M4, M4b and M5 are complete as first runs (one seed each, fp32, mixed precision off as a recorded deviation), the
baseline's ECE is measured, and a first sensor-failure probe has been run. The ablation chain M4 (no camera) -> M4b
(camera, no gate) -> M5 (camera + gate) says, for this setup: the camera accounts for the fog gain, the gate adds
nothing measurable in fog or under single-sensor failures. Everything up to the M4b entry is committed on
`model1/m3-run-setup-and-log` (pushed); the probe and its tools are committed in the next push.

Not done, in rough order of value: decide whether the camera reliability estimator should be trained on darkness/blur
(a protocol decision) and try a softer floor on r; the corruption engine proper (physical fog, snow, icing,
occlusion, calibration drift, dropout; evaluated on the test splits), which is the only way to measure ECE with two
modalities degraded and the other thesis metrics (lead time, worst-decile AP, recovery time); more seeds (the
between-model variation is of the order of 3 AP in the corrupted regime); an fp16-safe fix of the auxiliary heads; M6
(writeup).
