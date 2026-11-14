# Agent guide: read this every session

This repo is a two-person capstone (Dipin and Rohil). Claude Code loads this file
automatically. Keep it short and current: **update the "Current state" section
whenever a task lands, a branch merges, or a run starts or finishes.**

## Read first, in this order

1. `context.md`: project thesis, protocol invariants (§6) and working rules (§10).
   These override everything else, including this file.
2. `BASELINE_TASKS.md`: who owns which task and file, and the status table.
3. `baseline/README.md`: how to run the baseline.

## Ground rules

- The goal right now is to **reproduce the baseline**: PV-RCNN † (α=0), Hahner et al.
  ICCV 2021, trained from scratch on STF clear weather. Gate: Car 3D AP, R40, IoU 0.5,
  clear/moderate = **77.05 ± 2**.
- Never tune hyperparameters toward 77.05. If AP is off, check protocol first
  (`context.md` §3).
- Work on your own branch (`dipin/...` or `rohil/...`) and stay inside your file
  ownership in `BASELINE_TASKS.md` §2. Never push to `main` directly unless the
  person you're working for explicitly says so.
- Commit and push incrementally: one task per commit, with the task ID in the message.
- Never commit data, `.pkl`, checkpoints or logs.
- Report failures with the actual output. Don't call something done until it has run.

## Machine notes (Dipin's Windows box: RTX 4090, Docker Desktop)

- git is not on PATH: `C:\Users\dipin\AppData\Local\Programs\Git\cmd\git.exe`
- Python (CPU-side scripts 01, 02, 05, 06): `C:\Users\dipin\Desktop\298a\.python\pkg\tools\python.exe`
  (portable 3.12 with numpy, scipy, pillow and matplotlib, outside the repo)
- Raw STF data: `data/stf/` (gitignored). Phase 1 deliberately has only part of the
  camera data; see `context.md` §7.
- Converted data: `baseline/data/kitti_format/` (gitignored, ~21 GB, placeholder images)
- Training runs in Docker (`baseline/docker/`), with GPU passthrough confirmed working.

## Machine notes (Anushka's Windows box: RTX 5090, Docker Desktop)

- git is not installed system-wide: portable MinGit at `%LOCALAPPDATA%\MinGit\cmd\git.exe`.
- Python: standalone 3.12 at `%LOCALAPPDATA%\Python312\python.exe` (stdlib + `gdown`; no numpy/opencv: run the
  image-building scripts inside the Docker image).
- **Blackwell (sm_120): the Drive tar `stf-openpcdet-pinned.tar` does not run here** (torch 2.0.1 / CUDA 11.8, built on a
  4090). Build from `baseline/docker/Dockerfile` (CUDA 12.8, torch 2.7.1). The baseline numbers came from the 4090 image.
- `run_model1_docker.ps1` is run with Windows PowerShell 5.1 here (fixed to work under it).
- Data lives outside git under the repo's `data/` and `baseline/data/` (both gitignored). Logs of runs go to
  `run_logs/` next to the repo checkout; see `models/model1/README.md` ("Keep the logs").

## Rebuilding `baseline/data/kitti_format/` on another machine (~21 GB)

It is gitignored, so each machine builds its own from the raw data. The output is
identical, because frame IDs come from the split files in git. Needs Python 3.10+
with `numpy scipy pillow matplotlib`, and only these raw files under `data/stf/`
(~9 GB, not the full download):

- `lidar_hdl64_strongest/` (every `.z01`…`.z06` part plus `.zip`)
- `gt_labels/cam_left_labels_TMP.zip`, `gt_labels/gated_labels_TMPv2.zip`
- `calib_cam_stereo_left.json`, `calib_cam_stereo_right.json`, `calib_gated_bwv.json`, `calib_tf_tree_full.json`

```
cd baseline/scripts
python 01_extract_archives.py                      # -> baseline/data/kitti_source (~9 GB)
python 02_convert_to_kitti.py --placeholder-images # -> baseline/data/kitti_format (~21 GB)
python 05_verify_conversion.py                     # must print ALL CHECKS PASSED
```

Expect: 7829/7829 frames, "11,772 Car rows" remapped to DontCare in total across
splits, 7829 placeholder images. Do not drop `--placeholder-images`: Phase 1 has no
complete camera archive. `baseline/data/kitti_source/` can be deleted afterwards.

## Current state (update this)

_Last updated: 2026-11-13_

- **Model 1 (Sarvesh and Anushka):** code, data pipeline and 20 tests are merged and pass on the 5090. Real camera
  images rebuilt (7,829, verified). M3 (200-frame trial) is **in progress**: the loss goes NaN with the committed
  `USE_AMP: True`, and is finite with `USE_AMP: False`. Details and raw-log locations: `models/model1/docs/RUN_LOG.md`.
  Not yet trained on the full set; no Model 1 result exists yet.

- Both branches are merged into `main`. All D and R tasks are done except D6 (Phase 2 prep).
- Docker image `stf-openpcdet:pinned` is built on Dipin's machine, and data volume
  `stf_kitti` holds kitti_format plus infos.
- 1-epoch smoke test passed end to end (tag `smoke`): train → best-on-val → three
  test evals on the correct frame counts → gate report. 1-epoch AP is meaningless
  (clear/mod 61.62). It only proves the pipeline works.
- Measured: 6 min 56 s per epoch on one RTX 4090, so 80 epochs ≈ 9–10 h plus ≈ 2 h val evals.
- **Baseline reproduced: gate PASS.** Full from-scratch run, clear/moderate 76.59
  (band 75.05–79.05), best-on-val epoch 67. Table in `baseline/README.md`. Logs are in
  `baseline/output/kitti_models/pv_rcnn_stf/stf_baseline/baseline_logs/` on Dipin's
  machine (gitignored). Training actually took 8 h 44 m.
- Open: dense fog is 6.5 AP below the paper (Linear 298-61). Check protocol, don't tune.
- Next phase: Model 1 (`context.md` §5, §7), which needs the camera parts listed below.
- Task tracking is in Linear (team "298A - Team Rocks- Project"; baseline tasks
  298-47 to 298-62). Keep Linear in sync when a task lands.
- Commits for Rohil's tasks made on his behalf use `--author="rohilutture <rohil.utture4@gmail.com>"`
  (he agreed). Dipin's own work stays under Dipin.
- Phase 2 (model 1) needs the full `cam_stereo_left` archive. The Phase 1 download
  deliberately skipped parts `.z10`, `.z17`, `.z22` and the final `.zip`.
