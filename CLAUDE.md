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

## Current state (update this)

_Last updated: 2026-10-28_

- `main`: has the task split (`BASELINE_TASKS.md`) and this file.
- `dipin/baseline-data-checks`: D1–D5, D7 and D8 done and run on the real data. All 8
  conversion checks pass. Waiting to merge.
- `rohil/baseline-eval-fixes`: R1–R7 in progress (eval-split fix, checkpoint selection,
  Docker pinning, README).
- Next: merge both branches, build the Docker image, generate infos, then start the
  full 80-epoch training run from scratch.
