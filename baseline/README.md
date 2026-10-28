# Baseline reproduction — Hahner et al. ICCV 2021, PV-RCNN † (α=0)

Reproduces Table 2's clear/moderate = **77.05** Car 3D AP@0.5 IoU, R40
(gate: 75.05–79.05, see `../context.md` section 3), trained **from scratch** on
STF clear weather only.

Read `docs/REPRODUCTION_PLAN.md` for the rationale behind every design decision
(split source, class remap, DontCare handling, `USE_ROAD_PLANE: False`,
placeholder images, and so on). This README is the status and run order.

## Status

Done and verified on the full dataset (CPU side, `scripts/01`, `02`, `05`, `06`):

- Split counts match context.md exactly (3469/781/1847/946/786); 7829/7829 frames
  converted, with no missing lidar, label, calib or image files.
- Objects with no 3D box become DontCare (plan doc §3). All 8 checks in
  `05_verify_conversion.py` pass, including the lidar→camera extrinsic (points
  inside GT boxes) and projected-vs-labelled 2D box IoU (median 0.89).
- Images are shape-only placeholders (plan doc §6). PV-RCNN is lidar-only.

GPU side (`scripts/03`, `04`, `07`, `run_baseline_docker.ps1`, `docker/`):

- Docker image pinned to an OpenPCDet commit and package versions (`docker/`).
- `04_train_and_eval.sh` trains, keeps all 80 checkpoints, selects best-on-val
  automatically (`07_read_eval_logs.py`), evaluates each test split on its
  **own** info file, and prints the gate verdict.
- **Reproduced: gate PASS.** From-scratch 80-epoch run, best-on-val checkpoint
  epoch 67. Car 3D AP, R40, IoU 0.5:

  | Split | Easy | Moderate | Hard | Paper (E / M / H) |
  | --- | --- | --- | --- | --- |
  | Clear | 78.99 | **76.59** | 71.64 | 79.61 / 77.05 / 71.03 |
  | Light fog | 68.07 | 68.26 | 66.49 | 69.55 / 70.17 / 68.44 |
  | Dense fog | 38.23 | 39.49 | 38.75 | 45.03 / 46.00 / 45.08 |

  Clear/moderate 76.59 is inside 75.05–79.05. The moderate > easy inversion
  reproduces on both fog splits. Dense fog is 6.5 AP below the paper and is under
  investigation (Linear 298-61), not tuned toward. Single seed: run-to-run spread is
  not yet measured. Measured cost on one RTX 4090: 8 h 44 m training plus ~45 min
  of val evals.

## Run order (Windows + Docker Desktop + NVIDIA GPU)

From the repo root, after `01`/`02` have produced `baseline/data/kitti_format/`:

```powershell
# optional: 1-epoch smoke test of the whole pipeline (tag 'smoke', ~30-60 min)
powershell -File baseline/scripts/run_baseline_docker.ps1 -Smoke -Foreground

# the real run: builds the image and data volume if missing, generates infos,
# then trains 80 epochs + selects + evaluates in a detached container
powershell -File baseline/scripts/run_baseline_docker.ps1
docker logs -f stf-baseline
```

Results: `baseline/output/kitti_models/pv_rcnn_stf/stf_baseline/baseline_logs/`
(`train.log`, `eval_all_val.log`, `eval_test_*.log`). Re-print the gate at any time:

```powershell
python baseline/scripts/07_read_eval_logs.py gate --logs (Get-ChildItem baseline/output/kitti_models/pv_rcnn_stf/stf_baseline/baseline_logs/eval_test_*.log)
```

On a Linux GPU box without the PowerShell wrapper, the same steps are: deploy the
two configs into `<openpcdet>/tools/cfgs/{kitti_models,dataset_configs}/`, copy
`data/kitti_format` onto local disk at `<openpcdet>/data/kitti_format`, run
`03_create_infos.py`, then `OPENPCDET_ROOT=<openpcdet> ./04_train_and_eval.sh`.

**Reproducibility note (R6).** `--fix_random_seed` fixes OpenPCDet's seeds, but
cuDNN and spconv kernels are not bitwise deterministic, so two identical runs can
differ by a fraction of an AP point. Put that spread next to the ±2 gate when
reporting. Only spend GPU hours on a second seed if the result lands near a band
edge.
## Gate check

After step 4, read the Car AP@0.5, moderate column, from `eval_test_clear.log`.
Target: **77.05 ± 2** (75.05–79.05). If it's off, check protocol before touching
hyperparameters — context.md section 3 lists the likely causes in order (R11 vs
R40 first). **Never tune toward 77.05.**

**R11 vs R40 and IoU 0.5 vs 0.7, concretely** (both of context.md's #1 and #2
likely-mismatch causes are the same OpenPCDet quirk): `get_official_eval_result`
(in `pcdet/datasets/kitti/kitti_object_eval_python/eval.py`) hard-codes *two*
IoU-threshold rows per class (0.7 and 0.5 for Car) and computes *both* R11 and
R40 for each, so the printed log has **four** blocks for Car, not one:

```
Car AP@0.70, 0.70, 0.70:      <- R11, IoU 0.7 (KITTI's stricter official standard -- NOT this one)
Car AP_R40@0.70, 0.70, 0.70:  <- R40, IoU 0.7 (NOT this one either)
Car AP@0.70, 0.50, 0.50:      <- R11, IoU 0.5 (closer, but still not R40)
Car AP_R40@0.70, 0.50, 0.50:  <- R40, IoU 0.5 -- THIS is the block to read
  bbox AP: ...
  bev  AP: ...
  3d   AP: <easy>, <moderate>, <hard>   <- moderate value is the 77.05 gate number
```

The header's *first* number (0.70) is always the 2D-bbox IoU threshold and does
not change between the two rows — it's the **second** number (bev/3D IoU) that
distinguishes the 0.7-row from the 0.5-row for Car. Do not be misled by the
"0.70" appearing in the header of the row you actually want.

Separately: the *programmatic* `ret_dict` this function returns (what a
tensorboard/summary integration would log) only populates the **IoU-0.7**
numbers by default (`if i == 0` branch, and `i=0` is the `overlap_0_7` array) —
so if you ever read results from tensorboard/wandb instead of the raw text log,
you will silently get the wrong IoU threshold. **Always parse the text log for
this reproduction, specifically the `AP_R40@0.70, 0.50, 0.50:` block's `3d AP`
line**, not any structured/logged dict.

Also worth reproducing as a correctness signal, not just the headline number:
the moderate > easy inversion on both fog splits (context.md section 3 calls
this out explicitly — "positive evidence the protocol matches" if it
reproduces, a red flag if it doesn't).
