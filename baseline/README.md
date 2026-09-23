# Baseline reproduction — Hahner et al. ICCV 2021, PV-RCNN † (α=0)

Groundwork for reproducing Table 2's clear/moderate = **77.05** Car 3D AP@0.5 IoU
(gate: 75.05–79.05, see `../context.md` section 3). This machine is an Apple
Silicon Mac with no CUDA/spconv, so **everything here is CPU-side data
preparation** — the actual training/eval run needs a CUDA GPU box (context.md
section 8: ~150–250 GPU-hours estimated, Docker image with pinned CUDA/PyTorch
"planned", not built yet).

Read `docs/REPRODUCTION_PLAN.md` first — it has the full rationale for every
design decision below (why these exact split files, why the class remap, why
`USE_ROAD_PLANE: False`, etc). This README is just the run order.

## What's done (this machine, CPU-only) — complete, all checks passing

1. **Split files resolved** (`splits/*.txt`) — sourced from `MartinHahner/SeeingThroughFog`
   (branch `main`), the only split source found that reproduces context.md's
   exact counts (3469/781/1847/946/786). See plan doc §1 for the discrepancy
   with the upstream devkit's day/night split files (a superset, wrong counts).
2. **Data extracted** (`scripts/01_extract_archives.py`) — lidar
   (`lidar_hdl64_strongest`, strongest returns, 5→4 field truncation deferred to
   step 2) + labels (`gt_labels`) + calibration, filtered to exactly the 7829
   frames the five splits reference. Ran clean: 7829/7829 frames, 0 missing.
3. **Converted to KITTI layout** (`scripts/02_convert_to_kitti.py`) —
   `data/kitti_format/training/{image_2,velodyne,calib,label_2}` +
   `ImageSets/*.txt` (all five splits under `training/`, see plan doc §2 for
   why there is no `testing/` directory — a deliberate workaround for a
   hard-coded OpenPCDet path check, not an oversight). Includes a class remap
   (`PassengerCar→Car`, `RidableVehicle→Cyclist`), a from-scratch calibration
   derivation (STF's ROS-tf-tree JSON → KITTI's flat `calib.txt`), and image
   tone-mapping (STF's raw 16-bit tiffs → 8-bit PNG). Ran clean: all 7829
   frames converted, 0 missing lidar/label/image files, 77565 label objects
   (7423 remapped to DontCare — non-vocabulary STF classes, see plan doc §3).
4. **Calibration/label correctness verified**, two ways:
   - Visually: `data/sanity_check/000000_final_check.png` — every projected 3D
     box lands tightly on the real object in the camera image.
   - Programmatically, on the full converted dataset:
     `scripts/05_verify_conversion.py` — **all 6 checks pass**, including a
     fresh 20-frame random projection spot-check (131/131 = 100% of
     in-front-of-camera boxes project into image bounds).
   This was the single highest risk item (plan doc risk #4) and it passed
   cleanly.
5. **OpenPCDet configs prepared** (`configs/kitti_dataset_stf.yaml`,
   `configs/pv_rcnn_stf.yaml`) — stock `kitti_dataset.yaml` / `pv_rcnn.yaml`
   with only the minimal changes needed to point at our data (see each file's
   header comment for the itemized diff from stock).

Disk usage: `data/kitti_format` (final product) is ~21GB. `data/kitti_source`
(8.5GB, pre-conversion staging) and `data/_merged_zips` (37GB, the merged
cam_stereo_left archive used only for image extraction) are both safe to
delete once you've copied `kitti_format` to the GPU box — kept for now since
disk isn't tight (378GB free as of this writing) and re-generating
`_merged_zips` in particular is the slowest step to redo (~20+ min).

## What's NOT done (needs the GPU box)

- Running `scripts/03_create_infos.py` (needs `pcdet` installed, which needs
  spconv/CUDA) to generate the five `.pkl` info files + ground-truth database.
- Actual training (`scripts/04_train_and_eval.sh`) — 80 epochs, PV-RCNN,
  standard OpenPCDet policy, matching the paper's stated protocol.
- Checkpoint selection by best-on-clear-val performance (not last-epoch) —
  the paper's own stated selection rule, encoded as `--eval_all` in step 4.
- Evaluation on all three STF test splits (clear/light-fog/dense-fog) and the
  77.05 AP gate check itself.

That's it — everything else is done and verified. `03_create_infos.py` is the
next actual step, and it needs `pcdet` importable, which is the first thing
that needs CUDA/spconv.

## Run order on the GPU box

```bash
# 1. Verify the converted data one more time in the target environment
cd baseline/scripts
python 05_verify_conversion.py --kitti-format ../data/kitti_format

# 2. Deploy configs into an OpenPCDet checkout (see 04_train_and_eval.sh's
#    header comment for the exact required paths -- OpenPCDet resolves
#    _BASE_CONFIG_ relative to tools/, so placement matters)
cp ../configs/pv_rcnn_stf.yaml          $OPENPCDET_ROOT/tools/cfgs/kitti_models/
cp ../configs/kitti_dataset_stf.yaml    $OPENPCDET_ROOT/tools/cfgs/dataset_configs/
cp -r ../data/kitti_format              $OPENPCDET_ROOT/data/kitti_format
# ^ copy, not symlink, and onto local NVMe -- context.md section 8: don't train
#   off a mounted/network drive.

# 3. Generate info files + GT database (inside the OpenPCDet env)
python 03_create_infos.py --cfg ../configs/kitti_dataset_stf.yaml \
    --data-path $OPENPCDET_ROOT/data/kitti_format \
    --openpcdet-root $OPENPCDET_ROOT

# 4. Train + evaluate
OPENPCDET_ROOT=$OPENPCDET_ROOT ./04_train_and_eval.sh
```

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
