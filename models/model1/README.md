# Model 1: reliability-gated multimodal transformer

Fuses lidar (HDL-64 strongest) and the left RGB camera. Attention over modality
tokens is modulated by a learned per-stream reliability score. Built on OpenPCDet
as plug-in modules; OpenPCDet source and `../../baseline/` are not modified.

Full design and metric definitions: `docs/DESIGN.md`.

## Status

| Milestone | Work | Status |
| --- | --- | --- |
| M0 | Folder scaffold, README, DESIGN.md | done |
| M1 | Camera data: extract, demosaic, rectify and tone-map left-camera images for all 7,829 split frames; verify | done (`sarvesh/model1-data`): 7,829/7,829 images; alignment median (0, 0) px; all checks pass |
| M2 | Docker image, dataset config, model code, tools, tests | done, merged (`model1/integrate`). Re-verified 2026-11-13 on an RTX 5090: image rebuilt for sm_120, 20 / 20 tests pass |
| M1b | Real images rebuilt on the 5090 box from the raw `cam_stereo_left` archive; Docker volume and infos built | done 2026-11-13: 7,829 / 7,829 PNGs, `03_verify_images.py` ALL CHECKS PASSED, median shift (0, 0) px (`docs/RUN_LOG.md`) |
| M3 | 200-frame slice: overfit test + epoch timing | done 2026-11-13. The committed config (`USE_AMP: True`) goes NaN; with `USE_AMP: False` training is finite. One epoch = 4 min 51 s. fp16-safe fix not applied: M4/M5 ran in fp32, a recorded deviation (`docs/RUN_LOG.md`) |
| M4 | Lidar-only ablation, full training | done 2026-11-14 (one seed, fp32): moderate AP 74.57 / 70.97 / 44.09 (clear / light / dense fog) |
| M5 | Full Model 1: train, best-on-val, three test splits; AP + ECE + D-ECE + reliability | done 2026-11-14 as a first run (one seed, fp32): 77.70 / 76.89 / 67.85; ECE 0.162 / 0.171 / 0.150 |
| M4b | Ablation: camera without the reliability gate (`configs/model1_lidar_rgb_nogate.yaml`) | **running** (started 2026-11-14 10:33; about 7 h by estimate) |
| M6 | Writeup against the baseline | todo, after the baseline gate passes |

## Results so far

![Model 1 results overview](docs/figures/model1_results_overview.png)

Regenerate with `tools/make_results_figure.py` (reads the archived run logs). Moderate Car 3D AP (R40, IoU 0.5) and
ECE on the three STF test splits; one seed per model, fp32:

| Split | Baseline (PV-RCNN) AP | M4 lidar-only AP | M5 lidar + RGB AP | Baseline ECE | M4 ECE | M5 ECE |
| --- | --- | --- | --- | --- | --- | --- |
| Clear | 76.59 | 74.57 | 77.70 | 0.182 | 0.168 | 0.162 |
| Light fog | 68.26 | 70.97 | 76.89 | 0.200 | 0.180 | 0.171 |
| Dense fog | 39.49 | 44.09 | 67.85 | 0.263 | 0.108 | 0.150 |

What this does and does not show (details and caveats in `docs/RUN_LOG.md`):

- **AP:** with the same trunk, head, schedule and seed, the camera-off model (M4) behaves like the baseline in fog, so
  most of M5's fog gain comes with the camera stream and gated fusion, not the lidar trunk.
- **Calibration:** both Model 1 variants are better calibrated than the baseline, most of all in dense fog (0.263 down
  to 0.108-0.150). The baseline is over-confident (dense fog: confidence 0.43 for accuracy 0.16), Model 1 slightly
  under-confident. But the lidar-only model is as well calibrated as M5, so this comes with the trunk and loss, not the
  camera or the gate. Every ECE is still far from the 0.05 target, which is defined with two sensors degraded and is
  not measured yet.
- It does **not** show that the reliability gate matters (M4b, running, tests that).
- The lidar reliability score falls in fog the model never saw (median 0.80 to 0.42), but the camera reliability
  estimator is effectively constant (0.83), so it is not tracking camera health.

## Data

Phase 1 (the baseline) deliberately downloaded only part of the camera data
(`CLAUDE.md`, `context.md` §7) and uses blank placeholder images. Model 1 reads
pixels, so it needs the full `cam_stereo_left` archive
(`../../baseline/docs/REPRODUCTION_PLAN.md` §6: "Real images are needed again in Phase 2").
The 7 skipped parts (z02, z03, z06, z08, z11, z16, z19) were downloaded with
`download_stf_subset.py` on 2026-10-28. Right camera, `lidar_hdl64_last` and
`labeltool_labels_refined.zip` stay skipped (not used by Model 1 v1).

## Run order

```
# 1. Baseline conversion (lidar, labels, calib), see CLAUDE.md
# 2. Camera images (data_prep/, numpy + pillow; e.g. inside the image)
#    If data/stf/calib_cam_stereo_left.json is missing (it is not in the camera archive on its own),
#    recreate it from the committed tonemap_stats.json first:
python models/model1/data_prep/00_make_calib_from_stats.py
python models/model1/data_prep/01_extract_camera.py
python models/model1/data_prep/02_build_image_2.py build      # uses committed tonemap_stats.json
python models/model1/data_prep/03_verify_images.py
# 3. Docker: images, stf_kitti_mm volume, infos, then the tests
powershell -File models/model1/docker/run_model1_docker.ps1
# 4. Train + evaluate (80 epochs; MODEL1_CFG=cfgs/model1/model1_lidar_only.yaml for the ablation)
powershell -File models/model1/docker/run_model1_docker.ps1 -Detach -Cmd "bash /workspace/models/model1/tools/train_and_eval.sh"
```

Best-on-val selection and the AP read-out use `baseline/scripts/07_read_eval_logs.py`
(Rohil's R3, merged).

**Keep the logs.** Every training or evaluation run is evidence that it was run. Raw logs stay out of git
(`models/*/output/` is gitignored) but must be kept: copy them to `run_logs/<date>_<name>/` beside the repo
checkout with a `MANIFEST.sha256`, back them up (Drive), and add a dated entry to `docs/RUN_LOG.md`
(command, config, outcome, where the raw log is). Keep failed runs too.

To debug a run that goes non-finite, `tools/debug_train_hook.py` runs the real `train.py` and prints the loss,
auxiliary losses and gradient norm per iteration. Do not rely on `--set OPTIMIZATION.USE_AMP False`: `train.py`
reads `USE_AMP` before it applies `--set`, so edit the yaml instead.

## Folder layout

```
configs/     dataset config (data branch) + model configs: lidar_rgb, lidar_only
data_prep/   camera extraction, fixed tone map, image checks
rgfusion/    model code, registered into OpenPCDet at import time (register.py)
tools/       OpenPCDet entry point, train_and_eval.sh, ECE, reliability summary
tests/       unit + GPU end-to-end tests (20)
docker/      image built FROM the baseline's pinned image, runner script
docs/        DESIGN.md
```

Compute figures in this folder are estimates until M3 measures an epoch.
