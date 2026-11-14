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
| M3 | 200-frame slice: overfit test + epoch timing | **in progress, not complete.** Loss goes NaN under the committed config (`USE_AMP: True`); finite with `USE_AMP: False`. Fix not applied yet (`docs/RUN_LOG.md`) |
| M4 | Lidar-only ablation, full training | todo |
| M5 | Full Model 1: train, best-on-val, three test splits; AP + ECE + D-ECE + reliability | todo |
| M6 | Writeup against the baseline | todo, after the baseline gate passes |

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
