# Model 1: reliability-gated multimodal transformer

Fuses lidar (HDL-64 strongest) and the left RGB camera. Attention over modality
tokens is modulated by a learned per-stream reliability score. Built on OpenPCDet
as plug-in modules; OpenPCDet source and `../../baseline/` are not modified.

Full design and metric definitions: `docs/DESIGN.md`.

## Status

| Milestone | Work | Status |
| --- | --- | --- |
| M0 | Folder scaffold, README, DESIGN.md | done (this commit) |
| M1 | Camera data: unpack Drive zips, extract left-camera images for all 7,829 split frames, fixed global tone map, verify | blocked: needs the deferred camera parts (below) |
| M2 | Docker image, register modules, info files for `kitti_format_mm` | todo |
| M3 | 200-frame slice: overfit test + epoch timing | todo |
| M4 | Lidar-only ablation, full training | todo |
| M5 | Full Model 1: train, best-on-val, three test splits; AP + ECE + D-ECE + reliability | todo |
| M6 | Writeup against the baseline | todo, after the baseline gate passes |

## Data needed (and why it isn't all here yet)

Phase 1 (the baseline) deliberately downloaded only part of the camera data
(`CLAUDE.md`, `context.md` §7). The baseline uses blank placeholder images.
Model 1 reads pixels, so it needs the full `cam_stereo_left` archive
(`../../baseline/docs/REPRODUCTION_PLAN.md` §6: "Real images are needed again in Phase 2").

The skipped parts are inside the 13 Google Drive export zips not yet downloaded
(numbers 004, 017, 018, 021, 028, 035, 049, 054, 058–061, 063; ~1.5 GB each):

| STF archive parts | Needed for Model 1 v1 |
| --- | --- |
| `cam_stereo_left` z02, z03, z06, z08, z11, z16, z19 | **yes** |
| `cam_stereo_right` z09, z14, z15, z17, z22 | no |
| `lidar_hdl64_last` z04 | no |

Drive doesn't label which zip holds which part, so all 13 have to be downloaded.

## Folder layout

```
configs/     OpenPCDet dataset + model configs (M2)
data_prep/   camera extraction, tone map, image checks (M1)
rgfusion/    model code, registered into OpenPCDet at import time (M2+)
tools/       train/test wrappers, per-split eval, ECE, reliability dump (M2+)
tests/       unit tests: gate, projection under augmentation, dropout, ECE (M2+)
docker/      image built FROM the baseline's pinned image (M2)
docs/        DESIGN.md
```

Compute figures in this folder are estimates until M3 measures an epoch.
