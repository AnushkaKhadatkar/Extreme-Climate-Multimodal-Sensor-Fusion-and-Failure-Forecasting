# Baseline reproduction plan — Hahner et al. ICCV 2021, PV-RCNN † (α=0)

Target: Table 2, `PV-RCNN †`, α=0 row. Car 3D AP@0.5 IoU, clear/moderate = **77.05**
(gate: 75.05–79.05). See `../../context.md` §3 for why this exact row and not the
fog-augmented rows.

This doc is the reference for anyone (human or agent) picking this up later — it
records *why* each step is the way it is, not just what to run, because the source
data and tooling are scattered across three repos and none of them target OpenPCDet
directly.

## 0. Why this is harder than "point OpenPCDet at KITTI-format data"

context.md says "Labels are in KITTI format, which is why OpenPCDet detectors run
on this directly." True for the *label schema* (mostly), false for turnkey use:

- STF ships labels as 27-field extended-KITTI txt (`gt_labels/cam_left_labels_TMP`),
  not the KITTI directory layout OpenPCDet's `kitti_dataset.py` expects
  (`training/{image_2,velodyne,calib,label_2}` + `ImageSets/*.txt`).
- STF class names are `PassengerCar` / `RidableVehicle` / `Pedestrian` / ... —
  OpenPCDet's KITTI class filter expects `Car` / `Cyclist` / `Pedestrian`.
- STF lidar `.bin` files are `(N, 5)` float32 (x, y, z, intensity, channel/ring).
  OpenPCDet's stock KITTI loader hard-codes `(N, 4)` (x, y, z, intensity). The 5th
  field must be dropped when we write our own `.bin` — this is a data-conversion
  decision, not a change to OpenPCDet code, so it does not violate "do not modify
  the baseline."
- STF calibration is ROS-style JSON (`calib_cam_stereo_left.json` +
  `calib_tf_tree_full.json`, quaternion extrinsics) — not KITTI's flat
  `P0..P3 / R0_rect / Tr_velo_to_cam` text format. Must be converted per-frame.
- The paper's own code (`www.trace.ethz.ch/lidar_fog_simulation`, mirrored in
  `github.com/MartinHahner/LiDAR_fog_sim`) is the *fog simulator only*. The actual
  detector-training harness lives in a private OpenPCDet fork not published
  alongside the paper — no `dense_dataset.py` is public. So: convert STF into
  vanilla KITTI layout and run OpenPCDet's **unmodified** `kitti_dataset.py` +
  `pv_rcnn.yaml`. This is the safer, more auditable choice anyway (context.md:
  "Use released configs verbatim for the baseline").

## 1. Split files — resolved

context.md §2 gives exact counts to hit. Three candidate split sources were checked:

| Source | train | val | test(clear) | light fog | dense fog |
|---|---:|---:|---:|---:|---:|
| context.md target | 3469 | 781 | 1847 | 946 | 786 |
| `princeton-computational-imaging/SeeingThroughFog` (day+night summed) | 3526 | 808 | 1882 | 1052 | 887 |
| **`MartinHahner/SeeingThroughFog` (branch `main`, combined files)** | **3469** | **781** | **1847** | **946** | **786** |

`MartinHahner/SeeingThroughFog` is Hahner's own fork of the official devkit (referenced
via `LiDAR_fog_sim`'s `.gitmodules`) and its `splits/{train_clear,val_clear,test_clear,
test_light_fog,test_dense_fog}.txt` match context.md exactly, to the frame. This is the
authoritative split source for this reproduction — **use these files, not the upstream
princeton devkit's day/night split files**, which are a superset (some frames excluded,
reason not stated in either repo — likely a data-quality filter, e.g. `cvl/filter_missing_vlp_frames.py`
/ `remove_empty_dense_scenes.py` present in the same fork).

Mirrored locally at `baseline/splits/` (fetched via raw.githubusercontent.com on
2026-09-22). Each line is `sequence_timestamp,frame_id`, e.g.
`2019-09-11_18-41-27,02580`. This pairs with filenames in `gt_labels` as
`{sequence_timestamp}_{frame_id}.txt`, e.g. `2019-09-11_18-41-27_02580.txt`.

**Sensor**: HDL-64E only (these split files are the base ones, not the `_vlp32`
variants) — matches the paper ("Velodyne HDL-64E as their main LiDAR sensor").

## 2. Data layout decision

Convert STF → standard KITTI dataset layout under
`baseline/data/kitti_format/training/{image_2,velodyne,calib,label_2}` +
`ImageSets/{train,val,test_clear,test_light_fog,test_dense_fog}.txt`, driven 1:1
by the split files above.

**Everything lives under `training/`, including the three STF test splits —
there is no `testing/` directory.** This looks wrong against plain KITTI
convention (which splits `training/` vs `testing/`) but is required by
OpenPCDet's own `KittiDataset` source:

```python
self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')
```

That check is on the literal string `'test'`. Our test splits are named
`test_clear` / `test_light_fog` / `test_dense_fog` (not bare `test`) because
context.md needs them evaluated and reported as three independent columns,
matching Table 2 — so under OpenPCDet's own hard-coded logic they would
*already* resolve to `training/` regardless of what we intended, and any files
placed in `testing/` would simply never be read. Rather than fight that (which
would mean patching OpenPCDet source — a real violation of "do not modify the
baseline"), the data layout follows what the loader actually does: one flat
`training/` pool, five named `ImageSets/*.txt` files selecting disjoint frame-ID
ranges out of it. KITTI IDs are globally unique across all five splits (assigned
sequentially at conversion time, verified no collisions), so nothing conflicts.

Frame IDs use the pattern `{sequence}_{timestamp}` sanitized to a KITTI-style
zero-padded numeric ID via a generated mapping table
(`baseline/data/kitti_format/ImageSets/id_mapping.csv`) — KITTI filenames must be
numeric, STF's aren't.

## 3. Label conversion

- Source: `gt_labels/cam_left_labels_TMP/*.txt` (27 fields, see
  `read.py::get_kitti_object_list` from the official devkit for the schema).
- Class remap for the Car AP number (the only one this gate needs):
  `PassengerCar → Car`. (Other classes kept/mapped too — `RidableVehicle → Cyclist`,
  `Pedestrian → Pedestrian` — since `pv_rcnn.yaml` trains all three jointly even
  though we only gate on Car.)
- Fields 0–14 map directly to KITTI's 15-field label line (type, truncated,
  occluded, alpha, bbox x1y1x2y2, dimensions hwl, location xyz, ry). STF's fields
  15–22 (Euler angles + quaternion) and 23–26 (per-sensor visibility flags) are
  dropped — not part of KITTI's label schema and not consumed by
  OpenPCDet's KITTI loader.
- `DontCare` rows pass through unchanged (KITTI convention: bbox valid, 3D fields
  = -1 or similar sentinel — verified against real `DontCare` lines: STF has two
  kinds, true unannotated regions (dims/location = -1/-1000 sentinels, matching
  KITTI) and non-vocabulary classes we remap to DontCare (real geometry passed
  through). Both are fine — KITTI eval ignores DontCare 3D fields entirely
  regardless of content, only using the 2D bbox for an IoU-based ignore mask).
- **Protocol decision — objects with no 3D box become DontCare** (BASELINE_TASKS.md
  D1). About a third of STF's Car/Pedestrian/Cyclist rows carry only a 2D box: dims
  `-1`, location `-1000`. Passed through as real objects, they are ground truth no 3D
  detection can ever match. KITTI eval assigns difficulty from the 2D box,
  occlusion and truncation alone, so every one that qualifies is a guaranteed false
  negative, capping recall. `02_convert_to_kitti.py` now remaps them to `DontCare`, so
  their 2D box becomes an ignore region, which is KITTI's own convention for
  regions without labels. This changes evaluation ground truth and is recorded here as a protocol
  choice, not tuning. `--keep-sentinel-boxes` reproduces the old labels for
  comparison. Training is expected to be unaffected: these boxes sit at -1000 m, so
  OpenPCDet's range filter already drops them. Unverified against Hahner's own
  loader, which is not public.
- **Known limitation, does not affect the AP gate**: STF's `alpha` field
  (KITTI's observation angle, normally `ry - atan2(x,z)`) is a constant `-1.0`
  for every object, confirmed across a 1000-file / 4394-object random sample —
  STF's annotation pipeline never computed it. OpenPCDet's eval code only reads
  `alpha` for AOS (Average Orientation Similarity), a metric this reproduction
  does not report; Car 3D AP@0.5 IoU (the gate metric) does not use it. Do not
  attempt to backfill `alpha` from `ry`/location — that would be inventing data
  the paper's own baseline never had either, since Hahner et al. trained on the
  same STF labels.

## 4. Calibration conversion

- Source: `calib_cam_stereo_left.json` (intrinsics/projection `P`) +
  `calib_tf_tree_full.json` (ROS tf tree, quaternion extrinsics velodyne↔camera).
- Reference implementation: official devkit's
  `tools/DatasetViewer/lib/read.py::load_calib_data` — reused near-verbatim (same
  math: build 4×4 homogeneous transforms from quaternions, invert or compose as in
  that function, derive `Tr_velo_to_cam`). That function's `R` is explicitly the
  identity (STF's P already contains the rectification KITTI normally splits out
  into R0_rect) — write `R0_rect = I` in our `calib.txt`, matching the source's own
  documented convention, not a simplification of our own.
- One calibration file pair covers the whole dataset (STF calibration is static,
  not per-frame — unlike KITTI which is per-frame in principle). Same `calib.txt`
  content gets written for every frame for simplicity/compatibility with the
  loader's per-frame read.

## 5. LiDAR conversion

- Source: `lidar_hdl64_strongest/*.bin`, shape `(N, 5)` float32 — confirmed against
  official devkit's `read.py::load_velodyne_scan` (its own comment: `# [:, :4]`,
  i.e. even the reference tooling only keeps 4 of the 5 fields downstream).
  We use **strongest returns**, matching the paper's default and Table 1/2 baseline
  rows (no `‡` strongest∩last filter — that's a separate, non-default variant).
- Action: read `(N,5)`, write `(N,4)` (drop last column), float32, same binary
  layout OpenPCDet expects.

## 6. Images

- Source: `cam_stereo_left/*.tiff`. OpenPCDet's KITTI pipeline uses `image_2` only
  for FOV cropping / image shape metadata for PV-RCNN — not as network input
  (LiDAR-only detector). Convert to `.png` (KITTI convention) via a fixed,
  lossless conversion; do not need cam_stereo_right for this baseline.

## 7. What's built vs. what's still needed

Built in this pass (CPU-only groundwork, no training/eval that needs CUDA) —
**all complete and verified**:
- [x] Split-count verification against context.md §2 (this doc, §1)
- [x] `scripts/01_extract_archives.py` — extracts the split-zip archives.
      Ran clean: 7829/7829 frames, 0 missing.
- [x] `scripts/02_convert_to_kitti.py` — the STF → KITTI conversion described
      above. Ran clean on the full dataset: 7829 frames, 77565 label objects,
      0 missing lidar/label/image files.
- [x] `scripts/05_verify_conversion.py` — sanity checks (counts, per-frame file
      completeness, lidar shapes, calib matrix shapes, label class histogram,
      projection spot-check). All 6 checks pass on the full converted dataset,
      including a 100% (131/131) projection-into-image-bounds result on a
      fresh 20-frame random sample.
- [x] OpenPCDet `pv_rcnn_stf.yaml` + `kitti_dataset_stf.yaml` configs, pointed
      at the converted data (config only — unmodified detector code, per
      context.md "do not modify the baseline"). One real deviation from
      "verbatim": `USE_ROAD_PLANE: False` (STF ships no `planes/` dir) — see §2.
- [x] `scripts/03_create_infos.py`, `scripts/04_train_and_eval.sh` — the actual
      training/eval entry points, written and reviewed against OpenPCDet source
      but NOT run (need `pcdet` importable, i.e. CUDA/spconv, i.e. the GPU box).

Deliberately NOT attempted on this machine (no CUDA/spconv): training,
inference, mAP computation. Those need a GPU box; this groundwork is what gets
handed to it. See `README.md` for the exact run order.

## 8. Risks checked during groundwork, and how they resolved

1. **DontCare box format** — checked against real lines: STF has two kinds
   (true unannotated regions with `-1`/`-1000` sentinels matching KITTI; and
   non-vocabulary classes we remap to DontCare, which keep real geometry).
   Both are harmless — KITTI eval ignores DontCare 3D fields regardless of
   content. Resolved, no fix needed.
2. **R40 vs R11, and IoU 0.5 vs 0.7** — both turned out to be the same
   OpenPCDet quirk: `get_official_eval_result` prints *four* blocks per class
   (R11×IoU0.7, R40×IoU0.7, R11×IoU0.5, R40×IoU0.5) in the same log, and the
   *structured* `ret_dict` it returns only captures the IoU-0.7 numbers by
   default. Documented precisely in `README.md`'s Gate check section — this
   was the single highest-value finding from source-reading, since it would
   have caused a silent, confusing false gate failure on the GPU box.
3. **Best-checkpoint selection** — paper selects the snapshot with best
   clear-val performance, not the final epoch. Encoded as `--eval_all` in
   `04_train_and_eval.sh` — not yet run, this is a GPU-box step.
4. **Calibration correctness** — verified two ways: a visual projection
   overlay on a real image (every box lands tightly on the real object, see
   `data/sanity_check/000000_final_check.png`), and a programmatic spot-check
   across 20 random frames (131/131 = 100% of in-front-of-camera boxes project
   into image bounds). **Only partly resolved**: both of those checks project labels
   (already in camera coordinates) through `P2`, so they never exercise
   `Tr_velo_to_cam`, the transform PV-RCNN actually depends on. Checks 7
   (lidar points inside GT boxes) and 8 (projected-3D vs labelled-2D IoU) in
   `05_verify_conversion.py` cover that. Treat calibration as verified only once
   they pass.
5. **`testing/` vs `training/` directory layout** — found via source-reading
   (not by symptom/crash) that OpenPCDet's `KittiDataset` hard-codes
   `'training' if self.split != 'test' else 'testing'`, which would have
   silently misrouted all three of our named test splits. Fixed by keeping
   everything under `training/` — see §2.
6. **STF's `alpha` field is a constant -1.0`** (not a per-object computed
   angle like KITTI's) — confirmed across a 1000-file random sample. Does not
   affect the Car 3D AP gate metric (alpha only feeds AOS, not reported here).
   See §3.
