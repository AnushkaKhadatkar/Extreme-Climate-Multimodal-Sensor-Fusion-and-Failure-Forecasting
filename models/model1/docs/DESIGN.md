# Model 1 design

Status: **proposed, not built.** Every section below is the plan for M1–M5 and will
be updated as parts are implemented and measured.

## 1. Goal

`context.md` §5, model 1: cross-attention over modality tokens, with attention
weights modulated by a learned per-stream reliability score. Built first because
it is closest to the baseline, so one pass yields a validated multimodal pipeline,
an eval harness and a number comparable to the baseline.

What counts as progress (`context.md` §1): calibration and honest reliability
under degraded sensors, not AP in fog. An AP gain without a calibration gain is
reported as such.

## 2. Inputs

- **Lidar:** HDL-64 strongest returns, `(N, 4)` points from the baseline's
  `kitti_format` conversion. Intensity is on a 0–255 scale (D5 audit).
- **Camera:** left RGB (`cam_stereo_left`), 1920×1024, resized to 960×512.
- Calibration: the baseline's `calib.txt` (`P2`, `R0_rect = I`, `Tr_velo_to_cam`);
  the extrinsic is verified by checks 7 and 8 in `05_verify_conversion.py`.
- Not used in v1: right camera, radar, FIR, gated NIR (never downloaded).

### Tone mapping

The baseline's `tonemap_and_convert` stretches each frame between its own 1st and
99th percentiles. That normalises away fog's contrast loss (BASELINE_TASKS D6) and
would hide degradation from the camera stream. Model 1 uses one **fixed global**
mapping: the 1st/99th percentiles are computed once over the train split and
applied unchanged to every frame of every split.

### Data root

`<OpenPCDet>/data/kitti_format_mm/` holds symlinks to the baseline's `velodyne/`,
`calib/` and `label_2/` (post-D1 labels), a real `image_2/` and its own info
files. The baseline's data folder is not modified.

## 3. Architecture

1. **Lidar stream:** stock OpenPCDet `VoxelBackBone8x` → `HeightCompression` →
   `BaseBEVBackbone`, with the baseline's point-cloud range and voxel size. This
   shares a generic voxel encoder with PV-RCNN, not its keypoint or RoI-grid stages.
2. **Camera stream:** ResNet-50 + FPN (ImageNet weights), stride-8 features.
3. **Modality tokens:** each BEV cell (pooled to 0.8 m) lifts reference points at
   K heights. Train-time world augmentation (flip, rotation, scaling) is undone,
   then the points are projected through `Tr_velo_to_cam`, `R0_rect` and `P2`, and
   image features are sampled bilinearly. Per cell: one lidar token and one camera
   token. A learned "no-view" token replaces the camera token when the point falls
   outside the image.
4. **Reliability estimator:** one small MLP per stream, reading only that
   stream's own input (pooled features plus cheap statistics: lidar points per
   range bin; image gradient energy and saturated-pixel fraction). Output
   `r_m ∈ (0, 1)` per frame. A coarse spatial version is a possible later step.
5. **Gated fusion:** two layers of multi-head attention over the modality tokens
   at each cell. The attention logit for stream m is shifted by `log r_m`, so a
   stream with r → 0 drops out of the softmax.
6. **Head:** `AnchorHeadSingle` with the baseline's KITTI anchors (Car bottom
   z = −1.78 m). Sigmoid scores are the confidences used for ECE.

## 4. Training the gate without weather data (protocol decision)

`context.md` §6: training uses clear weather only, and weather corruption is applied
at evaluation time only. The gate therefore learns from generic, non-weather
signals:

- **Modality dropout:** per training sample, each stream is independently replaced,
  with probability p ≈ 0.25, by zeros, Gaussian noise or a random spatial mask.
  Both streams are never dropped at once. These perturbations are deliberately not
  shaped like fog, snow or icing.
- **Auxiliary heads:** each stream alone predicts a BEV Car heatmap. Reliability
  target: `r*_m = 0` if the stream was dropped, otherwise `exp(−L_aux,m / τ)`
  (stop-gradient; τ = running median of `L_aux,m`).
- Loss: detection loss + Σ aux losses + λ · BCE(r_m, r*_m).
- Same budget as the baseline: 80 epochs, `--fix_random_seed`, mixed precision.

This is recorded here as a protocol choice. If it turns out to need weather-shaped
perturbations to work, that is a finding to report, not a silent change.

## 5. Metrics

Definitions are fixed here before any number is produced and apply identically to
the baseline and all four models.

- **Car 3D AP:** OpenPCDet's KITTI eval, R40, IoU 0.5, easy/moderate/hard. Read
  from the text log block `AP_R40@0.70, 0.50, 0.50:`, `3d AP` line (parsed with
  `baseline/scripts/07_read_eval_logs.py` once Rohil's R3 lands). Each test split
  is evaluated on its own info file (`DATA_CONFIG.INFO_PATH.test`), never val.
  Logs must show 1847 / 946 / 786 frames.
- **ECE:** a detection is correct if it is a true positive at 3D IoU 0.5 (greedy
  matching by score, DontCare regions ignored as in KITTI eval). Its confidence is
  the detector score. 15 equal-width bins. ECE = Σ_b (n_b / N) · |acc_b − conf_b|.
- **D-ECE:** the same, computed separately for range bins 0–20, 20–40 and 40–70 m.
- **Reliability report:** distribution of `r_lidar` and `r_cam` per split
  (clear, light fog, dense fog). The thesis check: do the scores fall in real fog
  that the model never saw in training?
- **ECE with two modalities degraded** needs the corruption engine (separate
  work). The test tool exposes a hook for an evaluation-time corruption function.

## 6. Verification

- Unit tests: a dropped stream gives r < 0.2; projected boxes stay aligned under
  augmentation; dropout rates match configuration; ECE matches a hand-computed case.
- Eval parity: Model 1's eval path, run on the baseline's predictions, reproduces
  the baseline's logged AP to two decimals.
- Image overlays of lidar points on the new `image_2` for a sample of frames per split.

## 7. Risks

- World augmentation breaks lidar→image alignment if not inverted.
- GPU memory (RTX 4090, 24 GB) with 960×512 images plus spconv: fall back to
  batch 2 with gradient accumulation, or ResNet-34.
- Gate collapse (r ≈ 1 always, or constant): monitored per epoch on val with
  synthetic stream drops.
- Training time is an estimate until M3 measures an epoch.
