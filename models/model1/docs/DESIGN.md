# Model 1 design

Status: **code built and unit-tested (20 tests, including a GPU training step and
inference through OpenPCDet on synthetic frames). Not yet trained on STF.** No
number in this folder is a result until M3–M5 run.

## 1. Goal

`context.md` §5, model 1: cross-attention over modality tokens, with attention
weights modulated by a learned per-stream reliability score. Built first because
it is closest to the baseline, so one pass yields a validated multimodal pipeline,
an eval harness and a number comparable to the baseline.

What counts as progress (`context.md` §1): calibration and honest reliability
under degraded sensors, not AP in fog. An AP gain without a calibration gain is
reported as such.

## 2. Inputs and data (data branch: `data_prep/`, `configs/stf_multimodal_dataset.yaml`)

- **Lidar:** HDL-64 strongest returns, `(N, 4)` points from the baseline's
  `kitti_format` conversion. Intensity is on a 0–255 scale (D5 audit).
- **Camera:** left RGB (`cam_stereo_left`), 1920×1024, resized to 960×512 inside
  the model. The raw TIFFs are GRBG Bayer mosaics and are **unrectified**
  (K: f = 2613 px, barrel distortion D[0] = −0.568). Every frame's calibration
  uses the rectified P (f = 2356 px). `data_prep/02_build_image_2.py` therefore
  demosaics, then rectifies with the calibration file's own K, D, R, P. Measured
  effect on lidar-image alignment: median offset (+4, +12) px before
  rectification, (0, 0) px after (`03_verify_images.py`, 91 frames).
- **Calibration:** per frame from the dataset (`calib_matricies`):
  `trans_lidar_to_cam = R0_rect · Tr_velo_to_cam`, `trans_cam_to_img = P2`.
  The extrinsic is verified by checks 7 and 8 in `05_verify_conversion.py`.
- Not used in v1: right camera, radar, FIR, gated NIR (never downloaded).

### Tone mapping

The baseline's `tonemap_and_convert` stretches each frame between its own 1st and
99th percentiles. That normalises away fog's contrast loss (BASELINE_TASKS D6)
and would hide degradation from the camera stream. Model 1 uses one **fixed
global** mapping: 1st/99th percentiles over 300 seeded train frames
(`data_prep/tonemap_stats.json`, after one fixed grey-world white balance), applied
unchanged to every frame of every split. Recorded, not tested: on the sampled
frames, contrast relative to brightness (grey std / mean) is 0.68 in dense fog
against 1.00 in clear test. Whole-image contrast also tracks day vs night, so it
is not used as a pass/fail check.

### Protocol decision: no GT sampling

The baseline config pastes other frames' object points into the scene
(`gt_sampling`). The image has no matching pixels, so camera tokens at pasted
objects would show background and teach the fusion that the streams disagree on
clean data. Model 1's dataset config drops it. The lidar-only ablation
(`configs/model1_lidar_only.yaml`) uses the same augmentation, so comparisons
within Model 1 stay like-for-like.

### OpenPCDet detail that matters here

World flip/rotation/scaling move the points but not the image, so Model 1 maps
BEV points back to the original lidar frame before projecting
(`rgfusion/aug_tracking.py`). OpenPCDet's own `lidar_aug_matrix` cannot be used:
`set_lidar_aug_matrix` applies the flip only when `flip_y` exists, and this
config flips along x only, so the flip would be silently dropped.
`tests/test_projection.py` checks the inverse against OpenPCDet's own augmentation
functions.

## 3. Architecture (`rgfusion/`)

1. **Lidar stream:** OpenPCDet `MeanVFE` → `VoxelBackBone8x` → `HeightCompression`
   → `BaseBEVBackbone` (KITTI SECOND settings), BEV map 512 × 200 × 176 (0.4 m
   cells). Shares a generic voxel encoder with PV-RCNN, not its keypoint or
   RoI-grid stages.
2. **Camera stream:** ResNet-50 (ImageNet, **frozen**, FrozenBatchNorm) + a small
   trained FPN → 256-channel stride-8 map. Frozen because OpenPCDet's
   `adam_onecycle` uses one learning rate (3e-3) for every parameter, which would
   wreck pretrained features. It also saves memory next to spconv.
3. **Modality tokens:** each BEV cell's centre at K = 4 heights (−1.8, −1.2,
   −0.6, 0.0 m) is mapped back through the augmentation, projected into the image
   and sampled bilinearly. The camera token is the mean over samples inside the
   image and in front of the camera. Cells with none get a learned "no-view"
   token and are marked invalid. One lidar token and one camera token per cell
   (128-d).
4. **Reliability estimators:** one MLP per stream on detached inputs: pooled
   features (avg + max) plus cheap statistics. Lidar: log point counts in 7 range
   bins, mean intensity. Camera: mean, std, gradient energy, saturated and dark
   fractions. Output `r_m ∈ (0, 1)` per frame.
5. **Gated fusion:** a learned query plus cell position attends over the two
   tokens. Two layers, 8 heads. Attention logits get `+ log r_m` (and
   `+ log 1e-4` for an invalid camera token). `r` enters as a fixed bias
   (detached), so a reported r is what the estimator believes, not something the
   detection loss tuned. There is no residual path around the gate.
6. **Head:** OpenPCDet `AnchorHeadSingle` with the KITTI SECOND anchors (Car
   bottom z = −1.78 m) on the fused map (projected back to 512 channels).
   Sigmoid scores are the confidences used for ECE.

`FUSION.USE_CAMERA: False` skips 2–5: the lidar-only ablation.

## 4. Training the gate without weather data (protocol decision)

`context.md` §6: training uses clear weather only, and weather corruption is
applied at evaluation time only. The gate learns from generic, non-weather
signals.

- **Modality dropout** (`rgfusion/modality_dropout.py`), per training sample:
  with p = 0.25 the lidar is perturbed (sparse: keep 2% of points; noise: uniform
  random points; sector: remove a 40–80% azimuth sector). Otherwise, with
  p = 0.25, the camera is perturbed (zeros; uniform noise; grey rectangle over
  40–80% of the image). Never both, so 50% of samples stay clean. Each sample
  records the surviving fraction `keep_m`. None of these are shaped like fog,
  snow or icing.
- **Auxiliary heads:** each stream alone predicts a BEV Car-centre heatmap
  (CenterNet focal loss, per sample `L_m`), trained only on samples where that
  stream survived.
- **Reliability target:** `r*_m = keep_m · min(1, exp(1 − L_m / τ_m))`, where
  `τ_m` is an EMA (momentum 0.01) of `L_m` over clean samples. A typical or better
  frame for that stream → 1; twice its typical loss → e⁻¹ ≈ 0.37; wiped out → 0.
  τ is per stream, so the camera is not marked unreliable just for being worse
  than lidar at BEV localisation in general.
- **Loss:** detection loss + 0.5 · Σ aux losses + 1.0 · BCE(r_m, r*_m).
- Same budget as the baseline: 80 epochs, `--fix_random_seed`, batch 4, mixed
  precision (`USE_AMP`).

If it turns out to need weather-shaped perturbations to work, that is a finding
to report, not a silent change.

## 5. Metrics

Definitions are fixed here before any number is produced and apply identically
to the baseline and all four models.

- **Car 3D AP:** OpenPCDet's KITTI eval, R40, IoU 0.5, easy/moderate/hard. Read
  from the text log block `AP_R40@0.70, 0.50, 0.50:`, `3d AP` line, with
  `baseline/scripts/07_read_eval_logs.py` (Rohil's R3). Each test split is
  evaluated on its own info file (`DATA_CONFIG.INFO_PATH.test`), never val.
  Logs must show 1847 / 946 / 786 frames.
- **ECE** (`tools/eval_calibration.py`):
  - Detections: class Car, every detection OpenPCDet keeps (score ≥ 0.1).
  - Correct: a true positive at 3D IoU ≥ 0.5 (rotated boxes, lidar frame).
    Matching is greedy by descending score, and each ground-truth Car matches once.
  - Ground truth: the frame's Car boxes from the info file (post-D1 labels).
  - DontCare: an unmatched detection whose 2D box lies mostly inside a DontCare
    box (intersection / detection area > 0.5) is excluded.
  - 15 equal-width bins. ECE = Σ_b (n_b / N) · |accuracy_b − confidence_b|.
- **D-ECE:** the same, per range bin of the detection centre: 0–20, 20–40,
  40–70.4 m.
- **Reliability report** (`tools/summarize_reliability.py`): r_lidar, r_cam and
  the mean camera attention weight per split. The thesis check: do the scores
  fall in real fog the model never saw in training?
- **ECE with two modalities degraded** needs the corruption engine (separate
  work). The hook is ready: `M1_EVAL_CORRUPTION=<module>:<function>` applies a
  function to every test sample before voxelisation (`rgfusion/register.py`).

## 6. Verification

- `tests/` (run with `run_model1_docker.ps1`; default command): the inverse
  augmentation matches OpenPCDet's own functions; known-point projection; BEV
  grid ordering; dropout rates, exclusivity and keep fractions; attention
  weights sum to 1; a low-reliability or invalid camera is ignored; heatmap and
  focal-loss behaviour; the reliability target; ECE on hand-computed cases; and
  a full OpenPCDet training step (with backward) and inference for both configs
  on synthetic frames on the GPU.
- Planned once trained: eval parity (`eval_calibration.py` on the baseline's
  `result.pkl`) and the M3 overfit test on a 200-frame slice.

## 7. Risks

- GPU memory (RTX 4090, 24 GB) at batch 4 with 960×512 images plus spconv:
  measured at M3. Fall back to batch 2.
- Gate collapse (r ≈ 1 always, or constant): watch `m1_r_lidar` / `m1_r_cam`
  and `m1_tau_*` in TensorBoard. Dropped samples must pull r toward 0.
- Frozen ImageNet features may limit the camera stream. Unfreezing layer4 with a
  lower learning rate is the next step if camera aux loss stays flat.
- Training time is an estimate until M3 measures an epoch.
