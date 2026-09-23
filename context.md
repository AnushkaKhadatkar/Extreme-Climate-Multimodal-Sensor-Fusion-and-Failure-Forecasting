# Extreme-Climate Multimodal Sensor Fusion and Failure Forecasting

Context for Claude Code. Read this before touching anything in the repo.

DATA 298A/298B capstone, M.S. Applied Data Intelligence, San José State University.
Team 3, Section 11, Topic 17.

---

## 1. What this project is

Multi-sensor perception systems (camera + lidar + radar + gated NIR) assume sensor
**redundancy**: if one stream is uncertain, the others compensate. In extreme weather
that assumption fails, because sensors degrade **jointly, not independently** — the same
freezing rain coats a lidar dome and a camera lens, the same low sun saturates the camera
while wet roads scatter radar.

The consequence that matters: a conventional fusion model carries **no internal
representation of its own input quality**. It cannot distinguish a camera observing an
empty road from a camera observing ice on its own lens. So it keeps consuming the dead
stream at full weight and emits a confident, wrong prediction with no signal that anything
changed.

**The governing distinction for every design decision in this repo:**

> A visible accuracy drop is an engineering problem. A model that is confidently wrong and
> silent about it is a safety failure.

We are therefore **not** optimising for accuracy in fog. We are building a system that
estimates per-modality sensor health jointly with the detection task, without corruption
labels at inference time, and acts on that estimate: attenuating degraded streams, routing
through reliable ones, reporting honest uncertainty, and forecasting failure where the
physics permits.

If a proposed change improves AP but not calibration, it is not progress on this project's
thesis. Say so rather than shipping it silently.

---

## 2. Benchmark and data

### Primary: Seeing Through Fog (STF / DENSE)

Bijelic et al., CVPR 2020. ~13,500 labelled frames, ~100,000 labels, 10,000+ km across
northern Europe. Modalities: RGB stereo, gated NIR, scanning lidar, radar, FIR, plus
weather-station and road-friction readings. Access obtained (registration, non-commercial
research use).

**Splits — verify these counts before trusting any result:**

| Split            | Scenes |
| ---------------- | -----: |
| Clear train      |  3,469 |
| Clear val        |    781 |
| Clear test       |  1,847 |
| Light fog test   |    946 |
| Dense fog test   |    786 |

If a split file does not produce exactly these counts, stop and fix that before anything
else. A count mismatch invalidates every downstream comparison, and the resulting AP gap
looks exactly like a training bug.

Labels are in **KITTI format**, which is why OpenPCDet detectors run on this directly.

### Held-out: Canadian Adverse Driving Conditions (CADC)

Pitropov et al., IJRR 2021. 7,000 annotated winter frames, Region of Waterloo. 8 cameras +
Velodyne VLP-32C + GNSS/INS. **No gated NIR, no radar.**

Used for unseen-climate transfer only. **Never train on it. Never tune on it.** This is
298B work; nothing in 298A should touch it.

### Corruption engine (built by us, not downloaded)

Generates physically plausible, **temporally evolving, correlated** sensor failures on top
of STF frames — not independent random noise. It exists because no dataset anywhere labels
*which sensor degraded when*, and that ground-truth timeline is the only reason lead time
and recovery time are measurable at all.

| Channel            | Provenance                                  |
| ------------------ | ------------------------------------------- |
| Fog                | Hahner et al. ICCV 2021, public simulator   |
| Snow               | Hahner et al. CVPR 2022, public simulator   |
| Icing              | Ours — must be defended                     |
| Lens occlusion     | Ours — must be defended                     |
| Calibration drift  | Ours — must be defended                     |
| Telemetry dropout  | Ours — must be defended                     |

Anchor fog and snow to the published simulators rather than reimplementing them. Report
the four invented channels separately and label them synthetic in any output.

---

## 3. The baseline (this is not one of our models)

**Hahner et al., ICCV 2021**, "Fog Simulation on Real LiDAR Point Clouds for 3D Object
Detection in Adverse Weather", pp. 15283–15292. arXiv:2108.05249

**Exact result reproduced: Table 2, row `PV-RCNN †` at α = 0.**

- `†` = trained on clear weather only
- `α` = attenuation coefficient of their *simulated* fog augmentation
- **α = 0 = no fog augmentation.** This is the control row, and it matches our protocol
  exactly. Do not reproduce a fog-augmented row — that would bake weather augmentation into
  our reference point and confound every later comparison.

**Published Car 3D AP@0.5 IoU:**

| Split     | Easy  | Moderate  | Hard  |
| --------- | ----: | --------: | ----: |
| Clear     | 79.61 | **77.05** | 71.03 |
| Light fog | 69.55 |     70.17 | 68.44 |
| Dense fog | 45.03 | **46.00** | 45.08 |

Note the moderate > easy inversion on both fog splits, and the near-flat spread in dense
fog. These are real, not typos: difficulty tiers are separate evaluation sets (not nested
subsets), and KITTI difficulty is defined on *image-plane* visibility, which decouples from
actual lidar difficulty once backscatter dominates. **Reproducing the inversion is positive
evidence the protocol matches.**

Hahner's detectors are **lidar-only** — camera, gated NIR and radar are unused. Our
multimodal models therefore compete against a lidar-only reference from a richer input.
State that asymmetry explicitly in any writeup; do not let it pass silently.

### Gating criterion

**Reproduce 77.05 ± 2 AP** (pass band 75.05–79.05) on clear/moderate before anything else
counts. Until that passes, no number from any of our models is interpretable, because a
deviation of unknown origin contaminates every comparison.

### When reproduction misses, check protocol before touching hyperparameters

In order of likelihood:

1. **R11 vs R40** recall sampling in the eval config — most common cause, moves AP by several points
2. **IoU threshold** — must be 0.5, not KITTI's default 0.7 for cars
3. **3D vs BEV** — read the 3D column, not BEV
4. **Split file contents** — the counts in §2
5. **Difficulty tier** — moderate

Never tune hyperparameters to hit 77.05. That fits the pipeline to a target instead of
validating it, which defeats the entire purpose of the gate.

---

## 4. Metrics

Full metric spec: **Car 3D AP@0.5 IoU, KITTI easy/moderate/hard, 40 recall positions
(R40).** Every clause is load-bearing — dropping any one makes numbers incomparable.

| Metric | What it measures | Target | Baseline status |
| --- | --- | --- | --- |
| Car 3D AP@0.5 | Detection accuracy | Within 2 AP of 77.05 (clear/moderate) | Published |
| Expected calibration error | Whether confidence matches actual reliability | < 0.05 with **two modalities degraded** | Unoptimised — we measure it |
| Failure lead time | Warning before a 10% AP drop | Median ≥ 60 s (icing scenarios) | Zero by construction |
| Worst-decile AP | Tail risk, bottom 10% of scenarios | ≥ 70% of clear-weather AP | We measure it |
| Recovery time | Return to within 5% of pre-failure AP after sensor returns | ≤ 10 s | We measure it |

Only AP is published. The other four we compute ourselves **on the reproduced baseline**,
under conditions identical to our four models, so all five systems sit on the same axes.

**ECE is the thesis metric.** The "two modalities degraded" clause is the whole claim —
calibration on clean data proves nothing. Implementation decision that must be fixed early
and written down: a detection counts as correct if it is a true positive at the IoU
threshold, and its confidence is the detector score. Apply that definition identically to
the baseline and all four models. Consider D-ECE (conditioned on box position/size) as
well, since a detector can be calibrated on average and badly overconfident on distant
objects — which is exactly where weather bites.

**Recovery time exists to prevent a degenerate solution.** A model that aggressively
distrusts everything would score well on ECE and worst-decile risk while being useless.
Recovery punishes over-suppression; AP punishes under-performance. The five metrics bound
the solution space from both sides — do not optimise one in isolation.

---

## 5. The four models

The baseline is **not** one of these. It is a fifth, separately reported system. None of the
four reuses its architecture.

1. **Reliability-gated multimodal transformer** — cross-attention over modality tokens,
   attention weights modulated by a learned per-stream reliability score. **Built first**,
   because it is architecturally closest to the baseline, so one pass delivers a validated
   pipeline, a working eval harness and a comparable number.
2. **Product-of-experts probabilistic fusion** — each sensor predicts a full distribution;
   multiplying them means a high-variance (unreliable) sensor excludes itself. No explicit
   gate required.
3. **Physics-informed degradation forecaster** — predicts degradation from weather
   trajectories (ice accretion, solar geometry, thermal drift). This is where lead time
   comes from.
4. **Self-supervised test-time adaptation** — adapts without labels at deployment via
   cross-modal consistency, targeting unseen climates.

Expected outcome is **not** that one model dominates. Expect PoE to win calibration, the
transformer to win in-distribution accuracy, the physics-informed model to win lead time
and unseen-climate transfer. Characterising that trade-off is the contribution.

---

## 6. Protocol invariants — do not violate these

- **Train on STF clear weather only.** Adverse weather is never seen during training. This
  is inherited from the baseline and is what makes the benchmark a robustness test rather
  than a fog-fitting test.
- **Corruption is applied at evaluation time**, not training time. This is also why no
  experiment requires a retrain — train each detector once, freeze it, evaluate repeatedly.
- **No corruption labels at inference.** Sensor health must be estimated, not read off.
- **CADC is never trained or tuned on.**
- **All five systems evaluated under identical conditions** on all five metrics.
- **Fix seeds. Pin versions.** Reproducibility is the whole premise of the gate.

---

## 7. Current state and phasing

**Phase 1 — baseline reproduction (now).** Lidar + labels + calibration only; camera, gated
NIR and radar are not needed yet because the baseline is lidar-only. That is 7,829 scenes
across the five splits above (~15–30 GB), versus the full multimodal download.

Order of work:
1. Pull ~200 frames (100 clear, 100 dense fog) and debug the loaders, viewer, OpenPCDet
   config and environment on that slice. Every CUDA/spconv problem surfaces on 200 frames
   as reliably as on 7,829.
2. Verify split counts against §2.
3. One-time preprocessing pass to OpenPCDet's info format — do not convert on the fly.
4. Time one epoch. This converts our compute estimate into a measured number.
5. Train PV-RCNN on clear, evaluate, check the five protocol items, read AP.

**Phase 2 — corruption engine + model 1.** Add RGB camera. Start with the fog channel using
Hahner's public simulator; validate it by evaluating the already-trained PV-RCNN on
*simulated* fog and comparing against its *real* dense-fog score. If simulated fog degrades
the detector similarly to real fog, the channel is validated against reality — this is the
strongest available evidence for engine realism.

**Phase 3 — 298B.** Models 2–4, CADC transfer, consolidated trade-off map, benchmark release.

### 298A deliverables

Validated STF pipeline (sync + calibration confirmed) · corruption engine · reproduced
baseline with full five-metric profile · evaluation harness computing all five metrics ·
model 1 evaluated against the baseline.

---

## 8. Environment and tooling

- **OpenPCDet** — how PV-RCNN gets trained. Use released configs verbatim for the baseline.
- **Docker image with locked CUDA / PyTorch / OpenPCDet commit, fixed seeds.** Not yet
  built — this is planned, not done.
- Storage: training split on local NVMe. Do **not** train off a mounted Google Drive or an
  external spinning disk — random reads across the training set every epoch will bottleneck
  the GPU badly.
- Compute: estimated 150–250 GPU-hours across five training runs; currently ~60 GPU-hours a
  month available. Mixed precision throughout. Ablations on a 25% subsample.

---

## 9. Open questions

- **Are gated NIR and radar present in the STF release we hold?** Unresolved. The entire
  multimodal story depends on it. Resolve before Phase 2 planning.
- Exact STF licence terms — being confirmed in writing.
- CADC has no gated NIR or radar, so models must degrade gracefully to camera+lidar for the
  transfer study. Decide how, before 298B.

---

## 10. Working rules for Claude Code

- **Never tune toward 77.05.** Check protocol first. If AP is off, the cause is almost
  certainly the eval config, not the model.
- **Do not modify the baseline** beyond what is needed to run it on our splits. It is a
  reference point, not a competitor to be improved.
- **Keep the baseline separable** in code and in results. It is reported as a fifth system
  throughout — never folded into the four.
- **Flag AP-only improvements.** If a change helps accuracy but not calibration, say so
  explicitly rather than treating it as a win.
- **Preserve hedges in any generated writeup.** The Docker/Linear setup is *planned*, not
  done. Compute figures are *estimates*. The STF licence is *being confirmed*. Do not let a
  rewrite smooth these into past tense or fact.
- **Invented corruption channels stay labelled synthetic** in code comments, logs and
  outputs.
- When adding a metric implementation, write its definition into the repo docs at the same
  time. Ambiguous metric definitions applied inconsistently across five systems would
  invalidate the entire comparison.

---

## 11. Key references

| Ref | Use |
| --- | --- |
| Bijelic et al., CVPR 2020 — Seeing Through Fog — arXiv:1902.08913 | Dataset source |
| Hahner et al., ICCV 2021 — Fog simulation — arXiv:2108.05249 | **Baseline** + fog channel |
| Hahner et al., CVPR 2022 — Snowfall simulation | Snow channel |
| Pitropov et al., IJRR 2021 — CADC — doi:10.1177/0278364920979368 | Held-out transfer set |
| Guo et al., ICML 2017 — On calibration of modern neural networks — arXiv:1706.04599 | ECE definition |
| Hendrycks & Dietterich, ICLR 2019 — arXiv:1903.12261 | Corruption benchmarking design |
| Wang et al., ICLR 2021 — Tent — arXiv:2006.10726 | Model 4 foundation |
| Palladin et al., ECCV 2024 — SAMFusion — arXiv:2508.16408 | Closest prior work |

Code: `princeton-computational-imaging/SeeingThroughFog` (devkit) ·
`MartinHahner/LiDAR_fog_sim` (fog) · `SysCV/LiDAR_snow_sim` (snow) ·
`open-mmlab/OpenPCDet` (detectors)