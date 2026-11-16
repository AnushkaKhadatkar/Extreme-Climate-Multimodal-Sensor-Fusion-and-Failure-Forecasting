# Corruption engine: design

Shared by all five systems (the baseline and the four models), not Model 1's own tool. Lives at the repo root
(`corruption/`), not under `models/model1/`, because `context.md` section 8 ("all five systems evaluated under
identical conditions") requires the baseline and every model to see the exact same corrupted frame for a given
(scene, channel, severity, time). Model 1 already has a plug-in point for this
(`rgfusion/register.py`'s `M1_EVAL_CORRUPTION=<module>:<function>` hook); the baseline and the other models get the
same kind of hook once they need it.

Status: **design only. No corruption channel is implemented yet.** This file is written before any code so the
scope and the validation bar are fixed first, the same discipline `models/model1/docs/RUN_LOG.md` used before the
M5p/M4bp runs.

## 1. What this is not

- Not `models/model1/tools/eval_corruptions.py`. That file is a quick, deliberately simple probe (dead/occluded/
  dark/blurred camera, sparse/sectored lidar) built to answer one narrow question fast: does Model 1's reliability
  gate react to single-frame, single-sensor failures. It is kept, and it stays Model 1's own tool. It is not
  "physically plausible, temporally evolving, correlated" (`context.md` section 2) and was never meant to be.
- Not a replacement for Hahner's fog/snow simulators. Those are anchored to, not reimplemented (`context.md`
  section 2: "Anchor fog and snow to the published simulators rather than reimplementing them").

## 2. The six channels (`context.md` section 2)

| Channel | Provenance | Status |
| --- | --- | --- |
| Fog | Hahner et al., ICCV 2021, public simulator | **in progress** (this phase) |
| Snow | Hahner et al., CVPR 2022, public simulator | not started |
| Icing | Ours -- must be defended | not started |
| Lens occlusion | Ours -- must be defended | not started |
| Calibration drift | Ours -- must be defended | not started |
| Telemetry dropout | Ours -- must be defended | not started |

Fog first, for two reasons `context.md` section 7 already gives: it is the one channel we can validate against a
real held-out split (dense fog, 786 frames) that none of the five systems ever trained on, and validating it is
"the strongest available evidence for engine realism" for the whole corruption engine, not just the fog channel.

The four "ours" channels are reported separately from fog/snow and labelled synthetic in every output
(`context.md` section 2 and section 10's "invented corruption channels stay labelled synthetic in code comments,
logs and outputs" -- carried over verbatim as a rule for this package, not just for Model 1).

## 3. Interface

One corruption channel is a Python callable with the same signature Model 1's hook already expects, so the same
function registers with Model 1 and with whatever hook the baseline and models 2-4 eventually get:

```python
def apply(data_dict: dict, severity: float, rng: np.random.Generator) -> dict:
    """data_dict has at least 'points' (N, 4) float32 [x, y, z, intensity(0-255)] in the lidar frame,
    and may have 'images' (H, W, 3) float32 in [0, 1] if the model being evaluated uses a camera.
    severity in [0, 1]: a channel-specific mapping to its own physical parameter (e.g. for fog, alpha).
    Mutates and returns data_dict; does not change its keys or their dtypes/shapes beyond the corrupted
    array(s) themselves, so every downstream consumer (voxelisation, image resize) is unaffected."""
```

- **Deterministic per (frame, channel, severity, seed)**, exactly like `models/model1/tools/eval_corruptions.py`'s
  existing convention: seed from `zlib.crc32(frame_id)` so a frame always gets the same corruption on repeat runs,
  but different frames get different draws from the same `rng` state so the dataset isn't corrupted identically
  frame to frame.
- **severity, not alpha/mm-per-hour/etc, at the public interface.** Internally fog maps severity to Hahner's alpha
  (exact mapping fixed once the simulator's valid range is confirmed -- see section 5); snow will map to its own
  parameter the same way. This keeps "how bad is it" comparable across channels for the metrics in section 6, even
  though the underlying physics differ.
- **No corruption labels reach any model's reliability estimator at inference** (`context.md` section 6, already
  true of Model 1's modality-dropout training and the eval-corruption hook; carries over unchanged here). The
  severity argument is visible to the harness that orchestrates evaluation, never to the model under test.

## 4. Temporal evolution and correlation (the part the Model 1 probes could not test)

`context.md` section 2 calls for corruption that is "temporally evolving, correlated... not independent random
noise," specifically because that timeline is the only way to measure failure lead time and recovery time
(the two thesis metrics still unmeasured). STF frames are organised in sequences (`stf_sequence` in
`id_mapping.csv`); a severity *schedule* is a function of frame index within a sequence, not a constant:

```python
def schedule(frame_index_in_sequence: int, total_frames: int, rng: np.random.Generator) -> float:
    """-> severity in [0, 1] for this frame. Example shapes: a ramp up then plateau (onset), a plateau
    then ramp down (recovery), a slow sinusoid (intermittent). Each channel can use a different shape,
    but two channels sharing a schedule (e.g. fog severity driving both lidar and camera degradation
    together) is how "correlated" is expressed: pass the same schedule-derived severity into both
    channels' apply() for a given frame, rather than drawing each independently."""
```

Lead time and recovery time (section 6) are read off this schedule directly: the schedule defines a ground-truth
"failure started at frame k" and "failure ended at frame k+d" timeline that no real dataset provides, which is
exactly why `context.md` says this corruption engine is the only way to measure them.

Not solved by this design, deliberately left open: which STF sequences are long enough and how onset/plateau/
recovery durations map to real time (frame rate is sequence metadata, not yet pulled into this plan). That is
scoped for the "two sensors degraded, with a schedule" phase (section 7.2 below), not the fog-validation phase.

## 5. Fog channel (this phase)

- Wraps Hahner's `LiDAR_fog_sim` (public repo, cited in `context.md` section 11) for the lidar point cloud.
  Exact function signature, alpha range, and whether it needs any shipped lookup tables: being confirmed before
  any code is written against it (do not guess the interface).
- **No camera fog channel yet.** Hahner's simulator is lidar-only (`context.md` section 3: "Hahner's detectors are
  lidar-only"). A camera fog model (contrast/haze on the image, akin to `eval_corruptions.py`'s `cam_haze` probe
  but physically grounded) is out of scope for this phase; evaluating M5/M4b/M5p/M4bp under fog in this phase means
  the lidar is fogged and the camera is not, which is a known asymmetry to state plainly in any output, not hide.
- severity -> alpha mapping: fixed once section 5's open question (alpha's valid range and what the paper's own
  "light" vs "dense" fog correspond to) is answered.

### Validation bar (must pass before the fog channel is used for anything else)

`context.md` section 7: "validate it by evaluating the already-trained PV-RCNN on *simulated* fog and comparing
against its *real* dense-fog score. If simulated fog degrades the detector similarly to real fog, the channel is
validated against reality."

Concretely: take the baseline's trained checkpoint (already have it: clear/moderate 76.59, real dense-fog
moderate 39.49 AP -- `baseline/README.md`), apply the fog channel to the **clear test split** (1,847 frames, never
corrupted before) at the severity mapped to "dense fog" alpha, run the baseline's existing eval path unmodified, and
compare:

- **Pass bar (not yet chosen, see open question below):** simulated-fog AP within some band of 39.49 that
  reflects measurement noise, not a free parameter to tune toward. The baseline's own run-to-run spread has not
  been measured (`BASELINE_TASKS.md` "R6 ... Only spend GPU hours on a second seed if the gate result lands near a
  band edge" -- the baseline was run once), so the bar should probably be relative (same direction and rough
  magnitude of AP drop vs clear, e.g. "within 5 AP of the real drop") rather than an exact-match target, to avoid
  repeating the mistake the whole project is built to avoid: tuning a pipeline to hit a number instead of validating
  it. **Decide the exact band with the team before running this, not after seeing the number** -- RUN_LOG will
  record the chosen band before the validation run, the same discipline used for the M5p/M4bp evidence criteria.
- If it fails: check the alpha mapping and point cloud conventions (intensity scale, coordinate frame) before
  concluding the simulator itself is wrong -- same "check protocol before touching the model" discipline as
  `context.md` section 3's baseline-reproduction rule.

## 6. Metrics this unlocks (none measured yet)

| Metric | Needs from this engine |
| --- | --- |
| ECE with two modalities degraded | Any two channels (e.g. fog on lidar, lens occlusion on camera) applied together to one frame -- the single biggest thesis-metric gap today |
| Failure lead time | A severity schedule (section 4) with a known onset frame, read against when a model's reliability score (or AP) first drops |
| Worst-decile AP | Many corrupted frames at varied severity across channels -- the bottom 10% by AP, vs. clear-weather AP |
| Recovery time | A severity schedule with a known offset/recovery frame, read against when AP/reliability returns to within 5% of its pre-failure value |

None of these are implemented. They are downstream of having at least two validated channels (fog here, one "ours"
channel next, so "two modalities degraded" has two real things to combine) and a severity-schedule harness (section
4), not just the fog channel on its own.

## 7. Open questions

1. ~~Hahner's `LiDAR_fog_sim` interface~~ -- **answered.** `simulate_fog(p: ParameterSet, pc, noise, gain=False,
   noise_variant='v1', hard=True, soft=True) -> (augmented_pc, simulated_fog_pc, info_dict)`, vendored at
   `corruption/third_party/LiDAR_fog_sim/` (cloned from github.com/MartinHahner/LiDAR_fog_sim, CC BY-NC 4.0, fine
   for this non-commercial capstone with attribution; no pip package exists, so vendored as source, with the
   unrelated `SeeingThroughFog` submodule and its own nested `.git` removed since we don't use it -- our STF data
   comes through this repo's own baseline pipeline). `ParameterSet.alpha` default 0.06, range [0.003, 0.5]; 9
   shipped LUT alphas (0.005 to 0.2) cover `soft=True` (fog backscatter); `hard=True` (Beer-Lambert attenuation
   only, no LUT needed) works at any alpha. Point cloud: hard-target math only needs columns 0-3 (x, y, z,
   intensity); soft-target preserves a 5th column if present but does not require one. Confirm this with a real run
   before relying on it (see the still-open item below) -- the code has a `pc.shape[1] > 4` branch suggesting it
   was built against 5-column STF data (`n_features=5` is the CLI default) and STF's lidar here is 4 columns; needs
   a check that 4-column input is not silently mishandled before the wrapper in `fog.py` is written.
2. ~~Light-fog vs dense-fog alpha~~ -- **answered: there is no mapping, because the question was wrong.** Confirmed
   directly from the paper's Table 2 (ar5iv HTML, cross-checked against the exact AP numbers already in
   `context.md`): "light fog" and "dense fog" are STF's own real-weather test splits (946 and 786 scenes), evaluated
   at alpha = 0 in every baseline row. Alpha only appears as a training-time augmentation parameter (sampled from
   {0, 0.005, 0.01, 0.02, 0.03, 0.06} per training example), never as something tied to a real-weather split. So
   section 5's validation is not "find the alpha the paper used for dense fog" -- no such alpha exists -- it is
   exactly what `context.md` section 7 already says: pick an alpha, apply it to clear frames, and check whether the
   resulting AP drop resembles the real dense-fog AP drop. The pass/fail band (next item) is the real open question,
   not the alpha value.
3. The validation pass/fail band (section 5) -- a team decision, not a model one. Still open.
4. Whether frame rate / inter-frame timing metadata exists per STF sequence, needed before section 4's schedules can
   be given real time units (seconds) rather than just frame counts, which the lead-time and recovery-time metrics
   are defined in (`context.md` section 4: "Median >= 60 s", "<= 10 s"). Still open.
5. ~~Does `simulate_fog` handle STF's native 4-column lidar correctly?~~ -- **answered: yes**, verified by running it
   on a synthetic 2,000-point (N, 4) array. Hard-target mode leaves positions unchanged and attenuates intensity
   only (confirmed `positions unchanged: True`); soft-target mode correctly relocated 942/2000 points as fog
   backscatter returns; intensity attenuation moves the right direction with alpha (mean intensity ratio 0.82 at
   alpha=0.003 near the minimum, 0.068 at the default alpha=0.06 -- more fog, more attenuation, as expected). One
   real thing `fog.py`'s wrapper must handle: **`simulate_fog` silently returns `float64`** even when given
   `float32` input (confirmed: output `aug.dtype` was `float64` for a `float32` input `pc`), so the wrapper must
   cast back to `float32` before returning -- the rest of this repo's pipeline (voxelisation, `.bin` writing)
   expects `float32` lidar and a silent dtype upcast would be the kind of bug that passes every shape check and
   then shows up as something unrelated later.

## 8. What ships in this phase, concretely

- `corruption/fog.py`: the `apply(data_dict, severity, rng)` wrapper around Hahner's simulator, once Q1 above is
  answered.
- `corruption/validate_fog.py`: runs the validation in section 5 and prints/logs the comparison; refuses to call
  the channel "validated" anywhere else in the repo until this has been run and passed against the agreed band.
  Modelled on `baseline/scripts/05_verify_conversion.py`'s pattern: loud pass/fail, specific numbers, no silent
  partial success.
- Tests (CPU where possible, same split as `models/model1/tests/test_eval_corruptions.py`): shape/dtype
  preservation, determinism per (frame, severity, seed), severity=0 is a no-op, severity-monotonic degradation
  (more severity should not *improve* detectability of the point cloud, checked on a synthetic cloud, not asserted
  on real AP).
- Not shipped this phase: snow, the four "ours" channels, the schedule harness (section 4), any of the four metrics
  in section 6. Each is its own follow-on phase once fog is validated.
