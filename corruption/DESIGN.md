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

- `corruption/fog.py` -- **done.** `apply(data_dict, severity, rng, *, hard=True, soft=True, noise=10,
  noise_variant="v1")`. severity maps onto `ALPHAS_WITH_LUT` (the 9 shipped alphas, 0.005-0.2) by index, not
  interpolated onto the theoretical [0.003, 0.5] range -- see the module docstring and section 7 item 1 for why
  interpolating would silently snap to a different alpha than requested with no warning.
- **A second real bug found and fixed while writing this wrapper**, beyond the float32->float64 upcast in section
  7: `fog_simulation.py`'s soft-target (fog backscatter) noise draws from a **module-level global**
  `RNG = np.random.default_rng(seed=42)` set once at import time, not from any argument. Confirmed by reading the
  source (every `RNG.integers`/`RNG.uniform`/`RNG.beta` call targets that one shared object) and then proven with a
  deliberate before/after test: calling `fog.apply` for one frame, then for an unrelated frame, then for the first
  frame again with the identical seeded `rng` gave **different output** the second time (confirmed by breaking the
  fix and watching the test fail with a real `assert False`, not a crash) -- the fog added to a frame depended on
  how many other fog calls had happened earlier in the process, which is exactly the "deterministic per (frame,
  channel, severity, seed)" contract (section 3) silently violated. Fixed by reseeding that module global from the
  passed-in `rng` at the start of every call; `test_apply_is_deterministic_for_the_same_frame_seed_and_severity`
  is the regression test, written to fail without the fix (verified) and pass with it (verified).
- `corruption/tests/test_fog.py` -- 11 tests, CPU only (uses the real vendored simulator and LUTs, no mocking):
  severity->alpha mapping (always a shipped alpha, monotonic, hits both extremes), the determinism regression test
  above, cross-frame variation (determinism must not mean every frame gets an identical draw), dtype/shape
  preservation, images untouched, out-of-range severity raises, intensity attenuation direction (more severity ->
  lower mean intensity, hard-target only to isolate the effect from backscatter noise). All 23 tests in
  `corruption/tests/` (this file plus `test_base.py`) pass.
- `corruption/validate_fog.py` -- **not yet written.** Runs the validation in section 5 and prints/logs the
  comparison; refuses to call the channel "validated" anywhere else in the repo until this has been run and passed
  against the agreed band. Modelled on `baseline/scripts/05_verify_conversion.py`'s pattern: loud pass/fail,
  specific numbers, no silent partial success. **Blocked on section 7 item 3: the validation pass/fail band is a
  team decision, not written yet because nobody has agreed on it.** Do not run a validation claiming "passed" or
  "failed" before that band is agreed and recorded here.
- Not shipped this phase: snow, the four "ours" channels, the schedule harness (section 4), any of the four metrics
  in section 6. Each is its own follow-on phase once fog is validated.

## 9. Fog channel realism sweep (raw numbers only -- see section 7 item 3 and section 8: no pass/fail band exists yet)

Ran the baseline's trained checkpoint (`checkpoint_epoch_67.pth`, clear/moderate 76.59) through `corruption/fog.py`
on the clear test split (1,847 frames, never corrupted before this), one severity at a time, via
`corruption/eval_entry.py` (generalises Model 1's env-var hook pattern to any OpenPCDet detector without touching
`baseline/`). Real dense-fog moderate AP (`baseline/README.md`): **39.49**.

First pass, the 9 shipped LUT alphas (0.005-0.2):

| alpha | easy AP | moderate AP | hard AP |
| --- | --- | --- | --- |
| 0.005 | 76.84 | 76.01 | 72.86 |
| 0.01 | 75.77 | 74.63 | 71.41 |
| 0.02 | 72.14 | 70.49 | 66.80 |
| 0.03 | 66.77 | 64.76 | 60.73 |
| 0.06 | 43.63 | 41.68 | 38.95 |
| 0.1 | 23.70 | 22.37 | 19.92 |
| 0.12 | 17.98 | 16.78 | 14.68 |
| 0.15 | 12.40 | 11.38 | 9.74 |
| 0.2 | 7.49 | 6.76 | 5.59 |

alpha=0.06 moderate AP (41.68) came closest to the real dense-fog target (39.49), 2.19 AP over -- but the next
shipped alpha down, 0.03, moderate AP 64.76, is 25 AP away, and the next one up, 0.1, moderate AP 22.37, is 17 AP
away. The curve is steep through this region and the 9 shipped alphas are too coarse there to say more than "the
crossing point is near 0.06" -- which is why 4 more LUTs (0.04, 0.05, 0.07, 0.08) were generated (see git history,
`corruption/third_party/LiDAR_fog_sim/integral_lookup_tables/original/`) to refine around it.

Refined pass, the 5 new alphas bracketing 0.06:

| alpha | easy AP | moderate AP | hard AP | moderate AP vs. real (39.49) |
| --- | --- | --- | --- | --- |
| 0.04 | 57.25 | 55.97 | 52.16 | +16.48 |
| 0.05 | 50.53 | 48.44 | 45.54 | +8.95 |
| 0.06 | 43.63 | 41.68 | 38.95 | +2.19 |
| 0.07 | 39.21 | 38.27 | 34.59 | **-1.22** |
| 0.08 | 34.61 | 33.74 | 29.92 | -5.75 |

At this finer resolution, the real dense-fog moderate AP (39.49) falls **between alpha=0.06 and alpha=0.07**, closer
to 0.07 (undershoots by 1.22 AP) than 0.06 (overshoots by 2.19 AP) -- i.e. the single alpha value that best matches
real dense fog on this checkpoint is approximately **0.065-0.07**, not exactly 0.06 as the coarse sweep alone would
suggest. This is a raw observation, not a validation verdict: whether "within ~1-2 AP" counts as the channel being
realistic is exactly the pass/fail band section 7 item 3 says is a team decision, still not made. No alpha was
chosen or hardcoded as "the" fog severity based on this -- `fog.py`'s `severity_to_alpha` still maps evenly across
all 13 now-available alphas (see module docstring), not toward this crossing point specifically.

Logs: `fog_sweep_logs/fog_sev_*.log` (coarse) and `fog_sweep_logs/fog_refined_*.log` (refined), both gitignored
(run output, same convention as `baseline/output/`) -- numbers captured here instead.
