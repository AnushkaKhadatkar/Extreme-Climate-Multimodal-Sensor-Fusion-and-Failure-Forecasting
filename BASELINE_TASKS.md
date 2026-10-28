# Baseline fix-up: task split (Dipin / Rohil)

Working plan for finishing the PV-RCNN † (α=0) baseline reproduction. Written to be
read by a coding agent as well as by people. **Read `context.md` first** — its
protocol invariants (§6) and working rules (§10) override anything here.

Gate (unchanged): Car 3D AP, R40, IoU 0.5, clear/moderate = **77.05 ± 2**.

---

## 1. Why this work exists

A code review of `baseline/` (no data or checkpoints available at the time) found
problems that can make the gate number wrong regardless of how well the model trains.
Each task below cites the finding it addresses. None of them change detector
architecture, hyperparameters or `pv_rcnn.yaml` values — they fix the pipeline
around the detector, which `context.md` §10 permits.

---

## 2. Who owns what

Split by where the work runs: **Dipin** has the raw STF data on a local machine
(CPU-side conversion and checks); **Rohil** owns the GPU box (Docker, infos,
training, eval). If that assumption about machines is wrong, swap the columns,
not the task contents.

### File ownership (to avoid merge conflicts)

| Path | Owner |
| --- | --- |
| `baseline/scripts/01_extract_archives.py`, `02_convert_to_kitti.py`, `05_verify_conversion.py`, `06_data_statistics.py` | Dipin |
| `baseline/docs/REPRODUCTION_PLAN.md` | Dipin |
| `baseline/scripts/03_create_infos.py`, `04_train_and_eval.sh`, any new eval/selection script | Rohil |
| `baseline/configs/*.yaml`, `baseline/docker/*` | Rohil |
| `baseline/README.md` | Rohil (Dipin sends changes as PR comments or a note) |
| `BASELINE_TASKS.md` (this file) | Either — update the status table when a task lands |

If a task genuinely needs a file the other person owns, say so in the PR and keep
that change minimal.

---

## 3. Rohil's tasks (GPU box, training/eval side)

Branch: `rohil/baseline-eval-fixes` (create from `main`).

**R1 — Test evals currently run on val. Highest priority.**
`04_train_and_eval.sh` overrides only `DATA_CONFIG.DATA_SPLIT.test`. OpenPCDet's
`KittiDataset` loads frames and ground truth from `INFO_PATH[mode]`, which still
points at `kitti_infos_val.pkl`, so all three "test" logs report val numbers.
- Fix: in the per-split loop add
  `--set DATA_CONFIG.DATA_SPLIT.test "$split" DATA_CONFIG.INFO_PATH.test "kitti_infos_${split}.pkl"`.
- Verify: each log's "Generate label finished / total samples" line must show
  1847 / 946 / 786 frames, not 781.
- Then re-evaluate the **existing** best checkpoint (if an 80-epoch run already
  exists) on all three splits. No retraining needed for this.

**R2 — Remove dead config keys.** `EXTRA_TEST_SPLITS` / `EXTRA_TEST_INFO_PATH` in
`kitti_dataset_stf.yaml` are never read by OpenPCDet. Delete them and their
comment, and point the comment at the R1 mechanism instead.

**R3 — Make best-checkpoint selection reproducible.**
- Add `--max_ckpt_save_num 80` to training so epochs 1–50 are not discarded
  (default keeps only the last 30). If disk forbids it, document the limit instead.
- Add `--max_waiting_mins 0` to the `--eval_all` call.
- Write a small parser (`baseline/scripts/07_select_best_ckpt.py`) that reads the
  `--eval_all` text logs and picks the epoch with the best **val, Car,
  `AP_R40@0.70, 0.50, 0.50`, `3d AP`, moderate**. Do not use the tensorboard /
  `ret_dict` values — those only carry IoU 0.7 (see `baseline/README.md`).
- Have `04_train_and_eval.sh` call it instead of stopping for a manual `BEST_CKPT`
  (keep `BEST_CKPT` as an optional override).

**R4 — Fix the final hint in `04_train_and_eval.sh`.** It says grep
`'Car AP@0.50'`, which matches nothing. It should point at the
`Car AP_R40@0.70, 0.50, 0.50:` block, `3d AP` line, moderate column.

**R5 — Pin the Docker image.** `Dockerfile` clones OpenPCDet `master` with
`--depth 1` and leaves numba/easydict/etc unpinned, which breaks
`context.md` §6 "pin versions" and will eventually break the `n_sig == 11`
assertions in `patch_openpcdet.py`.
- Pin OpenPCDet to the exact commit SHA used for the existing run
  (`git -C /opt/OpenPCDet rev-parse HEAD` inside the current image).
- `pip freeze` the working image and pin every package.
- Keep `TORCH_CUDA_ARCH_LIST` but note in a comment that it targets the 4090 only.

**R6 — Record run-to-run spread.** `--fix_random_seed` does not make cuDNN /
spconv deterministic. Note this in the README next to the gate. Only spend GPU
hours on a second seed if the gate result lands near a band edge.

**R7 — README status drift.** `baseline/README.md` still says training is
not done and Docker is planned. Update to what is actually true, keeping
`context.md` §10's hedging rules (estimates stay estimates).

**R8 — After Dipin's D1 lands:** regenerate infos with `03_create_infos.py` on
the reconverted labels and re-evaluate the best checkpoint on all three test
splits. See §5 for why a retrain should not be needed.

---

## 4. Dipin's tasks (local machine, data/label side)

Branch: `dipin/baseline-data-checks` (create from `main`).

**D1 — Turn "no 3D box" objects into DontCare.** 33.8% of train
Car/Pedestrian/Cyclist rows carry STF's sentinel (dims −1, location −1000).
`02_convert_to_kitti.py` passes them through as real objects. In eval they stay
as ground truth no detection can match, capping recall.
- Remap them to `DontCare` in `convert_label` (keep the 2D box — KITTI eval uses it
  as an ignore region). Keep a flag to reproduce the old behaviour for comparison.
- Report counts per split and per class.
- This changes ground truth: record it as a protocol decision in
  `REPRODUCTION_PLAN.md`, not as tuning. If Hahner's dataset loader turns up
  publicly, check how it treats these rows.

**D2 — Verify the lidar→camera extrinsic.** Check 6 in `05_verify_conversion.py`
only exercises `P2`; `Tr_velo_to_cam` (what PV-RCNN actually depends on) is
never tested. Add a check: transform lidar points into the camera frame and
count points inside each real (non-sentinel) GT box. Near-all boxes within ~40 m
should contain points.

**D3 — Strengthen the projection check.** Compare each projected 3D box's 2D
extent with the label's own 2D bbox (fields 4–7) and report median IoU,
instead of "any corner inside a hard-coded 1920×1024".

**D4 — Audit the difficulty fields.** KITTI difficulty depends on `occluded`,
`truncated` and bbox height. Report their value distributions and the number of
Car GT per split per difficulty (after D1). If STF uses a different scale from
KITTI, every tier — and the moderate > easy inversion check — is wrong.

**D5 — Lidar audit.** Report intensity range (KITTI is 0–1; STF may be 0–255,
matters for the Phase 2 fog simulator), estimated ground height vs the
`[-3, 1]` z-range and −1.78 m car anchor, duplicate points, and what
distinguishes the ~2× point-count frames from the rest.

**D6 — Fixed tone-mapping (Phase 2 prep, low priority).** Per-frame percentile
stretching in `tonemap_and_convert` normalises away fog contrast loss — harmless
for lidar-only PV-RCNN, harmful once a model reads pixels. Add a fixed global
mapping option; do not change the default yet.

**D7 — Plan doc drift.** `REPRODUCTION_PLAN.md` says `id_mapping.csv` lives in
`baseline/splits/`; it is in `data/kitti_format/ImageSets/`. Also soften
"resolved with high confidence" on calibration until D2 passes.

**D8 — Hand-off.** After D1, reconvert labels and ship the new
`training/label_2/` to the GPU box (labels only; lidar/calib/images are
unchanged).

---

## 5. Order and dependencies

```
Rohil: R1 ──► re-eval existing ckpt (tells us how much of any gap R1 explained)
        R2, R3, R4, R5, R6, R7 in any order
Dipin:  D2, D3 (calibration confidence)   D4, D5 (audits)   D1 ──► D8 ──► Rohil R8
```

Why D1 should not need a retrain: the sentinel boxes sit at location −1000, so
OpenPCDet's `mask_points_and_boxes_outside_range` already drops them during
training, and they contain no points so `gt_sampling` filters them too. The
change mainly affects **eval** ground truth. Confirm by comparing per-class
training GT counts in `kitti_infos_train.pkl` before and after; retrain only if
they differ inside the point-cloud range.

Gate is read only after R1 + R3 + R8 are all in.

---

## 6. Rules for agents working on this

- Read `context.md` in full first. Never tune hyperparameters toward 77.05; if AP
  is off, check protocol (R11/R40, IoU, 3D vs BEV, split contents, difficulty).
- Work only on your own branch. Never push to `main` directly — open a PR.
- **Commit and push incrementally**: one task (or one coherent sub-step) per
  commit, push after each commit. Commit messages name the task ID, e.g.
  `R1: evaluate test splits on their own info files`.
- Stay inside your file-ownership column (§2).
- Do not edit `pv_rcnn_stf.yaml` model/optimisation values.
- Do not commit data, `.pkl` files, checkpoints or logs (see `.gitignore`).
- When a task changes what counts as ground truth or how a metric is read, write
  the definition into `REPRODUCTION_PLAN.md` / README in the same PR.
- Report results faithfully: if a check fails, say so with the output.
- Update the status table below when a task's PR merges.

---

## 7. Status

| Task | Owner | Status |
| --- | --- | --- |
| R1 test splits use own infos | Rohil | done, merged: smoke test evaluated 1847/946/786 frames |
| R2 dead config keys | Rohil | done, merged |
| R3 checkpoint selection | Rohil | done, merged (`07_read_eval_logs.py`) |
| R4 grep hint | Rohil | done, merged (automatic gate report) |
| R5 pin Docker | Rohil | done, merged: OpenPCDet 233f849, `requirements.lock.txt`, numba-cuda fix for the eval crash |
| R6 run-to-run spread | Rohil | done, merged (README note) |
| R7 README drift | Rohil | done, merged (+ `run_baseline_docker.ps1`) |
| R8 regen infos + re-eval after D1 | Rohil | done via 1-epoch smoke test: whole pipeline runs end to end |
| **Baseline run: 80 epochs from scratch + gate** | Dipin | **done: gate PASS**, clear/moderate 76.59 (epoch 67); results in `baseline/README.md` |
| Dense-fog gap vs paper (39.49 vs 46.00) | unassigned | todo (Linear 298-61) |
| Runner regenerated infos every run | Rohil | fix on `rohil/fix-runner-infos-check`, PR pending (Linear 298-62) |
| D1 sentinel → DontCare | Dipin | done on `dipin/baseline-data-checks`: 11,772 Car rows remapped (test_clear 3,130) |
| D2 extrinsic check | Dipin | done, passes: 61/61 boxes contain points |
| D3 2D IoU projection check | Dipin | done, passes: median IoU 0.89 |
| D4 difficulty-field audit | Dipin | done: KITTI scales, tiers valid |
| D5 lidar audit | Dipin | done: intensity 0–255, ground −1.95 m, 2× frames = one drive |
| D6 fixed tone-map option | Dipin | todo (Phase 2 prep) |
| D7 plan doc drift | Dipin | done |
| D8 ship reconverted data | Dipin | ready: `baseline/data/kitti_format/` (placeholder images, see REPRODUCTION_PLAN §6) |

### Hand-off notes for Rohil

- `ImageSets/*.txt` and frame IDs are unchanged, so existing checkpoints still apply.
- Copy the whole reconverted `baseline/data/kitti_format/training/` to the GPU box
  (labels changed; images are now shape-only placeholders), then run R8.
- Expect test GT Car counts to drop by the remapped rows above. That is the point of D1.
