#!/usr/bin/env python3
"""Tone-map the raw left-camera TIFFs into 8-bit KITTI image_2 PNGs with ONE
fixed global mapping.

Why not the baseline's tonemap_and_convert: it stretches every frame between
that frame's own 1st/99th percentiles. That undoes fog's contrast loss frame by
frame (BASELINE_TASKS.md D6), which would hide exactly the degradation Model 1's
camera reliability score has to notice. Here the 1st/99th percentiles are
computed once, over a seeded sample of TRAIN frames only, saved to
tonemap_stats.json (committed, so the mapping is reproducible), and applied
unchanged to every frame of every split.

PNG names are the KITTI ids from baseline/data/kitti_format/ImageSets/id_mapping.csv,
so image_2/<id>.png lines up with the baseline's velodyne/calib/label_2.

Usage (from the repo root; needs numpy + pillow, e.g. inside the baseline image):
    python models/model1/data_prep/02_build_image_2.py stats   # once, train split only
    python models/model1/data_prep/02_build_image_2.py build   # all 7,829 frames
"""

import argparse
import csv
import json
import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
STATS_FILE = HERE / "tonemap_stats.json"
ID_MAPPING = REPO / "baseline" / "data" / "kitti_format" / "ImageSets" / "id_mapping.csv"
IMAGE_SIZE = (1920, 1024)  # (width, height), matches the baseline's P2 / placeholders


def read_tiff(path: Path) -> np.ndarray:
    """Same reader as the baseline (PIL), as float32 HxW[xC]."""
    with Image.open(path) as img:
        return np.asarray(img).astype(np.float32)


def load_mapping() -> list[dict]:
    with open(ID_MAPPING, newline="") as f:
        return list(csv.DictReader(f))


def compute_stats(tiff_dir: Path, n_frames: int, seed: int) -> dict:
    train = [r for r in load_mapping() if r["split"] == "train"]
    rng = random.Random(seed)
    sample = rng.sample(train, min(n_frames, len(train)))
    values = []
    for r in sample:
        arr = read_tiff(tiff_dir / f"{r['stf_sequence']}_{r['stf_frame']}.tiff")
        values.append(arr[::8, ::8].ravel())  # every 8th pixel keeps memory small
    pooled = np.concatenate(values)
    lo, hi = np.percentile(pooled, [1, 99])
    return {
        "low": float(lo),
        "high": float(hi),
        "percentiles": [1, 99],
        "pooled_over": "all channels, every 8th pixel in each direction",
        "n_train_frames": len(sample),
        "seed": seed,
        "source_dtype_range_seen": [float(pooled.min()), float(pooled.max())],
    }


def tonemap(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    out = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return (out * 255.0 + 0.5).astype(np.uint8)


def _convert_one(job) -> str | None:
    src, dst, lo, hi = job
    if dst.exists():
        return None
    if not src.exists():
        return f"missing source {src.name}"
    img = Image.fromarray(tonemap(read_tiff(src), lo, hi)).convert("RGB")
    if img.size != IMAGE_SIZE:
        return f"{src.name}: size {img.size}, expected {IMAGE_SIZE}"
    tmp = dst.with_suffix(".tmp.png")
    img.save(tmp)
    tmp.replace(dst)  # never leave a half-written PNG under its final name
    return None


def build(tiff_dir: Path, out_dir: Path, workers: int) -> int:
    stats = json.loads(STATS_FILE.read_text())
    lo, hi = stats["low"], stats["high"]
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_mapping()
    jobs = [(tiff_dir / f"{r['stf_sequence']}_{r['stf_frame']}.tiff",
             out_dir / f"{r['kitti_id']}.png", lo, hi) for r in rows]
    print(f"tone map: low={lo:.1f} high={hi:.1f} (from {STATS_FILE.name}); {len(jobs)} frames")
    errors = []
    with Pool(workers) as pool:
        for i, err in enumerate(pool.imap_unordered(_convert_one, jobs, chunksize=16), 1):
            if err:
                errors.append(err)
            if i % 500 == 0:
                print(f"  {i}/{len(jobs)}")
    for e in errors[:20]:
        print("  ERROR", e)
    print(f"done: {len(jobs) - len(errors)}/{len(jobs)} ok, {len(errors)} errors -> {out_dir}")
    return 0 if not errors else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["stats", "build"])
    ap.add_argument("--tiff-dir", type=Path, default=REPO / "data" / "model1" / "cam_stereo_left_tiff")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "model1" / "image_2")
    ap.add_argument("--n-frames", type=int, default=300, help="train frames sampled for the stats")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if args.step == "stats":
        stats = compute_stats(args.tiff_dir.resolve(), args.n_frames, args.seed)
        STATS_FILE.write_text(json.dumps(stats, indent=2) + "\n")
        print(json.dumps(stats, indent=2))
        print(f"wrote {STATS_FILE}")
        return 0
    return build(args.tiff_dir.resolve(), args.out.resolve(), args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
