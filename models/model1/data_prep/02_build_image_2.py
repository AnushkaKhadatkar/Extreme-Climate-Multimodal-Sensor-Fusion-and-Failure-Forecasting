#!/usr/bin/env python3
"""Turn the raw left-camera TIFFs into 8-bit RGB KITTI image_2 PNGs with ONE
fixed global mapping.

The raw files are single-channel 16-bit Bayer mosaics, not grey images. In every
frame two of the four 2x2 positions have equal means (green) and the other two
differ (red and blue): a GRBG layout. OpenCV's code for it is
COLOR_BayerGB2RGB (OpenCV names the pattern from the second row/column). That
choice was checked on 12 frames across the dataset: its demosaiced R and B means
match the raw red/blue positions to within 7.6 summed counts, versus about 1,900
to 3,600 for the other three codes. Reading the mosaic as grey (as the
baseline's converter does; harmless there because the baseline uses
placeholder images) leaves a checkerboard and no colour.

The raw files are also UNRECTIFIED: calib_cam_stereo_left.json gives the raw
camera (K: f = 2612.9 px, strong barrel distortion D[0] = -0.568) and the
rectified projection P (f = 2355.7 px), and P is what every frame's calib.txt
uses as P2. Projecting with P onto raw pixels was measured off by a median 12 px
(03_verify_images.py), worse toward the edges. So each image is rectified with
that file's own K, D, R, P (cv2.initUndistortRectifyMap, the standard ROS
camera_info rectification), making pixels and P2 agree.

Steps per frame, all with constants computed ONCE:
  1. demosaic (COLOR_BayerGB2RGB)
  2. rectify (K, D, R -> P from calib_cam_stereo_left.json)
  3. white balance: fixed per-channel gains (grey-world over sampled TRAIN frames)
  4. tone map: fixed 1st/99th percentiles of the balanced train sample -> 0..255
Pixels that rectification maps from outside the sensor are black.

Why fixed and not per frame: the baseline's tonemap_and_convert stretches each
frame between its own percentiles, which undoes fog's contrast loss frame by
frame (BASELINE_TASKS.md D6) and would hide exactly the degradation Model 1's
camera reliability score has to notice. The constants live in
tonemap_stats.json (committed) so the mapping is reproducible.

PNG names are the KITTI ids from baseline/data/kitti_format/ImageSets/id_mapping.csv,
so image_2/<id>.png lines up with the baseline's velodyne/calib/label_2.

Usage (from the repo root; needs numpy, pillow, opencv, e.g. inside the baseline image):
    python models/model1/data_prep/02_build_image_2.py stats   # once, train split only
    python models/model1/data_prep/02_build_image_2.py build   # all 7,829 frames
"""

import argparse
import csv
import json
import random
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
STATS_FILE = HERE / "tonemap_stats.json"
ID_MAPPING = REPO / "baseline" / "data" / "kitti_format" / "ImageSets" / "id_mapping.csv"
IMAGE_SIZE = (1920, 1024)  # (width, height), matches the baseline's P2 / placeholders
BAYER_CODE = cv2.COLOR_BayerGB2RGB
CALIB_JSON = REPO / "data" / "stf" / "calib_cam_stereo_left.json"

_MAPS = {}  # per-process rectification maps (built once per worker)


def load_camera(calib_json: Path) -> dict:
    c = json.loads(Path(calib_json).read_text())
    return {"K": np.array(c["K"], np.float64).reshape(3, 3), "D": np.array(c["D"], np.float64),
            "R": np.array(c["R"], np.float64).reshape(3, 3), "P": np.array(c["P"], np.float64).reshape(3, 4),
            "size": (int(c["width"]), int(c["height"]))}


def init_rectify(calib_json: Path) -> None:
    cam = load_camera(calib_json)
    m1, m2 = cv2.initUndistortRectifyMap(cam["K"], cam["D"], cam["R"], cam["P"], cam["size"], cv2.CV_32FC1)
    ones = np.ones(cam["size"][::-1], np.float32)
    _MAPS.update(m1=m1, m2=m2, valid=cv2.remap(ones, m1, m2, cv2.INTER_LINEAR, borderValue=0) > 0.999)


def read_rgb(path: Path) -> np.ndarray:
    """Raw 16-bit Bayer TIFF -> rectified float32 (H, W, 3) RGB, in raw sensor units."""
    with Image.open(path) as img:
        raw = np.asarray(img).astype(np.uint16)
    rgb = cv2.cvtColor(raw, BAYER_CODE).astype(np.float32)
    return cv2.remap(rgb, _MAPS["m1"], _MAPS["m2"], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def load_mapping() -> list[dict]:
    with open(ID_MAPPING, newline="") as f:
        return list(csv.DictReader(f))


def compute_stats(tiff_dir: Path, n_frames: int, seed: int, calib_json: Path) -> dict:
    init_rectify(calib_json)
    valid = _MAPS["valid"][::8, ::8].reshape(-1)  # skip the black rectification border
    train = [r for r in load_mapping() if r["split"] == "train"]
    rng = random.Random(seed)
    sample = rng.sample(train, min(n_frames, len(train)))
    pixels = np.concatenate([
        read_rgb(tiff_dir / f"{r['stf_sequence']}_{r['stf_frame']}.tiff")[::8, ::8].reshape(-1, 3)[valid]
        for r in sample
    ])                                            # every 8th pixel keeps memory small
    means = pixels.mean(axis=0)
    gains = means[1] / means                      # grey world, green fixed at 1
    lo, hi = np.percentile(pixels * gains, [1, 99])
    cam = load_camera(calib_json)
    return {
        "bayer": "GRBG sensor layout, demosaiced with cv2.COLOR_BayerGB2RGB",
        "rectification": {"source": "calib_cam_stereo_left.json", "K": cam["K"].ravel().tolist(),
                          "D": cam["D"].tolist(), "R": cam["R"].ravel().tolist(), "P": cam["P"].ravel().tolist(),
                          "valid_pixel_fraction": float(_MAPS["valid"].mean())},
        "white_balance_gains_rgb": [float(g) for g in gains],
        "low": float(lo),
        "high": float(hi),
        "percentiles": [1, 99],
        "pooled_over": "all channels after white balance, every 8th pixel in each direction",
        "n_train_frames": len(sample),
        "seed": seed,
        "raw_channel_means_rgb": [float(m) for m in means],
    }


def tonemap(rgb: np.ndarray, gains, lo: float, hi: float) -> np.ndarray:
    out = np.clip((rgb * np.asarray(gains, np.float32) - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return (out * 255.0 + 0.5).astype(np.uint8)


def _convert_one(job) -> str | None:
    src, dst, gains, lo, hi = job
    if dst.exists():
        return None
    if not src.exists():
        return f"missing source {src.name}"
    img = Image.fromarray(tonemap(read_rgb(src), gains, lo, hi), mode="RGB")
    if img.size != IMAGE_SIZE:
        return f"{src.name}: size {img.size}, expected {IMAGE_SIZE}"
    tmp = dst.with_suffix(".tmp.png")
    img.save(tmp)
    tmp.replace(dst)  # never leave a half-written PNG under its final name
    return None


def build(tiff_dir: Path, out_dir: Path, workers: int, overwrite: bool, calib_json: Path) -> int:
    stats = json.loads(STATS_FILE.read_text())
    gains, lo, hi = stats["white_balance_gains_rgb"], stats["low"], stats["high"]
    out_dir.mkdir(parents=True, exist_ok=True)
    # Refuse to mix images made with different constants in one folder.
    marker = out_dir / "_tonemap_stats.json"
    if marker.exists() and json.loads(marker.read_text()) != stats and not overwrite:
        raise SystemExit(f"{out_dir} was built with different tonemap stats; rerun with --overwrite")
    if overwrite:
        for p in out_dir.glob("*.png"):
            p.unlink()
    marker.write_text(json.dumps(stats, indent=2))

    rows = load_mapping()
    jobs = [(tiff_dir / f"{r['stf_sequence']}_{r['stf_frame']}.tiff",
             out_dir / f"{r['kitti_id']}.png", gains, lo, hi) for r in rows]
    print(f"demosaic + rectify + gains {np.round(gains, 3).tolist()} + tone map [{lo:.1f}, {hi:.1f}]; {len(jobs)} frames")
    errors = []
    with Pool(workers, initializer=init_rectify, initargs=(calib_json,)) as pool:
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
    ap.add_argument("--overwrite", action="store_true", help="delete existing PNGs in --out first")
    ap.add_argument("--calib", type=Path, default=CALIB_JSON, help="STF calib_cam_stereo_left.json")
    args = ap.parse_args()

    if args.step == "stats":
        stats = compute_stats(args.tiff_dir.resolve(), args.n_frames, args.seed, args.calib.resolve())
        STATS_FILE.write_text(json.dumps(stats, indent=2) + "\n")
        print(json.dumps(stats, indent=2))
        print(f"wrote {STATS_FILE}")
        return 0
    return build(args.tiff_dir.resolve(), args.out.resolve(), args.workers, args.overwrite, args.calib.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
