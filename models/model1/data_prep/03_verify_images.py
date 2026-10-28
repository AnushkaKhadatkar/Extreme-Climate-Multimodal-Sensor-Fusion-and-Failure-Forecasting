#!/usr/bin/env python3
"""Checks on Model 1's real image_2 before anything trains on it.

  1. Coverage (pass/fail): every frame in all five ImageSets has
     image_2/<id>.png, 1920x1024, RGB.
  2. Lidar-image alignment (pass/fail), measured, not eyeballed. Lidar points
     are projected into the image. Along each scan line, consecutive points
     whose depth jumps by more than 1.5 m mark an object boundary (the nearer
     point). The image gradient under those boundary points is scored with the
     projection shifted by up to +/-24 px. If calibration and image agree, the
     score peaks at (or very near) zero shift. A wrong frame id, a wrong
     extrinsic or an unrectified image moves or flattens the peak. Frames with
     too few boundaries (e.g. dark night scenes) are skipped; the check fails
     as inconclusive if fewer than MIN_FRAMES frames could be measured.
  3. Brightness and contrast per split (information only, not pass/fail).
     Whole-image contrast mostly reflects scene content and day vs night (on the
     first run, clear train 53.3 vs clear test 40.4 grey-level std), so it cannot
     fairly test for fog by itself. The fixed tone map is guaranteed by
     construction in 02_build_image_2.py; these numbers are for the record.
  4. Overlays: a few frames per split with lidar points drawn on the image.

Needs the baseline conversion (velodyne/, calib/, ImageSets/) and numpy + pillow.

Usage (from the repo root):
    python models/model1/data_prep/03_verify_images.py
"""

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
SPLITS = ["train", "val", "test_clear", "test_light_fog", "test_dense_fog"]
IMAGE_SIZE = (1920, 1024)
MAX_SHIFT_PX = 8                     # full-resolution pixels
SHIFTS = range(-12, 13, 2)           # half-resolution pixels (+/-24 px full resolution)
MIN_EDGES = 30
MIN_FRAMES = 15


def read_ids(kitti: Path, split: str) -> list[str]:
    return [l.strip() for l in (kitti / "ImageSets" / f"{split}.txt").read_text().splitlines() if l.strip()]


def read_calib(path: Path):
    c = {}
    for line in path.read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            c[k.strip()] = np.array([float(x) for x in v.split()])
    return c["P2"].reshape(3, 4), c["R0_rect"].reshape(3, 3), c["Tr_velo_to_cam"].reshape(3, 4)


def project(points: np.ndarray, P2, R0, Tr):
    """Lidar xyz (N,3) -> pixel uv (M,2), depth (M,), and the kept points' original order index."""
    pts_h = np.hstack([points, np.ones((len(points), 1))])
    cam = (R0 @ (Tr @ pts_h.T)).T
    keep = np.nonzero(cam[:, 2] > 0.5)[0]
    cam = cam[keep]
    img = (P2 @ np.hstack([cam, np.ones((len(cam), 1))]).T).T
    uv = img[:, :2] / img[:, 2:3]
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < IMAGE_SIZE[0]) & (uv[:, 1] >= 0) & (uv[:, 1] < IMAGE_SIZE[1])
    return uv[inside], cam[inside, 2]


def check_coverage(kitti: Path, image_dir: Path) -> bool:
    print("=== 1. image_2 coverage, size and mode ===")
    ok = True
    for split in SPLITS:
        ids = read_ids(kitti, split)
        missing, wrong = 0, 0
        for kid in ids:
            p = image_dir / f"{kid}.png"
            if not p.exists():
                missing += 1
                continue
            with Image.open(p) as im:  # header only
                if im.size != IMAGE_SIZE or im.mode != "RGB":
                    wrong += 1
        good = missing == 0 and wrong == 0
        ok &= good
        print(f"  [{'OK' if good else 'FAIL'}] {split}: {len(ids) - missing}/{len(ids)} present, {wrong} wrong size/mode")
    return ok


def boundary_points(uv: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Nearer point of each depth jump > 1.5 m between neighbours on the same scan line (uv rows)."""
    row = np.round(uv[:, 1] / 3).astype(int)          # scan lines are ~3+ px apart in v
    order = np.lexsort((uv[:, 0], row))
    r, u, d = row[order], uv[order, 0], depth[order]
    same = (r[1:] == r[:-1]) & (u[1:] - u[:-1] < 25)
    jump = same & (np.abs(d[1:] - d[:-1]) > 1.5)
    near = np.where(d[:-1] < d[1:], order[:-1], order[1:])[jump]
    return uv[near]


def alignment_offset(img: np.ndarray, uv: np.ndarray, depth: np.ndarray):
    grey = img.mean(axis=2)[::2, ::2]
    gy, gx = np.gradient(grey)
    grad = np.hypot(gx, gy)
    h, w = grad.shape
    edges = boundary_points(uv, depth)
    if len(edges) < MIN_EDGES:
        return None
    eu, ev = (edges[:, 0] / 2).astype(int), (edges[:, 1] / 2).astype(int)
    scores = {}
    for dy in SHIFTS:
        for dx in SHIFTS:
            vv, uu = ev + dy, eu + dx
            m = (vv >= 0) & (vv < h) & (uu >= 0) & (uu < w)
            scores[(dx, dy)] = grad[vv[m], uu[m]].mean()
    (bx, by), best = max(scores.items(), key=lambda kv: kv[1])
    return bx * 2, by * 2, best / max(scores[(0, 0)], 1e-9)


def save_overlay(img: np.ndarray, uv: np.ndarray, depth: np.ndarray, path: Path) -> None:
    small = Image.fromarray(img.astype(np.uint8)).resize((960, 512))
    out = np.asarray(small).copy()
    t = np.clip(depth / 50.0, 0, 1)                    # near = red, far = blue
    colour = np.stack([255 * (1 - t), 255 * (1 - np.abs(2 * t - 1)), 255 * t], axis=1)
    u, v = (uv[:, 0] / 2).astype(int), (uv[:, 1] / 2).astype(int)
    for du in (-1, 0, 1):                              # 3x3 dots, drawn after resizing so they stay visible
        for dv in (-1, 0, 1):
            out[np.clip(v + dv, 0, 511), np.clip(u + du, 0, 959)] = colour
    Image.fromarray(out.astype(np.uint8)).save(path)


def check_alignment(kitti: Path, image_dir: Path, sanity_dir: Path, n_per_split: int, n_overlays: int) -> bool:
    print(f"\n=== 2-4. alignment, brightness/contrast, overlays ({n_per_split} frames/split) ===")
    sanity_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    all_dx, all_dy = [], []
    for split in SPLITS:
        ids = rng.sample(read_ids(kitti, split), n_per_split)
        dxs, dys, means, stds = [], [], [], []
        for i, kid in enumerate(ids):
            img = np.asarray(Image.open(image_dir / f"{kid}.png").convert("RGB")).astype(np.float32)
            grey = img.mean(axis=2)
            means.append(grey.mean())
            stds.append(grey.std())
            pts = np.fromfile(kitti / "training" / "velodyne" / f"{kid}.bin", dtype=np.float32).reshape(-1, 4)
            uv, depth = project(pts[:, :3], *read_calib(kitti / "training" / "calib" / f"{kid}.txt"))
            res = alignment_offset(img, uv, depth)
            if res is not None:
                dxs.append(res[0])
                dys.append(res[1])
            if i < n_overlays:
                save_overlay(img, uv, depth, sanity_dir / f"{split}_{kid}.png")
        all_dx += dxs
        all_dy += dys
        shift = f"dx={np.median(dxs):+.0f} dy={np.median(dys):+.0f} px" if dxs else "n/a"
        print(f"  {split}: best shift {shift} ({len(dxs)}/{n_per_split} frames measurable); "
              f"grey mean {np.mean(means):.1f}, std {np.mean(stds):.1f}, std/mean {np.mean(np.array(stds) / np.maximum(means, 1)):.2f}")

    if len(all_dx) < MIN_FRAMES:
        print(f"  [FAIL] only {len(all_dx)} frames measurable (need {MIN_FRAMES}): alignment inconclusive")
        return False
    mdx, mdy = float(np.median(all_dx)), float(np.median(all_dy))
    ok = abs(mdx) <= MAX_SHIFT_PX and abs(mdy) <= MAX_SHIFT_PX
    print(f"  [{'OK' if ok else 'FAIL'}] median best shift ({mdx:+.0f}, {mdy:+.0f}) px over {len(all_dx)} frames, "
          f"limit +/-{MAX_SHIFT_PX} px")
    print(f"  overlays: {sanity_dir}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kitti-format", type=Path, default=REPO / "baseline" / "data" / "kitti_format")
    ap.add_argument("--image-dir", type=Path, default=REPO / "data" / "model1" / "image_2")
    ap.add_argument("--sanity-dir", type=Path, default=REPO / "data" / "model1" / "sanity")
    ap.add_argument("--frames-per-split", type=int, default=20)
    ap.add_argument("--overlays-per-split", type=int, default=2)
    args = ap.parse_args()

    kitti, image_dir = args.kitti_format.resolve(), args.image_dir.resolve()
    results = [
        check_coverage(kitti, image_dir),
        check_alignment(kitti, image_dir, args.sanity_dir.resolve(), args.frames_per_split, args.overlays_per_split),
    ]
    print(f"\n=== {'ALL CHECKS PASSED' if all(results) else 'SOME CHECKS FAILED -- see above'} ===")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
