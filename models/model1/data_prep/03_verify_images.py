#!/usr/bin/env python3
"""Checks on Model 1's real image_2 before anything trains on it.

  1. Coverage: every frame in all five ImageSets has image_2/<id>.png at 1920x1024.
  2. Lidar-image alignment, measured, not eyeballed: project lidar points into
     the image, find lidar depth edges (neighbouring points more than 1.5 m apart),
     and score how strong the image gradient is under them. Repeat with the
     projection shifted by up to +/-24 px. If calibration and image are aligned
     the score peaks at (or very near) zero shift. A wrong image, a wrong frame
     id or a wrong extrinsic moves or flattens the peak.
  3. Contrast per split: mean per-image standard deviation of grey levels.
     With the fixed global tone map, dense fog should show LOWER contrast than
     clear. If it doesn't, the tone map is normalising fog away (the D6 problem).
  4. Overlays: a few frames per split with lidar points drawn on the image,
     saved for a visual check.

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
MAX_SHIFT_PX = 8          # full-resolution pixels; alignment peak must be this close to 0
SHIFTS = range(-12, 13, 2)  # half-resolution pixels (+/-24 px at full resolution)


def read_ids(kitti: Path, split: str) -> list[str]:
    return [l.strip() for l in (kitti / "ImageSets" / f"{split}.txt").read_text().splitlines() if l.strip()]


def read_calib(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = {}
    for line in path.read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            c[k.strip()] = np.array([float(x) for x in v.split()])
    return c["P2"].reshape(3, 4), c["R0_rect"].reshape(3, 3), c["Tr_velo_to_cam"].reshape(3, 4)


def project(points: np.ndarray, P2, R0, Tr) -> tuple[np.ndarray, np.ndarray]:
    """Lidar xyz (N,3) -> pixel uv (M,2) and depth (M,) for points in front of the camera."""
    pts_h = np.hstack([points, np.ones((len(points), 1))])
    cam = (R0 @ (Tr @ pts_h.T)).T
    keep = cam[:, 2] > 0.5
    cam = cam[keep]
    img = (P2 @ np.hstack([cam, np.ones((len(cam), 1))]).T).T
    uv = img[:, :2] / img[:, 2:3]
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < IMAGE_SIZE[0]) & (uv[:, 1] >= 0) & (uv[:, 1] < IMAGE_SIZE[1])
    return uv[inside], cam[inside, 2]


def check_coverage(kitti: Path, image_dir: Path) -> bool:
    print("=== 1. image_2 coverage and size ===")
    ok = True
    for split in SPLITS:
        ids = read_ids(kitti, split)
        missing, wrong = 0, 0
        for kid in ids:
            p = image_dir / f"{kid}.png"
            if not p.exists():
                missing += 1
                continue
            with Image.open(p) as im:  # reads the header only
                if im.size != IMAGE_SIZE:
                    wrong += 1
        good = missing == 0 and wrong == 0
        ok &= good
        print(f"  [{'OK' if good else 'FAIL'}] {split}: {len(ids) - missing}/{len(ids)} present, {wrong} wrong size")
    return ok


def alignment_offset(img: np.ndarray, uv: np.ndarray, depth: np.ndarray) -> tuple[int, int, float]:
    """Best (dx, dy) in full-res pixels and its score relative to the zero-shift score."""
    grey = img.mean(axis=2)[::2, ::2]               # half resolution
    gy, gx = np.gradient(grey)
    grad = np.hypot(gx, gy)
    h, w = grad.shape
    u = (uv[:, 0] / 2).astype(int)
    v = (uv[:, 1] / 2).astype(int)
    D = np.full((h, w), np.nan)
    order = np.argsort(-depth)                      # nearest point wins each pixel
    D[v[order], u[order]] = depth[order]

    # A pixel is a depth edge if a valid pixel within 3 px to its right is >1.5 m closer/farther.
    edge = np.zeros_like(D, dtype=bool)
    for s in (1, 2, 3):
        a, b = D[:, :-s], D[:, s:]
        jump = np.abs(a - b) > 1.5
        jump &= ~np.isnan(a) & ~np.isnan(b)
        edge[:, :-s] |= jump
    ev, eu = np.nonzero(edge)
    if len(ev) < 50:
        return 0, 0, float("nan")

    scores = {}
    for dy in SHIFTS:
        for dx in SHIFTS:
            vv, uu = ev + dy, eu + dx
            m = (vv >= 0) & (vv < h) & (uu >= 0) & (uu < w)
            scores[(dx, dy)] = grad[vv[m], uu[m]].mean()
    (bx, by), best = max(scores.items(), key=lambda kv: kv[1])
    return bx * 2, by * 2, best / max(scores[(0, 0)], 1e-9)


def check_alignment_and_contrast(kitti: Path, image_dir: Path, sanity_dir: Path,
                                 n_per_split: int, n_overlays: int) -> bool:
    print(f"\n=== 2-4. alignment ({n_per_split} frames/split), contrast, overlays ===")
    sanity_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    all_dx, all_dy = [], []
    contrast = {}
    for split in SPLITS:
        ids = rng.sample(read_ids(kitti, split), n_per_split)
        stds, dxs, dys = [], [], []
        for i, kid in enumerate(ids):
            img = np.asarray(Image.open(image_dir / f"{kid}.png").convert("RGB")).astype(np.float32)
            stds.append(img.mean(axis=2).std())
            pts = np.fromfile(kitti / "training" / "velodyne" / f"{kid}.bin", dtype=np.float32).reshape(-1, 4)
            uv, depth = project(pts[:, :3], *read_calib(kitti / "training" / "calib" / f"{kid}.txt"))
            dx, dy, gain = alignment_offset(img, uv, depth)
            if not np.isnan(gain):
                dxs.append(dx)
                dys.append(dy)
            if i < n_overlays:
                save_overlay(img, uv, depth, sanity_dir / f"{split}_{kid}.png")
        contrast[split] = float(np.mean(stds))
        all_dx += dxs
        all_dy += dys
        print(f"  {split}: median best shift dx={np.median(dxs):+.0f} dy={np.median(dys):+.0f} px "
              f"({len(dxs)} frames with enough depth edges), mean grey std {contrast[split]:.1f}")

    mdx, mdy = float(np.median(all_dx)), float(np.median(all_dy))
    aligned = abs(mdx) <= MAX_SHIFT_PX and abs(mdy) <= MAX_SHIFT_PX
    print(f"  [{'OK' if aligned else 'FAIL'}] overall median best shift ({mdx:+.0f}, {mdy:+.0f}) px, "
          f"limit +/-{MAX_SHIFT_PX} px")
    fog_kept = contrast["test_dense_fog"] < contrast["test_clear"]
    print(f"  [{'OK' if fog_kept else 'FAIL'}] dense-fog contrast {contrast['test_dense_fog']:.1f} "
          f"< clear contrast {contrast['test_clear']:.1f} (fixed tone map keeps fog's contrast loss)")
    print(f"  overlays: {sanity_dir}")
    return aligned and fog_kept


def save_overlay(img: np.ndarray, uv: np.ndarray, depth: np.ndarray, path: Path) -> None:
    out = img.copy()
    t = np.clip(depth / 60.0, 0, 1)                 # near = red, far = blue
    colour = np.stack([255 * (1 - t), 80 * np.ones_like(t), 255 * t], axis=1)
    u, v = uv[:, 0].astype(int), uv[:, 1].astype(int)
    for du in (0, 1):
        for dv in (0, 1):
            uu, vv = np.clip(u + du, 0, IMAGE_SIZE[0] - 1), np.clip(v + dv, 0, IMAGE_SIZE[1] - 1)
            out[vv, uu] = colour
    Image.fromarray(out.astype(np.uint8)).resize((960, 512)).save(path)


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
        check_alignment_and_contrast(kitti, image_dir, args.sanity_dir.resolve(),
                                     args.frames_per_split, args.overlays_per_split),
    ]
    print(f"\n=== {'ALL CHECKS PASSED' if all(results) else 'SOME CHECKS FAILED -- see above'} ===")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
