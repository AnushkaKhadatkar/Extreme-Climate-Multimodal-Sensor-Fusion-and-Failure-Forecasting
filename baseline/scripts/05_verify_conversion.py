#!/usr/bin/env python3
"""Sanity checks on the converted KITTI-format data, runnable on CPU (no
OpenPCDet/CUDA needed). Run this before shipping data/kitti_format/ to a GPU
box -- catching a conversion bug here is free; catching it after an 80-epoch
training run is not.

Checks:
  1. Split counts match context.md section 2 exactly.
  2. Every ImageSets id has all four files (velodyne, label_2, calib, image_2).
  3. Lidar bins are (N,4) float32 and non-empty.
  4. Calib matrices parse to the right shapes.
  5. Label class histogram (flags if remap looks off, e.g. zero Cars).
  6. A projection spot-check: for a handful of random frames, count how many
     non-DontCare 3D boxes with z>0 (in front of camera) project inside the
     image bounds. A badly wrong calibration produces ~0%; a correct one
     produces something close to 100% (a legitimately visible ground-truth box
     should, by construction, be in the camera's field of view -- STF's labels
     come from cam_left_labels_TMP, i.e. camera-visible annotations).
     Only exercises P2 -- see 7.
  7. Lidar->camera extrinsic: fraction of real GT boxes (within 40 m) that
     contain at least one lidar point after Tr_velo_to_cam + R0_rect.
  8. Median IoU between each projected 3D box's 2D extent and the label's own
     2D bbox -- catches calibration errors check 6 is too loose to see.

Usage:
    python 05_verify_conversion.py --kitti-format ../data/kitti_format
"""

import argparse
import random
from pathlib import Path

import numpy as np

EXPECTED_COUNTS = {
    "train": 3469,
    "val": 781,
    "test_clear": 1847,
    "test_light_fog": 946,
    "test_dense_fog": 786,
}

# All five splits live under 'training/' -- see 02_convert_to_kitti.py's
# module docstring / REPRODUCTION_PLAN.md section 2 for why (OpenPCDet's
# KittiDataset hard-codes 'testing/' only for a split literally named 'test',
# which none of our three named test splits are).
SPLIT_PHASE = {
    "train": "training", "val": "training",
    "test_clear": "training", "test_light_fog": "training", "test_dense_fog": "training",
}


def check_split_counts(kitti_format: Path) -> bool:
    print("=== 1. Split counts vs context.md section 2 ===")
    ok = True
    for name, expected in EXPECTED_COUNTS.items():
        f = kitti_format / "ImageSets" / f"{name}.txt"
        if not f.exists():
            print(f"  [FAIL] {name}.txt missing")
            ok = False
            continue
        n = len([l for l in f.read_text().splitlines() if l.strip()])
        status = "OK" if n == expected else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"  [{status}] {name}: {n} (expected {expected})")
    return ok


def check_files_present(kitti_format: Path) -> bool:
    print("\n=== 2. Per-frame file completeness ===")
    ok = True
    for split, phase in SPLIT_PHASE.items():
        f = kitti_format / "ImageSets" / f"{split}.txt"
        ids = [l.strip() for l in f.read_text().splitlines() if l.strip()]
        missing = {"velodyne": 0, "label_2": 0, "calib": 0, "image_2": 0}
        for kid in ids:
            for sub, ext in [("velodyne", "bin"), ("label_2", "txt"), ("calib", "txt"), ("image_2", "png")]:
                if not (kitti_format / phase / sub / f"{kid}.{ext}").exists():
                    missing[sub] += 1
        n_missing = sum(missing.values())
        status = "OK" if n_missing == 0 else "FAIL"
        print(f"  [{status}] {split}: {missing}")
        if n_missing:
            ok = False
    return ok


def check_lidar_shapes(kitti_format: Path, n_sample: int = 50) -> bool:
    print(f"\n=== 3. LiDAR bin shape check ({n_sample} random samples) ===")
    all_bins = list((kitti_format / "training" / "velodyne").glob("*.bin")) + \
               list((kitti_format / "testing" / "velodyne").glob("*.bin"))
    if not all_bins:
        print("  [FAIL] no .bin files found under training/velodyne or testing/velodyne "
              "-- data is not present, not just unsampled")
        return False
    ok = True
    for b in random.sample(all_bins, min(n_sample, len(all_bins))):
        scan = np.fromfile(b, dtype=np.float32)
        if scan.size % 4 != 0:
            print(f"  [FAIL] {b.name}: {scan.size} floats not divisible by 4")
            ok = False
        elif scan.size == 0:
            print(f"  [FAIL] {b.name}: empty")
            ok = False
    if ok:
        print(f"  [OK] all {min(n_sample, len(all_bins))} sampled bins are valid (N,4) float32")
    return ok


def check_calib(kitti_format: Path) -> bool:
    print("\n=== 4. Calibration matrix shapes ===")
    calib_file = kitti_format / "training" / "calib" / "000000.txt"
    if not calib_file.exists():
        print(f"  [FAIL] {calib_file} does not exist")
        return False
    calib = {}
    for line in calib_file.read_text().splitlines():
        key, vals = line.split(":", 1)
        calib[key] = np.array([float(v) for v in vals.split()])
    ok = True
    for key, expected_len in [("P2", 12), ("R0_rect", 9), ("Tr_velo_to_cam", 12)]:
        n = len(calib.get(key, []))
        status = "OK" if n == expected_len else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"  [{status}] {key}: {n} values (expected {expected_len})")
    return ok


def check_class_histogram(kitti_format: Path) -> bool:
    print("\n=== 5. Label class histogram (train split) ===")
    ids = [l.strip() for l in (kitti_format / "ImageSets" / "train.txt").read_text().splitlines() if l.strip()]
    hist = {}
    missing = 0
    for kid in ids:
        label_file = kitti_format / "training" / "label_2" / f"{kid}.txt"
        if not label_file.exists():
            missing += 1
            continue
        for line in label_file.read_text().splitlines():
            cls = line.split(" ")[0]
            hist[cls] = hist.get(cls, 0) + 1
    if missing:
        print(f"  [FAIL] {missing}/{len(ids)} label files missing -- cannot build a real histogram")
        return False
    for cls, n in sorted(hist.items(), key=lambda kv: -kv[1]):
        print(f"  {cls}: {n}")
    ok = hist.get("Car", 0) > 0 and hist.get("Pedestrian", 0) > 0
    if not ok:
        print("  [FAIL] expected nonzero Car and Pedestrian counts")
    else:
        print("  [OK] Car and Pedestrian both present in nonzero counts")
    return ok


def check_projection(kitti_format: Path, n_frames: int = 20) -> bool:
    print(f"\n=== 6. Label-to-image projection spot-check ({n_frames} random train frames) ===")
    ids = [l.strip() for l in (kitti_format / "ImageSets" / "train.txt").read_text().splitlines() if l.strip()]
    sample_ids = random.sample(ids, min(n_frames, len(ids)))

    def box3d_corners(h, w, l, x, y, z, ry):
        xc = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
        yc = [0, 0, 0, 0, -h, -h, -h, -h]
        zc = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
        corners = np.array([xc, yc, zc])
        R = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        return R @ corners + np.array([[x], [y], [z]])

    total_boxes = 0
    total_in_bounds = 0
    missing = 0
    for kid in sample_ids:
        calib_file = kitti_format / "training" / "calib" / f"{kid}.txt"
        label_file = kitti_format / "training" / "label_2" / f"{kid}.txt"
        if not calib_file.exists() or not label_file.exists():
            missing += 1
            continue

        calib = {}
        for line in calib_file.read_text().splitlines():
            key, vals = line.split(":", 1)
            calib[key] = np.array([float(v) for v in vals.split()])
        P2 = calib["P2"].reshape(3, 4)

        for line in label_file.read_text().splitlines():
            f = line.strip().split(" ")
            if not f or f[0] == "DontCare":
                continue
            h, w, l = float(f[8]), float(f[9]), float(f[10])
            x, y, z = float(f[11]), float(f[12]), float(f[13])
            ry = float(f[14])
            if z <= 0:
                continue
            total_boxes += 1
            corners = box3d_corners(h, w, l, x, y, z, ry)
            corners_h = np.vstack([corners, np.ones((1, 8))])
            proj = P2 @ corners_h
            proj = proj[:2] / proj[2]
            xs, ys = proj[0], proj[1]
            if xs.min() < 1920 and xs.max() > 0 and ys.min() < 1024 and ys.max() > 0:
                total_in_bounds += 1

    if missing:
        print(f"  [FAIL] {missing}/{len(sample_ids)} sampled frames had no calib/label file -- "
              "data is not present, cannot run the spot-check")
        return False

    frac = total_in_bounds / total_boxes if total_boxes else 0
    status = "OK" if frac > 0.9 else "FAIL"
    print(f"  [{status}] {total_in_bounds}/{total_boxes} in-front-of-camera boxes project into image bounds ({frac:.1%})")
    print("  (A badly wrong calibration produces near-0%; visually confirmed correct on frame "
          "000000 via data/sanity_check/000000_final_check.png during initial setup.)")
    print("  NOTE: this check only exercises P2. Labels are already in camera coordinates, so a "
          "wrong Tr_velo_to_cam passes it -- check 7 covers the extrinsic.")
    return frac > 0.9


def read_calib(calib_file: Path) -> dict[str, np.ndarray]:
    calib = {}
    for line in calib_file.read_text().splitlines():
        key, vals = line.split(":", 1)
        calib[key] = np.array([float(v) for v in vals.split()])
    return calib


def read_real_boxes(label_file: Path) -> list[tuple[str, list[float], list[float]]]:
    """(class, bbox2d x1y1x2y2, [h, w, l, x, y, z, ry]) for non-DontCare rows that
    carry a real 3D box (skips STF's dims -1 / location -1000 sentinel)."""
    out = []
    for line in label_file.read_text().splitlines():
        f = line.strip().split(" ")
        if len(f) < 15 or f[0] == "DontCare":
            continue
        box = [float(v) for v in f[8:15]]
        if min(box[:3]) <= 0 or box[5] <= -999:
            continue
        out.append((f[0], [float(v) for v in f[4:8]], box))
    return out


def box3d_corners_cam(h, w, l, x, y, z, ry) -> np.ndarray:
    """3x8 corners in KITTI camera coords (location = bottom centre, y down)."""
    xc = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    yc = [0, 0, 0, 0, -h, -h, -h, -h]
    zc = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    R = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    return R @ np.array([xc, yc, zc]) + np.array([[x], [y], [z]])


def check_extrinsic_points_in_boxes(kitti_format: Path, n_frames: int = 20, max_dist: float = 40.0) -> bool:
    """BASELINE_TASKS.md D2. Transform lidar points into the rectified camera frame
    with Tr_velo_to_cam and R0_rect -- the same path OpenPCDet uses to put labels
    into the lidar frame -- and count GT boxes within max_dist that contain at
    least one point. A wrong extrinsic leaves most boxes empty."""
    print(f"\n=== 7. Lidar->camera extrinsic: points inside GT boxes ({n_frames} random train frames, "
          f"boxes within {max_dist:.0f} m) ===")
    ids = [l.strip() for l in (kitti_format / "ImageSets" / "train.txt").read_text().splitlines() if l.strip()]
    n_boxes = n_hit = 0
    pts_per_box = []
    for kid in random.sample(ids, min(n_frames, len(ids))):
        calib = read_calib(kitti_format / "training" / "calib" / f"{kid}.txt")
        Tr = calib["Tr_velo_to_cam"].reshape(3, 4)
        R0 = calib["R0_rect"].reshape(3, 3)
        scan = np.fromfile(kitti_format / "training" / "velodyne" / f"{kid}.bin", dtype=np.float32).reshape(-1, 4)
        pts = R0 @ (Tr[:, :3] @ scan[:, :3].T + Tr[:, 3:4])  # 3xN, camera rect frame
        for _, _, (h, w, l, x, y, z, ry) in read_real_boxes(kitti_format / "training" / "label_2" / f"{kid}.txt"):
            if not 0 < z <= max_dist:
                continue
            # rotate points into the box frame: local = R(ry)^T (p - loc)
            c, s = np.cos(ry), np.sin(ry)
            d = pts - np.array([[x], [y], [z]])
            lx = c * d[0] - s * d[2]
            lz = s * d[0] + c * d[2]
            ly = d[1]
            inside = (np.abs(lx) <= l / 2) & (np.abs(lz) <= w / 2) & (ly <= 0) & (ly >= -h)
            n = int(inside.sum())
            n_boxes += 1
            n_hit += n > 0
            pts_per_box.append(n)
    if not n_boxes:
        print("  [FAIL] no real 3D boxes within range in the sample")
        return False
    frac = n_hit / n_boxes
    status = "OK" if frac > 0.9 else "FAIL"
    print(f"  [{status}] {n_hit}/{n_boxes} boxes contain >=1 lidar point ({frac:.1%}); "
          f"median points/box = {int(np.median(pts_per_box))}")
    return frac > 0.9


def check_projection_iou(kitti_format: Path, n_frames: int = 50) -> bool:
    """BASELINE_TASKS.md D3. Project each real 3D box with P2, take its clipped 2D
    extent, and compare with the label's own 2D bbox. Projected 3D hulls are a bit
    looser than tight 2D boxes, so expect a median IoU around 0.7-0.85, not 1.0."""
    print(f"\n=== 8. Projected-3D vs labelled-2D box IoU ({n_frames} random train frames) ===")
    from PIL import Image

    ids = [l.strip() for l in (kitti_format / "ImageSets" / "train.txt").read_text().splitlines() if l.strip()]
    ious = []
    for kid in random.sample(ids, min(n_frames, len(ids))):
        P2 = read_calib(kitti_format / "training" / "calib" / f"{kid}.txt")["P2"].reshape(3, 4)
        with Image.open(kitti_format / "training" / "image_2" / f"{kid}.png") as im:
            W, H = im.size
        for _, (x1, y1, x2, y2), (h, w, l, x, y, z, ry) in read_real_boxes(
                kitti_format / "training" / "label_2" / f"{kid}.txt"):
            corners = box3d_corners_cam(h, w, l, x, y, z, ry)
            if (corners[2] <= 0.1).any():
                continue
            proj = P2 @ np.vstack([corners, np.ones((1, 8))])
            u, v = proj[0] / proj[2], proj[1] / proj[2]
            px1, py1 = max(u.min(), 0), max(v.min(), 0)
            px2, py2 = min(u.max(), W - 1), min(v.max(), H - 1)
            iw, ih = max(0, min(px2, x2) - max(px1, x1)), max(0, min(py2, y2) - max(py1, y1))
            inter = iw * ih
            union = (px2 - px1) * (py2 - py1) + (x2 - x1) * (y2 - y1) - inter
            ious.append(inter / union if union > 0 else 0.0)
    if not ious:
        print("  [FAIL] no real 3D boxes in front of the camera in the sample")
        return False
    ious = np.array(ious)
    med = float(np.median(ious))
    status = "OK" if med > 0.5 else "FAIL"
    print(f"  [{status}] median IoU {med:.2f} over {len(ious)} boxes; "
          f"p10 {np.percentile(ious, 10):.2f}; {(ious < 0.3).mean():.1%} below 0.3")
    return med > 0.5


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kitti-format", type=Path, default=Path("../data/kitti_format"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    kitti_format = args.kitti_format.resolve()

    results = [
        check_split_counts(kitti_format),
        check_files_present(kitti_format),
        check_lidar_shapes(kitti_format),
        check_calib(kitti_format),
        check_class_histogram(kitti_format),
        check_projection(kitti_format),
        check_extrinsic_points_in_boxes(kitti_format),
        check_projection_iou(kitti_format),
    ]

    print(f"\n=== {'ALL CHECKS PASSED' if all(results) else 'SOME CHECKS FAILED -- see above'} ===")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
