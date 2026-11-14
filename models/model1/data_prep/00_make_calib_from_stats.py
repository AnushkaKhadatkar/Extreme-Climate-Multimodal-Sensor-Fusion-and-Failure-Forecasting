#!/usr/bin/env python3
"""Recreate data/stf/calib_cam_stereo_left.json from the committed tonemap_stats.json.

02_build_image_2.py rectifies the raw left-camera images with K, D, R, P from STF's
calib_cam_stereo_left.json. That file ships with the full STF download, not with the
camera archive on its own. tonemap_stats.json (committed) already stores the same
K, D, R and P under "rectification" (written by `02_build_image_2.py stats`), so this
script writes the JSON back out from it. The values are copied, not re-estimated.

Not copied from a source: "width" and "height". They are set to 1920 x 1024, the size
02_build_image_2.py already requires (IMAGE_SIZE) and the size the baseline's P2
principal point (about 988, 508) implies. 02_build_image_2.py checks every output
image's size and fails on a mismatch, so a wrong value here cannot pass silently.

Cross-check, when the baseline conversion is present: P must equal the P2 row in every
baseline/data/kitti_format/training/calib/*.txt (they were written from the same file).

Usage (from the repo root; stdlib only):
    python models/model1/data_prep/00_make_calib_from_stats.py
"""

import argparse
import glob
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stats", type=Path, default=HERE / "tonemap_stats.json")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "stf" / "calib_cam_stereo_left.json")
    ap.add_argument("--kitti-calib-dir", type=Path, default=REPO / "baseline" / "data" / "kitti_format" / "training" / "calib")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1024)
    args = ap.parse_args()

    rect = json.loads(args.stats.read_text())["rectification"]
    calib = {"K": rect["K"], "D": rect["D"], "R": rect["R"], "P": rect["P"],
             "width": args.width, "height": args.height}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(calib, indent=2) + "\n")
    print(f"wrote {args.out}  (K 9, D {len(calib['D'])}, R 9, P 12 values; {args.width}x{args.height})")

    files = sorted(glob.glob(str(args.kitti_calib_dir / "*.txt")))
    if not files:
        print(f"no baseline calib files under {args.kitti_calib_dir}: P cross-check skipped")
        return 0
    bad = 0
    for f in files:
        for line in Path(f).read_text().splitlines():
            if line.startswith("P2:"):
                p2 = [float(x) for x in line.split()[1:]]
                if max(abs(a - b) for a, b in zip(p2, calib["P"])) > 1e-6:
                    bad += 1
    print(f"P cross-check: {len(files) - bad}/{len(files)} baseline calib files have P2 == P")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
