#!/usr/bin/env python3
"""Extract the left-camera TIFFs for every frame in the five STF splits.

Model 1 reads pixels, so it needs the real cam_stereo_left images that Phase 1
skipped (see models/model1/README.md). This script only reads; it reuses the
split-archive reader from baseline/scripts/01_extract_archives.py rather than
copying it, so both pipelines read STF's archives the same way.

The archive is an Info-Zip split archive (cam_stereo_left.z01..z25 + .zip). The
reader walks it sequentially, so every part must be present: a gap makes the
scan fail loudly instead of silently skipping frames. This script checks the
parts first and stops with the list of missing ones.

Stdlib only, Python 3.10+ (the baseline module uses `X | None` annotations).

Usage (from the repo root):
    python models/model1/data_prep/01_extract_camera.py \
        --stf-root data/stf --out data/model1/cam_stereo_left_tiff
"""

import argparse
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
NUM_PARTS = 25            # cam_stereo_left.z01 .. z25, then cam_stereo_left.zip
PART_SIZE = 1572864000    # every part except the final .zip


def load_baseline_extractor():
    """Import baseline/scripts/01_extract_archives.py (its name starts with a
    digit, so a plain import statement can't load it)."""
    path = REPO / "baseline" / "scripts" / "01_extract_archives.py"
    spec = importlib.util.spec_from_file_location("baseline_extract_archives", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_parts(archive_dir: Path) -> list[Path]:
    """Return the ordered part list, or exit listing every missing/short part."""
    parts = [archive_dir / f"cam_stereo_left.z{i:02d}" for i in range(1, NUM_PARTS + 1)]
    parts.append(archive_dir / "cam_stereo_left.zip")
    problems = []
    for p in parts:
        if not p.exists():
            problems.append(f"missing {p.name}")
        elif p.suffix != ".zip" and p.stat().st_size != PART_SIZE:
            problems.append(f"{p.name} is {p.stat().st_size} bytes, expected {PART_SIZE}")
    if problems:
        sys.exit("cam_stereo_left archive is incomplete:\n  " + "\n  ".join(problems))
    return parts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stf-root", type=Path, default=REPO / "data" / "stf")
    ap.add_argument("--splits-dir", type=Path, default=REPO / "baseline" / "splits")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "model1" / "cam_stereo_left_tiff")
    ap.add_argument("--check-only", action="store_true",
                    help="Walk the whole archive without writing anything (integrity check)")
    args = ap.parse_args()

    extractor = load_baseline_extractor()
    parts = check_parts(args.stf_root.resolve() / "cam_stereo_left")
    print(f"all {len(parts)} cam_stereo_left parts present")

    split_keys = {}
    for split_file in sorted(args.splits_dir.glob("*.txt")):
        keys = []
        for line in split_file.read_text().splitlines():
            if line.strip():
                seq, frame = line.strip().split(",")
                keys.append(f"{seq}_{frame}")
        split_keys[split_file.stem] = keys
    wanted = set().union(*split_keys.values())
    print(f"{len(wanted)} unique frames referenced by {len(split_keys)} split files")

    out = args.out.resolve()
    # An empty wanted set walks every local header and writes nothing: a cheap way
    # to prove the archive is readable end to end.
    n = extractor.extract_from_split_archive(parts, out, set() if args.check_only else wanted, ".tiff")
    if args.check_only:
        print("integrity check passed: archive walked end to end")
        return 0

    print(f"extracted {n} tiffs -> {out}")
    ok = True
    for split, keys in split_keys.items():
        missing = [k for k in keys if not (out / f"{k}.tiff").exists()]
        status = "OK" if not missing else "FAIL"
        print(f"  [{status}] {split}: {len(keys) - len(missing)}/{len(keys)} images")
        if missing:
            ok = False
            print(f"      first missing: {missing[:5]}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
