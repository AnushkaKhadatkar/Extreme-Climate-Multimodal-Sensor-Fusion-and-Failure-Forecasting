#!/usr/bin/env python3
"""Extract the STF archives needed for Phase 1 (lidar + labels + calibration only).

Phase 1 does not need camera, gated NIR, radar, FIR, weather-station or
road-friction data -- see context.md section 7. This script only touches the
archives that feed the baseline reproduction:

  - lidar_hdl64_strongest  (split zip: lidar_hdl64_strongest.z01..z06 + .zip)
  - gt_labels              (two plain zips: cam_left_labels_TMP, gated_labels_TMPv2)
  - calib_*.json           (already extracted, not archived)
  - calib_tf_tree_full.json (already extracted, not archived)

Why this script exists instead of just running `unzip`: the multi-part archives
(lidar_hdl64_strongest.z01, .z02, ..., .zip) are Info-Zip "split archives", not
independently valid zips. The trailing `.zip` part holds the central directory but
is not byte 0 of the archive, so plain `unzip lidar_hdl64_strongest.zip` fails.
The fix is `zip -FF <the .zip part> --out <merged.zip>`, which Info-Zip's `zip`
(shipped on macOS) can do directly -- confirmed against this exact dataset's
archives, no 7z or py7zr dependency needed.

We also extract selectively: only the frames referenced by baseline/splits/*.txt,
not the full ~13k-scene archive, to avoid holding two copies of camera-scale data
on disk. (Lidar-only archives are cheap either way -- ~8.8GB total -- but this
keeps the code path identical for the camera archives we'll need in Phase 2.)

Usage:
    python 01_extract_archives.py --stf-root ../../data/stf --out-root ../data/kitti_source
    python 01_extract_archives.py --stf-root ../../data/stf --out-root ../data/kitti_source --full
        (--full extracts everything in the archive instead of just split-listed frames;
         useful for gt_labels since it's tiny and label lookups are easier unfiltered)
"""

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def load_frame_keys(splits_dir: Path) -> set[str]:
    """Return the set of 'sequence_frameid' keys (STF's own filename stem format)
    referenced by all five split files, e.g. '2019-09-11_18-41-27_02580'."""
    keys = set()
    for split_file in splits_dir.glob("*.txt"):
        for line in split_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            seq, frame = line.split(",")
            keys.add(f"{seq}_{frame}")
    return keys


def merge_split_zip(archive_dir: Path, basename: str, out_dir: Path) -> Path:
    """Merge <basename>.z01..zNN + <basename>.zip into a single valid zip via
    `zip -FF`. Returns path to the merged zip. Skips the merge if a merged copy
    already exists (idempotent -- this step is the expensive one)."""
    merged = out_dir / f"{basename}.merged.zip"
    if merged.exists():
        print(f"  [skip] {merged.name} already exists")
        return merged

    zip_part = archive_dir / f"{basename}.zip"
    if not zip_part.exists():
        raise FileNotFoundError(f"expected {zip_part} (the final split part)")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  merging {basename}.z01..zip -> {merged.name} (this reads all parts, be patient)")
    # zip -FF prompts for confirmation on some versions; -y / printf 'y\n' handles that.
    proc = subprocess.run(
        ["zip", "-FF", str(zip_part), "--out", str(merged)],
        input="y\n",
        capture_output=True,
        text=True,
    )
    if proc.returncode not in (0, 1):  # zip -FF often exits 1 even on success (warnings)
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:], file=sys.stderr)
        raise RuntimeError(f"zip -FF failed for {basename} (exit {proc.returncode})")
    if not merged.exists():
        raise RuntimeError(f"zip -FF reported success but {merged} was not created")
    return merged


def extract_selected(archive_path: Path, dest_dir: Path, wanted_keys: set[str] | None, ext: str) -> int:
    """Extract members from a zip archive into dest_dir (flat, no internal
    subdirectories kept). If wanted_keys is None, extract everything; else only
    members whose stem is in wanted_keys. Returns count of files extracted.

    Uses Python's zipfile directly rather than shelling out to `unzip`: macOS's
    stock Info-ZIP unzip (6.00) has no stdin-filelist flag (`-@` is a GNU
    extension), and passing thousands of names as argv risks the shell's arg
    length limit. zipfile.open() has neither problem and streams member-by-member
    so we never hold the whole (multi-GB, uncompressed) archive in memory."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path) as zf:
        members = [m for m in zf.namelist() if m.endswith(ext)]
        if wanted_keys is not None:
            members = [m for m in members if Path(m).stem in wanted_keys]

        if not members:
            print(f"  [warn] no matching members found in {archive_path.name}")
            return 0

        print(f"  extracting {len(members)} member(s) from {archive_path.name} -> {dest_dir}")
        for i, member in enumerate(members, 1):
            target = dest_dir / Path(member).name  # flatten: drop internal dir structure
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            if i % 1000 == 0:
                print(f"    ...{i}/{len(members)}")

    return len(members)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stf-root", type=Path, default=Path("../../data/stf"),
                     help="Root of the downloaded STF archives (data/stf)")
    ap.add_argument("--out-root", type=Path, default=Path("../data/kitti_source"),
                     help="Where to put extracted raw files, pre-KITTI-conversion")
    ap.add_argument("--splits-dir", type=Path, default=Path("../splits"))
    ap.add_argument("--full", action="store_true",
                     help="Extract every frame in the archive, not just split-listed ones")
    ap.add_argument("--tmp-dir", type=Path, default=Path("../data/_merged_zips"),
                     help="Where to stage merged split-zip files (kept for idempotency)")
    args = ap.parse_args()

    stf_root = args.stf_root.resolve()
    out_root = args.out_root.resolve()
    splits_dir = args.splits_dir.resolve()
    tmp_dir = args.tmp_dir.resolve()

    wanted_keys = None if args.full else load_frame_keys(splits_dir)
    if wanted_keys is not None:
        print(f"Loaded {len(wanted_keys)} unique frame keys from {splits_dir}")

    # 1) lidar_hdl64_strongest (split zip)
    print("\n=== lidar_hdl64_strongest ===")
    lidar_dir = stf_root / "lidar_hdl64_strongest"
    merged = merge_split_zip(lidar_dir, "lidar_hdl64_strongest", tmp_dir)
    n = extract_selected(merged, out_root / "lidar_hdl64_strongest", wanted_keys, ".bin")
    print(f"  -> {n} .bin files in {out_root / 'lidar_hdl64_strongest'}")

    # 2) gt_labels (two plain zips)
    print("\n=== gt_labels ===")
    for zip_name, subdir in [
        ("cam_left_labels_TMP.zip", "cam_left_labels_TMP"),
        ("gated_labels_TMPv2.zip", "gated_labels_TMP"),
    ]:
        zip_path = stf_root / "gt_labels" / zip_name
        n = extract_selected(zip_path, out_root / "gt_labels" / subdir, wanted_keys, ".txt")
        print(f"  -> {n} .txt files in {out_root / 'gt_labels' / subdir}")

    # 3) calibration -- not archived, just copy
    print("\n=== calibration (copy, not archived) ===")
    calib_out = out_root / "calib"
    calib_out.mkdir(parents=True, exist_ok=True)
    for calib_file in ["calib_cam_stereo_left.json", "calib_cam_stereo_right.json",
                        "calib_gated_bwv.json", "calib_tf_tree_full.json"]:
        src = stf_root / calib_file
        if src.exists():
            shutil.copy2(src, calib_out / calib_file)
            print(f"  copied {calib_file}")
        else:
            print(f"  [warn] {calib_file} not found at {src}")

    print("\nDone. Next: 02_convert_to_kitti.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
