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

The lidar component is handled by scanning local file headers sequentially
across the split parts directly (see MultiPartReader / extract_from_split_archive
below) rather than reconstructing a merged archive first. A merged-archive
approach (e.g. `zip -FF`, Info-Zip's own repair tool, shipped on macOS but not
Windows) needs admin rights to install cross-platform, and still isn't free of
work afterward: central-directory offsets in a split archive are relative to
the start of whichever disk they point into, and translating those to a single
merged stream's absolute offsets means patching every entry (and, past the
4GB mark, inserting zip64 extra fields for entries that didn't have them).
Scanning local headers directly sidesteps all of that -- confirmed by
inspection that this archive's entries use general-purpose bit 3 = 0, i.e.
sizes are known upfront in each local header, so no lookahead or central
directory is needed to walk the member list correctly.

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
import struct
import sys
import zipfile
import zlib
from pathlib import Path


class MultiPartReader:
    """Presents a sequence of Info-Zip split-archive parts (basename.z01,
    .z02, ..., basename.zip) as one continuous byte stream, without
    physically concatenating them to disk first.

    Why this exists instead of shelling out to `zip -FF`: that's a macOS/Linux
    Info-Zip tool with no Windows equivalent pre-installed, and this machine
    has no admin access to install one (7-Zip's installer needs an elevation
    prompt nobody here can click). Info-Zip split archives are a plain byte
    stream cut at fixed-size boundaries, with one wrinkle: the very first part
    is prefixed by a 4-byte legacy spanning marker (``PK\\x07\\x08``, i.e.
    0x08074b50 little-endian) that is not part of the logical archive and must
    be skipped. Every other part starts exactly where the previous one left
    off -- no marker, no trimming. (Verified against this exact dataset's
    archive by hand: byte 0 of part 1 is that exact 4-byte signature; parts
    2..N start with high-entropy compressed data, not a signature.)

    Only forward, sequential reads are needed here (we scan local file
    headers start to finish), so this only implements read(); no seek-back.
    """

    def __init__(self, parts: list[Path]):
        self._parts = []  # list of (path, start_offset_within_part, size_after_trim)
        for i, p in enumerate(parts):
            size = p.stat().st_size
            start = 4 if i == 0 else 0  # skip the spanning marker on part 1 only
            self._parts.append((p, start, size - start))
        self.total_size = sum(sz for _, _, sz in self._parts)
        self._part_idx = 0
        self._fh = None
        self._pos_in_part = 0
        self._open_part(0)

    def _open_part(self, idx: int):
        if self._fh is not None:
            self._fh.close()
        path, start, _ = self._parts[idx]
        self._fh = open(path, "rb")
        self._fh.seek(start)
        self._part_idx = idx
        self._pos_in_part = 0

    def read(self, n: int) -> bytes:
        out = bytearray()
        while n > 0:
            if self._part_idx >= len(self._parts):
                break
            _, _, remaining_in_part = self._parts[self._part_idx]
            remaining_in_part -= self._pos_in_part
            take = min(n, remaining_in_part)
            if take > 0:
                chunk = self._fh.read(take)
                out += chunk
                self._pos_in_part += len(chunk)
                n -= len(chunk)
                if len(chunk) < take:
                    # short read -- treat as end of this part, move on
                    take = len(chunk)
            if n > 0 and self._pos_in_part >= remaining_in_part:
                if self._part_idx + 1 >= len(self._parts):
                    break
                self._open_part(self._part_idx + 1)
        return bytes(out)

    def skip(self, n: int):
        # Same part-boundary-crossing logic as read(), but discards bytes
        # without copying -- used to fast-forward over unwanted member data.
        while n > 0:
            if self._part_idx >= len(self._parts):
                break
            _, _, remaining_in_part = self._parts[self._part_idx]
            remaining_in_part -= self._pos_in_part
            take = min(n, remaining_in_part)
            if take > 0:
                self._fh.seek(take, 1)
                self._pos_in_part += take
                n -= take
            if n > 0 and self._pos_in_part >= remaining_in_part:
                if self._part_idx + 1 >= len(self._parts):
                    break
                self._open_part(self._part_idx + 1)

    def close(self):
        if self._fh is not None:
            self._fh.close()


LOCAL_HEADER_SIG = b"PK\x03\x04"
CENTRAL_DIR_SIG = b"PK\x01\x02"


def extract_from_split_archive(parts: list[Path], dest_dir: Path,
                                 wanted_keys: set[str] | None, ext: str) -> int:
    """Scan local file headers sequentially across split-archive parts
    (see MultiPartReader) and extract members matching `ext` whose filename
    stem is in `wanted_keys` (or every member of that ext if wanted_keys is
    None). Stops at the first central-directory signature.

    Deliberately does not touch the central directory at all -- central
    directory offsets in a split archive are relative to the start of
    whichever disk they point into, not to the merged stream, and fixing that
    up correctly means patching every entry's offset plus any zip64 extra
    fields that offset overflows into. Local file headers carry no such
    indirection: general-purpose bit 3 is 0 for this archive (sizes are known
    upfront, confirmed by inspection), so each header's compressed size is
    enough to jump straight to the next header with no lookahead needed.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    reader = MultiPartReader(parts)
    n_extracted = 0
    n_scanned = 0
    try:
        while True:
            sig = reader.read(4)
            if len(sig) < 4 or sig == CENTRAL_DIR_SIG:
                break
            if sig != LOCAL_HEADER_SIG:
                raise RuntimeError(
                    f"expected local file header or central directory signature at "
                    f"scan position, got {sig.hex()} after {n_scanned} entries -- "
                    f"split-part boundary accounting is wrong, do not trust any "
                    f"output written so far"
                )
            rest = reader.read(26)
            (version, flags, comp_method, _mtime, _mdate, _crc,
             csize, usize, nlen, elen) = struct.unpack("<HHHHHIIIHH", rest)
            if flags & 0x8:
                raise RuntimeError(
                    "general-purpose bit 3 (streaming / data-descriptor sizes) is "
                    "set on an entry -- this archive is not the format this scan "
                    "was verified against, stop and investigate"
                )
            name = reader.read(nlen).decode("utf-8", "replace")
            if elen:
                reader.skip(elen)
            n_scanned += 1

            stem = Path(name).stem
            is_wanted = name.endswith(ext) and (wanted_keys is None or stem in wanted_keys)
            if is_wanted:
                raw = reader.read(csize)
                if comp_method == 0:
                    payload = raw
                elif comp_method == 8:
                    payload = zlib.decompressobj(-15).decompress(raw)
                else:
                    raise RuntimeError(f"unsupported compression method {comp_method} for {name}")
                if len(payload) != usize:
                    raise RuntimeError(
                        f"{name}: decompressed {len(payload)} bytes, header says {usize}"
                    )
                target = dest_dir / Path(name).name
                target.write_bytes(payload)
                n_extracted += 1
            else:
                reader.skip(csize)

            if n_scanned % 2000 == 0:
                print(f"    ...scanned {n_scanned} entries, extracted {n_extracted}")
    finally:
        reader.close()

    print(f"  scanned {n_scanned} entries total, extracted {n_extracted} matching {ext} members")
    return n_extracted


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


def find_split_parts(archive_dir: Path, basename: str) -> list[Path]:
    """Return [basename.z01, basename.z02, ..., basename.zip] in order.
    Raises if any expected part is missing."""
    parts = sorted(archive_dir.glob(f"{basename}.z[0-9][0-9]"))
    zip_part = archive_dir / f"{basename}.zip"
    if not zip_part.exists():
        raise FileNotFoundError(f"expected {zip_part} (the final split part)")
    if not parts:
        raise FileNotFoundError(f"expected {basename}.z01 etc. alongside {zip_part}, found none")
    return parts + [zip_part]


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
    args = ap.parse_args()

    stf_root = args.stf_root.resolve()
    out_root = args.out_root.resolve()
    splits_dir = args.splits_dir.resolve()

    wanted_keys = None if args.full else load_frame_keys(splits_dir)
    if wanted_keys is not None:
        print(f"Loaded {len(wanted_keys)} unique frame keys from {splits_dir}")

    # 1) lidar_hdl64_strongest (split zip)
    print("\n=== lidar_hdl64_strongest ===")
    lidar_dir = stf_root / "lidar_hdl64_strongest"
    parts = find_split_parts(lidar_dir, "lidar_hdl64_strongest")
    n = extract_from_split_archive(parts, out_root / "lidar_hdl64_strongest", wanted_keys, ".bin")
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
