#!/usr/bin/env python3
"""Download the scoped STF subset for the 298A/298B capstone.

Pulls only the components needed for the baseline (Hahner PV-RCNN, lidar-only)
plus all four proposed models (camera stereo left+right raw + lidar + radar +
weather/friction/CAN telemetry + FIR + labels + calibration), per context.md's
modality scoping. Skips gated NIR, rectification LUTs, raw history folders,
precomputed stereo depth (sgm/psmnet), and VLP-32 lidar — none of which are
needed by the scoped model set.

Total expected size: ~100 GB across ~81 files.

Usage:
    python3 download_stf_subset.py [--manifest stf.json] [--out ./data/stf] [--dry-run]
"""

import argparse
import json
import socket
import sys
import threading
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

MAX_RETRIES = 5
STALL_TIMEOUT = 90  # seconds with no new bytes before we abort and retry
DEFAULT_WORKERS = 8  # concurrent downloads

print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)

# Applies to every socket op (connect + each read), so a connection that goes
# silent mid-transfer (stays ESTABLISHED but stops sending data) gets forced
# to raise instead of hanging indefinitely.
socket.setdefaulttimeout(STALL_TIMEOUT)

INCLUDE_TOP_LEVEL = {
    "gt_labels",
    "labeltool_labels",
    "lidar_hdl64_last",
    "lidar_hdl64_strongest",
    "cam_stereo_left",
    "cam_stereo_right",
    "radar_targets",
    "weather_station",
    "road_friction",
    "fir_axis",
    "filtered_relevant_can_data",
}


def top_level(key: str) -> str:
    rel = key[len("SeeingThroughFog/"):] if key.startswith("SeeingThroughFog/") else key
    return rel.rstrip("/").split("/")[0]


def wanted(key: str) -> bool:
    if key.endswith("/"):
        return False
    t = top_level(key)
    return t in INCLUDE_TOP_LEVEL or t.startswith("calib_")


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def download_one(url: str, dest: Path, chunk_size: int = 1024 * 1024) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, MAX_RETRIES + 1):
        resume_from = tmp.stat().st_size if tmp.exists() else 0
        req = urllib.request.Request(url)
        if resume_from:
            req.add_header("Range", f"bytes={resume_from}-")

        try:
            resp = urllib.request.urlopen(req, timeout=STALL_TIMEOUT)
        except urllib.error.HTTPError as e:
            if e.code == 416:
                # Already fully downloaded.
                tmp.rename(dest)
                return dest.stat().st_size
            raise

        mode = "ab" if resume_from else "wb"
        written = resume_from
        try:
            with open(tmp, mode) as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    written += len(chunk)
        except (TimeoutError, socket.timeout, ConnectionError, urllib.error.URLError) as e:
            if attempt == MAX_RETRIES:
                raise
            wait = min(2 ** attempt, 30)
            safe_print(f"  [{dest.name}] transient error ({e!r}), retrying in {wait}s "
                       f"(attempt {attempt}/{MAX_RETRIES}, resuming from {human(written)})...")
            time.sleep(wait)
            continue

        tmp.rename(dest)
        return written

    raise RuntimeError("unreachable")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="stf.json", help="Path to stf.json")
    ap.add_argument("--out", default="./data/stf", help="Output directory")
    ap.add_argument("--dry-run", action="store_true", help="List what would be downloaded, without downloading")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Concurrent downloads (default: 8)")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    with open(manifest_path) as f:
        data = json.load(f)

    entries = [u for u in data["urls"] if wanted(u["key"])]
    print(f"{len(entries)} files match the scoped subset.\n")

    out_dir = Path(args.out)

    if args.dry_run:
        for e in entries:
            print(e["key"])
        return 0

    to_fetch = []
    total_written = 0
    for e in entries:
        rel = e["key"][len("SeeingThroughFog/"):] if e["key"].startswith("SeeingThroughFog/") else e["key"]
        dest = out_dir / rel
        if dest.exists():
            print(f"skip (exists): {rel}")
            total_written += dest.stat().st_size
        else:
            to_fetch.append((rel, dest, e["url"]))

    print(f"\nDownloading {len(to_fetch)} file(s) with {args.workers} concurrent workers...\n")

    lock = threading.Lock()
    start = time.time()

    def worker(rel: str, dest: Path, url: str):
        safe_print(f"-> downloading: {rel}", flush=True)
        try:
            size = download_one(url, dest)
        except urllib.error.HTTPError as err:
            safe_print(f"  ERROR {err.code} on {rel} -- the pre-signed URL may have expired.", file=sys.stderr)
            return
        except (TimeoutError, socket.timeout, ConnectionError, urllib.error.URLError) as err:
            safe_print(f"  ERROR on {rel} after {MAX_RETRIES} retries: {err!r} -- rerun the script to retry.", file=sys.stderr)
            return

        nonlocal total_written
        with lock:
            total_written += size
        elapsed = time.time() - start
        safe_print(f"  done: {rel} ({human(size)}) | total so far: {human(total_written)} | elapsed: {elapsed/60:.1f} min")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(worker, rel, dest, url) for rel, dest, url in to_fetch]
        for f in as_completed(futures):
            f.result()  # re-raise any unexpected exception

    print(f"\nFinished. Total downloaded: {human(total_written)} into {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
