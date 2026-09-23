#!/usr/bin/env python3
"""Extract the scoped subset of files (key + URL) from stf.json.

Scope (per context.md): baseline (lidar-only) + all four models
(camera stereo left+right + lidar + radar + weather/friction/CAN telemetry +
labels + calibration). Excludes gated NIR, rectification LUTs, raw history
folders, precomputed stereo depth (sgm/psmnet), VLP-32 lidar, and FIR.

Usage:
    python3 extract_required_files.py [stf.json] [--out required_files.json]
"""

import argparse
import json
from pathlib import Path

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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest", nargs="?", default="stf.json")
    ap.add_argument("--out", default="required_files.json")
    args = ap.parse_args()

    with open(args.manifest) as f:
        data = json.load(f)

    required = [u for u in data["urls"] if wanted(u["key"])]

    with open(args.out, "w") as f:
        json.dump({"urls": required}, f, indent=2)

    print(f"{len(required)} files matched out of {len(data['urls'])} total.")
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
