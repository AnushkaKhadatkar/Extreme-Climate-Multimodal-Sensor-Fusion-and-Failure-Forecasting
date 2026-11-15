#!/usr/bin/env python3
"""Collect the sensor-failure probe (tools/run_corruption_probe.sh) into markdown tables.

For every <model>_<condition> it reads, from the probe output folder:
    <model>_<cond>.log          Car 3D AP (R40, IoU 0.5), the "3d   AP:" row under "Car AP_R40@0.70, 0.50, 0.50:"
    ece_<model>_<cond>.json     ECE (tools/eval_calibration.py)
    rel_<model>_<cond>.jsonl    per-frame r_lidar, r_cam, cam_weight (camera models only)

Usage:  python summarize_corruption_probe.py <probe_dir> [--md out.md]
Stdlib only.
"""

import argparse
import json
import re
import statistics as st
from pathlib import Path

MODELS = ["M4", "M4b", "M5"]
CONDITIONS = ["clean", "cam_zeros", "cam_occlusion", "cam_dark", "cam_blur", "lidar_sparse", "lidar_sector", "both_dark_sparse"]


def read_ap(log: Path):
    if not log.exists():
        return None
    lines = log.read_text(errors="ignore").splitlines()
    idx = [i for i, l in enumerate(lines) if l.startswith("Car AP_R40@0.70, 0.50, 0.50")]
    if not idx:
        return None
    for l in lines[idx[-1] + 1: idx[-1] + 6]:
        m = re.match(r"3d\s+AP:\s*([\d.]+), ([\d.]+), ([\d.]+)", l)
        if m:
            return float(m.group(2))                      # moderate
    return None


def read_ece(path: Path):
    if not path.exists():
        return None
    o = json.loads(path.read_text())["overall"]
    n = sum(b["n"] for b in o["bins"])
    return o["ece"], o["n"], sum(b["n"] * b["confidence"] for b in o["bins"]) / n, sum(b["n"] * b["accuracy"] for b in o["bins"]) / n


def read_rel(path: Path):
    if not path.exists():
        return None
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return (st.median(r["r_lidar"] for r in rows), st.median(r["r_cam"] for r in rows),
            st.mean(r["cam_weight"] for r in rows))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path)
    ap.add_argument("--md", type=Path, default=None)
    args = ap.parse_args()
    d = args.dir
    data = {(m, c): {"ap": read_ap(d / f"{m}_{c}.log"), "ece": read_ece(d / f"ece_{m}_{c}.json"),
                     "rel": read_rel(d / f"rel_{m}_{c}.jsonl")} for m in MODELS for c in CONDITIONS}

    f = lambda v, p=2: "-" if v is None else f"{v:.{p}f}"
    out = ["**Moderate Car 3D AP and ECE on clear val (781 frames), per injected failure** (- = not run: a camera failure "
           "cannot change the lidar-only M4)", "",
           "| Condition | M4 AP | M4b AP | M5 AP | M4 ECE | M4b ECE | M5 ECE |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for c in CONDITIONS:
        row = [c]
        row += [f(data[(m, c)]["ap"]) for m in MODELS]
        row += [f(data[(m, c)]["ece"][0] if data[(m, c)]["ece"] else None, 3) for m in MODELS]
        out.append("| " + " | ".join(row) + " |")
    out += ["", "**What the reliability estimators believe** (median r_lidar / median r_cam / mean camera attention weight). "
            "In M4b r is estimated but does not drive attention.", "",
            "| Condition | M4b r_lidar | M4b r_cam | M4b cam weight | M5 r_lidar | M5 r_cam | M5 cam weight |",
            "| --- | --- | --- | --- | --- | --- | --- |"]
    for c in CONDITIONS:
        row = [c]
        for m in ("M4b", "M5"):
            rel = data[(m, c)]["rel"]
            row += [f(rel[i], 3) if rel else "-" for i in range(3)]
        out.append("| " + " | ".join(row) + " |")
    text = "\n".join(out)
    print(text)
    if args.md:
        args.md.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
