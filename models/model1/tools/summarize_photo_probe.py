#!/usr/bin/env python3
"""Tables for probe v3 (tools/run_photo_probe.sh), merged with probe v1 for the conditions v3 did not re-run.

    python summarize_photo_probe.py <corruption_probe_v1_dir> <corruption_probe_v3_dir> [--md out.md]

Models: M4b (camera, no gate), M5 (camera + gate), M4bp (no gate + photometric camera dropout), M5p (gate, floor 0.1,
+ photometric camera dropout). For each (model, condition) the v3 folder is read first, then v1. Stdlib only.
"""

import argparse
from pathlib import Path

from summarize_corruption_probe import read_ap, read_ece, read_rel

MODELS = ["M4b", "M5", "M4bp", "M5p"]
CONDITIONS = ["clean", "cam_zeros", "cam_occlusion", "cam_dark", "cam_blur", "cam_haze", "cam_motion_blur",
              "cam_color_cast", "lidar_sparse", "lidar_sector", "both_dark_sparse"]
FAMILY = {"cam_dark": "in-family for p", "cam_blur": "in-family for p", "cam_haze": "HELD-OUT", "cam_motion_blur": "HELD-OUT",
          "cam_color_cast": "HELD-OUT"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("v1", type=Path)
    ap.add_argument("v3", type=Path)
    ap.add_argument("--md", type=Path, default=None)
    args = ap.parse_args()

    def get(m, c):
        for d in (args.v3, args.v1):
            ap_, ece = read_ap(d / f"{m}_{c}.log"), read_ece(d / f"ece_{m}_{c}.json")
            if ap_ is not None:
                return {"ap": ap_, "ece": ece[0] if ece else None, "rel": read_rel(d / f"rel_{m}_{c}.jsonl")}
        return {"ap": None, "ece": None, "rel": None}

    data = {(m, c): get(m, c) for m in MODELS for c in CONDITIONS}
    f = lambda v, p=2: "-" if v is None else f"{v:.{p}f}"
    out = ["**Moderate Car 3D AP on clear val (781 frames) under injected failures** (synthetic; one seed; differences under "
           "about 3 AP are not evidence)", "",
           "| Condition | family | " + " | ".join(MODELS) + " | M5p - M4bp | M4bp - M4b | M5p - M5 |", "| --- | --- |" + " --- |" * (len(MODELS) + 3)]
    for c in CONDITIONS:
        v = {m: data[(m, c)]["ap"] for m in MODELS}
        d = lambda a, b: "-" if v[a] is None or v[b] is None else f"{v[a] - v[b]:+.2f}"
        out.append(f"| {c} | {FAMILY.get(c, '')} | " + " | ".join(f(v[m]) for m in MODELS) + f" | {d('M5p', 'M4bp')} | {d('M4bp', 'M4b')} | {d('M5p', 'M5')} |")
    out += ["", "**ECE**", "", "| Condition | " + " | ".join(MODELS) + " |", "| --- |" + " --- |" * len(MODELS)]
    for c in CONDITIONS:
        out.append(f"| {c} | " + " | ".join(f(data[(m, c)]["ece"], 3) for m in MODELS) + " |")
    out += ["", "**Median r_cam / median r_lidar / mean camera attention weight** (what the estimators believe; r drives attention "
            "only in M5 and M5p)", "", "| Condition | " + " | ".join(MODELS) + " |", "| --- |" + " --- |" * len(MODELS)]
    for c in CONDITIONS:
        cells = []
        for m in MODELS:
            rel = data[(m, c)]["rel"]
            cells.append("-" if not rel else f"{rel[1]:.2f} / {rel[0]:.2f} / {rel[2]:.2f}")
        out.append(f"| {c} | " + " | ".join(cells) + " |")
    text = "\n".join(out)
    print(text)
    if args.md:
        args.md.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
