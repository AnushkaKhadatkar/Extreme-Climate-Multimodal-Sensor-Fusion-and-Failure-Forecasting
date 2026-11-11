#!/usr/bin/env python3
"""Summarise Model 1's reliability scores per split.

Reads the JSONL files written during evaluation with M1_RELIABILITY_LOG set
(one line per frame: r_lidar, r_cam, cam_weight). The thesis check: the model
never saw fog in training, so do r_lidar / r_cam fall on the fog splits anyway?

Usage:
    python summarize_reliability.py reliability_test_clear.jsonl reliability_test_dense_fog.jsonl ...
"""

import json
import sys
from pathlib import Path

import numpy as np


def summarise(path: Path) -> dict:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    out = {"frames": len(rows)}
    for key in ("r_lidar", "r_cam", "cam_weight"):
        v = np.array([r[key] for r in rows], dtype=float)
        out[key] = {"mean": float(v.mean()), "p10": float(np.percentile(v, 10)),
                    "median": float(np.median(v)), "p90": float(np.percentile(v, 90))}
    return out


def main() -> int:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    print(f"{'file':<40} {'frames':>6}  {'r_lidar med':>11} {'r_cam med':>9} {'cam weight':>10}")
    for p in map(Path, sys.argv[1:]):
        s = summarise(p)
        print(f"{p.stem:<40} {s['frames']:>6}  {s['r_lidar']['median']:>11.3f} "
              f"{s['r_cam']['median']:>9.3f} {s['cam_weight']['mean']:>10.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
