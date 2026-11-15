#!/usr/bin/env python3
"""Tables for the gate-floor probe (tools/run_gate_floor_probe.sh) next to the sensor-failure probe v1.

    python summarize_gate_floor_probe.py <corruption_probe_v1_dir> <corruption_probe_v2_floor_dir> [--md out.md]

Columns: M4b (camera, no gate) and M5 with the original floor 1e-4, both from v1; then M5 at eval-time floors
0.05 / 0.2 / 0.5 from v2. Rows: the injected failures. Reuses the readers of summarize_corruption_probe.py. Stdlib only.
"""

import argparse
from pathlib import Path

from summarize_corruption_probe import CONDITIONS, read_ap, read_ece, read_rel

FLOORS = ["0.05", "0.2", "0.5"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("v1", type=Path)
    ap.add_argument("v2", type=Path)
    ap.add_argument("--md", type=Path, default=None)
    args = ap.parse_args()

    def get(d, label, cond):
        return {"ap": read_ap(d / f"{label}_{cond}.log"), "ece": read_ece(d / f"ece_{label}_{cond}.json"),
                "rel": read_rel(d / f"rel_{label}_{cond}.jsonl")}

    cols = [("M4b no gate", args.v1, "M4b"), ("M5 floor 1e-4", args.v1, "M5")] + [(f"M5 floor {f}", args.v2, f"M5f{f}") for f in FLOORS]
    data = {(name, c): get(d, label, c) for name, d, label in cols for c in CONDITIONS}
    f = lambda v, p=2: "-" if v is None else f"{v:.{p}f}"

    out = ["**Moderate Car 3D AP on clear val (781 frames), eval-time gate floor on the trained M5 checkpoint**", "",
           "| Condition | " + " | ".join(n for n, _, _ in cols) + " |", "| --- |" + " --- |" * len(cols)]
    for c in CONDITIONS:
        out.append(f"| {c} | " + " | ".join(f(data[(n, c)]["ap"]) for n, _, _ in cols) + " |")
    out += ["", "**ECE**", "", "| Condition | " + " | ".join(n for n, _, _ in cols) + " |", "| --- |" + " --- |" * len(cols)]
    for c in CONDITIONS:
        out.append(f"| {c} | " + " | ".join(f(data[(n, c)]["ece"][0] if data[(n, c)]["ece"] else None, 3) for n, _, _ in cols) + " |")
    out += ["", "**Mean attention weight on the camera token**", "", "| Condition | " + " | ".join(n for n, _, _ in cols) + " |",
            "| --- |" + " --- |" * len(cols)]
    for c in CONDITIONS:
        out.append(f"| {c} | " + " | ".join(f(data[(n, c)]["rel"][2] if data[(n, c)]["rel"] else None, 3) for n, _, _ in cols) + " |")
    text = "\n".join(out)
    print(text)
    if args.md:
        args.md.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
