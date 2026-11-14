#!/usr/bin/env python3
"""Draw the Model 1 results overview (AP, ECE, reliability) from the archived run logs.

Reads only files the runs wrote (nothing is typed in by hand except the baseline's published-protocol
reproduction numbers, which come from baseline/README.md):

    <run>/model1_logs/eval_<split>.log          Car 3D AP (R40, IoU 0.5), the "3d   AP:" row under
                                                "Car AP_R40@0.70, 0.50, 0.50:" (bbox@0.7, bev@0.5, 3d@0.5)
    <run>/model1_logs/ece_<split>.json          tools/eval_calibration.py output
    <run>/model1_logs/reliability_<split>.jsonl per-frame r_lidar, r_cam, cam_weight (camera runs only)

Usage (needs numpy + matplotlib; run from anywhere):

    python make_results_figure.py \
        --run M4=run_logs/2026-11-14_M4_lidar_only_fp32_seed666 \
        --run M5=run_logs/2026-11-14_M5_fp32_seed666 \
        [--run M4b=run_logs/2026-11-14_M4b_nogate_fp32_seed666] \
        --out models/model1/docs/figures/model1_results_overview.png

Run labels: M4 = lidar only, M5 = lidar + RGB with the reliability gate, M4b = lidar + RGB without the gate.
Colours follow the entity in a fixed order (baseline, M4, M5, M4b), never the rank, so adding M4b later does
not repaint the others. The palette is the dataviz reference palette's first four categorical slots, in its
fixed order. The palette validator (Node) was not run when this was written.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SPLITS = ["test_clear", "test_light_fog", "test_dense_fog"]
SPLIT_NAMES = ["Clear", "Light fog", "Dense fog"]

# Our reproduction of the baseline (PV-RCNN, alpha = 0), Car 3D AP moderate; baseline/README.md.
BASELINE_MODERATE = [76.59, 68.26, 39.49]

# Fixed entity order -> categorical slots 1..4 of the reference palette (light mode).
ENTITIES = [
    ("Baseline", "Baseline (PV-RCNN, lidar)", "#2a78d6"),
    ("M4", "M4: lidar only", "#eb6834"),
    ("M5", "M5: lidar + RGB, with gate", "#1baf7a"),
    ("M4b", "M4b: lidar + RGB, no gate", "#eda100"),
]
COLOR = {k: c for k, _, c in ENTITIES}
NAME = {k: n for k, n, _ in ENTITIES}

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def read_ap(log: Path) -> list[float]:
    """-> [easy, moderate, hard] Car 3D AP, R40, IoU 0.5, from the last evaluation in the log."""
    lines = log.read_text(errors="ignore").splitlines()
    idx = max(i for i, l in enumerate(lines) if l.startswith("Car AP_R40@0.70, 0.50, 0.50"))
    for l in lines[idx + 1: idx + 6]:
        m = re.match(r"3d\s+AP:\s*([\d.]+), ([\d.]+), ([\d.]+)", l)
        if m:
            return [float(x) for x in m.groups()]
    raise ValueError(f"no 3d AP row after the R40 header in {log}")


def read_run(run_dir: Path) -> dict:
    logs = run_dir / "model1_logs"
    out = {"ap": [], "ece": [], "rel": None}
    for s in SPLITS:
        out["ap"].append(read_ap(logs / f"eval_{s}.log")[1])
        out["ece"].append(json.loads((logs / f"ece_{s}.json").read_text())["overall"]["ece"])
    if (logs / f"reliability_{SPLITS[0]}.jsonl").exists():
        rel = {"r_lidar": [], "r_cam": [], "cam_weight": []}
        for s in SPLITS:
            rows = [json.loads(l) for l in (logs / f"reliability_{s}.jsonl").read_text().splitlines() if l.strip()]
            for k in ("r_lidar", "r_cam"):
                v = np.array([r[k] for r in rows])
                rel[k].append((float(np.median(v)), float(np.percentile(v, 10)), float(np.percentile(v, 90))))
            rel["cam_weight"].append(float(np.mean([r["cam_weight"] for r in rows])))
        out["rel"] = rel
    return out


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, fontsize=9, color=INK2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, labelsize=9, labelcolor=INK2)
    ax.set_xticks(range(len(SPLIT_NAMES)))
    ax.set_xticklabels(SPLIT_NAMES)


def grouped_bars(ax, series: list[tuple[str, list[float]]], fmt: str):
    n = len(series)
    width = 0.8 / max(n, 3)
    for j, (key, vals) in enumerate(series):
        xs = np.arange(len(vals)) + (j - (n - 1) / 2) * width
        ax.bar(xs, vals, width=width * 0.92, color=COLOR[key], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        for x, v in zip(xs, vals):
            ax.text(x, v, format(v, fmt), ha="center", va="bottom", fontsize=7.5, color=INK2, zorder=4)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=DIR", help="M4, M5 or M4b = run_logs folder")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--table-out", type=Path, default=None, help="also write the numbers as a markdown table")
    ap.add_argument("--baseline-ece", type=Path, default=None,
                    help="folder with ece_baseline_<split>.json (tools/eval_calibration.py on the baseline's result.pkl)")
    args = ap.parse_args()

    runs = {}
    for spec in args.run:
        label, path = spec.split("=", 1)
        if label not in COLOR or label == "Baseline":
            raise SystemExit(f"unknown run label {label!r}; use M4, M5 or M4b")
        runs[label] = read_run(Path(path))
    order = [k for k, _, _ in ENTITIES if k == "Baseline" or k in runs]
    base_ece = None
    if args.baseline_ece:
        base_ece = [json.loads((args.baseline_ece / f"ece_baseline_{s}.json").read_text())["overall"]["ece"] for s in SPLITS]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6), facecolor=SURFACE)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.80, bottom=0.115, hspace=0.42, wspace=0.18)

    # A: AP
    ax = axes[0, 0]
    style(ax, "Car 3D AP, moderate (higher is better)", "AP, R40, IoU 0.5")
    grouped_bars(ax, [(k, BASELINE_MODERATE if k == "Baseline" else runs[k]["ap"]) for k in order], ".1f")
    ax.set_ylim(0, 92)

    # B: ECE
    ax = axes[0, 1]
    style(ax, "Calibration error, ECE (lower is better)", "ECE, Car detections")
    ece_keys = [k for k in order if k != "Baseline"]
    ece_series = ([("Baseline", base_ece)] if base_ece else []) + [(k, runs[k]["ece"]) for k in ece_keys]
    grouped_bars(ax, ece_series, ".3f")
    ax.axhline(0.05, color=INK2, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    ax.set_xlim(-0.5, 3.0)                                                   # room at the right for the label
    ax.text(2.52, 0.054, "target 0.05*", ha="left", va="bottom", fontsize=8, color=INK2)
    if not base_ece:
        ax.text(0.01, 0.97, "baseline ECE not measured", transform=ax.transAxes, fontsize=8, color=MUTED, va="top")
    ax.set_ylim(0, 0.30)

    # C: reliability scores (median with 10th-90th percentile)
    ax = axes[1, 0]
    style(ax, "Reliability scores the model assigns (median, 10th-90th pct)", "score r in (0, 1)")
    rel_keys = [k for k in order if k != "Baseline" and runs[k]["rel"]]
    for j, k in enumerate(rel_keys):
        for name, ls, mk in (("r_lidar", "-", "o"), ("r_cam", (0, (4, 3)), "s")):
            d = np.array(runs[k]["rel"][name])
            xs = np.arange(3) + (j - (len(rel_keys) - 1) / 2) * 0.07
            ax.errorbar(xs, d[:, 0], yerr=[d[:, 0] - d[:, 1], d[:, 2] - d[:, 0]], color=COLOR[k], linestyle=ls,
                        marker=mk, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2, linewidth=2,
                        capsize=3, elinewidth=1.2, zorder=3)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.45, 2.45)
    ax.text(0.01, 0.04, "solid = r_lidar, dashed = r_cam (flat at its ceiling 0.83: not informative)",
            transform=ax.transAxes, fontsize=8, color=INK2)

    # D: camera attention weight
    ax = axes[1, 1]
    style(ax, "Mean attention weight on the camera token", "weight in (0, 1)")
    for j, k in enumerate(rel_keys):
        vals = runs[k]["rel"]["cam_weight"]
        xs = np.arange(3) + (j - (len(rel_keys) - 1) / 2) * 0.07
        ax.plot(xs, vals, color=COLOR[k], linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE,
                markeredgewidth=1.2, zorder=3)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.025, f"{v:.2f}", ha="center", fontsize=8, color=INK2)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.45, 2.45)

    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[k]) for k in order]
    fig.legend(handles, [NAME[k] for k in order], loc="upper left", bbox_to_anchor=(0.07, 0.905), ncol=len(order),
               frameon=False, fontsize=9.5, labelcolor=INK, handlelength=1.2)
    fig.text(0.07, 0.965, "Model 1 on Seeing Through Fog: what the runs show so far", fontsize=15, fontweight="bold",
             color=INK, va="top")
    fig.text(0.07, 0.935, "Trained on clear weather only; fog splits are never seen in training. One seed per model, fp32 "
             "(mixed precision off), RTX 5090 stack.", fontsize=9.5, color=INK2, va="top")
    fig.text(0.07, 0.03, "Read with care: the baseline differs from Model 1 in trunk, loss and software stack, so differences are "
             "not all due to the camera or the gate. ECE and reliability were measured on the test splits\n"
             "without any sensor corruption. *0.05 is the thesis target for ECE with two sensors degraded; that condition "
             "is not measured here. Source: run_logs/ (M4, M5" + (", M4b" if "M4b" in runs else "") +
             (", baseline ECE" if base_ece else "") + "), baseline/README.md.", fontsize=8.5, color=MUTED, va="bottom")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170, facecolor=SURFACE)
    print("wrote", args.out)

    ece_cols = ece_series
    rows = ["| Split | " + " | ".join(f"{NAME[k]} AP" for k in order) + " | " + " | ".join(f"{k} ECE" for k, _ in ece_cols) + " |",
            "| --- |" + " --- |" * (len(order) + len(ece_cols))]
    for i, s in enumerate(SPLIT_NAMES):
        ap_cells = [f"{(BASELINE_MODERATE if k == 'Baseline' else runs[k]['ap'])[i]:.2f}" for k in order]
        ece_cells = [f"{vals[i]:.3f}" for _, vals in ece_cols]
        rows.append(f"| {s} | " + " | ".join(ap_cells + ece_cells) + " |")
    table = "\n".join(rows)
    print(table)
    if args.table_out:
        args.table_out.write_text(table + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
