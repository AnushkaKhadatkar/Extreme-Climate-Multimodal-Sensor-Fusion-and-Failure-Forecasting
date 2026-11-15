#!/usr/bin/env python3
"""Draw the sensor-failure probe (AP and camera reliability under injected failures) from probe v1 + v3.

    python make_probe_figure.py <corruption_probe_v1_dir> <corruption_probe_v3_dir> --out docs/figures/model1_failure_probe.png

Reads the same files as summarize_photo_probe.py (v3 first, then v1). Needs numpy + matplotlib.
Models (fixed colour per entity, bars ordered no-gate then gate): M4b camera no gate, M4bp same + photometric camera
dropout, M5 camera + gate, M5p gate (floor 0.1) + photometric dropout. Colours are slots 4, 7, 3 and 5 of the dataviz
reference palette; its validator (Node) was not run. The failures are SYNTHETIC; clear val, one seed per model.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from summarize_corruption_probe import read_ap, read_rel  # noqa: E402

MODELS = [("M4b", "M4b: camera, no gate", "#eda100"), ("M4bp", "M4bp: no gate + photometric dropout", "#6250d6"),
          ("M5", "M5: camera + gate", "#1baf7a"), ("M5p", "M5p: gate (floor 0.1) + photometric dropout", "#e87ba4")]
CONDITIONS = ["clean", "cam_zeros", "cam_occlusion", "cam_dark", "cam_blur", "cam_haze", "cam_motion_blur", "cam_color_cast",
              "lidar_sparse", "lidar_sector", "both_dark_sparse"]
LABELS = ["no failure", "camera\ndead", "camera\noccluded", "camera\ndark", "camera\nblur", "camera\nhaze", "camera\nmotion blur",
          "camera\ncolour cast", "lidar\n5% points", "lidar\nsector gone", "dark camera +\nsparse lidar"]
HELD_OUT = {5, 6, 7}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("v1", type=Path)
    ap.add_argument("v3", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    def get(m, c):
        for d in (args.v3, args.v1):
            a = read_ap(d / f"{m}_{c}.log")
            if a is not None:
                rel = read_rel(d / f"rel_{m}_{c}.jsonl")
                return a, (rel[1] if rel else np.nan)
        return np.nan, np.nan

    ap_ = {m: [get(m, c)[0] for c in CONDITIONS] for m, _, _ in MODELS}
    rc_ = {m: [get(m, c)[1] for c in CONDITIONS] for m, _, _ in MODELS}

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13.5, 9.2), facecolor=SURFACE, gridspec_kw={"height_ratios": [1.5, 1]})
    fig.subplots_adjust(left=0.065, right=0.99, top=0.84, bottom=0.115, hspace=0.45)
    x = np.arange(len(CONDITIONS))
    w = 0.2
    for ax in (a1, a2):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, labelsize=9, labelcolor=INK2)
        ax.set_xticks(x)
        ax.set_xticklabels(LABELS, fontsize=8.5)
        for i in HELD_OUT:
            ax.axvspan(i - 0.5, i + 0.5, color="#f0efec", zorder=0)
    for j, (m, _, col) in enumerate(MODELS):
        a1.bar(x + (j - 1.5) * w, ap_[m], width=w * 0.92, color=col, edgecolor=SURFACE, linewidth=1.2, zorder=3)
        a2.plot(x + (j - 1.5) * 0.06, rc_[m], color=col, marker="o", markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1.0,
                linewidth=0, zorder=3)
    a1.set_ylim(0, 92)
    a1.set_title("Car 3D AP, moderate, under injected failures (higher is better; clear validation frames)", loc="left", fontsize=11,
                 fontweight="bold", color=INK, pad=10)
    a1.set_ylabel("AP, R40, IoU 0.5", fontsize=9, color=INK2)
    for i in HELD_OUT:
        a1.text(i, 88, "held out\nin training", ha="center", va="top", fontsize=7.5, color=MUTED)
    a2.set_ylim(-0.03, 1.0)
    a2.set_title("Median camera reliability score r_cam (what the model believes about its camera)", loc="left", fontsize=11,
                 fontweight="bold", color=INK, pad=10)
    a2.set_ylabel("r_cam", fontsize=9, color=INK2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in MODELS]
    fig.legend(handles, [n for _, n, _ in MODELS], loc="upper left", bbox_to_anchor=(0.065, 0.905), ncol=4, frameon=False,
               fontsize=9, labelcolor=INK, handlelength=1.2, columnspacing=1.6)
    fig.text(0.065, 0.965, "Model 1 under injected sensor failures: augmentation helps, the gate does not add to it", fontsize=15,
             fontweight="bold", color=INK, va="top")
    fig.text(0.065, 0.935, "Synthetic failures, one seed per model, fp32. Differences under about 3 AP are within the noise "
             "between separately trained models.", fontsize=9.5, color=INK2, va="top")
    fig.text(0.065, 0.03, "Shaded groups are camera failures the photometric-dropout variants (M4bp, M5p) never saw in training; dark and blur "
             "are in their training set, so those bars are partly teaching to the test.\nr_cam falls for a dead, occluded or blurred camera but not "
             "for dark, haze, motion blur or colour cast. Source: run_logs/2026-11-14_sensor_failure_probe_v1, 2026-11-15_photo_probe_v3.",
             fontsize=8.5, color=MUTED, va="bottom")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170, facecolor=SURFACE)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
