#!/usr/bin/env python3
"""Exploratory data analysis on the converted KITTI-format STF data.

Runs on CPU only, no OpenPCDet/CUDA needed -- same tier as
05_verify_conversion.py. Reads real labels and a representative sample of
real lidar scans (sampling, not synthesizing, per context.md's data rules)
and writes plots + a text summary to --out.

Usage:
    python 06_data_statistics.py --kitti-format ../data/kitti_format --out ../data/eda
"""

import argparse
import random
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Fixed categorical order/colors (dataviz skill: assign hues in fixed order,
# never cycled) -- slots 1-4 of the validated default palette.
CLASS_COLORS = {
    "Car": "#2a78d6",
    "Pedestrian": "#eb6834",
    "Cyclist": "#1baf7a",
    "DontCare": "#eda100",
}
CLASS_ORDER = ["Car", "Pedestrian", "Cyclist", "DontCare"]
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

SPLITS = ["train", "val", "test_clear", "test_light_fog", "test_dense_fog"]


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def load_split_ids(kitti_format: Path, split: str) -> list[str]:
    f = kitti_format / "ImageSets" / f"{split}.txt"
    return [l.strip() for l in f.read_text().splitlines() if l.strip()]


def parse_label_file(path: Path):
    """Return list of (cls, truncated, occluded, h, w, l, x, y, z) tuples."""
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        f = line.split(" ")
        cls = f[0]
        truncated, occluded = float(f[1]), int(f[2])
        h, w, l = float(f[8]), float(f[9]), float(f[10])
        x, y, z = float(f[11]), float(f[12]), float(f[13])
        rows.append((cls, truncated, occluded, h, w, l, x, y, z))
    return rows


# KITTI difficulty criteria, as in OpenPCDet's kitti_object_eval_python/eval.py
# clean_data(): (min 2D box height px, max occlusion, max truncation).
KITTI_DIFFICULTY = {"easy": (40, 0, 0.15), "moderate": (25, 1, 0.30), "hard": (25, 2, 0.50)}


def difficulty_audit(kitti_format: Path) -> list[str]:
    """BASELINE_TASKS.md D4. KITTI's easy/moderate/hard tiers are decided by
    occluded / truncated / 2D box height. If STF encodes occlusion or truncation
    on a different scale from KITTI, every tier is wrong. Report the raw value
    distributions and how many Car GT meet each tier's criteria per split."""
    lines = ["", "=== D4: difficulty-field audit (Car rows, real 3D box only) ==="]
    for split in SPLITS:
        occ, trunc = Counter(), Counter()
        tiers = Counter()
        n = 0
        for kid in load_split_ids(kitti_format, split):
            path = kitti_format / "training" / "label_2" / f"{kid}.txt"
            if not path.exists():
                continue
            for line in path.read_text().splitlines():
                f = line.strip().split(" ")
                if len(f) < 15 or f[0] != "Car" or float(f[8]) <= 0:
                    continue
                n += 1
                t, o = float(f[1]), int(float(f[2]))
                height = float(f[7]) - float(f[5])
                occ[o] += 1
                trunc[round(t, 1)] += 1
                for tier, (min_h, max_occ, max_trunc) in KITTI_DIFFICULTY.items():
                    if height >= min_h and o <= max_occ and t <= max_trunc:
                        tiers[tier] += 1
        lines.append(f"  {split}: {n} Car rows | meets easy {tiers['easy']}, moderate "
                     f"{tiers['moderate']}, hard {tiers['hard']} | none {n - tiers['hard']}")
        lines.append(f"    occluded values: {dict(sorted(occ.items()))}")
        lines.append(f"    truncated (rounded 0.1): {dict(sorted(trunc.items()))}")
    lines.append("  KITTI expects occluded in {0,1,2,3} and truncated in [0,1]. Anything else "
                 "means the tiers above do not mean what KITTI eval assumes.")
    return lines


def lidar_audit(kitti_format: Path, sample_ids: list[str], high_threshold: int = 90000) -> list[str]:
    """BASELINE_TASKS.md D5. Intensity scale, ground height vs PV-RCNN's KITTI
    assumptions (z range [-3, 1], car anchor bottom at -1.78 m), duplicate
    points, and which drives the ~2x point-count frames come from."""
    seq_of = {}
    mapping = kitti_format / "ImageSets" / "id_mapping.csv"
    if mapping.exists():
        for row in mapping.read_text().splitlines()[1:]:
            kid, seq, _, _ = row.split(",")
            seq_of[kid] = seq

    intensity_max, ground_z, dup_frac = [], [], []
    high_seqs, low_seqs = Counter(), Counter()
    for kid in sample_ids:
        scan = np.fromfile(kitti_format / "training" / "velodyne" / f"{kid}.bin", dtype=np.float32).reshape(-1, 4)
        intensity_max.append(float(scan[:, 3].max()))
        near = scan[(scan[:, 0] > 5) & (scan[:, 0] < 30) & (np.abs(scan[:, 1]) < 10)]
        if len(near):
            ground_z.append(float(np.percentile(near[:, 2], 5)))
        dup_frac.append(1 - len(np.unique(scan[:, :3], axis=0)) / len(scan))
        day = seq_of.get(kid, "?")[:10]
        (high_seqs if len(scan) > high_threshold else low_seqs)[day] += 1

    intensity_max, ground_z, dup_frac = map(np.array, (intensity_max, ground_z, dup_frac))
    lines = ["", f"=== D5: lidar audit ({len(sample_ids)} sampled train frames) ==="]
    lines.append(f"  intensity max per frame: median {np.median(intensity_max):.2f}, "
                 f"max {intensity_max.max():.2f}  (KITTI is 0-1; ~255 means a 0-255 scale)")
    lines.append(f"  ground z estimate (5th pct of z, 5-30 m ahead): median {np.median(ground_z):.2f} m, "
                 f"p5 {np.percentile(ground_z, 5):.2f}, p95 {np.percentile(ground_z, 95):.2f}  "
                 f"(KITTI ground is ~-1.73 m; POINT_CLOUD_RANGE z is [-3, 1])")
    lines.append(f"  duplicate xyz points: median {np.median(dup_frac):.1%} of a frame, "
                 f"max {dup_frac.max():.1%}")
    lines.append(f"  frames > {high_threshold} points by recording day: {dict(high_seqs)}")
    lines.append(f"  frames <= {high_threshold} points by recording day: {dict(low_seqs)}")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kitti-format", type=Path, default=Path("../data/kitti_format"))
    ap.add_argument("--out", type=Path, default=Path("../data/eda"))
    ap.add_argument("--lidar-sample-size", type=int, default=400,
                     help="number of real frames to sample for point-count stats (sampling, not synthesizing)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    kitti_format = args.kitti_format.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- gather per-split class counts and per-frame object counts ----
    split_class_counts = {}
    train_objects_per_frame = []
    train_cyclists_per_frame = []
    dims_by_class = {c: [] for c in ["Car", "Pedestrian", "Cyclist"]}
    dist_by_class = {c: [] for c in ["Car", "Pedestrian", "Cyclist"]}
    occlusion_by_class = Counter()
    n_sentinel_dims = 0
    n_vocab_rows = 0

    for split in SPLITS:
        ids = load_split_ids(kitti_format, split)
        counts = Counter()
        for kid in ids:
            rows = parse_label_file(kitti_format / "training" / "label_2" / f"{kid}.txt")
            counts.update(r[0] for r in rows)
            if split == "train":
                train_objects_per_frame.append(len(rows))
                train_cyclists_per_frame.append(sum(1 for r in rows if r[0] == "Cyclist"))
                for cls, trunc, occ, h, w, l, x, y, z in rows:
                    if cls not in dims_by_class:
                        continue
                    n_vocab_rows += 1
                    if h > 0 and w > 0 and l > 0:
                        dims_by_class[cls].append((l, w, h))
                        dist_by_class[cls].append(z)  # z = forward distance, camera coords
                        occlusion_by_class[(cls, occ)] += 1
                    else:
                        # h/w/l == -1.00 is STF's sentinel for "3D dims not measured"
                        # (paired with location == -1000.00) -- not a real 0-or-negative
                        # object size, exclude it rather than let it skew the box plots.
                        n_sentinel_dims += 1
        split_class_counts[split] = counts
        print(f"{split}: {len(ids)} frames, {sum(counts.values())} labeled objects -> {dict(counts)}")

    # ---- real lidar point-count sample (sampling real frames, not synthetic) ----
    train_ids = load_split_ids(kitti_format, "train")
    sample_ids = random.sample(train_ids, min(args.lidar_sample_size, len(train_ids)))
    point_counts = []
    for kid in sample_ids:
        scan = np.fromfile(kitti_format / "training" / "velodyne" / f"{kid}.bin", dtype=np.float32)
        point_counts.append(scan.size // 4)
    point_counts = np.array(point_counts)
    print(f"\nLidar points/frame over {len(sample_ids)} sampled train frames: "
          f"mean={point_counts.mean():.0f} std={point_counts.std():.0f} "
          f"min={point_counts.min()} max={point_counts.max()}")

    # ============================================================
    # Figure A: class distribution across all five splits (grouped bars)
    # ============================================================
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(SPLITS))
    width = 0.2
    for i, cls in enumerate(CLASS_ORDER):
        vals = [split_class_counts[s].get(cls, 0) for s in SPLITS]
        ax.bar(x + (i - 1.5) * width, vals, width, label=cls, color=CLASS_COLORS[cls],
               edgecolor="#fcfcfb", linewidth=0.5, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(SPLITS, color=INK)
    ax.set_ylabel("Labeled objects", color=INK)
    ax.set_title("Class distribution by split (after STF->KITTI conversion)", color=INK, fontsize=12, loc="left")
    style_axis(ax)
    ax.legend(frameon=False, labelcolor=INK, ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(out_dir / "A_class_distribution_by_split.png", dpi=150, facecolor="#fcfcfb")
    plt.close(fig)

    # ============================================================
    # Figure B: objects-per-frame histogram (train split)
    # ============================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(train_objects_per_frame, bins=range(0, max(train_objects_per_frame) + 2),
            color=CLASS_COLORS["Car"], edgecolor="#fcfcfb", linewidth=0.5, zorder=2)
    ax.set_xlabel("Labeled objects per frame", color=INK)
    ax.set_ylabel("Frame count", color=INK)
    ax.set_title(f"Objects per frame, train split (n={len(train_objects_per_frame)} frames, "
                 f"mean={np.mean(train_objects_per_frame):.1f})", color=INK, fontsize=12, loc="left")
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "B_objects_per_frame.png", dpi=150, facecolor="#fcfcfb")
    plt.close(fig)

    # ============================================================
    # Figure C: object dimensions by class (box plots: length, width, height)
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=False)
    dim_names = ["length (l)", "width (w)", "height (h)"]
    for dim_idx, (ax, dim_name) in enumerate(zip(axes, dim_names)):
        data = [[d[dim_idx] for d in dims_by_class[c]] for c in ["Car", "Pedestrian", "Cyclist"]]
        bp = ax.boxplot(data, tick_labels=["Car", "Pedestrian", "Cyclist"], patch_artist=True,
                         medianprops=dict(color=INK), widths=0.5)
        for patch, c in zip(bp["boxes"], ["Car", "Pedestrian", "Cyclist"]):
            patch.set_facecolor(CLASS_COLORS[c])
            patch.set_alpha(0.85)
        ax.set_title(dim_name, color=INK, fontsize=11)
        ax.set_ylabel("meters", color=INK)
        style_axis(ax)
    fig.suptitle("Object dimensions by class (KITTI camera-frame l/w/h)", color=INK, fontsize=12, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(out_dir / "C_dimensions_by_class.png", dpi=150, facecolor="#fcfcfb")
    plt.close(fig)

    # ============================================================
    # Figure D: forward-distance distribution by class
    # ============================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls in ["Car", "Pedestrian", "Cyclist"]:
        ax.hist(dist_by_class[cls], bins=40, range=(0, 80), alpha=0.6, label=cls,
                color=CLASS_COLORS[cls], zorder=2)
    ax.set_xlabel("Forward distance from sensor (m)", color=INK)
    ax.set_ylabel("Object count", color=INK)
    ax.set_title("Object distance distribution by class, train split", color=INK, fontsize=12, loc="left")
    style_axis(ax)
    ax.legend(frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out_dir / "D_distance_by_class.png", dpi=150, facecolor="#fcfcfb")
    plt.close(fig)

    # ============================================================
    # Figure E: lidar points-per-frame histogram (sampled real frames)
    # ============================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(point_counts, bins=30, color=CLASS_COLORS["Cyclist"], edgecolor="#fcfcfb", linewidth=0.5, zorder=2)
    ax.set_xlabel("Lidar points per frame", color=INK)
    ax.set_ylabel("Frame count", color=INK)
    ax.set_title(f"Lidar point count, {len(sample_ids)} sampled real train frames "
                 f"(mean={point_counts.mean():.0f})", color=INK, fontsize=12, loc="left")
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "E_lidar_points_per_frame.png", dpi=150, facecolor="#fcfcfb")
    plt.close(fig)

    # ---- text summary of findings ----
    total_counts = Counter()
    for c in split_class_counts.values():
        total_counts.update(c)
    pct_zero_cyclist = 100 * sum(1 for n in train_cyclists_per_frame if n == 0) / len(train_cyclists_per_frame)
    summary = out_dir / "summary.txt"
    lines = [
        "=== Data statistics summary (real data, computed by this script) ===",
        "",
        "Class totals across all 5 splits:",
    ]
    for cls in CLASS_ORDER:
        lines.append(f"  {cls}: {total_counts.get(cls, 0)}")
    lines += [
        "",
        f"Train split: {len(train_objects_per_frame)} frames, "
        f"mean {np.mean(train_objects_per_frame):.1f} objects/frame "
        f"(min {min(train_objects_per_frame)}, max {max(train_objects_per_frame)})",
        f"Lidar points/frame ({len(sample_ids)}-frame sample): "
        f"mean={point_counts.mean():.0f} std={point_counts.std():.0f}",
        "",
        "Findings:",
    ]
    train_car, train_ped = split_class_counts["train"]["Car"], split_class_counts["train"]["Pedestrian"]
    total_car, total_ped = total_counts["Car"], total_counts["Pedestrian"]
    if train_ped > train_car and total_car > total_ped:
        lines.append(
            f"  - Pedestrian outnumbers Car in the train split alone ({train_ped} vs {train_car}), "
            f"but Car leads once all 5 splits are combined ({total_car} vs {total_ped}) -- the test "
            f"splits skew more Car-heavy than train, worth checking for train/test distribution shift."
        )
    else:
        leader, other = ("Car", "Pedestrian") if total_car >= total_ped else ("Pedestrian", "Car")
        lines.append(f"  - {leader} is the most common class overall ({total_counts[leader]} vs "
                      f"{total_counts[other]} for {other}), consistent across splits.")
    lines += [
        f"  - DontCare makes up {100*total_counts['DontCare']/sum(total_counts.values()):.1f}% of all "
        f"labeled rows -- non-vocabulary STF classes plus Car/Pedestrian/Cyclist rows with no 3D box "
        f"(BASELINE_TASKS.md D1), both remapped during conversion, not noise.",
        f"  - {pct_zero_cyclist:.1f}% of train frames contain zero Cyclist instances -- the class is "
        f"real but sparse, worth watching in the per-class AP breakdown.",
        f"  - {n_sentinel_dims}/{n_vocab_rows} ({100*n_sentinel_dims/n_vocab_rows:.1f}%) of train-split "
        f"Car/Pedestrian/Cyclist rows carry STF's sentinel '-1.00' dims (3D box not measured for that "
        f"object) -- excluded from the dimension/distance plots above rather than left in to skew them.",
    ]
    pct_high = 100 * (point_counts > 90000).sum() / len(point_counts)
    if pct_high > 2:
        lines.append(
            f"  - Lidar points/frame is sharply bimodal, not a smooth spread: most sampled frames cluster "
            f"around {np.median(point_counts[point_counts < 90000]):.0f} points, but "
            f"{pct_high:.1f}% cluster near {np.median(point_counts[point_counts > 90000]):.0f} -- "
            f"roughly 2x, not weather-driven attenuation (that would be a continuous shift, not a clean "
            f"split). Open question, not yet explained: possibly a sensor/recording-session difference "
            f"across STF drives, worth checking before training leans on point density as a signal."
        )
    lines += difficulty_audit(kitti_format)
    lines += lidar_audit(kitti_format, sample_ids)
    summary.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nWrote 5 figures + summary.txt to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
