#!/usr/bin/env python3
"""Generate OpenPCDet .pkl info files for all five STF splits.

Run this ON THE GPU BOX, inside an activated OpenPCDet environment (needs
pcdet installed, which needs spconv/CUDA -- see REPRODUCTION_PLAN.md for why
this can't run on the dev machine that wrote this script).

Why not just use OpenPCDet's stock `python -m pcdet.datasets.kitti.kitti_dataset
create_kitti_infos <cfg>` CLI: that entrypoint hard-codes train/val/test as the
only three splits, and treats 'test' as label-free (has_label=False, matching
real KITTI where the test set is held out by the benchmark server). Our three
STF "testing" splits (test_clear, test_light_fog, test_dense_fog) all DO have
ground truth -- we need it to compute AP per context.md's five-metric spec. So
this script calls the same underlying KittiDataset.get_infos() the stock CLI
uses, once per split, all with has_label=True.

Usage (from an OpenPCDet checkout, with this repo's data already placed under
OpenPCDet's expected data/kitti_format path -- see kitti_dataset_stf.yaml):

    python 03_create_infos.py --cfg ../configs/kitti_dataset_stf.yaml \\
        --data-path /path/to/baseline/data/kitti_format
"""

import argparse
import pickle
import sys
from pathlib import Path

import yaml
from easydict import EasyDict


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", type=Path, required=True, help="kitti_dataset_stf.yaml")
    ap.add_argument("--data-path", type=Path, required=True, help="baseline/data/kitti_format")
    ap.add_argument("--openpcdet-root", type=Path, default=None,
                     help="Path to an OpenPCDet checkout (so pcdet is importable). "
                          "If omitted, assumes pcdet is already on PYTHONPATH / pip-installed.")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if args.openpcdet_root:
        sys.path.insert(0, str(args.openpcdet_root.resolve()))

    try:
        from pcdet.datasets.kitti.kitti_dataset import KittiDataset
    except ImportError as e:
        print("Could not import pcdet.datasets.kitti.kitti_dataset -- this must run "
              "inside an OpenPCDet environment with pcdet installed (needs spconv/CUDA). "
              "See REPRODUCTION_PLAN.md.", file=sys.stderr)
        raise

    dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg)))
    data_path = args.data_path.resolve()
    save_path = data_path

    class_names = ["Car", "Pedestrian", "Cyclist"]

    dataset = KittiDataset(dataset_cfg=dataset_cfg, class_names=class_names,
                            root_path=data_path, training=False)

    # (split_name, imageset_name, has_label) -- all five STF splits have real
    # ground truth, unlike stock KITTI's held-out test set.
    splits = [
        ("train", "train", True),
        ("val", "val", True),
        ("test_clear", "test_clear", True),
        ("test_light_fog", "test_light_fog", True),
        ("test_dense_fog", "test_dense_fog", True),
    ]

    infos_by_split = {}
    for split_name, imageset_name, has_label in splits:
        print(f"--- generating infos for {split_name} (ImageSets/{imageset_name}.txt) ---")
        dataset.set_split(split_name)
        infos = dataset.get_infos(num_workers=args.workers, has_label=has_label, count_inside_pts=True)
        infos_by_split[split_name] = infos
        out_file = save_path / f"kitti_infos_{split_name}.pkl"
        with open(out_file, "wb") as f:
            pickle.dump(infos, f)
        print(f"  wrote {len(infos)} infos -> {out_file}")

    # trainval, for parity with stock OpenPCDet tooling that expects it to exist
    trainval_file = save_path / "kitti_infos_trainval.pkl"
    with open(trainval_file, "wb") as f:
        pickle.dump(infos_by_split["train"] + infos_by_split["val"], f)
    print(f"wrote trainval -> {trainval_file}")

    print("--- building groundtruth database (for gt_sampling augmentation) ---")
    dataset.set_split("train")
    dataset.create_groundtruth_database(save_path / "kitti_infos_train.pkl", split="train")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
