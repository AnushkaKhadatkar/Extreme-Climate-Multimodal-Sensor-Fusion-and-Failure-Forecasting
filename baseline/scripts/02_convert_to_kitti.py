#!/usr/bin/env python3
"""Convert extracted STF raw files (from 01_extract_archives.py) into a standard
KITTI dataset layout that OpenPCDet's stock kitti_dataset.py can read unmodified.

See ../docs/REPRODUCTION_PLAN.md for the full rationale. Summary of the five
conversions this script does, per frame:

  1. Frame ID: STF's 'sequence_timestamp,frame_id' pair -> a KITTI-style 6-digit
     zero-padded numeric ID. STF IDs aren't numeric, so we assign IDs in split-file
     order and record the mapping (needed to trace a result back to a real STF
     scene later).
  2. LiDAR: (N,5) float32 [x,y,z,intensity,channel] -> (N,4) float32
     [x,y,z,intensity], written as a raw .bin (drop the channel/ring column;
     OpenPCDet's KITTI loader hard-codes num_features=4).
  3. Labels: STF's 27-field extended-KITTI txt -> KITTI's standard 15-field
     label_2 txt. Class remap: PassengerCar->Car, RidableVehicle->Cyclist,
     Pedestrian->Pedestrian (unchanged); everything else (Obstacle, Vehicle,
     LargeVehicle, *_is_group, stray 'person'/'train' typos) -> DontCare, since
     they're not part of KITTI's standard 3-class vocabulary and would otherwise
     silently mismatch OpenPCDet's default class filter.
  4. Calibration: STF's ROS-style camera-intrinsics JSON + tf-tree JSON (quaternion
     extrinsics) -> KITTI's flat calib.txt (P0-P3, R0_rect, Tr_velo_to_cam,
     Tr_imu_to_velo). One calibration is computed once (STF calibration is static
     across the whole dataset) and written identically for every frame.
  5. Images: cam_stereo_left/*.tiff -> image_2/*.png (lossless). Not used as
     network input by PV-RCNN (lidar-only), but OpenPCDet's KITTI pipeline reads
     image shape for FOV-based point filtering, so a placeholder is not safe --
     needs the real image.

All five splits (train/val/test_clear/test_light_fog/test_dense_fog) are written
under a single training/ directory, NOT split across training/ + testing/ the
way stock KITTI is laid out. This works around a hard-coded check in OpenPCDet's
KittiDataset (`'training' if self.split != 'test' else 'testing'` -- literally
any split name other than 'test' resolves to training/), which our named test
splits would otherwise silently miss. See REPRODUCTION_PLAN.md section 2.

Usage:
    python 02_convert_to_kitti.py --source ../data/kitti_source --stf-root ../../data/stf \\
        --out ../data/kitti_format --splits-dir ../splits
"""

import argparse
import csv
import json
import zipfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# STF class name -> KITTI class name. Anything not listed becomes DontCare.
CLASS_REMAP = {
    "PassengerCar": "Car",
    "RidableVehicle": "Cyclist",
    "Pedestrian": "Pedestrian",
    "DontCare": "DontCare",
}


def load_split(split_file: Path) -> list[tuple[str, str]]:
    """Return list of (sequence, frame_id) in file order."""
    out = []
    for line in split_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        seq, frame = line.split(",")
        out.append((seq, frame))
    return out


def compute_calibration(calib_dir: Path) -> dict[str, np.ndarray]:
    """Reproduce the official STF devkit's tools/DatasetViewer/lib/read.py
    ::load_calib_data, using scipy.spatial.transform.Rotation instead of
    pyquaternion (equivalent result: both normalize the quaternion before use,
    and STF's tf-tree quaternions are already unit-norm -- verified against a
    sample entry, see REPRODUCTION_PLAN.md risk #4).

    Returns matrices in KITTI's calib.txt convention:
      P2           : 3x4 camera projection (left cam, since STF's P already
                     encodes rectification -- see note on R0_rect below)
      R0_rect      : 3x3 identity (STF's calibration doesn't separate a rectifying
                     rotation from P; the source devkit itself hard-codes this)
      Tr_velo_to_cam: 3x4 lidar->camera extrinsic
    """
    with open(calib_dir / "calib_cam_stereo_left.json") as f:
        cam = json.load(f)
    with open(calib_dir / "calib_tf_tree_full.json") as f:
        tf_tree = json.load(f)

    cam_frame = "cam_stereo_left_optical"
    lidar_frame = "lidar_hdl64_s3_roof"  # HDL-64E ("roof"-mounted), matches lidar_hdl64_strongest

    T_cam = T_lidar = None
    for item in tf_tree:
        if item["child_frame_id"] == cam_frame:
            T_cam = item["transform"]
        elif item["child_frame_id"] == lidar_frame:
            T_lidar = item["transform"]
    if T_cam is None or T_lidar is None:
        raise RuntimeError(f"could not find {cam_frame!r} / {lidar_frame!r} in tf tree")

    def to_homogeneous(t: dict) -> np.ndarray:
        rot = t["rotation"]
        quat_xyzw = [rot["x"], rot["y"], rot["z"], rot["w"]]  # scipy wants x,y,z,w order
        R = Rotation.from_quat(quat_xyzw).as_matrix()
        trans = t["translation"]
        H = np.eye(4)
        H[:3, :3] = R
        H[:3, 3] = [trans["x"], trans["y"], trans["z"]]
        return H

    zero_to_camera = to_homogeneous(T_cam)
    zero_to_velodyne = to_homogeneous(T_lidar)

    # ROS tf: each transform maps *child* frame points into the *parent* frame,
    # i.e. it is parent_T_child. Both camera and lidar share the same parent
    # ('body'), so velodyne->camera = inv(parent_T_camera) @ parent_T_velodyne.
    velodyne_to_camera = np.linalg.inv(zero_to_camera) @ zero_to_velodyne

    P = np.reshape(cam["P"], (3, 4))
    R0_rect = np.eye(3)  # STF's P already contains rectification -- devkit convention, see plan doc

    Tr_velo_to_cam = velodyne_to_camera[:3, :]  # drop homogeneous row -> 3x4

    return {
        "P0": P, "P1": P, "P2": P, "P3": P,  # STF is not stereo-labeled per-camera in this schema; use left cam for all
        "R0_rect": R0_rect,
        "Tr_velo_to_cam": Tr_velo_to_cam,
        "Tr_imu_to_velo": np.zeros((3, 4)),  # unused by PV-RCNN; KITTI format requires the line to exist
    }


def write_calib_txt(path: Path, calib: dict[str, np.ndarray]) -> None:
    lines = []
    for key in ["P0", "P1", "P2", "P3"]:
        lines.append(f"{key}: " + " ".join(f"{v:.12e}" for v in calib[key].flatten()))
    lines.append("R0_rect: " + " ".join(f"{v:.12e}" for v in calib["R0_rect"].flatten()))
    lines.append("Tr_velo_to_cam: " + " ".join(f"{v:.12e}" for v in calib["Tr_velo_to_cam"].flatten()))
    lines.append("Tr_imu_to_velo: " + " ".join(f"{v:.12e}" for v in calib["Tr_imu_to_velo"].flatten()))
    path.write_text("\n".join(lines) + "\n")


def convert_lidar(src_bin: Path, dst_bin: Path) -> None:
    scan = np.fromfile(src_bin, dtype=np.float32).reshape(-1, 5)
    scan[:, :4].tofile(dst_bin)  # drop channel/ring column (field 4)


def convert_label(src_txt: Path, dst_txt: Path) -> tuple[int, int]:
    """Returns (n_objects, n_remapped_to_dontcare) for reporting."""
    if not src_txt.exists():
        # Some frames in the split files may not have a matching label export
        # (e.g. DontCare-only scenes trimmed by the source pipeline). Write an
        # empty label file rather than silently skipping -- KITTI tooling expects
        # every listed frame to have a label_2 entry, even if empty.
        dst_txt.write_text("")
        return 0, 0

    out_lines = []
    n_total = n_remapped = 0
    for line in src_txt.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        f = line.split(" ")
        if len(f) < 23:
            continue  # malformed line, skip rather than crash the whole conversion
        stf_class = f[0]
        n_total += 1
        kitti_class = CLASS_REMAP.get(stf_class)
        if kitti_class is None:
            kitti_class = "DontCare"
            n_remapped += 1
        # KITTI's 15 fields: type truncated occluded alpha bbox(4) dims(3) loc(3) ry
        kitti_fields = [kitti_class] + f[1:15]
        out_lines.append(" ".join(kitti_fields))

    dst_txt.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))
    return n_total, n_remapped


def tonemap_and_convert(raw_bytes_or_path, dst_png: Path) -> None:
    """STF's cam_stereo_left tiffs are raw 16-bit sensor readings with a narrow
    value range (observed ~360-3846 out of 0-65535 on a sample frame) -- a naive
    mode convert washes out to solid white. Percentile-stretch to 8-bit like any
    normal tone-mapping step, confirmed visually against a projected-label overlay
    (see REPRODUCTION_PLAN.md risk #4 -- this is the check that resolved it)."""
    from PIL import Image
    import io

    if isinstance(raw_bytes_or_path, (bytes, bytearray)):
        img = Image.open(io.BytesIO(raw_bytes_or_path))
    else:
        img = Image.open(raw_bytes_or_path)
    arr = np.array(img).astype(np.float32)
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    arr = (arr * 255).astype(np.uint8)
    Image.fromarray(arr).convert("RGB").save(dst_png)


class ImageSource:
    """Reads cam_stereo_left tiffs either from a pre-merged zip (fast path, no
    37GB re-extraction) or from loose files on disk. STF's cam_stereo_left
    archive is itself a split-zip (see 01_extract_archives.py's docstring for why
    that needs `zip -FF` first) -- this class doesn't do that merge itself, it
    just reads whichever form is available."""

    def __init__(self, stf_root: Path, merged_zip: Path | None):
        self.zf = zipfile.ZipFile(merged_zip) if merged_zip and merged_zip.exists() else None
        self.loose_dir = stf_root / "cam_stereo_left"
        self._names = set(self.zf.namelist()) if self.zf else None

    def convert(self, key: str, dst_png: Path) -> bool:
        if self.zf is not None:
            member = f"/cam_stereo_left/{key}.tiff"
            if member not in self._names:
                member = f"cam_stereo_left/{key}.tiff"
            if member not in self._names:
                return False
            with self.zf.open(member) as f:
                tonemap_and_convert(f.read(), dst_png)
            return True
        else:
            src = self.loose_dir / f"{key}.tiff"
            if not src.exists():
                return False
            tonemap_and_convert(src, dst_png)
            return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=Path("../data/kitti_source"),
                     help="Output of 01_extract_archives.py")
    ap.add_argument("--stf-root", type=Path, default=Path("../../data/stf"),
                     help="Original STF data root, used for images (not pre-extracted by step 1)")
    ap.add_argument("--out", type=Path, default=Path("../data/kitti_format"))
    ap.add_argument("--splits-dir", type=Path, default=Path("../splits"))
    ap.add_argument("--skip-images", action="store_true",
                     help="Skip image conversion (PV-RCNN doesn't need pixels, but OpenPCDet's "
                          "KITTI loader wants image shape metadata -- only skip this for a quick "
                          "lidar/label-only sanity pass)")
    ap.add_argument("--merged-image-zip", type=Path, default=Path("../data/_merged_zips/cam_stereo_left.merged.zip"),
                     help="Pre-merged cam_stereo_left zip (see 01_extract_archives.py docstring). "
                          "Falls back to loose tiffs under --stf-root/cam_stereo_left if not found.")
    args = ap.parse_args()

    source = args.source.resolve()
    out = args.out.resolve()
    splits_dir = args.splits_dir.resolve()
    stf_root = args.stf_root.resolve()
    image_source = None if args.skip_images else ImageSource(stf_root, args.merged_image_zip.resolve())

    # All five splits go under 'training/', not the KITTI-conventional
    # 'testing/' for the three test splits. This isn't a layout preference --
    # it works around a hard-coded check in OpenPCDet's KittiDataset:
    #   self.root_split_path = root_path / ('training' if self.split != 'test' else 'testing')
    # i.e. ANY split name other than the literal string 'test' resolves to
    # 'training/', regardless of what DATA_SPLIT / INFO_PATH says. Our three
    # test splits are named test_clear / test_light_fog / test_dense_fog (not
    # bare 'test', because context.md needs them evaluated as three independent
    # columns, matching Table 2) -- so under OpenPCDet's own logic they would
    # ALREADY resolve to training/, and a 'testing/' directory would just never
    # get read. KITTI IDs are unique across all five splits (see seen_keys
    # below), so nothing collides by keeping one flat pool. See
    # docs/REPRODUCTION_PLAN.md section 2 for the write-up.
    split_specs = [
        ("train_clear.txt", "training", "train"),
        ("val_clear.txt", "training", "val"),
        ("test_clear.txt", "training", "test_clear"),
        ("test_light_fog.txt", "training", "test_light_fog"),
        ("test_dense_fog.txt", "training", "test_dense_fog"),
    ]

    print("Computing calibration (once, shared by all frames)...")
    calib = compute_calibration(source / "calib")
    print("  Tr_velo_to_cam:\n", calib["Tr_velo_to_cam"])

    (out / "ImageSets").mkdir(parents=True, exist_ok=True)
    id_map_rows = []
    next_id = 0
    seen_keys: dict[str, int] = {}  # STF key -> assigned numeric id (shared across splits, e.g. clear-val may overlap frame numbering with test only by coincidence -- keep unique per key)

    total_objects = total_remapped = 0
    missing_lidar = missing_label = missing_image = 0

    for split_file, phase, imageset_name in split_specs:
        pairs = load_split(splits_dir / split_file)
        print(f"\n=== {split_file} -> {phase}/ ImageSets/{imageset_name}.txt ({len(pairs)} frames) ===")

        for sub in ["image_2", "velodyne", "calib", "label_2"]:
            (out / phase / sub).mkdir(parents=True, exist_ok=True)

        ids_this_split = []
        for seq, frame in pairs:
            key = f"{seq}_{frame}"
            if key not in seen_keys:
                seen_keys[key] = next_id
                next_id += 1
            kitti_id = f"{seen_keys[key]:06d}"
            ids_this_split.append(kitti_id)
            id_map_rows.append({"kitti_id": kitti_id, "stf_sequence": seq, "stf_frame": frame, "split": imageset_name})

            # lidar
            src_bin = source / "lidar_hdl64_strongest" / f"{key}.bin"
            dst_bin = out / phase / "velodyne" / f"{kitti_id}.bin"
            if src_bin.exists():
                convert_lidar(src_bin, dst_bin)
            else:
                missing_lidar += 1

            # label -- every split (including the three STF "test" ones) gets a
            # real label_2 written. Unlike stock KITTI's held-out test set, all
            # of our test frames have ground truth, which is what makes it
            # possible to compute AP on light/dense fog at all.
            src_label = source / "gt_labels" / "cam_left_labels_TMP" / f"{key}.txt"
            dst_label = out / phase / "label_2" / f"{kitti_id}.txt"
            if src_label.exists():
                n_obj, n_remap = convert_label(src_label, dst_label)
                total_objects += n_obj
                total_remapped += n_remap
            else:
                missing_label += 1
                dst_label.write_text("")

            # calib (identical file per frame -- STF calibration is static)
            write_calib_txt(out / phase / "calib" / f"{kitti_id}.txt", calib)

            # image
            if image_source is not None:
                dst_png = out / phase / "image_2" / f"{kitti_id}.png"
                if not image_source.convert(key, dst_png):
                    missing_image += 1

        (out / "ImageSets" / f"{imageset_name}.txt").write_text("\n".join(ids_this_split) + "\n")
        print(f"  wrote {len(ids_this_split)} ids to ImageSets/{imageset_name}.txt")

    with open(out / "ImageSets" / "id_mapping.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["kitti_id", "stf_sequence", "stf_frame", "split"])
        writer.writeheader()
        writer.writerows(id_map_rows)

    print(f"\n=== Summary ===")
    print(f"Unique frames converted: {len(seen_keys)}")
    print(f"Total label objects: {total_objects} ({total_remapped} remapped to DontCare)")
    print(f"Missing lidar files: {missing_lidar}")
    print(f"Missing label files (wrote empty label_2): {missing_label}")
    if image_source is not None:
        print(f"Missing image files: {missing_image}")
    print(f"\nOutput: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
