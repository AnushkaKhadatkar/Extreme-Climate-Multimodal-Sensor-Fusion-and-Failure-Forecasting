#!/usr/bin/env python3
"""ECE and D-ECE for Car detections, from OpenPCDet's saved predictions.

Definition (written into docs/DESIGN.md section 5, identical for every system):
  - Detections: class 'Car', every score OpenPCDet keeps (SCORE_THRESH 0.1).
  - Correct = true positive at 3D IoU >= 0.5 (rotated boxes in the lidar frame).
    Matching is greedy by descending score; each ground-truth Car matches once.
  - Ground truth: the frame's Car boxes from the info file (post-D1 labels, so Cars
    without a 3D box are DontCare already).
  - DontCare: an unmatched detection whose 2D image box lies mostly inside a
    DontCare 2D box (intersection / detection area > 0.5) is excluded, the same
    "ignore region" idea KITTI eval uses.
  - Confidence = detector score. 15 equal-width bins on [0, 1].
    ECE = sum_b (n_b / N) * |accuracy_b - mean confidence_b|.
  - D-ECE: the same per range bin of the detection centre (0-20, 20-40, 40-70.4 m).

Usage (inside the Model 1 image; IoU uses pcdet's GPU op):
    python eval_calibration.py --result <eval dir>/result.pkl \
        --infos /opt/OpenPCDet/data/kitti_format_mm/kitti_infos_test_clear.pkl --out ece_test_clear.json
Works for the baseline's result.pkl too, which is the point: one definition for all systems.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

N_BINS = 15
RANGE_BINS = [(0.0, 20.0), (20.0, 40.0), (40.0, 70.4)]


def greedy_match(iou: np.ndarray, scores: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    """iou (n_det, n_gt), scores (n_det,) -> bool (n_det,) true positives."""
    tp = np.zeros(len(scores), dtype=bool)
    if iou.size == 0:
        return tp
    taken = np.zeros(iou.shape[1], dtype=bool)
    for d in np.argsort(-scores, kind="stable"):
        cand = np.where(~taken, iou[d], -1.0)
        g = int(np.argmax(cand))
        if cand[g] >= thresh:
            tp[d] = True
            taken[g] = True
    return tp


def in_dontcare(det_bbox: np.ndarray, dc_bbox: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    """det_bbox (n, 4), dc_bbox (m, 4) as x1 y1 x2 y2 -> bool (n,)."""
    if len(det_bbox) == 0 or len(dc_bbox) == 0:
        return np.zeros(len(det_bbox), dtype=bool)
    x1 = np.maximum(det_bbox[:, None, 0], dc_bbox[None, :, 0])
    y1 = np.maximum(det_bbox[:, None, 1], dc_bbox[None, :, 1])
    x2 = np.minimum(det_bbox[:, None, 2], dc_bbox[None, :, 2])
    y2 = np.minimum(det_bbox[:, None, 3], dc_bbox[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = np.clip(det_bbox[:, 2] - det_bbox[:, 0], 1e-6, None) * np.clip(det_bbox[:, 3] - det_bbox[:, 1], 1e-6, None)
    return (inter / area[:, None]).max(axis=1) > thresh


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> dict:
    """Expected calibration error with equal-width bins; also returns the reliability diagram."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    total, bins = 0.0, []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        acc, cf = float(correct[m].mean()), float(conf[m].mean())
        total += m.sum() / len(conf) * abs(acc - cf)
        bins.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int(m.sum()),
                     "accuracy": acc, "confidence": cf})
    return {"ece": float(total) if len(conf) else float("nan"), "n": int(len(conf)), "bins": bins}


def collect(det_annos: list, gt_infos: list, iou_fn) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> confidences, correct flags, ranges (one entry per kept Car detection)."""
    gt_by_frame = {str(info["point_cloud"]["lidar_idx"]): info["annos"] for info in gt_infos}
    confs, correct, ranges = [], [], []
    for det in det_annos:
        gt = gt_by_frame[str(det["frame_id"])]
        car_det = det["name"] == "Car"
        boxes, scores, bbox2d = det["boxes_lidar"][car_det], det["score"][car_det], det["bbox"][car_det]
        n_obj = int((gt["name"] != "DontCare").sum())
        gt_names = gt["name"][:n_obj]
        gt_boxes = gt["gt_boxes_lidar"][:n_obj][gt_names == "Car"]
        dc_bbox = gt["bbox"][gt["name"] == "DontCare"]
        iou = iou_fn(boxes, gt_boxes) if len(boxes) and len(gt_boxes) else np.zeros((len(boxes), len(gt_boxes)))
        tp = greedy_match(iou, scores)
        keep = tp | ~in_dontcare(bbox2d, dc_bbox)
        confs.append(scores[keep])
        correct.append(tp[keep])
        ranges.append(np.linalg.norm(boxes[keep, :2], axis=1))
    cat = lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dt)
    return cat(confs, np.float64), cat(correct, bool), cat(ranges, np.float64)


def report(conf, correct, rng) -> dict:
    out = {"overall": ece(conf, correct), "by_range": {}}
    for lo, hi in RANGE_BINS:
        m = (rng >= lo) & (rng < hi)
        out["by_range"][f"{lo:g}-{hi:g}m"] = ece(conf[m], correct[m])
    return out


def pcdet_iou3d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    import torch
    from pcdet.ops.iou3d_nms import iou3d_nms_utils
    ta = torch.from_numpy(np.ascontiguousarray(a[:, :7], dtype=np.float32)).cuda()
    tb = torch.from_numpy(np.ascontiguousarray(b[:, :7], dtype=np.float32)).cuda()
    return iou3d_nms_utils.boxes_iou3d_gpu(ta, tb).cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--result", type=Path, required=True, help="result.pkl written by OpenPCDet's test.py")
    ap.add_argument("--infos", type=Path, required=True, help="kitti_infos_<split>.pkl for the same split")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    det_annos = pickle.load(open(args.result, "rb"))
    gt_infos = pickle.load(open(args.infos, "rb"))
    if len(det_annos) != len(gt_infos):
        raise SystemExit(f"{len(det_annos)} predictions vs {len(gt_infos)} infos: wrong split?")
    res = report(*collect(det_annos, gt_infos, pcdet_iou3d))
    res["frames"] = len(det_annos)
    o = res["overall"]
    print(f"Car ECE (IoU 0.5, {N_BINS} bins): {o['ece']:.4f} over {o['n']} detections, {res['frames']} frames")
    for k, v in res["by_range"].items():
        print(f"  D-ECE {k:>10}: {v['ece']:.4f} (n={v['n']})")
    if args.out:
        args.out.write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
