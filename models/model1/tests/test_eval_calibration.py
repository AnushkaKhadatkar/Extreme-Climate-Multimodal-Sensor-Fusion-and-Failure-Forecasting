"""ECE definition on hand-checkable cases (no GPU: the IoU function is injected)."""

import numpy as np

from eval_calibration import collect, ece, greedy_match, in_dontcare, report


def test_ece_hand_computed():
    conf = np.array([0.95, 0.95, 0.15, 0.15])
    correct = np.array([True, False, False, False])
    # bin of 0.95: acc 0.5 vs conf 0.95 -> 0.45, weight 0.5; bin of 0.15: acc 0 vs 0.15, weight 0.5
    assert abs(ece(conf, correct)["ece"] - (0.5 * 0.45 + 0.5 * 0.15)) < 1e-9


def test_perfectly_calibrated_is_zero():
    conf = np.full(10, 0.7)
    correct = np.array([True] * 7 + [False] * 3)
    assert ece(conf, correct)["ece"] < 1e-9                       # confidence 0.7, accuracy 0.7


def test_greedy_match_one_gt_once():
    iou = np.array([[0.8], [0.9]])
    tp = greedy_match(iou, np.array([0.9, 0.5]))
    assert tp.tolist() == [True, False]                          # higher score wins the only GT


def test_dontcare_overlap():
    det = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], dtype=float)
    dc = np.array([[0, 0, 8, 10]], dtype=float)
    assert in_dontcare(det, dc).tolist() == [True, False]


def test_collect_frame_with_no_detections():
    """OpenPCDet writes name=np.zeros(0) (float64), boxes (0, 7), bbox (0, 4) for an empty frame.
    That used to make collect() fail (`name == "Car"` is a scalar for an empty float array)."""
    gt_infos = [{"point_cloud": {"lidar_idx": "000002"}, "annos": {
        "name": np.array(["Car"]),
        "gt_boxes_lidar": np.array([[10, 0, -1, 4, 1.6, 1.5, 0]], dtype=float),
        "bbox": np.array([[0, 0, 5, 5]], dtype=float)}}]                 # one row per name
    empty = {"frame_id": "000002", "name": np.zeros(0), "score": np.zeros(0),
             "boxes_lidar": np.zeros((0, 7)), "bbox": np.zeros((0, 4))}
    one = {"frame_id": "000002", "name": np.array(["Car"]), "score": np.array([0.8]),
           "boxes_lidar": np.array([[10, 0, -1, 4, 1.6, 1.5, 0]], dtype=float), "bbox": np.array([[0, 0, 5, 5]], dtype=float)}
    fake_iou = lambda d, g: np.ones((len(d), len(g)))
    conf, correct, rng = collect([empty, one, empty], gt_infos, fake_iou)
    assert conf.tolist() == [0.8] and correct.tolist() == [True] and rng.shape == (1,)
    conf0, correct0, rng0 = collect([empty], gt_infos, fake_iou)       # only empty frames
    assert conf0.shape == (0,) and correct0.shape == (0,) and rng0.shape == (0,)


def test_collect_end_to_end_with_fake_iou():
    gt_infos = [{"point_cloud": {"lidar_idx": "000001"}, "annos": {
        "name": np.array(["Car", "Pedestrian", "DontCare"]),
        "gt_boxes_lidar": np.array([[10, 0, -1, 4, 1.6, 1.5, 0], [5, 2, -1, .8, .6, 1.7, 0]], dtype=float),
        "bbox": np.array([[0, 0, 50, 50], [60, 60, 70, 80], [200, 200, 300, 300]], dtype=float)}}]
    det_annos = [{"frame_id": "000001",
                  "name": np.array(["Car", "Car", "Car", "Pedestrian"]),
                  "score": np.array([0.9, 0.6, 0.3, 0.8]),
                  "boxes_lidar": np.array([[10, 0, -1, 4, 1.6, 1.5, 0], [30, 5, -1, 4, 1.6, 1.5, 0],
                                           [50, 0, -1, 4, 1.6, 1.5, 0], [5, 2, -1, .8, .6, 1.7, 0]], dtype=float),
                  "bbox": np.array([[0, 0, 50, 50], [500, 500, 550, 550], [210, 210, 260, 260], [60, 60, 70, 80]],
                                   dtype=float)}]
    fake_iou = lambda d, g: np.array([[0.9], [0.0], [0.0]])[: len(d)]
    conf, correct, rng = collect(det_annos, gt_infos, fake_iou)
    # Car 1 is a TP, car 2 a FP, car 3 a FP inside DontCare (dropped); the Pedestrian isn't a Car.
    assert conf.tolist() == [0.9, 0.6]
    assert correct.tolist() == [True, False]
    out = report(conf, correct, rng)
    assert out["by_range"]["0-20m"]["n"] == 1 and out["by_range"]["20-40m"]["n"] == 1
