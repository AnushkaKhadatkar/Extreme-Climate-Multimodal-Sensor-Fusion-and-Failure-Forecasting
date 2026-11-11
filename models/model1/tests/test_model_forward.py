"""End to end through OpenPCDet on synthetic frames: augmentation (incl. modality
dropout), voxelisation, collation, a training step with backward, and inference.
Needs a CUDA GPU (spconv) and the Model 1 dataset config, which lives on the data
branch (models/model1/configs/stf_multimodal_dataset.yaml) until both PRs merge."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

import rgfusion.register  # noqa: E402,F401
from pcdet.config import cfg_from_yaml_file  # noqa: E402
from pcdet.datasets.dataset import DatasetTemplate  # noqa: E402
from pcdet.models import build_network, load_data_to_gpu  # noqa: E402
from easydict import EasyDict  # noqa: E402

CFG_DIR = Path(__file__).resolve().parents[1] / "configs"
V2R = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
P2 = np.array([[1000, 0, 960, 0], [0, 1000, 512, 0], [0, 0, 1, 0]], dtype=np.float32)


class Synthetic(DatasetTemplate):
    def __len__(self):
        return 2

    def sample(self, i, rng):
        n = 6000
        x = rng.uniform(2, 68, n)
        y = rng.uniform(-0.6, 0.6, n) * x                     # roughly inside the camera FOV
        z = rng.uniform(-2.5, 0.5, n)
        pts = np.stack([x, y, z, rng.uniform(0, 255, n)], 1).astype(np.float32)
        boxes = np.array([[15, 1, -0.9, 3.9, 1.6, 1.5, 0.1], [30, -4, -0.9, 4.2, 1.7, 1.5, 1.4]], np.float32)
        return self.prepare_data({
            "frame_id": f"{i:06d}", "points": pts, "gt_boxes": boxes, "gt_names": np.array(["Car", "Car"]),
            "images": rng.uniform(0, 1, (1024, 1920, 3)).astype(np.float32),
            "trans_lidar_to_cam": V2R, "trans_cam_to_img": P2,
            "image_shape": np.array([1024, 1920], np.int32),
        })


def build(cfg_name, training):
    dataset_yaml = CFG_DIR / "stf_multimodal_dataset.yaml"
    if not dataset_yaml.exists():
        pytest.skip("stf_multimodal_dataset.yaml (data branch) not present")
    cfg = EasyDict()
    cwd = os.getcwd()
    tmp = Path(tempfile.mkdtemp(prefix="model1_cfg_"))   # the repo may be mounted read-only
    (tmp / "cfgs" / "dataset_configs").mkdir(parents=True, exist_ok=True)
    (tmp / "cfgs" / "dataset_configs" / dataset_yaml.name).write_text(dataset_yaml.read_text())
    os.chdir(tmp)                                   # _BASE_CONFIG_ resolves from tools/-like cwd
    try:
        cfg_from_yaml_file(str(CFG_DIR / cfg_name), cfg)
    finally:
        os.chdir(cwd)
    cfg.MODEL.FUSION.PRETRAINED_IMAGE_BACKBONE = False
    ds = Synthetic(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=training, root_path=tmp)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=ds).cuda()
    return ds, model


@pytest.mark.parametrize("cfg_name", ["model1_lidar_rgb.yaml", "model1_lidar_only.yaml"])
def test_train_step_and_inference(cfg_name):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    ds, model = build(cfg_name, training=True)
    model.train()
    batch = ds.collate_batch([ds.sample(i, rng) for i in range(2)])
    load_data_to_gpu(batch)
    with torch.cuda.amp.autocast():
        ret, tb, _ = model(batch)
    assert torch.isfinite(ret["loss"]), tb
    ret["loss"].backward()
    if cfg_name == "model1_lidar_rgb.yaml":
        for k in ("m1_aux_lidar", "m1_aux_cam", "m1_rel_loss", "m1_r_lidar", "m1_r_cam"):
            assert np.isfinite(tb[k]), k
        assert model.fusion.query.grad is not None
        assert not any(p.requires_grad for p in model.image_backbone.layer4.parameters())  # frozen

    ds_eval, _ = build(cfg_name, training=False)
    model.eval()
    batch = ds_eval.collate_batch([ds_eval.sample(i, rng) for i in range(2)])
    load_data_to_gpu(batch)
    with torch.no_grad():
        preds, _ = model(batch)
    assert len(preds) == 2 and "pred_scores" in preds[0]
