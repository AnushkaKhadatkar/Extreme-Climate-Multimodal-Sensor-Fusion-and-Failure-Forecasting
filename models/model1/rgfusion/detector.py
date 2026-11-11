"""Model 1 detector: lidar BEV + camera tokens, fused by reliability-gated attention.

Pipeline (OpenPCDet modules unless noted):
    MeanVFE -> VoxelBackBone8x -> HeightCompression -> BaseBEVBackbone   (lidar BEV, 512 ch)
    [Model 1] image backbone -> camera tokens on the BEV grid
    [Model 1] reliability r_lidar, r_cam -> gated attention over {lidar, camera} tokens
    AnchorHeadSingle on the fused BEV map                                  (same anchors as KITTI SECOND)

With FUSION.USE_CAMERA False the Model 1 parts are skipped and the head runs on
the lidar BEV map: the lidar-only ablation, same head and training setup.

Evaluation extras (environment variables, off by default):
    M1_RELIABILITY_LOG=<file.jsonl>  append {frame_id, r_lidar, r_cam, cam_weight} per frame
"""

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from pcdet.models.detectors.detector3d_template import Detector3DTemplate

from .aux_heads import BEVHeatmapHead, ReliabilityTarget, car_heatmap, focal_loss_per_sample
from .gated_fusion import GatedModalityAttention, stream_bias
from .image_backbone import ImageBackbone
from .modality_tokens import CameraToBEV, bev_cell_centers
from .reliability import ReliabilityEstimator, camera_stats, lidar_stats


class Model1Fusion(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

        f = model_cfg.FUSION
        self.use_camera = bool(f.USE_CAMERA)
        if not self.use_camera:
            return

        self.pc_range = [float(v) for v in dataset.point_cloud_range]
        stride = model_cfg.DENSE_HEAD.ANCHOR_GENERATOR_CONFIG[0]["feature_map_stride"]
        gx, gy = int(dataset.grid_size[0]) // stride, int(dataset.grid_size[1]) // stride
        self.grid_hw = (gy, gx)                                   # (ny, nx), e.g. (200, 176)
        bev_ch = self.backbone_2d.num_bev_features                # 512
        d = int(f.TOKEN_DIM)
        self.image_size = tuple(f.IMAGE_SIZE)                     # (H, W) fed to the image backbone

        self.image_backbone = ImageBackbone(int(f.IMAGE_CHANNELS), pretrained=bool(f.PRETRAINED_IMAGE_BACKBONE),
                                            freeze=bool(f.FREEZE_IMAGE_BACKBONE))
        self.cam_to_bev = CameraToBEV(int(f.IMAGE_CHANNELS), d, self.pc_range, self.grid_hw, f.SAMPLE_HEIGHTS)
        self.lidar_tokens = nn.Sequential(nn.Conv2d(bev_ch, d, 1, bias=False), nn.BatchNorm2d(d), nn.ReLU(inplace=True))
        self.fusion = GatedModalityAttention(d, int(f.NUM_HEADS), int(f.NUM_LAYERS))
        self.fused_to_bev = nn.Sequential(nn.Conv2d(d, bev_ch, 1, bias=False), nn.BatchNorm2d(bev_ch), nn.ReLU(inplace=True))
        self.rel_lidar = ReliabilityEstimator(bev_ch, n_stats=8)
        self.rel_cam = ReliabilityEstimator(int(f.IMAGE_CHANNELS), n_stats=5)
        self.aux_lidar = BEVHeatmapHead(d)
        self.aux_cam = BEVHeatmapHead(d)
        self.rel_target = ReliabilityTarget(2, float(f.TAU_MOMENTUM))
        self.aux_weight = float(f.AUX_WEIGHT)
        self.rel_weight = float(f.RELIABILITY_WEIGHT)

        H, W = self.grid_hw
        xy = bev_cell_centers(self.pc_range, self.grid_hw, torch.device("cpu"))
        lo = torch.tensor(self.pc_range[:2])
        hi = torch.tensor(self.pc_range[3:5])
        self.register_buffer("pos_xy", (xy - lo) / (hi - lo), persistent=False)   # (HW, 2) in [0, 1]
        self.m1 = {}

    # ------------------------------------------------------------------ forward
    def forward(self, batch_dict):
        for cur_module in self.module_list:
            if self.use_camera and cur_module is self.dense_head:
                batch_dict = self.fuse(batch_dict)
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)
            return {"loss": loss}, tb_dict, disp_dict

        pred_dicts, recall_dicts = self.post_processing(batch_dict)
        if self.use_camera:
            self._log_reliability(batch_dict)
        return pred_dicts, recall_dicts

    def fuse(self, batch_dict):
        bev = batch_dict["spatial_features_2d"]                                   # (B, 512, H, W)
        B, _, H, W = bev.shape
        images = batch_dict["images"]                                             # (B, 3, H_img, W_img) in [0, 1]
        image_hw = images.shape[-2:]
        small = F.interpolate(images, size=self.image_size, mode="bilinear", align_corners=False)
        img_feats = self.image_backbone(small)

        t_lidar = self.lidar_tokens(bev)
        t_cam, cam_valid = self.cam_to_bev(img_feats, batch_dict, image_hw)

        logit_l = self.rel_lidar(bev, lidar_stats(batch_dict["points"], B))
        logit_c = self.rel_cam(img_feats, camera_stats(small))
        r = torch.sigmoid(torch.stack([logit_l, logit_c], dim=1))               # (B, 2)

        tokens = torch.stack([t_lidar, t_cam], dim=-1)                            # (B, D, H, W, 2)
        tokens = tokens.flatten(2, 3).permute(0, 2, 3, 1)                         # (B, HW, 2, D)
        valid = torch.stack([torch.ones_like(cam_valid), cam_valid], dim=-1).flatten(1, 2)  # (B, HW, 2)
        bias = stream_bias(r.detach(), valid)
        fused, weights = self.fusion(tokens, bias, self.pos_xy)
        fused = fused.permute(0, 2, 1).reshape(B, -1, H, W)
        batch_dict["spatial_features_2d"] = self.fused_to_bev(fused)

        cam_weight = (weights[..., 1] * valid[..., 1]).sum(1) / valid[..., 1].sum(1).clamp(min=1)
        self.m1 = {"r_logits": torch.stack([logit_l, logit_c], dim=1), "r": r,
                   "t_lidar": t_lidar, "t_cam": t_cam, "cam_weight": cam_weight}
        return batch_dict

    # --------------------------------------------------------------------- loss
    def get_training_loss(self, batch_dict):
        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {"loss_rpn": loss_rpn.item(), **tb_dict}
        loss = loss_rpn
        if self.use_camera:
            m = self.m1
            B = m["r"].shape[0]
            target = car_heatmap(batch_dict["gt_boxes"], self.pc_range, self.grid_hw)
            l_lidar = focal_loss_per_sample(self.aux_lidar(m["t_lidar"]), target)
            l_cam = focal_loss_per_sample(self.aux_cam(m["t_cam"]), target)
            aux = torch.stack([l_lidar, l_cam], dim=1)                            # (B, 2)
            ones = torch.ones(B, device=aux.device)
            keep = torch.stack([batch_dict.get("m1_keep_lidar", ones).float(),
                                batch_dict.get("m1_keep_cam", ones).float()], dim=1)
            # A wiped-out stream's aux head would only learn noise: train it on surviving input only.
            aux_loss = (aux * (keep > 0).float()).sum() / B
            r_star = self.rel_target(aux.detach(), keep)
            rel_loss = F.binary_cross_entropy_with_logits(m["r_logits"].float(), r_star)
            loss = loss + self.aux_weight * aux_loss + self.rel_weight * rel_loss
            tb_dict.update({
                "m1_aux_lidar": l_lidar.mean().item(), "m1_aux_cam": l_cam.mean().item(),
                "m1_rel_loss": rel_loss.item(),
                "m1_r_lidar": m["r"][:, 0].mean().item(), "m1_r_cam": m["r"][:, 1].mean().item(),
                "m1_cam_weight": m["cam_weight"].mean().item(),
                "m1_tau_lidar": self.rel_target.tau[0].item(), "m1_tau_cam": self.rel_target.tau[1].item(),
            })
        tb_dict["loss"] = loss.item()
        return loss, tb_dict, {}

    # --------------------------------------------------------------------- eval
    def _log_reliability(self, batch_dict):
        path = os.environ.get("M1_RELIABILITY_LOG")
        if not path:
            return
        r = self.m1["r"].detach().cpu().tolist()
        w = self.m1["cam_weight"].detach().cpu().tolist()
        with open(path, "a") as f:
            for fid, (rl, rc), cw in zip(batch_dict["frame_id"], r, w):
                f.write(json.dumps({"frame_id": str(fid), "r_lidar": rl, "r_cam": rc, "cam_weight": cw}) + "\n")
