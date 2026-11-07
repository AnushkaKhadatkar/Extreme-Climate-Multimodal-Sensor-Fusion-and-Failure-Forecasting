"""Camera stream: ResNet-50 (ImageNet weights, frozen) + a small FPN to stride 8.

The ResNet is frozen in v1 (FrozenBatchNorm, no gradients). OpenPCDet's
adam_onecycle schedule uses one learning rate (3e-3) for every parameter, which
would wreck pretrained ImageNet features. Freezing also keeps memory within a
24 GB GPU next to spconv. Only the FPN is trained. Recorded in DESIGN.md.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops.misc import FrozenBatchNorm2d

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageBackbone(nn.Module):
    def __init__(self, out_channels: int = 256, pretrained: bool = True, freeze: bool = True):
        super().__init__()
        weights = "IMAGENET1K_V2" if pretrained else None
        r = torchvision.models.resnet50(weights=weights, norm_layer=FrozenBatchNorm2d)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4
        if freeze:
            for p in self.parameters():
                p.requires_grad_(False)
        self.frozen = freeze

        # FPN on C3 (stride 8, 512 ch), C4 (16, 1024), C5 (32, 2048) -> one stride-8 map.
        self.lat3 = nn.Conv2d(512, out_channels, 1)
        self.lat4 = nn.Conv2d(1024, out_channels, 1)
        self.lat5 = nn.Conv2d(2048, out_channels, 1)
        self.smooth = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 3, H, W) in [0, 1] -> (B, C, H/8, W/8)."""
        x = (images - self.mean) / self.std
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen):
            x = self.stem(x)
            c2 = self.layer1(x)
            c3 = self.layer2(c2)
            c4 = self.layer3(c3)
            c5 = self.layer4(c4)
        p5 = self.lat5(c5)
        p4 = self.lat4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        return self.smooth(p3)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:  # keep the frozen trunk in eval mode (FrozenBatchNorm anyway, but be explicit)
            for m in (self.stem, self.layer1, self.layer2, self.layer3, self.layer4):
                m.eval()
        return self
