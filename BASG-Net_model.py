"""
Public model-only implementation of BASG-Net.

This file releases only the model architecture for cervical cell segmentation.
It removes dataset loading, training, evaluation, visualization, local paths,
logs, checkpoints, and private experimental details.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv5x5(in_c, out_c, stride=1, dilation=1):
    """5 x 5 convolution with padding that preserves spatial resolution."""
    k = 5
    padding = (k // 2) * dilation
    return nn.Conv2d(
        in_c,
        out_c,
        kernel_size=k,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=False,
    )


def conv1x1(in_c, out_c, stride=1):
    """1 x 1 convolution."""
    return nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False)


class DoubleConv(nn.Module):
    """Conv(5 x 5) + BN + ReLU repeated twice."""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            conv5x5(in_c, out_c),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            conv5x5(out_c, out_c),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """MaxPool2d(2) + DoubleConv."""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_c, out_c)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    """Transposed-convolution upsampling + skip concatenation + DoubleConv."""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, in_c // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_c, out_c)

    def forward(self, x_up, x_skip):
        x_up = self.up(x_up)

        diff_y = x_skip.size(2) - x_up.size(2)
        diff_x = x_skip.size(3) - x_up.size(3)

        x_up = F.pad(
            x_up,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
            ],
        )

        x = torch.cat([x_skip, x_up], dim=1)
        x = self.conv(x)
        return x


class BASGNet(nn.Module):
    """
    BASG-Net: a U-Net-style segmentation network with an auxiliary
    background-aware branch and spatial gating of skip connections.
    """

    def __init__(self, in_channels=3, num_classes=1, base_c=64):
        super().__init__()

        c1 = base_c
        c2 = base_c * 2
        c3 = base_c * 4
        c4 = base_c * 8
        c5 = base_c * 16

        # Encoder
        self.inc = DoubleConv(in_channels, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)
        self.down4 = Down(c4, c5)

        # Decoder
        self.up1 = Up(c5, c4)
        self.up2 = Up(c4, c3)
        self.up3 = Up(c3, c2)
        self.up4 = Up(c2, c1)

        # Kept as an identity layer for compatibility with the original design.
        self.cam_layer = nn.Identity()

        # Segmentation head: 5 x 5 convolution followed by 1 x 1 projection.
        self.head = nn.Sequential(
            conv5x5(c1, c1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            conv1x1(c1, num_classes),
        )

        # Background-aware auxiliary branch attached to the H/4 encoder feature map.
        self.background_head = nn.Sequential(
            nn.Conv2d(c3, c3 // 2, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(c3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3 // 2, 1, kernel_size=1, bias=True),
        )

    def _gate_skip(self, skip: torch.Tensor, gate_keep: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial keep gate to skip-connection features.

        skip: [B, C, H, W]
        gate_keep: [B, 1, h, w]
        """
        if gate_keep.shape[2:] != skip.shape[2:]:
            gate = F.interpolate(
                gate_keep,
                size=skip.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        else:
            gate = gate_keep

        return skip * gate

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Auxiliary background branch.
        background_logits = self.background_head(x3)
        background_prob = torch.sigmoid(background_logits)

        # Complementary keep gate for foreground-related feature propagation.
        gate_keep = 1.0 - background_prob

        # Gated skip connections.
        x4_g = self._gate_skip(x4, gate_keep)
        x3_g = self._gate_skip(x3, gate_keep)
        x2_g = self._gate_skip(x2, gate_keep)
        x1_g = self._gate_skip(x1, gate_keep)

        # Decoder
        x = self.up1(x5, x4_g)
        x = self.up2(x, x3_g)
        x = self.up3(x, x2_g)
        x = self.up4(x, x1_g)

        x_cam = self.cam_layer(x)
        seg_logits = self.head(x_cam)

        return seg_logits, background_logits


class UNet(BASGNet):
    """
    Backward-compatible alias.

    The released model is BASG-Net, but this alias is kept in case users
    expect the original U-Net-style class name.
    """

    pass


def build_basg_net(in_channels=3, num_classes=1, base_c=64):
    return BASGNet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_c=base_c,
    )


if __name__ == "__main__":
    model = BASGNet(in_channels=3, num_classes=1, base_c=64)
    x = torch.randn(2, 3, 224, 224)

    seg_logits, background_logits = model(x)

    print("seg_logits:", tuple(seg_logits.shape))
    print("background_logits:", tuple(background_logits.shape))
