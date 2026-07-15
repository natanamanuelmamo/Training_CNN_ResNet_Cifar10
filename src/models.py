"""
models.py
=========
Two CIFAR-10 image classifiers built to be as directly comparable as possible:

  1. PlainCNN   - a "plain" stack of convolutions with NO shortcut connections.
  2. ResNetCNN  - the exact same layer layout, but each pair of conv layers is
                  wrapped in a residual block with a shortcut connection
                  (identity, or a 1x1 projection when shape changes).

This mirrors the controlled experiment in He et al. 2016 ("Deep Residual
Learning for Image Recognition"), Section 4.2, where a plain network and a
residual network are given IDENTICAL depth/width so that any performance
difference can only be attributed to the shortcut connections themselves.

Architecture (n=3 -> 20 weight layers, i.e. "ResNet-20" for CIFAR):
    stem:    3x3 conv, 16 channels
    stage1:  n blocks, 16  channels, stride 1
    stage2:  n blocks, 32  channels, first block stride 2 (downsample)
    stage3:  n blocks, 64  channels, first block stride 2 (downsample)
    head:    global average pool -> fully connected (10 classes)

Total weight layers = 1 (stem) + 2*n*3 (three stages, 2 conv layers/block) + 1 (fc)
                     = 6n + 2  ->  n=3 gives 20 layers, matching the paper's
                       smallest CIFAR ResNet variant.

Both networks share this exact skeleton, so parameter counts are almost
identical (the residual network has a handful of extra parameters only in
the three 1x1 projection shortcuts used when the spatial size / channel
count changes).
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class PlainBlock(nn.Module):
    """Two 3x3 conv-BN-ReLU layers, NO shortcut connection."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out)          # no residual addition here
        return out


class ResidualBlock(nn.Module):
    """
    Two 3x3 conv-BN layers PLUS a shortcut connection, implemented from
    scratch (no torchvision.models import).

    H(x) = ReLU( F(x) + shortcut(x) )

    - If in_channels == out_channels and stride == 1: shortcut is the
      identity (no extra parameters) -> H(x) = F(x) + x
    - Otherwise (channel count or spatial size changes): shortcut is a
      1x1 convolution ("projection shortcut", Eq. 12 in the paper),
      H(x) = F(x) + W_s * x
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.needs_projection = (stride != 1) or (in_channels != out_channels)
        if self.needs_projection:
            # 1x1 projection shortcut to match spatial size / channel count
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = out + identity          # <-- the residual (shortcut) connection
        out = self.relu(out)
        return out


# ---------------------------------------------------------------------------
# Full networks
# ---------------------------------------------------------------------------

class _BaseCIFARNet(nn.Module):
    """Shared skeleton: stem -> 3 stages -> global avg pool -> fc."""

    def __init__(self, block_cls, n=3, num_classes=10):
        super().__init__()
        self.in_channels = 16

        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.stage1 = self._make_stage(block_cls, 16, n, stride=1)
        self.stage2 = self._make_stage(block_cls, 32, n, stride=2)
        self.stage3 = self._make_stage(block_cls, 64, n, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

        self._init_weights()

    def _make_stage(self, block_cls, out_channels, n_blocks, stride):
        layers = [block_cls(self.in_channels, out_channels, stride=stride)]
        self.in_channels = out_channels
        for _ in range(n_blocks - 1):
            layers.append(block_cls(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class PlainCNN(_BaseCIFARNet):
    """Plain CNN, no shortcut connections. Model 1 in the assignment."""

    def __init__(self, n=3, num_classes=10):
        super().__init__(block_cls=PlainBlock, n=n, num_classes=num_classes)


class ResNetCNN(_BaseCIFARNet):
    """CNN with hand-implemented residual blocks. Model 2 in the assignment."""

    def __init__(self, n=3, num_classes=10):
        super().__init__(block_cls=ResidualBlock, n=n, num_classes=num_classes)


# ---------------------------------------------------------------------------
# Quick self-test / sanity check when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    x = torch.randn(2, 3, 32, 32)

    plain = PlainCNN(n=3)
    resnet = ResNetCNN(n=3)

    out_plain = plain(x)
    out_resnet = resnet(x)

    print("PlainCNN  output shape:", out_plain.shape)   # expect [2, 10]
    print("ResNetCNN output shape:", out_resnet.shape)  # expect [2, 10]

    n_plain = sum(p.numel() for p in plain.parameters() if p.requires_grad)
    n_resnet = sum(p.numel() for p in resnet.parameters() if p.requires_grad)
    print(f"PlainCNN  trainable params:  {n_plain:,}")
    print(f"ResNetCNN trainable params:  {n_resnet:,}")

    # Identity-fallback sanity check (mirrors the trainer's guide exercise):
    # zero out a residual block's conv weights -> block should output ~= input
    block = ResidualBlock(16, 16, stride=1)
    with torch.no_grad():
        for p in block.parameters():
            p.zero_()
        block.bn1.weight.fill_(1.0)  # keep BN a no-op-ish passthrough scale
        block.bn2.weight.fill_(1.0)
    test_in = torch.randn(1, 16, 8, 8)
    test_out = block(test_in)
    print("Identity fallback check - max abs diff:",
          (test_out - torch.relu(test_in)).abs().max().item())
