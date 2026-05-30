"""
ShallowConvNet: Shallow Convolutional Neural Network for EEG decoding.

Reference:
    Schirrmeister et al. (2017) "Deep learning with convolutional neural
    networks for EEG decoding and visualization."
    Human Brain Mapping, 38(11).

Adapted for BNCI2014_001 with n_classes=3.

Input: (batch, n_channels=22, n_times=512)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SquareActivation(nn.Module):
    """Squaring activation as in the original ShallowConvNet."""
    def forward(self, x):
        return x ** 2


class LogActivation(nn.Module):
    """Safe log activation: log(max(x, eps))."""
    def forward(self, x):
        return torch.log(torch.clamp(x, min=1e-6))


class ShallowConvNet(nn.Module):
    """
    ShallowConvNet for 3-class motor imagery.

    Architecture:
        1. Temporal Conv:   (1, 1, 1, 25) → 40 feature maps
        2. Spatial Conv:    (1, n_channels, 1, 1) depthwise → fuses channels
        3. BatchNorm → Square → AvgPool → Log → Dropout
        4. Linear classifier

    Args:
        n_channels   : EEG channels (22)
        n_times      : time samples (512)
        n_classes    : 3
        n_filters    : temporal filter count (default 40)
        filter_time  : temporal kernel length (default 25)
        pool_size    : average pooling size (default 75)
        pool_stride  : pooling stride (default 15)
        dropout      : dropout rate (default 0.5)
    """

    def __init__(
        self,
        n_channels=22,
        n_times=512,
        n_classes=3,
        n_filters=40,
        filter_time=25,
        pool_size=75,
        pool_stride=15,
        dropout=0.5,
    ):
        super().__init__()

        # Temporal convolution
        self.temporal_conv = nn.Conv2d(
            1, n_filters, (1, filter_time), bias=False
        )

        # Spatial (depthwise) convolution over channels
        self.spatial_conv = nn.Conv2d(
            n_filters, n_filters, (n_channels, 1), bias=False
        )

        self.bn = nn.BatchNorm2d(n_filters, momentum=0.1, eps=1e-5)
        self.square = SquareActivation()
        self.pool = nn.AvgPool2d(
            kernel_size=(1, pool_size),
            stride=(1, pool_stride),
        )
        self.log = LogActivation()
        self.drop = nn.Dropout(p=dropout)

        # Determine classifier input size
        self._flat_size = self._get_flat_size(n_channels, n_times, n_filters)
        self.classifier = nn.Linear(self._flat_size, n_classes)

        self._init_weights()

    def _get_flat_size(self, n_channels, n_times, n_filters):
        with torch.no_grad():
            x = torch.zeros(1, 1, n_channels, n_times)
            x = self.temporal_conv(x)
            x = self.spatial_conv(x)
            x = self.pool(x)
            return x.numel()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, C, T)
        x = x.unsqueeze(1)            # (B, 1, C, T)
        x = self.temporal_conv(x)     # (B, F, C, T-24)
        x = self.spatial_conv(x)      # (B, F, 1, T-24)
        x = self.bn(x)
        x = self.square(x)
        x = self.pool(x)
        x = self.log(x)
        x = self.drop(x)
        x = x.flatten(1)             # (B, flat)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = ShallowConvNet(n_channels=22, n_times=512, n_classes=3)
    x = torch.randn(4, 22, 512)
    out = model(x)
    print(f"ShallowConvNet output: {out.shape}")
    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params:,}")