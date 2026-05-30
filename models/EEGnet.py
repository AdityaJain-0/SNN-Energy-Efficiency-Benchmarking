"""
EEGNet: A Compact Convolutional Network for EEG-Based BCIs.

Reference:
    Lawhern et al. (2018) "EEGNet: A compact convolutional neural network
    for EEG-based brain–computer interfaces."
    J. Neural Eng., 15(5).

Adapted for BNCI2014_001 with n_classes=3.

Input: (batch, n_channels=22, n_times=512)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    """
    EEGNet-8,2 configuration adapted to 3-class motor imagery.

    Args:
        n_channels    : number of EEG channels (22)
        n_times       : number of time samples (512)
        n_classes     : 3
        F1            : temporal filters (default 8)
        D             : depth multiplier for depthwise conv (default 2)
        F2            : pointwise filters = F1 * D (default 16)
        kernel_length : temporal kernel size (default 64 ≈ 0.5 s at 128 Hz)
        dropout       : dropout rate (default 0.5)
    """

    def __init__(
        self,
        n_channels=22,
        n_times=512,
        n_classes=3,
        F1=8,
        D=2,
        F2=16,
        kernel_length=64,
        dropout=0.5,
    ):
        super().__init__()
        F2 = F1 * D

        # Block 1: Temporal + Depthwise Spatial Conv
        self.block1 = nn.Sequential(
            # Temporal conv across time
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
            # Depthwise spatial conv across channels
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )

        # Block 2: Separable Conv (depthwise + pointwise)
        self.block2 = nn.Sequential(
            nn.Conv2d(F2, F2, (1, 16), padding=(0, 8), groups=F2, bias=False),
            nn.Conv2d(F2, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        # Compute flattened size dynamically
        self._flat_size = self._get_flat_size(n_channels, n_times, F2)

        # Classifier
        self.classifier = nn.Linear(self._flat_size, n_classes)

        self._init_weights()

    def _get_flat_size(self, n_channels, n_times, F2):
        with torch.no_grad():
            x = torch.zeros(1, 1, n_channels, n_times)
            x = self.block1(x)
            x = self.block2(x)
            return x.numel()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d,)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, C, T)
        x = x.unsqueeze(1)       # (B, 1, C, T)
        x = self.block1(x)       # (B, F2, 1, T//4)
        x = self.block2(x)       # (B, F2, 1, T//32)
        x = x.flatten(1)         # (B, flat)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = EEGNet(n_channels=22, n_times=512, n_classes=3)
    x = torch.randn(4, 22, 512)
    out = model(x)
    print(f"EEGNet output: {out.shape}")   # (4, 3)
    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params:,}")