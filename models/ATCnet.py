"""
ATCNet: Attention Temporal Convolutional Network for EEG Motor Imagery.

Reference:
    Altaheri et al. (2022) "Physics-informed attention temporal convolutional
    network for EEG-based motor imagery classification."
    IEEE Trans. Ind. Informatics.

Adapted here for BNCI2014_001 with n_classes=3.

Input shape: (batch, n_channels=22, n_times=512)  [128 Hz × 4 s]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class EEGDepthwiseConv(nn.Module):
    """Temporal conv + depthwise spatial conv block (EEGNet-style front-end)."""

    def __init__(self, n_channels, n_filters=16, kernel_temporal=64, D=2, dropout=0.3):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, n_filters, (1, kernel_temporal), padding=(0, kernel_temporal // 2), bias=False),
            nn.BatchNorm2d(n_filters),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(n_filters, n_filters * D, (n_channels, 1), groups=n_filters, bias=False),
            nn.BatchNorm2d(n_filters * D),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        self.out_filters = n_filters * D

    def forward(self, x):
        # x: (B, C, T) → (B, 1, C, T)
        x = x.unsqueeze(1)
        x = self.temporal(x)
        x = self.spatial(x)          # (B, F, 1, T//8)
        x = x.squeeze(2)             # (B, F, T//8)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads=2, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.d_k = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape
        QKV = self.qkv(x).reshape(B, T, 3, self.h, self.d_k).permute(2, 0, 3, 1, 4)
        Q, K, V = QKV[0], QKV[1], QKV[2]
        attn = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = self.dropout(attn.softmax(dim=-1))
        out = (attn @ V).transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


class TCNBlock(nn.Module):
    """Single residual dilated causal conv block."""

    def __init__(self, in_ch, out_ch, kernel=4, dilation=1, dropout=0.3):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv1 = nn.utils.parametrize.register_parametrization if False else \
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation),
                nn.BatchNorm1d(out_ch), nn.ELU(), nn.Dropout(dropout),
            )
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation),
            nn.BatchNorm1d(out_ch), nn.ELU(), nn.Dropout(dropout),
        )
        self.trim = pad
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = self.conv1(x)[..., : x.shape[-1]]
        out = self.conv2(out)[..., : x.shape[-1]]
        return F.elu(out + self.res(x))


# ---------------------------------------------------------------------------
# ATCNet
# ---------------------------------------------------------------------------

class ATCNet(nn.Module):
    """
    ATCNet for 3-class motor imagery.

    Args:
        n_channels  : EEG channels (22 for BNCI2014_001)
        n_times     : time samples after resampling (512 = 4s @ 128 Hz)
        n_classes   : 3
        n_windows   : number of sliding attention windows
        eeg_filters : temporal conv output filters
        tcn_filters : TCN hidden dim
        tcn_depth   : stacked TCN blocks
        attn_heads  : multi-head attention heads
        dropout     : dropout rate
    """

    def __init__(
        self,
        n_channels=22,
        n_times=512,
        n_classes=3,
        n_windows=3,
        eeg_filters=16,
        D=2,
        tcn_filters=32,
        tcn_depth=2,
        attn_heads=2,
        dropout=0.3,
    ):
        super().__init__()
        self.n_windows = n_windows

        # Shared EEG front-end
        self.eeg_block = EEGDepthwiseConv(
            n_channels=n_channels,
            n_filters=eeg_filters,
            kernel_temporal=64,
            D=D,
            dropout=dropout,
        )
        feat_dim = eeg_filters * D  # = 32

        # Shared attention + TCN per window
        self.attn = MultiHeadSelfAttention(feat_dim, n_heads=attn_heads, dropout=dropout)
        self.attn_norm = nn.LayerNorm(feat_dim)

        tcn_blocks = []
        in_ch = feat_dim
        for i in range(tcn_depth):
            tcn_blocks.append(TCNBlock(in_ch, tcn_filters, kernel=4, dilation=2**i, dropout=dropout))
            in_ch = tcn_filters
        self.tcn = nn.Sequential(*tcn_blocks)

        # Window outputs → classifier
        self.classifier = nn.Sequential(
            nn.Linear(n_windows * tcn_filters, 64),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, C, T)
        feat = self.eeg_block(x)          # (B, F, T')
        T_feat = feat.shape[-1]
        win_size = T_feat // self.n_windows

        window_outputs = []
        for i in range(self.n_windows):
            start = i * win_size
            end = start + win_size if i < self.n_windows - 1 else T_feat
            w = feat[:, :, start:end]     # (B, F, win_size)

            # Attention (expects (B, T, F))
            w_t = w.permute(0, 2, 1)
            w_t = self.attn_norm(w_t + self.attn(w_t))
            w = w_t.permute(0, 2, 1)

            # TCN
            w = self.tcn(w)               # (B, tcn_filters, win_size)
            w = w.mean(dim=-1)            # (B, tcn_filters) – global avg
            window_outputs.append(w)

        combined = torch.cat(window_outputs, dim=-1)  # (B, n_windows * tcn_filters)
        return self.classifier(combined)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = ATCNet(n_channels=22, n_times=512, n_classes=3)
    x = torch.randn(4, 22, 512)
    out = model(x)
    print(f"ATCNet output: {out.shape}")   # (4, 3)
    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params:,}")