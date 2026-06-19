#Full SNN: 

"""
Spiking Neural Network (SNN) for EEG Motor Imagery Classification.

Framework : SpikingJelly  (pip install spikingjelly)
Neuron    : Parametric Leaky Integrate-and-Fire (PLIF) — learnable tau
Training  : STBP with ATan surrogate gradient

TWO VARIANTS — both available via make_snn(fully_spiking=True/False):

  HybridSNN (fully_spiking=False)  [original]
  ┌──────────────────────────────────────────────────────────────┐
  │ ANN CNN encoder (Temporal+Spatial Conv, BN, ELU, AvgPool)   │
  │   → dense feature vector                                     │
  │ → 3× spiking FC layers (PLIF)                               │
  │ → mean spike rate → logits                                   │
  └──────────────────────────────────────────────────────────────┘
  Pro : CNN front-end is a proven EEG feature extractor.
  Con : CNN layers consume ANN-level energy (dense MACs).

  FullySNN (fully_spiking=True)  [NEW — addresses novelty concern]
  ┌──────────────────────────────────────────────────────────────┐
  │ Spiking Conv2d  (temporal)   + PLIF                         │
  │ Spiking Conv2d  (spatial/DW) + PLIF                         │
  │ Spiking AvgPool + flatten                                    │
  │ → 3× spiking FC layers (PLIF)                               │
  │ → mean spike rate → logits                                   │
  └──────────────────────────────────────────────────────────────┘
  Pro : Every layer is spiking → SOP counting covers entire model.
        Energy is purely spike-driven end-to-end.
        Genuine novelty vs prior hybrid SNN-EEG papers.
  Con : Harder to train; may need more T_sim / lower LR.

Energy:
    SOPs counted via forward hooks on every linear/conv layer.
    Energy (µJ) = SOPs × 0.9 pJ  (Intel Loihi approximation).

References:
    Fang et al. (2021) "Incorporating Learnable Membrane Time Constants..."
    ICCV 2021.  (PLIF neuron)

    Kim & Panda (2021) "Visual Explanations from Spiking Neural Networks
    Using Inter-Spike Intervals." (fully-spiking conv SNN)
"""

import torch
import torch.nn as nn
import numpy as np

try:
    from spikingjelly.activation_based import neuron, layer, functional, surrogate
    SPIKINGJELLY_AVAILABLE = True
except ImportError:
    SPIKINGJELLY_AVAILABLE = False
    print("[SNN] SpikingJelly not installed. Install via: pip install spikingjelly")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _plif(tau_init=2.0):
    """Convenience factory for a PLIF node."""
    return neuron.ParametricLIFNode(
        init_tau=tau_init,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
        step_mode='s',
    )


def _register_sop_hook(module, sop_store: list):
    """
    Attach a forward hook to a Linear or Conv layer that counts
    spike-driven synaptic operations.

    sop_store is a mutable 1-element list so the outer object can read it.
    """
    def hook(mod, inp, out):
        # inp[0] is the pre-activation (spikes 0/1 for spiking layers,
        # or dense values for the CNN encoder in the hybrid model).
        # We count: number_of_spikes × fan_in
        x = inp[0].detach()
        # Treat any value > 0 as a spike (handles both binary and rate)
        n_spikes = (x > 0).float().sum().item()
        if isinstance(mod, nn.Linear):
            fan_in = mod.in_features
        else:  # Conv2d
            fan_in = mod.in_channels * mod.kernel_size[0] * mod.kernel_size[1]
        sop_store[0] += int(n_spikes) * fan_in
    module.register_forward_hook(hook)


class SpikeEncoder(nn.Module):
    """Converts the static analog EEG frame into a time-varying spike
    train. Constant current -> PLIF membrane dynamics produce different
    spikes at different t, even though the analog input never changes."""
    def __init__(self, n_channels, n_times, F0=4, mode='direct', tau_init=2.0):
        super().__init__()
        self.mode = mode
        if mode == 'direct':
            self.proj = nn.Conv2d(1, F0, (1, 1), bias=True)
            self.bn = nn.BatchNorm2d(F0)
            self.lif = _plif(tau_init)
        elif mode == 'poisson':
            pass
        else:
            raise ValueError(mode)

    def forward(self, frame):
        if self.mode == 'direct':
            cur = self.bn(self.proj(frame))
            return self.lif(cur)
        else:
            p = torch.sigmoid(frame)
            return torch.bernoulli(p)



# ---------------------------------------------------------------------------
# Non-spiking EEG encoder (used by HybridSNN only)
# ---------------------------------------------------------------------------

class _EEGEncoder(nn.Module):
    def __init__(self, n_channels, n_times, F1=8, D=2, kernel_length=64, dropout=0.25):
        super().__init__()
        F2 = F1 * D
        self.block = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F2, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
            nn.Conv2d(F2, F2, (1, 16), padding=(0, 8), groups=F2, bias=False),
            nn.Conv2d(F2, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            self.out_features = self.block(dummy).numel()

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.block(x).flatten(1)


# ---------------------------------------------------------------------------
# Variant 1: HybridSNN  (ANN encoder + spiking FC)
# ---------------------------------------------------------------------------

class HybridSNN(nn.Module):
    """
    CNN front-end encodes each EEG trial to a dense vector.
    That vector is fed (repeated T_sim times) into spiking FC layers.
    """

    def __init__(self, n_channels=22, n_times=512, n_classes=3,
                 hidden_dim=128, T_sim=8, dropout=0.5, tau_init=2.0):
        super().__init__()
        if not SPIKINGJELLY_AVAILABLE:
            raise ImportError("Install SpikingJelly: pip install spikingjelly")

        self.T_sim = T_sim
        self._sop = [0]  # mutable store for SOP hook

        self.encoder = _EEGEncoder(n_channels, n_times, dropout=dropout * 0.5)
        enc_dim = self.encoder.out_features

        self.fc1 = nn.Linear(enc_dim,    hidden_dim, bias=False)
        self.lif1 = _plif(tau_init)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.lif2 = _plif(tau_init)
        self.fc3 = nn.Linear(hidden_dim, n_classes,  bias=False)
        self.lif_out = _plif(tau_init)
        self.drop = nn.Dropout(dropout)

        # SOP hooks on spiking layers only
        for fc in [self.fc1, self.fc2, self.fc3]:
            _register_sop_hook(fc, self._sop)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def reset_sop(self):
        self._sop[0] = 0

    def get_energy_uJ(self, pJ_per_SOP=0.9):
        return self._sop[0] * pJ_per_SOP / 1e6

    def forward(self, x):
        functional.reset_net(self)
        self.reset_sop()

        feat = self.drop(self.encoder(x))  # (B, enc_dim)

        out_spikes = []
        for _ in range(self.T_sim):
            h = self.lif1(self.fc1(feat))
            h = self.lif2(self.fc2(h))
            s = self.lif_out(self.fc3(h))
            out_spikes.append(s)

        return torch.stack(out_spikes).mean(0)  # (B, n_classes)


# ---------------------------------------------------------------------------
# Variant 2: FullySNN  (all-spiking, first layer to last)
# ---------------------------------------------------------------------------

class FullySNN(nn.Module):
    """
    Fully spiking SNN: every layer (conv and FC) uses spiking neurons.
    No ANN layers after input encoding — energy is entirely spike-driven.

    Architecture:
        Input (B, C, T)
        → unsqueeze → (B, 1, C, T)                    [treat as single frame]
        → repeat T_sim times along batch
        → SpikingConv temporal  (1, F1, kernel_t)  + PLIF
        → SpikingConv spatial   (F1, F2, n_ch, 1) + PLIF
        → AvgPool (1,8) → flatten
        → SpikingFC hidden_dim  + PLIF
        → SpikingFC hidden_dim  + PLIF
        → SpikingFC n_classes   + PLIF
        → mean spike rate over T_sim → logits

    The time dimension of the EEG signal IS the simulation time axis:
        T_sim frames, each frame = one time slice of the EEG,
        so T_sim = n_times (after pooling) is the natural choice, though
        smaller T_sim values are used for speed (the signal is still
        presented fully via the temporal conv kernel).
    """

    def __init__(self, n_channels=22, n_times=512, n_classes=3,
             hidden_dim=128, T_sim=16, dropout=0.5, tau_init=2.0,
             F1=8, D=2, kernel_t=64, encoding='direct'):
        super().__init__()
        if not SPIKINGJELLY_AVAILABLE:
            raise ImportError("Install SpikingJelly: pip install spikingjelly")

        self.T_sim = T_sim
        self._sop = [0]
        F2 = F1 * D
        F0 = 4

        # NEW: spike encoder replaces "feed same analog frame T_sim times"
        self.encoder_spike = SpikeEncoder(n_channels, n_times, F0=F0,
                                        mode=encoding, tau_init=tau_init)

        # conv_temp now consumes spikes (F0 channels), not raw analog (1 channel)
        self.conv_temp = nn.Conv2d(F0, F1, (1, kernel_t),
                                padding=(0, kernel_t // 2), bias=False)
        self.bn_temp   = nn.BatchNorm2d(F1)
        self.lif_temp  = _plif(tau_init)

        self.conv_spat = nn.Conv2d(F1, F2, (n_channels, 1), groups=F1, bias=False)
        self.bn_spat   = nn.BatchNorm2d(F2)
        self.lif_spat  = _plif(tau_init)

        self.pool = nn.AvgPool2d((1, 8))

        # sizing dummy must route through F0 channels now
        with torch.no_grad():
            dummy_spk = torch.zeros(1, F0, n_channels, n_times)
            dummy = self.pool(self.bn_spat(self.conv_spat(
                            self.bn_temp(self.conv_temp(dummy_spk)))))
            flat_dim = dummy.numel()

        self.fc1     = nn.Linear(flat_dim,   hidden_dim, bias=False)
        self.lif_fc1 = _plif(tau_init)
        self.fc2     = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.lif_fc2 = _plif(tau_init)
        self.fc3     = nn.Linear(hidden_dim, n_classes,  bias=False)
        self.lif_out = _plif(tau_init)

        self.drop = nn.Dropout(dropout)

        sop_layers = [self.conv_temp, self.conv_spat, self.fc1, self.fc2, self.fc3]
        if encoding == 'direct':
            sop_layers.append(self.encoder_spike.proj)
        for m in sop_layers:
            _register_sop_hook(m, self._sop)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)

    def reset_sop(self):
        self._sop[0] = 0

    def get_energy_uJ(self, pJ_per_SOP=0.9):
        return self._sop[0] * pJ_per_SOP / 1e6

    def forward(self, x):
        functional.reset_net(self)   # resets encoder_spike's PLIF too
        self.reset_sop()

        frame = x.unsqueeze(1)  # (B, 1, C, T) -- analog, constant across t BY DESIGN

        out_spikes = []
        for _ in range(self.T_sim):
            spk_in = self.encoder_spike(frame)                       # varies per t
            h = self.lif_temp(self.bn_temp(self.conv_temp(spk_in)))
            h = self.lif_spat(self.bn_spat(self.conv_spat(h)))
            h = self.pool(h).flatten(1)
            h = self.drop(h)
            h = self.lif_fc1(self.fc1(h))
            h = self.lif_fc2(self.fc2(h))
            s = self.lif_out(self.fc3(h))
            out_spikes.append(s)

        return torch.stack(out_spikes).mean(0)


# ---------------------------------------------------------------------------
# Fallback (no SpikingJelly)
# ---------------------------------------------------------------------------

class _FallbackSNN(nn.Module):
    """Rate-sigmoid approximation. NOT a true SNN. Debug/CI only."""
    def __init__(self, n_channels=22, n_times=512, n_classes=3,
                 hidden_dim=128, T_sim=8, dropout=0.5, **kw):
        super().__init__()
        self._sop = [0]
        self.enc = _EEGEncoder(n_channels, n_times, dropout=dropout * 0.5)
        d = self.enc.out_features
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d, hidden_dim), nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid(),
            nn.Linear(hidden_dim, n_classes),
        )
    def forward(self, x):
        return self.net(self.enc(x))
    def get_energy_uJ(self, **kw):
        return 0.0
    def reset_sop(self):
        self._sop[0] = 0


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

# Keep old name working so Experiment.py import doesn't break
EEGSNN = HybridSNN

def make_snn(n_channels=22, n_times=512, n_classes=3,
             hidden_dim=128, T_sim=8, dropout=0.5,
             fully_spiking=False, tau_init=2.0):
    """
    Build an SNN model.

    Args:
        fully_spiking : bool
            False (default) → HybridSNN  (ANN encoder + spiking FC)
            True            → FullySNN   (all-spiking, every layer)

    Returns:
        model : nn.Module with .get_energy_uJ() and .reset_sop() methods.
    """
    if not SPIKINGJELLY_AVAILABLE:
        print("[SNN] Using fallback (non-spiking). Install SpikingJelly for true SNN.")
        return _FallbackSNN(n_channels=n_channels, n_times=n_times,
                            n_classes=n_classes, hidden_dim=hidden_dim,
                            T_sim=T_sim, dropout=dropout)

    cls = FullySNN if fully_spiking else HybridSNN
    label = "FullySNN" if fully_spiking else "HybridSNN"
    print(f"[SNN] Building {label}  (T_sim={T_sim}, hidden={hidden_dim})")

    return cls(n_channels=n_channels, n_times=n_times, n_classes=n_classes,
               hidden_dim=hidden_dim, T_sim=T_sim, dropout=dropout,
               tau_init=tau_init)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    x = torch.randn(4, 22, 512)

    for mode in [False, True]:
        label = "FullySNN" if mode else "HybridSNN"
        m = make_snn(22, 512, 3, fully_spiking=mode)
        out = m(x)
        params = sum(p.numel() for p in m.parameters())
        print(f"{label}: out={out.shape}  params={params:,}  energy={m.get_energy_uJ():.4f} µJ")




















# """
# Spiking Neural Network (SNN) for EEG Motor Imagery Classification.

# Framework : SpikingJelly (https://github.com/fangwei123456/spikingjelly)
# Neuron    : Parametric Leaky Integrate-and-Fire (PLIF) — learnable tau per neuron
# Training  : Spike-Timing Backpropagation (STBP) with surrogate gradient
#             (ATan surrogate from SpikingJelly)

# Architecture: EEG-SNN
#     1. EEG front-end  (temporal + spatial conv, same as EEGNet Block 1)
#     2. Three spiking FC layers with PLIF neurons
#     3. Voting readout: mean spike rate over T_sim timesteps → logits

# Input: (batch, n_channels=22, n_times=512)  – static frame input, repeated T_sim times
#        Alternatively, treat the time axis directly as the spike train axis.

# Energy:
#     SOP counting hook injected at forward pass to estimate synaptic operations.
#     Energy (µJ) = SOPs × 0.9 pJ (Loihi approximation).

# Reference:
#     Fang et al. (2021) "Incorporating Learnable Membrane Time Constants to
#     Enhance Learning of Spiking Neural Networks." ICCV 2021.
# """

# import torch
# import torch.nn as nn
# import numpy as np

# try:
#     from spikingjelly.activation_based import neuron, layer, functional, surrogate
#     SPIKINGJELLY_AVAILABLE = True
# except ImportError:
#     SPIKINGJELLY_AVAILABLE = False
#     print("[SNN] SpikingJelly not installed. Install via: pip install spikingjelly")


# # ---------------------------------------------------------------------------
# # EEG front-end (non-spiking CNN encoder shared with SNN)
# # ---------------------------------------------------------------------------

# class EEGEncoder(nn.Module):
#     """
#     Depthwise temporal + spatial CNN front-end that maps raw EEG to a
#     feature vector fed into the spiking layers.

#     Output shape: (B, out_features)
#     """

#     def __init__(self, n_channels=22, n_times=512, F1=8, D=2, kernel_length=64, dropout=0.25):
#         super().__init__()
#         F2 = F1 * D
#         self.block = nn.Sequential(
#             nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
#             nn.BatchNorm2d(F1),
#             nn.Conv2d(F1, F2, (n_channels, 1), groups=F1, bias=False),
#             nn.BatchNorm2d(F2),
#             nn.ELU(),
#             nn.AvgPool2d((1, 8)),
#             nn.Dropout(dropout),
#             nn.Conv2d(F2, F2, (1, 16), padding=(0, 8), groups=F2, bias=False),
#             nn.Conv2d(F2, F2, (1, 1), bias=False),
#             nn.BatchNorm2d(F2),
#             nn.ELU(),
#             nn.AvgPool2d((1, 4)),
#             nn.Dropout(dropout),
#         )
#         # Compute flat size
#         with torch.no_grad():
#             dummy = torch.zeros(1, 1, n_channels, n_times)
#             out = self.block(dummy)
#             self.out_features = out.numel()

#     def forward(self, x):
#         x = x.unsqueeze(1)   # (B, 1, C, T)
#         x = self.block(x)
#         return x.flatten(1)  # (B, out_features)


# # ---------------------------------------------------------------------------
# # SNN model
# # ---------------------------------------------------------------------------

# class EEGSNN(nn.Module):
#     """
#     SNN for 3-class EEG motor imagery.

#     The static EEG trial is encoded by a CNN front-end to a feature vector,
#     which is then repeated T_sim times and fed to spiking FC layers.
#     Final classification = mean spike rate of output layer.

#     Args:
#         n_channels  : EEG channels (22)
#         n_times     : time samples (512)
#         n_classes   : 3
#         hidden_dim  : hidden layer size for spiking layers
#         T_sim       : number of simulation timesteps (higher = better but slower)
#         dropout     : dropout before spiking layers
#         tau_init    : initial membrane time constant for PLIF
#     """

#     def __init__(
#         self,
#         n_channels=22,
#         n_times=512,
#         n_classes=3,
#         hidden_dim=128,
#         T_sim=8,
#         dropout=0.5,
#         tau_init=2.0,
#     ):
#         super().__init__()

#         if not SPIKINGJELLY_AVAILABLE:
#             raise ImportError("SpikingJelly is required. pip install spikingjelly")

#         self.T_sim = T_sim
#         self.n_classes = n_classes

#         # CNN encoder (non-spiking)
#         self.encoder = EEGEncoder(
#             n_channels=n_channels,
#             n_times=n_times,
#             F1=8, D=2,
#             kernel_length=64,
#             dropout=dropout * 0.5,
#         )
#         enc_dim = self.encoder.out_features

#         # Spiking FC layers
#         # Use step_mode='s' (single step) → we loop manually for T_sim steps
#         self.fc1 = nn.Linear(enc_dim, hidden_dim, bias=False)
#         self.lif1 = neuron.ParametricLIFNode(
#             init_tau=tau_init,
#             surrogate_function=surrogate.ATan(),
#             detach_reset=True,
#             step_mode='s',
#         )

#         self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
#         self.lif2 = neuron.ParametricLIFNode(
#             init_tau=tau_init,
#             surrogate_function=surrogate.ATan(),
#             detach_reset=True,
#             step_mode='s',
#         )

#         self.fc3 = nn.Linear(hidden_dim, n_classes, bias=False)
#         self.lif_out = neuron.ParametricLIFNode(
#             init_tau=tau_init,
#             surrogate_function=surrogate.ATan(),
#             detach_reset=True,
#             step_mode='s',
#         )

#         self.drop = nn.Dropout(dropout)

#         # SOP counter (set by forward hook)
#         self._sop_count = 0
#         self._register_sop_hooks()

#         self._init_weights()

#     # ------------------------------------------------------------------
#     # SOP (synaptic operations) counting hooks
#     # ------------------------------------------------------------------

#     def _register_sop_hooks(self):
#         """Register forward hooks to count spike-driven ops."""
#         def make_hook(layer_name):
#             def hook(module, inp, out):
#                 # out is the spike tensor (0/1); inp[0] is incoming activation
#                 spikes = out.detach()
#                 # SOPs = spikes × fan-in (= input dimension)
#                 fan_in = inp[0].shape[-1]
#                 self._sop_count += int(spikes.sum().item()) * fan_in
#             return hook

#         self.fc1.register_forward_hook(make_hook("fc1"))
#         self.fc2.register_forward_hook(make_hook("fc2"))
#         self.fc3.register_forward_hook(make_hook("fc3"))

#     def reset_sop_count(self):
#         self._sop_count = 0

#     # ------------------------------------------------------------------
#     # Weight init
#     # ------------------------------------------------------------------

#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)

#     # ------------------------------------------------------------------
#     # Forward
#     # ------------------------------------------------------------------

#     def forward(self, x):
#         """
#         Args:
#             x : (B, C, T) — raw EEG trial

#         Returns:
#             logits : (B, n_classes) — mean spike rate over T_sim steps
#         """
#         # Reset membrane potentials
#         functional.reset_net(self)
#         self.reset_sop_count()

#         # Encode EEG to static feature vector
#         feat = self.encoder(x)          # (B, enc_dim)
#         feat = self.drop(feat)

#         # Simulate over T_sim timesteps
#         out_spikes = []
#         for _ in range(self.T_sim):
#             h = self.fc1(feat)
#             h = self.lif1(h)
#             h = self.fc2(h)
#             h = self.lif2(h)
#             h = self.fc3(h)
#             s = self.lif_out(h)          # (B, n_classes) — 0/1 spikes
#             out_spikes.append(s)

#         # Mean firing rate → logits
#         logits = torch.stack(out_spikes, dim=0).mean(dim=0)   # (B, n_classes)
#         return logits

#     def get_energy_uJ(self, pJ_per_SOP=0.9):
#         """Estimate inference energy from accumulated SOP count."""
#         return self._sop_count * pJ_per_SOP / 1e6


# # ---------------------------------------------------------------------------
# # Fallback: pure-PyTorch approximation if SpikingJelly not available
# # ---------------------------------------------------------------------------

# class EEGSNNFallback(nn.Module):
#     """
#     Simplified SNN approximation using rate-coded sigmoid + threshold.
#     Use only if SpikingJelly is unavailable.
#     NOT a true SNN — for structural testing only.
#     """

#     def __init__(self, n_channels=22, n_times=512, n_classes=3, hidden_dim=128, T_sim=8, dropout=0.5, **kwargs):
#         super().__init__()
#         self.T_sim = T_sim
#         self.encoder = EEGEncoder(n_channels=n_channels, n_times=n_times, F1=8, D=2, dropout=dropout * 0.5)
#         enc_dim = self.encoder.out_features
#         self.net = nn.Sequential(
#             nn.Dropout(dropout),
#             nn.Linear(enc_dim, hidden_dim), nn.Sigmoid(),
#             nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid(),
#             nn.Linear(hidden_dim, n_classes),
#         )
#         self._sop_count = 0

#     def forward(self, x):
#         feat = self.encoder(x)
#         return self.net(feat)

#     def get_energy_uJ(self, pJ_per_SOP=0.9):
#         return 0.0  # Not countable in fallback


# def make_snn(n_channels=22, n_times=512, n_classes=3, hidden_dim=128, T_sim=8, dropout=0.5):
#     """Factory: returns EEGSNN if SpikingJelly available, else fallback."""
#     if SPIKINGJELLY_AVAILABLE:
#         return EEGSNN(n_channels=n_channels, n_times=n_times, n_classes=n_classes,
#                       hidden_dim=hidden_dim, T_sim=T_sim, dropout=dropout)
#     else:
#         print("[SNN] Using fallback (non-spiking) model. Install SpikingJelly for true SNN.")
#         return EEGSNNFallback(n_channels=n_channels, n_times=n_times, n_classes=n_classes,
#                                hidden_dim=hidden_dim, T_sim=T_sim, dropout=dropout)


# # ---------------------------------------------------------------------------
# # Quick sanity check
# # ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     model = make_snn(n_channels=22, n_times=512, n_classes=3)
#     x = torch.randn(4, 22, 512)
#     out = model(x)
#     print(f"SNN output: {out.shape}")
#     params = sum(p.numel() for p in model.parameters())
#     print(f"Params: {params:,}")
#     if SPIKINGJELLY_AVAILABLE:
#         print(f"SOPs per batch: {model._sop_count:,}")
#         print(f"Energy/batch (µJ): {model.get_energy_uJ():.4f}")







