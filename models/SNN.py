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

  FullySNN (fully_spiking=True)
  ┌──────────────────────────────────────────────────────────────┐
  │ Spike encoder (learnable proj -> PLIF, real time chunks)    │
  │ Spiking Conv2d  (temporal)   + PLIF                         │
  │ Spiking Conv2d  (spatial/DW) + PLIF                         │
  │ Spiking AvgPool + flatten + BatchNorm                       │
  │ → 3× spiking FC layers (PLIF, BatchNorm between)            │
  │ → mean spike rate → logits                                   │
  └──────────────────────────────────────────────────────────────┘
  Pro : Every layer is spiking → SOP counting covers entire model.
        Energy is purely spike-driven end-to-end.
  Con : Harder to train; needs careful threshold/BN tuning.

Energy:
    SOPs counted via forward hooks on every linear/conv layer.
    Each input spike contributes (fan_out) synaptic operations --
    i.e. one MAC per output neuron/channel it connects to, NOT
    fan_in (that would describe how many inputs ONE output neuron
    looks at, which is the wrong direction for spike-driven SOP
    counting).
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

def _plif(tau_init=2.0, v_threshold=1.0):
    """Convenience factory for a PLIF node."""
    return neuron.ParametricLIFNode(
        init_tau=tau_init,
        v_threshold=v_threshold,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
        step_mode='s',
    )


def _register_sop_hook(module, sop_store: list):
    """
    Attach a forward hook to a Linear or Conv layer that counts
    spike-driven synaptic operations (SOPs).

    Each incoming spike triggers one MAC per output neuron/channel
    it is connected to -- i.e. SOPs scale with FAN-OUT, not fan-in.
    Using fan-in would describe how many inputs a single output
    neuron reads, which is the wrong quantity for counting how many
    times a spike actually gets multiplied-and-accumulated downstream.

    For Conv2d, true SOP counting would also need to account for how
    many spatial output positions a single input pixel contributes to
    (depends on kernel size/stride/padding). Using out_channels alone
    is a standard simplifying approximation used in most SNN energy
    estimation papers (Loihi-style estimates) -- it ignores spatial
    fan-out overlap but is far more correct than using in_channels.

    sop_store is a mutable 1-element list so the outer object can read it.
    """
    def hook(mod, inp, out):
        x = inp[0].detach()
        n_spikes = (x > 0).float().sum().item()
        if isinstance(mod, nn.Linear):
            fan_out = mod.out_features
        else:  # Conv2d
            fan_out = mod.out_channels
        sop_store[0] += int(n_spikes) * fan_out
    module.register_forward_hook(hook)


class SpikeEncoder(nn.Module):
    """Converts an analog EEG frame/chunk into a time-varying spike
    train. In 'direct' mode, a learnable analog->current projection
    feeds a PLIF neuron, whose membrane dynamics produce different
    spikes at different t even from a constant-current input. In
    'poisson' mode, spikes are drawn stochastically each call with
    probability proportional to sigmoid(amplitude)."""
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

    Each of the T_sim simulation steps processes a REAL, sequential
    chunk of the EEG trial (not a repeated static frame), so the SNN's
    membrane dynamics integrate genuine temporal structure from the
    signal -- this matters a lot for motor-imagery EEG, where ERD/ERS
    power changes evolve over the trial's time course.
    """

    def __init__(self, n_channels=22, n_times=512, n_classes=3,
                 hidden_dim=128, T_sim=16, dropout=0.5, tau_init=2.0,
                 F1=8, D=2, kernel_t=8, encoding='direct'):
        super().__init__()
        if not SPIKINGJELLY_AVAILABLE:
            raise ImportError("Install SpikingJelly: pip install spikingjelly")

        self.T_sim = T_sim
        self.usable_times = (n_times // T_sim) * T_sim  # trim to nearest multiple
        self.chunk_size = self.usable_times // T_sim
        if self.usable_times != n_times:
            print(f"[FullySNN] n_times={n_times} not divisible by T_sim={T_sim}; "
                  f"trimming to {self.usable_times} samples "
                  f"({n_times - self.usable_times} dropped).")

        self._sop = [0]
        F2 = F1 * D
        F0 = 4

        # spike encoder operates on a single chunk_size-wide slice,
        # called once per timestep with a DIFFERENT real input slice
        self.encoder_spike = SpikeEncoder(n_channels, self.chunk_size, F0=F0,
                                           mode=encoding, tau_init=tau_init)

        self.conv_temp = nn.Conv2d(F0, F1, (1, kernel_t),
                                    padding=(0, kernel_t // 2), bias=False)
        self.bn_temp   = nn.BatchNorm2d(F1)
        self.lif_temp  = _plif(tau_init)

        self.conv_spat = nn.Conv2d(F1, F2, (n_channels, 1), groups=F1, bias=False)
        self.bn_spat   = nn.BatchNorm2d(F2)
        self.lif_spat  = _plif(tau_init)

        self.pool = nn.AvgPool2d((1, 2))  # smaller pool since chunk_size is already small

        # sizing dummy uses chunk_size, not full n_times
        with torch.no_grad():
            dummy_spk = torch.zeros(1, F0, n_channels, self.chunk_size)
            dummy = self.pool(self.bn_spat(self.conv_spat(
                              self.bn_temp(self.conv_temp(dummy_spk)))))
            flat_dim = dummy.numel()

        self.bn_flat = nn.BatchNorm1d(flat_dim)

        self.fc1     = nn.Linear(flat_dim,   hidden_dim, bias=True)
        self.lif_fc1 = _plif(tau_init, v_threshold=0.1)
        self.bn_fc1  = nn.BatchNorm1d(hidden_dim)

        self.fc2     = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.lif_fc2 = _plif(tau_init, v_threshold=0.1)
        self.bn_fc2  = nn.BatchNorm1d(hidden_dim)

        self.fc3     = nn.Linear(hidden_dim, n_classes,  bias=True)
        self.lif_out = _plif(tau_init, v_threshold=0.1)

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
        """
        x : (B, C, T) raw EEG trial.

        Splits T into T_sim REAL sequential chunks and feeds them one
        at a time, so membrane potentials integrate genuine temporal
        evolution of the signal across the trial.
        """
        functional.reset_net(self)
        self.reset_sop()

        x = x[:, :, :self.usable_times]  # trim trailing samples so it divides evenly

        out_spikes = []
        for t in range(self.T_sim):
            chunk = x[:, :, t * self.chunk_size:(t + 1) * self.chunk_size]
            frame = chunk.unsqueeze(1)  # (B, 1, C, chunk_size)

            spk_in = self.encoder_spike(frame)
            h = self.lif_temp(self.bn_temp(self.conv_temp(spk_in)))
            h = self.lif_spat(self.bn_spat(self.conv_spat(h)))
            h = self.pool(h).flatten(1)
            h = self.bn_flat(h)
            h = self.drop(h)
            h = self.lif_fc1(self.fc1(h))
            h = self.bn_fc1(h)
            h = self.lif_fc2(self.fc2(h))
            h = self.bn_fc2(h)
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

EEGSNN = HybridSNN  # keep old name working so Experiment.py import doesn't break

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