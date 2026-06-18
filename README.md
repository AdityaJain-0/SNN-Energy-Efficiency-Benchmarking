# EEG Motor Imagery Classification: ANN vs. SNN Comparison

This project benchmarks deep learning and spiking neural network (SNN) architectures on the **BNCI2014_001** dataset (BCI Competition IV Dataset 2a) for 3-class motor imagery classification: **Left Hand**, **Right Hand**, and **Feet**.

The central goal is to demonstrate the viability of SNNs for this task — showing that they can reach decoding accuracy competitive with established ANN baselines (ATCNet, EEGNet, ShallowConvNet) while consuming far less estimated inference energy. To that end, every model reports accuracy, F1 score, and estimated per-inference energy side by side, so accuracy/energy trade-offs are directly comparable rather than evaluated in isolation.

## Models

**ANN baselines** (for accuracy and energy comparison):

- **ATCNet** — Attention Temporal Convolutional Network
- **EEGNet** — compact convolutional network for EEG-based BCIs
- **ShallowConvNet** — shallow CNN baseline for EEG decoding

**SNN models** (the focus of this project):

- **HybridSNN** — CNN front-end + spiking fully-connected layers
- **FullySNN** — fully spiking network (every layer, including convolutions, is spike-driven), giving the cleanest end-to-end energy comparison since no layer relies on dense ANN-style computation

## Installation

```bash
pip install -r requirements.txt
```

This installs MOABB, MNE, PyTorch, scikit-learn, and SpikingJelly (required for the SNN models). No manual dataset downloads are needed — everything is fetched automatically the first time you run an experiment.

## Usage

```bash
# Quick test on 3 subjects
python Experiment.py --subjects 1 2 3

# Full run on all 9 subjects
python Experiment.py

# Run both SNN variants (HybridSNN and FullySNN) alongside the ANN models
python Experiment.py --snn_mode both
```

A full 9-subject run takes significantly longer than the 3-subject quick test, so it's worth validating your setup with `--subjects 1 2 3` first.

### SNN variants

The `--snn_mode` flag controls which spiking model(s) are included in the run:

| Flag value | Models run |
|---|---|
| `hybrid` (default) | HybridSNN |
| `full` | FullySNN |
| `both` | HybridSNN + FullySNN |

### Other useful options

| Flag | Description | Default |
|---|---|---|
| `--subjects` | Subject IDs (1–9) to include | all 9 |
| `--cv_mode` | Cross-validation strategy: `within_subject`, `loso`, or `legacy_pooled` | `within_subject` |
| `--n_splits` | K for K-fold (within_subject mode) | 5 |
| `--n_repeats` | Number of repeated fold assignments (within_subject mode) | 5 |
| `--epochs` | Max training epochs | 150 |
| `--patience` | Early-stopping patience | 20 |
| `--batch_size` | Training batch size | 64 |
| `--lr` | Learning rate | 1e-3 |
| `--verbose` | Print per-epoch training logs | off |

`within_subject` (the default) is recommended, since it avoids subject-level data leakage. `loso` (Leave-One-Subject-Out) is the strictest test of cross-subject generalization. `legacy_pooled` is kept only for quick debugging — it pools data across subjects and **will leak information across subjects**, inflating accuracy.

## Generated folders

- **`C-/`** — created automatically on first run. This caches the downloaded EEG dataset so it doesn't need to be re-fetched on subsequent runs.
- **`results/`** — created automatically after a run completes. Contains:
  - Confusion matrices per model/split
  - Training/validation loss and accuracy curves
  - A model comparison plot (accuracy, F1, and estimated energy per inference)
  - A CSV summary of all results (`summary_<cv_mode>.csv`)

## Notes

- No special manual downloads are required beyond `pip install -r requirements.txt` — SpikingJelly is included in that file and is required for the SNN models to run.
- If SpikingJelly is not installed, the SNN models fall back to a non-spiking approximation for structural testing only; install SpikingJelly for true spiking behavior and accurate energy estimates.
