"""
Data loading and preprocessing for BNCI2014_001 (BCI Competition IV Dataset 2a).

Classes kept (3-class subset):
    0 = Left Hand
    1 = Right Hand
    2 = Feet

Tongue class (3) is dropped.

CV STRATEGIES (three options, addressing data leakage):
  1. within_subject_rskf  — Repeated Stratified K-Fold within each subject in
                            isolation, then averaged. No subject-level leakage.
                            Multiple seeds → variance estimate of fold randomness.
  2. loso                 — Leave-One-Subject-Out: train on 8, test on 9th.
                            Strictest generalisation test; measures how well
                            the model transfers to unseen subjects.
  3. get_cv_splits        — LEGACY pooled split (kept for quick debugging only).
                            WARNING: causes subject-level data leakage.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
import torch
from torch.utils.data import TensorDataset, DataLoader


# ---------------------------------------------------------------------------
# MOABB loading — returns per-subject dict to support proper CV
# ---------------------------------------------------------------------------

def load_bnci2014_001(subjects=None, tmin=0.0, tmax=4.0, resample_freq=128):
    """
    Load BNCI2014_001 via MOABB, returning data per subject.

    Returns:
        subject_data : dict  { subject_id: {"X": ndarray (N,C,T), "y": ndarray (N,)} }
        meta         : dict  { sfreq, n_channels, n_times, subjects }
    """
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import MotorImagery

    dataset = BNCI2014_001()
    paradigm = MotorImagery(
        events=["left_hand", "right_hand", "feet"],
        n_classes=3,
        fmin=4.0,
        fmax=40.0,
        tmin=tmin,
        tmax=tmax,
        resample=resample_freq,
    )

    if subjects is None:
        subjects = dataset.subject_list

    label_map = {"left_hand": 0, "right_hand": 1, "feet": 2}
    subject_data = {}

    for subj in subjects:
        X, labels, _ = paradigm.get_data(dataset=dataset, subjects=[subj])
        y = np.array([label_map[l] for l in labels])
        subject_data[subj] = {"X": X, "y": y}

    # Infer shape from first subject
    sample_X = next(iter(subject_data.values()))["X"]
    meta = {
        "subjects": subjects,
        "sfreq": resample_freq,
        "n_channels": sample_X.shape[1],
        "n_times": sample_X.shape[2],
    }

    return subject_data, meta


def flatten_subject_data(subject_data):
    """
    Concatenate all subjects into a single (X, y) pair.
    Only use for legacy/quick-debug mode — causes subject leakage in CV.
    """
    Xs = [d["X"] for d in subject_data.values()]
    ys = [d["y"] for d in subject_data.values()]
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)


# ---------------------------------------------------------------------------
# Preprocessing (applied per-split to avoid leakage)
# ---------------------------------------------------------------------------

def preprocess(X: np.ndarray, y: np.ndarray):
    """
    Per-trial channel-wise z-score. Fit on the input set only.
    Call separately on train and val/test to avoid leakage.
    """
    n_epochs, n_ch, n_t = X.shape
    X_flat = X.reshape(n_epochs, -1)
    scaler = StandardScaler()
    X_flat = scaler.fit_transform(X_flat)
    X_norm = X_flat.reshape(n_epochs, n_ch, n_t).astype(np.float32)
    return X_norm, y.astype(np.int64)


def preprocess_split(X_train, y_train, X_val, y_val):
    """
    Fit scaler on train, apply to both train and val.
    Use this instead of bare preprocess() to guarantee no val leakage.
    """
    n_tr = X_train.shape[0]
    n_ch, n_t = X_train.shape[1], X_train.shape[2]

    scaler = StandardScaler()
    X_tr_flat = X_train.reshape(n_tr, -1)
    X_tr_flat = scaler.fit_transform(X_tr_flat)
    X_train_out = X_tr_flat.reshape(n_tr, n_ch, n_t).astype(np.float32)

    n_val = X_val.shape[0]
    X_val_flat = X_val.reshape(n_val, -1)
    X_val_flat = scaler.transform(X_val_flat)
    X_val_out = X_val_flat.reshape(n_val, n_ch, n_t).astype(np.float32)

    return X_train_out, y_train.astype(np.int64), X_val_out, y_val.astype(np.int64)


# ---------------------------------------------------------------------------
# CV Strategy 1: Within-Subject Repeated Stratified K-Fold
# ---------------------------------------------------------------------------

def within_subject_rskf(subject_data, n_splits=5, n_repeats=5, seeds=None):
    """
    Repeated Stratified K-Fold run independently for each subject.

    Why:
      - No subject-level data leakage: each subject's data is split in isolation.
      - Multiple repeats (different random fold assignments) → variance estimate
        of how sensitive results are to the arbitrary fold assignment.

    Args:
        subject_data : dict from load_bnci2014_001()
        n_splits     : K in K-fold (default 5)
        n_repeats    : number of times to repeat with different seeds (default 5)
        seeds        : list of ints for each repeat; None → [0,1,2,...,n_repeats-1]

    Yields (as iterator) or Returns (as list):
        {
          "subject": int,
          "repeat":  int,
          "fold":    int,
          "train_idx": np.ndarray,
          "val_idx":   np.ndarray,
          "X_train": ndarray, "y_train": ndarray,
          "X_val":   ndarray, "y_val":   ndarray,
        }

    Usage:
        for split in within_subject_rskf(subject_data):
            train_loader, val_loader = make_loaders(
                split["X_train"], split["y_train"],
                split["X_val"],   split["y_val"],
            )
    """
    if seeds is None:
        seeds = list(range(n_repeats))

    splits = []
    for subj_id, data in subject_data.items():
        X, y = data["X"], data["y"]
        for rep_idx, seed in enumerate(seeds):
            rskf = RepeatedStratifiedKFold(
                n_splits=n_splits, n_repeats=1, random_state=seed
            )
            for fold_idx, (tr_idx, val_idx) in enumerate(rskf.split(X, y)):
                X_tr, y_tr, X_val, y_val = preprocess_split(
                    X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]
                )
                splits.append({
                    "subject": subj_id,
                    "repeat":  rep_idx,
                    "fold":    fold_idx,
                    "train_idx": tr_idx,
                    "val_idx":   val_idx,
                    "X_train": X_tr, "y_train": y_tr,
                    "X_val":   X_val, "y_val":   y_val,
                })
    return splits


# ---------------------------------------------------------------------------
# CV Strategy 2: Leave-One-Subject-Out (LOSO)
# ---------------------------------------------------------------------------

def loso_splits(subject_data):
    """
    Leave-One-Subject-Out cross-validation.

    Each iteration: train on all subjects except one, test on the held-out subject.
    This is the gold-standard for subject-independent BCI generalisation.

    Args:
        subject_data : dict from load_bnci2014_001()

    Returns:
        list of dicts:
        {
          "test_subject": int,
          "X_train": ndarray, "y_train": ndarray,
          "X_test":  ndarray, "y_test":  ndarray,
        }

    Usage:
        for split in loso_splits(subject_data):
            print(f"Test subject: {split['test_subject']}")
            train_loader, val_loader = make_loaders(
                split["X_train"], split["y_train"],
                split["X_test"],  split["y_test"],
            )
    """
    subject_ids = list(subject_data.keys())
    splits = []

    for test_subj in subject_ids:
        train_subjs = [s for s in subject_ids if s != test_subj]

        X_train = np.concatenate([subject_data[s]["X"] for s in train_subjs], axis=0)
        y_train = np.concatenate([subject_data[s]["y"] for s in train_subjs], axis=0)
        X_test  = subject_data[test_subj]["X"]
        y_test  = subject_data[test_subj]["y"]

        X_train, y_train, X_test, y_test = preprocess_split(X_train, y_train, X_test, y_test)

        splits.append({
            "test_subject": test_subj,
            "X_train": X_train, "y_train": y_train,
            "X_test":  X_test,  "y_test":  y_test,
        })

    return splits


# ---------------------------------------------------------------------------
# LEGACY: pooled CV (data leakage — debug only)
# ---------------------------------------------------------------------------

def get_cv_splits(X, y, n_splits=5, seed=42):
    """
    LEGACY: Pooled StratifiedKFold across all subjects.
    WARNING: subject-level data leakage. Use within_subject_rskf or loso_splits.
    """
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = []
    for train_idx, val_idx in skf.split(X, y):
        splits.append((X[train_idx], y[train_idx], X[val_idx], y[val_idx]))
    return splits


# ---------------------------------------------------------------------------
# PyTorch DataLoaders
# ---------------------------------------------------------------------------

def make_loaders(X_train, y_train, X_val, y_val, batch_size=64):
    """Convert numpy arrays into PyTorch DataLoaders."""
    def to_loader(X, y, shuffle):
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.long)
        ds = TensorDataset(Xt, yt)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return to_loader(X_train, y_train, True), to_loader(X_val, y_val, False)