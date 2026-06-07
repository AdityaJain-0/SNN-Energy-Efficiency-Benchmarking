"""
Experiment runner: ATCNet, EEGNet, ShallowConvNet, HybridSNN, FullySNN
on BNCI2014_001  (3-class: Left Hand / Right Hand / Feet).

CV MODES (select via --cv_mode):

  within_subject  [DEFAULT — recommended]
    Repeated Stratified K-Fold run per subject in isolation.
    - No subject-level leakage
    - n_repeats independent random fold assignments → variance estimate
    - Reports mean ± SD per subject, then grand mean across subjects

  loso
    Leave-One-Subject-Out: train on 8 subjects, test on held-out 9th.
    - Strictest generalisation; measures cross-subject transfer
    - One result row per subject (= 9 rows total)

  legacy_pooled   [NOT RECOMMENDED — debug only]
    Original pooled StratifiedKFold across all subjects.
    WARNING: subject-level data leakage inflates accuracy.

Usage examples:
    python Experiment.py                                       # within_subject, all models
    python Experiment.py --cv_mode loso                       # LOSO
    python Experiment.py --cv_mode within_subject --n_repeats 10 --n_splits 5
    python Experiment.py --subjects 1 2 3 --epochs 30 --patience 10   # quick test
    python Experiment.py --snn_mode both                      # run Hybrid + Fully spiking
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from utils.data_loader import (
    load_bnci2014_001, flatten_subject_data,
    within_subject_rskf, loso_splits, get_cv_splits, make_loaders,
)
from utils.trainer import train_model, evaluate, count_params, estimate_ann_energy
from models.ATCnet       import ATCNet
from models.EEGnet       import EEGNet
from models.ShallowConvent import ShallowConvNet
from models.SNN          import make_snn, SPIKINGJELLY_AVAILABLE

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CLASS_NAMES  = ["Left Hand", "Right Hand", "Feet"]
RESULTS_DIR  = "results"
FIGURES_DIR  = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(name, n_channels, n_times, n_classes=3):
    if name == "ATCNet":
        return ATCNet(n_channels=n_channels, n_times=n_times, n_classes=n_classes)
    elif name == "EEGNet":
        return EEGNet(n_channels=n_channels, n_times=n_times, n_classes=n_classes)
    elif name == "ShallowConvNet":
        return ShallowConvNet(n_channels=n_channels, n_times=n_times, n_classes=n_classes)
    elif name == "HybridSNN":
        return make_snn(n_channels, n_times, n_classes, T_sim=8,  fully_spiking=False)
    elif name == "FullySNN":
        return make_snn(n_channels, n_times, n_classes, T_sim=16, fully_spiking=True)
    else:
        raise ValueError(f"Unknown model: {name}")


# ---------------------------------------------------------------------------
# Energy helper
# ---------------------------------------------------------------------------

def get_energy(model, model_name, X_sample, device):
    is_snn = model_name in ("HybridSNN", "FullySNN")
    if is_snn:
        model.eval()
        with torch.no_grad():
            model(X_sample.unsqueeze(0).to(device))
        sop_energy = model.get_energy_uJ()

        # For HybridSNN, also count the ANN CNN encoder
        if model_name == "HybridSNN":
            try:
                enc_energy, _ = estimate_ann_energy(model.encoder, X_sample.unsqueeze(0), device)
                return sop_energy + enc_energy
            except Exception as e:
                print(f"  [Energy] Encoder MAC count failed: {e}")
        return sop_energy
    else:
        try:
            energy, _ = estimate_ann_energy(model, X_sample, device)
            return energy
        except Exception as e:
            print(f"  [Energy] Could not estimate for {model_name}: {e}")
            return float("nan")

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm, model_name, label, save_dir):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{model_name} — {label}")
    plt.tight_layout()
    path = os.path.join(save_dir, f"cm_{model_name}_{label}.png")
    fig.savefig(path, dpi=120); plt.close(fig)


def plot_learning_curve(history, model_name, label, save_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ep = range(1, len(history["train_loss"]) + 1)
    ax1.plot(ep, history["train_loss"], label="Train")
    ax1.plot(ep, history["val_loss"],   label="Val")
    ax1.set_title(f"{model_name} — {label} Loss"); ax1.legend()
    ax2.plot(ep, history["train_acc"],  label="Train")
    ax2.plot(ep, history["val_acc"],    label="Val")
    ax2.set_title(f"{model_name} — {label} Accuracy"); ax2.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, f"curve_{model_name}_{label}.png")
    fig.savefig(path, dpi=120); plt.close(fig)


def plot_comparison(df, save_dir):
    agg = df.groupby("model").agg(
        acc_mean=("val_acc", "mean"), acc_std=("val_acc", "std"),
        f1_mean=("val_f1", "mean"),
        energy_mean=("energy_uJ", "mean"),
    ).reset_index()

    colors = ["#2196F3", "#9C27B0", "#FF5722", "#FF9800", "#4CAF50"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.bar(agg["model"], agg["acc_mean"], yerr=agg["acc_std"],
           color=colors[:len(agg)], capsize=5, alpha=0.85)
    ax.axhline(1/3, ls="--", color="gray", label="Chance (33%)")
    ax.set_ylabel("Accuracy"); ax.set_title("Mean Accuracy ± SD")
    ax.set_ylim(0, 1); ax.legend()

    ax = axes[1]
    ax.bar(agg["model"], agg["f1_mean"], color=colors[:len(agg)], alpha=0.85)
    ax.set_ylabel("F1 (macro)"); ax.set_title("Mean F1 Score (Macro)")
    ax.set_ylim(0, 1)

    ax = axes[2]
    ax.bar(agg["model"], agg["energy_mean"], color=colors[:len(agg)], alpha=0.85)
    ax.set_ylabel("Energy (µJ / inference)"); ax.set_title("Estimated Inference Energy")
    ax.set_yscale("log")

    for ax in axes:
        ax.set_xticklabels(agg["model"], rotation=20, ha="right")

    plt.tight_layout()
    path = os.path.join(save_dir, "model_comparison.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved comparison plot → {path}")


# ---------------------------------------------------------------------------
# Single train+eval pass
# ---------------------------------------------------------------------------

def run_one_split(model_name, X_train, y_train, X_val, y_val,
                  n_channels, n_times, args, device, split_label):
    train_loader, val_loader = make_loaders(X_train, y_train, X_val, y_val,
                                            batch_size=args.batch_size)
    model = build_model(model_name, n_channels, n_times)
    n_params = count_params(model)

    t0 = time.time()
    model, history = train_model(
        model=model, train_loader=train_loader, val_loader=val_loader,
        n_epochs=args.epochs, lr=args.lr, weight_decay=1e-4,
        patience=args.patience, device=device, verbose=args.verbose,
    )
    elapsed = time.time() - t0

    criterion = torch.nn.CrossEntropyLoss()
    _, val_acc, val_f1, cm = evaluate(model, val_loader, criterion, device)

    X_sample = torch.tensor(X_val[0], dtype=torch.float32)
    energy   = get_energy(model, model_name, X_sample, device)

    print(f"    [{split_label}] acc={val_acc:.4f}  f1={val_f1:.4f}  "
          f"energy={energy:.4f} µJ  t={elapsed:.1f}s")

    plot_confusion_matrix(cm, model_name, split_label, FIGURES_DIR)
    plot_learning_curve(history, model_name, split_label, FIGURES_DIR)

    return {
        "model":        model_name,
        "split_label":  split_label,
        "val_acc":      round(val_acc,  4),
        "val_f1":       round(val_f1,   4),
        "energy_uJ":    round(energy,   6) if not np.isnan(energy) else None,
        "n_params":     n_params,
        "train_time_s": round(elapsed,  1),
    }, cm


# ---------------------------------------------------------------------------
# CV modes
# ---------------------------------------------------------------------------

def run_within_subject(model_names, subject_data, meta, args, device):
    records, all_cms = [], {m: [] for m in model_names}
    seeds = list(range(args.n_repeats))

    splits = within_subject_rskf(subject_data,
                                 n_splits=args.n_splits,
                                 n_repeats=args.n_repeats,
                                 seeds=seeds)

    total = len(model_names) * len(splits)
    done  = 0
    for model_name in model_names:
        print(f"\n{'='*60}\n  Model: {model_name}\n{'='*60}")
        for sp in splits:
            label = f"subj{sp['subject']}_rep{sp['repeat']}_fold{sp['fold']}"
            rec, cm = run_one_split(
                model_name,
                sp["X_train"], sp["y_train"],
                sp["X_val"],   sp["y_val"],
                meta["n_channels"], meta["n_times"],
                args, device, label,
            )
            rec.update({"subject": sp["subject"],
                        "repeat":  sp["repeat"],
                        "fold":    sp["fold"]})
            records.append(rec)
            all_cms[model_name].append(cm)
            done += 1
            print(f"  Progress: {done}/{total}")

        agg_cm = np.sum(all_cms[model_name], axis=0)
        plot_confusion_matrix(agg_cm, model_name, "ALL_within", FIGURES_DIR)

    return records


def run_loso(model_names, subject_data, meta, args, device):
    records, all_cms = [], {m: [] for m in model_names}
    splits = loso_splits(subject_data)

    for model_name in model_names:
        print(f"\n{'='*60}\n  Model: {model_name}\n{'='*60}")
        for sp in splits:
            label = f"loso_test{sp['test_subject']}"
            rec, cm = run_one_split(
                model_name,
                sp["X_train"], sp["y_train"],
                sp["X_test"],  sp["y_test"],
                meta["n_channels"], meta["n_times"],
                args, device, label,
            )
            rec["test_subject"] = sp["test_subject"]
            records.append(rec)
            all_cms[model_name].append(cm)

        agg_cm = np.sum(all_cms[model_name], axis=0)
        plot_confusion_matrix(agg_cm, model_name, "ALL_loso", FIGURES_DIR)

    return records


def run_legacy(model_names, subject_data, meta, args, device):
    print("\n[WARNING] legacy_pooled mode has subject-level data leakage.\n")
    X_all, y_all = flatten_subject_data(subject_data)
    splits = get_cv_splits(X_all, y_all, n_splits=args.n_splits, seed=42)
    records = []
    for model_name in model_names:
        print(f"\n{'='*60}\n  Model: {model_name}\n{'='*60}")
        for fold_idx, (X_tr, y_tr, X_val, y_val) in enumerate(splits, 1):
            label = f"fold{fold_idx}"
            rec, _ = run_one_split(
                model_name, X_tr, y_tr, X_val, y_val,
                meta["n_channels"], meta["n_times"], args, device, label,
            )
            rec["fold"] = fold_idx
            records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build model list
    base_models  = ["ATCNet", "EEGNet", "ShallowConvNet"]
    snn_variants = {
        "hybrid": ["HybridSNN"],
        "full":   ["FullySNN"],
        "both":   ["HybridSNN", "FullySNN"],
    }
    model_names = base_models + snn_variants.get(args.snn_mode, ["HybridSNN"])
    print(f"Models: {model_names}")
    print(f"CV mode: {args.cv_mode}")

    # Load data
    print(f"\nLoading BNCI2014_001 (subjects={args.subjects}) ...")
    subject_data, meta = load_bnci2014_001(
        subjects=args.subjects, tmin=0.0, tmax=4.0, resample_freq=128
    )
    print(f"  Loaded {len(subject_data)} subjects  |  "
          f"channels={meta['n_channels']}  times={meta['n_times']}")

    # Run selected CV mode
    if args.cv_mode == "within_subject":
        records = run_within_subject(model_names, subject_data, meta, args, device)
    elif args.cv_mode == "loso":
        records = run_loso(model_names, subject_data, meta, args, device)
    else:
        records = run_legacy(model_names, subject_data, meta, args, device)

    # Save results
    df = pd.DataFrame(records)
    csv_path = os.path.join(RESULTS_DIR, f"summary_{args.cv_mode}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}")

    # Print summary
    summary = df.groupby("model").agg(
        acc_mean=("val_acc", "mean"),
        acc_std=("val_acc", "std"),
        f1_mean=("val_f1", "mean"),
        energy_uJ_mean=("energy_uJ", "mean"),
        n_params=("n_params", "first"),
    ).round(4)
    print("\n" + "="*60 + "\nFINAL SUMMARY\n" + "="*60)
    print(summary.to_string())

    plot_comparison(df, FIGURES_DIR)
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MOABB 3-class EEG model comparison")

    p.add_argument("--subjects",  nargs="+", type=int, default=None,
                   help="Subject IDs 1-9 (default: all)")
    p.add_argument("--cv_mode",   default="within_subject",
                   choices=["within_subject", "loso", "legacy_pooled"],
                   help="Cross-validation strategy (default: within_subject)")
    p.add_argument("--n_splits",  type=int, default=5,
                   help="K for K-fold  (within_subject mode, default 5)")
    p.add_argument("--n_repeats", type=int, default=5,
                   help="Repeats with different seeds (within_subject, default 5)")
    p.add_argument("--snn_mode",  default="hybrid",
                   choices=["hybrid", "full", "both"],
                   help="hybrid=HybridSNN  full=FullySNN  both=run both (default: hybrid)")
    p.add_argument("--epochs",    type=int,   default=150)
    p.add_argument("--patience",  type=int,   default=20)
    p.add_argument("--batch_size",type=int,   default=64)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--verbose",   action="store_true", default=False)

    main(p.parse_args())