"""
Shared training loop, metric computation, and energy estimation utilities.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix



# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.cpu().numpy())
    n = len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    return total_loss / n, acc, f1, cm


# ---------------------------------------------------------------------------
# Full training run
# ---------------------------------------------------------------------------

def train_model(
    model,
    train_loader,
    val_loader,
    n_epochs=150,
    lr=1e-3,
    weight_decay=1e-4,
    patience=20,
    device=None,
    verbose=True,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_val_acc = 0.0
    best_state = None
    no_improve = 0

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if verbose and epoch % 10 == 0:
            elapsed = time.time() - t0
            print(
                f"  Epoch {epoch:3d}/{n_epochs} | "
                f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} | "
                f"f1={val_f1:.3f} | t={elapsed:.1f}s"
            )

        if no_improve >= patience:
            if verbose:
                print(f"  Early stop at epoch {epoch}.")
            break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


# ---------------------------------------------------------------------------
# Energy / power estimation
# ---------------------------------------------------------------------------

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_ann_energy(model, X_sample: torch.Tensor, device, pJ_per_MAC=4.6):
    """
    Rough energy estimate for ANN via MAC counting.
    Uses 4.6 pJ/MAC (45-nm CMOS approximation, common in BCI literature).

    Args:
        X_sample : single input tensor (1, C, T)
        pJ_per_MAC : pico-joules per multiply-accumulate

    Returns:
        energy_uJ : float (micro-joules per inference)
        macs : int
    """
    from torch.utils.flop_counter import FlopCounterMode  # torch >= 2.0
    model = model.to(device).eval()
    x = X_sample.unsqueeze(0).to(device)
    with FlopCounterMode(model, display=False) as fcm:
        _ = model(x)
    flops = fcm.get_total_flops()
    macs = flops // 2  # 1 MAC ≈ 2 FLOPs
    energy_uJ = macs * pJ_per_MAC / 1e6
    return energy_uJ, macs


def estimate_snn_energy(n_synaptic_ops, pJ_per_SOP=0.9):
    """
    SNN energy estimate via synaptic operation (SOP) counting.
    Uses 0.9 pJ/SOP (Intel Loihi-inspired approximation).

    Args:
        n_synaptic_ops : total spike-driven operations over the trial
        pJ_per_SOP     : pico-joules per synaptic operation

    Returns:
        energy_uJ : float
    """
    return n_synaptic_ops * pJ_per_SOP / 1e6