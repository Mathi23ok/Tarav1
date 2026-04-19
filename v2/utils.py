import os
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def set_device() -> torch.device:
    """Return the best available device for training."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def save_checkpoint(model: torch.nn.Module, path: str) -> None:
    """Save model state dict to disk."""
    torch.save(model.state_dict(), path)


def compute_metrics(
    logits: torch.Tensor,
    score_preds: torch.Tensor,
    labels: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    """Compute evaluation metrics for classification and regression."""
    probs = torch.sigmoid(logits).cpu().numpy()
    label_preds = (probs >= 0.5).astype(np.int64)
    labels_np = labels.cpu().numpy().astype(np.int64)
    targets_np = targets.cpu().numpy().astype(np.float32)
    metrics = {
        "accuracy": float(accuracy_score(labels_np, label_preds)),
        "precision": float(precision_score(labels_np, label_preds, zero_division=0)),
        "recall": float(recall_score(labels_np, label_preds, zero_division=0)),
        "f1": float(f1_score(labels_np, label_preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels_np, probs)),
        "score_mae": float(np.mean(np.abs(score_preds.cpu().numpy() - targets_np))),
    }
    return metrics
