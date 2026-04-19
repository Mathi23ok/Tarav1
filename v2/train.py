import argparse
import os
import random
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import (
    FeatureDataset,
    create_balanced_sampler,
    load_npz_data,
    normalize_features,
)
from model import FusionNet
from utils import compute_metrics, save_checkpoint, set_device, ensure_dir


def set_seed(seed: int) -> None:
    """Seed random number generators for deterministic behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_batch(batch: list) -> Dict[str, torch.Tensor]:
    """Collate a list of samples into batched tensors."""
    features = torch.stack([item["features"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    scores = torch.stack([item["score"] for item in batch])
    return {"features": features, "label": labels, "score": scores}


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion_cls: nn.Module,
    criterion_score: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch and return average loss."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        features = batch["features"].to(device)
        labels = batch["label"].to(device)
        scores = batch["score"].to(device)
        optimizer.zero_grad()
        logits, score_preds = model(features)
        cls_loss = criterion_cls(logits, labels)
        score_loss = criterion_score(score_preds, scores)
        loss = cls_loss + 0.3 * score_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * features.size(0)
    return total_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion_cls: nn.Module,
    criterion_score: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate model on validation data and return loss plus metrics."""
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_scores = []
    all_labels = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            labels = batch["label"].to(device)
            scores = batch["score"].to(device)
            logits, score_preds = model(features)
            cls_loss = criterion_cls(logits, labels)
            score_loss = criterion_score(score_preds, scores)
            total_loss += (cls_loss + 0.3 * score_loss).item() * features.size(0)
            all_logits.append(logits)
            all_scores.append(score_preds)
            all_labels.append(labels)
            all_targets.append(scores)
    logits = torch.cat(all_logits)
    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)
    targets = torch.cat(all_targets)
    metrics = compute_metrics(logits, scores, labels, targets)
    return total_loss / len(loader.dataset), metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FusionNet for image forensics.")
    parser.add_argument("--train-path", type=str, required=True, help="Path to training NPZ feature file.")
    parser.add_argument("--val-path", type=str, required=True, help="Path to validation NPZ feature file.")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Directory to save model checkpoints.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden layer dimension.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--overfit-small", action="store_true", help="Train on small subset for overfitting check.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = set_device()
    ensure_dir(args.output_dir)

    train_features, train_labels, train_scores = load_npz_data(args.train_path)
    val_features, val_labels, val_scores = load_npz_data(args.val_path)

    if args.overfit_small:
        # Use only 100 samples for overfitting check
        indices = list(range(min(100, len(train_features))))
        train_features = train_features[indices]
        train_labels = train_labels[indices]
        train_scores = train_scores[indices]
        val_features = train_features
        val_labels = train_labels
        val_scores = train_scores

    train_features, mean, std = normalize_features(train_features)
    val_features, _, _ = normalize_features(val_features, mean, std)

    train_dataset = FeatureDataset(train_features, train_labels, train_scores)
    val_dataset = FeatureDataset(val_features, val_labels, val_scores)

    sampler = create_balanced_sampler(train_dataset.labels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate_batch,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    input_dim = train_features.shape[1]
    model = FusionNet(input_dim=input_dim, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_score = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_roc_auc = 0.0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    patience = 3
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion_cls, criterion_score, optimizer, device)
        val_loss, metrics = evaluate(model, val_loader, criterion_cls, criterion_score, device)
        print(
            f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} accuracy={metrics['accuracy']:.4f} "
            f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
            f"f1={metrics['f1']:.4f} roc_auc={metrics['roc_auc']:.4f}"
        )
        if metrics['roc_auc'] > best_roc_auc:
            best_roc_auc = metrics['roc_auc']
            save_checkpoint(model, os.path.join(args.output_dir, "fusionnet_best.pt"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping triggered")
                break
    print(f"Training complete. Best validation ROC-AUC: {best_roc_auc:.4f}")


if __name__ == "__main__":
    main()
