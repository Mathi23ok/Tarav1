import argparse
import cv2
import os
import time
import warnings
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.io import read_image
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore", category=UserWarning)

FFT_BINS = 32
FFT_SIZE = 128
NOISE_HIST_BINS = 16
NOISE_KERNEL = torch.ones(1, 1, 7, 7, dtype=torch.float32) / 49.0
PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_FFT_BIN_INDEX: Optional[np.ndarray] = None
_FFT_BIN_COUNTS: Optional[np.ndarray] = None


def _get_fft_bin_layout(size: int = FFT_SIZE, num_bins: int = FFT_BINS) -> Tuple[np.ndarray, np.ndarray]:
    """Cache radial-bin assignments for a fixed FFT resolution."""
    global _FFT_BIN_INDEX, _FFT_BIN_COUNTS

    if _FFT_BIN_INDEX is None or _FFT_BIN_COUNTS is None or _FFT_BIN_COUNTS.shape[0] != num_bins:
        y, x = np.indices((size, size), dtype=np.float32)
        center = (size - 1) / 2.0
        radius = np.sqrt((y - center) ** 2 + (x - center) ** 2)
        max_radius = radius.max()

        bin_index = np.floor(radius / (max_radius + 1e-6) * num_bins).astype(np.int64)
        bin_index = np.clip(bin_index, 0, num_bins - 1)
        bin_counts = np.bincount(bin_index.ravel(), minlength=num_bins).astype(np.float32)
        bin_counts = np.maximum(bin_counts, 1.0)

        _FFT_BIN_INDEX = bin_index
        _FFT_BIN_COUNTS = bin_counts

    return _FFT_BIN_INDEX, _FFT_BIN_COUNTS


def _resize_gray_batch(gray_images: Sequence[np.ndarray], size: int = FFT_SIZE) -> np.ndarray:
    """Resize grayscale images to a fixed square batch for CPU-side feature extraction."""
    resized = [
        cv2.resize(gray.squeeze(0), (size, size), interpolation=cv2.INTER_AREA).astype(np.float32, copy=False)
        for gray in gray_images
    ]
    return np.stack(resized, axis=0)


def get_feature_extractor() -> nn.Module:
    """Load pretrained ResNet50 without the classification head."""
    model = models.resnet50(pretrained=True)
    model.fc = nn.Identity()
    model.eval()
    return model


def preprocess_image(image: torch.Tensor) -> torch.Tensor:
    """Normalize image for pretrained model input."""
    return PREPROCESS(image.float() / 255.0)


def extract_cnn_embeddings(images: Sequence[torch.Tensor], model: nn.Module) -> np.ndarray:
    """Extract raw ResNet50 embeddings for a batch of images."""
    with torch.inference_mode():
        batch = torch.stack(list(images)).to(device, non_blocking=(device.type == "cuda"))
        embeddings = model(batch)
    return embeddings.cpu().numpy().astype(np.float32, copy=False)


def extract_fft_features(gray: torch.Tensor, num_bins: int = FFT_BINS) -> torch.Tensor:
    """Extract fast, normalized radial FFT bins from a downsampled grayscale image."""
    features, _ = extract_fft_features_batch([gray.detach().cpu().numpy()], num_bins=num_bins)
    return torch.from_numpy(features[0])


def extract_fft_features_batch(gray_images: Sequence[np.ndarray], num_bins: int = FFT_BINS) -> Tuple[np.ndarray, float]:
    """Extract FFT radial bins for a full batch of grayscale images."""
    fft_start = time.perf_counter()
    batch_gray = _resize_gray_batch(gray_images, size=FFT_SIZE)

    fft = np.fft.fft2(batch_gray, axes=(1, 2))
    mag = np.log1p(np.abs(np.fft.fftshift(fft, axes=(1, 2))))

    mean = mag.mean(axis=(1, 2), keepdims=True)
    std = mag.std(axis=(1, 2), keepdims=True) + 1e-6
    mag = (mag - mean) / std

    bin_index, bin_counts = _get_fft_bin_layout(FFT_SIZE, num_bins)
    flat_index = bin_index.ravel()
    flat_mag = mag.reshape(batch_gray.shape[0], -1).astype(np.float32, copy=False)
    features = np.zeros((batch_gray.shape[0], num_bins), dtype=np.float32)
    batch_index = np.broadcast_to(np.arange(batch_gray.shape[0])[:, None], flat_mag.shape)
    radial_index = np.broadcast_to(flat_index[None, :], flat_mag.shape)
    np.add.at(features, (batch_index, radial_index), flat_mag)
    features /= bin_counts[None, :]

    return features, time.perf_counter() - fft_start


def extract_noise_features(gray: torch.Tensor) -> torch.Tensor:
    """Extract noise statistics and histogram."""
    blurred = F.conv2d(gray.unsqueeze(0), NOISE_KERNEL, padding=3).squeeze(0)
    residual = (gray - blurred).flatten()

    mean_val = residual.mean()
    std_val = residual.std(unbiased=False)
    centered = residual - mean_val
    variance = residual.var(unbiased=False)

    if variance > 0:
        skew_val = (centered.pow(3).mean() / variance.sqrt().pow(3)).to(torch.float32)
        kurt_val = (centered.pow(4).mean() / variance.pow(2) - 3.0).to(torch.float32)
    else:
        skew_val = torch.tensor(0.0, dtype=torch.float32)
        kurt_val = torch.tensor(0.0, dtype=torch.float32)

    hist = torch.histc(residual, bins=NOISE_HIST_BINS, min=-1.0, max=1.0)
    hist = hist / residual.numel()

    stats = torch.stack([mean_val, std_val, skew_val, kurt_val]).to(torch.float32)
    return torch.cat([stats, hist.to(torch.float32)])


def extract_noise_features_batch(gray_images: Sequence[np.ndarray]) -> Tuple[np.ndarray, float]:
    """Extract noise features for a batch while preserving the existing feature definition."""
    noise_start = time.perf_counter()
    batch_gray = _resize_gray_batch(gray_images, size=FFT_SIZE)
    gray_tensor = torch.from_numpy(batch_gray).unsqueeze(1)

    blurred = F.conv2d(gray_tensor, NOISE_KERNEL, padding=3)
    residual = (gray_tensor - blurred).squeeze(1)
    residual_flat = residual.reshape(residual.size(0), -1)

    mean_val = residual_flat.mean(dim=1)
    std_val = residual_flat.std(dim=1, unbiased=False)
    centered = residual_flat - mean_val[:, None]
    variance = residual_flat.var(dim=1, unbiased=False)

    safe_std = torch.sqrt(variance + 1e-12)
    safe_var_sq = variance.pow(2) + 1e-12
    skew_val = centered.pow(3).mean(dim=1) / safe_std.pow(3)
    kurt_val = centered.pow(4).mean(dim=1) / safe_var_sq - 3.0

    valid = variance > 0
    skew_val = torch.where(valid, skew_val, torch.zeros_like(skew_val))
    kurt_val = torch.where(valid, kurt_val, torch.zeros_like(kurt_val))

    residual_np = residual_flat.numpy()
    hist_edges = np.linspace(-1.0, 1.0, NOISE_HIST_BINS + 1, dtype=np.float32)
    hist_idx = np.searchsorted(hist_edges, residual_np, side="right") - 1
    hist_idx = np.clip(hist_idx, 0, NOISE_HIST_BINS - 1)

    hist = np.zeros((residual_np.shape[0], NOISE_HIST_BINS), dtype=np.float32)
    batch_index = np.broadcast_to(np.arange(residual_np.shape[0])[:, None], hist_idx.shape)
    np.add.at(hist, (batch_index, hist_idx), 1.0)
    hist /= residual_np.shape[1]

    stats = torch.stack([mean_val, std_val, skew_val, kurt_val], dim=1).numpy().astype(np.float32, copy=False)
    noise_features = np.concatenate([stats, hist], axis=1)
    return noise_features, time.perf_counter() - noise_start


def load_image_inputs(path: Path) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray]]:
    """Load image once and prepare both CNN and CPU feature inputs."""
    try:
        image = read_image(str(path)).float()
    except Exception:
        print(f"Skipping corrupt image: {path}")
        return None, None

    if image.size(0) == 4:
        image = image[:3]
    if image.size(0) == 1:
        image = image.repeat(3, 1, 1)

    processed = preprocess_image(image)
    gray = image.mean(dim=0, keepdim=True)
    return processed, gray.numpy()


def collect_image_paths(root_dir: str, partition: str) -> List[Tuple[Path, int, float]]:
    """Collect image paths with labels and scores."""
    root_path = Path(root_dir) / partition
    paths = []
    for label_name, label in [("fake", 0), ("real", 1)]:
        class_dir = root_path / label_name
        if not class_dir.exists():
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for path in sorted(class_dir.glob(ext)):
                paths.append((path, label, float(label)))
    return paths


def save_features_to_npz(features: np.ndarray, labels: np.ndarray, scores: np.ndarray, output_path: str) -> None:
    """Save features, labels, scores to NPZ."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, X=features, y=labels, score=scores)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract features from image dataset.")
    parser.add_argument("--data-root", type=str, default="dataset", help="Root directory of dataset.")
    parser.add_argument("--output-dir", type=str, default="features", help="Output directory for NPZ files.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for CNN feature extraction.")
    args = parser.parse_args()

    model = get_feature_extractor()
    model.to(device)

    cnn_dim = 2048
    cpu_dim = FFT_BINS + 4 + NOISE_HIST_BINS
    feature_dim = cnn_dim + cpu_dim

    for partition in ["train", "test"]:
        paths = collect_image_paths(args.data_root, partition)
        feature_rows = []
        label_rows = []
        score_rows = []
        skipped_count = 0

        time_cnn = 0.0
        time_fft = 0.0
        time_noise = 0.0

        for start_idx in tqdm(
            range(0, len(paths), args.batch_size),
            total=(len(paths) + args.batch_size - 1) // args.batch_size,
            desc=f"{partition} batches",
        ):
            batch_items = paths[start_idx:start_idx + args.batch_size]
            valid_items = []
            processed_images = []
            gray_images = []

            for path, label, score in batch_items:
                processed, gray = load_image_inputs(path)
                if processed is None:
                    skipped_count += 1
                    continue

                valid_items.append((label, score))
                processed_images.append(processed)
                gray_images.append(gray)

            if not processed_images:
                continue

            cnn_start = time.perf_counter()
            cnn_embeddings = extract_cnn_embeddings(processed_images, model)
            time_cnn += time.perf_counter() - cnn_start

            fft_features, fft_elapsed = extract_fft_features_batch(gray_images)
            noise_features, noise_elapsed = extract_noise_features_batch(gray_images)
            cpu_features_batch = np.concatenate([fft_features, noise_features], axis=1)

            for (label, score), cnn_emb, cpu_features in zip(valid_items, cnn_embeddings, cpu_features_batch):
                combined_features = np.empty(feature_dim, dtype=np.float32)
                combined_features[:cnn_dim] = cnn_emb
                combined_features[cnn_dim:] = cpu_features
                feature_rows.append(combined_features)
                label_rows.append(label)
                score_rows.append(score)
            time_fft += fft_elapsed
            time_noise += noise_elapsed

        if feature_rows:
            features = np.stack(feature_rows).astype(np.float32, copy=False)
        else:
            features = np.empty((0, feature_dim), dtype=np.float32)
        labels = np.asarray(label_rows, dtype=np.int64)
        scores = np.asarray(score_rows, dtype=np.float32)

        output_path = os.path.join(args.output_dir, f"{partition}.npz")
        save_features_to_npz(features, labels, scores, output_path)
        print(f"Saved {len(features)} samples to {output_path}")
        print(f"Skipped {skipped_count} corrupt images")
        print(f"time_cnn={time_cnn:.4f}s")
        print(f"time_fft={time_fft:.4f}s")
        print(f"time_noise={time_noise:.4f}s")


if __name__ == "__main__":
    main()
