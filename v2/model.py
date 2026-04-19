import torch
import torch.nn as nn
from typing import Tuple


class FusionNet(nn.Module):
    """Fusion model with feature-specific branches and shared representation."""

    def __init__(self, input_dim: int = 2100, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        # Feature branches
        self.cnn_branch = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
        )
        self.fft_branch = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
        )
        self.noise_branch = nn.Sequential(
            nn.Linear(20, 32),
            nn.ReLU(inplace=True),
        )
        # Shared body
        self.body = nn.Sequential(
            nn.Linear(256 + 16 + 32, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, 1)
        self.score_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert features.size(1) == 2100, f"Expected input dim 2100, got {features.size(1)}"
        cnn_part = features[:, :2048]
        fft_part = features[:, 2048:2048 + 32]
        noise_part = features[:, 2048 + 32:]
        cnn_out = self.cnn_branch(cnn_part)
        fft_out = self.fft_branch(fft_part)
        noise_out = self.noise_branch(noise_part)
        fused = torch.cat([cnn_out, fft_out, noise_out], dim=1)
        hidden = self.body(fused)
        logits = self.classifier(hidden)
        score = torch.sigmoid(self.score_head(hidden))
        return logits.squeeze(-1), score.squeeze(-1)
