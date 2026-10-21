"""Transfer Learning module adapting pre-trained ImageNet/DCASE backbone (MobileNetV2) for industrial audio anomaly detection."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class TransferAcousticNet(nn.Module):
    """MobileNetV2 adapted for single-channel log-Mel spectrogram acoustic anomaly detection."""
    def __init__(self, embedding_dim: int = 128, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        base = models.mobilenet_v2(weights=weights)

        # 1-channel log-Mel adapter conv layer
        self.input_adapter = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.GELU(),
        )

        # Pre-trained convolutional feature extractor
        self.backbone = base.features

        # Projection head to latent metric space
        in_features = base.last_channel  # 1280
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_mels, n_frames)
        rgb_like = self.input_adapter(x)
        features = self.backbone(rgb_like)
        embedding = self.head(self.pool(features))
        return F.normalize(embedding, p=2, dim=1)


def fit_transfer_reference(embeddings: np.ndarray) -> dict[str, np.ndarray | float]:
    """Fit healthy baseline reference prototypes for transfer learning representations."""
    prototype = np.mean(embeddings, axis=0)
    prototype = prototype / max(np.linalg.norm(prototype), 1e-8)
    
    # Distance of training healthy points to prototype
    dists = np.linalg.norm(embeddings - prototype, axis=1)
    scale = float(max(np.median(dists), 1e-6))
    return {
        "prototype": prototype,
        "scale": scale,
        "mean_dist": float(np.mean(dists)),
        "std_dist": float(np.std(dists) + 1e-8),
    }


def transfer_anomaly_score(embedding: np.ndarray, reference: dict[str, np.ndarray | float]) -> float:
    """Compute normalized cosine-distance anomaly score."""
    norm_emb = embedding / max(np.linalg.norm(embedding), 1e-8)
    prototype = reference["prototype"]
    dist = float(np.linalg.norm(norm_emb - prototype))
    # Normalized score
    return float((dist - reference["mean_dist"]) / reference["std_dist"])
