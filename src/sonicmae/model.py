from __future__ import annotations
import torch
from torch import nn


class SonicMAE(nn.Module):
    """Compact spectrogram MAE with a pooled embedding head."""
    def __init__(self, n_mels: int = 64, width: int = 128, embedding_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(1, width, 5, stride=2, padding=2), nn.GELU(), nn.Conv2d(width, width, 3, stride=2, padding=1), nn.GELU())
        self.decoder = nn.Sequential(nn.ConvTranspose2d(width, width, 4, stride=2, padding=1), nn.GELU(), nn.ConvTranspose2d(width, 1, 4, stride=2, padding=1))
        self.embed = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(width, embedding_dim))

    def forward(self, spectrogram: torch.Tensor, mask_ratio: float = 0.35):
        mask = (torch.rand_like(spectrogram) < mask_ratio).float()
        z = self.encoder(spectrogram * (1 - mask))
        return self.decoder(z), self.embed(z), mask
