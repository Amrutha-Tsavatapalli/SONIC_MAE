"""Customized SonicConformerMAE architecture combining Depthwise Convolutions, Multi-Head Self-Attention, and Masked Reconstruction."""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardModule(nn.Module):
    """Macaron-style half-step feed-forward network with SwiGLU activation."""
    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        d_ff = d_model * expansion
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_model, d_ff)
        self.w3 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation: (swish(w1(x)) * w2(x))
        residual = x
        x_norm = self.norm(x)
        h = F.silu(self.w1(x_norm)) * self.w2(x_norm)
        return residual + 0.5 * self.dropout(self.w3(h))


class DepthwiseConvModule(nn.Module):
    """Depthwise separable convolution module capturing local spectral textures."""
    def __init__(self, d_model: int, kernel_size: int = 7, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.pointwise1 = nn.Linear(d_model, 2 * d_model)
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=kernel_size // 2, groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.pointwise2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        residual = x
        x_norm = self.norm(x)
        # Pointwise GLU projection
        h = self.pointwise1(x_norm)  # (B, T, 2D)
        a, b = h.chunk(2, dim=-1)
        h = a * torch.sigmoid(b)      # (B, T, D)
        
        # 1D Depthwise Conv along sequence dimension
        h = h.transpose(1, 2)         # (B, D, T)
        h = self.depthwise(h)
        h = self.batch_norm(h)
        h = F.silu(h)
        h = h.transpose(1, 2)         # (B, T, D)
        
        out = self.dropout(self.pointwise2(h))
        return residual + out


class ConformerBlock(nn.Module):
    """Full Conformer block: FFN1 (half) -> MHSA -> Depthwise Conv -> FFN2 (half) -> LayerNorm."""
    def __init__(self, d_model: int = 64, n_heads: int = 4, conv_kernel: int = 7, dropout: float = 0.1) -> None:
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, dropout=dropout)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)
        self.conv_module = DepthwiseConvModule(d_model, kernel_size=conv_kernel, dropout=dropout)
        self.ffn2 = FeedForwardModule(d_model, dropout=dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = self.ffn1(x)
        # Multi-Head Self-Attention
        res = x
        norm_x = self.attn_norm(x)
        attn_out, _ = self.self_attn(norm_x, norm_x, norm_x)
        x = res + attn_out
        # Depthwise Conv
        x = self.conv_module(x)
        # Second FFN
        x = self.ffn2(x)
        return self.final_norm(x)


class SonicConformerMAE(nn.Module):
    """Hybrid Convolutional-Transformer Masked Autoencoder with Conformer blocks."""
    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_blocks: int = 2,
        embedding_dim: int = 128,
        mask_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio
        self.d_model = d_model

        # Convolutional stem downsampling: (B, 1, 64, 320) -> (B, d_model, 16, 80)
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, d_model, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
        )

        # Conformer sequence encoder (sequences over time frames = 80, features = 16 * d_model = 1024 or projected)
        self.freq_proj = nn.Linear(16 * d_model, d_model)

        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(d_model=d_model, n_heads=n_heads, conv_kernel=7)
            for _ in range(n_blocks)
        ])

        # Global latent embedding head
        self.embedding_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, embedding_dim),
        )

        # Decoder stem: reconstructs back to (B, 1, 64, 320)
        self.decoder_proj = nn.Linear(d_model, 16 * d_model)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(d_model, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
        )

    def mask_patches(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply random patch masking (25% default ratio)."""
        if not self.training or self.mask_ratio <= 0.0:
            mask = torch.ones_like(x)
            return x, mask

        B, C, H, W = x.shape
        # Create patch grid of size 8x16
        ph, pw = 8, 16
        gh, gw = H // ph, W // pw
        num_patches = gh * gw
        num_mask = int(num_patches * self.mask_ratio)

        # Random binary mask
        mask = torch.ones(B, 1, gh, gw, device=x.device)
        for b in range(B):
            perm = torch.randperm(num_patches, device=x.device)[:num_mask]
            mask[b, 0, perm // gw, perm % gw] = 0.0

        # Upsample mask to full resolution
        mask_full = F.interpolate(mask, size=(H, W), mode="nearest")
        return x * mask_full, mask_full

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, 1, 64, 320)
        feat = self.stem(x)  # (B, d_model, 16, 80)
        B, C, F_bins, T_frames = feat.shape
        # Flatten frequency and channel: (B, T, C*F)
        tokens = feat.permute(0, 3, 1, 2).reshape(B, T_frames, C * F_bins)
        tokens = self.freq_proj(tokens)  # (B, T_frames, d_model)

        # Pass through Conformer blocks
        for block in self.conformer_blocks:
            tokens = block(tokens)

        # Global average pool over sequence frames
        latent = tokens.mean(dim=1)  # (B, d_model)
        embedding = self.embedding_head(latent)  # (B, embedding_dim)
        return tokens, embedding

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, T_frames=80, d_model)
        B, T_frames, _ = tokens.shape
        h = self.decoder_proj(tokens)  # (B, 80, 16 * d_model)
        feat = h.reshape(B, T_frames, self.d_model, 16).permute(0, 2, 3, 1)  # (B, d_model, 16, 80)
        recon = self.decoder(feat)  # (B, 1, 64, 320)
        return recon

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        masked_x, _ = self.mask_patches(x)
        tokens, embedding = self.encode(masked_x)
        reconstruction = self.decode(tokens)
        return reconstruction, embedding
