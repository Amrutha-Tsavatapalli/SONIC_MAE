"""Unit tests for Transfer Learning (MobileNetV2) and Customized Architecture (SonicConformerMAE)."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sonicmae.conformer import SonicConformerMAE, ConformerBlock
from sonicmae.transfer import TransferAcousticNet, fit_transfer_reference, transfer_anomaly_score


def test_conformer_block_forward():
    block = ConformerBlock(d_model=32, n_heads=2, conv_kernel=5, dropout=0.0)
    x = torch.randn(2, 40, 32)
    out = block(x)
    assert out.shape == (2, 40, 32)
    assert torch.isfinite(out).all()


def test_sonic_conformer_mae_full_pass():
    model = SonicConformerMAE(d_model=32, n_heads=2, n_blocks=1, embedding_dim=64, mask_ratio=0.25).eval()
    x = torch.randn(2, 1, 64, 320)
    recon, emb = model(x)
    
    assert recon.shape == (2, 1, 64, 320)
    assert emb.shape == (2, 64)
    assert torch.isfinite(recon).all()
    assert torch.isfinite(emb).all()


def test_transfer_acoustic_net_embedding_and_scoring():
    # Use uninitialized backbone for rapid unit test
    model = TransferAcousticNet(embedding_dim=64, pretrained=False).eval()
    x = torch.randn(3, 1, 64, 320)
    
    with torch.no_grad():
        emb = model(x).cpu().numpy()
        
    assert emb.shape == (3, 64)
    # Verify embeddings are normalized
    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)

    # Reference prototype fitting
    ref = fit_transfer_reference(emb)
    assert "prototype" in ref
    assert ref["prototype"].shape == (64,)

    # Anomaly scoring
    score = transfer_anomaly_score(emb[0], ref)
    assert np.isfinite(score)
