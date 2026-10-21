import sys
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import diagnose_root_cause, SonicMAE
from sonicmae.scoring import (
    fit_reference,
    anomaly_score,
    fit_multi_centroid_reference,
    multi_centroid_anomaly_score,
    pauc_at_fpr,
    roc_auc,
)


def test_roc_auc_and_pauc():
    # Synthetic perfect separation
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.15, 0.05, 0.8, 0.9, 0.7, 0.85])
    
    auc = roc_auc(labels, scores)
    assert auc == 1.0

    pauc = pauc_at_fpr(labels, scores, max_fpr=0.10)
    assert np.isfinite(pauc)
    assert pauc > 0.0


def test_multi_centroid_reference_and_scoring():
    np.random.seed(42)
    # Generate two distinct clusters of healthy embeddings
    cluster1 = np.random.normal(loc=-2.0, scale=0.2, size=(30, 128))
    cluster2 = np.random.normal(loc=2.0, scale=0.2, size=(30, 128))
    train_embeddings = np.concatenate([cluster1, cluster2], axis=0)
    reconstruction_errors = np.random.exponential(scale=0.05, size=60)

    ref = fit_multi_centroid_reference(train_embeddings, reconstruction_errors, n_clusters=2)
    assert "centroids" in ref
    assert ref["centroids"].shape == (2, 128)
    assert ref["k"] == 2

    # A normal point close to cluster 1
    normal_point = np.random.normal(loc=-2.0, scale=0.2, size=128)
    norm_score = multi_centroid_anomaly_score(normal_point, 0.04, ref)

    # An abnormal point far from both clusters
    anom_point = np.random.normal(loc=10.0, scale=0.2, size=128)
    anom_score = multi_centroid_anomaly_score(anom_point, 0.25, ref)

    assert anom_score > norm_score


def test_technician_diagnostics_rules():
    # High-frequency error dominance
    high_freq_map = np.zeros((64, 320))
    high_freq_map[50:64, :] = 1.0  # high frequency band
    diag_high = diagnose_root_cause(high_freq_map)
    assert "Bearing" in diag_high["fault_type"]

    # Low-frequency error dominance
    low_freq_map = np.zeros((64, 320))
    low_freq_map[0:15, :] = 1.0  # low frequency band
    diag_low = diagnose_root_cause(low_freq_map)
    assert "Cavitation" in diag_low["fault_type"]


def test_sonicmae_quantization_parity():
    model = SonicMAE(embedding_dim=128).eval()
    x = torch.randn(1, 1, 64, 320)
    
    with torch.no_grad():
        recon_fp32, emb_fp32 = model(x)
        
    assert recon_fp32.shape == (1, 1, 64, 320)
    assert emb_fp32.shape == (1, 128)
    assert torch.isfinite(recon_fp32).all()
    assert torch.isfinite(emb_fp32).all()
