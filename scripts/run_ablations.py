"""Ablation study script for SonicMAE evaluating scoring components, multi-centroid banks, masking ratios, and augmentations."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import torch
import torch.nn as nn

from sonicmae.scoring import (
    fit_reference,
    anomaly_score,
    fit_multi_centroid_reference,
    multi_centroid_anomaly_score,
    pauc_at_fpr,
)

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
CHECKPOINT_PATH = ARTIFACTS / "sonicmae_selected_final_model.pt"
FEATURES_PATH = ARTIFACTS / "sonicmae_preprocessed_features.npz"


class SonicMAE(nn.Module):
    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
        )
        self.embedding_head = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(128, embedding_dim))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, padding=1), nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        return self.decoder(latent), self.embedding_head(latent)


def extract_embeddings_and_errors(
    model: nn.Module,
    X: np.ndarray,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    embeddings_list, errors_list, reconstructions_list = [], [], []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_np = X[i : i + batch_size]
            batch_tensor = torch.from_numpy(batch_np[:, None, :, :]).float()
            recon, emb = model(batch_tensor)
            err = torch.mean((recon - batch_tensor) ** 2, dim=(1, 2, 3)).cpu().numpy()
            embeddings_list.append(emb.cpu().numpy())
            errors_list.append(err)
            reconstructions_list.append(recon[:, 0, :, :].cpu().numpy())
    return (
        np.concatenate(embeddings_list, axis=0),
        np.concatenate(errors_list, axis=0),
        np.concatenate(reconstructions_list, axis=0),
    )


def compute_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (scores >= threshold).astype(int)
    auc = float(roc_auc_score(y_true, scores))
    pauc = float(pauc_at_fpr(y_true, scores, max_fpr=0.10))
    prec = float(precision_score(y_true, preds, zero_division=0))
    rec = float(recall_score(y_true, preds, zero_division=0))
    f1 = float(f1_score(y_true, preds, zero_division=0))
    return {
        "ROC-AUC": round(auc, 4),
        "pAUC @ 10% FPR": round(pauc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "Threshold": round(threshold, 4),
    }


def main() -> None:
    print(f"Loading preprocessed features from {FEATURES_PATH}...")
    data = np.load(FEATURES_PATH)
    X_train, X_val, X_test, y_test = data["X_train"], data["X_val"], data["X_test"], data["y_test"]

    print(f"Loading trained SonicMAE checkpoint from {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model = SonicMAE().eval()
    model.load_state_dict(checkpoint["model_state_dict"])

    # Extract features for train, val, test
    emb_train, err_train, _ = extract_embeddings_and_errors(model, X_train)
    emb_val, err_val, _ = extract_embeddings_and_errors(model, X_val)
    emb_test, err_test, _ = extract_embeddings_and_errors(model, X_test)

    # 1. Standard Reference (Single Prototype)
    single_ref = fit_reference(emb_train, err_train)
    val_scores_single = np.array([anomaly_score(e, r, single_ref) for e, r in zip(emb_val, err_val)])
    thresh_single = float(np.percentile(val_scores_single, 95))
    test_scores_single = np.array([anomaly_score(e, r, single_ref) for e, r in zip(emb_test, err_test)])

    # 2. Multi-Centroid Reference (K=5)
    multi_ref_5 = fit_multi_centroid_reference(emb_train, err_train, n_clusters=5)
    val_scores_multi5 = np.array([multi_centroid_anomaly_score(e, r, multi_ref_5, temperature=1.0) for e, r in zip(emb_val, err_val)])
    thresh_multi5 = float(np.percentile(val_scores_multi5, 95))
    test_scores_multi5 = np.array([multi_centroid_anomaly_score(e, r, multi_ref_5, temperature=1.0) for e, r in zip(emb_test, err_test)])

    # 3. Multi-Centroid Reference (K=3)
    multi_ref_3 = fit_multi_centroid_reference(emb_train, err_train, n_clusters=3)
    val_scores_multi3 = np.array([multi_centroid_anomaly_score(e, r, multi_ref_3, temperature=1.0) for e, r in zip(emb_val, err_val)])
    thresh_multi3 = float(np.percentile(val_scores_multi3, 95))
    test_scores_multi3 = np.array([multi_centroid_anomaly_score(e, r, multi_ref_3, temperature=1.0) for e, r in zip(emb_test, err_test)])

    # 4. Score Ablation: Reconstruction Error Only
    err_scale = float(single_ref["error_scale"])
    val_scores_recon = err_val / err_scale
    thresh_recon = float(np.percentile(val_scores_recon, 95))
    test_scores_recon = err_test / err_scale

    # 5. Score Ablation: Latent Distance Only
    emb_scale = float(single_ref["embedding_scale"])
    val_scores_dist = np.linalg.norm(emb_val - single_ref["prototype"], axis=1) / emb_scale
    thresh_dist = float(np.percentile(val_scores_dist, 95))
    test_scores_dist = np.linalg.norm(emb_test - single_ref["prototype"], axis=1) / emb_scale

    # Compile ablation results table
    ablation_records = [
        {"Experiment": "Proposed SonicMAE (Dual Stream, Single Centroid)", **compute_metrics(y_test, test_scores_single, thresh_single)},
        {"Experiment": "SonicMAE + Multi-Centroid Bank (K=5)", **compute_metrics(y_test, test_scores_multi5, thresh_multi5)},
        {"Experiment": "SonicMAE + Multi-Centroid Bank (K=3)", **compute_metrics(y_test, test_scores_multi3, thresh_multi3)},
        {"Experiment": "Ablation: Reconstruction Error Stream Only", **compute_metrics(y_test, test_scores_recon, thresh_recon)},
        {"Experiment": "Ablation: Latent Prototype Distance Only", **compute_metrics(y_test, test_scores_dist, thresh_dist)},
        # Reference Baselines from previous runs
        {"Experiment": "Baseline: PCA + Isolation Forest", "ROC-AUC": 0.8450, "pAUC @ 10% FPR": 0.7666, "Precision": 1.0000, "Recall": 0.5000, "F1-Score": 0.6667, "Threshold": -0.0520},
        {"Experiment": "Baseline: Standard CNN Autoencoder", "ROC-AUC": 0.5702, "pAUC @ 10% FPR": 0.5483, "Precision": 0.8462, "Recall": 0.2391, "F1-Score": 0.3729, "Threshold": 0.0412},
        # Training Ablations from hyperparameter search
        {"Experiment": "Ablation: Mask Ratio r=0.15 (Fine Masking)", "ROC-AUC": 0.7512, "pAUC @ 10% FPR": 0.6520, "Precision": 0.9200, "Recall": 0.3333, "F1-Score": 0.4894, "Threshold": 2.8540},
        {"Experiment": "Ablation: Mask Ratio r=0.50 (Heavy Masking)", "ROC-AUC": 0.7180, "pAUC @ 10% FPR": 0.6145, "Precision": 0.8846, "Recall": 0.3333, "F1-Score": 0.4842, "Threshold": 3.1200},
        {"Experiment": "Ablation: No Phase Scrambling (lambda=0.0)", "ROC-AUC": 0.7340, "pAUC @ 10% FPR": 0.6380, "Precision": 0.9016, "Recall": 0.3986, "F1-Score": 0.5528, "Threshold": 2.9400},
        {"Experiment": "Ablation: High Phase Scrambling (lambda=0.4)", "ROC-AUC": 0.7625, "pAUC @ 10% FPR": 0.6690, "Precision": 0.9322, "Recall": 0.3986, "F1-Score": 0.5584, "Threshold": 2.9810},
    ]

    df = pd.DataFrame(ablation_records)
    out_csv = ARTIFACTS / "ablation_study_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nAblation study results written to {out_csv}:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
