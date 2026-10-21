"""Train and evaluate Transfer Learning (MobileNetV2) and Customized Architecture (SonicConformerMAE)."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sonicmae.conformer import SonicConformerMAE
from sonicmae.scoring import fit_reference, anomaly_score, pauc_at_fpr
from sonicmae.transfer import TransferAcousticNet, fit_transfer_reference, transfer_anomaly_score

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
FEATURES_PATH = ARTIFACTS / "sonicmae_preprocessed_features.npz"
CONFORMER_CKPT = ARTIFACTS / "sonic_conformer_mae.pt"
TRANSFER_CKPT = ARTIFACTS / "transfer_mobilenet_v2.pt"
COMPARISON_CSV = ARTIFACTS / "final_model_comparison.csv"
ABLATION_CSV = ARTIFACTS / "ablation_study_results.csv"


def infonce_loss(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """InfoNCE loss between dual augmented latent views."""
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    sim = torch.mm(z_a, z_b.t()) / temperature
    labels = torch.arange(len(z_a), device=z_a.device)
    return F.cross_entropy(sim, labels)


def train_conformer_mae(
    X_train: np.ndarray,
    X_val: np.ndarray,
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str = "cpu",
) -> SonicConformerMAE:
    print("\n--- Training SonicConformerMAE (Custom Hybrid Conv-Transformer) ---")
    model = SonicConformerMAE(d_model=64, n_heads=4, n_blocks=2, mask_ratio=0.25).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Normalize data
    train_tensor = torch.from_numpy(X_train[:, None, :, :]).float()
    loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_mse = 0.0, 0.0
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()

            recon, emb_a = model(batch_x)
            # Create second view for contrastive alignment
            noise = torch.randn_like(batch_x) * 0.05
            _, emb_b = model(batch_x + noise)

            mse = F.mse_loss(recon, batch_x)
            nce = infonce_loss(emb_a, emb_b)
            loss = mse + 0.20 * nce

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_x)
            total_mse += mse.item() * len(batch_x)

        avg_loss = total_loss / len(train_tensor)
        avg_mse = total_mse / len(train_tensor)
        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} - Loss: {avg_loss:.4f} (MSE: {avg_mse:.4f})")

    torch.save({"model_state_dict": model.state_dict(), "architecture": "SonicConformerMAE"}, CONFORMER_CKPT)
    print(f"SonicConformerMAE checkpoint saved to {CONFORMER_CKPT}")
    return model


def train_transfer_network(
    X_train: np.ndarray,
    epochs: int = 8,
    batch_size: int = 16,
    lr: float = 5e-4,
    device: str = "cpu",
) -> TransferAcousticNet:
    print("\n--- Training TransferAcousticNet (Pretrained MobileNetV2 Adapter) ---")
    model = TransferAcousticNet(embedding_dim=128, pretrained=True).to(device)
    # Freeze MobileNetV2 backbone features initially, train adapter & projection head
    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        list(model.input_adapter.parameters()) + list(model.head.parameters()),
        lr=lr, weight_decay=1e-4,
    )

    train_tensor = torch.from_numpy(X_train[:, None, :, :]).float()
    loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()

            emb_a = model(batch_x)
            noise = torch.randn_like(batch_x) * 0.05
            emb_b = model(batch_x + noise)

            loss = infonce_loss(emb_a, emb_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)

        avg_loss = total_loss / len(train_tensor)
        if epoch % 4 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} - InfoNCE Alignment Loss: {avg_loss:.4f}")

    torch.save({"model_state_dict": model.state_dict(), "architecture": "TransferAcousticNet"}, TRANSFER_CKPT)
    print(f"TransferAcousticNet checkpoint saved to {TRANSFER_CKPT}")
    return model


def evaluate_conformer(
    model: SonicConformerMAE,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    model.eval()
    def get_features(X):
        embs, errs = [], []
        with torch.no_grad():
            for i in range(0, len(X), 32):
                bx = torch.from_numpy(X[i : i + 32][:, None]).float()
                recon, emb = model(bx)
                err = torch.mean((recon - bx) ** 2, dim=(1, 2, 3)).cpu().numpy()
                embs.append(emb.cpu().numpy())
                errs.append(err)
        return np.concatenate(embs), np.concatenate(errs)

    emb_train, err_train = get_features(X_train)
    emb_val, err_val = get_features(X_val)
    emb_test, err_test = get_features(X_test)

    ref = fit_reference(emb_train, err_train)
    val_scores = np.array([anomaly_score(e, r, ref) for e, r in zip(emb_val, err_val)])
    thresh = float(np.percentile(val_scores, 95))

    test_scores = np.array([anomaly_score(e, r, ref) for e, r in zip(emb_test, err_test)])
    preds = (test_scores >= thresh).astype(int)

    auc = float(roc_auc_score(y_test, test_scores))
    pauc = float(pauc_at_fpr(y_test, test_scores, max_fpr=0.10))
    prec = float(precision_score(y_test, preds, zero_division=0))
    rec = float(recall_score(y_test, preds, zero_division=0))
    f1 = float(f1_score(y_test, preds, zero_division=0))

    return {
        "ROC-AUC": round(auc, 4),
        "pAUC @ 10% FPR": round(pauc, 4),
        "F1": round(f1, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "Threshold": round(thresh, 4),
    }


def evaluate_transfer(
    model: TransferAcousticNet,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    model.eval()
    def get_embs(X):
        embs = []
        with torch.no_grad():
            for i in range(0, len(X), 32):
                bx = torch.from_numpy(X[i : i + 32][:, None]).float()
                emb = model(bx)
                embs.append(emb.cpu().numpy())
        return np.concatenate(embs)

    emb_train = get_embs(X_train)
    emb_val = get_embs(X_val)
    emb_test = get_embs(X_test)

    ref = fit_transfer_reference(emb_train)
    val_scores = np.array([transfer_anomaly_score(e, ref) for e in emb_val])
    thresh = float(np.percentile(val_scores, 95))

    test_scores = np.array([transfer_anomaly_score(e, ref) for e in emb_test])
    preds = (test_scores >= thresh).astype(int)

    auc = float(roc_auc_score(y_test, test_scores))
    pauc = float(pauc_at_fpr(y_test, test_scores, max_fpr=0.10))
    prec = float(precision_score(y_test, preds, zero_division=0))
    rec = float(recall_score(y_test, preds, zero_division=0))
    f1 = float(f1_score(y_test, preds, zero_division=0))

    return {
        "ROC-AUC": round(auc, 4),
        "pAUC @ 10% FPR": round(pauc, 4),
        "F1": round(f1, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "Threshold": round(thresh, 4),
    }


def main():
    print(f"Loading data from {FEATURES_PATH}...")
    data = np.load(FEATURES_PATH)
    X_train, X_val, X_test, y_test = data["X_train"], data["X_val"], data["X_test"], data["y_test"]

    # 1. Train & Evaluate SonicConformerMAE
    t0 = time.time()
    conformer_model = train_conformer_mae(X_train, X_val, epochs=15)
    conformer_metrics = evaluate_conformer(conformer_model, X_train, X_val, X_test, y_test)
    print(f"SonicConformerMAE Test Metrics: {conformer_metrics} (Time: {time.time()-t0:.1f}s)")

    # 2. Train & Evaluate TransferAcousticNet
    t0 = time.time()
    transfer_model = train_transfer_network(X_train, epochs=8)
    transfer_metrics = evaluate_transfer(transfer_model, X_train, X_val, X_test, y_test)
    print(f"TransferAcousticNet Test Metrics: {transfer_metrics} (Time: {time.time()-t0:.1f}s)")

    # Update final_model_comparison.csv
    print("\nUpdating benchmark comparison table...")
    comparison_rows = [
        {"Model": "PCA + Isolation Forest", "ROC-AUC": 0.8450, "pAUC @ 10% FPR": 0.7666, "F1": 0.6667, "Precision": 1.0000, "Recall": 0.5000},
        {"Model": "SonicConformerMAE (Custom Hybrid)", "ROC-AUC": conformer_metrics["ROC-AUC"], "pAUC @ 10% FPR": conformer_metrics["pAUC @ 10% FPR"], "F1": conformer_metrics["F1"], "Precision": conformer_metrics["Precision"], "Recall": conformer_metrics["Recall"]},
        {"Model": "Proposed SonicMAE (Conv Baseline)", "ROC-AUC": 0.7862, "pAUC @ 10% FPR": 0.6911, "F1": 0.6040, "Precision": 0.9531, "Recall": 0.4420},
        {"Model": "Transfer Learning (MobileNetV2)", "ROC-AUC": transfer_metrics["ROC-AUC"], "pAUC @ 10% FPR": transfer_metrics["pAUC @ 10% FPR"], "F1": transfer_metrics["F1"], "Precision": transfer_metrics["Precision"], "Recall": transfer_metrics["Recall"]},
        {"Model": "CNN Autoencoder baseline", "ROC-AUC": 0.5702, "pAUC @ 10% FPR": 0.5483, "F1": 0.3729, "Precision": 0.8462, "Recall": 0.2391},
    ]

    df_comp = pd.DataFrame(comparison_rows)
    df_comp.to_csv(COMPARISON_CSV, index=False)
    print(f"Updated {COMPARISON_CSV}:")
    print(df_comp.to_string(index=False))

    # Also record in ablation study results
    if ABLATION_CSV.exists():
        df_abl = pd.read_csv(ABLATION_CSV)
        new_records = [
            {"Experiment": "Custom Arch: SonicConformerMAE (Conv-Transformer)", "ROC-AUC": conformer_metrics["ROC-AUC"], "pAUC @ 10% FPR": conformer_metrics["pAUC @ 10% FPR"], "Precision": conformer_metrics["Precision"], "Recall": conformer_metrics["Recall"], "F1-Score": conformer_metrics["F1"], "Threshold": conformer_metrics["Threshold"]},
            {"Experiment": "Transfer Learning: Pretrained MobileNetV2", "ROC-AUC": transfer_metrics["ROC-AUC"], "pAUC @ 10% FPR": transfer_metrics["pAUC @ 10% FPR"], "Precision": transfer_metrics["Precision"], "Recall": transfer_metrics["Recall"], "F1-Score": transfer_metrics["F1"], "Threshold": transfer_metrics["Threshold"]},
        ]
        df_abl = pd.concat([df_abl, pd.DataFrame(new_records)], ignore_index=True)
        df_abl.to_csv(ABLATION_CSV, index=False)
        print(f"Appended advanced models to {ABLATION_CSV}")


if __name__ == "__main__":
    main()
