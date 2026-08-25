from __future__ import annotations
import numpy as np


def fit_reference(embeddings: np.ndarray, reconstruction_errors: np.ndarray) -> dict[str, np.ndarray]:
    return {"prototype": embeddings.mean(0), "embedding_scale": np.array(max(np.median(np.linalg.norm(embeddings - embeddings.mean(0), axis=1)), 1e-6)), "error_scale": np.array(max(np.median(reconstruction_errors), 1e-6))}


def anomaly_score(embedding: np.ndarray, reconstruction_error: float, reference: dict[str, np.ndarray]) -> float:
    distance = np.linalg.norm(embedding - reference["prototype"]) / reference["embedding_scale"]
    return float(reconstruction_error / reference["error_scale"] + distance)


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives, negatives = labels == 1, labels == 0
    if not positives.any() or not negatives.any(): return float("nan")
    ranks = np.empty(scores.size, float); ranks[np.argsort(scores)] = np.arange(1, scores.size + 1)
    return float((ranks[positives].sum() - positives.sum() * (positives.sum() + 1) / 2) / (positives.sum() * negatives.sum()))
