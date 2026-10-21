from __future__ import annotations
import numpy as np


def fit_reference(embeddings: np.ndarray, reconstruction_errors: np.ndarray) -> dict[str, np.ndarray]:
    return {"prototype": embeddings.mean(0), "embedding_scale": np.array(max(np.median(np.linalg.norm(embeddings - embeddings.mean(0), axis=1)), 1e-6)), "error_scale": np.array(max(np.median(reconstruction_errors), 1e-6))}


def anomaly_score(embedding: np.ndarray, reconstruction_error: float, reference: dict[str, np.ndarray]) -> float:
    distance = np.linalg.norm(embedding - reference["prototype"]) / reference["embedding_scale"]
    return float(reconstruction_error / reference["error_scale"] + distance)


def pauc_at_fpr(labels: np.ndarray, scores: np.ndarray, max_fpr: float = 0.1) -> float:
    """Calculate partial AUC up to max_fpr (standard DCASE benchmark metric)."""
    positives = labels == 1
    negatives = labels == 0
    if not positives.any() or not negatives.any():
        return float("nan")
    # Sort negative scores to find threshold at max_fpr
    neg_scores = np.sort(scores[negatives])
    # Index of cutoff among negatives
    cutoff_idx = int(np.floor((1.0 - max_fpr) * len(neg_scores)))
    cutoff_idx = min(cutoff_idx, len(neg_scores) - 1)
    thresh = neg_scores[cutoff_idx]
    
    # Calculate standard ROC curve points below max_fpr
    sorted_indices = np.argsort(-scores)
    sorted_labels = labels[sorted_indices]
    
    tp, fp = 0, 0
    n_pos, n_neg = int(positives.sum()), int(negatives.sum())
    tpr_list, fpr_list = [0.0], [0.0]
    
    for label in sorted_labels:
        if label == 1:
            tp += 1
        else:
            fp += 1
        current_fpr = fp / n_neg
        current_tpr = tp / n_pos
        if current_fpr > max_fpr:
            # Interpolate to max_fpr
            prev_fpr = fpr_list[-1]
            prev_tpr = tpr_list[-1]
            alpha = (max_fpr - prev_fpr) / max(current_fpr - prev_fpr, 1e-12)
            interp_tpr = prev_tpr + alpha * (current_tpr - prev_tpr)
            fpr_list.append(max_fpr)
            tpr_list.append(interp_tpr)
            break
        fpr_list.append(current_fpr)
        tpr_list.append(current_tpr)
        
    if fpr_list[-1] < max_fpr:
        fpr_list.append(max_fpr)
        tpr_list.append(tpr_list[-1])
        
    # Trapezoidal integration divided by max_fpr
    pauc = np.trapz(tpr_list, fpr_list) / max_fpr
    return float(np.clip(pauc, 0.0, 1.0))


def fit_multi_centroid_reference(
    embeddings: np.ndarray,
    reconstruction_errors: np.ndarray,
    n_clusters: int = 5,
    random_state: int = 42,
) -> dict[str, np.ndarray | float]:
    """Fit multi-modal normal prototype bank using K-Means clustering."""
    from sklearn.cluster import KMeans

    k = min(n_clusters, len(embeddings))
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(embeddings)
    centroids = kmeans.cluster_centers_  # shape: (k, dim)

    # Distances of training points to their closest centroid
    min_dists = np.min(np.linalg.norm(embeddings[:, None, :] - centroids[None, :, :], axis=2), axis=1)
    embedding_scale = float(max(np.median(min_dists), 1e-6))
    error_scale = float(max(np.median(reconstruction_errors), 1e-6))

    return {
        "prototype": embeddings.mean(0),
        "centroids": centroids,
        "embedding_scale": np.array(embedding_scale),
        "error_scale": np.array(error_scale),
        "k": k,
    }


def multi_centroid_anomaly_score(
    embedding: np.ndarray,
    reconstruction_error: float,
    reference: dict[str, np.ndarray | float],
    temperature: float = 1.0,
) -> float:
    """Softmax-weighted multi-centroid distance combined with reconstruction error."""
    centroids = reference.get("centroids")
    if centroids is None:
        return anomaly_score(embedding, reconstruction_error, reference)

    dists = np.linalg.norm(centroids - embedding, axis=1)  # shape (k,)
    weights = np.exp(-dists / max(temperature, 1e-4))
    weights /= max(weights.sum(), 1e-8)
    soft_dist = float(np.sum(weights * dists))
    
    embedding_scale = float(reference["embedding_scale"])
    error_scale = float(reference["error_scale"])
    return float(reconstruction_error / error_scale + soft_dist / embedding_scale)


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute Area Under the Receiver Operating Characteristic Curve."""
    positives, negatives = labels == 1, labels == 0
    if not positives.any() or not negatives.any():
        return float("nan")
    ranks = np.empty(scores.size, float)
    ranks[np.argsort(scores)] = np.arange(1, scores.size + 1)
    return float((ranks[positives].sum() - positives.sum() * (positives.sum() + 1) / 2) / (positives.sum() * negatives.sum()))

