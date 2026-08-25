"""Interactive SonicMAE industrial-audio anomaly-detection dashboard."""
from __future__ import annotations

import io
import json
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
CHECKPOINT = ARTIFACTS / "sonicmae_selected_final_model.pt"
FEATURES = ARTIFACTS / "sonicmae_preprocessed_features.npz"
SUMMARY = ARTIFACTS / "final_project_summary.json"
COMPARISON = ARTIFACTS / "final_model_comparison.csv"
SR, N_FFT, HOP_LENGTH, N_MELS, N_FRAMES, EPS = 16_000, 1024, 512, 64, 320, 1e-8


class SonicMAE(nn.Module):
    """Architecture used by the Kaggle SonicMAE experiment."""
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


def preprocess(audio_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=SR, mono=True)
    audio = audio / max(float(np.max(np.abs(audio))), EPS)
    mel = librosa.feature.melspectrogram(y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0)
    spec = np.log(mel + 1e-6).astype(np.float32)
    if spec.shape[1] < N_FRAMES:
        spec = np.pad(spec, ((0, 0), (0, N_FRAMES - spec.shape[1])), mode="edge")
    else:
        spec = spec[:, :N_FRAMES]
    return audio, spec


@st.cache_resource(show_spinner="Loading SonicMAE and healthy reference profile...")
def load_system() -> tuple[SonicMAE, dict[str, float | np.ndarray]]:
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model = SonicMAE().eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    cache = np.load(FEATURES)
    train = cache["X_train"].astype(np.float32)
    validation = cache["X_val"].astype(np.float32)

    def encode(batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        embeddings, errors = [], []
        with torch.no_grad():
            for start in range(0, len(batch), 32):
                x = torch.from_numpy(batch[start:start + 32, None])
                reconstruction, embedding = model(x)
                embeddings.append(embedding.numpy())
                errors.append(((reconstruction - x).pow(2).mean(dim=(1, 2, 3)).numpy()))
        return np.concatenate(embeddings), np.concatenate(errors)

    train_embeddings, train_errors = encode(train)
    prototype = train_embeddings.mean(axis=0)
    train_distances = np.linalg.norm(train_embeddings - prototype, axis=1)
    val_embeddings, val_errors = encode(validation)
    val_distances = np.linalg.norm(val_embeddings - prototype, axis=1)

    error_mean, error_std = train_errors.mean(), train_errors.std() + EPS
    distance_mean, distance_std = train_distances.mean(), train_distances.std() + EPS
    validation_scores = (val_errors - error_mean) / error_std + (val_distances - distance_mean) / distance_std
    profile = {
        "mean": float(cache["mean"]), "std": float(cache["std"]), "prototype": prototype,
        "error_mean": float(error_mean), "error_std": float(error_std),
        "distance_mean": float(distance_mean), "distance_std": float(distance_std),
        "threshold": float(np.percentile(validation_scores, 95)),
    }
    return model, profile


def analyse(audio_bytes: bytes) -> dict[str, np.ndarray | float | str]:
    model, profile = load_system()
    waveform, raw_spec = preprocess(audio_bytes)
    normalized = (raw_spec - profile["mean"]) / profile["std"]
    x = torch.from_numpy(normalized[None, None])
    with torch.no_grad():
        reconstructed, embedding = model(x)
    reconstruction = reconstructed[0, 0].numpy()
    error_map = (normalized - reconstruction) ** 2
    error = float(error_map.mean())
    distance = float(np.linalg.norm(embedding[0].numpy() - profile["prototype"]))
    score = (error - profile["error_mean"]) / profile["error_std"] + (distance - profile["distance_mean"]) / profile["distance_std"]
    decision = "ANOMALY DETECTED" if score >= profile["threshold"] else "NORMAL OPERATION"
    return {"waveform": waveform, "spectrogram": raw_spec, "reconstruction": reconstruction * profile["std"] + profile["mean"], "error_map": error_map, "error": error, "distance": distance, "score": float(score), "threshold": profile["threshold"], "decision": decision}


def plot_analysis(result: dict[str, np.ndarray | float | str]) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    waveform = result["waveform"]
    axes[0, 0].plot(np.arange(len(waveform)) / SR, waveform, color="#38bdf8", linewidth=.55)
    axes[0, 0].set(title="Uploaded pump waveform", xlabel="Time (s)", ylabel="Amplitude")
    for ax, data, title, cmap in [
        (axes[0, 1], result["spectrogram"], "Input log-Mel spectrogram", "magma"),
        (axes[1, 0], result["reconstruction"], "SonicMAE reconstruction", "magma"),
        (axes[1, 1], result["error_map"], "Time-frequency anomaly heatmap", "hot"),
    ]:
        image = ax.imshow(data, origin="lower", aspect="auto", cmap=cmap)
        ax.set(title=title, xlabel="Time frames", ylabel="Mel bins")
        fig.colorbar(image, ax=ax, fraction=.046)
    fig.tight_layout()
    return fig


st.set_page_config(page_title="SonicMAE | Pump Health Monitor", page_icon="🔊", layout="wide")
st.markdown("""<style>
.stApp {background: #07111f;} .hero {padding: 1.5rem 0 .8rem;} .hero h1 {color:#e7f4ff; margin-bottom:.25rem;}
.hero p {color:#9db7cf;} [data-testid='stMetric'] {background:#101e30; border:1px solid #23415d; padding:13px; border-radius:12px;}
</style>""", unsafe_allow_html=True)
st.markdown("<div class='hero'><h1>🔊 SonicMAE Pump Health Monitor</h1><p>Self-supervised industrial acoustic anomaly detection • MIMII pump benchmark</p></div>", unsafe_allow_html=True)

if not CHECKPOINT.exists() or not FEATURES.exists():
    st.error("Model artifacts are missing. Add the Kaggle checkpoint and feature cache as described in README_UI.md.")
    st.stop()

tab_monitor, tab_results, tab_about = st.tabs(["Live analysis", "Experiment results", "About the model"])
with tab_monitor:
    uploaded = st.file_uploader("Upload a pump WAV recording", type=["wav"], help="The model expects a machine-sound WAV file. It resamples audio to 16 kHz automatically.")
    if uploaded:
        payload = uploaded.getvalue()
        st.audio(payload, format="audio/wav")
        result = analyse(payload)
        anomaly = result["decision"] == "ANOMALY DETECTED"
        st.markdown(f"### {'⚠️' if anomaly else '✅'} {result['decision']}")
        a, b, c, d = st.columns(4)
        a.metric("Anomaly score", f"{result['score']:.3f}")
        b.metric("Decision threshold", f"{result['threshold']:.3f}")
        c.metric("Reconstruction error", f"{result['error']:.4f}")
        d.metric("Healthy-prototype distance", f"{result['distance']:.3f}")
        st.caption("Decision threshold is the 95th percentile of unseen healthy validation recordings. Scores at or above it are flagged.")
        st.pyplot(plot_analysis(result), use_container_width=True)
    else:
        st.info("Upload a `.wav` recording to begin analysis.")

with tab_results:
    if SUMMARY.exists():
        summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
        cols = st.columns(4)
        cols[0].metric("SonicMAE ROC-AUC", f"{summary.get('roc_auc', 0):.4f}")
        cols[1].metric("pAUC @ 10% FPR", f"{summary.get('pauc_at_10_fpr', 0):.4f}")
        cols[2].metric("F1 score", f"{summary.get('f1_score', 0):.4f}")
        cols[3].metric("Recall", f"{summary.get('recall', 0):.4f}")
    if COMPARISON.exists():
        st.subheader("Model comparison")
        st.dataframe(pd.read_csv(COMPARISON), use_container_width=True, hide_index=True)
    else:
        st.info("Add `final_project_summary.json` and `final_model_comparison.csv` to show experiment metrics here.")

with tab_about:
    st.markdown("""### How SonicMAE works
1. WAV audio is converted to a peak-normalized 16 kHz log-Mel spectrogram.
2. A convolutional masked autoencoder reconstructs normal machine-sound patterns.
3. Contrastive learning makes healthy augmented views close in embedding space.
4. The final score combines reconstruction error and distance from the healthy prototype.

**Model selection:** the displayed checkpoint is the tuned `T2_lower_mask` SonicMAE configuration (mask ratio 0.25, contrastive weight 0.20).""")
