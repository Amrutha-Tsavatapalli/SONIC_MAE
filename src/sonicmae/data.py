from __future__ import annotations
from pathlib import Path
import numpy as np
from .audio import fixed_frames, load_wav, log_mel

ANOMALY_WORDS = ("anomaly", "abnormal", "fault")


def discover_wavs(root: str | Path, normal_only: bool = False) -> list[Path]:
    files = sorted(Path(root).rglob("*.wav"))
    if normal_only:
        files = [p for p in files if "normal" in str(p).lower()]
    return files


def label_for(path: Path) -> int | None:
    name = str(path).lower()
    if any(word in name for word in ANOMALY_WORDS): return 1
    if "normal" in name: return 0
    return None


def features(paths: list[Path], sample_rate: int, n_mels: int, frames: int) -> np.ndarray:
    if not paths: raise ValueError("No WAV files found. Check --data-dir and labels.")
    return np.stack([fixed_frames(log_mel(load_wav(p, sample_rate), sample_rate, n_mels=n_mels), frames) for p in paths])
