"""Dependency-light WAV and log-Mel preprocessing."""
from __future__ import annotations

import wave
from pathlib import Path
import numpy as np


def load_wav(path: str | Path, sample_rate: int = 16_000) -> np.ndarray:
    """Read PCM WAV, downmix, linearly resample, and peak-normalize."""
    with wave.open(str(path), "rb") as f:
        channels, width, rate, frames = f.getnchannels(), f.getsampwidth(), f.getframerate(), f.getnframes()
        raw = f.readframes(frames)
    if width == 1:
        signal = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128) / 128
    elif width == 2:
        signal = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
    elif width == 4:
        signal = np.frombuffer(raw, "<i4").astype(np.float32) / 2147483648
    else:
        raise ValueError(f"Unsupported PCM width: {width}")
    signal = signal.reshape(-1, channels).mean(axis=1)
    if rate != sample_rate and signal.size:
        out_n = round(signal.size * sample_rate / rate)
        signal = np.interp(np.linspace(0, signal.size - 1, out_n), np.arange(signal.size), signal).astype(np.float32)
    return signal / max(float(np.abs(signal).max()), 1e-8)


def _mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    hz_to_mel = lambda hz: 2595 * np.log10(1 + hz / 700)
    mel_to_hz = lambda mel: 700 * (10 ** (mel / 2595) - 1)
    hz = mel_to_hz(np.linspace(hz_to_mel(0), hz_to_mel(sample_rate / 2), n_mels + 2))
    bins = np.floor((n_fft + 1) * hz / sample_rate).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = bins[i : i + 3]
        if center > left: fb[i, left:center] = np.arange(left, center) / (center - left)
        if right > center: fb[i, center:right] = (right - np.arange(center, right)) / (right - center)
    return fb


def log_mel(signal: np.ndarray, sample_rate: int = 16_000, n_fft: int = 1024, hop: int = 256, n_mels: int = 64) -> np.ndarray:
    """Return a [mel, time] log-power spectrogram using NumPy only."""
    if signal.size < n_fft:
        signal = np.pad(signal, (0, n_fft - signal.size))
    frames = np.lib.stride_tricks.sliding_window_view(signal, n_fft)[::hop]
    power = np.abs(np.fft.rfft(frames * np.hanning(n_fft), axis=1)) ** 2
    return np.log(np.maximum(_mel_filterbank(sample_rate, n_fft, n_mels) @ power.T, 1e-10)).astype(np.float32)


def fixed_frames(spec: np.ndarray, frames: int = 256) -> np.ndarray:
    """Crop or right-pad a spectrogram to a deterministic model length."""
    if spec.shape[1] >= frames:
        return spec[:, :frames]
    return np.pad(spec, ((0, 0), (0, frames - spec.shape[1])), mode="edge")
