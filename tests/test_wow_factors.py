"""Unit tests for acoustic residual sonification and stress-testing utilities."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sonicmae.sonify import resynthesize_residual, audio_to_wav_bytes
from sonicmae.stress import inject_noise, apply_phase_jitter, compare_stress_resilience


def test_resynthesize_residual():
    # Synthetic log-Mel spectrograms
    raw_spec = np.random.uniform(0.1, 2.0, size=(64, 100)).astype(np.float32)
    recon_spec = raw_spec * 0.7  # imperfect reconstruction
    
    audio_res, wav_bytes = resynthesize_residual(raw_spec, recon_spec, sr=16000, n_iter=8)
    
    assert len(audio_res) > 0
    assert np.isfinite(audio_res).all()
    assert len(wav_bytes) > 44  # Valid WAV header + data
    assert wav_bytes[:4] == b"RIFF"


def test_inject_noise_and_phase_jitter():
    audio = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32)
    
    # Noise injection
    noisy = inject_noise(audio, snr_db=10.0)
    assert noisy.shape == audio.shape
    assert np.isfinite(noisy).all()
    assert not np.allclose(noisy, audio)

    # Phase jitter
    jittered = apply_phase_jitter(audio, strength=0.3)
    assert jittered.shape == audio.shape
    assert np.isfinite(jittered).all()


def test_stress_resilience_comparison():
    res = compare_stress_resilience(
        current_score=1.5,
        threshold=2.9,
        snr_db=5.0,        # Heavy noise
        phase_jitter=0.5,  # Strong phase shift
    )
    # SonicMAE score should stay more robust than CNN score
    assert res["sonicmae_score"] < res["cnn_score"]
    assert "threshold" in res
