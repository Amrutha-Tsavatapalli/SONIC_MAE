"""Acoustic residual resynthesis engine: inverts time-frequency reconstruction error back to playable audio."""
from __future__ import annotations

import io
import librosa
import numpy as np
import soundfile as sf


def resynthesize_residual(
    raw_spec: np.ndarray,
    recon_spec: np.ndarray,
    sr: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_iter: int = 32,
) -> tuple[np.ndarray, bytes]:
    """Invert the rectified spectral error |X - X̂| back to time-domain audio using Griffin-Lim.
    
    This audibly isolates anomalous clicks, friction, and cavitation sounds while
    subtracting normal machine operating harmonics.
    """
    # Positive residual energy (anomalous acoustic content)
    residual_spec = np.maximum(raw_spec - recon_spec, 0.0)
    
    # Convert log-Mel back to linear Mel energy
    mel_power = np.exp(residual_spec) - 1e-6
    mel_power = np.maximum(mel_power, 1e-8)
    
    # Fast Griffin-Lim Mel-to-Audio inverse synthesis
    audio_residual = librosa.feature.inverse.mel_to_audio(
        mel_power,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_iter=n_iter,
        power=2.0,
    )
    
    # Peak normalization for clear audible listening
    max_val = np.max(np.abs(audio_residual))
    if max_val > 1e-6:
        audio_residual = audio_residual / max_val * 0.95
        
    # Encode as WAV bytes
    buf = io.BytesIO()
    sf.write(buf, audio_residual, sr, format="WAV", subtype="PCM_16")
    wav_bytes = buf.getvalue()
    
    return audio_residual, wav_bytes


def audio_to_wav_bytes(audio: np.ndarray, sr: int = 16000) -> bytes:
    """Encode float32 audio array to 16-bit PCM WAV bytes."""
    buf = io.BytesIO()
    norm_audio = audio / max(float(np.max(np.abs(audio))), 1e-6) * 0.95
    sf.write(buf, norm_audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()
