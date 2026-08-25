"""Signal and spectrogram augmentations used by SonicMAE."""
from __future__ import annotations
import numpy as np


def phase_scramble(signal: np.ndarray, strength: float = 0.20, rng: np.random.Generator | None = None) -> np.ndarray:
    """Bounded random phase perturbation retaining real-signal FFT symmetry."""
    if not 0 <= strength <= 1:
        raise ValueError("strength must be in [0, 1]")
    rng = rng or np.random.default_rng()
    spectrum = np.fft.rfft(signal)
    # rfft/irfft preserves conjugate symmetry; DC and Nyquist stay real.
    offsets = rng.uniform(-np.pi * strength, np.pi * strength, spectrum.size)
    offsets[0] = 0
    if signal.size % 2 == 0: offsets[-1] = 0
    result = np.fft.irfft(np.abs(spectrum) * np.exp(1j * (np.angle(spectrum) + offsets)), n=signal.size)
    return result.astype(np.float32)


def augment_spec(spec: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    result = spec.copy() + rng.normal(0, 0.03, spec.shape).astype(np.float32)
    shift = int(rng.integers(-8, 9))
    return np.roll(result, shift, axis=1)
