import numpy as np
from sonicmae.audio import log_mel
from sonicmae.augment import phase_scramble


def test_log_mel_shape_and_finite():
    signal = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32)
    spec = log_mel(signal)
    assert spec.shape[0] == 64
    assert np.isfinite(spec).all()


def test_phase_scrambling_preserves_length_and_energy_reasonably():
    signal = np.random.default_rng(1).normal(size=2048).astype(np.float32)
    changed = phase_scramble(signal, .4, np.random.default_rng(2))
    assert changed.shape == signal.shape
    assert np.isclose(np.linalg.norm(changed), np.linalg.norm(signal), rtol=1e-5)
