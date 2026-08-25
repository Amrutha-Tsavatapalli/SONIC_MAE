# SonicMAE

SonicMAE is an unsupervised industrial anomalous-sound detection baseline based on the project brief. It trains on normal recordings, reconstructs masked log-Mel patches, learns invariant embeddings with InfoNCE, and scores recordings using both reconstruction error and distance to healthy embedding prototypes.

## Install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

## Expected data layout

Use WAV files in any nested folder structure. Files containing `normal` are used for training; files containing `anomaly`, `abnormal`, or `fault` are reserved for evaluation. For example:

```text
data/mimii/
  pump/id_00/normal/*.wav
  pump/id_00/anomaly/*.wav
```

This works with exports from MIMII, ToyADMOS2, and DCASE after preserving normal/anomaly words in the filenames or parent folders.

## Train and score

```powershell
sonicmae-train --data-dir data/mimii --output-dir runs/pump --epochs 30
sonicmae-score --checkpoint runs/pump/model.pt --reference runs/pump/reference.npz --audio data/mimii/pump/id_00/anomaly/example.wav
```

The training command writes a checkpoint, normal-reference embeddings, training history, and an evaluation JSON when held-out labelled files exist. The scoring command prints the combined anomaly score and writes a time-frequency heatmap array (`.npy`) identifying unusual regions.

## Method

1. WAV is converted to mono, resampled, peak-normalized, and represented as a log-Mel spectrogram.
2. The model masks random time-frequency patches and reconstructs them.
3. A second augmented view receives gain, time-shift, Gaussian noise, and optional conjugate-symmetric phase scrambling; InfoNCE aligns both embeddings.
4. Normal embeddings form a prototype bank. Final score = normalized reconstruction error + normalized prototype distance.

Phase scrambling changes spectral phase but retains magnitude and the real-signal conjugate constraint. It is deliberately bounded by `--phase-strength`; it should be ablated against the same model without it.

## Limitations

This is a research baseline, not a safety-certified maintenance system. Fit score normalization on normal validation audio from the intended deployment environment, and report ROC-AUC, pAUC, F-beta, and source/target harmonic means separately.
