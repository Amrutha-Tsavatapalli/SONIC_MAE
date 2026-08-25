"""Generate reproducible EDA for a labelled industrial-audio folder."""
from __future__ import annotations
import argparse, csv, json, wave
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sonicmae.audio import fixed_frames, load_wav, log_mel
from sonicmae.data import discover_wavs, label_for


def duration(path: Path) -> float:
    with wave.open(str(path), "rb") as f:
        return f.getnframes() / f.getframerate()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create EDA tables and plots from MIMII-style WAV files.")
    parser.add_argument("--data-dir", required=True); parser.add_argument("--output-dir", default="reports/eda")
    parser.add_argument("--sample-rate", type=int, default=16000); args = parser.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    records = []
    for path in discover_wavs(args.data_dir):
        label = label_for(path)
        if label is not None: records.append({"path": str(path), "label": "anomaly" if label else "normal", "duration_s": duration(path)})
    if not records: raise ValueError("No labelled WAV files found. Extract MIMII and pass its data directory.")
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, records[0].keys()); writer.writeheader(); writer.writerows(records)
    counts, durations = Counter(row["label"] for row in records), [row["duration_s"] for row in records]
    summary = {"files": len(records), "class_counts": dict(counts), "duration_seconds": {"total": sum(durations), "mean": float(np.mean(durations)), "min": min(durations), "max": max(durations)}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4)); ax.bar(counts.keys(), counts.values(), color=["#2374ab", "#d9534f"]); ax.set(title="MIMII files by label", ylabel="recordings"); fig.tight_layout(); fig.savefig(out / "class_balance.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4)); ax.hist(durations, bins=30, color="#2374ab"); ax.set(title="Recording-duration distribution", xlabel="seconds", ylabel="recordings"); fig.tight_layout(); fig.savefig(out / "durations.png", dpi=180); plt.close(fig)
    sample = Path(records[0]["path"]); signal = load_wav(sample, args.sample_rate); spec = fixed_frames(log_mel(signal, args.sample_rate))
    fig, axes = plt.subplots(2, 1, figsize=(10, 6)); axes[0].plot(np.arange(len(signal)) / args.sample_rate, signal, lw=.5); axes[0].set(title=f"Waveform: {sample.name}", xlabel="seconds"); image = axes[1].imshow(spec, origin="lower", aspect="auto", cmap="magma"); axes[1].set(title="Log-Mel spectrogram", xlabel="frames", ylabel="Mel bin"); fig.colorbar(image, ax=axes[1]); fig.tight_layout(); fig.savefig(out / "sample_waveform_spectrogram.png", dpi=180); plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
