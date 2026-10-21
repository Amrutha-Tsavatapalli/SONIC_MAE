from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from .audio import fixed_frames, load_wav, log_mel
from .model import SonicMAE
from .scoring import anomaly_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Score one WAV with a trained SonicMAE model.")
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--reference", required=True); parser.add_argument("--audio", required=True); parser.add_argument("--output-dir", default="scores")
    args = parser.parse_args()
    import torch
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = SonicMAE(payload["n_mels"]); model.load_state_dict(payload["state_dict"]); model.eval()
    spec = fixed_frames(log_mel(load_wav(args.audio, payload["sample_rate"]), payload["sample_rate"], n_mels=payload["n_mels"]), payload["frames"])
    tensor = torch.from_numpy(spec[None, None])
    with torch.no_grad(): reconstruction, embedding, _ = model(tensor, 0.0)
    heatmap = (reconstruction[0, 0].numpy() - spec) ** 2
    error = float(heatmap.mean()); reference = dict(np.load(args.reference))
    score = anomaly_score(embedding[0].numpy(), error, reference)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True); np.save(out / f"{Path(args.audio).stem}_heatmap.npy", heatmap)
    print(f"anomaly_score={score:.6f} reconstruction_error={error:.6f}")


if __name__ == "__main__": main()
