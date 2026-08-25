from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

from .augment import augment_spec, phase_scramble
from .audio import fixed_frames, load_wav, log_mel
from .data import discover_wavs, features
from .model import SonicMAE
from .scoring import fit_reference


def _info_nce(a, b, temperature: float = 0.2):
    import torch
    a, b = torch.nn.functional.normalize(a, dim=1), torch.nn.functional.normalize(b, dim=1)
    logits = a @ b.T / temperature
    labels = torch.arange(a.size(0), device=a.device)
    return (torch.nn.functional.cross_entropy(logits, labels) + torch.nn.functional.cross_entropy(logits.T, labels)) / 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SonicMAE only on normal WAV recordings.")
    parser.add_argument("--data-dir", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=30); parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sample-rate", type=int, default=16000); parser.add_argument("--n-mels", type=int, default=64); parser.add_argument("--frames", type=int, default=256)
    parser.add_argument("--mask-ratio", type=float, default=.35); parser.add_argument("--contrastive-weight", type=float, default=.2)
    parser.add_argument("--phase-strength", type=float, default=.2, help="0 disables controlled phase-scrambling augmentation")
    args = parser.parse_args()
    import torch
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    paths = discover_wavs(args.data_dir, normal_only=True)
    x = features(paths, args.sample_rate, args.n_mels, args.frames)
    # A phase-scrambled second view is made from waveform, retaining magnitude and real-signal validity.
    if args.phase_strength:
        rng = np.random.default_rng(7)
        second = np.stack([fixed_frames(log_mel(phase_scramble(load_wav(p, args.sample_rate), args.phase_strength, rng), args.sample_rate, n_mels=args.n_mels), args.frames) for p in paths])
    else: second = x.copy()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SonicMAE(args.n_mels).to(device); opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(42); history = []
    for epoch in range(args.epochs):
        order = rng.permutation(len(x)); losses = []
        model.train()
        for indices in (order[i:i + args.batch_size] for i in range(0, len(order), args.batch_size)):
            first = x[indices]; view = np.stack([augment_spec(second[j], rng) for j in indices])
            a = torch.from_numpy(first[:, None]).to(device); b = torch.from_numpy(view[:, None]).to(device)
            reconstruction, embedding, mask = model(a, args.mask_ratio)
            _, paired_embedding, _ = model(b, args.mask_ratio)
            rec_loss = ((reconstruction - a).pow(2) * mask).sum() / mask.sum().clamp_min(1)
            loss = rec_loss + args.contrastive_weight * _info_nce(embedding, paired_embedding)
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses))})
        print(f"epoch {epoch + 1}/{args.epochs}: loss={history[-1]['loss']:.4f}")
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(x[:, None]).to(device); reconstructed, embedded, _ = model(tensor, 0.0)
        errors = ((reconstructed - tensor).pow(2).mean(dim=(1, 2, 3))).cpu().numpy(); embeddings = embedded.cpu().numpy()
    reference = fit_reference(embeddings, errors)
    np.savez(out / "reference.npz", **reference)
    torch.save({"state_dict": model.state_dict(), "n_mels": args.n_mels, "frames": args.frames, "sample_rate": args.sample_rate}, out / "model.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out / "train_files.txt").write_text("\n".join(map(str, paths)), encoding="utf-8")


if __name__ == "__main__": main()
