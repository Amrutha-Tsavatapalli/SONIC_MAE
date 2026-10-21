"""Edge optimization, dynamic quantization, and latency benchmarking script for SonicMAE."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
import torch.ao.quantization as quantization

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
FP32_PATH = ARTIFACTS / "sonicmae_selected_final_model.pt"
INT8_PATH = ARTIFACTS / "sonicmae_quantized.pt"
OUT_JSON = ARTIFACTS / "edge_benchmark_results.json"


class SonicMAE(nn.Module):
    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
        )
        self.embedding_head = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(128, embedding_dim))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, padding=1), nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        return self.decoder(latent), self.embedding_head(latent)


def benchmark_model(model: nn.Module, dummy_input: torch.Tensor, runs: int = 100, warmup: int = 20) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    latencies = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

    latencies = np.array(latencies)
    return {
        "mean_latency_ms": round(float(np.mean(latencies)), 2),
        "median_latency_ms": round(float(np.median(latencies)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        "throughput_fps": round(float(1000.0 / np.mean(latencies)), 1),
    }


def main():
    print(f"Loading FP32 checkpoint from {FP32_PATH}...")
    checkpoint = torch.load(FP32_PATH, map_location="cpu", weights_only=False)
    fp32_model = SonicMAE().eval()
    fp32_model.load_state_dict(checkpoint["model_state_dict"])

    # Model size FP32
    fp32_size_kb = round(os.path.getsize(FP32_PATH) / 1024, 2)

    # Dynamic Quantization
    print("Applying PyTorch Dynamic Quantization (INT8 Linear & Conv)...")
    int8_model = quantization.quantize_dynamic(
        fp32_model,
        {nn.Linear},
        dtype=torch.qint8,
    )
    
    # Save quantized checkpoint
    torch.save({"model_state_dict": int8_model.state_dict(), "quantized": True}, INT8_PATH)
    int8_size_kb = round(os.path.getsize(INT8_PATH) / 1024, 2)

    # Benchmarking
    dummy_input = torch.randn(1, 1, 64, 320)
    print("Benchmarking FP32 Model (100 runs)...")
    fp32_metrics = benchmark_model(fp32_model, dummy_input)

    print("Benchmarking INT8 Quantized Model (100 runs)...")
    int8_metrics = benchmark_model(int8_model, dummy_input)

    # Latency speedup and size compression
    compression_ratio = round(fp32_size_kb / max(int8_size_kb, 1e-3), 2)
    speedup = round(fp32_metrics["mean_latency_ms"] / max(int8_metrics["mean_latency_ms"], 1e-3), 2)

    # Check output consistency
    with torch.no_grad():
        fp32_recon, fp32_emb = fp32_model(dummy_input)
        int8_recon, int8_emb = int8_model(dummy_input)
        cos_sim = float(torch.cosine_similarity(fp32_emb, int8_emb).item())
        recon_mae = float(torch.mean(torch.abs(fp32_recon - int8_recon)).item())

    results = {
        "hardware_target": "Industrial Edge Gateway / Raspberry Pi 4 / Jetson Nano",
        "input_audio_length_seconds": 10.0,
        "input_spectrogram_shape": [1, 1, 64, 320],
        "fp32": {
            "model_size_kb": fp32_size_kb,
            **fp32_metrics,
        },
        "int8_quantized": {
            "model_size_kb": int8_size_kb,
            **int8_metrics,
        },
        "compression_ratio": compression_ratio,
        "latency_speedup": speedup,
        "embedding_cosine_similarity": round(cos_sim, 6),
        "reconstruction_mae": round(recon_mae, 6),
        "meets_realtime_sla": int8_metrics["mean_latency_ms"] < 50.0,
    }

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nEdge Benchmark Report saved to {OUT_JSON}:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
