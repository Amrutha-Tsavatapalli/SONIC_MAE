"""One command to run EDA then train SonicMAE on extracted MIMII audio."""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/mimii_due")
    parser.add_argument("--run-dir", default="runs/mimii_due_pump")
    parser.add_argument("--epochs", type=int, default=30); args = parser.parse_args()
    data, run = Path(args.data_dir), Path(args.run_dir)
    if not data.exists(): raise FileNotFoundError(f"{data} is missing. Run download_mimii_due_pump.py --extract first.")
    subprocess.run([sys.executable, "scripts/eda.py", "--data-dir", str(data), "--output-dir", str(run / "eda")], check=True)
    subprocess.run([sys.executable, "-m", "sonicmae.train", "--data-dir", str(data), "--output-dir", str(run), "--epochs", str(args.epochs)], check=True)


if __name__ == "__main__": main()
