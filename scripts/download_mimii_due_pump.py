"""Resumable downloader for the official MIMII DUE pump development split."""
from __future__ import annotations
import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

URL = "https://zenodo.org/records/4740355/files/dev_data_pump.zip?download=1"
EXPECTED_BYTES = 1_100_000_000


def download(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    offset = destination.stat().st_size if destination.exists() else 0
    request = urllib.request.Request(URL, headers={"Range": f"bytes={offset}-"} if offset else {})
    print(f"Downloading from byte {offset:,}...")
    with urllib.request.urlopen(request) as response, destination.open("ab" if offset else "wb") as file:
        shutil.copyfileobj(response, file, length=1024 * 1024)
    print(f"Saved {destination.stat().st_size:,} bytes to {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default="data/archives/mimii_due_dev_data_pump.zip")
    parser.add_argument("--extract-dir", default="data/raw/mimii_due")
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args(); archive = Path(args.archive)
    download(archive)
    if args.extract:
        if archive.stat().st_size < EXPECTED_BYTES * .8: raise RuntimeError("Archive looks incomplete; re-run to resume before extracting.")
        Path(args.extract_dir).mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf: zf.extractall(args.extract_dir)
        print(f"Extracted to {args.extract_dir}")


if __name__ == "__main__": main()
