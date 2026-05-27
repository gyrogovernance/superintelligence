# scripts/download_bonsai.py
"""Download Bonsai-8B GGUF (Q1_0) from Hugging Face to data/models."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ID = "prism-ml/Bonsai-8B-gguf"
FILENAME = "Bonsai-8B-Q1_0.gguf"
LOCAL_DIR = Path("data/models/Bonsai-8B-gguf")
APPROX_SIZE_GB = 1.2


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def check_dependencies() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("ERROR: pip install huggingface_hub")
        sys.exit(1)


def check_disk_space(path: Path, required_gb: float) -> bool:
    path.mkdir(parents=True, exist_ok=True)
    _total, _used, free = shutil.disk_usage(path)
    free_gb = free / (1024**3)
    if free_gb < required_gb * 1.1:
        print(f"WARNING: low disk space: {free_gb:.1f} GB free, need ~{required_gb:.1f} GB")
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Bonsai-8B Q1_0 GGUF")
    parser.add_argument("--dry-run", action="store_true", help="Print plan only")
    args = parser.parse_args()

    root = repo_root()
    local_dir = root / LOCAL_DIR
    target = local_dir / FILENAME

    if target.is_file() and target.stat().st_size > 100_000_000:
        print(f"Already present: {target}")
        return 0

    if args.dry_run:
        print(f"Would download {REPO_ID}:{FILENAME} -> {local_dir}")
        return 0

    check_dependencies()
    if not check_disk_space(local_dir, APPROX_SIZE_GB):
        return 1

    from huggingface_hub import hf_hub_download

    print(f"Downloading {FILENAME} from {REPO_ID} ...")
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=str(local_dir),
    )
    print(f"Done: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
