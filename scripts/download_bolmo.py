# scripts/download_bolmo.py
"""
Download Bolmo-1B and Bolmo-7B from HuggingFace to data/models.

Usage:
    python scripts/download_bolmo.py              # download both
    python scripts/download_bolmo.py --model 1b   # download only 1B
    python scripts/download_bolmo.py --model 7b   # download only 7B
    python scripts/download_bolmo.py --dry-run    # show what would be downloaded
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

MODELS = {
    "1b": {
        "repo_id": "allenai/Bolmo-1B",
        "local_dir": Path("data/models/Bolmo-1B"),
        "approx_size_gb": 5.9,
    },
    "7b": {
        "repo_id": "allenai/Bolmo-7B",
        "local_dir": Path("data/models/Bolmo-7B"),
        "approx_size_gb": 15.0,
    },
}

REQUIRED_FILES = [
    "config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenization_bolmo.py",
    "configuration_bolmo.py",
    "modeling_bolmo.py",
    "generation_config.json",
]


def check_dependencies() -> None:
    """Verify required packages are available."""
    missing = []
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        missing.append("huggingface_hub")

    try:
        import transformers  # noqa: F401
    except ImportError:
        missing.append("transformers")

    if missing:
        print("ERROR: Missing required packages:")
        for pkg in missing:
            print(f"  pip install {pkg}")
        sys.exit(1)


def check_disk_space(path: Path, required_gb: float) -> bool:
    """Check if there is enough disk space at the given path."""
    path.mkdir(parents=True, exist_ok=True)
    total, used, free = shutil.disk_usage(path)
    free_gb = free / (1024 ** 3)
    if free_gb < required_gb * 1.1:  # 10% buffer
        print(f"  WARNING: Low disk space. Free: {free_gb:.1f} GB, Required: {required_gb:.1f} GB")
        return False
    return True


def is_already_downloaded(local_dir: Path) -> bool:
    """Check if the model appears to already be fully downloaded."""
    if not local_dir.exists():
        return False

    # Check for required metadata files
    for fname in REQUIRED_FILES:
        if not (local_dir / fname).exists():
            return False

    # Check for at least one weight shard
    safetensors = list(local_dir.glob("*.safetensors"))
    pytorch_bins = list(local_dir.glob("pytorch_model*.bin"))

    if not safetensors and not pytorch_bins:
        return False

    return True


def format_bytes(n: int) -> str:
    """Human-readable byte size."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} TB"


def download_model(
    repo_id: str,
    local_dir: Path,
    approx_size_gb: float,
    dry_run: bool = False,
) -> bool:
    """
    Download a single model from HuggingFace Hub.

    Returns True on success, False on failure.
    """
    print(f"\n{'=' * 60}")
    print(f"Model : {repo_id}")
    print(f"Target: {local_dir.resolve()}")
    print(f"Size  : ~{approx_size_gb:.1f} GB")
    print(f"{'=' * 60}")

    if is_already_downloaded(local_dir):
        print(f"  SKIPPING: Model already present at {local_dir}")
        print("  To re-download, delete the directory and run again.")
        return True

    if dry_run:
        print("  DRY RUN: Would download this model (skipping actual download).")
        return True

    # Check disk space
    check_disk_space(local_dir.parent, approx_size_gb)

    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import (
        EntryNotFoundError,
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
    )

    print("  Starting download...")
    print("  (This may take a while depending on your connection.)")
    t0 = time.time()

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            # Exclude olmo_core checkpoint format (large, only needed for training)
            # Keep the HF-format files needed for inference
            ignore_patterns=[
                "olmo_core/*",       # training-format checkpoints
                "*.msgpack",         # JAX weights
                "flax_model*",       # Flax weights
                "tf_model*",         # TensorFlow weights
                "rust_model*",       # Rust weights
            ],
        )
        elapsed = time.time() - t0
        print(f"  Download complete in {elapsed:.0f}s ({elapsed / 60:.1f} min)")

        # Verify critical files arrived
        missing = [f for f in REQUIRED_FILES if not (local_dir / f).exists()]
        if missing:
            print(f"  WARNING: Some expected files are missing: {missing}")
            print("  The download may have been incomplete.")
            return False

        # Report disk usage
        total_bytes = sum(
            f.stat().st_size
            for f in local_dir.rglob("*")
            if f.is_file()
        )
        print(f"  Disk usage: {format_bytes(total_bytes)}")
        print(f"  Location  : {local_dir.resolve()}")
        return True

    except RepositoryNotFoundError:
        print(f"  ERROR: Repository '{repo_id}' not found on HuggingFace.")
        print("  Check the repo ID or your HF_TOKEN if the model is gated.")
        return False

    except (EntryNotFoundError, LocalEntryNotFoundError) as e:
        print(f"  ERROR: A file was not found (or offline mode): {e}")
        return False

    except KeyboardInterrupt:
        print("\n  Download interrupted by user.")
        print("  Partial files kept. Re-run to resume.")
        return False

    except Exception as e:  # noqa: BLE001
        print(f"  ERROR: Unexpected error during download: {type(e).__name__}: {e}")
        return False

def print_post_download_usage(downloaded: list[str]) -> None:
    """Print usage instructions after download."""
    if not downloaded:
        return

    print("\n" + "=" * 60)
    print("USAGE WITH GYROLABE")
    print("=" * 60)

    for key in downloaded:
        info = MODELS[key]
        local = info["local_dir"]
        print(f"\n# {info['repo_id']}")
        print(f"model_path = \"{local}\"")

    print("""
# Basic HuggingFace inference:
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "data/models/Bolmo-1B"   # or Bolmo-7B
bolmo = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype="auto",
).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# GyroLabe coupling (same API as OLMo):
from src.tools.gyrolabe import GyroLabe, CouplingConfig, generate

labe = GyroLabe(bolmo, atlas_dir="data/atlas")
labe.install()

result = generate(
    model=bolmo,
    tokenizer=tokenizer,
    labe=labe,
    prompt="Language modeling is ",
    max_new_tokens=128,
    temperature=0.7,
    top_k=40,
)
print(result.text)
print(labe.stats())
""")

    print("NOTE: Bolmo uses trust_remote_code=True (custom modeling files).")
    print("NOTE: token_id & 0xFF gives the kernel driving byte for all 512 Bolmo tokens.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Bolmo-1B and/or Bolmo-7B from HuggingFace.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        choices=["1b", "7b", "both"],
        default="both",
        help="Which model(s) to download (default: both).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/models"),
        help="Base output directory (default: data/models).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually downloading.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token (or set HF_TOKEN env var). Not required for Bolmo.",
    )
    args = parser.parse_args()

    print("Bolmo Model Downloader")
    print("======================")

    check_dependencies()

    # Optional: set HF token if provided
    if args.hf_token:
        try:
            from huggingface_hub import login
            login(token=args.hf_token)
            print("Logged in to HuggingFace Hub.")
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: Could not log in with provided token: {e}")

    # Resolve output directory overrides
    targets: dict[str, dict] = {}
    for key, info in MODELS.items():
        if args.model in (key, "both"):
            entry = dict(info)
            # Remap local_dir relative to --out-dir
            entry["local_dir"] = args.out_dir / info["local_dir"].name
            targets[key] = entry

    if not targets:
        print("No models selected.")
        sys.exit(0)

    if args.dry_run:
        print("\nDRY RUN MODE: No files will be downloaded.\n")

    # Download selected models
    results: dict[str, bool] = {}
    for key, info in targets.items():
        success = download_model(
            repo_id=info["repo_id"],
            local_dir=info["local_dir"],
            approx_size_gb=info["approx_size_gb"],
            dry_run=args.dry_run,
        )
        results[key] = success

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    downloaded = []
    for key, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {MODELS[key]['repo_id']:30s}  [{status}]")
        if success and not args.dry_run:
            downloaded.append(key)

    failed = [k for k, v in results.items() if not v]
    if failed:
        print(f"\n{len(failed)} download(s) failed. Check errors above.")
        sys.exit(1)
    else:
        if not args.dry_run:
            print_post_download_usage(downloaded)
        print("\nDone.")


if __name__ == "__main__":
    main()