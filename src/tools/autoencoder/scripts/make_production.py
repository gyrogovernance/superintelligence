"""One-shot production regeneration driver.

Trains the four production checkpoints and writes the matching eval,
equivariance, and closed-form full-G reports, plus the verified-dictionary
audit and a single ``production_summary.json`` that points at every artifact
under the package's own data layout. Every report path is recorded with
forward slashes so the published JSONs are portable.

Run from the repository root:

    python -m src.tools.autoencoder.scripts.make_production

The script is intentionally short: it reuses the existing CLI subcommands so
the published reports come from the same code path as any user invocation.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
PACKAGE_ROOT = REPO_ROOT / "src" / "tools" / "autoencoder"
DATA_HOME = PACKAGE_ROOT / "data"
CHECKPOINTS = DATA_HOME / "checkpoints" / "production"
REPORTS = DATA_HOME / "reports"

ETA_DEFAULT = "0.03,0.03,0.03,0.03,0.03,0.03"


def _rel(p: Path) -> str:
    """Return a portable, forward-slash path relative to the repo root."""
    return p.resolve().relative_to(REPO_ROOT).as_posix()


def _run(cmd: list[str]) -> None:
    """Run a subprocess; abort the driver on non-zero exit so a failed
    artifact can never silently slip into the production summary."""
    print(f"$ {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    if proc.returncode != 0:
        print(
            f"command failed (rc={proc.returncode}): {' '.join(cmd)}",
            file=sys.stderr,
        )
        sys.exit(proc.returncode)


def _train(flag: str, *, ladder: str | None, extra: list[str], epochs: int) -> Path:
    """Run ``python -m src.tools.autoencoder.cli train`` for one production
    checkpoint and return the path to the saved ``.pt``."""
    out = CHECKPOINTS / f"{flag}.pt"
    cmd = [
        sys.executable,
        "-m",
        "src.tools.autoencoder.cli",
        "train",
        "--model",
        flag if ladder is None else f"spectral:{ladder}",
        "--output-dir",
        str(CHECKPOINTS),
        "--run-name",
        flag,
        "--epochs",
        str(epochs),
        "--learning-rate",
        "1e-3",
        "--val-fraction",
        "0.15",
        "--patience",
        "5",
    ]
    cmd.extend(extra)
    if flag == "mlp_full":
        # No rate penalty, no spectral rung.
        pass
    elif flag == "k4_full":
        pass
    elif flag == "spectral_full":
        pass
    elif flag == "spectral_bottleneck":
        cmd.extend(["--rate-weight", "1e-4"])
    _run(cmd)
    return out


def _train_denoise() -> Path:
    out = CHECKPOINTS / "spectral_denoise.pt"
    cmd = [
        sys.executable,
        "-m",
        "src.tools.autoencoder.cli",
        "train-denoise",
        "--output-dir",
        str(CHECKPOINTS),
        "--run-name",
        "spectral_denoise",
        "--ladder",
        "shell_radial",
        "--noise-rate",
        ETA_DEFAULT,
        "--epochs",
        "400",
        "--learning-rate",
        "1e-2",
        "--rate-weight",
        "0.0",
        "--report-file",
        str(REPORTS / "spectral_denoise_train.json"),
    ]
    _run(cmd)
    return out


def _evaluate(ckpt: Path, *, suffix: str) -> Path:
    out = REPORTS / f"{suffix}_eval.json"
    _run(
        [
            sys.executable,
            "-m",
            "src.tools.autoencoder.cli",
            "evaluate",
            "--checkpoint",
            str(ckpt),
            "--report-file",
            str(out),
        ]
    )
    return out


def _verify(ckpt: Path, *, suffix: str) -> Path:
    out = REPORTS / f"{suffix}_eq.json"
    _run(
        [
            sys.executable,
            "-m",
            "src.tools.autoencoder.cli",
            "verify-equivariance",
            "--checkpoint",
            str(ckpt),
            "--seed",
            "0",
            "--report-file",
            str(out),
        ]
    )
    return out


def _verify_full_g(ckpt: Path, *, suffix: str) -> Path:
    out = REPORTS / f"{suffix}_fullg.json"
    _run(
        [
            sys.executable,
            "-m",
            "src.tools.autoencoder.cli",
            "verify-full-g-exhaustive",
            "--checkpoint",
            str(ckpt),
            "--report-file",
            str(out),
        ]
    )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="reuse the existing checkpoints; only regenerate the reports",
    )
    args = parser.parse_args(argv)

    REPORTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, str]] = {}

    if args.skip_train:
        checkpoints = {
            "spectral_full": CHECKPOINTS / "spectral_full.pt",
            "spectral_bottleneck": CHECKPOINTS / "spectral_bottleneck.pt",
            "k4_full": CHECKPOINTS / "k4_full.pt",
            "mlp_full": CHECKPOINTS / "mlp_full.pt",
            "spectral_denoise": CHECKPOINTS / "spectral_denoise.pt",
        }
    else:
        checkpoints = {
            "spectral_full": _train("spectral_full", ladder=None, extra=[], epochs=5),
            "spectral_bottleneck": _train(
                "spectral_bottleneck", ladder=None, extra=[], epochs=800
            ),
            "k4_full": _train("k4_full", ladder=None, extra=[], epochs=700),
            "mlp_full": _train("mlp_full", ladder=None, extra=[], epochs=2000),
            "spectral_denoise": _train_denoise(),
        }

    for name, ckpt in checkpoints.items():
        if not ckpt.exists():
            print(f"missing checkpoint: {ckpt}", file=sys.stderr)
            return 2
        entry: dict[str, str] = {
            "checkpoint": _rel(ckpt),
            "eval": _rel(_evaluate(ckpt, suffix=name)),
            "equivariance": _rel(_verify(ckpt, suffix=name)),
        }
        # The closed-form full-G verifier is spectral-only; it inspects
        # ``bottleneck.gain`` (a Walsh-block gain vector) which only the
        # spectral ladder and the denoiser variant carry. K4 and MLP are
        # verified through the sampled equivariance check above.
        if name.startswith("spectral"):
            entry["full_g_closed_form"] = _rel(_verify_full_g(ckpt, suffix=name))
        summary[name] = entry

    closed_form = REPORTS / "exhaustive_full_g_verify.json"
    if closed_form.exists():
        summary["closed_form_certificate"] = {"report": _rel(closed_form)}

    out = REPORTS / "production_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
