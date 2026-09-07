#!/usr/bin/env python3
"""
hQVM CGM Percolation -- full study runner
=======================================

Runs analysis_1 (byte baseline), analysis_2 (canonical-word theory),
analysis_3 (structural completeness), analysis_4 (fiber-complete product cluster theorem),
saving combined stdout/stderr to hqvm_percolation_analysis_results.txt.

Usage:
  python experiments/hqvm_percolation_analysis_run.py
  python experiments/hqvm_percolation_analysis_run.py --only 1
  python experiments/hqvm_percolation_analysis_run.py --only 2
  python experiments/hqvm_percolation_analysis_run.py --only 3
  python experiments/hqvm_percolation_analysis_run.py --only 4
  python experiments/hqvm_percolation_analysis_run.py --only 3 --fast
  python experiments/hqvm_percolation_analysis_run.py --only 3 --sections 1-7,10,13,14 --fast
  python experiments/hqvm_percolation_analysis_run.py -o path/to/out.txt

Interpretation: docs/Findings/Analysis_hQVM_Percolation.md
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_EXPERIMENTS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = _EXPERIMENTS_DIR / "hqvm_percolation_analysis_results.txt"

SCRIPTS: dict[str, str] = {
    "1": "hqvm_percolation_analysis_1.py",
    "2": "hqvm_percolation_analysis_2.py",
    "3": "hqvm_percolation_analysis_3.py",
    "4": "hqvm_percolation_analysis_4.py",
}


def _configure_stdout_utf8() -> None:
    import codecs

    if hasattr(sys.stdout, "buffer"):
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")


def run_script(
    script_name: str,
    timeout_s: float | None,
    extra_args: list[str] | None = None,
) -> tuple[int, str, str, float]:
    path = _EXPERIMENTS_DIR / script_name
    if not path.is_file():
        return 127, "", f"missing file: {path}\n", 0.0

    cmd = [sys.executable, str(path)]
    if extra_args:
        cmd.extend(extra_args)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(_EXPERIMENTS_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        dt = time.perf_counter() - t0
        out = exc.stdout or ""
        err = str(exc.stderr or "") + f"\nTIMEOUT after {timeout_s}s\n"
        return 124, str(out), str(err), dt

    return (
        proc.returncode,
        proc.stdout or "",
        proc.stderr or "",
        time.perf_counter() - t0,
    )


def format_block(
    script_name: str, code: int, stdout: str, stderr: str, dt: float
) -> str:
    lines = [f"######## {script_name} ########", ""]
    if stdout:
        lines.append(stdout.rstrip())
        lines.append("")
    if stderr:
        lines.append("--- stderr ---")
        lines.append(stderr.rstrip())
        lines.append("")
    lines.append(f"exit={code}  duration={dt:.2f}s")
    lines.append("")
    return "\n".join(lines)


def run_all(
    output_path: Path,
    parts: tuple[str, ...],
    timeout_s: float | None,
    script_extra: dict[str, list[str]] | None = None,
) -> int:
    started = datetime.now(timezone.utc).astimezone()
    blocks: list[str] = [
        "hQVM CGM percolation study",
        f"started: {started.isoformat(timespec='seconds')}",
        f"python: {sys.executable}",
        f"parts: {', '.join(parts)}",
        "",
    ]

    worst_code = 0
    total_dt = 0.0
    for key in parts:
        name = SCRIPTS[key]
        print(f"Running {name} ...", flush=True)
        extra = (script_extra or {}).get(name, [])
        code, out, err, dt = run_script(name, timeout_s, extra)
        total_dt += dt
        worst_code = max(worst_code, code)
        blocks.append(format_block(name, code, out, err, dt))
        status = "ok" if code == 0 else f"exit {code}"
        print(f"  {status} ({dt:.1f}s)", flush=True)

    blocks.append(
        f"finished: {datetime.now().astimezone().isoformat(timespec='seconds')}"
    )
    blocks.append(f"total_duration={total_dt:.2f}s")
    blocks.append(f"worst_exit={worst_code}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(blocks), encoding="utf-8")
    print(f"Wrote {output_path}")
    return worst_code


def main() -> None:
    _configure_stdout_utf8()
    parser = argparse.ArgumentParser(
        description="Run hQVM percolation studies; save combined output to a text file.",
    )
    parser.add_argument(
        "--only",
        choices=("1", "2", "3", "4"),
        default=None,
        help="Run a single part instead of the full pipeline",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output file (default: {DEFAULT_OUTPUT.name})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        metavar="SEC",
        help="Per-script timeout in seconds (default: none)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Pass --fast to analysis_3 (skip expensive sections)",
    )
    parser.add_argument(
        "--sections",
        type=str,
        default=None,
        help="Pass --sections to analysis_3 when --only 3 is used",
    )
    args = parser.parse_args()

    parts = (args.only,) if args.only else ("1", "2", "3", "4")
    script_extra: dict[str, list[str]] = {}
    analysis3_args: list[str] = []
    if args.fast:
        analysis3_args.append("--fast")
    if args.sections:
        analysis3_args.extend(["--sections", args.sections])
    if analysis3_args:
        script_extra["hqvm_percolation_analysis_3.py"] = analysis3_args
    print("hQVM CGM Percolation -- runner")
    print("=" * 5)
    print(f"  Output: {args.output.resolve()}")
    raise SystemExit(run_all(args.output.resolve(), parts, args.timeout, script_extra))


if __name__ == "__main__":
    main()
