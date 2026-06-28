#!/usr/bin/env python3
"""
Run all executable CGM gravity scripts and save combined stdout/stderr to a text file.

Scripts (in run order): analysis_3, 2, 1, 4, 5, 6, 7, 8, 9, 10.
Skips hqvm_gravity_common.py (library only).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXPERIMENTS = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hqvm_gravity_common import configure_stdout_utf8

DEFAULT_OUTPUT = _EXPERIMENTS / "hqvm_gravity_analysis.txt"

GRAVITY_SCRIPTS: tuple[str, ...] = (
    "hqvm_gravity_analysis_3.py",
    "hqvm_gravity_analysis_2.py",
    "hqvm_gravity_analysis_1.py",
    "hqvm_gravity_analysis_4.py",
    "hqvm_gravity_analysis_5.py",
    "hqvm_gravity_analysis_6.py",
    "hqvm_gravity_analysis_7.py",
    "hqvm_gravity_analysis_8.py",
    "hqvm_gravity_analysis_9.py",
    "hqvm_gravity_analysis_10.py",
)


def run_script(script_name: str, timeout_s: float | None) -> tuple[int, str, str, float]:
    path = _EXPERIMENTS / script_name
    if not path.is_file():
        return 127, "", f"missing file: {path}\n", 0.0

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(_EXPERIMENTS),
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

    return proc.returncode, proc.stdout or "", proc.stderr or "", time.perf_counter() - t0


def format_block(script_name: str, code: int, stdout: str, stderr: str, dt: float) -> str:
    header = f"######## {script_name} ########"
    lines = [header, ""]
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


def run_all(output_path: Path, timeout_s: float | None) -> int:
    started = datetime.now(timezone.utc).astimezone()
    blocks: list[str] = [
        "CGM gravity script runner",
        f"started: {started.isoformat(timespec='seconds')}",
        f"python: {sys.executable}",
        f"scripts: {len(GRAVITY_SCRIPTS)}",
        "",
    ]

    worst_code = 0
    total_dt = 0.0
    for name in GRAVITY_SCRIPTS:
        print(f"Running {name} ...", flush=True)
        code, out, err, dt = run_script(name, timeout_s)
        total_dt += dt
        worst_code = max(worst_code, code)
        blocks.append(format_block(name, code, out, err, dt))
        status = "ok" if code == 0 else f"exit {code}"
        print(f"  {status} ({dt:.1f}s)", flush=True)

    blocks.append(f"finished: {datetime.now().astimezone().isoformat(timespec='seconds')}")
    blocks.append(f"total_duration={total_dt:.2f}s")
    blocks.append(f"worst_exit={worst_code}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(blocks), encoding="utf-8")
    print(f"Wrote {output_path}")
    return worst_code


def main() -> None:
    configure_stdout_utf8()
    parser = argparse.ArgumentParser(description="Run all CGM gravity scripts; save output.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output text file (default: {DEFAULT_OUTPUT.name})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        metavar="SEC",
        help="Per-script timeout in seconds (default: none)",
    )
    args = parser.parse_args()
    raise SystemExit(run_all(args.output.resolve(), args.timeout))


if __name__ == "__main__":
    main()
