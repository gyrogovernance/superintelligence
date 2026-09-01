#!/usr/bin/env python3
"""hqvm_runtime_analysis_run.py — Orchestrator for hQVM C medium cost study.

Usage:
  python src/tools/gyroscopic/helpers/hqvm_runtime_analysis_run.py
  python src/tools/gyroscopic/helpers/hqvm_runtime_analysis_run.py --only 1 --verbose
"""
from __future__ import annotations

import argparse
import importlib
import io
import re
import sys
import time
from pathlib import Path

_HELPERS = Path(__file__).resolve().parent
if str(_HELPERS) not in sys.path:
    sys.path.insert(0, str(_HELPERS))

SCRIPTS = {
    "1": "hqvm_runtime_analysis_1.py",
}


def _preflight() -> bool:
    try:
        __import__("numpy")
    except ImportError:
        print("ERROR: numpy is required. Install with:  pip install numpy")
        return False
    return True


def _load_runner(num: str):
    return importlib.import_module(f"hqvm_runtime_analysis_{num}").run


def _write_headlines(worknotes: Path, headlines: dict[str, str]) -> None:
    if not worknotes.is_file() or not headlines:
        return
    text = worknotes.read_text(encoding="utf-8")
    for key, val in headlines.items():
        line = f"{key}: {val}"
        pat = re.compile(rf"(?m)^{re.escape(key)}:.*$")
        if pat.search(text):
            text = pat.sub(line, text, count=1)
        elif "=====\nC SURFACE" in text:
            text = text.replace("=====\nC SURFACE", f"{line}\n\n=====\nC SURFACE", 1)
        else:
            text = text.rstrip() + f"\n{line}\n"
    text = re.sub(
        r"(?m)^# STATUS:.*$",
        "# STATUS: COMPLETE — measured C medium costs filled below",
        text,
        count=1,
    )
    worknotes.write_text(text, encoding="utf-8")


def main() -> None:
    choices = tuple(SCRIPTS) + ("all",)
    parser = argparse.ArgumentParser(description="hQVM runtime analysis runner")
    parser.add_argument("--only", choices=choices, default="all")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not _preflight():
        sys.exit(2)

    from hqvm_runtime_analysis_common import (
        RESULTS_PATH,
        WORKNOTES_PATH,
        ReportState,
        Tee,
        require_native,
        set_verbose,
    )

    set_verbose(args.verbose)
    try:
        require_native()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(2)

    selected = list(SCRIPTS) if args.only == "all" else [args.only]

    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = Tee(old, buf)

    print("=" * 5)
    print("hQVM RUNTIME ANALYSIS (C medium costs)")
    print("=" * 5)

    state = ReportState()
    t_all = time.perf_counter()
    try:
        for num in selected:
            t0 = time.perf_counter()
            _load_runner(num)(state)
            print(
                f"\n[{SCRIPTS[num]} finished in {time.perf_counter() - t0:.1f}s]",
                flush=True,
            )
    finally:
        sys.stdout = old

    passed = sum(1 for _, ok in state.gates if ok)
    failed = sum(1 for _, ok in state.gates if not ok)
    summary = (
        f'\n{"=" * 5}\n'
        f"SUMMARY: {passed} passed, {failed} failed out of {len(state.gates)} checks"
        f"  (total {time.perf_counter() - t_all:.1f}s)\n"
        f'{"=" * 5}\n'
    )
    if failed:
        summary += "".join(f"  FAIL: {label}\n" for label, ok in state.gates if not ok)

    print(summary, end="")
    RESULTS_PATH.write_text(buf.getvalue() + summary, encoding="utf-8")
    print(f"Wrote {RESULTS_PATH}")

    if not failed:
        _write_headlines(WORKNOTES_PATH, state.headlines)
        print(f"Updated {WORKNOTES_PATH}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
