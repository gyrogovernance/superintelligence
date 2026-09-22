#!/usr/bin/env python3
"""Run the genomics topology pipeline end to end.

    python -m src.tools.autoencoder.programs.genomics.topology.run
    python src/tools/autoencoder/programs/genomics/topology/run.py

Naming: see common.py. Stages are script_1_atlas through script_6_order.
Each RESULTS_script_N_*.json is the cache (script-hash validated). Use
--force to rebuild everything. Use --from KEY to rebuild from that stage
onward. Use --only KEY to rebuild a single stage when its inputs exist.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

_SCRIPT = Path(__file__).resolve()
_HERE = _SCRIPT.parent
_REPO = _SCRIPT.parent
for _ in range(8):
    _REPO = _REPO.parent
    if (_REPO / "src").is_dir():
        break
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from src.tools.autoencoder.programs.genomics.topology.common import (  # noqa: E402
    RESULT_JSON,
    RESULTS,
)

_STAGES: tuple[tuple[str, str], ...] = (
    ("atlas", "script_1_atlas.py"),
    ("null", "script_2_null.py"),
    ("narrow", "script_3_narrow.py"),
    ("claim", "script_4_claim.py"),
    ("holdout", "script_5_holdout.py"),
    ("order", "script_6_order.py"),
)
_STAGE_KEYS = tuple(key for key, _ in _STAGES)


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(script_name: str, extra: list[str] | None = None) -> int:
    script = _HERE / script_name
    if not script.is_file():
        print(f"missing stage script: {script}", file=sys.stderr)
        return 1
    cmd = [sys.executable, "-u", str(script), *(extra or [])]
    print("running", script_name, flush=True)
    return int(subprocess.call(cmd, cwd=str(_REPO)))


def _cached(key: str, script_name: str) -> bool:
    """True when RESULTS JSON exists and was produced by the current script file."""
    path = RESULT_JSON[key]
    script = _HERE / script_name
    if not path.is_file() or path.stat().st_size <= 0 or not script.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    recorded = str(
        ((payload.get("provenance") or {}).get("script") or {}).get("sha256") or ""
    )
    if not recorded:
        return False
    return recorded == _sha256_file(script)


def _stage_extras(args: argparse.Namespace) -> dict[str, list[str]]:
    seed = str(args.seed)
    permutations = str(args.permutations)
    null_draws = str(args.null_draws)
    device = str(args.device)
    return {
        "atlas": ["--seed", seed, "--permutations", permutations, "--device", device],
        "null": [
            "--seed",
            seed,
            "--permutations",
            permutations,
            "--null-draws",
            null_draws,
        ],
        "narrow": [
            "--seed",
            seed,
            "--permutations",
            permutations,
            "--null-draws",
            null_draws,
            "--device",
            device,
        ],
        "claim": [],
        "holdout": [],
        "order": ["--device", device, "--seed", seed, "--permutations", permutations],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild the membrane-topology atlas, nulls, Narrow scores, claim, holdout, and order stages."
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--null-draws", type=int, default=80)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild every stage even when RESULTS_script_N_*.json already exists.",
    )
    parser.add_argument(
        "--from",
        dest="from_stage",
        choices=_STAGE_KEYS,
        default=None,
        help="Rebuild this stage and every later stage.",
    )
    parser.add_argument(
        "--only",
        choices=_STAGE_KEYS,
        default=None,
        help="Rebuild only this stage.",
    )
    args = parser.parse_args()
    if args.from_stage and args.only:
        print("use only one of --from and --only", file=sys.stderr)
        return 1

    extras = _stage_extras(args)
    start_index = 0
    if args.from_stage:
        start_index = _STAGE_KEYS.index(args.from_stage)
    only_key = args.only

    rebuild_downstream = bool(args.force) or bool(args.from_stage)
    last_payload: dict | None = None
    for index, (key, script_name) in enumerate(_STAGES):
        if only_key is not None and key != only_key:
            continue
        if only_key is None and index < start_index:
            continue
        if only_key is None and not rebuild_downstream and _cached(key, script_name):
            print(
                f"skipping {script_name} (cache hit: {RESULT_JSON[key].name})",
                flush=True,
            )
            continue
        if (
            only_key is None
            and RESULT_JSON[key].is_file()
            and not rebuild_downstream
            and not _cached(key, script_name)
        ):
            print(
                f"cache stale for {script_name} (script changed); rebuilding",
                flush=True,
            )
        code = _run(script_name, extras.get(key, []))
        if code not in (0, 2):
            return code
        if only_key is None:
            rebuild_downstream = True
        if not RESULT_JSON[key].is_file():
            print(f"missing stage output: {RESULT_JSON[key].name}", file=sys.stderr)
            return 1
        last_payload = json.loads(RESULT_JSON[key].read_text(encoding="utf-8"))

    if only_key is None:
        for key, script_name in _STAGES:
            if not RESULT_JSON[key].is_file():
                print(f"missing stage cache: {RESULT_JSON[key].name}", file=sys.stderr)
                return 1
            if not _cached(key, script_name):
                print(
                    f"stage cache out of date: {RESULT_JSON[key].name} "
                    f"(does not match {script_name})",
                    file=sys.stderr,
                )
                return 1

    if last_payload is None and RESULT_JSON["claim"].is_file():
        last_payload = json.loads(RESULT_JSON["claim"].read_text(encoding="utf-8"))
    if last_payload is not None:
        claim = last_payload.get("claim") or last_payload.get("claim_sentence") or ""
        if claim:
            print(claim, flush=True)
        if "passed" in last_payload:
            print(f"passed={last_payload.get('passed')}", flush=True)
    print(f"wrote pipeline artifacts under {_HERE.name}/ and {RESULTS.name}", flush=True)
    return _report_gates(_stage_statuses())


def _report_gates(statuses: dict[str, dict[str, Any]]) -> int:
    """Print each failing stage and its failed gates, keeping exit codes."""
    failing = [
        (key, payload)
        for key, payload in statuses.items()
        if payload.get("status") != "ok" or payload.get("passed") is False
    ]
    if not failing:
        print("gates=all stages ok", flush=True)
        return 0
    print("failing stages:", flush=True)
    exit_code = 0
    for key, payload in failing:
        print(f"  stage={key} passed={payload.get('passed')}", flush=True)
        gates = payload.get("gates") or {}
        if gates:
            for name, value in gates.items():
                print(f"    gate={name} passed={value}", flush=True)
            failed = [name for name, value in gates.items() if not value]
            if failed:
                print(f"    failed_gates={','.join(failed)}", flush=True)
        else:
            print(f"    stage_error={payload.get('error', 'stage returned a failing status')}", flush=True)
        nested = payload.get("ae_gates")
        if isinstance(nested, dict):
            for section, values in nested.items():
                if isinstance(values, dict):
                    failed_nested = [
                        name for name, value in values.items() if not value
                    ]
                    if failed_nested:
                        print(
                            f"    ae_gates={section} failed_gates={','.join(failed_nested)}",
                            flush=True,
                        )
        exit_code = 2
    return exit_code


def _stage_statuses() -> dict[str, dict[str, Any]]:
    """Load every stage cache and its recorded pass state."""
    statuses: dict[str, dict[str, Any]] = {}
    for key, _ in _STAGES:
        path = RESULT_JSON[key]
        if not path.is_file():
            statuses[key] = {"status": "missing", "passed": None}
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            statuses[key] = {"status": "unreadable", "passed": None, "error": str(error)}
            continue
        # Atlas and null publish status=ok without a top-level passed flag.
        if "passed" not in payload and payload.get("status") == "ok":
            payload = {**payload, "passed": True}
        statuses[key] = payload
    return statuses


if __name__ == "__main__":
    raise SystemExit(main())
