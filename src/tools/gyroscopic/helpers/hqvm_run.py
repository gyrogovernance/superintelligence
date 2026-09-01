#!/usr/bin/env python3
"""hqvm_run.py — Standalone medium driver (Runtime Part II product).

Ingests a byte stream into the native cell pool, emits SLCP records at each
word closure, and optionally writes an append-only genealogy log. No llama
chassis is involved; all stepping is native C via runtime.c + kernel.c.

Usage:
  python helpers/hqvm_run.py --bytes "hello"
  python helpers/hqvm_run.py --hex aaab2a2b --seed omega --log /tmp/medium.glog
  python helpers/hqvm_run.py --file prompt.bin
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HELPERS = Path(__file__).resolve().parent
_REPO = _HELPERS.parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.tools.gyroscopic import ops  # noqa: E402

SEED_CHOICES = {
    "rest": ops.RT_SEED_REST,
    "equality_horizon": ops.RT_SEED_EQUALITY_HORIZON,
    "shell": ops.RT_SEED_SHELL,
    "omega": ops.RT_SEED_OMEGA,
}


def load_payload(args: argparse.Namespace) -> bytes:
    if args.file:
        return Path(args.file).read_bytes()
    if args.hex:
        h = args.hex.replace(" ", "").replace("0x", "")
        return bytes.fromhex(h)
    if args.bytes is not None:
        return args.bytes.encode("utf-8")
    if not sys.stdin.isatty():
        return sys.stdin.buffer.read()
    raise SystemExit("provide --bytes, --hex, --file, or pipe stdin")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the native gyroscopic medium")
    ap.add_argument("--bytes", type=str, default=None, help="UTF-8 string to ingest")
    ap.add_argument("--hex", type=str, default=None, help="hex byte stream")
    ap.add_argument("--file", type=str, default=None, help="read bytes from file")
    ap.add_argument("--seed", choices=sorted(SEED_CHOICES), default="rest")
    ap.add_argument("--capacity", type=int, default=64, help="cell pool capacity")
    ap.add_argument("--log", type=str, default="", help="genealogy log path (optional)")
    ap.add_argument("--no-slcp", action="store_true", help="ingest only, no SLCP print")
    ap.add_argument("--json", action="store_true", help="emit SLCP as JSON lines")
    args = ap.parse_args()

    ops.build_native(force=False)
    payload = load_payload(args)
    seed = SEED_CHOICES[args.seed]

    ops.rt_medium_open(args.log or None, seed, max(1, min(args.capacity, 4096)))
    try:
        ops.rt_medium_ingest(payload, emit_slcp=not args.no_slcp)
        snap = ops.rt_medium_cell_snapshot()
        slcp = ops.rt_medium_last_slcp()

        if args.json and slcp:
            print(json.dumps({"slcp": slcp, "cell": snap}, sort_keys=True))
        elif slcp:
            print("=" * 5)
            print(f"rule_hash={ops.rt_rule_hash():#x}")
            print(f"ingested={len(payload)} bytes step={snap['step']} words_closed={snap['step'] // 4}")
            print(f"omega12={snap['omega12']} chi6_key={snap['resonance_key']} omega_sig={snap['omega_sig']}")
            print(f"SLCP shell={slcp['shell']} horizon={slcp['horizon_distance']} ab={slcp['ab_distance']}")
            spec = slcp["spectral64"]
            print(f"spectral64[0:4]={spec[0]:.4f} {spec[1]:.4f} {spec[2]:.4f} {spec[3]:.4f}")
        else:
            print(f"ingested={len(payload)} bytes (no closed word yet) step={snap['step']}")

        if args.log:
            ev, rq = ops.rt_log_stats()
            print(f"genealogy log: events={ev} requests={rq} path={args.log}")
    finally:
        ops.rt_medium_close()

    print("MEDIUM_RUN OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
