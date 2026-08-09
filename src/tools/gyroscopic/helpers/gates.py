#!/usr/bin/env python3
"""
Canonical gyroscopic acceptance gates (NavPad §7). One module; no historical gates.

  python -m src.tools.gyroscopic.helpers.gates ledger
  python -m src.tools.gyroscopic.helpers.gates kv [--ppl] [--full]
  python -m src.tools.gyroscopic.helpers.gates codecs [--smoke-only] [--full]
  python -m src.tools.gyroscopic.helpers.gates causal
  python -m src.tools.gyroscopic.helpers.gates forward-probe

Prints measurements and PASS/FAIL only. Site probes are not product-mode certificates.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.tools.gyroscopic.config import (  # noqa: E402
    get_gyroscopic_llm_config,
    production_gyroscopic_env,
    resolve_llama_perplexity_path,
)
from src.tools.gyroscopic.ledger import default_ledger_path  # noqa: E402
from src.tools.gyroscopic.loader import run_llama_cli, run_llama_perplexity  # noqa: E402

CORPUS_FULL = _REPO_ROOT / "data" / "eval" / "ppl_corpus.txt"
CORPUS_TINY = _REPO_ROOT / "data" / "eval" / "ppl_tiny.txt"
SIDECAR = default_ledger_path()
MAX_RATIO = 1.05
PARIS_PROMPT = "The capital of France is"

LEDGER_PROMPTS = [
    ("paris", "The capital of France is", "Paris"),
    ("sun", "The Sun is a", None),
    ("math", "2 + 2 equals", None),
    ("water", "Water freezes at", None),
    ("dna", "DNA stands for", None),
]
LEDGER_ALLOW = (
    "attn_q.weight,attn_k.weight,attn_v.weight,attn_output.weight,"
    "ffn_gate.weight,ffn_up.weight,ffn_down.weight"
)
LEDGER_N_PREDICT = 16

N_LAYER = 36
LONG_FILLER_TOKENS = 2200


def _pass(ok: object) -> str:
    return "PASS" if ok else "FAIL"


def _clean_env() -> dict[str, str]:
    env = dict(os.environ)
    for k in list(env):
        if k.startswith(("GYRO_", "GYROSCOPIC_", "GGML_GYROSCOPIC")):
            env.pop(k, None)
    return env


def _gen(stdout: str, prompt: str) -> str:
    m = re.search(rf"> {re.escape(prompt)}\n(.*?)\[ Prompt:", stdout or "", re.S)
    if m:
        return m.group(1).strip()
    lines = [ln.strip() for ln in (stdout or "").splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _ensure_tiny(path: Path, min_bytes: int = 8000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    base = (
        "The capital of France is Paris. "
        "Language models estimate next-token probability. "
        "Perplexity measures predictive uncertainty under a fixed corpus. "
    )
    if path.is_file() and path.stat().st_size >= min_bytes:
        return path
    text = ""
    if CORPUS_FULL.is_file():
        text = CORPUS_FULL.read_text(encoding="utf-8", errors="ignore")[: min_bytes * 2]
    if len(text) < min_bytes:
        text = (base * ((min_bytes // len(base)) + 2))[:min_bytes]
    else:
        text = text[:min_bytes]
    path.write_text(text, encoding="utf-8")
    return path


def _ppl(cfg, env: dict[str, str], label: str, corpus_path: str, ctx: int, timeout_sec: int) -> float | None:
    t0 = time.perf_counter()
    try:
        r = run_llama_perplexity(
            cfg,
            corpus_path=corpus_path,
            env=env,
            timeout_sec=timeout_sec,
            extra_args=["-c", str(ctx)],
        )
    except subprocess.TimeoutExpired:
        sec = round(time.perf_counter() - t0, 2)
        print(f"  {label}_ppl=TIMEOUT sec={sec}")
        return None
    sec = round(time.perf_counter() - t0, 2)
    ppl = r.get("ppl") if isinstance(r, dict) else None
    print(f"  {label}_ppl={ppl} sec={sec}")
    return float(ppl) if ppl is not None else None


def _count_hits(stderr: str) -> tuple[int, int]:
    hits = 0
    n_lines = 0
    for ln in stderr.splitlines():
        if "displaced" in ln:
            n_lines += 1
            m = re.search(r"hits=(\d+)", ln)
            if m:
                hits = max(hits, int(m.group(1)))
    return hits, n_lines


def _arc3_env() -> dict[str, str]:
    env = _clean_env()
    env["GYRO_KV_KQ8"] = "1"
    env["GYRO_KV_V"] = "1"
    env["GYRO_HOLONOMIC_ATTN"] = "1"
    return env


def _codecs_base_env() -> dict[str, str]:
    env = _clean_env()
    env.update(production_gyroscopic_env(holonomic_kv=True))
    if SIDECAR.is_file():
        env["GYRO_LEDGER_PATH"] = str(SIDECAR)
    return env


def _codecs_variant_env(extra: dict[str, str]) -> dict[str, str]:
    env = _codecs_base_env()
    env.update(extra)
    return env


def _parse_counters(combined: str) -> dict[str, int | None]:
    hol = re.search(r"holonomic_score_calls=(\d+)", combined)
    stock = re.search(r"stock_score_calls=(\d+)", combined)
    vq8 = re.search(r"v_q8_calls=(\d+)", combined)
    rope_c = re.search(r"rope_codec_calls=(\d+)", combined)
    rope_s = re.search(r"rope_stock_calls=(\d+)", combined)
    return {
        "holonomic": int(hol.group(1)) if hol else None,
        "stock": int(stock.group(1)) if stock else None,
        "v_q8": int(vq8.group(1)) if vq8 else None,
        "rope_codec": int(rope_c.group(1)) if rope_c else None,
        "rope_stock": int(rope_s.group(1)) if rope_s else None,
    }


def _codecs_smoke(cfg, env: dict[str, str], label: str, timeout_sec: int = 300) -> tuple[bool, dict[str, int | None], str]:
    try:
        r = run_llama_cli(cfg, prompt=PARIS_PROMPT, n_predict=8, env=env, timeout_sec=timeout_sec)
    except subprocess.TimeoutExpired:
        print(f"  {label}_rc=TIMEOUT")
        return False, {}, ""
    gen = _gen(r.stdout or "", PARIS_PROMPT)
    combined = (r.stdout or "") + "\n" + (r.stderr or "")
    ctr = _parse_counters(combined)
    paris_ok = "Paris" in gen
    stock0 = ctr["stock"] == 0
    hol_pos = ctr["holonomic"] is not None and ctr["holonomic"] > 0
    v_pos = ctr["v_q8"] is not None and ctr["v_q8"] > 0
    ok = (r.returncode == 0) and paris_ok and stock0 and hol_pos and v_pos
    print(f"  {label}_rc={r.returncode} gen={gen!r}")
    print(
        f"  {label}_holonomic={ctr['holonomic']} stock={ctr['stock']} "
        f"v_q8={ctr['v_q8']} rope_codec={ctr['rope_codec']} rope_stock={ctr['rope_stock']}"
    )
    print(f"  {label}_paris  {_pass(paris_ok)}")
    print(f"  {label}_stock_score_calls=0  {_pass(stock0)}")
    print(f"  {label}_holonomic>0  {_pass(hol_pos)}")
    print(f"  {label}_v_q8>0  {_pass(v_pos)}")
    print(f"  {label}_smoke  {_pass(ok)}")
    return ok, ctr, combined


def _incomplete_forward_env(*, perturb: bool) -> dict[str, str]:
    env = _clean_env()
    env.update(production_gyroscopic_env(incomplete_forward=True))
    if SIDECAR.is_file():
        env["GYRO_LEDGER_PATH"] = str(SIDECAR)
    if perturb:
        env["GYRO_CGM_LIFT_PERTURB"] = "1"
    return env


def _causal_gen(stdout: str, prompt: str) -> str:
    m = re.search(rf"> {re.escape(prompt)}\n(.*?)\[ Prompt:", stdout or "", re.S)
    if m:
        return m.group(1).strip()
    lines = [ln.strip() for ln in (stdout or "").splitlines() if ln.strip()]
    keep = [
        ln for ln in lines
        if not ln.startswith((
            "load_", "llama_", "ggml_", "system_", "main:", "build:", "print_",
            "sampling:", "generate:", "slot ", "common_", "srv ", "clip ",
            "[hqvm", "=====",
        ))
    ]
    return " ".join(keep[-5:]) if keep else ""


def cmd_ledger(_args: argparse.Namespace) -> int:
    print("hQVM GATE LEDGER: prompt set + optional PPL (full Q1_0 displace)")
    print("=" * 5)
    cfg = get_gyroscopic_llm_config()
    if not SIDECAR.is_file():
        print(f"  sidecar missing  {_pass(False)}")
        return 1
    print(f"  sidecar={SIDECAR.name}  {_pass(True)}")
    print(f"  prompts={len(LEDGER_PROMPTS)}  n_predict={LEDGER_N_PREDICT}")

    ledger_env = _clean_env()
    ledger_env.update(production_gyroscopic_env())
    ledger_env["GYRO_LEDGER_ALLOW"] = LEDGER_ALLOW

    print("\n1. PROMPT SET (stock vs ledger)")
    print("=" * 5)
    n_ok = 0
    n_anchor = 0
    n_anchor_need = 0
    for name, prompt, must in LEDGER_PROMPTS:
        env_a = _clean_env()
        t0 = time.perf_counter()
        ra = run_llama_cli(cfg, prompt=prompt, n_predict=LEDGER_N_PREDICT, env=env_a, timeout_sec=400)
        sa = round(time.perf_counter() - t0, 2)
        ta = _gen(ra.stdout, prompt)

        t0 = time.perf_counter()
        rc = run_llama_cli(cfg, prompt=prompt, n_predict=LEDGER_N_PREDICT, env=ledger_env, timeout_sec=400)
        sc = round(time.perf_counter() - t0, 2)
        tc = _gen(rc.stdout, prompt)
        hits, n_disp = _count_hits(rc.stderr)

        ok_run = ra.returncode == 0 and rc.returncode == 0
        if must:
            n_anchor_need += 1
            ok_anchor = must in tc
            if ok_anchor:
                n_anchor += 1
        else:
            ok_anchor = True
        ok = ok_run and ok_anchor and (hits > 0 or n_disp > 0)
        if ok:
            n_ok += 1
        print(
            f"  {name}: stock_rc={ra.returncode} sec={sa} "
            f"ledger_rc={rc.returncode} sec={sc} hits={hits} disp_lines={n_disp} "
            f"anchor={_pass(ok_anchor if must else True)}"
        )
        print(f"    stock={ta!r}")
        print(f"    ledger={tc!r}")

    print(f"  prompt rows ok  {n_ok}/{len(LEDGER_PROMPTS)}  {_pass(n_ok == len(LEDGER_PROMPTS))}")
    if n_anchor_need:
        print(f"  anchor prompts  {n_anchor}/{n_anchor_need}  {_pass(n_anchor == n_anchor_need)}")

    print("\n2. PERPLEXITY (optional)")
    print("=" * 5)
    ppl_ok = True
    try:
        ppl_exe = resolve_llama_perplexity_path(cfg)
        print(f"  perplexity_exe={ppl_exe.name}  {_pass(True)}")
    except FileNotFoundError:
        print("  perplexity_exe missing  SKIP")
        ppl_exe = None
        ppl_ok = True

    if ppl_exe is not None and CORPUS_FULL.is_file():
        cfg_ppl = replace(cfg, n_ctx=512)
        env_a = _clean_env()
        t0 = time.perf_counter()
        ra = run_llama_perplexity(cfg_ppl, corpus_path=str(CORPUS_FULL), env=env_a, timeout_sec=1800)
        sa = round(time.perf_counter() - t0, 2)
        t0 = time.perf_counter()
        rc = run_llama_perplexity(
            cfg_ppl, corpus_path=str(CORPUS_FULL), env=ledger_env, timeout_sec=1800
        )
        sc = round(time.perf_counter() - t0, 2)
        ppl_stock = ra.get("ppl")
        ppl_led = rc.get("ppl")
        print("  n_ctx=512")
        print(f"  stock_ppl={ppl_stock} sec={sa} rc={ra.get('rc')}")
        print(f"  ledger_ppl={ppl_led} sec={sc} rc={rc.get('rc')}")
        if ppl_stock is None or ppl_led is None:
            err = (rc.get("stderr") or ra.get("stderr") or "")[-400:]
            print(f"  ppl parse  {_pass(False)}")
            if err:
                print(f"  stderr_tail={err!r}")
            ppl_ok = False
        else:
            ratio = float(ppl_led) / float(ppl_stock) if float(ppl_stock) > 0 else float("inf")
            print(f"  ratio ledger/stock={ratio:.4f}")
            ppl_ok = int(ra.get("rc", 1)) == 0 and int(rc.get("rc", 1)) == 0 and ratio < 2.0
            print(f"  ppl ratio < 2.0  {_pass(ppl_ok)}")
    elif ppl_exe is not None:
        print(f"  corpus missing {CORPUS_FULL}  SKIP")

    print("\n3. CHECKS")
    print("=" * 5)
    print(f"  all prompts ok  {_pass(n_ok == len(LEDGER_PROMPTS))}")
    print(f"  ppl (or skip)  {_pass(ppl_ok)}")
    overall = n_ok == len(LEDGER_PROMPTS) and ppl_ok
    print(f"  GATE_LEDGER  {_pass(overall)}")
    print("\nDONE")
    print("=" * 5)
    return 0 if overall else 1


def _kv_k(args: argparse.Namespace) -> int:
    print("hQVM GATE KV_K: Arc 2B-2 displaced Q8_0 K cache vs stock")
    print("=" * 5)
    print("  K_cache=Q8_0 reclaim=no-alloc")
    print(f"  mode={'PPL' if args.ppl else 'paris-only (default)'}")

    cfg = get_gyroscopic_llm_config()
    cfg_p = replace(cfg, n_ctx=max(256, int(args.ctx)))

    env_2b2 = _clean_env()
    env_2b2["GYRO_KV_KQ8"] = "1"
    env_2b2["GYRO_HOLONOMIC_ATTN"] = "1"

    print("\n1. PARIS SMOKE (2B-2)")
    print("=" * 5)
    r = run_llama_cli(cfg, prompt=PARIS_PROMPT, n_predict=8, env=env_2b2, timeout_sec=180)
    gen = _gen(r.stdout or "", PARIS_PROMPT)
    combined = (r.stdout or "") + "\n" + (r.stderr or "")
    print(f"  rc={r.returncode}")
    print(f"  gen={gen!r}")
    paris_ok = "Paris" in gen
    print(f"  Paris  {_pass(paris_ok)}")

    q8_ok = "K_cache=Q8_0" in combined
    none_f16 = "allocated_F16_K=NONE" in combined
    bytes_ok = "K_B_per_tok_layer=1088" in combined
    hol_on = ("holonomic_score_calls=" in combined) and ("stock_score_calls=0" in combined)
    print(f"  log_K_cache_Q8       {_pass(q8_ok)}")
    print(f"  log_no_F16_K         {_pass(none_f16)}")
    print(f"  log_K_bytes_1088     {_pass(bytes_ok)}")
    print(f"  holonomic_only       {_pass(hol_on)}")

    disp_ok = paris_ok and q8_ok and none_f16 and hol_on

    if not args.ppl:
        print(f"\n  GATE_KV_K  {_pass(disp_ok)}")
        print("  note=PPL deferred; run with --ppl when needed")
        return 0 if disp_ok else 1

    if args.full:
        corpus_path = CORPUS_FULL
        print(f"\n2. PPL TABLE corpus=FULL path={corpus_path}")
    else:
        corpus_path = _ensure_tiny(CORPUS_TINY, min_bytes=8000)
        print(f"\n2. PPL TABLE corpus=TINY path={corpus_path} bytes={corpus_path.stat().st_size}")
    print("=" * 5)

    try:
        ppl_exe = resolve_llama_perplexity_path(cfg)
        print(f"  perplexity_exe={ppl_exe.name}  {_pass(True)}")
    except FileNotFoundError:
        print("  perplexity_exe missing  FAIL")
        return 1

    ppl_stock = _ppl(cfg_p, _clean_env(), "stock", str(corpus_path), int(args.ctx), int(args.timeout))
    ppl_2b2 = _ppl(cfg_p, env_2b2, "2b2_q8_cache", str(corpus_path), int(args.ctx), int(args.timeout))

    env_pert = dict(env_2b2)
    env_pert["GYRO_COORD_PERTURB"] = "zero_kq8"
    ppl_pert = _ppl(cfg_p, env_pert, "2b2_zero_kq8", str(corpus_path), int(args.ctx), int(args.timeout))

    ratio_ok = False
    if ppl_stock and ppl_2b2 and float(ppl_stock) > 0:
        ratio = float(ppl_2b2) / float(ppl_stock)
        print(f"  2b2_ratio={ratio:.4f}")
        ratio_ok = ratio <= MAX_RATIO
    print(f"  PPL ratio <= {MAX_RATIO}  {_pass(ratio_ok)}")

    pert_ok = False
    if ppl_stock and ppl_pert and float(ppl_stock) > 0:
        pert_shift = abs(float(ppl_pert) / float(ppl_stock) - 1.0)
        print(f"  zero_kq8_rel_shift={pert_shift:.4f}")
        pert_ok = pert_shift > 0.05
    print(f"  zero_kq8 blows PPL  {_pass(pert_ok)}")

    overall = disp_ok and ratio_ok and pert_ok
    print(f"\n  GATE_KV_K  {_pass(overall)}")
    return 0 if overall else 1


def _kv_v(args: argparse.Namespace) -> int:
    print("\nhQVM GATE KV_V: Arc 3 displaced Q8_0 V cache + hqvm Attn@V")
    print("=" * 5)
    print(f"  mode={'PPL' if args.ppl else 'smoke/audit/perturb/long-context (default)'}")

    cfg = get_gyroscopic_llm_config()
    cfg_p = replace(cfg, n_ctx=max(256, int(args.ctx)))
    env_arc3 = _arc3_env()

    print("\n1. PARIS SMOKE + AUDIT (Gate G/F)")
    print("=" * 5)
    r = run_llama_cli(cfg, prompt=PARIS_PROMPT, n_predict=8, env=env_arc3, timeout_sec=180)
    gen = _gen(r.stdout or "", PARIS_PROMPT)
    combined = (r.stdout or "") + "\n" + (r.stderr or "")
    print(f"  rc={r.returncode}")
    print(f"  gen={gen!r}")
    paris_ok = "Paris" in gen
    print(f"  Paris  {_pass(paris_ok)}")

    v_q8 = combined.count("V_type=q8_0")
    f16v_none = combined.count("F16_V=NOT_ALLOCATED")
    k_q8 = combined.count("K_type=q8_0")
    expected = 2 * N_LAYER
    v_q8_ok = v_q8 == expected
    f16v_ok = f16v_none == expected
    k_ok = k_q8 == expected
    hol_on = ("v_q8_calls=" in combined) and ("stock_score_calls=0" in combined)
    v_calls = re.search(r"v_q8_calls=(\d+)", combined)
    v_calls_pos = v_calls is not None and int(v_calls.group(1)) > 0
    print(f"  V_type=q8_0 x{expected}  ({v_q8})  {_pass(v_q8_ok)}")
    print(f"  F16_V=NOT_ALLOCATED x{expected}  ({f16v_none})  {_pass(f16v_ok)}")
    print(f"  K_type=q8_0 x{expected}  ({k_q8})  {_pass(k_ok)}")
    print(f"  stock_score_calls=0  {_pass(hol_on)}")
    print(f"  v_q8_calls>0  ({v_calls.group(1) if v_calls else 0})  {_pass(v_calls_pos)}")

    disp_ok = paris_ok and v_q8_ok and f16v_ok and k_ok and hol_on and v_calls_pos

    print("\n2. GATE H (V perturb collapse)")
    print("=" * 5)
    env_pert = dict(env_arc3)
    env_pert["GYRO_V_PERTURB"] = "1"
    rp = run_llama_cli(cfg, prompt=PARIS_PROMPT, n_predict=8, env=env_pert, timeout_sec=180)
    gen_p = _gen(rp.stdout or "", PARIS_PROMPT)
    print(f"  perturbed gen={gen_p!r}")
    pert_ok = "Paris" not in gen_p
    print(f"  generation diverges from Paris  {_pass(pert_ok)}")

    print("\n3. GATE I (long-context Nk>2048, no abort, stock=0)")
    print("=" * 5)
    filler = ("alpha beta gamma delta epsilon zeta eta theta " * (LONG_FILLER_TOKENS // 8 + 2)).strip()
    q = "Question: the first city mentioned was Paris. What is the capital of Italy?"
    long_prompt = filler + " " + q
    cfg_l = replace(cfg, n_ctx=3072)
    rl = run_llama_cli(cfg_l, prompt=long_prompt, n_predict=8, env=env_arc3, timeout_sec=300)
    comb_l = (rl.stdout or "") + "\n" + (rl.stderr or "")
    aborted = "GGML_ABORT" in comb_l or "refusing" in comb_l
    stock_zero = "stock_score_calls=0" in comb_l
    vq8_l = re.search(r"v_q8_calls=(\d+)", comb_l)
    vq8_l_pos = vq8_l is not None and int(vq8_l.group(1)) > 0
    print(f"  rc={rl.returncode}")
    print(f"  no_abort  {_pass(not aborted)}")
    print(f"  stock_score_calls=0  {_pass(stock_zero)}")
    print(f"  v_q8_calls>0  ({vq8_l.group(1) if vq8_l else 0})  {_pass(vq8_l_pos)}")
    long_ok = (rl.returncode == 0) and (not aborted) and stock_zero and vq8_l_pos
    print(f"  Gate I  {_pass(long_ok)}")

    overall = disp_ok and pert_ok and long_ok

    if args.ppl:
        if args.full:
            corpus_path = CORPUS_FULL
            print(f"\n4. GATE J PPL corpus=FULL path={corpus_path}")
        else:
            corpus_path = _ensure_tiny(CORPUS_TINY, min_bytes=8000)
            print(f"\n4. GATE J PPL corpus=TINY path={corpus_path} bytes={corpus_path.stat().st_size}")
        print("=" * 5)
        try:
            ppl_exe = resolve_llama_perplexity_path(cfg)
            print(f"  perplexity_exe={ppl_exe.name}  {_pass(True)}")
        except FileNotFoundError:
            print("  perplexity_exe missing  FAIL")
            return 1

        ppl_stock = _ppl(cfg_p, _clean_env(), "stock", str(corpus_path), int(args.ctx), int(args.timeout))
        ppl_arc3 = _ppl(cfg_p, env_arc3, "arc3_q8v", str(corpus_path), int(args.ctx), int(args.timeout))
        env_pert2 = dict(env_arc3)
        env_pert2["GYRO_V_PERTURB"] = "1"
        ppl_vp = _ppl(cfg_p, env_pert2, "arc3_v_perturb", str(corpus_path), int(args.ctx), int(args.timeout))

        ratio_ok = False
        if ppl_stock and ppl_arc3 and float(ppl_stock) > 0:
            ratio = float(ppl_arc3) / float(ppl_stock)
            print(f"  arc3_ratio={ratio:.4f}")
            ratio_ok = ratio <= MAX_RATIO
        print(f"  PPL ratio <= {MAX_RATIO}  {_pass(ratio_ok)}")

        pert_shift_ok = False
        if ppl_stock and ppl_vp and float(ppl_stock) > 0:
            shift = abs(float(ppl_vp) / float(ppl_stock) - 1.0)
            print(f"  v_perturb_rel_shift={shift:.4f}")
            pert_shift_ok = shift > 0.05
        print(f"  v_perturb blows PPL  {_pass(pert_shift_ok)}")
        overall = overall and ratio_ok and pert_shift_ok

    print(f"\n  GATE_KV_V  {_pass(overall)}")
    return 0 if overall else 1


def cmd_kv(args: argparse.Namespace) -> int:
    print("hQVM GATE KV: Arc 2 K + Arc 3 V")
    print("=" * 5)
    rc_k = _kv_k(args)
    rc_v = _kv_v(args)
    overall = rc_k == 0 and rc_v == 0
    print("\nDONE")
    print("=" * 5)
    print(f"  GATE_KV  {_pass(overall)}")
    return 0 if overall else 1


def cmd_codecs(args: argparse.Namespace) -> int:
    print("hQVM GATE CODECS: certify live A/R/S vs ledger+KV base")
    print("=" * 5)

    if not SIDECAR.is_file():
        print(f"  sidecar missing  {_pass(False)}")
        return 1

    cfg = get_gyroscopic_llm_config()
    cfg_smoke = replace(cfg, n_ctx=512)
    cfg_p = replace(cfg, n_ctx=max(256, int(args.ctx)))

    variants: list[tuple[str, dict[str, str]]] = [
        ("Base", {}),
        ("A", {"GYRO_APERTURE_SOFTMAX": "1"}),
        ("R", {"GYRO_ROPE_CODEC": "1"}),
        ("S", {"GYRO_SILU_CODEC": "1"}),
        ("ARS", {
            "GYRO_APERTURE_SOFTMAX": "1",
            "GYRO_ROPE_CODEC": "1",
            "GYRO_SILU_CODEC": "1",
        }),
    ]

    print("\n1. PARIS SMOKE + COUNTERS")
    print("=" * 5)
    smoke_ok: dict[str, bool] = {}
    for name, extra in variants:
        print(f"\n  --- {name} ---")
        timeout = 600 if name in ("S", "ARS") else 300
        ok, ctr, combined = _codecs_smoke(cfg_smoke, _codecs_variant_env(extra), name, timeout_sec=timeout)
        if name in ("R", "ARS"):
            rc = ctr.get("rope_codec")
            rope_ctr = rc is not None and rc > 0
            rope_shadow = "[hqvm-rope-codec]" in combined
            print(f"  {name}_rope_codec_calls>0  {_pass(rope_ctr)}")
            print(f"  {name}_rope_shadow_log  {_pass(rope_shadow)}")
        smoke_ok[name] = ok

    if args.smoke_only:
        overall = all(smoke_ok.values())
        print("\nDONE")
        print("=" * 5)
        print(f"  GATE_CODECS (smoke)  {_pass(overall)}")
        return 0 if overall else 1

    print("\n2. PPL vs BASE")
    print("=" * 5)
    try:
        ppl_exe = resolve_llama_perplexity_path(cfg)
        print(f"  perplexity_exe={ppl_exe.name}  {_pass(True)}")
    except FileNotFoundError:
        print("  perplexity_exe missing  FAIL")
        return 1

    if args.full:
        corpus = CORPUS_FULL
        if not corpus.is_file():
            print(f"  corpus missing {corpus}  FAIL")
            return 1
        print(f"  corpus=FULL path={corpus}")
    else:
        corpus = _ensure_tiny(CORPUS_TINY)
        print(f"  corpus=TINY path={corpus} bytes={corpus.stat().st_size}")

    ppls: dict[str, float | None] = {}
    for name, extra in variants:
        ppls[name] = _ppl(cfg_p, _codecs_variant_env(extra), name, str(corpus), int(args.ctx), int(args.timeout))

    base = ppls.get("Base")
    ratio_ok: dict[str, bool] = {}
    print("\n3. RATIOS")
    print("=" * 5)
    if base is None or base <= 0:
        print(f"  base_ppl usable  {_pass(False)}")
        overall = False
    else:
        print(f"  base_ppl={base}")
        overall = True
        for name, _extra in variants:
            if name == "Base":
                ratio_ok[name] = True
                continue
            p = ppls.get(name)
            if p is None:
                print(f"  {name}_ratio=None  {_pass(False)}")
                ratio_ok[name] = False
                overall = False
                continue
            ratio = float(p) / float(base)
            ok = ratio <= MAX_RATIO
            ratio_ok[name] = ok
            print(f"  {name}_ratio={ratio:.4f}  {_pass(ok)}")
            if not ok:
                overall = False
        overall = overall and all(smoke_ok.values()) and all(ratio_ok.values())

    print("\n4. CHECKS")
    print("=" * 5)
    for name, _ in variants:
        print(f"  {name}_smoke  {_pass(smoke_ok.get(name, False))}")
        if name != "Base":
            print(f"  {name}_ppl_ratio<={MAX_RATIO}  {_pass(ratio_ok.get(name, False))}")
    print(f"  GATE_CODECS  {_pass(overall)}")
    print("\nDONE")
    print("=" * 5)
    return 0 if overall else 1


def cmd_causal(_args: argparse.Namespace) -> int:
    print("hQVM GATE CAUSAL: lift perturb changes decode")
    print("=" * 5)

    cfg = replace(get_gyroscopic_llm_config(), n_ctx=512)
    extra = ["--temp", "0", "--top-k", "1", "--seed", "1"]

    print("\n1. HYBRID (no perturb)")
    print("=" * 5)
    try:
        ra = run_llama_cli(
            cfg, prompt=PARIS_PROMPT, n_predict=10,
            env=_incomplete_forward_env(perturb=False), timeout_sec=600, extra_args=extra,
        )
    except Exception as e:
        print(f"  run_A_error={e!r}")
        print(f"  GATE_CAUSAL  {_pass(False)}")
        return 1
    gen_a = _causal_gen(ra.stdout or "", PARIS_PROMPT)
    comb_a = (ra.stdout or "") + "\n" + (ra.stderr or "")
    print(f"  rc={ra.returncode}")
    print(f"  gen_A={gen_a!r}")
    hits = re.search(r"\[hqvm-residual-hybrid\] hits=(\d+)", comb_a)
    stock = re.search(r"stock_score_calls=(\d+)", comb_a)
    print(f"  residual_hits={hits.group(1) if hits else None}")
    print(f"  stock_score_calls={stock.group(1) if stock else None}")
    print(f"  stock_score_calls=0  {_pass(stock is not None and int(stock.group(1)) == 0)}")
    if hits:
        h = int(hits.group(1))
        print(f"  residual_hits_near_72  {_pass(40 <= h <= 200)}")

    print("\n2. HYBRID + PERTURB")
    print("=" * 5)
    try:
        rb = run_llama_cli(
            cfg, prompt=PARIS_PROMPT, n_predict=10,
            env=_incomplete_forward_env(perturb=True), timeout_sec=600, extra_args=extra,
        )
    except Exception as e:
        print(f"  run_B_error={e!r}")
        print(f"  GATE_CAUSAL  {_pass(False)}")
        return 1
    gen_b = _causal_gen(rb.stdout or "", PARIS_PROMPT)
    print(f"  rc={rb.returncode}")
    print(f"  gen_B={gen_b!r}")

    differ = (gen_a != gen_b) and bool(gen_a) and bool(gen_b)
    print("\n3. CHECKS")
    print("=" * 5)
    print(f"  both_rc=0  {_pass(ra.returncode == 0 and rb.returncode == 0)}")
    print(f"  gen_nonempty  {_pass(bool(gen_a) and bool(gen_b))}")
    print(f"  token_seq_differs  {_pass(differ)}")
    overall = (ra.returncode == 0 and rb.returncode == 0 and differ)
    print(f"  GATE_CAUSAL  {_pass(overall)}")
    print("\nDONE")
    print("=" * 5)
    return 0 if overall else 1


def cmd_forward_probe(args: argparse.Namespace) -> int:
    print("hQVM GATE FORWARD-PROBE: Norm shadow + PPL Base/N/H/NH")
    print("=" * 5)
    print("  note=incomplete forward-site probe; not a product mode")
    print("  note=Norm COMMIT may FAIL until signed Δ-ruler")

    cfg = replace(get_gyroscopic_llm_config(), n_ctx=max(256, int(args.ctx)))
    try:
        print(f"  perplexity_exe={resolve_llama_perplexity_path(cfg).name}  {_pass(True)}")
    except FileNotFoundError:
        print("  perplexity_exe missing  FAIL")
        return 1

    print("\n1. NORM SHADOW COSINES")
    print("=" * 5)
    env_sh = _codecs_base_env()
    env_sh["GYRO_NORM_CODEC"] = "1"
    r = run_llama_cli(cfg, prompt=PARIS_PROMPT, n_predict=4, env=env_sh, timeout_sec=300)
    comb = (r.stdout or "") + "\n" + (r.stderr or "")
    cos = [float(m) for m in re.findall(r"\[hqvm-norm-codec\] cos=([0-9.]+)", comb)]
    mean_cos = sum(cos) / len(cos) if cos else None
    print(f"  n_cos={len(cos)} mean_cos={mean_cos}")
    cos_ok = mean_cos is not None and mean_cos >= 0.999
    print(f"  mean_cos>=0.999  {_pass(cos_ok)}")

    variants = [
        ("Base", {}),
        ("N", {"GYRO_NORM_CODEC": "1", "GYRO_NORM_COMMIT": "1"}),
        ("H", {"GYRO_CGM_LIFT": "1", "GYRO_RESIDUAL_HYBRID": "1"}),
        ("NH", {
            "GYRO_NORM_CODEC": "1", "GYRO_NORM_COMMIT": "1",
            "GYRO_CGM_LIFT": "1", "GYRO_RESIDUAL_HYBRID": "1",
        }),
    ]

    print("\n2. PPL")
    print("=" * 5)
    corpus = str(CORPUS_TINY)
    ppls: dict[str, float | None] = {}
    for name, extra in variants:
        env = _codecs_base_env()
        env.update(extra)
        ppls[name] = _ppl(cfg, env, name, corpus, int(args.ctx), int(args.timeout))

    base = ppls.get("Base")
    print("\n3. RATIOS")
    print("=" * 5)
    overall = base is not None and base > 0 and cos_ok
    if base and base > 0:
        print(f"  base_ppl={base}")
        for name, _ in variants:
            if name == "Base":
                continue
            p = ppls.get(name)
            if p is None:
                print(f"  {name}_ratio=None  {_pass(False)}")
                overall = False
                continue
            ratio = float(p) / float(base)
            ok = ratio <= MAX_RATIO
            print(f"  {name}_ratio={ratio:.4f}  {_pass(ok)}")
            if not ok:
                overall = False
    else:
        overall = False
        print(f"  base usable  {_pass(False)}")

    print(f"\n  GATE_FORWARD_PROBE  {_pass(overall)}")
    print("\nDONE")
    print("=" * 5)
    return 0 if overall else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Canonical gyroscopic acceptance gates (NavPad §7).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ledger", help="Ledger MatMul displace regression (Paris + PPL)")

    p2 = sub.add_parser("kv", help="Q8 K/V displace + holonomic Attn@V")
    p2.add_argument("--ppl", action="store_true", help="also run tiny-corpus PPL (slower)")
    p2.add_argument("--full", action="store_true", help="with --ppl: use full ppl_corpus.txt")
    p2.add_argument("--ctx", type=int, default=256, help="PPL context (-c)")
    p2.add_argument("--timeout", type=int, default=600, help="per-run PPL timeout seconds")

    pc = sub.add_parser("codecs", help="Aperture/RoPE/SwiGLU site probes vs Base")
    pc.add_argument("--full", action="store_true", help="use full ppl_corpus.txt")
    pc.add_argument("--ctx", type=int, default=256)
    pc.add_argument("--timeout", type=int, default=600)
    pc.add_argument("--smoke-only", action="store_true", help="skip PPL")

    sub.add_parser("causal", help="Lift perturb changes decode (coupling proof only)")

    ph = sub.add_parser(
        "forward-probe",
        help="Norm shadow + lift/residual PPL (incomplete sites; not a mode)",
    )
    ph.add_argument("--ctx", type=int, default=256)
    ph.add_argument("--timeout", type=int, default=2400)

    args = ap.parse_args()
    handlers = {
        "ledger": cmd_ledger,
        "kv": cmd_kv,
        "codecs": cmd_codecs,
        "causal": cmd_causal,
        "forward-probe": cmd_forward_probe,
    }
    return handlers[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
