#!/usr/bin/env python3
"""hqvm_runtime_analysis_1.py — Fat C medium cost measurements (hQVM_QuBEC_Theory.md §18).

Role: thin port smoke + ACTION/LEDGER + route2 synthesize + WHT climate + ACTION cache.
Inputs: hqvm_runtime_analysis_common / ops C backend (required).
Outputs: PASS/FAIL on ReportState; headline timings; --verbose tables.
Companion: hqvm_runtime_analysis_run.py. Not gates.py (real-weight product).
"""
from __future__ import annotations

import platform

import numpy as np

from hqvm_runtime_analysis_common import (
    ReportState,
    bench_n,
    check,
    info,
    ops,
    require_native,
    section,
    vprint,
)


def _rand_word(rng: np.random.RandomState, n: int) -> bytes:
    return bytes(int(x) for x in rng.randint(0, 256, n))


def _port_smoke(state: ReportState, o, rng: np.random.RandomState) -> None:
    """Thin trust checks only — not group re-proof."""
    section(state, "Port smoke (C trust)")
    g_ok = bool(o.wave_grammar_verify())
    check(
        state,
        "grammar",
        g_ok,
        quantity="C grammar verify returns ok",
        measured="ok" if g_ok else "fail",
        threshold="ok",
    )

    rest = o.pack_state12(0, 0)
    bad = 0
    n_trials = 64
    for _ in range(n_trials):
        w = _rand_word(rng, int(rng.randint(1, 9)))
        sig = o.sig13_compile(w)
        if o.trace_word_state12(rest, w) != o.sig13_apply(rest, sig):
            bad += 1
    check(
        state,
        "compile==apply",
          bad == 0,
        quantity="Port smoke: sig13 apply matches byte replay",
        measured=f"{n_trials - bad}/{n_trials}",
        threshold="all",
    )

    src = o.pack_state12(0, 0)
    tgt = o.pack_state12(37, 5)
    w = o.route2_witnesses(src, tgt)
    check(
        state,
        "route2 count",
        len(w) == 16,
        quantity="Port smoke: brute route2 returns 16",
        measured=str(len(w)),
        threshold="16",
    )


def _action_vs_ledger(state: ReportState, o, rng: np.random.RandomState) -> None:
    section(state, "ACTION vs LEDGER wall-clock")
    word_lens = (4, 16, 64, 256)
    batch_sizes = (256, 1024, 4096)
    vprint("  word_len | batch | ledger states/s | ACTION states/s | speedup")
    vprint("  " + "-" * 5)
    speedups: list[float] = []
    crossover_note = "ACTION faster on all sampled (word_len, batch)"
    found_ledger_win = False
    for n in word_lens:
        w = _rand_word(rng, n)
        sig = o.sig13_compile(w)
        for B in batch_sizes:
            states_b = [
                o.pack_state12(u, v) for u in range(64) for v in range(64)
            ][:B]

            def ledger(states_b=states_b, w=w):
                for s in states_b:
                    o.trace_word_state12(s, w)

            def action(states_b=states_b, sig=sig):
                o.sig13_apply_batch(states_b, sig)

            _t_l, lps = bench_n(ledger, B, repeat=3)
            _t_a, aps = bench_n(action, B, repeat=3)
            sp = aps / lps if lps > 0 else 0.0
            speedups.append(sp)
            if sp < 1.0:
                found_ledger_win = True
                crossover_note = f"LEDGER faster at word_len={n} batch={B} (ratio={sp:.2f})"
            vprint(f"  {n:8d} | {B:5d} | {lps:14.0f} | {aps:14.0f} | {sp:.2f}x")

    med = float(np.median(speedups)) if speedups else 0.0
    if not found_ledger_win:
        crossover_note = (
            f"ACTION faster on all sampled cells; median={med:.2f}x "
            f"(compile amortized excluded)"
        )
    state.headlines["ACTION vs LEDGER median speedup"] = f"{med:.2f}x"
    state.headlines["LEDGER/ACTION crossover note"] = crossover_note
    info(crossover_note)
    check(
        state,
        "Benchmark A",
        med >= 1.0,
        quantity="ACTION batch apply vs LEDGER byte replay (median)",
        measured=f"{med:.2f}x",
        threshold=">=1.0",
    )


def _route2_synth(state: ReportState, o, rng: np.random.RandomState) -> None:
    section(state, "Route2 synthesize vs brute")
    # Smoke: sets equal on coarse grid + random pairs
    mismatches = 0
    checked = 0
    for u in range(0, 64, 16):
        for v in range(0, 64, 16):
            src = o.pack_state12(u, v)
            for u2 in range(0, 64, 16):
                for v2 in range(0, 64, 16):
                    tgt = o.pack_state12(u2, v2)
                    brute = set(o.route2_witnesses(src, tgt))
                    syn = set(o.route2_synthesize(src, tgt))
                    checked += 1
                    if brute != syn or len(syn) != 16:
                        mismatches += 1
    for _ in range(32):
        src = o.pack_state12(int(rng.randint(0, 64)), int(rng.randint(0, 64)))
        tgt = o.pack_state12(int(rng.randint(0, 64)), int(rng.randint(0, 64)))
        brute = set(o.route2_witnesses(src, tgt))
        syn = set(o.route2_synthesize(src, tgt))
        checked += 1
        if brute != syn:
            mismatches += 1
    check(
        state,
        "synth==brute",
        mismatches == 0,
        quantity="Port smoke: synthesize set equals brute",
        measured=f"{checked - mismatches}/{checked}",
        threshold="all",
    )

    pairs = [
        (
            o.pack_state12(int(rng.randint(0, 64)), int(rng.randint(0, 64))),
            o.pack_state12(int(rng.randint(0, 64)), int(rng.randint(0, 64))),
        )
        for _ in range(64)
    ]

    def brute_all(pairs=pairs):
        for s, t in pairs:
            o.route2_witnesses(s, t)

    def syn_all(pairs=pairs):
        for s, t in pairs:
            o.route2_synthesize(s, t)

    t_b, _ = bench_n(brute_all, len(pairs), repeat=3)
    t_s, _ = bench_n(syn_all, len(pairs), repeat=5)
    us_b = 1e6 * t_b / len(pairs)
    us_s = 1e6 * t_s / len(pairs)
    ratio = us_b / us_s if us_s > 0 else 0.0
    state.headlines["route2 brute us/pair"] = f"{us_b:.2f}"
    state.headlines["route2 synthesize us/pair"] = f"{us_s:.3f}"
    vprint(f"  brute={us_b:.2f}us/pair  synthesize={us_s:.3f}us/pair  ratio={ratio:.1f}x")
    info(f"route2 synthesize {us_s:.3f}us vs brute {us_b:.2f}us ({ratio:.1f}x)")
    # Substance: synthesize much faster than brute
    ok_fast = us_s < 0.10 * us_b or us_s < 20.0
    check(
        state,
        "synth latency",
        ok_fast,
        quantity="Route2 synthesize latency vs brute",
        measured=f"syn={us_s:.3f}us brute={us_b:.2f}us",
        threshold="<10% of brute or <20us",
    )


def _climate_wht(state: ReportState, o, rng: np.random.RandomState) -> None:
    section(state, "Spectral climate WHT vs dense")
    f = [float(rng.randn()) for _ in range(64)]
    # Stabilize kernel scale
    s = sum(abs(v) for v in f) or 1.0
    f = [v / s for v in f]
    M, phi = o.climate_from_kernel(f)
    x0 = [float(rng.randn()) for _ in range(64)]
    n_smoke = 3
    xd = o.climate_dense_nstep(list(x0), M, n_smoke)
    xs = o.climate_spectral_nstep(list(x0), phi, n_smoke)
    max_abs = max(abs(a - b) for a, b in zip(xd, xs))
    check(
        state,
        "spectral==dense smoke",
        max_abs < 1e-3,
        quantity="Port smoke: spectral n-step matches dense circulant",
        measured=f"max_abs={max_abs:.2e}",
        threshold="<1e-3",
    )

    timings: list[str] = []
    speedups: list[float] = []
    vprint("  n | dense_s | spectral_s | speedup")
    vprint("  " + "-" * 5)
    for n in (10, 100, 1000):
        def dense(n=n, x0=x0, M=M):
            o.climate_dense_nstep(list(x0), M, n)

        def spectral(n=n, x0=x0, phi=phi):
            o.climate_spectral_nstep(list(x0), phi, n)

        td, _ = bench_n(dense, 1, repeat=3)
        ts, _ = bench_n(spectral, 1, repeat=5)
        sp = td / ts if ts > 0 else 0.0
        speedups.append(sp)
        timings.append(f"n={n}:{sp:.1f}x")
        vprint(f"  {n:4d} | {td:.6f} | {ts:.6f} | {sp:.2f}x")

    note = "; ".join(timings)
    state.headlines["WHT climate vs dense (n=10/100/1000)"] = note
    info(note)
    check(
        state,
        "climate speedup",
        min(speedups) >= 1.0 if speedups else False,
        quantity="Spectral climate wall-clock vs dense (all n)",
        measured=note,
        threshold=">=1.0x each",
    )

    # Shell7 timing (substance) + thin apply smoke
    section(state, "Shell7 radial apply")
    gains = [1.0, 0.9, 0.8, 0.7, 0.8, 0.9, 1.0]
    chi = [float(rng.rand()) for _ in range(64)]
    out = o.shell7_apply(list(chi), gains)
    # popcount shells: mass conserved up to gain weighting
    def shell_mass(vec):
        m = [0.0] * 7
        for i, v in enumerate(vec):
            m[bin(i).count("1")] += v
        return m

    m_in = shell_mass(chi)
    m_out = shell_mass(out)
    shell_ok = all(
        abs(m_out[n] - m_in[n] * gains[n]) < 1e-4 for n in range(7)
    )
    check(
        state,
        "shell7 mass",
        shell_ok,
        quantity="Port smoke: shell7 gains act on shell masses",
        measured="ok" if shell_ok else "mismatch",
        threshold="ok",
    )

    def shell_bench(chi=chi, gains=gains):
        for _ in range(1000):
            o.shell7_apply(list(chi), gains)

    t_sh, _ = bench_n(shell_bench, 1000, repeat=3)
    us = 1e6 * t_sh / 1000.0
    state.headlines["shell7 note"] = f"{us:.3f}us/apply (1000x loop)"
    info(state.headlines["shell7 note"])
    check(
        state,
        "shell7 timed",
        us < 1000.0,
        quantity="Shell7 apply latency",
        measured=f"{us:.3f}us",
        threshold="<1000us",
    )


def _equiv2080(state: ReportState, o, rng: np.random.RandomState) -> None:
    section(state, "G-equivariant 2080-sector layer")
    # Sector index: bijection (du,dv)->0..2079 with C(65,2)=2080 classes.
    idxs = set()
    for du in range(64):
        for dv in range(64):
            idxs.add(o.equiv2080_sector_index(du, dv))
    idx_ok = len(idxs) == 2080 and min(idxs) == 0 and max(idxs) == 2079
    check(
        state,
        "sector index",
        idx_ok,
        quantity="Sector index bijects XOR-diff pairs onto 0..2079",
        measured=f"{len(idxs)} classes",
        threshold="2080 in [0,2079]",
    )

    gains = [float(g) for g in rng.randn(2080)]

    # Build dense orbital matrix from the same gains (ground truth for equivariance).
    def gain_at(du: int, dv: int) -> float:
        return gains[o.equiv2080_sector_index(du, dv)]

    M = [0.0] * 4096 * 4096
    for su in range(64):
        for sv in range(64):
            row = (su * 64 + sv) * 4096
            for tu in range(64):
                du = su ^ tu
                base = row + tu * 64
                for tv in range(64):
                    M[base + tv] = gain_at(du, sv ^ tv)

    psi = [float(v) for v in rng.randn(4096)]

    # Equivariance probe: K(g·psi) == g·K(psi) on sampled group elements.
    # g = translation (tu,tv) and swap; permute psi by g before/after apply.
    def perm_apply(vec: list[float], t: tuple[int, int], swap: bool) -> list[float]:
        tu, tv = t
        out = [0.0] * 4096
        for u in range(64):
            for v in range(64):
                src = u * 64 + v
                wu, wv = (v ^ tv, u ^ tu) if swap else (u ^ tu, v ^ tv)
                out[wu * 64 + wv] = vec[src]
        return out

    max_rel = 0.0
    probes = [(5, 3, False), (37, 11, False), (9, 63, True), (63, 1, True)]
    for tu, tv, swap in probes:
        kp = o.equiv2080_apply(psi, gains)
        pg = perm_apply(psi, (tu, tv), swap)
        kpg = o.equiv2080_apply(pg, gains)
        want = perm_apply(kp, (tu, tv), swap)
        num = max(abs(a - b) for a, b in zip(kpg, want))
        den = max(1e-30, max(abs(v) for v in want))
        max_rel = max(max_rel, num / den)
        psi = pg  # chain probes across different vectors too

    check(
        state,
        "equivariance",
        max_rel < 1e-4,
        quantity="Equivariance: K(g·psi)==g·K(psi) over sampled G elements",
        measured=f"max_rel={max_rel:.2e}",
        threshold="<1e-4",
    )

    # Structured apply must equal its own dense expansion.
    y_s = o.equiv2080_apply(psi, gains)
    y_d = o.dense4096_matvec(M, psi)
    diff = max(abs(a - b) for a, b in zip(y_s, y_d))
    scale = max(abs(v) for v in y_d) or 1.0
    check(
        state,
        "structured==dense",
        diff / scale < 1e-4,
        quantity="Structured apply equals its dense orbital expansion",
        measured=f"max_abs={diff:.2e} (scale {scale:.2f})",
        threshold="rel<1e-4",
    )

    # Parameter count is the substance: 2080 vs dense.
    info(
        f"G-equivariant operator: {len(gains)} spectral params "
        f"vs {4096 * 4096} dense entries ({4096 * 4096 // len(gains)}x compression)"
    )

    # Wall-clock: structured vs dense matvec.
    def structured():
        o.equiv2080_apply(psi, gains)

    def dense():
        o.dense4096_matvec(M, psi)

    t_s, _ = bench_n(structured, 1, repeat=3)
    t_d2, _ = bench_n(dense, 1, repeat=3)
    sp = t_d2 / t_s if t_s > 0 else 0.0
    note = (
        f"equiv2080 apply {t_s * 1e3:.2f}ms vs dense4096 {t_d2 * 1e3:.2f}ms "
        f"({sp:.1f}x); 2080 params vs 16,777,216"
    )
    state.headlines["G-equivariant note"] = note
    info(note)
    check(
        state,
        "equiv timed",
        sp >= 1.0,
        quantity="Structured equivariant matvec vs dense 4096x4096 wall-clock",
        measured=f"{sp:.1f}x",
        threshold=">=1.0x",
    )


def _action_cache(state: ReportState, o, rng: np.random.RandomState) -> None:
    section(state, "ACTION cache (8192)")
    cache = o.sig13_cache_build()
    check(
        state,
        "cache identity fill",
        len(cache) == 8192 and cache[0] == 0 and cache[8191] == 8191,
        quantity="Port smoke: sig13 cache identity length 8192",
        measured=str(len(cache)),
        threshold="8192",
    )

    # Regime 1: compile tax amortized (B=4096). Expect ~1.0x: the apply loop
    # dominates and the cache is irrelevant. Gate is informational.
    B_big = 4096
    states_big = [o.pack_state12(u, v) for u in range(64) for v in range(64)][:B_big]
    words = [_rand_word(rng, 32) for _ in range(64)]
    sigs = [o.sig13_compile(w) for w in words]

    def with_recompile(states=states_big, words=words):
        o.sig13_compile_apply_many(states, words)

    def with_cache(sigs=sigs, states=states_big):
        o.sig13_apply_many_sigs(states, sigs, use_cache=True)

    t_c, _ = bench_n(with_recompile, len(sigs) * B_big, repeat=5)
    t_k, _ = bench_n(with_cache, len(sigs) * B_big, repeat=5)
    sp_amortized = t_c / t_k if t_k > 0 else 0.0

    # Regime 2: compile tax NOT amortized (B small). This is where the cache
    # must pay off; gate enforces >=1.0x here.
    B_small = 8
    states_small = states_big[:B_small]

    def recompile_small(states=states_small, words=words):
        o.sig13_compile_apply_many(states, words)

    def cache_small(sigs=sigs, states=states_small):
        o.sig13_apply_many_sigs(states, sigs, use_cache=True)

    t_c2, _ = bench_n(recompile_small, len(sigs) * B_small, repeat=7)
    t_k2, _ = bench_n(cache_small, len(sigs) * B_small, repeat=7)
    sp_unamort = t_c2 / t_k2 if t_k2 > 0 else 0.0

    note = (
        f"cache vs recompile: {sp_amortized:.2f}x at B={B_big} (amortized), "
        f"{sp_unamort:.2f}x at B={B_small} (unamortized); "
        f"64 sigs x 32B words, C loops"
    )
    state.headlines["ACTION cache note"] = note
    info(note)
    check(
        state,
        "cache throughput",
        sp_unamort >= 1.0,
        quantity="ACTION cache vs recompile (compile tax unamortized)",
        measured=f"{sp_unamort:.2f}x at B={B_small}",
        threshold=">=1.0x",
    )


def run(state: ReportState) -> None:
    require_native()
    o = ops()
    rng = np.random.RandomState(20260825)
    state.headlines["conditions"] = (
        f"{platform.system()} {platform.machine()} "
        f"python={platform.python_version()} numpy={np.__version__}"
    )

    _port_smoke(state, o, rng)
    _action_vs_ledger(state, o, rng)
    _route2_synth(state, o, rng)
    _climate_wht(state, o, rng)
    _equiv2080(state, o, rng)
    _action_cache(state, o, rng)


if __name__ == "__main__":
    st = ReportState()
    run(st)
    passed = sum(1 for _, ok in st.gates if ok)
    failed = sum(1 for _, ok in st.gates if not ok)
    print(f"\nSUMMARY: {passed} passed, {failed} failed out of {len(st.gates)}")
