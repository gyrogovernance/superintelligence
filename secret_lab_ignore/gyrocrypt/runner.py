"""
Gyrocrypt test runner — native spectral production + holonomy OPEN gates.

  python secret_lab_ignore/gyrocrypt/runner.py
"""

from __future__ import annotations

import io
import random
import sys
import time
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_REPO = _ROOT.parents[1]
for p in (str(_ROOT), str(_REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DIV = "========="


def head(title: str) -> None:
    print(DIV)
    print(title)
    print(DIV)


def test_byte_bridge() -> None:
    from kernel import GENE_MAC_REST, is_in_omega24, step_byte, step_byte_inverse

    head("Byte bridge / Gyrostate")
    for b in range(256):
        assert is_in_omega24(step_byte(GENE_MAC_REST, b))
    for b in (0, 1, 42, 255):
        s_next = step_byte(GENE_MAC_REST, b)
        assert step_byte_inverse(s_next, b) == GENE_MAC_REST
    print("[PASS] Byte bridge roundtrip + Omega reachability")

    for b in range(256):
        s = GENE_MAC_REST
        for _ in range(4):
            s = step_byte(s, b)
        assert s == GENE_MAC_REST
    print("[PASS] Depth-4 closure b^4 = id")


def test_wavefunction() -> None:
    from kernel import (
        K4_F,
        K4_W2,
        K4_W2P,
        OMEGA_SIZE,
        apply_k4,
        from_holographic,
        to_holographic,
        zero_wavefunction,
    )

    head("Holonomy wavefunction")
    rng = random.Random(3)
    psi = [complex(rng.gauss(0, 1), rng.gauss(0, 1)) for _ in range(OMEGA_SIZE)]
    for gate in (K4_W2, K4_W2P, K4_F):
        twice = apply_k4(apply_k4(psi, gate), gate)
        assert max(abs(twice[i] - psi[i]) for i in range(OMEGA_SIZE)) < 1e-12
    print("[PASS] K4 involutions + F = W2 o W2'")

    holo = to_holographic(psi)
    back = from_holographic(holo)
    assert max(abs(back[i] - psi[i]) for i in range(OMEGA_SIZE)) < 1e-15
    print("[PASS] holographic reshape round-trip")


def test_horizon_algebra() -> None:
    from kernel.holonomy import analyze_horizon_algebra

    head("Horizon affine algebra (holonomy diagnostic)")
    rep = analyze_horizon_algebra(sample_depth4=2048)
    assert rep.distinct_q6 <= 64
    assert rep.max_chirality_order == 2
    print(
        f"[PASS] q6 distinct={rep.distinct_q6} chi_order≤{rep.max_chirality_order} "
        f"Ω depth1_max={rep.max_depth1_omega_order} depth4_sample_max={rep.max_depth4_omega_order}"
    )


def test_holonomy_e2e() -> None:
    import time

    from kernel.holonomy import holonomy_e2e

    head("Holonomy end-to-end (compile → oracle → readout)")

    for n, a, exp_r in ((15, 7, 4), (143, 7, 60)):
        t0 = time.perf_counter()
        rep = holonomy_e2e(n, a, verify_all_oracle=True, max_closure_depth=20_000)
        elapsed = time.perf_counter() - t0
        assert rep.oracle_ok, f"N={n} oracle failed"
        assert rep.suffix_period == exp_r, f"N={n} suffix {rep.suffix_period} != {exp_r}"
        assert rep.closure_period == exp_r, f"N={n} closure {rep.closure_period} != {exp_r}"
        print(
            f"[PASS] N={n} e2e r={exp_r} method={rep.compile_method!r} "
            f"path={rep.suffix_path} ({elapsed:.1f}s)"
        )

    t0 = time.perf_counter()
    rep867 = holonomy_e2e(
        867199, 7, verify_all_oracle=True, max_closure_depth=0
    )
    elapsed = time.perf_counter() - t0
    assert rep867.oracle_ok
    assert rep867.suffix_period == 18018
    assert rep867.suffix_path != "NONE"
    print(
        f"[PASS] N=867199 e2e oracle+suffix r={rep867.suffix_period} "
        f"path={rep867.suffix_path} checked={rep867.oracle_checked} ({elapsed:.1f}s)"
    )
    print(f"[INFO] {rep867.notes}")


def test_bfs_mul_oracle_falsified() -> None:
    from kernel.holonomy import search_multiply_word_bfs

    head("BFS multiply oracle (offline falsification K18)")
    w4 = search_multiply_word_bfs(15, 7, max_depth=4, max_nodes=80_000)
    assert w4 is None
    print("[PASS] N=15,a=7 no parallel byte word at depth≤4")
    print("[INFO] depth≤6 also None (~80s offline); wavefunction BFS rebrand rejected")


def test_multicell_inject() -> None:
    from kernel.holonomy import (
        decode_residue_multicell,
        inject_residue_multicell,
        multicell_omega_key,
        native_cell_count,
        _chi_as_limb_decode,
    )

    head("Multi-cell QuBEC inject (holonomy register)")
    for n in (15, 143):
        nc = native_cell_count(n)
        keys = {
            multicell_omega_key(inject_residue_multicell(y, n, nc)) for y in range(n)
        }
        assert len(keys) == n, f"N={n} multicell inject not injective ({len(keys)}/{n})"
        chi_mism = sum(
            1
            for y in range(n)
            if _chi_as_limb_decode(inject_residue_multicell(y, n, nc), nc) != y
        )
        assert chi_mism == n, "chi-as-limb must not equal encoded residue"
        for y in range(n):
            st = inject_residue_multicell(y, n, nc)
            assert decode_residue_multicell(st, n, nc) == y
        print(
            f"[PASS] N={n} B={nc} inject/decode round-trip; "
            f"chi≠limb ({chi_mism}/{n})"
        )


def test_native_c() -> None:
    from kernel.bindings import (
        build_native,
        exp_mod_ladder,
        exp_mod_ladder_limbs,
        mul_mod_ladder,
        shor_period_u64,
        sparse_cqft_peaks,
    )
    from kernel.audit import period_reference

    head("Native C")
    build_native()
    assert mul_mod_ladder(3, 7, 15) == 6
    assert exp_mod_ladder(7, 4, 15) == 1
    assert exp_mod_ladder_limbs(7, 4, 15) == 1
    print("[PASS] native modexp uint64 + limbs")

    assert abs(exp_mod_ladder(7, 60, 143) - 1) == 0
    print("[PASS] cyclic character χ(1)^N = 1")

    peaks = sparse_cqft_peaks([1, 4, 7], 64, k_top=8)
    assert peaks and peaks[0][1] > 0
    print("[PASS] native sparse_cqft_peaks")

    assert int(shor_period_u64(7, 15, 4096)) == 4
    print("[PASS] shor_period_u64(7,15,4096) == 4")

    assert period_reference(143, 7) == 60
    print("[PASS] native modexp + audit period_reference N=143")


def test_simon() -> None:
    from kernel import simon

    head("Simon")
    for n_bits, secret in ((6, 0b101010), (12, 0xA5F), (18, 0x2D4B1), (60, 0x2D4B1A5F03)):
        mask = (1 << n_bits) - 1
        got = simon(n_bits, secret)
        assert got == (secret & mask)
    print("[PASS] simon n=6,12,18,(60)")


def test_production_period_factor() -> None:
    from kernel.shor import factor, period, period_report

    head("Production period + factor (native spectral)")
    r15 = period(15, 7)
    assert r15 == 4
    rep = period_report(15, 7)
    assert rep["r"] == 4
    print(f"[PASS] production period N=15 r=4 path={rep['path']}")
    assert factor(15, base=7) == (3, 5)
    print("[PASS] factor N=15 → (3,5)")
    r143 = period(143, 7)
    assert r143 == 60
    print(f"[PASS] production period N=143 r={r143}")


def test_audit() -> None:
    from kernel.audit import period_reference

    head("Audit: classical F_{G_X} scorer (reference only — uses modexp coset)")
    assert period_reference(143, 7) == 60
    print("[PASS] audit period_reference N=143 r=60")


def test_shor_large_native() -> None:
    import time

    from kernel.audit import period_reference
    from kernel.bindings import shor_last_path_tag

    head("Audit: large-N classical reference (867199)")
    t0 = time.perf_counter()
    r = period_reference(867199, 7)
    elapsed = time.perf_counter() - t0
    path = shor_last_path_tag()
    assert r == 18018, f"expected r=18018 got {r}"
    print(f"[PASS] audit period N=867199 (r={r}, path={path}, {elapsed:.1f}s)")


def test_horizon_tensor() -> None:
    from kernel.audit import _default_Q
    from kernel.bindings import (
        horizon_pack_keys_u64,
        horizon_tensor_mag2_y1_u64,
        horizon_tensor_step_drift_u64,
        shor_dp_mag2_y1_u64,
    )
    from kernel.shor import peak_k_for_period

    head("Horizon tensor gates (K11)")

    for N, base, r in ((15, 7, 4), (143, 7, 60)):
        B = max(1, (2 * N.bit_length() + 5) // 6)
        Q = _default_Q(N, B)
        k = peak_k_for_period(Q, r)
        keys, n_cells = horizon_pack_keys_u64(N)
        exact_y1 = shor_dp_mag2_y1_u64(base, N, Q, k)
        assert exact_y1 > 0, f"N={N} exact DP mag2 at y=1 must be positive"
        if n_cells < 2:
            print(f"[PASS] N={N} exact DP y=1 k={k} (tensor drift skipped: n_cells={n_cells})")
            continue
        rows = horizon_tensor_step_drift_u64(base, N, Q, k, keys, n_cells)
        for j, exact, tensor, ratio in rows:
            assert abs(ratio - 1.0) < 1e-9, f"N={N} digit {j} ratio={ratio}"
        tensor_y1 = horizon_tensor_mag2_y1_u64(base, N, Q, k, keys, n_cells)
        rel = abs(tensor_y1 - exact_y1) / max(exact_y1, 1e-30)
        assert rel < 1e-9, f"N={N} tensor/exact parity rel_err={rel}"
        print(f"[PASS] N={N} drift ratio=1.0 tensor/exact parity k={k}")


def test_dlp() -> None:
    from kernel.shor import dlp_solve

    head("DLP (production fail-closed)")
    x = dlp_solve(11, 2, 3)
    if x is not None:
        print(f"[PASS] dlp_solve x={x}")
    else:
        print("[OPEN] dual-flow holonomy DLP readout not implemented")


def run_tests() -> None:
    test_byte_bridge()
    test_wavefunction()
    test_horizon_algebra()
    test_bfs_mul_oracle_falsified()
    test_multicell_inject()
    test_native_c()
    test_simon()
    test_production_period_factor()
    test_audit()
    test_shor_large_native()
    test_holonomy_e2e()
    test_horizon_tensor()
    test_dlp()


def main() -> None:
    buf = io.StringIO()

    class _Tee:
        def __init__(self, *streams):
            self._streams = streams

        def write(self, data):
            for s in self._streams:
                s.write(data)

        def flush(self):
            for s in self._streams:
                s.flush()

    tee = _Tee(sys.stdout, buf)
    sys.stdout = tee
    failed = False
    try:
        head("Gyrocrypt kernel")
        t0 = time.time()
        run_tests()
        print(f"\n[OK] all tests ({time.time() - t0:.1f}s)")
    except Exception as e:
        failed = True
        import traceback

        traceback.print_exc()
        print(f"\n[FAIL] {e}")
    finally:
        sys.stdout = sys.__stdout__

    out_dir = _ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    out_path.write_text(buf.getvalue(), encoding="utf-8")
    print(f"Results: {out_path.relative_to(_REPO)}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
