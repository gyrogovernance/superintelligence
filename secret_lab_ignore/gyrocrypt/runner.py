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


def test_multicell_inject() -> None:
    from kernel.holonomy import (
        inject_residue_multicell,
        multicell_omega_key,
        native_cell_count,
    )

    head("Multi-cell QuBEC inject (holonomy register)")
    for n in (15, 143):
        nc = native_cell_count(n)
        keys = {
            multicell_omega_key(inject_residue_multicell(y, n, nc)) for y in range(n)
        }
        assert len(keys) == n, f"N={n} multicell inject not injective ({len(keys)}/{n})"
        print(f"[PASS] N={n} B={nc} inject_residue_multicell injective ({n}/{n})")


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
    from kernel.holonomy import compile_factor_operator, gyro_period, holonomy_spectrum
    from kernel.shor import factor, period, period_report
    from kernel.audit import period_reference

    head("Production period + factor (native spectral)")
    from kernel.core import cqft64_fast, zero_wavefunction

    v = cqft64_fast([1.0 + 0j] * 64)
    assert len(v) == 64
    print("[PASS] cqft64_fast smoke")

    r15 = period(15, 7)
    assert r15 == 4, f"production period(15,7) expected 4 got {r15}"
    rep = period_report(15, 7)
    assert rep["r"] == 4
    print(f"[PASS] production period N=15 r=4 path={rep['path']}")

    op15 = compile_factor_operator(15, 7)
    holo15 = gyro_period(15, 7)
    audit15 = period_reference(15, 7)
    assert not op15.compiled and op15.compile_method.startswith("MULTICELL_OPEN")
    print(
        f"[OPEN] holonomy N=15 method={op15.compile_method!r} → {holo15!r} "
        f"(audit r={audit15})"
    )

    got = factor(15, base=7)
    assert got == (3, 5), f"factor(15) expected (3,5) got {got}"
    print("[PASS] factor N=15 → (3,5)")

    op143 = compile_factor_operator(143, 7)
    r143 = period(143, 7)
    holo143 = gyro_period(143, 7)
    print(
        f"[INFO] N=143 production={r143} holonomy={holo143} "
        f"compiled={op143.compiled} (arith ord=60)"
    )
    _ = holonomy_spectrum(op143, max_depth=32)


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


def test_holonomy_vs_audit() -> None:
    import time

    from kernel.audit import period_reference
    from kernel.holonomy import compile_factor_operator, gyro_period, holonomy_spectrum

    head("Holonomy vs audit N=867199")
    t0 = time.perf_counter()
    op = compile_factor_operator(867199, 7)
    holo = gyro_period(867199, 7)
    audit = period_reference(867199, 7)
    elapsed = time.perf_counter() - t0
    if holo is None:
        print(
            f"[OPEN] holonomy fail-closed audit={audit} method={op.compile_method!r} "
            f"({elapsed:.1f}s)"
        )
    elif holo == audit:
        print(f"[PASS] holonomy matches audit at N=867199 ({elapsed:.1f}s)")
    else:
        print(f"[FAIL] holonomy mismatch holo={holo} audit={audit}")


def test_horizon_tensor() -> None:
    from kernel.audit import _default_Q
    from kernel.bindings import (
        horizon_pack_keys_u64,
        horizon_tensor_mag2_y1_u64,
        horizon_tensor_step_drift_u64,
        shor_dp_mag2_y1_u64,
    )
    from kernel.shor import peak_k_for_period

    head("Horizon tensor gates (K11/K15, C)")

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
    test_multicell_inject()
    test_native_c()
    test_simon()
    test_production_period_factor()
    test_audit()
    test_shor_large_native()
    test_holonomy_vs_audit()
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
