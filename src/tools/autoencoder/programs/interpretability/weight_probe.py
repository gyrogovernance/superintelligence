"""Probe frozen Bonsai weight rows through the CGM convertor.

Usage::

  python -m src.tools.autoencoder.programs.interpretability.weight_probe
  python -m src.tools.autoencoder.programs.interpretability.weight_probe --tensor output.weight --max-rows 64
  python -m src.tools.autoencoder.programs.interpretability.weight_probe --suite --max-rows 64
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.tools.autoencoder.helpers.evals_metrics import climate_readout
from src.tools.autoencoder.helpers.evals_run import json_safe, load_any_checkpoint
from src.tools.autoencoder.models.general import AffineSpectralCodec, K4Autoencoder
from src.tools.autoencoder.models.super import Super
from src.tools.autoencoder.paths import checkpoints_dir, ensure, reports_dir, tmp_dir
from src.tools.autoencoder.programs.interpretability.adapter import (
    DEFAULT_GGUF,
    load_tensor_rows,
)
from src.tools.autoencoder.programs.interpretability.converter import (
    chirality_from_tiles,
    energy_shell_hist,
    micro_frames,
    omega_pairs,
    participation,
    permute_tiles,
    q1_pair_mask,
    rows_to_tiles,
    shells_from_chi,
    tile_mode_energy,
    transport_shells,
)

CLIMATE_HIT_TV = 0.05
SUITE_TENSORS = (
    "token_embd.weight",
    "output.weight",
    "blk.0.attn_q.weight",
    "blk.17.attn_q.weight",
    "blk.35.attn_q.weight",
    "blk.0.attn_output.weight",
    "blk.17.attn_output.weight",
    "blk.35.attn_output.weight",
    "blk.0.ffn_down.weight",
    "blk.17.ffn_down.weight",
    "blk.35.ffn_down.weight",
)

_CACHE: dict[str, Any] = {}


def _stem(tensor_name: str) -> str:
    if "embd" in tensor_name:
        return "emb"
    if tensor_name.endswith("output.weight") and not tensor_name.startswith("blk."):
        return "out"
    return tensor_name.replace(".", "_").replace("/", "_")


def _tv(a: np.ndarray, b: np.ndarray) -> float:
    return 0.5 * float(np.abs(a - b).sum())


def _hist7(shells: np.ndarray) -> np.ndarray:
    h = np.bincount(np.asarray(shells).reshape(-1).astype(np.int64), minlength=7)
    p = h.astype(np.float64)
    return p / (p.sum() + 1e-30)


def _walsh_batch(codec: AffineSpectralCodec, rows: np.ndarray) -> dict[str, np.ndarray]:
    x = np.asarray(rows, dtype=np.float64)
    nrm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-30
    t = torch.as_tensor(x / nrm, dtype=torch.float32)
    coeff = codec.walsh_coefficients(t)
    energy = (coeff**2).detach().cpu().numpy().astype(np.float64)
    bid = codec.block_id.detach().cpu().numpy().astype(np.int64)
    n90 = []
    entropy = []
    diag = []
    for e in energy:
        block_e = np.bincount(bid, weights=e, minlength=2080).astype(np.float64)
        block_e = block_e / (block_e.sum() + 1e-30)
        csum = np.cumsum(block_e[np.argsort(block_e)[::-1]])
        n90.append(float(int(np.searchsorted(csum, 0.90) + 1)))
        nz = block_e[block_e > 0]
        entropy.append(float(-(nz * np.log(nz)).sum()) if len(nz) else 0.0)
        diag.append(float(sum(e[a * 64 + a] for a in range(64))) / (float(e.sum()) + 1e-30))
    return {
        "n_blocks_90": np.asarray(n90, dtype=np.float64),
        "block_entropy": np.asarray(entropy, dtype=np.float64),
        "diag_frac": np.asarray(diag, dtype=np.float64),
    }


def _maybe_load(path: Path) -> Any | None:
    key = str(path)
    if key in _CACHE:
        return _CACHE[key]
    if not path.is_file():
        _CACHE[key] = None
        return None
    model, _ = load_any_checkpoint(path)
    model.eval()
    _CACHE[key] = model
    return model


def _codec() -> AffineSpectralCodec:
    if "codec" not in _CACHE:
        codec = AffineSpectralCodec(frozen=True)
        codec.eval()
        _CACHE["codec"] = codec
    return _CACHE["codec"]


def _lambda_chi(n: int, t: int, rng: np.random.Generator, lam: float) -> np.ndarray:
    w = np.array([lam ** int(m).bit_count() for m in range(64)], dtype=np.float64)
    w /= w.sum()
    return rng.choice(64, size=(n, t), p=w).astype(np.int64)


def _super_climate(super_model: Super, frames: np.ndarray, cap: int = 1024) -> np.ndarray:
    climates = []
    with torch.no_grad():
        m = min(len(frames), cap)
        for i in range(0, m, 64):
            chunk = torch.as_tensor(frames[i : i + 64], dtype=torch.long)
            climates.append(super_model.encode(chunk).climate.cpu().numpy())
    return np.concatenate(climates, axis=0).mean(axis=0)


def _k4_energy(k4: K4Autoencoder, omega: np.ndarray) -> dict[str, float]:
    idx = np.asarray(omega, dtype=np.int64).reshape(-1)
    idx = idx[: min(len(idx), 4096)]
    with torch.no_grad():
        parts = k4.named_components(torch.as_tensor(idx, dtype=torch.long))
        return {k: float(v.pow(2).mean()) for k, v in parts.items()}


def run(
    *,
    gguf: Path,
    tensor_name: str,
    max_rows: int,
    seed: int = 7,
) -> dict[str, Any]:
    ensure()
    rng = np.random.default_rng(seed)
    rows = load_tensor_rows(gguf, tensor_name, max_rows)
    tiles = rows_to_tiles(rows)
    energy = tile_mode_energy(tiles)
    chi = chirality_from_tiles(tiles, rng=rng)
    shells = shells_from_chi(chi)
    transport = transport_shells(chi)
    n, t = chi.shape

    sign_null = rng.choice([-1.0, 1.0], size=tiles.shape)
    energy_pm = tile_mode_energy(sign_null)
    chi_pm = chirality_from_tiles(sign_null, rng=rng)
    chi_unif = rng.integers(0, 64, size=chi.shape, dtype=np.int64)
    chi_l4 = _lambda_chi(n, t, rng, 4.0)
    chi_perm = permute_tiles(chi, rng)

    e_sh = energy_shell_hist(energy)
    e_pm = energy_shell_hist(energy_pm)
    pr_real = float(participation(energy).mean())
    pr_pm = float(participation(energy_pm).mean())
    tv_energy_pm = _tv(e_sh, e_pm)

    qubec = climate_readout([1.0, 4.0])
    census = np.asarray(qubec["census_shell_histogram"], dtype=np.float64)
    l1 = np.asarray(qubec["ensemble_shell_histogram"][0], dtype=np.float64)
    l4 = np.asarray(qubec["ensemble_shell_histogram"][1], dtype=np.float64)

    real_sh = _hist7(shells)
    pm_sh = _hist7(shells_from_chi(chi_pm))
    unif_sh = _hist7(shells_from_chi(chi_unif))
    l4_sh = _hist7(shells_from_chi(chi_l4))
    tr_real = _hist7(transport)
    tr_perm = _hist7(transport_shells(chi_perm))
    tr_unif = _hist7(transport_shells(chi_unif))
    tr_pm = _hist7(transport_shells(chi_pm))

    within_m, across_m = q1_pair_mask(t)
    tr_within = _hist7(transport[:, within_m])
    tr_across = _hist7(transport[:, across_m])
    tr_perm_within = _hist7(transport_shells(chi_perm)[:, within_m])
    tr_perm_across = _hist7(transport_shells(chi_perm)[:, across_m])

    tv_pm = _tv(real_sh, pm_sh)
    tv_census = _tv(real_sh, census)
    tv_energy_census = _tv(e_sh, census)
    tv_tr_perm = _tv(tr_real, tr_perm)
    tv_q1 = _tv(tr_within, tr_across)

    n_spec = min(32, len(rows))
    codec = _codec()
    spec_real = _walsh_batch(codec, rows[:n_spec])
    shuf_rows = rows[:n_spec].astype(np.float64).copy()
    for i in range(n_spec):
        rng.shuffle(shuf_rows[i])
    spec_shuf = _walsh_batch(codec, shuf_rows)
    spec_sign = _walsh_batch(codec, np.sign(rows[:n_spec].astype(np.float64)))

    n90_real = float(spec_real["n_blocks_90"].mean())
    n90_shuf = float(spec_shuf["n_blocks_90"].mean())
    n90_sign = float(spec_sign["n_blocks_90"].mean())
    diag_real = float(spec_real["diag_frac"].mean())
    diag_shuf = float(spec_shuf["diag_frac"].mean())

    k4 = _maybe_load(checkpoints_dir() / "production" / "k4_full.pt")
    k4_read = None
    if isinstance(k4, K4Autoencoder):
        k4_read = {
            "adjacent_omega": _k4_energy(k4, omega_pairs(chi)),
            "order_shuffled_omega": _k4_energy(k4, omega_pairs(chi_perm)),
            "uniform_omega": _k4_energy(k4, omega_pairs(chi_unif)),
        }

    super_model = _maybe_load(checkpoints_dir() / "production" / "super.pt")
    super_read = None
    if isinstance(super_model, Super):
        cr = _super_climate(super_model, micro_frames(chi))
        cp = _super_climate(super_model, micro_frames(chi_perm))
        cu = _super_climate(super_model, micro_frames(chi_unif))
        cl = _super_climate(super_model, micro_frames(chi_l4))
        super_read = {
            "climate_real": cr.tolist(),
            "tv_vs_order_shuffle": _tv(cr, cp),
            "tv_vs_uniform_chi": _tv(cr, cu),
            "tv_vs_lambda4_chi": _tv(cr, cl),
            "tv_vs_census": _tv(cr, census),
        }

    stem = _stem(tensor_name)
    npz = tmp_dir() / f"interp_chi_{stem}.npz"
    np.savez_compressed(
        npz,
        allow_pickle=False,
        chi=chi.astype(np.uint8),
        shells=shells.astype(np.uint8),
        transport=transport.astype(np.uint8),
    )

    energy_is_sign_null = bool(tv_energy_pm < CLIMATE_HIT_TV)
    climate_is_sign_null = bool(tv_pm < CLIMATE_HIT_TV)
    order_hit = bool(tv_tr_perm >= CLIMATE_HIT_TV)
    q1_split_hit = bool(tv_q1 >= CLIMATE_HIT_TV)
    walsh_more_concentrated = bool(n90_shuf - n90_real >= 20.0)
    tile_sparser_than_pm1 = bool(pr_pm - pr_real >= 2.0)

    report: dict[str, Any] = {
        "tensor": tensor_name,
        "gguf": str(gguf).replace("\\", "/"),
        "max_rows": max_rows,
        "seed": seed,
        "n_rows": int(n),
        "chi_npz": str(npz).replace("\\", "/"),
        "convertor": (
            "64-WHT peak chirality of sign tiles; adjacent XOR as transport; "
            "adjacent chi pair as Ω; signed row as Ω Walsh"
        ),
        "claim": {
            "tile_walsh_energy_matches_pm1": energy_is_sign_null,
            "tv_energy_vs_pm1": tv_energy_pm,
            "tv_energy_vs_census": tv_energy_census,
            "tile_pr_real": pr_real,
            "tile_pr_pm1": pr_pm,
            "tile_sparser_than_pm1": tile_sparser_than_pm1,
            "tile_chi_matches_pm1_tiles": climate_is_sign_null,
            "tv_chi_vs_pm1_tiles": tv_pm,
            "tv_chi_vs_census": tv_census,
            "tv_chi_vs_uniform_chi": _tv(real_sh, unif_sh),
            "tv_chi_vs_lambda4": _tv(real_sh, l4_sh),
            "tv_chi_vs_qubec_l1": _tv(real_sh, l1),
            "tv_chi_vs_qubec_l4": _tv(real_sh, l4),
            "transport_order_is_not_a_shuffle": order_hit,
            "tv_transport_vs_order_shuffle": tv_tr_perm,
            "tv_transport_vs_iid_xor": _tv(tr_real, tr_unif),
            "tv_transport_vs_pm1_xor": _tv(tr_real, tr_pm),
            "q1_within_vs_across_differs": q1_split_hit,
            "tv_q1_within_vs_across": tv_q1,
            "tv_q1_within_vs_perm_within": _tv(tr_within, tr_perm_within),
            "tv_q1_across_vs_perm_across": _tv(tr_across, tr_perm_across),
            "signed_row_more_concentrated_than_shuffle": walsh_more_concentrated,
            "signed_row_n90_gap_vs_shuffle": n90_real - n90_shuf,
            "signed_row_n90_gap_vs_signs": n90_real - n90_sign,
            "signed_row_diag_gap_vs_shuffle": diag_real - diag_shuf,
        },
        "shell_hist_tile_energy": e_sh.tolist(),
        "shell_hist_pm1_energy": e_pm.tolist(),
        "shell_hist_tile_chi": real_sh.tolist(),
        "shell_hist_pm1": pm_sh.tolist(),
        "transport_hist": tr_real.tolist(),
        "transport_hist_order_shuffle": tr_perm.tolist(),
        "transport_hist_q1_within": tr_within.tolist(),
        "transport_hist_q1_across": tr_across.tolist(),
        "k4": k4_read,
        "super": super_read,
        "spectral_signed_rows": {
            "n90_real": n90_real,
            "n90_shuffle": n90_shuf,
            "n90_sign_only": n90_sign,
            "diag_real": diag_real,
            "diag_shuffle": diag_shuf,
            "entropy_real": float(spec_real["block_entropy"].mean()),
            "entropy_shuffle": float(spec_shuf["block_entropy"].mean()),
        },
    }
    out_path = reports_dir() / f"weight_probe_{stem}.json"
    out_path.write_text(json.dumps(json_safe(report), indent=2), encoding="utf-8")
    report["_written"] = str(out_path).replace("\\", "/")
    return report


def _print_report(report: dict[str, Any]) -> None:
    print(f"wrote {report['_written']}")
    c = report["claim"]
    en_s = "HIT" if c["tile_walsh_energy_matches_pm1"] else "MISS"
    sp_s = "HIT" if c["tile_sparser_than_pm1"] else "MISS"
    chi_s = "HIT" if c["tile_chi_matches_pm1_tiles"] else "MISS"
    or_s = "HIT" if c["transport_order_is_not_a_shuffle"] else "MISS"
    q1_s = "HIT" if c["q1_within_vs_across_differs"] else "MISS"
    wa_s = "HIT" if c["signed_row_more_concentrated_than_shuffle"] else "MISS"
    print(
        f"  tile Walsh energy vs ±1 (convertor rest): {en_s} "
        f"(tv={c['tv_energy_vs_pm1']:.4f} vs_census={c['tv_energy_vs_census']:.4f} "
        f"pr={c['tile_pr_real']:.2f}/{c['tile_pr_pm1']:.2f} sparse={sp_s})"
    )
    print(
        f"  tile-chi vs ±1 (tie-broken peak): {chi_s} "
        f"(tv={c['tv_chi_vs_pm1_tiles']:.4f} vs_census={c['tv_chi_vs_census']:.4f})"
    )
    print(
        f"  transport vs same-chi order-shuffle: {or_s} "
        f"(tv={c['tv_transport_vs_order_shuffle']:.4f})"
    )
    print(
        f"  Q1 within-block vs across-block: {q1_s} "
        f"(tv={c['tv_q1_within_vs_across']:.4f})"
    )
    print(
        f"  signed Ω Walsh vs entry-shuffle: {wa_s} "
        f"(n90_gap={c['signed_row_n90_gap_vs_shuffle']:+.1f} "
        f"vs_signs={c['signed_row_n90_gap_vs_signs']:+.1f} "
        f"diag_gap={c['signed_row_diag_gap_vs_shuffle']:+.4f})"
    )
    if report.get("super"):
        s = report["super"]
        print(
            f"  Super chi-micro vs order-shuffle: tv={s['tv_vs_order_shuffle']:.4f} "
            f"vs_unif={s['tv_vs_uniform_chi']:.4f} vs_l4={s['tv_vs_lambda4_chi']:.4f}"
        )
    if report.get("k4"):
        k = report["k4"]
        inv = k["adjacent_omega"]["z_invariant"]
        inv_p = k["order_shuffled_omega"]["z_invariant"]
        print(f"  K4 Ω-pair z_invariant: real={inv:.4f} order-shuffle={inv_p:.4f}")


def run_suite(*, gguf: Path, max_rows: int, seed: int) -> dict[str, Any]:
    rows = []
    for name in SUITE_TENSORS:
        try:
            report = run(gguf=gguf, tensor_name=name, max_rows=max_rows, seed=seed)
        except Exception as exc:
            rows.append({"tensor": name, "error": str(exc)})
            print(f"skip {name}: {exc}")
            continue
        _print_report(report)
        c = report["claim"]
        rows.append(
            {
                "tensor": name,
                "tile_energy_matches_pm1": c["tile_walsh_energy_matches_pm1"],
                "tv_energy_vs_pm1": c["tv_energy_vs_pm1"],
                "tv_energy_vs_census": c["tv_energy_vs_census"],
                "tile_pr_real": c["tile_pr_real"],
                "tile_pr_pm1": c["tile_pr_pm1"],
                "tile_sparser_than_pm1": c["tile_sparser_than_pm1"],
                "tile_chi_matches_pm1": c["tile_chi_matches_pm1_tiles"],
                "tv_chi_vs_pm1": c["tv_chi_vs_pm1_tiles"],
                "transport_order_hit": c["transport_order_is_not_a_shuffle"],
                "tv_transport_vs_order_shuffle": c["tv_transport_vs_order_shuffle"],
                "q1_split_hit": c["q1_within_vs_across_differs"],
                "tv_q1_within_vs_across": c["tv_q1_within_vs_across"],
                "walsh_hit": c["signed_row_more_concentrated_than_shuffle"],
                "n90_gap_vs_shuffle": c["signed_row_n90_gap_vs_shuffle"],
                "n90_gap_vs_signs": c["signed_row_n90_gap_vs_signs"],
                "super_tv_vs_order_shuffle": (
                    None if not report.get("super") else report["super"]["tv_vs_order_shuffle"]
                ),
                "report": report["_written"],
            }
        )
    summary = {
        "gguf": str(gguf).replace("\\", "/"),
        "max_rows": max_rows,
        "seed": seed,
        "tensors": rows,
    }
    out_path = reports_dir() / "weight_probe_suite.json"
    out_path.write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
    summary["_written"] = str(out_path).replace("\\", "/")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    ap.add_argument("--tensor", default="token_embd.weight")
    ap.add_argument("--max-rows", type=int, default=64)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--suite", action="store_true")
    args = ap.parse_args(argv)
    if not args.gguf.is_file():
        print(f"GGUF not found: {args.gguf}")
        return 2
    if args.suite:
        summary = run_suite(gguf=args.gguf, max_rows=args.max_rows, seed=args.seed)
        print(f"suite {summary['_written']}")
        return 0
    report = run(
        gguf=args.gguf,
        tensor_name=args.tensor,
        max_rows=args.max_rows,
        seed=args.seed,
    )
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
