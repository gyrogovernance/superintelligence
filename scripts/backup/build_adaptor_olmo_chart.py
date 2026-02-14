import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.getcwd())

import numpy as np
from transformers import AutoModelForCausalLM


MODEL_DIR = Path("data/models/Olmo-3-7B-Instruct")
OUT_PATH = Path("data/atlas/adaptor_olmo_k16_R16.npz")
ADAPTOR_VERSION = "3.0"

NB = 256
NF = 16
R = 0
K = 16
FAMILY_PROFILE = "attn_mix_mlp_up"
CHART_MODE = "xor_aa"

LAYERS = [3, 7, 11, 15, 19, 23, 27, 31]

PROFILE_RULES: dict[str, set[str]] = {
    "all": {"attn_proj", "attn_mix", "mlp_up_mix", "mlp_gate_mix", "lm_head_gram"},
    "attn_only": {"attn_proj"},
    "attn_plus_mix": {"attn_proj", "attn_mix"},
    "mix_only": {"attn_mix", "mlp_up_mix", "mlp_gate_mix"},
    "attn_mix_mlp_up": {"attn_mix", "mlp_up_mix"},
    "mlp_only": {"mlp_up_mix", "mlp_gate_mix"},
}
CHART_MODES = ("dense", "xor_aa")


def _np_weight(state_dict: dict[str, object], key: str) -> np.ndarray:
    t = state_dict[key]
    return t.detach().float().cpu().numpy()


def load_ops_from_model() -> tuple[dict[str, np.ndarray], dict[str, int]]:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        device_map="cpu",
    )
    model.eval()
    state_dict = model.state_dict()

    d = NB * NF
    ops: dict[str, np.ndarray] = {}
    op_phase: dict[str, int] = {}

    for idx, layer in enumerate(LAYERS):
        phase = idx % 4
        pfx = f"model.layers.{layer}"

        Wq = _np_weight(state_dict, f"{pfx}.self_attn.q_proj.weight")
        Wk = _np_weight(state_dict, f"{pfx}.self_attn.k_proj.weight")
        Wv = _np_weight(state_dict, f"{pfx}.self_attn.v_proj.weight")
        Wo = _np_weight(state_dict, f"{pfx}.self_attn.o_proj.weight")
        Wup = _np_weight(state_dict, f"{pfx}.mlp.up_proj.weight")
        Wgate = _np_weight(state_dict, f"{pfx}.mlp.gate_proj.weight")
        Wdown = _np_weight(state_dict, f"{pfx}.mlp.down_proj.weight")

        for name, W in (
            (f"L{layer}_q_proj", Wq),
            (f"L{layer}_k_proj", Wk),
            (f"L{layer}_v_proj", Wv),
            (f"L{layer}_o_proj", Wo),
        ):
            if W.shape == (d, d):
                ops[name] = W.astype(np.float32)
                op_phase[name] = phase

        A = Wo @ Wv
        Mup = Wdown @ Wup
        Mgate = Wdown @ Wgate
        for name, W in (
            (f"L{layer}_attn_mix", A),
            (f"L{layer}_mlp_up_mix", Mup),
            (f"L{layer}_mlp_gate_mix", Mgate),
        ):
            if W.shape == (d, d):
                ops[name] = W.astype(np.float32)
                op_phase[name] = phase

    lm_head = _np_weight(state_dict, "lm_head.weight")
    H = lm_head.T @ lm_head
    if H.shape == (d, d):
        ops["lm_head_gram"] = H.astype(np.float32)
        op_phase["lm_head_gram"] = -1
    return ops, op_phase


def op_category(name: str) -> str:
    if name.endswith("_q_proj") or name.endswith("_k_proj") or name.endswith("_v_proj") or name.endswith("_o_proj"):
        return "attn_proj"
    if name.endswith("_attn_mix"):
        return "attn_mix"
    if name.endswith("_mlp_up_mix"):
        return "mlp_up_mix"
    if name.endswith("_mlp_gate_mix"):
        return "mlp_gate_mix"
    if name == "lm_head_gram":
        return "lm_head_gram"
    return "other"


def select_ops_profile(
    ops: dict[str, np.ndarray],
    op_phase: dict[str, int],
    profile: str,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    allowed = PROFILE_RULES.get(profile)
    if allowed is None:
        valid = ", ".join(sorted(PROFILE_RULES.keys()))
        raise ValueError(f"Unknown family profile '{profile}'. Valid: {valid}")

    out_ops: dict[str, np.ndarray] = {}
    out_phase: dict[str, int] = {}
    for name, W in ops.items():
        if op_category(name) in allowed:
            out_ops[name] = W
            out_phase[name] = op_phase[name]
    return out_ops, out_phase


def boundary_coupling_matrix(W: np.ndarray) -> np.ndarray:
    d = NB * NF
    assert W.shape == (d, d)
    T = W.reshape(NB, NF, NB, NF)
    C = np.zeros((NB, NB), dtype=np.float64)
    for ho in range(NB):
        B = T[ho, :, :, :].reshape(NF, NB, NF)
        C[ho, :] = np.sum(B * B, axis=(0, 2))
    return (C + C.T) * 0.5


def build_chart(ops: dict[str, np.ndarray]) -> np.ndarray:
    C = np.zeros((NB, NB), dtype=np.float64)
    for W in ops.values():
        C += boundary_coupling_matrix(W)
        C += boundary_coupling_matrix(W.T)
    evals, evecs = np.linalg.eigh(C)
    idx = np.argsort(evals)[::-1]
    return evecs[:, idx].astype(np.float32)


def xor_aa_perm_U(nb: int = NB) -> np.ndarray:
    perm = np.array([i ^ 0xAA for i in range(nb)], dtype=np.int64)
    return np.eye(nb, dtype=np.float32)[perm]


def build_chart_from_mode(chart_mode: str, ops: dict[str, np.ndarray]) -> np.ndarray:
    mode = str(chart_mode)
    if mode == "dense":
        return build_chart(ops)
    if mode == "xor_aa":
        return xor_aa_perm_U(NB)
    raise ValueError(f"Unknown chart_mode '{mode}'")


def apply_chart(W: np.ndarray, U: np.ndarray) -> np.ndarray:
    d = NB * NF
    assert W.shape == (d, d)
    T = W.reshape(NB, NF, NB, NF)
    T1 = np.tensordot(U, T, axes=(1, 0))
    T2 = np.tensordot(T1, U.T, axes=(2, 0))
    T3 = np.transpose(T2, (0, 1, 3, 2))
    return T3.reshape(d, d).astype(np.float32)


def extract_phase_directions(Wp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract per-horizon direction vectors from charted operator Wp.
    Returns D[nb, nf] and per-horizon capture fractions.
    """
    T = Wp.reshape(NB, NF, NB, NF).astype(np.float32, copy=False)
    D = np.zeros((NB, NF), dtype=np.float32)
    frac = np.zeros(NB, dtype=np.float64)
    for h in range(NB):
        B = T[h].reshape(NF, NB * NF)
        U, S, _ = np.linalg.svd(B, full_matrices=False)
        d = U[:, 0].astype(np.float32, copy=False)
        if d[0] < 0.0:
            d = -d
        D[h] = d
        denom = float(np.sum(S * S)) + 1e-18
        frac[h] = float((S[0] * S[0]) / denom)
    return D, frac


def op_set_hash(ops: dict[str, np.ndarray]) -> str:
    payload = [f"{name}:{ops[name].shape}" for name in sorted(ops.keys())]
    return hashlib.sha256("|".join(payload).encode("utf-8")).hexdigest()


def main() -> None:
    out_path = OUT_PATH

    if not MODEL_DIR.exists():
        print(f"Error: {MODEL_DIR} not found.")
        return

    all_ops, all_phase = load_ops_from_model()
    ops, op_phase = select_ops_profile(all_ops, all_phase, FAMILY_PROFILE)
    if not ops:
        print("Error: no operators found.")
        return

    print(f"family_profile={FAMILY_PROFILE} operators={len(ops)} chart_mode={CHART_MODE}")
    U = build_chart_from_mode(CHART_MODE, ops)
    ortho = float(np.linalg.norm(U.T @ U - np.eye(NB, dtype=np.float32), ord="fro"))
    print(f"chart_orthogonality_error={ortho:.6e}")

    phase_accum = {
        0: np.zeros((NB * NF, NB * NF), dtype=np.float32),
        1: np.zeros((NB * NF, NB * NF), dtype=np.float32),
        2: np.zeros((NB * NF, NB * NF), dtype=np.float32),
        3: np.zeros((NB * NF, NB * NF), dtype=np.float32),
    }
    phase_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    for name, W in ops.items():
        Wt = apply_chart(W, U)
        phase = op_phase.get(name, -1)
        if phase in (0, 1, 2, 3):
            phase_accum[phase] += Wt
            phase_counts[phase] += 1

    D_phase = np.zeros((4, NB, NF), dtype=np.float32)
    mean_dir_frac = np.zeros(4, dtype=np.float64)
    min_dir_frac = np.zeros(4, dtype=np.float64)
    for p in range(4):
        if phase_counts[p] == 0:
            continue
        Wp = (phase_accum[p] / float(phase_counts[p])).astype(np.float32)
        Dp, frac = extract_phase_directions(Wp)
        D_phase[p] = Dp
        mean_dir_frac[p] = float(np.mean(frac))
        min_dir_frac[p] = float(np.min(frac))
        print(
            f"phase={p} direction_mean_frac={mean_dir_frac[p]:.6f} "
            f"direction_min_frac={min_dir_frac[p]:.6f} count={phase_counts[p]}"
        )

    min_mean = float(np.min(mean_dir_frac))
    min_min = float(np.min(min_dir_frac))
    print(f"min_direction_mean_frac={min_mean:.6f} min_direction_min_frac={min_min:.6f}")

    out: dict[str, np.ndarray] = {
        "adaptor_version": np.array(ADAPTOR_VERSION),
        "model_name": np.array(str(MODEL_DIR.name)),
        "nb": np.array(NB, dtype=np.int64),
        "nf": np.array(NF, dtype=np.int64),
        "R": np.array(R, dtype=np.int64),
        "operators": np.array(list(ops.keys())),
        "U": U.astype(np.float32),
        "R_fiber": np.eye(K, NF, dtype=np.float32),
        "D_phase": D_phase,
        "boundary_basis": np.array("lookup"),
        "family_profile": np.array(str(FAMILY_PROFILE)),
        "chart_mode": np.array(str(CHART_MODE)),
        "operator_set_hash": np.array(op_set_hash(ops)),
        "build_timestamp_utc": np.array(datetime.now(timezone.utc).isoformat()),
        "orthogonality_error": np.array(ortho, dtype=np.float64),
        "build_status": np.array("built"),
        "max_tail_fraction": np.array(0.0, dtype=np.float64),
        "direction_mean_frac_phase_0": np.array(mean_dir_frac[0], dtype=np.float64),
        "direction_mean_frac_phase_1": np.array(mean_dir_frac[1], dtype=np.float64),
        "direction_mean_frac_phase_2": np.array(mean_dir_frac[2], dtype=np.float64),
        "direction_mean_frac_phase_3": np.array(mean_dir_frac[3], dtype=np.float64),
        "direction_min_frac_phase_0": np.array(min_dir_frac[0], dtype=np.float64),
        "direction_min_frac_phase_1": np.array(min_dir_frac[1], dtype=np.float64),
        "direction_min_frac_phase_2": np.array(min_dir_frac[2], dtype=np.float64),
        "direction_min_frac_phase_3": np.array(min_dir_frac[3], dtype=np.float64),
        "phase_count_0": np.array(phase_counts[0], dtype=np.int64),
        "phase_count_1": np.array(phase_counts[1], dtype=np.int64),
        "phase_count_2": np.array(phase_counts[2], dtype=np.int64),
        "phase_count_3": np.array(phase_counts[3], dtype=np.int64),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), **out)  # type: ignore[arg-type]
    print(f"Saved adaptor to {out_path}")


if __name__ == "__main__":
    main()
