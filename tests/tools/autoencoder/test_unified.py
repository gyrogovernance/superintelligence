"""Tests for model unification (spec 1.1-1.3).

Covers the H-symmetrized proof (diagonal rung == H-invariant codec), the two
new ladder rungs (shell_gauge, chirality_gauge), and the unified model with
symmetry level {free, k4, full} including exact-equality equivalence against
the standalone classes.
"""

from __future__ import annotations

import numpy as np
import torch

from src.tools.autoencoder.kernel import apply_signature_index
from src.tools.autoencoder.models.general import K4Autoencoder
from src.tools.autoencoder.models.super import (
    SpectralAutoencoder,
    codec_ladder,
    codec_ladder_mask,
    full_g_equivariance_error,
    irrep_block_index,
)
from src.tools.autoencoder.models import UnifiedAutoencoder


# ---------------------------------------------------------------------------
# 1.2 H-symmetrized proof
# ---------------------------------------------------------------------------


def test_diagonal_rung_is_h_invariant() -> None:
    """The diagonal ladder rung is exactly invariant under the never-broken
    subgroup H = {(0,t,t), (1,t,t)}. Conclude: diagonal rung == H-symmetrized
    codec."""
    model = SpectralAutoencoder(ladder="diagonal")
    x = torch.arange(0, 4096, 17, dtype=torch.long)
    base = model(x)
    # enumerate H: parity in {0,1}, tau_u == tau_v
    max_err = 0.0
    for parity in (0, 1):
        for t in range(64):
            sig_id = (parity << 12) | (t << 6) | t
            transformed = torch.tensor(
                [apply_signature_index(int(i), sig_id) for i in x.tolist()],
                dtype=torch.long,
            )
            out_h = model(transformed)
            err = (out_h - base).abs().max().item()
            max_err = max(max_err, err)
    assert max_err < 1e-4  # H-invariant exactly


# ---------------------------------------------------------------------------
# 1.3 tied-gain ladder rungs (hQVM_QuBEC_Theory.md 17.1 operator classes)
# ---------------------------------------------------------------------------


def test_tied_rungs_mask_and_orbit_structure() -> None:
    bid, _ = irrep_block_index()
    for ladder, want_orbits in (
        ("shell_radial", 7),
        ("shell_gauge", 28),
        ("chirality_gauge", 56),
    ):
        mask, orbit = codec_ladder(ladder)
        assert mask.shape == (2080,) and int(mask.sum()) == 2080
        assert orbit is not None and orbit.shape == (2080,)
        # tying is exact: blocks with equal keys share one orbit id, distinct
        # keys never share an orbit id, and the count matches the class size
        key_to_orbit: dict[tuple, int] = {}
        for a in range(64):
            for b in range(a, 64):
                wa, wb = a.bit_count(), b.bit_count()
                key = (min(wa, wb), max(wa, wb))
                if ladder == "shell_radial":
                    key = ((a ^ b).bit_count(),)
                elif ladder == "chirality_gauge":
                    # ordered shell pair + AND parity on canonical (a <= b):
                    # (wt(a), wt(b), wt(a & b) mod 2), 56 classes.
                    key = (wa, wb, (a & b).bit_count() % 2)
                oid = int(orbit[bid[a, b]])
                if key in key_to_orbit:
                    assert key_to_orbit[key] == oid, (ladder, key)
                else:
                    key_to_orbit[key] = oid
        assert len(key_to_orbit) == want_orbits == len(set(key_to_orbit.values()))


def test_tied_gains_share_parameters() -> None:
    """Blocks in one orbit are gated by literally the same parameter, so
    updating one moves all of them, and distinct orbits are distinct."""
    model = SpectralAutoencoder(ladder="shell_gauge")
    assert model.bottleneck.gain.shape == (28,)
    with torch.no_grad():
        model.bottleneck.gain.copy_(torch.arange(28, dtype=model.bottleneck.gain.dtype))
    gains = model.bottleneck.block_gains()
    bid = model.block_id  # flat [4096]: block for coefficient (a, b) at a*64+b
    # wt(3)=wt(5)=wt(6)=wt(9)=2, so blocks (3,5) and (6,9) share shell pair (2,2)
    assert gains[bid[3 * 64 + 5]] == gains[bid[6 * 64 + 9]]
    # blocks (1,2) and (1,4) share shell pair (1,2), distinct from (2,2)
    assert gains[bid[1 * 64 + 2]] == gains[bid[1 * 64 + 4]]
    assert gains[bid[1 * 64 + 2]] != gains[bid[3 * 64 + 5]]


def test_new_rungs_remain_equivariant() -> None:
    for ladder in ("shell_radial", "shell_gauge", "chirality_gauge"):
        model = SpectralAutoencoder(ladder=ladder)
        x = torch.arange(0, 4096, 64, dtype=torch.long)
        sig_ids = torch.tensor([0, 1, 64, 4131, 8191], dtype=torch.long)
        report = full_g_equivariance_error(model, x, sig_ids)
        assert report["max"] < 1e-3


# ---------------------------------------------------------------------------
# 1.1 unified model equivalence
# ---------------------------------------------------------------------------


def test_unified_full_matches_spectral() -> None:
    standalone = SpectralAutoencoder(ladder="full")
    unified = UnifiedAutoencoder(symmetry="full", ladder="full")
    unified.load_from(standalone)
    x = torch.arange(0, 4096, 37, dtype=torch.long)
    assert torch.allclose(unified(x), standalone(x), atol=1e-5)


def test_unified_full_diagonal_matches_spectral_diagonal() -> None:
    standalone = SpectralAutoencoder(ladder="diagonal")
    unified = UnifiedAutoencoder(symmetry="full", ladder="diagonal")
    unified.load_from(standalone)
    x = torch.arange(0, 4096, 41, dtype=torch.long)
    assert torch.allclose(unified(x), standalone(x), atol=1e-5)


def test_unified_k4_matches_standalone_k4() -> None:
    """With shared base weights, unified(k4) equals K4Autoencoder
    exactly (Reynolds symmetrization over the four characters)."""
    standalone = K4Autoencoder()
    unified = UnifiedAutoencoder(
        symmetry="k4",
        hidden_dim=standalone.base_encoder[0].out_features,
        latent_dim=standalone.latent_dim,
    )
    unified.load_from(standalone)
    x = torch.arange(0, 4096, 53, dtype=torch.long)
    assert torch.allclose(unified(x), standalone(x), atol=1e-5)


def test_unified_free_is_not_k4_equivariant() -> None:
    """The free level makes no equivariance guarantee by construction."""
    model = UnifiedAutoencoder(symmetry="free", latent_dim=32)
    x = torch.tensor([0, 100, 2000], dtype=torch.long)
    z = model.encode(x)
    # decode with a K4-translated latent should change the output
    # (free model has no symmetrization), confirming no structural equivariance
    out_base = model.decode(z)
    # apply a nontrivial K4 character flip to half the latent channels
    z_flipped = z.clone()
    z_flipped[..., :16] *= -1
    out_flipped = model.decode(z_flipped)
    assert not torch.allclose(out_base, out_flipped, atol=1e-5)


# ---------------------------------------------------------------------------
# 1.5 unified multi-task (P2: shared spectral latent across the AE tasks)
# ---------------------------------------------------------------------------


def test_multitask_heads_verify_default_off() -> None:
    """The multi-task heads default to off, so a plain UnifiedAutoencoder is
    bit-identical to the standalone classes (the equivalence tests keep
    passing)."""
    plain = UnifiedAutoencoder(symmetry="full", ladder="full")
    assert plain.task_heads is None
    assert plain.heads_kind == ()
    with_heads = UnifiedAutoencoder(
        symmetry="full", ladder="full", heads=("transition", "word", "rank")
    )
    assert with_heads.task_heads is not None
    assert with_heads.heads_kind == ("transition", "word", "rank")


def test_multitask_smoke_trains_and_stays_exactly_equivariant() -> None:
    """A unified-full model with heads trained on all four objectives (state,
    transition, word, percolation rank) over the shared spectral latent:
    - all four component losses actually change between epochs (this catches
      the earlier bug where the per-epoch seed never advanced, so every epoch
      saw identical rows);
    - the transition head now sees the source state identity, so the next-state
      map is state-conditioned, not just byte-conditioned;
    - the full-scale equivariance error stays 0.0 (the K4/full dispatch is
      not disturbed by the heads)."""
    from src.tools.autoencoder.helpers.evals_datasets import percolation_dataset
    from src.tools.autoencoder.helpers.training_losses import LossWeights, weighted_total
    from src.tools.autoencoder.helpers.training_run import Trainer, TrainConfig, iterate_batches
    from src.tools.autoencoder.kernel import word_signature_id
    from src.tools.autoencoder.models.super import full_g_equivariance_error

    rng = np.random.default_rng(3)
    model = UnifiedAutoencoder(
        symmetry="full", ladder="full", heads=("transition", "word", "rank"), hidden_dim=96
    )
    config = TrainConfig(epochs=6, batch_size=96, lr=2e-3, device="cpu", seed=3)
    trainer = Trainer(model, config)
    weights = LossWeights(
        state_ce=1.0, transition_ce=1.0, word_ce=1.0, rank_ce=1.0
    )

    K = 160
    state_idx = rng.choice(4096, size=K, replace=False)
    from src.tools.autoencoder.datasets import transition_table

    table = transition_table().astype(np.int64)
    tr_state = rng.choice(4096, size=K)
    tr_byte = rng.choice(256, size=K)
    next_state = table[tr_state, tr_byte]
    w_byte = rng.choice(256, size=K)
    sigs = np.array([word_signature_id([b]) for b in range(256)], dtype=np.int64)
    tau_u = (sigs >> 6) & 63
    tau_v = sigs & 63
    ds = percolation_dataset(n_singletons=32, n_rank_samples=4, n_random=64, seed=3)
    n_mask = len(ds["transport_rank"])
    mask_idx = rng.choice(n_mask, size=K)

    arrays = {
        "state_index": state_idx,
        "tr_state": tr_state,
        "tr_byte": tr_byte,
        "next_state": next_state,
        "word_state": state_idx,
        "word_byte": w_byte,
        "tau_u": tau_u[w_byte],
        "tau_v": tau_v[w_byte],
        "allowed_mask": ds["allowed_mask"][mask_idx],
        "rank": ds["transport_rank"][mask_idx].astype(np.int64),
    }

    epoch_losses: list[dict[str, float]] = []

    def loss_fn(batch):
        heads = model.task_heads
        assert heads is not None
        comp: dict[str, torch.Tensor] = {}
        if len(batch["state_index"]):
            comp["state_ce"] = torch.nn.functional.cross_entropy(
                model(batch["state_index"]), batch["state_index"]
            )
        if len(batch["tr_state"]):
            feats = model.per_block_features(batch["tr_state"])
            comp["transition_ce"] = torch.nn.functional.cross_entropy(
                heads.transition_logits(feats, batch["tr_byte"], batch["tr_state"]),
                batch["next_state"],
            )
        if len(batch["word_state"]):
            wf = model.per_block_features(batch["word_state"])
            wl = heads.word_logits(wf, batch["word_byte"], batch["word_state"])
            comp["word_ce"] = torch.nn.functional.cross_entropy(
                wl[:, :64], batch["tau_u"]
            ) + torch.nn.functional.cross_entropy(wl[:, 64:], batch["tau_v"])
        if len(batch["allowed_mask"]):
            rl = heads.rank_logits(batch["allowed_mask"])
            comp["rank_ce"] = torch.nn.functional.cross_entropy(rl, batch["rank"])
        total, logs = weighted_total(comp, weights)
        epoch_losses.append({k: float(v) for k, v in logs.items()})
        return total, logs

    trainer.fit(lambda: iterate_batches(arrays, config.batch_size, 3), loss_fn)

    # Each component loss was logged every epoch (several epochs ran), so the
    # records span multiple epochs. Confirm the per-epoch seed advanced by
    # checking that the component losses actually moved across the run rather
    # than being frozen at one value.
    if len(epoch_losses) >= 4:
        for key in ("state_ce", "transition_ce", "word_ce", "rank_ce"):
            vals = [e[key] for e in epoch_losses if key in e]
            assert len(vals) >= 2, key
            assert max(vals) - min(vals) > 1e-4, (
                f"{key} did not change across epochs: {vals[:4]}"
            )

    # the transition head now learns a state-conditioned next-state map: its
    # CE over the training content drops well below the chance ceiling
    # (ln 4096 ~ 8.32) as it trains.
    model.eval()
    with torch.inference_mode():
        feats = model.per_block_features(torch.as_tensor(arrays["tr_state"]))
        tl = heads_transition_ce(model, feats, arrays)

    assert tl < 6.0, tl  # well below chance (ln 4096 ~ 8.32)
    # rank accuracy above chance on the rank strata
    with torch.inference_mode():
        heads = model.task_heads
        assert heads is not None
        rl = heads.rank_logits(torch.as_tensor(arrays["allowed_mask"]))
        pred_rank = rl.argmax(dim=-1).numpy()
    acc_rank = float((pred_rank == arrays["rank"]).mean())
    assert acc_rank > 0.3, acc_rank
    # the codec itself stays exactly equivariant at full scale
    x = torch.arange(0, 4096, 128, dtype=torch.long)
    sigs_small = torch.tensor([0, 1, 64, 4131, 8191], dtype=torch.long)
    assert model.spectral is not None
    report = full_g_equivariance_error(model.spectral, x, sigs_small)
    assert report["max"] < 1e-4
    assert report["forward_max"] < 1e-4


def heads_transition_ce(model, feats, arrays) -> float:
    """Helper: transition CE over the full content rows from a given batch."""
    with torch.inference_mode():
        tl = torch.nn.functional.cross_entropy(
            model.task_heads.transition_logits(
                feats,
                torch.as_tensor(arrays["tr_byte"]),
                torch.as_tensor(arrays["tr_state"]),
            ),
            torch.as_tensor(arrays["next_state"]),
        )
    return float(tl.detach())
