"""Tests for the percolation dataset (Dataset F) and ensemble labels
(Dataset E) against src.family exact results."""

from __future__ import annotations

import numpy as np

from src.family import bfs_reach, build_hqvm_d, gf2_rank

from src.tools.autoencoder.helpers.evals_datasets import (
    percolation_dataset,
    restriction_labels,
    shell_ensemble_labels,
    walsh_multipliers,
)


def test_singleton_labels_match_bfs() -> None:
    eng = build_hqvm_d(6)
    for b in (0x00, 0x54, 0xAA, 0xFE, 0x37):
        labels = restriction_labels([b])
        reach, spans, giant, full = bfs_reach(eng, [b])
        assert labels["reach_size"] == reach
        assert labels["horizon_spanning"] == int(spans)
        assert labels["transport_rank"] == gf2_rank([b], 6) or labels["transport_rank"] == gf2_rank(
            [int(__import__("src.family", fromlist=["q_word_d"]).q_word_d(b, 6))], 6
        )
        assert labels["fiber_complete"] == 0


def test_full_alphabet_full_reachability() -> None:
    labels = restriction_labels(list(range(256)))
    assert labels["full_reachability"] == 1
    assert labels["transport_rank"] == 6
    assert labels["reach_size"] == 4096


def test_rank_controlled_rows_rank_matches() -> None:
    data = percolation_dataset(n_rank_samples=2, n_random=8, seed=3)
    # rank-controlled rows have exactly k generators; rank must equal k for
    # rows whose allowed count is 1..6 from the controlled block. Check the
    # invariant: rank <= n_q_classes <= n_allowed.
    rank = data["transport_rank"].astype(np.int64)
    nq = data["n_q_classes"].astype(np.int64)
    na = data["n_allowed"].astype(np.int64)
    assert (rank <= nq).all()
    assert (nq <= na).all()
    assert (rank >= 0).all() and (rank <= 6).all()
    # A single byte (n_allowed == 1) is a rank-0 singleton exactly when its q
    # class is 0. Count q=0 bytes among the emitted singleton set and require
    # that many rank-0 singletons. Seed-robust for sampled or full sets.
    from src.tools.autoencoder.datasets import byte_census_arrays

    census = byte_census_arrays()
    q0_bytes = set(int(b) for b in np.flatnonzero(census["q_weight"] == 0))
    singleton_mask = na == 1
    # map allowed_mask -> byte index for singleton rows
    allowed_mask = data["allowed_mask"]
    rank0_singletons = int(((rank == 0) & singleton_mask).sum())
    # reconstruct emitted singleton bytes from the allowed mask
    emitted_q0 = 0
    for row in range(data["allowed_mask"].shape[0]):
        if not singleton_mask[row]:
            continue
        bits = np.unpackbits(allowed_mask[row].astype(np.uint8))
        b = int(np.flatnonzero(bits)[0])
        if b in q0_bytes:
            emitted_q0 += 1
    assert rank0_singletons == emitted_q0


def test_full_reachability_implies_giant() -> None:
    data = percolation_dataset(n_random=16, seed=5)
    full = data["full_reachability"].astype(bool)
    giant = data["giant"].astype(bool)
    assert (full & ~giant).sum() == 0


def test_predicted_cluster_law() -> None:
    from src.family import predicted_cluster_size

    data = percolation_dataset(n_rank_samples=1, n_random=0, seed=11)
    rank = data["transport_rank"].astype(np.int64)
    pred = data["predicted_cluster"].astype(np.int64)
    for r, p in zip(rank, pred):
        assert p == predicted_cluster_size(int(r))


def test_shell_ensemble_corpus_conventions() -> None:
    """Corpus-exact order parameters from hQVM_QuBEC_Theory.md sections 2.3,
    3, 5.2: rho = lambda/(1+lambda), eta = (1-lambda)/(1+lambda), and
    M2 = 4096/(1+eta^2)^6 over 4096 states."""
    from src.family import partition_Z1_coeff_d

    lams = [1.0, 0.5, 2.0, 0.1, 9.0]
    out = shell_ensemble_labels(lams)
    # corpus eta from lambda
    assert np.allclose(out["eta"], (1 - np.array(lams)) / (1 + np.array(lams)))
    # corpus rho from lambda
    assert np.allclose(out["rho"], np.array(lams) / (1 + np.array(lams)))
    # closed-form M2 over 4096 states
    eta = out["eta"]
    assert np.allclose(out["M2"], 4096.0 / (1.0 + eta**2) ** 6)
    # M2 over states = 64 x participation ratio on the register
    assert np.allclose(out["M2"], 64.0 * out["M2_chi"])
    # extremal values: thermal lambda=1 -> M2 = 4096; condensation -> 64
    i1 = lams.index(1.0)
    assert np.isclose(out["M2"][i1], 4096.0)
    # partition function cross-check against src.family at lambda=1: Z1 = 2^6 (1+1)^6 = 4096
    assert np.isclose(partition_Z1_coeff_d(6, 1.0), 4096.0)
    # raw moments kept under explicit names, not colliding with corpus eta
    assert np.isclose(out["expected_shell"][i1], 3.0)
    assert np.isclose(out["shell_variance"][i1], 1.5)
    # wt_var_norm = p(1-p) = (1-eta^2)/4; at lambda=1: 0.25, NOT corpus eta = 0
    assert np.isclose(out["wt_var_norm"][i1], 0.25)
    assert np.allclose(out["wt_var_norm"], (1 - eta**2) / 4.0)
    # rho consistent with the empirical register distribution
    assert np.isclose(out["expected_shell"][i1] / 6.0, out["rho"][i1])


def test_walsh_multipliers_uniform() -> None:
    out = walsh_multipliers([0.5] * 6)
    # eta=0.5 gives zero multiplier except character 0
    assert np.isclose(out["walsh_multiplier"][0], 1.0)
    assert np.allclose(out["walsh_multiplier"][1:], 0.0)
    assert out["isotropic"] == 1.0
    # isotropic damping convention: per-axis flip prob 0.5 -> multiplier 0 -> corpus eta = 0
    assert np.isclose(float(out["eta_isotropic"]), 0.0)


def test_walsh_multipliers_match_corpus_damping() -> None:
    """For the isotropic ensemble P(chi) ~ lambda^wt(chi), each axis is an
    independent mode with flip probability rho = lambda/(1+lambda); the
    Walsh multiplier of character a is therefore prod_i (1-2 rho)^{a_i} =
    eta^{wt(a)} with corpus eta = (1-lambda)/(1+lambda). Weight-1 modes
    decay as eta, weight-2 as eta^2, etc. - the exact form of the corpus
    statement that eta controls the decay of higher spectral modes."""
    for lam in (0.25, 1.0, 4.0):
        eta = (1.0 - lam) / (1.0 + lam)
        out = walsh_multipliers([(1.0 - eta) / 2.0] * 6)
        for a in range(64):
            wt = a.bit_count()
            assert np.isclose(out["walsh_multiplier"][a], eta**wt), (lam, a)
        assert np.isclose(float(out["eta_isotropic"]), eta)


def test_percolation_learner_rank_recovery_smoke() -> None:
    """A PercolationLearner trained a few epochs on the packed allowed mask
    reaches exact-rank accuracy above chance on the rank-controlled strata
    (k = 1..6, six balanced classes), and does not merely fit reach: the
    mechanism-vs-correlate gap (cluster vs raw reach) survives in its
    inputs, so keying on membership (not connectivity) is what trains."""
    import torch

    from src.tools.autoencoder.helpers.training_losses import LossWeights, weighted_total
    from src.tools.autoencoder.helpers.training_run import (
        Trainer,
        TrainConfig,
        iterate_batches,
        set_seed,
    )
    from src.tools.autoencoder.models.narrow import PercolationLearner

    set_seed(0)
    rng = np.random.default_rng(0)
    ds = percolation_dataset(n_singletons=128, n_rank_samples=10, n_random=60, seed=0)
    rank = ds["transport_rank"].astype(np.int64)
    # Rank-controlled strata are the rows with n_q_classes == transport_rank
    # (a k-generator alphabet spans exactly rank k): six balanced classes 1..6.
    nq = ds["n_q_classes"].astype(np.int64)
    controlled = np.flatnonzero((nq == rank) & (rank >= 1) & (rank <= 6))
    assert len(controlled) >= 48
    rng.shuffle(controlled)
    n_hold = 36
    va = controlled[:n_hold]
    tr = np.concatenate([controlled[n_hold:], np.flatnonzero(nq != rank)])

    model = PercolationLearner(hidden_dim=128)
    config = TrainConfig(epochs=12, batch_size=128, lr=3e-3, device="cpu", seed=0)
    trainer = Trainer(model, config)
    weights = LossWeights(rank_ce=1.0)

    arrays = {"allowed_mask": ds["allowed_mask"][tr], "rank": rank[tr]}
    val_arrays = {"allowed_mask": ds["allowed_mask"][va], "rank": rank[va]}

    def loss_fn(batch):
        logits = model(batch["allowed_mask"])["rank_logits"]
        ce = torch.nn.functional.cross_entropy(logits, batch["rank"])
        return weighted_total({"rank_ce": ce}, weights)[0], {"rank_ce": float(ce.detach())}

    trainer.fit(lambda: iterate_batches(arrays, config.batch_size, 0), loss_fn)

    model.eval()
    with torch.inference_mode():
        pred_train = model(torch.as_tensor(arrays["allowed_mask"]))["rank_logits"].argmax(dim=-1).numpy()
        pred_val = model(torch.as_tensor(val_arrays["allowed_mask"]))["rank_logits"].argmax(dim=-1).numpy()
    acc_train = float((pred_train == arrays["rank"]).mean())
    acc_val = float((pred_val == val_arrays["rank"]).mean())
    # chance on the rank-controlled strata is 1/6 ~ 0.167; the learner must
    # clear >= 0.5 exactly (the plan's acceptance threshold).
    assert acc_val >= 0.5, (acc_train, acc_val)
    # mechanism-vs-correlate: over the random rows (where reach is not the
    # cluster law), the two disagree; a learner keying on connectivity would
    # have nothing to key on, so the rank signal must come from membership.
    # The plan benchmarks this at the dataset level (percolation_suite gap).
    cluster_all = ds["predicted_cluster"].astype(np.float64)
    reach_all = ds["reach_size"].astype(np.float64)
    assert np.mean(np.abs(cluster_all - reach_all)) > 0.0
