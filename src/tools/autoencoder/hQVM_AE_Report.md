# hQVM Autoencoder: Test and Evaluation Report

This report inventories what the autoencoder package tests, what the evaluation helpers measure, and what the shipped production artifacts record.

**Product naming:** the suite is an exactly group-equivariant autoencoder package over a finite group-structured state space, with an exhaustive closed-form equivariance certificate, a Bayes-optimal spectral denoiser with analytic gains, and a hybrid Super process model: exact hQVM grammar and signature registers, learned provenance, a gated analytic climate head measured against the QuBEC λ ceiling, and a sequential residual on prefix context.

---

## 1. Test inventory by file

### test_anomaly.py (12 tests)

**Covers:** anomaly and perturbation datasets in `evals_datasets.py`, plus Walsh sector energy and shell distribution diagnostics in `evals_metrics.py`.

| Test | What it checks | Pass criterion |
|------|----------------|----------------|
| `test_biased_family_bytes_single_family` | Biased byte sampler draws from one intron family | All sampled bytes belong to the requested family |
| `test_biased_q_weight_bytes_exact_weight` | Q-weight biased sampler | Every byte has the requested Q popcount |
| `test_missing_q_class_alphabet_rank_and_labels` | Alphabet with one Q class removed | Rank and label arrays match kernel percolation logic |
| `test_gf2_rank_known_answers` | GF(2) rank helper | Matches hand-checked small matrices |
| `test_corrupted_mask_dataset_syndromes` | Mask-corruption dataset | Syndrome labels consistent with allowed-byte mask |
| `test_byte_perturbation_dataset_shadow_preserves_signature` | Byte perturbation corpus | Terminal signature unchanged under shadow perturbation |
| `test_byte_perturbation_labels_consistent_with_kernel` | Perturbation labels | Labels match kernel replay |
| `test_shell_distribution_ensemble_statistics` | Shell histogram ensemble | Mean and variance within expected ranges on lambda corpus |
| `test_walsh_sector_energy_partition` | Walsh sector energy | Energies partition the spectrum correctly |
| `test_hqvm_d_dataset_shapes_and_closure` | Dimension-d transition dataset | Shapes and closure properties hold |
| `test_hqvm_d6_matches_api_transitions` | d=6 dataset vs kernel API | Transition table matches `src.api` |
| `test_dimension_transfer_probe_structure` | Cross-dimension probe | Train/test split structure is well-formed |

**Result:** All pass.

---

### test_benchmarks.py (3 tests)

**Covers:** benchmark suites in `evals_run.py` (spec sections 3.1 and 3.3).

| Test | What it checks | Pass criterion |
|------|----------------|----------------|
| `test_percolation_suite_threshold_and_anchor` | `percolation_suite(seed=7)` | Threshold accuracy 1.0; singleton rank 1.0; mechanism gap >= 0 |
| `test_climate_sweep_regimes` | `climate_sweep([0.1, 1.0, 10.0])` | rho increases across lambda; regimes are condensed / thermal / condensed |
| `test_benchmarks_container_runs_all` | `Benchmarks()` container | Percolation and climate sub-suites return expected keys |

**Result:** All pass.

---

### test_byte_mechanism.py (8 tests)

**Covers:** byte mechanism datasets, factorization probes, shadow invariance, depth-4 frames, and narrow-tier heads.

| Test | What it checks | Pass criterion |
|------|----------------|----------------|
| `test_raw_byte_model_forward_shape` | `RawByteTransitionModel` output shape | Correct logits shape |
| `test_raw_byte_model_learns_transitions_smoke` | Short training on transition task | Loss decreases; parameters change |
| `test_factorization_targets_shape` | Factorization target arrays | Expected shapes |
| `test_factorization_probe_recovers_pairs_from_oracle_latent` | Linear probe on oracle latent | Recovers L/R factorization pairs |
| `test_shadow_invariance_zero_for_exact_table` | Shadow invariance metric | Zero error on kernel-exact transition table |
| `test_shadow_invariance_nonzero_for_untrained_mlp` | Contrast on untrained MLP | Error > 0 |
| `test_depth4_frame_signature_consistency` | Depth-4 frame dataset | Frame signatures match kernel replay |
| `test_frame_head_learns_signature_smoke` | `FrameHead` short training | Loss decreases |

**Result:** All pass.

---

### test_canonical_words.py (4 tests)

**Covers:** canonical K4/BU word dataset and word replay effects.

| Test | What it checks | Pass criterion |
|------|----------------|----------------|
| `test_canonical_words_rows_and_types` | Word table schema | Row count and field types |
| `test_word_k4_differs_from_gate_k4_except_f` | Word K4 vs gate K4 | Different except on gate F |
| `test_word_replay_effects_consistent` | Replay effects vs dataset | Involution, chirality, shell labels match |
| `test_sig_ids_replay` | Signature ID replay | End state matches kernel composition |

**Result:** All pass.

---

### test_codecs.py (4 tests)

**Covers:** deterministic narrow-tier chart codecs.

| Test | What it checks | Pass criterion |
|------|----------------|----------------|
| `test_exact_uv_codec_lossless` | ExactUV codec | Perfect round-trip on all 4096 states |
| `test_boundary_chirality_codec_lossless` | BoundaryChirality codec | Perfect round-trip |
| `test_chirality_only_codec_64_state_ambiguity` | ChiralityOnly codec | 64 states per chirality class (expected ambiguity) |
| `test_shell_only_codec_binomial_fibers` | ShellOnly codec | Fiber sizes follow binomial law |

**Result:** All pass.

---

### test_dataset_generation.py (23 tests)

**Covers:** generated census arrays, transition tables, K4 actions, signatures, manifests, and invariant checks in `datasets.py`.

Key checks include: byte census shape and kernel column match; family decomposition; full column set; chirality chart via bit ops; stepping paths agree with kernel; transport law; signature apply matches word replay; state census shape, index, shell fields, observables, K4 orbit partition, spin roundtrip; transition table and inverse roundtrip; K4 action arrays; signature dataset values and rest action; manifest save/load; invariant check failure reporting; entry byte count; generate determinism.

**Result:** All pass.

---

### test_dictionary.py (6 tests)

**Covers:** embedding corpus export (`corpus.py`) and dictionary audit (`evals_run.audit_dictionary`).

| Test | What it checks | Pass criterion |
|------|----------------|----------------|
| `test_export_embeddings_labels_match_census` | Export arrays vs census | Shapes match; byte_mask12 equals census |
| `test_audit_dictionary_green_on_exact_model` | Audit on identity AffineSpectralCodec | `passed` true; reconstruction, equivariance, H-invariance, shadow, frame parity, psi_hat gates pass |
| `test_audit_dictionary_on_tiny_trained_model` | Audit on 3-epoch trained AffineSpectralCodec | Equivariance still passes |
| `test_audit_gate_requires_only_core_invariants` | Gate logic | Only boolean core checks block `passed` |
| `test_audit_gate_fails_on_nonboolean_required` | Gate logic | Non-boolean required fields fail correctly |
| `test_audit_informational_pass_does_not_gate` | Informational checks | Probe accuracy reported but does not gate pass |

**Result:** All pass.

---

### test_ensembles.py (6 tests)

**Covers:** lambda-ensemble corpora and symmetry-breaking characterization in `evals_datasets.py`.

| Test | What it checks | Pass criterion |
|------|----------------|----------------|
| `test_lambda_chirality_distribution_normalizes` | P(chi) distribution | Sums to 1 |
| `test_chi_fiber_partitions_omega` | Chi fiber over Omega | Partitions the 4096 states |
| `test_sample_lambda_corpus_matches_expected_shell` | Sampled corpus | Shell histogram matches theory at given lambda |
| `test_stabilizer_characterization` | Stabilizer test helper | Correct membership |
| `test_never_broken_group_exhaustive` | Never-broken subgroup | Exhaustive composition closure |
| `test_w2_signature_ids_all_broken_shape` | W2 brokenness set | Expected count and shape |

**Result:** All pass.

---

### test_equivariance.py (7 tests)

**Covers:** exact K4 equivariance on `K4Autoencoder` (spec 6.4), exhaustive over all 4096 states and 4 gates.

| Test | What it checks | Pass criterion |
|------|----------------|----------------|
| `test_characters_are_group_homomorphisms` | K4 character table | Multiplication and involution laws |
| `test_rho_is_representation` | Latent action rho | rho(g) rho(h) = rho(gh) |
| `test_encoder_exactly_equivariant_exhaustive` | Encoder E(g.x) = rho(g) E(x) | Max error < 1e-4 per gate |
| `test_decoder_exactly_equivariant_exhaustive` | Decoder D(rho(g)z) = P_g D(z) | Max error < 1e-4 per gate |
| `test_equivariance_error_metrics` | `k4_equivariance_error` report | Max < 1e-4 |
| `test_reconstruction_is_exact_for_identity_latent` | Output validity | Finite logits; valid argmax indices |
| `test_k4_action_matrix_consistent` | K4 permutation matrix | Matches kernel apply |

**Result:** All pass.

---

### test_group_actions.py (6 tests)

**Covers:** kernel adapter indexing and group actions in `kernel.py`.

Checks: state index roundtrip for all 4096 states; step index matches kernel for sampled bytes; signature apply matches kernel; signature IDs enumerate the group; K4 orbit consistency; word signature ID matches replay.

**Result:** All pass.

---

### test_latent_components.py (4 tests)

**Covers:** named K4 latent components (`z_invariant`, `z_char_S`, `z_char_C`, `z_char_F`) and transition equivariance.

| Test | What it checks | Pass criterion |
|------|----------------|----------------|
| `test_named_latent_components_cover_latent` | Component partition | Named parts span full latent |
| `test_named_components_shapes_and_equivariance` | Per-component shapes | Each component equivariant under K4 |
| `test_transition_equivariance_zero_for_exact_kernel` | Transition head on exact model | Zero equivariance defect |
| `test_transition_equivariance_detects_untrained_model` | Contrast | Untrained model has nonzero defect |

**Result:** All pass.

---

### test_load.py (8 tests)

**Covers:** checkpoint save/load roundtrip and `build_model` config propagation.

Checks: task model get_config roundtrip for transition, rawbyte, word, percolation; K4 n_trivial/n_sign roundtrip for several configurations; build_model forwards n_trivial/n_sign.

**Result:** All pass.

---

### test_models.py (5 tests)

**Covers:** base model utilities and soft equivariance loss wiring.

Checks: state_to_bits roundtrip; MLP forward shapes; K4 generator batch matches kernel; soft equivariance loss finite; soft equivariance loss requires explicit latent_action.

**Result:** All pass.

---

### test_percolation.py (9 tests)

**Covers:** percolation datasets, shell ensemble labels, Walsh multipliers, and `PercolationLearner` smoke training.

Key checks: singleton labels match BFS; full alphabet gives full reachability; rank-controlled rows match declared rank; full reachability implies giant component; predicted cluster law; shell ensemble corpus conventions (rho, eta, M2); Walsh multipliers uniform and match corpus damping; learner rank recovery smoke.

**Result:** All pass.

---

### test_percolation_seed.py (2 tests)

**Covers:** determinism of percolation evaluation holdout split.

| Test | What it checks | Pass criterion |
|------|----------------|----------------|
| `test_percolation_eval_seed_deterministic` | Same seed | Identical holdout |
| `test_percolation_eval_seed_changes_holdout` | Different seed | Different holdout |

**Result:** All pass.

---

### test_psi_hat.py (4 tests)

**Covers:** Agrawal symmetry-breaking order parameter `psi_hat` in `evals_metrics.py`.

| Test | What it checks | Pass criterion |
|------|----------------|----------------|
| `test_psi_hat_unit_for_exact_encoder` | Exact K4 encoder | \|psi\| = 1 for broken generator |
| `test_psi_hat_trivial_generator_is_plus_one` | Trivial generator | psi = +1 |
| `test_psi_hat_single_character_encoder_is_signed_one` | Single-character latent | \|psi\| = 1 with correct sign |
| `test_psi_hat_unstructured_encoder_near_zero` | Random MLP | \|psi\| well below 1 |

**Result:** All pass.

---

### test_readouts.py (9 tests)

**Covers:** kernel-exact readouts in `evals_metrics.py` (spec sections 2.1 through 2.6 and climate synthesizer).

| Test | What it checks | Pass criterion |
|------|----------------|----------------|
| `test_climate_readout_rho_eta_m2` | Climate readout | rho, eta, M2 match theory on lambda grid |
| `test_plancherel_consistency_exact` | Plancherel law | Shell histogram consistent with weight moments |
| `test_anisotropy_eta_wt_law` | Anisotropic bath | eta vs weight law |
| `test_gauge_character_readout` | Gauge character | Matches kernel character table |
| `test_z2_sheet_readout` | Z2 sheet labels | Correct sheet assignment |
| `test_lift32_readout` | 32-bit lift | Membership arrays correct |
| `test_code_readout_membership` | Code sector | Membership matches kernel code map |
| `test_exact_denoiser_multipliers` | Closed-form denoiser | Multipliers match theory |
| `test_climate_synthesizer` | Climate synthesizer | Reconstructs target climate from Walsh gains |

**Result:** All pass.

---

### test_plan_completion.py

**Covers:** LUT scan vs kernel, prefix factor table, signature posterior vs short-word enumeration, write-protected caches, ladder aliases including `shell_climate`, `super_all` one-epoch checkpoint, encode/decode.

**Result:** Asserted by pytest.

---

### test_report_schemas.py (5 tests)

**Covers:** production JSON report schemas under `data/reports/`. Does not retrain; reads existing files.

Pinned reports: `k4_full_eval.json`, `mlp_full_eval.json`, `mlp_full_eq.json`, `super_gates.json` (requires `gate_version`), `production_summary.json` (keys include `k4_full`, `mlp_full`, `super`).

Additional check: all paths in `production_summary.json` use forward slashes.

**Result:** Pass when reports present; individual schema tests skip if file missing.

---

### test_scale.py (5 tests, 1 slow)

**Covers:** scale diagnostics, closed-form full-G verifier.

| Test | What it checks | Pass criterion |
|------|----------------|----------------|
| `test_operator_structure_commutant` | Operator structure readout | Commutant dimensions |
| `test_scalesuite_runs` | ScaleSuite container | All sub-readouts execute |
| `test_exhaustive_full_g_verify` | Full 4096 x 8192 sweep | `passed` true (**runslow**) |
| `test_verify_full_g_exhaustive_cli_handler` | Closed-form CLI verifier | Exit code 0, sub-second |

**Result:** 3 pass in default run; 1 requires `--runslow`.

---

### test_affine_codec.py (13 tests)

**Covers:** `AffineSpectralCodec`: Walsh transform, irrep blocks, spectral action, full-G equivariance sample, bottleneck, codec ladder masks.

Key thresholds: Walsh matrix orthogonal; 2080 blocks cover all frequencies; one-hot Walsh coeffs match kernel sign; roundtrip exact; spectral action matches translation; full-G equivariance sampled error zero; signature composition in spectrum; bottleneck preserves equivariance; ladder masks have correct structure; frozen mask zeros sectors; rate penalty ignores masked blocks; diagonal model equivariant exactly.

**Result:** All pass.

---

### test_null_corpus.py (5 tests)

**Covers:** Stage-0 null corpus invariants in `datasets.py`: byte fiber, canonical cycles, depth-2 witnesses, collisions, archetype flat-zero intron, cycle return to rest, generate write.

**Result:** All pass.

---

### test_null_splits.py (5 tests)

**Covers:** `NullCorpus` structured holdouts (micro-ref, signature-factor, collision cross-split, corruption transfer, frame helpers) and masked-byte leakage guard.

**Result:** All pass.

---

### test_super_trains.py (12 tests)

**Covers:** Super registry (`spectral`/`unified` absent), scan leakage guard, CLI Super train smoke, Theorem 1 grammar head, occupation readout, two-sided G6, G3 joint-word length, grammar-head no-peek flip test.

**Result:** All pass.

---

### test_task_training.py (15 tests)

**Covers:** CLI training wiring for model/task combinations, loss key parity, denoiser gain bound, Super masked-frame smoke.

Parametrized over mlp/k4 with state_ce, and narrow heads (transition, rawbyte, word, percolation_rank). Super has a dedicated one-epoch masked_frame smoke. Denoise trains `AffineSpectralCodec`.

| Test | What it checks | Pass criterion |
|------|----------------|----------------|
| `test_denoise_rate_weight_wired` | Rate weight in denoise CLI | Affects training |
| `test_loss_key_parity` | LossWeights fields | All CLI loss keys are valid fields |
| `test_task_primary_loss_key_nonzero` | Primary loss per task | Correct key has nonzero gradient |
| `test_denoise_smoke_passes_gain_bound` | Denoise train smoke (1 epoch) | Gain report mean_abs_error < 0.5 |
| `test_super_train_smoke` | Super masked_frame CLI | Checkpoint loads as Super |

**Result:** All pass in default run.

---

### test_training_smoke.py (6 tests)

**Covers:** training loop infrastructure in `training_run.py`.

Checks: smoke training reduces loss; training is deterministic with fixed seed; checkpoint roundtrip; validation and early stopping; JSONL log written.

**Result:** All pass.

---

### test_word_action.py (14 tests)

**Covers:** word/signature datasets, transition model, word action model, compositional consistency.

Checks: word signature ID convention; minimal representative words cover group; stratified sample labels exact and deterministic; commitment fields; same-signature-different-ledger pairs; shadow pair split no leakage; held-out Q class split; byte features shape; transition model forward and one-step smoke; word action parity exact; word action learns single-byte taus; compositional consistency with exact heads.

**Result:** All pass.

---

## 2. Evaluation helpers

### evals_datasets.py

Builds all evaluation and training corpora. Major groups:

- **Ensembles:** lambda chirality distribution, chi fiber, lambda corpus sampling, stabilizer characterization, never-broken subgroup (exhaustive), W2 brokenness set.
- **Anomaly:** biased family bytes, biased Q-weight bytes, missing Q-class alphabet, corrupted mask dataset, byte perturbation dataset.
- **Percolation:** restriction labels, BFS reachability, rank-controlled alphabet, percolation dataset with transport rank and cluster size law, shell ensemble labels, Walsh multipliers.
- **Words:** canonical words, word replay effects, stratified word sample, same-signature-different-ledger pairs, shadow pair split, held-out Q class split.
- **Byte mechanism:** fold targets, L/R factorization audit, depth-4 frame dataset, frame parity and mask checks.
- **Dimension transfer:** hqvm_d transition dataset, dimension transfer probe.

Tested by: `test_anomaly.py`, `test_ensembles.py`, `test_percolation.py`, `test_canonical_words.py`, `test_word_action.py`, `test_byte_mechanism.py`, `test_benchmarks.py`.

### evals_metrics.py

All metrics and readouts. Major groups:

- **Reconstruction:** exact accuracy, Hamming error, chirality accuracy, shell accuracy.
- **Equivariance:** K4 encoder/decoder error, generic full-G error, transition equivariance error.
- **Transition:** next-state accuracy, rollout accuracy.
- **Symmetry breaking:** psi_hat (Agrawal order parameter).
- **Byte mechanism:** factorization targets, probe from latent, shadow invariance error.
- **Readouts (spec 2.1 to 2.6):** climate (rho, eta, M2), Plancherel consistency, anisotropy, gauge character, Z2 sheet, 32-bit lift, code membership.
- **Denoiser:** exact denoiser multipliers, block multipliers, denoiser gain report.
- **Scale:** operator structure, Walsh sector energy, shell distribution ensemble.
- **Containers:** Readouts, ScaleSuite.

Tested by: `test_readouts.py`, `test_psi_hat.py`, `test_byte_mechanism.py`, `test_latent_components.py`, `test_anomaly.py`, `test_scale.py`, `test_equivariance.py`, `test_affine_codec.py`.

### evals_run.py

Evaluation harness and verification. Major functions:

| Function | Purpose | Tested by |
|----------|---------|-----------|
| `load_any_checkpoint` | Restore model from checkpoint with full config | `test_load.py`, `test_task_training.py`, `test_training_smoke.py` |
| `evaluate_reconstruction` | Argmax accuracy over state census | Production eval reports |
| `evaluate_checkpoint` | Full eval report writer | CLI evaluate; schema tests |
| `verify_k4_equivariance` | Per-gate K4 error with pass flag (tol 1e-4) | Production `*_eq.json` |
| `verify_full_g_equivariance` | Sampled full-group error | `test_affine_codec.py` |
| `exhaustive_full_g_verify` | Closed-form gain-symmetry certificate | `test_scale.py` (slow) |
| `audit_dictionary` | One-pass dictionary audit with pass gates | `test_dictionary.py` |
| `percolation_suite` | Benchmark 3.1 threshold and anchor | `test_benchmarks.py` |
| `anomaly_benchmark` | ROC/PR on perturbation corpus | Not directly unit-tested (invoked via Benchmarks) |
| `climate_sweep` | Benchmark 3.3 regime sweep | `test_benchmarks.py` |
| `Benchmarks` | Container for all benchmark suites | `test_benchmarks.py` |

### datasets.py (null corpus) and training_super.py

Null corpus builder (`dataset_null`), `NullCorpus` holdouts, Super task losses, and gates G1–G9 (including capacity sweep and latent factorization probe). Tested by `test_null_corpus.py`, `test_null_splits.py`, `test_super_trains.py`; gate numbers live in `super_gates.json`.

**CGM Null Dataset.** Formal definition and construction (archetype `0xAA`, rest state, then `byte_fiber` → `canonical_cycles` → `depth2_witnesses` → `signature_words` → `collisions` → `measures.json`) are stated in the README section "CGM Null Dataset". Builder: `generate_null_dataset()` in `datasets.py`.

### training_losses.py and training_run.py

Loss weighting (`LossWeights`, `weighted_total`) and the four-hook training loop with checkpointing, JSONL logging, early stopping. Tested by `test_training_smoke.py`, `test_task_training.py`.

---

## 3. Production evaluation artifacts

These JSON files under `data/reports/` are the machine-readable record of the shipped checkpoints. They are indexed by `production_summary.json` and their schemas are pinned by `test_report_schemas.py`.

### Summary table

| Checkpoint | Role | Recorded result |
|------------|------|-----------------|
| `super` | Exact grammar + provenance + λ climate + sequential residual | context 32, atom 8, 106,754 params; all mandatory gates pass. `claim`: `exact_grammar_plus_provenance_and_residual`. sha256 `42e8e5ba40c304e96d0b5d243dab712a0409c3bc4804d8bc86cb10185981c4ff`. See `super_gates.json`. |
| `k4_full` | Exact K4-equivariant state AE | Reconstruction exact 0.9995; K4 equivariance max 3.3e-11, passed |
| `mlp_full` | Symmetry-free contrast baseline | Reconstruction exact 0.850; equivariance max ~216 (multi-signature), passed false |
| `spectral_bottleneck` | AffineSpectralCodec for denoise / dictionary probes | Analytic gains; used by denoise ladder and dictionary probes |

### Super gates (`super_gates.json`)

Measured by `helpers.training_super` (`evaluate_gates`). Climate is analytic and gated; sequential completion adds a prefix-GRU arm plus an opt-in analytic previous-micro flip filter (`apply_markov`, used only on the Markov train/eval path so G6 stays calibrated). Grid in the JSON: context {16, 32}, atom {8}; capacity sweep ranks with measure gates (λ fraction, Markov Δ, then G4); the 32/8 winner then continue-trains 6 `super_all` epochs at lr 1e-3 and 2 Markov-path polish epochs. Regenerated end-to-end by `python -m src.tools.autoencoder.helpers.training_super gates`. G4_margin_full = 0.391.

| Gate | Meaning | Recorded |
|------|---------|----------|
| G1 | Masked canonical recovery on held-out micro-refs (256-way byte head) | 1.0 |
| G2 | One-site cycle fill and holonomy | 1.0 |
| G2_iterative | Autoregressive fill (separate metric) | 1.0 |
| G3 | Signature XOR-register composition on holdout words | 1.0 |
| G4 | Collision-ledger margin over untrained floor (cosine distance 0.1) | pass (rate 0.996, floor 0.605, margin +0.391). At margin 0.3: 0.896 |
| G5 | Margin over state-only Ω baseline | pass (0.996) |
| G6 | Uniform NLL, two-sided \|bits−8\|≤0.5 | 8.287, pass |
| G7 | Canonical next-byte bits/byte (threshold ≤ 2.5) | 1.545 (pos0≈6.147, pos1–3≈0.004–0.022), pass. Position 0 is the 6-bit micro floor with family fixed by frame index. |
| G8 | Size used for the shipped winner | 32 / 8, 106,754 params |
| G9 | Holdout signature labels and GRU linear/MLP probe | nunique parity 2, τu 32, τv 63; τ_u probe 0.0 (coverage check, not a fail) |
| residual_improvement_bits | Prior NLL − residual NLL on held-out incomplete-prior mix | −0.075 |
| λ residual | Improvement vs closed-form ceiling 8−H(p_λ) at λ=4 (ceiling 1.668 bits) | +1.501 bits, fraction 0.900 |
| Markov causal | Prefix-only last byte vs 6·h₂(p)+2 at p=0.1 | model 5.113 bits, bound 4.814, improvement +2.887, gap +0.299 |
| boundary | Exact signature-coset completion | valid 1.0, exact-byte 0.609 (ambiguity 2), mass on valid set 1.0 |
| corrupt replay | Holdout 2-byte recovery under rest→swapped→rest | 1.0 |

Grammar ablation (analytic frozen vs random-init free grammar, 3 epochs): analytic G1/G7 = 1.0 / 1.503; free G1/G7 = 0.016 / 11.038. The frozen grammar is what buys G1 in this budget.

Multi-seed (3 on 32/8, `super_all`, `args.epochs` default 8, measure-gate lite eval before continue-train): all seeds G1=1.0; G4 margins 0.216, 0.126, 0.350.

| context | atom | learnable params | 8-epoch G1 | 8-epoch G4 | 8-epoch λ | 8-epoch Markov Δ | shipped |
|---------|------|------------------|------------|------------|-----------|------------------|---------|
| 16 | 8 | 72,002 | 1.0 | +0.335 | 0.899 | +3.016 | no |
| 32 | 8 | 106,754 | 1.0 | +0.328 | 0.899 | +3.024 | yes (G4 +0.391, λ 0.900, Markov +2.887) |

### File reference

**Eval / equivariance** (narrow/general):

| File | Key numbers |
|------|-------------|
| `k4_full_eval.json` | exact 0.9995; eq max 3.3e-11; decoder max 9.4e-8 |
| `mlp_full_eval.json` | exact 0.850; eq max ~216, passed false |
| `k4_full_eq.json` | max 3.3e-11, mean 2.6e-12, passed true |
| `mlp_full_eq.json` | max ~216, mean ~87, passed false |

**Index** (`production_summary.json`): maps `super`, `k4_full`, and `mlp_full` to checkpoint paths, report paths, and sha256 digests. Super records `claim` (`exact_grammar_plus_provenance_and_residual`).

---

## 4. What is not tested here

- Kernel science itself (transition rule, family classification, Omega map) is assumed correct from the kernel and cited from the Features Report.
- One test requires `--runslow`: the naive exhaustive full-G sweep in `test_scale.py`. The denoiser smoke in `test_task_training.py` runs by default (1 epoch).
- GPU execution is not part of the suite; all tests run on CPU.
- The anomaly ROC/PR benchmark (`anomaly_benchmark`) is wired in `Benchmarks` but has no dedicated unit test file.
- Super trains on `dataset_null` only.

---

## 5. Regenerating artifacts

Production reports and checkpoints:

```
python -m src.tools.autoencoder.helpers.training_super production
```

Full test suite (autoencoder only):

```
python -m pytest tests/tools/autoencoder/
```

Slow tests included:

```
python -m pytest tests/tools/autoencoder/ --runslow
```
