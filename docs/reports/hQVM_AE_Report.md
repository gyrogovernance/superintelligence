# hQVM Autoencoder: Test and Evaluation Report

## What this report covers

This report records what the hQVM autoencoder package tests and what its shipped checkpoints actually achieved. It is the evaluation companion to the package specification, which covers the theory, the model tiers, and the design of the null corpus. If you want to know what the package is and why it is built the way it is, read the specification first. This document assumes that context and reports the measurements. The genomics programs that build on this package are reported separately in the [genomics report](hQVM_AE_Genomics_Report.md).

The package is an exactly group-equivariant autoencoder suite over a finite group-structured state space. It ships four certified pieces: models whose symmetries are built into the architecture rather than encouraged by a penalty, a closed-form equivariance certificate for the full affine group on the spectral codec, a Bayes-optimal spectral denoiser with analytic gains, and the Super process model, which combines an exact hQVM grammar and signature registers with learned provenance, a gated analytic climate head, and a sequential residual on prefix context.

The code lives in `src/tools/autoencoder/`: models in `models/`, training and evaluation helpers in `helpers/`, and unit tests under `tests/tools/autoencoder/`. Production checkpoints are under `data/checkpoints/production/` and the machine-readable evaluation records are under `data/reports/`. Everything in this report was produced on CPU. The regeneration commands are at the end.

## How the evaluation is organized

The package makes two different kinds of claim, and the way it is tested reflects that.

The first kind is an exactness claim. When the package says K4 equivariance holds exactly, or that the spectral codec's denoiser gains match the closed-form Bayes-optimal multipliers, it is asserting a relation that either holds or does not. These claims are tested over the full state space where that is tractable, and certified in closed form where a full sweep would be wasteful. A test of this kind passes a numeric gate or it fails.

The second kind is a measurement on a corpus. When the package says the Super climate head takes 90.0% of the closed-form QuBEC ceiling, or that the collision-ledger margin clears its untrained floor, it is reporting a number measured on a particular dataset. These numbers hold for that dataset and are regenerable, but they are not identities.

Most of the suite is exactness testing. The exception is the Super gate suite, which is a mixture: some gates are exact certificates, and some are measured margins with declared thresholds.

The tests are grouped by what they protect, not by folder order. One group checks the group symmetry itself. One group checks that the kernel adapter reproduces the kernel rather than reimplementing it. One group checks the generated datasets against kernel replay. One group checks the readouts and metrics against theory. One group checks that training actually wires up and produces usable checkpoints. One group checks the report schemas so the published JSON cannot silently drift. The domain-by-domain reference later in this report lists every file and what it covers.

Two structural facts about the suite are worth stating up front. First, all tests run on CPU; there is no GPU path in the suite. Second, exactly one test requires an opt-in slow flag: the naive exhaustive full-group sweep in `test_scale.py`. The closed-form verifier that replaces it runs by default and completes in under a second.

## The shipped artifacts

The package ships four production objects. Each one is a checkpoint with a corresponding report, and `production_summary.json` indexes them with paths and digests.

### k4_full: the exact K4 model

`k4_full` is the exact K4-equivariant state autoencoder, and it is the cleanest demonstration of what architectural symmetry buys. It reconstructs the state census at 0.9995 exact accuracy, and its K4 equivariance holds with a maximum error of 3.3e-11. That is not a small error that happens to be under a threshold. It is the residual of floating-point arithmetic on a relation that is exact by construction.

The decoder is measured separately. Its maximum equivariance error is 9.4e-8, which is the same story at a different point in the computation. The dedicated equivariance report records a maximum of 3.3e-11 and a mean of 2.6e-12, with the pass flag true. The test tolerances are set at 1e-4, which leaves roughly seven orders of magnitude of headroom on this model.

### mlp_full: the symmetry-free baseline

`mlp_full` is the contrast. It has the same kind of architecture as the K4 model but none of the symmetry. It reconstructs at 0.850 exact accuracy, and its K4 equivariance error reaches about 216 and fails the pass flag. The equivalent report file records a mean of about 87.

The role of this checkpoint is to make the K4 result mean something. A model without the symmetry does not have it, and the gap is not marginal. The comparison is the reason the K4 numbers are evidence rather than trivia.

### spectral_bottleneck: the analytic codec

`spectral_bottleneck` is the AffineSpectralCodec. Its denoiser gains match the closed-form Bayes-optimal multipliers, which is the analytic certificate for this piece. It is the checkpoint behind the denoise ladder and the dictionary probes, and it is the model the closed-form full-group certificate applies to.

### super: the process model

`super` is the hybrid process model, and it is the most constructed artifact in the package. The measured production object is exact grammar plus provenance plus residual, recorded with the claim `exact_grammar_plus_provenance_and_residual`. The shipped winner uses context 32 and atom 8, with 106,754 parameters, and its checkpoint digest is `42e8e5ba40c304e96d0b5d243dab712a0409c3bc4804d8bc86cb10185981c4ff`.

The headline measurements are these. The lambda-ensemble residual at lambda = 4 takes 90.0% of the closed-form ceiling, against a ceiling of 1.668 bits. Signature-constrained completion places mass 1.0 on the valid coset, with an ambiguity of 2. The Markov causal filter improves 2.89 bits over uniform on its holdout, leaving a 0.30-bit gap to the relevant bound, and it is opt-in precisely so the exact certificates stay exact. The full gate record is below.

## The Super gate suite

Super is evaluated by nine gates plus a set of named extra measurements. The gates are the contract for the shipped checkpoint, and the report is written by the Super training helper, whose gate routine also regenerates it end to end.

The gate suite is more involved than the rest of the package because Super has moving parts, and the gates exist to check that the parts stay separate. The Grammar gates (G1, G2, G2_iterative, G3) check the frozen exact grammar. G4 and G5 check the learned provenance against floors. G6 and G7 check the uniform and canonical distributions, with G6 two-sided so the model cannot drift while looking fine on one side. G8 records the size of the shipped winner. G9 is a coverage check on the holdout signature labels.

| Gate | What it checks | Recorded |
|------|----------------|----------|
| G1 | Masked canonical recovery on held-out micro-refs, 256-way byte head | 1.0 |
| G2 | One-site cycle fill and holonomy | 1.0 |
| G2_iterative | Autoregressive fill, a separate metric | 1.0 |
| G3 | Signature XOR-register composition on holdout words | 1.0 |
| G4 | Collision-ledger margin over the untrained floor, cosine distance 0.1 | pass, rate 0.996, floor 0.605, margin +0.391. At margin 0.3 the rate is 0.896. |
| G5 | Margin over the state-only Omega baseline | pass, 0.996 |
| G6 | Uniform NLL, two-sided, bits within 0.5 of 8 | 8.287, pass |
| G7 | Canonical next-byte bits per byte, threshold at most 2.5 | 1.545, pass. Position 0 is about 6.147, positions 1 through 3 are about 0.004 to 0.022. Position 0 is the 6-bit micro floor with the family fixed by frame index. |
| G8 | Size used for the shipped winner | 32 / 8, 106,754 parameters |
| G9 | Holdout signature labels and GRU probe | nunique parity 2, tau_u 32, tau_v 63, tau_u probe 0.0, a coverage check and not a failure |
| residual_improvement_bits | Prior NLL minus residual NLL on held-out incomplete-prior mix | -0.075 |
| lambda residual | Improvement against the closed-form ceiling 8 minus H(p_lambda) at lambda = 4, ceiling 1.668 bits | +1.501 bits, fraction 0.900 |
| Markov causal | Prefix-only last byte against 6*h2(p)+2 at p = 0.1 | model 5.113 bits, bound 4.814, improvement +2.887, gap +0.299 |
| boundary | Exact signature-coset completion | valid 1.0, exact-byte 0.609 with ambiguity 2, mass on the valid set 1.0 |
| corrupt replay | Holdout 2-byte recovery under rest, swapped, rest | 1.0 |

### Why the frozen grammar carries G1

The package claims that the frozen grammar, not the learned weights, is what produces the exact recovery on G1. That claim is tested by ablation. Running an analytic frozen grammar against a random-initialized free grammar for 3 epochs gives G1 of 1.0 and G7 of 1.503 for the analytic version, against G1 of 0.016 and G7 of 11.038 for the free version. The frozen grammar is what buys G1 at this training budget, and the ablation records it rather than asserting it.

### Stability across seeds and sizes

The gate numbers come from a specific run, so the suite also checks that they are not a lucky seed or a lucky size. Three seeds at 32/8 under the `super_all` task all reach G1 = 1.0, with G4 margins of 0.216, 0.126, and 0.350. The variation in G4 across seeds is the largest spread in the gate suite, which is consistent with G4 being a measured margin rather than an exact certificate.

The capacity sweep compares context 16 and context 32 at atom 8. Both reach G1 = 1.0. At 8 epochs, context 16 has a G4 margin of +0.335, lambda of 0.899, and Markov delta of +3.016, at 72,002 parameters. Context 32 has a G4 margin of +0.328, lambda of 0.899, and Markov delta of +3.024, at 106,754 parameters. Context 32 was shipped and continued training, reaching the final G4 of +0.391, lambda of 0.900, and Markov improvement of +2.887. The sweep makes the shipped size defensible rather than assumed.

## The test suite by domain

This section is the reference. It lists every test file, what it covers, and how it relates to the rest of the package. The grouping, rather than an alphabetical order, reflects that each group protects a different part of the contract.

### Exact group symmetry

`test_equivariance.py`, 7 tests. Covers exact K4 equivariance on `K4Autoencoder`, exhaustively over all 4096 states and four gates. It checks that the K4 characters are group homomorphisms, that the latent action rho is a representation, that the encoder satisfies E(g.x) = rho(g) E(x) and the decoder satisfies D(rho(g) z) = P_g D(z), each with a maximum error below 1e-4 per gate. It also checks the error metric itself, output validity for an identity latent, and consistency of the K4 permutation matrix with the kernel. All pass.

`test_latent_components.py`, 4 tests. Covers the named K4 latent components `z_invariant`, `z_char_S`, `z_char_C`, and `z_char_F`, plus transition equivariance. It checks that the named parts span the full latent, that each component is equivariant under K4, that the transition head has zero equivariance defect on the exact model, and that the same metric detects an untrained model. All pass.

`test_psi_hat.py`, 4 tests. Covers the Agrawal symmetry-breaking order parameter `psi_hat` in `evals_metrics.py`. It checks that an exact K4 encoder gives magnitude 1 for a broken generator, that a trivial generator gives +1, that a single-character latent gives magnitude 1 with the correct sign, and that an unstructured encoder sits well below 1. All pass.

`test_affine_codec.py`, 13 tests. Covers `AffineSpectralCodec`: the Walsh transform, the irrep blocks, the spectral action, a sampled full-group equivariance check, the bottleneck, and the codec ladder masks. Thresholds include orthogonality of the Walsh matrix, 2080 blocks covering all frequencies, one-hot Walsh coefficients matching the kernel sign, exact roundtrip, spectral action matching translation, sampled full-group error of zero, signature composition in the spectrum, equivariance preserved through the bottleneck, correct ladder mask structure, frozen masks zeroing sectors, the rate penalty ignoring masked blocks, and exact equivariance for the diagonal model. All pass.

### Kernel adapter fidelity

`test_group_actions.py`, 6 tests. Covers kernel adapter indexing and group actions in `kernel.py`. It checks the state index roundtrip for all 4096 states, that the step index matches the kernel for sampled bytes, that signature apply matches the kernel, that signature IDs enumerate the group, that K4 orbit consistency holds, and that the word signature ID matches replay. All pass.

`test_models.py`, 5 tests. Covers base model utilities and the soft equivariance loss wiring. It checks the state-to-bits roundtrip, MLP forward shapes, that the K4 generator batch matches the kernel, that the soft equivariance loss is finite, and that the loss requires an explicit latent action. All pass.

`test_load.py`, 8 tests. Covers checkpoint save and load roundtrip and `build_model` config propagation. It checks the task model `get_config` roundtrip for transition, rawbyte, word, and percolation, the K4 trivial and sign count roundtrip across several configurations, and that `build_model` forwards those counts. All pass.

### Datasets and generation

`test_dataset_generation.py`, 23 tests. Covers the generated census arrays, transition tables, K4 actions, signatures, manifests, and invariant checks in `datasets.py`. Checks include byte census shape and kernel column match, family decomposition, the full column set, the chirality chart via bit operations, stepping paths agreeing with the kernel, the transport law, signature apply matching word replay, state census shape and index and shell fields, observables, the K4 orbit partition, spin roundtrip, the transition table and inverse roundtrip, K4 action arrays, signature dataset values and rest action, manifest save and load, invariant check failure reporting, entry byte count, and generate determinism. All pass.

`test_null_corpus.py`, 5 tests. Covers the stage-0 null corpus invariants in `datasets.py`: the byte fiber, canonical cycles, depth-2 witnesses, collisions, the archetype flat-zero intron check, the cycle return to rest, and generate write. All pass.

`test_null_splits.py`, 5 tests. Covers the `NullCorpus` structured holdouts, including the micro-ref split, the signature-factor split, the collision cross-split, corruption transfer, frame helpers, and the masked-byte leakage guard. All pass.

`test_canonical_words.py`, 4 tests. Covers the canonical K4 and BU word dataset and word replay effects. It checks the word table schema, that word K4 differs from gate K4 except on gate F, that replay effects match the dataset on involution and chirality and shell labels, and that signature ID replay matches kernel composition. All pass.

`test_ensembles.py`, 6 tests. Covers lambda-ensemble corpora and symmetry-breaking characterization in `evals_datasets.py`. It checks that the chirality distribution normalizes, that the chi fiber partitions the 4096 states, that a sampled corpus matches the expected shell histogram at a given lambda, the stabilizer test helper, the never-broken subgroup by exhaustive composition closure, and the W2 brokenness set shape. All pass.

`test_percolation.py`, 9 tests. Covers percolation datasets, shell ensemble labels, Walsh multipliers, and a `PercolationLearner` smoke training. Checks include singleton labels matching BFS, the full alphabet giving full reachability, rank-controlled rows matching the declared rank, full reachability implying the giant component, the predicted cluster law, shell ensemble corpus conventions for rho, eta, and M2, Walsh multipliers being uniform and matching the corpus damping, and learner rank recovery. All pass.

`test_percolation_seed.py`, 2 tests. Covers determinism of the percolation evaluation holdout split. The same seed gives an identical holdout, and a different seed gives a different one. All pass.

### Readouts and metrics

`test_readouts.py`, 9 tests. Covers kernel-exact readouts in `evals_metrics.py`. It checks the climate readout (rho, eta, M2) against theory on the lambda grid, Plancherel consistency, the anisotropy eta law, the gauge character readout against the kernel character table, the Z2 sheet readout, the 32-bit lift, code sector membership, the closed-form denoiser multipliers, and the climate synthesizer reconstructing a target climate from Walsh gains. All pass.

`test_anomaly.py`, 12 tests. Covers the anomaly and perturbation datasets in `evals_datasets.py`, plus Walsh sector energy and shell distribution diagnostics in `evals_metrics.py`. It checks the biased family and Q-weight samplers, the missing Q-class alphabet, the GF(2) rank helper against hand-checked matrices, mask-corruption syndromes, byte perturbation shadow preservation and label consistency, shell histogram ensemble statistics, Walsh sector energy partition, the dimension-d transition dataset, that d = 6 matches the kernel API, and the cross-dimension probe structure. All pass.

`test_byte_mechanism.py`, 8 tests. Covers byte mechanism datasets, factorization probes, shadow invariance, depth-4 frames, and narrow-tier heads. It checks `RawByteTransitionModel` forward shape and short-training smoke, factorization target shapes, a linear probe recovering L and R pairs from an oracle latent, shadow invariance being zero on the kernel-exact table and nonzero for an untrained MLP, depth-4 frame signature consistency, and a `FrameHead` training smoke. All pass.

`test_scale.py`, 5 tests with 1 slow. Covers scale diagnostics and the closed-form full-group verifier. It checks the operator structure commutant, that the ScaleSuite container runs, the full 4096 by 8192 sweep (opt-in slow, requires `--runslow`), and the closed-form CLI verifier, which returns exit code 0 in under a second. Three tests pass in the default run.

### Codecs

`test_codecs.py`, 4 tests. Covers the deterministic narrow-tier chart codecs. It checks that ExactUV and BoundaryChirality round-trip perfectly, that ChiralityOnly maps 64 states per chirality class as expected, and that ShellOnly fiber sizes follow the binomial law. All pass.

### Training infrastructure

`test_training_smoke.py`, 6 tests. Covers the training loop in `training_run.py`. It checks that smoke training reduces loss, that training is deterministic with a fixed seed, checkpoint roundtrip, validation and early stopping, and that a JSONL log is written. All pass.

`test_task_training.py`, 15 tests. Covers CLI training wiring for model and task combinations, loss key parity, the denoiser gain bound, and the Super masked-frame smoke. It is parametrized over mlp and k4 with state_ce, plus the narrow heads (transition, rawbyte, word, percolation_rank). Super has a dedicated one-epoch masked_frame smoke, and the denoise path trains `AffineSpectralCodec`. It checks that the denoise rate weight is wired into training, that every CLI loss key is a valid field, that the primary loss key has a nonzero gradient, that the denoise smoke keeps its gain report mean absolute error below 0.5, and that a Super masked_frame checkpoint loads as Super. All pass in the default run.

`test_super_trains.py`, 12 tests. Covers the Super registry, the scan leakage guard, a CLI Super train smoke, the Theorem 1 grammar head, the occupation readout, two-sided G6, the G3 joint-word length, and the grammar-head no-peek flip test. It also checks that the retired `spectral` and `unified` names are absent from the registry. All pass.

### Word and signature action

`test_word_action.py`, 14 tests. Covers word and signature datasets, the transition model, the word action model, and compositional consistency. Checks include the word signature ID convention, minimal representative words covering the group, stratified sample labels being exact and deterministic, commitment fields, same-signature different-ledger pairs, the shadow pair split having no leakage, the held-out Q class split, byte feature shape, transition model forward and one-step smoke, exact word action parity, the word action learning single-byte taus, and compositional consistency with exact heads. All pass.

### Dictionary and reports

`test_dictionary.py`, 6 tests. Covers the embedding corpus export in `corpus.py` and the dictionary audit in `evals_run.audit_dictionary`. It checks that export arrays match the census, that the audit passes on the identity AffineSpectralCodec with every gate green, that equivariance still passes on a 3-epoch trained model, that only boolean core checks block the pass flag, that a non-boolean required field fails correctly, and that an informational probe accuracy does not gate the pass. All pass.

`test_report_schemas.py`, 5 tests. Covers the production JSON report schemas under `data/reports/`. It does not retrain; it reads existing files. Pinned reports are `k4_full_eval.json`, `mlp_full_eval.json`, `mlp_full_eq.json`, `super_gates.json` (which requires `gate_version`), and `production_summary.json` (whose keys include `k4_full`, `mlp_full`, and `super`). It also checks that all paths in `production_summary.json` use forward slashes. These pass when the reports are present, and individual schema tests skip if a file is missing. This is the guard that keeps the published JSON aligned with the CLI.

`test_plan_completion.py`. Covers the LUT scan against the kernel, the prefix factor table, the signature posterior against short-word enumeration, write-protected caches, ladder aliases including `shell_climate`, the `super_all` one-epoch checkpoint, and encode and decode. Asserted by pytest.

### The helpers under test

`evals_datasets.py` builds all evaluation and training corpora: the ensembles, the anomaly and perturbation sets, the percolation sets, the word and signature sets, the byte mechanism sets, and the dimension transfer probe. It is tested by the dataset and benchmark tests listed above.

`evals_metrics.py` holds the metrics and readouts: reconstruction, equivariance, transition, symmetry breaking, byte mechanism, the kernel readouts, the denoiser multipliers, the scale diagnostics, and the readout containers. It is tested by the readout, psi_hat, byte mechanism, latent component, anomaly, scale, equivariance, and affine codec tests.

`evals_run.py` drives the evaluations. It loads checkpoints, evaluates reconstruction, writes full evaluation reports, verifies K4 and full-group equivariance, runs the closed-form gain-symmetry certificate, audits the dictionary, and runs the percolation and climate benchmark suites. The benchmark container is `Benchmarks`.

`datasets.py` builds the null corpus and its holdouts. `training_super.py` holds the Super task losses and the gates G1 through G9, including the capacity sweep and the latent factorization probe. `training_losses.py` and `training_run.py` hold the loss weighting and the four-hook training loop with checkpointing, JSONL logging, and early stopping.

One note on the anomaly benchmark: `anomaly_benchmark` computes ROC and PR on the perturbation corpus and is wired into `Benchmarks`, but it has no dedicated unit test file of its own.

## Limits of this evaluation

The suite tests the package, not the kernel. The kernel science itself, meaning the transition rule, the family classification, and the Omega map, is assumed correct and cited from the Features Report. The package tests its own mapping onto that kernel.

The K4 and full-group equivariance claims are exact and are tested as exact. The Super gate suite mixes exact certificates with measured margins, and G4 is the clearest example of a measured number: it varies across seeds more than any other gate, and the report records the spread rather than a single value.

GPU execution is not part of the suite. All tests run on CPU.

Super trains on the CGM null dataset only, the same kernel-null corpus referred to elsewhere as the grammar-generated atlas. Its gate numbers are measurements on that corpus, and they should be read as such. They are not claims about biological data, which never enters training.

The report schemas are pinned, which means a change to the CLI that alters a published JSON shape will fail the schema tests on the next run. That is intentional, and it is why the schema tests are part of the suite rather than a separate check.

## Regenerating the artifacts

Production reports and checkpoints:

```text
python -m src.tools.autoencoder.helpers.training_super production
```

The full test suite for the autoencoder package:

```text
python -m pytest tests/tools/autoencoder/
```

Including the slow tests:

```text
python -m pytest tests/tools/autoencoder/ --runslow
```

The Super gate suite on its own, which rewrites `super_gates.json`:

```text
python -m src.tools.autoencoder.helpers.training_super gates
```

The regeneration paths are the same ones the package README documents. The difference is that this report tells you what the regenerated numbers are supposed to look like and which of them are exact contracts.
