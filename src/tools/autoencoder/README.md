# hQVM Group-Equivariant Autoencoder

## What this is

This package is a neuro-symbolic group-equivariant autoencoder suite over a finite group-structured state space. It ships four certified pieces together:

1. models whose symmetries are built into the architecture (not encouraged with a penalty);
2. an exhaustive closed-form equivariance certificate for the full affine group on the spectral codec;
3. a Bayes-optimal spectral denoiser whose gains have a known analytic form;
4. an exact-grammar process model (`Super`) whose learned parts are provenance discrimination and a residual climate head gated against QuBEC λ ceilings, with exact signature-coset completion.

The underlying machine is the hQVM, defined in the [Formalism](../../../docs/specs/hQVM_Specs_Formalism.md), [QuBEC](../../../docs/specs/hQVM_QuBEC_Theory.md), and [SDK](../../../docs/specs/hQVM_SDK_Quantum_Computing.md) specifications. It has exactly 4096 reachable states. Each of the 256 input bytes selects a fixed permutation of those states. Every move is exact and reversible.

An autoencoder compresses an input into a compact code and rebuilds the input from that code. What it keeps in the code is what it decided matters. Here the networks are trained on the 4096 states (and, for Super, on byte ledgers) so the code becomes a readable summary of structure that already exists in the kernel, not a free-form black box.

A network is equivariant when applying a symmetry to the input and then encoding gives the same result as encoding first and then applying the matching symmetry to the code. The models in this project satisfy that property exactly, before and after training.

The kernel is the only authority for the machine. No transition rule, gate, mask, or intron is reformulated in the learning stack, and no dataset is generated outside `src/api.py`. Every dataset, group action, and evaluation routes through the kernel.

## The state space

The reachable state space, written Ω, is the set of states the machine can occupy. Ω has 4096 elements, each a 24-bit gyroscopic state consisting of two conjugate 12-bit halves. Every state has the same bit density, and the space factors as Ω = U × V with 64 values per factor.

Each state carries a chirality word χ in GF(2)^6, a six-bit value that records the state's orientation. Reading a byte adds a byte-specific charge q to the chirality by XOR, which is an affine transport law on the chirality register. That translation is diagonalized exactly by the 64-point Walsh-Hadamard transform, so the chirality register has a natural Fourier basis.

Shells partition the states by the Hamming weight of χ, giving seven shells indexed 0 through 6. The compact code of the machine's climate, called the QuBEC, assigns an occupation probability to every state. Weighting states by λ^N for a real parameter λ and shell N yields the exact partition function Z₁(λ) = 64·(1+λ)^6, and from it the three order parameters ρ, η, and M₂ that describe the climate: occupation density, spectral damping, and effective support size. Condensed climates concentrate near the low shells; thermalized climates spread across all 4096 states.

The self-dual [12,6,2] mask code C64 is the transport space on the active face, with 64 codewords. Depth-four words close as involutory operators, so every sliding four-byte frame compiles to a pure translation whose signature has parity zero.

## The models

Models are grouped into three tiers by what symmetry they build in. Every tier
is one file; the registry and the symmetry selector live in `__init__.py`.
"Equivariant" is never in a file name because it is the defining property.

| Tier file | Model | Symmetry built in | How it works |
|---|---|---|---|
| `narrow.py` | `ExactUVCodec`, `BoundaryChiralityCodec`, `ChiralityOnlyCodec`, `ShellOnlyCodec` | none (deterministic) | exact chart codecs used as information-theoretic null models - not networks at all |
| `narrow.py` | `MLPAutoencoder` | none | plain encoder and decoder. MLP = Multi-Layer Perceptron |
| `narrow.py` | `TransitionModel`, `RawByteTransitionModel`, `WordActionModel`, `FrameHead` | none | byte-conditioned task models (next state, raw byte, word, frame), supervised by kernel-exact targets |
| `narrow.py` | `PercolationLearner` | none | reads the packed 256-bit allowed byte mask and predicts the kernel-exact percolation labels (transport rank, reach, full/horizon/giant flags) |
| `general.py` | `K4Autoencoder` | exactly K4 | averages the encoder over the four Klein gates (Reynolds symmetrization), giving a latent split into `z_invariant`, `z_char_S`, `z_char_C`, `z_char_F` |
| `general.py` | `AffineSpectralCodec` | full affine group | Walsh occupation codec |
| `super.py` | `Super` | affine ledger grammar | kernel scan, visible-sibling byte recovery, XOR signature registers, GRU ledger context |

Flags select a model (`--model mlp`, `k4`, `super`, `transition`, `rawbyte`,
`word`, `percolation`); a tier name (`narrow`, `general`, `super`) or
`all` selects every model in that tier for sweeping.

Super is the process model for byte ledgers. It scans a ledger with the kernel, recovers a masked canonical byte from the visible siblings and the frame position using a frozen analytic grammar, and composes the word signature with exact XOR registers. A GRU over depth-4 frames learns provenance: it separates collision ledgers that share a signature but are not the same word. Climate and sequential completion are separate residual arms. Climate is an analytic QuBEC tilt from the visible-byte q-class histogram, gated off on exact-grammar frames so G1 and G7 stay exact; a sequential MLP reads prefix-GRU context, and an opt-in previous-micro flip filter (`apply_markov`) supplies the Bayes-optimal Markov conditional without disturbing uniform/canonical certificates. Encode/decode uses a ledger-free `SuperCode` (exact signature, endpoint, climate, plus provenance). Production weights and the gate report are written by `helpers.training_super`.

The measured production object is exact grammar + provenance + residual (`claim`: `exact_grammar_plus_provenance_and_residual`; checkpoint sha256 `42e8e5ba40c304e96d0b5d243dab712a0409c3bc4804d8bc86cb10185981c4ff`). Winner size is context 32 / atom 8 (106,754 parameters), regenerated by `helpers.training_super gates` (measure-gate capacity sweep, then 6 continue-train + 2 Markov-path polish epochs). λ-ensemble residual at λ=4 takes 90.0% of the closed-form ceiling. Signature-constrained completion puts mass 1.0 on the valid coset (ambiguity 2). The Markov causal filter improves 2.89 bits versus uniform (gap +0.30 bits to the `6·h₂(0.1)+2` bound on this holdout). The Markov filter is opt-in (`apply_markov`) so G6/G7 certificates stay on the grammar. See `hQVM_AE_Report.md` and `data/reports/super_gates.json`.

AffineSpectralCodec is the Walsh occupation codec. `train-denoise` and dictionary export use it; the shipped denoiser gains match the closed-form Bayes-optimal multipliers.

## What the package contains

```
src/tools/autoencoder/
├── cli.py             all entry points: train, train-denoise, evaluate, verify, verify-groups, sample-ensemble, sweep-lambda, export-embeddings, audit-dictionary, generate, verify-full-g-exhaustive
├── kernel.py          kernel adapter: state indexing, stepping, gates, signatures, signature id packing, popcount6
├── datasets.py        kernel tables, census, null corpus (dataset_null + NullCorpus), manifests
├── corpus.py          the dictionary export: embeddings with exact kernel labels (charter artifact)
├── paths.py           data/ layout
├── README.md          this document
├── __init__.py        package registry
│
├── models/
│   ├── __init__.py   MODEL_KINDS registry + build_model
│   ├── narrow.py     codecs, MLP, byte-mechanism predictors, percolation learner
│   ├── general.py    K4Autoencoder, AffineSpectralCodec
│   └── super.py      Super
│
├── helpers/           naming convention <domain>_<role>.py
│   ├── training_run.py      shared trainer: callbacks, checkpointing, JSONL logs
│   ├── training_losses.py   weighted multi-objective losses
│   ├── training_super.py    Super tasks, gates G1-G9, production regeneration
│   ├── evals_run.py         checkpoint loading, evaluation, reports, verification
│   ├── evals_metrics.py     reconstruction/equivariance/transition/psi_hat metrics + readouts
│   └── evals_datasets.py    eval corpus builders: ensembles, percolation, words, byte mechanism
│
└── data/              ACTUAL DATA ONLY: *.npy + manifest.json (regenerable via `cli generate`)
    ├── dataset_null/          CGM null permutation atlas (see "CGM Null Dataset")
    ├── dataset_bytes/         byte census
    ├── dataset_states/        state census
    ├── dataset_transitions/   dense transition table
    ├── dataset_signatures/    8192-row group signature table
    ├── dataset_actions/       K4 action table
    ├── dataset_embeddings/    verified-dictionary corpus
    ├── dataset_ensembles/     lambda-ensemble artifact
    ├── checkpoints/           trained model weights (gitignored)
    ├── reports/               eval/verify/audit JSON reports (gitignored)
    └── tmp/                   scratch space (gitignored)
```

## CGM Null Dataset (`dataset_null`)

### Definition

The CGM Null Dataset is the exhaustive, kernel-exact catalog of hQVM instruction and trajectory structure used as Super's training and gate corpus. It is generated by `generate_null_dataset()` in `datasets.py` and loaded through `NullCorpus`. Every field is produced by calling the kernel (`src.api`, `src.constants`, `src.family`). Nothing is sampled from outside data.

"Null" means the uniform (maximum-entropy) occupation of that exact structure under the three policies in `measures.json`: uniform iid bytes, the canonical family frame sequence, and QuBEC λ-weighted micro occupation.

### Construction

Construction begins at the archetype and the rest state.

1. Archetype. The transcription archetype is `GENE_MIC_S = 0xAA`. For any byte `b`, the intron is `b XOR 0xAA`. The invariant check `archetype_flat_zero_intron` requires that byte `0xAA` is flat and has intron `0`.
2. Rest state. Every trajectory in the catalog starts from `GENE_MAC_REST`. Depth-4/8 cycles must return through the swapped mac state and close again at rest (`canonical_rest_swapped_rest`).
3. `byte_fiber` (256 rows). Enumerate all instruction bytes. Record intron, family phase, micro-ref, chirality weight, fold disagreement, flat flag, shadow partner, and single-byte signature.
4. `canonical_cycles` (64 × 8 rows). For each micro-ref `m` in `0..63`, build the eight-byte word that walks the four family phases twice: `(family 0..3 for m) repeated`. Step from rest. Store state before/after, chirality shell, prefix signature factors, and horizon flags. Step 3 must land on swapped; step 7 must land on rest.
5. `depth2_witnesses` (256 × 256 rows). From rest, apply every ordered pair `(b0, b1)`. Store intermediate state, endpoint, signature id, and transport.
6. `signature_words` (8192 rows). For each group signature, take one minimal representative word of length at most 4. Replay it from rest and require agreement with `action_on_rest` and with applying the signature operator to rest.
7. `collisions`. From depth-2 witnesses and signature words, collect pairs of distinct ledgers that share a signature (and related collision kinds). These rows separate terminal action from ledger provenance.
8. `measures.json`. Record the three occupation policies: uniform (`p_byte = 1/256`), canonical (family order `[0, 1, 2, 3]` on a fixed micro), and lambda (`p_m ∝ λ^{popcount(m)}` on a stated grid).
9. Manifest and invariants. Write `manifest.json` with shapes, kernel fingerprint, and the boolean invariant suite (`run_null_invariants`). Generation fails if any invariant fails.

`NullCorpus` wraps these arrays and adds deterministic train/holdout splits: micro-ref holdout, signature-factor holdout, and a collision connected-component ledger split.

Regenerate with:

```bash
python -m src.tools.autoencoder.cli generate --dataset null
```

## Quick start

```bash
# Generate the kernel-derived datasets (including the null corpus)
python -m src.tools.autoencoder.cli generate --dataset all

# Train a model
python -m src.tools.autoencoder.cli train --model super --task masked_frame --epochs 40
python -m src.tools.autoencoder.cli train --model super --task super_all --epochs 40
python -m src.tools.autoencoder.cli train --model k4 --epochs 5
python -m src.tools.autoencoder.cli train --model transition --task transition --epochs 5
python -m src.tools.autoencoder.cli train --model word --task word --epochs 5
python -m src.tools.autoencoder.cli train --model percolation --task percolation_rank --epochs 5
python -m src.tools.autoencoder.cli train-denoise --ladder diagonal_translation_radial --noise-rate 0.03,0.03,0.03,0.03,0.03,0.03

# Super gate suite (writes reports/super_gates.json and checkpoints/production/super.pt)
python -m src.tools.autoencoder.helpers.training_super gates

# Verify that a K4 checkpoint is exactly equivariant
python -m src.tools.autoencoder.cli verify-equivariance --checkpoint checkpoints/model.pt

# Export the verified dictionary and run the one-pass audit
# (no --output-dir: defaults to src/tools/autoencoder/data/dataset_embeddings/
#  identity export; a trained checkpoint writes <name>_<suffix>.npy so it
#  coexists with the identity export instead of overwriting it)
python -m src.tools.autoencoder.cli export-embeddings
python -m src.tools.autoencoder.cli audit-dictionary --report-file reports/embedding_corpus_audit.json

# Run the lambda-ensemble experiment (symmetry-breaking order parameter)
python -m src.tools.autoencoder.cli sweep-lambda --model mlp --epochs 15 --n 16384
```

All commands accept the long GNU-style flags (`--output-dir`, `--run-name`,
`--learning-rate`, `--noise-rate`, `--report-file`). The short forms
`--out`, `--name`, `--lr`, and `--eta` are accepted as aliases for this
release and will be removed at the next boundary.

Datasets are written to `src/tools/autoencoder/data/` with a JSON manifest per directory recording the schema version, a kernel fingerprint, array shapes and dtypes, and invariant-check results. The directory is regenerable and gitignored.

Model selection is by `--model`: one of the individual kinds (`mlp`, `k4`, `super`, `transition`, `rawbyte`, `word`, `percolation`). The tier names `narrow`, `general`, `super`, and `all` select every model in that tier (used for sweeps); for `train` a tier maps to its first member. `build_model` in `models/__init__.py` is the single constructor. `--hidden-dim` overrides the per-kind default width.

Trained production artifacts live in `src/tools/autoencoder/data/checkpoints/production/` (`super`, `k4_full`, `mlp_full`, plus `spectral_bottleneck` for the denoiser / dictionary probes), with reports in `src/tools/autoencoder/data/reports/` including `super_gates.json` and `production_summary.json`.

## Production regeneration

```bash
# One-shot: train narrow/general nulls, ensure Super gates, write production_summary.json.
python -m src.tools.autoencoder.helpers.training_super production

# Or: reuse the existing checkpoints and only regenerate the reports.
python -m src.tools.autoencoder.helpers.training_super production --skip-train
```

`tests/tools/autoencoder/test_report_schemas.py` pins the report schema so any drift between the CLI and the published JSON fails on the next `pytest` run.

## Kernel core vs external adapters

The model core knows only the carrier grammar and the kernel's exact group
interfaces. Material outside that boundary (for example physical observables or
weight tensors from other systems) enters only as an **external adapter**, not
as a new model and not as a new model tier. Adapters are pure data transforms
that read the census and byte surfaces already exposed by the package
(`datasets.byte_census_arrays` and related helpers) and produce the columns the
readouts consume. They never add a kernel fact to a model file and never
reformulate a transition rule; they only convert external material into the
structure the codec already understands. This is the boundary that keeps the
three tiers clean: the models learn or represent hQVM structure; adapters
compile external material into that structure; task heads make
application-specific predictions; readouts measure the resulting structure.

## Adding a model or specialization

A new model joins the tier whose symmetry it builds in (a new group would
justify a new tier file, nothing else). Before it ships it must satisfy:

1. **Symmetry containment** decides the tier: `narrow` (no built-in symmetry), `general` (K4 gates), `super` (ledger process).
2. **Kernel authority**: every label and action comes from the kernel adapter
   (`src/api.py`); zero reformulation inside the package.
3. **Paired null**: every structured model ships with its narrow null so the
   symmetry-breaking contrast is measurable.
4. **Exactness contract**: an equivariance/identity test with a numeric gate,
   plus an entry in the verify/audit path.
5. **Registry wiring**: `MODEL_KINDS`, `TIER_MEMBERS`, `_HIDDEN_DEFAULTS`,
   `build_model`, `load_any_checkpoint`, evaluate-task routing, and the
   "task actually trains" regression test.
6. **Benchmark**: a suite entry with kernel-exact labels.

`PercolationLearner` is the template: a supervised head on kernel-exact
labels, sitting in `narrow`, needing no external data. Physics probes and
similar readouts are adapters, not models.

## Where empirical data enters

Base training is kernel-null: the models self-supervise on exact labels, which
is why the four production artifacts exist. Empirical tensors from outside the
kernel are **not** mixed into the core models. They enter as a frozen-codec
head fine-tune: keep the codec's exact equivariance intact and train only the
task heads on compiled windows
(`--task empirical --init <production checkpoint>`, heads only). For LLM
weights specifically, tile the matrix to 64-wide blocks, compile the byte
stream through the census, keep the codec frozen, and read the block
features. No new model tier is required for that path.

## Verification

The suite asserts exact kernel relations rather than learned proxies:

- K4 equivariance holds to an error below 1e-4 over all 4096 states and all four gates: `E(gx) = ρ(g)E(x)` and `D(ρ(g)z) = P_g D(z)`.
- Full-group equivariance holds with zero observed error: the spectrum of `gx` equals `ρ(g)` applied to the spectrum of `x`. The closed-form certificate across all 4096 states and all 8192 signatures (a sub-second condition on the gain symmetry) is persisted to `src/tools/autoencoder/data/reports/exhaustive_full_g_verify.json`.
- Word composition holds exactly: the network's composition of two word signatures equals the kernel's.
- Depth-four frames compiling to pure translations is a kernel theorem cited from the Features Report (#64), and audited in the dictionary's frame-parity check.
- The two-byte witness routing count is a kernel theorem cited from the Features Report (#80/#120); it is not re-derived inside the suite.
- The dictionary audit recomposes reconstruction, equivariance, closed-form factorization probes, the H-invariance of the diagonal rung, shadow invariance, frame parity, and the psi_hat character-energy identity.

The test suite lives in `tests/tools/autoencoder/`. It covers the package's models, losses, metrics, datasets, reports, and the mapping to the kernel. Kernel feature facts are cited from the Features Report.

## Provenance

The four-hook callback protocol and the symmetry-regularized baseline concept are adapted from the MIT-licensed `ssb_detection_ising` repository (Del Maestro Group, 2019). No source code was copied from that repository, and none of its domain machinery appears here. Its license is retained in `third_party/LICENSE_ssb_detection_ising.md`.