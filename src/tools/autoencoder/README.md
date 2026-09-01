# hQVM Group-Equivariant Autoencoder

## What this is

This project trains neural networks to represent the state space of the hQVM, a holonomic virtual machine defined in the [Formalism](docs/specs/hQVM_Specs_Formalism.md), [QuBEC](docs/specs/hQVM_QuBEC_Theory.md), and [SDK](docs/specs/hQVM_SDK_Quantum_Computing.md) specifications. The machine has exactly 4096 reachable states, and each of the 256 possible input bytes selects a fixed permutation of those states. Reading a byte moves the machine from one state to another, and every such move is exact and reversible.

An autoencoder is a network that learns to compress its input into a compact code and then rebuild the input from that code. What it keeps in the code is what it decided matters. This project trains autoencoders on the 4096 states and asks each one what it must keep in order to rebuild a state exactly, so the code becomes a readable summary of the state's structure.

The distinguishing property of these networks is that they respect the machine's exact symmetries by construction. The hQVM state space carries group actions: the four Klein gates that permute states holonomically, and the larger affine signature group. A network is equivariant when applying a symmetry to the input and then encoding gives the same code as encoding first and then applying the corresponding symmetry to the code. The models in this project satisfy that property exactly, before and after training, because the symmetry is built into the architecture rather than encouraged with a penalty.

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
| `narrow.py` | `MLPAutoencoder` | none | a plain encoder and decoder with no architectural symmetry; the null baseline. MLP = Multi-Layer Perceptron, a stack of simple neuron layers |
| `narrow.py` | `TransitionModel`, `RawByteTransitionModel`, `WordActionModel`, `FrameHead` | none | byte-conditioned task models (next state, raw byte, word, frame), supervised by kernel-exact targets |
| `narrow.py` | `PercolationLearner` | none | reads the packed 256-bit allowed byte mask and predicts the kernel-exact percolation labels (transport rank, reach, full/horizon/giant flags) |
| `general.py` | `K4Autoencoder` | exactly K4 | averages the encoder over the four Klein gates (Reynolds symmetrization), giving a latent split into `z_inv`, `z_chi`, `z_shell`, `z_irrep` |
| `super.py` | `SpectralAutoencoder` | exactly the full affine group | applies the Walsh transform, one scalar gain per irreducible symmetry block (64 one-dimensional and 2016 two-dimensional blocks), and inverts the transform |
| `super.py` | `MultiCellSpectral` | exactly the full affine group | the spectral model extended to a product register of cells |
| `__init__.py` | `UnifiedAutoencoder` (optional `MultiTaskHeads`) | free, k4, or full, selectable | the symmetry selector: one class reproducing the standalone classes exactly at each level; optional multi-task heads (transition/word/percolation rank) read per-block pooled features of the shared spectral latent, leaving the codec's exact equivariance untouched |

Flags select a model (`--model mlp`, `k4`, `spectral`, `transition`, `rawbyte`,
`word`, `percolation`, `unified`); a tier name (`narrow`, `general`, `super`) or
`all` selects every model in that tier for sweeping.

The spectral model is exact even before training: with all gains set to one it is the identity codec, rebuilding its input perfectly. Because each gain lives inside one symmetry block, the gains commute with the group action, so the model stays exactly equivariant for any gain values. The lossy-codec ladder selects which blocks stay on: `full` keeps everything, `diagonal` keeps only the 64 one-dimensional blocks, `shell`, `offdiagonal`, `shell_gauge` and `chirality_gauge` keep their named sectors, and `trivial` keeps the trivial block. Three tied rungs share one gain per symmetry class: `shell_radial` uses 7 gains keyed by the carrier weight, `shell_gauge` uses 28 keyed by the unordered shell pair, and `chirality_gauge` uses 56 keyed by the ordered shell pair and its AND parity. An optional L1 rate term learns which frequencies survive, and a denoising objective trains the gains on bath-corrupted states against clean targets, where the closed-form optimal gains are known. Byte-conditioned models predict the next state that a byte takes the machine to. Word and action models compose byte signatures through the kernel group law, so the network's composition of two word signatures equals the kernel's composition exactly.

## What the package contains

```
src/tools/autoencoder/
├── cli.py             all entry points: train, train-denoise, evaluate, verify, verify-groups, sample-ensemble, sweep-lambda, export-embeddings, audit-dictionary, generate, verify-full-g-exhaustive, genomics
├── kernel.py          kernel adapter: state indexing, stepping, gates, signatures, signature id packing, popcount6
├── datasets.py        transition and inverse tables, action and signature tables, byte/state census, manifests, invariants
├── corpus.py          the dictionary export: embeddings with exact kernel labels (charter artifact)
├── README.md          this document
├── __init__.py        package registry
│
├── models/            the only nested folder - three tiers + the plug
│   ├── __init__.py   the plug: MODEL_KINDS registry + build_model + UnifiedAutoencoder
│   ├── narrow.py     no structure built in: codecs, MLP, byte-mechanism predictors, percolation learner
│   ├── general.py    builds in the K4 gate symmetry: K4Autoencoder
│   └── super.py      builds in the full group / multi-register: SpectralAutoencoder, MultiCellSpectral
│
├── helpers/           flat helper package, naming convention <domain>_<role>.py
│   ├── training_run.py        trainer: four-hook callbacks, checkpointing, JSONL logs, seeding
│   ├── training_losses.py     weighted multi-objective losses + popcount_tensor (torch popcount)
│   ├── evals_run.py           checkpoint loading, evaluation, reports, benchmark suites, and verification (verify_k4/full-G, exhaustive full-G verifier, audit_dictionary, write_audit_report)
│   ├── evals_metrics.py       reconstruction/equivariance/transition/psi_hat metrics + probe_from_latent, shadow_invariance_error + kernel-exact readouts (climate, anisotropy, gauge, Z2 sheet, lift, code, denoiser, synthesizer, operator_structure, genomics, walsh_sector_energy, shell_distribution_ensemble)
│   ├── evals_datasets.py      eval dataset and corpus builders: ensembles, percolation, words, byte mechanism
│   ├── genomics.py            genomics compile adapter: 9-layer GenomicCompile, 24 NCBI nucleotide encodings, climate summary
│   └── ingest_genomics.py     populates dataset_genomics/ from the science catalog or public sources (--skip-network)
│
└── data/              ACTUAL DATA ONLY: *.npy + manifest.json. Zero .py files (regenerable via `cli generate`)
    ├── dataset_bytes/         byte census (kernel arrays)
    ├── dataset_states/        state census (kernel arrays)
    ├── dataset_transitions/   dense transition table
    ├── dataset_signatures/    8192-row group signature table
    ├── dataset_actions/       K4 action table
    ├── dataset_embeddings/    verified-dictionary corpus (identity export + per-checkpoint exports)
    ├── dataset_ensembles/     lambda-ensemble artifact (symmetry-breaking order parameter)
    ├── dataset_genomics/      frozen genomics catalog (ingested once via `helpers.ingest_genomics`)
    ├── checkpoints/           trained model weights (gitignored)
    ├── reports/               eval/verify/audit JSON reports (gitignored)
    └── tmp/                   scratch space (gitignored)
```

## Quick start

```bash
# Generate the kernel-derived datasets (byte census, state census, transition tables, signatures)
python -m src.tools.autoencoder.cli generate --dataset all

# Train a model
python -m src.tools.autoencoder.cli train --model spectral --epochs 5          # identity codec baseline
python -m src.tools.autoencoder.cli train --model spectral:shell --epochs 5    # lossy codec rung
python -m src.tools.autoencoder.cli train --model k4 --epochs 5                # K4 equivariant
python -m src.tools.autoencoder.cli train --model unified --symmetry full --epochs 5   # unified, full symmetry
python -m src.tools.autoencoder.cli train --model transition --task transition --epochs 5  # byte-conditioned next-state model
python -m src.tools.autoencoder.cli train --model word --task word --epochs 5   # per-byte signature model
python -m src.tools.autoencoder.cli train --model percolation --task percolation_rank --epochs 5  # rank-recovery learner
python -m src.tools.autoencoder.cli train --model unified --task unified_multi --symmetry full --epochs 5  # one shared latent, all four objectives
python -m src.tools.autoencoder.cli train-denoise --ladder shell_radial --noise-rate 0.03,0.03,0.03,0.03,0.03,0.03  # spectral denoiser

# Verify that a checkpoint is exactly equivariant (closed-form full-G certificate)
python -m src.tools.autoencoder.cli verify-equivariance --checkpoint checkpoints/model.pt

# Export the verified dictionary and run the one-pass audit
# (no --output-dir: defaults to src/tools/autoencoder/data/dataset_embeddings/
#  identity export; a trained checkpoint writes <name>_<suffix>.npy so it
#  coexists with the identity export instead of overwriting it)
python -m src.tools.autoencoder.cli export-embeddings
python -m src.tools.autoencoder.cli audit-dictionary --report-file reports/embedding_corpus_audit.json

# Run the lambda-ensemble experiment (symmetry-breaking order parameter)
python -m src.tools.autoencoder.cli sweep-lambda --model mlp --epochs 15 --n 16384

# Genomics compile adapter (data-only; no model involved)
# 1) Populate the genomics catalog (one-time, deterministic; --skip-network hashes an existing copy):
python -m src.tools.autoencoder.helpers.ingest_genomics
# or point at a different science checkout:
python -m src.tools.autoencoder.helpers.ingest_genomics --science-catalog /path/to/science/data/catalogs/genomics
# or disable the science-copy branch entirely (download from public URLs only):
python -m src.tools.autoencoder.helpers.ingest_genomics --science-catalog ""
# 2) Compile any sequence file to the certified 9-layer GenomicCompile:
python -m src.tools.autoencoder.cli genomics --input-file src/tools/autoencoder/data/dataset_genomics/ecoli_k12_cds.fna.gz --max-bases 500000 --enc 0
```

All commands accept the long GNU-style flags (`--output-dir`, `--run-name`,
`--learning-rate`, `--noise-rate`, `--report-file`). The short forms
`--out`, `--name`, `--lr`, and `--eta` are accepted as aliases for this
release and will be removed at the next boundary.

Datasets are written to `src/tools/autoencoder/data/` with a JSON manifest per directory recording the schema version, a kernel fingerprint, array shapes and dtypes, and invariant-check results. The directory is regenerable and gitignored.

Model selection is by `--model`: one of the individual kinds (`mlp`, `k4`, `spectral`, `transition`, `rawbyte`, `word`, `percolation`, `unified`) or a `spectral:<ladder>` rung. The tier names `narrow`, `general`, `super`, and `all` select every model in that tier (used for sweeps); for `train` a tier maps to its first member. `build_model` in `models/__init__.py` is the single constructor, so the CLI never keeps a parallel copy of the model defaults. `--hidden-dim` overrides the per-kind default width (default: 128 for narrow models, the codec's own value for spectral); useful when an existing checkpoint was trained at a non-default width.

Trained production artifacts live in `src/tools/autoencoder/data/checkpoints/production/` (`spectral_full`, `spectral_bottleneck`, `k4_full`, `mlp_full`, and the `spectral_denoise` denoiser), each with a per-checkpoint evaluation and equivariance report in `src/tools/autoencoder/data/reports/` and a compiled `production_summary.json`. The denoiser adds a `denoiser_gain_report` to its evaluation (machine-checked `pass` flag with the published `tol`; the trained gains track the closed-form shrinkage multipliers). Two dictionary corpora exist side by side in `src/tools/autoencoder/data/dataset_embeddings/`: the bare-filename identity export and the trained-checkpoint export whose files are suffixed with the cleaned checkpoint name (e.g. `byte_byte_spectral_bottleneck.npy`, `manifest_spectral_bottleneck.json`). The trained `spectral_bottleneck` export is audited green by `audit-dictionary`. The closed-form full-G certificate `src/tools/autoencoder/data/reports/exhaustive_full_g_verify.json` covers all three trained spectral checkpoints (full, bottleneck, denoiser). Every report path in the published JSON uses forward slashes so the bundle is portable across OSes.

## Production regeneration

```bash
# One-shot: retrain the five checkpoints and write the matching eval,
# equivariance, and closed-form full-G reports, plus production_summary.json.
python -m src.tools.autoencoder.scripts.make_production

# Or: reuse the existing checkpoints and only regenerate the reports.
python -m src.tools.autoencoder.scripts.make_production --skip-train
```

`tests/tools/autoencoder/test_report_schemas.py` pins the report schema so any drift between the CLI and the published JSON fails on the next `pytest` run.

## Kernel core vs external adapters

The model core knows only the carrier grammar and the kernel's exact group
interfaces. Everything outside that - physical observables, genomics, and
LLM-weight tensors - is an **external adapter**, not a new model and not a new
model tier. Adapters are pure data transforms that read the census/byte
surfaces already exposed by the package (`datasets.byte_census_arrays`,
`helpers.evals_metrics.genomics_compile`, `helpers.genomics.compile_interval`)
and produce the exact columns the readouts consume. They never add a kernel fact to a model file and never
reformulate a transition rule; they only convert external material into the
structure the codec already understands. This is the boundary that keeps the
three tiers clean: the models learn or represent hQVM structure; adapters
compile external material into that structure; task heads make
application-specific predictions; readouts measure the resulting structure.

## Adding a model or specialization

A new model joins the tier whose symmetry it builds in (a new group would
justify a new tier file, nothing else). Before it ships it must satisfy:

1. **Symmetry containment** decides the tier: `narrow` (no built-in symmetry),
   `general` (K4 gates), `super` (full affine group / multi-register).
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
labels, sitting in `narrow`, needing no external data. Genomics and physics
probes are adapters, not models.

## Where empirical data enters

Base training is kernel-null: the models self-supervise on exact labels, which
is why the four production artifacts exist. Empirical data (genomics
sequences, LLM-weight matrices) is **not** mixed into the core models. It
enters as a frozen-codec head fine-tune: keep the codec's exact equivariance
intact and train only the task heads on compiled windows
(`--task empirical --init <production checkpoint>`, heads only). For LLM
weights specifically, tile the matrix to 64-wide blocks, compile the byte
stream through the census, keep the codec frozen, and read the block
features - no new "LLM tier" is needed. Genomics follows the same adapter pattern: a data-only transform of the sequence through the certified 9-layer `GenomicCompile` (byte_fold_w, fold_poles, family_sheet, omega_signature, depth4_parity, chi_shells, qubec_order, ab_horizon, boundary_keys). The catalog is populated once by `python -m src.tools.autoencoder.helpers.ingest_genomics` (or `--skip-network` against an existing copy) into `data/dataset_genomics/`, and any sequence window is compiled by `cli genomics --input-file ...`. Nothing is re-derived; every value is read from the compile layers.

## Verification

The suite asserts exact kernel relations rather than learned proxies:

- K4 equivariance holds to an error below 1e-4 over all 4096 states and all four gates: `E(gx) = ρ(g)E(x)` and `D(ρ(g)z) = P_g D(z)`.
- Full-group equivariance holds with zero observed error: the spectrum of `gx` equals `ρ(g)` applied to the spectrum of `x`. The closed-form certificate across all 4096 states and all 8192 signatures (a sub-second condition on the gain symmetry) is persisted to `src/tools/autoencoder/data/reports/exhaustive_full_g_verify.json`.
- Word composition holds exactly: the network's composition of two word signatures equals the kernel's.
- Depth-four frames compiling to pure translations is a kernel theorem cited from the Features Report (#64), and audited in the dictionary's frame-parity check.
- The two-byte witness routing count is a kernel theorem cited from the Features Report (#80/#120); it is not re-derived inside the suite.
- The dictionary audit recomposes reconstruction, equivariance, closed-form factorization probes, the H-invariance of the diagonal rung, shadow invariance, frame parity, and the psi_hat character-energy identity.

The test suite lives in `tests/tools/autoencoder/`. The pytest collector is the single source of truth for the count (`pytest tests/tools/autoencoder --collect-only -q`): 204 tests are collected. The default run executes 202 (the closed-form full-G regression in `test_scale.py` and the denoiser training smoke in `test_task_training.py` are gated behind `--runslow`); all 202 pass on CPU. The suite covers the autoencoder's own objects: models, losses, metrics, bottlenecks, datasets, artifacts, the genomics compile adapter, the production report schemas, and the mapping to the kernel. Kernel feature facts are cited from the Features Report, never re-proved here.

## Provenance

The four-hook callback protocol and the symmetry-regularized baseline concept are adapted from the MIT-licensed `ssb_detection_ising` repository (Del Maestro Group, 2019). No source code was copied from that repository, and none of its domain machinery appears here. Its license is retained in `third_party/LICENSE_ssb_detection_ising.md`.