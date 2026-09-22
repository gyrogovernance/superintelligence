# hQVM Group-Equivariant Autoencoder

## What this is

A neuro-symbolic autoencoder suite over a finite group-structured state space. The models build their symmetries into the architecture rather than encouraging them with a penalty, and the package ships an exact K4 equivariant autoencoder, a closed-form equivariance certificate for the full affine group on the spectral codec, a Bayes-optimal spectral denoiser with analytic gains, and the Super process model.

The underlying machine is the hQVM, defined in the [Formalism](../../../docs/specs/hQVM_Specs_Formalism.md), [QuBEC](../../../docs/specs/hQVM_QuBEC_Theory.md), and [SDK](../../../docs/specs/hQVM_SDK_Quantum_Computing.md) specifications. The kernel is the only authority for the machine. No transition rule, gate, mask, or intron is reformulated in the learning stack, and no dataset is generated outside the kernel adapter.

For the theory, the model tiers, the state space, the null dataset, and the rules for adding a model, see [hQVM_AE_Specs.md](../../../docs/specs/hQVM_AE_Specs.md). For measured results and the gate record, see [hQVM_AE_Report.md](../../../docs/reports/hQVM_AE_Report.md). This file covers how to run the code.

## Layout

```
src/tools/autoencoder/
├── cli.py             entry points: train, train-denoise, evaluate, verify, verify-groups, sample-ensemble, sweep-lambda, export-embeddings, audit-dictionary, generate, verify-full-g-exhaustive
├── kernel.py          kernel adapter: state indexing, stepping, gates, signatures, signature id packing, popcount6
├── datasets.py        kernel tables, census, null corpus (dataset_null + NullCorpus), manifests
├── corpus.py          dictionary export: embeddings with exact kernel labels
├── paths.py           data/ layout
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
    ├── dataset_null/          CGM null permutation atlas
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
python -m src.tools.autoencoder.cli export-embeddings
python -m src.tools.autoencoder.cli audit-dictionary --report-file reports/embedding_corpus_audit.json

# Run the lambda-ensemble experiment (symmetry-breaking order parameter)
python -m src.tools.autoencoder.cli sweep-lambda --model mlp --epochs 15 --n 16384
```

Model selection is by `--model`. The individual kinds are `mlp`, `k4`, `super`, `transition`, `rawbyte`, `word`, and `percolation`. The tier names `narrow`, `general`, `super`, and `all` select every model in that tier for sweeps; for `train` a tier maps to its first member. `build_model` in `models/__init__.py` is the single constructor, and `--hidden-dim` overrides the per-kind default width.

All commands accept the long GNU-style flags (`--output-dir`, `--run-name`, `--learning-rate`, `--noise-rate`, `--report-file`). The short forms `--out`, `--name`, `--lr`, and `--eta` are accepted as aliases for this release and will be removed at the next boundary.

Datasets are written to `src/tools/autoencoder/data/` with a JSON manifest per directory recording the schema version, a kernel fingerprint, array shapes and dtypes, and invariant-check results. The directory is regenerable and gitignored.

## Production regeneration

```bash
# One-shot: train narrow/general nulls, run Super gates, write production_summary.json.
python -m src.tools.autoencoder.helpers.training_super production

# Or: reuse the existing checkpoints and only regenerate the reports.
python -m src.tools.autoencoder.helpers.training_super production --skip-train
```

Trained production artifacts live in `data/checkpoints/production/` (`super`, `k4_full`, `mlp_full`, plus `spectral_bottleneck` for the denoiser and dictionary probes), with reports in `data/reports/` including `super_gates.json` and `production_summary.json`. `tests/tools/autoencoder/test_report_schemas.py` pins the report schema, so any drift between the CLI and the published JSON fails on the next `pytest` run.

## Tests

```bash
python -m pytest tests/tools/autoencoder/
python -m pytest tests/tools/autoencoder/ --runslow
```

The suite is documented domain by domain in the [report](../../../docs/reports/hQVM_AE_Report.md).

## Provenance

The four-hook callback protocol and the symmetry-regularized baseline concept are adapted from the MIT-licensed `ssb_detection_ising` repository (Del Maestro Group, 2019). No source code was copied from that repository, and none of its domain machinery appears here. Its license is retained in `third_party/LICENSE_ssb_detection_ising.md`.
