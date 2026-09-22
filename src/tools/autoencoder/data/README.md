# Autoencoder data directory

Layout for `src/tools/autoencoder/data/`.

| Path | Git | Role |
|------|-----|------|
| `checkpoints/production/` | shipped | Frozen production weights (`k4_full.pt`, `mlp_full.pt`, `spectral_bottleneck.pt`, `super.pt`) |
| `reports/` | shipped | Gate and eval JSON indexed by `production_summary.json` |
| `dataset_genomics/` | local | Community catalogs from `programs/genomics/ingest_genomics.py` |
| `dataset_topology/` | local | UniProt TSV from `ingest_topology.py`; Gamble Table S1 placed by hand |
| `dataset_null/`, `dataset_ensembles/`, `dataset_embeddings/` | local | Regenerable AE training and dictionary artifacts |
| `tmp/` | local | Scratch; safe to delete |

Everything under this tree is gitignored except `checkpoints/production/` and `reports/`. Rebuild catalogs with the genomics ingest scripts; rebuild null and embedding corpora with the autoencoder CLI.
