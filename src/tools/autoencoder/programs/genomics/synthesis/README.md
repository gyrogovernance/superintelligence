# Synthesis domain

The Synthesis domain of the hQVM AE Genomics program evaluates the frozen hQVM model components on real genomic catalogs for artificial gene synthesis and design capacity. The models train on grammar-generated sequences. Reproducible contact with living catalogs under composition controls is the measured signal.

The domain asks four questions:

1. Does the K4 layer preserve the exact symmetry required by the coordinate system?
2. Do real coding sequences occupy the carrier in a coherent, measurable way?
3. Do the frozen models distinguish biological sequence order from shuffled or synonymous alternatives?
4. Do the model readouts remain informative when sequence composition is held fixed?

The evaluation uses *E. coli*, yeast, SARS-CoV-2, and human chromosome 22 splice-flank catalogs. Comparison datasets preserve selected properties such as GC content, amino-acid sequence, or byte composition so that the tested signal is explicit.

## Run

Prepare the catalogs, then run the complete suite:

```text
python -m src.tools.autoencoder.programs.genomics.ingest_genomics
python -m src.tools.autoencoder.programs.genomics.synthesis.run
```

Useful subsets are:

```text
python -m src.tools.autoencoder.programs.genomics.synthesis.run --only g4
python -m src.tools.autoencoder.programs.genomics.synthesis.run --hosts ecoli,yeast
```

## Inputs

The Synthesis domain uses the catalogs created by `ingest_genomics.py` and the frozen checkpoints in `src/tools/autoencoder/data/checkpoints/production/`. The checkpoints are `k4_full.pt`, `mlp_full.pt`, `spectral_bottleneck.pt`, and `super.pt`.

## Outputs

`RESULTS.txt` gives a readable account of every evaluation. `gates.json` contains the same results in a machine-readable form, including the comparison used for each test.

## Program files

| File | Purpose |
|------|---------|
| `run.py` | Command-line entry point for the Synthesis suite |
| `cli.py` | Catalog and reference-climate command-line tools |
| `censuses.py` | Evaluation definitions, comparisons, and result assembly |
| `constellation.py` | Frozen and randomly initialized model loading |
| `reads.py` | Host-specific catalog preparation |
| `climate.py` | Sequence-level model and kernel readouts |
| `field.py` | Exact path-order features |
| `registry.py` | Names and inclusion records for the evaluations |

See the parent [genomics README](../README.md) for the shared coordinate system, data sources, and complete program context.
