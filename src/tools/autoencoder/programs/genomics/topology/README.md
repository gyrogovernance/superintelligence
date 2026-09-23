# Topology domain

The Topology domain of the hQVM AE Genomics program studies codon order and membrane topology in *E. coli*. It compares coding windows annotated as transmembrane with cytoplasmic windows from the same gene. This within-gene comparison keeps the organism, gene, encoded protein, and host codon-usage background fixed while the annotated topological domain changes.

The main kernel measurement is mean shell for adjacent codon pairs. The domain also evaluates frozen Narrow and Super model readouts on the same windows. A related analysis ranks expression in the Gamble 2016 library of synonymous variants that encode the same peptide, using Super climate.

## Design

The atlas stage assembles 590 eligible *E. coli* K-12 integral membrane proteins and computes the transmembrane-minus-cytoplasmic contrast for each gene. The null stage redraws synonymous codons while preserving the encoded protein, including a version that follows observed host codon usage. The holdout stage tests whether frozen model features improve classification of topology when genes are kept together across training and test folds.

The order stage uses Gamble Table S1. Each peptide group contains synonymous nine-base variants with the same encoded amino acids. The analysis measures whether model readouts rank with the reported expression values inside each peptide group.

## Pipeline

| Stage | Purpose |
|-------|---------|
| `atlas` | Build the within-gene topology cohort and compute the primary measurements |
| `null` | Test the result against amino-acid-preserving and usage-matched synonymous alternatives |
| `narrow` | Score the frozen Narrow model on the atlas windows |
| `claim` | Verify the primary kernel result and its source and cohort records |
| `holdout` | Test the incremental predictive value of frozen model readouts |
| `order` | Test codon-order ranking against Gamble expression measurements |

## Run

Prepare the inputs and run the full pipeline:

```text
python -m src.tools.autoencoder.programs.genomics.ingest_genomics
python -m src.tools.autoencoder.programs.genomics.ingest_topology --check
python -m src.tools.autoencoder.programs.genomics.topology.run
```

Rebuild the pipeline or selected stages when needed:

```text
python -m src.tools.autoencoder.programs.genomics.topology.run --force
python -m src.tools.autoencoder.programs.genomics.topology.run --from order
python -m src.tools.autoencoder.programs.genomics.topology.run --only order
```

## Inputs

The domain uses the *E. coli* CDS catalog from `dataset_genomics/`, the pinned UniProt K-12 proteome TSV, the Gamble Table S1 workbook, and the frozen production checkpoints. The Gamble workbook is obtained from the journal supplement and saved as `data/dataset_topology/gamble_2016__NIHMS800838-supplement-6.xlsx`.

## Outputs

`RESULTS.txt` summarizes every stage. Each `RESULTS_script_*.json` file contains the machine-readable result and provenance for one stage. The `run.py` command reuses a stage result when its JSON file is present and its recorded script hash matches the current script.

## Program files

| File | Purpose |
|------|---------|
| `run.py` | Runs the Topology pipeline and stage selection |
| `common.py` | Shared cohort, scoring, and result helpers |
| `script_1_atlas.py` | Builds the within-gene topology atlas |
| `script_2_null.py` | Runs synonymous and usage-matched null analyses |
| `script_3_narrow.py` | Computes frozen Narrow readouts |
| `script_4_claim.py` | Verifies the primary kernel result |
| `script_5_holdout.py` | Runs gene-grouped held-out classification |
| `script_6_order.py` | Runs the Gamble fixed-peptide order analysis |

See the parent [genomics README](../README.md) for data acquisition and the shared scientific background.
