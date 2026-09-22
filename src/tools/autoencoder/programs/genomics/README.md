# hQVM AE Genomics

The Genomics Program advances programmable nucleic acid research on DNA and RNA through grammar-trained autoencoders of the hQVM AE suite, grounded in first principles. The models read biological sequences through the group-equivariant coordinate system and a formal algebra for nucleotides, codons, and codon-pair transitions derived from our CGM theory.

With training restricted to that coordinate system and no biological sequence in the corpus, every reproducible contact between readout and genomic structure that survives composition controls is a signature of the underlying physics. Even the faintest of those marks remains informative, and opens a concrete frontier for genomics.

## Theory: Physics of Genomics

The hQVM coding-sequence coordinate system, as set out in our theoretical analysis, represents a nucleotide as a two-bit coordinate, a codon as a point of the 64-state horizon, and an ordered codon pair as a point of the 4096-state carrier. Adjacent codons form ordered transitions to which the kernel assigns a local cost called shell; a coding sequence is a path through those states whose exact properties the kernel computes.

The autoencoders learn to reconstruct masked positions in grammar-generated paths and to provide learned summaries of sequence order. The coordinate system was derived from the CGM axioms, and the independent [genomics analysis](https://github.com/gyrogovernance/science/blob/main/docs/Findings/Analysis_hQVM_CGM_Genomics.md) records the derivation and its 301 verification checks.

## Sub-Programs

### **Synthesis**: Artificial Gene Synthesis and Design.

Synthetic genomics seeks to design and construct entire genomes to mechanistically dissect fundamental questions and advance research focused on health and medicine. Our probes aim to inform artificial gene synthesis and design, opening alternative paths for nucleic-acid work under synonymous freedom: codon-order rearrangements are scored and ranked while protein and composition remain fixed. 

CGM Genomics Analysis supplies the features, while our code maps each feature to a published AE layer (Narrow, K4, spectral, or Super), attributing credit across the deterministic kernel, architecture-visible structure, and trained weights. The primary design quantities are path memory (order-dependent displacement along the byte walk) and climate (shell census, radial modes, and ledger summaries), evaluated on *E. coli*, yeast, SARS-CoV-2, and human chromosome 22 splice-flank catalogs.

Entry point: `python -m src.tools.autoencoder.programs.genomics.synthesis.run`. See the [synthesis README](synthesis/README.md).

### **Topology**: Biological Membrane Topology Analysis.

Membrane topology describes the number of membrane-spanning segments in a protein and how its parts orient relative to the inside and outside of a biological membrane. The program expands into a frontier analysis of membrane-topology inside individual *E. coli* genes. 

For each eligible integral membrane protein, transmembrane-helix coding windows are compared with cytoplasmic windows of the same gene, so the organism, gene, encoded protein, and host codon-usage background remain fixed while the annotated topological domain changes. 

The primary kernel measurement is mean shell for adjacent codon pairs, with frozen Narrow and Super reads scored on the same windows.

A related capacity holds the peptide fixed and ranks measured expression across synonymous spellings in the Gamble 2016 yeast library, with Super climate as the ranking read.

Entry point: `python -m src.tools.autoencoder.programs.genomics.topology.run`. See the [topology README](topology/README.md).

Full results for both programs are in [hQVM_AE_Genomics_Report.md](../../../../../docs/reports/hQVM_AE_Genomics_Report.md); design and methodology are in the [genomics specification](../../../../../docs/programs/hQVM_AE_Genomics_Specs.md).

## Results at a glance

Within a single *E. coli* gene, transmembrane-helix coding uses codon pairs with lower grammar cost than the cytoplasmic domains of that same gene: a mean shell difference of `-0.2262`, negative in 492 of 590 genes, with `p = 0.0002`, and the contrast holds under protein-fixed and usage-matched nulls.

Across the suite the same grammar signatures appear. A frozen Narrow read adds `+0.009094` held-out AUC over a controlled baseline. Exact K4 symmetry holds to a maximum error of `3.32e-11`, and trained-minus-random order-memory margins run between `+0.1931` and `+0.2271` across the tested catalogs. Against GC-matched shuffles, Super climate AUC is `0.8408` in *E. coli* and `0.7113` in yeast. In the Gamble 2016 synonymous variant library, the within-peptide Spearman correlation of Super climate with expression is `+0.0196` with `p = 0.0038`.

## Run the analyses

Obtain the public catalogs and verify the topology inputs before running the programs:

```text
python -m src.tools.autoencoder.programs.genomics.ingest_genomics
python -m src.tools.autoencoder.programs.genomics.ingest_topology --check
python -m src.tools.autoencoder.programs.genomics.synthesis.run
python -m src.tools.autoencoder.programs.genomics.topology.run
```

The topology command can rebuild the full pipeline or selected stages:

```text
python -m src.tools.autoencoder.programs.genomics.topology.run --force
python -m src.tools.autoencoder.programs.genomics.topology.run --from order
python -m src.tools.autoencoder.programs.genomics.topology.run --only order
```

## Data and repository contents

The repository includes the frozen model files and evaluation reports needed to run the analyses. Large source catalogs, derived datasets, and temporary files are generated locally.

| Path | Source | Used by |
|------|--------|---------|
| `data/checkpoints/production/*.pt` | Included in the repository | Synthesis and topology |
| `data/reports/*.json` | Included in the repository | Model and gate reports |
| `data/dataset_genomics/` | Downloaded by `ingest_genomics.py` | Synthesis and topology |
| `data/dataset_topology/uniprot_ecoli_k12.tsv` | Downloaded by `ingest_topology.py` | Topology atlas and null analyses |
| `data/dataset_topology/gamble_2016__NIHMS800838-supplement-6.xlsx` | Obtained from the journal supplement | Fixed-peptide order analysis |
| `data/dataset_null/`, `dataset_ensembles/`, `dataset_embeddings/` | Rebuilt by the autoencoder tools | Training and dictionary analyses |

The Gamble workbook is distributed with the journal supplement. Save it as `gamble_2016__NIHMS800838-supplement-6.xlsx` in `data/dataset_topology/`, then run `ingest_topology.py --check` to verify its checksum.

## Terms used in the reports

- **Shell:** The local cost the fixed hQVM kernel assigns to an adjacent codon pair, so a coding sequence carries a path cost rather than an endpoint score.
- **Carrier path:** A DNA window compiled by the grammar to a trajectory through the 4096-state carrier. Such a window is also called a grammar-generated path.
- **Order-memory margin:** The trained-minus-random increment on permutation pairs that share a signature and byte multiset but differ in order. It is the instrument certificate for the learned measure.
- **Super climate:** An aggregate model readout computed across a sequence window.
- **Narrow read:** A window-level representation produced by the frozen Narrow model.
- **Usage-matched null:** A synonymous comparison set sampled using observed *E. coli* codon frequencies.
- **Holdout:** A gene-grouped evaluation in which genes used for fitting are kept separate from genes used for testing.
