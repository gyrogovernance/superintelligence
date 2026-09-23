# hQVM AE Genomics Program Specification

## What this program is

The hQVM AE Genomics program applies the grammar-trained autoencoder models from the hQVM autoencoder package to biological sequences. The models train on sequences generated from the hQVM coding-sequence coordinate system. The representation they carry is that grammar. The program measures how that coordinate system and those models contact real biological sequence.

This is a program specification. It describes what the program is, how the analyses are built, and the estimand of each analysis. Numbers, pass/fail records, and file inventories live in the genomics report.

The program has two domains. **Synthesis** evaluates the frozen models on genomic catalogs for artificial gene synthesis and design capacity. **Topology** studies codon order and membrane topology in *E. coli*, and includes a fixed-peptide analysis on the Gamble 2016 synonymous variant library. Together they cover three levels of sequence organization: whole genomic catalogs, individual genes, and synonymous codon order within a fixed peptide. Every claim in the program is a scoring claim: it ranks or locates native sequences relative to the grammar. Masked completion is certified by the production gates, so the same measure can also be sampled from.

## Why the program is built this way

The program rests on a separation that the autoencoder package is built to enforce. The coordinate system represents nucleotides and codons as exact algebraic states. Adjacent codons form ordered transitions, and the kernel assigns each transition a local cost called shell. A coding sequence is a path through these states, and the kernel computes exact properties of that path. The autoencoders learn to reconstruct masked positions in grammar-generated paths and provide learned summaries of sequence order. The shell cost is a path quantity, the same class of object the field's kinetic design literature uses when it separates folding paths from endpoint energies.

The coordinate system was derived from the CGM axioms. The independent [genomics analysis](https://github.com/gyrogovernance/science/blob/main/docs/Findings/Analysis_hQVM_CGM_Genomics.md) records that derivation and its 301 verification checks. The Topology contrast the program reports is a consequence of the coordinate system's cost function.

Training uses no biological sequence. Reproducible contact between these readouts and genomic structure, under composition controls, is therefore a mark of the underlying physics. Coherent marks at modest effect size are the expected regime, and every analysis is built for that scale.

The RNA design literature weighs a candidate sequence by uniform sampling, thermodynamic energy, or a prior learned from biological families. A measure induced by a formal grammar holds a separate seat. The field's measures treat agreement with biology as the objective. This program's measure treats grammar-biology contact as the signal.

The program works across both domains because one level of organization is incomplete. Synthesis asks how the models behave on real genomic catalogs. Topology asks whether adjacent codon-pair shell associates with membrane topology within individual genes, and whether a frozen model readout ranks synonymous expression when the peptide is held fixed. Each level has its own controls and estimand.

## The models the program uses

The program evaluates frozen model components from the autoencoder package. Narrow and Super are the two learned components it uses.

Narrow provides a compact representation of each Topology window. Super performs masked prediction and supplies an aggregate window readout called climate. Randomly initialized models with the same architectures provide a comparison for the contribution of trained weights.

The hQVM kernel compiles a DNA window into a sequence of carrier states. It records exact path properties, including parity, complementarity, word signatures, and the shell cost of adjacent codon pairs. These quantities are computed from the coordinate system. The kernel statistic used most directly in Topology is the mean shell of adjacent codon pairs.

Randomly initialized models with the same architectures are the controlled comparison that shows what published weights add beyond architecture alone.

## Synthesis domain

The Synthesis domain evaluates the frozen models on genomic catalogs. It asks four distinct questions.

Does the K4 layer preserve the exact symmetry required by the coordinate system? Do real coding sequences occupy the carrier in a coherent, measurable way? Do the frozen models distinguish biological sequence order from shuffled or synonymous alternatives? Do the model readouts remain informative when sequence composition is held fixed?

The suite evaluates K4, Narrow, spectral, and Super components on four catalogs: *E. coli*, yeast, SARS-CoV-2, and human chromosome 22 splice flanks. Comparison datasets preserve selected properties such as GC content, amino-acid sequence, or byte composition so that the tested signal is explicit. The symmetry evaluation checks that the K4 layer preserves the required group symmetry on real sequence. The distributional comparison measures how living catalogs sit relative to the uniform carrier atlas. The order-memory test constructs sequence pairs with the same algebraic signature and byte multiset while changing only path order, and measures whether trained Super retains information about order beyond the fixed carrier quantities. The composition-controlled tests compare Super readouts against GC-matched shuffles and against protein-fixed synonymous resamples.

Each evaluation is recorded with the comparison used for it. The readable account is in RESULTS.txt. The machine-readable form is in gates.json. A number is interpretable together with the comparison it was measured against.

## Topology domain

The Topology domain studies codon order and membrane topology in *E. coli*. It compares coding windows annotated as transmembrane with cytoplasmic windows from the same gene. This within-gene comparison keeps the organism, gene, encoded protein, and host codon-usage background fixed while the annotated topological domain changes. The main kernel measurement is mean shell for adjacent codon pairs.

The study is built as a pipeline with six stages, and each stage has a job.

The atlas stage assembles the within-gene topology cohort and computes the primary measurements. The null stage redraws synonymous codons while preserving the encoded protein, including a version that follows observed host codon usage. The narrow stage scores the frozen Narrow model on the atlas windows. The claim stage verifies the primary kernel result and its source and cohort records. The holdout stage tests the incremental predictive value of frozen model readouts with genes kept together across training and test folds. The order stage tests codon-order ranking against Gamble expression measurements.

The null analysis answers specific questions with separate controls. An amino-acid-preserving null redraws codons while keeping the encoded protein fixed. A usage-matched null samples synonymous codons using the host's observed codon frequencies. Boundary composition is examined two ways. Linear adjustment for codon-boundary dinucleotide content tests a residual after a broad composition correction. Exact preservation of every boundary dinucleotide isolates junction composition as its own channel by fixing the bases where synonymous freedom normally sits. That control places the Topology contrast in codon-identity order, the same design freedom the Synthesis domain scores. Length matching pairs each transmembrane window with the closest cytoplasmic window from the same gene and keeps length differences within 10%.

These controls answer different questions, and the program reports each one separately. The linear adjustment measures the residual after a broad composition correction. The exact match localizes the contrast to codon-identity order once junction composition is pinned. Each answer has its own use.

The claim stage gates on the amino-acid-preserving residual. The usage-matched residual carries a lower effect size and is reported for the host-usage question.

The held-out analysis keeps genes together in cross-validation folds so that windows from one gene stay on one side of the train-test split. The baseline contains kernel shell, GC content, window length, and the model read-truncation flag. The question is whether adding frozen model readouts improves classification of topology beyond that baseline. Narrow supplies the main unique contribution over the controlled baseline. Adding Super to Narrow gives a further increment. Super alone gives a much smaller one. That pattern shows which model contributes what.

The program also records the frozen Narrow and Super readouts on the atlas windows directly, not only in the held-out classification. Narrow heterogeneity has a mean transmembrane-minus-cytoplasmic difference. Super climate has a mean difference in the opposite direction. These are descriptive measurements on the cohort, separate from the classification question.

## Fixed-peptide analysis (Topology domain)

The fixed-peptide analysis uses Gamble 2016 Table S1, measured in yeast, which contains 35,811 nine-base variants across 5,148 encoded tripeptides. Within each tripeptide, the encoded protein and sequence length are fixed while codon order varies, and expression is measured for every variant in the synGFPSEQ expression column. This design isolates synonymous order while holding peptide composition fixed.

The eligible analysis set contains 28,504 variants in 2,597 peptide groups with at least five synonymous spellings. Frozen Super climate, Super context, Narrow heterogeneity, and kernel mean shell are scored before expression enters the analysis. GC content and codon-count entropy are reported as additional composition controls.

The primary readout is Super climate. The analysis measures the within-peptide Spearman correlation with the reported expression measure, and it also runs a peptide-held-out evaluation across folds. The estimand is ranking with expression under fixed peptide identity.

The held-out evaluation checks that the same mean rank survives when whole peptide groups are held out, so the signal is measured across peptide groups.

## Scope of the estimands

The program reports associations and ranked signals. The Topology estimand is association between adjacent codon-pair order and annotated membrane topology within genes. The fixed-peptide estimand is ranking of synonymous expression under fixed peptide identity. The Synthesis estimand is exact symmetry and measurable order information on real genomic catalogs. Each analysis has its own estimand. Causal mechanism statements sit outside those estimands.

## Inputs and outputs

The program uses the frozen model files and evaluation reports from the autoencoder package, plus public biological catalogs. The repository includes the frozen model files and evaluation reports needed to run the analyses. Large source catalogs, derived datasets, and temporary files are generated locally.

The Synthesis domain uses the catalogs created by the genomics ingest command and the frozen checkpoints from the autoencoder production directory. The Topology domain uses the *E. coli* CDS catalog, the pinned UniProt K-12 proteome TSV, the Gamble Table S1 workbook, and the frozen production checkpoints. The Gamble workbook is distributed with the journal supplement and is saved with a specific filename so the ingest check can verify its checksum.

Each analysis writes a readable RESULTS.txt and a set of machine-readable JSON result files. The Topology pipeline reuses a stage result when its JSON file is present and its recorded script hash matches the current script. This is what makes partial reruns safe. When one stage changes, stages that are still current stay in place.

## Reproducibility

The program is meant to be rerunnable from the public catalogs. The ingest commands obtain and verify the inputs. The Synthesis and Topology commands run the analyses. The Topology command can rebuild the full pipeline or selected stages, which matters when a script changes or when only one stage needs to be repeated.

The independent analysis that defines the coordinate system is maintained separately. The genomics program depends on that coordinate system and inherits its updates.
