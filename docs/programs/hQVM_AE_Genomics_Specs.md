# hQVM AE Genomics Program Specification

## What this program is

The hQVM AE Genomics program takes the grammar-trained autoencoder models from the hQVM autoencoder package and applies them to biological sequences from three angles. The models were trained only on sequences generated from the hQVM coding-sequence coordinate system, so the representation they carry is the grammar's own and nothing borrowed from biology. The program tests whether structure learned from that coordinate system is still measurable when you look at real biological sequence.

This is a program specification. It describes what the program is, how the analyses are built, and what they can and cannot claim. It does not report results. If you want the numbers, the pass/fail records, and the file inventories, those are in the genomics report.

The program contains two analyses. The synthesis analysis evaluates the frozen models on genomic catalogs. The topology analysis studies the relationship between codon order and membrane topology in E. coli, and it also runs a separate fixed-peptide analysis on the Gamble 2016 synonymous variant library. The two are related but independent. Together they cover three levels of sequence organization: whole genomic catalogs, individual genes, and synonymous codon order within a fixed peptide. Every claim in the program is a scoring claim: it ranks or locates native sequences relative to the grammar. The same measure can also be sampled from, since masked completion is certified exactly by the production gates, though generation for a design task is not claimed here.

## Why the program is built this way

The program rests on a separation that the autoencoder package is built to enforce. The coordinate system represents nucleotides and codons as exact algebraic states. Adjacent codons form ordered transitions, and the kernel assigns each transition a local cost called shell. A coding sequence is a path through these states, and the kernel computes exact properties of that path. The autoencoders learn to reconstruct masked positions in grammar-generated paths and provide learned summaries of sequence order. The shell cost is a path quantity, the same class of object the field's kinetic design literature uses when it separates folding paths from endpoint energies.

The coordinate system is not fitted to biology. It was derived from the CGM axioms, and the independent [genomics analysis](https://github.com/gyrogovernance/science/blob/main/docs/Findings/Analysis_hQVM_CGM_Genomics.md) records that derivation and its 301 verification checks. The topology contrast the program reports is therefore a consequence of the coordinate system's cost function rather than a fit to it.

Agreement between these models and biology cannot serve as evidence, because the training data contain no biological sequence: any agreement would trace to the grammar rather than to learning. The program therefore measures the reverse relation. It asks how far native biological sequence departs from what the grammar alone would generate, and whether that departure is reproducible across the catalogs. Departures of this size are expected to be small, which is why every analysis is designed around a small but coherent effect rather than a large, obvious one.

The RNA design literature knows three ways to weigh a candidate sequence: sample uniformly, score a thermodynamic energy, or apply a prior learned from biological families. A measure induced by a formal grammar, and fitted to no biological data, is not one of them. It holds a separate seat, and the reading of every result follows from that seat. Where the field's measures treat agreement with biology as the objective, a grammar-derived measure treats the departure of native sequence from the grammar's expectation as the signal.

The program works at three levels because one level is not enough to tell the story. The synthesis suite asks whether the models behave as expected when applied to real genomic catalogs at all. The topology study asks whether a specific kernel measure, adjacent codon-pair shell, is associated with a specific biological annotation, membrane topology, within individual genes. The fixed-peptide study asks whether a frozen model readout ranks synonymous expression above chance when the peptide is held fixed. Each level has its own controls and its own estimand, and the program keeps them separate rather than merging them into one narrative.

## The models the program uses

The program evaluates frozen model components from the autoencoder package. Narrow and Super are the two learned components it uses.

Narrow provides a compact representation of each topology window. Super performs masked prediction and supplies an aggregate window readout called climate. Randomly initialized models with the same architectures provide a comparison for the contribution of trained weights.

The hQVM kernel compiles a DNA window into a sequence of carrier states. It records exact path properties, including parity, complementarity, word signatures, and the shell cost of adjacent codon pairs. These quantities are computed from the coordinate system rather than learned from biological data. The kernel statistic used most directly in the topology study is the mean shell of adjacent codon pairs.

The program also uses randomly initialized models with the same architectures. These are not afterthoughts. They are the controlled comparison that tells you whether a trained weight is doing something beyond what the architecture alone would do.

## The synthesis analysis

The synthesis analysis evaluates the frozen models on genomic catalogs. It asks four questions, and it is worth stating them because they are not the same question asked four ways.

Does the K4 layer preserve the exact symmetry required by the coordinate system? Do real coding sequences occupy the carrier in a coherent, measurable way? Do the frozen models distinguish biological sequence order from shuffled or synonymous alternatives? Do the model readouts remain informative when sequence composition is held fixed?

The suite evaluates K4, Narrow, spectral, and Super components on four catalogs: E. coli, yeast, SARS-CoV-2, and human chromosome 22 splice flanks. Comparison datasets preserve selected properties such as GC content, amino-acid sequence, or byte composition so that the tested signal is explicit. The symmetry evaluation checks that the K4 layer preserves the required group symmetry on real sequence. The distributional comparison measures how far the biological catalogs sit from the uniform carrier atlas. The order-memory test constructs sequence pairs with the same algebraic signature and byte multiset while changing only path order, and measures whether the trained Super model retains information about order beyond the fixed carrier quantities. The composition-controlled tests compare Super readouts against GC-matched shuffles and against protein-fixed synonymous resamples.

Each evaluation is recorded with the comparison used for it. The readable account is in RESULTS.txt. The machine-readable form is in gates.json. The reason to record the comparison alongside the result is that a number without the comparison it was measured against is not interpretable.

## The topology analysis

The topology analysis studies the relationship between codon order and membrane topology in E. coli. It compares coding windows annotated as transmembrane with cytoplasmic windows from the same gene. This within-gene comparison is the design choice that makes the study tractable. It keeps the organism, gene, encoded protein, and host codon-usage background fixed while the annotated topological domain changes. The main kernel measurement is mean shell for adjacent codon pairs.

The study is built as a pipeline with six stages, and each stage has a job.

The atlas stage assembles the within-gene topology cohort and computes the primary measurements. The null stage redraws synonymous codons while preserving the encoded protein, including a version that follows observed host codon usage. The narrow stage scores the frozen Narrow model on the atlas windows. The claim stage verifies the primary kernel result and its source and cohort records. The holdout stage tests the incremental predictive value of frozen model readouts with genes kept together across training and test folds. The order stage tests codon-order ranking against Gamble expression measurements.

The null analysis is where the study does its real work, because this is where it answers specific questions instead of producing a single generic control. An amino-acid-preserving null redraws codons while keeping the encoded protein fixed. A usage-matched null samples synonymous codons using the host's observed codon frequencies. Boundary composition is examined two ways. Linear adjustment for codon-boundary dinucleotide content tests a residual after a broad composition correction. Exact preservation of every boundary dinucleotide tests a much stricter constraint with less statistical power. Length matching pairs each transmembrane window with the closest cytoplasmic window from the same gene and keeps length differences within 10%.

These controls answer different questions, and the program reports each one separately. The linear adjustment and the exact dinucleotide match are not two attempts at the same thing. The linear adjustment asks whether a broad composition correction leaves a residual. The exact match asks whether the signal survives a much stricter constraint, with the understood cost that fewer synonymous codons are free to vary. Collapsing them into one conclusion would lose the distinction that makes them both useful.

The claim stage gates on the amino-acid-preserving residual. The usage-matched residual carries a lower effect size and is reported for the host-usage question rather than as the gated statistic.

The held-out analysis keeps genes together in cross-validation folds so that windows from one gene do not appear on both sides of the train-test split. The baseline contains kernel shell, GC content, window length, and the model read-truncation flag. The question is whether adding frozen model readouts improves classification of topology beyond that baseline. Narrow supplies the main unique contribution over the controlled baseline. Adding Super to Narrow gives a further increment. Super alone gives a much smaller one. This pattern is the result that matters, because it tells you which model is contributing what.

The program also records the frozen Narrow and Super readouts on the atlas windows directly, not only in the held-out classification. Narrow heterogeneity has a mean transmembrane-minus-cytoplasmic difference. Super climate has a mean difference in the opposite direction. These are descriptive measurements on the cohort, separate from the classification question.

## The fixed-peptide analysis

The fixed-peptide analysis uses Gamble 2016 Table S1, measured in yeast, which contains 35,811 nine-base variants across 5,148 encoded tripeptides. Within each tripeptide, the encoded protein and sequence length are fixed while codon order varies, and expression is measured for every variant in the synGFPSEQ expression column. This design isolates synonymous order while holding peptide composition fixed.

The eligible analysis set contains 28,504 variants in 2,597 peptide groups with at least five synonymous spellings. Frozen Super climate, Super context, Narrow heterogeneity, and kernel mean shell are scored before expression enters the analysis. GC content and codon-count entropy are reported as additional composition controls.

The primary readout is Super climate. The analysis measures the within-peptide Spearman correlation with the reported expression measure, and it also runs a peptide-held-out evaluation across folds. The question is whether the same frozen readout ranks with expression under fixed peptide identity, not whether it fully predicts expression.

The held-out evaluation guards against a within-peptide correlation inflated by group-level effects. Holding out whole peptide groups and checking that the same mean rank survives is the check that the signal is not an artifact of the peptide grouping.

## What the program does not claim

The program reports associations and ranked signals, not mechanisms. The topology result does not by itself establish thermodynamic stability, folding kinetics, insertion mechanism, or a causal effect. The fixed-peptide result does not show that the model readout fully predicts expression or that the observed rank difference is the only cause of the expression variation. The synthesis results show that grammar-trained models retain exact symmetries and measurable order information on real genomic catalogs, but they do not identify the biological mechanism responsible for that information.

This is not hedging for the sake of hedging. It is the actual scope of the analyses. Each one has its own estimand, and none of them is a mechanism study. You can read the topology result as evidence that adjacent codon-pair order is associated with annotated membrane topology within genes. You cannot read it as evidence that the kernel measure causes the topology or that the topology causes the kernel measure. Those are different claims, and the program does not make the second one.

## Inputs and outputs

The program uses the frozen model files and evaluation reports from the autoencoder package, plus public biological catalogs. The repository includes the frozen model files and evaluation reports needed to run the analyses. Large source catalogs, derived datasets, and temporary files are generated locally.

The synthesis program uses the catalogs created by the genomics ingest command and the frozen checkpoints from the autoencoder production directory. The topology program uses the E. coli CDS catalog, the pinned UniProt K-12 proteome TSV, the Gamble Table S1 workbook, and the frozen production checkpoints. The Gamble workbook is distributed with the journal supplement and is saved with a specific filename so the ingest check can verify its checksum.

Each analysis writes a readable RESULTS.txt and a set of machine-readable JSON result files. The topology pipeline reuses a stage result when its JSON file is present and its recorded script hash matches the current script. This is what makes partial reruns safe. If you change one stage, the pipeline does not recompute the stages that are still current.

## Reproducibility

The program is meant to be rerunnable from the public catalogs. The ingest commands obtain and verify the inputs. The synthesis and topology commands run the analyses. The topology command can rebuild the full pipeline or selected stages, which matters when a script changes or when only one stage needs to be repeated.

The independent analysis that defines the coordinate system is maintained separately. The genomics program depends on that coordinate system but does not redefine it. If the coordinate system changes, that is a different document's problem, but the genomics program will feel it.
