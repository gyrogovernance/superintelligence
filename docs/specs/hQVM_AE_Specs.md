# hQVM Autoencoder Specification

## What this document is

This is the specification for the hQVM group-equivariant autoencoder package. It covers what the package is, the theory it rests on, how its models are organized, where the data comes from, and what it means to add something new to it. If you want a list of every test and the numbers it recorded, that is a different document. This one is about the product and the reasoning behind it.

The package is an autoencoder suite over a finite group-structured state space. That sentence is the definition, so it is worth taking slowly. An autoencoder compresses an input into a compact code and rebuilds the input from that code. What the code keeps is what the model decided matters. The twist here is that the thing being modeled already has symmetries, and the models are built so those symmetries hold exactly rather than being nudged toward by a loss term. The underlying machine is the hQVM kernel. It has exactly 4096 reachable states and a reversible byte transducer as its transition rule.

This specification assumes the formalism in the hQVM Specifications Formalism document and the theory in the QuBEC document. When this text cites a kernel fact, that fact lives in one of those documents. This specification is about the learning package that sits on top of the kernel, not about the kernel itself.

## Why the package exists

The package sits at the meeting point of two things that are often tangled together: a fixed coordinate system for sequence, and learned representations of sequence order.

The coordinate system represents nucleotides and codons as exact algebraic states. Adjacent codons form ordered transitions, and the kernel assigns each transition a local cost called shell. A coding sequence is a path through these states, and the kernel computes exact properties of that path. The autoencoders in this package learn to reconstruct masked positions in grammar-generated paths, and in doing so they produce learned summaries of sequence order.

The design matters because it separates what is fixed from what is learned. The kernel determines the coordinate system. No transition rule, gate, mask, or intron is reformulated inside the learning stack, and no dataset is generated outside the kernel adapter. Every dataset, group action, and evaluation goes through the kernel. The models do not invent a new machine. They learn or represent structure that is already there.

That separation decides what the package can honestly claim. The training data contain no biological genomes, no annotations, no expression measurements, and no phenotypes, so the learned representations are grammar-derived references rather than models fitted to biological fitness. That also fixes what evidence would look like. Agreement with biology could not be credited to the models, since nothing in training could produce it except the grammar; the informative signal is the distance between native sequence and the grammar's own expectation, and effects of that kind are small. The package produces a representation with known structure and lets you verify that structure. Testing it on real genomes, protein topology, and synonymous expression is the job of the genomics programs.

The package also separates exact symmetry from learned provenance. Some models carry no learned symmetry at all. They are exact chart codecs or controlled baselines. Others build a specific symmetry into the architecture. The Super process model adds a learned ledger component on top of an exact grammar. The way to tell these apart is structural: a model belongs to the tier that matches the symmetry it builds in, and it ships with a narrow null so you can measure what the symmetry is actually doing.

## The state the models operate on

The reachable state space, written Omega, is the set of states the machine can occupy. It has 4096 elements, each a 24-bit gyroscopic state made of two conjugate 12-bit halves. Every state has the same bit density, and the space factors as Omega = U x V with 64 values per factor.

Each state carries a chirality word chi in GF(2)^6, a six-bit value that records the state's orientation. Reading a byte adds a byte-specific charge q to the chirality by XOR. That is an affine transport law on the chirality register, and it is diagonalized exactly by the 64-point Walsh-Hadamard transform. The chirality register therefore has a natural Fourier basis.

Shells partition the states by the Hamming weight of chi, giving seven shells indexed 0 through 6. The compact description of the machine's occupation over these states is called the QuBEC. It assigns an occupation probability to every state. Weight states by lambda^N for a real parameter lambda and shell N, and you get the exact partition function Z1(lambda) = 64 * (1 + lambda)^6. From that come the three order parameters that describe the climate: rho for occupation density, eta for spectral damping, and M2 for effective support size. Condensed climates sit near the low shells. Thermalized climates spread across all 4096 states.

The self-dual [12,6,2] mask code C64 is the transport space on the active face, with 64 codewords. Depth-four words close as involutory operators, so every sliding four-byte frame compiles to a pure translation whose signature has parity zero.

You do not need to recompute any of this to use the package. It is stated here to keep the model design legible. The models are not generic networks dropped onto an arbitrary state space. They are built around a space with a known factorization, a known Fourier basis on the chirality register, a known shell structure, and a known closure property at depth four. When a model in this package has a symmetry, that symmetry is a symmetry of this specific machine.

## How the models are organized

Models are grouped into three tiers by the symmetry they build in. Each tier is one file. The registry and the symmetry selector live in the package init. The word equivariant does not appear in a file name because it is the defining property of the package, not a feature you toggle.

The three tiers are narrow, general, and super. Narrow means no built-in symmetry. General means exactly K4. Super means the affine ledger grammar. A model joins the tier that matches the symmetry it actually has.

### The narrow tier

The narrow tier holds two different kinds of thing, and it is worth keeping them straight because they serve different roles.

The first kind is deterministic chart codecs. These are not networks. They are exact, lossless codings of part of the kernel state space. They serve as information-theoretic null models and as the building blocks of the codec ladder. The shipped codecs are ExactUVCodec, BoundaryChiralityCodec, ChiralityOnlyCodec, and ShellOnlyCodec. Each one documents its fiber structure exactly, including the ambiguity where a codec intentionally collapses distinctions. For example, ChiralityOnlyCodec maps 64 states per chirality class, which is the expected ambiguity for a codec that keeps only chirality.

The second kind is learned models with no built-in symmetry. MLPAutoencoder is a plain encoder and decoder. The task models in this tier are byte-conditioned predictors supervised by kernel-exact targets. TransitionModel predicts the next state. RawByteTransitionModel predicts the raw input byte. WordActionModel predicts the word signature. FrameHead predicts the frame signature. PercolationLearner reads the packed 256-bit allowed byte mask and predicts the kernel-exact percolation labels.

These narrow models exist to make the symmetry contrast measurable. If you want to know what K4 equivariance or the Super ledger is actually buying you, you compare against a narrow model with the same architecture. The narrow tier is the controlled comparison for the rest of the package.

### The general tier

The general tier holds models that build a specific group symmetry into the architecture.

K4Autoencoder is exactly K4-equivariant. It averages the encoder over the four Klein gates using Reynolds symmetrization. That gives a latent split into four named components: the invariant part and the three character parts for the S, C, and F generators. The symmetry holds before and after training. It is exact over all 4096 states and all four gates, and the test suite measures it with a numeric gate.

AffineSpectralCodec is the Walsh occupation codec. It builds in the full affine group on the spectral chart, which is why its shipped denoiser gains match the closed-form Bayes-optimal multipliers. That match is the analytic certificate for the model. It is the model used by the denoise command and by the dictionary probes.

### The Super tier

Super is the process model for byte ledgers, and it is the most constructed model in the package because it combines an exact grammar with learned parts.

The exact grammar part does three things. It scans a ledger with the kernel. It recovers a masked canonical byte from the visible siblings and the frame position using a frozen analytic grammar. It composes the word signature with exact XOR registers. The grammar is frozen. That is what lets the model achieve exact canonical recovery on the relevant gates.

The learned parts are provenance discrimination and a residual climate head. A GRU over depth-4 frames learns to separate collision ledgers that share a signature but are not the same word. Climate is an analytic QuBEC tilt from the visible-byte q-class histogram, and it is gated off on exact-grammar frames so the exact certificates stay exact. A sequential MLP reads prefix-GRU context. An opt-in previous-micro flip filter supplies the Bayes-optimal Markov conditional without disturbing the uniform and canonical certificates.

Encode and decode use a ledger-free SuperCode that carries exact signature, endpoint, climate, and provenance. Production weights and the gate report are written by the Super training helper.

The measured production object is exact grammar plus provenance plus residual. The shipped winner is context 32 by atom 8, with 106,754 parameters. The lambda-ensemble residual at lambda = 4 takes 90.0% of the closed-form ceiling. Signature-constrained completion puts mass 1.0 on the valid coset with ambiguity 2. The Markov causal filter improves 2.89 bits versus uniform, leaving a gap of 0.30 bits to the relevant bound on that holdout. The Markov filter is opt-in precisely so the G6 and G7 certificates stay on the grammar.

### How model selection works

You select a model by name. The individual kinds are mlp, k4, super, transition, rawbyte, word, and percolation. The tier names narrow, general, super, and all select every model in that tier for sweeping. For training, a tier maps to its first member. The single constructor is build_model in the models init. The hidden-dim flag overrides the per-kind default width.

This is not just a naming convention. The tier is the symmetry contract. If you pick a tier, you are picking the symmetry you want the model to carry, and the package is responsible for making that symmetry real.

## What the package contains

The package is organized around one boundary: the kernel core must never be contaminated by external material. Everything else flows from that.

The command-line entry points are in cli.py. The kernel adapter is in kernel.py and covers state indexing, stepping, gates, signatures, signature id packing, and popcount6. datasets.py holds kernel tables, the census, and the null corpus with its structured holdouts. corpus.py is the dictionary export with exact kernel labels. paths.py documents the data layout. The models live in the models directory by tier. The helpers directory follows a <domain>_<role> naming convention.

The data directory is the only place that holds actual data files. It contains numpy arrays and manifest.json files, and it is regenerable through the generate command. Checkpoints, reports, and temporary files are gitignored. The structure is:

- data/dataset_null: the CGM null permutation atlas
- data/dataset_bytes: byte census
- data/dataset_states: state census
- data/dataset_transitions: dense transition table
- data/dataset_signatures: 8192-row group signature table
- data/dataset_actions: K4 action table
- data/dataset_embeddings: verified-dictionary corpus
- data/dataset_ensembles: lambda-ensemble artifact
- data/checkpoints: trained model weights
- data/reports: eval, verify, and audit JSON reports
- data/tmp: scratch space

The listing is not there to make you memorize paths. It shows where the package keeps things that are real data versus things that are derived artifacts versus things that are temporary. That distinction matters when you are trying to understand what is regenerable and what is pinned.

## The CGM null dataset

The CGM null dataset is the exhaustive, kernel-exact catalog of hQVM instruction and trajectory structure. It is the training and gate corpus for Super. It is generated by the null dataset builder and loaded through the null corpus class. Every field is produced by calling the kernel. Nothing is sampled from outside data.

Null means the uniform, maximum-entropy occupation of that exact structure under three policies recorded in measures.json. The three policies are uniform independent bytes, the canonical family frame sequence, and QuBEC lambda-weighted micro occupation.

Construction begins at the archetype and the rest state. The transcription archetype is 0xAA. For any byte b, the intron is b XOR 0xAA. The invariant check requires that byte 0xAA is flat and has intron 0. Every trajectory in the catalog starts from the rest state, and depth-4 and depth-8 cycles must return through the swapped mac state and close again at rest.

The catalog then enumerates the structure in stages. The byte fiber records all 256 instruction bytes with their intron, family phase, micro-ref, chirality weight, fold disagreement, flat flag, shadow partner, and single-byte signature. The canonical cycles build the eight-byte word that walks the four family phases twice for each of the 64 micro-refs. The depth-2 witnesses apply every ordered pair of bytes from rest and record the intermediate state, endpoint, signature id, and transport. The signature words take one minimal representative word of length at most 4 for each group signature and replay it from rest. The collision set collects pairs of distinct ledgers that share a signature, which separates terminal action from ledger provenance. measures.json records the three occupation policies. A manifest and a boolean invariant suite close the generation, and generation fails if any invariant fails.

The null corpus class wraps these arrays and adds deterministic train and holdout splits: a micro-ref holdout, a signature-factor holdout, and a collision connected-component ledger split.

This dataset is the reason the package can talk about training at all. Super is trained on grammar-generated structure. The null dataset is that structure, laid out exhaustively and exactly. When you read a gate number for Super, you are reading a measurement on this dataset.

## Where empirical data enters

Base training is kernel-null. The models self-supervise on exact labels, which is why the production artifacts exist. Empirical tensors from outside the kernel are not mixed into the core models. They enter only as a frozen-codec head fine-tune. You keep the codec's exact equivariance intact and train only the task heads on compiled windows. For LLM weights specifically, you tile the matrix to 64-wide blocks, compile the byte stream through the census, keep the codec frozen, and read the block features. No new model tier is required for that path.

The boundary that keeps this clean is the external adapter boundary. Material outside the kernel core, such as physical observables or weight tensors from other systems, enters only as an external adapter. It does not become a new model and it does not become a new model tier. Adapters are pure data transforms that read the census and byte surfaces already exposed by the package and produce the columns the readouts consume. They never add a kernel fact to a model file and they never reformulate a transition rule. They only convert external material into the structure the codec already understands.

This boundary is what keeps the three tiers clean. The models learn or represent hQVM structure. Adapters compile external material into that structure. Task heads make application-specific predictions. Readouts measure the resulting structure.

## Adding a model or specialization

A new model joins the tier whose symmetry it builds in. A new group would justify a new tier file, and nothing else. Before it ships it must satisfy six things.

First, symmetry containment decides the tier. Narrow is for no built-in symmetry. General is for K4 gates. Super is for the ledger process. Second, kernel authority: every label and action comes from the kernel adapter, with zero reformulation inside the package. Third, a paired null: every structured model ships with its narrow null so the symmetry-breaking contrast is measurable. Fourth, an exactness contract: an equivariance or identity test with a numeric gate, plus an entry in the verify or audit path. Fifth, registry wiring: the model kind registry, the tier members, the hidden defaults, the constructor, the checkpoint loader, the evaluate-task routing, and the test that the task actually trains. Sixth, a benchmark: a suite entry with kernel-exact labels.

PercolationLearner is the template. It is a supervised head on kernel-exact labels, sitting in narrow, needing no external data. Physics probes and similar readouts are adapters, not models.

The package has a hard boundary between what is kernel structure and what is learned structure, and a new model has to respect that boundary in a visible way. Each of the six requirements is a place where that boundary could be violated, and each one is checked.

## How verification works

The suite asserts exact kernel relations rather than learned proxies.

K4 equivariance holds to an error below 1e-4 over all 4096 states and all four gates. The encoder satisfies E(gx) = rho(g) E(x) and the decoder satisfies D(rho(g) z) = P_g D(z).

Full-group equivariance holds with zero observed error. The spectrum of gx equals rho(g) applied to the spectrum of x. The closed-form certificate across all 4096 states and all 8192 signatures is a sub-second condition on the gain symmetry, and it is persisted to a report file.

Word composition holds exactly. The network's composition of two word signatures equals the kernel's.

Depth-four frames compiling to pure translations is a kernel theorem cited from the Features Report, and it is audited in the dictionary's frame-parity check. The two-byte witness routing count is a kernel theorem cited from the Features Report, and it is not re-derived inside the suite.

The dictionary audit recomposes reconstruction, equivariance, closed-form factorization probes, the H-invariance of the diagonal rung, shadow invariance, frame parity, and the psi_hat character-energy identity.

The test suite lives in the tests directory. It covers the package's models, losses, metrics, datasets, reports, and the mapping to the kernel. Kernel feature facts are cited from the Features Report.

What this yields is a package whose claims can be checked. Exact equivariance is not a regularization target. It is a property that either holds or it does not, and the test suite measures it where it can and certifies it in closed form where it can. If you are using this package, you should know which of its claims are asserted as exact and which are measurements on a particular corpus.

## Provenance

The four-hook callback protocol and the symmetry-regularized baseline concept are adapted from the MIT-licensed ssb_detection_ising repository from the Del Maestro Group in 2019. No source code was copied from that repository, and none of its domain machinery appears here. Its license is retained in the third-party directory.
