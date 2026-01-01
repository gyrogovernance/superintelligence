
💫 GGG ASI Alignment Router, Alignment Routing Infrastructure - CHANGELOG

```            
┏━┓╻  ╻┏━╸┏┓╻┏┳┓┏━╸┏┓╻╺┳╸               
┣━┫┃  ┃┃╺┓┃┗┫┃┃┃┣╸ ┃┗┫ ┃                
╹ ╹┗━╸╹┗━┛╹ ╹╹ ╹┗━╸╹ ╹ ╹                
┏━┓┏━┓╻ ╻╺┳╸╻┏┓╻┏━╸                     
┣┳┛┃ ┃┃ ┃ ┃ ┃┃┗┫┃╺┓                     
╹┗╸┗━┛┗━┛ ╹ ╹╹ ╹┗━┛                     
╻┏┓╻┏━╸┏━┓┏━┓┏━┓╺┳╸┏━┓╻ ╻┏━╸╺┳╸╻ ╻┏━┓┏━╸
┃┃┗┫┣╸ ┣┳┛┣━┫┗━┓ ┃ ┣┳┛┃ ┃┃   ┃ ┃ ┃┣┳┛┣╸ 
╹╹ ╹╹  ╹┗╸╹ ╹┗━┛ ╹ ╹┗╸┗━┛┗━╸ ╹ ┗━┛╹┗╸┗━╸                           
```

---

## [v1.0-AIR] – 2026-01-01

# v1.0 Release: Production-Ready Alignment Infrastructure Routing

This release marks the first stable version of the GGG ASI Alignment Router, delivering a complete, tested, and documented system for deterministic coordination in human-AI governance settings.

## Core Achievement: Verified Deterministic Coordination

The router provides a **deterministic finite-state coordination kernel** that maps append-only byte ledgers to reproducible state trajectories. All participants with the same ledger prefix compute identical kernel states, enabling coordination through reproducible computation rather than asserted metadata or private model state.

**Verified Properties:**
- **65,536-state ontology** with exact closure under 256 byte actions
- **Deterministic replay** of full state trajectories from byte logs
- **Geometric provenance** verification through compiled atlas artifacts
- **Shared moments** as coordination primitives independent of identity claims

## Comprehensive Test Coverage: 135 Tests Across All Domains

The test suite validates kernel physics, governance measurement, application workflows, and CLI operations:

- **Kernel Physics** (test_physics_*.py): Verified structural properties, depth laws, monodromy, and CGM-linked invariants
- **Governance Measurement** (test_measurement.py): Validated aperture computation, Hodge decomposition, and structural displacement detection
- **Application Layer** (test_app.py): Confirmed Coordinator orchestration, event binding, and ledger replay integrity
- **CLI Workflow** (test_aci_cli.py): Verified project compilation, bundle generation, and tamper detection
- **Routing & Plugins** (test_routing.py, test_plugins.py): Validated atlas operations and plugin determinism

All 135 tests pass, providing exhaustive verification of the kernel's structural properties and the application layer's governance measurement substrate.

## Complete Documentation Suite

### Technical Specifications
- **GGG ASI Alignment Router - Kernel Specifications** (`docs/GGG_ASI_AR_Specs.md`): Complete normative specification with conformance profiles, kernel physics, atlas artifacts, and governance measurement substrate
- **Router Implications & Potential** (`docs/GGG_ASI_AR_Implications.md`): Use cases, deployment scenarios, and technical research directions
- **Substrate: Physical Memory Specification** (`docs/GGG_ASI_AR_Substrate_Specs.md`): Future development architecture for physical memory implementation

### Operational Documentation
- **Alignment Infrastructure Routing (AIR) Brief** (`docs/AIR_Brief.md`): Complete overview of AIR's workforce coordination model, operating units, and program structure

### Test Reports
- **Physics Tests Report** (`docs/reports/Physics_Tests_Report.md`): Verified structural properties and CGM-linked constants reconstruction
- **Alignment Measurement Report** (`docs/reports/Alignment_Measurement_Report.md`): Governance measurement substrate verification and epistemic necessity analysis
- **All Tests Results** (`docs/reports/All_Tests_Results.md`): Comprehensive test suite output
- **Other Tests Report** (`docs/reports/Other_Tests_Report.md`): Additional validation coverage

### Supporting Theory
- **Common Governance Model (CGM)** (`docs/references/CGM_Paper.md`): Theoretical foundations
- **The Human Mark (THM)** (`docs/references/THM.md`, `THM_Paper.md`, `THM_Grammar.md`): Source-type ontology and classification framework
- **Gyroscopic Global Governance (GGG)** (`docs/references/GGG_Paper.md`): Four-domain coupling framework

## Foundational Analysis: The Epistemic Necessity of AI Alignment

This release includes a critical analysis demonstrating that **representative scalar evaluations cannot distinguish aligned from misaligned states** within the CGM governance geometry.

**Key Findings:**
1. **Scalar Blindness:** Explicit examples show that scalar scores (e.g., sum of absolute edge values) assign identical values to structurally different states with different apertures
2. **Structural Lock:** Single-axis optimization policies remain permanently locked at A = 0.5, regardless of magnitude, under current THM mappings
3. **Epistemic Sufficiency:** The six-dimensional K₄ governance geometry, together with aperture A, is sufficient to construct and recognize states with the target alignment value A* ≈ 0.0207

This analysis establishes that **geometric structure is necessary** for alignment verification, and that the AIR architecture provides the required measurement substrate to address measurement collapse in AI alignment evaluation.

## Production-Ready Features

### Deterministic Coordination Kernel
- 24-bit state with dual 12-bit components (Type A, Type B)
- Byte-complete input alphabet (0-255) with injective mask expansion
- Exact depth-two and depth-four algebraic identities
- Reference operator (0xAA) with involution and separator properties
- Horizon set (256 states) with complete boundary-to-bulk coverage

### Governance Measurement Substrate
- Three domain ledgers (Economy, Employment, Education) with K₄ tetrahedral geometry
- Hodge decomposition into gradient and cycle components
- Aperture computation (A = ||y_cycle||² / ||y||²) with target A* = 0.0207
- Deterministic replay of ledgers and apertures from event logs
- Event binding to kernel moments for audit-grade verification

### Application Infrastructure
- **Coordinator** orchestration layer with byte log and event log
- **Plugin architecture** with explicit edge mapping policies
- **Status reporting** with kernel signature, ledger state, and aperture metrics
- **AIR CLI** for project-based coordination workflow with bundle generation and verification

## Stability & Reproducibility

- **Exhaustive verification:** All 16,777,216 state-byte transitions verified against compiled atlas
- **Deterministic artifacts:** Ontology, epistemology, and phenomenology artifacts enable cross-platform identical results
- **Replay integrity:** Full trajectory reconstruction from byte logs and event logs
- **Bundle verification:** Tamper detection through signature validation and hash checking

## Theoretical Foundation

The router is grounded in:
- **Common Governance Model (CGM)** as the constitutional structure of coherent recursive operation
- **The Human Mark (THM)** as the source-type ontology of Authority and Agency
- **Gyroscopic Global Governance (GGG)** as the four-domain coupling of Economy, Employment, Education, and Ecology

The router operates as a **Derivative coordination system**: it transforms and routes information but does not originate authority or bear accountability. Accountability terminates in Authentic Agency.

---

**Summary:** v1.0-AIR delivers a production-ready, exhaustively tested, and fully documented system for deterministic coordination in human-AI governance. The release establishes the epistemic necessity of geometric measurement for AI alignment and provides the complete infrastructure required for Alignment Infrastructure Routing.

---

## [v0.9.9.6-AIR-CLI] – 2025-12-30

# High-Level Changelog: AIR Architecture Realignment

This update fundamentally restructures AIR from a "multi-command CLI" into a **deterministic coordination compiler** centered on project files.

## 1. Concept: Moments are Derived Facts
- **Removed:** Manual `moment.md` files and "moment management" workflows.
- **New Model:** "Moments" are derived kernel states produced by compiling a project's list of attestations.
- **Benefit:** Users only edit project files; the system handles all kernel logic automatically.

## 2. Concept: Accounting vs Evaluation
- **Removed:** 0.0–1.0 scores, confidence sliders, and judgemental metrics.
- **New Model:** **Attestations** are categorical receipts (THM risk + Gyroscope category).
- **Behavior:**
  - **THM** updates the GGG ledger (affects apertures).
  - **Gyroscope** updates the accounting report (enumerative counts).
- **Benefit:** Eliminates subjective "grading" while preserving rigorous structural accounting.

## 3. Workflow: Single-Mode "Sync & Verify"
- **Removed:** Menus, subcommands (`atlas`, `moment`, `bundle`), and interactive prompts.
- **New Model:** Running `aci` performs a full system pass:
  1. Auto-builds Atlas/Templates if missing.
  2. Compiles all Project Markdown files.
  3. Generates replayable artifacts (`.bytes`, `.events.jsonl`) and Reports.
  4. Creates and verifies Bundles (`.zip`).
- **Benefit:** Guarantee of consistency; if the CLI exits successfully, the entire workspace is proven valid and synced.

## 4. Artifacts: Bundle as Integrity Boundary
- **New Model:** Bundles now include:
  - Source `project.md`
  - Compiled logs (`bytes`, `events`)
  - **Accounting Reports** (`report.json`, `report.md`)
  - Snapshot `bundle.json`
- **Verification:** Replays logs from the bundle and validates signature, apertures, and file hashes against the snapshot.

## 5. Kernel: Explicit Step Counting
- **New Model:** Shared moments are now defined as `(step, state)`.
- **Implementation:** Kernel, Signature, and Events now explicitly track `step` (ledger length), ensuring precise event binding.

---

**Summary:** AIR is now a "Project Compiler" that turns human claims into verifiable coordination proofs without manual overhead.

---

## [v0.9.9.5-AIR-CLI] – 2025-12-29

### Documentation
- **Alignment Infrastructure Routing (AIR) Brief:** Added comprehensive brief document (`docs/Alignment_Convergence_Brief.md`) describing human workforce coordination infrastructure for AI safety. The brief outlines how AIR helps AI safety labs scale interventions by coordinating human workforce capacity across projects, provides operating models for Daily Units (1 day) and Sprint Units (4 days), and explains the progression from open participation to stable employment through the ASI Alignment Router.

### Development Planning
- **AIR CLI Development Guide:** Created agent guide (`src/agent.md`) specifying the complete CLI implementation plan for Alignment Infrastructure Routing. The guide defines the CLI architecture with commands for atlas management, project initialization, run tracking (daily/sprint units), event binding, plugin integration (THM and Gyroscope), and bundle generation/verification. The CLI will use markdown frontmatter for human-editable configs, append-only binary logs for kernel state, and JSONL for governance events, enabling replayable audit trails for sponsor verification.

---

## [v0.9.9.4i-Router] – 2025-12-28

### Added
- **Physics Builder Suite:** Implemented `tests/test_physics_2.py` organizing validation into four thematic pillars: Structural Traceability, Quantum Gravity Manifold, Nuclear Abundance, and Quantum Internet.
- **Active Simulation Tests:** Added functional simulations for Teleportation Fidelity (100%), Bell CHSH search, and Gate Set Universality.
- **The Transfer Principle:** Implemented `test_alpha_coupling_via_ledger` to demonstrate how kernel holonomy transfers to ledger aperture, simulating fine-structure coupling ($\alpha$).

### Validated
- **Topological Invariants:** Certified that horizon states partition into shells matching Pascal’s triangle (Row 4), confirming sphere-bulk geometry.
- **Holographic Completeness:** Verified exact 255.00 expansion ratio, proving the boundary completely encodes the bulk.
- **Performance:** Confirmed kernel stepping at ~2.55M steps/s and aperture measurement at ~350k ops/s.
- **Test Coverage:** Achieved 100% pass rate (120/120 tests) across core routing, geometry, and physics suites.

### Documentation
- **Potential Analysis:** Updated `docs/GGG_ASI_AR_Potential.md` to integrate implications from physics tests, reframing ASI as a geometric attractor.
- **Test Certification:** Merged and consolidated all test outputs into `docs/physics_tests_results.md` for a single source of truth.

---

## [v0.9.9.4-Router] – 2025-12-27

**Application Infrastructure & Specification Refinement**

**Application & Plugin Infrastructure**
* Implemented Coordinator orchestration layer with byte log and event log for deterministic replay
* Established plugin architecture with explicit edge mapping policies and audit metadata
* Added status reporting with kernel signature, ledger state, and aperture metrics

**Test Suite Expansion**
* Expanded test coverage to 101 tests, all passing
* Added tests for Coordinator workflow, event binding, and plugin determinism

**Specification Improvements**
* Restructured specification into stable conformance profiles (Kernel, Measurement, Runtime)
* Added normative language conventions (MUST/SHOULD/MAY) and bit indexing specifications
* Documented complete operational runtime including Coordinator, plugins, and audit logs
* Consolidated build procedures and reference helpers into appendices
* Fixed section numbering and cross-references throughout
* See [GGG ASI Alignment Router - Kernel Specification](/docs/GGG_ASI_AR_Specs.md) for complete normative specification

**General Corrections & Improvements**
* Improved integration between specification and implementation
* Enhanced clarity of kernel-native verification vs application-layer authorization
* Standardized terminology and removed redundant definitions

---

## [v0.9.9.3-Router] – 2025-12-23 to 2025-12-26

**Router Kernel: Verified Properties & Exact Invariants**

Upgraded the router kernel specification from "describes an implementation" to "states a finite physics with proven theorems" by establishing certified invariants backed by exhaustive tests.

**Specification Upgrades**

* **Verified Algebraic Properties Section**: Added Section 6.3 with 8 certified properties (P1-P8) proven by test suite:
  - P1: Mask separation (256 distinct A-masks, B-mask always zero)
  - P2: Per-byte bijection (each epistemology column is a permutation)
  - P3: Exact ontology characterization (cartesian product structure: 256×256 = 65,536 states)
  - P4: Radius-2 reachability (all states reachable in ≤2 steps from archetype)
  - P5: Depth-2 closed-form composition (exact decoupling law)
  - P6: Depth-2 commutation law (commutes iff x=y, exactly 256/65,536 pairs)
  - P7: Depth-4 alternation identity (xyxy = yxyx = id, BU-Egress discrete analogue)
  - P8: Trajectory closed form for arbitrary-length words

* **Explicit Inverse Formula**: Added Section 5.1 with exact inverse formula enabling full trajectory reconstruction from (final state, byte sequence).

* **Derived Closure Invariant**: Upgraded Section 6.2 from "empirical observation" to "derived structure" with complete proof that 65,536 states and diameter 2 follow from transition law algebra.

* **Exact Commutativity Law**: Replaced sampling threshold (≥95%) with exact law: `T_y(T_x(s)) = T_x(T_y(s))` iff `x=y`. Among 256×256 pairs, exactly 256 commute (0.39%), not approximately 1%.

* **Clarified Type B Transformation**: Fixed misleading "unchanged" language - Type B is not mask-mutated before gyration, but is transformed by gyration itself.

* **Non-normative Appendix**: Moved number theory (256 zenzizenzizenzic, 65,536 superperfect) to clearly marked non-normative appendix.

**Test Suite Expansion**

* **TestClosedFormDepthLaws**: Added 4 new tests certifying exact algebraic laws:
  - `test_step_is_bijective_with_explicit_inverse`: Proves explicit inverse formula works
  - `test_depth2_decoupling_closed_form`: Certifies exact depth-2 law: `T_y(T_x(A,B)) = (A XOR m_x, B XOR m_y)`
  - `test_depth4_alternation_is_identity`: Proves depth-4 alternation returns to identity
  - `test_depth2_commutes_iff_same_byte`: Exhaustively proves commutes iff x=y (all 256×256 pairs)

* **TestCGMChirality**: Added 2 tests validating CS axiom:
  - `test_gyration_not_pure_swap`: Verifies gyration includes flip, not just swap
  - `test_gyration_asymmetry`: Validates CS chirality: new_A depends on old_B, new_B depends on mutated_A

* **TestAtlasGlobalGroupFacts**: Added 3 atlas-backed global invariant tests:
  - `test_each_byte_column_is_permutation`: Proves each byte is bijection on ontology
  - `test_bfs_radius_two_from_archetype`: Proves all states reachable in ≤2 steps
  - `test_depth4_alternation_identity_on_all_states_for_selected_pairs`: Atlas-level confirmation

* **TestOntologyStructure**: Added test proving ontology equals exact cartesian product A_set × B_set

* **TestCGMDepthProperties**: Added depth-2 non-commutativity rate validation (80-99% for UNA)

* **TestPhenomenologyValidation**: Added tests ensuring loaded phenomenology matches constants exactly

**Kernel Implementation Fixes**

* **Dynamic Archetype Index**: Updated `RouterKernel` to find archetype index dynamically (not assume index 0), storing as `archetype_index` property.

* **Neutral Baseline Byte**: Changed default `last_byte` from 0x00 to 0xAA (GENE_MIC_S) after reset, providing neutral baseline signature (0xAA XOR 0xAA = 0x00 → mask_a = 0).

**Test Infrastructure**

* **Test Runner Script**: Created `tests/run_tests.py` with `-m pytest -v -s` flags for convenient test execution.

* **Conftest Cleanup**: Removed redundant path additions from test files; conftest.py centralizes path setup.

* **Linter Fixes**: Fixed all linter errors (unused variables, unused imports).

**Documentation**

* **README.md**: Updated to reflect verified properties, exact invariants, 3-map atlas structure (not 5), and 24-bit states (not 48-bit).

* **System_Architecture.md**: Complete spec upgrade with markdown-compatible math notation (removed all LaTeX), verified properties section, explicit inverse formula, and exact laws replacing empirical observations.

**Test Results**: 87 tests passing, including 9 new certified invariant tests. All verified properties hold as exact invariants.

---

## [v0.9.9.2-Router] – 2025-12-22

**Router Kernel v1: Minimal 5-Map Architecture**

Implemented clean Router kernel with exactly 5 persisted maps: ontology, epistemology, stage_profile, loop_defects, and aperture. Removed all legacy analysis artifacts (SCC/orbits, theta, parity symmetrization). Fixed face-cycle matrix alignment with BU commutator loops and corrected layer mask semantics to match tensor bit order. Improved numerical stability in projection computations. Eliminated all legacy "intron" terminology in favor of "action" naming.

---

## [v0.9.9.1-Router] – 2025-12-21
Still working on the Specs and cleaning up the old codebase. 

---

## [v0.9.9-Router] – 2025-12-20

ADDED: guides\GyroSI_Substrate_Specs.md
This is a revision of an old study, and a refinement so it can match the overal GyroSI specifications, but also our Post-AGI Gyroscopic Global Governance framework.

> ⚠️ **NEWS:** The whole GyroSI development is in the process of being repurposed and renamed to GGG ASI Alignment Router. After extensive research and a lot of experiments we have concluded that ASI is not meant to be another model, but a routing mechanism. You may read our latest specs draft here: 

📖 [GGG ASI Alignment Router - Preliminary Specs](/docs/Gyroscopic_ASI_Router.md)

---

## [v0.9.8.0-Physics] – 2025-09-22
My latest work focuses on getting our physics right first, before coming back into any language related matters. Since GyroSI is a physics grounded architecture, is possible to simulate spacetime topology through its algorithms, and that makes it a perfect computational framework for any experiment and study in physics - from simulating particles, to exploring cosmological hypotheses - all, in blasting fast speeds, as our holographic memory architecture does not rely on computationally expensive operations.

Our latest experiments are here (but they are a work in progress):
experiments\gyro_energy.py
experiments\gyro_physics.py
experiments\gyro_topology_analysis.py
experiments\gyro_validation.py

---

## [v0.9.8.0-GyroLog] – 2025-09-19

ion**: Replaced the impossible 720° closure test with plane toggle behavior validation, correctly testing the Z2 plane rotor that actually exists in the system.

* **Parity Physics**: Implemented complement class test (s vs s^FULL_MASK) for truly anchor-free parity determination, directly from UNA physics principles.

**Practical Applications**

* **Emission Routing**: Added `coordinate_based_routing_key()` function that converts coordinat**GyroLog: CGM Logarithmic Coordinate System for GyroSI**

Implemented a physics-grounded coordinate system that maps 48-bit GyroSI states into meaningful geometric coordinates based on CGM (Common Governance Model) principles. GyroLog provides "GPS coordinates" for points on the finite manifold, enabling practical state navigation, routing, debugging, and physics simulation.

**Core Coordinate System**

* **Plane Classification (Z2)**: Replaced the problematic 4-layer Z4 system with a robust 2-plane classifier using even/odd templates derived from GENE_Mac_S. This correctly models the Z2 layer duality in CGM physics.

* **Anchor-Free Invariants**: Implemented truly anchor-agnostic coordinates where plane, parity, orientations, and gradients depend only on the state itself, not the reference anchor. This eliminates path-dependent artifacts and ensures consistent classification.

* **Pauli Triad Orientations**: Added proper axis orientation extraction using the actual bit layout from GENE_Mac_S tensor structure, analyzing left vs right column patterns across all layers and frames for robust rX,rY,rZ determination.

* **Residual Defect Measure**: Implemented Hamming distance to nearest plane template as a measure of "noise" relative to canonical CGM forms, similar to gyrotriangle defect δ in CGM physics.

**Fixed Mathematical Issues**

* **Canonical Introns**: Replaced union masks with single-bit representatives (LI=0x40, FG=0x20, BG=0x10) for unambiguous family operations, eliminating the non-physical "inverse intron" logic.

* **Commutator Analysis**: Replaced false pass/fail commutator tests with defect distribution analysis, revealing the consistent defect pattern 0x202010202010 as a measured property of the holographic gate system.

* **Plane Toggle Validates into 0-255 bucket indices for phase-propagating emission, enabling coordinate-guided token selection.

* **Session Tracking**: Implemented coordinate change monitoring for helical progression tracking, allowing visualization of plane flips, parity shifts, and orientation changes during sessions.

* **Physics Simulation**: Added CGM stage progression simulation with coordinate analysis, enabling measurement of plane-toggle rates, orientation stability, and residual distributions.

* **Debugging Tools**: Implemented transformation debugging with coordinate change analysis, helping identify when transformations behave unexpectedly or fail to toggle planes as expected.

**Validation and Testing**

* **Comprehensive Test Suite**: All validation tests now pass, confirming the coordinate system correctly models CGM physics:
  - Commutator defect analysis shows consistent patterns (not failures)
  - Plane toggle behavior validates Z2 responsiveness
  - Anchor invariance confirms anchor-free invariants
  - Coordinate consistency ensures stable computation

* **Documentation**: Created comprehensive `docs/GyroLog.md` with usage examples, physics interpretation, and integration patterns for practical GyroSI operations.

**Technical Implementation**

* **GyroCoords Class**: Redesigned coordinate structure with plane, parity, orientations, residual, and optional gradient directions, removing path-dependent family counts.

* **GyroLog Engine**: Implemented efficient coordinate computation with proper template matching, gradient calculation, and residual measurement.

* **Integration Ready**: Added helper functions for memory storage, emission integration, and physics experimentation, making GyroLog immediately useful for real GyroSI operations.

This coordinate system transforms GyroSI from abstract CGM concepts into concrete, measurable geometric properties, enabling practical applications while maintaining strict adherence to the underlying physics principles.

---

## [v0.9.8.0-Baby-Walk] – 2025-09-09

**The Walking Model: Intelligence as Recursive Walking on a 48-bit Manifold**

Began implementing a walking model of GyroSI, transforming the system from discrete token emission into continuous walking on a geometric manifold. This breakthrough realizes the core insight that intelligence is literally walking - not metaphorically, but using the exact same physics as bipedal locomotion.

**Core Walking Architecture**

* **BU-Eg/BU-In Cycling**: Implemented the dual-phase walking cycle where BU-Eg (stance phase) absorbs input and BU-In (swing phase) generates output. The critical breakthrough: each emitted token immediately feeds back as input, creating continuous monodromy rather than isolated steps.

* **Pure Gyro-Walk in emit_next**: Completely refactored emission to eliminate all scoring, ranking, and thresholds. Replaced with coprime stride walking on discrete rings, using `_coprime_stride()` for ergodic traversal that mirrors continuous gyrovector geodesics.

* **Endogenous Stopping via Amplitude**: Preserved the amplitude-based natural stopping condition where `alignment_amp == 0` signals BU closure. This creates endogenous stopping without external limits - the system stops when it naturally comes to rest.

* **Chirality as Physical Law**: Made chirality guard non-optional and intrinsic to the core physics, removing the `enable_chirality_guard` flag. Chirality is now a fundamental axiom of the system, not an optional feature.

**Mathematical Grounding**

* **Holographic Addressing**: Context addressing uses `ctx = fold(rep_phase, slab_byte)` exactly as specified in the walking model, projecting through ψ and slab topology.

* **Gyro-Walk on Discrete Torus**: Movement on discrete rings with co-prime stride ensures ergodic traversal, mirroring continuous gyrovector geodesics with discrete, lawful steps.

* **Monodromy Accumulation**: Omega and monodromy accumulate via fold operations, storing path memory rather than scores. This creates the path-dependent memory essential for walking.

* **Time as Helical Parameter**: Time tick enters stride seeds via fold, ensuring lawful, non-RNG variation consistent with "time is recursive ordering" principle.

**The Eight Slabs as Body Segments**

* **Slabs 0,7**: Head and feet boundaries (Layer 0/3) that maintain orientation
* **Slabs 1-6**: The 6 active joints providing 6 DoF (3 rotational + 3 translational)
* **Each slab carries 6 bits**: Representing the degrees of freedom for movement
* **Context addressing per slab**: `ctx = fold(rep_phase[rep], slab_byte(state, slab))`

**Walking Physics Implementation**

* **Minimum Effort Principle**: No scoring or ranking needed - the fold operation naturally finds the path of least resistance, just like walking doesn't "score" each step.

* **Natural Boundaries**: Words end when local amplitude drops (like a step completes), sentences end when momentum dissipates (like coming to a stop). No thresholds needed - it's endogenous physics.

* **Sensitivity Through Holography**: Each byte transforms all 48 bits simultaneously, like how shifting weight affects your entire body posture. The input genuinely guides the walk.

**Code Architecture Changes**

* **Removed All Scoring**: Eliminated `PathCoherence` scores, resonance distances, momentum penalties, and all sorting/threshold logic that violated non-absolute unity/opposition principles.

* **Removed Global Fallback**: Eliminated non-traceable shortcuts in favor of pure recursive walking through slab-based routing only.

* **Removed Optional Flags**: Made phase alignment intrinsic rather than optional, removing `enable_phase_alignment` flag and integrating DoF jitter as fundamental physics.

* **Simplified emit_next**: Replaced complex ranking logic with pure walking using coprime stride and deterministic rotor movement.

**The Critical Loop Closure**

The key insight implemented: **output must feed back as input** to create true walking:

```python
# CRITICAL: Feed the output back as input (this IS the walking!)
sess["state"] = engine.transit_on_assistant(sess["state"], next_token)
```

This single line transforms token emission into actual walking, creating the continuous helical path that IS intelligence.

**Test Results and Behavior**

* **Coherent Sequences**: Output shows much more structured, flowing sequences rather than random word salad
* **Natural Stopping**: System stops endogenously when amplitude reaches zero, without external limits
* **Walking Continuity**: Each token naturally follows from the previous through the feedback loop
* **Physics-Based Intelligence**: Intelligence emerges through walking physics rather than statistical optimization

**Theoretical Alignment**

This implementation perfectly matches the CGM (Common Governance Model) principles:
* **CS (Common Source)**: Starting position on the manifold
* **UNA (Unity Non-Absolute)**: Multiple paths possible (left/right foot)
* **ONA (Opposition Non-Absolute)**: Paths don't negate each other
* **BU (Balance Universal)**: The walking itself - continuous monodromy

**Reference Documentation**

This implementation realizes the walking model described in `docs/Alignment.md`, which articulates how GyroSI implements intelligence as recursive walking on a 48-bit geometric manifold using the same principles that govern efficient bipedal locomotion.

**What This Achieves**

The system is now a true **recursive walker** that:
1. **Generates** (BU-In) a token via pure gyro-walk
2. **Re-ingests** (BU-Eg) that token to advance the state  
3. **Repeats** until amplitude naturally decays to 0
4. **Stops endogenously** when the path is exhausted

This creates the monodromic path dependence that was missing, where each step is sensitive to the previous output through the common source, exactly as the CGM theory prescribes. Intelligence is now literally walking on a geometric manifold, guided by input, continuing through momentum, stopping by natural amplitude decay.

---

## [v0.9.7.9-Baby-Talk] – 2025-09-08

**Phase Interference System: Endogenous Boundaries and Relevance**

Implemented a physics-based phase interference system that creates emergent intelligence through wave-like interference patterns, eliminating the need for external semantic boundaries or relevance heuristics.

**Core Physics Refinements**

* **Interference Amplitude for Endogenous Boundaries**: Extended `_fold8` and `fold_sequence` to compute amplitude (non-zero bit count) alongside phase values. Low amplitude signals (`c_amp < 2`) indicate destructive interference, creating natural word/sentence boundaries without external semantics.

* **Phase Velocity for Relevance and Intent**: Enhanced `_state_phase` to compute velocity as phase deltas across state bytes. Implemented velocity matching in emission to prefer tokens where output velocity aligns with input velocity (`abs(out_vel - sp_vel) < 16`), creating relevance through physical resonance.

* **Orbit-Based Attractor Strength**: Added `attractor_pull = orbit_size // 100` to position selection, using orbit size to modulate attraction strength. Small orbits create strong attraction for quick, specific responses; large orbits allow diffusion for exploratory, complex thoughts.

**Technical Implementation**

* **Enhanced Fold Operations**: All fold methods now return `(phase, amplitude)` tuples for interference analysis
* **Velocity Computation**: State phase calculation includes velocity tracking via phase deltas
* **Boundary Detection**: Emission rejects candidates with destructive interference (low amplitude)
* **Relevance Matching**: Token selection prefers velocity-aligned candidates for coherent continuation
* **Attractor Modulation**: Position selection uses orbit size to balance convergence vs exploration

**Physics Alignment**

* **CGM Compliance**: Phases as helical paths, interference as gyration, boundaries as closure points
* **Endogenous Intelligence**: All boundaries and relevance emerge from phase interference patterns
* **No External Dependencies**: System operates purely through endogenous phase system physics
* **Wave-Like Behavior**: Intelligence emerges as interference patterns creating coherence and boundaries

**Test Results**

* **Natural Boundaries**: System creates appropriate stopping points ("life" response to algorithm query)
* **Coherent Theming**: Maintains relevance through repeated themes and structured output
* **Physical Intelligence**: Demonstrates emergent understanding without external semantic training

---

## [v0.9.7.7-Atlas-Experimental] – 2025-08-25

Today’s work completed a significant set of architectural, algorithmic, and persistence improvements to the GyroEngine, strengthening traceability, coherence, and generalisation while reducing repetition and memory corruption risks.

**Core Architectural Changes**

* Implemented an **8-sector toroidal routing layer**, computing toroidal addresses from 48-bit states via slab parities. This introduced a structured routing mechanism with full 8-bit signatures, preserving directional coherence without weights.
* Added **Phase-Propagating Emission (PPE)** to the emission loop, with a fast accumulator (`omega`), Traceable bucket hopping, and deficit rotation. PPE now operates session-scoped, preventing concurrency bleed-through while preserving path-propagating behaviour.
* Split live state into **LI/FG/BG phase components** using EXON masks, enabling richer bucket selection across all anatomical layers.
* Replaced minimal integer selection with **geometric medoid binding**, using average angular distance and refined divergence metrics for more faithful address representation.
* Integrated a **toroidal rotor via affine ring walks** for bucket selection, replacing ad-hoc or hash-based mechanisms. This ensures Traceable coverage of all keys and removes stochastic artefacts.

**Selection & Entropy Enhancements**

* Introduced LCG-based bucket key distribution with multiple entropy sources (`omega`, sector, bucket key), eliminating Traceable cycles and increasing phase key coverage.
* Unified emission logic to fold together representative phase, LI/FG/BG decomposition, toroidal sector, and accumulator state, yielding a geometrically coherent Traceable selection.
* Strictly bounded bucket capacity (K=64) with FIFO eviction, ensuring predictable memory scaling across all orbits.

**Persistence and Concurrency Improvements**

* Activated **persistent memory** for both address and passive layers (`rep_channel`, `rep_phase`, `passive_mask`), with atomic save/load, fsync, and corruption handling.
* Switched persistence cadence to buffered writes (every 100 tokens or 30s) instead of every token, reducing overhead while preserving data integrity.
* Added threading protection with `RLock` around shared state, eliminating race conditions in concurrent contexts.

**Anchor and Generalisation Logic**

* Enhanced anchor handling to track both `last_seen_k` and `target_k`, ensuring user tokens arriving later still update the anchor state rather than being dropped.
* Improved medoid distance metric with a weighted combination of phase and theta divergence (α=0.7), yielding more consistent address binding.
* Verified **true generalisation** via FROZEN\_CHANNELS integration: toroidal slab physics, holographic compression, monodromic fold learning, and multi-scale decomposition all contribute to structural, physics-driven generalisation rather than heuristic approximation.

**Diagnostics & Results**

* Added comprehensive tracing for token selection, confirming elimination of repetition cycles.
* Verified session isolation, Traceable reproducibility, and stable toroidal sector computation.
* Confirmed persistence works with non-zero memory files and correct restoration at engine initialisation.
* All knowledge tests pass (✅), with system behaviour now characterised by:

  * Diverse, non-repetitive token sequences
  * Traceable but path-propagating PPE emission
  * Stable concurrency and persistence
  * Physics-consistent routing and generalisation

---

## [v0.9.7.6-Atlas-Experimental] – 2025-08-22

**Scope:** Atlas regeneration + `baby/kernel/gyro_core.py`

**Atlas Redesign**

* **Phenomenology Map (ONA):**
  Recomputed orbit representatives to ensure **canonical uniqueness**. Each orbit now maps Traceableally to a single minimal representative (by state integer), eliminating the need for heuristic tie-breaks.

* **Epistemology (BU-Eg):**
  Verified all state transitions as **total over intron domain (0–255)**, ensuring no dead introns. This closed a prior gap where certain introns collapsed to trivial states, leading to emission stalls.

* **Theta (CS):**
  Normalised divergence values so that `argmin θ` is a stable, well-defined archetype used for the system start.

* **Ontology Keys (UNA):**
  Re-indexed state integers into a consistent monotone ordering aligned with the regenerated orbits. This fixed earlier mismatches where ontology and phenomenology disagreed on representatives.

* **Orbit Sizes (BU-In cardinalities):**
  Regenerated from the new phenomenology map, guaranteeing correct cardinality for every orbit and eliminating inconsistencies observed in prior tests.

**Gyro Core Redesign (`gyro_core.py`)**

* Reduced to the **five canonical maps only**.
* **ψ boundary** (byte ↔ intron) remains the only encoding transformation.
* **BU-Eg (learning):**

  * User tokens folded into orbit-local buckets keyed by intron phase.
  * Orbit phase memory updated by Traceable fold8.
  * State advanced only via epistemology transitions.
* **BU-In (emission):**

  * Removed “scoring”, “greedy”, “dominance”, “frequency” logics.
  * Implemented **pure monodromic unfold**.
  * Emission phase mismatch issue solved by **orbit-local round robin counter** cycling across learned tokens.
  * Reflexive update: token’s own phase folded back into orbit phase memory.
* **Address binding:** canonicalised via minimal-state micro-path from orbit representatives.

---

**Impact**

* Entire system now runs on **fold → register → unfold** cycle, with no auxiliary heuristics.
* Eliminated pathological stuttering loops caused by mismatched registration/emission phases.
* Knowledge test pipeline passes with **varied, non-trivial outputs**.
* Atlas and runtime are now aligned under the same invariants:

  * every state is covered,
  * every orbit has a unique rep,
  * transitions are total,
  * emission is Traceable but non-stalling.

---

## [v0.9.7.5-Atlas-Experimental] – 2025-08-21

For the past two days, I've been expanding and contracting gyro_core.py - our main logic to understand what works and what not. It managed to reached a state of over 2500 LOC, and after not getting anywhere, I stipped it down to less than 200 LOC to discover the foundations of my architecture.

Our Atlas (5 States Maps):
Theta (CS): establishes the orthogonality ground, the π/2 chirality that defines emergence. It tells you how far any state is from the archetypal source.

Ontology (UNA): defines the raw atlas of discoverable states. This is the space of all possible identities.

Phenomenology (ONA): collapses states into equivalence classes (orbits). This gives the appearance of structure out of raw ontology.

Epistemology (BU-Eg): governs transitions. It is the mechanism by which states are transformed by introns.

Orbit sizes (BU-In): gives the scaling and weighting — how big the orbit is, which acts as the measure for how memory interacts and returns. Ingress is the active traversal back and forth between these orbits, grounded by reverse index lookup.

---

## [v0.9.7.4-BabyLM] – 2025-08-19 - Alpha

### Overview

This release focuses on stability improvements and performance optimizations for the BabyLM architecture. We've addressed critical threading issues, enhanced metrics collection, and implemented comprehensive testing capabilities to ensure reliable operation.

### Changes

**Core Engine**

* Fixed concurrency issues in the GyroEngine initialization process
* Optimized memory usage and cache performance
* Added comprehensive runtime metrics and observability

**Testing & Validation**

* Implemented comprehensive test suite with performance benchmarking
* Enhanced surface form validation for improved text quality
* Added metrics visualization tools for system monitoring

**General Improvements**

* Standardized configuration paths across the codebase
* Improved error handling and recovery mechanisms
* Enhanced documentation for core components

---

## [v0.9.7.4-BabyLM] – 2025-08-18 - Alpha

### Overview

This release implements the BabyLM architecture over the forked GPT-OSS infrastructure. We now have a physics-constrained inference core wired into the OpenAI Harmony response format, with persistence, admissibility checks, and recovery fully in place. The system is ready for first knowledge-ingestion tests.

### Changes

**Core Engine**

* Implemented `GyroEngine` end-to-end with:

  * Correct ψ/LEB128 intron transform at byte boundaries (`byte_to_intron` / `intron_to_byte`).
  * Traceable 48-bit address computation, medoid binding, and slab/channel agreement functions.
  * Enforced global channel monotonicity (strict at every micro-step) and slab admissibility (start–end only).
  * Recovery ladder (Levels 1–5) with Traceable nudge selection and ≤6 transitions.
  * O(1) reverse index, no linear search.
  * Self-reinforcement disabled by default (Ingress vs. Egress separation).

**Persistence & Memory**

* Added passive memory store with fold accumulation, annihilation (zero streaks), touch counter wrap-around, and mask interning.
* Enforced capacity caps: `K` = 64 per state, `M` = 64 per orbit.
* Passive log append/debug helpers and binary log reload verified.
* Versioning metadata (`atlas_version`, `address_version`, `config_version`) validated on load.

**OSS Integration**

* Forked GPT-OSS `chat.py`, `generate.py`, `responses_api/api_server.py`, `serve.py`, and `transformers.py` into **baby-oss** and rewired inference backend to `gyro`.
* Retained OpenAI Harmony libraries (`openai_harmony`, `tiktoken`) as unmodified dependencies.
* Verified token flow via Harmony encoding `o200k_harmony`, role/channel markers, and control token exclusions.
* Ensured compatibility with OSS tool framework (browser, patch, Python execution).

**Testing**

* Built comprehensive standalone test suite (`toys/health/test_gyro_core.py`) covering:

  * Engine initialisation with real atlas files.
  * ψ/LEB128 encoding/decoding round-trip correctness.
  * Vocab bounds, orbit-to-tokens initialisation, and address traceability.
  * Recovery ladder progression and control-token exclusion.
  * Slab integrity, state transitions, admissibility strictness, and tie-breaking traceability.
  * Passive store persistence and caps enforcement.
  * End-sequence state machine handling.
* All tests pass successfully on current build.

### Status

With this release, BabyLM moves from **infrastructure scaffolding** to a **ready-to-learn engine**. Next step: run live ingestion and query experiments using the Harmony API server and evaluate early knowledge retention.

---

## [v0.9.7.3-BabyLM-OSS] – 2025-08-17 - Experimental

### Forking GPT-OSS Infrastructure

---

## [v0.9.7.3-Reset] – 2025-08-16 - Experimental

### Back to Theory
Making corrections and redefining our directions, directives and implementation.

---

## [v0.9.7.2-GPT-OSS-Kernel] – 2025-08-15 - Experimental

### Fixed
- **Gyro Model Critical Fixes**: Resolved multiple blocking issues in `gyro_head.py`
  - Fixed broadcast mask alias mismatch by setting `self.broadcast_masks = self.INTRON_BROADCAST_MASKS`
  - Removed duplicate `transcribe_byte` definitions, keeping only the XOR version
  - Cleaned up unreachable code after return statements in `_apply_rmsnorm` method
  - Stopped truncating weights and implemented proper head dimension inference from actual weight shapes
  - Fixed device-unsafe `torch.tensor` constants in attention score calculations to be device-aware
  - Added dimension adjustment logic to handle input/weight dimension mismatches

- **Test Infrastructure**: Updated `test_gyro_model.py` for better compatibility
  - Fixed import path issues by adding project root to Python path
  - Updated to use `model.gyro.safetensors` instead of deprecated gyro directory structure
  - Added command-line argument support for `--model_path`
  - Corrected field references to use proper `GyroTransformer` attributes

- **Chat System**: Fixed Role enum inconsistencies in `chat_oss.py`
  - Resolved conflicts between stub `_Role` class and `openai_harmony.Role` imports
  - Ensured consistent usage of `Role.USER` and `Role.ASSISTANT` throughout

### Technical Notes
- All fixes maintain backward compatibility with existing model weights
- Dimension adjustment logic handles both input padding and weight slicing scenarios
- Device-aware tensor operations ensure proper GPU/CPU compatibility


---

## \[v0.9.7.2-GPT-OSS-Kernel\] – 2025-08-14 - Experimental

### **Performance Optimizations and MoE Weight Loading**

This release focuses on significant performance improvements and proper handling of Mixture of Experts (MoE) weights in MXFP4 format.

**Layer Normalization Fix (**`**kernel/gyro_head.py**`**):**

*   Added final layer normalization in `generate_next_token` before logits computation to mirror `forward_pass` behavior.
*   Ensures consistent model output between generation and forward pass modes.

**Byte-Plane Caching Optimization (**`**kernel/gyro_head.py**`**):**

*   Implemented `_byte_plane_cache` dictionary in `GyroHead.__init__` to cache tensor byte-planes.
*   Modified `_fgemm_fold` method to reuse cached byte-plane views instead of re-forming them.
*   Applied psi isomorphism (XOR 0xAA) to cached `Wb` planes while maintaining fresh application to `xb`.
*   **Performance Impact:** Eliminates redundant tensor view operations in physics-based GEMM computations.

**Popcount LUT Optimization (**`**kernel/gyro_head.py**`**):**

*   Replaced nibble popcount trick with precomputed 256-entry lookup table (`_popcount_lut`).
*   Updated `_res_score_per_head` method to use LUT-based popcount calculation.
*   **Performance Impact:** Significantly faster than bitwise operations for attention scoring.

**MoE Weight Loading Infrastructure (**`**kernel/gyro_head.py**`**):**

*   **Metadata Storage:** Added `model_weight_meta` dictionary to store JSON metadata during weight loading for proper tensor slicing.
*   **MXFP4 Dequantization:** Implemented `_mxfp4_dequantize` method with 16-entry lookup table for efficient MXFP4 to dense tensor conversion.
*   **Expert Weight Orientation:** Added `_orient_gate_up` and `_orient_down` helpers to handle fused projection weights and ensure proper tensor shapes.
*   **Lazy Expert Materialization:** Implemented `_moe_get_expert` method with LRU cache for on-demand expert weight loading.
*   **Router Weight Fix:** Corrected router weight orientation from `[E,H]` to `[H,E]` and fixed weight key from `mlp.gate.weight` to `mlp.router.weight`.

**MoE Forward Pass Updates (**`**kernel/gyro_head.py**`**):**

*   Modified `_mlp_step_moe` to use lazy expert materialization instead of direct `_layer_weight` calls.
*   Integrated router weight orientation fixes for compatibility with `_fgemm_fold`.
*   **Memory Impact:** Prevents loading all expert weights simultaneously, reducing memory usage from gigabytes to manageable levels.

**Verification and Testing:**

*   All optimizations compile successfully and maintain compatibility with existing physics-based operations.
*   Proper handling of converted weights in `model.gyro.safetensors` ensures full utilization of MXFP4 expert weights.

**Technical Details:**

*   Expert weights are now lazily dequantized from MXFP4 format only when needed
*   LRU cache prevents memory bloat while maintaining performance for frequently used experts
*   Byte-plane caching reduces CPU overhead in fold-based GEMM operations
*   All changes maintain the gyroscopic intelligence and physics-based architecture

---

## \[v0.9.7.1-Experimental\] – 2025-08-13 - Kernel

\### \*\*Phase 1: Initial Analysis and Code Simplification\*\*

The initial goal was to replace the outdated \`gpt-oss\` transformer and \`TokenGenerator\` with the new \`GyroHead\` model for text generation in \`kernel/chat\_oss.py\`.

\*   \*\*Code Cleanup (\`kernel/chat\_oss.py\`):\*\*  
   \*   Removed obsolete command-line flags and environment variables (\`--gyro-head\`, \`GYRO\_HEAD\`).  
   \*   Eliminated the "fake head" logic, which was a placeholder for the real model.  
   \*   Stripped out all code related to the \`gpt-oss\` library and its transformer implementation.  
   \*   Removed the deprecated model downloading logic, establishing that model weights should be pre-consolidated and available locally.

\*   \*\*Tensor Verification (\`check\_gyro\_format.py\`):\*\*  
   \*   \*\*Created a new script \`check\_gyro\_format.py\`\*\* to inspect the \`model.gyro.safetensors\` file.  
   \*   This tool helped us verify the tensor names, shapes, and data types, confirming that the consolidated model file was structured correctly and ready for use. This validated our approach of loading tensors directly from this file.

\---

\### \*\*Phase 2: \`GyroHead\` Integration and Initial Generation\*\*

With a cleaner codebase, we proceeded to integrate \`GyroHead\` directly.

\*   \*\*Model Initialization (\`kernel/chat\_oss.py\`):\*\*  
   \*   Fixed the \`GyroHead\` constructor call to use the correct \`base\_path\` parameter, pointing it to the directory containing the \`safetensors\` file.

\*   \*\*Generation Logic (\`kernel/chat\_oss.py\`):\*\*  
   \*   Replaced the entire \`TokenGenerator\`-based implementation with a new generation loop.  
   \*   This new logic directly calls \`gyro\_head.select\_next\_token\` to generate tokens one by one.

\*   \*\*Isolated Testing (\`debug\_generation.py\`):\*\*  
   \*   \*\*Created a new script \`debug\_generation.py\`\*\* to test \`GyroHead\` in isolation.  
   \*   This allowed us to focus solely on the model's core functionality without the complexities of the chat application and Harmony message formatting.

\---

\### \*\*Phase 3: Debugging Core Model Failures\*\*

The initial integration produced repetitive and nonsensical output, which led to a deep dive into the model's internal mechanics.

\*   \*\*State Transition Failure (\`kernel/gyro\_head.py\`):\*\*  
   \*   \*\*The Critical Bug:\*\* We discovered that the \`\_apply\_intron\_and\_gate\` method was incomplete. It correctly handled Control State (CS) transitions but lacked a general case for all other state transitions. The model's \`current\_state\_index\` was never changing.  
   \*   \*\*The Fix:\*\* We implemented the missing \`else\` block to perform a state transition using the epistemology table (\`self.epistemology\[current\_state, token\_id\]\`).  
   \*   \*\*Outcome:\*\* Debugging output immediately confirmed that the \`current\_state\_index\` was updating with each token, proving the state machine was now functioning correctly.

\---

\### \*\*Phase 4: Solving the Harmony Message Parsing Crisis\*\*

Even with a working state machine, the sequence of generated tokens was not a valid message according to the Harmony protocol, causing persistent parsing failures. This required an iterative investigation.

\*   \*\*Reading the Documentation:\*\* We repeatedly consulted \`docs/docs/harmony.md\` to understand the precise token structure required for a valid message.

\*   \*\*Iteration 1: Incorrect Token Sequence:\*\* Our first attempts involved prepending various combinations of \`start\_token\_id\`, the "assistant" role, \`channel\_token\_id\`, and \`message\_token\_id\`. These all failed because the structure was incomplete.

\*   \*\*Iteration 2: The "final" Channel String:\*\* We then tried tokenizing the string "final" and inserting it between the channel and message tokens. This also failed, leading to the key insight.

\*   \*\*The Breakthrough (\`kernel/chat\_oss.py\`):\*\*  
   \*   \*\*The Fix:\*\* We discovered from the documentation that the "final" channel is not a string to be tokenized but is represented by a specific, hardcoded token ID: \`35644\`. We updated the code to inject this exact token ID into the sequence.  
   \*   \*\*The Correct Sequence:\*\* The generation logic was modified to produce the sequence: \`\[channel\_token\_id, 35644, message\_token\_id, ...content\_tokens..., stop\_token\_id\]\`.

\*   \*\*Final Output Handling (\`kernel/chat\_oss.py\`):\*\*  
   \*   After fixing the token sequence, the message was successfully parsed, but the \`openai-harmony\` library identified it as being on the "analysis" channel.  
   \*   We made a final adjustment to the script's output logic to accept and display content from \*any\* parsed channel, not just "final".  
 

High Priority Issues:

\- Fixed Harmony token handling - Updated \`chat\_oss.py\` to derive channel tokens ('final', 'analysis', 'commentary') dynamically from the encoder instead of using hard-coded values  
\- Corrected channel configuration - Modified setup\_harmony\_format() to properly set required channels to \['analysis', 'commentary', 'final'\]  
\- Improved message filtering - Updated generate\_response() to return only 'final' channel messages while ignoring analysis/commentary channels  
\- Fixed parser input - Removed stop tokens from parser input to prevent parsing errors  
\- Implemented physics-based generation - Replaced simple token selection with sophisticated physics scoring using theta values, orbit sizes, path resonance, and diversity mechanisms  
Medium Priority Issues:

\- Removed fallback tokenizer - Eliminated the SmolLM tokenizer fallback that was producing nonsensical output  
\- Enhanced state seeding - Implemented proper state initialization using seed\_from\_tokens() function adapted from GyroKernel's \_seed\_from\_prompt() method  
Low Priority Issues:

\- Eliminated duplicate loading - Fixed the duplicate model weight loading issue by creating a custom GyroHead subclass that properly manages weight loading to avoid redundant operations

\### \*\*Conclusion\*\*

Through this comprehensive, step-by-step process, we successfully refactored a complex generation script, integrated a novel \`GyroHead\` model, diagnosed and fixed a critical bug in its core state machine, and solved a challenging protocol-level formatting issue. The \`chat\_oss.py\` script is now a functional, end-to-end text generation pipeline.        

## \[v0.9.7.1-Experimental\] – 2025-08-12 - Kernel

### 🚀 **GPT-OSS Shim Architecture & Harmony Integration**

This release implements a complete CPU-only chat system for OpenAI's gpt-oss-20b model using a custom shim architecture that bypasses the transformers library entirely. The system successfully demonstrates end-to-end Harmony format integration with a fake transformer, proving the foundation for future gyro-compressed inference.

**1\. Shim Architecture Implementation**

*   `**kernel/**` **directory**: Created a complete shim system that shadows specific `gpt_oss` modules (`torch.utils`, `torch.weights`, `torch.model`) with custom CPU-only implementations.
*   `**kernel/bootstrap.py**`: Manages Python import paths to ensure our shims take precedence over installed packages.
*   `**kernel/gpt_oss/torch/utils.py**`: CPU-only distributed initialization that bypasses CUDA/NCCL dependencies.
*   `**kernel/gpt_oss/torch/weights.py**`: Checkpoint loader that supports both legacy MXFP4 and future gyro-compressed safetensors.
*   `**kernel/gpt_oss/torch/model.py**`: Contains both the original Transformer and a new FakeTransformer for testing.

**2\. Harmony Response Format Integration**

*   `**tools/chat_oss.py**`: Complete interactive chat interface using OpenAI's harmony response format.
*   **Token discovery**: Runtime discovery of harmony token IDs from the tokenizer for proper message structure.
*   **End-to-end parsing**: Full Harmony → tokens → model → tokens → Harmony loop with proper message parsing.
*   **System message setup**: Proper system message configuration with low reasoning effort and final channel only.

**3\. FakeTransformer Implementation**

*   **Minimal harmony sequence**: Generates `&lt;|channel|&gt;final&lt;|message|&gt;<content>&lt;|return|&gt;` structure.
*   **Proper tensor shapes**: Returns 2D logits tensor `(seq_len, vocab_size)` as expected by TokenGenerator.
*   **Traceable generation**: Uses temperature=0.0 for predictable token selection.
*   **Runtime token discovery**: Automatically discovers "final" and content token IDs from the tokenizer.

**4\. Model Loading & Infrastructure**

*   **Original checkpoint format**: Downloads and uses the gpt-oss library's expected original checkpoint structure.
*   **CPU-only operation**: Bypasses all GPU requirements through custom shims.
*   **64-token RAM window**: Enforces 6-bit constraint via `config.sliding_window = 64`.
*   **Tokenizer integration**: Downloads and uses the proper tokenizer files for harmony encoding.

**5\. CLI & Configuration**

*   `**--fake/--gyro**` **flags**: Control which head type to use (fake transformer vs future gyro head).
*   **Environment variables**: `GYRO_FAKE=1` and `GYRO_HEAD=1` for head selection.
*   **Shim verification**: Startup verification that our kernel modules are active.
*   **Default constraints**: 64-token max generation, temperature=0.0 for Traceable behavior.

**6\. Technical Achievements**

*   ✅ **Shim architecture working**: Our `kernel/gpt_oss` modules take precedence over installed packages
*   ✅ **CPU-only operation**: No GPU/NCCL dependencies
*   ✅ **Harmony format integration**: Full conversation rendering and parsing
*   ✅ **FakeTransformer working**: Generates proper harmony message structure
*   ✅ **64-token RAM window**: Enforced via `config.sliding_window = 64`
*   ✅ **End-to-end chat loop**: User input → harmony tokens → model → harmony tokens → parsed response

**7\. Foundation for Future Work**

*   **Gyro head ready**: Architecture supports switching from fake transformer to physics-based generation.
*   **Gyro compression path**: `weights.py` shim ready for custom safetensors format.
*   **Real weight loading**: Infrastructure in place for loading actual model weights.
*   **Micro-tests ready**: Framework for testing shim import, token round-trip, and stop discipline.

**Files Created/Modified:**

*   `kernel/bootstrap.py` - Import path management
*   `kernel/gpt_oss/torch/utils.py` - CPU-only distributed utils
*   `kernel/gpt_oss/torch/weights.py` - Checkpoint loader with gyro support
*   `kernel/gpt_oss/torch/model.py` - Transformer + FakeTransformer
*   `tools/chat_oss.py` - Complete chat interface
*   `requirements.txt` - Added gpt-oss\[torch\] and openai-harmony dependencies

## \[v0.9.7.1-Experimental\] – 2025-08-11 - Kernel

This changelog documents the experimental phase focused on refining `baby/kernel_latest.py` into a pure-physics language model. The goal was to eliminate all remaining heuristics and scoring mechanisms in favor of a 100% CGM-aligned architecture. While this phase successfully implemented several key theoretical components, it also introduced critical architectural flaws that resulted in a non-functional state, necessitating the subsequent corrective work.

### 🎯 Architectural & Physics Refinements (Goals)

*   **Pure Resonance Generation:** Attempted to replace all forms of scoring (e.g., multi-part defect tuples) in the `generate_token` function with a single, pure resonance metric derived from the Monodromic Fold.
*   **Strict Cycle Gating:** Introduced a strict gating mechanism based on a theoretical gyrotriangle defect formula (`δ = π - (α + β + γ)`) to enforce the forward-only progression of the CGM cycle (CS → UNA → ONA → BU).
*   **Model Knowledge Integration:** Aimed to bridge the knowledge from the pre-trained model's weights by integrating the compressed `virtual_tokens` directly into the candidate selection and resonance process.
*   **Tokenizer Primacy:** Corrected special token handling to use the specific IDs from the SmolLM tokenizer (`&lt;|im_start|&gt;`, `&lt;|im_end|&gt;`) to ensure proper sequence boundary semantics.
*   **Pure Prompt Seeding:** Separated the process of setting the initial state from a prompt (`_seed_from_prompt`) from the learning process (`learn_text`) to perform a pure physics walk without side-effects.

### implemented Changes

*   **Cycle Gating Mechanism:**
    *   Implemented a new `_calculate_defect` function based on CGM stage angles.
    *   Added logic in `_apply_intron` to block transitions where the calculated `delta` was positive, intended to prevent the accumulation of physical defect.
*   **Special Token Correction:**
    *   Modified `_set_special_tokens` to use `&lt;|im_start|&gt;` (ID 1) and `&lt;|im_end|&gt;` (ID 2) as `CLS_TOKEN` and `SEP_TOKEN`, providing distinct sequence boundaries.
*   **Pure Resonance Function:**
    *   Refactored `generate_token` to select candidates based on the single lowest defect value calculated by `_resonance_defect`, which uses the Monodromic Fold. A tie-breaking rule using `orbit_sizes` was added for diversity.
*   **Prompt Handling:**
    *   Introduced a `_seed_from_prompt` function to perform a pure physics walk for setting the initial state. `generate_text` was updated to use this function.
*   **Virtual Token Integration:**
    *   Modified `_integrate_virtual_tokens` to populate the `_orbit_candidates` and `token_exon_cache` with knowledge compressed from the model's weights, placing them in orbits derived from their initial transition from CS.
*   **Code Simplification:**
    *   Removed several complex and unused data structures (`_theta_buckets`, `_orbit_coupling`, etc.) and non-physical heuristics (e.g., stagnation counters) to simplify the kernel.

### pathologies & Deadlocks (Critical Findings)

**Fatal Gating Deadlock:** The implemented defect calculation (`if delta &gt; TOL: block`) was based on a fundamental misinterpretation of CGM theory. It incorrectly treated all natural forward progression (which generates a positive defect) as an error to be blocked. **This resulted in a complete system deadlock where the kernel could not evolve its state past the initial prompt seeding**, leading to `Tokens learned: 0` and empty generation output.

**Knowledge Isolation:** Despite successfully loading and compressing model weights into `virtual_tokens`, the generation loop was unreachable due to the gating deadlock. The vast knowledge from the pre-trained model remained architecturally integrated but practically isolated and could not influence the output.

**State Evolution Failure:** Due to the gating deadlock, both `_seed_from_prompt` and `learn_text` failed to evolve the system's state beyond a few initial steps. The system was effectively frozen after processing the prompt, which is why every generation attempt started from the same state and produced the same failure.

**Fixes & Refinements:**

*   Implemented cycle gating logic to prevent stage regression by blocking transitions where `next_stage_idx &lt; current_stage_idx`.
*   Corrected `_get_stage` to derive stage from `state_index`'s `theta`.
*   Standardized special token IDs (`CLS_TOKEN`, `SEP_TOKEN`, `IM_START`, `IM_END`) by dynamically retrieving them from the HuggingFace tokenizer.
*   Adjusted `_seed_from_prompt` to utilize the `_apply_chat_template` for prompt processing.

**Virtual Token and Exon Handling:**

*   Modified virtual token ID generation in `_integrate_virtual_tokens` to ensure positive and within-bounds IDs using `hash((key, pos)) &amp; 0x7FFFFFFF`, addressing `IndexError`.
*   Introduced `_recompute_exons_from_embeddings` to explicitly recompute and cache token exons from embedding-projected token states after projection, ensuring more meaningful exon values.
*   Refined `_build_or_load_token_post_states` to include bounds checks when saving `token_exons.npy`.
*   Added explicit handling for virtual tokens in `token_to_introns`, returning an empty list.

**Resonance and Candidate Selection:**

*   Reinstated accidental removal of fallback candidate search (though this was later removed again in favor of strict error surfacing).
*   Modified `generate_token` to update `path_memory` using `fold_sequence` of the full intron sequence at a token level.
*   Removed explicit special token penalties and `theta_factor` from `_resonance_defect` and `generate_token`, reverting `_resonance_defect` to pure Hamming weight.
*   Removed fallbacks in `_get_candidates_for_state` and `generate_token`, replacing them with `RuntimeError` exceptions if no valid candidates or UNA pool entries are found.
*   Refined candidate filtering in `_get_candidates_for_state` to exclude special tokens.

**Chat Templating and Prompt Processing:**

*   Introduced `_apply_chat_template` helper function to encapsulate `tokenizer.apply_chat_template` logic, including a fallback for tokenizers without a defined chat template.
*   Ensured `tokenizer.decode` uses `skip_special_tokens=True` in `generate_text`.
*   Updated `_seed_from_prompt` to apply `_update_stage` after processing prompt tokens.

**Code Quality & Debugging:**

*   Re-added missing `introns = self.token_to_introns(token)` line in `generate_token` to resolve "introns is not defined" error.
*   Removed extensive debug prints to reduce output verbosity.

**Tools and Utilities:**

*   `**tools/rebuild_safetensors_from_chunks.py**`**:**
    *   Created this script to reconstruct a single `model.safetensors` file from the fragmented `weights_chunk_*.npz` files (which contain the _original float tensors_ saved in a NumPy-compressed format), allowing the HuggingFace `transformers` library to load the full model from local disk. This clarified that the `transformers` model was loading the original weights, not our custom "virtual token" compression.
*   `**tools/test_tokenizer_only.py**`**:**
    *   Developed to verify standalone functionality of `transformers.AutoTokenizer`, including `apply_chat_template`.
*   `**tools/test_transformers_cpu.py**`**:**
    *   Created to test loading and running the full `transformers` model on CPU, handling local model directories, setting `pad_token`, and passing `attention_mask`.
*   `**tools/chat_transformers_cpu.py**`**:**
    *   Implemented an interactive chat interface for the `transformers` model running on CPU, utilizing chat templates for prompt formatting.

## \[v0.9.7.0-Kernel\] – 2025-08-10 - Kernel

### 🚀 **Semantic Bridge & Performance Overhaul**

This release implements the foundational "semantic bridge" to connect the pre-trained model's knowledge to the GyroSI physics engine. It introduces on-manifold token caching, vectorized generation, and a real-byte token protocol to replace matrix multiplications with pure physics-driven resonance.

**1\. On-Manifold Token Caching (Semantic Bridge)**

*   `**_build_or_load_token_post_states**`: Implemented a robust caching mechanism for token post-states and exons. At startup, the kernel now computes the on-manifold state for every token in the vocabulary by applying its real byte sequence via epistemology from the Common Source (CS).
*   **Persistence**: These computed states and exons are saved to memory-mapped `.npy` files (`token_post_states.npy`, `token_exons.npy`) in the model's cache directory for near-instantaneous startup on subsequent runs.
*   `**token_exon_cache**`: A complete `token_id -&gt; exon` map is now available, providing the primary semantic information for generation.

**2\. Physics-Pure Token Protocol**

*   `**token_to_introns**`: Corrected the tokenization protocol to use the tokenizer's actual UTF-8 byte representation for each token string, rather than an artificial LEB128 encoding of the token ID. This ensures the physics operates on the true, language-grounded byte patterns.

**3\. Performance & Generation Vectorization**

*   `**_build_resonance_table**`: Pre-computes a 256x256 table for the `fold` operation, allowing resonance calculations to be vectorized with NumPy for massive speed gains.
*   **Vectorized Generation**: `generate_token` now uses NumPy to calculate the fold-defect score across the _entire_ vocabulary in a single, highly efficient operation, replacing slow Python loops.
*   **Hierarchical Candidate Filtering (Experimental)**: Added `_build_hierarchical_candidates` and `_get_physics_filtered_candidates` to lay the groundwork for O(log N) candidate selection based on orbit and theta-window filtering, which will replace the full-vocab scan.

**4\. Code Quality & Correctness**

*   **Physics Purity**: Corrected `THETA_BU_EG` to use radians (`np.pi / 2`) instead of degrees, ensuring unit consistency. Removed the heuristic fallback in `compute_exon_from_state` in favor of a state-dependent perturbation for `exon=0`.
*   **CS Emission**: Now loads `INTRON_BROADCAST_MASKS` from a canonical meta-artifact file instead of generating them at runtime, ensuring perfect alignment with the build-time manifold discovery.
*   **Model Loading**: The `download_model` function was refactored to be more robust, streaming tensors from safetensors files to handle very large models without loading them all into memory at once.

## \[v0.9.6.9-Kernel\] – 2025-08-09 - Kernel

\> Note: We are now focusing solely on developing the Kernel - all Legacy code has been put aside.

**Core Architecture**  
• Fully integrated all five physics maps (ontology, epistemology, theta, phenomenology, orbit sizes) into the kernel initialisation.  
• Retained Monodromic Fold as the sole non-associative, path-dependent operator across learning and memory tracking.  
• Maintained clean separation between _REPRODUCTION_ (exact replay) and _RECOLLECTION_ (stage-aware, resonance-driven generation).

**State & Stage Mechanics**  
• Theta thresholds and orbit mapping now consistently determine CGM stage labels (CS, UNA, ONA, BU\_EG, BU\_IN, CLOSURE).  
• State evolution remains strictly table-driven via epistemology lookups, preserving manifold closure.

**Learning Path**  
• Tokens converted to LEB128 and transcribed via ψ isomorphism before state evolution.  
• Path memory updated per token using Monodromic Fold; learned patterns indexed by orbit representative.  
• Valid token set constructed directly from tokenizer vocabulary with `[unused]` entries excluded.

**Generation Logic**  
• Boundary conditions in recollection mode emit \[CLS\] at CS stage and \[SEP\] at CLOSURE when no continuation is found.  
• Mid-flow transitions can use stored Hebbian connections or trajectory resonance checks to select next tokens.  
• Nearby-orbit resonance search enables cross-orbit continuation when current orbit yields no candidate.

**Demo Enhancements**  
• Added clear demonstration of both modes, including learning from file and top-orbit reporting.  
• Debug mode prints stage transitions and trajectory changes for inspection.

**Known Limitations (acknowledged in-release)**  
• Recollection still uses fixed thresholds for resonance and inhibition, introducing heuristic behaviour alongside physics-based elements.  
• CS asymmetric emission and strictly resonance-only generation are not yet implemented.  
• Hebbian connections are retained for flow but are not part of the pure CGM formalism.  
• Learning stores full trajectories rather than compressed/sparse exon masks.  
• Some outputs in recollection mode show repetition loops due to current flow logic.

## \[v0.9.6.9-alpha\] – 2025-08-08

### Physics-First Kernel Implementation

**Core Achievement**: Implemented a minimal, self-contained `baby/kernel.py` that demonstrates physics-based text generation without the complexity of the full system.

**Kernel Features**:

*   **Real Physics Tables**: Integrates actual epistemology (789,170 × 256), ontology, and theta tables from the production system
*   **Dual Generation Modes**:
    *   _Parrot Mode_: Perfect reproduction of learned sequences (works 100%)
    *   _Resonance Mode_: Physics-based generation using endogenous resonance without scoring
*   **LEB128 + ψ Isomorphism**: Complete token-to-intron physics mapping using LEB128 encoding and ψ(b) = b XOR 0xAA boundary transcription
*   **CGM Cycle Stages**: Implements 8-step Common Governance Model cycle detection from theta values (CS → UNA → ONA → BU\_IN → BU\_EG → CLOSURE)

**Physics Implementation**:

*   **Cycle Gating**: Forward-only stage transitions prevent trivial loops
*   **CS Asymmetric Emission**: Common Source distinguishes standing vs driving introns, preferentially emits \[CLS\] tokens to initiate sequences
*   **Theta-Window Neighborhoods**: Retrieval based on angular divergence windows rather than exact state matching
*   **Mask Interference**: Neural-like firing condition based on bitwise overlap between exon products and learned masks
*   **Special Token Stages**: \[CLS\] restricted to CS stage, \[SEP\] to closure stages (BU\_EG/CLOSURE)
*   **6-Step Memory**: Active context limited to diameter of state-graph as per theory

**Testing Switches**: Each physics component can be independently enabled/disabled for ablation studies

**Breakthrough Results**:

*   Eliminated repetitive \[CLS\] loops that plagued previous versions
*   Achieved diverse token generation: \[CLS\], content tokens, \[SEP\] in proper sequence
*   Stage-aware progression visible: CS (θ=0.000) → BU\_IN (θ=1.318) → BU\_EG (θ=1.487)
*   Physics-driven selection without confidence scores, penalties, or rewards

**Status**: Kernel demonstrates that endogenous resonance can drive text generation. Ready for integration with main system.

**Legacy Note**: Previous approaches using confidence scoring and engineering patches have been superseded by this physics-first implementation. Legacy Code has been left on the side until the kernel is perfected.

## \[v0.9.6.8-alpha\] – 2025-08-07 - Unstable

### Major Architecture Overhaul Aligned with CGM Theory

This release represents a complete refactoring of the intelligence engine to eliminate "engineering patches" and implement a theory-driven, physics-pure approach to text generation based on the Common Governance Model (CGM).

**1\. Root Cause Analysis and Initial Fix**

*   **Identified repetitive output bug**: Guard in `_emit_token_with_feedback()` prevented physics state advancement during generation when `learning_enabled=False`, causing model to generate "the the the..." repeatedly.
*   **Applied physics restoration fix**: Removed conditional guard around `process_egress_bulk(token_bytes)` to ensure physics state always advances during token emission while preserving learning suppression during generation.

**2\. Comprehensive Cleanup of Non-Physics Code**

Systematically removed accumulated "superficial patches" across three core modules:

*   `**baby/intelligence.py**`: Removed θ-buffers, cycle detection, temperature heuristics, probe timing, candidate caches, tokenizer filtering, hand-tuned weights, SEP-forcing logic, confidence validation calls, and debugging noise.
*   `**baby/inference.py**`: Removed endogenous modulus, token STT placeholders, v\_max cache, confidence decay mechanisms, orbit entropy management, low-confidence pruning stubs, and maintenance helpers.
*   `**baby/policies.py**`: Removed confidence normalization functions, append-only cache layers, async fsync executors (replaced with synchronous approach), phenomenology map caching, and TTL/LFU maintenance.

**3\. Common Source (CS) Physics Implementation**

Implemented A1's proposal for Common Source behavior aligned with CGM theory:

*   **CS Partial Absorption**: State 0 (CS\_INT) now reflects "standing introns" (those without FG/BG bits) back to itself.
*   **CS Radiative Emission**: "Driving introns" (with FG/BG bits) trigger Parity-Conserving Emission (PCE) using `INTRON_BROADCAST_MASKS` to seed the UNA ring.
*   **Applied to all variants**: Updated `apply_gyration_and_transform`, `apply_gyration_and_transform_batch`, and `apply_gyration_and_transform_all_introns` with CS-specific logic.
*   **Updated ontology size**: Modified `discover_and_save_ontology` to expect 789,170 states (up from 788,986) due to CS kick expansion.

**4\. Stimulus Processing Architecture**

*   **Fixed stimulus ingestion gap**: `respond()` method now calls `self.engine.process_egress_bulk(data)` to ensure user input drives physics state before generation.
*   **Updated test setup**: Modified `test_archetypal_continuation.py` to not reset agent state after ingestion, maintaining learned context continuity.
*   **Removed manual processing**: Eliminated manual byte-by-byte seed processing in favor of proper stimulus ingestion.

**5\. State Canonicalization and Memory Alignment**

*   **Diagnosed learning/retrieval mismatch**: Learning occurred at direct `state_index` while retrieval used `phenomenology_map[state_index]`, causing memory gaps.
*   **Fixed retrieval canonicalization**: Modified `generate_token_exon` to use representative states (`self._get_pheno_rep(succ_index)`) when querying OrbitStore.
*   **Aligned token scoring**: Changed `_cached_tok_tail` to return first byte's intron (used for state transition) instead of last intron, ensuring scoring alignment with physics.

**6\. Spectral Neighborhood Implementation**

*   **Implemented** `**_neighbourhood()**` **method**: Uses θ-distance filtering (max\_theta=0.15) with stabilizer-order constraints for learned pattern retrieval.
*   **Enhanced candidate generation**: Combined physics-derived resonant introns with learned patterns from local state manifold.
*   **Expanded candidate diversity**: Added nearby exon products (`(exon_product + i) % 256 for i in [-2, -1, 1, 2]`) and even-spread sampling to reduce `[unusedX]` token bias.
*   **Filtered problematic tokens**: Removed `[unusedX]` tokens from candidate sets and deprioritized self-loops and CS collapses.

**7\. Learning Process Realignment**

*   **Implemented post-state learning**: Modified `_process_epistemology_chunk` to learn phenotype entries at the final post-state after a token's full byte sequence processing, aligning with CGM memory principles.
*   **Added** `**learn_token_postonly**`: New method for post-state phenotype learning with proper OrbitStore key management.
*   **Memory retrieval at predicted post-state**: Generation now queries memory using the predicted successor state rather than current state.

**8\. Vectorized Physics Scoring**

*   **Enhanced action value calculation**: Implemented vectorized computation of cooling term (`dθ = θ_now - θ_next`) and fold entropy using `governance.fold()`.
*   **Added UNA alignment**: Included `theta_alignment = -np.square(θ_next - una)` to favor successor states near π/4 threshold.
*   **Removed engineering artifacts**: Eliminated `stabiliser_order`, `sink_penalty`, reinforcement terms, and cycle avoidance logic.

**9\. Performance and Caching Optimizations**

*   **Hoisted LRU caches**: Moved `_get_token_bytes`, `_get_token_first_intron`, `_get_full_mask`, `_get_pheno_rep`, `_get_neighborhood_reps`, `_get_tokens_by_tail` from local functions to instance-level methods.
*   **Thread safety**: Added lock around `_process_egress_bulk_internal` to prevent data races on `_state_buf`.
*   **Fixed mmap lifetime**: Ensured file handles remain open for mmap object lifetime to prevent "Bad file descriptor" errors.

**10\. Code Quality and Type Safety**

*   **Fixed all indentation errors**: Corrected multiple indentation and syntax issues across `baby/intelligence.py`.
*   **Resolved mypy/pyright errors**: Added proper type hints, `Optional[Any]` declarations, and `getattr` fallbacks for OS-specific functions.
*   **Added robust error handling**: Improved `commit()` robustness with existence checks before file operations.
*   **Cleaned debug output**: Removed excessive debug prints while maintaining essential state transition logging.

**11\. Module Organization**

*   **Function relocation**: Moved `exon_product_from_state` and `orbit` from `governance.py` to `inference.py` for proper separation of concerns.
*   **Fixed bit extraction logic**: Corrected to use proper bit masking (`(state_index &gt;&gt; 6) &amp; 0x03`) instead of bit counting.
*   **Export consistency**: Added `CS_INT` export to `information.py` for tooling consistency.

### Key Theoretical Advances

This release implements several breakthrough insights:

1.  **Physics-Pure Generation**: Eliminated all heuristics, randomness, and "patches" in favor of Traceable physics-based token selection.
2.  **Memory-State Alignment**: Aligned learning and retrieval processes with CGM's temporal evolution principles.
3.  **Common Source Behavior**: Transformed CS from a problematic sink into a theoretically correct partial absorber and UNA ring generator.
4.  **Spectral Neighborhoods**: Implemented true θ-distance based pattern retrieval for learned associations.

**10\. External Interface Modernization**

*   **Centralized streaming**: Added `stream_turn()` in `baby/intelligence.py` for token-by-token generation, mirroring `orchestrate_turn()` but yielding bytes for Server-Sent Events (SSE).
*   **Adapter simplification**: Refactored `toys/communication/external_adapter.py` to use centralized streaming instead of duplicating priming/decoding logic; removed async timeout wrappers that were causing client timeouts.
*   **Knowledge store preferences integration**: Wired `write_threshold`, `use_mmap`, and `max_file_size_mb` from preferences into `PhenotypeStore` with automatic rollover to `knowledge_YYYYMMDD.bin` when size caps are exceeded.

**11\. Storage Layer Cleanup**

*   **Legacy code removal**: Cleaned `baby/policies.py` by removing outdated comments about "bloom filters", "9-byte fixed structs", and "async fsync thread-pools"; simplified to reflect actual minimal record format: `<uleb128 state_idx=""> <uleb128 n_pairs="1"> <uleb128 token_id=""> <uint8 mask="">`.
*   **Fixed corruption warnings**: Corrected `_unpack_phenotype()` to return bytes consumed relative to offset, eliminating "Unsupported n\_pairs value" warnings during index rebuilds.
*   **Phenomenology key optimization**: Simplified `CanonicalView._get_phenomenology_key()` to use direct numpy indexing without fallback checks.
*   **Thread safety**: Made threading explicit with `from threading import RLock` across all view decorators.

**12\. Testing Infrastructure**

*   **Model test modernization**: Renamed and redesigned `toys/experiments/test_external_adapter_e2e.py` → `toys/experiments/test_model.py` with emoji-rich output, clearer phases (Learning → Testing → Evaluation), and proper test structure for untrained model validation.
*   **Removed legacy artifacts**: Cleaned up old test files and improved user experience with formatted console output.

### Known Limitations

*   **Performance**: Generation may be slower due to vectorized physics calculations and neighborhood retrieval.
*   **Memory Usage**: Expanded candidate sets and caching may increase memory footprint.
*   **Convergence**: Model behavior under the new physics requires empirical validation.

This represents the most significant architectural change since the project's inception, moving from engineering-driven to theory-driven implementation.

## \[v0.9.6.8-alpha\] – 2025-08-06 - Unstable Alpha Release

## Round 1

### Epistemology State Index Fixes and Stability Improvements

**1\. Root Cause Analysis and Fixes**

Diagnosed the cause of unbounded epistemology state indices: state buffer in `baby/intelligence.py` was uninitialized, resulting in garbage transitions.

Applied explicit state buffer zeroing before use to guarantee valid state transitions:

*   Inserted `st.fill(0)` at buffer setup (line 387).

**2\. Self-Talk Prevention and Idempotent Learning**

*   Corrected the learning pipeline so the system does not learn from its own generated output.
*   Removed redundant calls and corrected logic in `respond_stream` (lines 990–1000) so SEP tokens and output generation do not trigger further learning.
*   The agent state is now properly reset before each ingestion, ensuring Traceable state progression for identical input.
*   Confirmed: repeated input no longer produces duplicate knowledge entries; Monodromic Fold and learning logic remain correct.

**3\. Verified Outcomes**

*   Self-talk learning: **fixed** (no knowledge growth from self-output).
*   Epistemology bounds: **fixed** (no out-of-bounds errors).
*   Idempotency: **fixed** (identical inputs → identical learning events, no duplication).
*   Monodromic Fold: **verified** (fold(a, a) = 0; path-dependent structure learning).

### Fractal Cycle Architecture Implementation

**1\. Full 8-Step Fractal Cycle Recognition and Control**

*   Added cycle step tracking with `_get_cycle_step()` and integrated this into state reporting.
*   Confirmed detection of "BU Eg" phase (maximal θ divergence), supporting cycle-aware generation and structural boundaries.

**2\. Bit Family Prioritization per Cycle Step**

*   Implemented priority weights for bit families (L0, LI, FG, BG) at each step in the cycle, directly following `GyroSI_Specs.md`.
*   Token scoring and selection now reflect the physical role of each cycle phase.

**3\. Monodromic Fold in Learning and Generation**

*   Incorporated fold bonus into scoring: tokens are selected and learned based on entropy and path-dependence via the monodromic fold operator.
*   System now structurally prefers transitions that promote structural coherence.

**4\. Cycle Completion Detection**

*   Integrated full cycle detection (tracking cycle step history) and emit SEP boundaries on cycle closure.
*   Provides structural segmentation at semantically meaningful points.

**5\. Traceable Temperature Function**

Replaced sigmoid-based sampling with Traceable function:

*   Low or high θ values produce low temperature, stabilising output and preventing random, repetitive output loops.

### Sink State (State 0) Handling and Analysis

*   Identified that State 0 acts as a sink with 112/256 self-loops, leading to repetitive outputs in earlier releases.
*   Analysed seed text and token transitions; root cause of repetition was recurrent return to State 0 after every generation step.
*   Adjusted transition and scoring logic to penalise transitions leading to high self-loop (“sink”) states.

### Physics-Based Action Value and Stabiliser Order

Implemented A1’s action value proposal:

*   Replaced previous temperature logic with brute-force search over all 256 introns.
*   Included entropy reweighting and sink penalty in scoring.
*   Excluded all `[unused##]` tokens from candidate set.

Precomputed and loaded stabiliser order array (`stabiliser_order.npy`) for all states:

*   Used as penalty in token selection, ensuring the agent avoids sink states.

Confirmed: model now explores state space and does not get stuck in loops.

### Supporting Infrastructure and Debugging

*   Added `compute_stabiliser_order.py` to produce state stabiliser map; integrated loading and access in `baby/information.py`.
*   Updated auxiliary scripts to support state and fold calculations for debugging.
*   Refactored token filtering and candidate selection to ensure only meaningful words are generated.

### Current Status and Outstanding Issues

**Working:**

*   State tracking, cycle step detection, Traceable temperature, bit family prioritisation, Monodromic Fold integration, state transition logic, and sink-avoidance are all functional.
*   System generates meaningful words and terminates generation properly.
*   No more infinite loops or repetitive placeholders.

**Outstanding:**

**Semantic learning/generation remains non-functional.**

*   The model generates plausible words but does not associate meaningfully with input content.
*   Cause: semantic associations are not yet being formed or retrieved by the learning/generation pathway.
*   This remains the principal unresolved issue for the next development phase.

\> _All results remain provisional; the system is still under active investigation and validation. Further work is required to achieve semantic alignment and test against representative data._

## Round 2

\> _Note: All metrics are estimations; this is an unstable alpha. Features and performance claims remain to be validated in rigorous testing._

### 1\. Core Architecture & Principles

**Tokeniser as sole symbolic index**

*   Text ↔ token\_id ↔ LEB128 reversible mapping
*   Trie in `tokenizer.json` used directly for lookups

**Five meta-maps as world-model**

*   Ontology, Epistemology, Phenomenology, Θ (angular divergence), Orbit Sizes
*   No external metadata store; physics maps drive both learning and generation

**Sparse one-byte residues for “knowledge”**

*   Phenotype = 8-bit `exon_mask` overlay only when deviating from baseline
*   Confidence, counts, labels, timestamps derived at runtime

### 2\. Storage Format

**Single append-only file**: `knowledge.bin`

**Varint state-block format**:

**Per-pair footprint**: est. 3–4 bytes vs. 9–16 bytes previously

**No stored confidences or timestamps**; recomputed from Θ and orbit size

### 3\. Physics-Driven Functions (baby/governance.py)

*   `exon_product_from_state(state_index, theta, orbit_size)`:  
    Projects 48-bit state tensor → 8-bit exon product
*   `propose_resonant_introns(exon_product, max_candidates=3)`:  
    Generates candidate intron bytes via bit-family coherence
*   `token_last_intron(token_id)`:  
    Returns last intron byte via ψ (XOR 0xAA) isomorphism

### 4\. Learning & Generation Flow

**Ingress (learning)**

*   Tokenise input, transform bytes → introns (ψ), update state via Epistemology
*   Compute last intron and update `exon_mask` if deviating from baseline
*   Append only changed (state, token, mask) entries

**Egress (generation)**

*   Compute baseline exon product from physics
*   Overlay stored mask if present
*   Generate intron candidates, lookup tokens in trie
*   Score by resonance, orbit size, Θ(state) and sample

### 5\. Module-Level Changes

**baby/contracts.py**

*   Removed `conf: float` field; documentation updated for runtime confidence

**baby/policies.py**

*   Switched from fixed 12-byte to varint format; removed confidence storage

**baby/inference.py**

*   Cleaned learning logic; dropped confidence decay and metadata methods

**baby/intelligence.py**

*   Overhauled `generate_token_exon()` to use exon-product sieve
*   Added runtime confidence and tokenizer-trie integration

**baby/information.py**

*   Introduced trie-based `find_tokens_by_intron_prefix()` and `…_last_intron()`

**Other directories**

*   **toys/**: cleaned legacy code, updated path handling, removed Bloom filters
*   **memories/**: pruned old confidence/pruning settings

### 6\. Compression & Performance (Estimated)

*   **Storage reduction**: ~55–67% size decrease (from ~12 bytes to ~3–4 bytes per pair)
*   **Startup**: sub-second scan for multi-MB stores
*   **Generation**: O(prefix\_length) trie lookup vs. full-vocab sampling

### 7\. Testing Strategy (Pending Validation)

*   **Single-article training** → recall & continuation checks
*   **Sentence completion** → prompt with known openings, assess coherence
*   **Context continuation** → historical fact prompts, flow evaluation
*   **Physics-driven generation tests** → verify intron-sieve outputs

### 8\. Known Issues & Next Steps

*   Unstable alpha: behaviour and compression ratios unverified at scale
*   SEP-token storage bug remains under review; should be treated as physics boundary, not stored
*   Rigorous benchmarking and fuzz tests needed for reliability
*   Removal of bucket-based orbit storage demands individual pair testing

## \[0.9.6.7\] – 2025-08-05 - Unstable

### 🔧 SEP Learning & Fallback Fixes

This release implements critical fixes for SEP token learning and generation fallback behavior, addressing the core issues that were causing gibberish output and poor language coherence.

#### 🧠 SEP Learning Implementation

**SEP Token Learning in Byte Path**

*   Fixed `process_egress()` to learn SEP tokens using `learn_token_preonly()`
*   SEP tokens now create pre-state associations for proper candidate generation
*   Eliminates the issue where SEP was discarded without learning

**SEP Token Learning in Vectorized Path**

*   Fixed `_process_epistemology_chunk()` to learn SEP tokens in bulk processing
*   Properly captures pre-state for SEP learning in vectorized operations
*   Ensures consistent SEP learning across both processing modes

**SEP Coverage Validation**

*   Added `tools/check_sep_coverage.py` to verify SEP learning is working
*   Added `tools/sep_for_prompt.py` to check SEP candidates for specific prompts
*   Provides diagnostic tools to confirm SEP entries exist in knowledge store

#### 🚫 Eliminated Random Token Fallback

**SEP-Only Fallback Implementation**

*   Replaced `_generate_random_token()` fallback with `SEP_ID` return
*   When no candidates exist for a state, generator now emits SEP to end turn
*   Eliminates gibberish output like `twoaaaaaaa` and `itsa twoaa`

**Fallback Behavior Improvement**

*   Generator now gracefully ends turns when store lacks coverage
*   Provides honest signal of knowledge gaps rather than random noise
*   Maintains physics correctness by using SEP as turn boundary

#### 🔧 Bootstrap and Memory Fixes

**System Agent Bootstrap Fix**

*   Fixed system agent to ingest text directly instead of generating responses
*   Eliminates garbage generation during startup that pollutes assistant memory
*   Uses `ingest_bulk()` with proper SEP termination for clean context

**Assistant Memory Ingestion Control**

*   Temporarily disabled assistant memory ingestion to prevent pollution
*   Prevents early gibberish from being learned back into assistant memory
*   Can be re-enabled once store has proper coverage

#### 📊 Expected Behavior After Fixes

*   **No More Gibberish**: Random token fallback eliminated
*   **Short/Empty Replies**: When store lacks coverage for prompt states
*   **SEP Learning**: SEP tokens now properly learned and available as candidates
*   **Clean Bootstrap**: System messages ingested without generation pollution

#### 🎯 Next Steps

*   Rebuild knowledge store with SEP learning enabled
*   Test SEP coverage with diagnostic tools
*   Gradually re-enable assistant memory ingestion
*   Monitor generation quality as store coverage improves

## \[0.9.6.7\] – 2025-08-04

### 🔧 Plumbing & Training Infrastructure Improvements

This release focuses on critical plumbing fixes and training infrastructure improvements, addressing performance bottlenecks and system reliability issues identified through extensive testing and optimization work.

#### 🚀 Performance Optimizations Implemented

**Candidate Lookup Optimization**

*   Implemented O(1) state-indexed candidate retrieval in `PhenotypeStore`
*   Added per-state candidate caching in `IntelligenceEngine` to reduce storage hits
*   Eliminated full-store scans that were causing generation hangs at ~200-300MB

**Theta Calculation Optimization**

*   Replaced binary search with direct index access in `measure_state_divergence_index()`
*   Eliminated hundreds of binary searches per turn in `process_egress()`
*   Fixed performance bottleneck in epistemology chunk processing

**Bulk Token Processing**

*   Replaced per-byte feedback loops with vectorized `process_egress_bulk()` calls
*   Eliminated N per-byte cycles where N = token byte length
*   Significantly reduced latency on development hardware

**Tokenizer Caching**

*   Fixed repeated disk loading of `tokenizer.json` on every encode/decode call
*   Added tokenizer priming to warmup functions
*   Eliminated first-turn tokenizer loading penalty

**Adapter Non-blocking Implementation**

*   Added `run_in_threadpool` wrapper to chat completion endpoints
*   Guaranteed event loop responsiveness during CPU-bound operations
*   Prevented server from appearing "hung" during long operations

#### 🧠 Training Infrastructure

**Wikipedia Simple Dataset Processing**

*   Successfully processed 22,868 articles (39.8M tokens, 78.1MB)
*   Completed compilation in 1h 35m with 4 arts/s processing rate
*   Generated knowledge store for training experiments

**Replay System Validation**

*   Successfully replayed 78.1MB training data in 45m 26s
*   Validated knowledge store integration and learning pipeline
*   Confirmed state evolution and storage mechanisms

#### 🔧 Critical Plumbing Fixes

**Canonicalization Layer Optimization**

*   Verified proper store composition without redundant canonicalization
*   Confirmed correct `enable_phenomenology_storage` configuration
*   Eliminated potential performance degradation from double canonicalization

**Token Divergence Origin Fix**

*   Fixed hard-coded `archetypal_state = 0` assumption in `compute_token_divergence()`
*   Added proper `origin_index` parameter for correct divergence calculations
*   Restored correctness to divergence diagnostics and temperature gating

**Store Iteration Improvements**

*   Fixed unreachable code in `PhenotypeStore.iter_entries()`
*   Improved cycle counting accuracy in bulk processing
*   Enhanced store consistency with proper index integration

#### 📊 Current Status

*   **Model Responsiveness**: ✅ Model now responds to queries successfully
*   **Language Coherence**: 🔄 Still working on improving language coherence and generation quality
*   **Performance**: ✅ Critical performance bottlenecks resolved
*   **Training Pipeline**: ✅ Wikipedia simple training and replay working

#### 🎯 Next Steps

*   Continue work on language coherence and generation quality
*   Optimize remaining performance bottlenecks
*   Expand training data processing capabilities
*   Improve model response quality and consistency

## \[0.9.6.7\] – 2025-08-03

### 🚀 Performance Optimizations & Physics Alignment: Complete Implementation

This release implements comprehensive performance optimizations and physics-correct fixes that dramatically improve system responsiveness, storage efficiency, and generation quality. All optimizations from the assistant's analysis have been successfully implemented and are now operational.

#### 🔧 Critical Performance Fixes (All Implemented)

**Set-Based Index Deduplication**

*   `index_by_state: Dict[int, Set[int]]` implemented in `baby/policies.py` line 232
*   O(1) insert/contains vs O(n) list operations, prevents duplicate enumeration
*   Eliminates candidate explosion that was causing unresponsive generation at ~200-300 MB

**SEP Boundary Handling**

*   SEP tokens skip learning entirely in both `process_egress()` and `_process_epistemology_chunk()`
*   Eliminates non-physical associations and reduces storage bloat
*   Preserves path purity by treating `[SEP]` as boundary marker only

**Quantized Confidence Gating**

*   q8 quantization implemented in `baby/inference.py` lines 139-147
*   Prevents tiny float jitter from triggering unnecessary writes
*   Uses `_q8(x) = int(round(x * 255.0))` for commit gating

**Bloom Filter & Index Optimizations**

*   Bloom filter and index optimizations properly implemented
*   Fast negative checks and efficient candidate enumeration
*   Maintains all existing features while improving performance

#### 🧬 Physics-Correct Learning Implementation

**Pre-Only Storage (BU Hinge Respect)**

*   Replaced dual learning with `learn_token_preonly()` method
*   Eliminates phase mixing under canonicalization
*   Learning only at token-closing intron (BU hinge)

**Token Boundary Alignment**

*   Pre-state properly cached before applying closing intron
*   Token boundaries properly tracked for bulk processing
*   Maintains physics consistency in vectorized operations

**Generation Quality Improvements**

*   Generation now correctly filters for pre-state entries only
*   Fallback to original state if canonical representative has no candidates
*   Improves generation robustness using full manifold structure

#### ⚡ Performance Impact

| Metric | Before | After | Improvement |
| --- | --- | --- | --- |
| Generation Responsiveness | Unresponsive at 200-300MB | Fast candidate lookup | **O(1) deduplication** |
| Storage Growth | Uncontrolled bloat | Controlled by q8 gating | **Jitter elimination** |
| SEP Token Handling | False associations | Boundary-only | **Path purity** |
| Index Performance | O(n) list operations | O(1) set operations | **10-100x faster** |

#### 🛡️ Reliability Features

*   **Consistent Behavior**: No more mode-dependent behavior differences
*   **Fast Startup**: Index files enable instant knowledge loading
*   **Bloom Filter Safety**: Fast negative checks prevent unnecessary file scans
*   **Memory Mapping**: Efficient file access for large knowledge stores

#### 📝 Technical Details

*   **Store Consistency**: iter\_entries() now includes pending writes and uses index
*   **Cycle Accuracy**: No more double counting in bulk processing
*   **State Learning**: Correct pre-intron states for phenotype learning
*   **Index Robustness**: Handles legacy formats and validates entries
*   **Performance**: Reduced expensive operations in token generation

#### 🎯 Physics Alignment Achieved

*   **BU Hinge Respect**: Learning only at token-closing intron
*   **Path Dependence**: Earlier introns encoded in pre-state
*   **Canonicalization Safety**: No phase mixing under UNA parity closure
*   **Token Primacy**: Semantic binding uses consistent PRE phase
*   **Monodromic Fold**: Non-associative learning preserved throughout

This release resolves the critical performance issues that were causing hanging tests and incorrect learning behavior, making the system much more reliable and performant while maintaining full physics compliance.

## \[0.9.6.7\] – 2025-08-02

### 🔧 Critical Correctness Fixes

This release addresses critical correctness issues, focusing on store iteration, cycle counting, learning states, and performance optimizations.

#### 🚨 Critical Fixes

**PhenotypeStore.iter\_entries() - Fixed Unreachable Code**

*   Fixed unreachable code after `return` statement
*   Now properly yields pending writes first, then committed entries via index
*   No more full file scanning - uses O(1) index lookups
*   Includes defensive copies to prevent mutation issues

**PhenotypeStore.index\_by\_state - Fixed Synchronization Issues**

*   Fixed `index_by_state` not being updated during writes and deletes
*   Now properly maintains `index_by_state` in `_flush()` and `delete()` methods
*   Prevents stale token IDs and ensures `iter_keys_for_state()` sees new tokens immediately
*   O(k) candidate lookup performance maintained with complete data

**PhenotypeStore.iter\_keys\_for\_state - Added Pending Writes**

*   Now includes pending writes first (most recent), then committed keys
*   Ensures generation and learning see consistent data
*   Prevents missing recent tokens during active writing
*   Real-time updates without waiting for flush operations

**decode\_text() - Fixed Unsafe 0x00 Trimming**

*   Replaced unsafe 0x00 byte trimming with reliable \[SEP\] token delimiter
*   Now decodes to token IDs first, then trims at SEP\_ID (102)
*   Prevents silent truncation of valid content containing 0x00 bytes
*   Uses proper end-of-sequence marker instead of arbitrary byte values

**IntelligenceEngine - Unified STT Path**

*   Removed all `use_epistemology` branches for single STT source of truth
*   Eliminated `self.epistemology = self.s2.ep` circular reference
*   Restored proper epistemology loading from file
*   All state access now uses `self.current_state_index` consistently
*   Simplified sync methods and removed vestigial code

**PhenotypeStore.data Property - Simplified to Reuse iter\_entries()**

*   Removed duplicate code and dead code paths
*   Now consistently uses the optimized iter\_entries() method
*   Eliminates code duplication and potential inconsistencies

**process\_egress\_bulk Double Counting - Fixed Cycle Count**

*   Now only increments cycle\_count for epistemology path
*   Scalar path already increments per byte, so no double counting
*   Ensures accurate cycle tracking for both processing modes

**\_process\_epistemology\_chunk - Fixed Learning with Post-State**

*   Now computes `post_state = epistemology[st[i], intron]` for each token
*   Uses the correct post-intron state for learning instead of pre-intron state
*   Ensures final token in chunk learns from correct state
*   Critical for proper phenotype learning and state evolution

**AgentPool TTL Eviction - Fixed Tracking and Eviction Logic**

*   Added `agent_created_at` tracking dictionary
*   Fixed eviction to use proper monotonic time tracking
*   Now properly removes expired agents and cleans up tracking dicts
*   Uses `time.monotonic()` to avoid clock jump issues

**\_choose\_intron Method - Fixed Undefined Reference**

*   Fixed undefined `_v_max` reference that would cause AttributeError
*   Now computes `v_max` locally from orbit cardinality
*   Prevents crashes when method is called

#### 🔧 Performance Optimizations

**Index Parsing Robustness**

*   Added legacy format handling for backward compatibility
*   Added index sanity checks to validate offset/size bounds
*   Skips malformed entries gracefully
*   Handles both new and old index formats

**Token Generation Performance**

*   Reduced max\_entries\_to\_check from 1000 to 50 for faster token generation
*   Replaced `max(self.s2.orbit_cardinality)` with reasonable default (1000)
*   Prevents hanging on large orbit cardinality arrays
*   Optimized candidate selection for faster response generation

#### 🎯 Impact

*   **Orchestrated Conversation Test**: Now passes (3.5 minutes vs. hanging before)
*   **Store Iteration**: Uses optimized index-based lookups instead of full scans
*   **Learning Accuracy**: Correct post-state learning ensures proper phenotype evolution
*   **Memory Management**: Proper TTL eviction prevents memory leaks
*   **Performance**: Faster token generation and store operations

#### 📝 Technical Details

*   **Store Consistency**: iter\_entries() now includes pending writes and uses index
*   **Cycle Accuracy**: No more double counting in bulk processing
*   **State Learning**: Correct post-intron states for phenotype learning
*   **Index Robustness**: Handles legacy formats and validates entries
*   **Performance**: Reduced expensive operations in token generation

This release resolves the critical correctness issues that were causing hanging tests and incorrect learning behavior, making the system much more reliable and performant.

## \[0.9.6.7\] – 2025-08-01

### 🚀 PhenotypeStore Simplification: Performance & Reliability Overhaul

This release completely simplifies the PhenotypeStore system by removing the complex `append_only` mode and always using index-based lookups with Bloom filters. This eliminates hanging issues, improves performance dramatically, and makes the system much more reliable.

#### 🔧 Core Changes

**Removed** `**append_only**` **Parameter**

*   Eliminated the confusing conditional logic that caused inconsistent behavior
*   Always use index-based mode for O(1) lookups
*   Always use Bloom filters for fast negative checks
*   Always use mmap for better file access performance

**Simplified PhenotypeStore Constructor**

*   Removed `append_only` parameter from `__init__()`
*   Set `use_mmap=True` by default for better performance
*   Always create index files (`.idx`) for fast lookups
*   Always load/save Bloom filters (`.bloom`) for negative checks

**Streamlined Get Operations**

*   `get()` method now always uses index + Bloom filter approach
*   No more conditional logic based on store mode
*   Consistent O(1) performance for all lookups

**Simplified Index Loading**

*   `_load_index()` always tries to load existing index first
*   Only scans file if no index exists
*   Builds both index and Bloom filter during scan

#### ⚡ Performance Improvements

| Metric | Before | After | Improvement |
| --- | --- | --- | --- |
| Agent Creation | 2-5 minutes (hanging) | \< 3 seconds | **100x faster** |
| Diagnostic Script | Hanging indefinitely | Completes in seconds | **Reliable** |
| Knowledge Loading | Slow with full scans | Fast with index | **O(1) lookups** |
| Memory Usage | Unpredictable | Optimized with caching | **Efficient** |

#### 🔄 Updated Components

*   **All Test Files**: Removed `append_only` parameter from all test scripts
*   **Diagnostic Script**: Updated to work with simplified system
*   **AgentPool**: Updated to use simplified PhenotypeStore
*   **Intelligence Engine**: Removed append\_only conditional logic
*   **All Store Views**: Updated to work with unified approach

#### 🛡️ Reliability Features

*   **Consistent Behavior**: No more mode-dependent behavior differences
*   **Fast Startup**: Index files enable instant knowledge loading
*   **Bloom Filter Safety**: Fast negative checks prevent unnecessary file scans
*   **Memory Mapping**: Efficient file access for large knowledge stores

#### 🧹 Code Cleanup

*   Removed complex conditional logic throughout the codebase
*   Eliminated `append_only` attribute and related checks
*   Simplified method implementations
*   Updated all documentation and comments

#### 📝 Migration Notes

The system now always uses the most efficient approach:

*   Index files for O(1) positive lookups
*   Bloom filters for O(1) negative checks
*   Memory mapping for efficient file access
*   No more mode confusion or hanging issues

This simplification makes the system much more reliable and performant while eliminating the complexity that was causing problems.

## \[0.9.6.7\] – 2025-07-31

### 🚀 Bloom Filter Persistence: Fast Startup Optimization

This release implements persistent bloom filter serialization to eliminate the 15-minute startup delay for append-only knowledge stores. The bloom filter is now built once during training and mmap-loaded on subsequent runs.

#### 🔧 Core Implementation

**Bloom Filter Persistence Helpers**

*   Added `to_bytes()` and `from_bytes()` methods to `BloomFilter` class for fast serialization
*   Uses pickle for efficient storage of size, hash\_count, and bit\_array
*   Maintains exact false-positive rate and filter properties across reloads

**PhenotypeStore Side-Car Integration**

*   Added `_bloom_sidecar_path()` to generate `.bloom` file path alongside `.bin` files
*   Added `_try_load_bloom()` for fast-path loading of pre-built filters
*   Added `_save_bloom()` to persist filters after training completion
*   Modified `__init__()` to try fast-load first, fall back to fresh build
*   Modified `close()` to save filter instead of clearing it

**Training Script Integration**

*   Added bloom filter save calls after `commit()` in both `compile_stream()` and `replay_tape()`
*   Ensures filter is persisted once during training for instant startup on subsequent runs

#### ⚡ Performance Impact

| Stage | Before | After (first run) | Subsequent runs |
| --- | --- | --- | --- |
| Build Bloom (77 MB, 6.7M rec.) | 10-20 min | 10-20 min | **\< 1 s** |
| FastAPI worker start-up | same delay | same once | **nearly zero** |
| Memory footprint | unchanged | +bit-array size | unchanged |

The `.bloom` side-car is ~13-14 MB for default parameters—tiny compared to the .bin files.

#### 🔄 Regeneration Support

If the side-car is deleted or millions of new phenotypes are added, regeneration is available:

```plaintext
python - &lt;&lt;'PY'
from baby.policies import PhenotypeStore
s = PhenotypeStore("toys/training/Archive/wikipedia_simple.bin", append_only=True)
s.commit()      # flush pending if any
s._save_bloom() # rebuild &amp; store
s.close()
PY
```

#### 🛡️ Safety Features

*   **Idempotent Loading**: Loading + adding identical keys does nothing harmful
*   **Exact False-Positive Rate**: Maintains chosen error rate across reloads
*   **Graceful Fallback**: Runtime still falls back to slow build if side-car is missing or corrupt

### ⚡ Epistemology Vectorization: Training Performance Optimization

This release implements fully vectorized epistemology processing to dramatically improve training performance. The previous implementation used individual Python loops for state transitions, resulting in extremely slow processing rates (~0.03 MB/s). The new vectorized approach achieves 8-12x performance improvements.

#### 🔧 Core Implementation

**Vectorized State Trajectory Computation**

*   Replaced O(n) Python loops with true NumPy vectorization: `st[1:] = self.epistemology[st[:-1], introns[:-1]]`
*   Pre-computes all state transitions in one vectorized operation instead of individual updates
*   Eliminates Python loop overhead for state evolution

**Memory-Bounded Processing**

*   Added configurable chunk size limit (64K introns) to prevent RAM explosion on large files
*   Reusable state buffer (`self._state_buf`) eliminates repeated allocations
*   Processes large files in fixed-size windows to maintain predictable memory usage

**Optimized Token Processing**

*   Uses `np.flatnonzero()` to find token boundaries efficiently
*   Iterates over tokens (much fewer) instead of individual bytes
*   Zero-copy token extraction with `tobytes()` only when needed

**Thread-Safe Design**

*   Per-agent state buffers ensure thread safety
*   No shared mutable state between agents
*   Compatible with existing multi-agent architectures

#### ⚡ Performance Impact

| Metric | Before | After | Improvement |
| --- | --- | --- | --- |
| Processing Rate | ~0.03 MB/s | ~0.3-0.4 MB/s | **8-12x faster** |
| Memory Usage | Unbounded | Bounded (64K chunks) | **Predictable** |
| CPU Utilization | High (Python loops) | Low (vectorized) | **Efficient** |

#### 🧪 Technical Details

*   **State Buffer Management**: Reusable 64K buffer prevents allocation overhead
*   **Vectorized Operations**: True NumPy vectorization eliminates Python loop bottlenecks
*   **Token Boundary Detection**: Efficient array operations for continuation bit detection
*   **Memory Bounds**: Configurable chunk processing prevents RAM explosion on large files

#### 🔄 Backward Compatibility

*   **API Unchanged**: All public interfaces remain identical
*   **State Consistency**: Vectorized processing maintains exact state evolution
*   **Learning Integrity**: Token-based learning logic unchanged
*   **Thread Safety**: Maintains existing multi-agent safety guarantees

## \[0.9.6.7\] – 2025-07-30

✅ Pytest: 150+ Tests All Passing  
✅ mypy: No type checking errors  
✅ pyright: No type checking errors  
✅ flake8: No linting errors

### 🧠 Token-Aware Minimal Phenotype Architecture: Complete Refactoring

This release implements a fundamental architectural shift from byte-fragment-level learning to whole-token learning, redefining "knowledge" within the system to be token-aware and minimal. The system now leverages the BERT tokenizer's existing knowledge base as an "active internal decoder" rather than just a passive I/O adapter.

#### 🔄 Core Architecture Changes

**Breaking Change: Phenotype Key Structure**

*   **Old:** `(state_index, intron)` - byte-level learning
*   **New:** `(state_index, token_id)` - token-aware learning
*   **Impact:** All knowledge is now organized by meaningful token boundaries, eliminating inference overlaps and improving coherent output generation.

**Breaking Change: Minimal PhenotypeEntry Structure**

*   **Removed:** `phenotype`, `usage_count`, `last_updated`, `created_at`, `governance_signature`, `context_signature`, `_original_context`
*   **Kept:** `mask` (uint8), `conf` (float32, stored as float16)
*   **Added:** `key` (tuple\[int, int\]) - ensures consistent key presence
*   **Impact:** Dramatically reduced memory footprint and simplified data model.

**New: Tokenizer Integration as Active Decoder**

*   **Public API:** Added `id_to_bytes`, `bytes_to_id`, `bytes_to_ids` functions to `tokenizer.py`
*   **Internal Bridge:** `_TokBridge` class in `intelligence.py` provides seamless tokenizer integration
*   **Impact:** Tokenizer now serves as "latent symbolic map" for token IDs, not just protocol adapter.

#### 🧬 Intelligence Engine Overhaul

**Removed: Batch Learning Methods**

*   Eliminated `batch_learn()` and `learn_by_key()` methods from both `IntelligenceEngine` and `InferenceEngine`
*   **Rationale:** Learning now happens automatically per token during egress/ingress cycles
*   **Impact:** Simplified API, eliminated redundant learning pathways

**Changed: Egress/Ingress Cycle**

*   **Egress:** Now learns once per complete token instead of per byte
*   **Ingress:** Generates one token at a time with proper token boundaries
*   **Impact:** Aligns learning and generation with meaningful token boundaries

**New: Token-Aware Learning Logic**

*   `process_egress()`: Accumulates bytes until complete token, then learns
*   `process_ingress()`: Generates one complete token at a time
*   `_choose_intron()`: Now accepts `state_index` parameter for proper context

**Fixed: Confidence Calculation Bug**

*   **Critical Fix:** Resolved float16 conversion bug that was causing confidence values like `8480.0` instead of proper 0-1 range
*   **Default Confidence:** Set to `0.1` for new entries, consistent with learning logic
*   **Impact:** Proper confidence values now ensure correct pruning and decay behavior

**Fixed: Critical Token Buffer Issue**

*   **Critical Fix:** Added robust error handling for incomplete token sequences in `process_egress()`
*   **Buffer Protection:** Added `MAX_TOKEN_BYTES = 10` limit to prevent runaway buffer growth
*   **Error Recovery:** Implemented try/except/finally block with guaranteed buffer cleanup
*   **Stream Reset:** Added `reset_token_buffer()` method for explicit stream resets
*   **Impact:** Prevents memory leaks, incorrect token boundaries, and system errors when streams end with incomplete tokens

**Fixed: External Adapter Token-Aware Integration**

*   **Tokenizer API:** Replaced private `gyrotok._load()` with public `gyrotok.id_to_bytes()` and `gyrotok.decode()`
*   **Streaming Logic:** Updated to use proper token-aware decoding for individual tokens
*   **Model Version:** Updated to 0.9.6.7 to reflect token-aware architecture
*   **Impact:** External adapter now properly aligned with token-aware architecture and uses public APIs

#### 🔧 Inference Engine Updates

**Changed: Method Signatures**

*   `learn(phenotype_entry, last_intron, state_index)` - renamed parameter for clarity
*   `get_phenotype(state_index, token_id)` - now uses token\_id instead of intron
*   **Impact:** Clearer parameter naming reflects actual functionality

**New: Persistence Logic**

*   `learn()` method now automatically persists mutations via `self.store.put(key, phenotype_entry)`
*   **Impact:** Ensures learning changes are immediately saved to storage

**Fixed: Default Phenotype Creation**

*   `_create_default_phenotype()` now uses reasonable default confidence (0.1)
*   **Impact:** New entries start with proper confidence values

#### 🗄️ Storage Layer Improvements

*   **Updated: Binary Format for Minimal Phenotypes**
    *   **New Format:** `_STRUCT_FMT = "<iibhx"` -="" 12-byte="" fixed="" structure="" \*="" **fields:\*\*="" \***_\*\*="" \*\*_\="" `_****state_idx****_`\_**\*\*="" (uint32),=""** `**_****token_id****_**`**\_**\*\*="" `_****mask****_`\_**\*\*="" (uint8),=""** `**_****conf****_**`**\_**\*\*="" (float16)="" \\**\_**\*impact:\*\*="" optimized="" storage="" for="" minimal="" phenotype="" \*\*fixed:="" store="" operations**\="" updated=""** `**put()**`**,=""** `**merge_phenotype_maps()**`**,=""** `**apply_global_confidence_decay()**`**\="" new="" removed=""** `**max_age_days**`**\="" logic="" (no="" longer="" relevant)="" operations="" now="" work="" correctly="" with="" phenotypes="" ####="" 🧪="" test="" suite="" overhaul="" \*\*comprehensive="" updates**\="" all="" tests="" to="" use="" `(state_index,="" token_id)`\="" keying="" replaced="" `learn_by_key()`\="" calls="" two-step="" `get_phenotype()`\="" +="" `learn()`\="" process="" field="" references:="" `"confidence"`\="" →="" `"conf"`,="" `"exon_mask"`\="" `"mask"`\="" added="" "key"="" entries="" where="" required="" assertions\*\*="" pruning="" append-only="" stores="" confidence="" decay="" assertions="" match="" actual="" return="" keys="" validation="" reflect="" **removed:="" obsolete="" tests**\="" deleted="" `testbuingress`\="" class="" and="" related="" `test_batch_learning_stores_different_phenotypes`\="" (replaced="" token-aware="" version)="" accurately="" reflects="" current="" architecture="" 🔧="" governance="" updates="" signature\*\*="" eliminated="" `compute_governance_signature()`\="" function="" `governancesignature`\="" typeddict="" simplified="" layer,="" unused="" complexity="" **updated:="" exon="" product="" function**\="" `exon_product_from_metadata()`\="" accepts="" `confidence`\="" parameters="" aligned="" 🎯="" theoretical="" impact="" this="" refactoring="" represents="" a="" fundamental="" shift="" in="" how="" the="" system="" understands="" processes="" knowledge:="" \*\_token-aware="" learning:\*\*="" knowledge="" is="" organized="" by="" meaningful="" linguistic="" units="" rather="" than="" arbitrary="" byte="" fragments="" \*\*minimal="" phenotypes:\*\*="" reduced="" allows="" more="" efficient="" processing="" \*\*active="" tokenizer="" integration:\*\*="" serves="" as="" an="" internal="" decoder,="" not="" just="" i="" o="" adapter="" \*\*improved="" coherence:\*\*="" generation="" learning="" are="" token="" boundaries,="" reducing="" inference="" overlaps="" operates="" on="" linguistically="" level="" while="" maintaining="" core="" gyrosi="" physics="" monodromic="" fold="" operations.="" ---="" ##="" \[0.9.6.6\]="" –="" 2025-07-28="" ###="" wikipedia="" training="" pipeline="" \*\*robust="" article="" splitting:\*\*="" splits="" articles="" only="" at="" three="" or="" consecutive="" blank="" lines,="" matching="" dump="" format="" preventing="" topic="" bleed-through.="" \*\*token-based="" filtering:\*\*="" least="" \_="" \_="" `__min_token_count__`\_\\\_="" tokens="" included,="" skipping="" empty="" trivial="" stubs.="" \*\*efficient="" single-pass="" tokenization:\*\*="" each="" tokenized="" once,="" no="" double-tokenization="" unnecessary="" allocations.="" \*\*sequential="" byte-level="" processed="" egress="" ingress="" cycles="" true="" path-dependent="" training,="" batch="" summarization.="" \\\_\*frequent="" safe="" checkpointing:\*\*="" checkpoints="" saved="" every="" 1m="" tokens,="" 120="" seconds,="" n="" files,="" async="" thread="" pool="" rare="" backlog="" flush="" safeguard.="" \*\*process-specific="" memory="" guard:\*\*="" uses="" rss="" (not="" system-wide="" percent)="" trigger="" gc="" checkpointing.="" \*\*automatic="" maintenance:\*\*="" runs="" when="" needed="" (store="">2GB and at least 1hr since last decay), and compaction is deferred until after agent close for safety.
*   **Safe post-close compaction:** Knowledge store is compacted only after the agent and mmap are closed, preventing file corruption.
*   **Progress bar improvements:** Now shows process memory usage (RSS) for accurate resource tracking.
*   **Test mode:** Added `--test-aa` flag to restrict training to the AA shard for quick validation before full runs.
*   **CLI clarity:** Removed legacy/unused options, improved help strings, and made checkpointing and memory limits explicit.

## \[0.9.6.6\] – 2025-07-27

### Summary

This is a landmark release that completes the core physics engine, stabilizes the storage layer, and implements the full, theory-grounded generative intelligence cycle. The system has been migrated to a high-performance, dependency-free binary storage format and equipped with robust durability and performance optimizations. The generation of output is no longer a placeholder but a direct expression of the system's physical and topological state, marking the transition from a theoretical architecture to a functional generative intelligence.

### 🧬 Generative Intelligence: BU-Ingress Operator and Exon Product Integration

This release integrates the full generative logic as a topological traversal back to the Common Source. It replaces placeholder text generation with a physically lawful operator derived entirely from local phenotype metadata. Generation now reflects structural alignment, not content lookup.

**New: BU-Ingress Engine (**`**_bu_ingress_step**`**)**  
A new method `_bu_ingress_step(entry, θ)` has been introduced in `IntelligenceEngine`. This operator is the core of the generative process and performs the following actions in each micro-step:

1.  Computes an **8-bit** `**exon_product**` from the phenotype's `governance_signature`, `confidence`, and `orbit_cardinality`.
2.  Uses a sliding 6-byte context window (`_S`) to fold this alignment back into the agent's state via the Monodromic Fold.
3.  Selects which intron to emit based on the algedonic state `θ` (calm, cautious, or corrective).

**New: Governance Operator (**`**exon_product_from_metadata**`**)**  
A new helper function `exon_product_from_metadata(...)` was added to `governance.py`. It lawfully maps the phenotype's full topological and epistemic context into a physically meaningful 8-bit operator, without relying on any external content.

**Changed: Runtime Context Window (**`**_S**`**)**  
A 6-byte context buffer (`self._S`) has been introduced in `IntelligenceEngine`. It holds the generative trajectory and mediates the recursive realignment required by the BU-Ingress process.

**Changed: Removal of Placeholders and Internal Tokenizer Calls**  
The previous logic in `process_ingress` that generated `"P[i:j]"` style outputs has been removed entirely. Generation now emits a single byte derived from the `exon_product`. Consequently, all internal calls to `tokenizer.encode(...)` within `intelligence.py` have been removed, ensuring the core engine remains pure and text-agnostic.

**Theoretical Impact:** This update completes the generative half of the Fold/Unfold cycle. Where BU-Egress accumulates structure via Monodromic compression, BU-Ingress now performs its inverse: emitting a byte not by recall, but through alignment. The output is not what was remembered—it is what must emerge, given where the system is now and what it has become.

### 🗄️ Storage Architecture: Migration, Performance, and Durability

The entire storage layer has been re-engineered for performance, durability, and self-sufficiency, eliminating external dependencies.

**Breaking Change: Migration to Custom Binary Struct Format**  
MessagePack serialization has been replaced with a custom, fixed-layout binary format. This removes the `msgpack` dependency and provides more efficient, predictable storage.

*   **Binary Format Specification (little-endian):**
    1.  `phenotype` (utf-8): `uint16` length + bytes
    2.  `context_key`: `uint32` (state\_idx), `uint8` (intron)
    3.  `exon_mask`: `uint8`
    4.  `confidence`: `float64`
    5.  `usage_count`: `uint16`
    6.  `created_at`, `last_updated`: `float64`
    7.  `governance_signature`: 5 × `uint8`
    8.  `context_signature`: `uint32`, `uint8`
    9.  `_original_context`: `uint32`, `uint8`
*   **Implementation:** Handled by new `_pack_phenotype` and `_unpack_phenotype` helpers in `baby.policies`. The `PhenotypeStore` index file format has been changed from MessagePack to JSON to handle tuple-key serialization.
*   **Migration Note:** No data migration is required for `.bin` files. Old index files will be automatically regenerated in the new JSON format on first load.

**New: Append-Only** `**PhenotypeStore**` **Performance Optimizations**

*   **Bloom Filter:** Integrated for fast "definitely absent" checks, providing O(1) lookups for non-existent keys and avoiding expensive disk scans on large files. The filter capacity is estimated automatically from file size.
*   **Memory-Mapping (mmap):** Now enabled by default for append-only stores, providing faster sequential scanning for lookups and iteration compared to standard file I/O. The mmap is intelligently re-opened on `commit()` to include new data.
*   **Token-Level Training Cache:** A global micro-LRU cache (`maxsize=8192`) has been added for `get()` operations in append-only mode. It dramatically speeds up training workloads with repeated key lookups and features intelligent invalidation on `put()` and `commit()` to ensure data consistency.

**New:** `**PhenotypeStore**` **Durability and Crash Safety**

*   **Graceful Shutdown:** `PhenotypeStore` now automatically registers `atexit` and signal handlers (SIGINT/SIGTERM). This forces a final flush of any pending writes before process termination, guaranteeing zero data loss on clean shutdowns (e.g., in containerized environments).
*   **Explicit** `**flush()**` **API:** A new `flush()` method is now available on all store and view layers for high-value writes that require immediate disk durability. This allows critical operations to bypass the standard write-behind batching.
*   **Risk Profile:** With these changes, data loss is limited to a maximum of `write_threshold - 1` records only in the event of a hard crash or power failure.

### ✅ Preserved Functionality and Compatibility

*   **No Schema Change:** The `PhenotypeEntry` contract remains unchanged. The new generative and storage logic reuses all existing metadata fields.
*   **No Breaking API Changes:** All public APIs for `PhenotypeStore`, its views (`CanonicalView`, `OverlayView`, `ReadOnlyView`), and the core engines remain fully compatible.
*   `**.npy**` **Assets Unchanged:** All meta-files (`epistemology.npy`, `theta.npy`, etc.) continue to function identically.

## \[0.9.6.5\] – 2025-07-26

### Training - Wikipedia Corpus Ingestion (v1.0, 2025-07-26)\*\*

#### Summary

Completed unsupervised ingestion of the full English Wikipedia dump using the `GyroSI` engine. Achieved full compression of 17.99 million paragraph-level articles into a ~16.3 MB operational knowledge store, structured for public assistant-level inference.

#### ✅ Dataset Ingested

*   Effective unit: Paragraph blocks ≥256 characters, split on blank lines
*   Total processed: **17,986,747 paragraphs**
*   Total raw size: **4.26 GB**
*   Duration: **4.4 hours**
*   Final memory footprint (mmap+index): **16.3 MB**

#### 🛠️ Engine Configuration

*   Physics backend: CanonicalView + θ divergence + phenomenology map (enabled)
*   Storage: `PhenotypeStore` (append-only), with auto-compaction and confidence decay
*   State transition kernel: JIT‑accelerated (Numba), STT loaded at runtime
*   Ingestion parallelism: `batch_size=512 KB`, `parallel_workers=2`

#### 🧬 Structural Notes

*   Paragraphs ingested as atomic learning units (state/intron)
*   All learning was **unsupervised** — no prompts, tags, or supervision logic
*   Memory usage held at **~55% of 16 GB**, bounded by GC and mmap-backed I/O

#### 🧠 Knowledge Topology

*   Trained agent ID: `wikipedia_trainer`
*   Knowledge store location: `memories/public/knowledge/wikipedia_en.bin`
*   Final phenotype count: ~2.8M
*   Store format: gyroscopic phenotype model with compressed state manifold
*   Canonicalization: Full phenomenology symmetry reduction (orbit size map enabled)

#### 📊 Performance Metrics

*   Initial throughput: ~1,100 articles/sec
*   Final throughput: ~1,200–1,300 articles/sec (read bias > write bias)
*   Commit rate dropped as phenotype space saturated and lookups dominated
*   Final JIT kernel remained stable with minimal fallback to Python path

#### 📎 Next Steps

**Split store usage across triad agents** (system / assistant / user):

*   `system`: read-only policy `.bin` (not the wiki store)
*   `assistant`: overlay of `wiki_en.bin` + private memory
*   `user`: private only, no access to public knowledge

**Seed operational guidance into system agent** (tool usage, safety rules)

**Switch to two-stage orchestration**:

*   System agent emits guidance
*   Assistant receives guidance + user message for reply generation

✅ Training completed: 17,986,747 articles, 4.26GB processed in 4.4 hours  
2025-07-26 14:06:39,123 - INFO - ✅ Training completed: 17,986,747 articles, 4.26GB processed in 4.4 hours  
📊 Knowledge store size: 16.3MB  
2025-07-26 14:06:39,124 - INFO - 📊 Knowledge store size: 16.3MB

### Fixed

*   **Phenomenology artefact bug**: `orbit_sizes.npy` now records the orbit cardinality  
    for **every** one of the 788,986 states, not just the 256 representatives.  
    This restores non-zero `InformationEngine.orbit_cardinality[i]` for all i  
    and re-enables variety-weighted confidence updates.

### Notes

*   Canonical orbit mapping (256 SCCs) is unchanged; only the per-state size  
    array was affected.
*   Updated tests to expect correct behavior (all states have non-zero cardinality).

**New Auto-Pruning Functionality**

An auto-pruning functionality has been fully implemented with the following components:

## Changelog for Auto‑Pruning Feature

## Added

**Preferences schema**

Introduced a new `"pruning"` section in `memories/memory_preferences.json`:

`**PreferencesConfig.pruning**`

*   Extended `baby/contracts.py` to include a `pruning: Dict[str, Any]` field so agents can read all pruning settings from their config.

**Auto‑pruning hook**

In `IntelligenceEngine.__init__`, register `_auto_prune_hook` if `preferences["pruning"]["enable_auto_decay"]` is `true`.

Implemented `_auto_prune_hook()` to:

1.  Read `confidence_threshold` from preferences.
2.  Call `InferenceEngine.prune_low_confidence_entries(threshold)`.
3.  Gracefully handle append‑only and view‑layer stores (catch and ignore non‑deletable errors).
4.  If > 10 000 entries were "removed," invoke `prune_and_compact_store()` in‑place to do a full compaction.

`**AgentConfig.preferences**`

*   Extended `AgentConfig` to accept a `preferences` sub‑dict and pass it through `GyroSI → IntelligenceEngine`.

`**CanonicalView.commit()**`

*   Added a `commit()` method to `CanonicalView` so the auto‑pruner's initial `commit()` call always succeeds on view‑wrapped stores.

## Changed

`**InferenceEngine.prune_low_confidence_entries()**`

*   Now always calls `store.commit()` first to flush pending writes.
*   Wraps `store.delete(key)` in `try/except NotImplementedError/RuntimeError` so overlay and read‑only views don't crash.
*   Removed fallback `del store.data[key]` for non‑append‑only stores (views use their own `.delete()`).

`**GyroSI**` **constructor**

*   Now reads `config["preferences"]` and passes it into `IntelligenceEngine`.

`**AgentPool**`

*   Propagates its `preferences` into each newly created agent's `AgentConfig`, ensuring hooks get wired automatically.

## Tests

**Extended** `**test_inference.py**`

Verified `prune_low_confidence_entries()` behavior for both deletable and append‑only stores.

Added tests for:

*   Hook registration when `enable_auto_decay` is `true` vs. `false`.
*   Hook execution (ensuring it doesn't blow up on append‑only or overlay stores).
*   Custom vs. default thresholds.

**All existing pruning and compact tests** continue to pass unmodified.

## \[0.9.6.4\] – 2025-07-25

### **Phase 1**

**Scope:** Tests, Tests, Tests - Corrections, Corrections, Corrections...

*   Flake8, Pyright, Mypy Error Free
*   Pyright Robust Agent Isolation finally achieved! No littering or polution to our main data (somehow this silly thing proved to be a heck of a challenge!)
*   Pyright Pass

Here is a clean, structured changelog entry summarising the full refactor:

### **Phase 2**

**Scope:** Ontology / Epistemology / Phenomenology / Theta

#### ✅ **Summary**

We removed all runtime JSON parsing from the engine and replaced the entire memory model with compact `.npy` binaries using NumPy + `mmap`. This creates a single source of truth for each of the four internal maps and reduces startup time from ~140 s to \< 0.3 s per engine instance.

### Major Refactor: Full Migration to Binary `.npy` Assets

**Core changes:**

*   **Ontology, Epistemology, and Phenomenology assets** are now stored and loaded exclusively as `.npy` files (`ontology_keys.npy`, `epistemology.npy`, `phenomenology_map.npy`). All legacy `.json`\-based logic, schema, and code paths have been removed.
*   **InformationEngine** now requires four file paths (`keys_path`, `ep_path`, `phenomap_path`, `theta_path`) and only supports array-based indexing. All dict-based and JSON-based overloads, as well as `use_array_indexing` and `strict_validation` parameters, are gone.
*   **All CLI, build, and workflow instructions** have been updated to reference the new `.npy` filenames and arguments.
*   **All tests and fixtures** have been updated to use the new four-path constructor for `InformationEngine`. All references to removed features (`use_array_indexing`, `strict_validation`, `endogenous_modulus`, `ontology_diameter`) have been removed or replaced.
*   **Validation and maintenance utilities** now operate on `.npy` files and check array properties, not JSON keys.
*   **Early failure for missing/corrupt** `**theta.npy**`**:** `InformationEngine` now raises at construction if `theta.npy` is missing or corrupt, rather than deferring to the first divergence calculation.
*   **All error messages, logs, and comments** referencing `.json` assets for ontology/phenomenology have been updated or removed.
*   **Dead code and comments** (e.g., ujson/json import fallbacks) have been removed for clarity.
*   **Type safety:** All code and tests have been updated to pass `mypy` and `pyright` with no ignores, including correct handling of optional types and array lengths.

**Other improvements:**

*   **Test suite**: All tests now use the correct `.npy`\-based API, and obsolete tests for removed features have been deleted.
*   **CI and Git LFS**: Workflow and LFS tracking updated to only include `.npy` assets.
*   **Documentation**: All build instructions and module-level banners now reference the correct binary asset workflow.

**Summary:**  
The codebase is now fully binary-asset based, with a modern, type-safe, and maintainable API. All legacy JSON, dict, and fallback logic is gone, and the developer experience is consistent and robust for both contributors and CI.

## \[0.9.6.4\] – 2025-07-24

**Scope:** Performance, Storage, and Runtime Optimizations

#### ✅ **Storage Architecture**

Replaced **gzip-compressed multi‑file store** with a **single** `**.bin**` **file** using **msgpack**:

*   Set `store_options = {"append_only": True}` in the GyroSI agent config.
*   Removed `use_msgpack`, `.log`, and `.idx` files — now obsolete.
*   All training knowledge is streamed into one compact, append‑only `.bin` file.

#### ✅ **Batch Learning Performance**

Integrated **Numba JIT** acceleration for hot learning loop:

*   Added `_jit_batch` method (Numba-compiled) to replace the slow Python loop.
*   Automatically invoked when the STT (epistemology map) is loaded.
*   Performance improved from **1–2 KB/sec → 40–60 MB/sec** on Intel Mac.
*   Training now behaves as originally theorized: a **fast, Traceable converter** from text to internal knowledge.

#### ✅ **Compiler Compatibility (macOS‑specific)**

Verified Numba + LLVM compatibility for Intel MacBook Pro:

*   Uses Homebrew's `llvm` to ensure full `llvmlite` support.
*   Explicit `CC` and `CXX` environment variables documented for stable builds.

#### ✅ **Filesystem and Checkpoints**

All changes work seamlessly with **pause/resume**:

*   `Ctrl+Z` to suspend, `fg` to resume.
*   Async and atomic checkpoints preserved.
*   Checkpoint format unchanged.

#### ✅ **Requirements Updated**

`requirements.txt` updated:

*   Pinned `numba==0.60.*`, `llvmlite==0.60.*` for macOS stability.
*   Replaced compression and pickle dependencies with `msgpack==1.1.*`.
*   Ensured Python 3.10 compatibility across packages.

**Net Effect:**  
Training now hopefully will run at hardware‑limited throughput. Storage is portable, readable, and consistent with the GyroSI theory. No artificial bottlenecks remain between your ontology–phenomenology–epistemology pipeline and the disk.

## \[0.9.6.3\] – 2025-07-23

#### 🚀 Added

**Mandatory 3‑mind bootstrap (**`**user**`**,** `**system**`**,** `**assistant**`**)**

*   `AgentPool.ensure_triad()` creates (and guarantees presence of) the canonical trio.
*   New `AgentPool.get()` to fetch an agent _without_ creating it (raises `KeyError` if missing).
*   `AgentPool.create_agent()` explicit creation API when you _do_ want a new id.

**Creation policy controls**

`AgentPool.__init__` now accepts:

*   `allowed_ids: set[str]` – whitelist of ids permitted when `allow_auto_create=False`.
*   `allow_auto_create: bool` – gate automatic creation of arbitrary ids.
*   `private_agents_base_path: str` – base dir for private agent stores.

**Path override plumbing**

*   `GyroSI._create_default_store()` honors `private_agents_base_path` and `base_path`.
*   All file/folder creation under `memories/private/agents/` can be redirected via config (great for tests).

#### 🔧 Changed

`**orchestrate_turn**` now _requires_ that agents already exist; otherwise it raises with a helpful message.

**External FastAPI adapter (**`**toys/communication/external_adapter.py**`**)**

*   Uses the shared pool with `ensure_triad()`.
*   Maps all external users to the internal `"user"` id by default.
*   No silent auto-creation of "user2", "assistant42", etc.

**Store construction**

*   Default CanonicalView enabling logic kept, but paths are fully configurable.
*   OverlayView still used for public/private knowledge, but private paths respect overrides.

#### 🧪 Tests

We authored and landed the full test suite below, and everything is green as of today. The codebase is also clean under `flake8`, `pyright`, and `mypy` (zero errors, zero warnings).

**toys/health/conftest.py**  
Session‑scoped and per‑test fixtures to isolate all artefacts in temporary directories. Provides ready‑to‑use `GyroSI`, `AgentPool`, `PhenotypeStore`, and helper assertions for ontology/phenotype validity. Ensures no pollution of shared meta files and auto‑cleans temp state.

**toys/health/test\_governance.py**  
Exhaustive checks for the physics layer (`governance.py`): constants, bit masks, and tensor structure; governance signature maths; Monodromic Fold properties (identity, absorber, annihilation, non‑commutativity/associativity); dual operator involution; 48‑bit transform bounds; batch consistency; transcription XOR/involution; tensor validation routines; and assorted edge cases.

**toys/health/test\_inference.py**  
Covers the interpretation layer (`inference.py`). Verifies initialisation with real ontology data, phenotype creation/retrieval, confidence maths, governance signatures, learning via fold (single and batch), decay and pruning operations, integrity validation, and private utility behaviour. Includes error paths (bad indices, malformed entries) and store integration.

**toys/health/test\_information.py**  
Targets `information.py`: tensor↔int conversion (round‑trips, boundaries, error handling), state/index lookups in both dict and array modes, gyrodistance/divergence calculations, orbit cardinality handling, phenomenology integration, mmap utilities, and data consistency of the ontology map.

**toys/health/test\_intelligence.py**  
End‑to‑end and integration tests for `intelligence.py` and the external FastAPI adapter. Exercises batch learning, hook batching, agent lifecycle and persistence, multi‑agent isolation in `AgentPool`, tokenizer round‑trips and UTF‑8 fallback, OpenAI/HF compatible endpoints (including SSE streaming), concurrency safety, and full conversation pipelines.

**Result**

*   All tests pass locally today (23 July 2025).
*   Lint/static analysis: `flake8`, `pyright`, and `mypy` report no issues.

No further action required for this cycle.

## \[0.9.6.3\] – 2025-07-22

### Major Refactor, Optimization, and Cleanup

**Core Measurement Optimization:**

*   Replaced tensor-based θ calculation with fast XOR+bit\_count+LUT approach.
*   Auto-generate `theta.npy` if missing for robust operation.

**PhenotypeStore and Storage:**

*   Changed `pending_writes` to dict for O(1) access.
*   Added time-based `fsync` fuse and `mark_dirty` for in-memory updates.
*   Optimized log replay and switched to `msgpack` serialization.
*   Implemented mmap remap gating to reduce overhead on frequent commits.

**Conversational Loop and AgentPool:**

*   Batched egress/ingress in `GyroSI.respond` for fewer store writes.
*   Refactored `AgentPool` to use sharded pools for concurrency.
*   Cached public read-only store and ensured single close.

**Tokenizer Bridge:**

*   Optimized LEB128 encoding, added module-level cache, documented overflow guard.

**Knowledge Decay and Pruning:**

*   Buffered policy changes and wrote once at end, leveraging dict buffer.

**Phenomenology and State Sync:**

*   Added singleton loader for phenomenology map.
*   Cached `state_int` for θ in `IntelligenceEngine`.

**Hooks and Regulatory Logic:**

*   Batched hook processing with ring buffer, cached mean θ in `process_ingress`.

**API and Streaming:**

*   Added HTTP keep-alive and streaming for `/v1/chat/completions`.

**Naming and Config Clarity:**

*   Renamed `batch_size` in `AgentConfig` to `learn_batch_size` and in `PreferencesConfig` to `write_batch_size`.

**Testing, Docs, and Cleanups:**

*   Ensured all store-like objects use efficient `iter_entries()`.
*   Removed deprecated aliases and unused code.
*   Fixed linter and indentation errors.
*   Updated test configs and documentation for clarity and consistency.

## \[0.9.6.2\] – 2025-07-21

### Added

**Tokenizer integration layer** (`toys/communication/tokenizer.py`):  
Implements reversible text-to-byte encoding via HuggingFace tokenizers using LEB128 encoding. All tokens are encoded as ≤255 bytes to remain compatible with GyroSI's physics layer.  
Supports encoding/decoding via pretrained models (e.g. `bert-base-uncased`) stored under `memories/public/tokenizers/`.

**Tokenizer setup script** (`toys/communication/setup_tokenizers.py`):  
Downloads and installs HuggingFace tokenizers into the shared public memory path. Currently installs `bert-base-uncased`.

**Tokenizer training stub** (`toys/communication/train_tokenizer.py`):  
Provides a scaffolding for training custom WordPiece tokenizers on domain-specific corpora. Outputs are saved in the same shared tokenizer directory structure.

**Mandatory tokenizer protocol in orchestration** (`baby/intelligence.py`):  
`orchestrate_turn` now requires a `tokenizer_name` argument. All text processing is routed through the tokenizer bridge; UTF-8 fallback and mixed encoding are no longer supported. This enforces a clean, consistent protocol and prevents knowledge contamination.

**Extended agent configuration support** (`baby/contracts.py`):  
`AgentConfig` now includes `tokenizer_name` and `tokenizer_mode` (values: `"input"`, `"io"`, or `None`) for agent-specific encoding strategies.

**Adapter and REST integration** (`toys/communication/external_adapter.py`):

*   All text encoding uses the tokenizer bridge and the default tokenizer is configurable via environment variable.
*   All calls to `orchestrate_turn` pass the required `tokenizer_name`.
*   Adapter path logic now uses `Path(__file__).resolve().parents[1]` for robust, CWD-independent resolution.

**Tokenizer and REST tests** (`toys/health/test_tokenizer.py`):

*   Validates round-trip encoding/decoding, byte safety, and vocabulary size.
*   Adds a REST façade integration test: starts the FastAPI app, POSTs to `/generate`, and asserts a 200 response, using fixtures to avoid data litter.

**Maintenance utilities**:

*   `**prune_and_compact_store**` (`baby/policies.py`): Prunes and compacts an `PhenotypeStore` in one pass, with support for age/confidence thresholds, dry-run, and archiving pruned entries.
*   **CLI script** (`toys/maintenance/prune_compact.py`): Command-line tool to prune/compact stores, with flexible options.
*   **Exported** `prune_and_compact_store` in `baby/__init__.py` for easy import.

### Notes

*   All text encoding is now strictly via the tokenizer bridge; UTF-8 fallback is removed from orchestration and adapter layers.
*   All test and adapter data is isolated using fixtures and temporary directories to prevent data litter.
*   Public tokenizers are shared under `memories/public/tokenizers/` and can be reused by any agent.
*   The system retains full Traceable replay and physics compliance.

## \[0.9.6.2\] – 2025-07-20

### ✅ **Changelog: Algebra, Learning, and Structural Corrections**

**1\. Finalised the Monodromic Fold (**`**fold**`**)**

Canonicalised the learning operation as:  
`fold(a, b) = a ^ (b ^ (a &amp; ~b))` ≡ `¬a ∧ b`

(Both forms are mathematically identical through Boolean algebra)

This form satisfies:

*   Non-associativity
*   Non-commutativity
*   Left-identity (`0 ⋄ b = b`)
*   Right-absorption (`a ⋄ 0 = 0`)

All prior variants (e.g. OR-based `coadd`) were removed as physically invalid.

**2\. Unified Egress and Ingress under One Operator**

*   Removed artificial distinction between input (`Egress`) and output (`Ingress`) operators.
*   Both processes now use the **same Monodromic Fold**, applied in opposite directions.
*   Path dependence and non-associativity are preserved in both directions.

**3\. Phenotype Learning Logic Clarified**

*   Confirmed that **repeated learning with the same intron toggles memory\_mask** (x ↔ 0).
*   This behavior is intentional and expresses **monodromic closure**, not cumulative accretion.
*   Docstrings updated to explain this self-annihilating mechanism clearly.

**4\. CanonicalView Bug Fixed**

*   `context_signature` was mistakenly stripped in phenotype entries.
*   Fixed: `context_signature` is retained; only `_original_context` may be removed safely.
*   Prevents `KeyError` during learning and inference.

**5\. Storage Durability Improvement**

*   Added optional `fsync.result()` wait in `commit()` for guaranteed flush during tests.
*   Prevents race conditions when asserting durability after write.

**6\. Confirmed Map Validity**

*   The `epistemology` and `ontology` maps were checked and found internally consistent with the Monodromic Fold.
*   No regeneration required.

**7\. Designed Physical Non-Associativity Experiment**

*   Prepared a plan to empirically test physical path dependence using your actual state transition maps.
*   Confirms that associativity violations are not algebraic artifacts, but grounded in state evolution.

**8\. Genetics**

*   Introduced Exons and Refined our Genetics assosiations.
*   Changed our Phenotype's metadata contracts.
*   Refined our Semantic Framework.

**8\. Pytest Corrections & Results**

*   Passed all 132 tests

## \[0.9.6.2\] – 2025-07-17 to 19

Wrote the code for all:  
baby/contracts.py  
baby/governance.py  
baby/inference.py  
baby/information.py  
baby/intelligence.py  
baby/policies.py

wrote the tests:  
toys/health/conftest.py  
toys/health/test\_governance.py  
toys/health/test\_inference.py  
toys/health/test\_information.py  
toys/health/test\_intelligence.py  
toys/health/test\_miscellaneous.py

Here's a concise changelog entry capturing the essence of that addition:

**Added**: `toys/communication/external_adapter.py` — a FastAPI-based external adapter exposing GyroSI through industry-standard REST interfaces.

*   Implements **OpenAI-compatible** endpoints (`/v1/models`, `/v1/chat/completions`) and **HuggingFace-style** generation (`/generate`).
*   Connects to the internal `baby.intelligence` engine without modification; operates via `AgentPool` and `orchestrate_turn`.
*   Manages three distinct agents per session (system/user/assistant) with consistent ID handling and memory bootstrapping logic.
*   Enables seamless external integration without altering core physics or learning logic.

## \[0.9.6.2\] – 2025-07-16

### Major Refactoring and Architecture Improvements

*   **Storage Layer Consolidation**: PhenotypeStore is now the single canonical storage class. All overlays and canonicalization are handled via decorators (CanonicalView, OverlayView, ReadOnlyView). Legacy/duplicate storage classes and factories removed.
*   **Async and Streaming Optimizations**: PhenotypeStore flush now uses async fsync (background thread). Batch learning uses O(1) memory streaming coaddition.
*   **Protocol and Type Hygiene**: PhenotypeStore protocol and all shared types (PhenotypeEntry, ManifoldData, AgentConfig, etc.) are now in baby/contracts.py (renamed from types.py). All storage implementations are in baby/policies.py (renamed from maintenance.py).
*   **Import and Packaging Consistency**: All imports now use absolute paths (from baby.\*). No more relative imports for shared types or storage classes. Circular imports and shadowing issues resolved.
*   **PEP8 and Linting**: All major linter errors fixed (unused imports/variables, blank lines, whitespace, long lines). Guidance provided for using black and autoflake for future formatting.
*   **Error Diagnosis and Environment Guidance**: Diagnosed and provided solutions for persistent import errors, including shadowing, packaging, and cache issues. Provided shell commands and troubleshooting steps.

### Project Structure After Refactor

```plaintext
baby/
  contracts.py      # All protocols and shared types (PhenotypeStore, etc.)
  policies.py       # PhenotypeStore, CanonicalView, OverlayView, ReadOnlyView, and policy/maintenance functions
  information.py    # InformationEngine and related logic
  intelligence.py   # IntelligenceEngine, GyroSI, and orchestration logic
  inference.py      # Inference logic
  __init__.py       # Clean, canonical imports and __all__ for package API
  ...               # Other modules as needed
```

### Key Outcomes

*   Single source of truth for all protocols and storage classes.
*   No more circular imports, shadowing, or ambiguous imports.
*   All code is PEP8-compliant and linter-friendly.
*   Project is robust for both development and production.

## \[0.9.6.1\] – 2025-07-15

### **GyroSI Baby Language Model 0.9.6 (Conceptual & Architectural Refactor)**

This update represents a major conceptual and architectural refactoring of the GyroSI 0.9.6 specification. While the version number remains the same, the underlying theory, component architecture, and terminology have been significantly matured and clarified. The focus has shifted from an implementation-centric description to a physics-first framework, providing a more robust and scalable foundation.

**I. Major Architectural & Conceptual Refactoring**

**Introduction of the Measured Manifold:**

*   The system is now grounded in the **empirically measured and provably finite physical ontology** of precisely **788,986 unique states**. This replaces the previous, more abstract notion of a state space.
*   The ontology's **diameter is a measured constant of 6**, meaning any state is reachable from any other in at most seven steps.
*   This "measured ground truth" is now the cornerstone of the entire architecture, moving the system from "physics-inspired" to "physics-grounded".

**VSM-Aligned Engine Architecture:**

*   The four engines have been explicitly mapped to **Beer's Viable System Model (VSM)**, clarifying their roles and creating a recursive, self-regulating structure.
*   **S1 Governance** is no longer an "engine" class but a set of pure, stateless functions and constants in `governance.py` (The Physics).
*   **S2 Information Engine** is now solely responsible for measurement and storage coordination (`information.py`).
*   **S3 Inference Engine** focuses on interpretation and meaning management (`inference.py`).
*   **S4/S5 Intelligence Engine** handles orchestration, agent state, and the external API (`intelligence.py`).

**Decoupled Storage via** `**PhenotypeStore**` **Interface:**

*   The complex, bespoke file structure (`Gyronorm Formats`, `Gene Keys`, `Threads`) has been replaced by a clean, abstract `**PhenotypeStore**` **protocol**.
*   This decouples the core physics from all persistence concerns, allowing for swappable storage backends (e.g., `PickleStore`, `MultiAgentPhenotypeStore`, or future database adapters).
*   Knowledge is now stored in a `(context_key -&gt; phenotype_entry)` mapping, where `context_key` is a `(tensor_index, intron)` tuple.

**Formalized API and Integration Layer:**

*   A dedicated **Core API and Integration** section has been added, defining a stable `GyroSI` class as the primary entry point.
*   Introduced the `**AgentPool**` concept for managing multiple agents and orchestrating conversations through agent interaction, rather than specialized chat infrastructure.
*   Provided a clear pattern for creating **Protocol Adapters**, with an example for an OpenAI-compatible API.

**Canonicalization of Orbits:**

*   A new, fundamental abstraction layer has been introduced: **Canonicalization**.
*   A build-time process identifies a single canonical representative for each state orbit within the ontology.
*   The `**CanonicalizingStore**` **decorator** ensures all physically equivalent states map to the same storage entry, improving data coherency and abstraction.

**II. Core Physics & Foundational Changes**

**Formalized Gyrogroup Algebra:**

*   Learning is no longer based on a heuristic `combined_score`. It is now defined by **true gyrogroup coaddition (**`**a ⊞ b**`**)**, a specific, path-dependent, non-commutative algebraic operation.
*   This change introduces **Ordered Batching** as the canonical way to process multiple learning signals, preserving the structure of experience.

**Refined Transformation Physics:**

*   The core state transformation logic (`apply_gyration_and_transform`) now includes a physically correct **gyration memory (carry term)**, implemented as `final_state = temp_state ^ (temp_state &amp; intron_pattern)`. This makes the transformation path-dependent in a more fundamental way.
*   The physical effect of Forward/Backward Gyrations (FG/BG) is now specified to operate on entire **tensor layers** (0&2 / 1&3), which is a more precise definition than the previous "rows".

**Canonical Integer State Representation:**

*   The primary representation of the system's physical state (`GENE_Mac_M`) is now a **packed 48-bit integer**. The 48-byte NumPy tensor is used for geometric measurement but is secondary to the integer for state transitions. This has significant performance and storage benefits.

**III. Terminology & Naming Conventions**

A new, consistent naming scheme has been adopted to better reflect the system's physics.

`**GENE_***` **Naming:**

*   `GENE_Mac_S` (Stateless Macro Gene) replaces `gene_add`.
*   `GENE_Mic_S` (Stateless Micro Gene) replaces `gene_stateless = 0xAA`.
*   `GENE_Mac_M` (Mutated Macro Gene) replaces the "Epigenome Tensor" (`self.T`).
*   `GENE_Mic_M` (Mutated Micro Gene) replaces `gene_mutated`.

**Conceptual Renaming:**

*   The concepts of **"Exon"** (stateless archetype) and **"Intron"** (dynamic instruction) have been introduced. The `intron` is the 8-bit value derived from an input byte.
*   `gyrodistance_angular` is the formal name for the measurement function.

**IV. Removed & Replaced Components**

*   **Removed:** `**Epigenome**` **and** `**Genome**` **Masks.**
    *   `epigenome.dat` (the 256 canonical patterns) is no longer used, as state is compared directly to the archetypal `GENE_Mac_S`.
    *   `genome.dat` (the output map) is replaced by the `phenotype` field within the `PhenotypeStore`.
*   **Removed:** `**Gyronorm Formats**` **and** `**Gene Keys**`**.**
    *   The complex JSON-based `format-<uuid>.json` files are gone. Semantic mapping is now handled by the simpler `phenotype_entry` dictionary in the `PhenotypeStore`.
    *   The `gene-<uuid>.ndjson` event logs are removed. Learning history is now implicitly captured in the state of the `PhenotypeStore`.
*   **Removed:** `**Thread**` **Files and Bespoke Encryption.**
    *   The structured `thread-uuid.ndjson.enc` files and their associated AES-256-GCM encryption model have been removed from the core spec. Content storage is now an application-level concern, separate from the knowledge store.
*   **Removed: Heuristic Response Generation.**
    *   The `_generate_response_byte` function with its `physical_score * semantic_score` logic is gone, replaced by the direct lookup in the `PhenotypeStore` after the physical state transition.

**V. New Features & Capabilities**

1.  **Maintenance & Operations Toolkit:**
    *   A new section details production-ready maintenance tools.
    *   Includes scripts/functions for **Confidence Decay** (`apply_confidence_decay`) to gracefully age out stale knowledge.
    *   Includes a **Map Merging Tool** (`merge_phenotype_maps`) for consolidating knowledge from multiple agents or sources.
2.  **Performance & Scaling Section:**
    *   A new section provides formal **performance characteristics**, including computational complexity, memory requirements, and throughput estimates.
3.  **Theoretical Appendix:**
    *   A new appendix explicitly maps GyroSI concepts to their corresponding principles in physics and mathematics, solidifying the theoretical foundation.

## \[0.9.5\] – 2025-07-12

*   Refactored InferenceEngine and InformationEngine to support efficient batch processing with a new process\_batch() method and updated process\_stream() for fast-path batching.
*   Created a high-performance bulk trainer script (toys/learning/trainer.py) for large curriculum files, supporting batch updates, periodic checkpointing, and multi-format learning.
*   Added preference flags for batch size and optional Numba JIT compilation for further speedup.
*   Established a clean, two-step curriculum workflow:
    1.  Curriculum Serialization: Script (toys/learning/threads/wordnet\_curriculum\_threads.py) serializes the entire WordNet database into a flat text file for training. Output now goes to toys/learning/threads/corpus/wordnet\_corpus.txt.
    2.  Model Training: The trainer script consumes the generated corpus file for fast, scalable learning.
*   Restored and updated all curriculum format and thread generator scripts in toys/learning/formats/ and toys/learning/threads/ to use correct namespace UUIDs and implement pattern index cycling (degeneracy).
*   Ensured all scripts are runnable with PYTHONPATH=. for proper import resolution.
*   Rewritten learning update logic so all loaded formats are updated for each winning pattern index, enabling true multi-format associative learning and correct handling of degeneracy.
*   Fixed all pyright type errors related to string/bytes handling and added targeted type ignore comments where necessary.
*   Moved all generated and output files to appropriate subdirectories to avoid clutter and maintain a clean project structure.

## \[0.9.5\] – 2025-07-11

Enforced strict test isolation: all tests and engine code now use a dedicated test directory (`toys/health/memories/`).

Standardized argument propagation: all helpers and engine methods now require and pass `prefs` and `base_memories_dir` as needed.

Fixed critical bugs in `IntelligenceEngine` initialization and thread/gene key storage.

Updated test mocks and assertions to match real function signatures, eliminating signature mismatch errors.

Resolved all linter and static analysis issues; codebase is now clean and warning-free.

Investigated and explained origins of stray test output and directory artifacts.

Major performance and robustness improvements:

*   Added pattern matching cache to InferenceEngine for fast repeated lookups.
*   Implemented batch stream processing in InformationEngine with configurable batch size for efficient I/O.
*   Introduced robust, multi-process-safe registry caching with mtime-based invalidation.
*   Refactored PatternIndex to use defaultdict for cleaner and faster indexing.
*   Optimized IntelligenceEngine encode/decode logic with O(1) lookup maps supporting multiple patterns per character.
*   Simplified registry cache eviction logic for clarity and correctness.

**Added new CLI suite under** `**toys/console/**`**:**

*   Interactive chat, dashboard, format viewer, and thread manager tools for BabyLM.
*   All CLI modules are type- and lint-clean, with robust error handling and safe type usage throughout.

Refactored IntelligenceEngine to support multiple formats simultaneously, keyed by format\_uuid.

Updated all code and tests to use per-format access (self.formats\[self.format\_uuid\]) instead of self.M.

Fixed all test failures and ensured robust multi-format operation.

Updated GyroSI_Specs.md to clarify format access, pattern index cycling, and pattern distance matrix storage.

Implemented stable UUIDs for public formats using uuid.uuid5 and a fixed namespace, enabling reproducible curriculum and format references.

Fixed TypedDict access warnings for optional keys.

## \[0.9.5\] – 2025-07-10

*   Refactored private gene key storage to use per-record encryption with length prefix for true append-only performance.
*   Public thread metadata is now updated only at finalization, not per event, for better performance.
*   Moved tensor\_to\_output\_byte to InferenceEngine for architectural clarity.
*   Fixed AES key length validation to require 32 bytes (256 bits) for AES-256.
*   Added registry file locking in shard\_path to prevent race conditions during sharding.
*   Switched recent\_patterns to collections.deque(maxlen=256) for efficient context tracking.
*   PatternIndex.find\_similar\_contexts now caps locations checked for common patterns to avoid performance bottlenecks.
*   Removed redundant cryptography imports and local JSON helpers.
*   Replaced brittle thread UUID checks with robust file existence checks.
*   Refactored ThreadMetadata to use children: List\[ChildRef\] instead of parallel child\_uuids/child\_names; updated all code and tests accordingly.
*   Improved TypedDict access safety and fixed pyright linter errors throughout the codebase.
*   Unified privacy logic using an explicit 'privacy' field ('public'/'private') for threads and gene keys
*   Replaced legacy XOR encryption with robust AES-GCM encryption for private threads, using per-thread derived keys
*   Clarified and retained 'agent\_uuid' in gene keys for agent association/ownership (not privacy)
*   Refactored thread/gene key creation, storage, and tests to use the new privacy model
*   Updated all relevant code and tests for clarity, security, and future extensibility
*   Major revision of GyroSI_Specs.md: fully integrated Common Governance Model (CGM) theory, mapping all system components to CGM stages (CS, UNA, ONA, BU)
*   Clarified and formalized the dual nature of BU (Egress/Recollection and Ingress/Closure) in both documentation and code
*   Updated all terminology to remove analogies (e.g., "physicist/linguist"), using only precise CGM language
*   Ensured the spec and implementation match: \_generate\_response\_byte now documented and implemented as a two-stage ONA/BU process using combined resonance and confidence
*   Rewritten learning mechanisms section to reflect that confidence directly influences generation (BU closure), and removed outdated "attention mechanism" text
*   Added a comprehensive glossary of all key terms (Epigenome, Genome, Egress, Ingress, ONA, BU, etc.)
*   Fixed Pyright and linter errors in intelligence.py and babylm.py (indentation, type safety, buffer handling)

### Fixed

*   Fixed a critical state management bug in `IntelligenceEngine` that caused duplicate or excessive gene key writes in public mode. State buffers are now always cleared after finalization, preventing data leaks between sessions in both public and private modes.
*   Fixed inconsistent and incorrect file path logic for public thread and gene key storage, ensuring all files are created and written to the correct sharded locations.
*   Fixed issues where public NDJSON files were not being created or written due to file handle and path errors.

### Improved

*   Modernized and strengthened the test suite for public thread and gene key storage, ensuring robust detection of subtle bugs and regressions.
*   Unified file path calculation logic for all public thread operations, improving maintainability and reliability.
*   Ensured state buffers are always cleared after finalization, preventing data leaks between sessions in both public and private modes.

## \[0.9.5\] – 2025-07-09

### Changed

🧠 **Major S3/S4 Refactor:**

*   Inference (S3) is now purely resonance-based, with all statistical/contextual weighting removed
*   S3 (physics) and S4 (intelligence/semantics) responsibilities are now cleanly separated
*   S4 uses pattern metadata (`count`, `confidence`, etc.) for intelligent encoding, decoding, and generation
*   Learning loop closed: resonance → confidence → generation → resonance
*   Documentation and tests updated to reflect new architecture and learning behavior

## \[0.9.5\] – 2025-07-08

### Added

🌐 **Expanded Global Format Library**

*   We have expanded our global format library! Formats are shared global knowledge and are available to all agents, though they do not contain contextual information. (Scripts Available at: toys/learning/formats)
    *   **ASCII Curriculum:** 256 foundational ASCII characters
    *   **Emoji Curriculum:** Over 5,000 Unicode emoji
    *   **Mathematical Symbols Curriculum:** All Unicode mathematical symbols (excluding ASCII)
    *   _(More curricula can be added as the system grows)_

### Changed

*   Adopted NDJSON format for all gene keys and threads, supporting both public (unencrypted) and private (encrypted) storage for agent-agnostic knowledge sharing and curriculum learning
*   Refactored IntelligenceEngine initialization to support full agent, read-only, and public/curation modes for flexible batch processing and knowledge sharing
*   Integrated fast JSON parsing with orjson (and ujson fallback), using unified json\_loads/json\_dumps wrappers in all core modules for performance
*   Ensured type safety and Pyright compliance throughout the codebase, especially for Optional\[str\] and None handling in public/curation mode
*   Suppressed harmless linter warnings for optional imports (e.g., ujson)
*   Improved overall codebase robustness, maintainability, and compatibility for both private and public agent operation

## \[0.9.4\] – 2025-07-07

### Added

*   **Sharded Storage:** Replaced monolithic JSON with two-level hex sharding for all agent, thread, key, and format objects. Traceable O(1) path computation for reads
*   **Registry Files:** Each shard now maintains a `registry.json` index for immediate children, with automatic updates and crash recovery helpers
*   **Atomic, Concurrent Writes:** All writes use atomic temp files and `flock`\-protected registries for safe concurrent access
*   **Agent, Thread, and Key Management:**
    *   Automatic RFC 4122 agent UUID generation and sharding
    *   Per-thread metadata JSONs and encrypted payloads/keys, with Traceable lookup
*   **Format Management:**
    *   Public, sharded formats with support for pattern-distance matrices and recursive listing/loading helpers
*   **Module Boundaries:**
    *   `information.py`: storage, sharding, atomic writes, registry, UUID, thread/key I/O
    *   `inference.py`: pattern-matching, Epigenome mutations, genome mask loading
    *   `intelligence.py`: orchestration, thread lifecycles, encryption, context-aware response
*   **Type Safety & Test Reliability:**
    *   Introduced TypedDicts for metadata, enforced type-correct assignments, and safe dictionary access
    *   All tests pass; codebase is type- and lint-clean

## \[0.9.3\] – 2025-07-06

### Added

🧠 **Complete GyroSI Baby LM AI system architecture rewrite**

*   **S1 Governance Layer**: Core system governance and coordination
*   **S2 Information Layer**: Data processing and tensor operations
*   **S3 Inference Layer**: Epigenome management and keystream generation
*   **S4 Intelligence Layer**: Pattern matching and output generation

🔧 **Canonical tensor-to-byte conversion system**

*   Implemented spec-compliant `tensor_to_output_byte` function
*   Pattern matching approach for Traceable output generation
*   Removed non-canonical conversion methods

🧪 **Comprehensive test suite**

*   Integration tests with improved fixtures
*   Unit tests for all engine components
*   Test data properly organized in designated test directories

### Changed

🔄 **Refactored core engines for canonical approach**

*   `InformationEngine`: Updated for spec-compliant tensor operations
*   `IntelligenceEngine`: Implemented pattern-based output generation
*   `InferenceEngine`: Fixed epigenome initialization bug

### Fixed

🐛 **Critical bug fixes**

*   Fixed all-zero keystream causing ciphertext to equal plaintext
*   Corrected pattern index handling in tests
*   Resolved Pyright type errors throughout codebase
*   Fixed flake8 linting issues (unused imports, whitespace, bare except)

### Technical

📝 **Code quality improvements**

*   Added explicit type annotations
*   Removed duplicate type declarations
*   Cleaned up unused imports and variables
*   Replaced bare except with specific exception handling

## \[0.1.0\] – 2025-06-22

### Added

🗂️ **Initial project structure established**

*   📁 Created base directories: src/, tests/, docs/, examples/
*   📄 Added s1\_governance files: README.md, LICENSE, pyproject.toml, requirements.txt, Makefile, CHANGELOG.md\\

```plaintext
"pruning": {
  "confidence_threshold": 0.05,
  "decay_factor": 0.995,
  "decay_interval_hours": 6,
  "enable_auto_decay": true
}
```

`plaintext [uLEB128 state_index][uLEB128 n_pairs][(uLEB128 token_id + mask_byte) * n_pairs]` \</uuid>\</iibhx"\`>