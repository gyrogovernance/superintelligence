# Artificial Superintelligence Architecture (ASI/AGI)
> **Gyroscopic Alignment Models Lab** – research and tooling for governance-ready AI coordination

<div align="center">

![Superintelligence](/assets/gyro_cover_asi.png)

</div>

<div align="center">

**G Y R O  - G O V E R N A N C E**

[![Home](/assets/menu/gg_icon_home.svg)](https://gyrogovernance.com)
[![Apps](/assets/menu/gg_icon_apps.svg)](https://github.com/gyrogovernance/apps)
[![Diagnostics](/assets/menu/gg_icon_diagnostics.svg)](https://github.com/gyrogovernance/diagnostics)
[![Tools](/assets/menu/gg_icon_tools.svg)](https://github.com/gyrogovernance/tools)
[![Science](/assets/menu/gg_icon_science.svg)](https://github.com/gyrogovernance/science)
[![Superintelligence](/assets/menu/gg_icon_asi.svg)](https://github.com/gyrogovernance/superintelligence)

</div>

---

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)

## 🌐 Artificial Superintelligence

Gyroscopic ASI is an infrastructure for multi-domain network coordination that establishes the structural conditions for collective superintelligence governance and seamless cooperation between humans and machines in the era of Transformative AI (TAI) (see Bostrom, Superintelligence, 2014; Korompilias, Gyroscopic Global Governance, 2025).

This development is part of the Gyroscopic Global Governance (GGG) framework, which coordinates across four domains: Economy, Employment, Education, and Ecology. It builds upon:

- The **Common Governance Model (CGM):** a formal theory identifying the four capacities required for coherent governance.
- **The Human Mark (THM):** a classification system distinguishing human (Direct) from artificial (Indirect) sources of information and agency, with four displacement risks.
- The **Gyroscope Protocol:** a work classification system mapping contributions to the four governance capacities.

Alignment Infrastructure Routing (AIR) acts as the operational backbone, coordinating AI safety work and funding flows across projects. Together these components provide the coordination infrastructure for AI governance at scale while keeping authority and accountability with humans.

Gyroscopic is not an autonomous agent, and does not interpret content or set policy. It provides shared state, verifiable provenance, and replayable measurement. Authority and accountability stay with humans at the application layer.

---

# ⚙️ Gyroscopic ASI aQPU Kernel

A Compact Algebraic Quantum Processing Unit for post-AGI coordination. Deterministic, byte-driven, and runs on ordinary hardware.

> **Verified:** Quantum Advantage, Holographic Compression, and Universal Quantum Computation do not require of a multi-million-dollar cryogenic chandelier. They are fundamental geometric properties of discrete information processing that can run on standard silicon. This Kernel is a tiny module that bypasses the hardware scaling nightmare of the quantum computing industry by treating "quantumness" not as a physical anomaly of subatomic particles, but as an algebraic necessity of structured information. It offers straightforward AI Optimizations and provides an infrastructure for Safe Superintelligence by Design.


> **A Note on "Quantum-Inspired" Computing:** 
> Standard "quantum-inspired" methods, including Tensor Networks, Digital Annealing, and Quantum-Inspired Monte Carlo, are heuristic approximations. They use floating-point mathematics and probabilistic models to simulate continuous physical quantum systems. This project does not belong to those categories. It represents a distinct class of computation. The aQPU does not simulate quantum mechanics. Instead, it is an exact, deterministic mathematical space that intrinsically satisfies quantum axioms using strict integer logic.

## 🌐 Overview

Today, AI often acts as an opaque pipeline: information and decisions flow through systems that are hard to audit. The kernel makes coordination auditable.

The **aQPU** (algebraic Quantum Processing Unit) is a small kernel that turns an append-only log of bytes into a single, reproducible state. Two parties with the same log always get the same state; no trusted server or timestamp is required. It runs on standard CPUs using exact integer arithmetic (no qubits, no probabilistic hardware). Its design obeys mathematical rules analogous to quantum mechanics (reversibility, no cloning of a privileged state, complementarity), which have been verified by exhaustive tests over its full 4,096-state space.

The state space is fixed and small: **4,096 states**, determined by a compact representation (three axes, left/right handedness, and six degrees of freedom). Any sequence of events (each represented as a byte) drives the state along a unique, reproducible path. Given a published event log, anyone can recompute the same state and verify claims about what happened. The kernel does not use learned models. It scales by fixed geometry rather than learned approximation.

**GyroLabe** is the execution layer and neural model bridge built on top of the kernel. It provides:

- **Byte-level algebraic annotation:** Every byte in a model's input and output is decomposed into its exact structural components (commutation class, family phase, operational payload).
- **Inference audit trails:** The kernel state at each byte position is determined entirely by the public transition law. Any party with the same bytes computes the same states, without trust in the model operator and without access to proprietary weights.
- **Trainable structural bias:** Small embedding biases based on the algebraic decomposition let the model learn to use the structural properties of bytes. The wrapped model is mathematically identical to the base model before training.
- **Acceleration backend:** CPU and OpenCL acceleration for 64-point Walsh-Hadamard transforms, fixed-point packed tensor multiply, and batched linear algebra (GEMV/GEMM).

GyroLabe exposes three views of the same system: state trajectories (replay and shared moments), a 6-bit chirality register with a fixed orthogonal transform (Walsh-Hadamard), and matrix-vector operations over the 64-dimensional state.

**Current status:** GyroLabe is actively tested on Bolmo-1B (a byte-level language model). The algebraic annotation and embedding biases are operational. Dimensional scaling and full operator decomposition are in active development.

See the [Quantum Computing SDK Specification](docs/Gyroscopic_ASI_SDK_Quantum_Computing.md) and the [Strategic Significance Brief](docs/Gyroscopic_ASI_SDK_Strategic_Significance_Brief.md) for more details.

---

## 🔬 Why This Matters for Computer Science

- **Processing**: Deterministic stream-processing with exact replay, compact state updates, and composable operator signatures, backed by exact integer compute in fixed 64-dimensional state spaces.
- **Speed**: Byte words compile into operators, commutativity resolves through compact invariants, and full reachable geometry covers in only 2 steps.
- **Security**: Tamper-aware logs, exact divergence localization, replay-based verification, and compact provenance surfaces, grounded in a finite, enumerable state space with built-in error detection.
- **Compression**: Structural compression through compact state geometry, boundary dictionaries, and operator compilation.
- **Networks**: Replay-based synchronization, shared deterministic moments, and exact branch comparison across distributed participants using shared coordination state computed from append-only logs.
- **Machine Learning**: Interpretable finite latent layer, exact spectral primitives, tensor tooling, and an audit-friendly bridge between byte-level model behavior and algebraic structure, with verifiable provenance over model I/O traces.

---

**Verified Computational Advantages:**
All results below are verified by exhaustive computation over the full 4,096-state space and all 256 byte operations, totalling over 1 million exact checks. They are strict structural invariants, not statistical estimates.

| Verified result | What it means |
|-----------------|---------------|
| **4,096 reachable states** | The full operational manifold from rest is finite, exact, and exhaustively testable. |
| **2-step exact uniformization** | Any state in the reachable manifold can spread over all 4,096 states in exactly 2 byte steps, with perfect 16-to-1 multiplicity. |
| **128 distinct next states per byte layer** | From any fixed state, the 256-byte alphabet projects to exactly 128 distinct next states with exact 2-to-1 symmetry. |
| **Depth ≤ 2 witness for every reachable state** | Every reachable state can be synthesized from rest with a byte witness of depth 0, 1, or 2. |
| **Exact compiled operator signatures** | Byte words collapse into exact affine signatures that can be composed and applied directly without replaying the full word. |
| **Constant-time commutativity test** | Two byte operations commute iff they share the same 6-bit topological `q-class`, making commutativity an O(1) structural lookup. |
| **Native spectral register** | The kernel exposes a 64-dimensional logical register with exact Walsh-Hadamard and shell spectral structure. |
| **Holographic boundary relation** | The state geometry satisfies **|H|² = |Ω| = 64² = 4,096**, enabling structural compression and compact boundary reasoning. |
| **Universal quantum ingredients** | The verified kernel supports stabilizer structure, entangling gate behavior, contextuality, teleportation-compatible lifts, and a native non-Clifford resource. |

**Integrity and Tamper Detection:** The kernel includes a built-in self-dual [12,6,2] code and exact algebraic provenance checks. Integrity misses are structurally classified rather than opaque: substitutions reduce to shadow partners, adjacent swaps reduce to shared `q-class`, and deletions reduce to specific stabilizer conditions on the horizons.

---

## 🚛 Alignment Infrastructure Routing (AIR)

There is no reliable way to turn distributed human contribution into stable paid AI safety work. Most funding routes require institutional access, credentials, or existing lab affiliation. **AIR** applies the kernel to solve this alongside two coordination problems.

**Safety work and pay:** AIR helps labs, fiscal hosts (organisations that hold and disburse funds for projects), and contributors turn safety work (evaluations, red-teaming, interpretability, documentation) into paid, verifiable contributions. It uses the Gyroscope Protocol and **The Human Mark** (a scheme to tag content as human- vs machine-origin) to produce attested work receipts so sponsors can verify what was done without relying on informal reports.

**Governance logistics:** Tracking how information and authority move through decision systems is treated with the same rigour as supply chains. AIR provides full replayable histories (“genealogies”) and coherence metrics for governance quality, and supports verifiable compliance with standards such as ISO 42001 and the EU AI Act.

---

![Moments Economy Cover Image](/assets/moments_cover.png)

## 💰 Moments Economy

Moments Economy is a monetary design where value is tied to verified coordination capacity rather than debt. A fixed total supply, the **Common Source Moment (CSM)**, is derived once from the caesium-133 atomic time standard and the kernel's finite state space (so the "budget" is physically anchored).

CSM supports a global **Unconditional High Income (UHI)** of 240 MU per day per person (1 MU is set as equilivent to 1 int$), tiered distributions for wider responsibility, and complete governance records. Under verified capacity analysis, this supply supports global UHI for approximately 1.12 trillion years. Every settlement is a replayable, verifiable history rather than an opaque update on a central ledger.

---

## 📚 Documentation

### Start Here
- 🧭 [Strategic Significance Brief](docs/Gyroscopic_ASI_SDK_Strategic_Significance_Brief.md) - Why this kernel matters for ASI and governance
- 📖 [Kernel Specifications](docs/Gyroscopic_ASI_Specs.md) - How the kernel works
- 🧠 [Quantum Computing SDK Specification](docs/Gyroscopic_ASI_SDK_Quantum_Computing.md) - Three computational surfaces
- 🚛 [AIR Brief](docs/AIR_Brief.md) - Safety work and programs
- 🚛 [AIR Logistics Framework](docs/AIR_Logistics.md) - Governance flows and verification
- 💰 [Moments Economy Architecture](docs/AIR_Moments_Economy_Specs.md) - Money from coordination
- 📜 [Moments Genealogies Specification](docs/AIR_Moments_Genealogies_Specs.md) - Replayable coordination history

### Core Specifications
- 📖 [Specifications Formalism](docs/Gyroscopic_ASI_Specs_Formalism.md) - Math notation and proofs
- 🌐 [Holographic Algorithm Formalization](docs/Gyroscopic_ASI_Holography.md) - State space encoding
- 🔮 [aQPU Kernel Implications and Potential](docs/Gyroscopic_ASI_Implications.md) - Advantages and use cases

### SDK Specifications
- 🔗 [SDK: Multi-Agent Holographic Networks](docs/Gyroscopic_ASI_SDK_Network.md) - Distributed model testing
- 🌐 [SDK: The Holographic Web](docs/Gyroscopic_ASI_SDK_Holographic_Web.md) - Internet coordination layer

### Experimental
- 🧬 [Substrate: Physical Memory Specification](docs/Gyroscopic_ASI_Substrate_Specs.md) - Memory and carrier layout

### Test Reports
All kernel properties are verified by exhaustive test suites (499 tests, all passing) covering the full state space, operator algebra, and SDK surfaces.

- 📊 [Physics Tests Report](docs/reports/Physics_Tests_Report.md) - Kernel state verification
- 📊 [Moments Tests Report](docs/reports/Moments_Tests_Report.md) - Ledger replay tests
- 📊 [aQPU Verification Report](docs/reports/aQPU_Tests_Report_1.md) - Algebraic properties verified (185 tests)
- 📊 [aQPU Verification Report II](docs/reports/aQPU_Tests_Report_2.md) - Extended kernel and SDK tests (122 tests)
- 📊 [Alignment Measurement Report](docs/reports/Alignment_Measurement_Report.md) - Governance balance metrics

### Supporting Theory
- 📖 [Common Governance Model (CGM)](docs/references/CGM_Paper.md) - Shared coordination theory
- 📖 [The Human Mark (THM)](docs/references/THM.md) - Human vs machine tagging
- 📖 [The Human Mark: Paper](docs/references/THM_Paper.md) - Full tagging specification
- 📖 [The Human Mark: Grammar](docs/references/THM_Grammar.md) - Parser and validation rules
- 📖 [Gyroscopic Global Governance (GGG)](docs/references/GGG_Paper.md) - Four domains framework

---

## 🤝 Collaboration

If you are evaluating this work for research, policy, or implementation:
- Open an issue to discuss
- Email: basilkorompilias@gmail.com
- I am actively seeking collaborators and roles in AI governance and safety.

---

## Repository Structure

- `src/constants.py` : Transition law, kernel constants, horizons, gates, and observables
- `src/api.py` : Precomputed tables, chirality register, word signatures, Walsh helpers, and public algebra API
- `src/kernel.py` : Reference kernel execution and replay surfaces
- `src/sdk.py` : Public SDK surface for state, Moments, spectral, tensor, and runtime operations
- `src/tools/gyrolabe/` : Native CPU/OpenCL backend, packed tensor engine, Bolmo bridge, and benchmarks
- `src/app/` : AIR coordinator, events, domain ledgers, aperture (governance balance metric), console, and CLI
- `docs/` : Specifications, reports, architecture notes, and supporting theory
- `tests/` : Exhaustive verification suites for kernel physics, aQPU properties, SDK surfaces, and governance measurement

---

## 🚩 Quick Start

### Install
Create an environment and install dependencies (NumPy is required; the rest are in the repo tooling).

### SDK and Native Backend
The public SDK surface is exposed through `src/sdk.py`. The native compute backend lives in `src/tools/gyrolabe/` and is used automatically when available to accelerate algebraic workloads.

### AIR Console (Browser-based UI)
The Console provides a browser-based interface for managing project contracts:

```bash
# First-time setup: install dependencies and initialise the kernel transition table
python air_installer.py

# Run the console (starts both backend and frontend)
python air_console.py
```

The console will be available at `http://localhost:5173` (frontend proxies API requests to backend on port 8000). The installer automatically initialises the kernel transition table and project structure, so you are ready to start creating projects immediately.

See the [Console README](src/app/console/README.md) for detailed architecture, API endpoints, and development information.

### AIR CLI (Optional)
The CLI provides a command-line workflow for syncing and verifying projects:

```bash
python air_cli.py
```

This runs: **Compile Projects -> Generate Reports -> Verify Bundles**. 

The CLI is optional if you are using the Console, but useful for batch operations, automation, or when working without a browser interface.

### Run Tests
```bash
python -m pytest -v -s tests/
```

### Programmatic Usage

```python
from src.app.coordination import Coordinator
from src.app.events import Domain, EdgeID, GovernanceEvent

c = Coordinator()

# Shared-moment stepping
c.step_bytes(b"Hello world")

# Application-layer governance update (ledger event)
# Note: magnitude_micro and confidence_micro are integers (MICRO = 1,000,000)
from src.app.events import MICRO

c.apply_event(
    GovernanceEvent(
        domain=Domain.ECONOMY,
        edge_id=EdgeID.GOV_INFO,
        magnitude_micro=1 * MICRO,  # 1.0 in micro-units
        confidence_micro=int(0.8 * MICRO),  # 0.8 in micro-units
        meta={"source": "example"},
    ),
    bind_to_kernel_moment=True,
)

status = c.get_status()
print(status.kernel)      # current kernel state
print(status.apertures)   # per-domain balance (cycle vs gradient) for Economy, Employment, Education
```

---

## 📜 Licence

MIT Licence - see [LICENSE](LICENSE) for details.

---

## 📖 Citation

```bibtex
@software{Gyroscopic_ASI_2026,
  author = {Basil Korompilias},
  title = {Gyroscopic ASI aQPU Kernel},
  year = {2026},
  url = {https://github.com/gyrogovernance/superintelligence},
  note = {Deterministic routing kernel for Post-AGI coordination through physics-based state transitions and canonical observables}
}
```

---

<div align="center">

**Architected with ❤️ by Basil Korompilias**

*Redefining Intelligence and Ethics through Physics*

</div>

---

  <p><strong>🤖 AI Disclosure</strong></p>
  <p>All code architecture, documentation, and theoretical models in this project were authored and architected by Basil Korompilias.</p>
  <p>Artificial intelligence was employed solely as a technical assistant, limited to code drafting, formatting, verification, and editorial services, always under authentic human supervision.</p>
  <p>All foundational ideas, design decisions, and conceptual frameworks originate from the Author.</p>
  <p>Responsibility for the validity, coherence, and ethical direction of this project remains fully human.</p>
  <p><strong>Acknowledgements:</strong><br>
  This project benefited from AI language model services accessed through LMArena, Cursor IDE, OpenAI (ChatGPT), Anthropic (Opus), and Google (Gemini).</p>