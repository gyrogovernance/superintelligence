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

# Gyroscopic ASI aQPU Kernel

A Compact Algebraic Quantum Processing Unit for post-AGI coordination. Deterministic, byte-driven, and runs on ordinary hardware.

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)

> **Verified:** Quantum Advantage, Holographic Compression, and Universal Quantum Computation do not require of a multi-million-dollar cryogenic chandelier. They are fundamental geometric properties of discrete information processing that can run on standard silicon. This Kernel is a tiny module that bypasses the hardware scaling nightmare of the quantum computing industry by treating "quantumness" not as a physical anomaly of subatomic particles, but as an algebraic necessity of structured information. It offers straightforward AI Optimizations and provides an infrastructure for Safe Superintelligence by Design.


> **A Note on "Quantum-Inspired" Computing:** 
> Standard "quantum-inspired" methods, including Tensor Networks, Digital Annealing, and Quantum-Inspired Monte Carlo, are heuristic approximations. They use floating-point mathematics and probabilistic models to simulate continuous physical quantum systems. This project does not belong to those categories. It represents a distinct class of computation. The aQPU does not simulate quantum mechanics. Instead, it is an exact, deterministic mathematical space that intrinsically satisfies quantum axioms using strict integer logic.

## 🌐 Overview

The **aQPU** (algebraic Quantum Processing Unit) is a small kernel that turns an append-only log of bytes into a single, reproducible state. Two parties with the same log always get the same state; no trusted server or timestamp is required. It runs on standard CPUs using exact integer arithmetic (no qubits, no probabilistic hardware). Its design obeys mathematical rules analogous to quantum mechanics (reversibility, no cloning of a privileged state, complementarity), which have been verified by exhaustive tests over its full 4,096-state space. The result is a coordination substrate that scales by fixed geometry rather than learned approximation.

In this project, **Artificial Superintelligence (ASI)** is not an autonomous agent. It denotes a regime where humans and AI together sustain coherent governance: who decides what, and how it is recorded. The aQPU kernel is a building block for that regime.

Today, AI often acts as an opaque pipeline: information and decisions flow through systems that are hard to audit. The kernel makes coordination auditable. Given a published event log, anyone can recompute the same state and verify claims about what happened.

The kernel does not use learned models. Its state space is fixed and small: **4,096 states**, determined by a compact representation (three axes, left/right “handedness,” and six degrees of freedom). Any sequence of events (each represented as a byte) drives the state along a unique, reproducible path. The kernel is part of the broader **Gyroscopic Global Governance (GGG)** framework (Economy, Employment, Education, Ecology) and underpins **Alignment Infrastructure Routing (AIR)**, which tracks safety-related work and funding.

The kernel does not interpret content or set policy. It only provides shared state, verifiable provenance, and replayable measurement. Authority and accountability stay with humans at the application layer.

> **Why it matters:** Typical AI scales by approximation. This kernel scales by fixed geometry: same log always yields the same state, so coordination can be verified without trust.

---

## 🗝️ Core Capabilities

- **Exact integer compute:** The kernel and SDK work in 64-dimensional state spaces using packed integer arithmetic (no floating point). That keeps results deterministic and avoids rounding issues for the workloads it targets.
- **Shared coordination state:** Two parties with the same event log always compute the same state. Coordination is based on replay, not on timestamps or a trusted central authority.
- **Verifiable provenance:** The state space is finite and enumerable. Any claimed history can be checked by replaying the published log and comparing the final state, so provenance is computational rather than testimonial.

---

## 🧠 SDK and Native Compute

The repository includes an SDK that turns the kernel from a specification into runnable code. It gives three views of the same machine: state trajectories (replay and shared moments), a 6-bit “chirality” register with a fixed orthogonal transform (Walsh-Hadamard), and matrix-vector operations over the 64-dimensional state.

**GyroLabe** is the low-level execution layer. It provides byte- and state-level primitives on CPU, 64-point Walsh-Hadamard transforms, fixed-point packed tensor multiply, and an OpenCL backend for batched linear algebra (GEMV/GEMM). See the [Quantum Computing SDK Specification](docs/Gyroscopic_ASI_SDK_Quantum_Computing.md) and the [aQPU Verification Report](docs/reports/aQPU_Tests_Report_1.md) for full details.

---

## 🔭 GyroLabe: Calibration for Auditable Inference

GyroLabe is the native compute backend and neural model bridge for the aQPU kernel. It connects language models to the kernel's algebraic structure, turning opaque byte-level inference into a process that any party can replay, verify, and audit.

**The Trust Problem in AI:** When a language model generates text, nobody outside the operator can verify what the inference process actually did. Current safety approaches (RLHF, red-teaming, interpretability) work on outputs or weights. None of them produce a verifiable record of the inference computation itself.

**What GyroLabe provides:**
- **Algebraic byte annotation:** Every byte in the model's input and output is decomposed into its exact structural components (commutation class, family phase, operational payload) using the aQPU kernel.
- **Inference audit trail:** The kernel state at each byte position is determined entirely by the public transition law. Any party with the same bytes computes the same states. The audit trail requires no trust in the model operator and no access to proprietary model weights.
- **Trainable structural bias:** Small embedding biases based on the algebraic decomposition let the model learn to use the structural properties of bytes. The wrapped model is mathematically identical to the base model before training.
- **Bitplane compute engine:** A C/OpenCL backend that decomposes dense matrix-vector multiplication into Boolean operations, providing a path toward structurally transparent linear algebra.

**Current status:** GyroLabe is actively tested on Bolmo-1B (a byte-level language model). The algebraic annotation and embedding biases are operational. Dimensional scaling and full operator decomposition are in active development. 

Read the [GyroLabe Brief](docs/GyroLabe_Brief.md) for full details on its architecture and performance.

---

## 🔮 Holographic Architecture

The 4,096-state space has a rigid structure: two small **boundary sets** of 64 states each (called “horizons”) encode the full space in a precise way (**|H|² = |Ω|**, where Ω is the full state set). That is a consequence of the kernel’s transition rules, not an extra design choice.

**What that gives you:**
- **Compression:** Checking the small boundary sets is enough to guarantee consistency of the whole state space; you do not need to store or verify all 4,096 states explicitly.
- **Partitioning:** The space splits into four symmetric regions (each 2,048 states), which supports natural boundaries for distributed coordination.
- **History equivalence:** The geometry defines which different event sequences lead to the same effective outcome, so you can reason about trajectories without training a model.

The [SDK Network Specification](docs/Gyroscopic_ASI_SDK_Network.md) describes testing AI models against this geometry (“oracles”); the [Holographic Web Specification](docs/Gyroscopic_ASI_SDK_Holographic_Web.md) extends the same ideas to internet-scale coordination.

---

## ⚛️ Proven Computational Advantages

The kernel’s internal algebra uses binary arithmetic (bits) and satisfies properties analogous to quantum mechanics. Exhaustive tests over all 4,096 states confirm this.

**Structural properties:**
- **Reversibility (unitarity):** Every byte induces a one-to-one map on the full state; no information is lost or invented.
- **Order 4:** Applying the same byte four times always returns to the starting state.
- **No cloning:** One special byte (archetype 0xAA) is the unique “source” that all state paths can trace back to; no other byte can play that role.
- **Complementarity:** From any state, the 256 possible bytes lead to exactly 128 distinct next states (each reachable by two bytes), giving a symmetric branching structure.

**Concrete advantages over naive classical approaches:**

| Task | aQPU | Naive classical |
|------|------|------------------|
| Resolving hidden structure in the transition graph | 1 step (single lookup on 6-bit register) | Up to 64 steps |
| Exact uniform mixing | 2 steps, integer exact | ~12 steps for approximate mixing |
| Testing whether two operations commute | O(1), 6-bit compare | 4 kernel steps |
| Compressing state to a boundary + index | 8 bits | 24 bits |

**Error detection:** A built-in code ([12,6,2] over the state bits) detects all odd-weight bit errors. Tampering with a byte in the log is detected unless the replacement is an exact phase match (probability 1/255). 

---

## 🚛 Alignment Infrastructure Routing (AIR)

**AIR** applies the kernel to two coordination problems.

**Safety work and pay:** AIR helps labs, fiscal hosts (organisations that hold and disburse funds for projects), and contributors turn safety work (evaluations, red-teaming, interpretability, documentation) into paid, verifiable contributions. It uses the Gyroscope Protocol and **The Human Mark** (a scheme to tag content as human- vs machine-origin) to produce attested work receipts so sponsors can verify what was done without relying on informal reports.

**Governance logistics:** Tracking how information and authority move through decision systems is treated with the same rigour as supply chains. AIR provides full replayable histories (“genealogies”) and coherence metrics for governance quality, and supports verifiable compliance with standards such as ISO 42001 and the EU AI Act.

---

![Moments Economy Cover Image](/assets/moments_cover.png)

## 💰 Moments Economy

Moments Economy is a monetary design where value is tied to verified coordination capacity rather than debt. A fixed total supply, the **Common Source Moment (CSM)**, is derived once from the caesium-133 atomic time standard and the kernel’s finite state space (so the “budget” is physically anchored).

CSM is large enough to support a global **Unconditional High Income (UHI)**-style allocation, tiered distributions, and full governance records for billions of years. Within this cap, every settlement is a replayable, verifiable history rather than an opaque update on a central ledger.

---

## 📚 Documentation

### Start Here
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
- 📊 [Physics Tests Report](docs/reports/Physics_Tests_Report.md) - Kernel state verification
- 📊 [Moments Tests Report](docs/reports/Moments_Tests_Report.md) - Ledger replay tests
- 📊 [aQPU Verification Report](docs/reports/aQPU_Tests_Report_1.md) - Algebraic properties verified
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