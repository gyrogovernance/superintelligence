# Artificial Superintelligence Architecture (ASI/AGI)
> **Gyroscopic Alignment Models Lab**

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

# GGG ASI Alignment Router
Alignment Infrastructure Routing for Post‑AGI Coordination

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)

Status: research implementation. The kernel physics and measurement substrate are specified and tested. The AIR CLI provides a project-based coordination workflow.

## 🌐 Overview

GGG ASI Alignment Router is a multi-domain network coordination algorithm for focused and well-distributed coordination of contributions. As a collective superintelligence implementation, it is a network of humans and machines (Superintelligence, Bostrom 2014) amplifying rather than outperforming single agent potential, while preserving the constitutive conditions of governance and intelligibility (Gyroscopic Global Governance, Korompilias 2025).

The algorithm is a deterministic finite-state coordination system for routing and audit in human–AI governance settings. It maps an append-only byte ledger to a reproducible state trajectory and exports a compact routing signature and governance observables.

In this project, Artificial Superintelligence is treated as an operational regime: stable coordination across heterogeneous human and machine capabilities that maintains four constitutive governance principles across economy, employment, education, and ecology:
- **Governance Management Traceability**
- **Information Curation Variety**
- **Inference Interaction Accountability**
- **Intelligence Cooperation Integrity**

The router does not interpret content and does not decide policy. It provides shared moments, geometric provenance, and replayable measurement so that authorization and accountability remain under Original human agency at the application layer.

## 🗝️ Key ideas

- **Shared moments:** Participants who share the same archetype and the same ledger prefix compute the same kernel state at the same step. Coordination is grounded in reproducible computation rather than asserted metadata (timestamps, approvals, identity claims) or private model state.

- **Geometric provenance:** Valid states lie in a finite ontology Ω compiled into an atlas. Membership and replay are structurally checkable from artifacts.

- **Governance measurement substrate:** Application-layer GovernanceEvents update per-domain K4 edge ledgers. Hodge decomposition splits each ledger into gradient and cycle components, and aperture is computed as the cycle-energy fraction. This yields a replayable coordination metric independent of model internals.

---

## 🌐 Theoretical Foundation

The Router is grounded in:

- **Common Governance Model (CGM)** as the constitutional structure of coherent recursive operation.
- **The Human Mark (THM)** as the source-type ontology of Authority and Agency in sociotechnical systems.
- **Gyroscopic Global Governance (GGG)** as the four-domain coupling of Economy, Employment, Education, and Ecology.

The Router operates as a Derivative coordination system: it transforms and routes information but does not originate authority or bear accountability. Accountability terminates in Original Agency.

Mathematical formalism uses Hodge decomposition over K4 tetrahedral geometry, with face-cycle matrices aligned to BU commutator loops.

---

## 📚 Documentation

### Getting Started
- 📋 [**Alignment Infrastructure Routing (AIR) Brief**](docs/AIR_Brief.md) - Overview of AIR, workforce coordination, and operating model

### Technical Specifications
- 📖 [**GGG ASI Alignment Router** - Kernel Specifications](docs/GGG_ASI_AR_Specs.md) - Complete technical specification for implementation
- 🔮 [**Router Implications & Potential**](docs/GGG_ASI_AR_Implications.md) - Use cases and deployment scenarios
- 🧬 [**Substrate: Physical Memory Specification**](docs/GGG_ASI_AR_Substrate_Specs.md) - Future development: physical memory architecture

### Test Reports
- 📊 [**Physics Tests Report**](docs/reports/Physics_Tests_Report.md) - Verified structural properties and CGM-linked invariants
- 📊 [**Alignment Measurement Report**](docs/reports/Alignment_Measurement_Report.md) - Governance measurement substrate verification
- 📊 [**All Tests Results**](docs/reports/All_Tests_Results.md) - Comprehensive test suite results
- 📊 [**Other Tests Report**](docs/reports/Other_Tests_Report.md) - Additional test coverage and validation

### Supporting Theory
- 📖 [**Common Governance Model (CGM)**](docs/references/CGM_Paper.md) - Theoretical foundations
- 📖 [**The Human Mark (THM)**](docs/references/THM.md) - Source-type ontology overview
- 📖 [**The Human Mark - Paper**](docs/references/THM_Paper.md) - Complete THM specification
- 📖 [**The Human Mark - Grammar**](docs/references/THM_Grammar.md) - PEG specification for tagging and validation
- 📖 [**Gyroscopic Global Governance (GGG)**](docs/references/GGG_Paper.md) - Four-domain coupling framework

---

## Repository structure

- `src/router/` kernel physics, atlas builder, kernel runtime
- `src/app/` coordinator, governance events, domain ledgers, aperture
- `src/plugins/` analytics helpers, adapters, framework connectors
- `docs/` specifications and notes
- `src/tests/` exhaustive kernel and measurement verification

---

## 🚩 Quick start

### Install
Create an environment and install dependencies (NumPy is required; the rest are in the repo tooling).

### AIR CLI
The CLI provides a project-based, MD-first workflow:

```bash
python air_cli.py
```

This runs automatically: **Auto-build Atlas → Compile Projects → Generate Reports → Verify Bundles**. 

**Project Format:** Projects are markdown files in `data/projects/` using bracket notation. Copy `_template.md` to create a new project (e.g., `cp _template.md my-project.md`). See Section 5.4 of the specifications for the complete project format specification.

### AIR Console (Browser-based UI)
The Console provides a browser-based interface for managing project contracts:

```bash
# First-time setup: install dependencies
python air_installer.py

# Run the console (starts both backend and frontend)
python air_console.py
```

The console will be available at `http://localhost:5173` (frontend proxies API requests to backend on port 8000).

### Build the atlas (Manual)
The atlas compiles the kernel physics into three artifacts: ontology, epistemology, and phenomenology.

```bash
python -m src.router.atlas --out data/atlas
```

### Run tests
```bash
python -m pytest -v -s tests/
```

### Programmatic usage

```python
from pathlib import Path
from src.app.coordination import Coordinator
from src.app.events import Domain, EdgeID, GovernanceEvent

c = Coordinator(Path("data/atlas"))

# Shared-moment stepping
c.step_bytes(b"Hello world")

# Application-layer governance update (ledger event)
c.apply_event(
    GovernanceEvent(
        domain=Domain.ECONOMY,
        edge_id=EdgeID.GOV_INFO,
        magnitude=1.0,
        confidence=0.8,
        meta={"source": "example"},
    ),
    bind_to_kernel_moment=True,
)

status = c.get_status()
print(status.kernel)
print(status.apertures)
```

---

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 📖 Citation

```bibtex
@software{GGG_ASI_AR_2025,
  author = {Basil Korompilias},
  title = {GGG ASI Alignment Router},
  year = {2025},
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
  <p>Artificial intelligence was employed solely as a technical assistant, limited to code drafting, formatting, verification, and editorial services, always under direct human supervision.</p>
  <p>All foundational ideas, design decisions, and conceptual frameworks originate from the Author.</p>
  <p>Responsibility for the validity, coherence, and ethical direction of this project remains fully human.</p>
  <p><strong>Acknowledgements:</strong><br>
  This project benefited from AI language model services accessed through LMArena, Cursor IDE, OpenAI (ChatGPT), Anthropic (Opus), and Google (Gemini).</p>

