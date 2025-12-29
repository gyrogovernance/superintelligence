```markdown
# Agent Guide: ACI CLI Development

> **Document purpose**: This file guides AI assistants developing the Alignment Convergence Infrastructure (ACI) CLI. Follow this document precisely. Do not expand scope.

---

## 1. What ACI Is

ACI (Alignment Convergence Infrastructure) is a CLI tool for coordinating AI safety work contributions. It provides:

1. **Shared moments** via deterministic kernel stepping
2. **Governance events** bound to those moments
3. **Aperture computation** from domain ledgers
4. **Replayable bundles** for sponsor verification

ACI does **not** evaluate work, process deliverables, or validate content. It provides the coordination substrate and audit trail. Evaluation happens externally by sponsors.

---

## 2. Theoretical Foundation (Do Not Deviate)

### Kernel Shared Moments

The GGG ASI Router kernel provides deterministic state transitions. Each byte input advances the kernel by one step, producing a new shared moment.

**Canonical time mapping:**

| Work Unit | Duration | Kernel Steps | Bytes |
|-----------|----------|--------------|-------|
| Daily | 1 day | 1 | `0x01` |
| Sprint | 4 days | 4 | `0x01 0x02 0x03 0x04` |

Moments are dimensionless. We use 1-day and 4-day as the two canonical dimensionful references.

### Governance Events

Events update K₄ edge ledgers per domain (Economy, Employment, Education). Events are bound to kernel moments for audit trail.

### Aperture

Aperture measures cycle-to-total energy ratio via Hodge decomposition on K₄. Target aperture A* ≈ 0.0207.

---

ASCI Branding for our CLI

```            
┏━┓╻  ╻┏━╸┏┓╻┏┳┓┏━╸┏┓╻╺┳╸               
┣━┫┃  ┃┃╺┓┃┗┫┃┃┃┣╸ ┃┗┫ ┃                
╹ ╹┗━╸╹┗━┛╹ ╹╹ ╹┗━╸╹ ╹ ╹                
┏━╸┏━┓┏┓╻╻ ╻┏━╸┏━┓┏━╸┏━╸┏┓╻┏━╸┏━╸       
┃  ┃ ┃┃┗┫┃┏┛┣╸ ┣┳┛┃╺┓┣╸ ┃┗┫┃  ┣╸        
┗━╸┗━┛╹ ╹┗┛ ┗━╸╹┗╸┗━┛┗━╸╹ ╹┗━╸┗━╸       
╻┏┓╻┏━╸┏━┓┏━┓┏━┓╺┳╸┏━┓╻ ╻┏━╸╺┳╸╻ ╻┏━┓┏━╸
┃┃┗┫┣╸ ┣┳┛┣━┫┗━┓ ┃ ┣┳┛┃ ┃┃   ┃ ┃ ┃┣┳┛┣╸ 
╹╹ ╹╹  ╹┗╸╹ ╹┗━┛ ╹ ╹┗╸┗━┛┗━╸ ╹ ┗━┛╹┗╸┗━╸                           
```

## 3. Existing Codebase (Use These, Do Not Duplicate)

### Router Layer (`src/router/`)

```python
from src.router.kernel import RouterKernel
from src.router.atlas import build_all, AtlasPaths
from src.router.constants import ARCHETYPE_STATE24, GENE_MIC_S
```

- `RouterKernel(atlas_dir)` — loads atlas, provides `step_byte()`, `signature()`
- `build_all(base_dir)` — builds ontology, epistemology, phenomenology

### App Layer (`src/app/`)

```python
from src.app.coordination import Coordinator, CoordinationStatus
from src.app.events import GovernanceEvent, Domain, EdgeID
from src.app.ledger import DomainLedgers, compute_aperture, hodge_decomposition
```

- `Coordinator(atlas_dir)` — owns kernel + ledgers, provides `step_byte()`, `apply_event()`, `get_status()`
- `GovernanceEvent` — immutable event with domain, edge_id, magnitude, confidence, meta
- `DomainLedgers` — holds three 6-element edge vectors, computes aperture

### Plugins (`src/plugins/`)

```python
from src.plugins.frameworks import THMDisplacementPlugin, GyroscopeWorkMixPlugin, PluginContext
```

- `THMDisplacementPlugin().emit_events(payload, ctx)` — converts THM signals to GovernanceEvents
- `GyroscopeWorkMixPlugin().emit_events(payload, ctx)` — converts Gyroscope signals to GovernanceEvents

---

## 4. CLI Folder Structure

All CLI code lives in one folder:

```
src/app/cli/
  __init__.py
  main.py        # argparse commands + optional interactive menu
  store.py       # workspace I/O, frontmatter parsing, replay
  ui.py          # ANSI formatting, tables, prompts
  schemas.py     # constants (A*, canonical bytes), validation helpers
```

Root runner:

```
aci.py          # entry point: calls src.app.cli.main.main()
```

---

## 5. Data Model (Markdown + Frontmatter)

All human-editable config uses `.md` files with YAML frontmatter. Binary/append-only logs are separate.

### Workspace Layout

```
.aci/
  projects/
    <project_slug>/
      project.md
      runs/
        <run_id>/
          run.md
          bytes.bin       # append-only binary
          events.jsonl    # append-only JSON lines
```

### `project.md` Template

```markdown
---
project_name: Example Lab Pilot
atlas_dir: data/atlas
sponsor: Example Safety Lab
created_at: 2025-12-29T12:00:00Z
---

# Example Lab Pilot

Project notes go here. Contributors and sponsors can edit freely.
```

### `run.md` Template

```markdown
---
run_id: 2025-12-29_daily_001
unit: daily
title: Jailbreak eval set #12
domain_focus: education
status: active
contributor_id: ""
day_index: 1
created_at: 2025-12-29T12:05:00Z
links: []
thm:
  primary: ""
  secondary: []
  grammar: ""
---

# Jailbreak eval set #12

## Notes

Describe what you did.

## Evidence

Add links to PRs, files, external resources.
```

### `bytes.bin`

Raw bytes, append-only. For daily: 1 byte. For sprint: up to 4 bytes.

### `events.jsonl`

One JSON object per line, matching `GovernanceEvent.as_dict()`:

```json
{"domain":2,"edge_id":0,"magnitude":0.1,"confidence":0.9,"meta":{"plugin":"thm_displacement","signal":"GTD"},"kernel_state_index":123,"kernel_last_byte":1}
```

---

## 6. CLI Commands (Complete Surface)

### Atlas

```bash
aci atlas build --out data/atlas
aci atlas doctor --atlas data/atlas
```

### Projects

```bash
aci project init "Name" --atlas data/atlas [--sponsor "..."]
aci project ls
aci project show <project_slug>
```

### Runs

```bash
aci run start daily --project <slug> --title "..." --domain education
aci run start sprint --project <slug> --title "..." --domain employment
aci run ls --project <slug>
aci run status <run_id>
aci run next-day <run_id>     # sprint only: steps to next day
aci run close <run_id>        # marks status=closed
```

**On `run start`:**
- Create run folder
- Write `run.md` with frontmatter scaffold
- Step kernel once (daily) or once for day 1 (sprint)
- Write byte(s) to `bytes.bin`

**On `run next-day`:**
- Validate run is sprint and day_index < 4
- Step kernel once with next canonical byte
- Append byte to `bytes.bin`
- Increment day_index in `run.md` frontmatter

### Events (Manual)

```bash
aci event add <run_id> --domain education --edge GOV_INFO --mag 0.1 [--conf 0.9] [--note "..."]
```

- Creates `GovernanceEvent`
- Appends to `events.jsonl`
- Binds to current kernel moment

### Plugins (Internal)

```bash
aci plugin thm <run_id> --domain education --GTD 0.1 --IAD 0.05 [--confidence 0.9]
aci plugin gyro <run_id> --GM 0.1 --ICu -0.1 [--confidence 0.8]
```

- Calls existing plugin `.emit_events()`
- Appends resulting events to `events.jsonl`

### Bundles

```bash
aci bundle make <run_id> --out bundles/<run_id>.zip
aci bundle verify <path_to_zip>
```

**Bundle contents:**
- `project.md`
- `run.md`
- `bytes.bin`
- `events.jsonl`
- `bundle.json` (generated snapshot)

**`bundle.json` schema:**

```json
{
  "run_id": "2025-12-29_daily_001",
  "generated_at": "2025-12-29T18:00:00Z",
  "kernel": {
    "state_index": 12345,
    "state_hex": "0a1b2c",
    "a_hex": "0aa",
    "b_hex": "555",
    "last_byte": 1
  },
  "logs": {
    "byte_count": 1,
    "event_count": 3,
    "bytes_sha256": "abc123...",
    "events_sha256": "def456..."
  },
  "apertures": {
    "economy": 0.0184,
    "employment": 0.0211,
    "education": 0.0340
  }
}
```

**Verify process:**
1. Unpack to temp directory
2. Create fresh `Coordinator`
3. Replay `bytes.bin` via `step_byte()`
4. Replay `events.jsonl` via `apply_event()`
5. Compute final status
6. Compare with `bundle.json`
7. Print PASS or FAIL

---

## 7. Implementation Details

### `store.py` Core Functions

```python
from pathlib import Path
from typing import Tuple, Dict, Any
import re
import json
import hashlib

# Use PyYAML if available, otherwise implement minimal parser
try:
    import yaml
except ImportError:
    yaml = None  # fallback to minimal parser


def get_workspace_root() -> Path:
    """Returns .aci/ in current working directory."""
    return Path.cwd() / ".aci"


def parse_frontmatter(path: Path) -> Tuple[Dict[str, Any], str]:
    """Parse markdown file with YAML frontmatter."""
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.+?)\n---\n?(.*)", text, re.DOTALL)
    if not match:
        return {}, text
    if yaml:
        fm = yaml.safe_load(match.group(1))
    else:
        fm = _minimal_yaml_parse(match.group(1))
    return fm, match.group(2)


def write_frontmatter(path: Path, meta: Dict[str, Any], body: str) -> None:
    """Write markdown file with YAML frontmatter."""
    if yaml:
        fm_str = yaml.dump(meta, default_flow_style=False, sort_keys=False)
    else:
        fm_str = _minimal_yaml_dump(meta)
    path.write_text(f"---\n{fm_str}---\n{body}", encoding="utf-8")


def append_bytes(path: Path, data: bytes) -> None:
    """Append bytes to binary log."""
    with open(path, "ab") as f:
        f.write(data)


def read_bytes(path: Path) -> bytes:
    """Read all bytes from binary log."""
    if not path.exists():
        return b""
    return path.read_bytes()


def append_event(path: Path, event_dict: Dict[str, Any]) -> None:
    """Append event as JSON line."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event_dict) + "\n")


def read_events(path: Path) -> list:
    """Read all events from JSONL file."""
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").strip().split("\n"):
        if line:
            events.append(json.loads(line))
    return events


def file_sha256(path: Path) -> str:
    """Compute SHA256 hash of file."""
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay_run(atlas_dir: Path, run_dir: Path) -> "CoordinationStatus":
    """Replay bytes + events to reconstruct state."""
    from src.app.coordination import Coordinator
    from src.app.events import GovernanceEvent, Domain, EdgeID
    
    coord = Coordinator(atlas_dir)
    
    # Replay bytes
    for b in read_bytes(run_dir / "bytes.bin"):
        coord.step_byte(b)
    
    # Replay events
    for ev_dict in read_events(run_dir / "events.jsonl"):
        ev = GovernanceEvent(
            domain=Domain(ev_dict["domain"]),
            edge_id=EdgeID(ev_dict["edge_id"]),
            magnitude=ev_dict["magnitude"],
            confidence=ev_dict.get("confidence", 1.0),
            meta=ev_dict.get("meta", {}),
        )
        coord.apply_event(ev, bind_to_kernel_moment=False)
    
    return coord.get_status()
```

### `schemas.py` Constants

```python
# Canonical target aperture (from CGM)
A_STAR = 0.0207

# Canonical byte sequences for work units
DAILY_BYTES = bytes([0x01])
SPRINT_BYTES = bytes([0x01, 0x02, 0x03, 0x04])

# Domain name mapping
DOMAIN_NAMES = {
    "economy": 0,
    "employment": 1,
    "education": 2,
}

# Edge name mapping (matches EdgeID enum)
EDGE_NAMES = {
    "GOV_INFO": 0,
    "GOV_INFER": 1,
    "GOV_INTEL": 2,
    "INFO_INFER": 3,
    "INFO_INTEL": 4,
    "INFER_INTEL": 5,
}
```

### `ui.py` Helpers

```python
import sys

# ANSI codes (disable if not TTY)
USE_COLOR = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    if not USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"

def bold(text: str) -> str:
    return _c("1", text)

def green(text: str) -> str:
    return _c("32", text)

def yellow(text: str) -> str:
    return _c("33", text)

def red(text: str) -> str:
    return _c("31", text)

def cyan(text: str) -> str:
    return _c("36", text)

def header(title: str) -> str:
    return f"\n{bold('ACI')} — Alignment Convergence Infrastructure\n{bold(title)}\n"

def kv(key: str, value: str, indent: int = 0) -> str:
    pad = "  " * indent
    return f"{pad}{key}: {cyan(str(value))}"

def table(rows: list, headers: list) -> str:
    """Simple fixed-width table."""
    widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    lines = []
    hdr = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    lines.append(bold(hdr))
    lines.append("-" * len(hdr))
    for row in rows:
        lines.append("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))
    return "\n".join(lines)

def success(msg: str) -> None:
    print(green("✓ " + msg))

def error(msg: str) -> None:
    print(red("✗ " + msg))

def warn(msg: str) -> None:
    print(yellow("! " + msg))
```

### `main.py` Structure

```python
import argparse
import sys
from pathlib import Path

from . import store, ui, schemas


def cmd_atlas_build(args):
    from src.router.atlas import build_all
    build_all(Path(args.out))
    ui.success(f"Atlas built at {args.out}")


def cmd_atlas_doctor(args):
    # Load and verify atlas
    ...


def cmd_project_init(args):
    # Create project folder + project.md
    ...


def cmd_run_start(args):
    # Create run folder + run.md
    # Step kernel with canonical bytes
    # Write bytes.bin
    ...


def cmd_run_status(args):
    # Replay and print status
    ...


def cmd_event_add(args):
    # Create event, append to jsonl
    ...


def cmd_plugin_thm(args):
    # Call THMDisplacementPlugin, append events
    ...


def cmd_bundle_make(args):
    # Zip run folder + bundle.json
    ...


def cmd_bundle_verify(args):
    # Unpack, replay, compare
    ...


def main():
    parser = argparse.ArgumentParser(
        prog="aci",
        description="Alignment Convergence Infrastructure CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Atlas commands
    atlas = subparsers.add_parser("atlas")
    atlas_sub = atlas.add_subparsers(dest="subcommand", required=True)
    
    atlas_build = atlas_sub.add_parser("build")
    atlas_build.add_argument("--out", default="data/atlas")
    atlas_build.set_defaults(func=cmd_atlas_build)
    
    atlas_doctor = atlas_sub.add_parser("doctor")
    atlas_doctor.add_argument("--atlas", default="data/atlas")
    atlas_doctor.set_defaults(func=cmd_atlas_doctor)

    # Project commands
    project = subparsers.add_parser("project")
    project_sub = project.add_subparsers(dest="subcommand", required=True)
    
    project_init = project_sub.add_parser("init")
    project_init.add_argument("name")
    project_init.add_argument("--atlas", default="data/atlas")
    project_init.add_argument("--sponsor", default="")
    project_init.set_defaults(func=cmd_project_init)
    
    # ... etc for all commands

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
```

### `aci.py` (Root Runner)

```python
#!/usr/bin/env python3
"""ACI CLI entry point."""

from src.app.cli.main import main

if __name__ == "__main__":
    main()
```

---

## 8. Scope Boundaries (Hard Rules)

### DO

- Use existing `Coordinator`, `RouterKernel`, `DomainLedgers`, plugins
- Store logs as append-only files
- Reconstruct state by replay
- Compute apertures from ledgers
- Generate bundles for external verification

### DO NOT

- Evaluate deliverables or datasets
- Parse or validate deliverable content
- Add external dependencies beyond PyYAML (optional)
- Create complex plugin architectures
- Add web servers or async code
- Expand command surface beyond what's listed
- Process files uploaded by contributors

---

## 9. Status Output Format

When implementing `aci run status`, output should look like:

```
ACI — Alignment Convergence Infrastructure
Run: 2025-12-29_daily_001

Unit:        daily
Domain:      education
Status:      active
Contributor: gyro:basil
Day:         1/1

Kernel
  state_index: 12345
  state_hex:   0a1b2c
  A: 0aa  B: 555
  last_byte:   01

Logs
  bytes:  1
  events: 3

Apertures
  Economy:    0.0184
  Employment: 0.0211
  Education:  0.0340

Target A*: 0.0207
```

---

## 10. Verification Output Format

When implementing `aci bundle verify`, output should look like:

**Pass:**
```
ACI — Bundle Verification

Bundle: bundles/2025-12-29_daily_001.zip
Run:    2025-12-29_daily_001

Replaying...
  bytes:  1 ✓
  events: 3 ✓

Comparing final state...
  kernel:    ✓ match
  apertures: ✓ match
  hashes:    ✓ match

✓ PASS — Bundle verified successfully
```

**Fail:**
```
ACI — Bundle Verification

Bundle: bundles/2025-12-29_daily_001.zip
Run:    2025-12-29_daily_001

Replaying...
  bytes:  1 ✓
  events: 3 ✓

Comparing final state...
  kernel:    ✗ mismatch
    expected state_index: 12345
    computed state_index: 12346
  apertures: ✓ match
  hashes:    ✓ match

✗ FAIL — Bundle verification failed
```

---

## 11. Testing Strategy

### Unit Tests

Test each store function:
- `parse_frontmatter` / `write_frontmatter`
- `append_bytes` / `read_bytes`
- `append_event` / `read_events`
- `replay_run`

### Integration Tests

Test full workflows:
1. `atlas build` → `project init` → `run start daily` → `event add` → `status` → `bundle make` → `bundle verify`
2. Same for sprint with `next-day` steps

### Replay Determinism

Verify that:
- Same bytes + events → same final state
- Bundle verification passes for correctly generated bundles
- Bundle verification fails when logs are tampered

---

## 12. Implementation Order

Build in this sequence:

1. **`schemas.py`** — constants only, no logic
2. **`store.py`** — file I/O + replay function
3. **`ui.py`** — formatting helpers
4. **`main.py`** — commands in order:
   - `atlas build`, `atlas doctor`
   - `project init`, `project ls`, `project show`
   - `run start`, `run ls`, `run status`
   - `run next-day`, `run close`
   - `event add`
   - `plugin thm`, `plugin gyro`
   - `bundle make`, `bundle verify`
5. **`aci.py`** — root runner

---

## 13. Dependencies

### Required
- Python 3.10+
- numpy (already in project)

### Optional
- PyYAML (for cleaner frontmatter parsing)

If PyYAML is unavailable, implement minimal YAML parser for flat key-value + simple lists. Do not add other dependencies.

---

## 14. Reference: Example Session

```bash
# Setup
python aci.py atlas build --out data/atlas
python aci.py project init "Safety Lab Q1" --atlas data/atlas --sponsor "Example Lab"

# Daily run
python aci.py run start daily --project safety-lab-q1 --title "Jailbreak batch #7" --domain education
# User edits .aci/projects/safety-lab-q1/runs/2025-01-15_daily_001/run.md
python aci.py plugin thm 2025-01-15_daily_001 --domain education --IAD 0.15 --confidence 0.9
python aci.py run status 2025-01-15_daily_001
python aci.py run close 2025-01-15_daily_001
python aci.py bundle make 2025-01-15_daily_001 --out bundles/2025-01-15_daily_001.zip

# Sponsor verifies
python aci.py bundle verify bundles/2025-01-15_daily_001.zip

# Sprint run
python aci.py run start sprint --project safety-lab-q1 --title "Circuit analysis" --domain employment
python aci.py plugin gyro 2025-01-15_sprint_001 --GM 0.1 --ICu -0.05
python aci.py run next-day 2025-01-15_sprint_001
python aci.py event add 2025-01-15_sprint_001 --domain employment --edge INFER_INTEL --mag 0.08
python aci.py run next-day 2025-01-15_sprint_001
python aci.py run next-day 2025-01-15_sprint_001
python aci.py run status 2025-01-15_sprint_001
python aci.py bundle make 2025-01-15_sprint_001 --out bundles/2025-01-15_sprint_001.zip
```

---

**End of agent guide. Follow this document. Do not expand scope.**
```