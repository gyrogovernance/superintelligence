```markdown
# agent.md – AIR Governance Console

This document specifies the AIR Governance Console, a browser-based UI for managing AIR program contracts. The console is a thin view layer over existing AIR logic. It does not reimplement any parsing, attestation, or computation.

---

## 1. Architecture Principle

The console is a **remote control** for `src/app`. It does not define new data structures or logic. It:

1. Reads artifacts produced by existing code.
2. Displays them visually.
3. Writes user edits back to markdown files.
4. Calls existing sync functions to recompute.

All data structures in the frontend must mirror the shapes already defined in Python. The backend imports and calls existing modules directly.

---

## 2. Existing Code (Do Not Modify)

The following modules are the source of truth. The console wraps them.

### 2.1 CLI and Store

**`src/app/cli/store.py`** provides:

- `get_programs_dir()` – returns `data/programs/` path.
- `get_aci_dir()` – returns `data/programs/.aci/` path.
- `get_bundles_dir()` – returns `data/programs/bundles/` path.
- `get_atlas_dir()` – returns `data/atlas/` path.
- `parse_program_from_markdown(path)` – returns `(slug, domain_counts, principle_counts, unit)`.
- `ensure_program_id(slug)` – returns stable UUID, creates if missing.
- `sync_program(atlas_dir, program_path)` – compiles attestations, writes `.bytes`, `.events.jsonl`, `.report.json`, `.report.md`.
- `bundle_program(atlas_dir, program_path)` – creates verified `.zip` bundle.
- `verify_bundle(atlas_dir, bundle_path)` – returns boolean.

**`src/app/cli/templates.py`** provides:

- `PROJECT_TEMPLATE_MD` – the canonical markdown template string.

**`src/app/cli/schemas.py`** provides:

- `A_STAR = 0.0207` – the canonical target aperture.

### 2.2 Coordination

**`src/app/coordination.py`** provides:

- `Coordinator` – steps the kernel and applies events.
- `CoordinationStatus` – dataclass with `kernel`, `ledgers`, `apertures`.

**`src/app/events.py`** provides:

- `Domain` enum: `ECONOMY = 0`, `EMPLOYMENT = 1`, `EDUCATION = 2`.
- `EdgeID` enum for K₄ edges.
- `GovernanceEvent` dataclass.

**`src/app/ledger.py`** provides:

- `DomainLedgers` – holds edge vectors and computes apertures.
- `compute_aperture()` – Hodge decomposition.

### 2.3 Router

**`src/router/kernel.py`** provides:

- `RouterKernel` – steps through atlas states.
- `Signature` dataclass: `step`, `state_index`, `state_hex`, `a_hex`, `b_hex`.

**`src/router/atlas.py`** provides:

- `build_all()` – builds ontology, epistemology, phenomenology.

### 2.4 Report Structure

`sync_program` writes `.aci/{slug}.report.json` with this structure:

```python
{
    "program_slug": str,
    "program_id": str,
    "compilation": {
        "attestation_count": int,
        "processed_attestations": int,
        "skipped_attestations": list,
        "byte_count": int,
        "kernel": {
            "step": int,
            "state_index": int,
            "state_hex": str,
            "a_hex": str,
            "b_hex": str,
            "last_byte": int
        },
        "hashes": {
            "bytes_sha256": str,
            "events_sha256": str
        }
    },
    "accounting": {
        "thm": {
            "totals": {"GTD": int, "IVD": int, "IAD": int, "IID": int},
            "by_domain": {
                "economy": {"GTD": int, "IVD": int, "IAD": int, "IID": int},
                "employment": {"GTD": int, "IVD": int, "IAD": int, "IID": int},
                "education": {"GTD": int, "IVD": int, "IAD": int, "IID": int}
            }
        },
        "gyroscope": {
            "totals": {"GMT": int, "ICV": int, "IIA": int, "ICI": int},
            "by_domain": {
                "economy": {"GMT": int, "ICV": int, "IIA": int, "ICI": int},
                "employment": {"GMT": int, "ICV": int, "IIA": int, "ICI": int},
                "education": {"GMT": int, "ICV": int, "IIA": int, "ICI": int}
            }
        }
    },
    "ledger": {
        "y_econ": list[float],  # 6 elements
        "y_emp": list[float],
        "y_edu": list[float]
    },
    "apertures": {
        "A_econ": float,
        "A_emp": float,
        "A_edu": float
    },
    "warnings": {}  # or dict with warning details
}
```

The frontend must use this exact structure. Do not flatten or rename fields.

---

## 3. File Structure

```
src/app/console/
├── api/
│   ├── __init__.py
│   └── server.py
├── ui/
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   ├── index.html
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── api.ts
│       ├── types.ts
│       ├── theme.ts
│       ├── components/
│       │   ├── Header.tsx
│       │   ├── BalancePanel.tsx
│       │   ├── DomainsPanel.tsx
│       │   ├── PrinciplesPanel.tsx
│       │   ├── Stepper.tsx
│       │   ├── KernelPanel.tsx
│       │   ├── ReportPanel.tsx
│       │   ├── GlossaryModal.tsx
│       │   └── ConfirmModal.tsx
│       └── index.css
└── README.md
```

---

## 4. Backend Specification

The backend is a thin HTTP layer that imports and calls existing modules.

### 4.1 Server Setup

```python
# src/app/console/api/server.py

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from pydantic import BaseModel
import json
import re

from src.app.cli import store, templates
from src.app.cli.schemas import A_STAR

app = FastAPI()

# Paths
def programs_dir() -> Path:
    return store.get_programs_dir()

def aci_dir() -> Path:
    return store.get_aci_dir()

def bundles_dir() -> Path:
    return store.get_bundles_dir()

def atlas_dir() -> Path:
    return store.get_atlas_dir()
```

### 4.2 List Programs

```python
@app.get("/api/programs")
def list_programs():
    programs = []
    for f in sorted(programs_dir().glob("*.md")):
        if f.name.startswith("_"):
            continue
        slug = f.stem
        program_id = None
        id_path = aci_dir() / f"{slug}.id"
        if id_path.exists():
            program_id = id_path.read_text(encoding="utf-8").strip()
        programs.append({"slug": slug, "program_id": program_id})
    return {"programs": programs}
```

### 4.3 Create Program

```python
class CreateProgramRequest(BaseModel):
    slug: str

@app.post("/api/programs")
def create_program(req: CreateProgramRequest):
    slug = req.slug.strip().lower()
    
    # Validate slug
    if not re.match(r'^[a-z0-9][a-z0-9\-]*[a-z0-9]$|^[a-z0-9]$', slug):
        raise HTTPException(400, "Invalid slug. Use lowercase letters, numbers, and hyphens.")
    
    program_path = programs_dir() / f"{slug}.md"
    if program_path.exists():
        raise HTTPException(400, "Program already exists.")
    
    # Write template
    program_path.write_text(templates.PROJECT_TEMPLATE_MD, encoding="utf-8")
    
    # Generate ID
    program_id = store.ensure_program_id(slug)
    
    # Sync to create initial artifacts
    store.sync_program(atlas_dir(), program_path)
    
    return {"status": "created", "slug": slug, "program_id": program_id}
```

### 4.4 Get Program

```python
@app.get("/api/programs/{slug}")
def get_program(slug: str):
    program_path = programs_dir() / f"{slug}.md"
    if not program_path.exists():
        raise HTTPException(404, "Program not found.")
    
    # Parse editable fields from markdown
    parsed_slug, domain_counts, principle_counts, unit = store.parse_program_from_markdown(program_path)
    
    # Read computed report
    report_path = aci_dir() / f"{slug}.report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = None
    
    return {
        "editable": {
            "slug": parsed_slug,
            "unit": unit,
            "domain_counts": domain_counts,
            "principle_counts": principle_counts
        },
        "report": report
    }
```

### 4.5 Update Program

The update endpoint writes to the markdown file using the same regex patterns as the parser.

```python
class UpdateProgramRequest(BaseModel):
    unit: str
    domain_counts: dict
    principle_counts: dict

@app.put("/api/programs/{slug}")
def update_program(slug: str, req: UpdateProgramRequest):
    program_path = programs_dir() / f"{slug}.md"
    if not program_path.exists():
        raise HTTPException(404, "Program not found.")
    
    content = program_path.read_text(encoding="utf-8")
    
    # Update domain counts
    content = re.sub(
        r'(Economy[^:]*:\s*)\[(\d+)\]',
        rf'\g<1>[{req.domain_counts["economy"]}]',
        content
    )
    content = re.sub(
        r'(Employment[^:]*:\s*)\[(\d+)\]',
        rf'\g<1>[{req.domain_counts["employment"]}]',
        content
    )
    content = re.sub(
        r'(Education[^:]*:\s*)\[(\d+)\]',
        rf'\g<1>[{req.domain_counts["education"]}]',
        content
    )
    
    # Update unit
    content = re.sub(
        r'(Unit:\s*)\[(daily|sprint)\]',
        rf'\g<1>[{req.unit}]',
        content,
        flags=re.IGNORECASE
    )
    
    # Update principle counts
    for abbrev in ["GMT", "GTD", "ICV", "IVD", "IIA", "IAD", "ICI", "IID"]:
        if abbrev in ["GMT", "ICV", "IIA", "ICI"]:
            pattern = rf'({abbrev}\s+Alignment\s+Incidents:\s*)\[(\d+)\]'
        else:
            pattern = rf'({abbrev}\s+Displacement\s+Incidents:\s*)\[(\d+)\]'
        content = re.sub(
            pattern,
            rf'\g<1>[{req.principle_counts[abbrev]}]',
            content,
            flags=re.IGNORECASE
        )
    
    # Write updated content
    program_path.write_text(content, encoding="utf-8")
    
    # Sync immediately
    store.sync_program(atlas_dir(), program_path)
    
    # Return updated state
    return get_program(slug)
```

### 4.6 Delete Program

```python
@app.delete("/api/programs/{slug}")
def delete_program(slug: str):
    program_path = programs_dir() / f"{slug}.md"
    if not program_path.exists():
        raise HTTPException(404, "Program not found.")
    
    # Remove markdown
    program_path.unlink()
    
    # Remove artifacts
    for ext in [".bytes", ".events.jsonl", ".report.json", ".report.md", ".id"]:
        artifact = aci_dir() / f"{slug}{ext}"
        if artifact.exists():
            artifact.unlink()
    
    # Remove bundle
    bundle = bundles_dir() / f"{slug}.zip"
    if bundle.exists():
        bundle.unlink()
    
    return {"status": "deleted"}
```

### 4.7 Download Bundle

```python
@app.get("/api/programs/{slug}/bundle")
def download_bundle(slug: str):
    bundle_path = bundles_dir() / f"{slug}.zip"
    if not bundle_path.exists():
        raise HTTPException(404, "Bundle not found. Sync the program first.")
    return FileResponse(bundle_path, filename=f"{slug}.zip", media_type="application/zip")
```

### 4.8 Glossary

```python
@app.get("/api/glossary")
def get_glossary():
    return {
        "A_STAR": A_STAR,
        "domains": {
            "economy": {
                "name": "Economy",
                "description": "The domain of CGM operations and systemic resource flows."
            },
            "employment": {
                "name": "Employment",
                "description": "The domain of Gyroscope work categories and human contribution patterns."
            },
            "education": {
                "name": "Education",
                "description": "The domain of THM capacities and epistemic development."
            }
        },
        "alignment": {
            "GMT": {
                "name": "Governance Management Traceability",
                "description": "The capacity to understand and maintain the chain of authority from human sources to outputs."
            },
            "ICV": {
                "name": "Information Curation Variety",
                "description": "The capacity to recognise and preserve diversity in information sources."
            },
            "IIA": {
                "name": "Inference Interaction Accountability",
                "description": "The capacity to accept responsibility for conclusions and decisions."
            },
            "ICI": {
                "name": "Intelligence Cooperation Integrity",
                "description": "The capacity to maintain coherent reasoning over time and context."
            }
        },
        "displacement": {
            "GTD": {
                "name": "Governance Traceability Displacement",
                "risk": "Approaching Indirect Authority and Agency as Direct.",
                "description": "Occurs when a derivative system is treated as if it were an autonomous direct source."
            },
            "IVD": {
                "name": "Information Variety Displacement",
                "risk": "Approaching Indirect Authority without Agency as Direct.",
                "description": "Occurs when derivative authority is treated as direct authority."
            },
            "IAD": {
                "name": "Inference Accountability Displacement",
                "risk": "Approaching Indirect Agency without Authority as Direct.",
                "description": "Occurs when derivative agency is treated as direct agency."
            },
            "IID": {
                "name": "Intelligence Integrity Displacement",
                "risk": "Approaching Direct Authority and Agency as Indirect.",
                "description": "Occurs when direct authority and agency are devalued as inferior to derivative processing."
            }
        }
    }
```

---

## 5. Frontend Types

The frontend types must mirror the backend response shapes exactly.

```typescript
// src/types.ts

// Matches store.parse_program_from_markdown output
export interface EditableState {
  slug: string;
  unit: 'daily' | 'sprint';
  domain_counts: {
    economy: number;
    employment: number;
    education: number;
  };
  principle_counts: {
    GMT: number;
    GTD: number;
    ICV: number;
    IVD: number;
    IIA: number;
    IAD: number;
    ICI: number;
    IID: number;
  };
}

// Matches .report.json structure exactly
export interface ReportKernel {
  step: number;
  state_index: number;
  state_hex: string;
  a_hex: string;
  b_hex: string;
  last_byte: number;
}

export interface ReportHashes {
  bytes_sha256: string;
  events_sha256: string;
}

export interface ReportCompilation {
  attestation_count: number;
  processed_attestations: number;
  skipped_attestations: Array<{ index: number; id: string | null; reason: string }>;
  byte_count: number;
  kernel: ReportKernel;
  hashes: ReportHashes;
}

export interface THMTotals {
  GTD: number;
  IVD: number;
  IAD: number;
  IID: number;
}

export interface GyroTotals {
  GMT: number;
  ICV: number;
  IIA: number;
  ICI: number;
}

export interface DomainBreakdown<T> {
  economy: T;
  employment: T;
  education: T;
}

export interface ReportAccounting {
  thm: {
    totals: THMTotals;
    by_domain: DomainBreakdown<THMTotals>;
  };
  gyroscope: {
    totals: GyroTotals;
    by_domain: DomainBreakdown<GyroTotals>;
  };
}

export interface ReportLedger {
  y_econ: number[];
  y_emp: number[];
  y_edu: number[];
}

export interface ReportApertures {
  A_econ: number;
  A_emp: number;
  A_edu: number;
}

export interface Report {
  program_slug: string;
  program_id: string;
  compilation: ReportCompilation;
  accounting: ReportAccounting;
  ledger: ReportLedger;
  apertures: ReportApertures;
  warnings: Record<string, unknown>;
}

// API response shape
export interface ProgramResponse {
  editable: EditableState;
  report: Report | null;
}

export interface ProgramSummary {
  slug: string;
  program_id: string | null;
}

export interface ProgramListResponse {
  programs: ProgramSummary[];
}

// Glossary
export interface GlossaryEntry {
  name: string;
  description: string;
  risk?: string;
}

export interface Glossary {
  A_STAR: number;
  domains: Record<string, GlossaryEntry>;
  alignment: Record<string, GlossaryEntry>;
  displacement: Record<string, GlossaryEntry>;
}
```

---

## 6. Frontend API Client

```typescript
// src/api.ts

import type {
  ProgramListResponse,
  ProgramResponse,
  EditableState,
  Glossary
} from './types';

const BASE = '/api';

export async function listPrograms(): Promise<ProgramListResponse> {
  const res = await fetch(`${BASE}/programs`);
  if (!res.ok) throw new Error('Failed to list programs');
  return res.json();
}

export async function getProgram(slug: string): Promise<ProgramResponse> {
  const res = await fetch(`${BASE}/programs/${slug}`);
  if (!res.ok) throw new Error('Failed to load program');
  return res.json();
}

export async function createProgram(slug: string): Promise<{ status: string; slug: string; program_id: string }> {
  const res = await fetch(`${BASE}/programs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ slug })
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Failed to create program');
  }
  return res.json();
}

export async function updateProgram(
  slug: string,
  data: {
    unit: string;
    domain_counts: EditableState['domain_counts'];
    principle_counts: EditableState['principle_counts'];
  }
): Promise<ProgramResponse> {
  const res = await fetch(`${BASE}/programs/${slug}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  });
  if (!res.ok) throw new Error('Failed to update program');
  return res.json();
}

export async function deleteProgram(slug: string): Promise<{ status: string }> {
  const res = await fetch(`${BASE}/programs/${slug}`, { method: 'DELETE' });
  if (!res.ok) throw new Error('Failed to delete program');
  return res.json();
}

export function bundleUrl(slug: string): string {
  return `${BASE}/programs/${slug}/bundle`;
}

export async function getGlossary(): Promise<Glossary> {
  const res = await fetch(`${BASE}/glossary`);
  if (!res.ok) throw new Error('Failed to load glossary');
  return res.json();
}
```

---

## 7. Theme Configuration

```javascript
// tailwind.config.js

export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        sans: ['Nunito', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Consolas', 'monospace'],
      },
      colors: {
        balance: {
          balanced: '#10b981',
          warning: '#f59e0b',
          critical: '#ef4444',
          rigid: '#8b5cf6',
          fragmented: '#f97316',
        },
        alignment: {
          light: '#d1fae5',
          dark: '#064e3b',
        },
        displacement: {
          light: '#fee2e2',
          dark: '#7f1d1d',
        },
      },
    },
  },
  tools: [],
};
```

```typescript
// src/theme.ts

export type Theme = 'light' | 'dark' | 'system';

export function getStoredTheme(): Theme {
  const stored = localStorage.getItem('theme');
  if (stored === 'light' || stored === 'dark' || stored === 'system') {
    return stored;
  }
  return 'system';
}

export function applyTheme(theme: Theme): void {
  const root = document.documentElement;
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;

  root.classList.remove('light', 'dark');

  if (theme === 'dark' || (theme === 'system' && prefersDark)) {
    root.classList.add('dark');
  } else {
    root.classList.add('light');
  }

  localStorage.setItem('theme', theme);
}

export function initTheme(): void {
  applyTheme(getStoredTheme());

  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    if (getStoredTheme() === 'system') {
      applyTheme('system');
    }
  });
}
```

---

## 8. Global Styles

```css
/* src/index.css */

@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  html {
    font-family: 'Nunito', system-ui, sans-serif;
  }

  body {
    @apply bg-white text-gray-900;
    @apply dark:bg-gray-950 dark:text-gray-100;
    @apply transition-colors duration-200;
  }
}

@layer components {
  .card {
    @apply rounded-xl p-4;
    @apply bg-gradient-to-br from-white to-gray-50;
    @apply dark:from-gray-900 dark:to-gray-800;
    @apply border border-gray-200 dark:border-gray-700;
    @apply shadow-sm;
  }

  .card-alignment {
    @apply bg-gradient-to-br from-alignment-light to-emerald-50;
    @apply dark:from-alignment-dark dark:to-emerald-950;
    @apply border-emerald-200 dark:border-emerald-800;
  }

  .card-displacement {
    @apply bg-gradient-to-br from-displacement-light to-red-50;
    @apply dark:from-displacement-dark dark:to-red-950;
    @apply border-red-200 dark:border-red-800;
  }

  .btn {
    @apply px-4 py-2 rounded-lg font-semibold;
    @apply transition-all duration-150;
    @apply focus:outline-none focus:ring-2 focus:ring-offset-2;
    @apply dark:focus:ring-offset-gray-900;
  }

  .btn-primary {
    @apply bg-gradient-to-r from-blue-500 to-blue-600;
    @apply hover:from-blue-600 hover:to-blue-700;
    @apply text-white shadow-md hover:shadow-lg;
    @apply focus:ring-blue-500;
  }

  .btn-secondary {
    @apply bg-gray-100 dark:bg-gray-800;
    @apply hover:bg-gray-200 dark:hover:bg-gray-700;
    @apply text-gray-700 dark:text-gray-300;
    @apply focus:ring-gray-400;
  }

  .btn-danger {
    @apply bg-gradient-to-r from-red-500 to-red-600;
    @apply hover:from-red-600 hover:to-red-700;
    @apply text-white;
    @apply focus:ring-red-500;
  }

  .stepper-btn {
    @apply w-11 h-11 rounded-lg font-bold text-xl;
    @apply flex items-center justify-center;
    @apply bg-gray-100 dark:bg-gray-800;
    @apply hover:bg-gray-200 dark:hover:bg-gray-700;
    @apply focus:outline-none focus:ring-2 focus:ring-blue-500;
    @apply disabled:opacity-40 disabled:cursor-not-allowed;
    @apply transition-colors duration-150;
  }

  .gauge-track {
    @apply h-3 rounded-full bg-gray-200 dark:bg-gray-700;
    @apply relative overflow-hidden;
  }

  .gauge-fill {
    @apply absolute top-0 h-full rounded-full;
    @apply transition-all duration-300 ease-out;
  }

  .gauge-centre {
    @apply absolute top-0 left-1/2 w-0.5 h-full;
    @apply bg-gray-400 dark:bg-gray-500;
    @apply -translate-x-1/2;
  }
}
```

---

## 9. Component Specifications

### 9.1 App.tsx

Responsibilities:

- Initialise theme on mount.
- Fetch program list and glossary on mount.
- Manage selected program slug.
- Fetch program data when slug changes.
- Pass data and handlers to child components.
- Handle debounced saves (500ms after last edit).

State:

```typescript
interface AppState {
  programs: ProgramSummary[];
  selectedSlug: string | null;
  program: ProgramResponse | null;
  glossary: Glossary | null;
  status: 'idle' | 'loading' | 'saving' | 'error';
  error: string | null;
  theme: Theme;
}
```

### 9.2 Header.tsx

Contains:

- Title: "AIR".
- Theme toggle (three states: light, dark, system).
- Program selector dropdown.
- Create program button.
- Delete program button.
- Unit toggle (Daily / Sprint).
- Status indicator.

### 9.3 BalancePanel.tsx

Props:

```typescript
interface BalancePanelProps {
  apertures: ReportApertures | null;
  aStar: number;
}
```

Displays three gauges for A_econ, A_emp, A_edu.

Each gauge:

- Label (Economy, Employment, Education).
- Horizontal track with centre mark at A*.
- Fill extends left (rigid, purple) or right (fragmented, orange) from centre.
- Percentage deviation label.
- Colour coding: balanced (green), warning (amber), critical (red).

Calculation:

```typescript
function formatDeviation(aperture: number, aStar: number) {
  const deviation = ((aperture - aStar) / aStar) * 100;
  const text = `${deviation >= 0 ? '+' : ''}${deviation.toFixed(1)}%`;
  const abs = Math.abs(deviation);
  const status = abs <= 5 ? 'balanced' : abs <= 10 ? 'warning' : 'critical';
  const direction = deviation < -1 ? 'rigid' : deviation > 1 ? 'fragmented' : 'balanced';
  return { text, status, direction };
}
```

### 9.4 DomainsPanel.tsx

Props:

```typescript
interface DomainsPanelProps {
  counts: EditableState['domain_counts'];
  onChange: (domain: 'economy' | 'employment' | 'education', value: number) => void;
  onInfo: (domain: string) => void;
}
```

Three cards, each with:

- Label.
- Stepper.
- Proportion bar (percentage of total).
- Info button.

### 9.5 PrinciplesPanel.tsx

Props:

```typescript
interface PrinciplesPanelProps {
  counts: EditableState['principle_counts'];
  onChange: (principle: string, value: number) => void;
  onInfo: (principle: string) => void;
}
```

Two sections:

- Alignment (GMT, ICV, IIA, ICI) with green gradient.
- Displacement (GTD, IVD, IAD, IID) with red gradient.

Each card has abbreviation, full name, stepper, info button.

### 9.6 Stepper.tsx

Props:

```typescript
interface StepperProps {
  value: number;
  min?: number;
  label: string;
  onChange: (value: number) => void;
}
```

Accessible stepper with large touch targets (44×44 minimum).

### 9.7 KernelPanel.tsx

Props:

```typescript
interface KernelPanelProps {
  kernel: ReportKernel | null;
}
```

Displays:

- Step count.
- State hex.

Monospace font, muted styling.

### 9.8 ReportPanel.tsx

Props:

```typescript
interface ReportPanelProps {
  report: Report | null;
  onDownloadBundle: () => void;
}
```

Displays:

- THM totals.
- Gyroscope totals.
- Distribution by domain.
- Download bundle button.

### 9.9 GlossaryModal.tsx

Props:

```typescript
interface GlossaryModalProps {
  isOpen: boolean;
  onClose: () => void;
  glossary: Glossary | null;
  type: 'domain' | 'alignment' | 'displacement';
  itemKey: string;
}
```

Full screen on mobile, centred dialog on desktop.

Trap focus, close on Escape, close on background click.

### 9.10 ConfirmModal.tsx

Props:

```typescript
interface ConfirmModalProps {
  isOpen: boolean;
  title: string;
  message: string;
  confirmLabel: string;
  variant: 'danger' | 'default';
  onConfirm: () => void;
  onCancel: () => void;
}
```

Used for delete confirmation.

---

## 10. Accessibility

- All buttons have `aria-label`.
- Steppers use `role="group"` with `aria-label`.
- Status changes use `aria-live="polite"`.
- Modals use `role="dialog"`, `aria-modal="true"`, `aria-labelledby`.
- Focus is trapped within open modals.
- Focus returns to trigger on modal close.
- Visible focus indicators on all interactive elements.

---

## 11. Responsive Layout

Mobile-first. Panels stack vertically on narrow screens, arrange in rows on wider screens.

Use Tailwind responsive prefixes: `sm:`, `md:`, `lg:`.

Minimum touch target: 44×44 pixels.

---

## 12. Development Workflow

### 12.1 Backend

```bash
cd src/app/console/api
uvicorn server:app --reload --port 8000
```

### 12.2 Frontend

```bash
cd src/app/console/ui
npm install
npm run dev
```

Vite config:

```typescript
// vite.config.ts
import { defineConfig } from 'vite';
import react from '@vitejs/tool-react';

export default defineConfig({
  tools: [react()],
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
});
```

### 12.3 Production Build

```bash
cd src/app/console/ui
npm run build
```

Output in `dist/`. Backend can serve as static files.

---

## 13. Acceptance Criteria

The console is complete when:

1. Users can create, edit, and delete programs from the UI.
2. Edits to counts and unit save automatically and trigger sync.
3. Balance gauges show percentage deviation from A* with correct colours.
4. Kernel step and state are displayed from `report.compilation.kernel`.
5. Report summary is displayed from `report.accounting`.
6. Bundle download works.
7. Glossary modal opens for any domain or principle.
8. Light, dark, and system themes work.
9. UI is accessible and responsive.
10. Empty state prompts user to create a program.
11. Errors are handled gracefully.
12. Existing CLI continues to work on program files.
```