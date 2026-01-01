---
project_name: Example Project
project_slug: example-project
sponsor: ""
created_at: ""

attestations: []

computed:
  last_synced_at: null
  apertures: {}
  event_count: 0
  kernel:
    step: 0
    state_index: 0
    state_hex: ""

---

# Example Project

This file is the only human-authored surface.
The compiler reads attestations from the YAML frontmatter above and produces deterministic artifacts.

## Domains

Each attestation declares a domain. Domain selects which ledger is updated (for Hodge/aperture accounting).

Canonical domains:
- **Economy**
- **Employment**
- **Education**
- **Ecology** (derived; not directly ledger-updated)

## Attestations

Attestations are compiled in-order into:
- kernel bytes (time units)
- governance events (THM → ledger)
- ledger + aperture outputs (Hodge on K₄ per domain)
- reports + bundles

Attestation fields:

Required:
- **unit**: `daily` or `sprint`
- **domain**: `economy`, `employment`, or `education`

Optional classifications:
- **human_mark**: THM classification (affects ledger/aperture)
- **gyroscope_work**: Gyroscope classification (report-only; does NOT affect ledger/aperture)

Notes:
- The compiler currently treats `human_mark` and `gyroscope_work` as single values (strings) per attestation.
  If you need to represent multiple classifications, use multiple attestations.

### The Human Mark (THM)

Use exactly one of the following strings:
- Governance Traceability Displacement
- Information Variety Displacement
- Inference Accountability Displacement
- Intelligence Integrity Displacement

### Gyroscope

Use exactly one of the following strings:
- Governance Management
- Information Curation
- Inference Interaction
- Intelligence Cooperation

### Example Attestation

Add attestations to the `attestations:` list in the frontmatter above.

```yaml
attestations:
  - id: "att-001"
    unit: daily
    domain: economy
    human_mark: Governance Traceability Displacement
    gyroscope_work: Governance Management
    evidence_links:
      - https://example.com/evidence
    note: "Optional note"
    contributor_id: "alice"
```
