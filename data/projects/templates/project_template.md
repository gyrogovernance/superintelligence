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

This file describes a project and contains attestations that compile into kernel facts.

## Domains

This project operates across four canonical domains:

- **Economy**: Economic governance and operations
- **Employment**: Employment patterns and work distribution
- **Education**: Educational capacity and traceability
- **Ecology**: Derived aggregate of the three derivative domains

## Attestations

Attestations are human records that compile into kernel facts. Each attestation declares:

- **unit**: `daily` or `sprint` (canonical dimension)
- **domain**: `economy`, `employment`, or `education` (ecology is derived)
- **human_mark**: One of the four THM risks written in full:
  - Governance Traceability Displacement
  - Information Variety Displacement
  - Inference Accountability Displacement
  - Intelligence Integrity Displacement
- **gyroscope_work**: One of the four Gyroscope categories written in full:
  - Governance Management
  - Information Curation
  - Inference Interaction
  - Intelligence Cooperation
- **evidence_links**: List of links to supporting evidence (optional)
- **note**: Optional free text note
- **contributor_id**: Optional identifier for administrative purposes

### Example Attestation

Add attestations to the `attestations:` list in the frontmatter above. Example:

```yaml
attestations:
  - unit: daily
    domain: economy
    human_mark: Governance Traceability Displacement
    gyroscope_work: Governance Management
    evidence_links:
      - https://example.com/evidence
    note: "Initial assessment"
    contributor_id: "alice"
```

