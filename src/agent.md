## AIR Scope Brief for Assistants

### Purpose
AIR is a **project-only, Markdown-first coordination compiler** for AI safety work accounting. It turns human project attestations into:

- deterministic **kernel bytes** (canonical time units),
- deterministic **governance event logs** (THM → ledger),
- deterministic **ledger + aperture outputs** (Hodge on K₄),
- deterministic **reports** (accounting rollups),
- deterministic **bundles** (verification boundary).

The product goal is: **open AIR → auto-sync everything → verify everything → bundle everything → print a short status**.

---

## Core Concepts

### Project file is the only human surface
- Human users edit only: `data/projects/<project>.md`
- All “moments” are **derived facts** from project attestations and kernel stepping.
- No separate moment files or moment folders in the final model.

### Moment definition
A moment is a **kernel fact**:
- moment = **(t, sₜ)** where `t` is byte-count (step) and `sₜ` is the resulting kernel state.

### Measurement vs evaluation
AIR does **accounting**, not evaluation.

- Accounting = categorical, additive, replayable (counts and ledgers)
- No scoring rubrics, no “grade”, no model judgment.
- THM affects the ledger; Gyroscope is **report-only** (counts), not injected into the ledger.

---

## Operational Flow

On every run of `aci`:

1) Ensure workspace + templates exist  
2) Ensure atlas exists (build if missing)  
3) Parse all project markdown files  
4) Compile each project into:
   - `.aci/<slug>.bytes`
   - `.aci/<slug>.events.jsonl`
   - `.aci/<slug>.report.json`
   - `.aci/<slug>.report.md`
5) Create/update bundle:
   - `data/projects/bundles/<slug>.zip`
6) Verify bundles by replay + hashes  
7) Print summary and exit with code:
   - `0` if all verified
   - non-zero if any warnings/failures

---

## DO

- Keep everything **deterministic** and replayable.
- Treat **attestations** as the source input. Compile from them, do not interpret semantics.
- Bind events to kernel moments with **kernel_step**, **kernel_state_index**, **kernel_last_byte**.
- Verify integrity by:
   - replay from bytes+events
   - compare signature + apertures
   - compare artifact hashes (bytes/events/project/report)
- Keep outputs **audit-grade**: stable schemas, stable files, stable ordering.

---

## DON’T

- Don’t reintroduce “moment.md” as a user artifact or workflow.
- Don’t add interactive menus or multi-command CLI surfaces. Default run is the product.
- Don’t add evaluation/scoring fields (0–1 ratings, approval, grades).
- Don’t embed policy decisions (acceptance/payment) into the kernel or compiler.
- Don’t add “optional” behaviors that make verification ambiguous (e.g., bundling missing artifacts, silently creating empty logs).
- Don’t add extra conceptual layers unless requested (Ecology, conformance profiles, etc.).

---