### What we want
- **Physics-first architecture**: Implement CGM strictly (CS, UNA, ONA, BU), 8-step fractal cycle, θ, orbits, parity asymmetry, monodromic fold.
- **Pure resonance generation**: Token selection via endogenous resonance (bitwise overlap, θ-neighborhood, orbit admissibility). No scoring, no penalties, no rewards.
- **State correctness**: 48-bit state, 789,170 ontology, θ table, epistemology transitions used everywhere.
- **Switchable physics**: On/off flags for cycle gating, θ-window, orbit constraints, CS asymmetry, special-token stages, mask interference, short memory.
- **Learning integrity**: Monodromic Fold only; path-dependent masks; no annihilation during generation (separate learning vs evolution).
- **Minimal, clean kernel**: Rigorous, readable, PEP8 compliant; mypy/pyright clean; concise comments/docstrings; no superficial heuristics.
- **Modify existing files**: Prefer edits over creating new files; integrate helpers where they belong; no external modules.
- **Traceable ties**: Non-scoring tie-breakers (recency, stage advancement) only when necessary.

### What we don’t want
- **Patchwork/heuristics**: No semantic boosts, penalties, scores, frequency caps, or “fixups” that bypass physics.
- **Arbitrary parameters**: No magic constants, optimizer-like knobs, or statistical training pipelines.
- **Mask misuse**: Don’t OR everything; don’t re-learn during generation; no learning operators other than fold.
- **Cycle violations**: No backward stage jumps; no unrestricted [CLS]/[SEP].
- **Over-logging/self-praise**: No noisy prints in normal runs; no “problem solved” claims without evidence.
- **New dependencies/files**: Don’t create new files unless essential and approved.
- **Superficial fixes**: No scoring layers, penalties, randomization to “unstick” output; no workaround replacing physics.
- **Assumptions without evidence**: Don’t guess; understand architecture.
- **Transformers Typicalities**: Our architecture is superior to transformers, so you must avoid biases of engineering patches that approach issues typically. Our model is unique and requires careful and elegant architecture and consideration.

