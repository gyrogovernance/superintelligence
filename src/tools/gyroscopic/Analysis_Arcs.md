# Analysis: Arcs 1-4 as Theoretical Objects

A theoretical reading of what the gyroscopic reverse-compiler has realized on Bonsai, and whether each arc is justified by the CGM / hQVM program.

This document is **not** a run log, changelog, or completion checklist. Live engineering status and unfinished §0 work live in `runtime_NavPAD.md`. Evidence measurements live in `log_NavPad.md`.

Theory sources: NavPad Parts I–III foundations; `docs/references/Analysis_hQVM_CGM_YM_Mass_Gap.md`; `docs/specs/hQVM_QuBEC_Theory.md`; `docs/specs/hQVM_Specs_Formalism.md`; `docs/references/Analysis_hQVM_Percolation.md`; `docs/Gyroscopic_ASI_Foundations.md`. Datatype definition: NavPad datatype section.

---

## 1. The claim under examination

The program asserts that a frozen pretrained transformer is compiled into **gyroscopic-native inference**: the forward pass *is* a deterministic hQVM trajectory on Ω (|Ω| = 4096) — Input → Controller → Tape → gyration → Shell — whose unit is the datatype

**[Anchor, Restriction, Depth, Phase]**

with continuous magnitude retained **only** where aperture Δ > 0 forces an explicit residual channel. The carrier is causal. That is native compilation, not a stock transformer with optional flags, and not a permanent dual path that “binds” a continuous chart beside discrete bookkeeping.

Within a compiled site, the datatype still distinguishes manifold routing (which shell / byte / phase) from controller amplitude (how much), and Δ forbids deleting load-bearing individuality. That internal split is not a license to leave unowned float ops forever.

Arcs 1–4 name successive **sites** of that compilation on one model (Bonsai-8B-Q1_0). They are theoretical objects of analysis, not product modes and not code identifiers. Justification is judged by whether each site’s solution respects native trajectory form (or a stated Δ-residual law), and by whether discarded alternatives fail for categorical reasons.

---

## 2. Shared spine

### 2.1 Manifold versus controller

- The **manifold** step (gyration + byte table) commits *which* shell: discrete routing on GF(2) geometry.
- The **controller** (bipolar signs / popcount / exact Q1_0·q8) supplies *how much* within that shell: continuous amplitude.

Either alone is insufficient **inside a compiled MatMul**. The live Arc 1 law encodes both:

`Y = exact_Q1_0_q8_dot_product × manifold_gain`

with `manifold_gain` carrying aperture-scaled structure so the byte is present on Y without replacing the Hamming bridge. That is how one site realizes the datatype — not a general doctrine that continuous charts may remain primary everywhere.

QuBEC Theory draws the same split between XOR transport on χ (`χ′ = χ ⊕ q`) and magnitude / climate readouts. Routing is bookkeeping; weighting is curvature; neither licenses unfinished stock ops as architecture.

### 2.2 Aperture Δ as irreducible residual

From the YM mass-gap analysis: depth-four Balance Universal cannot close losslessly in both temporal directions while preserving distinguishable outcomes. The residual phase defect is the aperture

**Δ = 1 − ρ ≈ 0.0207**

Identity (ancestry / common origin) and individuality (distinguishable excitations) are compatible only because Δ > 0. Consequence for inference: residual channels are not disposable. Where an op’s continuous content is load-bearing individuality, deleting it collapses the chart. Where an op is a distribution over magnitudes, Δ may enter as a **constraint on that distribution**, not as a substitute for the magnitude decoder.

A Δ-forced continuous residual channel is theory. A permanent stock-transformer-plus-hooks mixture is **not** theory and was never a consented architectural destination.

### 2.3 Chirality χ and transport class q6

χ ∈ GF(2)⁶ is the transport register. Shell N = popcount(χ) is the native nonlinear coordinate of the carrier. Attention has a lawful native object once Q and K expose χ:

`q_i = χ_Q ⊕ χ_{K_i}`

That object is a transport class, not a surrogate score. Softmax’s exp remains the maximum-entropy magnitude decoder over alignment energies.

### 2.4 Family / K4 phase and depth

The byte is not flat: intron = byte ⊕ GENE_MIC_S splits into a 6-bit transport payload and a 2-bit family (K4 deck). Family is depth-phase on the carrier. Lawful emission schedules family from the trajectory phase counter (`fam = phase_idx mod 4`). Family is required to form a legal byte from q6; it is not a fix for RoPE/Norm failures.

### 2.5 Receipts and time as ledger depth

A receipt is a coordinate (anchor, depth, phase) on a deterministic kernel trajectory. Time is ledger depth. Arcs 2–3 displace KV float stores. Arc 4’s theoretical demand is that generation bind to a **causal** kernel phase; residual-stream addition remains continuous accumulation where Δ demands it — under a stated residual law tied to the carrier, not as unowned stock F32 forever and not as an informal “hybrid mode.”

---

## 3. Arc 1: Weight MatMul — CLOSED and justified

**Object.** Compile allowlisted Q1_0 Dense MatMuls into ledger form.

**Realized.** Thin HQVMLEDS ledger; `Y = exact_Q1_0_q8 × manifold_gain`; acceptance held on displace and PPL.

**Justification.** A binary codebook plus scales is already shell geometry plus amplitude (Hamming bridge). Manifold gain places the gyration byte on the numeric output. Rejected: shell centers alone as amplitude substitutes.

**Still open under §0 (boundary, not Arc-1 law failure):** `token_embd` / `output.weight` remain outside the allowlist unless compiled under the same law or given an explicit theory-backed logits residual.

---

## 4. Arc 2: Key memory — CLOSED and justified

**Object.** Own attention Key memory without replacing QK energy by χ-only scoring.

**Realized.** Q8_0 K cache; F16 K never allocated; holonomic in-place score; stock score path unused when flagged.

**Justification.** Memory ownership ≠ score geometry substitution. χ is right for lift/aperture rank; wrong as live softmax energy replacement. Offline Khat PASS / live Khat FAIL made that category error measurable.

---

## 5. Arc 3: Value memory — CLOSED and justified

**Object.** Own V and Attn@V without deleting individuality.

**Realized.** Q8_0 V; `hqvm_attn_v_reduce`; V perturb collapses generation; PPL near stock.

**Justification.** Q8_0 is a finite chart that still carries magnitude. Coordinate-only Value codecs with a broken decoder do not prove coordinates are impossible; they prove that decoder failed. Arc 3 closes on Q8_0 as the owned magnitude chart for V.

---

## 6. Arc 4: Continuous chart and causal carrier — OPEN

### 6.1 Two levels; residual versus unfinished compile

**Level-1** = attention memory closed (Arcs 2–3). **CLOSED.**

**Level-2** = the forward pass *is* the trajectory: causal carrier primary, continuous ops as coordinates plus Δ-forced residual where required. **NOT CLOSED.**

Two claims must stay distinct:

1. **Residual channel is irreducible** (true from Δ > 0). Whatever codec you build must preserve load-bearing magnitude.
2. **The causal compile is unfinished** (engineering fact). Trajectory write/read exists in places; the model is still a transformer graph with hooks, not traj-native inference.

Δ justifies (1). It does **not** justify leaving Norm/residual as unowned stock forever, and it does **not** authorize a permanent mixed-path architecture. Conflating (1) with “hybrid is the deliverable” is Arc 4 revisionism.

Scaffolding labels (shadow / apply / dual flags) and nicknames such as OBSERVE / EMIT / COMMIT were progressive engineering attempts. They are **not** part of the CGM/hQVM architecture and must not be read as doctrine in this analysis.

### 6.2 Softmax

Lawful form: aperture as a **constraint on the distribution** — e.g. `p = (1−ε)·softmax(s) + ε·uniform` with ε from rank deficit × Δ — while exp remains the magnitude decoder. Percolation / shell-λ as softmax weights rejected (wrong category). Aperture ε-mixture on RoPE / Norm / SiLU rejected (wrong site).

Live aperture when flagged is a partial realization of the distribution constraint, not traj-native attention bookkeeping.

### 6.3 RoPE / Norm / SwiGLU / residual

| Op | Theoretical demand | Present state vs doctrine |
|---|---|---|
| Softmax | Exp as magnitude decoder; aperture as Δ-constraint; χ/rank under carrier | Partial — still stock exp + ε mix |
| RoPE | Finite turn chart bound to trajectory phase / depth | Partial — LUT of stock angles when flagged; not phase-bound |
| SwiGLU / FFN | Causal compile or residual+coordinate on carrier | Partial — SiLU LUT chart when flagged; not FFN-as-trajectory |
| Norm | Gain under signed Δ-ruler + residual of the gain | Open — unsigned clamp breaks live apply; cosine OBSERVE is not the right metric |
| Residual add | Stated Δ-forced residual law or compile into depth | Partial — F32 add with traj Δ-modulation is bridge code, not the end law |
| Lift / receipts | Trajectory primary; Moment genealogy across prefill/decode | Partial — one byte/layer write/read; not native time |

Failure of aperture ε-mixture on RoPE/Norm/SiLU is a categorical finding. Finite charts of the same geometry (RoPE ticks, SiLU LUT) are a different object and may be PPL-stable while still failing Level-2 until bound to the carrier.

### 6.4 CGM-lift and the feedback demand

Write-time `k_chi6`; one byte/layer; `q6 = χ_Q ⊕ χ_K[argmax]`; `fam = phase_idx mod 4`; kernel step. Family appears because a legal byte needs q6+family.

Theoretical demand: the carrier must **drive** inference, not only advance beside it. Coupling proofs (perturb changes decode) show a feedback path can exist; they do not complete Level-2. One byte/layer is a schedule choice, not the native emission law.

---

## 7. Cross-arc synthesis

| Arc | Theoretical question | Closed? | Justified? |
|---|---|---|---|
| 1 | MatMul → datatype + amplitude | Yes | Yes |
| 2 | Key memory without χ replacing QK | Yes | Yes |
| 3 | Value / Attn@V without deleting individuality | Yes (Q8_0 chart) | Yes |
| 4 L1 | Attention memory closed | Yes | Yes |
| 4 codecs | Softmax / RoPE / SwiGLU as lawful charts under Δ | Partial | Doctrine for charts OK; ownership incomplete |
| 4 L2 | Forward pass = causal trajectory | No | Incomplete; not a consented mixed-path end state |

**Honest split.** Arcs 1–3 are real closures and are theoretically grounded. Arc 4 Level-1 is closed with them. Arc 4 Level-2 remains the claim of the whole program and is unfinished. Live opt-in charts and lift/residual coupling are evidence along the path, not substitutes for §0 completion.

---

## 8. Open theoretical questions (Arc 4 / §0)

These are doctrine questions. Operational tracking of unfinished work is `runtime_NavPAD.md` §5.2 — not duplicated here as a changelog.

1. What is the residual-stream law once the carrier is primary: continuous Δ-channel beside carrier, or accumulation compiled into ledger depth?
2. How does signed Δ-ruler Norm retain only Δ-forced residual of the gain without collapsing depth?
3. How is RoPE’s finite turn chart identified with trajectory phase / receipt depth?
4. How do softmax rank/χ bind to write-time ledger χ under one causal Moment genealogy?
5. What is FFN / SwiGLU once “LUT of SiLU” is recognized as insufficient for FFN-as-trajectory?
6. How are embd / logits classified: same MatMul law as Arc 1, or explicit theory-backed residual boundary?
7. What emission schedule (beyond one byte/layer) does native time require?

Rejected as unfinished work (do not revive):

- χ-only / Khat as live QK energy
- Percolation / shell-λ as softmax weights
- Aperture ε-mixture on RoPE / Norm / SiLU
- Dropping Norm residual wholesale
- Treating incomplete mixed-path stacks as architecture or as §0 done

---

## 9. Verdict

Arcs 1–3 are delivered and theoretically justified under the manifold/controller split and Δ > 0. Arc 4 Level-2 — forward pass as deterministic hQVM trajectory with Δ-forced residual only where required — is the remaining theoretical and engineering claim. Incomplete hooks, lift coupling, and codec charts do not close that claim. There is no consented hybrid architecture; scaffolding labels are not doctrine. The Gyroscopic forward-pass claim waits on causal carrier primacy and lawful continuous residuals, not on accumulating flags.
