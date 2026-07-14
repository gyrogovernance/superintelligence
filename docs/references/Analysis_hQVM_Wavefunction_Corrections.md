# Analysis: hQVM Wavefunction Corrections

## hQVM Wavefunction Kernel — Coordinate and Signature Reference

**Scope.** This note records the canonical `W₂` / `W₂'` coordinate action and the
K4 operator signatures for the hQVM wavefunction kernel. It is a reference for
experiments that read the `(u6, v6)` coordinates and the `(τ_u6, τ_v6)` signatures
directly. All statements are verified against the kernel (`src/constants.py`,
`src/api.py`) by finite computation on the full 4096-state manifold Ω, and
reconciled with the theory ([QuBEC Theory](docs/specs/hQVM_QuBEC_Theory.md), [hQVM Specs Formalism](docs/specs/hQVM_Specs_Formalism.md)).
No free parameters.

---

## 1. Omega12 chart conventions

Ω is charted by two 6-bit coordinates `(u6, v6)` with:

```
A12 = GENE_MAC_A12 ^ word6_to_pairdiag12(u6)      # GENE_MAC_A12 = 0xAAA
B12 = GENE_MAC_A12 ^ word6_to_pairdiag12(v6)
chirality6(u6, v6) = u6 ^ v6
code_shell = popcount(u6 ^ v6)
```

The rest state `GENE_MAC_REST = (A12=0xAAA, B12=0x555)` is:

```
rest:  u6 = 0,  v6 = 63,  chi = 63 = 111111,  code_shell = 6   (complement horizon)
```

**Two shell conventions coexist and must not be mixed:**

| Convention | Source | Complement horizon | Equality horizon |
|------------|--------|--------------------|------------------|
| `code_shell = popcount(χ)` | QuBEC §1.3, kernel `OmegaState12.shell` | 6 | 0 |
| `doc_shell = 6 − code_shell` | wavefunction pole labels | 0 | 6 |

A pole swap is the same map under either label: `code_shell` 6 → 0, or
`doc_shell` 0 → 6. Reports must state which convention they use; the kernel
exposes `code_shell`.

---

## 2. The byte as a gyrating instruction (theory basis)

The per-byte transition is the fused `[L][R]` operation (Formalism §5; wavefunction
§16.11). On the Omega12 chart the single-byte action is:

```
(u', v') = (v ^ e_A(fam),  u ^ m ^ e_B(fam))
```

where `m` is the 6-bit micro-reference, and the family bits select complements:

```
e_A(fam) = 63 if fam in {01, 11} else 0     (intron bit 0)
e_B(fam) = 63 if fam in {10, 11} else 0     (intron bit 7)
```

The four family cases are the K4 gates (QuBEC §10.1):

```
fam 00: (u, v) -> (v, u)            = gate S (pure swap)
fam 01: (u, v) -> (v ^ 63, u)       = gate C (swap + A-complement)
fam 10: (u, v) -> (v, u ^ 63)       = swap + B-complement
fam 11: (u, v) -> (v ^ 63, u ^ 63)  = gate F (complement-swap)
```

**Key structural point.** Each byte has a swap in its R-step, so a **single byte**
has linear part "swap" (parity 1). A **two-byte depth-4 word** has parity
`length mod 2 = 0`, hence identity linear part: the `(u,v)` coordinates are
**not** exchanged, only translated. This is why `W₂` (a 2-byte word) preserves
the `(u,v)` order, while a single byte does not.

---

## 3. W₂ and W₂' coordinate action

The depth-4 half-words act on the Omega12 chart as:

```
W2  = [byte(fam 00, m), byte(fam 01, m)]   ->  (u, v) -> (u ^ m ^ 63, v ^ m)
W2' = [byte(fam 10, m), byte(fam 11, m)]   ->  (u, v) -> (u ^ m,         v ^ m ^ 63)
```

The coordinate order `(u, v)` is **preserved** (no swap). The defect corrected
here was writing `W₂` as `(u, v) → (v ⊕ m ⊕ 63, u ⊕ m)` — that is the **single
byte fam-01** action (`R_01`), not the composed two-byte operator. The chirality
conclusion `χ' = χ ⊕ 63` is unaffected because XOR is symmetric, but the
intermediate `(u6, v6)` values are not.

Derivation of `W₂` for `m = 0` (fam 00 then fam 01), from the byte rule in §2:

```
start (0, 63)
byte fam00: (v, u)                            = (63, 0)
byte fam01: (v ^ 63, u)                       = (0 ^ 63, 63) = (63, 63)
=> W2(0, 63) = (63, 63)
```

From rest with `m = 0`:

```
W2:   (0, 63) -> (63, 63)   chi 63 -> 0     code_shell 6 -> 0   (pole swap)
W2':  (0, 63) -> (0, 0)     chi 63 -> 0     code_shell 6 -> 0   (pole swap)
```

Both land on the equality horizon (`A12 = B12`, χ = 0). `W₂(W₂(s)) = s` for all
`s ∈ Ω` (involution). The kernel's `step_omega12_by_byte` reproduces these
exactly, and they match the single-byte rule composed in §2.

---

## 4. K4 operator signatures

The signature is `(parity, τ_u6, τ_v6)`, the affine action
`(u, v) → (u ⊕ τ_u6, v ⊕ τ_v6)` for `parity = 0` (identity linear part). The
canonical signatures for `m = 0` are:

| Operator | Signature (parity, τ_u6, τ_v6) | Chirality map | rest (0,63) → |
|----------|-------------------------------|---------------|---------------|
| id | (0, 0, 0) | s → s | (0, 63) |
| W₂ (m=0) | (0, 63, 0) | s → 6−s | (63, 63) |
| W₂' (m=0) | (0, 0, 63) | s → 6−s | (0, 0) |
| F (W₂ ∘ W₂') | (0, 63, 63) | s → s | (63, 0) |

Gate **F** on Omega12 is the global complement `(u, v) → (u ⊕ 63, v ⊕ 63)` (QuBEC
§10.1, gate F), consistent with `F = W₂ ∘ W₂'`. The corrected `W₂` / `W₂'` values
reproduce the verified trajectories in §3 and match `omega_word_signature` for the
corresponding byte words (`W₂ = (0xAA, 0xAB)`, `W₂' = (0x2A, 0x2B)`).

The `τ` values are `m`-dependent for `W₂` and `W₂'` (they shift by `m`); the
structural result `χ' = χ ⊕ 63` is universal over all 64 micro_refs.

---

## 5. T2 consistency and the shell-labeling resolution

Theorem T2 (`W₂` maps shell `s → 6−s`) holds over all 4096 states. Verified:

```
chi' != chi ^ 63 over Ω : 0 mismatches
shell' != 6 - shell   over Ω : 0 mismatches   (shell = code_shell)
W2 from rest -> state 0x555555 (equality horizon) : True
W2^2 == id : True
```

The apparent contradiction "χ ⊕ 63 from rest gives shell 0, not 6" comes from
mixing the two conventions in §1:

- Under `code_shell = popcount(χ)`: rest is 6, after `W₂` is 0 → `6 → 0`.
- Under `doc_shell = 6 − code_shell`: rest is 0, after `W₂` is 6 → `0 → 6`.

Both describe the same pole swap. T2's "s → 6−s" uses the inverted (`doc_shell`)
label; in QuBEC's native `code_shell` it is "6−s → s", i.e. `N → 6−N`. The physics
is identical.

---

## 6. Representative shadow pairs under W₂

The complement-horizon ↔ equality-horizon pairing under the corrected action:

| Complement horizon | Equality horizon shadow |
|-------------------|------------------------|
| (63, 0) χ=111111 | (0, 0) χ=000000 |
| (62, 1) χ=111111 | (1, 1) χ=000000 |
| (0, 63) χ=111111 rest | (63, 63) χ=000000 |

Each pair is an exact `W₂`-inverse: applying `W₂` to the shadow reconstructs the
original. (The first column uses `code_shell` 6; the second column is `code_shell`
0.)

---

## 7. Verification

Integrated in ``experiments/hqvm_wavefunction_kernel.py`` section **H**
(`verify_k4_w2`, `print_k4_w2_verification`):

```
python `experiments/hqvm_wavefunction_kernel.py`          # A–H
python `experiments/hqvm_wavefunction_kernel.py` --k4-only  # H only
```

Checks: W₂/W₂′ signatures, rest trajectories, T2 over all 4096 Ω states,
QuBEC single-byte step rule vs `step_omega12_by_byte`, affine `(u^m^63,v^m)`
vs swapped `(v^m^63,u^m)` (fam-01 only). Exact integer arithmetic; no sampling.

---

*Reference note for experiments using the hQVM wavefunction kernel. The kernel
(`src/constants.py`, `src/api.py`) is the source of truth; the theory
([QuBEC Theory](docs/specs/hQVM_QuBEC_Theory.md) §1.3, §10.1; [hQVM Specs Formalism](docs/specs/hQVM_Specs_Formalism.md) §5) supplies
the conventions. This note records the corrected canonical coordinates and
signatures for reproducibility across repositories.*
