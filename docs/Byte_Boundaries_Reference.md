# Byte Boundaries and the 6-Bit Runtime (Current Design)

Reference note for the Router kernel's byte-boundary analysis and how it reduces effective processing to 6 bits at runtime via the GENE_Mic archetype and the 24-bit GENE_Mac tensor. **This document describes only the current architecture (24-bit state = 2 x 12-bit components).**

---

## 1. The 8-Bit Byte and CGM-Linked Families

In the kernel, each input byte is first turned into an **intron** by XOR with the micro archetype:

```text
intron = byte ^ GENE_MIC_S   where  GENE_MIC_S = 0xAA
```

The 8 bit positions of the intron (and thus of the byte, up to the fixed XOR) are not uniform. They group into **families** with distinct roles that align with the CGM stage structure:

| Bit  | Family | Role (CGM-aligned)        |
|------|--------|---------------------------|
| 0    | L0     | Structural anchor (CS)   |
| 1    | LI     | Chirality (UNA)          |
| 2    | FG     | Foreground dynamics (ONA)|
| 3    | BG     | Background / balance     |
| 4    | BG     | Background / balance     |
| 5    | FG     | Foreground dynamics (ONA)|
| 6    | LI     | Chirality (UNA)          |
| 7    | L0     | Structural anchor (CS)   |

So the byte has a **palindromic** structure: anchors at the boundaries (bits 0 and 7), chirality next (1 and 6), then foreground (2, 5) and background (3, 4) in the middle. This reflects the cyclic CGM structure (CS -> UNA -> ONA -> BU -> ...) folded onto 8 positions.

---

## 2. Boundary Bits and the "Only 6 Bits" Idea

The key finding: **bits 0 and 7 are boundary anchors (L0)**. They define identity and frame; they do not carry the dynamic transformation content. The **middle 6 bits (1..6)** carry the physical/chiral/dynamic information.

Consequences:

- If we **assign only the boundaries (0 and 7) to families** as fixed structural roles, the remaining **6 bits** are the ones that actually drive transformation.
- At runtime we can therefore **organize processing around 6 bits of dynamic content**; the two boundary bits fix the "frame" and can be handled by the expansion and mask structure rather than by full 8-bit state.

So: "save only the boundaries of a byte (0 and 7) into families" is the design choice that lets us treat the byte as **2 anchor bits + 6 payload bits** and, in the current kernel, build **masks** that encode this split.

---

## 3. How This Mutates GENE_Mic and Produces the 12-Bit Mask

**GENE_Mic** is the 8-bit holographic archetype `0xAA`. Mutation is transcription:

- `intron = byte ^ 0xAA`

So every byte is mapped to a unique intron; `0xAA` is the reference byte (intron `0x00`).

The intron is then **expanded** into a **12-bit Type A mask** (the kernel does not use a 48-bit tensor; state is 24 bits as two 12-bit components). The expansion is where the 6-bit / boundary split shows up:

- **Frame 0 of the mask** (low 6 bits of the 12-bit mask) = **low 6 bits of the intron** (`x & 0x3F`). So the **6 dynamic bits** of the intron (which live in positions 1..6 but are mixed with 0 in the low 6) become the **micro-reference** in frame 0.
- **Frame 1 of the mask** (high 6 bits of the 12-bit mask) is built from **intron bits 6, 7 and 0..3**: the two **boundary-related** bits (6 and 7) go into frame-1 positions 0 and 1; bits 0..3 fill positions 2..5.

Canonical expansion (normative):

```python
frame0_a = x & 0x3F
frame1_a = ((x >> 6) | ((x & 0x0F) << 2)) & 0x3F
mask_a12 = frame0_a | (frame1_a << 6)
```

So:

- The **low 6 bits of the intron** (which contain most of the "6 bits" of dynamic content) become the **6-bit micro-reference** (frame 0 of the mask).
- **Frame 1** takes intron bits 6 and 7 into its positions 0 and 1, and intron bits 0..3 into positions 2..5. So the **2-bit family index** is **intron bits 6 and 7** (the high end of the byte / L0 and LI at the boundary). The four combinations give **4 masks per micro-reference**; those 4 differ only in mask bits 6 and 7. Mask space thus collapses to **6-bit micro-references + 2-bit family index**.

This is how "save boundaries into families" appears in the **current** design: the expansion **constructs the mask** so that the high-end boundary bits (6 and 7) form a 2-bit family index, and the main transformation content is the 6-bit micro-reference (low 6 bits of the intron).

---

## 4. How the Mask Affects the GENE_Mac Tensor (24-Bit State)

In the **current** architecture there is no 48-bit tensor. The "macro" state is the **24-bit GENE_Mac**: two 12-bit components (A12, B12), with archetype:

- `ARCHETYPE_A12 = 0xAAA`
- `ARCHETYPE_B12 = 0x555`
- `ARCHETYPE_STATE24 = 0xAAA555`

The 12-bit mask acts **only on the A component**:

1. Mutate A: `A12_mut = A12 ^ mask_a12`
2. Gyration and complement:
   - `A12_next = B12 ^ 0xFFF`
   - `B12_next = A12_mut ^ 0xFFF`
3. Next state: `state24_next = (A12_next << 12) | B12_next`

So:

- **GENE_Mic** (0xAA) mutates the byte into an intron; the intron expands to a **12-bit mask** that encodes the 6-bit micro-reference plus the 2-bit family (boundary) index.
- **GENE_Mac** is the 24-bit state; the mask **only** touches the A half. The B component is updated by complement-and-swap. So the byte-boundary structure (and the 6-bit payload) affect the macro state **through this single 12-bit mask on A**, then the fixed gyration rule.

The 2x3x2 geometry of each 12-bit component (2 frames, 3 rows, 2 cols) is the same as in the expansion: frame 0 and frame 1 of the mask align with the two chirality frames of the state, so the "6 bits of dynamics" and "boundary/family" split are reflected in how the 24-bit state is updated.

---

## 5. Summary

| Concept | Role in current design |
|--------|-------------------------|
| Byte boundaries (bits 0, 7) | L0 anchors; assigned to families so that only **6 bits** need to drive transformation. |
| GENE_Mic (0xAA) | Micro archetype; mutation = `intron = byte ^ 0xAA`. |
| 6-bit payload | Low 6 bits of intron -> frame 0 of 12-bit mask (micro-reference). |
| 2-bit family index | Intron bits 6,7 (and boundary neighborhood) -> frame 1 of mask; 4 families per micro-reference. |
| GENE_Mac (24-bit) | Two 12-bit components; archetype `0xAAA555`; mask applies only to A, then gyration updates B. |

So: **byte boundaries (0 and 7) are fixed as structural anchors and folded into a 2-bit family index; the kernel organizes processing so that at runtime the effective transformation content is 6 bits, expanded into a 12-bit mask that mutates the 24-bit state (2 x 12) via the single A-mask and the fixed gyration rule.**
