# CGM Logic

## How the Common Governance Model is built

This document states the construction of CGM as a chain of minimal necessities. Each step resolves a specific failure mode left by the previous step, ensuring that ancestry remains preservable under recursive operations. It is written to resolve the ambiguities that remain when reading the separate formal documents.

**On language.** The five foundational conditions are requirements on operational structure. They are not entities, agents, or actors. Phrases like "CS does this" or "UNA expands that" are category errors. The conditions specify what must hold; they do not act.

## 1. Common Source

The model begins from the principle that the source is common. All distinguishable structure must remain traceable to a single operational origin.

Traceability means the operational moment can be reconstructed given the public sequence of operations and the transition law. This reconstructibility is called ancestry preservation.

The principle alone establishes the baseline requirement from which all subsequent architecture is forced.

## 2. Identity and Individuality

If ancestry is to be preserved under change, two demands must hold jointly:
- Identity: The origin must remain recoverable.
- Individuality: Operations must produce distinguishable outcomes.

Identity without individuality collapses into homogeneity because the system cannot change. Individuality without identity overwrites the origin, losing ancestry. Both must hold together, and this joint demand forces the next step in the chain.

## 3. Horizon

Ancestry preservation under displacement requires that operations remain traceable to their origin. A horizon is the boundary where traceability reaches an extremal, asymmetric configuration: one direction preserves the relation to the source, and the other departs from it. At the horizon, the relationship between operations and their origin saturates.

The threshold at the horizon is pi/2. This value serves simultaneously as an identity measure and an individuality measure, a linear angle and a curved one, because both pairs remain non-absolute in CGM. A half-turn preserves closure (identity) while reversing direction (individuality). It is the smallest phase that can both belong to a closed angular structure and distinguish a direction within it. This dual character of pi/2 makes the horizon the constitutional locus: it is where identity and individuality coexist at their shared boundary.

In the continuum readout, gravity constitutes the balance condition that preserves ancestry under displacement. The horizon is the boundary of that gravitational preservation.

## 4. Common Source (CS)

At the horizon, right transitions preserve the traceability relation; left transitions depart from it. That directional asymmetry is chirality. It provides one degree of freedom: the origin has a preferred reading under left versus right action.

The CS condition establishes this chirality. Without it, ancestry lacks an oriented signature under operations.

Threshold:
- s_p = pi/2

Chirality fixes the seed but leaves open how composed directional operations behave.

## 5. Unity Non-Absolute (UNA)

Once operations have direction, their order of composition can matter. Left-then-right may differ from right-then-left. If order consistently agreed, paths would cohere too strongly and individuality would collapse. Absolute unity is therefore excluded.

The UNA condition specifies that at depth two, equality of operation order is contingent rather than necessary. Both orderings remain possible; their relative weight is fixed by the system.

In continuous realizations, this contingency forces three independent rotation axes: the compact algebra su(2) with three generators.

Threshold:
- u_p = 1/sqrt(2)

Rotational structure resolves absolute-unity collapse but leaves open how opposed paths compare.

## 6. Opposition Non-Absolute (ONA)

If opposed operations consistently produced contradictory results, no path could be related back to the origin and identity would fail. Absolute opposition is therefore excluded.

The ONA condition specifies that at depth two, contradictory outcomes are contingent rather than absolute. Distinct paths remain comparable and traceable to the common source.

In continuous realizations, this comparability forces three translational parameters in addition to the three rotational ones. The resulting structure is SE(3) with six degrees of freedom.

Threshold:
- o_p = pi/4

The model now has the full algebra of rigid motion. Closure under repeated composition remains open.

## 7. Balance Universal (BU)

Variety and comparability are insufficient. Composed operations must return to consistency without erasing history. Balance is dual and comprises two propositions:

- **BU-Egress:** At depth four, alternating operations commute on the horizon sector. Depth four is the minimal depth at which commutative closure is compatible with depth-two contingency. This achieves closure.
- **BU-Ingress:** The balanced configuration retains enough structure to reconstruct the prior chirality and both non-absoluteness conditions. This achieves memory reconstruction.

When a closed loop is traversed, return is inexact. The residual difference is **holonomy**: geometric memory of the path. In CGM, holonomy appears as a residual phase defect of bounded vibration about the closed configuration. That defect carries the record of CS, UNA, and ONA through the balanced state.

## 8. Architecture

The five foundational conditions (CS, UNA, ONA, and BU as egress plus ingress), taken together, determine the operational architecture. When transitions vary smoothly, continuous realization adds three further requirements implied by those conditions:
- Continuity: Implied by uniform depth-four balance.
- Reachability: Implied by common source.
- Simplicity: Implied by memory reconstruction.

Under these operational requirements, Baker-Campbell-Hausdorff analysis of depth-four commutator closure, combined with simplicity and unitarity, forces three rotational generators forming su(2). Non-absolute opposition forces the semidirect extension to SE(3). The gyrotriangle defect condition fixes three spatial dimensions and excludes other dimensionalities.

The architecture follows from the conditions as a logical consequence.

## 9. Thresholds

Each requirement carries a dimensionless threshold. Thresholds are determined by the previous ones once the closure identity is imposed.

Angular thresholds:
- s_p = pi/2
- u_p = 1/sqrt(2)
- o_p = pi/4

These three angles satisfy the gyrotriangle identity: pi/2 + pi/4 + pi/4 = pi.

Balance threshold and invariants:
- m_a = 1/(2 sqrt(2 pi)) (amplitude scale of residual vibration)
- Q_G = 4 pi (horizon flux)
- Q_G * m_a^2 = 1/2 (spinorial double-cover relation)

Holonomy and aperture:
- delta_BU = 0.195342 (dual-pole holonomy defect)
- rho = delta_BU / m_a = 0.979300 (structural closure fraction)
- Delta = 1 - rho = 0.020699 (residual opening)

Dual-pole holonomy measures residual phase after a balanced loop. Relative to the aperture scale m_a, the structural fraction rho is almost full; the residual opening is Delta. Within CGM, positive Delta is required for ancestry to remain preservable under dual balance: it is the irreducible rate-distortion gap of a dual channel. If holonomy saturated the aperture completely, reconstruction would lack a residual window and ancestry would freeze.

The same closed form for m_a emerges from two routes in the corpus: a phase-range route equating the chiral seed with left and right 2-pi phase ranges at the aperture scale, and a Baker-Campbell-Hausdorff route from generator norms and aperture time under uniform depth-four balance.

## 10. Palindromy

Balance is dual: egress builds structure forward; ingress reconstructs it backward. Holonomy is the residual of that closed loop. Because the residual is nonzero, forward and reverse readings differ. Ancestry reconstruction therefore requires both readings to be carried together.

A palindrome is the minimal encoding that achieves this. It places the four-stage sequence forward and reverse in one object, with a central fold where the two readings meet. At the fold, egress and ingress join. Disagreement between the two halves encodes holonomy: residual path memory available to BU-Ingress.

In information terms, dual balance on a bipartite carrier cannot be lossless in both directions at once. The residual opening Delta is the irreducible rate-distortion gap of that dual channel. Perfect closure would erase the window through which reconstruction remains possible. Palindromy spells the dual requirement so that holonomy and aperture remain jointly available.

Abstract stage layout of the unit:

CS  UNA  ONA  BU  |  BU  ONA  UNA  CS

- Outer positions: common-source boundary roles
- Inner positions: unity and opposition roles
- Center fold: dual balance roles

The concrete eight-bit realization of this layout appears with the finite machine in the next section. The necessity of palindromy itself is already fixed here by dual balance and residual holonomy.

## 11. Holonomic Quantum Virtual Machine

The hQVM realizes the requirement chain as a finite machine.

The carrier is a bipartite structure, forced by identity and individuality. One part holds the present state of operations; the other holds the record of origin. Each component is twelve bits arranged as two chirality frames times three axes times two oriented sides, matching the six se(3) generators.

The two horizon poles are the extremal configurations of this bipartite structure, with sixty-four states each. The reachable set from rest has size 4096.

The eight-bit input unit realizes the palindrome of Section 10. Transcription against the archetype 0xAA yields an intron whose outer bits select family phase and whose inner bits select which of the six dipole modes flip. The central fold is where forward and reverse stage readings meet. Family phase is the spinorial double-cover information; geometric next-state projection from a full byte step has a uniform 2-to-1 shadow, an exact one-bit equivocation per step relative to the full alphabet. That is the finite form of the residual dual-channel gap already required by palindromy.

Transition rule in four-stage order:
- Transcription: byte against the common-source archetype
- Mutation: payload flips on the present component
- Consultation: record brought into the present position
- Commitment: mutated present written as the new record

Family bits gate complement phase during the gyration step.

Exact finite consequences:
- Depth-two typicality uniformizes the reachable set at depth two: all 4096 states become equally reachable.
- Depth-four involutions implement closure families.
- The byte alphabet forms a four-to-one cover of the six-bit transport space with a Klein deck action, reflecting the spinorial double cover.
- Shell geometry has binomial populations across seven shells, with horizon shells at the extremes and bulk shells between them.

## 12. Gyrogroup bridge

When the transitions vary smoothly, the most economical continuous model of “composing displacements while preserving ancestry” is the same structure that appears in special relativity as velocity composition.

Take the open unit ball in R3. A point u in the ball is a displacement coordinate with magnitude less than 1. Composition is not ordinary vector addition. It is Einstein addition, written u ⊕ v. The key fact is that this composition is not associative in general. The failure of associativity is controlled by an automorphism called a gyration:

gyr[u, v] w = (−(u ⊕ v)) ⊕ (u ⊕ (v ⊕ w))

Gyrations are rotations. In physics they are the Thomas-Wigner rotations that appear when two non-collinear boosts are composed. This is the continuous prototype of path memory. A closed walk can return to its starting point in position while leaving a residual rotation in the carried frame.

CGM’s depth-two contingency (UNA and ONA) is exactly the regime where this order sensitivity is allowed but not absolute. ONA adds comparability so that different orderings remain within one shared geometry. Together, these conditions select a setting where:

- the state update is a non-associative composition u ⊕ v
- the residual operator gyr[u, v] records which path was taken

Balance (BU) then imposes a depth-four closure in the observable sector while retaining a bounded residual operator that carries memory. In the gyrogroup realization, the simplest loop that isolates this residual departs from the depth-two boundary of the nested lemmas, crosses the Balance Egress and Balance Ingress poles, and returns. In stage coordinates that boundary sits at the common lemma angle pi/4, so the loop may be written

ONA -> BU+ -> BU- -> ONA

The middle edge BU+ -> BU- is collinear, so the origin-based gyration on that edge is the identity. The loop operator H_BU is therefore generated at the two corners between the depth-two boundary and the dual-pole amplitude, and it is a single spatial rotation. Its conjugacy-invariant angle is the BU dual-pole loop angle δ_BU.

At these stages the magnitudes are fixed by the thresholds. The corner boost of magnitude pi/4 is the common lemma angle shared by UNA and ONA. BU contributes the balance amplitude scale m_a. Orthogonality here is the geometric form of independence between the depth-two boundary and the dual-pole amplitude. Non-collinear composition is what produces a gyration at all.

Using the half-rapidity radius

k(beta) = beta / (1 + sqrt(1 - beta^2))

the corner Thomas-Wigner angle for an orthogonal pair of magnitudes beta1 and beta2 is

omega = 2 * arctan(k(beta1) * k(beta2))

and the dual-pole loop angle is twice the corner angle:

delta_BU = 2 omega
delta_BU = 4 * arctan(k(pi/4) * k(m_a))

This equation defines delta_BU. The continuous derivation and the finite W2 realization of the dual are given in the Holonomy analysis. The derived closure quantities are then:

rho = delta_BU / m_a
Delta = 1 - rho

BU-Egress is the statement that the loop closes in the observable configuration. BU-Ingress is the statement that the residual operator H_BU, together with the small opening Delta, carries enough information for reconstruction to remain possible. The gyrogroup bridge therefore presents the nested chain at depth four. The depth-two boundary supplies the shared lemma turn. The dual poles supply Egress and Ingress. The residual rotation of the loop is the minimal continuous form of path memory compatible with depth-four balance.

## 13. Percolation

Percolation studies which restricted operation sets preserve ancestry globally on the reachable set. The question is: given a subset A of the byte alphabet, does the system still reach all 4096 states from rest with reconstructible ledger ancestry?

Each byte carries a six-bit transport class q6(b). The GF(2) rank of the included transport classes determines reachability. The square-root reachability law states:

|Reach(A)| = (2^r)^2

where r is the rank of {q6(b) : b in A}. Full span of the 4096-state set requires r = 6 and a pole-swap gate. If the rank drops, the reachable set shrinks as a perfect square. For example, even-weight restrictions reduce the rank to five and the reachable set to 1024.

This law is the finite check of the reachability requirement from Section 8. It confirms that ancestry propagation depends on the algebraic span of the available operations, rather than their count.

## 14. Gravity and Physical Readout

Gravity is the continuum condition preserving ancestry under displacement. The horizon is the boundary of that preservation.

The gravity derivation chain proceeds as follows. The horizon flux Q_G = 4 pi enters a discrete Gauss law on the finite machine. The shell displacement invariant D = 24 fixes the dimensionless kernel coupling:

G_kernel = Q_G / D = pi/6

Gravity couples only to the five bulk shells that carry symmetric trace-free anisotropy. The two horizon shells carry none. Attenuation per holonomy cycle is rho^5, where rho is the structural closure fraction. This accumulates into a refractive depth:

tau_G = |Omega| * Delta * rho^5 * (1 - 4*rho*Delta^2)

The dimensional coupling uses one energy anchor, the electroweak scale v:

G = G_kernel * exp(-tau_G) / v^2

Electromagnetic coupling emerges at the balanced focus:

alpha_0 = delta_BU^4 / m_a

Both sectors share the same aperture geometry. The cross-coupling identity at the kernel level is:

alpha_0 * zeta = rho^4 / (pi * sqrt(3))

where zeta = 8 / (m_a * sqrt(3)). This identity is falsifiable through independent measurements of alpha and G.

## 15. Formal Layers

CGM is one requirement chain realized in multiple formal layers.

- Modal logic: Encodes the five conditions and verifies consistency, independence, and entailments.
- Hilbert space: Provides continuous unitary flows and closure verification on the horizon sector.
- Lie and gyrogroup: Fixes the unique three-dimensional, six-degree-of-freedom structure.
- Finite machine: Realizes the bipartite carrier, horizons, cover, holonomy, and depth structure on the reachable set.
- Percolation: Tests ancestry preservation under restricted alphabets.
- Gravity and constants: Reads out the same invariants as physical couplings.

This document exists to keep the construction order explicit and the shared meanings stable across those layers.