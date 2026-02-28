"""
Post-training analysis of HGT Phase 1.

Characterizes:
  1. Output distribution: Is the model producing uniform noise or structured predictions?
  2. Physics alignment: Do predictions respect family structure, vertex charge, mask geometry?
  3. Curriculum coverage: How does the model behave on each curriculum family?
  4. Head decomposition: Are family_head and micro_head learning independently?
  5. Embedding structure: Have BL1/TL1 embeddings differentiated from initialization?
  6. Gradient diagnostics: Which components received signal during training?

Usage:
    pytest secret_lab_ignore/autobots/tests/test_analysis.py -v -s
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter

import pytest
import torch
import torch.nn.functional as F
import numpy as np

_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from secret_lab_ignore.autobots.config import HGTConfig
from secret_lab_ignore.autobots.model import HGTForCausalLM
from secret_lab_ignore.autobots import physics

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MODEL_DIR = _root / "data" / "autobots" / "model"


@pytest.fixture(scope="module")
def model():
    if not MODEL_DIR.exists():
        pytest.skip(f"Trained model not found at {MODEL_DIR}")
    m = HGTForCausalLM.from_pretrained(MODEL_DIR)
    m.eval()
    return m


@pytest.fixture(scope="module")
def mask12_table():
    return physics.compute_mask12_table()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def make_sequence(byte_list: list[int]) -> torch.Tensor:
    """Create a [1, seq] input tensor from a list of byte values."""
    return torch.tensor([byte_list], dtype=torch.long)


def get_logits(model: HGTForCausalLM, byte_list: list[int]) -> torch.Tensor:
    """Run forward pass, return logits [1, seq, 256]."""
    with torch.no_grad():
        out = model(input_ids=make_sequence(byte_list))
    return out.logits


def top_k_bytes(logits_1d: torch.Tensor, k: int = 10) -> list[tuple[int, float]]:
    """Return top-k (byte_value, probability) from a single position's logits."""
    probs = F.softmax(logits_1d, dim=-1)
    topk = torch.topk(probs, k)
    return [(int(topk.indices[i]), float(topk.values[i])) for i in range(k)]


def byte_to_physics(b: int) -> dict:
    """Compute all physics properties for a single byte."""
    intr = physics.intron(b)
    m12 = physics.expand_intron_to_mask12(intr)
    return {
        "byte": b,
        "intron": intr,
        "family": (intr >> 6) & 0x3,
        "micro_ref": intr & 0x3F,
        "mask12": m12,
        "vertex": physics.vertex_charge(m12),
        "mask_weight": bin(m12).count("1"),
    }


# ===========================================================================
# 1. OUTPUT DISTRIBUTION ANALYSIS
# ===========================================================================

class TestOutputDistribution:
    """Characterize what the model is actually producing."""

    def test_entropy_vs_uniform(self, model):
        """
        Measure output entropy across diverse inputs.
        Uniform over 256 = ln(256) ≈ 5.545 nats.
        If entropy ≈ 5.5, model is producing noise.
        If entropy < 4.0, model is concentrating predictions.
        """
        print("\n" + "=" * 70)
        print("OUTPUT DISTRIBUTION ANALYSIS")
        print("=" * 70)

        test_sequences = [
            ("single_byte_0xAA", [0xAA]),
            ("single_byte_0x00", [0x00]),
            ("ascii_hello", list(b"Hello")),
            ("random_walk", [23, 187, 42, 99, 201, 15, 88, 134]),
            ("closure_xyxy", [10, 20, 10, 20]),
            ("all_family_0", [0xAA, 0xAA, 0xAA, 0xAA]),  # intron 0x00, family 0
            ("family_sweep", [0xAA, 0xEA, 0x2A, 0x6A]),  # families 0,1,2,3 (via intron)
        ]

        uniform_entropy = np.log(256)
        print(f"\nUniform entropy (ln(256)): {uniform_entropy:.4f} nats")
        print(f"{'Sequence':<20} {'Last-pos entropy':>17} {'Top-1 prob':>11} "
              f"{'Top-1 byte':>11} {'Top-5 mass':>11}")
        print("-" * 70)

        entropies = []
        for name, seq in test_sequences:
            logits = get_logits(model, seq)
            last_logits = logits[0, -1]
            probs = F.softmax(last_logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
            entropies.append(entropy)
            top1_prob = probs.max().item()
            top1_byte = probs.argmax().item()
            top5 = torch.topk(probs, 5).values.sum().item()
            print(f"{name:<20} {entropy:>17.4f} {top1_prob:>11.4f} "
                  f"{top1_byte:>11d} {top5:>11.4f}")

        mean_entropy = np.mean(entropies)
        verdict = "uniform noise" if mean_entropy > 5.0 else (
            "concentrated" if mean_entropy < 4.5 else "some structure"
        )
        print(f"\nMean entropy: {mean_entropy:.4f} -> {verdict}")

        for name, seq in test_sequences:
            if len(seq) > 1:
                logits = get_logits(model, seq)
                pos_entropies = []
                for p in range(logits.shape[1]):
                    probs = F.softmax(logits[0, p], dim=-1)
                    e = -torch.sum(probs * torch.log(probs + 1e-10)).item()
                    pos_entropies.append(e)
                var_ent = np.var(pos_entropies)
                print(f"  {name}: pos entropy var={var_ent:.4f} (0=flat)")
                break

    def test_prediction_consistency(self, model):
        """
        Does the model predict the same thing for the same input?
        (Sanity check that deterministic forward pass is consistent.)
        """
        seq = [10, 20, 30, 40, 50]
        logits1 = get_logits(model, seq)
        logits2 = get_logits(model, seq)
        diff = (logits1 - logits2).abs().max().item()
        print(f"\nDeterminism check: max logit difference = {diff:.2e} "
              f"({'PASS' if diff < 1e-5 else 'FAIL'})")
        assert diff < 1e-5, "Forward pass is not deterministic"

    def test_position_sensitivity(self, model):
        """
        Does the model produce different outputs for different positions
        of the same byte? If logits are identical across positions,
        the temporal context (attention + L4) is not functioning.
        """
        print("\n--- Position Sensitivity ---")
        seq = [65, 66, 67, 68, 65]  # 'A','B','C','D','A'
        logits = get_logits(model, seq)

        # Compare logits at position 0 (first 'A') vs position 4 (second 'A')
        pos0 = logits[0, 0]
        pos4 = logits[0, 4]
        cosine = F.cosine_similarity(pos0.unsqueeze(0), pos4.unsqueeze(0)).item()
        l2_dist = (pos0 - pos4).norm().item()
        print(f"Same byte 'A' at pos 0 vs 4: cosine={cosine:.4f}, L2={l2_dist:.4f}")
        differentiated = cosine < 0.99
        print(f"  -> {'Positions differentiated' if differentiated else 'WARNING: Nearly identical'}")
        assert differentiated, "Model outputs same logits for same byte at different positions"


# ===========================================================================
# 2. PHYSICS ALIGNMENT
# ===========================================================================

class TestPhysicsAlignment:
    """Does the model's output respect the structural byte decomposition?"""

    def test_family_prediction_bias(self, model):
        """
        For each input family, what family does the model predict next?
        In random FSM walks, all families should be roughly equally likely.
        """
        print("\n" + "=" * 70)
        print("PHYSICS ALIGNMENT: FAMILY STRUCTURE")
        print("=" * 70)

        family_to_bytes = {f: [] for f in range(4)}
        for b in range(256):
            intr = physics.intron(b)
            fam = (intr >> 6) & 0x3
            family_to_bytes[fam].append(b)

        print(f"\nFamily sizes: {[len(v) for v in family_to_bytes.values()]} "
              f"(should all be 64)")

        # For each family, create short sequences and check predicted family
        print(f"\n{'Input family':>13} {'Pred fam 0':>11} {'Pred fam 1':>11} "
              f"{'Pred fam 2':>11} {'Pred fam 3':>11}")
        print("-" * 58)

        for input_fam in range(4):
            fam_counts = Counter()
            for b in family_to_bytes[input_fam]:
                logits = get_logits(model, [b])
                pred_byte = logits[0, -1].argmax().item()
                pred_fam = (physics.intron(pred_byte) >> 6) & 0x3
                fam_counts[pred_fam] += 1
            total = sum(fam_counts.values())
            fracs = [fam_counts[f] / total for f in range(4)]
            dom = max(range(4), key=lambda f: fracs[f])
            print(f"  Family {input_fam}:   "
                  + "  ".join(f"{fr:>9.1%}" for fr in fracs)
                  + f"  (dom={dom})")

    def test_vertex_charge_structure(self, model):
        """
        Group all 256 bytes by vertex charge (0-3).
        Check: do bytes in the same vertex class get similar logit patterns?
        """
        print("\n--- Vertex Charge Clustering ---")

        vertex_to_bytes = {v: [] for v in range(4)}
        for b in range(256):
            props = byte_to_physics(b)
            vertex_to_bytes[props["vertex"]].append(b)

        print(f"Vertex class sizes: {[len(v) for v in vertex_to_bytes.values()]}")

        # For each vertex class, compute mean logit vector
        vertex_means = {}
        for v in range(4):
            logit_vecs = []
            for b in vertex_to_bytes[v]:
                logits = get_logits(model, [b])
                logit_vecs.append(logits[0, -1])
            vertex_means[v] = torch.stack(logit_vecs).mean(dim=0)

        # Cross-vertex cosine similarity
        off_diag = []
        print(f"\n{'':>8}", end="")
        for v in range(4):
            print(f"{'V' + str(v):>8}", end="")
        print()
        for vi in range(4):
            print(f"  V{vi}:  ", end="")
            for vj in range(4):
                cos = F.cosine_similarity(
                    vertex_means[vi].unsqueeze(0),
                    vertex_means[vj].unsqueeze(0)
                ).item()
                if vi != vj:
                    off_diag.append(cos)
                print(f"{cos:>8.3f}", end="")
            print()
        avg_off = np.mean(off_diag)
        print(f"  Off-diag mean: {avg_off:.3f} "
              f"({'vertex separation' if avg_off < 0.99 else 'all collapsed'})")

    def test_mask_weight_effect(self, model):
        """
        Check: does mask weight (popcount of mask12) affect output?
        Group bytes by mask weight and compare predictions.
        """
        print("\n--- Mask Weight Effect ---")

        weight_groups = {}
        for b in range(256):
            props = byte_to_physics(b)
            w = props["mask_weight"]
            weight_groups.setdefault(w, []).append(b)

        print(f"{'Weight':>7} {'Count':>6} {'Mean entropy':>13} {'Mean top-1 prob':>16}")
        print("-" * 45)

        for w in sorted(weight_groups.keys()):
            ents = []
            top1s = []
            for b in weight_groups[w]:
                logits = get_logits(model, [b])
                probs = F.softmax(logits[0, -1], dim=-1)
                ent = -torch.sum(probs * torch.log(probs + 1e-10)).item()
                ents.append(ent)
                top1s.append(probs.max().item())
            print(f"{w:>7} {len(weight_groups[w]):>6} "
                  f"{np.mean(ents):>13.4f} {np.mean(top1s):>16.4f}")


# ===========================================================================
# 3. CURRICULUM FAMILY ANALYSIS
# ===========================================================================

class TestCurriculumFamilies:
    """How does the model behave on each type of training data?"""

    def test_random_walk_loss(self, model):
        """Loss on typical random walk sequences."""
        print("\n" + "=" * 70)
        print("CURRICULUM FAMILY ANALYSIS")
        print("=" * 70)

        import random
        rng = random.Random(42)

        losses = []
        for _ in range(20):
            seq = [rng.randint(0, 255) for _ in range(64)]
            inp = make_sequence(seq)
            with torch.no_grad():
                out = model(input_ids=inp, labels=inp)
            if out.loss is not None:
                losses.append(out.loss.item())

        mean_loss = np.mean(losses)
        uniform = np.log(256)
        print(f"\nRandom walks (n=20, len=64):")
        print(f"  Mean loss: {mean_loss:.4f} (uniform={uniform:.4f})")
        print(f"  Std loss:  {np.std(losses):.4f}")
        print(f"  Gap from uniform: {mean_loss - uniform:+.4f} "
              f"({'no learning' if abs(mean_loss - uniform) < 0.05 else 'learning'})")

    def test_closure_detection(self, model):
        """
        P7 test: Does the model predict differently after xyxy (closure)
        vs xyxz (non-closure)?
        """
        print("\n--- P7 Closure Detection ---")

        closure_entropies = []
        nonclosure_entropies = []

        for x in range(0, 256, 32):
            for y in range(0, 256, 32):
                if x == y:
                    continue
                z = (y + 1) % 256
                if z == x:
                    z = (z + 1) % 256

                # Closure: xyxy
                logits_c = get_logits(model, [x, y, x, y])
                probs_c = F.softmax(logits_c[0, -1], dim=-1)
                ent_c = -torch.sum(probs_c * torch.log(probs_c + 1e-10)).item()
                closure_entropies.append(ent_c)

                # Non-closure: xyxz
                logits_nc = get_logits(model, [x, y, x, z])
                probs_nc = F.softmax(logits_nc[0, -1], dim=-1)
                ent_nc = -torch.sum(probs_nc * torch.log(probs_nc + 1e-10)).item()
                nonclosure_entropies.append(ent_nc)

        print(f"  Closure (xyxy) mean entropy:     {np.mean(closure_entropies):.4f}")
        print(f"  Non-closure (xyxz) mean entropy:  {np.mean(nonclosure_entropies):.4f}")
        diff = np.mean(nonclosure_entropies) - np.mean(closure_entropies)
        print(f"  Difference:                       {diff:+.4f} "
              f"({'closure has lower entropy [OK]' if diff > 0.01 else 'no differentiation'})")

    def test_reference_byte_special(self, model):
        """
        Byte 0xAA is the reference (intron=0, zero mutation).
        Does the model treat it differently?
        """
        print("\n--- Reference Byte 0xAA ---")

        ref_logits = get_logits(model, [0xAA])
        ref_probs = F.softmax(ref_logits[0, -1], dim=-1)
        ref_entropy = -torch.sum(ref_probs * torch.log(ref_probs + 1e-10)).item()
        ref_top5 = top_k_bytes(ref_logits[0, -1], 5)

        # Compare with a random byte
        other_logits = get_logits(model, [0x55])
        other_probs = F.softmax(other_logits[0, -1], dim=-1)
        other_entropy = -torch.sum(other_probs * torch.log(other_probs + 1e-10)).item()

        print(f"  0xAA (reference) entropy: {ref_entropy:.4f}")
        print(f"  0x55 (complement) entropy: {other_entropy:.4f}")
        print(f"  0xAA top-5 predictions: {ref_top5}")


# ===========================================================================
# 4. HEAD DECOMPOSITION ANALYSIS
# ===========================================================================

class TestHeadDecomposition:
    """Are the family_head and micro_head producing structured outputs?"""

    def test_head_logit_decomposition(self, model):
        """
        Directly inspect the family_head (4 logits) and micro_head (64 logits)
        before they are combined into 256.
        """
        print("\n" + "=" * 70)
        print("HEAD DECOMPOSITION ANALYSIS")
        print("=" * 70)

        # Run a forward pass manually to get the intermediate head outputs
        seq = list(b"Test")
        input_ids = make_sequence(seq)

        with torch.no_grad():
            introns = (input_ids ^ model.config.gene_mic_s).to(torch.uint8)
            families = (torch.bitwise_right_shift(introns.int(), 6) & 0x3).long()
            micro_refs = (introns & 0x3F).long()
            mt = getattr(model, "mask12_table")
            mask12s = mt[input_ids.clamp(0, 255)].long() & 0xFFF

            l1_states = physics.compute_l1_trajectory(introns)
            vertices = physics.compute_vertex_batch(mask12s, model.config.q0, model.config.q1)
            l2_a8, l2_b8 = physics.compute_l2_trajectory(introns)
            l3_a12, l3_b12 = physics.compute_l3_trajectory(introns, mask12s)
            l4_O, l4_E = physics.compute_l4_commitments(mask12s)

        # Get to the head inputs by running the full model but intercepting
        head = model.head
        with torch.no_grad():
            # Full forward to get bl3, tl3
            out = model(input_ids=input_ids)
            # We can't easily intercept — use the head's subcomponents directly
            # Instead, check the head weights
            fam_w = head.family_head.weight.data  # [4, dim]
            micro_w = head.micro_head.weight.data  # [64, dim]

        fam_norms = fam_w.norm(dim=1)
        micro_norms = micro_w.norm(dim=1)

        print(f"\nFamily head weight norms (4 classes):")
        for i in range(4):
            print(f"  Family {i}: {fam_norms[i]:.4f}")

        print(f"\nMicro head weight norms (64 classes):")
        print(f"  Mean: {micro_norms.mean():.4f}")
        print(f"  Std:  {micro_norms.std():.4f}")
        print(f"  Min:  {micro_norms.min():.4f}")
        print(f"  Max:  {micro_norms.max():.4f}")

        # Check: are the family logits differentiated in the output?
        logits_256 = out.logits[0, -1]  # [256]

        # Reconstruct per-family marginal
        family_marginals = []
        for f in range(4):
            fam_bytes = [(f << 6 | m) ^ physics.GENE_MIC_S for m in range(64)]
            fam_logits = logits_256[fam_bytes]
            family_marginals.append(fam_logits.mean().item())

        print(f"\nPer-family mean logits in output:")
        for f in range(4):
            print(f"  Family {f}: {family_marginals[f]:.4f}")

        spread = max(family_marginals) - min(family_marginals)
        print(f"  Spread: {spread:.4f} "
              f"({'families differentiated' if spread > 0.1 else 'families collapsed'})")


# ===========================================================================
# 5. EMBEDDING ANALYSIS
# ===========================================================================

class TestEmbeddingStructure:
    """Have the embeddings differentiated from random initialization?"""

    def test_bl1_embedding_structure(self, model):
        """Check if byte, family, micro embeddings have learned structure."""
        print("\n" + "=" * 70)
        print("EMBEDDING STRUCTURE ANALYSIS")
        print("=" * 70)

        byte_emb = model.bl1.byte_embed.weight.data  # [256, dim//2]
        fam_emb = model.bl1.family_embed.weight.data  # [4, dim//4]
        micro_emb = model.bl1.micro_embed.weight.data  # [64, dim//4]

        print(f"\nByte embedding [256, {byte_emb.shape[1]}]:")
        print(f"  Mean norm: {byte_emb.norm(dim=1).mean():.4f}")
        print(f"  Std norm:  {byte_emb.norm(dim=1).std():.4f}")

        # Exhaustive: all same-family pairs vs cross-family pairs
        family_to_bytes = {f: [] for f in range(4)}
        for b in range(256):
            fam = (physics.intron(b) >> 6) & 0x3
            family_to_bytes[fam].append(b)

        same_fam_cos = []
        diff_fam_cos = []
        for fi in range(4):
            bi_list = family_to_bytes[fi]
            for i, bi in enumerate(bi_list):
                for j in range(i + 1, len(bi_list)):
                    bj = bi_list[j]
                    cos = F.cosine_similarity(
                        byte_emb[bi].unsqueeze(0), byte_emb[bj].unsqueeze(0)
                    ).item()
                    same_fam_cos.append(cos)
            for fj in range(4):
                if fi == fj:
                    continue
                for bi in bi_list[:16]:
                    for bj in family_to_bytes[fj][:16]:
                        cos = F.cosine_similarity(
                            byte_emb[bi].unsqueeze(0), byte_emb[bj].unsqueeze(0)
                        ).item()
                        diff_fam_cos.append(cos)

        same_mean = np.mean(same_fam_cos)
        diff_mean = np.mean(diff_fam_cos)
        delta = same_mean - diff_mean
        print(f"\n  Same-family pairs: {len(same_fam_cos)} cosine mean={same_mean:.4f}")
        print(f"  Cross-family pairs: {len(diff_fam_cos)} cosine mean={diff_mean:.4f}")
        print(f"  Delta (same - cross): {delta:+.4f} "
              f"({'family clustering' if delta > 0.02 else 'no structure'})")

        print(f"\nFamily embedding [4, {fam_emb.shape[1]}]:")
        for i in range(4):
            for j in range(i + 1, 4):
                cos = F.cosine_similarity(
                    fam_emb[i].unsqueeze(0), fam_emb[j].unsqueeze(0)
                ).item()
                print(f"  F{i}-F{j} cosine: {cos:.4f}")

    def test_tl1_embedding_structure(self, model):
        """Check TL1: L1 states differing by 1 bit vs 4+ bits."""
        print("\n--- TL1 (L1 State) Embeddings ---")

        l1_emb = model.tl1.l1_state_embed.weight.data  # [256, dim]
        vtx_emb = model.tl1.vertex_embed.weight.data  # [4, 4]

        def hamming(a: int, b: int) -> int:
            return bin((a ^ b) & 0xFF).count("1")

        adjacent_cos = []
        distant_cos = []
        for s in range(256):
            for flip in [1, 2, 4, 8]:
                neighbor = s ^ flip
                cos = F.cosine_similarity(
                    l1_emb[s].unsqueeze(0), l1_emb[neighbor].unsqueeze(0)
                ).item()
                adjacent_cos.append(cos)
            distant = s ^ 0x0F
            if hamming(s, distant) >= 2:
                cos = F.cosine_similarity(
                    l1_emb[s].unsqueeze(0), l1_emb[distant].unsqueeze(0)
                ).item()
                distant_cos.append(cos)

        adj_mean = np.mean(adjacent_cos)
        dist_mean = np.mean(distant_cos)
        print(f"  1-bit flip (4096 pairs): cosine mean={adj_mean:.4f}")
        print(f"  4-bit flip s^0x0F (256 pairs): cosine mean={dist_mean:.4f}")
        diff = adj_mean - dist_mean
        print(f"  Delta (1bit - 4bit): {diff:+.4f} "
              f"({'1bit more similar' if diff > 0.01 else 'flat'})")

        print(f"\n  Vertex embeddings [4, 4]:")
        for i in range(4):
            print(f"    V{i}: {vtx_emb[i].tolist()}")


# ===========================================================================
# 6. GRADIENT DIAGNOSTICS
# ===========================================================================

class TestGradientFlow:
    """Which components received meaningful gradient during training?"""

    def test_parameter_norms_by_component(self, model):
        """
        Print parameter norms for each named component.
        Components with near-initialization norms haven't been updated.
        """
        print("\n" + "=" * 70)
        print("PARAMETER NORMS BY COMPONENT")
        print("=" * 70)

        components = {}
        for name, param in model.named_parameters():
            # Group by top-level component
            top = name.split(".")[0]
            if top not in components:
                components[top] = {"total_norm": 0.0, "count": 0, "params": 0}
            components[top]["total_norm"] += param.data.norm().item() ** 2
            components[top]["count"] += 1
            components[top]["params"] += param.numel()

        print(f"\n{'Component':<20} {'Params':>10} {'RMS norm':>12} {'# tensors':>10}")
        print("-" * 55)
        for comp, info in sorted(components.items()):
            rms = (info["total_norm"] / info["count"]) ** 0.5
            print(f"{comp:<20} {info['params']:>10,} {rms:>12.4f} {info['count']:>10}")

    def test_gradient_one_step(self, model):
        """
        Run one forward+backward and check gradient norms per component.
        Components with zero gradients are disconnected from the loss.
        """
        print("\n--- Gradient Norms (single step) ---")

        model.train()
        inp = torch.randint(0, 256, (2, 16))
        out = model(input_ids=inp, labels=inp)
        out.loss.backward()

        components = {}
        for name, param in model.named_parameters():
            top = name.split(".")[0]
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
            else:
                grad_norm = 0.0
            if top not in components:
                components[top] = {"max_grad": 0.0, "any_nonzero": False}
            if grad_norm > components[top]["max_grad"]:
                components[top]["max_grad"] = grad_norm
            if grad_norm > 0:
                components[top]["any_nonzero"] = True

        print(f"\n{'Component':<20} {'Max grad norm':>14} {'Has gradient':>13}")
        print("-" * 50)
        for comp, info in sorted(components.items()):
            status = "yes" if info["any_nonzero"] else "DEAD"
            print(f"{comp:<20} {info['max_grad']:>14.6f} {status:>13}")

        model.eval()
        model.zero_grad()


# ===========================================================================
# 7. GENERATION SAMPLE
# ===========================================================================

class TestGeneration:
    """What does the model actually produce?"""

    def test_generate_samples(self, model):
        """Generate a few byte sequences and display them."""
        print("\n" + "=" * 70)
        print("GENERATION SAMPLES")
        print("=" * 70)

        prompts = [
            ("from_reference", [0xAA]),
            ("from_hello", list(b"Hello")),
            ("from_random", [42, 137, 200]),
        ]

        for name, prompt in prompts:
            generated = model.generate(
                make_sequence(prompt),
                max_new_tokens=32,
                temperature=1.0,
                do_sample=True,
            )
            gen_bytes = generated[0].tolist()
            new_bytes = gen_bytes[len(prompt):]

            # Analyze generated bytes
            gen_families = [(physics.intron(b) >> 6) & 0x3 for b in new_bytes]
            fam_counts = Counter(gen_families)

            try:
                text = bytes(new_bytes).decode("utf-8", errors="replace")
                safe = text[:40].encode("ascii", errors="replace").decode("ascii")
            except Exception:
                safe = "<decode error>"

            print(f"\n  Prompt '{name}': {prompt}")
            print(f"  Generated bytes: {new_bytes[:20]}{'...' if len(new_bytes) > 20 else ''}")
            print(f"  As text (ascii-safe): {repr(safe)}")
            print(f"  Family distribution: {dict(fam_counts)}")
            print(f"  Unique bytes: {len(set(new_bytes))}/32")


# ===========================================================================
# SUMMARY
# ===========================================================================

class TestSummary:
    """Print a concise summary of model status."""

    def test_print_summary(self, model):
        """One-page summary of model state."""
        print("\n" + "=" * 70)
        print("MODEL SUMMARY")
        print("=" * 70)

        n_params = sum(p.numel() for p in model.parameters())
        n_buffers = sum(b.numel() for b in model.buffers())
        config = model.config

        print(f"\n  Architecture: HGT ({config.model_type})")
        print(f"  Resolution dims: {config.resolution_dims}")
        print(f"  Attention heads:  {config.num_heads}")
        print(f"  Total parameters: {n_params:,}")
        print(f"  Total buffers:    {n_buffers:,}")
        print(f"  Physics constants:")
        print(f"    GENE_MIC_S:    0x{config.gene_mic_s:02X}")
        print(f"    Q0:            0x{config.q0:03X}")
        print(f"    Q1:            0x{config.q1:03X}")
        print(f"    Archetype:     0x{config.archetype_state24:06X}")
        mask12_ok = hasattr(model, "mask12_table") and getattr(model, "mask12_table").shape[0] == 256
        print(f"    mask12 buffer: {'intact' if mask12_ok else 'MISSING'}")

        # Loss by regime (correct yardstick: structured vs unstructured)
        import random as rnd
        rng = rnd.Random(42)
        regimes = [
            ("random", [rng.randint(0, 255) for _ in range(64)]),
            ("closure_xyxy", [10, 20, 10, 20] * 16),
            ("family_locked", [(0 << 6 | rng.randint(0, 63)) ^ physics.GENE_MIC_S
                              for _ in range(64)]),
            ("separator", [rng.randint(0, 255), 0xAA] * 32),
        ]
        uniform_loss = np.log(256)
        print(f"\n  Loss by regime (uniform={uniform_loss:.4f}):")
        regime_losses = {}
        for name, seq in regimes:
            inp = make_sequence(seq)
            with torch.no_grad():
                out = model(input_ids=inp, labels=inp)
            loss = out.loss.item() if out.loss is not None else float("nan")
            regime_losses[name] = loss
            print(f"    {name}: {loss:.4f}")
        loss_closure = regime_losses["closure_xyxy"]
        loss_fam = regime_losses["family_locked"]
        loss_rand = regime_losses["random"]
        struct_mean = (loss_closure + loss_fam + regime_losses["separator"]) / 3
        status = (
            "LEARNED (structure)" if struct_mean < 3.5 and loss_closure < 2.0 else
            "PARTIALLY LEARNED" if struct_mean < 4.5 else
            "NOT LEARNED" if struct_mean > 5.0 else
            "SOME LEARNING"
        )
        print(f"  Learning status: {status} (Phase-1 uses structured curriculum; random~ln256 is expected)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])