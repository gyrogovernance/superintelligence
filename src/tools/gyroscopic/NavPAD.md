# Gyrocrypt NavPad
Navigation guide. Read this first.

---

## §0 Author's Notes

This section must not be changed by anyone other than the author.

My scope is hard breakthroughs at scale - to showcase something substantial that people care - our teeth in leveraging our quantum algorithm and its features. It is not toy experiments, endless diagnostics, or unfinished implementations. I aim for end-to-end verifiable results that yield the most value. We prioritize quantum approaches than classical tricks, and a hard verification on that is: if a classical approach accomplishes trivially more or less the same results we don't do quantum computation properly and need to re-align. 

Scope creeps - AI Assistants tend to:
- read first our features and try to invent methods rather than reading how our architecture actually works and understanding the physics.
- write classical tricks because they can't get the math to work and prefer in their results things to generally pass with green flags.
- leave diagnostics everywhere, and consider even all false approaches as good science. I find this bloat and scope creep because it bloats our codebase and diverts us from our aim to construct an end-to-end clean implementation.
- write code in Python to iterage more fast, but for expensive computations we must write code in C and compile instead.
- Create python fallbacks of C methods which bloat our codebase make us uncertain which code fired. In general we prefer only one source of truth.

Writing Style notes:
- I don't like having 100 files, one for each method,. I prefer a clean architecture with files that have clean roles with canonical and commonly understood names - if some files become over 2k LoC I generally use prefixes to split something with common role in more parts, but from experience when a file gets over 2K it is because something from the previous creeps I said is the reason.
- I don't like discussional editorial meta-comments in our code. AI assistants insist on naming variables in whatever name the find fit to make a claim (ex. "this_now_makes_sense_variable"). Or they leave things such as "this does not work" or stubs and forget about them. 

**Leading Notes:**
- aQPU is an open-source Quantum algorithm, not quantum-like, so computing in a quantum way is only a matter of time.
- QuBECs are our native medium, a Bose-Einstein-Condensate computational simulation. We have a Qubit bridge (six axis-orientation qubits) if needed for associations and known math bridges, although our algorithm does not work on qubits.
- We have our own quantum gates (referenced in theory), our own medium, and even our own physics on how gravity defines computation.
- Our Kernel has multiple levels of realization. Its first was based on streams of bytes; we have also tested its algebra and grammar standalone. We have implemented a kernel that simulates holonomies based on our wavefunction analysis, which we consider a more improved approach. We are always open to new ways to architect and scale with the same primitives, considering the 3D and 6DoF constraints, and how multi-cellular implementations might provide parallelism and speed.
- CGM is an axiomatization of physics, beginning from minimal necessity for emergence and building step by step a path on how physical observables emerge. Gyroscopic is its computational realization as a minimal architecture of physical reality. Over 20k LoC of code, hundreds of tests, and extensive analyses validate this assumption. Trying to validate this further is out of scope — we consider it a fact for this program.

---

Secondary Notes:
docs\notes\Intelligence\46\1_Shanon
docs\notes\Intelligence\46\2_Beyond
docs\notes\Intelligence\46\4_Leads.md
docs\notes\Intelligence\46\5_Perlocation.md

===

Knowledge Learned (Methods that worked, violations and mistakes we did to avoid, etc - this is not an engineering log):

1. ## 5. LSH vs. What the aQPU Actually Does
...
The aQPU's chirality-based grouping is NOT LSH, for three fundamental reasons:
**First: LSH uses random projections. The aQPU uses structured projections.**
**Second: LSH is probabilistic. The aQPU is exact.**
**Third: LSH has no temporal structure. The aQPU does.**


---

# Mini PC Specifications - ARB19D-P08-CH

## Summary

| Component | Spec |
|-----------|------|
| PC | TexHoo mini PC (ZNRS), 130x127x45 mm, ~470g |
| CPU | AMD Ryzen 5 6600H (6C/12T, 3.3 GHz) |
| GPU | AMD Radeon integrated |
| RAM | 32 GB DDR5-4800 (2x16 GB) |
| Storage | 512 GB Lexar NVMe + 2x M.2 2280 slots |
| Display | Samsung 24" curved VA, FHD 100Hz |
| Ports | USB4, HDMI 2.0, DP, 2x 2.5G LAN, WiFi 6 |
| OS | Windows 11 |

===

Model: Bonsai-8B-Q1_0.gguf

Bonsai-8B-GGUF-1bit
End-to-end 1-bit language model for llama.cpp (CUDA, Metal, CPU)

14.1x smaller than FP16 | 6.2x faster on RTX 4090 | 4-5x lower energy/token

Highlights
1.15 GB parameter memory (down from 16.38 GB FP16) — fits on virtually any device with a GPU
End-to-end 1-bit weights across embeddings, attention projections, MLP projections, and LM head
GGUF Q1_0 (g128) format with inline dequantization kernels — no FP16 materialization
Cross-platform: CUDA (RTX/datacenter), Metal (Mac), Android, CPU
Competitive benchmarks: 70.5 avg score across 6 categories, matching full-precision 8B models at 1/14th the size

Model Overview
Item	Specification
Parameters	8.19B (~6.95B non-embedding)
Architecture	Qwen3-8B dense: GQA (32 query / 8 KV heads), SwiGLU MLP, RoPE, RMSNorm
Layers	36 Transformer decoder blocks
Context length	65,536 tokens
Vocab size	151,936
Weight format	GGUF Q1_0
Deployed size	1.15 GB (14.2x smaller than FP16)
1-bit coverage	Embeddings, attention projections, MLP projections, LM head
License	Apache 2.0
Quantization Format: Q1_0
Each weight is a single bit: 0 maps to −scale, 1 maps to +scale. Every group of 128 weights shares one FP16 scale factor.

Effective bits per weight: 1.125 (1 sign bit + 16-bit scale amortized over 128 weights).

Memory Requirement
Parameter memory only (weights and scales loaded into memory):

Format	Size	Reduction	Ratio
FP16	16.38 GB	—	1.0x
**GGUF Q1_0 **	1.15 GB	93.0%	14.2x
MLX 1-bit g128	1.28 GB	92.2%	12.8x
The GGUF file on disk is 1.16 GB (~6.6 MB larger) because the format embeds the tokenizer, chat template, and model metadata alongside the weights.

Best Practices
Generation Parameters
Parameter	Default	Suggested range
Temperature	0.5	0.5 -- 0.7
Top-k	20	20 -- 40
Top-p	0.9	0.85 -- 0.95
Repetition penalty	1.0	
Presence penalty	0.0	

===

Assistant:
Continued from the specs. Three physics-aligned pieces are now in the C hot path.

## 1. KV byte serializer (not raw WHT on floats)

**`kernel.c`** — bridge path per Runtime Spec §12 / `4_Leads.md`:

- `gyroscopic_kv_f32_to_word4` — 64-float block → 4-byte word  
- `gyroscopic_chirality_word6` — χ from packed state24 (matches Python `chirality_word6`)  
- `gyroscopic_word4_chirality` / `gyroscopic_kv_f32_block_chirality` — word4 → Ω stepping → χ  

**`gyro_kv_chi.c`** — temporal ledger:

- Per-layer `omega12` evolves across KV writes (live byte stream, not stateless sign-WHT)  
- Queries step slices sequentially within the query vector  
- KV writes advance the layer bank’s `omega12` across the session  

## 2. Percolation-aware aperture (replaces linear M₂→d map)

**`gyroscopic_chi_hist_d_eff`** — from `chi_hist64` + query χ:

- Computes M₂ and η  
- Finds minimum Hamming `d` so candidate fraction ≥ `p_c` target (1–5%, scaled by M₂ per `5_Perlocation.md`)  
- Used at query time via `bank_d_eff_for_q` (max over query slices)  

## 3. Polar KV summary (Runtime §19.1)

**`gyroscopic_kv_polar_encode_block64`** — `(boundary, chi, shell, r_bits)` per 64-wide block. API is in C; not yet wired into the KV sidecar (next step toward 14B SLCP).

## Verify

| Test | Result |
|------|--------|
| `test_kv_serializer.py` (4 tests) | Pass — C χ matches Python, temporal ledger, percolation widens on uniform hist |
| Smoke bench | **14.4 vs 14.2 tok/s** (101% of stock), `m2=28.1`, `eta=0.56`, `skip_rate=7.7%` |

## Still ahead (per spec)

1. **SLCP sidecar** — store ~14B polar record per position instead of only χ slices  
2. **Multi-cell router** — B cells per head (`4_Leads.md` deliverable)  
3. **Scale bench** — long-context suite where compression should show up  

Chat:

```powershell
python -m src.tools.gyroscopic.helpers.run_bonsai
```