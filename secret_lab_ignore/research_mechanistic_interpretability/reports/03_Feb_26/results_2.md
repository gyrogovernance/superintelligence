(.venv) PS F:\Development\superintelligence> & f:/Development/superintelligence/.venv/Scripts/python.exe f:/Development/superintelligence/research_mechanistic_interpretability/cgm_tomography.py
CGM Holographic Phase Diagram
-----------------------------
[SETUP] Loading Model & Atlas...
Loading weights: 100%|█| 355/355 [00:00<00:00, 2844.22it/s, Material
Loaded codec from data\models\Olmo-3-7B-Instruct\gyro_codebook.npz: 100278 tokens, 4096D
[SETUP] Sampling Tokens...
[SETUP] Prepared 1972 tokens (Balanced K4).

[PHASE MAP] Scanning 32 layers...
L   | Aperture | Stiffness | Monodromy | Stasis  
---------------------------------------------
... Running forward pass to capture states ...
0   | 0.0000   | 0.2705    | 1.0021    | 0.37
1   | 0.0000   | 0.4272    | 0.9796    | 0.34
2   | 0.0000   | 0.4016    | 0.9713    | 0.34
3   | 0.0000   | 0.5816    | 1.0094    | 0.34
4   | 0.0000   | 0.5960    | 0.8636    | 0.34
5   | 0.0000   | 0.5401    | 0.9548    | 0.34
6   | 0.0000   | 0.7166    | 0.9916    | 0.34
7   | 0.0000   | 1.2240    | 1.0300    | 0.34
8   | 0.0000   | 0.6471    | 1.0165    | 0.34
9   | 0.0000   | 0.8623    | 1.0077    | 0.34
10  | 0.0000   | 0.9272    | 0.9823    | 0.34
11  | 0.0000   | 0.8628    | 1.0202    | 0.34
12  | 0.0000   | 0.8615    | 1.0456    | 0.34
13  | 0.0000   | 0.7352    | 1.0271    | 0.34
14  | 0.0000   | 0.6321    | 0.9702    | 0.34
15  | 0.0000   | 1.1065    | 0.9970    | 0.34
16  | 0.0000   | 0.8080    | 1.0223    | 0.34
17  | 0.0000   | 1.1085    | 1.0228    | 0.34
18  | 0.0000   | 0.8734    | 1.0787    | 0.34
19  | 0.0000   | 1.0763    | 0.9977    | 0.34
20  | 0.0000   | 0.5993    | 0.9723    | 0.34
21  | 0.0000   | 1.4326    | 1.0068    | 0.34
22  | 0.0000   | 1.0866    | 0.9881    | 0.34
23  | 0.0000   | 0.8259    | 0.9677    | 0.34
24  | 0.0000   | 0.9178    | 0.9508    | 0.34
25  | 0.0000   | 1.0383    | 0.9388    | 0.34
26  | 0.0000   | 1.1196    | 0.9399    | 0.34
27  | 0.0000   | 1.1187    | 0.9257    | 0.34
28  | 0.0000   | 0.7781    | 0.8472    | 0.34
29  | 0.0000   | 0.2898    | 1.0696    | 0.34
30  | 0.0000   | 0.8329    | 1.0148    | 0.33
31  | 0.0000   | 0.9018    | 1.2089    | 0.33
32  | 0.0000   | 1.0987    | 0.0000    | 0.33
---------------------------------------------

[DIAGNOSIS]
Final Aperture: 0.0000 (Target ~ 0.0207)
>> SYSTEM OVER-DAMPED (Too much Gradient)
Avg Spectral Stiffness: 0.8272 (Lower is more Tetrahedral)
Max MLP Twist: 1.2089

Total dt=1223.4s

(.venv) PS F:\Development\superintelligence> & f:/Development/superintelligence/.venv/Scripts/python.exe f:/Development/superintelligence/research_mechanistic_interpretability/cgm_full_tomography.py
============================================================
CGM FULL TOMOGRAPHY v2
Rigorous Mechanistic Analysis with Checkpointing
============================================================
[RUN ID] d93be88e
[CHECKPOINT] Loaded existing progress: 1 parts completed

[SETUP] Loading model...
`torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|█| 355/355 [00:00<00:00, 3241.57it/s, Material
  Model: Olmo-3-7B-Instruct
  Layers: 32
  Device: cpu

============================================================
RUNNING: weight_structure
============================================================
  Computing random baseline for chirality...
  Baseline chirality (random): 0.7043 ± 0.0035
  Layer 0: chirality mean=0.3953 (+-0.3090 vs baseline)
  Layer 8: chirality mean=0.5451 (+-0.1591 vs baseline)
  Layer 16: chirality mean=0.5290 (+-0.1753 vs baseline)
  Layer 24: chirality mean=0.5174 (+-0.1869 vs baseline)

  Analyzing MLP 256×43 structure via WHT...
  Layer 0: WHT top-16 mass=0.8682, n_sig=1 (baseline: 1.0)
  Layer 8: WHT top-16 mass=0.8812, n_sig=1 (baseline: 1.0)
  Layer 16: WHT top-16 mass=0.7992, n_sig=1 (baseline: 1.0)
  Layer 24: WHT top-16 mass=0.8955, n_sig=1 (baseline: 1.0)
  Layer 31: WHT top-16 mass=0.8302, n_sig=1 (baseline: 1.0)
[CHECKPOINT] Saved: weight_structure

============================================================
RUNNING: subspace_stability
============================================================
  Running forward pass on 512 tokens...
  Computing subspace bases...
    Layer 0: eff_rank=481, top8_ratio=0.0534
    Layer 8: eff_rank=471, top8_ratio=0.2122
    Layer 16: eff_rank=415, top8_ratio=0.3553
    Layer 24: eff_rank=410, top8_ratio=0.3315
    Layer 32: eff_rank=447, top8_ratio=0.3188
  Computing subspace transitions...

  Phase transition detected at layer 20
    Stability before: 0.7386
    Stability after: 0.7685
[CHECKPOINT] Saved: subspace_stability

============================================================
RUNNING: word_algebra
============================================================
  Building controlled byte-token alphabet...
    Found 207 single-byte tokens (coverage: 80.9%)

  Testing depth-2 non-commutativity...
    Final layer non-commutativity: 0.6591

  Testing depth-4 alternation closure...
    D4/D2 ratio: 0.8574
    Closure score: 0.5384
    (Kernel closure score would be 1.0)
[CHECKPOINT] Saved: word_algebra

============================================================
RUNNING: kernel_equivariance
============================================================
  Generating 100 random byte sequences...
  Running model on sequences...
  Training probes and testing equivariance...

    layer_0:
      Vertex accuracy: 1.0000 (baseline: 0.2514)
      Horizon correlation: 0.2008

    layer_16:
      Vertex accuracy: 1.0000 (baseline: 0.2568)
      Horizon correlation: -0.0405

    layer_final:
      Vertex accuracy: 1.0000 (baseline: 0.3005)
      Horizon correlation: -0.0171
[CHECKPOINT] Saved: kernel_equivariance

============================================================
SYNTHESIS
============================================================

KEY FINDINGS:
  1. Subspace phase transition detected at layer 20. Stability jumps from 0.739 to 0.768.
  2. Depth-4 alternation closure is weak (score: 0.538). The kernel has closure_score=1.0. This is a fundamental architectural gap.       
  3. Kernel vertex structure is partially recoverable from hidden states (best layer: 0, accuracy: 1.000).

ARCHITECTURAL IMPLICATIONS:
  1. Consider adding explicit 4-step recurrence or weight-tying between layers to improve depth-4 closure.

SCORES:
  chirality: value=0.5098, above_baseline=-0.1945, significant=False 
  phase_transition: layer=20, stability_jump=0.1223
  depth_4_closure: value=0.5384, ratio=0.8574
  kernel_equivariance: best_layer=0, vertex_accuracy=1.0000, horizon_correlation=0.2008
[CHECKPOINT] Saved: synthesis
[CHECKPOINT] Saved: runtime
[CHECKPOINT] Final results saved to: research_mechanistic_interpretability\results_v2\results_d93be88e.json

============================================================
COMPLETE
============================================================
  Runtime: 1181.8s
  Results: research_mechanistic_interpretability\results_v2\results_d93be88e.json
(.venv) PS F:\Development\superintelligence> 

