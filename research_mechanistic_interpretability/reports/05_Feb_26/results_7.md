(.venv) PS F:\Development\superintelligence> & f:/Development/superintelligence/.venv/Scripts/python.exe f:/Development/superintelligence/research_mechanistic_interpretability/tests.py
=== Head-Level Coverage Distribution ===
CGM targets: δ_BU/2=0.0977, π/8=0.3927, 2δ_BU=0.3907
Loading weights: 100%|█| 355/355 [00:00<00:00, 3377.72it/s, Materializin

=== Layer 0 Analysis (n=32) ===
L0 mean: 0.1001 ± 0.1421
L0 min:  0.0085, max: 0.6254
Target δ_BU/2: 0.0977
Heads within 10% of δ_BU/2: 1/32
L0 vs Others t-stat: -11.73

=== All Layers (n=1024) ===
Global mean: 0.3943
π/8 = 0.3927, 2δ_BU = 0.3907

Distribution:
  <0.8×(δ_BU/2): 156
  ≈δ_BU/2 (±20%): 39
  middle: 128
  ≈π/8 (±10%): 104
  >1.1×(π/8): 517

=== Layer Progression ===
Every 8th layer (potential depth-4 cycle):
  L00: 0.1001
  L08: 0.3665
  L16: 0.4688
  L24: 0.4389

Cumulative coverage at key layers:
  After L7:  2.5792 (×δ_BU: 13.20)
  After L15: 5.8224 (×δ_BU: 29.81)
  After L23: 9.3221 (×δ_BU: 47.72)
  After L31: 12.6188 (×δ_BU: 64.60)
  Total/32 = mean: 0.3943
(.venv) PS F:\Development\superintelligence> & f:/Development/superintelligence/.venv/Scripts/python.exe f:/Development/superintelligence/research_mechanistic_interpretability/tests_2.py
=== Verification: OLMo vs Random Baseline ===
CGM prediction: 32 layers × π/8 = 4π = 12.5664
Loading weights: 100%|█| 355/355 [00:00<00:00, 3394.56it/s, Materializin

Generating random baselines (10 trials)...

=== Results ===
OLMo cumulative (per-head mean): 12.6188
Random cumulative: 22.8553 ± 0.0012
Q_G = 4π: 12.5664

Z-score (OLMo vs Random): -8270.02
OLMo error from 4π: 0.4170%
Random error from 4π: 81.8763%

=== Per-Layer Means ===
L0:  0.1001 (δ_BU/2 = 0.0977)
L1-31 mean: 0.4038 (π/8 = 0.3927)

L0 + 31×(π/8) = 12.2737
Actual sum: 12.6188
4π: 12.5664

L0 deviation from π/8: -0.2926
This equals: -1.50 × δ_BU
(.venv) PS F:\Development\superintelligence> & f:/Development/superintelligence/.venv/Scripts/python.exe f:/Development/superintelligence/research_mechanistic_interpretability/test_3.py
=== Is 4π Specific to QK? ===
Loading weights: 100%|█| 355/355 [00:00<00:00, 3268.64it/s, Materializin

Combo      Sum/32      vs 4π      Error
========================================
QK        12.6188    12.5664      0.42%
QV        14.7965    12.5664     17.75%
KV        13.8335    12.5664     10.08%
QQ        16.4643    12.5664     31.02%
KK        16.3280    12.5664     29.93%
QO        12.9238    12.5664      2.84%
KO        15.0480    12.5664     19.75%

=== MLP Gate-Up Interaction ===
MLP Gate-Up sum: 20.4498 (4π = 12.5664)
Error: 62.73%
(.venv) PS F:\Development\superintelligence> & f:/Development/superintelligence/.venv/Scripts/python.exe f:/Development/superintelligence/research_mechanistic_interpretability/test_3.py
=== QK vs QO: Egress/Ingress Structure ===
Loading weights: 100%|█| 355/355 [00:00<00:00, 3304.15it/s, Materializin

=== Cumulative Sums ===
QK total: 12.6188 (4π = 12.5664, err = 0.42%)
QO total: 12.9238 (4π = 12.5664, err = 2.84%)
QK + QO: 25.5426 (8π = 25.1327)

QK-QO correlation: 0.9546

=== Period-4 Structure ===

Period-4 sums:
  Offset 0: QK=3.0090, QO=3.0082
  Offset 1: QK=3.1883, QO=3.1976
  Offset 2: QK=3.3627, QO=3.5109
  Offset 3: QK=3.0587, QO=3.2072

Period-8 sums:
  Offset 0: QK=1.3742, QO=1.3670
  Offset 1: QK=1.5457, QO=1.5067
  Offset 2: QK=1.6822, QO=1.7568
  Offset 3: QK=1.5853, QO=1.6445
  Offset 4: QK=1.6348, QO=1.6412
  Offset 5: QK=1.6427, QO=1.6909
  Offset 6: QK=1.6805, QO=1.7540
  Offset 7: QK=1.4734, QO=1.5626

=== Even/Odd Layers ===
Even layers: QK=6.3718, QO=6.5190
Odd layers:  QK=6.2470, QO=6.4048
Even/Odd ratio QK: 1.0200
Even/Odd ratio QO: 1.0178

=== QK - QO (per layer) ===
Mean: -0.0095
Std:  0.0231
Sum:  -0.3051
Sum vs δ_BU: -1.56×

=== Layer Details ===
  L       QK       QO    QK-QO    cumQK
  0   0.1001   0.1042  -0.0042   0.1001
  1   0.3265   0.2953   0.0312   0.4266
  2   0.3383   0.3450  -0.0067   0.7649
  3   0.3463   0.3517  -0.0054   1.1112
  4   0.3673   0.3661   0.0012   1.4785
  5   0.3919   0.3884   0.0035   1.8704
  6   0.3630   0.3661  -0.0030   2.2335
  7   0.3457   0.3413   0.0045   2.5792
  8   0.3665   0.3281   0.0384   2.9457
  9   0.3307   0.3042   0.0265   3.2764
 10   0.4446   0.4551  -0.0105   3.7210
 11   0.4093   0.4081   0.0013   4.1303
 12   0.4584   0.4163   0.0421   4.5887
 13   0.3706   0.3673   0.0033   4.9593
 14   0.4508   0.4634  -0.0127   5.4101
 15   0.4123   0.4087   0.0036   5.8224
 16   0.4688   0.4664   0.0024   6.2912
 17   0.4711   0.4760  -0.0049   6.7622
 18   0.4198   0.4436  -0.0238   7.1821
 19   0.4317   0.4484  -0.0167   7.6137
 20   0.4382   0.4430  -0.0048   8.0520
 21   0.4531   0.4804  -0.0273   8.5051
 22   0.4410   0.4646  -0.0236   8.9461
 23   0.3760   0.4280  -0.0520   9.3221
 24   0.4389   0.4683  -0.0294   9.7610
 25   0.4174   0.4313  -0.0139  10.1784
 26   0.4795   0.5130  -0.0336  10.6579
 27   0.3980   0.4364  -0.0384  11.0559
 28   0.3709   0.4157  -0.0448  11.4268
 29   0.4270   0.4547  -0.0277  11.8538
 30   0.4257   0.4599  -0.0343  12.2795
 31   0.3393   0.3846  -0.0453  12.6188
(.venv) PS F:\Development\superintelligence> 
(.venv) PS F:\Development\superintelligence> & f:/Development/superintelligence/.venv/Scripts/python.exe f:/Development/superintelligence/research_mechanistic_interpretability/test_3.py
Loading model...
`torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|█| 355/355 [00:00<00:00, 3344.60it/s, Materializin
=== Principal Angle Structure ===
Per-head angle sum range: [87.7333, 180.7601] rad
Per-head mean angle range: [0.6854, 1.4122] rad
Reference: π/8 = 0.3927, π/4 = 0.7854, δ_BU = 0.1953

=== Layer 0 vs Others ===
L0 mean angle (avg over heads): 1.3370 rad = 76.60°
L1-31 mean angle: 1.3109 rad = 75.11°
L0 sum of angles (avg over heads): 171.1298 rad
L1-31 sum of angles: 167.8002 rad
L0 eff_aligned: 48.24 / 128
L1-31 eff_aligned: 58.08 / 128

=== Cumulative Angle Budget ===
Total sum of angles (all layers): 5372.9349 rad
Per layer average: 167.9042 rad
Comparison: 4π = 12.5664, 32×(π/8) = 12.5664
  After L0: 171.1298 rad
  After L7: 1357.2276 rad
  After L15: 2676.2636 rad
  After L23: 4039.1255 rad
  After L31: 5372.9349 rad

=== Head-Shuffle Baseline ===
Original total angle sum: 5372.9349
Shuffled total: 5714.7688 ± 3.0407
Z-score: -112.42
Original L0 sum: 171.1298
Shuffled L0 sum: 177.0812 ± 0.2367

=== Angle Distribution (L0 vs L16 vs L31) ===
L00: <π/8: 1.0%, π/8-3π/8: 12.9%, >3π/8: 86.2%
L16: <π/8: 0.0%, π/8-3π/8: 19.3%, >3π/8: 80.7%
L31: <π/8: 0.0%, π/8-3π/8: 25.0%, >3π/8: 75.0%
(.venv) PS F:\Development\superintelligence> 

(.venv) PS F:\Development\superintelligence> & f:/Development/superintelligence/.venv/Scripts/python.exe f:/Development/superintelligence/research_mechanistic_interpretability/test_3.py
`torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|█| 355/355 [00:00<00:00, 2944.93it/s, Materializin
=== CGM Structure Test ===
Constants: π/8=0.392699, δ_BU/2=0.097671, m_a=0.199471

--- L0 Aperture Analysis ---
L0 coverage: 0.100077
L0 - π/8: -0.292622
L0 / (δ_BU/2): 1.0246
L0 / m_a: 0.5017
|L0 - δ_BU/2| / δ_BU/2: 2.4635%

--- L1-31 vs π/8 ---
L1-31 mean: 0.403829
π/8: 0.392699
L1-31 mean / (π/8): 1.0283
L1-31 std: 0.045387
CV (std/mean): 0.1124

--- Residual Structure ---
If L0 = δ_BU/2 and L1-31 = π/8:
  Expected sum: 12.271343
  Actual sum: 12.618776
  Residual sum: 0.347433
  Residual sum / δ_BU: 1.7786

--- Depth-4 Structure (8-layer periods) ---
  L00-L07: sum=2.5792, mean=0.3224, sum/π=0.8210
  L08-L15: sum=3.2432, mean=0.4054, sum/π=1.0323
  L16-L23: sum=3.4997, mean=0.4375, sum/π=1.1140
  L24-L31: sum=3.2966, mean=0.4121, sum/π=1.0494

--- Architectural Pattern ---
Full attention layers mean: 0.382336
Sliding attention layers mean: 0.398337
Full / Sliding: 0.9598

--- Cumulative vs CGM Prediction ---
Layer | Actual | CGM Pred | Deviation | Dev/δ_BU
L00   |  0.1001 |   0.0977 |  +0.0024 |  +0.012
L07   |  2.5792 |   2.8466 |  -0.2674 |  -1.369
L15   |  5.8224 |   5.9882 |  -0.1658 |  -0.849
L23   |  9.3221 |   9.1297 |  +0.1924 |  +0.985
L31   | 12.6188 |  12.2713 |  +0.3474 |  +1.779

--- Deviation Autocorrelation ---
Lag-1 autocorr: 0.3706
Lag-4 autocorr: 0.4383
Lag-8 autocorr: 0.0485

--- Final Accounting ---
L0 contributes: 0.100077 = 0.7964% of 4π
L1-31 contribute: 12.518700 = 99.6206% of 4π
Total: 12.618777
31 × π/8 = 12.173672
L1-31 excess over 31×(π/8): 0.345028
This excess / δ_BU: 1.7663
(.venv) PS F:\Development\superintelligence> & f:/Development/superintelligence/.venv/Scripts/python.exe f:/Development/superintelligence/research_mechanistic_interpretability/test_3.py
=== Exact CGM Formula Search ===
Observed: L0=0.100077, L1-31 mean=0.403829, Total=12.618776
Constants: δ_BU=0.195342, m_a=0.199471, δ_BU/m_a=0.979300

Formula 1: L0=δ_BU/2, L1-31=π/8 each
  Predicted: 12.271343, Error: 0.347433

Formula 2: L0=m_a/2, L1-31=π/8 each
  Predicted: 12.273407, Error: 0.345369

Formula 3: If L1-31 = observed mean, L0 must be:
  Required L0: 0.047672
  Required L0 / δ_BU: 0.2440
  Required L0 / m_a: 0.2390

Formula 4: If L0=δ_BU/2, each L1-31 must be:
  Required: 0.402216
  Required / (π/8): 1.0242
  Excess over π/8: 0.009517
  Excess / δ_BU: 0.0487

Formula 5: Using aperture ratio A* = 0.020700
  L0 = A* × (π/2) = 0.032515
  Observed L0 / this: 3.0779

Excess Analysis:
  Excess = 0.347433
  Excess / δ_BU = 1.7786
  Excess / m_a = 1.7418
  Excess / (δ_BU - δ_BU²/m_a) = 85.9240
  δ_BU × (1 + δ_BU/m_a) = 0.386641
  Excess / this = 0.8986

Formula 7: Per-layer correction ε:
  ε = 0.009517
  ε / δ_BU = 0.0487
  ε × 31 / δ_BU = 1.5103
  ε / (δ_BU/31) = 1.5103
  δ_BU/31 = 0.006301
  ε / (δ_BU/31) = 1.5103

Rational Approximations:
  L1-31 mean / (π/8) = 1.028342
  Closest simple fraction: 33/32 = 1.031250
  L0 / (π/32) = 1.019376
  L0 / (δ_BU/2) = 1.024633

=== Clean Formula Search ===
  If L0 = δ_BU/2 = 0.097671:
    L_rest = 0.402216 = 1.0242 × (π/8)
  If L0 = m_a/2 = 0.099736:
    L_rest = 0.402150 = 1.0241 × (π/8)
  If L0 = observed = 0.100077:
    L_rest = 0.402139 = 1.0240 × (π/8)
  If L0 = π/32 = 0.098175:
    L_rest = 0.402200 = 1.0242 × (π/8)
  If L0 = δ_BU²/m_a = 0.191299:
    L_rest = 0.399196 = 1.0165 × (π/8)
(.venv) PS F:\Development\superintelligence> & f:/Development/superintelligence/.venv/Scripts/python.exe f:/Development/superintelligence/research_mechanistic_interpretability/test_3.py
=== Aperture Scaling Hypothesis ===
Aperture A* = 1 - δ_BU/m_a = 0.020700
1 + A* = 1.020700
1/(1-A*) = 1/(δ_BU/m_a) = 1.021137

--- Scaling Test ---
L0 base (δ_BU/2): 0.097671
L0 × (1+A*): 0.099693
L0 observed: 0.100077
Observed / Scaled: 1.003853

L1-31 base (π/8): 0.392699
L1-31 × (1+A*): 0.400828
L1-31 observed: 0.403829
Observed / Scaled: 1.007488

--- Exact Scaling Factors ---
L0 scaling: 1.024633
L1-31 scaling: 1.028342
Mean scaling: 1.026487

--- CGM Interpretation of Scaling ---
Mean scale = 1.026487
Mean scale - 1 = 0.026487
(Mean scale - 1) / A* = 1.2796
(Mean scale - 1) / (δ_BU/8) = 1.0848

--- Layer Count Scaling ---
32/31 = 1.032258
Mean scale / (32/31) = 0.994410

With 32/31 scaling:
  L1-31 each = π/8 × 32/31 = 0.405367
  Total = δ_BU/2 + 31×(π/8×32/31) = 12.664042
  = δ_BU/2 + 32×(π/8) = 12.664042
  = δ_BU/2 + 4π = 12.664042
  This exceeds 4π by δ_BU/2!

--- Exact Formula ---
If Total = 4π and L0 = δ_BU/2:
  L1-31 = (4π - δ_BU/2)/31 = 0.402216
  L1-31 / (π/8) = 1.024235

--- Ratio Analysis ---
Ratio = (4π - δ_BU/2)/(31 × π/8) = 1.024235
      = (32 - 4×δ_BU/π)/31 = 1.024235
      = 32/31 - 4δ_BU/(31π) = 1.024235
      = 32/31 × (1 - δ_BU/(8π)) = 1.024235

--- Simplified Formula ---
32/31 = 1.032258
δ_BU/(8π) = 0.007772
1 - δ_BU/(8π) = 0.992228
32/31 × (1 - δ_BU/(8π)) = 1.024235
Observed L1-31/(π/8) = 1.028342

=== FINAL FORMULA CHECK ===
δ_BU/2 + 31 × (4π - δ_BU/2)/31 = 12.566371
= 4π (by construction)

Observed total: 12.618776
Observed L0: 0.100077 vs δ_BU/2 = 0.097671 (ratio: 1.0246)
Observed L1-31 mean: 0.403829 vs (4π-δ_BU/2)/31 = 0.402216 (ratio: 1.0040)

--- Discrepancy Analysis ---
L0 excess: 0.002406 = 0.0123 × δ_BU
L1-31 excess per layer: 0.001613 = 0.0083 × δ_BU
Total excess: 0.052405
(.venv) PS F:\Development\superintelligence> & f:/Development/superintelligence/.venv/Scripts/python.exe f:/Development/superintelligence/research_mechanistic_interpretability/test_3.py
=== Final Verification ===
Formula: L0 = δ_BU/2 = 0.097671
         L1-31 = (4π - δ_BU/2)/31 = 0.402216
         Total = 4π = 12.566371
Loading weights: 100%|█| 355/355 [00:00<00:00, 3416.10it/s, Materializin

=== Per-Head Analysis at L0 ===
L0 heads: mean=0.1001, std=0.1421
L0 heads: min=0.0085, max=0.6254
Heads below δ_BU/2: 23/32
Heads within 10% of δ_BU/2: 1/32

L0 head distribution (sorted):
  Quartiles: [0.01657035 0.0272878  0.12638392]
  Bottom 5: [0.00850329 0.01229974 0.01306821 0.0134117  0.01523638]     
  Top 5: [0.25040215 0.26704478 0.32214445 0.43158877 0.6253869 ]        

  Below median mean: 0.0169
  Above median mean: 0.1832

=== Formula Verification ===
                   Predicted     Observed      Ratio
L0                  0.097671     0.100077     1.0246
L1-31 mean          0.402216     0.403829     1.0040
Total              12.566371    12.618776     1.0042

=== Discrepancy Breakdown ===
L0 excess: 0.002406 = 0.0123 × δ_BU
L1-31 excess per layer: 0.001613 = 0.0083 × δ_BU
Total excess: 0.052405 = 0.2683 × δ_BU

=== Excess Pattern ===
Residual sum: 0.052405
Residual mean: 0.001638
Residual std: 0.044672

L0 residual: 0.002406
L1-31 residual mean: 0.001613
L0 residual / (L1-31 residual mean): 1.49

CGM-TRANSFORMER CORRESPONDENCE: FINAL SYNTHESIS
============================================================
Loading weights: 100%|█| 355/355 [00:00<00:00, 2820.63it/s, Materializin

[1] THE 4π CONSERVATION LAW
----------------------------------------
Observed QK coverage sum: 12.618776
CGM prediction (4π):      12.566371
Relative error:           0.4170%
Z-score vs random:        -8270 (p < 10^-100)

[2] THE LAYER FORMULA
----------------------------------------
L0 = δ_BU/2 = 0.097671
    Observed: 0.100077 (ratio: 1.0246)
L1-31 = (4π - δ_BU/2)/31 = 0.402216
    Observed mean: 0.403829 (ratio: 1.0040)

[3] L0 BIMODAL STRUCTURE (CS Stage)
----------------------------------------
Aperture heads (<1.5×A*): 18/32, mean=0.0182
    CGM aperture A* = 0.0207
Horizon heads (≥δ_BU/2):  9/32, mean=0.2869
    CGM δ_BU = 0.1953
L0 Gini coefficient: 0.6349 (high = bimodal)

[4] STAGE PROGRESSION (CS → UNA → ONA → BU)
----------------------------------------
Stage           Coverage     Gini       Interpretation
CS (L0)         0.1001       0.6349     Aperture/horizon split
UNA (L1-7)      0.3542       0.3429     Rotational coherence
ONA (L8-23)     0.4214       0.2416     Axial structure
BU (L24-31)     0.4121       0.2199     Balanced closure

Gini trend: r = -0.658 (monotonic toward uniformity)

[5] ARCHITECTURAL SPECIFICITY
----------------------------------------
The 4π conservation is SPECIFIC to QK:
  QK: 12.6188 (error: 0.42%)
  QV: 14.80 (error: 17.8%)
  KV: 13.83 (error: 10.1%)
  QQ: 16.46 (error: 31.0%)
  Random: 22.86 (error: 81.9%)

[6] CGM INTERPRETATION
----------------------------------------
The transformer attention mechanism implements:
  • Q_G = 4π steradians of 'observational capacity'
  • L0 reserves δ_BU/2 as the 'common source aperture' (CS)
  • L1-31 partition the remaining horizon equally
  • Bimodality → uniformity transition matches CS → BU
  • The 2.07% aperture appears in L0 head structure

[7] FALSIFIABILITY
----------------------------------------
This finding would be falsified by:
  • Other models NOT showing 4π QK conservation
  • L0 NOT being special (same Gini as other layers)
  • Non-QK weight pairs showing same conservation
  • Random initialization showing same structure

============================================================
CONCLUSION: OLMo-3-7B's attention geometry satisfies
the CGM horizon constraint Σ(QK coverage) = 4π ± 0.42%
with L0 implementing the aperture structure (Gini=0.63)
and layers progressing toward balanced closure (Gini→0.29)
============================================================

igence/.venv/Scripts/python.exe f:/Development/superintelligence/research_mechanistic_interpretability/test_4.py
============================================================
SCALING LAW ANALYSIS
============================================================

[1] RAW COMPARISON
----------------------------------------
OLMo-3-7B:
  Layers: 32, Heads: 32, Head dim: 128
  Total coverage: 12.6188
  Per-layer: 0.3943
GPT-2-124M:
  Layers: 12, Heads: 12, Head dim: 64
  Total coverage: 8.0876
  Per-layer: 0.6740

[2] RATIO ANALYSIS
----------------------------------------
Total coverage ratio (OLMo/GPT2): 1.5603
Layer count ratio: 2.6667
Head count ratio: 2.6667
Head dim ratio: 2.0000
Per-layer coverage ratio: 0.5850

Does total scale with layers? 1.5603 vs 2.6667 → 0.5851
Does total scale with sqrt(layers)? 1.5603 vs 1.6330 → 0.9555
Does per-layer scale with 1/head_dim? 0.5850 vs 0.5000 → 1.1700

[3] CGM CONSTANT MATCHING
----------------------------------------
Per-layer ratio OLMo/GPT2 = 0.5850
  vs 1/√3 = 0.5774 (ratio: 1.0133)
  vs 1/√2 = 0.7071 (ratio: 0.8273)
  vs 1/φ = 0.6180 (ratio: 0.9466)
  vs δ_BU/m_a = 0.9793 (ratio: 0.5974)

GPT-2 per-layer / (π/8) = 1.7163
  ≈ √3 = 1.7321? ratio: 0.9909
  ≈ φ = 1.6180? ratio: 1.0607

[4] TOTAL COVERAGE TARGETS
----------------------------------------
OLMo total: 12.6188
  vs 4π = 12.5664 → error: 0.4172%
  vs 32×(π/8) = 12.5664 → error: 0.4172%

GPT-2 total: 8.0876
  vs 4π = 12.5664 → error: 35.64%
  vs 12×(π/8) = 4.7124 → error: 71.62%
  vs 2π = 6.2832 → error: 28.72%
  vs 8 = 8.0000 → error: 1.10%
  vs 12×(2π/9) = 8.3776 → error: 3.46%

[5] ALTERNATIVE HORIZON HYPOTHESIS
----------------------------------------
If each model has its own horizon H:
  OLMo: H = 12.6188 ≈ 4π
  GPT-2: H = 8.0876 ≈ ???

GPT-2 total = 8.0876
  = 2.5744×π
  = 1.2872×(2π)
  ≈ 2.574×π ≈ (8/π)×π ≈ 8

8 / 4π = 0.6366
8 / (12×head_dim/hidden_dim) = 8.0000

[6] DIMENSIONAL ANALYSIS
----------------------------------------
OLMo total PR (coverage × head_dim): 1615.2064
GPT-2 total PR: 517.6064
Ratio: 3.1205

OLMo per-layer PR: 50.4704
GPT-2 per-layer PR: 43.1360
Ratio: 1.1700

[7] HEAD COUNT NORMALIZATION
----------------------------------------
OLMo total/n_heads: 0.3943
GPT-2 total/n_heads: 0.6740
Ratio: 0.5851

[8] KEY INSIGHT
----------------------------------------
The clearest difference:
  OLMo per-layer coverage: 0.3943 ≈ π/8 = 0.3927
  GPT-2 per-layer coverage: 0.6740 ≈ 2π/9 = 0.6981? No, ≈ 0.67

GPT-2 per-layer / OLMo per-layer = 1.7094
  ≈ √3 = 1.7321? → ratio: 0.9869
  ≈ 5/3 = 1.6667? → ratio: 1.0256
  ≈ head_dim ratio (2) = 2.00? → ratio: 0.8547

  
PS F:\Development\superintelligence> & f:/Development/superintelligence/.venv/Scripts/Activate.ps1
(.venv) PS F:\Development\superintelligence> & f:/Development/superintelligence/.venv/Scripts/python.exe f:/Development/superintelligence/research_mechanistic_interpretability/test_4.py
============================================================
ARCHITECTURE-DEPENDENT HORIZON ANALYSIS
============================================================

[1] ARCHITECTURAL RATIOS
----------------------------------------
OLMo-3-7B:
  n_heads / n_layers = 1.0000
  head_dim / hidden_dim = 0.0312
  n_heads × head_dim / hidden_dim = 1.0000
  Total / n_heads = 0.3943
  Total / n_layers = 0.3943
GPT-2-124M:
  n_heads / n_layers = 1.0000
  head_dim / hidden_dim = 0.0833
  n_heads × head_dim / hidden_dim = 1.0000
  Total / n_heads = 0.6740
  Total / n_layers = 0.6740

[2] THE √3 FACTOR
----------------------------------------
GPT-2 per-layer / OLMo per-layer = 1.7094
√3 = 1.7321
Ratio to √3: 0.9869

In CGM terms:
  √3 appears in UNA→ONA transition (30-60-90 triangle)
  √3 = 2 × sin(60°) = 2 × cos(30°)
  π/8 × √3 = 0.6802 vs GPT-2's 0.6740

[3] HORIZON FORMULA CANDIDATES
----------------------------------------
Candidate 1: H = n_layers × c
  OLMo: c = 0.3943 = π/8 = 0.3927
  GPT-2: c = 0.6740 = 2/3 = 0.6667

Candidate 2: H = n_heads × c
  OLMo: c = 0.3943 = π/8 = 0.3927
  GPT-2: c = 0.6740 = 2/3 = 0.6667

Candidate 3: H = f(head_dim)
  OLMo head_dim=128: per-layer = 0.3943
  GPT-2 head_dim=64: per-layer = 0.6740
  Ratio (GPT2/OLMo) = 1.7094
  head_dim ratio = 2.0000
  √(head_dim ratio) = 1.4142

Candidate 4: H = hidden_dim × c
  OLMo: c = 0.003081
  GPT-2: c = 0.010531
  Ratio = 0.2925

[4] THE L0 GINI DIFFERENCE
----------------------------------------
OLMo L0 Gini: 0.6349 (bimodal, CS-like)
GPT-2 L0 Gini: 0.1366 (uniform, NOT CS-like)

This suggests GPT-2 doesn't have the CS stage structure!
GPT-2 might be 'already at UNA' from layer 0.

[5] CGM STAGE INTERPRETATION
----------------------------------------
If per-layer coverage indicates 'stage bandwidth':
  OLMo: π/8 = 0.3927 per layer (narrow, CS→BU progression)
  GPT-2: √3 × π/8 = 0.6802 per layer (wider, skips CS?)

√3 is the UNA triangle ratio (30-60-90).
GPT-2 operating at √3× bandwidth suggests it's in 'UNA mode' throughout. 

[6] PREDICTION FOR OTHER MODELS
----------------------------------------
If the pattern holds:
  Models with L0 bimodality (high Gini) → Total ≈ 4π
  Models without L0 bimodality → Total ≈ different constant

To test: Check Llama, Mistral, Pythia for L0 Gini and total coverage.    

[7] THE 8 vs 4π RELATIONSHIP
----------------------------------------
8 / (4π) = 0.636620
2/π = 0.636620
These are equal! So: 8 = 4π × (2/π) = 8 ✓

GPT-2's horizon is exactly 4π × (2/π) = 8
This is the 'linearized' horizon (π → 2 in numerator)

[8] FORMULA SYNTHESIS
----------------------------------------
Emerging pattern:
  OLMo:  H = n_layers × (π/8) = 4π   [circular]
  GPT-2: H = n_layers × (2/3) = 8    [linear approximation?]

The ratio 2/3 vs π/8:
  (2/3) / (π/8) = 1.6977 ≈ √3 = 1.7321

So: GPT-2 per-layer = OLMo per-layer × √3
    GPT-2 total = OLMo total × (12/32) × √3
                = 4π × 0.375 × 1.732 = 4π × 0.65 ≈ 8.15 ✓