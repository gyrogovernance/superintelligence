(.venv) PS F:\Development\superintelligence> & f:/Development/superintelligence/.venv/Scripts/python.exe f:/Development/superintelligence/research_mechanistic_interpretability/gyroscopic_tomography.py
Loading OLMo from data\models\Olmo-3-7B-Instruct...
`torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|█| 355/355 [00:00<00:00, 2785.21it/s, Materializin
Loaded in 1.0s  hidden=4096  layers=32  heads=32
=====
CROSS-LAYER HORIZON MRI
=====
Layer  0: horizon row var/median 1.472±2.329  perm 1.117±0.686  effect_d 0.21
Layer  7: horizon row var/median 1.094±0.611  perm 1.077±0.466  effect_d 0.03
Layer 15: horizon row var/median 1.079±0.319  perm 1.041±0.371  effect_d 0.11
Layer 23: horizon row var/median 1.050±0.351  perm 0.989±0.338  effect_d 0.18
Layer 31: horizon row var/median 1.170±0.704  perm 1.033±0.516  effect_d 0.22
=====
HORIZON HOLONOMY AT LAYERS 7 AND 31
=====

Layer 7:
  r=0.05  horizon16 h_norm 4.701e-01±2.1e-02  proj256 h_norm 1.213e-01±8.2e-04  ratio 3.875
  r=0.20  horizon16 h_norm 5.106e-01±2.5e-02  proj256 h_norm 2.033e-01±8.2e-03  ratio 2.511

Layer 31:
  r=0.05  horizon16 h_norm 4.626e-01±2.6e-02  proj256 h_norm 1.885e-01±1.7e-02  ratio 2.454
  r=0.20  horizon16 h_norm 5.206e-01±9.9e-03  proj256 h_norm 2.339e-01±2.1e-02  ratio 2.226
=====
HORIZON ALGEBRA AT LAYER 31
=====
Horizon row index: 121
  singular values (generators) top5: [203.31573692914344, 168.84480495514597, 145.49733961278858, 95.0426595101901, 84.24472774981308]
  dim99 span{X,Y,[X,Y]}: 3  dim99 with higher commutators: 4
  cos([X,[X,Y]], Y): -0.883  cos([Y,[X,Y]], X): 0.780
Null row index: 122
  singular values (generators) top5: [179.97785130690613, 158.42082351432745, 125.26583647262807, 97.10874314919656, 91.58983143391183]
  dim99 span{X,Y,[X,Y]}: 3  dim99 with higher commutators: 5
  cos([X,[X,Y]], Y): -0.767  cos([Y,[X,Y]], X): 0.765
Total time: 23.0 min
(.venv) PS F:\Development\superintelligence> 

(.venv) PS F:\Development\superintelligence> & f:/Development/superintelligence/.venv/Scripts/python.exe f:/Development/superintelligence/research_mechanistic_interpretability/gyroscopic_tomography.py
`torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|█| 355/355 [00:00<00:00, 3244.70it/s, Materializin
Loaded OLMo in 0.8s  hidden=4096  layers=32  heads=32
=====
BINDING MRI (LOW8 VS BYTES4)
=====
L00  d_low8 +0.19  d_bytes4 -0.10
L07  d_low8 +0.18  d_bytes4 +0.11
L15  d_low8 +0.28  d_bytes4 +0.19
L31  d_low8 +0.10  d_bytes4 +0.22
=====
CURVATURE CONCENTRATION (R=0.20)
=====
L07 low8  h16 0.534  h256 0.231  ratio 2.315
L07 b4  h16 0.503  h256 0.231  ratio 2.183
L15 low8  h16 0.548  h256 0.163  ratio 3.367
L15 b4  h16 0.454  h256 0.163  ratio 2.786
L31 low8  h16 0.552  h256 0.229  ratio 2.410
L31 b4  h16 0.511  h256 0.229  ratio 2.233
=====
NULL SUITE (DIM3/DIM5/MED||G||)
=====
L07 low8 H  d3 3  d5 5  medG 118.96
L07 low8 N  d3 3  d5 4  medG 151.40
L07 low8 R  d3 3  d5 5  medG 146.47
L07 low8 P  d3 3  d5 5  medG 59.47
L07 b4 H  d3 3  d5 4  medG 100.36
L07 b4 N  d3 3  d5 5  medG 111.88
L07 b4 R  d3 3  d5 5  medG 146.47
L07 b4 P  d3 3  d5 5  medG 59.47
L15 low8 H  d3 3  d5 5  medG 105.85
L15 low8 N  d3 3  d5 5  medG 136.89
L15 low8 R  d3 3  d5 5  medG 151.24
L15 low8 P  d3 3  d5 5  medG 133.21
L15 b4 H  d3 3  d5 5  medG 138.60
L15 b4 N  d3 3  d5 5  medG 131.37
L15 b4 R  d3 3  d5 5  medG 151.24
L15 b4 P  d3 3  d5 5  medG 133.21
L31 low8 H  d3 3  d5 5  medG 156.29
L31 low8 N  d3 3  d5 5  medG 88.51
L31 low8 R  d3 3  d5 5  medG 148.60
L31 low8 P  d3 3  d5 5  medG 143.39
L31 b4 H  d3 3  d5 4  medG 133.13
L31 b4 N  d3 3  d5 4  medG 145.53
L31 b4 R  d3 3  d5 5  medG 148.60
L31 b4 P  d3 3  d5 5  medG 143.39
Total time: 11.4 min