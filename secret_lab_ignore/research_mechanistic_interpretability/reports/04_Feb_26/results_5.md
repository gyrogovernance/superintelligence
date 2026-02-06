(.venv) PS F:\Development\superintelligence> & f:/Development/superintelligence/.venv/Scripts/python.exe f:/Development/superintelligence/research_mechanistic_interpretability/gyroscopic_tomography.py
Loading OLMo from data\models\Olmo-3-7B-Instruct...
`torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|█| 355/355 [00:00<00:00, 3142.23it/s, Materializin
Loaded in 0.8s  hidden=4096  layers=32  heads=32
=====
INTEGRITY
=====
Self-loop floor h_norm: 6.596e-07
Gauge invariance |Δh_norm|: 4.470e-08
Neighbor ablation ratio (independent/shared): 1.000
(k,q) stability rel_std: 0.294
=====
HORIZON MRI
=====
Horizon row var/median: 1.152±0.464  permuted: 1.026±0.354  effect_d: 0.30
=====
HORIZON HOLONOMY
=====
r=0.05  horizon16 h_norm 4.687e-01±2.0e-02  proj256 h_norm 1.182e-01±1.7e-02  ratio 3.966
r=0.20  horizon16 h_norm 4.638e-01±2.5e-02  proj256 h_norm 1.464e-01±4.9e-03  ratio 3.168
=====
HORIZON ALGEBRA
=====
Horizon row index: 236
  singular values (generators) top5: [234.342124328315, 181.02611316541316, 149.68624308963493, 126.64242808167864, 83.92582874032465]
  dim99 span{X,Y,[X,Y]}: 3  dim99 with higher commutators: 5
  cos([X,[X,Y]], Y): -0.738  cos([Y,[X,Y]], X): 0.788
Null row index: 237
  singular values (generators) top5: [205.78129625445663, 143.15473106177697, 127.97056232080155, 94.7575778177083, 52.02801832504818]
  dim99 span{X,Y,[X,Y]}: 3  dim99 with higher commutators: 4
  cos([X,[X,Y]], Y): -0.928  cos([Y,[X,Y]], X): 0.698
Total time: 13.7 min
(.venv) PS F:\Development\superintelligence> 

