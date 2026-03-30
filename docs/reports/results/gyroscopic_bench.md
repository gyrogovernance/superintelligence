(.venv) PS F:\Development\superintelligence> python scripts\bench_gyrolabe.py
All 2 critical native symbols available
GyroLabe benchmark (repeats=8, native=YES)

=== exact kernel ops ===

--- n=256 ---
  signature_scan python: n=256, avg 1.933 ms, 132440.91 items/s
  signature_scan native: n=256, avg 0.012 ms, 22260869.07 items/s (168x vs py)
  qmap_extract python: n=256, avg 3.949 ms, 64831.67 items/s
  qmap_extract native: n=256, avg 0.021 ms, 12032901.38 items/s (186x vs py)
  extract_scan native: n=256, avg 0.028 ms, 9155120.22 items/s
  chirality_distance python: n=256, avg 2.507 ms, 102108.48 items/s
  chirality_distance native: n=256, avg 0.022 ms, 11729668.44 items/s (115x vs py)
  chir_dist_adjacent python: n=256, avg 2.516 ms, 101743.25 items/s
  chir_dist_adjacent native: n=256, avg 0.013 ms, 19015783.67 items/s (187x vs py)
  apply_sig_batch python: n=256, avg 2.179 ms, 117508.00 items/s
  apply_sig_batch native: n=256, avg 0.013 ms, 20398404.79 items/s (174x vs py)
  state_scan python: n=256, avg 1.668 ms, 153493.32 items/s
  state_scan native: n=256, avg 0.009 ms, 27125830.40 items/s (177x vs py) 
  omega_sig_scan python: n=256, avg 2.218 ms, 115431.01 items/s
  omega_sig_scan native: n=256, avg 0.009 ms, 27713133.11 items/s (240x vs py)
  omega12_scan python: n=256, avg 1.847 ms, 138581.57 items/s
  omega12_scan native: n=256, avg 0.013 ms, 20418754.19 items/s (147x vs py)
  shell_hist_state24 python: n=256, avg 3.024 ms, 84654.68 items/s
  shell_hist_state24 native: n=256, avg 0.014 ms, 18400723.01 items/s (217x vs py)
  shell_hist_omega12 python: n=256, avg 3.143 ms, 81445.66 items/s
  shell_hist_omega12 native: n=256, avg 0.008 ms, 30476196.00 items/s (374x vs py)

--- n=4096 ---
  signature_scan python: n=4096, avg 30.657 ms, 133607.28 items/s
  signature_scan native: n=4096, avg 0.018 ms, 224746221.16 items/s (1682x vs py)
  qmap_extract python: n=4096, avg 62.075 ms, 65984.94 items/s
  qmap_extract native: n=4096, avg 0.027 ms, 153768189.12 items/s (2330x vs py)
  extract_scan native: n=4096, avg 0.043 ms, 95366706.31 items/s
  chirality_distance python: n=4096, avg 38.830 ms, 105486.57 items/s
  chirality_distance native: n=4096, avg 0.068 ms, 60546934.06 items/s (574x vs py)
  chir_dist_adjacent python: n=4096, avg 39.161 ms, 104594.09 items/s
  chir_dist_adjacent native: n=4096, avg 0.037 ms, 109922844.76 items/s (1051x vs py)
  apply_sig_batch python: n=4096, avg 34.100 ms, 120118.40 items/s
  apply_sig_batch native: n=4096, avg 0.015 ms, 266623261.19 items/s (2220x vs py)
  state_scan python: n=4096, avg 25.937 ms, 157924.08 items/s
  state_scan native: n=4096, avg 0.015 ms, 279590397.64 items/s (1770x vs py)
  omega_sig_scan python: n=4096, avg 34.123 ms, 120034.93 items/s
  omega_sig_scan native: n=4096, avg 0.017 ms, 246932892.51 items/s (2057x vs py)
  omega12_scan python: n=4096, avg 29.289 ms, 139846.05 items/s
  omega12_scan native: n=4096, avg 0.014 ms, 303407386.46 items/s (2170x vs py)
  shell_hist_state24 python: n=4096, avg 47.565 ms, 86113.76 items/s
  shell_hist_state24 native: n=4096, avg 0.043 ms, 94979708.04 items/s (1103x vs py)
  shell_hist_omega12 python: n=4096, avg 49.349 ms, 83000.54 items/s
  shell_hist_omega12 native: n=4096, avg 0.015 ms, 281029264.30 items/s (3386x vs py)

--- n=65536 ---
  signature_scan python: n=65536, avg 502.729 ms, 130360.51 items/s
  signature_scan native: n=65536, avg 0.133 ms, 493215426.48 items/s (3783x vs py)
  qmap_extract python: n=65536, avg 955.985 ms, 68553.34 items/s
  qmap_extract native: n=65536, avg 0.125 ms, 523137095.23 items/s (7631x vs py)
  extract_scan native: n=65536, avg 0.283 ms, 231811470.04 items/s
  chirality_distance python: n=65536, avg 622.694 ms, 105245.91 items/s
  chirality_distance native: n=65536, avg 0.412 ms, 159072789.77 items/s (1511x vs py)
  chir_dist_adjacent python: n=65536, avg 617.855 ms, 106070.21 items/s
  chir_dist_adjacent native: n=65536, avg 0.421 ms, 155547380.48 items/s (1466x vs py)
  apply_sig_batch python: n=65536, avg 546.541 ms, 119910.42 items/s
  apply_sig_batch native: n=65536, avg 0.051 ms, 1281564315.44 items/s (10688x vs py)
  state_scan python: n=65536, avg 412.662 ms, 158812.95 items/s
  state_scan native: n=65536, avg 0.105 ms, 626688959.99 items/s (3946x vs py)
  omega_sig_scan python: n=65536, avg 541.187 ms, 121096.78 items/s
  omega_sig_scan native: n=65536, avg 0.133 ms, 493958926.29 items/s (4079x vs py)
  omega12_scan python: n=65536, avg 461.756 ms, 141927.81 items/s
  omega12_scan native: n=65536, avg 0.078 ms, 843313469.33 items/s (5942x vs py)
  shell_hist_state24 python: n=65536, avg 795.951 ms, 82336.75 items/s
  shell_hist_state24 native: n=65536, avg 0.318 ms, 206128564.28 items/s (2503x vs py)
  shell_hist_omega12 python: n=65536, avg 851.888 ms, 76930.27 items/s
  shell_hist_omega12 native: n=65536, avg 0.125 ms, 524812791.86 items/s (6822x vs py)

=== OpenCL reference ===

--- batch=64 ---
  OpenCL_f32.gemm_batch: n=64, avg 0.339 ms, 188804.48 items/s
  parity warning for OpenCL_f32 batch=64: max_abs=0.001369
  OpenCL_i32.gemm_batch: n=64, avg 0.250 ms, 255782.58 items/s

--- batch=256 ---
  OpenCL_f32.gemm_batch: n=256, avg 0.743 ms, 344392.69 items/s
  parity warning for OpenCL_f32 batch=256: max_abs=0.001405
  OpenCL_i32.gemm_batch: n=256, avg 0.381 ms, 672202.72 items/s

=== QuBEC holographic matmul ===

--- 64×64 ---
  numpy i64 matmul: n=4096, avg 0.110 ms, 37402121.53 items/s
  torch i64 matmul: n=4096, avg 0.112 ms, 36600022.05 items/s (98% vs numpy)
  matmul_i32_exact: n=4096, avg 0.093 ms, 44221322.76 items/s (118% vs numpy)
  qubec_holographic: n=4096, avg 0.101 ms, 40514342.81 items/s (108% vs numpy)
  qubec_holographic (torch): n=4096, avg 0.100 ms, 41129659.18 items/s (110% vs numpy)
  mixed profile [high dynamic range input]:
    q: bulk=0 spin=19 dense=45
    k: bulk=0 spin=21 dense=43
    pairs: bb=0 bs=0 bd=0 ss=399 sd=1762 dd=1935 total=4096
  qubec symmetric: n=4096, avg 0.083 ms, 49127439.48 items/s (131% vs numpy)
  standard profile [default range input]: init=0.00ms eval=1.00ms total_pairs=4096
    q: bulk=64 spin=0 dense=0
    k: bulk=64 spin=0 dense=0
    pairs: bb=4096 bs=0 bd=0 ss=0 sd=0 dd=0

--- 128×64 ---
  numpy i64 matmul: n=16384, avg 0.423 ms, 38713412.36 items/s
  torch i64 matmul: n=16384, avg 0.471 ms, 34816978.97 items/s (90% vs numpy)
  matmul_i32_exact: n=16384, avg 0.153 ms, 107181290.50 items/s (277% vs numpy)
  qubec_holographic: n=16384, avg 0.140 ms, 116715940.01 items/s (301% vs numpy)
  qubec_holographic (torch): n=16384, avg 0.147 ms, 111332710.39 items/s (288% vs numpy)
  mixed profile [high dynamic range input]:
    q: bulk=0 spin=49 dense=79
    k: bulk=0 spin=41 dense=87
    pairs: bb=0 bs=0 bd=0 ss=2009 sd=7502 dd=6873 total=16384
  qubec symmetric: n=16384, avg 0.130 ms, 126346630.80 items/s (326% vs numpy)
  standard profile [default range input]: init=0.00ms eval=0.00ms total_pairs=16384
    q: bulk=128 spin=0 dense=0
    k: bulk=128 spin=0 dense=0
    pairs: bb=16384 bs=0 bd=0 ss=0 sd=0 dd=0

--- 256×64 ---
  numpy i64 matmul: n=65536, avg 1.766 ms, 37105387.88 items/s
  torch i64 matmul: n=65536, avg 1.846 ms, 35492252.13 items/s (96% vs numpy)
  matmul_i32_exact: n=65536, avg 0.435 ms, 150648812.62 items/s (406% vs numpy)
  qubec_holographic: n=65536, avg 0.252 ms, 260257131.79 items/s (701% vs numpy)
  qubec_holographic (torch): n=65536, avg 0.291 ms, 224996996.94 items/s (606% vs numpy)
  mixed profile [high dynamic range input]:
    q: bulk=0 spin=86 dense=170
    k: bulk=0 spin=69 dense=187
    pairs: bb=0 bs=0 bd=0 ss=5934 sd=27812 dd=31790 total=65536
  qubec symmetric: n=65536, avg 0.201 ms, 326272945.45 items/s (879% vs numpy)
  standard profile [default range input]: init=0.00ms eval=0.00ms total_pairs=65536
    q: bulk=256 spin=0 dense=0
    k: bulk=256 spin=0 dense=0
    pairs: bb=65536 bs=0 bd=0 ss=0 sd=0 dd=0

--- 512×64 ---
  numpy i64 matmul: n=262144, avg 7.460 ms, 35138415.57 items/s
  torch i64 matmul: n=262144, avg 7.427 ms, 35295072.01 items/s (100% vs numpy)
  matmul_i32_exact: n=262144, avg 1.489 ms, 176101033.79 items/s (501% vs numpy)
  qubec_holographic: n=262144, avg 1.001 ms, 261842879.30 items/s (745%   qubec_holographic: n=262144, avg 1.001 ms, 261842879.30 items/s   qubec_holographic: n=262144, avg 1.001 ms, 261842879.30 items/s (745% vs numpy)
  qubec_holographic (torch): n=262144, avg 0.744 ms, 352415135.88 items/s (10x vs numpy)
  mixed profile [high dynamic range input]:
    q: bulk=0 spin=178 dense=334
    k: bulk=0 spin=180 dense=332
    pairs: bb=0 bs=0 bd=0 ss=32040 sd=119216 dd=110888 total=262144  
  qubec symmetric: n=262144, avg 0.893 ms, 293394143.59 items/s (835% vs numpy)
  standard profile [default range input]: init=0.00ms eval=0.00ms total_pairs=262144
    q: bulk=512 spin=0 dense=0
    k: bulk=512 spin=0 dense=0
    pairs: bb=262144 bs=0 bd=0 ss=0 sd=0 dd=0
(.venv) PS F:\Development\superintelligence>

===

(.venv) PS F:\Development\superintelligence> pytest tests\tools\test_gyrolabe_encode.py -v -s
========================== test session starts ===========================
platform win32 -- Python 3.14.2, pytest-9.0.2, pluggy-1.6.0 -- F:\Development\superintelligence\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: F:\Development\superintelligence
configfile: pytest.ini
plugins: anyio-4.12.1
collected 26 items                                                        

tests/tools/test_gyrolabe_encode.py::test_exact_boundary_zero_transcendentals 
[exact boundary - zero transcendentals]
  prompt: 'The QuBEC climate is finite, shell-exact, and byte-native.'
  patch_count: 58
  mean_bytes_per_patch: 1.000
  exact boundary path completed with transcendental calls blocked.
PASSED
tests/tools/test_gyrolabe_encode.py::test_chirality_vs_cosine_speed_fidelity
[chirality vs cosine - speed]
  chirality_distance_adjacent: 0.000011 s
  mock_cosine_adjacent: 0.002950 s
  speedup: 274.7x
  chirality path runs without transcendental operations.

  [fidelity] Chirality distance stratification:
    Chirality Dist 0: count = 1
    Chirality Dist 1: count = 7
    Chirality Dist 2: count = 14
    Chirality Dist 3: count = 23
    Chirality Dist 4: count = 23
    Chirality Dist 5: count = 6
PASSED
tests/tools/test_gyrolabe_encode.py::test_m2_modulated_boundary_threshold  
[M2-modulated boundary threshold]
  M2=64 (condensed): patch_count = 31
  M2=4096 (thermalized): patch_count = 74
  M2 modulates effective threshold; monotonic relationship observed.       
PASSED
tests/tools/test_gyrolabe_encode.py::test_encode_extract_fields_speed_report
[encode extract_fields speed]
  batch: 2
  tokens: 8192
  valid_bytes: 8192
  avg_ms: 1.648
  bytes_per_sec: 4970067.06
  valid_bytes_per_sec: 4970067.06
PASSED
tests/tools/test_gyrolabe_encode.py::test_gyrolabe_opencl_climate_projection_verbose
[gyrolabe OpenCL climate projection]
  batch_shape: (64, 64)
  max_err: 0.0
PASSED
tests/tools/test_gyrolabe_encode.py::TestQubecHolographicMatmul::test_qubec_small_exact PASSED
tests/tools/test_gyrolabe_encode.py::TestQubecHolographicMatmul::test_qubec_symmetric PASSED
tests/tools/test_gyrolabe_encode.py::TestQubecHolographicMatmul::test_qubec_asymmetric_shapes PASSED
tests/tools/test_gyrolabe_encode.py::TestQubecHolographicMatmul::test_qubec_wide_columns PASSED
tests/tools/test_gyrolabe_encode.py::TestQubecHolographicMatmul::test_qubec_width_64_exact PASSED
tests/tools/test_gyrolabe_encode.py::TestQubecHolographicMatmul::test_qubec_extreme_values PASSED
tests/tools/test_gyrolabe_encode.py::TestQubecHolographicMatmul::test_qubec_zero_matrix PASSED
tests/tools/test_gyrolabe_encode.py::TestQubecHolographicMatmul::test_qubec_single_row PASSED
tests/tools/test_gyrolabe_encode.py::TestQubecHolographicMatmul::test_qubec_sparse_values PASSED
tests/tools/test_gyrolabe_encode.py::TestQubecHolographicMatmul::test_qubec_spin_heavy PASSED
tests/tools/test_gyrolabe_encode.py::TestQubecHolographicMatmul::test_qubec_dense_heavy PASSED
tests/tools/test_gyrolabe_encode.py::TestQubecHolographicMatmul::test_qubec_mixed_bulk_spin_dense PASSED
tests/tools/test_gyrolabe_encode.py::TestQubecHolographicMatmul::test_qubec_profile_fields PASSED
tests/tools/test_gyrolabe_encode.py::TestQubecHolographicMatmul::test_qubec_profile_symmetric PASSED
tests/tools/test_gyrolabe_encode.py::TestQubecHolographicMatmul::test_qubec_torch_input PASSED
tests/tools/test_gyrolabe_encode.py::TestQubecHolographicMatmul::test_qubec_matches_matmul_i32 PASSED
tests/tools/test_gyrolabe_encode.py::TestQubecBolmoIntegration::test_qubec_wht64_exact PASSED
tests/tools/test_gyrolabe_encode.py::TestQubecBolmoIntegration::test_qubec_embedding_projection
[qubec embedding projection]
  embed slice: 64×64
  query batch: 16×64
  exact match: YES
PASSED
tests/tools/test_gyrolabe_encode.py::TestQubecBolmoIntegration::test_qubec_attention_score_simulation
[qubec attention simulation]
  seq_len=32, head_dim=64
  asymmetric: exact
  symmetric: exact
PASSED
tests/tools/test_gyrolabe_encode.py::TestQubecBolmoIntegration::test_qubec_speed_vs_torch
[qubec speed vs torch]
  64×64: qubec=0.119ms torch=0.117ms ratio=0.98x
  128×64: qubec=0.152ms torch=0.568ms ratio=3.73x
  256×64: qubec=0.371ms torch=2.000ms ratio=5.40x
PASSED
tests/tools/test_gyrolabe_encode.py::TestQubecBolmoIntegration::test_qubec_real_weight_matmul_speed
[qubec on real Bolmo weights]
  name                                                      shape  bulk  spin dense     qubec    torch  ratio
  model.local_encoder.byte_embedding.weight               256×256     0   model.local_encoder.byte_embedding.weight               256×256    0   256    2.568ms   6.624ms  2.58x
  model.local_encoder.subword_embedding.weight            256×256     0    13   243    3.206ms   6.761ms  2.11x
  model.local_encoder.layers.0.xlstm.q.weight             256×256     0     0   256    2.598ms   7.013ms  2.70x
  model.local_encoder.layers.0.xlstm.k.weight             256×256     0     0   256    2.475ms   7.449ms  3.01x
  model.local_encoder.layers.0.xlstm.v.weight             256×256     0     0   256    2.675ms   7.445ms  2.78x
  model.local_encoder.layers.0.xlstm.ogate_preact.weight      256×256     0     0   256    2.750ms   7.999ms  2.91x
PASSED

========================== 26 passed in 27.68s ===========================
(.venv) PS F:\Development\superintelligence>

===

(.venv) PS F:\Development\superintelligence> & f:/Development/superintelligence/.venv/Scripts/python.exe f:/Development/superintelligence/scripts/bench_gyrograph.py      
native_available=True, opencl_available=True
bench scales: [256, 4096, 65536], repeats: 20
=== trace_word4_batch ===
trace_word4_batch parity OK for n=256
trace_word4_batch native: 256 cells, 20 runs, avg 0.029 ms, 8755130 cells/s, 35020522 words/s
trace_word4_batch indexed opencl: 256 cells, 20 runs, avg 0.283 ms, 905329 cells/s, 3621318 words/s
trace_word4_batch python: 256 cells, 20 runs, avg 0.602 ms, 425044 cells/s, 1700178 words/s
trace_word4_batch parity OK for n=4096
trace_word4_batch native: 4096 cells, 20 runs, avg 0.106 ms, 38525207 cells/s, 154100827 words/s
trace_word4_batch indexed opencl: 4096 cells, 20 runs, avg 0.554 ms, 7391701 cells/s, 29566802 words/s
trace_word4_batch python: 4096 cells, 20 runs, avg 10.910 ms, 375452 cells/s, 1501808 words/s
trace_word4_batch parity OK for n=65536
trace_word4_batch native: 65536 cells, 20 runs, avg 0.351 ms, 186682999 cells/s, 746731997 words/s
trace_word4_batch indexed opencl: 65536 cells, 20 runs, avg 1.522 ms, 43065782 cells/s, 172263129 words/s
trace_word4_batch python: 65536 cells, 20 runs, avg 176.258 ms, 371818 cells/s, 1487271 words/s
=== apply_trace_word4_batch ===
apply_trace_word4_batch parity OK for n=256
apply_trace_word4_batch: 256 cells, 20 runs, avg 0.095 ms, 2700992 cells/s, 10803967 words/s
apply_trace_word4_batch parity OK for n=4096
apply_trace_word4_batch: 4096 cells, 20 runs, avg 0.277 ms, 14764085 cells/s, 59056339 words/s
apply_trace_word4_batch parity OK for n=65536
apply_trace_word4_batch: 65536 cells, 20 runs, avg 4.770 ms, 13739995 cells/s, 54959982 words/s
=== ingest_word4_batch ===
ingest_word4_batch parity OK for n=256
ingest_word4_batch: 256 cells, 20 runs, avg 0.076 ms, 3382664 cells/s, 13530655 words/s
ingest_word4_batch parity OK for n=4096
ingest_word4_batch: 4096 cells, 20 runs, avg 0.264 ms, 15500766 cells/s, 62003066 words/s
ingest_word4_batch parity OK for n=65536
ingest_word4_batch: 65536 cells, 20 runs, avg 4.550 ms, 14403659 cells/s, 57614636 words/s
=== indexed non-contiguous ids ===
ingest_word4_batch_indexed_noncontiguous: 256 cells, 20 runs, avg 0.099 ms, 2595428 cells/s, 10381710 words/s
ingest_word4_batch_indexed_noncontiguous: 4096 cells, 20 runs, avg 0.523 ms, 7834661 cells/s, 31338644 words/s
ingest_word4_batch_indexed_noncontiguous: 65536 cells, 20 runs, avg 8.600 ms, 7620084 cells/s, 30480336 words/s
=== GyroGraph.ingest end-to-end ===
GyroGraph.ingest parity OK for n=256
GyroGraph.ingest native (in-place): 256 cells, 20 runs, avg 0.150 ms, 1701449 cells/s, 6805796 words/s
GyroGraph.ingest python (in-place): 256 cells, 20 runs, avg 7.422 ms, 34492 cells/s, 137966 words/s
GyroGraph.ingest native (reset each run): 256 cells, 20 runs, avg 0.090 ms, 2857143 cells/s, 11428571 words/s
GyroGraph.ingest python (reset each run): 256 cells, 20 runs, avg 6.750 ms, 37924 cells/s, 151695 words/s
GyroGraph.ingest parity OK for n=4096
GyroGraph.ingest native (in-place): 4096 cells, 20 runs, avg 0.263 ms, 15570592 cells/s, 62282369 words/s
GyroGraph.ingest python (in-place): 4096 cells, 20 runs, avg 113.233 ms, 36173 cells/s, 144693 words/s
GyroGraph.ingest native (reset each run): 4096 cells, 20 runs, avg 0.449 ms, 9131545 cells/s, 36526178 words/s
GyroGraph.ingest python (reset each run): 4096 cells, 20 runs, avg 105.458 ms, 38840 cells/s, 155360 words/s
GyroGraph.ingest parity OK for n=65536
GyroGraph.ingest native (in-place): 65536 cells, 20 runs, avg 1.458 ms, 44949400 cells/s, 179797599 words/s
GyroGraph.ingest python (in-place): 65536 cells, 20 runs, avg 1777.776 ms, 36864 cells/s, 147456 words/s
GyroGraph.ingest native (reset each run): 65536 cells, 20 runs, avg 3.533 ms, 18551066 cells/s, 74204264 words/s
GyroGraph.ingest python (reset each run): 65536 cells, 20 runs, avg 1830.029 ms, 35811 cells/s, 143246 words/s
(.venv) PS F:\Development\superintelligence> 

===

PS F:\Development\superintelligence> & f:/Development/superintelligence/.venv/Scripts/Activate.ps1
(.venv) PS F:\Development\superintelligence> pytest tests\tools\test_gyrograph_decode.py tests\tools\test_gyrolabe_encode.py -v -s
=============================== test session starts ================================
platform win32 -- Python 3.14.2, pytest-9.0.2, pluggy-1.6.0 -- F:\Development\superintelligence\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: F:\Development\superintelligence
configfile: pytest.ini
plugins: anyio-4.12.1
collected 13 items                                                                  

tests/tools/test_gyrograph_decode.py::test_exact_selection_zero_transcendentals 
[decode exact - zero transcendentals]
  selected_token: 329
  exact selector completed with transcendental calls blocked.
PASSED
tests/tools/test_gyrograph_decode.py::test_qsector_collapse_drought_elimination      
[q-sector collapse - Drought elimination]
  raw_support_count_mean: 3.35
  exact_support_count_mean: 3.35
  phase_redundancy_mean: 0.0
  512-way flat selection collapsed to 64-sector exact selection.
PASSED
tests/tools/test_gyrograph_decode.py::test_speed_comparison
[speed comparison]
  exact_qsector_select vs softmax+argmax: 1.15x
  chirality_distance_adjacent: 0.000105 s
  wht64 vs numpy WHT: 1.32x
PASSED
tests/tools/test_gyrograph_decode.py::test_decode_bridge_step_speed_report
[decode bridge step speed] batch=1
  boundary_hook avg_ms: 0.263
  select_hook avg_ms: 0.607
  full step avg_ms: 1.046
  full tokens_per_sec: 38236.71

[decode bridge step speed] batch=4
  boundary_hook avg_ms: 1.001
  select_hook avg_ms: 1.810
  full step avg_ms: 2.137
  full tokens_per_sec: 74878.15

[decode bridge step speed] batch=8
  boundary_hook avg_ms: 1.438
  select_hook avg_ms: 2.668
  full step avg_ms: 6.239
  full tokens_per_sec: 51288.71

[decode bridge step speed] batch=16
  boundary_hook avg_ms: 4.056
  select_hook avg_ms: 7.257
  full step avg_ms: 7.575
  full tokens_per_sec: 84493.05
PASSED
tests/tools/test_gyrograph_decode.py::test_generation_overhead_report
[bolmo generation overhead]
  prompt_tokens: 77
  raw_generated: 82
  bridged_generated: 82
  raw_ms: 8628.425
  bridged_ms: 6308.211
  slowdown_ratio: 0.731
  raw_tokens_per_s: 9.50
  bridged_tokens_per_s: 13.00
PASSED
tests/tools/test_gyrograph_decode.py::test_decode_generation_language_quality_metrics

[decode language quality metrics]
  text[:220]: In 2026, exact byte-level decoding should still produce coherent language about the same lenghen as was already available as of 10 years ago as a response from a similar syslog message from a syslog-server ran on a simil
  ascii_ratio: 1.0000
  max_run: 2
  unique_char_ratio: 0.1271
  patch_count: 32
  mean_bpp: 4.938
PASSED
tests/tools/test_gyrograph_decode.py::test_strict_mode_forces_exact_selector
[strict mode exact selector]
  exact_qsector_select calls: 75
PASSED
tests/tools/test_gyrograph_decode.py::test_gyrograph_opencl_backend_usage_verbose    
[gyrograph backend usage]
  backend_counts: {'python': 0, 'cpu_indexed': 0, 'opencl_indexed': 14}
PASSED
tests/tools/test_gyrolabe_encode.py::test_exact_boundary_zero_transcendentals 
[exact boundary - zero transcendentals]
  prompt: 'The QuBEC climate is finite, shell-exact, and byte-native.'
  patch_count: 58
  mean_bytes_per_patch: 1.000
  exact boundary path completed with transcendental calls blocked.
PASSED
tests/tools/test_gyrolabe_encode.py::test_chirality_vs_cosine_speed_fidelity
[chirality vs cosine - speed]
  chirality_distance_adjacent: 0.000011 s
  mock_cosine_adjacent: 0.003214 s
  speedup: 284.1x
  chirality path runs without transcendental operations.

  [fidelity] Chirality distance stratification:
    Chirality Dist 0: count = 1
    Chirality Dist 1: count = 7
    Chirality Dist 2: count = 14
    Chirality Dist 3: count = 23
    Chirality Dist 4: count = 23
    Chirality Dist 5: count = 6
PASSED
tests/tools/test_gyrolabe_encode.py::test_m2_modulated_boundary_threshold
[M2-modulated boundary threshold]
  M2=64 (condensed): patch_count = 31
  M2=4096 (thermalized): patch_count = 74
  M2 modulates effective threshold; monotonic relationship observed.
PASSED
tests/tools/test_gyrolabe_encode.py::test_encode_extract_fields_speed_report
[encode extract_fields speed]
  batch: 2
  tokens: 8192
  valid_bytes: 8192
  avg_ms: 1.959
  bytes_per_sec: 4182504.65
  valid_bytes_per_sec: 4182504.65
PASSED
tests/tools/test_gyrolabe_encode.py::test_gyrolabe_opencl_climate_projection_verbose 
[gyrolabe OpenCL climate projection]
  batch_shape: (64, 64)
  max_err: 0.0
PASSED

========================== 13 passed in 88.64s (0:01:28) =========================== 
(.venv) PS F:\Development\superintelligence> 


