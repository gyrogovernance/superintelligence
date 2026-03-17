PS F:\Development\superintelligence> & f:/Development/superintelligence/.venv/Scripts/Activate.ps1
(.venv) PS F:\Development\superintelligence> & f:/Development/superintelligence/.venv/Scripts/python.exe f:/Development/superintelligence/scripts/bench_gyrolabe.py
=== exact kernel ops ===
--- n=256 ---
signature_scan python: n=256, avg 1.922 ms, 133175.53 items/s
signature_scan native: n=256, avg 0.017 ms, 14862118.85 items/s
qmap_extract python: n=256, avg 3.831 ms, 66831.35 items/s
qmap_extract native: n=256, avg 0.020 ms, 12953823.44 items/s
extract_scan native: n=256, avg 0.031 ms, 8379705.46 items/s
chirality_distance python: n=256, avg 2.424 ms, 105606.20 items/s
chirality_distance native: n=256, avg 0.035 ms, 7311674.95 items/s
chirality_distance_adjacent python: n=256, avg 2.428 ms, 105449.06 items/s
chirality_distance_adjacent native: n=256, avg 0.013 ms, 20439120.48 items/s
apply_signature_batch python: n=256, avg 2.180 ms, 117424.46 items/s
apply_signature_batch native: n=256, avg 0.018 ms, 13998632.18 items/s
state_scan_from_state python: n=256, avg 1.606 ms, 159424.58 items/s
state_scan_from_state native: n=256, avg 0.009 ms, 28643371.72 items/s
omega_signature_scan python: n=256, avg 2.123 ms, 120606.09 items/s
omega_signature_scan native: n=256, avg 0.009 ms, 29897811.21 items/s
omega12_scan_from_omega12 python: n=256, avg 1.801 ms, 142161.01 items/s
omega12_scan_from_omega12 native: n=256, avg 0.009 ms, 29425284.78 items/s
shell_histogram_state24 python: n=256, avg 2.966 ms, 86308.98 items/s
shell_histogram_state24 native: n=256, avg 0.008 ms, 31801230.03 items/s
shell_histogram_omega12 python: n=256, avg 3.071 ms, 83371.33 items/s
shell_histogram_omega12 native: n=256, avg 0.008 ms, 31219496.09 items/s
wht64 python ref: n=256, avg 0.024 ms, 10633436.55 items/s
wht64 native: n=256, avg 0.063 ms, 4045032.48 items/s
wht64_metal_first: n=256, avg 0.065 ms, 3922620.06 items/s
--- n=4096 ---
signature_scan python: n=4096, avg 26.480 ms, 154683.80 items/s
signature_scan native: n=4096, avg 0.017 ms, 246932892.51 items/s
qmap_extract python: n=4096, avg 59.612 ms, 68711.06 items/s
qmap_extract native: n=4096, avg 0.025 ms, 163594646.00 items/s
extract_scan native: n=4096, avg 0.040 ms, 103011620.02 items/s
chirality_distance python: n=4096, avg 38.336 ms, 106844.11 items/s
chirality_distance native: n=4096, avg 0.065 ms, 62858237.65 items/s
chirality_distance_adjacent python: n=4096, avg 38.257 ms, 107066.21 items/s
chirality_distance_adjacent native: n=4096, avg 0.036 ms, 112604809.48 items/s
apply_signature_batch python: n=4096, avg 33.553 ms, 122074.79 items/s
apply_signature_batch native: n=4096, avg 0.014 ms, 289725936.80 items/s
state_scan_from_state python: n=4096, avg 25.520 ms, 160501.17 items/s
state_scan_from_state native: n=4096, avg 0.013 ms, 304535303.50 items/s
omega_signature_scan python: n=4096, avg 34.173 ms, 119861.28 items/s
omega_signature_scan native: n=4096, avg 0.024 ms, 172918216.98 items/s
omega12_scan_from_omega12 python: n=4096, avg 28.665 ms, 142890.10 items/s
omega12_scan_from_omega12 native: n=4096, avg 0.013 ms, 327025927.63 items/s
shell_histogram_state24 python: n=4096, avg 45.900 ms, 89237.18 items/s
shell_histogram_state24 native: n=4096, avg 0.024 ms, 168994308.44 items/s
shell_histogram_omega12 python: n=4096, avg 65.230 ms, 62792.74 items/s
shell_histogram_omega12 native: n=4096, avg 0.023 ms, 176266792.52 items/s
wht64 python ref: n=4096, avg 0.291 ms, 14065933.80 items/s
wht64 native: n=4096, avg 0.212 ms, 19286639.03 items/s
wht64_metal_first: n=4096, avg 0.184 ms, 22269946.56 items/s
--- n=65536 ---
signature_scan python: n=65536, avg 440.935 ms, 148629.49 items/s
signature_scan native: n=65536, avg 0.121 ms, 540224627.88 items/s
qmap_extract python: n=65536, avg 959.006 ms, 68337.44 items/s
qmap_extract native: n=65536, avg 0.119 ms, 549683379.41 items/s
extract_scan native: n=65536, avg 0.229 ms, 286245899.74 items/s
chirality_distance python: n=65536, avg 610.520 ms, 107344.52 items/s
chirality_distance native: n=65536, avg 0.449 ms, 145980231.51 items/s
chirality_distance_adjacent python: n=65536, avg 610.917 ms, 107274.88 items/s
chirality_distance_adjacent native: n=65536, avg 0.398 ms, 164694350.63 items/s
apply_signature_batch python: n=65536, avg 540.639 ms, 121219.48 items/s
apply_signature_batch native: n=65536, avg 0.052 ms, 1257285275.13 items/s
state_scan_from_state python: n=65536, avg 408.611 ms, 160387.11 items/s
state_scan_from_state native: n=65536, avg 0.091 ms, 721861489.07 items/s
omega_signature_scan python: n=65536, avg 537.802 ms, 121858.95 items/s
omega_signature_scan native: n=65536, avg 0.125 ms, 523293739.93 items/s
omega12_scan_from_omega12 python: n=65536, avg 475.132 ms, 137932.30 items/s
omega12_scan_from_omega12 native: n=65536, avg 0.077 ms, 847402623.51 items/s
shell_histogram_state24 python: n=65536, avg 799.112 ms, 82011.03 items/s
shell_histogram_state24 native: n=65536, avg 0.290 ms, 226161681.38 items/s
shell_histogram_omega12 python: n=65536, avg 834.364 ms, 78546.06 items/s
shell_histogram_omega12 native: n=65536, avg 0.152 ms, 431441738.66 items/s
wht64 python ref: n=65536, avg 6.319 ms, 10371794.71 items/s
wht64 native: n=65536, avg 2.172 ms, 30175890.98 items/s
wht64_metal_first: n=65536, avg 1.623 ms, 40388564.89 items/s
=== tensor and operator ops ===
--- batch=64 ---
bitplane_gemv python: n=64, avg 386.066 ms, 165.77 items/s
bitplane_gemv native: n=64, avg 0.119 ms, 540084.39 items/s
torch mv: n=64, avg 0.003 ms, 24037572.72 items/s
parity warning for PackedBitplaneMatrix64.gemv parity batch=64: max_abs=0.000865
PackedBitplaneMatrix64.gemv: n=64, avg 0.079 ms, 806807.44 items/s
parity warning for PackedBitplaneMatrix64.gemm_packed_batch parity batch=64: max_abs=0.001241
pack_vector_batch64: n=64, avg 0.060 ms, 1063564.61 items/s
PackedBitplaneMatrix64.gemm_packed_batch: n=64, avg 3.741 ms, 17107.38 items/s
torch mm: n=64, avg 0.012 ms, 5327784.94 items/s
PackedBitplaneMatrix64I32.gemv_packed: n=64, avg 0.058 ms, 1108705.15 items/s        
OpenCLPackedMatrix64.gemm_packed_batch: n=64, avg 0.379 ms, 168715.19 items/s
parity warning for OpenCLPackedMatrix64 parity batch=64: max_abs=0.001368
OpenCLPackedMatrix64I32.gemm_packed_batch: n=64, avg 0.371 ms, 172471.87 items/s
--- batch=256 ---
bitplane_gemv python: n=256, avg 1573.745 ms, 162.67 items/s
bitplane_gemv native: n=256, avg 0.446 ms, 573669.47 items/s
torch mv: n=256, avg 0.003 ms, 78168038.37 items/s
parity warning for PackedBitplaneMatrix64.gemv parity batch=256: max_abs=0.000932
PackedBitplaneMatrix64.gemv: n=256, avg 0.282 ms, 906917.00 items/s
parity warning for PackedBitplaneMatrix64.gemm_packed_batch parity batch=256: max_abs=0.001390
pack_vector_batch64: n=256, avg 0.222 ms, 1152050.42 items/s
PackedBitplaneMatrix64.gemm_packed_batch: n=256, avg 4.188 ms, 61128.31 items/s
torch mm: n=256, avg 0.027 ms, 9394496.62 items/s
PackedBitplaneMatrix64I32.gemv_packed: n=256, avg 0.063 ms, 4079680.96 items/s       
OpenCLPackedMatrix64.gemm_packed_batch: n=256, avg 0.733 ms, 349065.12 items/s
parity warning for OpenCLPackedMatrix64 parity batch=256: max_abs=0.001405
OpenCLPackedMatrix64I32.gemm_packed_batch: n=256, avg 0.429 ms, 597328.36 items/s
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