```markdown
(.venv) PS F:\Development\superintelligence> python -m src.tools.gyrolabe.helpers.test_aqpu_matmul
aQPU bitplane matrix-vector multiplication
Backends: Python (ref) | C unpacked | C packed | OpenCL
----------------------------------------

--- Test 1: Random matrix ---
  random:
    Python (ref): max_err=1.09e-05, pass=True
    C unpacked:   max_err=1.09e-05, pass=True, match_py=True
    C packed:     max_err=7.45e-06, pass=True
    Ops: 4096 float muls/row, 262144 total

--- Test 2: Real Bolmo block ---
Loading weights: 100%|█| 263/263 [00:00<00:00, 3123.82it/s, Mate
  Bolmo q_proj block:
    Python (ref): max_err=8.36e-07, pass=True
    C unpacked:   max_err=8.37e-07, pass=True, match_py=True     
    C packed:     max_err=3.60e-07, pass=True
    Ops: 4096 float muls/row, 262144 total

--- Test 3: Identity matrix ---
  identity:
    Python (ref): max_err=3.46e-05, pass=True
    C unpacked:   max_err=3.46e-05, pass=True, match_py=True     
    C packed:     max_err=3.48e-05, pass=True
    Ops: 4096 float muls/row, 262144 total

--- Test 4: Single nonzero ---
  single nonzero:
    Python (ref): max_err=3.05e-05, pass=True
    C unpacked:   max_err=3.05e-05, pass=True, match_py=True     
    C packed:     max_err=0.00e+00, pass=True
    Ops: 4096 float muls/row, 262144 total

--- Test 5: Batched GEMM (OpenCL vs CPU packed) ---
  batch=64: CPU vs OpenCL max_err=2.98e-08, pass=True

--- Test 6: Integer-native OpenCL vs CPU (i32) ---
  i32 batch=64: CPU vs OpenCL max_err=0, pass=True

----------------------------------------
(.venv) PS F:\Development\superintelligence> python -m src.tools.gyrolabe.helpers.benchmark --all 

GyroLabe Phase 4: Compute Benchmarking
C library: loaded
------------------------------------------------------------
signature_scan n=    64: Python 0.684ms, C 0.009ms, speedup 77.8x
signature_scan n=   256: Python 1.918ms, C 0.009ms, speedup 206.2x
signature_scan n=  1024: Python 6.858ms, C 0.011ms, speedup 647.0x
signature_scan n=  4096: Python 27.139ms, C 0.016ms, speedup 1675.3x
signature_scan n= 16384: Python 106.774ms, C 0.038ms, speedup 2787.8x
signature_scan n= 65536: Python 419.968ms, C 0.124ms, speedup 3376.0x
chirality_distance n=   256: Python 2.444ms, C 0.019ms, speedup 128.6x (13.47M pairs/s)
chirality_distance n=  1024: Python 9.870ms, C 0.024ms, speedup 413.0x (42.85M pairs/s)
chirality_distance n=  4096: Python 39.256ms, C 0.043ms, speedup 921.5x (96.15M pairs/s)
chirality_distance n= 16384: Python 156.868ms, C 0.118ms, speedup 1327.1x (138.61M pairs/s)
chirality_distance n= 65536: Python 632.705ms, C 0.430ms, speedup 1472.1x (152.48M pairs/s)
chirality_distance n=262144: Python 2549.475ms, C 1.627ms, speedup 1567.5x (161.17M pairs/s)
chirality vs cosine (2048d) n=  256: chirality 0.019ms, cosine 0.153ms, speedup 7.9x
chirality vs cosine (2048d) n= 1024: chirality 0.035ms, cosine 2.462ms, speedup 70.5x
chirality vs cosine (2048d) n= 4096: chirality 0.063ms, cosine 10.335ms, speedup 165.1x
chirality vs cosine (2048d) n=16384: chirality 0.118ms, cosine 41.687ms, speedup 353.0x
qmap_extract n=   256: Python 5.622ms, C 0.030ms, speedup 189.3x 
qmap_extract n=  1024: Python 15.465ms, C 0.020ms, speedup 769.4x
qmap_extract n=  4096: Python 60.680ms, C 0.025ms, speedup 2466.7x
qmap_extract n= 16384: Python 245.956ms, C 0.043ms, speedup 5667.2x
qmap_extract n= 65536: Python 1009.698ms, C 0.123ms, speedup 8222.3x
wht64 batch=    1: C 0.024ms, py_wht 0.004ms, dense 0.002ms | vs_py 0.2x, vs_dense 0.1x
wht64 batch=   16: C 0.025ms, py_wht 0.010ms, dense 0.006ms | vs_py 0.4x, vs_dense 0.2x
wht64 batch=   64: C 0.035ms, py_wht 0.012ms, dense 0.008ms | vs_py 0.3x, vs_dense 0.2x
wht64 batch=  256: C 0.046ms, py_wht 0.018ms, dense 0.013ms | vs_py 0.4x, vs_dense 0.3x
wht64 batch= 1024: C 0.119ms, py_wht 0.042ms, dense 0.056ms | vs_py 0.4x, vs_dense 0.5x
wht64 batch= 4096: C 0.495ms, py_wht 0.192ms, dense 0.192ms | vs_py 0.4x, vs_dense 0.4x
wht64 batch=16384: C 2.416ms, py_wht 0.635ms, dense 0.542ms | vs_py 0.3x, vs_dense 0.2x
aqpu_bitplane batch=   1: torch 0.01ms, CPU packed 0.16ms (0.0x vs torch), CPU i32 0.12ms, OpenCL 0.19ms (0.0x vs torch, 0.8x vs CPU), OpenCL i32 0.26ms
aqpu_bitplane batch=  16: torch 0.06ms, CPU packed 0.29ms (0.2x vs torch), CPU i32 1.01ms, OpenCL 0.26ms (0.2x vs torch, 1.1x vs CPU), OpenCL i32 0.32ms
aqpu_bitplane batch=  64: torch 0.25ms, CPU packed 0.92ms (0.3x vs torch), CPU i32 7.37ms, OpenCL 0.42ms (0.6x vs torch, 2.2x vs CPU), OpenCL i32 0.45ms
aqpu_bitplane batch= 256: torch 1.34ms, CPU packed 3.89ms (0.3x vs torch), CPU i32 21.94ms, OpenCL 1.00ms (1.3x vs torch, 3.9x vs CPU), OpenCL i32 0.88ms
aqpu_bitplane batch=1024: torch 5.99ms, CPU packed 13.27ms (0.5x vs torch), CPU i32 67.60ms, OpenCL 3.74ms (1.6x vs torch, 3.6x vs CPU), OpenCL i32 2.66ms
aqpu_bitplane batch=4096: torch 21.95ms, CPU packed 53.57ms (0.4x vs torch), CPU i32 269.12ms, OpenCL 14.29ms (1.5x vs torch, 3.7x vs CPU), OpenCL i32 9.57ms
```

