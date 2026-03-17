# GyroLabe C compilation notes

This folder builds native C shared libraries used by
`src/tools/gyrolabe/ops.py` and exposes an optional OpenCL backend from
`src/tools/gyrolabe/opencl_backend.py`.

## Files built from C

- `gyrolabe.c` -> `_build/gyrolabe_<hash>.<dll|dylib|so>`
- `gyrolabe_opencl.c` -> `_build/gyrolabe_opencl_<hash>.<dll|dylib|so>` (OpenCL only)

If a matching compiled library already exists in `_build/`, it is reused rather
than rebuilt. This allows prebuilt binaries (for example, Windows `.dll` files)
to be checked into the repository and used directly on systems without a local
C compiler.

The `<hash>` part is derived from:

- source hash
- host platform + Python version
- selected compiler
- compiler mode (`native` / `portable`)
- OpenMP/SIMD flags
- OpenCL hash flag when used

## Compiler toolchain detected

`ops.py` and `opencl_backend.py` share these search rules:

- Linux / macOS: prefer `cc`, then `clang`, then `gcc`
- Windows: prefer `gcc` from `PATH` or `C:/msys64/{ucrt64,mingw64}/bin`,
  then `cl` / `clang-cl`, then `clang`

Fallback errors:

- `No suitable C compiler was found for GyroLabe`
- `No C compiler found for OpenCL backend`

## Compile flags and optional features

When compiling `gyrolabe.c` (`_build_shared_library`):

- `-O3`
- `-std=c11`
- `-shared` (or `-dynamiclib` on macOS)
- `-fPIC` for non-Windows non-macOS
- OpenMP:
  - `-fopenmp` for GCC/clang
  - `/openmp` for MSVC
- Native mode (`GYRO_BUILD_MODE=native` on x86_64):
  - `-mavx2 -mfma` (GCC/clang)
  - `/arch:AVX2` (MSVC)

When compiling `gyrolabe_opencl.c` (`opencl_backend.py`):

- `-O3`
- `-std=c11`
- `-shared` (or `-dynamiclib` on macOS)
- `-fPIC` on non-Windows
- OpenCL include and lib flags from environment (below)

## Environment variables used

- `GYRO_BUILD_MODE`
  - `native` enables AVX2+FMA on supported x86_64
  - any other value uses portable flags
- `GYROLABE_NO_OPENMP`
  - set to `1` to disable OpenMP
- `GYROLABE_NO_NATIVE`
  - set to `1` to disable native CPU flags
- `OPENCL_INCLUDE`
  - adds `-I` include path for `gyrolabe_opencl.c`
- `OPENCL_LIB`
  - adds `-L` library path and `-lOpenCL`
- `OPENCL_SDK_PATH`, `OPENCL_ROOT`
  - fallback SDK roots for include + library
- `LLVM_HOME`, `LLVM_PATH`, `LLVM_ROOT`
  - helps find LLVM/clang on Windows when not on PATH
- `MINGW64_BIN`, `MSYS2_PATH`
  - extra DLL search candidates on Windows

## Runtime DLL search paths used on Windows

- compiled compiler binary directory
- `C:/msys64/{ucrt64,mingw64}/bin`
- `LLVM_HOME` / `LLVM_PATH` / `LLVM_ROOT` (and `/bin` variants)
- `MINGW64_BIN` / `MSYS2_PATH` (if set)

On Windows builds created with MSYS2/MinGW, the compiled `.dll` may depend on
MSYS2 runtime DLLs (for example `libgcc_s_seh-1.dll`, `libwinpthread-1.dll`).
Ensure these are discoverable via the documented search paths if you distribute
prebuilt binaries.

## When the C backend is used

`gyrolabe/ops.py` compiles/loads the C library lazily when needed.
If compilation or loading fails, the module falls back to pure Python
implementations and emits a warning.

## Helpful terminal checks

From repo root:

```bash
python -c "from src.tools.gyrolabe import build_native; print(build_native())"
python -c "from src.tools.gyrolabe import native_available; print(native_available())"
python -c "from src.tools.gyrolabe import opencl_backend; print(opencl_backend.available())"
```

If `native_available()` returns `False`, the Python fallback implementations
remain correct but slower. On macOS and Linux, installing standard build tools
(Xcode Command Line Tools on macOS, or `build-essential` on Debian/Ubuntu) is
usually sufficient to enable the native backend.

Use a shell path that points to the same compiler/runtime you want to validate,
for example your MSVC Developer Prompt or MSYS2 shell.
