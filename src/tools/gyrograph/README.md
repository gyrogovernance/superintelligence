# GyroGraph C compilation notes

This folder builds native C shared libraries (CPU and optional OpenCL) used by
`src/tools/gyrograph/ops.py`.

## Files built from C

- `gyrograph.c` -> `_build/gyrograph_<hash>.<dll|dylib|so>`
- `gyrograph_opencl.c` -> `_build/gyrograph_opencl_<hash>.<dll|dylib|so>`

If a matching compiled library already exists in `_build/`, it is reused rather
than rebuilt. This allows prebuilt binaries (for example, Windows `.dll` files)
to be checked into the repository and used directly on systems without a local
C compiler.

The `<hash>` part is derived from:

- source hash
- host platform + Python version
- selected compiler
- compiler mode (`native` / `portable`)
- active compile flags
- OpenCL enabled flag

## Compiler toolchain detected

`ops.py` chooses compilers with this order:

- Linux / macOS: `cc` then `clang` then `gcc`
- Windows: `gcc` or MSYS2 `gcc` first,
  then `cl` / `clang-cl`, then `clang`

Fallback errors:

- `No suitable C compiler was found for GyroGraph`

## Compile flags

Shared library build (`_build_shared_library`) uses:

- `-O3`
- `-std=c11`
- `-shared` (or `-dynamiclib` on macOS)
- OpenCL include flags when needed
- OpenMP:
  - `-fopenmp` for GCC/clang
  - `/openmp` for MSVC
- `-fPIC` for non-Windows non-macOS
- optional SIMD:
  - `-mavx2 -mfma` on x86_64 with `GYRO_BUILD_MODE=native`
  - `/arch:AVX2` on MSVC with native mode

For `gyrograph_opencl.c` the OpenCL flags are added from:

- include path finder (`OPENCL_INCLUDE`, `OPENCL_SDK_PATH`, `OPENCL_ROOT`)
- lib flags (`OPENCL_LIB` or `OPENCL_SDK_PATH` / `OPENCL_ROOT` + `lib/.../` + `-lOpenCL`)

## Environment variables used

- `GYRO_BUILD_MODE`
  - `native` enables SIMD flags on supported x86_64
  - other values use portable mode
- `GYROGRAPH_NO_OPENMP`
  - set to `1` to disable OpenMP
- `OPENCL_INCLUDE`
  - optional explicit OpenCL include directory
- `OPENCL_LIB`
  - optional library directory passed with `-L` plus `-lOpenCL`
- `OPENCL_SDK_PATH`, `OPENCL_ROOT`
  - fallback roots for include + library
- `LLVM_PATH`, `LLVM_ROOT`, `MINGW64_BIN`, `MSYS2_PATH`
  - used for compiler/DLL discovery on Windows

## Windows runtime DLL search paths

- compiler directory from the selected compiler
- `C:/msys64/ucrt64/bin`, `C:/msys64/mingw64/bin`
- `LLVM_PATH` / `LLVM_ROOT` and `/bin` forms
- `MINGW64_BIN` / `MSYS2_PATH`

## Runtime behavior

`ops.py` compiles and loads CPU or OpenCL libraries lazily.
If a native library cannot be loaded, it falls back to Python implementations
and logs a warning.

## Helpful terminal checks

From repo root:

```bash
python - <<'PY'
from src.tools import gyrograph
print("cpu available:", gyrograph.native_available())
print("opencl available:", gyrograph.opencl_available())
PY
```

If `native_available()` returns `False`, the Python fallback implementations
remain correct but slower. On macOS and Linux, installing standard build tools
(Xcode Command Line Tools on macOS, or `build-essential` on Debian/Ubuntu) is
usually sufficient to enable the native backend.

Use a shell with your desired compiler visible in `PATH` (MSVC developer prompt,
MSYS2, or LLVM/clang setup) so compiled dependencies are found correctly.
