"""GyroLabe -- Native codec and execution substrate for the Gyroscopic ASI stack.

Naming Conventions (canonical - all new code must follow these)
==============================================================

This codebase uses two operational domains with distinct naming grammars.
Every public symbol must belong to exactly one domain.

Domain 1: GyroLabe (codec / extraction / structural operations)
---------------------------------------------------------------
- Python functions: gyrolabe_<op>  (e.g. gyrolabe_signature_scan)
- Python classes: GyroLabe<Name> (e.g. GyroLabeBolmoEncodeBridge)
- C functions: gyrolabe_<op> (e.g. gyrolabe_chirality_distance)
- Error messages: "GyroLabe: ..."

Domain 2: GyroMatMul (multiplication / matmul / model integration)
----------------------------------------------------------------
- Python functions: gyromatmul_<op>  (e.g. gyromatmul_i32, gyromatmul_i32_dense)
- Python classes: GyroMatMul<Name> (e.g. GyroMatMulAttention, GyroMatMulMLP)
- C functions: gyromatmul_<op> (e.g. gyromatmul_i32, gyromatmul_gemm_i16x64)
- Error messages: "GyroMatMul: ..."

Prohibited patterns
-------------------
- No bare "gyro_" prefix on any public Python name. Use gyrolabe_ or gyromatmul_.
- No "qubec", "holographic", "universal", or "exact_" as intent-repeating prefixes.
- No "metal" in names or comments when the backend is OpenCL.
- No public aliases that duplicate an existing canonical name.
- No "_impl" suffix on public functions (use it only for internal helpers
  that are genuinely shared, such as autograd Function forward/backward helpers).

Structural rules
----------------
- Every function does what its name says. No silent fallback to a different
  operation. If a required backend is unavailable, raise; do not substitute.
- The K4 chart matmul (gyromatmul_i32) never silently degrades to the
  dense path (gyromatmul_i32_dense). These are two distinct operations.
- Every GYRO_EXPORT C function must have a Python consumer.
  Every _FUNC_SPECS entry must correspond to a real call path.
- Internal helpers are underscore-prefixed and excluded from __all__.

Shared C header (gyrolabe_core.h)
---------------------------------
- Shared inline helpers use the "gyro_" prefix (e.g. gyro_dot_i32_n, gyro_popcnt64).
  This prefix is reserved for the C layer only.
- No "gyro_" function is part of the Python public API.
"""

def gyromatmul_runtime_caps():
    from src.tools.gyroscopic.gyrolabe import ops as _ops

    return _ops.gyromatmul_runtime_caps()


__all__ = ["gyromatmul_runtime_caps"]

