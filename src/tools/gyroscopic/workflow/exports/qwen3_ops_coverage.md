# Qwen3.5-4B-Q8_0 export-graph-ops coverage

Model: `data/models/unsloth-Qwen3.5-4B-GGUF/Qwen3.5-4B-Q8_0.gguf`

Binary: `external/llama.cpp/build/bin/Release/export-graph-ops.exe` (built with `GGML_CPU_BACKEND_SUBDIR=ggml-gyroscopic`, `GGML_GYROSCOPIC=ON` per `scripts/build_llama_cpp_windows.ps1`).

Raw export: `src/tools/gyroscopic/workflow/exports/qwen3_ops.txt` (one line per unique structural pattern; PP and TG merged into one set).

Tool stderr summary: pp 49 unique patterns / 1833 nodes; tg 40 unique / 1833 nodes; 89 unique patterns total in set.

Decoded `ggml_op`: first integer on each line of `qwen3_ops.txt`, mapped to names in `external/llama.cpp/ggml/include/ggml.h` (enum order at this repo snapshot). Row counts are how many distinct patterns use that op (same op, different shapes, multiple lines). `GGML_OP_UNARY` / `GGML_OP_GLU` subtypes sit in `op_params`; not expanded here.

- **2** -- GGML_OP_ADD (rows in export file: 4)
- **7** -- GGML_OP_MUL (rows in export file: 14)
- **22** -- GGML_OP_CONCAT (rows in export file: 2)
- **25** -- GGML_OP_RMS_NORM (rows in export file: 8)
- **28** -- GGML_OP_L2_NORM (rows in export file: 2)
- **29** -- GGML_OP_MUL_MAT (rows in export file: 16)
- **32** -- GGML_OP_SCALE (rows in export file: 2)
- **34** -- GGML_OP_CPY (rows in export file: 7)
- **35** -- GGML_OP_CONT (rows in export file: 2)
- **40** -- GGML_OP_GET_ROWS (rows in export file: 8)
- **42** -- GGML_OP_SET_ROWS (rows in export file: 2)
- **48** -- GGML_OP_ROPE (rows in export file: 4)
- **73** -- GGML_OP_FLASH_ATTN_EXT (rows in export file: 2)
- **75** -- GGML_OP_SSM_CONV (rows in export file: 2)
- **85** -- GGML_OP_GATED_DELTA_NET (rows in export file: 2)
- **86** -- GGML_OP_UNARY (rows in export file: 10)
- **95** -- GGML_OP_GLU (rows in export file: 2)

In-scope matmul-related ops from `agent.md` appearing in this export:

- `GGML_OP_MUL_MAT` (29) -- present.
- `GGML_OP_MUL_MAT_ID` -- not among the distinct op integers in this file.
- `GGML_OP_OUT_PROD` -- not among the distinct op integers in this file.
