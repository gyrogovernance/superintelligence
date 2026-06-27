#pragma once

/*
 * Gyroscopic public API for the ggml-gyroscopic backend and Python ctypes bindings.
 *
 * Inference hooks (ggml-gyroscopic/):
 *   - Q1_0 matmul: per-layer gravity scale (quants.c)
 *   - Attention: M₂ percolation prefilter + 14-byte SLCP KV sidecar (gyro_kv_chi.c)
 *   - Optional G(psi) attention coupling (GYROSCOPIC_GRAVITY_ATTN=1)
 */

#include "kernel.h"
