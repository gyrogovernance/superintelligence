#pragma once
/*
 * hQVM temporal ledger.
 *
 * HQVMLEDS v1 (sidecar / production): shared byte table + bin edges + allowlist
 * only. Weight signs/scales are read from the live ggml Q1_0 tensor in RAM.
 * This is the thin ledger: temporal inference extras, not a second model file.
 * Fat HQVMLEDG v2 deleted (chassis second-route cleanup).
 */

#include <stddef.h>
#include <stdint.h>

#include "codec.h"
#include "kernel.h"

#if defined(_WIN32) || defined(_WIN64)
#  ifndef HQVM_EXPORT
#    define HQVM_EXPORT __declspec(dllexport)
#  endif
#else
#  ifndef HQVM_EXPORT
#    define HQVM_EXPORT __attribute__((visibility("default")))
#  endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Q1_0 weight view (shared with layer block weights). */
typedef struct hqvm_q1_weight {
    const void *q1_data;
    int64_t     n_rows;
    int64_t     n_cols;
    size_t      row_stride_bytes;
} hqvm_q1_weight_t;

/* ---- Thin sidecar v1 (production) ---- */

#define HQVM_SIDECAR_MAX_ALLOW 64
#define HQVM_SIDECAR_ALLOW_LEN 96

typedef struct hqvm_sidecar {
    uint32_t version;
    uint32_t n_bin;
    uint32_t n_allow;
    int16_t *byte_table; /* [64][n_bin], -1 = identity */
    float   *bin_edges;  /* [n_bin+1] */
    char     allow[HQVM_SIDECAR_MAX_ALLOW][HQVM_SIDECAR_ALLOW_LEN];
} hqvm_sidecar;

HQVM_EXPORT int hqvm_sidecar_load(hqvm_sidecar *S, const char *path);
HQVM_EXPORT void hqvm_sidecar_free(hqvm_sidecar *S);

/* True if tensor name contains any allow substring. */
HQVM_EXPORT int hqvm_sidecar_allows(const hqvm_sidecar *S, const char *tensor_name);

/* Apply env GYRO_LEDGER_ALLOW=pat1,pat2 (replaces file allowlist if set). */
HQVM_EXPORT void hqvm_sidecar_apply_env_allow(hqvm_sidecar *S);

/* Per-32 q8 activation quant. xd_h stores fp16 bit-patterns (n/32 scales). n % 32 == 0. */
HQVM_EXPORT void hqvm_quantize_x_q8(const float *x, int64_t n, int8_t *qx, uint16_t *xd_h);

/*
 * Dyad→q8 quantizer: IEEE754 bits in dyad → block q8 (no hqvm_dyad32_to_f32 chart).
 * Bit-identical to the prior dyad→f32→q8 path for finite values.
 */
HQVM_EXPORT void hqvm_quantize_dyad_q8(const hqvm_dyad32_t *x, int64_t n, int8_t *qx, uint16_t *xd_h);

/* Quantize one dyad row into ggml Q8_0 blocks (n multiple of 32). */
HQVM_EXPORT void hqvm_quantize_dyad_row_q8(const hqvm_dyad32_t *row, int64_t n, void *out_q8);

/* Dequantize 64 columns starting at col0 (must be 64-aligned) on one Q1_0 row. */
HQVM_EXPORT int hqvm_q1_dequant_row_cols(
    const void * q1_row,
    size_t       row_stride_bytes,
    int64_t      col0,
    float        out64[64]);

/* Dequantize a 64x64 tile at (row0, col0) from Q1_0 row-major storage. */
HQVM_EXPORT int hqvm_q1_dequant_tile64(
    const void * q1_data,
    size_t       row_stride_bytes,
    int64_t      n_rows,
    int64_t      n_cols,
    int64_t      row0,
    int64_t      col0,
    float        out64x64[4096]);

/*
 * Live displace from ggml Q1_0 weight memory (row-major block_q1_0).
 * ncols multiple of 128. Processes rows [row0, row1).
 * Pre-quantized activation path (qx, xd_h from hqvm_quantize_x_q8).
 *
 * Frozen per-site law (NavPAD §14.6 CLOSED):
 *   Y = exact_Q1_0_q8_dot_product * manifold_gain
 * where manifold_gain = 1 + APERTURE_GAP * mean(sign(parity(mismatch) XOR chi'_bit0))
 * via Q16 aperture chart, and chi' comes from byte_table -> step_uv6. Do not replace the
 * dot with bin_centers / Krawtchouk (Gate B FAIL 0.836).
 */
HQVM_EXPORT int hqvm_forward_q1_0_q8(
    const hqvm_sidecar *S,
    const void *q1_data,
    int64_t nrows,
    int64_t ncols,
    size_t row_stride_bytes,
    const int8_t *qx,
    const uint16_t *xd_h,
    float *Y,
    int64_t row0,
    int64_t row1
);

/*
 * Same P_Q + manifold law as hqvm_forward_q1_0_q8, but stores dyad rows.
 * Retires hosting float-Y pack (H2 float-Y). Controller accumulate may still use
 * float inside the kernel — that is not a returned float-Y buffer.
 * D_Q remains gate-only (never hot-path full-tile dequant).
 */
HQVM_EXPORT int hqvm_forward_q1_0_q8_dyad(
    const hqvm_sidecar *S,
    const void *q1_data,
    int64_t nrows,
    int64_t ncols,
    size_t row_stride_bytes,
    const int8_t *qx,
    const uint16_t *xd_h,
    hqvm_dyad32_t *Y,
    int64_t row0,
    int64_t row1
);

/* Portable signed two-limb two's-complement integer. */
typedef struct hqvm_i128 {
    uint64_t lo;
    uint64_t hi;
} hqvm_i128;

/* Exact signed comparison: -1, 0, or 1. */
HQVM_EXPORT int hqvm_i128_cmp(hqvm_i128 a, hqvm_i128 b);

/* Add coefficient * 2^shift exactly; returns -1 on invalid shift or signed overflow. */
HQVM_EXPORT int hqvm_i128_add_shifted_i64(hqvm_i128 *acc, int64_t coefficient, uint32_t shift);

/* Decode finite fp16 bits as M * 2^e. Negative and non-finite values are rejected. */
HQVM_EXPORT int hqvm_fp16_decode_nonnegative(uint16_t bits, uint16_t *mantissa, int32_t *exponent);

/*
 * Exact Q1_0-by-Q8 row argmax over [row0,row1). Scores are result * 2^64.
 * Rows are scanned ascending and ties retain the least row.
 */
HQVM_EXPORT int hqvm_argmax_q1_0_q8_exact(
    const hqvm_sidecar *S,
    const void *q1_data,
    int64_t nrows,
    int64_t ncols,
    size_t row_stride_bytes,
    const int8_t *qx,
    const uint16_t *xd_h,
    int64_t row0,
    int64_t row1,
    int32_t *best_row,
    hqvm_i128 *best_score
);

/* Quantize x then call hqvm_forward_q1_0_q8 (convenience / non-hook callers). */
HQVM_EXPORT int hqvm_forward_q1_0_f32(
    const hqvm_sidecar *S,
    const void *q1_data,
    int64_t nrows,
    int64_t ncols,
    size_t row_stride_bytes,
    const float *x,
    float *Y,
    int64_t row0,
    int64_t row1
);

HQVM_EXPORT void hqvm_step_uv6(uint8_t u, uint8_t v, uint8_t byte, uint8_t *u_out, uint8_t *v_out);
HQVM_EXPORT uint8_t hqvm_shell_uv6(uint8_t u, uint8_t v);

/* Dequantize first 64 cols of one Q1_0 row (tile probes / gates). */
HQVM_EXPORT int hqvm_q1_dequant_row64(
    const void * q1_row, size_t row_stride_bytes, float out64[64]);

/* ---- Product MatMul + gate session (ex-hosting) ---- */
typedef struct hqvm_gate_counters {
    uint64_t matmul_calls;
    uint64_t matmul_pq_calls;
    uint64_t matmul_dq_calls;
    uint64_t norm_calls;
    uint64_t rope_calls;
    uint64_t attn_score_calls;
    uint64_t v_reduce_calls;
    uint64_t swiglu_calls;
    uint64_t not_implemented;
} hqvm_gate_counters_t;

HQVM_EXPORT int  hqvm_sidecar_ready(void);
HQVM_EXPORT void hqvm_sidecar_reset_session(void);
HQVM_EXPORT void hqvm_gate_counters_reset(void);
HQVM_EXPORT void hqvm_gate_counters_snapshot(hqvm_gate_counters_t * out);
HQVM_EXPORT void hqvm_gate_counters_print(const char * tag);
HQVM_EXPORT void hqvm_gate_counters_inc_norm(void);
HQVM_EXPORT void hqvm_gate_counters_inc_rope(void);
HQVM_EXPORT void hqvm_gate_counters_inc_attn(void);
HQVM_EXPORT void hqvm_gate_counters_inc_vreduce(void);
HQVM_EXPORT void hqvm_gate_counters_inc_swiglu(void);
HQVM_EXPORT void hqvm_gate_counters_inc_stub(void);

HQVM_EXPORT int hqvm_matmul_dyad(
    const hqvm_q1_weight_t * W,
    const hqvm_dyad32_t *    x,
    hqvm_dyad32_t *          y);
HQVM_EXPORT int hqvm_matmul_dq_selftest(
    const hqvm_q1_weight_t * W,
    const int8_t *           qx,
    const uint16_t *         xd,
    int64_t *                out_dq_rows);

#ifdef __cplusplus
}
#endif
