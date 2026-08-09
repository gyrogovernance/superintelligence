#pragma once
/*
 * hQVM temporal ledger.
 *
 * HQVMLEDG v2 (fat / bridge): packed signs + scales + byte table. Early live
 * displace bridge only. Do not scale this to the full model.
 *
 * HQVMLEDS v1 (sidecar / production): shared byte table + bin edges + allowlist
 * only. Weight signs/scales are read from the live ggml Q1_0 tensor in RAM.
 * This is the thin ledger: temporal inference extras, not a second model file.
 */

#include <stddef.h>
#include <stdint.h>

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

/* ---- Fat v2 (verify / bridge) ---- */

typedef struct hqvm_ledger {
    uint32_t version;
    uint32_t n_rows;
    uint32_t n_blocks;
    uint32_t block_w;
    uint32_t n_bin;
    uint32_t n_scale_blocks;
    uint8_t *signs;
    float   *scales;
    int16_t *byte_table;
    float   *bin_edges;
    float   *coefs;
    uint32_t n_coef;
} hqvm_ledger;

HQVM_EXPORT int hqvm_ledger_load(hqvm_ledger *L, const char *path);
HQVM_EXPORT void hqvm_ledger_free(hqvm_ledger *L);

HQVM_EXPORT void hqvm_ledger_forward_bits(
    const hqvm_ledger *L,
    const uint8_t *x01,
    float *Y,
    int16_t *shells
);

HQVM_EXPORT void hqvm_ledger_forward_f32(
    const hqvm_ledger *L,
    const float *x,
    float *Y
);

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

/* Per-32 q8 activation quant (stock-like). xd has n/32 scales. n % 32 == 0. */
HQVM_EXPORT void hqvm_quantize_x_q8(const float *x, int64_t n, int8_t *qx, float *xd);

/*
 * Live displace from ggml Q1_0 weight memory (row-major block_q1_0).
 * ncols multiple of 128. Processes rows [row0, row1).
 * Pre-quantized activation path (qx, xd from hqvm_quantize_x_q8).
 *
 * Frozen per-site law (NavPAD §14.6 CLOSED):
 *   Y = exact_Q1_0_q8_dot_product * manifold_gain
 * where manifold_gain = 1 + APERTURE_GAP * mean(sign(parity(mismatch) XOR chi'_bit0))
 * and chi' comes from byte_table -> step_uv6. Do not replace the dot with
 * bin_centers / Krawtchouk (Gate B FAIL 0.836).
 */
HQVM_EXPORT int hqvm_forward_q1_0_q8(
    const hqvm_sidecar *S,
    const void *q1_data,
    int64_t nrows,
    int64_t ncols,
    size_t row_stride_bytes,
    const int8_t *qx,
    const float *xd,
    float *Y,
    int64_t row0,
    int64_t row1
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
HQVM_EXPORT void hqvm_pack12_bits(const uint8_t *x01, uint8_t *u, uint8_t *v);

#ifdef __cplusplus
}
#endif
