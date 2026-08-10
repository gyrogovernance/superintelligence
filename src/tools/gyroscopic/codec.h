#pragma once
/*
 * Continuous op codecs: Norm / RoPE / SiLU finite charts + residual measurement.
 * Also owns aperture Norm / SiLU helpers used by the attention path.
 */

#include <stddef.h>
#include <stdint.h>

#include "constants.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(_WIN64)
#  ifndef HQVM_EXPORT
#    define HQVM_EXPORT __declspec(dllexport)
#  endif
#else
#  ifndef HQVM_EXPORT
#    define HQVM_EXPORT __attribute__((visibility("default")))
#  endif
#endif

/* ---- Norm codec ---- */
HQVM_EXPORT int hqvm_norm_codec_enabled(void);
HQVM_EXPORT int hqvm_norm_commit_enabled(void);
HQVM_EXPORT float hqvm_rms_gain(const float *x, int64_t n, float eps);
/* Fixed-point sum(x^2)/n + eps then reciprocal sqrt (LUT + 1 Newton). */
HQVM_EXPORT float hqvm_rms_gain_fixed(const float *x, int64_t n, float eps);
/* Signed Delta-ruler: n = round(log2(g/g0)/Delta), sat int16. Never clamp negatives to 0. */
HQVM_EXPORT void hqvm_norm_set_g0(float g0);
HQVM_EXPORT float hqvm_norm_g0(void);
/* Export/runtime: g0 = geomean(|g|) of Norm weight vector (once). */
HQVM_EXPORT float hqvm_norm_geomean_gains(const float *g, int64_t n);
HQVM_EXPORT void hqvm_norm_set_g0_from_gains(const float *g, int64_t n);
HQVM_EXPORT int16_t hqvm_norm_encode_gain16(float g, float g0, float Delta);
HQVM_EXPORT float hqvm_norm_decode_gain16(int16_t n, float g0, float Delta);
HQVM_EXPORT float hqvm_norm_pow2_delta(int16_t n); /* 2^(n*Delta) via LUT */
/* Channel-wise Delta-ruler on Norm weights: w[i] <- g0 * 2^(n16[i]*Delta). */
HQVM_EXPORT void hqvm_norm_apply_gain_ruler(float *w, int64_t n, float g0, float Delta);
/* Legacy unsigned wrappers (encode16 + clamp to [0, LAYER_MASK_12] for storage). */
HQVM_EXPORT uint16_t hqvm_norm_encode_gain12(float g, float g0, float Delta);
HQVM_EXPORT float hqvm_norm_decode_gain12(uint16_t q, float g0, float Delta);
HQVM_EXPORT uint8_t hqvm_norm_encode_gain(float g, float g0, float Delta);
HQVM_EXPORT float hqvm_norm_decode_gain(uint8_t q, float g0, float Delta);
HQVM_EXPORT void hqvm_norm_codec_shadow(
    const float *x, int64_t n, float eps, float Delta);

/* ---- RoPE codec (T_256 turn ticks) ---- */
#define HQVM_ROPE_TICKS 256
#define HQVM_ROPE_MAX_FREQ 128

HQVM_EXPORT void hqvm_rope_codec_init(void);
HQVM_EXPORT int hqvm_rope_codec_enabled(void);
HQVM_EXPORT uint8_t hqvm_rope_encode_tick(float theta);
/* Turn-tick identity: theta_tick = (pos * dtheta_k) & 255; no atan2. */
HQVM_EXPORT void hqvm_rope_init_dtheta(
    int64_t n_dims, float theta_scale, float freq_scale, const float *freq_factors);
HQVM_EXPORT void hqvm_rope_ticks_from_pos(
    int64_t pos, int64_t n_dims, uint8_t *ticks_out);
HQVM_EXPORT void hqvm_rope_apply_pair(
    float x0, float x1, uint8_t tick, float sin_sign, float *y0, float *y1);
HQVM_EXPORT void hqvm_rope_apply_row(
    const float *src, float *dst, const uint8_t *ticks, int64_t n_dims,
    int64_t n_offset, int is_neox, float sin_sign);
HQVM_EXPORT void hqvm_rope_codec_shadow(
    const float *src, const float *dst_stock, const float *cache,
    int64_t n_dims, int64_t n_offset, int is_neox, float sin_sign);
HQVM_EXPORT void hqvm_rope_codec_audit_reset(void);
HQVM_EXPORT void hqvm_rope_codec_audit_report(void);
HQVM_EXPORT void hqvm_rope_stock_inc(void);
HQVM_EXPORT void hqvm_rope_codec_counters_get(uint64_t *codec_calls, uint64_t *stock_calls);

/* ---- FFN shell gate (no SiLU) ---- */
HQVM_EXPORT int hqvm_ffn_shell_gate_enabled(void);
HQVM_EXPORT void hqvm_ffn_shell_gate_init(void);
HQVM_EXPORT void hqvm_ffn_shell_gate_apply(
    float *dst, const float *gate, const float *up, int64_t n,
    uint8_t fam, uint8_t Nc);
HQVM_EXPORT void hqvm_stock_silu_inc(void);
HQVM_EXPORT uint64_t hqvm_stock_silu_calls(void);

/* ---- SiLU codec ---- */
#define HQVM_SILU_BINS 256

HQVM_EXPORT void hqvm_silu_codec_init(void);
HQVM_EXPORT void hqvm_silu_codec_init_range(float clip);
HQVM_EXPORT int hqvm_silu_codec_enabled(void);
HQVM_EXPORT void hqvm_silu_apply(float *x, int64_t n, float clip);
HQVM_EXPORT void hqvm_swiglu_apply(
    float *dst, const float *gate, const float *up, int64_t n, float clip);
HQVM_EXPORT void hqvm_silu_codec_shadow(const float *x, int64_t n, float clip);
HQVM_EXPORT void hqvm_swiglu_codec_shadow(
    const float *gate, const float *up, int64_t n, float clip);

/* ---- Aperture Norm / SiLU variants ---- */
HQVM_EXPORT int hqvm_aperture_silu_enabled(void);
HQVM_EXPORT void hqvm_aperture_rms_scale(float *row, int64_t n, float Delta);
HQVM_EXPORT void hqvm_aperture_silu(float *row, int64_t n, float Delta);

#ifdef __cplusplus
}
#endif
