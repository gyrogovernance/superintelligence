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

/* ---- FFN joint law (H7) ----
 * Documented production L2 (Theory_Drop Runtime §4.1.2):
 *   silu(gate)·up·(1+Δ·m)·(1+0.25Δ·m_req)
 * Shell-only `up*simplex(λ^N)` is forbidden (discards controller magnitude).
 * Product path remains exact stock SwiGLU until this opt-in face is enabled
 * (`GYRO_FFN_NATIVE=1`, alias `GYRO_FFN_SHELL_GATE`) and Paris holds.
 * Analysis §7.7 eventual shell-native nonlinearity is not a softstep hack.
 */
HQVM_EXPORT int hqvm_ffn_shell_gate_enabled(void);
/* Alias: GYRO_FFN_NATIVE (same as hqvm_ffn_shell_gate_enabled). */
HQVM_EXPORT int hqvm_ffn_native_enabled(void);
HQVM_EXPORT void hqvm_ffn_shell_gate_init(void);
HQVM_EXPORT void hqvm_ffn_shell_gate_apply_native(
    float *dst, const float *gate, const float *up, int64_t n,
    uint8_t fam, uint8_t Nc);
HQVM_EXPORT void hqvm_stock_silu_inc(void);
HQVM_EXPORT uint64_t hqvm_stock_silu_calls(void);

/* ---- Integer-owned finite binary32 chart (native magnitude lane) ----
 * hqvm_dyad32_t is the controller magnitude object. to_f32/from_f32 are chart
 * adapters for stock interoperability — never proof that a site is native.
 */
typedef struct hqvm_dyad32 {
    uint32_t bits;
} hqvm_dyad32_t;

/* Product faces (ex-hosting): Delta-ruler norm, YaRN RoPE ticks, FFN SwiGLU|L2. */
#ifndef HQVM_HEAD_DIM
#define HQVM_HEAD_DIM 128
#endif
HQVM_EXPORT int hqvm_norm_ruler_dyad(
    const hqvm_dyad32_t * x_in,
    hqvm_dyad32_t *       x_out,
    int64_t               n,
    const float *         g,
    float                 g0);
HQVM_EXPORT int hqvm_rope_qk_dyad(
    hqvm_dyad32_t * Q,
    hqvm_dyad32_t * K,
    int32_t         n_heads,
    int32_t         gqa_ratio,
    int32_t         token_pos);
HQVM_EXPORT int hqvm_ffn_gate_dyad(
    const hqvm_dyad32_t * gate,
    const hqvm_dyad32_t * up,
    hqvm_dyad32_t *       dst,
    int64_t               n,
    uint8_t               fam,
    uint8_t               Nc);

HQVM_EXPORT int hqvm_dyad32_is_finite(hqvm_dyad32_t x);
HQVM_EXPORT int hqvm_dyad32_sign(hqvm_dyad32_t x);
HQVM_EXPORT hqvm_dyad32_t hqvm_dyad32_abs(hqvm_dyad32_t x);
HQVM_EXPORT int hqvm_dyad32_is_zero(hqvm_dyad32_t x);
HQVM_EXPORT int hqvm_dyad32_from_i32(int32_t x, hqvm_dyad32_t *out);
HQVM_EXPORT int hqvm_dyad32_add(hqvm_dyad32_t a, hqvm_dyad32_t b, hqvm_dyad32_t *out);
HQVM_EXPORT int hqvm_dyad32_mul(hqvm_dyad32_t a, hqvm_dyad32_t b, hqvm_dyad32_t *out);
HQVM_EXPORT int hqvm_dyad32_div(hqvm_dyad32_t a, hqvm_dyad32_t b, hqvm_dyad32_t *out);
HQVM_EXPORT int hqvm_dyad32_mul_rational(
    hqvm_dyad32_t x, int32_t num, int32_t den, hqvm_dyad32_t *out);
/* Chassis adapters copy the binary32 object representation; they do no arithmetic. */
HQVM_EXPORT hqvm_dyad32_t hqvm_dyad32_from_f32(float x);
HQVM_EXPORT float hqvm_dyad32_to_f32(hqvm_dyad32_t x);
/* Exact integer magnitude * 2^exp2 packed as a dyad object (round-to-nearest-even). */
HQVM_EXPORT int hqvm_dyad32_pack_i128(uint32_t sign, uint64_t sig, int exp2, hqvm_dyad32_t *out);

/* Committed-chart access. */
HQVM_EXPORT float hqvm_norm_rsqrt_mantissa(double mant);
HQVM_EXPORT float hqvm_norm_commit_gain(float inv_gain);
HQVM_EXPORT float hqvm_norm_weight_commuted(float w, float ref, float Delta);

/* ---- Dyadic aperture / scale charts (F32 algebra closure) ---- */
/* Residual mixer: gain = 1 + APERTURE_GAP * (Nc-3)/3 as num/den (den = 3<<16). */
HQVM_EXPORT void hqvm_residual_gain_q16(uint8_t Nc, int32_t *num, int32_t *den);
HQVM_EXPORT float hqvm_residual_gain_from_Nc(uint8_t Nc);
/* Manifold MatMul gain: 1 + APERTURE_GAP * 0.5 * (s0+s1), s in {±1}; den = 1<<16. */
HQVM_EXPORT void hqvm_manifold_gain_q16(
    uint8_t chi_bit0, uint8_t p0, uint8_t chi_bit1, uint8_t p1,
    int32_t *num, int32_t *den);
HQVM_EXPORT float hqvm_manifold_gain_from_bits(
    uint8_t chi_bit0, uint8_t p0, uint8_t chi_bit1, uint8_t p1);

#ifdef __cplusplus
}
#endif
