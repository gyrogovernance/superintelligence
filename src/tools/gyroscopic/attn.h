#pragma once
/*
 * Attention path: KV Q8 caches, holonomic QK/Attn@V, CGM-lift (owns the single traj).
 */

#include <stddef.h>
#include <stdint.h>

#include "constants.h"
#include "kernel.h"
#include "codec.h"
#include "layer.h"

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

#define HQVM_KV_N_KV_HEAD   8
#define HQVM_KV_N_LAYER     36
#define HQVM_KV_HEAD_DIM    128
#define HQVM_Q8_BLOCK 32

/* ---- KV / Q8 ---- */
HQVM_EXPORT int hqvm_kv_keys_enabled(void);
HQVM_EXPORT int hqvm_holonomic_attn_enabled(void);
HQVM_EXPORT int hqvm_coord_perturb_flip(void);
HQVM_EXPORT void hqvm_holonomic_counters_inc(int holonomic);
HQVM_EXPORT void hqvm_holonomic_counters_get(uint64_t *holonomic, uint64_t *stock);
HQVM_EXPORT void hqvm_holonomic_counters_print(void);
HQVM_EXPORT void hqvm_v_q8_inc(void);
HQVM_EXPORT uint16_t hqvm_f32_to_f16(float f);
HQVM_EXPORT float hqvm_f16_to_f32(uint16_t h);
HQVM_EXPORT float hqvm_q8_cache_row_score(
    const float *q_head128, const void *k_q8_blocks, float scale);
HQVM_EXPORT void hqvm_attn_v_accum_q8(
    float *VKQ32, int64_t DV, const void *v_q8_row, float a);
HQVM_EXPORT void hqvm_quantize_row_q8(const float *row, int64_t n, void *out_q8);
HQVM_EXPORT uint8_t hqvm_intron_from_q8_row(const void *k_q8_row);
HQVM_EXPORT uint8_t hqvm_k_chi6_from_row(const float *k_head128);
HQVM_EXPORT uint8_t hqvm_k_chi6_from_dyad_head(const hqvm_dyad32_t *k_head128);

/* ---- Attention scores (H5) ----
 * Stock: float Q · Q8 K. Native: dyad Q × Q8 integer accumulate (no float Q row).
 * Product: hqvm_attn_head_scores_dyad (GYRO_NATIVE_ATTN_SCORES). Converter
 * hqvm_q8_cache_row_score_dyad deleted.
 */
HQVM_EXPORT float hqvm_dyad_q8_cache_row_score(
    const hqvm_dyad32_t *q_head_dyad, const void *k_q8_blocks, float scale);
HQVM_EXPORT int hqvm_attn_scores_native_enabled(void);
HQVM_EXPORT int hqvm_attn_head_scores_weights_stock(
    hqvm_dyad32_t *weights_dyad,
    const hqvm_dyad32_t *q_head_dyad,
    const void *k_q8_base, const float *k_f32_base,
    const float *k_fallback_f32,
    size_t k_row_stride, size_t floats_per_tok, size_t k_per_head_q8,
    const uint8_t *chi_layer, int64_t n_kv_heads, int kv_head,
    int64_t kv_len, uint8_t Nc, int top_k, int attn_level, float attn_scale);
HQVM_EXPORT int hqvm_attn_head_scores_weights_native(
    hqvm_dyad32_t *weights_dyad,
    const hqvm_dyad32_t *q_head_dyad,
    const void *k_q8_base, const float *k_f32_base,
    const float *k_fallback_f32,
    size_t k_row_stride, size_t floats_per_tok, size_t k_per_head_q8,
    const uint8_t *chi_layer, int64_t n_kv_heads, int kv_head,
    int64_t kv_len, uint8_t Nc, int top_k, int attn_level, float attn_scale);
HQVM_EXPORT int hqvm_attn_head_scores_dyad(
    hqvm_dyad32_t *       weights_dyad,
    const hqvm_dyad32_t * q_head_dyad,
    const void *          k_q8_base,
    const float *         k_f32_base,
    const float *         k_fallback_f32,
    size_t                k_row_stride,
    size_t                floats_per_tok,
    size_t                k_per_head_q8,
    const uint8_t *       chi_layer,
    int64_t               n_kv_heads,
    int                   kv_head,
    int64_t               kv_len,
    uint8_t               Nc,
    int                   top_k,
    int                   attn_level,
    float                 attn_scale);

/* ---- Softmax / Attn@V / shadows ----
 * hqvm_attn_v_reduce is STOCK float V-reduce (H6 product path until native face lands).
 */
HQVM_EXPORT void hqvm_softmax_inplace(float *scores, int64_t Nk, float M);
HQVM_EXPORT void hqvm_stock_softmax_inc(void);
HQVM_EXPORT uint64_t hqvm_stock_softmax_calls(void);
/* Phase 5: shell-bucket poly-lambda + within-shell QK top-k (no exp). */
#define HQVM_ATTN_SHELL_TOPK 16
#define HQVM_LAMBDA_MAX_ATTN 8.0f
HQVM_EXPORT void hqvm_attn_weight_shell_qk(
    float *scores, const float *q_head128, const void *k_chi_base,
    int kv_head, int64_t Nk, uint8_t Nc, int top_k);
/* Same law; chi from flat native KV table [pos * n_kv_heads + kv_head]. */
HQVM_EXPORT void hqvm_attn_weight_shell_qk_flat(
    float *scores, const float *q_head128, const uint8_t *k_chi6,
    int64_t n_kv_heads, int kv_head, int64_t Nk, uint8_t Nc, int top_k);
HQVM_EXPORT void hqvm_attn_v_reduce(
    float *out, int64_t DV, const float *weights, int64_t Nk,
    const void *v_base, size_t v_row_stride, int v_is_q8, int v_perturb);
/* H6 native: dyad weights × Q8 V → dyad out (Analysis §7.3). Not stock wrap. */
HQVM_EXPORT int hqvm_attn_v_reduce_dyad_q8(
    hqvm_dyad32_t *out_dyad, int64_t DV,
    const hqvm_dyad32_t *weights_dyad, int64_t Nk,
    const void *v_q8_base, size_t v_row_stride);
/* H6 product face: native Q8 when enabled; else stock float unpack/reduce/pack. */
HQVM_EXPORT int hqvm_vreduce_native_enabled(void);
HQVM_EXPORT int hqvm_v_reduce_dyad(
    hqvm_dyad32_t *       out_dyad,
    int64_t               DV,
    const hqvm_dyad32_t * weights_dyad,
    int64_t               Nk,
    const void *          v_base,
    size_t                v_row_stride,
    int                   v_is_q8);
HQVM_EXPORT int hqvm_v_perturb_enabled(void);

/* ---- CGM-lift (native driver carrier / Pi anchor) ---- */
HQVM_EXPORT void hqvm_cgm_lift_init(void);
HQVM_EXPORT uint8_t hqvm_byte_of_q6_fam(uint8_t q6, uint8_t fam);
HQVM_EXPORT uint8_t hqvm_q6_of_byte(uint8_t byte);
HQVM_EXPORT uint8_t hqvm_fam_of_byte(uint8_t byte);
HQVM_EXPORT int hqvm_byte_of_q6_fam_ok(void);
HQVM_EXPORT int hqvm_cgm_lift_enabled(void);
HQVM_EXPORT int hqvm_cgm_lift_layer(void);
HQVM_EXPORT int hqvm_cgm_lift_bump_layer(void);
HQVM_EXPORT int hqvm_cgm_lift_traj_ready(void);
HQVM_EXPORT uint32_t hqvm_cgm_lift_state24(void);
HQVM_EXPORT uint8_t hqvm_cgm_lift_last_byte(void);
/* Request-scoped sequence reset: traj + Genealogy (CS from Pi or GENE_MAC_REST). */
HQVM_EXPORT void hqvm_cgm_lift_reset_sequence(void);
HQVM_EXPORT int hqvm_cgm_lift_seq_active(void);
/*
 * Named entry contract Pi_summary_sign12:
 *   bipolar signs of embd dims [0..5] → u6, dims [6..11] → v6 (bit i set iff coord < 0).
 * GGUF embedding storage may remain chassis F32; the request CS anchor is these 12 bits.
 * Finite apply: hqvm_pi_summary_sign12_from_bits. Diagnostic off-switch: GYRO_PI_FROM_EMBD=0.
 */
HQVM_EXPORT void hqvm_pi_summary_sign12_from_embd(const float *e, int64_t n);
HQVM_EXPORT void hqvm_pi_summary_sign12_from_bits(uint8_t u6, uint8_t v6);
/* Alias retained for call sites; forwards to hqvm_pi_summary_sign12_from_embd. */
HQVM_EXPORT void hqvm_pi_stash_from_embd_row(const float *e, int64_t n);
HQVM_EXPORT int hqvm_pi_applied(void);
/* Carrier (u6,v6) after Pi / reset (GENE_MAC_REST bits if no Pi). */
HQVM_EXPORT void hqvm_cgm_lift_get_uv6(uint8_t *u6, uint8_t *v6);
HQVM_EXPORT uint8_t hqvm_cgm_lift_carrier_shell(void);
HQVM_EXPORT uint8_t hqvm_cgm_lift_fam(void);

/* GyroClock: depth = token_pos * HQVM_N_LAYER + layer_idx (Bonsai L=36). */
HQVM_EXPORT uint64_t hqvm_genealogy_depth(uint32_t token_pos, uint32_t layer_idx);
HQVM_EXPORT uint32_t hqvm_genealogy_token_pos(uint64_t depth);
HQVM_EXPORT uint32_t hqvm_genealogy_layer(uint64_t depth);
HQVM_EXPORT uint64_t hqvm_genealogy_depth_start(void);
HQVM_EXPORT uint64_t hqvm_genealogy_depth_end(void);
HQVM_EXPORT uint64_t hqvm_genealogy_step_count(void);
HQVM_EXPORT uint64_t hqvm_genealogy_span(void);
HQVM_EXPORT int hqvm_genealogy_n_layer(void);
HQVM_EXPORT void hqvm_genealogy_counters_print(void);
/* Prefill: record iq1; decode: return seq cursor (never padded Nk-1). */
HQVM_EXPORT void hqvm_genealogy_observe_prefill(uint32_t token_pos);
HQVM_EXPORT uint32_t hqvm_genealogy_decode_token_pos(void);
HQVM_EXPORT uint32_t hqvm_genealogy_seq_len(void);
/* RoPE audit: last lift token_pos (layer-0 bind check). */
HQVM_EXPORT void hqvm_rope_clock_token_pos_set(uint32_t token_pos);
HQVM_EXPORT uint32_t hqvm_rope_clock_token_pos_get(void);

HQVM_EXPORT void hqvm_cgm_lift_counters_get(
    uint64_t *lift_calls, uint64_t *chi6_writes, uint64_t *invariant_fails);
HQVM_EXPORT void hqvm_cgm_lift_counters_print(void);

#ifdef __cplusplus
}
#endif

