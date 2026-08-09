#pragma once
/*
 * Attention path: KV Q8 caches, holonomic QK/Attn@V, CGM-lift (owns the single traj).
 */

#include <stddef.h>
#include <stdint.h>

#include "constants.h"
#include "kernel.h"

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

#define HQVM_KV_REC_MAX     65536
#define HQVM_KV_CHI_BINS    64
#define HQVM_KV_N_KV_HEAD   8
#define HQVM_KV_N_LAYER     36
#define HQVM_KV_HALF        64
#define HQVM_KV_HEAD_DIM    128
#define HQVM_KEY_COORD_BYTES 20
#define HQVM_Q8_BLOCK 32

typedef struct gyro_kv_coord {
    uint8_t  chi6;
    int8_t   q8[HQVM_KV_HALF];
    float    d;
    float    mean_abs;
} gyro_kv_coord_t;

typedef struct gyro_kv_cell {
    gyro_kv_coord_t key_coords[2];
} gyro_kv_cell_t;

typedef struct gyro_key_coord {
    int8_t q8_0[HQVM_KV_HALF];
    int8_t q8_1[HQVM_KV_HALF];
    float  d0;
    float  d1;
} gyro_key_coord_t;

typedef struct hqvm_kv_rec {
    uint32_t token_i;
    uint8_t  chi6;
    uint8_t  shell;
    uint8_t  delta_chi;
    uint8_t  _pad;
    float    mean_abs;
} hqvm_kv_rec;

typedef struct hqvm_kv_ledger {
    uint32_t n_rec;
    uint32_t cap;
    uint8_t  prev_chi;
    uint8_t  has_prev;
    hqvm_kv_rec *recs;
    uint32_t n_coord;
    uint32_t coord_cap;
    gyro_kv_coord_t *coords;
} hqvm_kv_ledger;

enum {
    HQVM_HOL_MODE_DOT = 0,
    HQVM_HOL_MODE_ZERO = 1,
    HQVM_HOL_MODE_RANDOM = 2,
};

/* ---- KV / Q8 ---- */
HQVM_EXPORT hqvm_kv_ledger *hqvm_kv_ledger_global(void);
HQVM_EXPORT void hqvm_kv_ledger_reset(hqvm_kv_ledger *L);
HQVM_EXPORT int hqvm_kv_ledger_enabled(void);
HQVM_EXPORT int hqvm_kv_keys_enabled(void);
HQVM_EXPORT int hqvm_holonomic_attn_enabled(void);
HQVM_EXPORT int hqvm_holonomic_attn_mode(unsigned *seed_out);
HQVM_EXPORT int hqvm_coord_perturb_flip(void);
HQVM_EXPORT void hqvm_holonomic_counters_inc(int holonomic);
HQVM_EXPORT void hqvm_holonomic_counters_get(uint64_t *holonomic, uint64_t *stock);
HQVM_EXPORT void hqvm_holonomic_counters_print(void);
HQVM_EXPORT void hqvm_v_q8_inc(void);
HQVM_EXPORT void hqvm_kv_project_plane64(const float *plane64, gyro_kv_coord_t *out);
HQVM_EXPORT void hqvm_kv_project_head128(const float *head128, gyro_kv_cell_t *out);
HQVM_EXPORT uint16_t hqvm_f32_to_f16(float f);
HQVM_EXPORT float hqvm_f16_to_f32(uint16_t h);
HQVM_EXPORT float hqvm_q8_cache_row_score(
    const float *q_head128, const void *k_q8_blocks, float scale);
HQVM_EXPORT void hqvm_attn_v_accum_q8(
    float *VKQ32, int64_t DV, const void *v_q8_row, float a);
HQVM_EXPORT void hqvm_quantize_row_q8(const float *row, int64_t n, void *out_q8);
HQVM_EXPORT uint8_t hqvm_intron_from_q8_row(const void *k_q8_row);
HQVM_EXPORT uint8_t hqvm_k_chi6_from_row(const float *k_head128);
HQVM_EXPORT int hqvm_kv_ledger_append_f32(
    hqvm_kv_ledger *L, uint32_t token_i, const float *x, int64_t n);
HQVM_EXPORT void hqvm_kv_ledger_gather_histogram(
    const hqvm_kv_ledger *L, float H[HQVM_KV_CHI_BINS]);
HQVM_EXPORT uint32_t hqvm_kv_ledger_count(const hqvm_kv_ledger *L);
HQVM_EXPORT uint32_t hqvm_kv_ledger_coord_count(const hqvm_kv_ledger *L);

/* ---- Softmax / Attn@V / shadows ---- */
HQVM_EXPORT void hqvm_softmax_inplace(float *scores, int64_t Nk, float M);
HQVM_EXPORT void hqvm_percolation_shadow(const float *raw_scores, int64_t Nk, float M);
HQVM_EXPORT void hqvm_attn_v_reduce(
    float *out, int64_t DV, const float *weights, int64_t Nk,
    const void *v_base, size_t v_row_stride, int v_is_q8, int v_perturb);
HQVM_EXPORT void hqvm_receipts_on_layer(uint8_t intron_byte, int layer_i, int64_t Nk);
HQVM_EXPORT int hqvm_v_perturb_enabled(void);
HQVM_EXPORT int hqvm_percolation_shadow_enabled(void);
HQVM_EXPORT int hqvm_receipts_enabled(void);
HQVM_EXPORT void hqvm_residual_shadow_log(const float *row, int64_t n, int is_f16);
HQVM_EXPORT void hqvm_shell_norm_shadow_log(const float *x, int64_t n);
HQVM_EXPORT void hqvm_percolation_softmax(
    float *logits, const float *q_head128, const void *k_base_q8,
    size_t k_row_stride, int64_t Nk);
HQVM_EXPORT int hqvm_percolation_enabled(void);
HQVM_EXPORT void hqvm_percolation_gates_report(void);
HQVM_EXPORT void hqvm_shell_softmax(
    float *logits, const float *q_head128, const void *k_base_q8,
    size_t k_row_stride, int64_t Nk, float lambda);
HQVM_EXPORT int hqvm_shell_softmax_enabled(void);
HQVM_EXPORT float hqvm_shell_softmax_lambda(void);
HQVM_EXPORT void hqvm_aperture_softmax(
    float *logits, const float *q_head128, const void *k_base_q8,
    size_t k_row_stride, int64_t Nk, float Delta, float eps_max,
    const void *k_chi_base, int kv_head);
HQVM_EXPORT int hqvm_aperture_enabled(void);
HQVM_EXPORT void hqvm_aperture_shadow(
    const float *logits, const float *q_head128, const void *k_base_q8,
    size_t k_row_stride, int64_t Nk, float Delta, float eps_max,
    const void *k_chi_base, int kv_head);
HQVM_EXPORT void hqvm_aperture_rope_mix(
    float *dst, const float *src, int64_t n_dims, int64_t n_offset,
    int is_neox, float Delta, int64_t pos);
HQVM_EXPORT int hqvm_aperture_rope_enabled(void);
HQVM_EXPORT int hqvm_aperture_rms_enabled(void);

/* ---- CGM-lift (single traj owner) ---- */
typedef struct {
    uint8_t chi_q;
    uint8_t q6;
    uint8_t fam;
    uint8_t byte;
    int     argmax;
    uint8_t rank_r;
    float   eps;
    uint32_t phase_idx;
    uint32_t state24;
} gyro_lift_attn_t;

HQVM_EXPORT void hqvm_cgm_lift_init(void);
HQVM_EXPORT uint8_t hqvm_byte_of_q6_fam(uint8_t q6, uint8_t fam);
HQVM_EXPORT uint8_t hqvm_q6_of_byte(uint8_t byte);
HQVM_EXPORT uint8_t hqvm_fam_of_byte(uint8_t byte);
HQVM_EXPORT int hqvm_byte_of_q6_fam_ok(void);
HQVM_EXPORT int hqvm_cgm_lift_enabled(void);
HQVM_EXPORT int hqvm_cgm_lift_perturb_enabled(void);
HQVM_EXPORT int hqvm_cgm_lift_layer(void);
HQVM_EXPORT int hqvm_cgm_lift_bump_layer(void);
HQVM_EXPORT int hqvm_cgm_lift_traj_ready(void);
HQVM_EXPORT uint32_t hqvm_cgm_lift_state24(void);
HQVM_EXPORT uint8_t hqvm_cgm_lift_last_byte(void);
HQVM_EXPORT int hqvm_residual_hybrid_enabled(void);
HQVM_EXPORT float hqvm_residual_gain(void);
HQVM_EXPORT void hqvm_residual_hybrid_hit(void);
HQVM_EXPORT uint64_t hqvm_residual_hybrid_hits(void);
HQVM_EXPORT void hqvm_residual_hybrid_counters_print(void);
HQVM_EXPORT void hqvm_cgm_lift_counters_get(
    uint64_t *lift_calls, uint64_t *chi6_writes, uint64_t *invariant_fails);
HQVM_EXPORT void hqvm_cgm_lift_counters_print(void);
HQVM_EXPORT void hqvm_k_chi6_store(
    const void *k_base, int64_t idx, const float *row_f32, int64_t n_heads, int64_t head_dim);
HQVM_EXPORT uint8_t hqvm_k_chi6_get(const void *k_base, int64_t idx, int head);
HQVM_EXPORT int hqvm_k_chi6_has(const void *k_base);
HQVM_EXPORT void hqvm_lift_attention_phase(
    const float *scores, const void *k_base, int64_t Nk, int head,
    uint8_t chi_q, int depth, float Delta, float eps_max, gyro_lift_attn_t *out);

#ifdef __cplusplus
}
#endif
