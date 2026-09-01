#pragma once
/*
 * Native gyroscopic forward for Bonsai (L=36).
 *
 * Architecture = driver loops + block primitive:
 *   hqvm_forward_prefill / hqvm_forward_decode_step
 *     → for each (t, ell): depth = t*L + ell; hqvm_block_forward(...)
 *
 * Production path does not execute stock ggml block graph.
 */

#include <stddef.h>
#include <stdint.h>

#include "constants.h"
#include "codec.h"
#include "ledger.h"

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

#ifndef HQVM_HIDDEN_DIM
#define HQVM_HIDDEN_DIM 4096
#endif
#ifndef HQVM_FFN_DIM
#define HQVM_FFN_DIM 12288
#endif
#ifndef HQVM_N_HEAD
#define HQVM_N_HEAD 32
#endif
#ifndef HQVM_HEAD_DIM
#define HQVM_HEAD_DIM 128
#endif

typedef struct hqvm_block_weights {
    hqvm_q1_weight_t attn_q;
    hqvm_q1_weight_t attn_k;
    hqvm_q1_weight_t attn_v;
    hqvm_q1_weight_t attn_o;
    hqvm_q1_weight_t ffn_gate;
    hqvm_q1_weight_t ffn_up;
    hqvm_q1_weight_t ffn_down;
    const float *attn_norm_g; /* [HQVM_HIDDEN_DIM] */
    const float *ffn_norm_g;
    const float *attn_q_norm_g; /* [HQVM_HEAD_DIM] Qwen3 Q-RMSNorm */
    const float *attn_k_norm_g; /* [HQVM_HEAD_DIM] Qwen3 K-RMSNorm */
    float        attn_norm_g0;
    float        ffn_norm_g0;
    float        attn_q_norm_g0;
    float        attn_k_norm_g0;
} hqvm_block_weights_t;

/* Compatibility alias (older call sites). */
typedef hqvm_block_weights_t hqvm_layer_weights_t;

typedef struct hqvm_tail_weights {
    hqvm_q1_weight_t output;
    const float     *output_norm_g; /* [HQVM_HIDDEN_DIM] */
    float            output_norm_g0;
} hqvm_tail_weights_t;

typedef struct hqvm_block_kv {
    void   *k_q8;
    void   *v_q8;
    float  *k_f32; /* optional debug path: GYRO_NATIVE_KV=f32 */
    float  *v_f32;
    uint8_t *k_chi6;
    int64_t  n_ctx;
    int64_t  n_kv_heads;
    int64_t  kv_pos;
    size_t   k_row_stride; /* Q8 bytes / token, or f32 floats*4 */
    size_t   v_row_stride;
    int      use_f32;      /* 1 → score/reduce from k_f32/v_f32 */
} hqvm_block_kv_t;

typedef hqvm_block_kv_t hqvm_layer_kv_t;

HQVM_EXPORT int hqvm_native_forward_enabled(void);
HQVM_EXPORT int hqvm_native_weights_ready(void);
HQVM_EXPORT void hqvm_native_bypass_set(int on);
HQVM_EXPORT int hqvm_native_bypass_active(void);

/* Stock ggml block-graph invocations (must be 0 on production path). */
HQVM_EXPORT void hqvm_stock_block_forward_inc(void);
HQVM_EXPORT uint64_t hqvm_stock_block_forward_calls(void);
HQVM_EXPORT void hqvm_stock_block_forward_reset(void);
/* Aliases */
HQVM_EXPORT void hqvm_stock_graph_layer_inc(void);
HQVM_EXPORT uint64_t hqvm_stock_graph_layer_calls(void);
HQVM_EXPORT void hqvm_stock_graph_layer_reset(void);

HQVM_EXPORT void hqvm_native_block_inc(void);
HQVM_EXPORT uint64_t hqvm_native_block_calls(void);
HQVM_EXPORT void hqvm_native_layer_inc(void);
HQVM_EXPORT uint64_t hqvm_native_layer_calls(void);

/*
 * One block (layer) under Genealogy depth.
 * depth = token_pos * HQVM_N_LAYER + layer_idx; fam = depth & 3.
 * Mutates (*u6,*v6); writes K/V at kv_pos; writes x_out.
 */
HQVM_EXPORT int hqvm_block_forward(
    const hqvm_dyad32_t *x_in,
    int32_t token_pos,
    int32_t layer_idx,
    uint64_t depth,
    const hqvm_block_weights_t *W,
    uint8_t *u6,
    uint8_t *v6,
    hqvm_block_kv_t *KV,
    hqvm_dyad32_t *x_out);

/* Alias: same as hqvm_block_forward (depth computed if 0). */
HQVM_EXPORT int hqvm_layer_forward(
    const float *x_in,
    int32_t token_pos,
    int32_t layer_idx,
    int32_t n_tokens,
    const hqvm_layer_weights_t *W,
    uint8_t *u6,
    uint8_t *v6,
    hqvm_layer_kv_t *KV,
    float *x_out);

HQVM_EXPORT void hqvm_block_register_weights(int32_t layer_idx, const hqvm_block_weights_t *W);
HQVM_EXPORT const hqvm_block_weights_t *hqvm_block_get_weights(int32_t layer_idx);
HQVM_EXPORT void hqvm_layer_register_weights(int32_t layer_idx, const hqvm_layer_weights_t *W);
HQVM_EXPORT const hqvm_layer_weights_t *hqvm_layer_get_weights(int32_t layer_idx);

HQVM_EXPORT void hqvm_tail_register_weights(const hqvm_tail_weights_t *W);
HQVM_EXPORT const hqvm_tail_weights_t *hqvm_tail_get_weights(void);
HQVM_EXPORT int hqvm_native_tail_weights_ready(void);
HQVM_EXPORT int hqvm_native_tail_prepare(const hqvm_dyad32_t *x, int64_t n_tokens,
    const int32_t *out_ids, int64_t n_outputs);
HQVM_EXPORT int hqvm_native_tail_copy_norm(float *result_norm, size_t norm_stride_bytes);
HQVM_EXPORT int hqvm_native_tail_project(float *result_output, size_t output_stride_bytes,
    int64_t row0, int64_t row1, int32_t worker_id);
HQVM_EXPORT int hqvm_native_emission_enabled(void);
/* H-EMIT: commit authority — stock_selector_fallback must stay 0 under product profile.
 * Do not treat stem-Paris alone as emission ownership if committed=0 / fallback>0. */
HQVM_EXPORT int hqvm_native_emission_workers_begin(int32_t n_workers);
HQVM_EXPORT int hqvm_native_emission_reduce(int32_t n_workers);
HQVM_EXPORT int32_t hqvm_native_emission_selected_token(int64_t output_idx);
HQVM_EXPORT void hqvm_native_emission_reset(void);
HQVM_EXPORT uint64_t hqvm_native_selector_calls(void);
HQVM_EXPORT uint64_t hqvm_native_selector_scored_rows(void);
HQVM_EXPORT uint64_t hqvm_exact_tail_rows(void);
HQVM_EXPORT uint64_t hqvm_exact_tail_calls(void);
HQVM_EXPORT uint64_t hqvm_stock_selector_fallback_calls(void);
HQVM_EXPORT uint64_t hqvm_stock_selector_calls(void);
HQVM_EXPORT uint64_t hqvm_native_emission_committed_calls(void);
HQVM_EXPORT void hqvm_native_emission_commit_inc(void);
HQVM_EXPORT void hqvm_stock_selector_fallback_inc(void);
HQVM_EXPORT void hqvm_stock_selector_inc(void);
HQVM_EXPORT int64_t hqvm_native_tail_vocab(void);
HQVM_EXPORT int64_t hqvm_native_tail_n_outputs(void);
HQVM_EXPORT void hqvm_native_tail_clear(void);
HQVM_EXPORT void hqvm_native_tail_inc(void);
HQVM_EXPORT uint64_t hqvm_native_tail_calls(void);

/* Finite residual entry: original sign12 anchor plus Q8 coordinates/fp16 scale bits.
 * The decoded row is the current F32 interoperability ABI, not the owned storage law. */
HQVM_EXPORT int hqvm_entry_q8_encode_decode(
    hqvm_dyad32_t *x, int32_t T, uint8_t *pi_u6, uint8_t *pi_v6);
HQVM_EXPORT uint64_t hqvm_entry_q8_rows(void);
HQVM_EXPORT uint64_t hqvm_entry_q8_scale_blocks(void);
HQVM_EXPORT uint64_t hqvm_dyad_residual_rows(void);
HQVM_EXPORT uint64_t hqvm_dyad_residual_coordinates(void);
HQVM_EXPORT uint64_t hqvm_float_residual_storage_calls(void);
HQVM_EXPORT uint64_t hqvm_float_residual_adapter_calls(void);
HQVM_EXPORT uint64_t hqvm_dyad_scratch_rows(void);
HQVM_EXPORT uint64_t hqvm_dyad_scratch_bytes(void);

/* Request reset: embd Pi → (u6,v6) or explicit finite sign12 bits. */
HQVM_EXPORT void hqvm_reset_request(const float *embd_row0, int64_t n_embd);
HQVM_EXPORT void hqvm_reset_request_bits(uint8_t pi_u6, uint8_t pi_v6);

/*
 * Prefill: for t in 0..T-1, for ell in 0..35, depth=t*36+ell, block_forward.
 * x is residual [T * hidden]; tokens optional (Pi from embd_row0 if provided).
 */
HQVM_EXPORT int hqvm_forward_prefill(
    hqvm_dyad32_t *x, int32_t T,
    const float *embd_row0, int64_t n_embd,
    uint8_t *u6, uint8_t *v6, hqvm_block_kv_t *KV);

/*
 * Decode one token position t (already written into x row t): run 36 blocks.
 */
HQVM_EXPORT int hqvm_forward_decode_step(
    hqvm_dyad32_t *x_row, int32_t t,
    uint8_t *u6, uint8_t *v6, hqvm_block_kv_t *KV);

/* Legacy ubatch helper = prefill over residual stream without separate Pi row. */
HQVM_EXPORT int hqvm_native_forward_ubatch(
    float *x, int32_t n_tokens, uint8_t *u6, uint8_t *v6, hqvm_layer_kv_t *KV);

/* Native KV cache lifecycle (persistent across graph evals). */
HQVM_EXPORT hqvm_block_kv_t *hqvm_native_kv_get(int64_t n_ctx);
HQVM_EXPORT void hqvm_native_kv_reset(void);
HQVM_EXPORT void hqvm_native_kv_free(void);
HQVM_EXPORT int64_t hqvm_native_kv_pos(void);

/* Reset native KV + prefill state for a new sequence. */
HQVM_EXPORT void hqvm_native_sequence_reset(void);
HQVM_EXPORT int hqvm_native_prefill_done(void);
HQVM_EXPORT void hqvm_native_mark_prefill_done(void);

/* Executor integrity counters (Gate 0A). Increment only when stock op actually runs. */
HQVM_EXPORT void hqvm_stock_flash_attn_inc(void);
HQVM_EXPORT uint64_t hqvm_stock_flash_attn_calls(void);
HQVM_EXPORT void hqvm_stock_rope_inc(void);
HQVM_EXPORT uint64_t hqvm_stock_rope_calls(void);
HQVM_EXPORT void hqvm_stock_rmsnorm_inc(void);
HQVM_EXPORT uint64_t hqvm_stock_rmsnorm_calls(void);
HQVM_EXPORT void hqvm_stock_swiglu_inc(void);
HQVM_EXPORT uint64_t hqvm_stock_swiglu_calls(void);
HQVM_EXPORT void hqvm_stock_add_inc(void);
HQVM_EXPORT uint64_t hqvm_stock_add_calls(void);
HQVM_EXPORT void hqvm_stock_set_rows_inc(void);
HQVM_EXPORT uint64_t hqvm_stock_set_rows_calls(void);
HQVM_EXPORT void hqvm_stock_tail_inc(void);
HQVM_EXPORT uint64_t hqvm_stock_tail_calls(void);

HQVM_EXPORT void hqvm_kv_null_read_inc(void);
HQVM_EXPORT void hqvm_kv_null_write_inc(void);
HQVM_EXPORT uint64_t hqvm_kv_null_reads(void);
HQVM_EXPORT uint64_t hqvm_kv_null_writes(void);
HQVM_EXPORT void hqvm_kv_write_inc(uint64_t n);
HQVM_EXPORT void hqvm_kv_read_inc(uint64_t n);
HQVM_EXPORT uint64_t hqvm_kv_k_writes(void);
HQVM_EXPORT uint64_t hqvm_kv_v_writes(void);
HQVM_EXPORT uint64_t hqvm_kv_chi_writes(void);
HQVM_EXPORT uint64_t hqvm_kv_reads(void);

/* Owned-call receipts: observational counters per claimed-native site. */
HQVM_EXPORT void     hqvm_norm_ruler_commit_inc(void);
HQVM_EXPORT uint64_t hqvm_norm_ruler_commits(void);
HQVM_EXPORT void     hqvm_rope_codec_row_inc(void);
HQVM_EXPORT uint64_t hqvm_rope_codec_rows(void);
HQVM_EXPORT void     hqvm_attn_shell_weight_inc(void);
HQVM_EXPORT uint64_t hqvm_attn_shell_weight_calls(void);
HQVM_EXPORT void     hqvm_ffn_shell_gate_inc(void);
HQVM_EXPORT uint64_t hqvm_ffn_shell_gate_calls(void);
HQVM_EXPORT void     hqvm_lift_step_inc(void);
HQVM_EXPORT uint64_t hqvm_lift_steps(void);
HQVM_EXPORT void     hqvm_matmul_q1_inc(void);
HQVM_EXPORT uint64_t hqvm_matmul_q1_calls(void);
HQVM_EXPORT void     hqvm_score_dot_head_inc(void);
HQVM_EXPORT uint64_t hqvm_score_dot_heads(void);
HQVM_EXPORT void     hqvm_vkq_reduce_head_inc(void);
HQVM_EXPORT uint64_t hqvm_vkq_reduce_heads(void);
HQVM_EXPORT void     hqvm_node_seen_inc(void);
HQVM_EXPORT void     hqvm_node_bypassed_inc(void);
HQVM_EXPORT uint64_t hqvm_nodes_seen(void);
HQVM_EXPORT uint64_t hqvm_nodes_bypassed(void);

HQVM_EXPORT int hqvm_native_request_id(void);
HQVM_EXPORT void hqvm_native_request_begin(int is_prefill, int32_t T);
HQVM_EXPORT void hqvm_native_counters_print(const char *tag, int is_prefill, int32_t T, int64_t kv_pos);
HQVM_EXPORT void hqvm_native_counters_reset_request(void);

/* Debug ladder: GYRO_ATTN_LEVEL=0|1|2 (default 2). FFN/V-reduce native: hosting GYRO_FFN_NATIVE / GYRO_NATIVE_VREDUCE. */
HQVM_EXPORT int hqvm_attn_level(void);

#ifdef __cplusplus
}
#endif
