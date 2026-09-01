/*
 * Native gyroscopic forward: driver loops + hqvm_block_forward (L=36).
 *
 * Activations in this file are hqvm_dyad32_t (dyadic magnitude). That is the
 * native controller lane — not a float with a new name. Sites that still unpack
 * to float for stock math must be labeled stock (H6/H7 today), never "native."
 * Owners: codec (norm/RoPE/FFN), ledger (MatMul), attn (scores/V-reduce).
 */

#include "layer.h"

#include "attn.h"
#include "codec.h"
#include "constants.h"
#include "kernel.h"
#include "ledger.h"
#include "runtime.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t s_stock_block = 0;
static uint64_t s_native_block = 0;
static int s_bypass = 0;
static hqvm_block_weights_t s_W[HQVM_N_LAYER];
static uint8_t s_W_ok[HQVM_N_LAYER];
static hqvm_tail_weights_t s_tail_W;
static int s_tail_W_ok = 0;
static hqvm_sidecar s_tail_side;
static int s_tail_side_ok = 0;
static hqvm_dyad32_t *s_tail_norm_d = NULL;
static int8_t *s_tail_qx = NULL;
static uint16_t *s_tail_xd = NULL;
static int64_t s_tail_cap_outputs = 0;
static int64_t s_tail_n_outputs = 0;
static uint64_t s_native_tail = 0;
static int s_native_emission = -1;
static int32_t s_emission_workers = 0;
static int64_t s_emission_outputs = 0;
static int32_t *s_emission_rows = NULL;
static hqvm_i128 *s_emission_scores = NULL;
static uint64_t s_native_selector = 0;
static uint64_t s_native_selector_rows = 0;
static uint64_t s_exact_tail_rows = 0;
static uint64_t s_exact_tail_calls = 0;
static uint64_t s_entry_q8_rows = 0;
static uint64_t s_entry_q8_scale_blocks = 0;
static uint64_t s_dyad_residual_rows = 0;
static uint64_t s_dyad_residual_coordinates = 0;
static uint64_t s_float_residual_storage_calls = 0;
static uint64_t s_float_residual_adapter_calls = 0;
static uint64_t s_stock_selector_fallback = 0;
static uint64_t s_stock_selector = 0;
static uint64_t s_native_emission_committed = 0;

/* Owned-call receipts: each claimed-native site counts its own work here so
 * "native" is observational (an executed call), not inferred from bypass.
 * Node telemetry lives in ggml-cpu.c: nodes_seen/nodes_bypassed count graph
 * nodes classified as block work and skipped while the driver owns them. */
static uint64_t s_norm_ruler_commits = 0;      /* Δ-ruler norm rows committed */
static uint64_t s_rope_codec_rows = 0;         /* rows turned via finite tick chart */
static uint64_t s_attn_shell_weight_calls = 0; /* algebraic shell weight law calls */
static uint64_t s_ffn_shell_gate_calls = 0;    /* shell-family FFN gate calls */
static uint64_t s_lift_steps = 0;              /* carrier (u,v) steps in driver */
static uint64_t s_matmul_q1_calls = 0;         /* ledger MatMul dispatches */
static uint64_t s_score_dot_heads = 0;         /* per-head exact score passes */
static uint64_t s_vkq_reduce_heads = 0;        /* per-head Attn@V reductions */
static uint64_t s_nodes_seen = 0;              /* stock block-work nodes observed */
static uint64_t s_nodes_bypassed = 0;          /* of those, skipped by driver */

static hqvm_block_kv_t s_kv;
static int s_kv_init = 0;
static int s_native_prefill_done = 0;

static uint64_t s_dyad_scratch_rows = 0;
static uint64_t s_dyad_scratch_bytes = 0;

/* Persistent block scratch (avoids ~11 malloc/free per hqvm_block_forward). */
#define HQVM_BLK_DYADS_FIXED (8 * HQVM_HIDDEN_DIM + 3 * HQVM_FFN_DIM + HQVM_HIDDEN_DIM)
static hqvm_dyad32_t *s_blk_dyad = NULL;
static int64_t       s_blk_dyad_cap = 0;

static int block_dyad_scratch_ensure(int64_t kv_slots) {
    const int64_t need = HQVM_BLK_DYADS_FIXED + 2 * (kv_slots > 0 ? kv_slots : 1);
    if (s_blk_dyad && need <= s_blk_dyad_cap) {
        return 0;
    }
    free(s_blk_dyad);
    s_blk_dyad = (hqvm_dyad32_t *)malloc((size_t)need * sizeof(hqvm_dyad32_t));
    if (!s_blk_dyad) {
        s_blk_dyad_cap = 0;
        return -1;
    }
    s_blk_dyad_cap = need;
    return 0;
}

#define BLK_OFF_X_NORM  0
#define BLK_OFF_X_N2    (1 * HQVM_HIDDEN_DIM)
#define BLK_OFF_Q       (2 * HQVM_HIDDEN_DIM)
#define BLK_OFF_K       (3 * HQVM_HIDDEN_DIM)
#define BLK_OFF_V       (4 * HQVM_HIDDEN_DIM)
#define BLK_OFF_ATTN    (5 * HQVM_HIDDEN_DIM)
#define BLK_OFF_ATTN_O  (6 * HQVM_HIDDEN_DIM)
#define BLK_OFF_X_MID   (7 * HQVM_HIDDEN_DIM)
#define BLK_OFF_GATE    (8 * HQVM_HIDDEN_DIM)
#define BLK_OFF_UP      (BLK_OFF_GATE + HQVM_FFN_DIM)
#define BLK_OFF_FFN_H   (BLK_OFF_UP + HQVM_FFN_DIM)
#define BLK_OFF_FFN_O   (BLK_OFF_FFN_H + HQVM_FFN_DIM)
#define BLK_OFF_WT(k)   (BLK_OFF_FFN_O + HQVM_HIDDEN_DIM)
#define BLK_OFF_LWT(k)  (BLK_OFF_WT(k) + (k))

static uint64_t s_stock_flash_attn = 0;
static uint64_t s_stock_rope = 0;
static uint64_t s_stock_rmsnorm = 0;
static uint64_t s_stock_swiglu = 0;
static uint64_t s_stock_add = 0;
static uint64_t s_stock_set_rows = 0;
static uint64_t s_stock_tail = 0;
static uint64_t s_kv_null_reads = 0;
static uint64_t s_kv_null_writes = 0;
static uint64_t s_kv_k_writes = 0;
static uint64_t s_kv_v_writes = 0;
static uint64_t s_kv_chi_writes = 0;
static uint64_t s_kv_reads = 0;
static int s_request_id = 0;
static uint64_t s_native_block_req0 = 0;

static void norm_apply_ruler_dyad(
    const hqvm_dyad32_t *x_in, hqvm_dyad32_t *x_out, int64_t n,
    const float *g, float g0);
static int matmul_q1_dyad_to_dyad(
    const hqvm_q1_weight_t *W, const hqvm_dyad32_t *x, hqvm_dyad32_t *y);

hqvm_block_kv_t *hqvm_native_kv_get(int64_t n_ctx) {
    const int64_t n_kv_heads = HQVM_KV_N_KV_HEAD;
    const int64_t head_dim = HQVM_HEAD_DIM;
    const int64_t blk_per_head = head_dim / HQVM_Q8_BLOCK;
    const size_t bytes_per_head = (size_t)blk_per_head * 34u;
    const size_t row_stride_q8 = bytes_per_head * (size_t)n_kv_heads;
    const size_t row_stride_f32 = (size_t)n_kv_heads * (size_t)head_dim * sizeof(float);
    const size_t layer_bytes_q8 = (size_t)n_ctx * row_stride_q8;
    const size_t layer_bytes_f32 = (size_t)n_ctx * row_stride_f32;
    int use_f32 = 0;

    if (s_kv_init) return &s_kv;
    if (n_ctx <= 0) {
        const char *e = getenv("GYRO_NATIVE_N_CTX");
        n_ctx = (e && e[0]) ? (int64_t)atoll(e) : 4096;
    }
    {
        const char *e = getenv("GYRO_NATIVE_KV");
        use_f32 = (e && (strcmp(e, "f32") == 0 || strcmp(e, "float") == 0)) ? 1 : 0;
    }

    s_kv.k_q8 = calloc((size_t)HQVM_N_LAYER * layer_bytes_q8, 1);
    s_kv.v_q8 = calloc((size_t)HQVM_N_LAYER * layer_bytes_q8, 1);
    s_kv.k_chi6 = calloc((size_t)HQVM_N_LAYER * (size_t)n_ctx * (size_t)n_kv_heads, 1);
    s_kv.k_f32 = NULL;
    s_kv.v_f32 = NULL;
    s_kv.use_f32 = use_f32;
    if (use_f32) {
        s_kv.k_f32 = (float *)calloc((size_t)HQVM_N_LAYER * layer_bytes_f32, 1);
        s_kv.v_f32 = (float *)calloc((size_t)HQVM_N_LAYER * layer_bytes_f32, 1);
    }
    if (!s_kv.k_q8 || !s_kv.v_q8 || !s_kv.k_chi6 || (use_f32 && (!s_kv.k_f32 || !s_kv.v_f32))) {
        free(s_kv.k_q8);
        free(s_kv.v_q8);
        free(s_kv.k_chi6);
        free(s_kv.k_f32);
        free(s_kv.v_f32);
        memset(&s_kv, 0, sizeof(s_kv));
        return NULL;
    }
    s_kv.n_ctx = n_ctx;
    s_kv.n_kv_heads = n_kv_heads;
    s_kv.k_row_stride = use_f32 ? (int64_t)row_stride_f32 : (int64_t)row_stride_q8;
    s_kv.v_row_stride = s_kv.k_row_stride;
    s_kv.kv_pos = -1;
    s_kv_init = 1;

    fprintf(stderr,
        "[hqvm-kv] native KV allocated n_ctx=%lld n_layers=%d n_kv_heads=%lld "
        "row_stride=%zu use_f32=%d bytes_per_head_q8=%zu\n",
        (long long)n_ctx, HQVM_N_LAYER, (long long)n_kv_heads,
        (size_t)s_kv.k_row_stride, use_f32, bytes_per_head);
    fflush(stderr);
    return &s_kv;
}

void hqvm_native_kv_reset(void) {
    if (s_kv_init) s_kv.kv_pos = -1;
}

void hqvm_native_kv_free(void) {
    if (!s_kv_init) return;
    free(s_kv.k_q8);
    free(s_kv.v_q8);
    free(s_kv.k_f32);
    free(s_kv.v_f32);
    free(s_kv.k_chi6);
    memset(&s_kv, 0, sizeof(s_kv));
    s_kv_init = 0;
}

int64_t hqvm_native_kv_pos(void) {
    return s_kv_init ? s_kv.kv_pos : -1;
}

void hqvm_native_sequence_reset(void) {
    hqvm_native_kv_reset();
    s_native_prefill_done = 0;
    /* New sequence boundary: fresh request cell in the genealogy loop. */
    if (hqvm_rt_enabled()) {
        hqvm_rt_request_reset(HQVM_RT_SEED_REST);
    }
}

int hqvm_native_prefill_done(void) {
    return s_native_prefill_done;
}

void hqvm_native_mark_prefill_done(void) {
    s_native_prefill_done = 1;
}

int hqvm_native_forward_enabled(void) {
    static int s = -1;
    if (s < 0) {
        const char *e = getenv("GYRO_NATIVE_FORWARD");
        s = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    return s;
}

int hqvm_native_weights_ready(void) {
    int32_t l;
    for (l = 0; l < HQVM_N_LAYER; ++l) {
        if (!s_W_ok[l]) return 0;
        if (!s_W[l].attn_q.q1_data || !s_W[l].attn_k.q1_data || !s_W[l].attn_v.q1_data
            || !s_W[l].attn_o.q1_data || !s_W[l].ffn_gate.q1_data || !s_W[l].ffn_up.q1_data
            || !s_W[l].ffn_down.q1_data) {
            return 0;
        }
    }
    return 1;
}

int hqvm_native_tail_weights_ready(void) {
    return s_tail_W_ok && s_tail_W.output.q1_data && s_tail_W.output_norm_g
        && s_tail_W.output.n_cols == HQVM_HIDDEN_DIM
        && s_tail_W.output.n_rows > 0 && s_tail_W.output.row_stride_bytes > 0;
}

void hqvm_native_bypass_set(int on) { s_bypass = on ? 1 : 0; }
int hqvm_native_bypass_active(void) { return s_bypass; }

void hqvm_stock_block_forward_inc(void) { s_stock_block++; }
uint64_t hqvm_stock_block_forward_calls(void) { return s_stock_block; }
void hqvm_stock_block_forward_reset(void) { s_stock_block = 0; }

void hqvm_stock_graph_layer_inc(void) { hqvm_stock_block_forward_inc(); }
uint64_t hqvm_stock_graph_layer_calls(void) { return hqvm_stock_block_forward_calls(); }
void hqvm_stock_graph_layer_reset(void) { hqvm_stock_block_forward_reset(); }

void hqvm_native_block_inc(void) { s_native_block++; }
uint64_t hqvm_native_block_calls(void) { return s_native_block; }
void hqvm_native_layer_inc(void) { hqvm_native_block_inc(); }
uint64_t hqvm_native_layer_calls(void) { return hqvm_native_block_calls(); }
void hqvm_native_tail_inc(void) { s_native_tail++; }
uint64_t hqvm_native_tail_calls(void) { return s_native_tail; }

int hqvm_native_emission_enabled(void) {
    if (s_native_emission < 0) {
        const char * e = getenv("GYRO_NATIVE_EMISSION");
        s_native_emission = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    return s_native_emission;
}

void hqvm_native_emission_reset(void) {
    free(s_emission_rows);
    free(s_emission_scores);
    s_emission_rows = NULL;
    s_emission_scores = NULL;
    s_emission_workers = 0;
    s_emission_outputs = 0;
}

int hqvm_native_emission_workers_begin(int32_t n_workers) {
    hqvm_native_emission_reset();
    if (!hqvm_native_emission_enabled() || n_workers <= 0 || s_tail_n_outputs <= 0) return 0;
    s_emission_rows = (int32_t *)malloc((size_t)n_workers * (size_t)s_tail_n_outputs * sizeof(int32_t));
    s_emission_scores = (hqvm_i128 *)malloc((size_t)n_workers * (size_t)s_tail_n_outputs * sizeof(hqvm_i128));
    if (!s_emission_rows || !s_emission_scores) {
        hqvm_native_emission_reset();
        return -1;
    }
    s_emission_workers = n_workers;
    s_emission_outputs = s_tail_n_outputs;
    memset(s_emission_rows, 0xff, (size_t)n_workers * (size_t)s_tail_n_outputs * sizeof(int32_t));
    return 0;
}

int hqvm_native_emission_reduce(int32_t n_workers) {
    int64_t t;
    int32_t w;
    if (!s_emission_rows || !s_emission_scores || n_workers != s_emission_workers) return 0;
    for (t = 0; t < s_tail_n_outputs; ++t) {
        int32_t best_row = -1;
        hqvm_i128 best_score = { 0, 0 };
        for (w = 0; w < n_workers; ++w) {
            const int32_t row = s_emission_rows[(size_t)w * s_tail_n_outputs + t];
            const hqvm_i128 score = s_emission_scores[(size_t)w * s_tail_n_outputs + t];
            {
                const int cmp = best_row < 0 ? 1 : hqvm_i128_cmp(score, best_score);
                if (row >= 0 && (cmp > 0 || (cmp == 0 && row < best_row))) {
                    best_score = score;
                    best_row = row;
                }
            }
        }
        s_emission_rows[t] = best_row;
        s_emission_scores[t] = best_score;
        s_native_selector_rows += (uint64_t)s_tail_W.output.n_rows;
        s_exact_tail_rows += (uint64_t)s_tail_W.output.n_rows;
        s_exact_tail_calls += (uint64_t)n_workers;
        s_native_selector++;
    }
    return 0;
}

int32_t hqvm_native_emission_selected_token(int64_t output_idx) {
    if (output_idx < 0 || output_idx >= s_emission_outputs || !s_emission_rows) return -1;
    return s_emission_rows[output_idx];
}
uint64_t hqvm_native_selector_calls(void) { return s_native_selector; }
uint64_t hqvm_native_selector_scored_rows(void) { return s_native_selector_rows; }
uint64_t hqvm_exact_tail_rows(void) { return s_exact_tail_rows; }
uint64_t hqvm_exact_tail_calls(void) { return s_exact_tail_calls; }
uint64_t hqvm_stock_selector_fallback_calls(void) { return s_stock_selector_fallback; }
uint64_t hqvm_stock_selector_calls(void) { return s_stock_selector; }
uint64_t hqvm_native_emission_committed_calls(void) { return s_native_emission_committed; }
void hqvm_native_emission_commit_inc(void) { s_native_emission_committed++; }
void hqvm_stock_selector_fallback_inc(void) { s_stock_selector_fallback++; }
void hqvm_stock_selector_inc(void) { s_stock_selector++; }

void hqvm_stock_flash_attn_inc(void) { s_stock_flash_attn++; }
uint64_t hqvm_stock_flash_attn_calls(void) { return s_stock_flash_attn; }
void hqvm_stock_rope_inc(void) { s_stock_rope++; }
uint64_t hqvm_stock_rope_calls(void) { return s_stock_rope; }
void hqvm_stock_rmsnorm_inc(void) { s_stock_rmsnorm++; }
uint64_t hqvm_stock_rmsnorm_calls(void) { return s_stock_rmsnorm; }
void hqvm_stock_swiglu_inc(void) { s_stock_swiglu++; }
uint64_t hqvm_stock_swiglu_calls(void) { return s_stock_swiglu; }
void hqvm_stock_add_inc(void) { s_stock_add++; }
uint64_t hqvm_stock_add_calls(void) { return s_stock_add; }
void hqvm_stock_set_rows_inc(void) { s_stock_set_rows++; }
uint64_t hqvm_stock_set_rows_calls(void) { return s_stock_set_rows; }
void hqvm_stock_tail_inc(void) { s_stock_tail++; }
uint64_t hqvm_stock_tail_calls(void) { return s_stock_tail; }

void hqvm_kv_null_read_inc(void) { s_kv_null_reads++; }
void hqvm_kv_null_write_inc(void) { s_kv_null_writes++; }
uint64_t hqvm_kv_null_reads(void) { return s_kv_null_reads; }
uint64_t hqvm_kv_null_writes(void) { return s_kv_null_writes; }
void hqvm_kv_write_inc(uint64_t n) {
    s_kv_k_writes += n;
    s_kv_v_writes += n;
    s_kv_chi_writes += n;
}
void hqvm_kv_read_inc(uint64_t n) { s_kv_reads += n; }
uint64_t hqvm_kv_k_writes(void) { return s_kv_k_writes; }
uint64_t hqvm_kv_v_writes(void) { return s_kv_v_writes; }
uint64_t hqvm_kv_chi_writes(void) { return s_kv_chi_writes; }
uint64_t hqvm_kv_reads(void) { return s_kv_reads; }

/* Owned-call receipt accessors (see statics above). */
void hqvm_norm_ruler_commit_inc(void) { s_norm_ruler_commits++; }
uint64_t hqvm_norm_ruler_commits(void) { return s_norm_ruler_commits; }
void hqvm_rope_codec_row_inc(void) { s_rope_codec_rows++; }
uint64_t hqvm_rope_codec_rows(void) { return s_rope_codec_rows; }
void hqvm_attn_shell_weight_inc(void) { s_attn_shell_weight_calls++; }
uint64_t hqvm_attn_shell_weight_calls(void) { return s_attn_shell_weight_calls; }
void hqvm_ffn_shell_gate_inc(void) { s_ffn_shell_gate_calls++; }
uint64_t hqvm_ffn_shell_gate_calls(void) { return s_ffn_shell_gate_calls; }
void hqvm_lift_step_inc(void) { s_lift_steps++; }
uint64_t hqvm_lift_steps(void) { return s_lift_steps; }
void hqvm_matmul_q1_inc(void) { s_matmul_q1_calls++; }
uint64_t hqvm_matmul_q1_calls(void) { return s_matmul_q1_calls; }
void hqvm_score_dot_head_inc(void) { s_score_dot_heads++; }
uint64_t hqvm_score_dot_heads(void) { return s_score_dot_heads; }
void hqvm_vkq_reduce_head_inc(void) { s_vkq_reduce_heads++; }
uint64_t hqvm_vkq_reduce_heads(void) { return s_vkq_reduce_heads; }
void hqvm_node_seen_inc(void) { s_nodes_seen++; }
void hqvm_node_bypassed_inc(void) { s_nodes_bypassed++; }
uint64_t hqvm_nodes_seen(void) { return s_nodes_seen; }
uint64_t hqvm_nodes_bypassed(void) { return s_nodes_bypassed; }

int hqvm_native_request_id(void) { return s_request_id; }

void hqvm_native_counters_reset_request(void) {
    s_stock_block = 0;
    s_native_block = 0;
    s_stock_flash_attn = 0;
    s_stock_rope = 0;
    s_stock_rmsnorm = 0;
    s_stock_swiglu = 0;
    s_stock_add = 0;
    s_stock_set_rows = 0;
    s_stock_tail = 0;
    s_native_tail = 0;
    s_native_selector = 0;
    s_native_selector_rows = 0;
    s_exact_tail_rows = 0;
    s_exact_tail_calls = 0;
    s_entry_q8_rows = 0;
    s_entry_q8_scale_blocks = 0;
    s_dyad_residual_rows = 0;
    s_dyad_residual_coordinates = 0;
    s_float_residual_storage_calls = 0;
    s_float_residual_adapter_calls = 0;
    s_dyad_scratch_rows = 0;
    s_dyad_scratch_bytes = 0;
    s_stock_selector_fallback = 0;
    s_stock_selector = 0;
    s_native_emission_committed = 0;
    s_norm_ruler_commits = 0;
    s_rope_codec_rows = 0;
    s_attn_shell_weight_calls = 0;
    s_ffn_shell_gate_calls = 0;
    s_lift_steps = 0;
    s_matmul_q1_calls = 0;
    s_score_dot_heads = 0;
    s_vkq_reduce_heads = 0;
    s_nodes_seen = 0;
    s_nodes_bypassed = 0;
    hqvm_native_emission_reset();
    s_kv_null_reads = 0;
    s_kv_null_writes = 0;
    s_kv_k_writes = 0;
    s_kv_v_writes = 0;
    s_kv_chi_writes = 0;
    s_kv_reads = 0;
    s_native_block_req0 = 0;
    hqvm_rt_counters_request_reset();
}

void hqvm_native_request_begin(int is_prefill, int32_t T) {
    if (is_prefill) {
        s_request_id++;
        hqvm_native_counters_reset_request();
        fprintf(stderr,
            "[hqvm-native] ===== request_begin id=%d T=%d is_prefill=1 =====\n",
            s_request_id, (int)T);
        fflush(stderr);
    }
    s_native_block_req0 = s_native_block;
    (void)T;
}

void hqvm_native_counters_print(const char *tag, int is_prefill, int32_t T, int64_t kv_pos) {
    const uint64_t nat = hqvm_native_block_calls();
    const uint64_t nat_delta = nat - s_native_block_req0;
    fprintf(stderr,
        "[hqvm-native] %s req=%d T=%d is_prefill=%d kv_pos=%lld "
        "native_block_calls=%llu native_block_delta=%llu "
        "stock_block_forward_calls=%llu stock_flash_attn_calls=%llu "
        "stock_softmax_calls=%llu stock_rope_calls=%llu stock_rmsnorm_calls=%llu "
        "stock_swiglu_calls=%llu stock_silu_calls=%llu stock_add_calls=%llu "
        "set_rows_calls=%llu native_tail_calls=%llu stock_tail_calls=%llu "
        "native_selector_calls=%llu native_selector_scored_rows=%llu "
        "exact_tail_rows=%llu exact_tail_calls=%llu "
        "stock_selector_fallback_calls=%llu stock_selector_calls=%llu "
        "native_emission_committed_calls=%llu "
        "entry_q8_rows=%llu entry_q8_scale_blocks=%llu "
        "dyad_residual_rows=%llu dyad_residual_coordinates=%llu "
        "float_residual_storage_calls=%llu float_residual_adapter_calls=%llu "
        "dyad_scratch_rows=%llu dyad_scratch_bytes=%llu "
        "kv_null_reads=%llu kv_null_writes=%llu "
        "K_writes=%llu V_writes=%llu chiK_writes=%llu kv_reads=%llu "
        "owned_norm_ruler=%llu owned_rope_rows=%llu "
        "owned_shell_weight=%llu owned_ffn_gate=%llu owned_lift_steps=%llu "
        "owned_matmul_q1=%llu owned_score_heads=%llu owned_vkq_heads=%llu "
        "nodes_seen=%llu nodes_bypassed=%llu "
        "rt_stock_ops_total=%llu rt_prefilter_calls=%llu rt_prefilter_skipped=%llu "
        "rt_group_calls=%llu rt_group_rows=%llu rt_group_groups=%llu "
        "rt_genealogy_events=%llu rt_genealogy_requests=%llu "
        "attn_level=%d ffn_native=%d vreduce_native=%d attn_scores_native=%d pi_applied=%d\n",
        tag ? tag : "summary",
        s_request_id, (int)T, is_prefill, (long long)kv_pos,
        (unsigned long long)nat,
        (unsigned long long)nat_delta,
        (unsigned long long)hqvm_stock_block_forward_calls(),
        (unsigned long long)hqvm_stock_flash_attn_calls(),
        (unsigned long long)hqvm_stock_softmax_calls(),
        (unsigned long long)hqvm_stock_rope_calls(),
        (unsigned long long)hqvm_stock_rmsnorm_calls(),
        (unsigned long long)hqvm_stock_swiglu_calls(),
        (unsigned long long)hqvm_stock_silu_calls(),
        (unsigned long long)hqvm_stock_add_calls(),
        (unsigned long long)hqvm_stock_set_rows_calls(),
        (unsigned long long)hqvm_native_tail_calls(),
        (unsigned long long)hqvm_stock_tail_calls(),
        (unsigned long long)hqvm_native_selector_calls(),
        (unsigned long long)hqvm_native_selector_scored_rows(),
        (unsigned long long)hqvm_exact_tail_rows(),
        (unsigned long long)hqvm_exact_tail_calls(),
        (unsigned long long)hqvm_stock_selector_fallback_calls(),
        (unsigned long long)hqvm_stock_selector_calls(),
        (unsigned long long)hqvm_native_emission_committed_calls(),
        (unsigned long long)hqvm_entry_q8_rows(),
        (unsigned long long)hqvm_entry_q8_scale_blocks(),
        (unsigned long long)hqvm_dyad_residual_rows(),
        (unsigned long long)hqvm_dyad_residual_coordinates(),
        (unsigned long long)hqvm_float_residual_storage_calls(),
        (unsigned long long)hqvm_float_residual_adapter_calls(),
        (unsigned long long)hqvm_dyad_scratch_rows(),
        (unsigned long long)hqvm_dyad_scratch_bytes(),
        (unsigned long long)hqvm_kv_null_reads(),
        (unsigned long long)hqvm_kv_null_writes(),
        (unsigned long long)hqvm_kv_k_writes(),
        (unsigned long long)hqvm_kv_v_writes(),
        (unsigned long long)hqvm_kv_chi_writes(),
        (unsigned long long)hqvm_kv_reads(),
        (unsigned long long)hqvm_norm_ruler_commits(),
        (unsigned long long)hqvm_rope_codec_rows(),
        (unsigned long long)hqvm_attn_shell_weight_calls(),
        (unsigned long long)hqvm_ffn_shell_gate_calls(),
        (unsigned long long)hqvm_lift_steps(),
        (unsigned long long)hqvm_matmul_q1_calls(),
        (unsigned long long)hqvm_score_dot_heads(),
        (unsigned long long)hqvm_vkq_reduce_heads(),
        (unsigned long long)hqvm_nodes_seen(),
        (unsigned long long)hqvm_nodes_bypassed(),
        (unsigned long long)hqvm_rt_stock_ops_total(),
        (unsigned long long)hqvm_rt_prefilter_calls(),
        (unsigned long long)hqvm_rt_prefilter_skipped(),
        (unsigned long long)hqvm_rt_group_calls(),
        (unsigned long long)hqvm_rt_group_rows(),
        (unsigned long long)hqvm_rt_group_groups(),
        (unsigned long long)hqvm_rt_log_events(),
        (unsigned long long)hqvm_rt_log_requests(),
        hqvm_attn_level(),
        hqvm_ffn_native_enabled(),
        hqvm_vreduce_native_enabled(),
        hqvm_attn_scores_native_enabled(),
        hqvm_pi_applied());
    fflush(stderr);
}

int hqvm_attn_level(void) {
    static int s = -99;
    if (s == -99) {
        const char *e = getenv("GYRO_ATTN_LEVEL");
        if (e && strcmp(e, "softmax") == 0) s = -1; /* debug stock softmax */
        else if (e && e[0]) s = atoi(e);
        else s = 2; /* full law */
        if (s < -1) s = -1;
        if (s > 2) s = 2;
    }
    return s;
}

void hqvm_block_register_weights(int32_t layer_idx, const hqvm_block_weights_t *W) {
    if (!W || layer_idx < 0 || layer_idx >= HQVM_N_LAYER) return;
    s_W[layer_idx] = *W;
    s_W_ok[layer_idx] = 1;
}

const hqvm_block_weights_t *hqvm_block_get_weights(int32_t layer_idx) {
    if (layer_idx < 0 || layer_idx >= HQVM_N_LAYER || !s_W_ok[layer_idx]) return NULL;
    return &s_W[layer_idx];
}

void hqvm_layer_register_weights(int32_t layer_idx, const hqvm_layer_weights_t *W) {
    hqvm_block_register_weights(layer_idx, W);
}

const hqvm_layer_weights_t *hqvm_layer_get_weights(int32_t layer_idx) {
    return hqvm_block_get_weights(layer_idx);
}

void hqvm_tail_register_weights(const hqvm_tail_weights_t *W) {
    if (!W) return;
    s_tail_W = *W;
    s_tail_W_ok = 1;
}

const hqvm_tail_weights_t *hqvm_tail_get_weights(void) {
    return hqvm_native_tail_weights_ready() ? &s_tail_W : NULL;
}

static int hqvm_tail_sidecar_ready(void) {
    if (s_tail_side_ok == 0) {
        const char *path = getenv("GYRO_LEDGER_PATH");
        if (path && path[0] && hqvm_sidecar_load(&s_tail_side, path) == 0) {
            hqvm_sidecar_apply_env_allow(&s_tail_side);
            s_tail_side_ok = 1;
        } else {
            s_tail_side_ok = -1;
        }
    }
    return s_tail_side_ok > 0;
}

int hqvm_native_tail_prepare(const hqvm_dyad32_t *x, int64_t n_tokens, const int32_t *out_ids, int64_t n_outputs) {
    int64_t t;
    if (!hqvm_native_tail_weights_ready() || !x || !out_ids || n_tokens <= 0 || n_outputs <= 0) return -1;
    if (!hqvm_tail_sidecar_ready() || n_outputs > s_tail_cap_outputs) {
        if (n_outputs > s_tail_cap_outputs) {
            free(s_tail_norm_d);
            s_tail_norm_d = (hqvm_dyad32_t *)malloc((size_t)n_outputs * HQVM_HIDDEN_DIM * sizeof(hqvm_dyad32_t));
            s_tail_cap_outputs = s_tail_norm_d ? n_outputs : 0;
        }
        if (!hqvm_tail_sidecar_ready() || !s_tail_norm_d) return -1;
    }
    for (t = 0; t < n_outputs; ++t) {
        const int32_t src = out_ids[t];
        if (src < 0 || src >= n_tokens) return -1;
        norm_apply_ruler_dyad(x + (size_t)src * HQVM_HIDDEN_DIM,
            s_tail_norm_d + (size_t)t * HQVM_HIDDEN_DIM, HQVM_HIDDEN_DIM,
            s_tail_W.output_norm_g, s_tail_W.output_norm_g0);
    }
    /* dyad-owned tail scratch: final norm rows (4096 dyad each) */
    s_dyad_scratch_rows += (uint64_t)n_outputs;
    s_dyad_scratch_bytes += (uint64_t)n_outputs * (uint64_t)HQVM_HIDDEN_DIM * sizeof(hqvm_dyad32_t);
    free(s_tail_qx);
    free(s_tail_xd);
    s_tail_qx = (int8_t *)malloc((size_t)n_outputs * HQVM_HIDDEN_DIM);
    s_tail_xd = (uint16_t *)malloc((size_t)n_outputs * (HQVM_HIDDEN_DIM / 32) * sizeof(uint16_t));
    if (!s_tail_qx || !s_tail_xd) return -1;
    for (t = 0; t < n_outputs; ++t) {
        hqvm_quantize_dyad_q8(s_tail_norm_d + (size_t)t * HQVM_HIDDEN_DIM, HQVM_HIDDEN_DIM,
            s_tail_qx + (size_t)t * HQVM_HIDDEN_DIM,
            s_tail_xd + (size_t)t * (HQVM_HIDDEN_DIM / 32));
    }
    s_tail_n_outputs = n_outputs;
    return 0;
}

int hqvm_native_tail_copy_norm(float *result_norm, size_t norm_stride_bytes) {
    int64_t t;
    if (!result_norm || !s_tail_norm_d || s_tail_n_outputs <= 0) return -1;
    for (t = 0; t < s_tail_n_outputs; ++t) {
        float *dst = (float *)((char *)result_norm + (size_t)t * norm_stride_bytes);
        const hqvm_dyad32_t *src = s_tail_norm_d + (size_t)t * HQVM_HIDDEN_DIM;
        for (int64_t i = 0; i < HQVM_HIDDEN_DIM; ++i) dst[i] = hqvm_dyad32_to_f32(src[i]);
    }
    return 0;
}

int hqvm_native_tail_project(float *result_output, size_t output_stride_bytes, int64_t row0, int64_t row1, int32_t worker_id) {
    int64_t t;
    if (!result_output || !s_tail_qx || !s_tail_xd || s_tail_n_outputs <= 0 || !hqvm_tail_sidecar_ready()) return -1;
    for (t = 0; t < s_tail_n_outputs; ++t) {
        float *out = (float *)((char *)result_output + (size_t)t * output_stride_bytes);
        const int8_t *qx = s_tail_qx + (size_t)t * HQVM_HIDDEN_DIM;
        const uint16_t *xd = s_tail_xd + (size_t)t * (HQVM_HIDDEN_DIM / 32);
        if (hqvm_forward_q1_0_q8(&s_tail_side, s_tail_W.output.q1_data,
                s_tail_W.output.n_rows, s_tail_W.output.n_cols, s_tail_W.output.row_stride_bytes,
                qx, xd, out, row0, row1) != 0) return -1;
        if (s_emission_rows && worker_id >= 0 && worker_id < s_emission_workers) {
            int32_t best_row;
            hqvm_i128 best_score;
            if (hqvm_argmax_q1_0_q8_exact(&s_tail_side, s_tail_W.output.q1_data,
                    s_tail_W.output.n_rows, s_tail_W.output.n_cols, s_tail_W.output.row_stride_bytes,
                    qx, xd, row0, row1, &best_row, &best_score) != 0) return -1;
            s_emission_rows[(size_t)worker_id * s_tail_n_outputs + t] = best_row;
            s_emission_scores[(size_t)worker_id * s_tail_n_outputs + t] = best_score;
        }
    }
    return 0;
}

int64_t hqvm_native_tail_vocab(void) { return s_tail_W.output.n_rows; }
int64_t hqvm_native_tail_n_outputs(void) { return s_tail_n_outputs; }
void hqvm_native_tail_clear(void) { s_tail_n_outputs = 0; }

uint64_t hqvm_entry_q8_rows(void) { return s_entry_q8_rows; }
uint64_t hqvm_entry_q8_scale_blocks(void) { return s_entry_q8_scale_blocks; }
uint64_t hqvm_dyad_residual_rows(void) { return s_dyad_residual_rows; }
uint64_t hqvm_dyad_residual_coordinates(void) { return s_dyad_residual_coordinates; }
uint64_t hqvm_float_residual_storage_calls(void) { return s_float_residual_storage_calls; }
uint64_t hqvm_float_residual_adapter_calls(void) { return s_float_residual_adapter_calls; }
uint64_t hqvm_dyad_scratch_rows(void) { return s_dyad_scratch_rows; }
uint64_t hqvm_dyad_scratch_bytes(void) { return s_dyad_scratch_bytes; }

int hqvm_entry_q8_encode_decode(hqvm_dyad32_t *x, int32_t T, uint8_t *pi_u6, uint8_t *pi_v6) {
    const int64_t n = HQVM_HIDDEN_DIM;
    const int64_t nb = n / HQVM_Q8_BLOCK;
    int8_t *q = NULL;
    uint16_t *d = NULL;
    float *row = NULL;
    int32_t t;
    if (!x || T <= 0 || !pi_u6 || !pi_v6) return -1;

    *pi_u6 = 0;
    *pi_v6 = 0;
    for (int i = 0; i < 6; ++i) {
        if (hqvm_dyad32_sign(x[i])) *pi_u6 |= (uint8_t)(1u << i);
        if (hqvm_dyad32_sign(x[6 + i])) *pi_v6 |= (uint8_t)(1u << i);
    }

    q = (int8_t *)malloc((size_t)T * (size_t)n);
    d = (uint16_t *)malloc((size_t)T * (size_t)nb * sizeof(uint16_t));
    row = (float *)malloc((size_t)n * sizeof(float));
    if (!q || !d || !row) {
        free(q);
        free(d);
        free(row);
        return -2;
    }

    for (t = 0; t < T; ++t) {
        hqvm_dyad32_t *dyad_row = x + (size_t)t * (size_t)n;
        int8_t *qr = q + (size_t)t * (size_t)n;
        uint16_t *dr = d + (size_t)t * (size_t)nb;
        for (int64_t k = 0; k < n; ++k) row[k] = hqvm_dyad32_to_f32(dyad_row[k]);
        s_float_residual_adapter_calls++; /* entry_q8_quantize_adapter */
        hqvm_quantize_x_q8(row, n, qr, dr);
        for (int64_t b = 0; b < nb; ++b) {
            const float scale = hqvm_f16_to_f32(dr[b]);
            for (int64_t j = 0; j < HQVM_Q8_BLOCK; ++j) {
                const int64_t k = b * HQVM_Q8_BLOCK + j;
                dyad_row[k] = hqvm_dyad32_from_f32((float)qr[k] * scale);
            }
        }
    }

    s_entry_q8_rows += (uint64_t)T;
    s_entry_q8_scale_blocks += (uint64_t)T * (uint64_t)nb;
    free(q);
    free(d);
    free(row);
    return 0;
}

void hqvm_reset_request(const float *embd_row0, int64_t n_embd) {
    if (embd_row0 && n_embd >= 12) {
        hqvm_pi_summary_sign12_from_embd(embd_row0, n_embd);
    }
    hqvm_cgm_lift_reset_sequence();
}

void hqvm_reset_request_bits(uint8_t pi_u6, uint8_t pi_v6) {
    hqvm_pi_summary_sign12_from_bits(pi_u6, pi_v6);
    hqvm_cgm_lift_reset_sequence();
}

static uint8_t popc6(uint8_t x) {
#if defined(_MSC_VER)
    return (uint8_t)__popcnt((unsigned)(x & 63));
#else
    return (uint8_t)__builtin_popcount((unsigned)(x & 63));
#endif
}

/* H3: Delta-ruler via codec. Dyad in/out — not a float dual-malloc bridge. */
static void norm_apply_ruler_dyad(
    const hqvm_dyad32_t *x_in, hqvm_dyad32_t *x_out, int64_t n,
    const float *g, float g0)
{
    if (!x_in || !x_out || n <= 0) return;
    (void)hqvm_norm_ruler_dyad(x_in, x_out, n, g, g0 > 0.0f ? g0 : 1.0f);
}

static int matmul_q1_dyad_to_dyad(
    const hqvm_q1_weight_t *W, const hqvm_dyad32_t *x, hqvm_dyad32_t *y)
{
    int rc;
    if (!W || !y || W->n_rows <= 0) return -1;
    hqvm_matmul_q1_inc();
    rc = hqvm_matmul_dyad(W, x, y);
    return rc;
}

/* Per-head RMSNorm (Qwen3 attn_q_norm / attn_k_norm), in-place via codec. */
static void rms_norm_heads_dyad(
    hqvm_dyad32_t *X, int32_t n_heads, const float *g, float g0)
{
    hqvm_dyad32_t tmp[HQVM_HEAD_DIM];
    int h;
    if (!X || n_heads <= 0) return;
    for (h = 0; h < n_heads; ++h) {
        hqvm_dyad32_t *drow = X + h * HQVM_HEAD_DIM;
        if (hqvm_norm_ruler_dyad(drow, tmp, HQVM_HEAD_DIM, g, g0 > 0.0f ? g0 : 1.0f) != 0)
            return;
        memcpy(drow, tmp, sizeof(tmp));
    }
}

/* Dyadic residual add with shell gain (medium residual law). Not a float adapter. */
static int residual_add_dyad(
    const hqvm_dyad32_t *x, const hqvm_dyad32_t *y, hqvm_dyad32_t *out,
    int64_t n, uint8_t u6, uint8_t v6)
{
    static int s_id = -1;
    int64_t i;
    int32_t num = 1, den = 1;
    hqvm_dyad32_t gain;
    if (s_id < 0) {
        const char *e = getenv("GYRO_NATIVE_RESIDUAL");
        s_id = (e && e[0] == '0') ? 1 : 0;
    }
    if (!s_id) {
        hqvm_dyad32_t dn, dd;
        const uint8_t Nc = popc6((uint8_t)(u6 ^ v6));
        hqvm_residual_gain_q16(Nc, &num, &den);
        if (hqvm_dyad32_from_i32(num, &dn) != 0 ||
            hqvm_dyad32_from_i32(den, &dd) != 0 ||
            hqvm_dyad32_div(dn, dd, &gain) != 0) return -1;
    } else {
        if (hqvm_dyad32_from_i32(1, &gain) != 0) return -1;
    }
    for (i = 0; i < n; ++i) {
        hqvm_dyad32_t product;
        if (!hqvm_dyad32_is_finite(x[i]) || !hqvm_dyad32_is_finite(y[i]) ||
            hqvm_dyad32_mul(y[i], gain, &product) != 0 ||
            hqvm_dyad32_add(x[i], product, &out[i]) != 0) return -1;
    }
    s_dyad_residual_rows++;
    s_dyad_residual_coordinates += (uint64_t)n;
    return 0;
}

/* H4: RoPE via codec (YaRN ticks on dyad heads). */
static void rope_qk_heads_dyad(
    hqvm_dyad32_t *Q, hqvm_dyad32_t *K, int32_t n_heads, int32_t gqa_ratio, int32_t token_pos)
{
    if (!Q || !K || n_heads <= 0 || gqa_ratio <= 0) return;
    (void)hqvm_rope_qk_dyad(Q, K, n_heads, gqa_ratio, token_pos);
}

/* P0 residual capture (GYRO_RESIDUAL_CAPTURE=path.bin). */
typedef struct {
    FILE *fp;
    int enabled;
    int max_layers;
    int max_tokens;
    uint64_t n_written;
} rcap_state_t;

static rcap_state_t s_rcap = { NULL, 0, -1, 8, 0 };

static void rcap_init(void) {
    static int s_inited = 0;
    const char *path;
    uint32_t ver;
    uint32_t hid;
    const char *e;
    if (s_inited) return;
    s_inited = 1;
    path = getenv("GYRO_RESIDUAL_CAPTURE");
    if (!path || !path[0]) return;
    s_rcap.fp = fopen(path, "wb");
    if (!s_rcap.fp) return;
    fwrite("GRCP", 1, 4, s_rcap.fp);
    ver = 1u;
    hid = (uint32_t)HQVM_HIDDEN_DIM;
    fwrite(&ver, sizeof(ver), 1, s_rcap.fp);
    fwrite(&hid, sizeof(hid), 1, s_rcap.fp);
    s_rcap.enabled = 1;
    s_rcap.max_layers = -1;
    e = getenv("GYRO_RESIDUAL_CAPTURE_MAX_LAYERS");
    if (e && e[0]) s_rcap.max_layers = atoi(e);
    s_rcap.max_tokens = 8;
    e = getenv("GYRO_RESIDUAL_CAPTURE_MAX_TOKENS");
    if (e && e[0]) s_rcap.max_tokens = atoi(e);
}

static void rcap_dyad(uint8_t phase, int32_t t, int32_t ell, const hqvm_dyad32_t *d, int64_t n) {
    int32_t tok;
    int32_t layer;
    uint8_t ph;
    float row[HQVM_HIDDEN_DIM];
    int64_t i;
    if (!d || n != HQVM_HIDDEN_DIM) return;
    rcap_init();
    if (!s_rcap.enabled || !s_rcap.fp) return;
    if (s_rcap.max_tokens >= 0 && t >= s_rcap.max_tokens) return;
    if (s_rcap.max_layers >= 0 && ell >= s_rcap.max_layers) return;
    for (i = 0; i < HQVM_HIDDEN_DIM; ++i) row[i] = hqvm_dyad32_to_f32(d[i]);
    tok = t;
    layer = ell;
    ph = phase;
    fwrite(&tok, sizeof(tok), 1, s_rcap.fp);
    fwrite(&layer, sizeof(layer), 1, s_rcap.fp);
    fwrite(&ph, sizeof(ph), 1, s_rcap.fp);
    fwrite(row, sizeof(float), (size_t)HQVM_HIDDEN_DIM, s_rcap.fp);
    fflush(s_rcap.fp);
    s_rcap.n_written++;
}

static void numstat_row(const char *tag, int32_t t, int32_t ell, const float *x, int64_t n) {
    static int s_en = -1;
    static int s_n = 0;
    int64_t i;
    double s = 0.0, s2 = 0.0;
    float mn = 0.0f, mx = 0.0f;
    int nan = 0;
    if (s_en < 0) {
        const char *e = getenv("GYRO_NATIVE_NUMSTAT");
        s_en = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    if (!s_en || !x || n <= 0 || s_n >= 24) return;
    mn = mx = x[0];
    for (i = 0; i < n; ++i) {
        const float v = x[i];
        if (v != v) { nan++; continue; }
        s += (double)v;
        s2 += (double)v * (double)v;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    fprintf(stderr,
        "[hqvm-numstat] %s t=%d ell=%d mean=%.5g rms=%.5g min=%.5g max=%.5g nan=%d "
        "first=[%.7g,%.7g,%.7g] last=[%.7g,%.7g,%.7g]\n",
        tag, (int)t, (int)ell, s / (double)n, sqrt(s2 / (double)n), (double)mn, (double)mx, nan,
        (double)x[0], (double)x[n > 1 ? 1 : 0], (double)x[n > 2 ? 2 : 0],
        (double)x[n > 2 ? n - 3 : 0], (double)x[n > 1 ? n - 2 : 0], (double)x[n - 1]);
    fflush(stderr);
    s_n++;
}

/* Prefill/decode block: one L-step of native forward (owners, not hosting). */
int hqvm_block_forward(
    const hqvm_dyad32_t *x_in,
    int32_t token_pos,
    int32_t layer_idx,
    uint64_t depth,
    const hqvm_block_weights_t *W,
    uint8_t *u6,
    uint8_t *v6,
    hqvm_block_kv_t *KV,
    hqvm_dyad32_t *x_out)
{
    hqvm_dyad32_t *x_norm_d = NULL;
    hqvm_dyad32_t *x_n2_d = NULL;
    hqvm_dyad32_t *Q = NULL, *K = NULL, *V = NULL;
    float *K_fb = NULL, *V_fb = NULL;
    hqvm_dyad32_t *attn = NULL;
    hqvm_dyad32_t *attn_o = NULL;
    hqvm_dyad32_t *x_mid = NULL;
    hqvm_dyad32_t *gate = NULL, *up = NULL, *ffn_h = NULL;
    hqvm_dyad32_t *ffn_o = NULL;
    hqvm_dyad32_t *weights_d = NULL;
    hqvm_dyad32_t *lift_weights_d = NULL;
    uint8_t fam;
    uint8_t Nc;
    int32_t head, n_heads = HQVM_N_HEAD;
    int64_t kv_len;
    int64_t kv_slots = 1;
    int rc = 0;

    if (!x_in || !W || !u6 || !v6 || !x_out) return -1;
    if (layer_idx < 0 || layer_idx >= HQVM_N_LAYER) return -1;

    if (KV && KV->n_ctx > 0) {
        kv_slots = KV->n_ctx;
    }
    if (block_dyad_scratch_ensure(kv_slots) != 0) {
        return -2;
    }
    x_norm_d = s_blk_dyad + BLK_OFF_X_NORM;
    x_n2_d   = s_blk_dyad + BLK_OFF_X_N2;
    Q        = s_blk_dyad + BLK_OFF_Q;
    K        = s_blk_dyad + BLK_OFF_K;
    V        = s_blk_dyad + BLK_OFF_V;
    attn     = s_blk_dyad + BLK_OFF_ATTN;
    attn_o   = s_blk_dyad + BLK_OFF_ATTN_O;
    x_mid    = s_blk_dyad + BLK_OFF_X_MID;
    gate     = s_blk_dyad + BLK_OFF_GATE;
    up       = s_blk_dyad + BLK_OFF_UP;
    ffn_h    = s_blk_dyad + BLK_OFF_FFN_H;
    ffn_o    = s_blk_dyad + BLK_OFF_FFN_O;

    {
        static int s_id = -1;
        static int s_max = -2; /* -2 unset, -1 means all layers */
        if (s_id < 0) {
            const char *e = getenv("GYRO_NATIVE_IDENTITY");
            s_id = (e && e[0] && e[0] != '0') ? 1 : 0;
            if (s_id) {
                fprintf(stderr, "[hqvm-native] IDENTITY block: x_out = x_in\n");
                fflush(stderr);
            }
        }
        if (s_max == -2) {
            const char *e = getenv("GYRO_NATIVE_MAX_LAYER");
            s_max = (e && e[0]) ? atoi(e) : -1;
            if (s_max >= 0) {
                fprintf(stderr, "[hqvm-native] MAX_LAYER=%d (identity above)\n", s_max);
                fflush(stderr);
            }
        }
        if (s_id || (s_max >= 0 && layer_idx > s_max)) {
            memcpy(x_out, x_in, HQVM_HIDDEN_DIM * sizeof(hqvm_dyad32_t));
            hqvm_native_block_inc();
            return 0;
        }
    }

    {
        const uint64_t expect =
            (uint64_t)token_pos * (uint64_t)HQVM_N_LAYER + (uint64_t)layer_idx;
        if (depth != expect) depth = expect;
    }
    fam = (uint8_t)(depth & 3ull);

    rcap_dyad(0, token_pos, layer_idx, x_in, HQVM_HIDDEN_DIM);

    /* Dyad: norms + Q/K/V + attn + attn_o + gate/up/ffn_h/ffn_o. Norm+RoPE+FFN-gate hosted (H3/H4/H7). */
    s_dyad_scratch_rows += 8;
    s_dyad_scratch_bytes += (uint64_t)(8 * HQVM_HIDDEN_DIM * sizeof(hqvm_dyad32_t));
    s_dyad_scratch_bytes += (uint64_t)(3 * HQVM_FFN_DIM * sizeof(hqvm_dyad32_t));

    /* --- Attention sublayer --- */
    norm_apply_ruler_dyad(x_in, x_norm_d, HQVM_HIDDEN_DIM, W->attn_norm_g, W->attn_norm_g0);

    if (matmul_q1_dyad_to_dyad(&W->attn_q, x_norm_d, Q) != 0
        || matmul_q1_dyad_to_dyad(&W->attn_k, x_norm_d, K) != 0
        || matmul_q1_dyad_to_dyad(&W->attn_v, x_norm_d, V) != 0) {
        rc = -3;
        goto done;
    }

    {
        const int32_t n_kv_heads = (int32_t)(KV ? KV->n_kv_heads : (int64_t)n_heads);
        const int32_t gqa_ratio = n_heads / (n_kv_heads > 0 ? n_kv_heads : n_heads);
        /* Qwen3: RMSNorm each Q/K head before RoPE. */
        if (!W->attn_q_norm_g || !W->attn_k_norm_g) {
            static int s_qn = 0;
            if (s_qn < 3) {
                fprintf(stderr,
                    "[hqvm-native] missing Q/K norm layer=%d q=%p k=%p\n",
                    (int)layer_idx, (void *)W->attn_q_norm_g, (void *)W->attn_k_norm_g);
                fflush(stderr);
                s_qn++;
            }
        }
        {
            static int s_skip_qn = -1;
            if (s_skip_qn < 0) {
                const char *e = getenv("GYRO_NATIVE_SKIP_QKNORM");
                s_skip_qn = (e && e[0] && e[0] != '0') ? 1 : 0;
                if (s_skip_qn) {
                    fprintf(stderr, "[hqvm-native] SKIP Q/K RMSNorm\n");
                    fflush(stderr);
                }
            }
            if (!s_skip_qn) {
                rms_norm_heads_dyad(Q, n_heads, W->attn_q_norm_g, W->attn_q_norm_g0);
                rms_norm_heads_dyad(K, n_kv_heads > 0 ? n_kv_heads : n_heads,
                               W->attn_k_norm_g, W->attn_k_norm_g0);
            }
        }
        rope_qk_heads_dyad(Q, K, n_heads, gqa_ratio, token_pos);
    }

    {
        const int32_t n_kv_heads = (int32_t)(KV ? KV->n_kv_heads : (int64_t)n_heads);
        const int use_f32_kv = (KV && KV->use_f32 && KV->k_f32 && KV->v_f32);
        const int64_t kv_elems =
            (int64_t)(n_kv_heads > 0 ? n_kv_heads : n_heads) * HQVM_HEAD_DIM;

        if (use_f32_kv) {
            int64_t ii;
            K_fb = (float *)malloc((size_t)kv_elems * sizeof(float));
            V_fb = (float *)malloc((size_t)kv_elems * sizeof(float));
            if (!K_fb || !V_fb) { rc = -2; goto done; }
            for (ii = 0; ii < kv_elems; ++ii) {
                K_fb[ii] = hqvm_dyad32_to_f32(K[ii]);
                V_fb[ii] = hqvm_dyad32_to_f32(V[ii]);
            }
        }
    }

    kv_len = 1;
    if (KV && KV->n_ctx > 0 && KV->kv_pos >= 0 && KV->kv_pos < KV->n_ctx &&
        ((KV->use_f32 && KV->k_f32 && KV->v_f32) || (KV->k_q8 && KV->v_q8))) {
        const int64_t pos = KV->kv_pos;
        const size_t layer_kv_bytes =
            (size_t)KV->n_ctx * (size_t)KV->k_row_stride;
        uint8_t *chi_layer = KV->k_chi6
            ? KV->k_chi6 + (size_t)layer_idx * (size_t)KV->n_ctx * (size_t)KV->n_kv_heads
            : NULL;
        if (KV->use_f32) {
            float *k_layer = KV->k_f32 + ((size_t)layer_idx * layer_kv_bytes) / sizeof(float);
            float *v_layer = KV->v_f32 + ((size_t)layer_idx * layer_kv_bytes) / sizeof(float);
            const size_t floats_per_head = (size_t)HQVM_HEAD_DIM;
            for (head = 0; head < (int)KV->n_kv_heads; ++head) {
                float *krow = k_layer + (size_t)pos * (size_t)(KV->k_row_stride / sizeof(float))
                    + (size_t)head * floats_per_head;
                float *vrow = v_layer + (size_t)pos * (size_t)(KV->v_row_stride / sizeof(float))
                    + (size_t)head * floats_per_head;
                memcpy(krow, K_fb + head * HQVM_HEAD_DIM, HQVM_HEAD_DIM * sizeof(float));
                memcpy(vrow, V_fb + head * HQVM_HEAD_DIM, HQVM_HEAD_DIM * sizeof(float));
                if (chi_layer) {
                    chi_layer[(size_t)pos * (size_t)KV->n_kv_heads + (size_t)head] =
                        hqvm_k_chi6_from_dyad_head(K + head * HQVM_HEAD_DIM);
                }
            }
        } else {
            char *k_layer = (char *)KV->k_q8 + (size_t)layer_idx * layer_kv_bytes;
            char *v_layer = (char *)KV->v_q8 + (size_t)layer_idx * layer_kv_bytes;
            for (head = 0; head < (int)KV->n_kv_heads; ++head) {
                char *krow = k_layer + (size_t)pos * KV->k_row_stride
                    + (size_t)head * (KV->k_row_stride / (size_t)KV->n_kv_heads);
                char *vrow = v_layer + (size_t)pos * KV->v_row_stride
                    + (size_t)head * (KV->v_row_stride / (size_t)KV->n_kv_heads);
                hqvm_quantize_dyad_row_q8(K + head * HQVM_HEAD_DIM, HQVM_HEAD_DIM, krow);
                hqvm_quantize_dyad_row_q8(V + head * HQVM_HEAD_DIM, HQVM_HEAD_DIM, vrow);
                if (chi_layer) {
                    chi_layer[(size_t)pos * (size_t)KV->n_kv_heads + (size_t)head] =
                        hqvm_k_chi6_from_dyad_head(K + head * HQVM_HEAD_DIM);
                }
            }
        }
        hqvm_kv_write_inc((uint64_t)KV->n_kv_heads);
        kv_len = pos + 1;
    } else {
        hqvm_kv_null_write_inc();
    }

    memset(attn, 0, HQVM_HIDDEN_DIM * sizeof(hqvm_dyad32_t));
    Nc = popc6((uint8_t)((*u6) ^ (*v6)));
    kv_len = (kv_len > 0) ? kv_len : 1;
    weights_d = s_blk_dyad + BLK_OFF_WT(kv_len);
    lift_weights_d = s_blk_dyad + BLK_OFF_LWT(kv_len);

    {
        const int32_t n_kv_heads = (int32_t)(KV ? KV->n_kv_heads : (int64_t)n_heads);
        const int32_t gqa_ratio = n_heads / (n_kv_heads > 0 ? n_kv_heads : n_heads);
        const size_t layer_kv_bytes = (KV && KV->n_ctx > 0)
            ? (size_t)KV->n_ctx * (size_t)KV->k_row_stride : 0;
        const int use_f32 = (KV && KV->use_f32 && KV->k_f32 && KV->v_f32);
        const char *k_layer_q8 = (!use_f32 && KV && KV->k_q8)
            ? (const char *)KV->k_q8 + (size_t)layer_idx * layer_kv_bytes : NULL;
        const char *v_layer_q8 = (!use_f32 && KV && KV->v_q8)
            ? (const char *)KV->v_q8 + (size_t)layer_idx * layer_kv_bytes : NULL;
        const float *k_layer_f32 = (use_f32)
            ? KV->k_f32 + ((size_t)layer_idx * layer_kv_bytes) / sizeof(float) : NULL;
        const float *v_layer_f32 = (use_f32)
            ? KV->v_f32 + ((size_t)layer_idx * layer_kv_bytes) / sizeof(float) : NULL;
        const uint8_t *chi_layer = (KV && KV->k_chi6)
            ? KV->k_chi6 + (size_t)layer_idx * (size_t)KV->n_ctx * (size_t)KV->n_kv_heads
            : NULL;
        const int attn_lvl = hqvm_attn_level();
        const float attn_scale = 1.0f / sqrtf((float)HQVM_HEAD_DIM);

        if (!k_layer_q8 && !k_layer_f32) hqvm_kv_null_read_inc();

        for (head = 0; head < n_heads; ++head) {
            const int32_t kv_head = head / (gqa_ratio > 0 ? gqa_ratio : 1);
            const hqvm_dyad32_t *qh_d = Q + head * HQVM_HEAD_DIM;
            hqvm_dyad32_t *ah_d = attn + head * HQVM_HEAD_DIM;
            const size_t k_per_head_q8 = (KV && KV->n_kv_heads > 0 && !use_f32)
                ? (size_t)(KV->k_row_stride / KV->n_kv_heads) : 0;
            const size_t floats_per_tok = (KV && use_f32)
                ? (size_t)(KV->k_row_stride / sizeof(float)) : 0;
            const float *k_fallback = K_fb + kv_head * HQVM_HEAD_DIM;

            /* H5: attn owns stock vs native score face. */
            if (hqvm_attn_head_scores_dyad(weights_d, qh_d,
                    k_layer_q8, k_layer_f32, k_fallback,
                    (size_t)(KV ? KV->k_row_stride : 0), floats_per_tok, k_per_head_q8,
                    chi_layer, KV ? KV->n_kv_heads : (int64_t)n_heads, kv_head,
                    kv_len, Nc, HQVM_ATTN_SHELL_TOPK, attn_lvl, attn_scale) != 0) {
                rc = -2; goto done;
            }
            hqvm_score_dot_head_inc();
            if (k_layer_q8 || k_layer_f32) hqvm_kv_read_inc((uint64_t)kv_len);

            if (head == 0) {
                memcpy(lift_weights_d, weights_d, (size_t)kv_len * sizeof(hqvm_dyad32_t));
            }

            /* H6: single attn face (native Q8 or stock float). */
            {
                const void *v_base = NULL;
                size_t v_stride = 0;
                int v_is_q8 = 0;
                if (v_layer_f32) {
                    v_base = v_layer_f32 + (size_t)kv_head * (size_t)HQVM_HEAD_DIM;
                    v_stride = floats_per_tok * sizeof(float);
                    v_is_q8 = 0;
                } else if (v_layer_q8) {
                    const size_t v_per_head =
                        (size_t)(KV->v_row_stride / KV->n_kv_heads);
                    v_base = v_layer_q8 + (size_t)kv_head * v_per_head;
                    v_stride = (size_t)KV->v_row_stride;
                    v_is_q8 = 1;
                } else {
                    v_base = V_fb + kv_head * HQVM_HEAD_DIM;
                    v_stride = (size_t)HQVM_HEAD_DIM * sizeof(float);
                    v_is_q8 = 0;
                }
                if (hqvm_v_reduce_dyad(
                        ah_d, HQVM_HEAD_DIM, weights_d, kv_len,
                        v_base, v_stride, v_is_q8) != 0) {
                    rc = -2; goto done;
                }
                hqvm_vkq_reduce_head_inc();
            }
        }
    }

    /* Carrier step from attn argmax transport (head 0, matching cgm_lift). */
    {
        uint64_t qsigns = 0;
        uint8_t chi_q, chi_k = 0, q6, byte;
        int d, argmax = 0;
        float best = -INFINITY;
        int64_t j;
        const int32_t lift_kv_head = 0;
        {
            uint64_t _qs = 0;
            for (d = 0; d < 64; ++d) if (!hqvm_dyad32_sign(Q[d])) _qs |= (1ull << d);
            qsigns = _qs;
        }
        chi_q = gyroscopic_chirality_from_signs64(qsigns);
        for (j = 0; j < kv_len; ++j) {
            const float w = hqvm_dyad32_to_f32(lift_weights_d[j]);
            if (w > best) { best = w; argmax = (int)j; }
        }
        if (KV && KV->k_chi6 && argmax >= 0 && argmax < kv_len) {
            const uint8_t *chi_layer = KV->k_chi6
                + (size_t)layer_idx * (size_t)KV->n_ctx * (size_t)KV->n_kv_heads;
            chi_k = chi_layer[(size_t)argmax * (size_t)KV->n_kv_heads
                + (size_t)lift_kv_head];
        }
        q6 = (uint8_t)((chi_q ^ chi_k) & 63u);
        byte = hqvm_byte_of_q6_fam(q6, fam);
        hqvm_step_uv6(*u6, *v6, byte, u6, v6);
        hqvm_lift_step_inc();
    }

    if (matmul_q1_dyad_to_dyad(&W->attn_o, attn, attn_o) != 0) {
        rc = -3; goto done;
    }
    {
        static int s_skip_attn = -1;
        if (s_skip_attn < 0) {
            const char *e = getenv("GYRO_NATIVE_SKIP_ATTN");
            s_skip_attn = (e && e[0] && e[0] != '0') ? 1 : 0;
        }
        if (s_skip_attn) memset(attn_o, 0, HQVM_HIDDEN_DIM * sizeof(hqvm_dyad32_t));
    }
    if (residual_add_dyad(x_in, attn_o, x_mid, HQVM_HIDDEN_DIM, *u6, *v6) != 0) {
        rc = -5; goto done;
    }
    rcap_dyad(2, token_pos, layer_idx, x_mid, HQVM_HIDDEN_DIM);

    /* --- FFN sublayer --- */
    norm_apply_ruler_dyad(x_mid, x_n2_d, HQVM_HIDDEN_DIM, W->ffn_norm_g, W->ffn_norm_g0);
    if (matmul_q1_dyad_to_dyad(&W->ffn_gate, x_n2_d, gate) != 0
        || matmul_q1_dyad_to_dyad(&W->ffn_up, x_n2_d, up) != 0) {
        rc = -3;
        goto done;
    }
    /* H7: codec FFN on dyad lanes — product stock SwiGLU unless GYRO_FFN_NATIVE=1. */
    Nc = popc6((uint8_t)((*u6) ^ (*v6)));
    if (hqvm_ffn_gate_dyad(gate, up, ffn_h, HQVM_FFN_DIM, fam, Nc) != 0) {
        rc = -2;
        goto done;
    }
    if (matmul_q1_dyad_to_dyad(&W->ffn_down, ffn_h, ffn_o) != 0) {
        rc = -3; goto done;
    }
    {
        static int s_skip_ffn = -1;
        if (s_skip_ffn < 0) {
            const char *e = getenv("GYRO_NATIVE_SKIP_FFN");
            s_skip_ffn = (e && e[0] && e[0] != '0') ? 1 : 0;
        }
        if (s_skip_ffn) memset(ffn_o, 0, HQVM_HIDDEN_DIM * sizeof(hqvm_dyad32_t));
    }

    {
        uint64_t gsigns = 0;
        uint8_t chi_g, byte;
        int d;
        for (d = 0; d < 64; ++d) if (!hqvm_dyad32_sign(gate[d])) gsigns |= (1ull << d);
        chi_g = gyroscopic_chirality_from_signs64(gsigns);
        byte = hqvm_byte_of_q6_fam(chi_g, fam);
        hqvm_step_uv6(*u6, *v6, byte, u6, v6);
        hqvm_lift_step_inc();
    }
    if (residual_add_dyad(x_mid, ffn_o, x_out, HQVM_HIDDEN_DIM, *u6, *v6) != 0) {
        rc = -5; goto done;
    }
    rcap_dyad(3, token_pos, layer_idx, x_out, HQVM_HIDDEN_DIM);

    if ((token_pos == 0 && (layer_idx == 0 || layer_idx == HQVM_N_LAYER - 1)) ||
        (token_pos == 1 && layer_idx == 0)) {
        /* numstat is diagnostic; keep but adapt to dyad scratch (use tmp floats) */
        {
            float tmpf[HQVM_HIDDEN_DIM];
            float tmp_ffn[HQVM_FFN_DIM];
            int64_t ii;
            for (ii = 0; ii < HQVM_HIDDEN_DIM; ++ii) tmpf[ii] = hqvm_dyad32_to_f32(x_in[ii]);
            numstat_row("x_in", token_pos, layer_idx, tmpf, HQVM_HIDDEN_DIM);
            for (ii = 0; ii < HQVM_HIDDEN_DIM; ++ii) tmpf[ii] = hqvm_dyad32_to_f32(x_out[ii]);
            numstat_row("x_out", token_pos, layer_idx, tmpf, HQVM_HIDDEN_DIM);
            for (ii = 0; ii < HQVM_HIDDEN_DIM; ++ii) tmpf[ii] = hqvm_dyad32_to_f32(Q[ii]);
            numstat_row("Q", token_pos, layer_idx, tmpf, HQVM_HIDDEN_DIM);
            for (ii = 0; ii < HQVM_KV_N_KV_HEAD * HQVM_HEAD_DIM; ++ii) tmpf[ii] = hqvm_dyad32_to_f32(V[ii]);
            numstat_row("V", token_pos, layer_idx, tmpf, HQVM_KV_N_KV_HEAD * HQVM_HEAD_DIM);
            for (ii = 0; ii < HQVM_HIDDEN_DIM; ++ii) tmpf[ii] = hqvm_dyad32_to_f32(attn[ii]);
            numstat_row("attn", token_pos, layer_idx, tmpf, HQVM_HIDDEN_DIM);
            for (ii = 0; ii < HQVM_HIDDEN_DIM; ++ii) tmpf[ii] = hqvm_dyad32_to_f32(attn_o[ii]);
            numstat_row("attn_o", token_pos, layer_idx, tmpf, HQVM_HIDDEN_DIM);
            for (ii = 0; ii < HQVM_FFN_DIM; ++ii) tmp_ffn[ii] = hqvm_dyad32_to_f32(gate[ii]);
            numstat_row("gate", token_pos, layer_idx, tmp_ffn, HQVM_FFN_DIM);
            for (ii = 0; ii < HQVM_FFN_DIM; ++ii) tmp_ffn[ii] = hqvm_dyad32_to_f32(up[ii]);
            numstat_row("up", token_pos, layer_idx, tmp_ffn, HQVM_FFN_DIM);
            for (ii = 0; ii < HQVM_FFN_DIM; ++ii) tmp_ffn[ii] = hqvm_dyad32_to_f32(ffn_h[ii]);
            numstat_row("ffn_h", token_pos, layer_idx, tmp_ffn, HQVM_FFN_DIM);
            for (ii = 0; ii < HQVM_HIDDEN_DIM; ++ii) tmpf[ii] = hqvm_dyad32_to_f32(ffn_o[ii]);
            numstat_row("ffn_o", token_pos, layer_idx, tmpf, HQVM_HIDDEN_DIM);
        }
        if (layer_idx == 0) {
            fprintf(stderr,
                "[hqvm-numstat] dims q=%lldx%lld k=%lldx%lld gate=%lldx%lld down=%lldx%lld\n",
                (long long)W->attn_q.n_rows, (long long)W->attn_q.n_cols,
                (long long)W->attn_k.n_rows, (long long)W->attn_k.n_cols,
                (long long)W->ffn_gate.n_rows, (long long)W->ffn_gate.n_cols,
                (long long)W->ffn_down.n_rows, (long long)W->ffn_down.n_cols);
            fprintf(stderr,
                "[hqvm-numstat] norms attn_g0=%.5g ffn_g0=%.5g qn=%p k_n=%p "
                "attn_g[0]=%.5g attn_g[1]=%.5g qn_g[0]=%.5g\n",
                (double)W->attn_norm_g0, (double)W->ffn_norm_g0,
                (void *)W->attn_q_norm_g, (void *)W->attn_k_norm_g,
                W->attn_norm_g ? (double)W->attn_norm_g[0] : -1.0,
                W->attn_norm_g ? (double)W->attn_norm_g[1] : -1.0,
                W->attn_q_norm_g ? (double)W->attn_q_norm_g[0] : -1.0);
            fflush(stderr);
        }
    }

    hqvm_native_block_inc();
    rc = 0;

done:
    free(K_fb);
    free(V_fb);
    return rc;
}

int hqvm_layer_forward(
    const float *x_in,
    int32_t token_pos,
    int32_t layer_idx,
    int32_t n_tokens,
    const hqvm_layer_weights_t *W,
    uint8_t *u6,
    uint8_t *v6,
    hqvm_layer_kv_t *KV,
    float *x_out)
{
    const uint64_t depth =
        (uint64_t)token_pos * (uint64_t)HQVM_N_LAYER + (uint64_t)layer_idx;
    hqvm_dyad32_t *din = NULL, *dout = NULL;
    int rc;
    (void)n_tokens;
    din = (hqvm_dyad32_t *)malloc(HQVM_HIDDEN_DIM * sizeof(*din));
    dout = (hqvm_dyad32_t *)malloc(HQVM_HIDDEN_DIM * sizeof(*dout));
    if (!din || !dout) { free(din); free(dout); return -2; }
    for (int64_t i = 0; i < HQVM_HIDDEN_DIM; ++i) din[i] = hqvm_dyad32_from_f32(x_in[i]);
    s_float_residual_storage_calls++;
    rc = hqvm_block_forward(din, token_pos, layer_idx, depth, W, u6, v6, KV, dout);
    if (rc == 0) for (int64_t i = 0; i < HQVM_HIDDEN_DIM; ++i) x_out[i] = hqvm_dyad32_to_f32(dout[i]);
    free(din); free(dout);
    return rc;
}

int hqvm_forward_decode_step(
    hqvm_dyad32_t *x_row, int32_t t,
    uint8_t *u6, uint8_t *v6, hqvm_block_kv_t *KV)
{
    int32_t ell;
    hqvm_dyad32_t *scratch = NULL;
    hqvm_dyad32_t *xin, *xou;
    if (!x_row || !u6 || !v6 || t < 0) return -1;
    if (!hqvm_native_weights_ready()) return -2;
    scratch = (hqvm_dyad32_t *)malloc(HQVM_HIDDEN_DIM * sizeof(hqvm_dyad32_t));
    if (!scratch) return -3;
    xin = x_row;
    xou = scratch;
    if (KV) KV->kv_pos = t;
    for (ell = 0; ell < HQVM_N_LAYER; ++ell) {
        const uint64_t depth =
            (uint64_t)t * (uint64_t)HQVM_N_LAYER + (uint64_t)ell;
        if (hqvm_block_forward(xin, t, ell, depth, &s_W[ell], u6, v6, KV, xou) != 0) {
            free(scratch);
            return -4;
        }
        {
            hqvm_dyad32_t *sw = xin;
            xin = xou;
            xou = sw;
        }
    }
    if (xin != x_row) {
        memcpy(x_row, xin, HQVM_HIDDEN_DIM * sizeof(hqvm_dyad32_t));
    }
    free(scratch);
    return 0;
}

int hqvm_forward_prefill(
    hqvm_dyad32_t *x, int32_t T,
    const float *embd_row0, int64_t n_embd,
    uint8_t *u6, uint8_t *v6, hqvm_block_kv_t *KV)
{
    int32_t ell, t;
    hqvm_dyad32_t *scratch = NULL;
    if (!x || !u6 || !v6 || T <= 0) return -1;
    if (!hqvm_native_weights_ready()) return -2;
    scratch = (hqvm_dyad32_t *)malloc((size_t)T * HQVM_HIDDEN_DIM * sizeof(hqvm_dyad32_t));
    if (!scratch) return -3;
    if (embd_row0) {
        hqvm_reset_request(embd_row0, n_embd);
    }
    hqvm_cgm_lift_get_uv6(u6, v6);

    /* Prefill must be layer-major. Each layer computes K/V for all prompt
     * tokens before any later layer consumes the resulting residual stream. */
    for (ell = 0; ell < HQVM_N_LAYER; ++ell) {
        for (t = 0; t < T; ++t) {
            const uint64_t depth = (uint64_t)t * (uint64_t)HQVM_N_LAYER + (uint64_t)ell;
            if (KV) KV->kv_pos = t;
            if (hqvm_block_forward(
                    x + (size_t)t * HQVM_HIDDEN_DIM, t, ell, depth,
                    &s_W[ell], u6, v6, KV,
                    scratch + (size_t)t * HQVM_HIDDEN_DIM) != 0) {
                free(scratch);
                return -4;
            }
        }
        memcpy(x, scratch, (size_t)T * HQVM_HIDDEN_DIM * sizeof(hqvm_dyad32_t));
    }
    if (KV) KV->kv_pos = T - 1;
    free(scratch);
    return 0;
}

int hqvm_native_forward_ubatch(
    float *x, int32_t n_tokens, uint8_t *u6, uint8_t *v6, hqvm_layer_kv_t *KV)
{
    hqvm_dyad32_t *dx;
    int rc;
    if (!x || n_tokens <= 0) return -1;
    dx = (hqvm_dyad32_t *)malloc((size_t)n_tokens * HQVM_HIDDEN_DIM * sizeof(*dx));
    if (!dx) return -2;
    for (int64_t i = 0; i < (int64_t)n_tokens * HQVM_HIDDEN_DIM; ++i) dx[i] = hqvm_dyad32_from_f32(x[i]);
    s_float_residual_storage_calls++;
    rc = hqvm_forward_prefill(dx, n_tokens, x, HQVM_HIDDEN_DIM, u6, v6, KV);
    if (rc == 0) for (int64_t i = 0; i < (int64_t)n_tokens * HQVM_HIDDEN_DIM; ++i) x[i] = hqvm_dyad32_to_f32(dx[i]);
    free(dx);
    return rc;
}
