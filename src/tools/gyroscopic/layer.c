/*
 * Native gyroscopic forward: driver loops + hqvm_block_forward (L=36).
 */

#include "layer.h"

#include "attn.h"
#include "codec.h"
#include "constants.h"
#include "kernel.h"
#include "ledger.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_MSC_VER)
#  include <intrin.h>
#endif

static uint64_t s_stock_block = 0;
static uint64_t s_native_block = 0;
static int s_bypass = 0;
static hqvm_block_weights_t s_W[HQVM_N_LAYER];
static uint8_t s_W_ok[HQVM_N_LAYER];

static hqvm_block_kv_t s_kv;
static int s_kv_init = 0;
static int s_native_prefill_done = 0;

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
    s_kv_null_reads = 0;
    s_kv_null_writes = 0;
    s_kv_k_writes = 0;
    s_kv_v_writes = 0;
    s_kv_chi_writes = 0;
    s_kv_reads = 0;
    s_native_block_req0 = 0;
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
        "set_rows_calls=%llu stock_tail_calls=%llu "
        "kv_null_reads=%llu kv_null_writes=%llu "
        "K_writes=%llu V_writes=%llu chiK_writes=%llu kv_reads=%llu "
        "attn_level=%d ffn_level=%d pi_applied=%d\n",
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
        (unsigned long long)hqvm_stock_tail_calls(),
        (unsigned long long)hqvm_kv_null_reads(),
        (unsigned long long)hqvm_kv_null_writes(),
        (unsigned long long)hqvm_kv_k_writes(),
        (unsigned long long)hqvm_kv_v_writes(),
        (unsigned long long)hqvm_kv_chi_writes(),
        (unsigned long long)hqvm_kv_reads(),
        hqvm_attn_level(),
        hqvm_ffn_level(),
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

int hqvm_ffn_level(void) {
    static int s = -1;
    if (s < 0) {
        const char *e = getenv("GYRO_FFN_LEVEL");
        if (e && e[0]) s = atoi(e);
        else s = 2;
        if (s < 0) s = 0;
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

void hqvm_reset_request(const float *embd_row0, int64_t n_embd) {
    if (embd_row0 && n_embd >= 12) {
        hqvm_pi_stash_from_embd_row(embd_row0, n_embd);
    }
    hqvm_cgm_lift_reset_sequence();
}

static uint8_t popc6(uint8_t x) {
#if defined(_MSC_VER)
    return (uint8_t)__popcnt((unsigned)(x & 63));
#else
    return (uint8_t)__builtin_popcount((unsigned)(x & 63));
#endif
}

static void norm_apply_ruler(
    const float *x_in, float *x_out, int64_t n,
    const float *g, float g0)
{
    static int s_plain = -1;
    float scale;
    int64_t i;
    if (!x_in || !x_out || n <= 0) return;
    if (s_plain < 0) {
        const char *e = getenv("GYRO_NATIVE_NORM");
        /* plain = float RMS * weight, no Δ-ruler (debug). */
        s_plain = (e && strcmp(e, "plain") == 0) ? 1 : 0;
        if (s_plain) {
            fprintf(stderr, "[hqvm-norm] native using plain RMS (no Delta-ruler)\n");
            fflush(stderr);
        }
    }
    if (s_plain) {
        /* Bonsai/Qwen3 GGUF: attention.layer_norm_rms_epsilon = 1e-6. */
        scale = hqvm_rms_gain(x_in, n, 1e-6f);
        for (i = 0; i < n; ++i) {
            const float wi = g ? g[i] : 1.0f;
            x_out[i] = x_in[i] * scale * wi;
        }
        return;
    }
    scale = hqvm_rms_gain_fixed(x_in, n, 1e-6f);
    {
        /* RMS inverse gain and learned Norm weights are distinct moments.
         * The former is dimensionless around unity; the latter uses the
         * tensor-local geomean supplied by the loader. */
        const int16_t ns = hqvm_norm_encode_gain16(scale, 1.0f, (float)APERTURE_GAP);
        scale = hqvm_norm_decode_gain16(ns, 1.0f, (float)APERTURE_GAP);
    }
    for (i = 0; i < n; ++i) {
        float wi = 1.0f;
        if (g) {
            const float gi = fabsf(g[i]);
            const int16_t n16 = hqvm_norm_encode_gain16(
                gi > 0.0f ? gi : g0, g0, (float)APERTURE_GAP);
            wi = hqvm_norm_decode_gain16(n16, g0, (float)APERTURE_GAP);
            if (g[i] < 0.0f) wi = -wi;
        }
        x_out[i] = x_in[i] * scale * wi;
    }
}

static int matmul_q1(
    const hqvm_q1_weight_t *W, const float *x, float *y)
{
    static hqvm_sidecar s_side;
    static int s_side_ok = 0;
    const hqvm_sidecar *S = NULL;
    size_t stride;
    if (!W || !W->q1_data || !x || !y || W->n_rows <= 0 || W->n_cols <= 0) {
        return -1;
    }
    if (s_side_ok == 0) {
        const char *path = getenv("GYRO_LEDGER_PATH");
        if (path && path[0] && hqvm_sidecar_load(&s_side, path) == 0) {
            hqvm_sidecar_apply_env_allow(&s_side);
            s_side_ok = 1;
        } else {
            s_side_ok = -1;
        }
    }
    if (s_side_ok > 0) S = &s_side;
    if (!S) return -5; /* native MatMul requires HQVMLEDS sidecar */
    stride = W->row_stride_bytes;
    if (stride == 0) {
        const int64_t nblk = W->n_cols / 128;
        stride = (size_t)nblk * 20;
    }
    return hqvm_forward_q1_0_f32(
        S, W->q1_data, W->n_rows, W->n_cols, stride,
        x, y, /*row0*/ 0, /*row1*/ W->n_rows);
}

static float native_rope_freq_base(void) {
    static float s = -1.0f;
    if (s < 0.0f) {
        const char *e = getenv("GYRO_ROPE_FREQ_BASE");
        /* Bonsai-8B is Qwen3: GGUF qwen3.rope.freq_base = 1e6 (not Llama-2 1e4). */
        s = (e && e[0]) ? (float)atof(e) : 1000000.0f;
        if (s <= 0.0f) s = 1000000.0f;
        fprintf(stderr, "[hqvm-rope] native freq_base=%.0f\n", (double)s);
        fflush(stderr);
    }
    return s;
}

static float native_rope_freq_scale(void) {
    static float s = -1.0f;
    if (s < 0.0f) {
        const char *e = getenv("GYRO_ROPE_FREQ_SCALE");
        /* YaRN factor 4 → llama.cpp stores freq_scale = 1/factor = 0.25. */
        s = (e && e[0]) ? (float)atof(e) : 0.25f;
        if (s <= 0.0f) s = 0.25f;
        fprintf(stderr, "[hqvm-rope] native freq_scale=%g\n", (double)s);
        fflush(stderr);
    }
    return s;
}

/* Per-head RMSNorm (Qwen3 attn_q_norm / attn_k_norm), in-place. */
static void rms_norm_heads(
    float *X, int32_t n_heads, const float *g, float g0)
{
    float tmp[HQVM_HEAD_DIM];
    int h;
    if (!X || n_heads <= 0) return;
    for (h = 0; h < n_heads; ++h) {
        float *row = X + h * HQVM_HEAD_DIM;
        norm_apply_ruler(row, tmp, HQVM_HEAD_DIM, g, g0 > 0.0f ? g0 : 1.0f);
        memcpy(row, tmp, sizeof(tmp));
    }
}

static void residual_add(
    const float *x, const float *y, float *out, int64_t n, uint8_t u6, uint8_t v6)
{
    static int s_id = -1;
    int64_t i;
    if (s_id < 0) {
        const char *e = getenv("GYRO_NATIVE_RESIDUAL");
        /* 0 = identity x+y (debug); default = Δ-law */
        s_id = (e && e[0] == '0') ? 1 : 0;
    }
    if (s_id) {
        for (i = 0; i < n; ++i) out[i] = x[i] + y[i];
        return;
    }
    {
        const uint8_t Nc = popc6((uint8_t)(u6 ^ v6));
        const float m = (float)((int)Nc - 3) / 3.0f;
        const float gain = 1.0f + (float)APERTURE_GAP * m;
        for (i = 0; i < n; ++i) out[i] = x[i] + y[i] * gain;
    }
}

static void rope_qk_heads(
    float *Q, float *K, int32_t n_heads, int32_t gqa_ratio, int32_t token_pos)
{
    static int s_mode = -1; /* 0=tick codec, 1=skip, 2=float stock-like */
    float tmp[HQVM_HEAD_DIM];
    int h, kv_h, n_kv;
    const float freq_base = native_rope_freq_base();
    const float freq_scale = native_rope_freq_scale();
    if (s_mode < 0) {
        const char *e = getenv("GYRO_NATIVE_ROPE");
        if (e && e[0] == '0') s_mode = 1;
        else if (e && strcmp(e, "float") == 0) s_mode = 2;
        else s_mode = 0;
        fprintf(stderr, "[hqvm-rope] native mode=%s freq_base=%.0f freq_scale=%g\n",
            s_mode == 1 ? "skip" : (s_mode == 2 ? "float" : "tick"),
            (double)freq_base, (double)freq_scale);
        fflush(stderr);
    }
    if (s_mode == 1 || n_heads <= 0 || gqa_ratio <= 0) return;

    n_kv = n_heads / gqa_ratio;
    if (n_kv <= 0) n_kv = 1;

    if (s_mode == 2) {
        /* Stock-like NeoX float RoPE + YaRN mix (ext_factor=1, mscale≈1 after cancel). */
        int i;
        const int np = HQVM_HEAD_DIM / 2;
        const int n_dims = HQVM_HEAD_DIM;
        const float n_ctx_orig = 16384.0f;
        const float beta_fast = 32.0f;
        const float beta_slow = 1.0f;
        const float ext_factor = 1.0f;
        const float mscale = 1.0f; /* after llama.cpp yarn cancel ≈ 1 */
        float corr0, corr1;
        float cos_t[64], sin_t[64];
        float theta = (float)token_pos;
        const float theta_scale = powf(freq_base, -2.0f / (float)n_dims);
        {
            const float start = (float)n_dims * logf(n_ctx_orig / (beta_fast * 2.0f * (float)GYRO_M_PI))
                / (2.0f * logf(freq_base));
            const float end = (float)n_dims * logf(n_ctx_orig / (beta_slow * 2.0f * (float)GYRO_M_PI))
                / (2.0f * logf(freq_base));
            corr0 = start < 0.0f ? 0.0f : floorf(start);
            corr1 = end > (float)(n_dims - 1) ? (float)(n_dims - 1) : ceilf(end);
        }
        for (i = 0; i < np; ++i) {
            const float theta_extrap = theta;
            const float theta_interp = freq_scale * theta_extrap;
            float ramp = 1.0f - ((float)i - corr0) / (corr1 - corr0 + 1e-3f);
            if (ramp < 0.0f) ramp = 0.0f;
            if (ramp > 1.0f) ramp = 1.0f;
            ramp *= ext_factor;
            {
                const float th = theta_interp * (1.0f - ramp) + theta_extrap * ramp;
                cos_t[i] = cosf(th) * mscale;
                sin_t[i] = sinf(th) * mscale;
            }
            theta *= theta_scale;
        }
        for (h = 0; h < n_heads; ++h) {
            float *qh = Q + h * HQVM_HEAD_DIM;
            for (i = 0; i < np; ++i) {
                const int i0 = 2 * i;
                const float x0 = qh[i0], x1 = qh[i0 + 1];
                qh[i0] = x0 * cos_t[i] - x1 * sin_t[i];
                qh[i0 + 1] = x0 * sin_t[i] + x1 * cos_t[i];
            }
        }
        for (kv_h = 0; kv_h < n_kv; ++kv_h) {
            float *kh = K + kv_h * HQVM_HEAD_DIM;
            for (i = 0; i < np; ++i) {
                const int i0 = 2 * i;
                const float x0 = kh[i0], x1 = kh[i0 + 1];
                kh[i0] = x0 * cos_t[i] - x1 * sin_t[i];
                kh[i0 + 1] = x0 * sin_t[i] + x1 * cos_t[i];
            }
        }
        return;
    }

    {
        uint8_t ticks[HQVM_ROPE_MAX_FREQ];
        const float theta_scale = powf(freq_base, -2.0f / (float)HQVM_HEAD_DIM);
        hqvm_rope_codec_init();
        hqvm_rope_init_dtheta(HQVM_HEAD_DIM, theta_scale, freq_scale, NULL);
        hqvm_rope_ticks_from_pos(token_pos, HQVM_HEAD_DIM, ticks);
        for (h = 0; h < n_heads; ++h) {
            float *qh = Q + h * HQVM_HEAD_DIM;
            memcpy(tmp, qh, sizeof(tmp));
            hqvm_rope_apply_row(tmp, qh, ticks, HQVM_HEAD_DIM, HQVM_HEAD_DIM / 2, 1, 1.0f);
        }
        for (kv_h = 0; kv_h < n_kv; ++kv_h) {
            float *kh = K + kv_h * HQVM_HEAD_DIM;
            memcpy(tmp, kh, sizeof(tmp));
            hqvm_rope_apply_row(tmp, kh, ticks, HQVM_HEAD_DIM, HQVM_HEAD_DIM / 2, 1, 1.0f);
        }
    }
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

/* Attn-L0: top-1 by QK score → one-hot weights. */
static void attn_weights_l0(float *scores, int64_t kv_len) {
    int64_t j, best = 0;
    float best_s = -INFINITY;
    for (j = 0; j < kv_len; ++j) {
        if (scores[j] > best_s) { best_s = scores[j]; best = j; }
    }
    for (j = 0; j < kv_len; ++j) scores[j] = 0.0f;
    if (kv_len > 0) scores[best] = 1.0f;
}

/* Debug: standard causal softmax (stock chart) for isolation. */
static void attn_weights_softmax(float *scores, int64_t kv_len) {
    int64_t j;
    float M = -INFINITY, Z = 0.0f;
    for (j = 0; j < kv_len; ++j) if (scores[j] > M) M = scores[j];
    for (j = 0; j < kv_len; ++j) {
        scores[j] = expf(scores[j] - M);
        Z += scores[j];
    }
    if (Z <= 0.0f) Z = 1.0f;
    for (j = 0; j < kv_len; ++j) scores[j] /= Z;
}

/* Attn-L1: within-shell top-k, equal shell weights (λ=1 → lam_pow=1). */
static void attn_weights_l1_flat(
    float *scores, const float *qh, const uint8_t *chi_layer,
    int64_t n_kv_heads, int kv_head, int64_t kv_len)
{
    /* Nc=3 → λ=1.0 in table → equal shell weights after normalize. */
    hqvm_attn_weight_shell_qk_flat(
        scores, qh, chi_layer, n_kv_heads, kv_head, kv_len, /*Nc*/ 3, HQVM_ATTN_SHELL_TOPK);
}

int hqvm_block_forward(
    const float *x_in,
    int32_t token_pos,
    int32_t layer_idx,
    uint64_t depth,
    const hqvm_block_weights_t *W,
    uint8_t *u6,
    uint8_t *v6,
    hqvm_block_kv_t *KV,
    float *x_out)
{
    float *x_norm = NULL;
    float *Q = NULL, *K = NULL, *V = NULL, *attn = NULL, *attn_o = NULL;
    float *x_mid = NULL, *x_n2 = NULL;
    float *gate = NULL, *up = NULL, *ffn_h = NULL, *ffn_o = NULL;
    float *scores = NULL;
    float *lift_scores = NULL;
    uint8_t fam;
    uint8_t Nc;
    int32_t head, n_heads = HQVM_N_HEAD;
    int64_t kv_len;
    int rc = 0;

    if (!x_in || !W || !u6 || !v6 || !x_out) return -1;
    if (layer_idx < 0 || layer_idx >= HQVM_N_LAYER) return -1;

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
            memcpy(x_out, x_in, HQVM_HIDDEN_DIM * sizeof(float));
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

    x_norm = (float *)malloc(HQVM_HIDDEN_DIM * sizeof(float));
    Q = (float *)malloc(HQVM_HIDDEN_DIM * sizeof(float));
    K = (float *)malloc(HQVM_HIDDEN_DIM * sizeof(float));
    V = (float *)malloc(HQVM_HIDDEN_DIM * sizeof(float));
    attn = (float *)malloc(HQVM_HIDDEN_DIM * sizeof(float));
    attn_o = (float *)malloc(HQVM_HIDDEN_DIM * sizeof(float));
    x_mid = (float *)malloc(HQVM_HIDDEN_DIM * sizeof(float));
    x_n2 = (float *)malloc(HQVM_HIDDEN_DIM * sizeof(float));
    gate = (float *)malloc(HQVM_FFN_DIM * sizeof(float));
    up = (float *)malloc(HQVM_FFN_DIM * sizeof(float));
    ffn_h = (float *)malloc(HQVM_FFN_DIM * sizeof(float));
    ffn_o = (float *)malloc(HQVM_HIDDEN_DIM * sizeof(float));
    if (!x_norm || !Q || !K || !V || !attn || !attn_o || !x_mid || !x_n2
        || !gate || !up || !ffn_h || !ffn_o) {
        rc = -2;
        goto done;
    }

    /* --- Attention sublayer --- */
    norm_apply_ruler(x_in, x_norm, HQVM_HIDDEN_DIM, W->attn_norm_g, W->attn_norm_g0);

    if (matmul_q1(&W->attn_q, x_norm, Q) != 0
        || matmul_q1(&W->attn_k, x_norm, K) != 0
        || matmul_q1(&W->attn_v, x_norm, V) != 0) {
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
                rms_norm_heads(Q, n_heads, W->attn_q_norm_g, W->attn_q_norm_g0);
                rms_norm_heads(K, n_kv_heads > 0 ? n_kv_heads : n_heads,
                               W->attn_k_norm_g, W->attn_k_norm_g0);
            }
        }
        rope_qk_heads(Q, K, n_heads, gqa_ratio, token_pos);
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
                memcpy(krow, K + head * HQVM_HEAD_DIM, HQVM_HEAD_DIM * sizeof(float));
                memcpy(vrow, V + head * HQVM_HEAD_DIM, HQVM_HEAD_DIM * sizeof(float));
                if (chi_layer) {
                    chi_layer[(size_t)pos * (size_t)KV->n_kv_heads + (size_t)head] =
                        hqvm_k_chi6_from_row(K + head * HQVM_HEAD_DIM);
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
                hqvm_quantize_row_q8(K + head * HQVM_HEAD_DIM, HQVM_HEAD_DIM, krow);
                hqvm_quantize_row_q8(V + head * HQVM_HEAD_DIM, HQVM_HEAD_DIM, vrow);
                if (chi_layer) {
                    chi_layer[(size_t)pos * (size_t)KV->n_kv_heads + (size_t)head] =
                        hqvm_k_chi6_from_row(K + head * HQVM_HEAD_DIM);
                }
            }
        }
        hqvm_kv_write_inc((uint64_t)KV->n_kv_heads);
        kv_len = pos + 1;
    } else {
        hqvm_kv_null_write_inc();
    }

    memset(attn, 0, HQVM_HIDDEN_DIM * sizeof(float));
    Nc = popc6((uint8_t)((*u6) ^ (*v6)));
    scores = (float *)malloc((size_t)kv_len * sizeof(float));
    lift_scores = (float *)malloc((size_t)kv_len * sizeof(float));
    if (!scores || !lift_scores) { rc = -2; goto done; }

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
            const float *qh = Q + head * HQVM_HEAD_DIM;
            float *ah = attn + head * HQVM_HEAD_DIM;
            int64_t j;
            const size_t k_per_head_q8 = (KV && KV->n_kv_heads > 0 && !use_f32)
                ? (size_t)(KV->k_row_stride / KV->n_kv_heads) : 0;
            const size_t floats_per_tok = (KV && use_f32)
                ? (size_t)(KV->k_row_stride / sizeof(float)) : 0;

            for (j = 0; j < kv_len; ++j) {
                float s = 0.0f;
                if (k_layer_f32) {
                    const float *kh = k_layer_f32
                        + (size_t)j * floats_per_tok
                        + (size_t)kv_head * (size_t)HQVM_HEAD_DIM;
                    int d;
                    for (d = 0; d < HQVM_HEAD_DIM; ++d) s += qh[d] * kh[d];
                    s *= attn_scale;
                } else if (k_layer_q8) {
                    const char *krow = k_layer_q8
                        + (size_t)j * KV->k_row_stride
                        + (size_t)kv_head * k_per_head_q8;
                    s = hqvm_q8_cache_row_score(qh, krow, attn_scale);
                } else {
                    const float *kh = K + kv_head * HQVM_HEAD_DIM;
                    int d;
                    for (d = 0; d < HQVM_HEAD_DIM; ++d) s += qh[d] * kh[d];
                    s *= attn_scale;
                }
                scores[j] = s;
            }
            if (k_layer_q8 || k_layer_f32) hqvm_kv_read_inc((uint64_t)kv_len);

            if (attn_lvl < 0) {
                attn_weights_softmax(scores, kv_len);
            } else if (attn_lvl == 0) {
                attn_weights_l0(scores, kv_len);
            } else if (attn_lvl == 1) {
                if (chi_layer) {
                    attn_weights_l1_flat(
                        scores, qh, chi_layer, KV->n_kv_heads, kv_head, kv_len);
                } else {
                    hqvm_attn_weight_shell_qk(
                        scores, qh, NULL, kv_head, kv_len, /*Nc*/ 3, HQVM_ATTN_SHELL_TOPK);
                }
            } else {
                if (chi_layer) {
                    hqvm_attn_weight_shell_qk_flat(
                        scores, qh, chi_layer, KV->n_kv_heads, kv_head,
                        kv_len, Nc, HQVM_ATTN_SHELL_TOPK);
                } else {
                    hqvm_attn_weight_shell_qk(
                        scores, qh, NULL, kv_head, kv_len, Nc, HQVM_ATTN_SHELL_TOPK);
                }
            }

            if (head == 0) {
                memcpy(lift_scores, scores, (size_t)kv_len * sizeof(float));
            }

            if (v_layer_f32) {
                int64_t j2;
                memset(ah, 0, HQVM_HEAD_DIM * sizeof(float));
                for (j2 = 0; j2 < kv_len; ++j2) {
                    const float *vh = v_layer_f32
                        + (size_t)j2 * floats_per_tok
                        + (size_t)kv_head * (size_t)HQVM_HEAD_DIM;
                    int d;
                    for (d = 0; d < HQVM_HEAD_DIM; ++d) ah[d] += scores[j2] * vh[d];
                }
            } else if (v_layer_q8) {
                const size_t v_per_head =
                    (size_t)(KV->v_row_stride / KV->n_kv_heads);
                const char *v_head_base =
                    v_layer_q8 + (size_t)kv_head * v_per_head;
                hqvm_attn_v_reduce(
                    ah, HQVM_HEAD_DIM, scores, kv_len,
                    v_head_base, (size_t)KV->v_row_stride, 1, 0);
            } else {
                const float *vh = V + kv_head * HQVM_HEAD_DIM;
                int64_t j2;
                memset(ah, 0, HQVM_HEAD_DIM * sizeof(float));
                for (j2 = 0; j2 < kv_len; ++j2) {
                    int d;
                    for (d = 0; d < HQVM_HEAD_DIM; ++d) ah[d] += scores[j2] * vh[d];
                }
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
        for (d = 0; d < 64; ++d) if (Q[d] >= 0.0f) qsigns |= (1ull << d);
        chi_q = gyroscopic_chirality_from_signs64(qsigns);
        for (j = 0; j < kv_len; ++j) {
            if (lift_scores[j] > best) { best = lift_scores[j]; argmax = (int)j; }
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
    }

    if (matmul_q1(&W->attn_o, attn, attn_o) != 0) { rc = -3; goto done; }
    {
        static int s_skip_attn = -1;
        if (s_skip_attn < 0) {
            const char *e = getenv("GYRO_NATIVE_SKIP_ATTN");
            s_skip_attn = (e && e[0] && e[0] != '0') ? 1 : 0;
        }
        if (s_skip_attn) memset(attn_o, 0, HQVM_HIDDEN_DIM * sizeof(float));
    }
    residual_add(x_in, attn_o, x_mid, HQVM_HIDDEN_DIM, *u6, *v6);

    /* --- FFN sublayer --- */
    norm_apply_ruler(x_mid, x_n2, HQVM_HIDDEN_DIM, W->ffn_norm_g, W->ffn_norm_g0);
    if (matmul_q1(&W->ffn_gate, x_n2, gate) != 0
        || matmul_q1(&W->ffn_up, x_n2, up) != 0) {
        rc = -3;
        goto done;
    }
    Nc = popc6((uint8_t)((*u6) ^ (*v6)));
    {
        const int ffn_lvl = hqvm_ffn_level();
        if (ffn_lvl <= 0) {
            static int s_exact = -1;
            if (s_exact < 0) {
                const char *e = getenv("GYRO_NATIVE_SILU");
                s_exact = (e && strcmp(e, "exact") == 0) ? 1 : 0;
            }
            if (s_exact) {
                int64_t i;
                for (i = 0; i < HQVM_FFN_DIM; ++i) {
                    const float z = gate[i];
                    const float sig = 1.0f / (1.0f + expf(-z));
                    ffn_h[i] = z * sig * up[i];
                }
            } else {
                hqvm_swiglu_apply(ffn_h, gate, up, HQVM_FFN_DIM, 8.0f);
            }
        } else if (ffn_lvl == 1) {
            hqvm_ffn_shell_gate_apply(ffn_h, gate, up, HQVM_FFN_DIM, /*fam*/ 0, Nc);
        } else {
            hqvm_ffn_shell_gate_apply(ffn_h, gate, up, HQVM_FFN_DIM, fam, Nc);
        }
    }
    if (matmul_q1(&W->ffn_down, ffn_h, ffn_o) != 0) { rc = -3; goto done; }
    {
        static int s_skip_ffn = -1;
        if (s_skip_ffn < 0) {
            const char *e = getenv("GYRO_NATIVE_SKIP_FFN");
            s_skip_ffn = (e && e[0] && e[0] != '0') ? 1 : 0;
        }
        if (s_skip_ffn) memset(ffn_o, 0, HQVM_HIDDEN_DIM * sizeof(float));
    }

    {
        uint64_t gsigns = 0;
        uint8_t chi_g, byte;
        int d;
        for (d = 0; d < 64; ++d) if (gate[d] >= 0.0f) gsigns |= (1ull << d);
        chi_g = gyroscopic_chirality_from_signs64(gsigns);
        byte = hqvm_byte_of_q6_fam(chi_g, fam);
        hqvm_step_uv6(*u6, *v6, byte, u6, v6);
    }
    residual_add(x_mid, ffn_o, x_out, HQVM_HIDDEN_DIM, *u6, *v6);

    if ((token_pos == 0 && (layer_idx == 0 || layer_idx == HQVM_N_LAYER - 1)) ||
        (token_pos == 1 && layer_idx == 0)) {
        numstat_row("x_in", token_pos, layer_idx, x_in, HQVM_HIDDEN_DIM);
        numstat_row("x_out", token_pos, layer_idx, x_out, HQVM_HIDDEN_DIM);
        numstat_row("Q", token_pos, layer_idx, Q, HQVM_HIDDEN_DIM);
        numstat_row("V", token_pos, layer_idx, V, HQVM_KV_N_KV_HEAD * HQVM_HEAD_DIM);
        numstat_row("attn", token_pos, layer_idx, attn, HQVM_HIDDEN_DIM);
        numstat_row("attn_o", token_pos, layer_idx, attn_o, HQVM_HIDDEN_DIM);
        numstat_row("gate", token_pos, layer_idx, gate, HQVM_FFN_DIM);
        numstat_row("up", token_pos, layer_idx, up, HQVM_FFN_DIM);
        numstat_row("ffn_h", token_pos, layer_idx, ffn_h, HQVM_FFN_DIM);
        numstat_row("ffn_o", token_pos, layer_idx, ffn_o, HQVM_HIDDEN_DIM);
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
    free(x_norm); free(Q); free(K); free(V);
    free(attn); free(attn_o); free(x_mid); free(x_n2);
    free(gate); free(up); free(ffn_h); free(ffn_o); free(scores); free(lift_scores);
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
    (void)n_tokens;
    return hqvm_block_forward(x_in, token_pos, layer_idx, depth, W, u6, v6, KV, x_out);
}

int hqvm_forward_decode_step(
    float *x_row, int32_t t,
    uint8_t *u6, uint8_t *v6, hqvm_block_kv_t *KV)
{
    int32_t ell;
    float *scratch = NULL;
    float *xin, *xou;
    if (!x_row || !u6 || !v6 || t < 0) return -1;
    if (!hqvm_native_weights_ready()) return -2;
    scratch = (float *)malloc(HQVM_HIDDEN_DIM * sizeof(float));
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
            float *sw = xin;
            xin = xou;
            xou = sw;
        }
    }
    if (xin != x_row) {
        memcpy(x_row, xin, HQVM_HIDDEN_DIM * sizeof(float));
    }
    free(scratch);
    return 0;
}

int hqvm_forward_prefill(
    float *x, int32_t T,
    const float *embd_row0, int64_t n_embd,
    uint8_t *u6, uint8_t *v6, hqvm_block_kv_t *KV)
{
    int32_t ell, t;
    float *scratch = NULL;
    if (!x || !u6 || !v6 || T <= 0) return -1;
    if (!hqvm_native_weights_ready()) return -2;
    scratch = (float *)malloc((size_t)T * HQVM_HIDDEN_DIM * sizeof(float));
    if (!scratch) return -3;
    hqvm_reset_request(embd_row0 ? embd_row0 : x, embd_row0 ? n_embd : HQVM_HIDDEN_DIM);
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
        memcpy(x, scratch, (size_t)T * HQVM_HIDDEN_DIM * sizeof(float));
    }
    if (KV) KV->kv_pos = T - 1;
    free(scratch);
    return 0;
}

int hqvm_native_forward_ubatch(
    float *x, int32_t n_tokens, uint8_t *u6, uint8_t *v6, hqvm_layer_kv_t *KV)
{
    return hqvm_forward_prefill(x, n_tokens, x, HQVM_HIDDEN_DIM, u6, v6, KV);
}
