#include "attn.h"

#include "kernel.h"
#include "codec.h"
#include "ledger.h"
#include "runtime.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_MSC_VER)
#  include <intrin.h>
#  include <windows.h>
#else
#  include <stdatomic.h>
#endif

/*
 * Attention sites: H5 dyad×Q8 scores + H6 dyad×Q8 V-reduce (Analysis §7.3).
 * Stock float paths remain for opt-out / f32 V fallback / diagnostics.
 */

/* ===== KV / Q8 (chassis holonomic score hook) ===== */

int hqvm_kv_keys_enabled(void) {
    const char *e = getenv("GYRO_KV_KQ8");
    return (e && e[0] && e[0] != '0') ? 1 : 0;
}

int hqvm_coord_perturb_flip(void) {
    const char *e = getenv("GYRO_COORD_PERTURB");
    return (e && (strcmp(e, "flip_signs") == 0 || strcmp(e, "1") == 0)) ? 1 : 0;
}

int hqvm_holonomic_attn_enabled(void) {
    const char *e = getenv("GYRO_HOLONOMIC_ATTN");
    return (e && e[0] && e[0] != '0') ? 1 : 0;
}

#if defined(_MSC_VER)
static volatile LONG64 g_hol_calls = 0;
static volatile LONG64 g_stock_calls = 0;
static volatile LONG64 g_v_q8_calls = 0;
#else
static _Atomic uint64_t g_hol_calls = 0;
static _Atomic uint64_t g_stock_calls = 0;
static _Atomic uint64_t g_v_q8_calls = 0;
#endif

void hqvm_v_q8_inc(void) {
#if defined(_MSC_VER)
    InterlockedIncrement64(&g_v_q8_calls);
#else
    atomic_fetch_add(&g_v_q8_calls, 1);
#endif
}

void hqvm_holonomic_counters_inc(int holonomic) {
#if defined(_MSC_VER)
    if (holonomic) InterlockedIncrement64(&g_hol_calls);
    else InterlockedIncrement64(&g_stock_calls);
#else
    if (holonomic) atomic_fetch_add(&g_hol_calls, 1);
    else atomic_fetch_add(&g_stock_calls, 1);
#endif
}

void hqvm_holonomic_counters_get(uint64_t *holonomic, uint64_t *stock) {
#if defined(_MSC_VER)
    if (holonomic) *holonomic = (uint64_t)InterlockedCompareExchange64(&g_hol_calls, 0, 0);
    if (stock) *stock = (uint64_t)InterlockedCompareExchange64(&g_stock_calls, 0, 0);
#else
    if (holonomic) *holonomic = atomic_load(&g_hol_calls);
    if (stock) *stock = atomic_load(&g_stock_calls);
#endif
}

void hqvm_holonomic_counters_print(void) {
    uint64_t h = 0, s = 0, vq8 = 0;
    hqvm_holonomic_counters_get(&h, &s);
#if defined(_MSC_VER)
    vq8 = (uint64_t)InterlockedCompareExchange64(&g_v_q8_calls, 0, 0);
#else
    vq8 = atomic_load(&g_v_q8_calls);
#endif
    fprintf(stderr, "[hqvm-holonomic] holonomic_score_calls=%llu stock_score_calls=%llu v_q8_calls=%llu\n",
            (unsigned long long)h, (unsigned long long)s, (unsigned long long)vq8);
    /* Native driver / site ownership counters (resolved at link from layer/codec). */
    {
        extern uint64_t hqvm_stock_block_forward_calls(void);
        extern uint64_t hqvm_native_block_calls(void);
        extern uint64_t hqvm_stock_softmax_calls(void);
        extern uint64_t hqvm_stock_silu_calls(void);
        fprintf(stderr,
            "[hqvm-native] stock_block_forward_calls=%llu native_block_calls=%llu "
            "stock_softmax_calls=%llu stock_silu_calls=%llu pi_applied=%d\n",
            (unsigned long long)hqvm_stock_block_forward_calls(),
            (unsigned long long)hqvm_native_block_calls(),
            (unsigned long long)hqvm_stock_softmax_calls(),
            (unsigned long long)hqvm_stock_silu_calls(),
            hqvm_pi_applied());
    }
    fflush(stderr);
}

/* Arc 2B-2: score directly from ggml Q8_0 K-cache blocks.
 * Layout per block (34 B): fp16 scale + 32 int8 quants. 4 blocks per head128. */
typedef struct { uint16_t d; int8_t q[HQVM_Q8_BLOCK]; } hqvm_q8_blk;

/*
 * H5 STOCK float Q·Q8 score. Chassis chart — not a native claim.
 */
float hqvm_q8_cache_row_score(
    const float *q_head128,
    const void *k_q8_blocks,
    float scale)
{
    const hqvm_q8_blk *blk = (const hqvm_q8_blk *)k_q8_blocks;
    float s = 0.0f;
    int b, i;
    static int s_zero = -1;
    if (s_zero < 0) {
        const char *e = getenv("GYRO_COORD_PERTURB");
        s_zero = (e && strcmp(e, "zero_kq8") == 0) ? 1 : 0;
    }
    if (!q_head128 || !blk) return 0.0f;
    if (s_zero) return 0.0f;
    for (b = 0; b < 4; ++b) {
        const float d = hqvm_f16_to_f32(blk[b].d);
        float part = 0.0f;
        for (i = 0; i < HQVM_Q8_BLOCK; ++i) {
            part += q_head128[b * HQVM_Q8_BLOCK + i] * (float)blk[b].q[i];
        }
        s += part * d;
    }
    return s * scale;
}

/* IEEE binary32 parts (same law as codec dyad_unpack; local to avoid exporting). */
static int attn_dyad_parts(hqvm_dyad32_t x, uint32_t *sign, uint64_t *sig, int *exp2) {
    const uint32_t ef = (x.bits >> 23) & 0xffu;
    const uint32_t frac = x.bits & 0x7fffffu;
    if (ef == 0xffu) return -1;
    *sign = x.bits >> 31;
    if (ef == 0) {
        *sig = frac;
        *exp2 = -149;
    } else {
        *sig = (uint64_t)(0x800000u | frac);
        *exp2 = (int)ef - 150;
    }
    return 0;
}

/*
 * H5 native controller score: dyad Q × Q8 K without materializing float Q[128].
 * Uses dyad mantissa/exp + integer fp16 K scale; accumulates in double (same
 * algebraic product as stock float Q·Q8) so we do not truncate negative shifts.
 * One float at the end for shell-law interop — not a dyad→float→stock converter.
 */
float hqvm_dyad_q8_cache_row_score(
    const hqvm_dyad32_t *q_head_dyad,
    const void *k_q8_blocks,
    float scale)
{
    const hqvm_q8_blk *blk = (const hqvm_q8_blk *)k_q8_blocks;
    double acc = 0.0;
    int b, i;
    static int s_zero = -1;
    if (s_zero < 0) {
        const char *e = getenv("GYRO_COORD_PERTURB");
        s_zero = (e && strcmp(e, "zero_kq8") == 0) ? 1 : 0;
    }
    if (!q_head_dyad || !blk) return 0.0f;
    if (s_zero) return 0.0f;
    for (b = 0; b < 4; ++b) {
        uint16_t mw;
        int32_t ew;
        double d_scale;
        if (hqvm_fp16_decode_nonnegative(blk[b].d, &mw, &ew) != 0) {
            /* Match stock: fall back to f16→f32 chart if decode rejects. */
            d_scale = (double)hqvm_f16_to_f32(blk[b].d);
            mw = 0;
            ew = 0;
        } else {
            d_scale = ldexp((double)mw, ew);
        }
        for (i = 0; i < HQVM_Q8_BLOCK; ++i) {
            uint32_t sign;
            uint64_t sig;
            int exp2;
            int8_t kq = blk[b].q[i];
            double q;
            const int idx = b * HQVM_Q8_BLOCK + i;
            if (attn_dyad_parts(q_head_dyad[idx], &sign, &sig, &exp2) != 0) continue;
            q = ldexp((double)sig, exp2);
            if (sign) q = -q;
            acc += q * (double)kq * d_scale;
        }
    }
    return (float)(acc * (double)scale);
}

/* Arc 3D: fused Attn@V accumulate from displaced
 * Q8_0 V blocks — blockwise dequant + accumulate, no float V materialized. */
void hqvm_attn_v_accum_q8(
    float *VKQ32,
    int64_t DV,
    const void *v_q8_row,
    float a)
{
    const hqvm_q8_blk *blk = (const hqvm_q8_blk *)v_q8_row;
    int64_t b, d;
    int i;
    if (!VKQ32 || !blk || DV <= 0 || a == 0.0f) return;
    d = 0;
    for (b = 0; b < DV / HQVM_Q8_BLOCK; ++b) {
        const float ds = hqvm_f16_to_f32(blk[b].d);
        const float ad = a * ds;
        for (i = 0; i < HQVM_Q8_BLOCK; ++i) {
            VKQ32[d + i] += ad * (float)blk[b].q[i];
        }
        d += HQVM_Q8_BLOCK;
    }
}

/*
 * H6 native value reduce (Analysis §7.3): dyad attention weights × Q8_0 V.
 * Owns the accumulate (double acc, same product as stock float×Q8); packs dyad
 * once per head dim. Not a wrap of hqvm_attn_v_reduce + pack.
 */
int hqvm_attn_v_reduce_dyad_q8(
    hqvm_dyad32_t *out_dyad,
    int64_t DV,
    const hqvm_dyad32_t *weights_dyad,
    int64_t Nk,
    const void *v_q8_base,
    size_t v_row_stride)
{
    double *acc = NULL;
    int64_t ik, d;
    int i;
    if (!out_dyad || !weights_dyad || !v_q8_base || DV <= 0 || Nk <= 0) return -1;
    if (DV % HQVM_Q8_BLOCK != 0) return -1;
    acc = (double *)malloc((size_t)DV * sizeof(double));
    if (!acc) return -1;
    for (d = 0; d < DV; ++d) acc[d] = 0.0;
    for (ik = 0; ik < Nk; ++ik) {
        uint32_t sign;
        uint64_t sig;
        int exp2;
        double a;
        const hqvm_q8_blk *blk;
        if (attn_dyad_parts(weights_dyad[ik], &sign, &sig, &exp2) != 0) continue;
        a = ldexp((double)sig, exp2);
        if (sign) a = -a;
        if (a == 0.0) continue;
        hqvm_v_q8_inc();
        blk = (const hqvm_q8_blk *)((const char *)v_q8_base + (size_t)ik * v_row_stride);
        d = 0;
        for (; d + HQVM_Q8_BLOCK <= DV; d += HQVM_Q8_BLOCK) {
            uint16_t mw;
            int32_t ew;
            double ds;
            const int64_t b = d / HQVM_Q8_BLOCK;
            if (hqvm_fp16_decode_nonnegative(blk[b].d, &mw, &ew) != 0) {
                ds = (double)hqvm_f16_to_f32(blk[b].d);
            } else {
                ds = ldexp((double)mw, ew);
            }
            for (i = 0; i < HQVM_Q8_BLOCK; ++i) {
                acc[d + i] += a * ds * (double)blk[b].q[i];
            }
        }
    }
    for (d = 0; d < DV; ++d) {
        out_dyad[d] = hqvm_dyad32_from_f32((float)acc[d]);
    }
    free(acc);
    return 0;
}

/* Arc 4 Lift: chi6 of a float K head — WHT peak (matches lift store). */
uint8_t hqvm_k_chi6_from_row(const float *k_head128) {
    uint64_t signs = 0;
    int i;
    if (!k_head128) return 0;
    for (i = 0; i < 64; ++i) {
        if (k_head128[i] >= 0.0f) signs |= (1ull << i);
    }
    return gyroscopic_chirality_from_signs64(signs);
}

uint8_t hqvm_k_chi6_from_dyad_head(const hqvm_dyad32_t *k_head128) {
    uint64_t signs = 0;
    int i;
    if (!k_head128) return 0;
    for (i = 0; i < 64; ++i) {
        if (!hqvm_dyad32_sign(k_head128[i])) signs |= (1ull << i);
    }
    return gyroscopic_chirality_from_signs64(signs);
}

/* Arc 2B-2/3C: quantize one F16/F32 row into Q8_0 blocks (set_rows write path).
 * n dims must be a multiple of HQVM_Q8_BLOCK. fp16 scale + int8 quants. */
void hqvm_quantize_row_q8(
    const float *row,
    int64_t n,
    void *out_q8)
{
    /* Match ggml quantize_row_q8_0: scale d = amax/127 is always >= 0. */
    hqvm_q8_blk *blk = (hqvm_q8_blk *)out_q8;
    int64_t b;
    int i;
    if (!row || !out_q8 || n <= 0) return;
    for (b = 0; b < n / HQVM_Q8_BLOCK; ++b) {
        float amax = 0.0f;
        float d, id;
        for (i = 0; i < HQVM_Q8_BLOCK; ++i) {
            const float ax = fabsf(row[b * HQVM_Q8_BLOCK + i]);
            if (ax > amax) amax = ax;
        }
        d = amax / 127.0f;
        id = d > 0.0f ? 1.0f / d : 0.0f;
        blk[b].d = hqvm_f32_to_f16(d);
        for (i = 0; i < HQVM_Q8_BLOCK; ++i) {
            int qv = (int)lrintf(row[b * HQVM_Q8_BLOCK + i] * id);
            if (qv > 127) qv = 127;
            if (qv < -127) qv = -127;
            blk[b].q[i] = (int8_t)qv;
        }
    }
}

/* Arc 4B: fold a Q8_0 K row into an intron byte (sign XOR + scale). */
uint8_t hqvm_intron_from_q8_row(const void *k_q8_row) {
    const hqvm_q8_blk *blk = (const hqvm_q8_blk *)k_q8_row;
    uint8_t intron = 0;
    int i;
    if (!blk) return 0;
    for (i = 0; i < HQVM_Q8_BLOCK; ++i) intron ^= (uint8_t)blk[0].q[i];
    intron ^= (uint8_t)blk[0].d;
    return intron;
}

/* Byte table for receipts / lift (single owner in this file). */
#if defined(_MSC_VER)
static int hqvm_popcount64(uint64_t x) {
    return (int)__popcnt64(x);
}
#else
static int hqvm_popcount64(uint64_t x) {
    return __builtin_popcountll(x);
}
#endif

uint16_t hqvm_f32_to_f16(float f) {
    union { float f; uint32_t u; } v;
    uint32_t x, sign, mant;
    int exp;
    v.f = f;
    x = v.u;
    sign = (x >> 16) & 0x8000u;
    exp = (int)((x >> 23) & 0xFFu) - 127 + 15;
    mant = x & 0x7FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant = (mant | 0x800000u) >> (1 - exp);
        return (uint16_t)(sign | ((mant + 0x1000u) >> 13));
    }
    if (exp >= 31) {
        /* Inf or NaN: keep the low mantissa bits so NaN stays NaN with payload. */
        if (mant == 0) return (uint16_t)(sign | 0x7C00u);
        return (uint16_t)(sign | 0x7C00u | (mant >> 13));
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | ((mant + 0x1000u) >> 13));
}

float hqvm_f16_to_f32(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    union { float f; uint32_t u; } v;
    if (exp == 0) {
        if (mant == 0) {
            v.u = sign;
            return v.f;
        }
        exp = 127 - 15 + 1;
        while ((mant & 0x400u) == 0) {
            mant <<= 1;
            exp--;
        }
        mant &= 0x3FFu;
        v.u = sign | ((uint32_t)exp << 23) | (mant << 13);
        return v.f;
    }
    if (exp == 31) {
        v.u = sign | 0x7F800000u | (mant << 13);
        return v.f;
    }
    v.u = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    return v.f;
}


/* ===== Softmax / Attn@V ===== */
/*
 * Holonomic attention math — gyroscopic-owned (thin ggml hook in ops.cpp).
 */



#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_MSC_VER)
#  include <windows.h>
#endif

/* GF(2)^6 rank of a 6-bit vector set (kernel transport space). */
static int hqvm_rank_gf2_6(const uint8_t *vals, int64_t n) {
    int basis[6] = {0, 0, 0, 0, 0, 0};
    int64_t i;
    int rank = 0;
    for (i = 0; i < n; ++i) {
        int x = vals[i] & 63;
        int b;
        for (b = 5; b >= 0; --b) {
            if (!(x & (1 << b))) continue;
            if (!basis[b]) { basis[b] = x; rank++; break; }
            x ^= basis[b];
        }
    }
    return rank;
}

/* Percolation coverage on the kernel: |Reach| = (2^r)^2, rank-5 parity-obstructed
 * plateau 1024 (even-weight only). q8 K row -> chi6 via kernel sign/WHT. */
static void hqvm_krow_chi_pair(const void *k_q8_row, uint8_t chi[2]) {
    uint64_t signs0 = 0, signs1 = 0;
    const uint8_t *b = (const uint8_t *) k_q8_row;
    int i;
    /* Q8_0 row of 128: 4 blocks of (f16 scale + 32 int8). Sign bits only. */
    for (i = 0; i < 64; ++i) {
        const int blk = i >> 5;
        const int off = 2 + blk * 34 + (i & 31);
        if (((const int8_t *) b)[off] >= 0) signs0 |= (1ull << i);
    }
    for (i = 0; i < 64; ++i) {
        const int blk = 2 + (i >> 5);
        const int off = 2 + blk * 34 + (i & 31);
        if (((const int8_t *) b)[off] >= 0) signs1 |= (1ull << i);
    }
    chi[0] = gyroscopic_chirality_from_signs64(signs0);
    chi[1] = gyroscopic_chirality_from_signs64(signs1);
}

int hqvm_v_perturb_enabled(void) {
    static int s = -1;
    if (s < 0) {
        const char *e = getenv("GYRO_V_PERTURB");
        s = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    return s;
}

void hqvm_softmax_inplace(float *scores, int64_t Nk, float M) {
    float Ssum = 0.0f;
    int64_t i;
    float invS;
    if (!scores || Nk <= 0) return;
    for (i = 0; i < Nk; ++i) {
        if (scores[i] > -1e20f) {
            scores[i] = expf(scores[i] - M);
            Ssum += scores[i];
        } else {
            scores[i] = 0.0f;
        }
    }
    invS = Ssum > 0.0f ? 1.0f / Ssum : 0.0f;
    for (i = 0; i < Nk; ++i) scores[i] *= invS;
}

static uint64_t s_stock_softmax_calls = 0;

void hqvm_stock_softmax_inc(void) {
    s_stock_softmax_calls++;
}

uint64_t hqvm_stock_softmax_calls(void) {
    return s_stock_softmax_calls;
}

static float hqvm_attn_lambda_from_Nc(uint8_t Nc) {
    /* m=(Nc-3)/3; λ=(1+m)/(1-m). Nc=0 is the singular pole: use λ=1 so
     * shell masses stay finite (λ=0 would zero every N>0 term). */
    static const float lam[7] = {
        1.0f, 0.2f, 0.5f, 1.0f, 2.0f, 5.0f, HQVM_LAMBDA_MAX_ATTN
    };
    if (Nc > 6) Nc = 6;
    return lam[Nc];
}

/* Shared shell-weight core. chi from flat native KV table when present. */
static void attn_weight_shell_qk_impl(
    float *scores, uint8_t chi_q,
    const uint8_t *k_chi6_flat, int64_t n_kv_heads, int kv_head_flat,
    int64_t Nk, uint8_t Nc, int top_k)
{
    int64_t j;
    float lam, lam_pow[7];
    float *w = NULL;
    int *shell_of = NULL;
    uint8_t *taken = NULL;
    int count[7];
    float denom = 0.0f;
    int i, N;
    if (!scores || Nk <= 0) return;
    if (top_k <= 0) top_k = HQVM_ATTN_SHELL_TOPK;
    if (kv_head_flat < 0) kv_head_flat = 0;
    if (n_kv_heads <= 0) n_kv_heads = HQVM_KV_N_KV_HEAD;
    lam = hqvm_attn_lambda_from_Nc(Nc);
    lam_pow[0] = 1.0f;
    for (i = 1; i <= 6; ++i) lam_pow[i] = lam_pow[i - 1] * lam;

    w = (float *)malloc((size_t)Nk * sizeof(float));
    shell_of = (int *)malloc((size_t)Nk * sizeof(int));
    taken = (uint8_t *)malloc((size_t)Nk);
    if (!w || !shell_of || !taken) { free(w); free(shell_of); free(taken); return; }
    memset(taken, 0, (size_t)Nk);
    for (N = 0; N < 7; ++N) count[N] = 0;

    for (j = 0; j < Nk; ++j) {
        uint8_t chi_k;
        int Nj;
        w[j] = 0.0f;
        shell_of[j] = -1;
        if (scores[j] <= -1e20f) continue;
        chi_k = (k_chi6_flat && kv_head_flat >= 0 && kv_head_flat < (int)n_kv_heads)
            ? k_chi6_flat[j * n_kv_heads + kv_head_flat] : 0;
#if defined(_MSC_VER)
        Nj = (int)__popcnt((unsigned)((chi_q ^ chi_k) & 63));
#else
        Nj = __builtin_popcount((unsigned)((chi_q ^ chi_k) & 63));
#endif
        if (Nj > 6) Nj = 6;
        shell_of[j] = Nj;
        count[Nj]++;
    }

    /* Within each shell: algebraic top-k by QK score. */
    for (N = 0; N < 7; ++N) {
        int need = top_k;
        int got = 0;
        if (count[N] <= 0) continue;
        if (need > count[N]) need = count[N];
        while (got < need) {
            int64_t best = -1;
            float best_s = -INFINITY;
            for (j = 0; j < Nk; ++j) {
                if (shell_of[j] != N || taken[j]) continue;
                if (scores[j] > best_s) { best_s = scores[j]; best = j; }
            }
            if (best < 0) break;
            taken[best] = 1u;
            got++;
        }
    }

    /* Joint: peaked controller chart of QK energy times lambda^N. */
    {
        float M = -INFINITY;
        int any = 0;
        for (j = 0; j < Nk; ++j) {
            if (!taken[j]) continue;
            if (scores[j] > M) M = scores[j];
            any = 1;
        }
        denom = 0.0f;
        if (any) {
            for (j = 0; j < Nk; ++j) {
                int Nj;
                float e;
                if (!taken[j]) {
                    w[j] = 0.0f;
                    continue;
                }
                Nj = shell_of[j];
                if (Nj < 0) Nj = 0;
                if (Nj > 6) Nj = 6;
                e = expf(scores[j] - M);
                w[j] = e * lam_pow[Nj];
                denom += w[j];
            }
        }
    }

    if (denom <= 0.0f) {
        /* Fallback: uniform over unmasked */
        int nu = 0;
        for (j = 0; j < Nk; ++j) if (scores[j] > -1e20f) nu++;
        for (j = 0; j < Nk; ++j)
            scores[j] = (scores[j] > -1e20f && nu > 0) ? (1.0f / (float)nu) : 0.0f;
    } else {
        for (j = 0; j < Nk; ++j) scores[j] = w[j] / denom;
    }
    free(w);
    free(shell_of);
    free(taken);
}

static uint8_t attn_chi_q_from_float_head(const float *q_head128) {
    uint64_t qsigns = 0;
    int i;
    if (!q_head128) return 0;
    for (i = 0; i < 64; ++i) if (q_head128[i] >= 0.0f) qsigns |= (1ull << i);
    return gyroscopic_chirality_from_signs64(qsigns);
}

static uint8_t attn_chi_q_from_dyad_head(const hqvm_dyad32_t *q_head_dyad) {
    uint64_t qsigns = 0;
    int i;
    if (!q_head_dyad) return 0;
    for (i = 0; i < 64; ++i) if (!hqvm_dyad32_sign(q_head_dyad[i])) qsigns |= (1ull << i);
    return gyroscopic_chirality_from_signs64(qsigns);
}

void hqvm_attn_weight_shell_qk(
    float *scores, const float *q_head128, const void *k_chi_base_unused,
    int kv_head_unused, int64_t Nk, uint8_t Nc, int top_k)
{
    (void)k_chi_base_unused;
    (void)kv_head_unused;
    attn_weight_shell_qk_impl(scores, attn_chi_q_from_float_head(q_head128),
                              NULL, 0, 0, Nk, Nc, top_k);
}

void hqvm_attn_weight_shell_qk_flat(
    float *scores, const float *q_head128, const uint8_t *k_chi6,
    int64_t n_kv_heads, int kv_head, int64_t Nk, uint8_t Nc, int top_k)
{
    attn_weight_shell_qk_impl(scores, attn_chi_q_from_float_head(q_head128),
                              k_chi6, n_kv_heads, kv_head, Nk, Nc, top_k);
}

/*
 * H6 STOCK float V-reduce (product path). Not a native dyad accumulate.
 * Do not wrap this under hosting and call the hazard closed.
 */
void hqvm_attn_v_reduce(
    float *out,
    int64_t DV,
    const float *weights,
    int64_t Nk,
    const void *v_base,
    size_t v_row_stride,
    int v_is_q8,
    int v_perturb)
{
    int64_t ik, d;
    if (!out || !weights || !v_base || DV <= 0 || Nk <= 0) return;
    for (d = 0; d < DV; ++d) out[d] = 0.0f;
    for (ik = 0; ik < Nk; ++ik) {
        const float a = weights[ik];
        const char *v_row;
        if (a == 0.0f) continue;
        v_row = (const char *)v_base + (size_t)ik * v_row_stride;
        if (v_is_q8) {
            hqvm_v_q8_inc();
            if (!v_perturb) hqvm_attn_v_accum_q8(out, DV, v_row, a);
        } else {
            const float *vf = (const float *)v_row;
            for (d = 0; d < DV; ++d) out[d] += a * vf[d];
        }
    }
}

/*
 * H5 score+weight orchestration.
 * Stock: float Q row + hqvm_q8_cache_row_score (honest chassis).
 * Native: hqvm_dyad_q8_cache_row_score (no float Q[128]); chi from dyad signs.
 * Product must call through hosting — do not claim *_dyad alone is native.
 */
int hqvm_attn_head_scores_weights_stock(
    hqvm_dyad32_t *weights_dyad,
    const hqvm_dyad32_t *q_head_dyad,
    const void *k_q8_base, const float *k_f32_base,
    const float *k_fallback_f32,
    size_t k_row_stride, size_t floats_per_tok, size_t k_per_head_q8,
    const uint8_t *chi_layer, int64_t n_kv_heads, int kv_head,
    int64_t kv_len, uint8_t Nc, int top_k, int attn_level, float attn_scale)
{
    float *scores = NULL;
    float q[HQVM_HEAD_DIM];
    int64_t j;
    int i;
    static int s_polar_env = -1;
    int polar_on = 0;
    uint64_t q_anchor = 0;
    uint8_t q_chi6 = 0;
    int64_t n_consulted = 0;
    int64_t n_kept = 0;
    if (s_polar_env < 0) {
        const char *e = getenv("GYRO_NATIVE_POLAR_PREFILTER");
        s_polar_env = (e && e[0] && e[0] != '0') ? 1 : 0;
        if (s_polar_env && !hqvm_rt_request_cell()) s_polar_env = 0;
    }
    polar_on = s_polar_env;
    if (!weights_dyad || !q_head_dyad || kv_len <= 0) return -1;
    if (top_k <= 0) top_k = HQVM_ATTN_SHELL_TOPK;
    for (i = 0; i < HQVM_HEAD_DIM; ++i) {
        q[i] = hqvm_dyad32_to_f32(q_head_dyad[i]);
        if (q[i] >= 0.0f) q_anchor |= (1ull << i);
    }
    if (polar_on) {
        q_chi6 = gyroscopic_chirality_from_signs64(q_anchor & 0xFFFFFFFFFFFFull);
    }
    scores = (float *)malloc((size_t)kv_len * sizeof(float));
    if (!scores) return -1;
    for (j = 0; j < kv_len; ++j) {
        float s = 0.0f;
        if (polar_on && chi_layer && n_kv_heads > 0
            && kv_head >= 0 && kv_head < (int)n_kv_heads) {
            const uint8_t chi_k =
                chi_layer[(size_t)j * (size_t)n_kv_heads + (size_t)kv_head];
            hqvm_rt_polar_summary qs, ks;
            qs.chi6 = q_chi6; qs.anchor64 = q_anchor; qs.radius = 1.0f;
            ks.chi6 = chi_k; ks.anchor64 = 0; ks.radius = 1.0f;
            ++n_consulted;
            if (hqvm_rt_polar_score(&qs, &ks) <= 0.0f) {
                scores[j] = -INFINITY;
                continue;
            }
            ++n_kept;
        }
        if (k_f32_base) {
            const float *kh = k_f32_base + (size_t)j * floats_per_tok + (size_t)kv_head * (size_t)HQVM_HEAD_DIM;
            int d; for (d = 0; d < HQVM_HEAD_DIM; ++d) s += q[d] * kh[d];
            s *= attn_scale;
        } else if (k_q8_base) {
            const char *krow = (const char *)k_q8_base + (size_t)j * k_row_stride + (size_t)kv_head * k_per_head_q8;
            s = hqvm_q8_cache_row_score(q, krow, attn_scale);
        } else {
            const float *kh;
            if (k_fallback_f32) {
                kh = k_fallback_f32 + kv_head * HQVM_HEAD_DIM;
            } else {
                static int s_selfscore_warned = 0;
                if (!s_selfscore_warned) {
                    fprintf(stderr,
                        "[hqvm-attn] no K source (f32/q8/fallback); scoring Q against Q\n");
                    fflush(stderr);
                    s_selfscore_warned = 1;
                }
                kh = q;
            }
            int d; for (d = 0; d < HQVM_HEAD_DIM; ++d) s += q[d] * kh[d];
            s *= attn_scale;
        }
        scores[j] = s;
    }
    if (polar_on) hqvm_rt_prefilter_report(n_consulted, n_kept);
    if (attn_level < 0) {
        float M = -INFINITY, Z = 0.0f;
        for (j = 0; j < kv_len; ++j) if (scores[j] > M) M = scores[j];
        for (j = 0; j < kv_len; ++j) { scores[j] = expf(scores[j] - M); Z += scores[j]; }
        if (Z <= 0.0f) Z = 1.0f;
        for (j = 0; j < kv_len; ++j) scores[j] /= Z;
        hqvm_stock_softmax_inc();
    } else if (attn_level == 0) {
        int64_t best = 0; float best_s = -INFINITY;
        for (j = 0; j < kv_len; ++j) if (scores[j] > best_s) { best_s = scores[j]; best = j; }
        for (j = 0; j < kv_len; ++j) scores[j] = 0.0f;
        if (kv_len > 0) scores[best] = 1.0f;
    } else {
        hqvm_attn_shell_weight_inc();
        if (chi_layer) hqvm_attn_weight_shell_qk_flat(scores, q, chi_layer, n_kv_heads, kv_head, kv_len,
            attn_level == 1 ? 3 : Nc, top_k);
        else hqvm_attn_weight_shell_qk(scores, q, NULL, kv_head, kv_len,
            attn_level == 1 ? 3 : Nc, top_k);
    }
    for (j = 0; j < kv_len; ++j) weights_dyad[j] = hqvm_dyad32_from_f32(scores[j]);
    free(scores);
    return 0;
}

/*
 * H5 native: dyad×Q8 integer scores + shell joint law. f32 K falls back to stock
 * for that row source (debug KV). No float Q[128] on the Q8 path.
 */
int hqvm_attn_head_scores_weights_native(
    hqvm_dyad32_t *weights_dyad,
    const hqvm_dyad32_t *q_head_dyad,
    const void *k_q8_base, const float *k_f32_base,
    const float *k_fallback_f32,
    size_t k_row_stride, size_t floats_per_tok, size_t k_per_head_q8,
    const uint8_t *chi_layer, int64_t n_kv_heads, int kv_head,
    int64_t kv_len, uint8_t Nc, int top_k, int attn_level, float attn_scale)
{
    float *scores = NULL;
    int64_t j;
    uint8_t chi_q;
    static int s_polar_env = -1;
    int polar_on = 0;
    uint64_t q_anchor = 0;
    int64_t n_consulted = 0;
    int64_t n_kept = 0;
    int i;

    if (!weights_dyad || !q_head_dyad || kv_len <= 0) return -1;
    /* f32 KV / missing Q8: not the native chart — use stock for honesty. */
    if (!k_q8_base || k_f32_base) {
        return hqvm_attn_head_scores_weights_stock(
            weights_dyad, q_head_dyad, k_q8_base, k_f32_base, k_fallback_f32,
            k_row_stride, floats_per_tok, k_per_head_q8,
            chi_layer, n_kv_heads, kv_head, kv_len, Nc, top_k, attn_level, attn_scale);
    }
    if (top_k <= 0) top_k = HQVM_ATTN_SHELL_TOPK;
    if (s_polar_env < 0) {
        const char *e = getenv("GYRO_NATIVE_POLAR_PREFILTER");
        s_polar_env = (e && e[0] && e[0] != '0') ? 1 : 0;
        if (s_polar_env && !hqvm_rt_request_cell()) s_polar_env = 0;
    }
    polar_on = s_polar_env;
    chi_q = attn_chi_q_from_dyad_head(q_head_dyad);
    for (i = 0; i < 64; ++i) if (!hqvm_dyad32_sign(q_head_dyad[i])) q_anchor |= (1ull << i);

    scores = (float *)malloc((size_t)kv_len * sizeof(float));
    if (!scores) return -1;
    for (j = 0; j < kv_len; ++j) {
        const char *krow = (const char *)k_q8_base + (size_t)j * k_row_stride
            + (size_t)kv_head * k_per_head_q8;
        if (polar_on && chi_layer && n_kv_heads > 0
            && kv_head >= 0 && kv_head < (int)n_kv_heads) {
            const uint8_t chi_k =
                chi_layer[(size_t)j * (size_t)n_kv_heads + (size_t)kv_head];
            hqvm_rt_polar_summary qs, ks;
            qs.chi6 = chi_q; qs.anchor64 = q_anchor; qs.radius = 1.0f;
            ks.chi6 = chi_k; ks.anchor64 = 0; ks.radius = 1.0f;
            ++n_consulted;
            if (hqvm_rt_polar_score(&qs, &ks) <= 0.0f) {
                scores[j] = -INFINITY;
                continue;
            }
            ++n_kept;
        }
        scores[j] = hqvm_dyad_q8_cache_row_score(q_head_dyad, krow, attn_scale);
    }
    if (polar_on) hqvm_rt_prefilter_report(n_consulted, n_kept);
    if (attn_level < 0) {
        float M = -INFINITY, Z = 0.0f;
        for (j = 0; j < kv_len; ++j) if (scores[j] > M) M = scores[j];
        for (j = 0; j < kv_len; ++j) { scores[j] = expf(scores[j] - M); Z += scores[j]; }
        if (Z <= 0.0f) Z = 1.0f;
        for (j = 0; j < kv_len; ++j) scores[j] /= Z;
        hqvm_stock_softmax_inc();
    } else if (attn_level == 0) {
        int64_t best = 0; float best_s = -INFINITY;
        for (j = 0; j < kv_len; ++j) if (scores[j] > best_s) { best_s = scores[j]; best = j; }
        for (j = 0; j < kv_len; ++j) scores[j] = 0.0f;
        if (kv_len > 0) scores[best] = 1.0f;
    } else {
        hqvm_attn_shell_weight_inc();
        attn_weight_shell_qk_impl(
            scores, chi_q,
            chi_layer, n_kv_heads, kv_head,
            kv_len, attn_level == 1 ? 3 : Nc, top_k);
    }
    for (j = 0; j < kv_len; ++j) weights_dyad[j] = hqvm_dyad32_from_f32(scores[j]);
    free(scores);
    (void)k_fallback_f32;
    return 0;
}


/* ===== CGM-lift — owns the single trajectory instance ===== */


#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_MSC_VER)
#  include <intrin.h>
#endif

/* BYTE_OF_Q6_FAM lives in kernel.c; attn re-exports via attn.h for lift path. */

static int s_init = 0;
static int s_table_ok = 0;

static uint64_t s_lift_calls = 0;
static uint64_t s_invariant_fails = 0;

static gyro_trajectory_state_t s_traj;
static int s_traj_init = 0;

/* GyroClock — genealogy = law + CS anchor + depth (no per-step byte storage). */
static int s_seq_active = 0;
static int s_layer = 0;
static uint64_t s_depth_min = 0;
static uint64_t s_depth_max = 0;
static uint64_t s_step_count = 0;
static uint64_t s_depth_start = 0; /* first depth recorded this sequence */
static uint64_t s_depth_end = 0;   /* max+1 after last step (span end) */
static uint32_t s_rope_clock_token_pos = 0;
static uint32_t s_seq_len = 0;     /* tokens observed; next decode index */
static int s_decode_mode = 0;      /* last lift was decode (Nq==1) */
static uint8_t s_last_byte = 0;
static uint8_t s_pi_u6 = 0;
static uint8_t s_pi_v6 = 0;
static int s_pi_pending = 0;
static int s_pi_applied = 0;

void hqvm_cgm_lift_init(void) {
    if (s_init) return;
    hqvm_byte_table_init();
    s_table_ok = hqvm_byte_table_ok();
    if (!s_table_ok) {
        fprintf(stderr, "[hqvm-cgm-lift] BYTE_OF_Q6_FAM verify FAIL\n");
    } else {
        fprintf(stderr, "[hqvm-cgm-lift] BYTE_OF_Q6_FAM verify PASS\n");
    }
    s_init = 1;
}

int hqvm_byte_of_q6_fam_ok(void) {
    if (!s_init) hqvm_cgm_lift_init();
    return s_table_ok;
}

int hqvm_cgm_lift_enabled(void) {
    static int s = -1;
    if (s < 0) {
        const char *e = getenv("GYRO_CGM_LIFT");
        s = (e && e[0] && e[0] != '0') ? 1 : 0;
        if (s) hqvm_cgm_lift_init();
    }
    return s;
}

int hqvm_cgm_lift_layer(void) {
    return s_layer % HQVM_N_LAYER;
}

int hqvm_cgm_lift_bump_layer(void) {
    const int prev = s_layer % HQVM_N_LAYER;
    s_layer = (s_layer + 1) % HQVM_N_LAYER;
    /* After layer L-1 of a decode token, advance sequence cursor (not Nk-1). */
    if (s_decode_mode && prev == (HQVM_N_LAYER - 1)) {
        s_seq_len++;
        s_decode_mode = 0;
    }
    return s_layer;
}

int hqvm_cgm_lift_traj_ready(void) {
    return s_traj_init;
}

uint32_t hqvm_cgm_lift_state24(void) {
    return s_traj_init ? (s_traj.state24 & 0xFFFFFFu) : 0u;
}

uint8_t hqvm_cgm_lift_last_byte(void) {
    return s_last_byte;
}

void hqvm_pi_summary_sign12_from_bits(uint8_t u6, uint8_t v6) {
    static int s_pi_from_embd = -1;
    if (s_pi_from_embd < 0) {
        const char *env = getenv("GYRO_PI_FROM_EMBD");
        /* Default on. Set 0 to keep GENE_MAC_REST. */
        s_pi_from_embd = (env && env[0] == '0') ? 0 : 1;
    }
    if (!s_pi_from_embd) {
        s_pi_pending = 0;
        return;
    }
    s_pi_u6 = (uint8_t)(u6 & 63);
    s_pi_v6 = (uint8_t)(v6 & 63);
    s_pi_pending = 1;
}

void hqvm_pi_summary_sign12_from_embd(const float *e, int64_t n) {
    uint8_t u = 0, v = 0;
    int i;
    if (!e || n < 12) return;
    for (i = 0; i < 6; ++i) {
        if (e[i] < 0.0f) u |= (uint8_t)(1u << i);
        if (e[6 + i] < 0.0f) v |= (uint8_t)(1u << i);
    }
    hqvm_pi_summary_sign12_from_bits(u, v);
}

void hqvm_pi_stash_from_embd_row(const float *e, int64_t n) {
    hqvm_pi_summary_sign12_from_embd(e, n);
}

int hqvm_pi_applied(void) {
    return s_pi_applied;
}

void hqvm_cgm_lift_get_uv6(uint8_t *u6, uint8_t *v6) {
    if (u6) *u6 = (uint8_t)(s_pi_u6 & 63);
    if (v6) *v6 = (uint8_t)(s_pi_v6 & 63);
}

uint8_t hqvm_cgm_lift_carrier_shell(void) {
    uint8_t chi;
    if (s_traj_init) {
        chi = gyroscopic_chirality_word6(s_traj.state24);
    } else {
        chi = (uint8_t)((s_pi_u6 ^ s_pi_v6) & 63);
    }
#if defined(_MSC_VER)
    return (uint8_t)__popcnt((unsigned)(chi & 63));
#else
    return (uint8_t)__builtin_popcount((unsigned)(chi & 63));
#endif
}

uint8_t hqvm_cgm_lift_fam(void) {
    if (!s_traj_init) return 0;
    return (uint8_t)(s_traj.phase_idx & 3u);
}

void hqvm_cgm_lift_reset_sequence(void) {
    if (!s_init) hqvm_cgm_lift_init();
    hqvm_traj_reset(&s_traj);
    s_pi_applied = 0;
    if (s_pi_pending) {
        /* Pack (u6,v6) into A12/B12 low+high nibbles as CS anchor. */
        const uint16_t A = (uint16_t)(((s_pi_u6 & 63) | ((uint16_t)(s_pi_u6 & 63) << 6)) & 0xFFF);
        const uint16_t B = (uint16_t)(((s_pi_v6 & 63) | ((uint16_t)(s_pi_v6 & 63) << 6)) & 0xFFF);
        s_traj.state24 = ((uint32_t)A << 12) | (uint32_t)B;
        s_pi_applied = 1;
        s_pi_pending = 0;
        fprintf(stderr,
            "[hqvm-gyro-clock] reset_sequence L=%d Pi_applied u6=%u v6=%u state24=%06x\n",
            HQVM_N_LAYER, (unsigned)s_pi_u6, (unsigned)s_pi_v6,
            (unsigned)(s_traj.state24 & 0xFFFFFFu));
    } else {
        fprintf(stderr, "[hqvm-gyro-clock] reset_sequence L=%d anchor=GENE_MAC_REST\n",
                HQVM_N_LAYER);
    }
    s_traj_init = 1;
    s_layer = 0;
    s_depth_min = UINT64_MAX;
    s_depth_max = 0;
    s_step_count = 0;
    s_depth_start = 0;
    s_depth_end = 0;
    s_rope_clock_token_pos = 0;
    s_seq_len = 0;
    s_decode_mode = 0;
    s_last_byte = 0;
    s_seq_active = 1;
    fflush(stderr);
}

void hqvm_genealogy_observe_prefill(uint32_t token_pos) {
    s_decode_mode = 0;
    if (token_pos + 1u > s_seq_len) {
        s_seq_len = token_pos + 1u;
    }
}

uint32_t hqvm_genealogy_decode_token_pos(void) {
    s_decode_mode = 1;
    return s_seq_len;
}

uint32_t hqvm_genealogy_seq_len(void) {
    return s_seq_len;
}

int hqvm_cgm_lift_seq_active(void) {
    return s_seq_active;
}

uint64_t hqvm_genealogy_depth(uint32_t token_pos, uint32_t layer_idx) {
    return (uint64_t)token_pos * (uint64_t)HQVM_N_LAYER + (uint64_t)(layer_idx % HQVM_N_LAYER);
}

uint32_t hqvm_genealogy_token_pos(uint64_t depth) {
    return (uint32_t)(depth / (uint64_t)HQVM_N_LAYER);
}

uint32_t hqvm_genealogy_layer(uint64_t depth) {
    return (uint32_t)(depth % (uint64_t)HQVM_N_LAYER);
}

uint64_t hqvm_genealogy_depth_start(void) {
    return s_depth_start;
}

uint64_t hqvm_genealogy_depth_end(void) {
    return s_depth_end;
}

uint64_t hqvm_genealogy_step_count(void) {
    return s_step_count;
}

uint64_t hqvm_genealogy_span(void) {
    if (s_step_count == 0 || s_depth_min == UINT64_MAX) return 0;
    return s_depth_max - s_depth_min + 1ull;
}

int hqvm_genealogy_n_layer(void) {
    return HQVM_N_LAYER;
}

void hqvm_genealogy_counters_print(void) {
    fprintf(stderr,
        "[hqvm-gyro-clock] depth_start=%llu depth_end=%llu steps=%llu span=%llu "
        "L=%d layer=%d seq_len=%u\n",
        (unsigned long long)s_depth_start,
        (unsigned long long)s_depth_end,
        (unsigned long long)s_step_count,
        (unsigned long long)hqvm_genealogy_span(),
        HQVM_N_LAYER,
        s_layer % HQVM_N_LAYER,
        (unsigned)s_seq_len);
    fflush(stderr);
}

void hqvm_rope_clock_token_pos_set(uint32_t token_pos) {
    s_rope_clock_token_pos = token_pos;
}

uint32_t hqvm_rope_clock_token_pos_get(void) {
    return s_rope_clock_token_pos;
}

void hqvm_cgm_lift_counters_get(
    uint64_t *lift_calls, uint64_t *chi6_writes, uint64_t *invariant_fails)
{
    if (lift_calls) *lift_calls = s_lift_calls;
    if (chi6_writes) *chi6_writes = 0;
    if (invariant_fails) *invariant_fails = s_invariant_fails;
}

void hqvm_cgm_lift_counters_print(void) {
    fprintf(stderr,
        "[hqvm-cgm-lift] cgm_lift_calls=%llu invariant_fails=%llu "
        "phase_idx=%u state24=%06x table_ok=%d\n",
        (unsigned long long)s_lift_calls,
        (unsigned long long)s_invariant_fails,
        s_traj_init ? (unsigned)s_traj.phase_idx : 0u,
        s_traj_init ? (unsigned)(s_traj.state24 & 0xFFFFFFu) : 0u,
        s_table_ok);
    fflush(stderr);
    hqvm_genealogy_counters_print();
}

int hqvm_attn_scores_native_enabled(void) {
    static int s = -1;
    if (s < 0) {
        const char *e = getenv("GYRO_NATIVE_ATTN_SCORES");
        s = (e && e[0] == '1') ? 1 : 0;
    }
    return s;
}

int hqvm_vreduce_native_enabled(void) {
    static int s = -1;
    if (s < 0) {
        const char *e = getenv("GYRO_NATIVE_VREDUCE");
        s = (e && e[0] == '1') ? 1 : 0;
    }
    return s;
}

int hqvm_attn_head_scores_dyad(
    hqvm_dyad32_t *weights_dyad,
    const hqvm_dyad32_t *q_head_dyad,
    const void *k_q8_base,
    const float *k_f32_base,
    const float *k_fallback_f32,
    size_t k_row_stride,
    size_t floats_per_tok,
    size_t k_per_head_q8,
    const uint8_t *chi_layer,
    int64_t n_kv_heads,
    int kv_head,
    int64_t kv_len,
    uint8_t Nc,
    int top_k,
    int attn_level,
    float attn_scale)
{
    static int s_logged = 0;
    int native = hqvm_attn_scores_native_enabled();
    int rc;

    hqvm_gate_counters_inc_attn();
    if (!s_logged) {
        fprintf(stderr, "[hqvm-attn] mode=%s\n",
            native ? "NATIVE-dyad-q8" : "stock-float-QK");
        fflush(stderr);
        s_logged = 1;
    }
    if (native) {
        rc = hqvm_attn_head_scores_weights_native(
            weights_dyad, q_head_dyad, k_q8_base, k_f32_base, k_fallback_f32,
            k_row_stride, floats_per_tok, k_per_head_q8,
            chi_layer, n_kv_heads, kv_head, kv_len, Nc, top_k, attn_level, attn_scale);
    } else {
        rc = hqvm_attn_head_scores_weights_stock(
            weights_dyad, q_head_dyad, k_q8_base, k_f32_base, k_fallback_f32,
            k_row_stride, floats_per_tok, k_per_head_q8,
            chi_layer, n_kv_heads, kv_head, kv_len, Nc, top_k, attn_level, attn_scale);
    }
    return rc;
}

int hqvm_v_reduce_dyad(
    hqvm_dyad32_t *out_dyad,
    int64_t DV,
    const hqvm_dyad32_t *weights_dyad,
    int64_t Nk,
    const void *v_base,
    size_t v_row_stride,
    int v_is_q8)
{
    static int s_logged = 0;
    int native = hqvm_vreduce_native_enabled();
    int v_pert = hqvm_v_perturb_enabled();
    float *weights_f = NULL;
    float *out_f = NULL;
    int64_t j, d;
    int rc = 0;

    hqvm_gate_counters_inc_vreduce();
    if (!s_logged) {
        fprintf(stderr, "[hqvm-vreduce] mode=%s\n",
            (native && !v_pert) ? "NATIVE-dyad-q8" : "stock-float");
        fflush(stderr);
        s_logged = 1;
    }
    if (!out_dyad || !weights_dyad || !v_base || DV <= 0 || Nk <= 0) return -1;

    if (native && v_is_q8 && !v_pert) {
        return hqvm_attn_v_reduce_dyad_q8(
            out_dyad, DV, weights_dyad, Nk, v_base, v_row_stride);
    }

    weights_f = (float *)malloc((size_t)Nk * sizeof(float));
    out_f = (float *)malloc((size_t)DV * sizeof(float));
    if (!weights_f || !out_f) {
        free(weights_f);
        free(out_f);
        return -2;
    }
    for (j = 0; j < Nk; ++j) weights_f[j] = hqvm_dyad32_to_f32(weights_dyad[j]);
    hqvm_attn_v_reduce(
        out_f, DV, weights_f, Nk, v_base, v_row_stride, v_is_q8, v_pert);
    for (d = 0; d < DV; ++d) out_dyad[d] = hqvm_dyad32_from_f32(out_f[d]);
    free(weights_f);
    free(out_f);
    return rc;
}


