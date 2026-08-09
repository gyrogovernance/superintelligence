#include "attn.h"

#include "kernel.h"

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

/* ===== KV / Q8 ===== */
/*
 * hQVM KV coordinate ledger (Arc 2 Phases 3/6/7) + Arc 4B trajectory/receipts.
 */


#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_MSC_VER)
#  include <windows.h>
#else
#  include <stdatomic.h>
#endif

static hqvm_kv_ledger g_kv;
#if defined(_MSC_VER)
static volatile LONG g_kv_lock = 0;
#else
static atomic_flag g_kv_lock = ATOMIC_FLAG_INIT;
#endif

static void hqvm_kv_lock(void) {
#if defined(_MSC_VER)
    while (InterlockedCompareExchange(&g_kv_lock, 1, 0) != 0) { }
#else
    while (atomic_flag_test_and_set(&g_kv_lock)) { }
#endif
}

static void hqvm_kv_unlock(void) {
#if defined(_MSC_VER)
    InterlockedExchange(&g_kv_lock, 0);
#else
    atomic_flag_clear(&g_kv_lock);
#endif
}

static uint8_t hqvm_popcount8(uint8_t v) {
#if defined(_MSC_VER)
    return (uint8_t)__popcnt(v);
#else
    return (uint8_t)__builtin_popcount((unsigned)v);
#endif
}

static void hqvm_wht64(float data[64]) {
    int stride, i, j;
    for (stride = 32; stride >= 1; stride >>= 1) {
        for (i = 0; i < 64; i += 2 * stride) {
            for (j = 0; j < stride; ++j) {
                const float a = data[i + j];
                const float b = data[i + j + stride];
                data[i + j] = a + b;
                data[i + j + stride] = a - b;
            }
        }
    }
}

hqvm_kv_ledger *hqvm_kv_ledger_global(void) {
    return &g_kv;
}

int hqvm_kv_ledger_enabled(void) {
    const char *e = getenv("GYRO_KV_LEDGER");
    return (e && e[0] && e[0] != '0') ? 1 : 0;
}

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
    fflush(stderr);
}

static int hqvm_streq_ci(const char *a, const char *b) {
    if (!a || !b) return 0;
    while (*a && *b) {
        char ca = *a, cb = *b;
        if (ca >= 'A' && ca <= 'Z') ca = (char)(ca - 'A' + 'a');
        if (cb >= 'A' && cb <= 'Z') cb = (char)(cb - 'A' + 'a');
        if (ca != cb) return 0;
        ++a; ++b;
    }
    return *a == 0 && *b == 0;
}

/* Mode chosen once at first call — no hot-path string compares after. */
static int g_hol_mode = -1;
static unsigned g_hol_seed = 1u;

int hqvm_holonomic_attn_mode(unsigned *seed_out) {
    if (g_hol_mode < 0) {
        const char *e = getenv("GYRO_HOLONOMIC_ATTN_MODE");
        g_hol_mode = HQVM_HOL_MODE_DOT;
        g_hol_seed = 1u;
        if (e && e[0]) {
            if (hqvm_streq_ci(e, "zero_scores")) {
                g_hol_mode = HQVM_HOL_MODE_ZERO;
            } else if (strncmp(e, "random_scores", 13) == 0) {
                const char *p = e + 13;
                if (*p == ':' || *p == '(') {
                    unsigned long v = strtoul(p + 1, NULL, 10);
                    if (v != 0) g_hol_seed = (unsigned)v;
                }
                g_hol_mode = HQVM_HOL_MODE_RANDOM;
            } else if (hqvm_streq_ci(e, "dot") || hqvm_streq_ci(e, "int8")) {
                g_hol_mode = HQVM_HOL_MODE_DOT;
            }
        }
        fprintf(stderr, "[hqvm-holonomic] mode_init=%d seed=%u\n", g_hol_mode, g_hol_seed);
        fflush(stderr);
    }
    if (seed_out) *seed_out = g_hol_seed;
    return g_hol_mode;
}

void hqvm_kv_project_plane64(const float *plane64, gyro_kv_coord_t *out) {
    float data[64];
    float best_mag = -1.0f;
    int best_k = 0;
    int k;
    float sumsq = 0.0f;
    float amax = 0.0f;

    if (!plane64 || !out) return;
    memset(out, 0, sizeof(*out));
    for (k = 0; k < 64; ++k) {
        const float x = plane64[k];
        float ax = x < 0.0f ? -x : x;
        data[k] = (x >= 0.0f) ? 1.0f : -1.0f;
        sumsq += x * x;
        if (ax > amax) amax = ax;
    }
    hqvm_wht64(data);
    for (k = 0; k < 64; ++k) {
        float mag = data[k] < 0.0f ? -data[k] : data[k];
        if (mag > best_mag) {
            best_mag = mag;
            best_k = k;
        }
    }
    out->chi6 = (uint8_t)(best_k & 63);
    out->mean_abs = sqrtf(sumsq);
    out->d = (amax > 0.0f) ? (amax / 127.0f) : 0.0f;
    {
        const float id = (out->d > 0.0f) ? (1.0f / out->d) : 0.0f;
        for (k = 0; k < 64; ++k) {
            int v = (int)lrintf(plane64[k] * id);
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            out->q8[k] = (int8_t)v;
        }
    }
}

void hqvm_kv_project_head128(const float *head128, gyro_kv_cell_t *out) {
    if (!head128 || !out) return;
    hqvm_kv_project_plane64(head128, &out->key_coords[0]);
    hqvm_kv_project_plane64(head128 + 64, &out->key_coords[1]);
}

void hqvm_kv_ledger_reset(hqvm_kv_ledger *L) {
    if (!L) return;
    hqvm_kv_lock();
    free(L->recs);
    free(L->coords);
    memset(L, 0, sizeof(*L));
    hqvm_kv_unlock();
}

uint32_t hqvm_kv_ledger_count(const hqvm_kv_ledger *L) {
    return L ? L->n_rec : 0;
}

uint32_t hqvm_kv_ledger_coord_count(const hqvm_kv_ledger *L) {
    return L ? L->n_coord : 0;
}

int hqvm_kv_ledger_store_coord(hqvm_kv_ledger *L, const gyro_kv_coord_t *coord) {
    if (!L || !coord) return -1;
    hqvm_kv_lock();
    if (!L->coords) {
        L->coord_cap = 4096;
        L->coords = (gyro_kv_coord_t *)malloc((size_t)L->coord_cap * sizeof(gyro_kv_coord_t));
        if (!L->coords) {
            hqvm_kv_unlock();
            return -1;
        }
    }
    if (L->n_coord >= L->coord_cap) {
        uint32_t ncap = L->coord_cap * 2u;
        gyro_kv_coord_t *nr;
        if (ncap > HQVM_KV_REC_MAX) ncap = HQVM_KV_REC_MAX;
        if (L->n_coord >= ncap) {
            hqvm_kv_unlock();
            return -1;
        }
        nr = (gyro_kv_coord_t *)realloc(L->coords, (size_t)ncap * sizeof(gyro_kv_coord_t));
        if (!nr) {
            hqvm_kv_unlock();
            return -1;
        }
        L->coords = nr;
        L->coord_cap = ncap;
    }
    L->coords[L->n_coord++] = *coord;
    hqvm_kv_unlock();
    return 0;
}

void hqvm_kv_ledger_gather_histogram(const hqvm_kv_ledger *L, float H[HQVM_KV_CHI_BINS]) {
    uint32_t i;
    if (!H) return;
    for (i = 0; i < HQVM_KV_CHI_BINS; ++i) H[i] = 0.0f;
    if (!L || !L->coords) return;
    for (i = 0; i < L->n_coord; ++i) {
        H[L->coords[i].chi6 & 63] += 1.0f;
    }
}

int hqvm_kv_ledger_append_f32(
    hqvm_kv_ledger *L,
    uint32_t token_i,
    const float *x,
    int64_t n)
{
    gyro_kv_coord_t coord;
    uint8_t chi, shell, dchi;
    hqvm_kv_rec *rec;
    int64_t n64;

    if (!L || !x || n <= 0) return -1;

    n64 = n < 64 ? n : 64;
    {
        float plane[64];
        int64_t i;
        for (i = 0; i < 64; ++i) {
            plane[i] = (i < n64) ? x[i] : 0.0f;
        }
        hqvm_kv_project_plane64(plane, &coord);
    }
    chi = coord.chi6;
    shell = hqvm_popcount8(chi);
    (void)hqvm_kv_ledger_store_coord(L, &coord);

    hqvm_kv_lock();
    if (!L->recs) {
        L->cap = 4096;
        L->recs = (hqvm_kv_rec *)malloc((size_t)L->cap * sizeof(hqvm_kv_rec));
        if (!L->recs) {
            hqvm_kv_unlock();
            return -1;
        }
    }
    if (L->n_rec >= L->cap) {
        uint32_t ncap = L->cap * 2u;
        hqvm_kv_rec *nr;
        if (ncap > HQVM_KV_REC_MAX) ncap = HQVM_KV_REC_MAX;
        if (L->n_rec >= ncap) {
            hqvm_kv_unlock();
            return -1;
        }
        nr = (hqvm_kv_rec *)realloc(L->recs, (size_t)ncap * sizeof(hqvm_kv_rec));
        if (!nr) {
            hqvm_kv_unlock();
            return -1;
        }
        L->recs = nr;
        L->cap = ncap;
    }
    dchi = L->has_prev ? (uint8_t)(chi ^ L->prev_chi) : chi;
    rec = &L->recs[L->n_rec++];
    rec->token_i = token_i;
    rec->chi6 = chi;
    rec->shell = shell;
    rec->delta_chi = dchi;
    rec->_pad = 0;
    rec->mean_abs = coord.mean_abs;
    L->prev_chi = chi;
    L->has_prev = 1;
    hqvm_kv_unlock();
    return 0;
}

/* Removed (dead, superseded by Q8_0 cache scoring):
 * hqvm_kv_ledger_load_khat, hqvm_kv_chi_score (Khat never used by live score),
 * hqvm_pack_key_coord / hqvm_key_coord_score_* (Arc 2B-1 ring, superseded),
 * hqvm_kv_keys_{ensure,store,get} (key ring never read by holonomic path),
 * hqvm_kv_shadow_{accum,print} (coord_shadow accumulators never queried). */

/* Arc 2B-2: score directly from ggml Q8_0 K-cache blocks.
 * Layout per block (34 B): fp16 scale + 32 int8 quants. 4 blocks per head128. */
typedef struct { uint16_t d; int8_t q[HQVM_Q8_BLOCK]; } hqvm_q8_blk;

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

/* Arc 2B-2/3C: quantize one F16/F32 row into Q8_0 blocks (set_rows write path).
 * n dims must be a multiple of HQVM_Q8_BLOCK. fp16 scale + int8 quants. */
void hqvm_quantize_row_q8(
    const float *row,
    int64_t n,
    void *out_q8)
{
    hqvm_q8_blk *blk = (hqvm_q8_blk *)out_q8;
    int64_t b;
    int i;
    if (!row || !out_q8 || n <= 0) return;
    for (b = 0; b < n / HQVM_Q8_BLOCK; ++b) {
        float amax = 0.0f;
        int64_t imax = 0;
        float d, id;
        for (i = 0; i < HQVM_Q8_BLOCK; ++i) {
            const float ax = fabsf(row[b * HQVM_Q8_BLOCK + i]);
            if (ax > amax) { amax = ax; imax = i; }
        }
        d = row[b * HQVM_Q8_BLOCK + imax] / 127.0f;
        id = d ? 1.0f / d : 0.0f;
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
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
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

/* Coverage table: r -> |Reach| (rank-5 parity plateau 1024). */
static const int64_t HQVM_REACH_BY_RANK[7] = {2, 4, 16, 64, 256, 1024, 4096};
static uint64_t s_gate_sum = 0, s_gate_n = 0;
static int s_gate_layer = -1;
static uint64_t s_gate_sum_l[36], s_gate_n_l[36];

void hqvm_percolation_gates_report(void) {
    if (s_gate_n == 0) return;
    fprintf(stderr, "[hqvm-percolation] gates=%llu/%llu (%.4f)\n", (unsigned long long)s_gate_sum, (unsigned long long)s_gate_n, (double)s_gate_sum / (double)s_gate_n);
}

void hqvm_percolation_gates(
    const float *q_head128,
    const void *k_base_q8,
    size_t k_row_stride,
    int64_t Nk,
    uint8_t *gates)
{
    uint8_t chi_q[2];
    int64_t j;
    uint64_t qsigns0 = 0, qsigns1 = 0;
    int i;
    if (!q_head128 || !k_base_q8 || !gates || Nk <= 0) return;
    for (i = 0; i < 64; ++i) {
        if (q_head128[i] >= 0.0f) qsigns0 |= (1ull << i);
        if (q_head128[64 + i] >= 0.0f) qsigns1 |= (1ull << i);
    }
    chi_q[0] = gyroscopic_chirality_from_signs64(qsigns0);
    chi_q[1] = gyroscopic_chirality_from_signs64(qsigns1);

    {
        int basis[6] = {0,0,0,0,0,0};
        int rank = 0;
        for (j = 0; j < Nk; ++j) {
            uint8_t chi_k[2];
            uint8_t t[2];
            const char *krow = (const char *) k_base_q8 + (size_t)j * k_row_stride;
            hqvm_krow_chi_pair(krow, chi_k);
            t[0] = (uint8_t)(chi_q[0] ^ chi_k[0]);
            t[1] = (uint8_t)(chi_q[1] ^ chi_k[1]);
            if (j == 0) {
                gates[0] = 1;
            } else {
                int b, grew = 0;
                for (i = 0; i < 2; ++i) {
                    int x = t[i];
                    for (b = 5; b >= 0; --b) {
                        if (!(x & (1 << b))) continue;
                        if (!basis[b]) { basis[b] = x; rank++; grew = 1; break; }
                        x ^= basis[b];
                    }
                }
                gates[j] = grew ? 1 : 0;
            }
            s_gate_sum += gates[j];
            s_gate_n++;
            if (s_gate_layer >= 0 && s_gate_layer < 36) {
                s_gate_sum_l[s_gate_layer] += gates[j];
                s_gate_n_l[s_gate_layer]++;
            }
        }
    }
}

int hqvm_v_perturb_enabled(void) {
    static int s = -1;
    if (s < 0) {
        const char *e = getenv("GYRO_V_PERTURB");
        s = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    return s;
}

int hqvm_percolation_shadow_enabled(void) {
    static int s = -1;
    if (s < 0) {
        const char *e = getenv("GYRO_PERCOLATION_SOFTMAX");
        s = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    return s;
}

int hqvm_percolation_enabled(void) {
    static int s = -1;
    if (s < 0) {
        const char *e = getenv("GYRO_PERCOLATION");
        s = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    return s;
}

int hqvm_receipts_enabled(void) {
    static int s = -1;
    if (s < 0) {
        const char *e = getenv("GYRO_RECEIPTS");
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

/* Shell-occupation softmax (QuBEC §2.2, §23.1). Replaces exp(score) with the
 * native polynomial shell weight: w_j = lambda^N_j, N_j = popcount(chi_Q xor
 * chi_K). Partition (1+lambda)^6 is exact/finite; no exp, no sqrt, no percolation
 * gate. lambda is the shell occupation control (rho = lambda/(1+lambda)).
 * N convention: chi = q XOR k, so N=0 (max alignment) gets weight 1 and N=6 gets
 * lambda^6 — lambda < 1 sharpens toward aligned keys. */
/* Aperture-constrained softmax (YM mass-gap Δ as irreducible opening).
 * Softmax stays as the max-entropy magnitude decoder; CGM enters as a global
 * aperture: r = GF(2) rank of the transport set {chi_Q xor chi_K_i} over the
 * window, deficit (6-r) * Delta sets the mixing weight eps toward uniform.
 * r is one scalar per window (no per-key gating). exp is not replaced. */
void hqvm_aperture_softmax(
    float *logits, const float *q_head128, const void *k_base_q8,
    size_t k_row_stride, int64_t Nk, float Delta, float eps_max,
    const void *k_chi_base, int kv_head)
{
    int64_t j;
    uint8_t chi_q[2];
    uint64_t qsigns0 = 0, qsigns1 = 0;
    int i;
    float m = -INFINITY, sum = 0.0f;
    int basis0[6] = {0,0,0,0,0,0}, basis1[6] = {0,0,0,0,0,0};
    int r0 = 0, r1 = 0, r, deficit;
    float eps, uni;
    const int use_store = (k_chi_base != NULL) && hqvm_k_chi6_has(k_chi_base);
    if (!logits || !q_head128 || !k_base_q8 || Nk <= 0) return;
    for (i = 0; i < 64; ++i) {
        if (q_head128[i] >= 0.0f) qsigns0 |= (1ull << i);
        if (q_head128[64 + i] >= 0.0f) qsigns1 |= (1ull << i);
    }
    chi_q[0] = gyroscopic_chirality_from_signs64(qsigns0);
    chi_q[1] = gyroscopic_chirality_from_signs64(qsigns1);

    for (j = 0; j < Nk; ++j) {
        uint8_t chi_k[2];
        int b, x;
        if (logits[j] <= -1e20f) continue;
        if (use_store) {
            chi_k[0] = hqvm_k_chi6_get(k_chi_base, j, kv_head);
            chi_k[1] = chi_k[0];
        } else {
            const char *krow = (const char *) k_base_q8 + (size_t)j * k_row_stride;
            hqvm_krow_chi_pair(krow, chi_k);
        }
        x = (chi_q[0] ^ chi_k[0]) & 63;
        for (b = 5; b >= 0; --b) {
            if (!(x & (1 << b))) continue;
            if (!basis0[b]) { basis0[b] = x; r0++; break; }
            x ^= basis0[b];
        }
        x = (chi_q[1] ^ chi_k[1]) & 63;
        for (b = 5; b >= 0; --b) {
            if (!(x & (1 << b))) continue;
            if (!basis1[b]) { basis1[b] = x; r1++; break; }
            x ^= basis1[b];
        }
    }
    r = (r0 + r1 + 1) / 2;
    deficit = 6 - r;
    if (deficit < 0) deficit = 0;
    eps = (float)deficit * Delta;
    if (eps > eps_max) eps = eps_max;
    if (eps < 0.0f) eps = 0.0f;
    uni = 1.0f / (float)Nk;

    for (j = 0; j < Nk; ++j) if (logits[j] > m) m = logits[j];
    for (j = 0; j < Nk; ++j) {
        const float e = logits[j] > -1e20f ? expf(logits[j] - m) : 0.0f;
        logits[j] = e;
        sum += e;
    }
    if (sum > 0.0f) {
        for (j = 0; j < Nk; ++j)
            logits[j] = (1.0f - eps) * (logits[j] / sum) + eps * uni;
    } else {
        for (j = 0; j < Nk; ++j) logits[j] = uni;
    }
}

int hqvm_aperture_enabled(void) {
    static int s = -1;
    if (s < 0) {
        const char *e = getenv("GYRO_APERTURE_SOFTMAX");
        s = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    return s;
}

/* Shadow: run aperture softmax on a copy, compare to stock on the same logits.
 * Logs KL(stock||ap), top-1 agreement, eps, r. Measurement only. */
void hqvm_aperture_shadow(
    const float *logits, const float *q_head128, const void *k_base_q8,
    size_t k_row_stride, int64_t Nk, float Delta, float eps_max,
    const void *k_chi_base, int kv_head)
{
    float *ap, *st;
    int64_t j;
    float m = -INFINITY, ssum = 0.0f;
    double kl = 0.0;
    int top_s = -1, top_a = -1;
    float bs = -1.0f, ba = -1.0f;
    static int s_print = 0;
    if (!logits || !q_head128 || !k_base_q8 || Nk <= 0) return;
    ap = (float *) malloc((size_t)Nk * sizeof(float));
    st = (float *) malloc((size_t)Nk * sizeof(float));
    if (!ap || !st) { free(ap); free(st); return; }
    memcpy(ap, logits, (size_t)Nk * sizeof(float));
    hqvm_aperture_softmax(ap, q_head128, k_base_q8, k_row_stride, Nk, Delta, eps_max,
                          k_chi_base, kv_head);
    for (j = 0; j < Nk; ++j) {
        if (logits[j] > m) m = logits[j];
    }
    for (j = 0; j < Nk; ++j) {
        st[j] = logits[j] > -1e20f ? expf(logits[j] - m) : 0.0f;
        ssum += st[j];
    }
    if (ssum > 0.0f) for (j = 0; j < Nk; ++j) st[j] /= ssum;
    for (j = 0; j < Nk; ++j) {
        const float p = st[j], q = ap[j];
        if (p > 0.0f && q > 0.0f) kl += p * logf(p / q);
        if (p > bs) { bs = p; top_s = (int)j; }
        if (q > ba) { ba = q; top_a = (int)j; }
    }
    if (s_print < 40) {
        fprintf(stderr, "[hqvm-aperture] KL=%.4f top1_agree=%d eps_top_s=%.4g eps_top_a=%.4g\n",
            kl, top_s == top_a, bs, ba);
        s_print++;
    }
    free(ap);
    free(st);
}

int hqvm_aperture_rope_enabled(void) {
    static int s = -1;
    if (s < 0) {
        const char *e = getenv("GYRO_APERTURE_ROPE");
        s = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    return s;
}

int hqvm_aperture_rms_enabled(void) {
    static int s = -1;
    if (s < 0) {
        const char *e = getenv("GYRO_APERTURE_RMS");
        s = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    return s;
}


void hqvm_shell_softmax(
    float *logits, const float *q_head128, const void *k_base_q8,
    size_t k_row_stride, int64_t Nk, float lambda)
{
    int64_t j;
    uint8_t chi_q[2];
    uint64_t qsigns0 = 0, qsigns1 = 0;
    int i;
    float lam_pow[7];
    float sum = 0.0f;
    if (!logits || !q_head128 || !k_base_q8 || Nk <= 0) return;
    for (i = 0; i < 64; ++i) {
        if (q_head128[i] >= 0.0f) qsigns0 |= (1ull << i);
        if (q_head128[64 + i] >= 0.0f) qsigns1 |= (1ull << i);
    }
    chi_q[0] = gyroscopic_chirality_from_signs64(qsigns0);
    chi_q[1] = gyroscopic_chirality_from_signs64(qsigns1);
    lam_pow[0] = 1.0f;
    for (i = 1; i <= 6; ++i) lam_pow[i] = lam_pow[i - 1] * lambda;
    for (j = 0; j < Nk; ++j) {
        uint8_t chi_k[2];
        const char *krow = (const char *) k_base_q8 + (size_t)j * k_row_stride;
        int N0, N1, N;
        hqvm_krow_chi_pair(krow, chi_k);
        N0 = gyroscopic_chirality_distance(chi_q[0], chi_k[0]);
        N1 = gyroscopic_chirality_distance(chi_q[1], chi_k[1]);
        N = (N0 + N1) / 2;
        logits[j] = lam_pow[N & 6];
        sum += logits[j];
    }
    if (sum > 0.0f) {
        for (j = 0; j < Nk; ++j) logits[j] /= sum;
    } else {
        for (j = 0; j < Nk; ++j) logits[j] = 1.0f / (float)Nk;
    }
}

int hqvm_shell_softmax_enabled(void) {
    static int s = -1;
    if (s < 0) {
        const char *e = getenv("GYRO_SHELL_SOFTMAX");
        s = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    return s;
}

float hqvm_shell_softmax_lambda(void) {
    static float s_lam = -1.0f;
    if (s_lam < 0.0f) {
        const char *e = getenv("GYRO_SHELL_LAMBDA");
        s_lam = e ? (float) atof(e) : 0.5f;
        if (s_lam <= 0.0f) s_lam = 0.5f;
    }
    return s_lam;
}

/* REJECTED (Arc 4): percolation θ as softmax gate — category error.
 * Reachability climate is not per-key energy. Kept behind flag for audit only. */
void hqvm_percolation_softmax(
    float *logits, const float *q_head128, const void *k_base_q8,
    size_t k_row_stride, int64_t Nk)
{
    int64_t j;
    float m = -INFINITY, sum = 0.0f;
    uint8_t *gates;
    if (!logits || !q_head128 || !k_base_q8 || Nk <= 0) return;
    gates = (uint8_t *) malloc((size_t)Nk);
    if (!gates) return;
    hqvm_percolation_gates(q_head128, k_base_q8, k_row_stride, Nk, gates);
    for (j = 0; j < Nk; ++j) if (logits[j] > m) m = logits[j];
    for (j = 0; j < Nk; ++j) {
        const float e = logits[j] > -1e20f ? expf(logits[j] - m) : 0.0f;
        logits[j] = gates[j] ? e : 0.0f;
        sum += logits[j];
    }
    if (sum > 0.0f) {
        for (j = 0; j < Nk; ++j) logits[j] /= sum;
    } else {
        for (j = 0; j < Nk; ++j) logits[j] = 1.0f / (float)Nk;
    }
    free(gates);
}

void hqvm_percolation_shadow(const float *raw_scores, int64_t Nk, float M) {
    /* θ(gap) = exp(-2*gap) with gap = M - raw_score; compare KL vs stock softmax.
     * Measurement only — does not change committed weights. */
    float *theta = NULL;
    float *stock = NULL;
    float tsum = 0.0f, ssum = 0.0f;
    double kl = 0.0;
    int top_stock = -1, top_perc = -1;
    float bs = -1.0f, bp = -1.0f;
    int64_t i;
    static int s_print = 0;

    if (!raw_scores || Nk <= 0 || M <= -1e20f) return;
    theta = (float *) malloc((size_t)Nk * sizeof(float));
    stock = (float *) malloc((size_t)Nk * sizeof(float));
    if (!theta || !stock) {
        free(theta);
        free(stock);
        return;
    }
    for (i = 0; i < Nk; ++i) {
        float th = 0.0f, st = 0.0f;
        if (raw_scores[i] > -1e20f) {
            const float gap = M - raw_scores[i];
            th = expf(-2.0f * gap);
            st = expf(raw_scores[i] - M);
        }
        theta[i] = th;
        stock[i] = st;
        tsum += th;
        ssum += st;
    }
    {
        const float invT = tsum > 0.0f ? 1.0f / tsum : 0.0f;
        const float invS = ssum > 0.0f ? 1.0f / ssum : 0.0f;
        for (i = 0; i < Nk; ++i) {
            const float p = stock[i] * invS;
            const float q = theta[i] * invT;
            if (p > bs) { bs = p; top_stock = (int)i; }
            if (q > bp) { bp = q; top_perc = (int)i; }
            if (p > 1e-12f && q > 1e-12f) kl += (double)p * log((double)p / (double)q);
        }
    }
    s_print += 1;
    if (s_print == 1 || (s_print % 36) == 0) {
        fprintf(stderr,
                "[hqvm-perc] KL(stock||perc)=%.4f top1_stock=%d top1_perc=%d agree=%d\n",
                kl, top_stock, top_perc, top_stock == top_perc ? 1 : 0);
        fflush(stderr);
    }
    free(theta);
    free(stock);
}

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


void hqvm_receipts_on_layer(uint8_t intron_byte, int layer_i, int64_t Nk) {
    (void)intron_byte;
    (void)Nk;
    if (!hqvm_receipts_enabled()) return;
    /* Single carrier: seal from lift traj only. No parallel trajectory. */
    if (!hqvm_cgm_lift_traj_ready()) {
        static int s_note = 0;
        if (s_note < 1) {
            fprintf(stderr, "[hqvm-receipt] skip: canonical traj not ready (enable GYRO_CGM_LIFT)\n");
            fflush(stderr);
            s_note = 1;
        }
        return;
    }
    {
        hqvm_receipt_t r;
        r.anchor12 = (uint16_t)(hqvm_cgm_lift_state24() & LAYER_MASK_12);
        r.k4_family = (uint8_t)((hqvm_cgm_lift_state24() >> 12) & FAMILY_MASK);
        r.state24 = hqvm_cgm_lift_state24();
        r.depth = (uint32_t)(hqvm_cgm_lift_layer() + 1);
        r.fnv1a = hqvm_receipt_seal(&r);
        if ((layer_i % GYROSCOPIC_DEFAULT_TOTAL_LAYERS) == (GYROSCOPIC_DEFAULT_TOTAL_LAYERS - 1)) {
            hqvm_receipt_print(&r);
        }
    }
}

void hqvm_residual_shadow_log(const float *row, int64_t n, int is_f16) {
    static int s_en = -1;
    static int s_print = 0;
    static int s_depth = 0;
    double s2 = 0.0;
    int64_t i, nn;
    (void)is_f16;
    if (s_en < 0) {
        const char *e = getenv("GYRO_RESIDUAL_SHADOW");
        s_en = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    if (!s_en || !row || n <= 0) return;
    s_print += 1;
    if (!(s_print == 1 || (s_print % 72) == 0)) return;
    nn = n <= (int64_t)OMEGA_SIZE ? n : (int64_t)OMEGA_SIZE;
    for (i = 0; i < nn; ++i) s2 += (double)row[i] * (double)row[i];
    s_depth += 1;
    fprintf(stderr, "[hqvm-resid] add_call=%d depth=%d rms=%.4f ne0=%lld\n",
            s_print, s_depth, (nn > 0 ? sqrt(s2 / (double)nn) : 0.0), (long long)n);
    fflush(stderr);
}

void hqvm_shell_norm_shadow_log(const float *x, int64_t n) {
    static int s_en = -1;
    static int s_print = 0;
    double s2 = 0.0, rms;
    uint64_t signbits = 0;
    int64_t i, nb, nn;
    uint8_t chi6;
    int shell;
    if (s_en < 0) {
        const char *e = getenv("GYRO_SHELL_NORM");
        s_en = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    if (!s_en || !x || n <= 0) return;
    s_print += 1;
    if (!(s_print == 1 || (s_print % 36) == 0)) return;
    nn = n <= (int64_t)OMEGA_SIZE ? n : (int64_t)OMEGA_SIZE;
    for (i = 0; i < nn; ++i) s2 += (double)x[i] * (double)x[i];
    rms = sqrt(s2 / (nn > 0 ? (double)nn : 1.0));
    nb = nn < 64 ? nn : 64;
    for (i = 0; i < nb; ++i) if (x[i] >= 0.0f) signbits |= (1ull << i);
    chi6 = gyroscopic_chirality_from_signs64(signbits);
#if defined(_MSC_VER)
    shell = (int)__popcnt((unsigned)chi6);
#else
    shell = __builtin_popcount((unsigned)chi6);
#endif
    fprintf(stderr, "[hqvm-shell] rms=%.4f chi6=%u shell=%d\n",
            rms, (unsigned)chi6, shell);
    fflush(stderr);
}


/* ===== CGM-lift — owns the single trajectory instance ===== */


#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_MSC_VER)
#  include <intrin.h>
#endif

/* BYTE_OF_Q6_FAM[q6][fam] — inverse of intron = byte^GENE_MIC_S split. */
static uint8_t s_byte_of_q6_fam[64][4];
static int s_init = 0;
static int s_table_ok = 0;

static uint64_t s_lift_calls = 0;
static uint64_t s_chi6_writes = 0;
static uint64_t s_invariant_fails = 0;

/* chi6 side-store: identify stream by K tensor base pointer. */
#define HQVM_CHI_STREAMS 96
#define HQVM_CHI_CAP     8192
typedef struct {
    const void *base;
    uint8_t chi[HQVM_CHI_CAP][HQVM_KV_N_KV_HEAD];
    int64_t max_idx;
} hqvm_chi_stream_t;

static hqvm_chi_stream_t s_chi[HQVM_CHI_STREAMS];
static int s_chi_n = 0;

static gyro_trajectory_state_t s_traj;
static int s_traj_init = 0;

static uint8_t fam_of_intron(uint8_t intron) {
    return (uint8_t)((intron & 1u) | ((intron >> 6) & 2u));
}

static uint8_t q6_of_intron(uint8_t intron) {
    return (uint8_t)((intron >> 1) & CHIRALITY_MASK_6);
}

void hqvm_cgm_lift_init(void) {
    int b, ok = 1;
    int q, f;
    if (s_init) return;
    memset(s_byte_of_q6_fam, 0, sizeof(s_byte_of_q6_fam));
    for (b = 0; b < 256; ++b) {
        const uint8_t intron = (uint8_t)(b ^ (int)GENE_MIC_S);
        const uint8_t q6 = q6_of_intron(intron);
        const uint8_t fam = fam_of_intron(intron);
        s_byte_of_q6_fam[q6][fam] = (uint8_t)b;
    }
    for (b = 0; b < 256; ++b) {
        const uint8_t intron = (uint8_t)(b ^ (int)GENE_MIC_S);
        const uint8_t q6 = q6_of_intron(intron);
        const uint8_t fam = fam_of_intron(intron);
        if (s_byte_of_q6_fam[q6][fam] != (uint8_t)b) ok = 0;
        if (q6_of_intron((uint8_t)(b ^ (int)GENE_MIC_S)) != q6) ok = 0;
        if (fam_of_intron((uint8_t)(b ^ (int)GENE_MIC_S)) != fam) ok = 0;
    }
    for (q = 0; q < 64; ++q) {
        for (f = 0; f < 4; ++f) {
            const uint8_t byte = s_byte_of_q6_fam[q][f];
            const uint8_t intron = (uint8_t)(byte ^ (int)GENE_MIC_S);
            if (q6_of_intron(intron) != (uint8_t)q) ok = 0;
            if (fam_of_intron(intron) != (uint8_t)f) ok = 0;
        }
    }
    s_table_ok = ok;
    if (!ok) {
        fprintf(stderr, "[hqvm-cgm-lift] BYTE_OF_Q6_FAM verify FAIL\n");
    } else {
        fprintf(stderr, "[hqvm-cgm-lift] BYTE_OF_Q6_FAM verify PASS\n");
    }
    s_init = 1;
}

uint8_t hqvm_byte_of_q6_fam(uint8_t q6, uint8_t fam) {
    if (!s_init) hqvm_cgm_lift_init();
    return s_byte_of_q6_fam[q6 & 63][fam & 3];
}

uint8_t hqvm_q6_of_byte(uint8_t byte) {
    return q6_of_intron((uint8_t)(byte ^ (int)GENE_MIC_S));
}

uint8_t hqvm_fam_of_byte(uint8_t byte) {
    return fam_of_intron((uint8_t)(byte ^ (int)GENE_MIC_S));
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

int hqvm_cgm_lift_perturb_enabled(void) {
    static int s = -1;
    if (s < 0) {
        const char *e = getenv("GYRO_CGM_LIFT_PERTURB");
        s = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    return s;
}

static int s_layer = 0;
static uint8_t s_last_byte = 0;
static uint64_t s_residual_hits = 0;

int hqvm_cgm_lift_layer(void) {
    return s_layer % 36;
}

int hqvm_cgm_lift_bump_layer(void) {
    s_layer = (s_layer + 1) % 36;
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

int hqvm_residual_hybrid_enabled(void) {
    static int s = -1;
    if (s < 0) {
        const char *e = getenv("GYRO_RESIDUAL_HYBRID");
        s = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    return s;
}

float hqvm_residual_gain(void) {
    uint8_t chi6;
    int shell;
    float m;
    if (!s_traj_init || !hqvm_cgm_lift_enabled()) return 1.0f;
    chi6 = gyroscopic_chirality_word6(s_traj.state24);
#if defined(_MSC_VER)
    shell = (int)__popcnt((unsigned)chi6);
#else
    shell = __builtin_popcount((unsigned)chi6);
#endif
    m = (float)(shell - 3) / 3.0f;
    return 1.0f + (float)APERTURE_GAP * m;
}

void hqvm_residual_hybrid_hit(void) {
    s_residual_hits++;
    if (s_residual_hits == 1ull || (s_residual_hits % 72ull) == 0ull) {
        hqvm_residual_hybrid_counters_print();
    }
}

uint64_t hqvm_residual_hybrid_hits(void) {
    return s_residual_hits;
}

void hqvm_residual_hybrid_counters_print(void) {
    fprintf(stderr,
        "[hqvm-residual-hybrid] hits=%llu gain=%.6f state24=%06x\n",
        (unsigned long long)s_residual_hits,
        (double)hqvm_residual_gain(),
        s_traj_init ? (unsigned)(s_traj.state24 & 0xFFFFFFu) : 0u);
    fflush(stderr);
}

void hqvm_cgm_lift_counters_get(
    uint64_t *lift_calls, uint64_t *chi6_writes, uint64_t *invariant_fails)
{
    if (lift_calls) *lift_calls = s_lift_calls;
    if (chi6_writes) *chi6_writes = s_chi6_writes;
    if (invariant_fails) *invariant_fails = s_invariant_fails;
}

void hqvm_cgm_lift_counters_print(void) {
    fprintf(stderr,
        "[hqvm-cgm-lift] cgm_lift_calls=%llu k_chi6_write_calls=%llu invariant_fails=%llu "
        "phase_idx=%u state24=%06x table_ok=%d\n",
        (unsigned long long)s_lift_calls,
        (unsigned long long)s_chi6_writes,
        (unsigned long long)s_invariant_fails,
        s_traj_init ? (unsigned)s_traj.phase_idx : 0u,
        s_traj_init ? (unsigned)(s_traj.state24 & 0xFFFFFFu) : 0u,
        s_table_ok);
    fflush(stderr);
}

static hqvm_chi_stream_t *chi_stream(const void *base, int create) {
    int i;
    if (!base) return NULL;
    for (i = 0; i < s_chi_n; ++i) {
        if (s_chi[i].base == base) return &s_chi[i];
    }
    if (!create || s_chi_n >= HQVM_CHI_STREAMS) return NULL;
    s_chi[s_chi_n].base = base;
    s_chi[s_chi_n].max_idx = -1;
    memset(s_chi[s_chi_n].chi, 0, sizeof(s_chi[s_chi_n].chi));
    return &s_chi[s_chi_n++];
}

void hqvm_k_chi6_store(
    const void *k_base, int64_t idx, const float *row_f32, int64_t n_heads, int64_t head_dim)
{
    hqvm_chi_stream_t *S;
    int64_t h;
    if (!k_base || !row_f32 || idx < 0 || idx >= HQVM_CHI_CAP) return;
    if (n_heads <= 0) n_heads = HQVM_KV_N_KV_HEAD;
    if (head_dim <= 0) head_dim = HQVM_KV_HEAD_DIM;
    S = chi_stream(k_base, 1);
    if (!S) return;
    s_chi6_writes++;
    {
        static int s_store_note = 0;
        if (s_store_note < 2) {
            fprintf(stderr, "[hqvm-cgm-lift] chi6_store base=%p idx=%lld n_heads=%lld\n",
                k_base, (long long)idx, (long long)n_heads);
            s_store_note++;
        }
    }
    for (h = 0; h < n_heads && h < HQVM_KV_N_KV_HEAD; ++h) {
        uint64_t signs = 0;
        int i;
        const float *plane = row_f32 + h * head_dim;
        for (i = 0; i < 64 && i < head_dim; ++i) {
            if (plane[i] >= 0.0f) signs |= (1ull << i);
        }
        S->chi[idx][h] = gyroscopic_chirality_from_signs64(signs);
    }
    if (idx > S->max_idx) S->max_idx = idx;
}

uint8_t hqvm_k_chi6_get(const void *k_base, int64_t idx, int head) {
    hqvm_chi_stream_t *S = chi_stream(k_base, 0);
    int i;
    if (idx < 0 || idx >= HQVM_CHI_CAP) return 0;
    if (head < 0 || head >= HQVM_KV_N_KV_HEAD) return 0;
    if (S) return S->chi[idx][head];
    for (i = 0; i < s_chi_n; ++i) {
        if (s_chi[i].max_idx >= idx) return s_chi[i].chi[idx][head];
    }
    return 0;
}

int hqvm_k_chi6_has(const void *k_base) {
    int i;
    if (chi_stream(k_base, 0) != NULL) return 1;
    for (i = 0; i < s_chi_n; ++i) if (s_chi[i].max_idx >= 0) return 1;
    return 0;
}

void hqvm_lift_attention_phase(
    const float *scores, const void *k_base, int64_t Nk, int head,
    uint8_t chi_q, int depth, float Delta, float eps_max, gyro_lift_attn_t *out)
{
    int64_t i, i_star = -1;
    float best = -INFINITY;
    uint8_t *qvals = NULL;
    int r = 0;
    float eps = 0.0f;
    uint8_t q6, fam, byte;
    static int s_print = 0;
    static int s_inv_note = 0;

    if (!out) return;
    memset(out, 0, sizeof(*out));
    out->chi_q = chi_q;
    out->argmax = -1;
    if (!scores || !k_base || Nk <= 0) return;
    if (!s_init) hqvm_cgm_lift_init();
    if (!s_traj_init) { hqvm_traj_reset(&s_traj); s_traj_init = 1; }

    for (i = 0; i < Nk; ++i) {
        if (scores[i] > best) { best = scores[i]; i_star = i; }
    }
    if (i_star < 0 || best <= -1e20f) return;

    out->argmax = (int)i_star;
    q6 = (uint8_t)((chi_q ^ hqvm_k_chi6_get(k_base, i_star, head)) & 63);
    /* Canonical depth-phase: fam from emission counter, not layer index. */
    fam = (uint8_t)(s_traj.phase_idx & 3u);
    out->phase_idx = s_traj.phase_idx;

    if (hqvm_cgm_lift_perturb_enabled()) {
        /* Flip q6 bit0 and fam — magnitude path unchanged; traj must diverge. */
        q6 = (uint8_t)((q6 ^ 1u) & 63);
        fam = (uint8_t)((fam ^ 1u) & 3);
    }

    byte = hqvm_byte_of_q6_fam(q6, fam);
    if (hqvm_q6_of_byte(byte) != q6 || hqvm_fam_of_byte(byte) != fam) {
        s_invariant_fails++;
        if (s_inv_note < 3) {
            fprintf(stderr,
                "[hqvm-cgm-lift] INVARIANT FAIL byte=%02x q6=%u/%u fam=%u/%u\n",
                (unsigned)byte, (unsigned)q6, (unsigned)hqvm_q6_of_byte(byte),
                (unsigned)fam, (unsigned)hqvm_fam_of_byte(byte));
            s_inv_note++;
        }
    }

    out->q6 = q6;
    out->fam = fam;
    out->byte = byte;
    s_last_byte = byte;

    qvals = (uint8_t *) malloc((size_t)Nk);
    if (qvals) {
        for (i = 0; i < Nk; ++i) {
            qvals[i] = (uint8_t)((chi_q ^ hqvm_k_chi6_get(k_base, i, head)) & 63);
        }
        r = hqvm_rank_gf2_6(qvals, Nk);
        free(qvals);
    }
    out->rank_r = (uint8_t)r;
    eps = (float)(6 - r) * Delta;
    if (eps < 0.0f) eps = 0.0f;
    if (eps > eps_max) eps = eps_max;
    out->eps = eps;

    hqvm_traj_step(&s_traj, out->byte);
    s_traj.phase_idx++;
    s_lift_calls++;
    out->state24 = s_traj.state24;

    if (s_print < 12) {
        fprintf(stderr,
            "[hqvm-cgm-lift] layer=%d head=%d argmax=%d q6=%u fam=%u byte=%02x "
            "phase=%u r=%d eps=%.4f state=%06x\n",
            depth, head, out->argmax, (unsigned)out->q6, (unsigned)out->fam,
            (unsigned)out->byte, (unsigned)out->phase_idx, r, eps,
            (unsigned)(s_traj.state24 & 0xFFFFFFu));
        s_print++;
    }
}

