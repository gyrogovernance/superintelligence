/*
 * hQVM temporal ledger (thin HQVMLEDS + Q1_0 forward).
 * step_uv6 matches science/src.family.step_uv(d=6).
 * Fat HQVMLEDG v2 deleted.
 */

#include "ledger.h"
#include "constants.h"
#include "attn.h"
#include "codec.h"
#include "kernel.h"

#include <math.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_MSC_VER)
#  include <intrin.h>
#endif
#if defined(__AVX2__) || defined(GGML_AVX2)
#  include <immintrin.h>
#endif

#define HQVM_D CHIRALITY_QUBITS_6
#define HQVM_MASK6 CHIRALITY_MASK_6

static uint8_t intron_from_byte6(uint8_t byte) {
    return (uint8_t)((byte ^ (uint8_t)GENE_MIC_S) & UINT8_MASK);
}

static uint8_t intron_micro6(uint8_t intron) {
    return (uint8_t)((intron >> 1) & HQVM_MASK6);
}

static uint8_t eps_a6(uint8_t intron) {
    return (intron & 1u) ? HQVM_MASK6 : 0u;
}

static uint8_t eps_b6(uint8_t intron) {
    return ((intron >> (HQVM_D + 1)) & 1u) ? HQVM_MASK6 : 0u;
}

void hqvm_step_uv6(uint8_t u, uint8_t v, uint8_t byte, uint8_t *u_out, uint8_t *v_out) {
    uint8_t intron = intron_from_byte6(byte);
    uint8_t micro = intron_micro6(intron);
    *u_out = (uint8_t)((v ^ eps_a6(intron)) & HQVM_MASK6);
    *v_out = (uint8_t)((u ^ micro ^ eps_b6(intron)) & HQVM_MASK6);
}

uint8_t hqvm_shell_uv6(uint8_t u, uint8_t v) {
    uint8_t chi = (uint8_t)((u ^ v) & HQVM_MASK6);
#if defined(_MSC_VER)
    return (uint8_t)__popcnt(chi);
#else
    return (uint8_t)__builtin_popcount((unsigned)chi);
#endif
}

/* ---- Thin sidecar + Q1_0 forward from ggml memory ---- */

#define HQVM_QK1_0 ((int)BOUNDARY_SIZE) /* ggml Q1_0 block width; coincides with BOUNDARY_SIZE on Bonsai */

typedef struct {
    uint16_t d;
    uint8_t qs[HQVM_QK1_0 / 8];
} hqvm_block_q1_0;

static int bin_shell_edges(const float *edges, uint32_t n_bin, float s) {
    uint32_t lo = 0, hi = n_bin + 1;
    while (lo < hi) {
        uint32_t mid = (lo + hi) / 2;
        if (edges[mid] <= s) lo = mid + 1;
        else hi = mid;
    }
    int idx = (int)lo - 1;
    if (idx < 0) idx = 0;
    if ((uint32_t)idx >= n_bin) idx = (int)n_bin - 1;
    return idx;
}

static float gyro_manifold_gain_2half(uint8_t chi_bit0, uint8_t p0, uint8_t chi_bit1, uint8_t p1) {
    return hqvm_manifold_gain_from_bits(chi_bit0, p0, chi_bit1, p1);
}

#if defined(_MSC_VER)
static int popcount64_(uint64_t x) {
    return (int)__popcnt64(x);
}
#else
static int popcount64_(uint64_t x) {
    return (int)__builtin_popcountll(x);
}
#endif

void hqvm_sidecar_free(hqvm_sidecar *S) {
    if (!S) return;
    free(S->byte_table);
    free(S->bin_edges);
    memset(S, 0, sizeof(*S));
}

int hqvm_sidecar_load(hqvm_sidecar *S, const char *path) {
    FILE *f;
    char magic[8];
    uint32_t i, n_bt, n_edges;
    uint16_t plen;

    if (!S || !path) return -1;
    memset(S, 0, sizeof(*S));
    f = fopen(path, "rb");
    if (!f) return -1;
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "HQVMLEDS", 8) != 0) {
        fclose(f);
        return -1;
    }
    if (fread(&S->version, 4, 1, f) != 1 || S->version != 1 ||
        fread(&S->n_bin, 4, 1, f) != 1 ||
        fread(&S->n_allow, 4, 1, f) != 1) {
        fclose(f);
        return -1;
    }
    if (S->n_allow > HQVM_SIDECAR_MAX_ALLOW) {
        fclose(f);
        return -1;
    }
    n_bt = 64u * S->n_bin;
    n_edges = S->n_bin + 1u;
    S->byte_table = (int16_t *)malloc(n_bt * sizeof(int16_t));
    S->bin_edges = (float *)malloc(n_edges * sizeof(float));
    if (!S->byte_table || !S->bin_edges) {
        hqvm_sidecar_free(S);
        fclose(f);
        return -1;
    }
    if (fread(S->byte_table, sizeof(int16_t), n_bt, f) != n_bt ||
        fread(S->bin_edges, sizeof(float), n_edges, f) != n_edges) {
        hqvm_sidecar_free(S);
        fclose(f);
        return -1;
    }
    for (i = 0; i < S->n_allow; ++i) {
        if (fread(&plen, 2, 1, f) != 1 || plen == 0 || plen >= HQVM_SIDECAR_ALLOW_LEN) {
            hqvm_sidecar_free(S);
            fclose(f);
            return -1;
        }
        if (fread(S->allow[i], 1, plen, f) != plen) {
            hqvm_sidecar_free(S);
            fclose(f);
            return -1;
        }
        S->allow[i][plen] = '\0';
    }
    fclose(f);
    return 0;
}

void hqvm_sidecar_apply_env_allow(hqvm_sidecar *S) {
    const char *env;
    char buf[512];
    char *tok;
    char *ctx = NULL;
    uint32_t n = 0;

    if (!S) return;
    env = getenv("GYRO_LEDGER_ALLOW");
    if (!env || !env[0]) return;
    strncpy(buf, env, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';
#if defined(_MSC_VER)
    tok = strtok_s(buf, ",;", &ctx);
#else
    tok = strtok_r(buf, ",;", &ctx);
#endif
    while (tok && n < HQVM_SIDECAR_MAX_ALLOW) {
        while (*tok == ' ' || *tok == '\t') ++tok;
        if (*tok) {
            strncpy(S->allow[n], tok, HQVM_SIDECAR_ALLOW_LEN - 1);
            S->allow[n][HQVM_SIDECAR_ALLOW_LEN - 1] = '\0';
            ++n;
        }
#if defined(_MSC_VER)
        tok = strtok_s(NULL, ",;", &ctx);
#else
        tok = strtok_r(NULL, ",;", &ctx);
#endif
    }
    S->n_allow = n;
}

int hqvm_sidecar_allows(const hqvm_sidecar *S, const char *tensor_name) {
    uint32_t i;
    if (!S || !tensor_name || !tensor_name[0]) return 0;
    for (i = 0; i < S->n_allow; ++i) {
        if (S->allow[i][0] && strstr(tensor_name, S->allow[i]) != NULL) {
            return 1;
        }
    }
    return 0;
}

void hqvm_quantize_dyad_q8(const hqvm_dyad32_t *x, int64_t n, int8_t *qx, uint16_t *xd_h) {
    int64_t i;
    if (!x || !qx || !xd_h || n <= 0 || (n % 32) != 0) { return; }
    for (i = 0; i < n; i += 32) {
        float lane[32];
        int j;
        for (j = 0; j < 32; ++j) {
            memcpy(&lane[j], &x[i + j].bits, sizeof(float));
        }
        hqvm_quantize_x_q8(lane, 32, qx + i, xd_h + i / 32);
    }
}

typedef struct { uint16_t d; int8_t q[32]; } hqvm_q8_blk_local;

void hqvm_quantize_dyad_row_q8(const hqvm_dyad32_t *row, int64_t n, void *out_q8) {
    hqvm_q8_blk_local *blk = (hqvm_q8_blk_local *)out_q8;
    int64_t b;
    if (!row || !out_q8 || n <= 0 || (n % 32) != 0) return;
    for (b = 0; b < n / 32; ++b) {
        int8_t qx[32];
        uint16_t xd;
        hqvm_quantize_dyad_q8(row + b * 32, 32, qx, &xd);
        blk[b].d = xd;
        memcpy(blk[b].q, qx, 32);
    }
}

/* Reused across hqvm_matmul_dyad calls (max n_cols = FFN inner). */
static int8_t  *s_matmul_qx = NULL;
static uint16_t *s_matmul_xd = NULL;
static int64_t   s_matmul_cap_cols = 0;

static int matmul_scratch_ensure(int64_t n_cols) {
    if (n_cols <= s_matmul_cap_cols && s_matmul_qx && s_matmul_xd) {
        return 0;
    }
    free(s_matmul_qx);
    free(s_matmul_xd);
    s_matmul_qx = (int8_t *)malloc((size_t)n_cols);
    s_matmul_xd = (uint16_t *)malloc((size_t)(n_cols / 32) * sizeof(uint16_t));
    if (!s_matmul_qx || !s_matmul_xd) {
        free(s_matmul_qx);
        free(s_matmul_xd);
        s_matmul_qx = NULL;
        s_matmul_xd = NULL;
        s_matmul_cap_cols = 0;
        return -1;
    }
    s_matmul_cap_cols = n_cols;
    return 0;
}

int hqvm_q1_dequant_row64(const void *q1_row, size_t row_stride_bytes, float out64[64]) {
    return hqvm_q1_dequant_row_cols(q1_row, row_stride_bytes, 0, out64);
}

int hqvm_q1_dequant_row_cols(
    const void *q1_row, size_t row_stride_bytes, int64_t col0, float out64[64])
{
    const hqvm_block_q1_0 *row;
    const hqvm_block_q1_0 *blk;
    float scale;
    int i;
    int64_t blk_idx;
    int half;
    int base;
    if (!q1_row || !out64 || row_stride_bytes < sizeof(hqvm_block_q1_0)) return -1;
    if ((col0 & 63) != 0) return -1;
    row = (const hqvm_block_q1_0 *)q1_row;
    blk_idx = col0 / 128;
    half = (int)((col0 / 64) & 1);
    if ((size_t)(blk_idx + 1) * sizeof(hqvm_block_q1_0) > row_stride_bytes) return -1;
    blk = &row[blk_idx];
    scale = hqvm_f16_to_f32(blk->d);
    base = half * 64;
    for (i = 0; i < 64; ++i) {
        const int qi = base + i;
        const uint8_t bit = (uint8_t)((blk->qs[qi >> 3] >> (qi & 7)) & 1u);
        out64[i] = bit ? scale : -scale;
    }
    return 0;
}

int hqvm_q1_dequant_tile64(
    const void *q1_data,
    size_t row_stride_bytes,
    int64_t n_rows,
    int64_t n_cols,
    int64_t row0,
    int64_t col0,
    float out64x64[4096])
{
    int r;
    if (!q1_data || !out64x64 || n_rows <= 0 || n_cols <= 0) return -1;
    if ((row0 & 63) != 0 || (col0 & 63) != 0) return -1;
    if (row0 + 64 > n_rows || col0 + 64 > n_cols) return -1;
    for (r = 0; r < 64; ++r) {
        const char *row_ptr = (const char *)q1_data + (size_t)(row0 + r) * row_stride_bytes;
        if (hqvm_q1_dequant_row_cols(row_ptr, row_stride_bytes, col0,
                out64x64 + (size_t)r * 64) != 0) {
            return -1;
        }
    }
    return 0;
}

void hqvm_quantize_x_q8(const float *x, int64_t n, int8_t *qx, uint16_t *xd_h) {
    int64_t i, j;
    for (i = 0; i < n; i += 32) {
        float amax = 0.0f;
#if defined(__AVX2__) || defined(GGML_AVX2)
        {
            __m256 v0 = _mm256_loadu_ps(x + i);
            __m256 v1 = _mm256_loadu_ps(x + i + 8);
            __m256 v2 = _mm256_loadu_ps(x + i + 16);
            __m256 v3 = _mm256_loadu_ps(x + i + 24);
            const __m256 sign = _mm256_set1_ps(-0.0f);
            v0 = _mm256_andnot_ps(sign, v0);
            v1 = _mm256_andnot_ps(sign, v1);
            v2 = _mm256_andnot_ps(sign, v2);
            v3 = _mm256_andnot_ps(sign, v3);
            {
                __m256 m = _mm256_max_ps(_mm256_max_ps(v0, v1), _mm256_max_ps(v2, v3));
                __m128 lo = _mm256_castps256_ps128(m);
                __m128 hi = _mm256_extractf128_ps(m, 1);
                __m128 t = _mm_max_ps(lo, hi);
                t = _mm_max_ps(t, _mm_movehdup_ps(t));
                t = _mm_max_ps(t, _mm_movehl_ps(t, t));
                amax = _mm_cvtss_f32(t);
            }
        }
#else
        for (j = 0; j < 32; ++j) {
            const float ax = fabsf(x[i + j]);
            if (ax > amax) amax = ax;
        }
#endif
        {
            const float d = amax / 127.0f;
            const float id = (d > 0.0f) ? (1.0f / d) : 0.0f;
            xd_h[i / 32] = hqvm_f32_to_f16(d);
            for (j = 0; j < 32; ++j) {
                int v = (int)roundf(x[i + j] * id);
                if (v > 127) v = 127;
                if (v < -127) v = -127;
                qx[i + j] = (int8_t)v;
            }
        }
    }
}

static int forward_q1_0_q8_impl(
    const hqvm_sidecar *S,
    const void *q1_data,
    int64_t nrows,
    int64_t ncols,
    size_t row_stride_bytes,
    const int8_t *qx,
    const uint16_t *xd_h,
    float *Y_f,
    hqvm_dyad32_t *Y_d,
    int64_t row0,
    int64_t row1)
{
    /* Frozen per-site law (NavPAD §14.6 CLOSED):
     * Y = exact_Q1_0_q8_dot * manifold_gain(parity(mismatch) XOR chi'_bit0). */
    const int64_t n_q1 = ncols / HQVM_QK1_0;
    const int64_t n_blk64 = ncols / 64;
    int64_t r, bi, i;
    uint64_t *x_sign = NULL;
    uint8_t u0, v0, chi0;

    if (!S || !q1_data || !qx || !xd_h) return -1;
    if (!Y_f && !Y_d) return -1;
    if (ncols <= 0 || nrows <= 0 || (ncols % HQVM_QK1_0) != 0) return -1;
    if (row_stride_bytes < (size_t)n_q1 * sizeof(hqvm_block_q1_0)) return -1;
    if (row0 < 0) row0 = 0;
    if (row1 > nrows) row1 = nrows;
    if (row0 >= row1) return 0;

    x_sign = (uint64_t *)malloc((size_t)n_blk64 * sizeof(uint64_t));
    if (!x_sign) return -1;

    for (bi = 0; bi < n_blk64; ++bi) {
        uint64_t sign_word = 0;
        for (i = 0; i < 64; ++i) {
            if (qx[bi * 64 + i] >= 0) sign_word |= ((uint64_t)1u << (uint64_t)i);
        }
        x_sign[bi] = sign_word;
    }

    u0 = (uint8_t)(x_sign[0] & 63u);
    v0 = (uint8_t)((x_sign[0] >> 6) & 63u);
    chi0 = u0 ^ v0;

    for (r = row0; r < row1; ++r) {
        const hqvm_block_q1_0 *row =
            (const hqvm_block_q1_0 *)((const char *)q1_data + (size_t)r * row_stride_bytes);
        float acc = 0.0f;
        for (bi = 0; bi < n_q1; ++bi) {
            const hqvm_block_q1_0 *blk = &row[bi];
            const float scale = hqvm_f16_to_f32(blk->d);
            int half;
            uint8_t chi_bit[2];
            uint8_t parity_bit[2];
            float manifold_gain;

            for (half = 0; half < 2; ++half) {
                const int64_t b64 = bi * 2 + half;
                uint64_t wsign;
                uint64_t mismatch;
                int matches, shell64;
                int16_t bsel;
                uint8_t uu, vv;

                memcpy(&wsign, blk->qs + (size_t)half * 8u, 8u);
                mismatch = (uint64_t)(wsign ^ x_sign[b64]);
                matches = (int)popcount64_(~mismatch);
                shell64 = 64 - matches;
                bsel = S->byte_table[chi0 * S->n_bin + (uint32_t)bin_shell_edges(S->bin_edges, S->n_bin, (float)shell64)];
                uu = u0; vv = v0;
                if (bsel >= 0) {
                    hqvm_step_uv6(u0, v0, (uint8_t)bsel, &uu, &vv);
                }
                chi_bit[half] = (uint8_t)(((uu ^ vv) & 1u) ? 1u : 0u);
                parity_bit[half] = (uint8_t)(popcount64_(mismatch) & 1u);
            }
            manifold_gain = gyro_manifold_gain_2half(chi_bit[0], parity_bit[0], chi_bit[1], parity_bit[1]);

#if defined(__AVX2__) || defined(GGML_AVX2)
            {
                const __m256i ones_8 = _mm256_set1_epi8(1);
                const __m256i ones_16 = _mm256_set1_epi16(1);
                const __m256i zero = _mm256_setzero_si256();
                const __m256i byte_shuf = _mm256_setr_epi8(
                    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                    2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
                const __m256i bit_masks = _mm256_setr_epi8(
                    1, 2, 4, 8, 16, 32, 64, (char)-128, 1, 2, 4, 8, 16, 32, 64, (char)-128,
                    1, 2, 4, 8, 16, 32, 64, (char)-128, 1, 2, 4, 8, 16, 32, 64, (char)-128);
                const uint32_t *qs32 = (const uint32_t *)blk->qs;
                const int8_t *qbase = qx + bi * HQVM_QK1_0;
                const uint16_t *xdb = xd_h + bi * 4;
                __m256 acc_block;
                int k;
                {
                    const __m256i qy = _mm256_loadu_si256((const __m256i *)(qbase + 0));
                    const __m256i sm = _mm256_cmpeq_epi8(
                        _mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int)qs32[0]), byte_shuf), bit_masks),
                        zero);
                    const __m256i sy = _mm256_sub_epi8(_mm256_xor_si256(qy, sm), sm);
                    const __m256i s32 = _mm256_madd_epi16(_mm256_maddubs_epi16(ones_8, sy), ones_16);
                    acc_block = _mm256_mul_ps(_mm256_set1_ps(hqvm_f16_to_f32(xdb[0])), _mm256_cvtepi32_ps(s32));
                }
                for (k = 1; k < 4; ++k) {
                    const __m256i qy = _mm256_loadu_si256((const __m256i *)(qbase + k * 32));
                    const __m256i sm = _mm256_cmpeq_epi8(
                        _mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int)qs32[k]), byte_shuf), bit_masks),
                        zero);
                    const __m256i sy = _mm256_sub_epi8(_mm256_xor_si256(qy, sm), sm);
                    const __m256i s32 = _mm256_madd_epi16(_mm256_maddubs_epi16(ones_8, sy), ones_16);
                    acc_block = _mm256_fmadd_ps(_mm256_set1_ps(hqvm_f16_to_f32(xdb[k])), _mm256_cvtepi32_ps(s32), acc_block);
                }
                {
                    __m128 lo = _mm256_castps256_ps128(acc_block);
                    __m128 hi = _mm256_extractf128_ps(acc_block, 1);
                    __m128 s = _mm_add_ps(lo, hi);
                    s = _mm_add_ps(s, _mm_movehdup_ps(s));
                    s = _mm_add_ss(s, _mm_movehl_ps(s, s));
                    acc += scale * _mm_cvtss_f32(s) * manifold_gain;
                }
            }
#else
            {
                for (half = 0; half < 2; ++half) {
                    const int64_t b64 = bi * 2 + half;
                    uint64_t wsign;
                    float amp = 0.0f;
                    memcpy(&wsign, blk->qs + (size_t)half * 8u, 8u);
                    for (i = 0; i < 64; ++i) {
                        int64_t ix = b64 * 64 + i;
                        int8_t q = qx[ix];
                        float mag = (float)(q >= 0 ? q : -q) * hqvm_f16_to_f32(xd_h[ix / 32]);
                        int mismatch = (int)((wsign >> (uint64_t)i) & 1u) ^
                                       (int)((x_sign[b64] >> (uint64_t)i) & 1u);
                        amp += mismatch ? -mag : mag;
                    }
                    acc += amp * scale * manifold_gain;
                }
            }
#endif
        }
        if (Y_d) Y_d[r] = hqvm_dyad32_from_f32(acc);
        else Y_f[r] = acc;
    }

    free(x_sign);
    return 0;
}

int hqvm_forward_q1_0_q8(
    const hqvm_sidecar *S,
    const void *q1_data,
    int64_t nrows,
    int64_t ncols,
    size_t row_stride_bytes,
    const int8_t *qx,
    const uint16_t *xd_h,
    float *Y,
    int64_t row0,
    int64_t row1)
{
    return forward_q1_0_q8_impl(
        S, q1_data, nrows, ncols, row_stride_bytes, qx, xd_h, Y, NULL, row0, row1);
}

int hqvm_forward_q1_0_q8_dyad(
    const hqvm_sidecar *S,
    const void *q1_data,
    int64_t nrows,
    int64_t ncols,
    size_t row_stride_bytes,
    const int8_t *qx,
    const uint16_t *xd_h,
    hqvm_dyad32_t *Y,
    int64_t row0,
    int64_t row1)
{
    return forward_q1_0_q8_impl(
        S, q1_data, nrows, ncols, row_stride_bytes, qx, xd_h, NULL, Y, row0, row1);
}

int hqvm_i128_cmp(hqvm_i128 a, hqvm_i128 b) {
    const uint64_t sa = a.hi >> 63;
    const uint64_t sb = b.hi >> 63;
    if (sa != sb) return sa ? -1 : 1;
    if (a.hi != b.hi) return a.hi < b.hi ? -1 : 1;
    if (a.lo != b.lo) return a.lo < b.lo ? -1 : 1;
    return 0;
}

int hqvm_i128_add_shifted_i64(hqvm_i128 *acc, int64_t coefficient, uint32_t shift) {
    hqvm_i128 term, sum;
    uint64_t mag, carry;
    const uint64_t old_sign = acc ? (acc->hi >> 63) : 0;
    uint64_t term_sign;

    if (!acc || shift >= 128) return -1;
    mag = coefficient < 0 ? (uint64_t)(-(coefficient + 1)) + 1u : (uint64_t)coefficient;
    term.lo = 0;
    term.hi = 0;
    if (shift < 64) {
        term.lo = mag << shift;
        term.hi = shift ? (mag >> (64u - shift)) : 0;
        if (shift && (term.hi >> shift) != 0) return -1;
    } else {
        const uint32_t hi_shift = shift - 64u;
        term.hi = mag << hi_shift;
        if (hi_shift && (term.hi >> hi_shift) != mag) return -1;
    }
    if (coefficient >= 0) {
        if (term.hi >> 63) return -1;
    } else {
        if ((term.hi >> 63) && (term.hi != UINT64_C(0x8000000000000000) || term.lo != 0)) return -1;
        term.lo = ~term.lo + 1u;
        term.hi = ~term.hi + (term.lo == 0);
    }

    term_sign = term.hi >> 63;
    sum.lo = acc->lo + term.lo;
    carry = sum.lo < acc->lo;
    sum.hi = acc->hi + term.hi + carry;
    if (old_sign == term_sign && (sum.hi >> 63) != old_sign) return -1;
    *acc = sum;
    return 0;
}

int hqvm_fp16_decode_nonnegative(uint16_t bits, uint16_t *mantissa, int32_t *exponent) {
    const uint16_t exp = (uint16_t)((bits >> 10) & 31u);
    const uint16_t frac = (uint16_t)(bits & 1023u);
    if (!mantissa || !exponent || (bits & 0x8000u) || exp == 31u) return -1;
    if (exp == 0) {
        *mantissa = frac;
        *exponent = -24;
    } else {
        *mantissa = (uint16_t)(1024u + frac);
        *exponent = (int32_t)exp - 25;
    }
    return 0;
}

/* Add one Q1_0 block contribution into score. Returns -1 on ledger failure. */
static int hqvm_exact_add_q1_block(
    const hqvm_sidecar *S,
    const hqvm_block_q1_0 *blk,
    const uint64_t *x_sign,
    const int8_t *qx,
    const uint16_t *mx,
    const int32_t *ex,
    uint8_t u0,
    uint8_t v0,
    uint8_t chi0,
    int64_t bi,
    hqvm_i128 *score)
{
    uint16_t mw;
    int32_t ew, gain_num, gain_den;
    uint8_t chi_bit[2], parity_bit[2];
    int half, k;
    int64_t i;
    if (hqvm_fp16_decode_nonnegative(blk->d, &mw, &ew) != 0) return -1;
    for (half = 0; half < 2; ++half) {
        uint64_t wsign, mismatch;
        int shell64;
        int16_t bsel;
        uint8_t uu = u0, vv = v0;
        memcpy(&wsign, blk->qs + (size_t)half * 8u, 8u);
        mismatch = wsign ^ x_sign[bi * 2 + half];
        shell64 = (int)popcount64_(mismatch);
        bsel = S->byte_table[chi0 * S->n_bin + (uint32_t)bin_shell_edges(S->bin_edges, S->n_bin, (float)shell64)];
        if (bsel < -1 || bsel > 255) return -1;
        if (bsel >= 0) hqvm_step_uv6(u0, v0, (uint8_t)bsel, &uu, &vv);
        chi_bit[half] = (uint8_t)((uu ^ vv) & 1u);
        parity_bit[half] = (uint8_t)(popcount64_(mismatch) & 1u);
    }
    hqvm_manifold_gain_q16(chi_bit[0], parity_bit[0], chi_bit[1], parity_bit[1], &gain_num, &gain_den);
    if (gain_den != 65536 || gain_num < 0 || gain_num > 66893) return -1;
    for (k = 0; k < 4; ++k) {
        int32_t p = 0;
        int64_t coefficient;
        const int64_t base = bi * HQVM_QK1_0 + k * 32;
        const uint32_t shift = (uint32_t)(ew + ex[bi * 4 + k] + 48);
        for (i = 0; i < 32; ++i) {
            const uint8_t wb = (uint8_t)((blk->qs[(k * 32 + i) >> 3] >> ((k * 32 + i) & 7)) & 1u);
            p += wb ? (int32_t)qx[base + i] : -(int32_t)qx[base + i];
        }
        if (p < -4064 || p > 4064) return -1;
        if (shift > 58) return -1;
        coefficient = (int64_t)p * (int64_t)mw * (int64_t)mx[bi * 4 + k] * (int64_t)gain_num;
        if (hqvm_i128_add_shifted_i64(score, coefficient, shift) != 0) return -1;
    }
    return 0;
}

static int hqvm_pack_x_signs(
    const int8_t *qx,
    int64_t n_blk64,
    uint64_t *x_sign,
    uint8_t *u0_out,
    uint8_t *v0_out,
    uint8_t *chi0_out)
{
    int64_t bi, i;
    for (bi = 0; bi < n_blk64; ++bi) {
        uint64_t sign_word = 0;
        for (i = 0; i < 64; ++i) {
            const int8_t q = qx[bi * 64 + i];
            if (q == INT8_MIN) return -1;
            if (q >= 0) sign_word |= UINT64_C(1) << (uint32_t)i;
        }
        x_sign[bi] = sign_word;
    }
    *u0_out = (uint8_t)(x_sign[0] & 63u);
    *v0_out = (uint8_t)((x_sign[0] >> 6) & 63u);
    *chi0_out = (uint8_t)(*u0_out ^ *v0_out);
    return 0;
}

static int hqvm_argmax_q1_0_q8_exact_fill(
    const hqvm_sidecar *S,
    const void *q1_data,
    int64_t nrows,
    int64_t ncols,
    size_t row_stride_bytes,
    const int8_t *qx,
    const uint16_t *xd_h,
    int64_t row0,
    int64_t row1,
    int32_t *best_row,
    hqvm_i128 *best_score,
    hqvm_i128 *scores_out)
{
    const int64_t n_q1 = ncols / HQVM_QK1_0;
    const int64_t n_blk64 = ncols / 64;
    uint64_t *x_sign = NULL;
    uint16_t *mx = NULL;
    int32_t *ex = NULL;
    uint8_t u0, v0, chi0;
    int64_t r, bi, i;
    int have_best = 0;

    if (!S || !S->byte_table || !S->bin_edges || S->n_bin == 0 || !q1_data || !qx || !xd_h || !best_row || !best_score) return -1;
    if (ncols <= 0 || nrows <= 0 || (ncols % HQVM_QK1_0) != 0 || n_blk64 <= 0) return -1;
    if (row_stride_bytes < (size_t)n_q1 * sizeof(hqvm_block_q1_0)) return -1;
    if (row0 < 0) row0 = 0;
    if (row1 > nrows) row1 = nrows;
    if (row0 >= row1 || row1 > INT32_MAX) return -1;

    x_sign = (uint64_t *)malloc((size_t)n_blk64 * sizeof(uint64_t));
    mx = (uint16_t *)malloc((size_t)(ncols / 32) * sizeof(uint16_t));
    ex = (int32_t *)malloc((size_t)(ncols / 32) * sizeof(int32_t));
    if (!x_sign || !mx || !ex) goto fail_exact;
    for (i = 0; i < ncols / 32; ++i) {
        if (hqvm_fp16_decode_nonnegative(xd_h[i], &mx[i], &ex[i]) != 0) goto fail_exact;
    }
    if (hqvm_pack_x_signs(qx, n_blk64, x_sign, &u0, &v0, &chi0) != 0) goto fail_exact;

    for (r = row0; r < row1; ++r) {
        const hqvm_block_q1_0 *row = (const hqvm_block_q1_0 *)((const char *)q1_data + (size_t)r * row_stride_bytes);
        hqvm_i128 score = { 0, 0 };
        for (bi = 0; bi < n_q1; ++bi) {
            if (hqvm_exact_add_q1_block(S, &row[bi], x_sign, qx, mx, ex, u0, v0, chi0, bi, &score) != 0) goto fail_exact;
        }
        if (scores_out) scores_out[r] = score;
        if (!have_best || hqvm_i128_cmp(score, *best_score) > 0) {
            *best_row = (int32_t)r;
            *best_score = score;
            have_best = 1;
        }
    }
    free(x_sign);
    free(mx);
    free(ex);
    return have_best ? 0 : -1;

fail_exact:
    free(x_sign);
    free(mx);
    free(ex);
    return -1;
}

int hqvm_argmax_q1_0_q8_exact(
    const hqvm_sidecar *S,
    const void *q1_data,
    int64_t nrows,
    int64_t ncols,
    size_t row_stride_bytes,
    const int8_t *qx,
    const uint16_t *xd_h,
    int64_t row0,
    int64_t row1,
    int32_t *best_row,
    hqvm_i128 *best_score)
{
    return hqvm_argmax_q1_0_q8_exact_fill(
        S, q1_data, nrows, ncols, row_stride_bytes, qx, xd_h, row0, row1, best_row, best_score, NULL);
}


int hqvm_forward_q1_0_f32(
    const hqvm_sidecar *S,
    const void *q1_data,
    int64_t nrows,
    int64_t ncols,
    size_t row_stride_bytes,
    const float *x,
    float *Y,
    int64_t row0,
    int64_t row1)
{
    int8_t *qx;
    uint16_t *xd_h;
    int rc;

    if (!x) return -1;
    if (ncols <= 0 || (ncols % 32) != 0) return -1;
    qx = (int8_t *)malloc((size_t)ncols);
    xd_h = (uint16_t *)malloc((size_t)(ncols / 32) * sizeof(uint16_t));
    if (!qx || !xd_h) {
        free(qx);
        free(xd_h);
        return -1;
    }
    hqvm_quantize_x_q8(x, ncols, qx, xd_h);
    rc = hqvm_forward_q1_0_q8(S, q1_data, nrows, ncols, row_stride_bytes, qx, xd_h, Y, row0, row1);
    free(qx);
    free(xd_h);
    return rc;
}

#ifndef HQVM_TILE
#define HQVM_TILE 64
#endif

static hqvm_sidecar s_sidecar;
static int          s_sidecar_ok = 0; /* 0 unset, 1 ok, -1 missing */
static hqvm_gate_counters_t s_cnt;

static const hqvm_sidecar * matmul_sidecar(void) {
    if (s_sidecar_ok == 0) {
        const char *path = getenv("GYRO_LEDGER_PATH");
        if (path && path[0] && hqvm_sidecar_load(&s_sidecar, path) == 0) {
            hqvm_sidecar_apply_env_allow(&s_sidecar);
            s_sidecar_ok = 1;
        } else {
            s_sidecar_ok = -1;
        }
    }
    return (s_sidecar_ok > 0) ? &s_sidecar : NULL;
}

static size_t matmul_row_stride(const hqvm_q1_weight_t *W) {
    size_t stride = W->row_stride_bytes;
    if (stride == 0 && W->n_cols > 0) {
        const int64_t nblk = W->n_cols / 128;
        stride = (size_t)nblk * 20;
    }
    return stride;
}

int hqvm_sidecar_ready(void) {
    return matmul_sidecar() != NULL;
}

void hqvm_sidecar_reset_session(void) {
    if (s_sidecar_ok > 0) {
        hqvm_sidecar_free(&s_sidecar);
    }
    memset(&s_sidecar, 0, sizeof(s_sidecar));
    s_sidecar_ok = 0;
    hqvm_gate_counters_reset();
}

void hqvm_gate_counters_reset(void) {
    memset(&s_cnt, 0, sizeof(s_cnt));
}

void hqvm_gate_counters_snapshot(hqvm_gate_counters_t *out) {
    if (out) *out = s_cnt;
}

void hqvm_gate_counters_print(const char *tag) {
    fprintf(stderr,
        "[hqvm-gate] %s matmul=%llu pq=%llu dq=%llu norm=%llu rope=%llu "
        "attn=%llu vred=%llu swiglu=%llu stub=%llu sidecar=%s\n",
        tag ? tag : "counters",
        (unsigned long long)s_cnt.matmul_calls,
        (unsigned long long)s_cnt.matmul_pq_calls,
        (unsigned long long)s_cnt.matmul_dq_calls,
        (unsigned long long)s_cnt.norm_calls,
        (unsigned long long)s_cnt.rope_calls,
        (unsigned long long)s_cnt.attn_score_calls,
        (unsigned long long)s_cnt.v_reduce_calls,
        (unsigned long long)s_cnt.swiglu_calls,
        (unsigned long long)s_cnt.not_implemented,
        hqvm_sidecar_ready() ? "ready" : "missing");
    fflush(stderr);
}

void hqvm_gate_counters_inc_norm(void) { s_cnt.norm_calls++; }
void hqvm_gate_counters_inc_rope(void) { s_cnt.rope_calls++; }
void hqvm_gate_counters_inc_attn(void) { s_cnt.attn_score_calls++; }
void hqvm_gate_counters_inc_vreduce(void) { s_cnt.v_reduce_calls++; }
void hqvm_gate_counters_inc_swiglu(void) { s_cnt.swiglu_calls++; }
void hqvm_gate_counters_inc_stub(void) { s_cnt.not_implemented++; }

static void matmul_dequant_x64(
    const int8_t * qx,
    const uint16_t * xd,
    int64_t          col0,
    float            x64[64])
{
    int i;
    for (i = 0; i < 64; ++i) {
        const int64_t idx = col0 + i;
        const float scale = hqvm_f16_to_f32(xd[idx / 32]);
        x64[i] = (float)qx[idx] * scale;
    }
}

static float matmul_pchi_dot_row(
    const float * Wtile,
    int           ri,
    const float * x64)
{
    float f[HQVM_TILE];
    float y = 0.0f;
    int j;
    gyroscopic_project_chi_coeffs(Wtile, f);
    for (j = 0; j < HQVM_TILE; ++j) {
        y += f[ri ^ j] * x64[j];
    }
    return y;
}

static float matmul_dq_row(
    const hqvm_q1_weight_t * W,
    size_t                   stride,
    int64_t                  row,
    const int8_t *           qx,
    const uint16_t *         xd)
{
    float Wtile[HQVM_TILE * HQVM_TILE];
    float x64[HQVM_TILE];
    const int64_t r0 = (row / HQVM_TILE) * HQVM_TILE;
    const int     ri = (int)(row - r0);
    int64_t cb;
    float dq = 0.0f;

    if ((W->n_cols % HQVM_TILE) != 0) return 0.0f;
    for (cb = 0; cb < W->n_cols / HQVM_TILE; ++cb) {
        const int64_t c0 = cb * HQVM_TILE;
        if (hqvm_q1_dequant_tile64(
                W->q1_data, stride, W->n_rows, W->n_cols, r0, c0, Wtile) != 0) {
            continue;
        }
        matmul_dequant_x64(qx, xd, c0, x64);
        dq += gyroscopic_tile_hybrid_dot_row(Wtile, ri, x64)
            - matmul_pchi_dot_row(Wtile, ri, x64);
    }
    return dq;
}

int hqvm_matmul_dyad(
    const hqvm_q1_weight_t *W,
    const hqvm_dyad32_t *x,
    hqvm_dyad32_t *y)
{
    const hqvm_sidecar *S;
    size_t stride;
    int rc = 0;
    static int s_logged = 0;

    s_cnt.matmul_calls++;
    if (!W || !W->q1_data || !x || !y || W->n_rows <= 0 || W->n_cols <= 0) return -1;
    if ((W->n_cols % 32) != 0) return -1;

    S = matmul_sidecar();
    if (!S) return -5;

    if (!s_logged) {
        fprintf(stderr, "[hqvm-matmul] mode=Q1xq8-manifold-dyadY\n");
        fflush(stderr);
        s_logged = 1;
    }

    if (matmul_scratch_ensure(W->n_cols) != 0) {
        return -2;
    }

    hqvm_quantize_dyad_q8(x, W->n_cols, s_matmul_qx, s_matmul_xd);
    stride = matmul_row_stride(W);

    s_cnt.matmul_pq_calls++;
    rc = hqvm_forward_q1_0_q8_dyad(
        S, W->q1_data, W->n_rows, W->n_cols, stride,
        s_matmul_qx, s_matmul_xd, y, 0, W->n_rows);

    return rc;
}

int hqvm_matmul_dq_selftest(
    const hqvm_q1_weight_t * W,
    const int8_t *           qx,
    const uint16_t *         xd,
    int64_t *                out_dq_rows)
{
    int64_t r;
    size_t stride;
    int64_t n_dq = 0;

    if (!W || !W->q1_data || !qx || !xd || !out_dq_rows) return -1;
    if ((W->n_cols % HQVM_TILE) != 0) return -1;
    if (W->n_rows > HQVM_TILE || W->n_cols > 256) return -2;

    stride = matmul_row_stride(W);
    for (r = 0; r < W->n_rows; ++r) {
        const float dq = matmul_dq_row(W, stride, r, qx, xd);
        if (dq != 0.0f) {
            n_dq++;
        }
        out_dq_rows[r] = (int64_t)(dq != 0.0f);
    }
    s_cnt.matmul_dq_calls += (uint64_t)n_dq;
    return 0;
}
