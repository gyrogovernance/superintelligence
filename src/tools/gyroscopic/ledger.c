/*
 * hQVM temporal ledger v2 implementation.
 * step_uv6 matches science/src.family.step_uv(d=6).
 */

#include "ledger.h"
#include "constants.h"

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

void hqvm_pack12_bits(const uint8_t *x01, uint8_t *u, uint8_t *v) {
    uint8_t uu = 0, vv = 0;
    int i;
    for (i = 0; i < 6; ++i) {
        if (x01[i]) uu |= (uint8_t)(1u << i);
        if (x01[6 + i]) vv |= (uint8_t)(1u << i);
    }
    *u = uu;
    *v = vv;
}

static int bin_shell(const hqvm_ledger *L, float s) {
    uint32_t lo = 0, hi = L->n_bin + 1;
    while (lo < hi) {
        uint32_t mid = (lo + hi) / 2;
        if (L->bin_edges[mid] <= s) lo = mid + 1;
        else hi = mid;
    }
    int idx = (int)lo - 1;
    if (idx < 0) idx = 0;
    if ((uint32_t)idx >= L->n_bin) idx = (int)L->n_bin - 1;
    return idx;
}

static uint8_t sign_bit(const uint8_t *packed8, uint32_t i) {
    return (uint8_t)((packed8[i >> 3] >> (i & 7u)) & 1u);
}

static uint8_t popcount64_(uint64_t v) {
#if defined(_MSC_VER)
    return (uint8_t)__popcnt64(v);
#else
    return (uint8_t)__builtin_popcountll(v);
#endif
}

static const uint8_t *block_signs(const hqvm_ledger *L, uint32_t r, uint32_t b) {
    const uint32_t bytes_per = L->block_w / 8u;
    return L->signs + ((size_t)r * L->n_blocks + b) * bytes_per;
}

static float block_scale(const hqvm_ledger *L, uint32_t r, uint32_t b) {
    /* two 64-wide blocks share one Q1_0 128-bundle scale */
    uint32_t sb = b / 2u;
    if (sb >= L->n_scale_blocks) sb = L->n_scale_blocks - 1u;
    return L->scales[(size_t)r * L->n_scale_blocks + sb];
}

int hqvm_ledger_load(hqvm_ledger *L, const char *path) {
    FILE *f;
    char magic[8];
    size_t n_sign_bytes, n_scale, n_bt, n_edges;

    if (!L || !path) return -1;
    memset(L, 0, sizeof(*L));
    f = fopen(path, "rb");
    if (!f) return -1;
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "HQVMLEDG", 8) != 0) {
        fclose(f);
        return -1;
    }
    if (fread(&L->version, 4, 1, f) != 1) {
        fclose(f);
        return -1;
    }
    if (L->version != 2) {
        /* v1 (fp32 coefs) no longer supported in this payoff path */
        fclose(f);
        return -1;
    }
    if (fread(&L->n_rows, 4, 1, f) != 1 ||
        fread(&L->n_blocks, 4, 1, f) != 1 ||
        fread(&L->block_w, 4, 1, f) != 1 ||
        fread(&L->n_bin, 4, 1, f) != 1 ||
        fread(&L->n_scale_blocks, 4, 1, f) != 1) {
        fclose(f);
        return -1;
    }
    n_sign_bytes = (size_t)L->n_rows * (size_t)L->n_blocks * (size_t)(L->block_w / 8u);
    n_scale = (size_t)L->n_rows * (size_t)L->n_scale_blocks;
    n_bt = 64u * (size_t)L->n_bin;
    n_edges = (size_t)L->n_bin + 1u;

    L->signs = (uint8_t *)malloc(n_sign_bytes);
    L->scales = (float *)malloc(n_scale * sizeof(float));
    L->byte_table = (int16_t *)malloc(n_bt * sizeof(int16_t));
    L->bin_edges = (float *)malloc(n_edges * sizeof(float));
    if (!L->signs || !L->scales || !L->byte_table || !L->bin_edges) {
        hqvm_ledger_free(L);
        fclose(f);
        return -1;
    }
    if (fread(L->signs, 1, n_sign_bytes, f) != n_sign_bytes ||
        fread(L->scales, sizeof(float), n_scale, f) != n_scale ||
        fread(L->byte_table, sizeof(int16_t), n_bt, f) != n_bt ||
        fread(L->bin_edges, sizeof(float), n_edges, f) != n_edges) {
        hqvm_ledger_free(L);
        fclose(f);
        return -1;
    }
    fclose(f);
    return 0;
}

void hqvm_ledger_free(hqvm_ledger *L) {
    if (!L) return;
    free(L->signs);
    free(L->scales);
    free(L->byte_table);
    free(L->bin_edges);
    free(L->coefs);
    memset(L, 0, sizeof(*L));
}

void hqvm_ledger_forward_bits(
    const hqvm_ledger *L,
    const uint8_t *x01,
    float *Y,
    int16_t *shells)
{
    uint8_t u0, v0, chi0;
    uint32_t r, b, c;
    const uint32_t bw = L->block_w;

    hqvm_pack12_bits(x01, &u0, &v0);
    chi0 = (uint8_t)((u0 ^ v0) & HQVM_MASK6);

    for (r = 0; r < L->n_rows; ++r) {
        float acc = 0.0f;
        for (b = 0; b < L->n_blocks; ++b) {
            const uint8_t *ps = block_signs(L, r, b);
            const uint8_t *xb = x01 + b * bw;
            uint32_t pop = 0; /* XNOR matches */
            for (c = 0; c < bw; ++c) {
                pop += (uint32_t)(sign_bit(ps, c) == xb[c]);
            }
            /* shell64 = popcount(w ^ x) = bw - pop */
            double shell64 = (double)(bw - pop);
            int tgt = bin_shell(L, (float)shell64);
            int16_t bsel = L->byte_table[chi0 * L->n_bin + (uint32_t)tgt];
            uint8_t uu = u0, vv = v0;
            if (bsel >= 0) {
                hqvm_step_uv6(u0, v0, (uint8_t)bsel, &uu, &vv);
            }
            uint8_t sh = hqvm_shell_uv6(uu, vv);
            if (shells) {
                shells[(size_t)r * L->n_blocks + b] = (int16_t)sh;
            }
            float scale = block_scale(L, r, b);
            acc += (float)((2.0 * (double)pop - (double)bw) * (double)scale);
        }
        Y[r] = acc;
    }
}

void hqvm_ledger_forward_f32(
    const hqvm_ledger *L,
    const float *x,
    float *Y)
{
    /* Routing: XNOR-popcount on sign(x) → shell64 → byte table → step_uv6.
     * Amplitude: exact per-block sum_i (1-2*w_i)*x_i * Q1_0_scale
     * (mean(|x|) fold failed Gate C text; this is NavPAD §14.3). */
    const uint32_t bw = L->block_w;
    const uint32_t words_per_block = bw / 64u; /* 1 for block_w=64 */
    uint32_t in_dim = L->n_blocks * bw;
    uint32_t in_words = in_dim / 64u;
    uint64_t *xw = (uint64_t *)malloc((size_t)in_words * sizeof(uint64_t));
    uint32_t r, b, w, i;
    uint8_t u0, v0, chi0;

    if (!xw) return;
    for (w = 0; w < in_words; ++w) {
        uint64_t word = 0;
        uint32_t base = w * 64u;
        for (i = 0; i < 64u; ++i) {
            if (x[base + i] < 0.0f) word |= (uint64_t)1 << i;
        }
        xw[w] = word;
    }
    u0 = (uint8_t)(xw[0] & HQVM_MASK6);
    v0 = (uint8_t)((xw[0] >> 6) & HQVM_MASK6);
    chi0 = (uint8_t)((u0 ^ v0) & HQVM_MASK6);

    for (r = 0; r < L->n_rows; ++r) {
        float acc = 0.0f;
        for (b = 0; b < L->n_blocks; ++b) {
            const uint8_t *ps = block_signs(L, r, b);
            const float *xb = x + b * bw;
            uint32_t pop = 0;
            double amp = 0.0;
            for (w = 0; w < words_per_block; ++w) {
                uint64_t wsign;
                memcpy(&wsign, ps + w * 8u, 8u);
                uint64_t xw_block = xw[b * words_per_block + w];
                uint64_t xnor = ~(wsign ^ xw_block);
#if defined(_MSC_VER)
                pop += (uint32_t)__popcnt64(xnor);
#else
                pop += (uint32_t)__builtin_popcountll(xnor);
#endif
            }
            /* exact amplitude: ggml Q1_0 bit=1 → +1, bit=0 → -1 (see
             * ggml_vec_dot_q1_0_q8_0). Export/Python used (1-2*bit); live
             * must match ggml or the whole attn_q row is sign-flipped. */
            for (i = 0; i < bw; ++i) {
                uint8_t wb = sign_bit(ps, i);
                double wbipolar = (wb ? 1.0 : -1.0);
                amp += wbipolar * (double)xb[i];
            }
            double shell64 = (double)(bw - pop);
            int tgt = bin_shell(L, (float)shell64);
            int16_t bsel = L->byte_table[chi0 * L->n_bin + (uint32_t)tgt];
            if (bsel >= 0) {
                uint8_t uu, vv;
                hqvm_step_uv6(u0, v0, (uint8_t)bsel, &uu, &vv);
                (void)hqvm_shell_uv6(uu, vv);
            }
            float scale = block_scale(L, r, b);
            acc += (float)(amp * (double)scale);
        }
        Y[r] = acc;
    }
    free(xw);
}

/* ---- Thin sidecar + Q1_0 forward from ggml memory ---- */

#define HQVM_QK1_0 ((int)BOUNDARY_SIZE) /* ggml Q1_0 block width; coincides with BOUNDARY_SIZE on Bonsai */

typedef struct {
    uint16_t d;
    uint8_t qs[HQVM_QK1_0 / 8];
} hqvm_block_q1_0;

static float hqvm_fp16_to_f32(uint16_t h) {
#if defined(__F16C__) || defined(GGML_F16C) || defined(__AVX2__) || defined(GGML_AVX2)
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)h)));
#else
    const unsigned sign = (h >> 15) & 1u;
    const unsigned exp = (h >> 10) & 0x1fu;
    const unsigned mant = h & 0x3ffu;
    float f;
    if (exp == 0) {
        f = (mant == 0) ? 0.0f : (float)ldexp((double)mant, -24);
    } else if (exp == 31) {
        f = (mant == 0) ? (float)INFINITY : (float)NAN;
    } else {
        f = (float)ldexp((double)(mant | 0x400u), (int)exp - 25);
    }
    return sign ? -f : f;
#endif
}

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

static float gyro_sign01(uint8_t bit01) {
    return bit01 ? 1.0f : -1.0f;
}

static float gyro_manifold_gain_2half(uint8_t chi_bit0, uint8_t p0, uint8_t chi_bit1, uint8_t p1) {
    const float s0 = gyro_sign01((uint8_t)(p0 ^ chi_bit0));
    const float s1 = gyro_sign01((uint8_t)(p1 ^ chi_bit1));
    return 1.0f + (float)APERTURE_GAP * 0.5f * (s0 + s1);
}

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

void hqvm_quantize_x_q8(const float *x, int64_t n, int8_t *qx, float *xd) {
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
            xd[i / 32] = d;
            for (j = 0; j < 32; ++j) {
                int v = (int)lrintf(x[i + j] * id);
                if (v > 127) v = 127;
                if (v < -127) v = -127;
                qx[i + j] = (int8_t)v;
            }
        }
    }
}

int hqvm_forward_q1_0_q8(
    const hqvm_sidecar *S,
    const void *q1_data,
    int64_t nrows,
    int64_t ncols,
    size_t row_stride_bytes,
    const int8_t *qx,
    const float *xd,
    float *Y,
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

    if (!S || !q1_data || !qx || !xd || !Y) return -1;
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
            const float scale = hqvm_fp16_to_f32(blk->d);
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
                const float *xdb = xd + bi * 4;
                __m256 acc_block;
                int k;
                {
                    const __m256i qy = _mm256_loadu_si256((const __m256i *)(qbase + 0));
                    const __m256i sm = _mm256_cmpeq_epi8(
                        _mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int)qs32[0]), byte_shuf), bit_masks),
                        zero);
                    const __m256i sy = _mm256_sub_epi8(_mm256_xor_si256(qy, sm), sm);
                    const __m256i s32 = _mm256_madd_epi16(_mm256_maddubs_epi16(ones_8, sy), ones_16);
                    acc_block = _mm256_mul_ps(_mm256_set1_ps(xdb[0]), _mm256_cvtepi32_ps(s32));
                }
                for (k = 1; k < 4; ++k) {
                    const __m256i qy = _mm256_loadu_si256((const __m256i *)(qbase + k * 32));
                    const __m256i sm = _mm256_cmpeq_epi8(
                        _mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int)qs32[k]), byte_shuf), bit_masks),
                        zero);
                    const __m256i sy = _mm256_sub_epi8(_mm256_xor_si256(qy, sm), sm);
                    const __m256i s32 = _mm256_madd_epi16(_mm256_maddubs_epi16(ones_8, sy), ones_16);
                    acc_block = _mm256_fmadd_ps(_mm256_set1_ps(xdb[k]), _mm256_cvtepi32_ps(s32), acc_block);
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
                        float mag = (float)(q >= 0 ? q : -q) * xd[ix / 32];
                        int mismatch = (int)((wsign >> (uint64_t)i) & 1u) ^
                                       (int)((x_sign[b64] >> (uint64_t)i) & 1u);
                        amp += mismatch ? -mag : mag;
                    }
                    acc += amp * scale * manifold_gain;
                }
            }
#endif
        }
        Y[r] = acc;
    }

    free(x_sign);
    return 0;
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
    float *xd;
    int rc;

    if (!x) return -1;
    if (ncols <= 0 || (ncols % 32) != 0) return -1;
    qx = (int8_t *)malloc((size_t)ncols);
    xd = (float *)malloc((size_t)(ncols / 32) * sizeof(float));
    if (!qx || !xd) {
        free(qx);
        free(xd);
        return -1;
    }
    hqvm_quantize_x_q8(x, ncols, qx, xd);
    rc = hqvm_forward_q1_0_q8(S, q1_data, nrows, ncols, row_stride_bytes, qx, xd, Y, row0, row1);
    free(qx);
    free(xd);
    return rc;
}
