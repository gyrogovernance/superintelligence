#include "gyrolabe_qubec_matmul.h"

#include "gyrolabe.h"
#include "gyrolabe_registry.h"
#include "gyrograph_policy.h"

#include "quants.h"
#include "gyrolabe_wht.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <limits.h>

const char * gyrolabe_qubec_residual_mode_label(void) {
    const gyro_policy * policy = gyro_policy_get();
    if (policy != NULL && policy->mode == GYRO_MODE_GYROSCOPIC) {
        return "packed-native";
    }
    return "dense-native";
}
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#if defined(_MSC_VER)
#include <intrin.h>
#endif

#if defined(_MSC_VER)
#define GYRO_QUBEC_TLS __declspec(thread)
#else
#define GYRO_QUBEC_TLS _Thread_local
#endif

#define GYRO_WITNESS_PERIOD 48

#if defined(GGML_CPU_FP16_TO_FP32)
#define GYRO_FP16_TO_F32(v) GGML_CPU_FP16_TO_FP32((ggml_fp16_t) (v))
#elif defined(GGML_FP16_TO_FP32)
#define GYRO_FP16_TO_F32(v) GGML_FP16_TO_FP32((ggml_fp16_t) (v))
#else
#define GYRO_FP16_TO_F32(v) ggml_fp16_to_fp32((ggml_fp16_t) (v))
#endif

/*
 * QuBEC Coordinate Identity for 64-wide blocks:
 * x[i] with i = 0..63 is the amplitude at chi state chi = i.
 * Index i is the 6-bit chi coordinate in GF(2)^6.
 * shell N is the popcount of i.
 * WHT maps between chi-space and spectral-space.
 */

static GYRO_QUBEC_TLS gyrolabe_qubec_call_stats g_last_call_stats;
static GYRO_QUBEC_TLS gyrolabe_qubec_dispatch_stats g_last_dispatch_stats;

static int gyromatmul_vec_dot_q8_0_q8_0_avx2(
    int n_blocks,
    const gyromatmul_block_q8_0 * GYRO_RESTRICT x,
    const gyromatmul_block_q8_0 * GYRO_RESTRICT y,
    float * GYRO_RESTRICT out
);

static int gyromatmul_vec_dot_q8_0_q8_0_avx2_condensed(
    int n_blocks,
    const gyromatmul_block_q8_0 * GYRO_RESTRICT x,
    const gyromatmul_block_q8_0 * GYRO_RESTRICT y,
    float * GYRO_RESTRICT out,
    int use_condensed
);

static uint32_t gyrolabe_popcount_u64(uint64_t x) {
#if defined(_MSC_VER)
    return (uint32_t) __popcnt64((unsigned __int64) x);
#else
    return (uint32_t) __builtin_popcountll((unsigned long long) x);
#endif
}

static void dequantize_row64_q8_0(
    const struct gyromatmul_block_q8_0 * row64,
    float * out64
) {
    int sub;
    for (sub = 0; sub < 2; ++sub) {
        const float d = ggml_fp16_to_fp32((ggml_fp16_t) row64[sub].d);
        int i;
        for (i = 0; i < 32; ++i) {
            out64[(size_t) sub * 32u + (size_t) i] = (float) row64[sub].qs[i] * d;
        }
    }
}

static inline void gyrolabe_accumulate_dense_rows(
    int local_rows,
    int panel_row0,
    int lda_blocks,
    int q8_offset,
    int ldc,
    int col_idx,
    const struct gyromatmul_block_q8_0 * a,
    const struct gyromatmul_block_q8_0 * wr,
    float * c,
    int use_condensed_chart
) {
    int local_row;
    for (local_row = 0; local_row < local_rows; ++local_row) {
        const int panel_row = panel_row0 + local_row;
        const int row_off = panel_row * lda_blocks;
        const struct gyromatmul_block_q8_0 * ar =
            a + (size_t) row_off + (size_t) q8_offset;
        float dot = 0.0f;
        if (gyromatmul_vec_dot_q8_0_q8_0_avx2_condensed(
                2, wr, ar, &dot, use_condensed_chart) == 0) {
            c[(size_t) panel_row + (size_t) col_idx * (size_t) ldc] += dot;
        }
    }
}

/* CHI_INVARIANT P_Q: WHT, diagonal phi from registry, WHT, then 1/64 (see GyroLabe spectral chart). */
static void apply_spectral_tile64_f32(
    const float * x64,
    const float * phi_f32,
    float * y64
) {
    float tmp[64];
    memcpy(tmp, x64, 64u * sizeof(float));
    int i;
    gyrolabe_wht64_f32_inplace(tmp);
    for (i = 0; i < 64; ++i) {
        tmp[i] *= phi_f32[i];
    }
    gyrolabe_wht64_f32_inplace(tmp);
    for (i = 0; i < 64; ++i) {
        /* Unnormalized WHT: H(H(x)) = 64*x; one 1/64 closes P_Q for diagonal spectral weights. */
        y64[i] = tmp[i] * (1.0f / 64.0f);
    }
}

static inline void k4_gemv64_avx2(const gyrolabe_block_info_t * info, const float * x, float * y) {
    int r;
    int j;
    int k;
    for (r = 0; r < 64; ++r) {
        float acc = 0.0f;
        const uint64_t row_sign = info->packed_DQ.sign_mask[r];
        for (j = 0; j < 64; ++j) {
            int32_t mag = 0;
            int32_t qv;
            const uint64_t bit = (1ULL << j);
            for (k = 0; k < 16; ++k) {
                if ((info->packed_DQ.bitplanes[r][k] & bit) != 0ULL) {
                    mag |= (1 << k);
                }
            }
            qv = (row_sign & bit) != 0ULL ? -mag : mag;
            acc += (float) qv * x[j];
        }
        y[r] = acc * info->packed_DQ.scale_w;
    }
}

static int apply_block_pq(const gyrolabe_block_info_t * info, const float * x, float * y) {
    int i;
    for (i = 0; i < 64; ++i) y[i] = 0.0f;
    if (info == NULL) {
        return -1;
    }
    if (info->class_id == GYRO_CLASS_CHI_INVARIANT) {
        apply_spectral_tile64_f32(x, info->eigenvalues.phi_64, y);
        return 0;
    } else if (info->class_id == GYRO_CLASS_SHELL_RADIAL) {
        float tmp[64];
        memcpy(tmp, x, sizeof(tmp));
        gyrolabe_wht64_f32_inplace(tmp);
        for (i = 0; i < 64; ++i) {
            tmp[i] *= (1.0f / 64.0f);
        }
        for (i = 0; i < 64; ++i) {
            const int r = (int) gyrolabe_popcount_u64((uint64_t) i);
            tmp[i] *= (float) info->eigenvalues.lambda_7[r];
        }
        gyrolabe_wht64_f32_inplace(tmp);
        for (i = 0; i < 64; ++i) {
            tmp[i] *= (1.0f / 64.0f);
        }
        memcpy(y, tmp, sizeof(tmp));
        return 0;
    }
    return -1;
}

static void gyrolabe_update_witness_row(
    float y_acc,
    const struct gyromatmul_block_q8_0 * wr,
    const struct gyromatmul_block_q8_0 * ar,
    int use_condensed_chart
) {
    float y_ref = 0.0f;
    if (gyromatmul_vec_dot_q8_0_q8_0_avx2_condensed(2, wr, ar, &y_ref, use_condensed_chart) != 0) {
        y_ref = 0.0f;
    }
    {
        const float y_err = fabsf(y_acc - y_ref);
        if (y_err > g_last_call_stats.max_abs_row_error) {
            g_last_call_stats.max_abs_row_error = y_err;
        }
        if (y_err > 1.0e-3f) {
            g_last_call_stats.parity_mismatch_rows += 1;
        } else {
            g_last_call_stats.exact_witness_rows += 1;
        }
    }
}

void gyrolabe_qubec_get_last_call_stats(gyrolabe_qubec_call_stats * out_stats) {
    if (out_stats == NULL) {
        return;
    }
    *out_stats = g_last_call_stats;
}

void gyrolabe_qubec_get_last_dispatch_stats(gyrolabe_qubec_dispatch_stats * out_stats) {
    if (out_stats == NULL) {
        return;
    }
    *out_stats = g_last_dispatch_stats;
}

/*
 * QuBEC matmul contract (names are historical; roles are fixed):
 * - a, lda_bytes: weight matrix panel (ggml src0 Q8_0), row-major blocks along k.
 * - w, ldb_bytes: quantized activation slab (ggml params->wdata / src1 Q8), one row per output col.
 * - m: rows of this panel in a (weight / dst row dim); n: output columns; ldc >= m.
 * - Registry: (row_block_global, k_block) over the weight tensor only; jb is output-col blocks, not a registry row index.
 * - weight_col0: global output-column origin; row_start: global weight-row origin.
 */
int gyrolabe_qubec_matmul_q8_0(
    int m,
    int n,
    int k,
    const struct gyromatmul_block_q8_0 * a,
    int lda_bytes,
    const struct gyromatmul_block_q8_0 * w,
    int ldb_bytes,
    float * c,
    int ldc,
    int weight_col0,
    const struct ggml_tensor * w_tensor,
    const void * w_q8_lookup_base,
    int row_start,
    int k_block,
    int32_t cell_idx
) {
    const int q8_block_bytes = (int) sizeof(struct gyromatmul_block_q8_0);
    const int lda_blocks = lda_bytes / q8_block_bytes;
    const int ldb_blocks = ldb_bytes / q8_block_bytes;
    const int k64_blocks = k / 64;
    const int k_tail32 = (k % 64) / 32;
    gyrolabe_qubec_dispatch_stats local_dispatch = {0};
    const int row_end = row_start + m;

    memset(&g_last_call_stats, 0, sizeof(g_last_call_stats));
    g_last_call_stats.max_abs_row_error = 0.0f;
    const int use_condensed_chart = 0;
    (void) cell_idx;

    if (m <= 0 || n <= 0 || k <= 0) {
        g_last_dispatch_stats = local_dispatch;
        return -1;
    }
    if (a == NULL || w == NULL || c == NULL) {
        g_last_dispatch_stats = local_dispatch;
        return -1;
    }
    if ((k % 32) != 0) {
        g_last_dispatch_stats = local_dispatch;
        return -1;
    }
    if (lda_bytes <= 0 || ldb_bytes <= 0 || ldc < m) {
        g_last_dispatch_stats = local_dispatch;
        return -1;
    }
    if (lda_blocks < (k / 32) || ldb_blocks < (k / 32)) {
        g_last_dispatch_stats = local_dispatch;
        return -1;
    }
    if (k64_blocks <= 0) {
        local_dispatch.no_k64_blocks = 1;
    }

    for (int j = 0; j < n; ++j) {
        for (int row = 0; row < m; ++row) {
            c[(size_t) row + (size_t) j * (size_t) ldc] = 0.0f;
        }
    }

    {
        const int row_block_first = row_start >> 6;
        const int row_block_last = (row_end - 1) >> 6;
        const int j_block_min = weight_col0 >> 6;
        const int j_block_max = (weight_col0 + n - 1) >> 6;

        for (int row_block_global = row_block_first; row_block_global <= row_block_last; ++row_block_global) {
            const int global_block_row0 = row_block_global << 6;
            const int global_block_row1 = global_block_row0 + 64;
            const int active_row0 = row_start > global_block_row0 ? row_start : global_block_row0;
            const int active_row1 = row_end < global_block_row1 ? row_end : global_block_row1;
            const int panel_row0 = active_row0 - row_start;
            const int local_rows = active_row1 - active_row0;

            if (local_rows <= 0) {
                continue;
            }
            gyrolabe_registry_entry_t reg_entry = gyrolabe_registry_find_entry(w_tensor, w_q8_lookup_base);

            if (reg_entry != NULL) {
                for (int b = 0; b < k64_blocks; ++b) {
                    const int q8_offset = b * 2;
                    const gyrolabe_block_info_t * info_blk =
                        gyrolabe_registry_get_block_from_entry(reg_entry, row_block_global, k_block + b);

                    for (int jb = j_block_min; jb <= j_block_max; ++jb) {
                        const gyrolabe_block_info_t * info = info_blk;
                        int jj0 = (jb << 6) - weight_col0;
                        int jj1 = ((jb + 1) << 6) - weight_col0;
                        int ncol;

                        if (jj0 < 0) jj0 = 0;
                        if (jj1 > n) jj1 = n;
                        if (jj0 >= jj1) continue;
                        ncol = jj1 - jj0;
                        local_dispatch.scanned_blocks += ncol;

                        if (info != NULL) {
                            int jj;
                            for (jj = jj0; jj < jj1; ++jj) {
                                float y_p[64];
                                float y_d[64];
                                float x_act[64];
                                int local_row;
                                int used_structured = 0;

                                dequantize_row64_q8_0(
                                    w + (size_t) jj * (size_t) ldb_blocks + (size_t) q8_offset,
                                    x_act);

                                if (apply_block_pq(info, x_act, y_p) == 0) {
                                    used_structured = 1;
                                    if (info->class_id == GYRO_CLASS_CHI_INVARIANT) {
                                        g_last_call_stats.used_chi = 1;
                                    } else if (info->class_id == GYRO_CLASS_SHELL_RADIAL) {
                                        g_last_call_stats.used_radial = 1;
                                    } else if (info->class_id == GYRO_CLASS_CHI_X_GAUGE) {
                                        g_last_call_stats.used_chi_gauge = 1;
                                    }
                                } else {
                                    memset(y_p, 0, sizeof(y_p));
                                }

                                if (info->dq_lattice_empty) {
                                    memset(y_d, 0, sizeof(y_d));
                                } else {
                                    k4_gemv64_avx2(info, x_act, y_d);
                                    if (!used_structured) {
                                        g_last_call_stats.spectral_sparse_rows += local_rows;
                                    }
                                }

                                if (used_structured || !info->dq_lattice_empty) {
                                    for (local_row = 0; local_row < local_rows; ++local_row) {
                                        const int panel_row = panel_row0 + local_row;
                                        const int rel_row = (row_start + panel_row) - global_block_row0;
                                        const struct gyromatmul_block_q8_0 * wr =
                                            w + (size_t) jj * (size_t) ldb_blocks + (size_t) q8_offset;
                                        const struct gyromatmul_block_q8_0 * ar =
                                            a + (size_t) panel_row * (size_t) lda_blocks + (size_t) q8_offset;
                                        float y_acc;
                                        if (rel_row < 0 || rel_row >= 64) {
                                            continue;
                                        }
                                        y_acc = y_p[rel_row] + y_d[rel_row];
                                        if (((panel_row + jj + b) % GYRO_WITNESS_PERIOD) == 0) {
                                            gyrolabe_update_witness_row(y_acc, wr, ar, use_condensed_chart);
                                        }
                                        c[(size_t) panel_row + (size_t) jj * (size_t) ldc] += y_acc;
                                    }
                                    g_last_call_stats.structured_attempt_rows += local_rows;
                                    if (used_structured) {
                                        g_last_call_stats.structured_rows += local_rows;
                                        local_dispatch.structured_blocks += 1;
                                    }
                                } else {
                                    const struct gyromatmul_block_q8_0 * wr =
                                        w + (size_t) jj * (size_t) ldb_blocks + (size_t) q8_offset;
                                    gyrolabe_accumulate_dense_rows(
                                        local_rows, panel_row0, lda_blocks, q8_offset,
                                        ldc, jj, a, wr, c, use_condensed_chart);
                                    g_last_call_stats.dense_rows += local_rows;
                                }
                            }
                        } else {
                            for (int jj = jj0; jj < jj1; jj++) {
                                const struct gyromatmul_block_q8_0 * wr =
                                    w + (size_t) jj * (size_t) ldb_blocks + (size_t) q8_offset;
                                gyrolabe_accumulate_dense_rows(
                                    local_rows, panel_row0, lda_blocks, q8_offset,
                                    ldc, jj, a, wr, c, use_condensed_chart);
                            }
                            g_last_call_stats.dense_rows += local_rows * ncol;
                        }
                    }
                }
            } else {
                for (int b = 0; b < k64_blocks; ++b) {
                    const int q8_offset = b * 2;
                    int j;
                    for (j = 0; j < n; ++j) {
                        const struct gyromatmul_block_q8_0 * wr =
                            w + (size_t) j * (size_t) ldb_blocks + (size_t) q8_offset;
                        gyrolabe_accumulate_dense_rows(
                            local_rows, panel_row0, lda_blocks, q8_offset,
                            ldc, j, a, wr, c, use_condensed_chart);
                    }
                }
                g_last_call_stats.dense_rows += local_rows * n * k64_blocks;
            }

            if (k_tail32 != 0) {
                const int q8_offset = k64_blocks * 2;
                int j;
                for (j = 0; j < n; ++j) {
                    const struct gyromatmul_block_q8_0 * weight_row_panel =
                        w + (size_t) j * (size_t) ldb_blocks;
                    const struct gyromatmul_block_q8_0 * weight_tail =
                        weight_row_panel + (size_t) q8_offset;
                    int local_row;
                    for (local_row = 0; local_row < local_rows; ++local_row) {
                        const int panel_row = panel_row0 + local_row;
                        const int row_off = panel_row * lda_blocks;
                        const struct gyromatmul_block_q8_0 * activation_tail =
                            a + (size_t) row_off + (size_t) q8_offset;
                        float tail_dot = 0.0f;
                        if (gyromatmul_vec_dot_q8_0_q8_0_avx2_condensed(
                                1, weight_tail, activation_tail, &tail_dot, use_condensed_chart) == 0) {
                            c[(size_t) panel_row + (size_t) j * (size_t) ldc] += tail_dot;
                        }
                    }
                }
                g_last_call_stats.dense_rows += local_rows * n;
            }
        }
    }

    g_last_dispatch_stats.scanned_blocks = local_dispatch.scanned_blocks;
    g_last_dispatch_stats.structured_blocks = local_dispatch.structured_blocks;
    g_last_dispatch_stats.no_structured_fallback = 0;
    g_last_dispatch_stats.no_k64_blocks = local_dispatch.no_k64_blocks;
    return 0;
}

int gyromatmul_vec_dot_q8_0_q8_0_ref(
    int n_blocks,
    const gyromatmul_block_q8_0 * GYRO_RESTRICT x,
    const gyromatmul_block_q8_0 * GYRO_RESTRICT y,
    float * GYRO_RESTRICT out
) {
    int b;
    if (out == NULL || x == NULL || y == NULL || n_blocks < 0) {
        return -1;
    }

    float acc = 0.0f;

    for (b = 0; b < n_blocks; ++b) {
        int32_t d00 = 0, d01 = 0, d10 = 0, d11 = 0;
        int i;

        for (i = 0; i < 32; ++i) {
            const int8_t vx = x[b].qs[i];
            const int8_t vy = y[b].qs[i];
            const int32_t lx = (int32_t) ((uint8_t) vx & 0x0Fu);
            const int32_t ly = (int32_t) ((uint8_t) vy & 0x0Fu);
            const int32_t hx = (int32_t) (vx >> 4);
            const int32_t hy = (int32_t) (vy >> 4);
            d00 += lx * ly;
            d01 += lx * hy;
            d10 += hx * ly;
            d11 += hx * hy;
        }

        const int32_t block_dot = d00 + 16 * (d01 + d10) + 256 * d11;
        const float scale =
            ggml_fp16_to_fp32((ggml_fp16_t) x[b].d) *
            ggml_fp16_to_fp32((ggml_fp16_t) y[b].d);
        acc += scale * (float) block_dot;
    }

    *out = acc;
    return 0;
}

#if defined(__AVX2__)
static inline __m256i gyro_k4_split_lo_nibble(__m256i v) {
    return _mm256_and_si256(v, _mm256_set1_epi8(0x0F));
}

static inline __m256i gyro_k4_split_hi_nibble_signed(__m256i v) {
    const __m256i mask = _mm256_set1_epi8(0x0F);
    const __m256i logical_hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), mask);
    const __m256i sign_bit = _mm256_and_si256(logical_hi, _mm256_set1_epi8(0x08));
    const __m256i sign_mask = _mm256_cmpeq_epi8(sign_bit, _mm256_set1_epi8(0x08));
    const __m256i sign_fill = _mm256_and_si256(sign_mask, _mm256_set1_epi8((char) 0xF0));
    return _mm256_or_si256(logical_hi, sign_fill);
}

static inline __m256i gyro_dot_s8s8_to_i32(__m256i a, __m256i b) {
    const __m256i abs_a = _mm256_sign_epi8(a, a);
    const __m256i adj_b = _mm256_sign_epi8(b, a);
    const __m256i dot16 = _mm256_maddubs_epi16(abs_a, adj_b);
    return _mm256_madd_epi16(_mm256_set1_epi16(1), dot16);
}

static inline __m256i gyro_dot_u8u8_to_i32(__m256i a, __m256i b) {
    const __m256i dot16 = _mm256_maddubs_epi16(a, b);
    return _mm256_madd_epi16(_mm256_set1_epi16(1), dot16);
}

static inline __m256i gyro_dot_u8s8_to_i32(__m256i u, __m256i s) {
    const __m256i dot16 = _mm256_maddubs_epi16(u, s);
    return _mm256_madd_epi16(_mm256_set1_epi16(1), dot16);
}

static int gyromatmul_vec_dot_q8_0_q8_0_avx2_condensed(
    int n_blocks,
    const gyromatmul_block_q8_0 * GYRO_RESTRICT x,
    const gyromatmul_block_q8_0 * GYRO_RESTRICT y,
    float * GYRO_RESTRICT out,
    int use_condensed
) {
    __m256 acc = _mm256_setzero_ps();
    (void) use_condensed;
    int b;

    if (out == NULL || x == NULL || y == NULL || n_blocks < 0) {
        return -1;
    }

    for (b = 0; b < n_blocks; ++b) {
        const __m256i qx = _mm256_loadu_si256((const __m256i *) x[b].qs);
        const __m256i qy = _mm256_loadu_si256((const __m256i *) y[b].qs);

        const __m256i lx = gyro_k4_split_lo_nibble(qx);
        const __m256i ly = gyro_k4_split_lo_nibble(qy);
        const __m256i hx = gyro_k4_split_hi_nibble_signed(qx);
        const __m256i hy = gyro_k4_split_hi_nibble_signed(qy);

        const __m256i d00_v = gyro_dot_u8u8_to_i32(lx, ly);
        const __m256i d01_v = gyro_dot_u8s8_to_i32(lx, hy);
        const __m256i d10_v = gyro_dot_u8s8_to_i32(ly, hx);
        const __m256i d11_v = gyro_dot_s8s8_to_i32(hx, hy);

        const __m256i cross_v = _mm256_add_epi32(d01_v, d10_v);
        const __m256i dot_v = _mm256_add_epi32(
            d00_v,
            _mm256_add_epi32(_mm256_slli_epi32(cross_v, 4), _mm256_slli_epi32(d11_v, 8))
        );

        const float scale = GYRO_FP16_TO_F32((ggml_fp16_t) x[b].d) * GYRO_FP16_TO_F32((ggml_fp16_t) y[b].d);
        const __m256 dot_f = _mm256_cvtepi32_ps(dot_v);
        const __m256 sc = _mm256_set1_ps(scale);
#if defined(__FMA__)
        acc = _mm256_fmadd_ps(sc, dot_f, acc);
#else
        acc = _mm256_add_ps(_mm256_mul_ps(sc, dot_f), acc);
#endif
    }

    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
    *out = _mm_cvtss_f32(sum);
    return 0;
}

int gyromatmul_vec_dot_q8_0_q8_0_avx2(
    int n_blocks,
    const gyromatmul_block_q8_0 * GYRO_RESTRICT x,
    const gyromatmul_block_q8_0 * GYRO_RESTRICT y,
    float * GYRO_RESTRICT out
) {
    return gyromatmul_vec_dot_q8_0_q8_0_avx2_condensed(n_blocks, x, y, out, 0);
}
#else
static int gyromatmul_vec_dot_q8_0_q8_0_avx2_condensed(
    int n_blocks,
    const gyromatmul_block_q8_0 * GYRO_RESTRICT x,
    const gyromatmul_block_q8_0 * GYRO_RESTRICT y,
    float * GYRO_RESTRICT out,
    int use_condensed
) {
    (void) use_condensed;
    return gyromatmul_vec_dot_q8_0_q8_0_ref(n_blocks, x, y, out);
}

int gyromatmul_vec_dot_q8_0_q8_0_avx2(
    int n_blocks,
    const gyromatmul_block_q8_0 * GYRO_RESTRICT x,
    const gyromatmul_block_q8_0 * GYRO_RESTRICT y,
    float * GYRO_RESTRICT out
) {
    return gyromatmul_vec_dot_q8_0_q8_0_ref(n_blocks, x, y, out);
}
#endif
