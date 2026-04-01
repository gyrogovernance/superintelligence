#include "core.h"

#include <immintrin.h>
#include <stdlib.h>
#include <string.h>

static int gyro_kernel_is_ref(void) {
    static int cached = -1;
    if (cached >= 0) {
        return cached;
    }
    const char * v = getenv("GGML_GYROSCOPIC_KERNEL");
    if (v == NULL || v[0] == '\0') {
        cached = 0;
        return 0;
    }
    cached = strcmp(v, "ref") == 0 ? 1 : 0;
    return cached;
}

static inline float gyro_fp16_to_f32(uint16_t h) {
    const uint32_t sign = ((uint32_t) h & 0x8000u) << 16;
    uint32_t exp  = ((uint32_t) h >> 10) & 0x1Fu;
    uint32_t frac = (uint32_t) h & 0x03FFu;

    uint32_t out;
    if (exp == 0) {
        if (frac == 0) {
            out = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((frac & 0x0400u) == 0) {
                frac <<= 1;
                exp--;
            }
            frac &= 0x03FFu;
            out = sign | (exp << 23) | (frac << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7F800000u | (frac << 13);
    } else {
        out = sign | ((exp + (127 - 15)) << 23) | (frac << 13);
    }

    float f;
    memcpy(&f, &out, sizeof(f));
    return f;
}

static inline int32_t gyro_hsum_i32_256(__m256i v) {
    int32_t tmp[8];
    _mm256_storeu_si256((__m256i *) tmp, v);
    return tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
}

void gyromatmul_runtime_query(gyromatmul_runtime_caps * out_caps) {
    if (out_caps == NULL) {
        return;
    }
    out_caps->avx2_enabled = 1u;
#if defined(__F16C__) || defined(_M_AMD64)
    out_caps->f16c_enabled = 1u;
#else
    out_caps->f16c_enabled = 0u;
#endif
#if defined(__FMA__) || defined(_M_AMD64)
    out_caps->fma_enabled = 1u;
#else
    out_caps->fma_enabled = 0u;
#endif
    out_caps->reserved = 0u;
}

int gyromatmul_vec_dot_f32_ref(
    int n,
    const float * GYRO_RESTRICT x,
    const float * GYRO_RESTRICT y,
    float * GYRO_RESTRICT out
) {
    if (n < 0 || x == NULL || y == NULL || out == NULL) {
        return -1;
    }

    float acc = 0.0f;
    for (int i = 0; i < n; ++i) {
        acc += x[i] * y[i];
    }

    *out = acc;
    return 0;
}

int gyromatmul_vec_dot_f32_avx2(
    int n,
    const float * GYRO_RESTRICT x,
    const float * GYRO_RESTRICT y,
    float * GYRO_RESTRICT out
) {
    if (n < 0 || x == NULL || y == NULL || out == NULL) {
        return -1;
    }

    float acc = 0.0f;
    int i = 0;

    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);
        __m256 vp = _mm256_mul_ps(vx, vy);

        float tmp[8];
        _mm256_storeu_ps(tmp, vp);

        acc += tmp[0];
        acc += tmp[1];
        acc += tmp[2];
        acc += tmp[3];
        acc += tmp[4];
        acc += tmp[5];
        acc += tmp[6];
        acc += tmp[7];
    }

    for (; i < n; ++i) {
        acc += x[i] * y[i];
    }

    *out = acc;
    return 0;
}

int gyromatmul_vec_dot_q8_0_q8_0_ref(
    int n_blocks,
    const gyromatmul_block_q8_0 * GYRO_RESTRICT x,
    const gyromatmul_block_q8_0 * GYRO_RESTRICT y,
    float * GYRO_RESTRICT out
) {
    if (n_blocks < 0 || x == NULL || y == NULL || out == NULL) {
        return -1;
    }

    float acc = 0.0f;

    for (int b = 0; b < n_blocks; ++b) {
        int32_t isum = 0;
        for (int i = 0; i < 32; ++i) {
            isum += (int32_t) x[b].qs[i] * (int32_t) y[b].qs[i];
        }

        const float d =
            gyro_fp16_to_f32(x[b].d) *
            gyro_fp16_to_f32(y[b].d);

        acc += d * (float) isum;
    }

    *out = acc;
    return 0;
}

int gyromatmul_vec_dot_q8_0_q8_0_avx2(
    int n_blocks,
    const gyromatmul_block_q8_0 * GYRO_RESTRICT x,
    const gyromatmul_block_q8_0 * GYRO_RESTRICT y,
    float * GYRO_RESTRICT out
) {
    if (n_blocks < 0 || x == NULL || y == NULL || out == NULL) {
        return -1;
    }

    float acc = 0.0f;

    for (int b = 0; b < n_blocks; ++b) {
        const int8_t * px = x[b].qs;
        const int8_t * py = y[b].qs;

        __m128i x_lo_8 = _mm_loadu_si128((const __m128i *) (px + 0));
        __m128i x_hi_8 = _mm_loadu_si128((const __m128i *) (px + 16));
        __m128i y_lo_8 = _mm_loadu_si128((const __m128i *) (py + 0));
        __m128i y_hi_8 = _mm_loadu_si128((const __m128i *) (py + 16));

        __m256i x_lo_16 = _mm256_cvtepi8_epi16(x_lo_8);
        __m256i x_hi_16 = _mm256_cvtepi8_epi16(x_hi_8);
        __m256i y_lo_16 = _mm256_cvtepi8_epi16(y_lo_8);
        __m256i y_hi_16 = _mm256_cvtepi8_epi16(y_hi_8);

        __m256i p_lo = _mm256_madd_epi16(x_lo_16, y_lo_16);
        __m256i p_hi = _mm256_madd_epi16(x_hi_16, y_hi_16);

        int32_t isum = gyro_hsum_i32_256(p_lo) + gyro_hsum_i32_256(p_hi);

        const float d =
            gyro_fp16_to_f32(x[b].d) *
            gyro_fp16_to_f32(y[b].d);

        acc += d * (float) isum;
    }

    *out = acc;
    return 0;
}

int gyromatmul_vec_dot_f32(
    int n,
    const float * GYRO_RESTRICT x,
    const float * GYRO_RESTRICT y,
    float * GYRO_RESTRICT out
) {
    if (gyro_kernel_is_ref()) {
        return gyromatmul_vec_dot_f32_ref(n, x, y, out);
    }
    return gyromatmul_vec_dot_f32_avx2(n, x, y, out);
}

int gyromatmul_vec_dot_q8_0_q8_0(
    int n_blocks,
    const gyromatmul_block_q8_0 * GYRO_RESTRICT x,
    const gyromatmul_block_q8_0 * GYRO_RESTRICT y,
    float * GYRO_RESTRICT out
) {
    if (gyro_kernel_is_ref()) {
        return gyromatmul_vec_dot_q8_0_q8_0_ref(n_blocks, x, y, out);
    }
    return gyromatmul_vec_dot_q8_0_q8_0_avx2(n_blocks, x, y, out);
}

int gyromatmul_gemm_f32(
    int m,
    int n,
    int k,
    const float * GYRO_RESTRICT a,
    int lda,
    const float * GYRO_RESTRICT b,
    int ldb,
    float * GYRO_RESTRICT c,
    int ldc
) {
    if (m < 0 || n < 0 || k < 0 || a == NULL || b == NULL || c == NULL) {
        return -1;
    }
    if (ldc < m) {
        return -1;
    }

    for (int i = 0; i < m; ++i) {
        const float * ai = a + (size_t) i * (size_t) lda;
        for (int j = 0; j < n; ++j) {
            const float * bj = b + (size_t) j * (size_t) ldb;
            float s = 0.0f;
            if (gyromatmul_vec_dot_f32(k, ai, bj, &s) != 0) {
                return -1;
            }
            c[i + (size_t) j * (size_t) ldc] = s;
        }
    }

    return 0;
}

int gyromatmul_gemm_q8_0_q8_0(
    int m,
    int n,
    int k,
    const gyromatmul_block_q8_0 * GYRO_RESTRICT a,
    int lda_blocks,
    const gyromatmul_block_q8_0 * GYRO_RESTRICT b,
    int ldb_blocks,
    float * GYRO_RESTRICT c,
    int ldc
) {
    if (m < 0 || n < 0 || k < 0 || (k % 32) != 0 || a == NULL || b == NULL || c == NULL) {
        return -1;
    }
    if (ldc < m) {
        return -1;
    }

    const int k_blocks = k / 32;

    for (int i = 0; i < m; ++i) {
        const gyromatmul_block_q8_0 * ai = a + (size_t) i * (size_t) lda_blocks;
        for (int j = 0; j < n; ++j) {
            const gyromatmul_block_q8_0 * bj = b + (size_t) j * (size_t) ldb_blocks;
            float s = 0.0f;
            if (gyromatmul_vec_dot_q8_0_q8_0(k_blocks, ai, bj, &s) != 0) {
                return -1;
            }
            c[i + (size_t) j * (size_t) ldc] = s;
        }
    }

    return 0;
}

int gyromatmul_out_prod_f32(
    int rows,
    int cols,
    const float * GYRO_RESTRICT x,
    const float * GYRO_RESTRICT y,
    float * GYRO_RESTRICT out,
    int ld_out
) {
    if (rows < 0 || cols < 0 || x == NULL || y == NULL || out == NULL) {
        return -1;
    }

    for (int i = 0; i < rows; ++i) {
        const float xi = x[i];
        float * row = out + (size_t) i * (size_t) ld_out;
        for (int j = 0; j < cols; ++j) {
            row[j] = xi * y[j];
        }
    }

    return 0;
}
