#include "gyrolabe_wht.h"
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <assert.h>

#if defined(_MSC_VER)
#  define GYRO_WHT_ALIGN32 __declspec(align(32))
#else
#  define GYRO_WHT_ALIGN32 __attribute__((aligned(32)))
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#endif

static void gyrolabe_wht64_f32_inplace_scalar(float * x) {
    int h;
    int i;
    int j;
    for (h = 1; h < 64; h <<= 1) {
        for (i = 0; i < 64; i += h << 1) {
            for (j = 0; j < h; ++j) {
                float u = x[i + j];
                float v = x[i + h + j];
                x[i + j] = u + v;
                x[i + h + j] = u - v;
            }
        }
    }
}

void gyrolabe_wht64_f32_inplace(float * x) {
#if defined(__AVX2__)
    /* Stack float[64] is often only 16-byte aligned on Windows; AVX2 path uses __m256 stores. */
    GYRO_WHT_ALIGN32 float buf[64];
    memcpy(buf, x, 64u * sizeof(float));
    gyrolabe_wht64_f32_inplace_avx2(buf);
    memcpy(x, buf, 64u * sizeof(float));
#else
    gyrolabe_wht64_f32_inplace_scalar(x);
#endif
}

static uint64_t gyrolabe_precond_xorshift64star(uint64_t * s) {
    uint64_t x = *s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x * 2685821657736338717ULL;
    return *s;
}

void gyrolabe_precond_fill_dr_dc(uint64_t seed, float dr[64], float dc[64]) {
    uint64_t s = seed ? seed : 0x9E3779B97F4A7C15ULL;
    int i;
    for (i = 0; i < 64; ++i) {
        const uint64_t u = gyrolabe_precond_xorshift64star(&s);
        dr[i] = (u & 1u) ? 1.0f : -1.0f;
    }
    for (i = 0; i < 64; ++i) {
        const uint64_t u = gyrolabe_precond_xorshift64star(&s);
        dc[i] = (u & 1u) ? 1.0f : -1.0f;
    }
}

void gyrolabe_precond_apply_Dc_vec(float * x, const float dc[64]) {
    int i;
    for (i = 0; i < 64; ++i) {
        x[i] *= dc[i];
    }
}

void gyrolabe_precond_apply_Dr_vec(float * x, const float dr[64]) {
    int i;
    for (i = 0; i < 64; ++i) {
        x[i] *= dr[i];
    }
}

void gyrolabe_precond_apply_DrDc_mat(float * M, const float dr[64], const float dc[64]) {
    int r;
    int c;
    for (r = 0; r < 64; ++r) {
        for (c = 0; c < 64; ++c) {
            M[(size_t) r * 64u + (size_t) c] *= dr[r] * dc[c];
        }
    }
}

void gyrolabe_hadamard_cols64(float * M) {
    float col[64];
    int c;
    int r;
    for (c = 0; c < 64; ++c) {
        for (r = 0; r < 64; ++r) {
            col[r] = M[(size_t) r * 64u + (size_t) c];
        }
        gyrolabe_wht64_f32_inplace(col);
        for (r = 0; r < 64; ++r) {
            M[(size_t) r * 64u + (size_t) c] = col[r];
        }
    }
}

void gyrolabe_hadamard_rows64(float * M) {
    float row[64];
    int r;
    for (r = 0; r < 64; ++r) {
        memcpy(row, M + (size_t) r * 64u, 64u * sizeof(float));
        gyrolabe_wht64_f32_inplace(row);
        memcpy(M + (size_t) r * 64u, row, 64u * sizeof(float));
    }
}

static void gyrolabe_wht64_int32_scalar(int32_t * x) {
    int h;
    int i;
    int j;
    for (h = 1; h < 64; h <<= 1) {
        for (i = 0; i < 64; i += h << 1) {
            for (j = 0; j < h; ++j) {
                int32_t u = x[i + j];
                int32_t v = x[i + h + j];
                x[i + j] = u + v;
                x[i + h + j] = u - v;
            }
        }
    }
}

void gyrolabe_wht64_int32(int32_t * x) {
#if defined(__AVX2__)
    GYRO_WHT_ALIGN32 int32_t buf[64];
    memcpy(buf, x, 64u * sizeof(int32_t));
    gyrolabe_wht64_int32_avx2(buf);
    memcpy(x, buf, 64u * sizeof(int32_t));
#else
    gyrolabe_wht64_int32_scalar(x);
#endif
}

int gyrolabe_wht64_int32_safe(int32_t * x) {
    for (int h = 1; h < 64; h <<= 1) {
        for (int i = 0; i < 64; i += h<<1) {
            for (int j = 0; j < h; ++j) {
                const int32_t u = x[i + j];
                const int32_t v = x[i + h + j];
                const int64_t s = (int64_t) u + (int64_t) v;
                const int64_t d = (int64_t) u - (int64_t) v;
                if (s < (int64_t) INT32_MIN || s > (int64_t) INT32_MAX ||
                    d < (int64_t) INT32_MIN || d > (int64_t) INT32_MAX) {
                    return -1;
                }
                x[i + j] = (int32_t) s;
                x[i + h + j] = (int32_t) d;
            }
        }
    }
    return 0;
}

void gyrolabe_wht64_verify_self_inverse(void) {
    int i, j;
    for (i = 0; i < 64; ++i) {
        int32_t v[64] = {0};
        v[i] = 1;
        gyrolabe_wht64_int32(v);
        gyrolabe_wht64_int32(v);
        for (j = 0; j < 64; ++j) {
            assert(v[j] == (j == i ? 64 : 0));
        }
    }
}

#if defined(__AVX2__)
void gyrolabe_wht64_f32_inplace_avx2(float * x) {
    __m256 * v = (__m256 *) x;
    int i;
    int j;

    for (i = 0; i < 8; ++i) {
        __m256 v0 = v[i];
        __m256 t0;

        t0 = _mm256_shuffle_ps(v0, v0, 0xB1);
        v0 = _mm256_blend_ps(_mm256_add_ps(v0, t0), _mm256_sub_ps(t0, v0), 0xAA);

        t0 = _mm256_shuffle_ps(v0, v0, 0x4E);
        v0 = _mm256_blend_ps(_mm256_add_ps(v0, t0), _mm256_sub_ps(t0, v0), 0xCC);

        t0 = _mm256_permute2f128_ps(v0, v0, 0x01);
        v[i] = _mm256_blend_ps(_mm256_add_ps(v0, t0), _mm256_sub_ps(t0, v0), 0xF0);
    }

    for (i = 0; i < 8; i += 2) {
        __m256 a = v[i];
        __m256 b = v[i + 1];
        v[i] = _mm256_add_ps(a, b);
        v[i + 1] = _mm256_sub_ps(a, b);
    }

    for (i = 0; i < 8; i += 4) {
        for (j = 0; j < 2; ++j) {
            __m256 a = v[i + j];
            __m256 b = v[i + j + 2];
            v[i + j] = _mm256_add_ps(a, b);
            v[i + j + 2] = _mm256_sub_ps(a, b);
        }
    }

    for (j = 0; j < 4; ++j) {
        __m256 a = v[j];
        __m256 b = v[j + 4];
        v[j] = _mm256_add_ps(a, b);
        v[j + 4] = _mm256_sub_ps(a, b);
    }
}

void gyrolabe_wht64_int32_avx2(int32_t * x) {
    __m256i * const vi = (__m256i *) x;
    int i;
    int j;

    for (i = 0; i < 8; ++i) {
        __m256i v0 = vi[i];
        __m256i t0;
        __m256i sum;
        __m256i diff;

        t0 = _mm256_shuffle_epi32(v0, 0xB1);
        sum = _mm256_add_epi32(v0, t0);
        diff = _mm256_sub_epi32(t0, v0);
        v0 = _mm256_blend_epi32(sum, diff, 0xAA);

        t0 = _mm256_shuffle_epi32(v0, 0x4E);
        sum = _mm256_add_epi32(v0, t0);
        diff = _mm256_sub_epi32(t0, v0);
        v0 = _mm256_blend_epi32(sum, diff, 0xCC);

        t0 = _mm256_permute2f128_si256(v0, v0, 0x01);
        sum = _mm256_add_epi32(v0, t0);
        diff = _mm256_sub_epi32(t0, v0);
        vi[i] = _mm256_blend_epi32(sum, diff, 0xF0);
    }

    for (i = 0; i < 8; i += 2) {
        __m256i a = vi[i];
        __m256i b = vi[i + 1];
        vi[i] = _mm256_add_epi32(a, b);
        vi[i + 1] = _mm256_sub_epi32(a, b);
    }

    for (i = 0; i < 8; i += 4) {
        for (j = 0; j < 2; ++j) {
            __m256i a = vi[i + j];
            __m256i b = vi[i + j + 2];
            vi[i + j] = _mm256_add_epi32(a, b);
            vi[i + j + 2] = _mm256_sub_epi32(a, b);
        }
    }

    for (j = 0; j < 4; ++j) {
        __m256i a = vi[j];
        __m256i b = vi[j + 4];
        vi[j] = _mm256_add_epi32(a, b);
        vi[j + 4] = _mm256_sub_epi32(a, b);
    }
}
#endif

#ifdef GYROLABE_SELF_TEST
#include "gyrolabe_chi_gauge_tile.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

void test_wht16_involution(void) {
    float x[16];
    float y[16];
    int i;

    for (i = 0; i < 16; ++i) {
        x[i] = (float)(rand() % 100);
    }
    memcpy(y, x, sizeof(x));

    gyrolabe_wht16_f32_inplace(y);
    gyrolabe_wht16_f32_inplace(y);
    for (i = 0; i < 16; ++i) {
        y[i] *= 1.0f / 16.0f;
    }

    for (i = 0; i < 16; ++i) {
        assert(fabsf(y[i] - x[i]) < 1e-5f);
    }
}
#endif
