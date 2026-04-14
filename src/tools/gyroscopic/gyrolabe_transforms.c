#include "gyrolabe_transforms.h"
#include "gyrolabe_wht.h"
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

static const int32_t KRAWTCHOUK_7[7][7] = {
    {1, 1, 1, 1, 1, 1, 1},
    {6, 4, 2, 0, -2, -4, -6},
    {15, 5, -1, -3, -1, 5, 15},
    {20, 0, -4, 0, 4, 0, -20},
    {15, -5, -1, 3, -1, -5, 15},
    {6, -4, 2, 0, -2, 4, -6},
    {1, -1, 1, -1, 1, -1, 1}
};

void gyrolabe_wht64_float(float *data) {
    gyrolabe_wht64_f32_inplace(data);
    for (int i = 0; i < 64; ++i) {
        data[i] *= (1.0f / 64.0f);
    }
}

#if defined(__AVX2__)
static void gyrolabe_wht64_batch_int32_avx2_chunk(int32_t ** arrays, int chunk) {
    __m256i lanes[64];
    int pos;
    int h;
    int i;
    int j;
    int b;

    if (arrays == NULL || chunk <= 0) {
        return;
    }
    if (chunk > 8) {
        chunk = 8;
    }

    for (pos = 0; pos < 64; ++pos) {
        int32_t tmp[8];
        for (b = 0; b < chunk; ++b) {
            tmp[b] = arrays[b][pos];
        }
        for (; b < 8; ++b) {
            tmp[b] = 0;
        }
        lanes[pos] = _mm256_loadu_si256((__m256i *) tmp);
    }

    for (h = 1; h < 64; h <<= 1) {
        for (i = 0; i < 64; i += h << 1) {
            for (j = 0; j < h; ++j) {
                __m256i u = lanes[i + j];
                __m256i v = lanes[i + h + j];
                lanes[i + j] = _mm256_add_epi32(u, v);
                lanes[i + h + j] = _mm256_sub_epi32(u, v);
            }
        }
    }

    for (pos = 0; pos < 64; ++pos) {
        int32_t tmp[8];
        _mm256_storeu_si256((__m256i *) tmp, lanes[pos]);
        for (b = 0; b < chunk; ++b) {
            arrays[b][pos] = tmp[b];
        }
    }
}
#endif

void gyrolabe_wht64_batch_int32(int32_t ** arrays, int batch_size) {
    int base;
    if (arrays == NULL || batch_size <= 0) {
        return;
    }
#if defined(__AVX2__)
    for (base = 0; base < batch_size; base += 8) {
        int chunk = batch_size - base;
        if (chunk > 8) {
            chunk = 8;
        }
        gyrolabe_wht64_batch_int32_avx2_chunk(arrays + base, chunk);
    }
#else
    for (base = 0; base < batch_size; ++base) {
        gyrolabe_wht64_int32(arrays[base]);
    }
#endif
}

void gyrolabe_wht64_batch_float(float ** arrays, int batch_size) {
    for (int b = 0; b < batch_size; ++b) {
        gyrolabe_wht64_float(arrays[b]);
    }
}

void gyrolabe_krawtchouk7_int32(const int32_t shell_hist[7], int32_t spectral[7]) {
    for (int k = 0; k < 7; ++k) {
        int64_t acc = 0;
        for (int w = 0; w < 7; ++w) {
            acc += (int64_t)shell_hist[w] * KRAWTCHOUK_7[w][k];
        }
        spectral[k] = (int32_t)(acc / 64);
    }
}

void gyrolabe_krawtchouk7_inverse_int32(const int32_t spectral[7], int32_t shell_hist[7]) {
    for (int w = 0; w < 7; ++w) {
        int64_t acc = 0;
        for (int k = 0; k < 7; ++k) {
            acc += (int64_t)spectral[k] * KRAWTCHOUK_7[k][w];
        }
        shell_hist[w] = (int32_t)acc;
    }
}

/*
 * Float Krawtchouk-7 (QuBEC / GyroLabe): unnormalized forward
 *   spectral[k] = sum_w shell_hist[w] * K[w][k]
 * inverse applies K^T and divides by 64 so
 *   inverse(forward(h)) == h, forward(inverse(s)) == s
 * (int32 path keeps integer /64 in forward for lattice counts.)
 */
void gyrolabe_krawtchouk7_float(const float shell_hist[7], float spectral[7]) {
    for (int k = 0; k < 7; ++k) {
        float acc = 0.0f;
        for (int w = 0; w < 7; ++w) {
            acc += shell_hist[w] * (float)KRAWTCHOUK_7[w][k];
        }
        spectral[k] = acc;
    }
}

void gyrolabe_krawtchouk7_inverse_float(const float spectral[7], float shell_hist[7]) {
    for (int w = 0; w < 7; ++w) {
        float acc = 0.0f;
        for (int k = 0; k < 7; ++k) {
            acc += spectral[k] * (float)KRAWTCHOUK_7[k][w];
        }
        shell_hist[w] = acc / 64.0f;
    }
}

void gyrolabe_k4char4_int32(const int32_t family_hist[4], int32_t character[4]) {
    character[0] = family_hist[0] + family_hist[1] + family_hist[2] + family_hist[3];
    character[1] = family_hist[0] - family_hist[1] + family_hist[2] - family_hist[3];
    character[2] = family_hist[0] + family_hist[1] - family_hist[2] - family_hist[3];
    character[3] = family_hist[0] - family_hist[1] - family_hist[2] + family_hist[3];
}

GYROLABE_EXPORT void gyrolabe_k4char4_float(const float family_hist[4], float character[4]) {
    character[0] = family_hist[0] + family_hist[1] + family_hist[2] + family_hist[3];
    character[1] = family_hist[0] - family_hist[1] + family_hist[2] - family_hist[3];
    character[2] = family_hist[0] + family_hist[1] - family_hist[2] - family_hist[3];
    character[3] = family_hist[0] - family_hist[1] - family_hist[2] + family_hist[3];
}

void gyrolabe_k4_decompose_int32(int32_t v, int16_t *L, int16_t *H) {
    *L = (int16_t)(((uint32_t)v & 0xFFFFu) << 16 >> 16);
    *H = (int16_t)(v >> 16);
}

void gyrolabe_k4_contract(
    const int32_t *q,
    const int32_t *k,
    int n,
    int64_t *D00,
    int64_t *D01,
    int64_t *D10,
    int64_t *D11
) {
    int64_t d00 = 0;
    int64_t d01 = 0;
    int64_t d10 = 0;
    int64_t d11 = 0;
    int i;
    for (i = 0; i < n; ++i) {
        int16_t Lq = 0;
        int16_t Hq = 0;
        int16_t Lk = 0;
        int16_t Hk = 0;
        gyrolabe_k4_decompose_int32(q[i], &Lq, &Hq);
        gyrolabe_k4_decompose_int32(k[i], &Lk, &Hk);
        d00 += (int64_t)Lq * (int64_t)Lk;
        d01 += (int64_t)Lq * (int64_t)Hk;
        d10 += (int64_t)Hq * (int64_t)Lk;
        d11 += (int64_t)Hq * (int64_t)Hk;
    }
    *D00 = d00;
    *D01 = d01;
    *D10 = d10;
    *D11 = d11;
}

int64_t gyrolabe_k4_dot(const int32_t *q, const int32_t *k, int n) {
    int64_t D00 = 0;
    int64_t D01 = 0;
    int64_t D10 = 0;
    int64_t D11 = 0;
    gyrolabe_k4_contract(q, k, n, &D00, &D01, &D10, &D11);
    return D00 + 65536LL * (D01 + D10) + 65536LL * 65536LL * D11;
}