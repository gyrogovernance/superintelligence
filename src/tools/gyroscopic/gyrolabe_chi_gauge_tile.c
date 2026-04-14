#include "gyrolabe_chi_gauge_tile.h"

/* Chi x Gauge lifted tile (Transform Algebra 8.1): WHT16 sectors, 256 spectral params. */

#include <stddef.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(__AVX2__)
void gyrolabe_wht16_f32_inplace(float * x) {
    __m256 v0 = _mm256_loadu_ps(x);
    __m256 v1 = _mm256_loadu_ps(x + 8);
    __m256 t0, t1;

    // Stage 1 (h=1): Pairs (0,1), (2,3), (4,5), (6,7)
    t0 = _mm256_shuffle_ps(v0, v0, 0xB1);
    t1 = _mm256_shuffle_ps(v1, v1, 0xB1);
    v0 = _mm256_blend_ps(_mm256_add_ps(v0, t0), _mm256_sub_ps(t0, v0), 0xAA);
    v1 = _mm256_blend_ps(_mm256_add_ps(v1, t1), _mm256_sub_ps(t1, v1), 0xAA);

    // Stage 2 (h=2): Pairs (0,2), (1,3), (4,6), (5,7)
    t0 = _mm256_shuffle_ps(v0, v0, 0x4E);
    t1 = _mm256_shuffle_ps(v1, v1, 0x4E);
    v0 = _mm256_blend_ps(_mm256_add_ps(v0, t0), _mm256_sub_ps(t0, v0), 0xCC);
    v1 = _mm256_blend_ps(_mm256_add_ps(v1, t1), _mm256_sub_ps(t1, v1), 0xCC);

    // Stage 3 (h=4): Swap 128-bit lanes
    t0 = _mm256_permute2f128_ps(v0, v0, 0x01);
    t1 = _mm256_permute2f128_ps(v1, v1, 0x01);
    v0 = _mm256_blend_ps(_mm256_add_ps(v0, t0), _mm256_sub_ps(t0, v0), 0xF0);
    v1 = _mm256_blend_ps(_mm256_add_ps(v1, t1), _mm256_sub_ps(t1, v1), 0xF0);

    // Stage 4 (h=8): Mix the two vectors
    t0 = _mm256_add_ps(v0, v1);
    t1 = _mm256_sub_ps(v0, v1);

    _mm256_storeu_ps(x, t0);
    _mm256_storeu_ps(x + 8, t1);
}
#else
void gyrolabe_wht16_f32_inplace(float * x) {
    int h;
    for (h = 1; h < 16; h <<= 1) {
        const int stride = h << 1;
        int i;
        for (i = 0; i < 16; i += stride) {
            int j;
            for (j = 0; j < h; ++j) {
                const float a = x[i + j];
                const float b = x[i + h + j];
                x[i + j]     = a + b;
                x[i + h + j] = a - b;
            }
        }
    }
}
#endif

void gyrolabe_chi_gauge_input_wht(const float * x64, float xwht[4][16]) {
    int go, h;
    for (go = 0; go < 4; ++go) {
        for (h = 0; h < 16; ++h) {
            xwht[go][h] = x64[(size_t) go * 16u + (size_t) h];
        }
        gyrolabe_wht16_f32_inplace(xwht[go]);
    }
}

void gyrolabe_apply_spectral_tile64_gauge4_lifted(
    const float xwht[4][16],
    const float * params256,
    float * y64
) {
    float tmp_spectral[16];
    int go;
    int gd;
    int h;

    for (go = 0; go < 4; ++go) {
        for (h = 0; h < 16; ++h) {
            tmp_spectral[h] = 0.0f;
        }
        for (gd = 0; gd < 4; ++gd) {
            const int gi = go ^ gd;
            const float * p_sector = params256 + (size_t) go * 64u + (size_t) gd * 16u;
            for (h = 0; h < 16; ++h) {
                tmp_spectral[h] += p_sector[h] * xwht[gi][h];
            }
        }
        gyrolabe_wht16_f32_inplace(tmp_spectral);
        for (h = 0; h < 16; ++h) {
            /*
             * Full 64-point inverse is (H16 inverse per lane) * (H4 inverse across lanes).
             * Each H16 inverse contributes 1/16; missing H4 would bias output by 4x.
             */
            y64[(size_t) go * 16u + (size_t) h] = tmp_spectral[h] * (1.0f / 64.0f);
        }
    }
}

static void apply_chi_gauge_tile64(
    const float * x64,
    const float * params256,
    float * y64
) {
    float xwht[4][16];
    gyrolabe_chi_gauge_input_wht(x64, xwht);
    gyrolabe_apply_spectral_tile64_gauge4_lifted(xwht, params256, y64);
}

void gyrolabe_apply_spectral_tile64_gauge4(
    const float * x64,
    const float * gauge_eigen256,
    float * y64
) {
    apply_chi_gauge_tile64(x64, gauge_eigen256, y64);
}
