#include "gyrolabe.h"
#include "gyrograph_types.h"
#include "gyrolabe_chi_gauge_tile.h"
#include "gyrolabe_transforms.h"
#include "gyrolabe_wht.h"
#include <math.h>
#include <string.h>

#if defined(_MSC_VER)
#include <intrin.h>
static inline int gyro_popcnt_i(int x) {
    return (int)__popcnt((unsigned int)x);
}
#else
static inline int gyro_popcnt_i(int x) {
    return __builtin_popcount(x);
}
#endif

static float frobenius_norm_64x64(const float *W) {
    double acc = 0.0;
    for (int i = 0; i < 4096; ++i) acc += (double)W[i] * W[i];
    return (float)sqrt(acc);
}

static const int16_t K7[7][7] = {
    {1, 1, 1, 1, 1, 1, 1},
    {6, 4, 2, 0, -2, -4, -6},
    {15, 5, -1, -3, -1, 5, 15},
    {20, 0, -4, 0, 4, 0, -20},
    {15, -5, -1, 3, -1, -5, 15},
    {6, -4, 2, 0, -2, 4, -6},
    {1, -1, 1, -1, 1, -1, 1},
};

static int is_shell_radial_exact_i32(const int32_t B[64][64], int32_t lambda[7]) {
    for (int r = 0; r < 7; ++r) {
        int64_t sum = 0, cnt = 0;
        for (int i = 0; i < 64; ++i) if (gyro_popcnt_i(i) == r)
            for (int j = 0; j < 64; ++j) if (gyro_popcnt_i(j) == r) {
                sum += B[i][j];
                cnt++;
            }
        lambda[r] = cnt ? (int32_t)(sum / cnt) : 0;
    }
    for (int i = 0; i < 64; ++i) for (int j = 0; j < 64; ++j) {
        int ri = gyro_popcnt_i(i), rj = gyro_popcnt_i(j);
        int64_t v = 0;
        for (int r = 0; r < 7; ++r) v += (int64_t)K7[r][ri] * lambda[r] * K7[r][rj];
        if (B[i][j] != (int32_t)(v / 64)) return 0;
    }
    return 1;
}

static int is_chi_invariant_exact_i32(const int32_t B[64][64], int32_t phi[64]) {
    int32_t tmp[64];
    for (int i = 1; i < 64; ++i) for (int j = 0; j < 64; ++j) {
        if (B[i][j] != B[0][j ^ i]) return 0;
    }
    for (int j = 0; j < 64; ++j) tmp[j] = B[0][j];
    gyrolabe_wht64_int32(tmp);
    /* Normalized spectrum for gyrolabe_apply_structured_64 (uses gyrolabe_wht64_float per WHT). */
    for (int j = 0; j < 64; ++j) phi[j] = tmp[j] / 64;
    return 1;
}

static void reconstruct_shell_P_from_lambda(int32_t P[64][64], const int32_t lambda[7]) {
    for (int i = 0; i < 64; ++i) for (int j = 0; j < 64; ++j) {
        int ri = gyro_popcnt_i(i), rj = gyro_popcnt_i(j);
        int64_t v = 0;
        for (int r = 0; r < 7; ++r) v += (int64_t)K7[r][ri] * lambda[r] * K7[r][rj];
        P[i][j] = (int32_t)(v / 64);
    }
}

static void reconstruct_chi_P_from_B(int32_t P[64][64], const int32_t B[64][64]) {
    for (int i = 0; i < 64; ++i) for (int j = 0; j < 64; ++j) P[i][j] = B[0][j ^ i];
}

static float scr_frobenius_ratio_i32(const int32_t P[64][64], const int32_t B[64][64]) {
    double sp = 0.0, sb = 0.0;
    for (int i = 0; i < 64; ++i) for (int j = 0; j < 64; ++j) {
        double bi = (double)B[i][j], pi = (double)P[i][j];
        sb += bi * bi;
        sp += pi * pi;
    }
    if (sb < 1e-30) return 0.0f;
    return (float)(sqrt(sp) / sqrt(sb));
}

GYROLABE_EXPORT int gyrolabe_analyze_operator_64(
    const float *W_block,
    float threshold,
    gyrolabe_operator_report *out
) {
    if (!W_block ||!out) return -1;
    memset(out, 0, sizeof(*out));

    float norm_W = frobenius_norm_64x64(W_block);
    if (norm_W < 1e-9f) {
        out->op_class = GYROLABE_CLASS_GENERIC;
        out->scr = 0.0f;
        return 0;
    }

    (void) threshold;
    {
        int32_t B[64][64];
        int32_t phi[64];
        int32_t lambda[7];
        for (int i = 0; i < 64; ++i) for (int j = 0; j < 64; ++j) {
            B[i][j] = (int32_t) lrintf(W_block[i * 64 + j]);
        }
        // Spec order: shell-radial before chi-invariant.
        if (is_shell_radial_exact_i32(B, lambda)) {
            int32_t P[64][64];
            reconstruct_shell_P_from_lambda(P, lambda);
            for (int s = 0; s < 7; ++s) out->eigenvalues_256[s] = (float) lambda[s];
            out->op_class = GYROLABE_CLASS_SHELL_RADIAL;
            out->scr = scr_frobenius_ratio_i32(P, B);
            out->defect_norm = 0.0f;
            out->eigenvalues_valid = 1;
            return 0;
        }
        if (is_chi_invariant_exact_i32(B, phi)) {
            int32_t P[64][64];
            reconstruct_chi_P_from_B(P, B);
            for (int i = 0; i < 64; ++i) out->eigenvalues_256[i] = (float) phi[i];
            out->op_class = GYROLABE_CLASS_CHI_INVARIANT;
            out->scr = scr_frobenius_ratio_i32(P, B);
            out->defect_norm = 0.0f;
            out->eigenvalues_valid = 1;
            return 0;
        }
    }

    out->op_class = GYROLABE_CLASS_GENERIC;
    out->scr = 0.0f;
    out->defect_norm = norm_W;
    return 0;
}

GYROLABE_EXPORT int gyrolabe_apply_structured_64(
    const gyrolabe_operator_report *report,
    const float *x,
    float *y
) {
    if (!report ||!x ||!y) return -1;
    if (!report->eigenvalues_valid) return -1;

    switch (report->op_class) {
        case GYROLABE_CLASS_CHI_INVARIANT: {
            float tmp[64];
            memcpy(tmp, x, 64 * sizeof(float));
            gyrolabe_wht64_float(tmp);
            for (int i = 0; i < 64; ++i) tmp[i] *= report->eigenvalues_256[i];
            gyrolabe_wht64_float(tmp);
            for (int i = 0; i < 64; ++i) y[i] = tmp[i];
            return 0;
        }
        case GYROLABE_CLASS_SHELL_RADIAL: {
            float tmp[64];
            memcpy(tmp, x, 64 * sizeof(float));
            gyrolabe_wht64_float(tmp);
            for (int i = 0; i < 64; ++i) {
                const int s = gyro_popcnt_i(i);
                tmp[i] *= report->eigenvalues_256[s];
            }
            gyrolabe_wht64_float(tmp);
            for (int i = 0; i < 64; ++i) y[i] = tmp[i];
            return 0;
        }
        case GYROLABE_CLASS_SHELL_GAUGE:
            return -2;
        case GYROLABE_CLASS_CHI_GAUGE:
            gyrolabe_apply_spectral_tile64_gauge4(x, report->eigenvalues_256, y);
            return 0;
        default:
            return -1;
    }
}