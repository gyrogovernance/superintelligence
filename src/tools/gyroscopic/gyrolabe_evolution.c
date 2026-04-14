#include "gyrolabe_evolution.h"
#include "gyrolabe_transforms.h"
#include <math.h>
#include <string.h>

void gyrolabe_chirality_evolve_n(
    const int32_t chi_hist[64],
    const int32_t byte_ensemble[64],
    uint64_t n_steps,
    int32_t out[64]
) {
    float spec_climate[64];
    float spec_ensemble[64];
    int i;
    for (i = 0; i < 64; ++i) {
        spec_climate[i] = (float)chi_hist[i];
        spec_ensemble[i] = (float)byte_ensemble[i];
    }
    /* QuBEC Transform Algebra 9.1: unnormalized WHT; >>= 6 only after final WHT. */
    gyrolabe_wht64_f32_inplace(spec_climate);
    gyrolabe_wht64_f32_inplace(spec_ensemble);

    for (i = 0; i < 64; ++i) {
        double base = (double)spec_ensemble[i] / 64.0;
        double acc = 1.0;
        uint64_t n = n_steps;
        double b = base;
        while (n != 0u) {
            if ((n & 1u) != 0u) {
                acc *= b;
            }
            b *= b;
            n >>= 1u;
        }
        spec_climate[i] = (float)((double)spec_climate[i] * acc);
    }

    gyrolabe_wht64_f32_inplace(spec_climate);
    for (i = 0; i < 64; ++i) {
        double v = (double)spec_climate[i] / 64.0;
        if (v >= 0.0) {
            out[i] = (int32_t)(v + 0.5);
        } else {
            out[i] = (int32_t)(v - 0.5);
        }
    }
}

void gyrolabe_shell_evolve_n(
    const int32_t shell_hist[7],
    const float eigenvals[7],
    uint64_t n_steps,
    int32_t out[7]
) {
    float shell_f[7];
    float spec[7];
    int r;
    for (r = 0; r < 7; ++r) {
        shell_f[r] = (float)shell_hist[r];
    }
    gyrolabe_krawtchouk7_float(shell_f, spec);
    for (r = 0; r < 7; ++r) {
        double base = (double)eigenvals[r];
        double acc = 1.0;
        uint64_t n = n_steps;
        double b = base;
        while (n != 0u) {
            if ((n & 1u) != 0u) {
                acc *= b;
            }
            b *= b;
            n >>= 1u;
        }
        spec[r] = (float)((double)spec[r] * acc);
    }
    {
        float out_f[7];
        gyrolabe_krawtchouk7_inverse_float(spec, out_f);
        /* inverse_float applies K^T and /64; spatial shell counts need no extra scale */
        for (r = 0; r < 7; ++r) {
            out[r] = (int32_t)(out_f[r] + 0.5f);
        }
    }
}

int gyrolabe_horizon_proximity(const int32_t chi_hist[64], float threshold) {
    float spec[64];
    int i;
    float max_val = 0.0f;
    float max_err = 0.0f;
    for (i = 0; i < 64; ++i) {
        spec[i] = (float)chi_hist[i];
    }
    gyrolabe_wht64_float(spec);
    for (i = 0; i < 64; ++i) {
        float a = fabsf(spec[i]);
        if (a > max_val) {
            max_val = a;
        }
    }
    if (max_val < 1e-6f) {
        return 0;
    }
    for (i = 0; i < 64; ++i) {
        float norm = spec[i] / max_val;
        float rounded = roundf(norm);
        float err = fabsf(norm - rounded);
        if (err > max_err) {
            max_err = err;
        }
    }
    return max_err < (1.0f - threshold) ? 1 : 0;
}

void gyrolabe_anisotropy_extract(const int32_t byte_ensemble[256], float eta_vec[6]) {
    int64_t total = 0;
    int axis;
    int b;
    for (b = 0; b < 256; ++b) {
        total += (int64_t)byte_ensemble[b];
    }
    if (total == 0) {
        for (axis = 0; axis < 6; ++axis) {
            eta_vec[axis] = 1.0f;
        }
        return;
    }

    for (axis = 0; axis < 6; ++axis) {
        int64_t count = 0;
        for (b = 0; b < 256; ++b) {
            uint8_t intron = (uint8_t)((unsigned int)b ^ 0xAAu);
            uint8_t payload = (uint8_t)((intron >> 1) & 0x3Fu);
            if ((payload >> axis) & 1) {
                count += (int64_t)byte_ensemble[b];
            }
        }
        {
            float p = (float)count / (float)total;
            eta_vec[axis] = 1.0f - 2.0f * p;
        }
    }
}
