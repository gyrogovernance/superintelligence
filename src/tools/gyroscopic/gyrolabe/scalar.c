#include "core.h"
#ifndef GYRO_EXPORT
#define GYRO_EXPORT GYROLABE_EXPORT
#endif

// gyrolabe_scalar.c — Structural scalar operations.
//
// Functions here use comparisons, counting, thresholding, WHT,
// and linear arithmetic. They do not use expf, tanhf, or sqrtf.

/* Forward declarations for codec functions used by fused extractor */
extern void gyrolabe_init(void);
extern void gyrolabe_extract_scan(
    const uint8_t* bytes, int64_t len,
    uint8_t* q_class_out, uint8_t* family_out, uint8_t* micro_ref_out,
    int32_t* signatures_out, int32_t* states_out
);
extern void gyrolabe_state24_to_omega12_batch(
    const int32_t* states_in, int64_t n, int32_t* omega12_out, uint8_t* valid_out
);

// Exact support-based attention weighting replacing softmax.
// Identifies support set within support_delta of row maximum,
// assigns uniform 1/count weight, zeros everything else.
GYRO_EXPORT void gyrolabe_attention_uniform_weights_f32(
    const float* raw_scores,
    int64_t B, int64_t H, int64_t Tq, int64_t Tk,
    float support_delta,
    int64_t hard_cap,
    float* weights_out,
    int32_t* support_out
) {
    if (!raw_scores || !weights_out || !support_out ||
        B <= 0 || H <= 0 || Tq <= 0 || Tk <= 0) return;

    const int64_t bh = B * H;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(bh * Tq >= 1024)
#endif
    for (int64_t g = 0; g < bh * Tq; ++g) {
        const int64_t row_idx = g;
        const float* row = raw_scores + row_idx * Tk;
        float* wrow = weights_out + row_idx * Tk;
        
        float mx = row[0];
        for (int64_t t = 1; t < Tk; ++t) {
            if (row[t] > mx) mx = row[t];
        }

        float thresh = mx - support_delta;
        int32_t sup = 0;
        for (int64_t t = 0; t < Tk; ++t) {
            if (row[t] >= thresh) sup++;
        }

        if (hard_cap > 0 && sup > (int32_t)hard_cap)
            sup = (int32_t)hard_cap;
        if (sup < 1) sup = 1;

        support_out[row_idx] = sup;

        float w = 1.0f / (float)sup;

        if (hard_cap > 0 && sup < Tk) {
            for (int64_t t = 0; t < Tk; ++t) wrow[t] = 0.0f;
            int32_t assigned = 0;
            for (int64_t t = Tk - 1; t >= 0 && assigned < sup; --t) {
                if (row[t] >= thresh) {
                    wrow[t] = w;
                    assigned++;
                }
            }
        } else {
            for (int64_t t = 0; t < Tk; ++t) {
                wrow[t] = (row[t] >= thresh) ? w : 0.0f;
            }
        }
    }
}


GYRO_EXPORT void gyrolabe_hidden_geometry_distance6_f32(
    const float* hidden_states,
    int64_t B, int64_t T, int64_t H,
    uint8_t* out
) {
    if (!hidden_states || !out || B <= 0 || T <= 0 || H <= 0) return;

    for (int64_t b = 0; b < B; ++b) {
        uint64_t prev_sign = 0;
        int has_prev = 0;

        for (int64_t t = 0; t < T; ++t) {
            const float* row = hidden_states + (b * T + t) * H;

            float folded[64];
            if (H == 64) {
                for (int i = 0; i < 64; ++i) folded[i] = row[i];
            } else if (H > 64) {
                for (int i = 0; i < 64; ++i) folded[i] = 0.0f;
                int64_t usable = (H / 64) * 64;
                for (int64_t j = 0; j < usable; ++j) {
                    folded[j % 64] += row[j];
                }
            } else {
                for (int i = 0; i < 64; ++i) {
                    folded[i] = (i < H) ? row[i] : 0.0f;
                }
            }

            for (int stride = 1; stride < 64; stride <<= 1) {
                int jump = stride << 1;
                for (int base = 0; base < 64; base += jump) {
                    for (int j = 0; j < stride; ++j) {
                        float u = folded[base + j];
                        float v = folded[base + j + stride];
                        folded[base + j] = u + v;
                        folded[base + j + stride] = u - v;
                    }
                }
            }

            uint64_t sign = 0;
            for (int i = 0; i < 64; ++i) {
                if (folded[i] < 0.0f) sign |= (1ULL << i);
            }

            uint8_t dist6 = 0;
            if (has_prev) {
                uint32_t diff64 = (uint32_t)gyro_popcnt64(sign ^ prev_sign);
                dist6 = (uint8_t)((diff64 * 6 + 32) / 64);
                if (dist6 > 6) dist6 = 6;
            }

            out[b * T + t] = dist6;
            prev_sign = sign;
            has_prev = 1;
        }
    }
}

GYRO_EXPORT void gyrolabe_chirality_distance_adjacent_2d(
    const int32_t* states,
    int64_t B, int64_t T,
    int32_t lookahead,
    uint8_t* out
) {
    if (!states || !out || B <= 0 || T <= 0 || lookahead < 0) return;
    for (int64_t b = 0; b < B; ++b) {
        const int32_t* row = states + b * T;
        uint8_t* orow = out + b * T;
        for (int64_t i = 0; i < T; ++i) {
            int64_t j = i + (int64_t)lookahead;
            orow[i] = (j >= T) ? 0 : gyro_chirality_distance_pair((uint32_t)row[i], (uint32_t)row[j]);
        }
    }
}

GYRO_EXPORT void gyrolabe_extract_fields_fused(
    const uint8_t* bytes, int64_t n,
    uint8_t* q_class_out,
    uint8_t* family_out,
    uint8_t* micro_ref_out,
    int32_t* signatures_out,
    int32_t* states_out,
    int32_t* omega12_out,
    uint8_t* omega12_valid_out,
    uint8_t* chirality6_out,
    uint8_t* shell_out,
    int32_t* q_hist64_out,
    int32_t* family_hist4_out,
    int32_t* micro_hist64_out,
    int32_t* shell_hist7_out,
    int32_t* q_weight_hist7_out,
    int32_t* bit_excitation6_out
) {
    if (!bytes || n <= 0) return;

    gyrolabe_init();

    for (int i = 0; i < 64; ++i) q_hist64_out[i] = 0;
    for (int i = 0; i < 4; ++i) family_hist4_out[i] = 0;
    for (int i = 0; i < 64; ++i) micro_hist64_out[i] = 0;
    for (int i = 0; i < 7; ++i) shell_hist7_out[i] = 0;
    for (int i = 0; i < 7; ++i) q_weight_hist7_out[i] = 0;
    for (int i = 0; i < 6; ++i) bit_excitation6_out[i] = 0;

    gyrolabe_extract_scan(
        bytes, n,
        q_class_out, family_out, micro_ref_out,
        signatures_out, states_out
    );

    for (int64_t i = 0; i < n; ++i) {
        gyrolabe_state24_to_omega12_batch(states_out + i, 1, omega12_out + i, omega12_valid_out + i);
        uint8_t valid = omega12_valid_out[i];

        if (valid) {
            uint32_t o12 = (uint32_t)omega12_out[i];
            uint8_t u6 = (uint8_t)((o12 >> 6) & 0x3F);
            uint8_t v6 = (uint8_t)(o12 & 0x3F);
            uint8_t chi6 = u6 ^ v6;
            uint8_t sh = (uint8_t)gyro_popcnt32((uint32_t)chi6);
            chirality6_out[i] = chi6;
            shell_out[i] = sh;

            uint8_t q6 = q_class_out[i];
            uint8_t fam = family_out[i];
            uint8_t mic = micro_ref_out[i];
            uint8_t qw = (uint8_t)gyro_popcnt32((uint32_t)(q6 & 0x3F));

            q_hist64_out[q6 & 0x3F]++;
            family_hist4_out[fam & 0x03]++;
            micro_hist64_out[mic & 0x3F]++;
            if (sh < 7) shell_hist7_out[sh]++;
            if (qw < 7) q_weight_hist7_out[qw]++;
            for (int j = 0; j < 6; ++j) {
                if ((q6 >> j) & 1) bit_excitation6_out[j]++;
            }
        } else {
            chirality6_out[i] = 0;
            shell_out[i] = 0;
        }
    }
}
